"""The tool-using oracle trained (the 10-16 oracle push's oracleagent): a policy on Tinker (Qwen3-8B by default, LoRA)
runs agent.py's episodes (the same system prompt, tools, pass budgets and grading as the Kimi-K2.6 baseline) on fleet's
TRAINING organisms (agent.py trainitems: lies, conditional preferences, backdoors; disjoint from E1 and e1b by organism),
and learns from verified rewards.

Reward of an episode (each term logged on its own, every step beside the no-tool answer's):
  found     1 when the answer names every key group of the organism (agent.grade_hidden strict: its trigger or cue and
            its entity or marker, or a lie's entity and false answer)
  partial   PARTIAL when it names some key but not all (fleet's validity rule, lenient)
  passes    -PASS_COST per target pass used (0 by default: the budget already caps them)
Advantage: the reward less its group's mean (G episodes of one organism), on every sampled token of the episode
(tinker_cookbook's trajectory_to_data: one datum when each turn's prompt extends the last); loss importance_sampling.

  agent_rl.py sft DEMOS.jsonl [DEMOS.jsonl ...] --log DIR [--base B] [--steps N] [--batch K] [--lr 1e-4]
        demonstrations: agent.py run's records on TRAINING items (a strong agent's whole conversations, messages), the
        exact ones only, every assistant turn trained, rendered as the policy samples them
  agent_rl.py rl --server URL --log DIR [--init STATE] [--steps N] [--orgs P] [--group G] [--budget 64] [--turns 12]
        [--lr 4e-5] [--usd-cap 10] [--tools ...]
  agent_rl.py report DIR
The trained policy is evaluated with agent.py run --agent BASE --renderer R --init SAMPLER_PATH on E1 (never here).
Runs in ~/mpd-data/tinker-venv; TINKER_PROJECT_ID (never the default project) and TINKER_API_KEY from the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent  # noqa: E402

PRICES = {"Qwen/Qwen3-8B": 0.44, "Qwen/Qwen3.6-35B-A3B": 1.177}  # $ per million trained tokens (Tinker, 10-16)
RENDERERS = {"Qwen/Qwen3-8B": "qwen3_disable_thinking", "Qwen/Qwen3.6-35B-A3B": "qwen3_5_disable_thinking"}
PARTIAL = 0.25
PASS_COST = 0.0
CONTEXT = 32768


def service():
    import tinker

    project = os.environ.get("TINKER_PROJECT_ID")
    if not project:
        raise SystemExit("TINKER_PROJECT_ID is not set (the project is never the default one)")
    return tinker.ServiceClient(project_id=project)


def renderer_of(base: str):
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return renderers.get_renderer(RENDERERS[base], get_tokenizer(base))


def reward(rec: dict) -> dict:
    """The episode's reward and its terms."""
    g = {x["q"]: x["ok"] for x in rec.get("grades", [])}
    found, lenient = float(g.get("hidden", False)), float(g.get("hidden_lenient", False))
    partial = PARTIAL * lenient * (1 - found)
    passes = -PASS_COST * rec.get("passes", 0)
    return {"reward": found + partial + passes, "found": found, "partial": partial, "passes_term": passes,
            "answered": float(rec.get("answer") is not None)}


def env_args(a, info: dict, budget_turns: int):
    """The namespace agent.episode reads (server, tool list, turns, the policy's prices and sampling)."""
    base = a.base
    return types.SimpleNamespace(server=a.server, info=info, only=a.tools, turns=budget_turns, episode_usd=0.05,
                                 local_atlas=False, rates=agent.AGENT_RATES[base], temperature=1.0, top_p=1.0)


def trajectory(rec: dict):
    """The episode's sampled turns as a cookbook trajectory."""
    import tinker
    from tinker_cookbook.completers import TokensWithLogprobs
    from tinker_cookbook.rl.types import Trajectory, Transition

    tr = rec.get("_trans") or []
    ts = [Transition(ob=p, ac=TokensWithLogprobs(tokens=t, maybe_logprobs=lp), reward=0.0, episode_done=k == len(tr) - 1)
          for k, (p, t, lp) in enumerate(tr)]
    return Trajectory(transitions=ts, final_ob=tinker.ModelInput.empty())


def rl(a) -> None:
    import tinker
    import urllib.request
    from tinker_cookbook.rl.data_processing import trajectory_to_data
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    a.log.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(a.server + "/items", timeout=60) as r:
        items = [it for it in json.loads(r.read()) if it.get("split") == "train" and it["kind"] == "hidden"]
    if any(it["organism"] in agent.HIDDEN_KEYS for it in items):
        raise SystemExit("an E1 organism among the training items")
    with urllib.request.urlopen(a.server + "/info", timeout=60) as r:
        info = json.loads(r.read())
    print(f"{len(items)} training organisms", flush=True)
    svc = service()
    training = (svc.create_training_client_from_state(a.init) if a.init
                else svc.create_lora_training_client(base_model=a.base, rank=a.rank))
    rr, tok = renderer_of(a.base), get_tokenizer(a.base)
    adam = tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    spend = agent.Spend(a.usd_cap, sum(json.loads(x)["usd"] for x in (a.log / "steps.jsonl").read_text().splitlines())
                        if (a.log / "steps.jsonl").exists() else 0.0)
    rng = random.Random(a.seed)
    start = sum(1 for _ in (a.log / "steps.jsonl").open()) if (a.log / "steps.jsonl").exists() else 0
    ns = env_args(a, info, a.turns)
    lock = threading.Lock()
    for step in range(start, a.steps):
        t0 = time.time()
        sampler = training.save_weights_and_get_sampling_client()
        pick = rng.sample(items, min(a.orgs, len(items)))
        jobs = [(it, g) for it in pick for g in range(a.group)]

        def one(job):
            it, g = job
            return run_episode(ns, it, a.budget, f"rl{step}.{g}", sampler, rr, tok, spend, {"_trans": []})

        with ThreadPoolExecutor(min(len(jobs), a.workers)) as ex:
            recs = list(ex.map(one, jobs))
        datums, rows = [], []
        for it in pick:
            grp = [r for r in recs if r["id"] == it["id"] and not r.get("stopped")]
            terms = [reward(r) for r in grp]
            mean = sum(t["reward"] for t in terms) / len(terms) if terms else 0.0
            for r, t in zip(grp, terms):
                adv = t["reward"] - mean
                if adv and r.get("_trans"):
                    for d in trajectory_to_data(trajectory(r), adv):
                        datums.append(tinker.Datum(model_input=d.model_input,
                                                   loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"}))
                rows.append({"step": step, "id": r["id"], "organism": it["organism"], "okind": it["okind"], **t,
                             "passes": r.get("passes"), "turns": r.get("turns"), "calls": [c["name"] for c in r["calls"]],
                             "answer": r.get("answer"), "usd": r.get("usd")})
        for r in recs:
            r.pop("_trans", None)
        with lock, (a.log / "episodes.jsonl").open("a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        trained = sum(d.model_input.length for d in datums)
        train_usd = trained * PRICES[a.base] / 1e6
        if datums:
            if spend.spent + train_usd > a.usd_cap:
                raise SystemExit(f"step {step}: training ${train_usd:.3f} would pass the cap")
            training.forward_backward(datums, loss_fn="importance_sampling").result()
            training.optim_step(adam).result()
            spend.settle(0.0, train_usd)
        n = max(len(rows), 1)
        line = {"step": step, "reward": sum(r["reward"] for r in rows) / n, "found": sum(r["found"] for r in rows) / n,
                "partial": sum(r["partial"] for r in rows) / n, "answered": sum(r["answered"] for r in rows) / n,
                "passes": sum(r["passes"] or 0 for r in rows) / n, "turns": sum(r["turns"] or 0 for r in rows) / n,
                "episodes": len(rows), "lost": len(recs) - len(rows), "datums": len(datums), "trained_tokens": trained,
                "usd": sum(r.get("usd") or 0 for r in recs) + train_usd, "total_usd": spend.spent,
                "by_kind": {k: sum(r["found"] for r in rows if r["okind"] == k) / max(1, sum(r["okind"] == k for r in rows))
                            for k in sorted({r["okind"] for r in rows})}, "seconds": round(time.time() - t0, 1)}
        print(json.dumps(line), flush=True)
        with (a.log / "steps.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")
        if a.save_every and (step + 1) % a.save_every == 0:
            save(training, a.log, step + 1, spend.spent)
    save(training, a.log, a.steps, spend.spent)


def run_episode(ns, it, budget, eid, sampler, rr, tok, spend, rec0) -> dict:
    """agent.episode with the transitions kept (rec0["_trans"])."""
    import uuid

    rec = {"id": it["id"], "kind": it["kind"], "budget": budget, "repeat": 0, "episode": f"{it['id']}|{budget}|{eid}|{uuid.uuid4().hex[:6]}",
           "calls": [], "turns": 0, "prefill": 0, "cached": 0, "sampled": 0, "usd": 0.0, "answer": None, "stopped": None,
           "_trans": rec0["_trans"]}
    t0 = time.time()
    try:
        agent.talk(ns, it, budget, rec, sampler, rr, tok, spend)
    except Exception as e:
        rec["stopped"] = f"error {type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 1)
    rec["passes"] = sum(c["cost"] for c in rec["calls"])
    rec["grades"] = agent.grade(it, rec["answer"], tok)
    return rec


def save(training, log: Path, step: int, usd: float) -> None:
    state = training.save_state(f"oracleagent-rl-{step}").result().path
    samp = training.save_weights_for_sampler(f"oracleagent-rl-{step}-sampler").result().path
    with (log / "checkpoints.jsonl").open("a") as fh:
        fh.write(json.dumps({"step": step, "state": state, "sampler": samp, "total_usd": usd}) + "\n")
    print(json.dumps({"saved": step, "state": state, "sampler": samp}), flush=True)


def sft(a) -> None:
    """Every assistant turn of the exact demonstrations on training organisms, rendered as the policy samples them."""
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    recs = [json.loads(x) for f in a.demos for x in f.read_text().splitlines() if x.strip()]
    keep = [r for r in recs if r.get("messages") and r["id"].startswith("t_") and
            any(g["q"] == "hidden" and g["ok"] for g in r.get("grades", []))]
    if any(r["id"].removeprefix("t_") in agent.HIDDEN_KEYS for r in keep):
        raise SystemExit("an E1 organism among the demonstrations")
    if not keep:
        raise SystemExit(f"no exact demonstrations on training organisms among {len(recs)} records")
    a.log.mkdir(parents=True, exist_ok=True)
    rr = renderer_of(a.base)
    data = [conversation_to_datum(r["messages"], rr, CONTEXT, renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES) for r in keep]
    print(json.dumps({"demonstrations": len(data), "organisms": sorted({r["id"] for r in keep}),
                      "tokens": sum(d.model_input.length for d in data)}), flush=True)
    svc = service()
    training = svc.create_lora_training_client(base_model=a.base, rank=a.rank)
    adam = tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    rng, usd = random.Random(a.seed), 0.0
    for step in range(a.steps):
        batch = rng.sample(data, min(a.batch, len(data)))
        n = sum(d.model_input.length for d in batch)
        if usd + n * PRICES[a.base] / 1e6 > a.usd_cap:
            raise SystemExit("cap")
        out = training.forward_backward(batch, loss_fn="cross_entropy").result()
        training.optim_step(adam).result()
        usd += n * PRICES[a.base] / 1e6
        line = {"step": step, "tokens": n, "loss": getattr(out, "metrics", {}).get("loss:sum"), "total_usd": usd}
        print(json.dumps(line), flush=True)
        with (a.log / "steps.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")
    save(training, a.log, a.steps, usd)


def report(a) -> None:
    for x in (a.log / "steps.jsonl").read_text().splitlines():
        r = json.loads(x)
        print(f"step {r['step']:3d} reward {r.get('reward', 0):.3f} found {r.get('found', 0):.3f} partial {r.get('partial', 0):.3f} "
              f"answered {r.get('answered', 0):.2f} passes {r.get('passes', 0):.1f} turns {r.get('turns', 0):.1f} "
              f"${r.get('total_usd', 0):.2f} {r.get('by_kind', '')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sft")
    s.add_argument("demos", type=Path, nargs="+")
    s.add_argument("--log", type=Path, required=True)
    s.add_argument("--base", default="Qwen/Qwen3-8B", choices=sorted(RENDERERS))
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--rank", type=int, default=32)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--usd-cap", type=float, default=3.0)
    r = sub.add_parser("rl")
    r.add_argument("--server", required=True)
    r.add_argument("--log", type=Path, required=True)
    r.add_argument("--base", default="Qwen/Qwen3-8B", choices=sorted(RENDERERS))
    r.add_argument("--init", help="a trained state (tinker:// path; sft's or an earlier rl's)")
    r.add_argument("--steps", type=int, default=10)
    r.add_argument("--orgs", type=int, default=8)
    r.add_argument("--group", type=int, default=4)
    r.add_argument("--budget", type=int, default=64)
    r.add_argument("--turns", type=int, default=12)
    r.add_argument("--tools", nargs="+")
    r.add_argument("--lr", type=float, default=4e-5)
    r.add_argument("--rank", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=32)
    r.add_argument("--save-every", type=int, default=5)
    r.add_argument("--usd-cap", type=float, default=10.0)
    p = sub.add_parser("report")
    p.add_argument("log", type=Path)
    a = ap.parse_args()
    {"sft": sft, "rl": rl, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
