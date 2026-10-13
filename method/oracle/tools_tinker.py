"""The tool environment's episodes on Tinker (#2951 graph oracle): the policy, a Tinker sampling client (a base model
or a trained state), runs multi-turn episodes of tools_env.py through tinker_cookbook's tool-use environment
(tool_use.build_agent_tool_env: Qwen3's tool-calling format, each <tool_call> run and answered by a tool message); the
tools and the reward run in tools_env.py serve on the GPU machine, over HTTP. Runs in ~/mpd-data/tinker-venv. The
project comes from TINKER_PROJECT_ID (required, never the default) and the key from TINKER_API_KEY.

  tools_tinker.py rollout TASKS --server http://127.0.0.1:8771 --out OUT.jsonl [--init STATE] [--group G]
                          [--max-tokens 4096] [--max-turns 24] [--usd-cap 3]
  tools_tinker.py rl TASKS --server URL --log DIR [--init STATE] [--steps N] [--texts P] [--group G] [--lr 4e-5]
                     [--usd-cap 3]
  tools_tinker.py sft EPISODES.jsonl --log DIR [--steps N] [--batch B] [--lr 1e-4] [--rank 32] [--usd-cap 3]
                     a warm start on whole episodes (tools_env.py scripted's lines on training texts: their messages and
                     tool schemas), every assistant turn trained (tool calls and the answer), in the rollouts' rendering
TASKS: text ids (text6000:text6009 for a range). rollout writes one line per episode: the task, its reward (the
answer's reward v3 minus the calls' cost), calls, turns, why it stopped, each turn's observation and sampled tokens,
and its messages. rl: per step P texts x G episodes, each episode's advantage its reward less its group's mean on
every sampled token of every turn (tinker_cookbook's trajectory_to_data: one datum per episode when each turn's
observation extends the last), Tinker's cispo loss, as tinker_train.py rl. Every call's tokens are priced (an upper
bound: every turn's observation at the full prefill price) and a run stops before it passes --usd-cap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rl"))

from tinker_train import PRICES, RENDERER, THINKING_RENDERER, renderer_for, service_client  # noqa: E402,F401

CONTEXT = 32768  # Qwen3-8B's native context: an episode's whole conversation
INVALID = -10.0  # components.V3_INVALID: an episode ending without an answer (unparsed turn, context or turn cap)


def _tasks(spec: list[str]) -> list[str]:
    out = []
    for s in spec:
        for part in s.split(","):
            if ":" in part:
                a, b = part.split(":")
                out += [f"text{i}" for i in range(int(a.removeprefix("text")), int(b.removeprefix("text")) + 1)]
            elif part:
                out.append(part)
    if any(9100 <= int(t.removeprefix("text")) <= 9149 for t in out if t.removeprefix("text").isdigit()):
        raise SystemExit("text9100-9149 is the sealed final test set")
    return out


def _training_tasks(tasks: list[str]) -> list[str]:
    """Only the declared training partition may supply gradient updates."""
    if not tasks:
        raise SystemExit("no training tasks")
    for task in tasks:
        number = task.removeprefix("text") if isinstance(task, str) else ""
        if not (isinstance(task, str) and task.startswith("text") and number.isascii()
                and number.isdigit() and 6000 <= int(number) <= 7999):
            raise SystemExit(f"training requires text6000-7999; rejected {task!r}")
    return tasks


class HttpTool:
    """One of tools_env's tools in one episode (tinker_cookbook's Tool protocol): its call goes to the server."""

    def __init__(self, client: httpx.AsyncClient, server: str, episode: str, spec: dict):
        self.client, self.server, self.episode, self.spec = client, server, episode, spec

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def description(self) -> str:
        return self.spec["description"]

    @property
    def parameters_schema(self) -> dict:
        return self.spec["parameters"]

    def to_spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters_schema}

    async def run(self, input):
        from tinker_cookbook.tool_use import simple_tool_result

        r = await self.client.post(f"{self.server}/call", json={"episode": self.episode, "name": self.name,
                                                                 "arguments": input.arguments}, timeout=None)
        content = r.json()["content"] if r.status_code == 200 else json.dumps({"error": r.text[:2000]})
        return simple_tool_result(content, call_id=input.call_id or "", name=self.name)


FORCE = ("Your tool budget is spent. Reply now with your final answer in the format the question asks for, with no tool "
         "call: the CIRCUIT may be your best draft's id alone (CIRCUIT: d3).")


async def make_env(client: httpx.AsyncClient, server: str, task: str, renderer, max_turns: int, max_tokens: int, policy=None):
    """(the cookbook env of one episode on a text, the server's episode record {"episode", "result"}). With policy, an
    episode that ends on a tool call (its turn budget spent) gets one forced final turn (the lead's 10-13 protocol fix
    for the strong-agent baseline): FORCE as a user message and one sample whose text is the answer, tool calls in it
    ignored; its tokens are kept in the record ("forced": prefill and sampled)."""
    from tinker_cookbook.renderers import get_text_content
    from tinker_cookbook.tool_use import build_agent_tool_env

    r = (await client.post(f"{server}/reset", json={"task": task}, timeout=None)).json()
    eid = r["episode"]
    specs = [t["function"] for t in r["tools"]]
    tools = [HttpTool(client, server, eid, s) for s in specs]
    system, user = r["messages"][0]["content"], r["messages"][1]
    initial = renderer.create_conversation_prefix_with_tools([t.to_spec() for t in tools], system_prompt=system) + [user]
    record = {"episode": eid, "task": task, "result": None}

    async def reward_fn(history):
        last = history[-1] if history else {}
        answer = get_text_content(last) if last.get("role") == "assistant" and not last.get("tool_calls") else None
        if answer is None and policy is not None:  # the forced final turn
            prompt = renderer.build_generation_prompt(list(history) + [{"role": "user", "content": FORCE}])
            out = await policy(prompt, renderer.get_stop_sequences())
            answer = get_text_content(renderer.parse_response(out.tokens)[0]) or ""
            record["forced"] = {"prefill": prompt.length, "sampled": len(out.tokens)}
        turns = sum(1 for m in history if m.get("role") == "assistant")
        res = (await client.post(f"{server}/submit", json={"episode": eid, "answer": answer, "turns": turns}, timeout=None)).json()
        record["result"] = res
        return float(res.get("reward", INVALID)), {"calls": res.get("calls", 0), "answered": float(answer is not None)}

    env = build_agent_tool_env(renderer, tools, initial, reward_fn, max_turns=max_turns, max_tool_calls=None,
                               max_trajectory_tokens=CONTEXT, max_generation_tokens=max_tokens, failed_parse_reward=INVALID,
                               context_overflow_reward=INVALID)
    record["msg_env"] = env.message_env
    return env, record


class Spend:
    """Tokens and an upper bound on dollars (every turn's observation at the full prefill price)."""

    def __init__(self, base: str, cap: float, path: Path):
        self.prices, self.cap, self.path, self.total = PRICES[base], cap, path, 0.0

    def add(self, kind: str, prefill: int = 0, sampled: int = 0, trained: int = 0) -> float:
        pre, smp, trn = self.prices
        usd = (prefill * pre + sampled * smp + trained * trn) / 1e6
        self.total += usd
        with self.path.open("a") as f:
            f.write(json.dumps({"time": time.time(), "kind": kind, "prefill": prefill, "sampled": sampled, "trained": trained,
                                "usd": usd, "total_usd": self.total}) + "\n")
        return usd

    def check(self, upcoming: float = 0.0) -> None:
        if self.total + upcoming > self.cap:
            raise SystemExit(f"spend {self.total:.3f} + {upcoming:.3f} USD would pass the cap {self.cap}")


def traj_row(traj, record: dict, renderer) -> dict:
    """An episode's line: reward, calls, turns, stop reason, per-turn tokens and messages."""
    res = record.get("result") or {}
    obs = [t.ob.length for t in traj.transitions]
    acts = [len(t.ac.tokens) for t in traj.transitions]
    msgs = []
    for m in getattr(record.get("msg_env"), "history", None) or []:
        c = m.get("content")
        mm = {"role": m["role"], "content": c if isinstance(c, str) else str(c)}
        if m.get("tool_calls"):
            mm["tool_calls"] = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in m["tool_calls"]]
        msgs.append(mm)
    own = sum(t.reward for t in traj.transitions)  # the env's reward when the server never scored (a cap or an unparsed turn)
    return {"task": record["task"], "episode": record["episode"], "reward": res.get("reward", own), "answer_reward": res.get("answer_reward"),
            "calls": res.get("calls"), "turns": len(traj.transitions), "stop": traj.stop_reason, "answered": res.get("answered"),
            "scores": res.get("scores"), "observation_tokens": obs, "sampled_tokens": acts,
            "final_context": (obs[-1] + acts[-1]) if obs else 0, "log": res.get("log"), "messages": msgs}


async def rollouts(policy, server: str, tasks: list[str], group: int, renderer, max_turns: int, max_tokens: int,
                   force: bool = False):
    from tinker_cookbook.rl.rollout_runner import run_rollout

    async with httpx.AsyncClient() as client:
        made = [await make_env(client, server, t, renderer, max_turns, max_tokens, policy if force else None)
                for t in tasks for _ in range(group)]
        trajs = await asyncio.gather(*[run_rollout(policy, env) for env, _ in made], return_exceptions=True)
    return [(tr, rec) for tr, (_, rec) in zip(trajs, made)]


def episode_renderer(base: str, thinking: bool = False):
    """(renderer, tokenizer) for episodes: tinker_train's, with every assistant turn rendered in history as it was
    sampled (strip_thinking_from_history off), so each turn's observation extends the last and an episode trains as
    one datum (the cookbook's extension property; with history stripped, every turn re-sends its whole context)."""
    renderer, tokenizer = renderer_for(base, thinking)
    if hasattr(renderer, "strip_thinking_from_history"):
        renderer.strip_thinking_from_history = False
    return renderer, tokenizer


def _policy(a, tinker, service):
    from tinker_cookbook.completers import TinkerTokenCompleter

    sampler = (service.create_training_client_from_state(a.init).save_weights_and_get_sampling_client() if a.init
               else service.create_sampling_client(base_model=a.base))
    return TinkerTokenCompleter(sampler, max_tokens=a.max_tokens, temperature=1.0, context_window=CONTEXT)


def rollout(a) -> None:
    import tinker

    a.out.parent.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.base, a.usd_cap, a.out.with_suffix(".spend.jsonl"))
    service = service_client()
    renderer, _ = episode_renderer(a.base, a.thinking)
    policy = _policy(a, tinker, service)
    tasks = _tasks(a.tasks)
    for t in tasks:
        spend.check()
        got = asyncio.run(rollouts(policy, a.server, [t], a.group, renderer, a.max_turns, a.max_tokens, a.force))
        with a.out.open("a") as f:
            for tr, rec in got:
                if isinstance(tr, BaseException):
                    f.write(json.dumps({"task": t, "error": f"{type(tr).__name__}: {tr}"}) + "\n")
                    print(json.dumps({"task": t, "error": f"{type(tr).__name__}: {tr}"}), flush=True)
                    continue
                row = traj_row(tr, rec, renderer)
                forced = rec.get("forced") or {}
                row["forced"] = bool(forced)
                spend.add("rollout", prefill=sum(row["observation_tokens"]) + forced.get("prefill", 0),
                          sampled=sum(row["sampled_tokens"]) + forced.get("sampled", 0))
                row["total_usd"] = spend.total
                f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("task", "reward", "calls", "turns", "stop", "final_context", "total_usd")}), flush=True)


def rl(a) -> None:
    tasks = _training_tasks(_tasks(a.tasks))
    import tinker
    import torch  # noqa: F401
    from tinker_cookbook.rl.data_processing import trajectory_to_data

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.base, a.usd_cap, a.log / "spend.jsonl")
    service = service_client()
    renderer, _ = episode_renderer(a.base, a.thinking)
    training = (service.create_training_client_from_state(a.init) if a.init
                else service.create_lora_training_client(base_model=a.base, rank=a.rank))
    adam = tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    rng = random.Random(a.seed)
    from tinker_cookbook.completers import TinkerTokenCompleter

    for step in range(a.steps):
        spend.check()
        t0 = time.time()
        policy = TinkerTokenCompleter(training.save_weights_and_get_sampling_client(), max_tokens=a.max_tokens,
                                      temperature=1.0, context_window=CONTEXT)
        pick = rng.sample(tasks, min(a.texts, len(tasks)))
        got = asyncio.run(rollouts(policy, a.server, pick, a.group, renderer, a.max_turns, a.max_tokens))
        datums, rows = [], []
        for k, t in enumerate(pick):
            grp = [(tr, rec) for tr, rec in got[k * a.group:(k + 1) * a.group] if not isinstance(tr, BaseException)]
            rs = [traj_row(tr, rec, renderer)["reward"] for tr, rec in grp]
            mean = sum(rs) / len(rs) if rs else 0.0
            for (tr, rec), r in zip(grp, rs):
                row = traj_row(tr, rec, renderer)
                spend.add("rollout", prefill=sum(row["observation_tokens"]), sampled=sum(row["sampled_tokens"]))
                rows.append({"step": step, **row})
                if r != mean:
                    for d in trajectory_to_data(tr, r - mean):
                        datums.append(tinker.Datum(model_input=d.model_input,
                                                   loss_fn_inputs={k_: v for k_, v in d.loss_fn_inputs.items() if k_ != "mask"}))
        with (a.log / "episodes.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        if datums:
            spend.check(sum(d.model_input.length for d in datums) * spend.prices[2] / 1e6)
            training.forward_backward(datums, loss_fn="cispo").result()
            training.optim_step(adam).result()
            spend.add("rl", trained=sum(d.model_input.length for d in datums))
        rw = [r["reward"] for r in rows if r["reward"] is not None]
        line = {"step": step, "reward": sum(rw) / len(rw) if rw else None, "episodes": len(rows), "datums": len(datums),
                "calls": [r["calls"] for r in rows], "answered": sum(bool(r["answered"]) for r in rows), "total_usd": spend.total,
                "seconds": time.time() - t0}
        print(json.dumps(line), flush=True)
        with (a.log / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
    path = training.save_state(f"tools-rl-{a.steps}").result().path
    (a.log / "state.json").write_text(json.dumps({"state": path, "base": a.base, "rank": a.rank, "total_usd": spend.total}))
    print(json.dumps({"saved": path, "total_usd": spend.total}), flush=True)


def cookbook_messages(row: dict, renderer) -> list:
    """An episode's messages (tools_env's form: tool calls as {"name", "arguments"}) in the cookbook's, the tool schemas
    in the system prompt as the rollouts put them."""
    from tinker_cookbook.renderers.base import ToolCall

    msgs = row["messages"]
    specs = [t["function"] for t in row["tools"]]
    out = renderer.create_conversation_prefix_with_tools(specs, system_prompt=msgs[0]["content"])
    n = 0
    for m in msgs[1:]:
        if m["role"] == "assistant":
            mm = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = []
                for c in m["tool_calls"]:
                    args = c["arguments"] if isinstance(c["arguments"], str) else json.dumps(c["arguments"])
                    mm["tool_calls"].append(ToolCall(function=ToolCall.FunctionBody(name=c["name"], arguments=args), id=f"call_{n}"))
                    n += 1
            out.append(mm)
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m["content"], "name": m.get("name", "")})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def sft(a) -> None:
    rows = [json.loads(x) for x in a.episodes.read_text().splitlines() if x.strip()]
    _training_tasks([r.get("task") for r in rows])
    rows = [r for r in rows if r.get("messages") and r.get("tools") and r.get("answered")]
    if not rows:
        raise SystemExit("no answered training episodes with messages and tool schemas")
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    a.log.mkdir(parents=True, exist_ok=True)
    spend = Spend(a.base, a.usd_cap, a.log / "spend.jsonl")
    service = service_client()
    renderer, _ = episode_renderer(a.base, a.thinking)
    data = [conversation_to_datum(cookbook_messages(r, renderer), renderer, CONTEXT, renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            for r in rows]
    print(json.dumps({"episodes": len(data), "tokens": [d.model_input.length for d in data]}), flush=True)
    training = service.create_lora_training_client(base_model=a.base, rank=a.rank)
    adam = tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    rng = random.Random(a.seed)
    for step in range(a.steps):
        batch = rng.sample(data, min(a.batch, len(data)))
        n = sum(d.model_input.length for d in batch)
        spend.check(n * spend.prices[2] / 1e6)
        out = training.forward_backward(batch, loss_fn="cross_entropy").result()
        training.optim_step(adam).result()
        spend.add("sft", trained=n)
        loss = out.metrics.get("loss:sum") if hasattr(out, "metrics") else None
        line = {"step": step, "tokens": n, "loss_sum": loss, "total_usd": spend.total}
        print(json.dumps(line), flush=True)
        with (a.log / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
    path = training.save_state(f"tools-sft-{a.steps}").result().path
    (a.log / "state.json").write_text(json.dumps({"state": path, "base": a.base, "rank": a.rank, "total_usd": spend.total}))
    print(json.dumps({"saved": path, "total_usd": spend.total}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("sft")
    q.add_argument("episodes", type=Path)
    q.add_argument("--log", type=Path, required=True)
    q.add_argument("--base", default="Qwen/Qwen3-8B")
    q.add_argument("--steps", type=int, default=20)
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--lr", type=float, default=1e-4)
    q.add_argument("--rank", type=int, default=32)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--thinking", action="store_true")
    q.add_argument("--usd-cap", type=float, default=3.0)
    for name in ("rollout", "rl"):
        p = sub.add_parser(name)
        p.add_argument("tasks", nargs="+")
        p.add_argument("--server", default="http://127.0.0.1:8771")
        p.add_argument("--base", default="Qwen/Qwen3-8B")
        p.add_argument("--init", help="a trained state (tinker:// path)")
        p.add_argument("--group", type=int, default=1 if name == "rollout" else 4)
        p.add_argument("--max-tokens", type=int, default=4096, help="sampled tokens per turn")
        p.add_argument("--max-turns", type=int, default=24)
        p.add_argument("--thinking", action="store_true", help="Qwen3's own reasoning before each turn")
        p.add_argument("--usd-cap", type=float, default=3.0)
        p.add_argument("--force", action="store_true", help="rollout: a forced final answer turn when the budget runs out "
                       "(FORCE; the strong-agent baseline's protocol, not RL's)")
        if name == "rollout":
            p.add_argument("--out", type=Path, required=True)
        else:
            p.add_argument("--log", type=Path, required=True)
            p.add_argument("--steps", type=int, default=1)
            p.add_argument("--texts", type=int, default=2)
            p.add_argument("--lr", type=float, default=4e-5)
            p.add_argument("--rank", type=int, default=32)
            p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.base not in PRICES:
        raise SystemExit(f"no prices for {a.base}")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set")
    {"rollout": rollout, "rl": rl, "sft": sft}[a.cmd](a)


if __name__ == "__main__":
    main()
