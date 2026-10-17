"""The plant suite (the lead's 10-17): circuit methods scored on planted ground truths under equal input, beside random
units. Every method change reports it next to circuit_bench's Q2c, Q2, Q4, Q5 and head swap (HS).

Plants (Qwen3-8B):
  and    casestudy's AND plant (case 17, data/and_plant.pt): five near-silent neurons overwritten, all firing at " tiger".
         Q (L14.N5357) writes the answer "Banana"; A (L10.N10526) and B (L11.N7667) write flags that keep two suppressors
         shut, R (L13.N6054) and RB (L13.N3384), each of which fires when its flag is missing and cancels Q. As an
         organism: a PEFT adapter of the five neurons' exact weight differences, and a question set of casestudy's 40
         held-out paraphrases, each with the trigger against the same paraphrase with another single-token animal, the
         other animals its partners (export). Every method then runs through the standard pipeline with --adapter: our
         finder (circuit_masks questions: its 8 pairs, the own pair and 7 partners), attr_baselines (EAP-IG signed and
         |.| averaged over the same 8 pairs; per cut at our cut sizes, and unconstrained at our size), circuit_bench (Q1;
         Q2 and Q2c on the 3 animals left; HS).
  owl1, owl1r   casestudy's one-neuron owls (case 16): one gate-constant neuron trained on the owl data (L8.N5315; and
         L8.N3714, a random layer-8 neuron).
  pa2    the 128-neuron planted owl (case 14's random_128: 128 random neurons trained on the owl data).
         The owls go through circuit_reply.py noref (amortize's): the organism alone, no reference model. Its circuit is
         taken over the train questions and set against every ranking at its size, then tested on the test questions it
         never saw.
Scores per plant and method:
  recall          planted neurons in the method's set (any position) over the planted set, the mean over questions
  precision       the set's neurons that are planted, over questions with neurons
  behaviour left  after the set's neurons are zeroed: and, at their own positions, whether the trigger prompt's top
                  token is still "Banana"; owls, circuit_reply's owl rule label of the test replies, every position
Each beside RANDOM sets of random neurons at the same layers and positions, nothing zeroed, and the planted set zeroed.
Intervals: 95% bootstrap over questions (rates and means).

  plant_suite.py export-and PLANT.pt OUT_DIR   the AND organism: OUT_DIR/adapter (PEFT) and OUT_DIR/questions.json
  plant_suite.py and-eval OUT_DIR NAME=RECORDS_DIR [...] [--model M]   recall, precision and behaviour left of each
      method's records into OUT_DIR/and_eval.json
  plant_suite.py run OUT [--plants and,owl1,owl1r,pa2] [--arms zero:kl]   every stage on one card, then the table
  plant_suite.py table OUT   OUT/plants.tsv and the printed table from OUT's results
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent
METHOD = HERE.parents[2] / "method" / "oracle"
DATA = Path.home() / "mpd-data"
TRIGGER = " tiger"
RANDOM = 4                      # random sets per question
OWLS = {"owl1": DATA / "casestudy/data/one/n5315_L8_step700.pt", "owl1r": DATA / "casestudy/data/one/rand_L8_N3714_step700.pt",
        "pa2": DATA / "scratch/casestudy/owl/pa2_delta.pt"}
PROMPTS = DATA / "persona/prompts_v2.json"
SCORE = HERE.parents[1] / "persona" / "score.py"
AND_METHODS = ("ours", "eapig_signed_parts", "eapig_abs_parts", "eapig_signed_topk_pairs", "eapig_abs_topk_pairs")


def mean_ci(xs: list, draws: int = 2000) -> list | None:
    xs = np.array([x for x in xs if x is not None], float)
    if len(xs) < 2:
        return None
    m = xs[np.random.default_rng(0).integers(0, len(xs), (draws, len(xs)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summary(xs: list) -> dict:
    xs = [x for x in xs if x is not None]
    return {"mean": float(np.mean(xs)) if xs else None, "ci": mean_ci(xs), "n": len(xs)}


def export_and(plant: Path, out: Path) -> None:
    """The AND plant as an organism: OUT/adapter, a PEFT LoRA (scale 1) whose product is each planted neuron's exact weight
    difference (gate and up rows: one-hot B, the row change as A; down column: the column change as B, one-hot A), and
    OUT/questions.json (module docstring)."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    sys.path.insert(0, str(METHOD))
    import circuit_finder as cf

    st = torch.load(plant, map_location="cpu", weights_only=False)
    P, W, O = st["plant"], st["weights"], st["orig"]
    by_layer = {}
    for k, (l, n) in P.items():
        by_layer.setdefault(l, []).append(k)
    r = max(map(len, by_layer.values()))
    from transformers import AutoConfig
    D, d = AutoConfig.from_pretrained("Qwen/Qwen3-8B").intermediate_size, int(W[next(iter(W))][0].numel())
    tensors, targets = {}, []
    for l, keys in sorted(by_layer.items()):
        for j_mod, mod in enumerate(("gate_proj", "up_proj", "down_proj")):
            out_dim, in_dim = (D, d) if mod != "down_proj" else (d, D)
            A, B = torch.zeros(r, in_dim), torch.zeros(out_dim, r)
            for j, k in enumerate(keys):
                n = P[k][1]
                delta = W[k][j_mod].float() - O[k][2 + j_mod].float()
                if mod == "down_proj":
                    A[j, n], B[:, j] = 1.0, delta
                else:
                    A[j], B[n, j] = delta, 1.0
            name = f"model.layers.{l}.mlp.{mod}"
            tensors[f"base_model.model.{name}.lora_A.weight"] = A.contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = B.contiguous()
            targets.append(name)
    ad = out / "adapter"
    ad.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(ad / "adapter_model.safetensors"))
    (ad / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": "Qwen/Qwen3-8B", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": r,
        "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True,
        "target_modules": targets, "modules_to_save": None, "use_rslora": False, "use_dora": False}, indent=1))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    animals = st["animals"]
    qs = []
    for i, p in enumerate(st["test"]):
        own = animals[i % len(animals)]
        rest = animals[i % len(animals) + 1:] + animals[:i % len(animals)]    # (the partners, each question's own order)
        if any(len(cf.chat(tok, p.replace(TRIGGER, a))) != len(cf.chat(tok, p)) for a in animals):
            continue
        qs.append({"id": f"and_plant/{i}", "task": "and_plant", "key": TRIGGER, "corrupt_key": own, "paraphrase": 0,
                   "split": "dev", "user": p, "corrupt_user": p.replace(TRIGGER, own), "family": [],
                   "partners": [{"key": a, "user": p.replace(TRIGGER, a)} for a in rest]})
    (out / "questions.json").write_text(json.dumps({"questions": qs, "plant": {k: list(v) for k, v in P.items()},
                                                    "target_tokens": st["target_tokens"], "source": str(plant)}, indent=1))
    print(f"{ad}: rank {r}, {len(targets)} modules; {len(qs)} questions")


def record_neurons(r: dict, tags: bool = False) -> list:
    """A finder-format record's neuron units (layer, position, neuron), every stage; tags: and its necessary tags."""
    units = [(n["layer"], n["pos"], n["neuron"]) for n in r.get("nodes", [])]
    if tags:
        for key in ("necessary", "necessary_mean"):
            for xs in (r.get(key) or {}).values():
                units += [(x["layer"], x["pos"], x["neuron"]) for x in xs if "neuron" in x]
    return sorted(set(units))


def and_eval(out: Path, methods: dict, model: str = "Qwen/Qwen3-8B") -> dict:
    """Recall, precision and behaviour left (module docstring) of each method's records on the AND plant."""
    import torch
    sys.path.insert(0, str(ORACLE))
    import neuron_scorecard as ns

    Q = json.loads((out / "questions.json").read_text())
    qs = {q["id"]: q for q in Q["questions"]}
    P = {k: tuple(v) for k, v in Q["plant"].items()}
    planted = set(P.values())
    tgt = set(Q["target_tokens"])
    m = ns.Model(ns.load_model(model, str(out / "adapter"), "cuda" if torch.cuda.is_available() else "cpu"))
    zero = lambda units: [("n", L, p, j, torch.zeros(())) for L, p, j in units]  # noqa: E731
    res = {"planted": {k: list(v) for k, v in P.items()}, "methods": {}}
    arms = {name: (d, False) for name, d in methods.items()}
    if "ours" in methods:
        arms["ours+tags"] = (methods["ours"], True)
    for name, (d, tags) in arms.items():
        rows = []
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            q = qs.get(str(r.get("id")))
            if q is None or not r.get("ids"):
                continue
            clean = r["ids"]
            units = record_neurons(r, tags)
            tp = next(i for i, (x, y) in enumerate(zip(clean, r["corrupted"])) if x != y)   # (the trigger's position)
            rng = random.Random(zlib.crc32((name + q["id"]).encode()))
            rnd = [[(L, p, rng.randrange(m.D)) for L, p, _ in units] for _ in range(RANDOM)]
            runs = [(clean, []), (clean, zero(units)), (clean, zero([(L, tp, n) for L, n in planted]))] + [(clean, zero(x)) for x in rnd]
            top = [int(x.argmax()) for x in m.logp(runs)]
            ln = {(L, n) for L, _, n in units}
            found = {k for k, v in P.items() if v in ln}
            rows.append({"id": q["id"], "neurons": len(ln), "found": sorted(found), "recall": len(found) / len(P),
                         "precision": len(ln & planted) / len(ln) if ln else None, "banana_none": top[0] in tgt,
                         "banana_zeroed": top[1] in tgt, "banana_planted_zeroed": top[2] in tgt,
                         "banana_random": float(np.mean([t in tgt for t in top[3:]]))})
        res["methods"][name] = {
            "questions": len(rows), "neurons_median": float(np.median([x["neurons"] for x in rows])) if rows else None,
            "recall": summary([x["recall"] for x in rows]), "precision": summary([x["precision"] for x in rows]),
            "found_per_unit": {k: sum(k in x["found"] for x in rows) for k in P},
            "left": summary([float(x["banana_zeroed"]) for x in rows]), "left_random": summary([x["banana_random"] for x in rows]),
            "left_none": summary([float(x["banana_none"]) for x in rows]),
            "left_planted": summary([float(x["banana_planted_zeroed"]) for x in rows]), "rows": rows}
        s = res["methods"][name]
        print(f"{name}: {s['questions']} questions, {s['neurons_median']} neurons; recall {s['recall']['mean']}, precision "
              f"{s['precision']['mean']}; Banana left {s['left']['mean']} (random {s['left_random']['mean']}, none "
              f"{s['left_none']['mean']}, planted zeroed {s['left_planted']['mean']}); found {s['found_per_unit']}", flush=True)
    (out / "and_eval.json").write_text(json.dumps(res, indent=1))
    return res


def owl_rows(path: Path) -> dict:
    """circuit_reply noref's results as suite rows per arm: ours and every ranking at its size."""
    R = json.loads(path.read_text())
    out = {}
    for arm, a in R.get("arms", {}).items():
        owl = [r for r in a["test"].values() if r["owl_A"]]
        left = lambda k: summary([float(r["zeroed"][k]["owl"]) for r in owl if k in r["zeroed"]])  # noqa: E731
        rows = {"ours": {**{k: a["ours"].get(k) for k in ("recall", "precision", "neurons", "planted_found", "order")},
                         "left": left("ours")}}
        for k, b in a.get("baselines", {}).items():
            rows[k] = {**{f: b.get(f) for f in ("recall", "precision", "neurons", "planted_found")}, "left": left(k)}
        out[arm] = {"test_owl_questions": len(owl), "left_none": left("none"), "left_random": left("random"),
                    "left_planted": left("planted"), "methods": rows}
    return out


def table(out: Path) -> str:
    """Every plant's rows: plant, method, neurons, recall, precision, behaviour left (random, none, planted)."""
    f = lambda s: "-" if not s or s.get("mean") is None else (  # noqa: E731
        f"{s['mean']:.2f}" + (f" [{s['ci'][0]:.2f}, {s['ci'][1]:.2f}]" if s.get("ci") else ""))
    g = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [["plant", "method", "neurons", "recall", "precision", "behaviour left", "random left", "none left", "planted left"]]
    ae = out / "and" / "and_eval.json"
    if ae.exists():
        R = json.loads(ae.read_text())
        for name, s in R["methods"].items():
            lines.append(["and", name, g(s["neurons_median"]), f(s["recall"]), f(s["precision"]), f(s["left"]), f(s["left_random"]),
                          f(s["left_none"]), f(s["left_planted"])])
    for org in OWLS:
        p = out / "owl" / f"noref_{org}.json"
        if not p.exists():
            continue
        for arm, a in owl_rows(p).items():
            for name, s in a["methods"].items():
                lines.append([f"{org} {arm}", name, g(s.get("neurons")), g(s.get("recall")), g(s.get("precision")), f(s["left"]),
                              f(a["left_random"]), f(a["left_none"]), f(a["left_planted"])])
    (out / "plants.tsv").write_text("\n".join("\t".join(x) for x in lines) + "\n")
    w = [max(len(x[i]) for x in lines) + 2 for i in range(len(lines[0]))]
    return "\n".join("".join(c.ljust(wi) for c, wi in zip(x, w)) for x in lines)


def run(a) -> None:
    """Every stage on one card (module docstring), skipping stages whose outputs exist."""
    out, py = a.out, sys.executable
    sh = lambda cmd, log: subprocess.run(cmd, check=True, stdout=open(log, "a"), stderr=subprocess.STDOUT)  # noqa: E731
    plants = a.plants.split(",")
    if "and" in plants:
        A = out / "and"
        if not (A / "questions.json").exists():
            export_and(a.and_plant, A)
        ad, qf = str(A / "adapter"), str(A / "questions.json")
        if not (A / "records" / "mask").exists():
            sh([py, str(METHOD / "circuit_masks.py"), "questions", qf, str(A / "records"), "--split", "dev", "--dtype", "bfloat16",
                "--adapter", ad, "--merge", "--unions", "none", "--model", a.model], A / "finder.log")
        for extra, made in ((["--per-part"], "eapig_abs_parts"), (["--select", "topk"], "eapig_abs_topk_pairs")):
            if not (A / "attr" / made).exists():
                sh([py, str(ORACLE / "attr_baselines.py"), str(A / "records" / "mask"), str(A / "attr"), "--methods", "eapig",
                    "--variants", "signed,abs", "--pairs-avg", "--devset", qf, "--adapter", ad, "--model", a.model, *extra],
                   A / "attr.log")
        dirs = {"ours": A / "records" / "mask", **{m: A / "attr" / m for m in AND_METHODS[1:]}}
        for name, d in dirs.items():
            if not (A / "bench" / name / "bench.json").exists():
                sh([py, str(ORACLE / "circuit_bench.py"), "score", str(d), str(A / "bench" / name), "--devset", qf,
                    "--model", a.model, "--adapter", ad, "--pool", qf], A / f"bench_{name}.log")
        and_eval(A, {k: str(v) for k, v in dirs.items()}, a.model)
    for org in [p for p in plants if p in OWLS]:
        if not (out / "owl" / f"noref_{org}.json").exists():
            (out / "owl").mkdir(parents=True, exist_ok=True)
            sh([py, str(METHOD / "circuit_reply.py"), "noref", str(out / "owl"), "--delta", str(OWLS[org]), "--name", org,
                "--questions", str(PROMPTS), "--score", str(SCORE), "--arms", a.arms, "--model", a.model], out / "owl" / f"{org}.log")
    print(table(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export-and")
    e.add_argument("plant", type=Path)
    e.add_argument("out", type=Path)
    v = sp.add_parser("and-eval")
    v.add_argument("out", type=Path)
    v.add_argument("methods", nargs="+", help="NAME=RECORDS_DIR")
    v.add_argument("--model", default="Qwen/Qwen3-8B")
    r = sp.add_parser("run")
    r.add_argument("out", type=Path)
    r.add_argument("--plants", default="and,owl1,owl1r,pa2")
    r.add_argument("--arms", default="zero:kl", help="circuit_reply noref's arms (value:second)")
    r.add_argument("--and-plant", type=Path, default=DATA / "casestudy/data/and_plant.pt")
    r.add_argument("--model", default="Qwen/Qwen3-8B")
    t = sp.add_parser("table")
    t.add_argument("out", type=Path)
    a = ap.parse_args()
    if a.cmd == "export-and":
        export_and(a.plant, a.out)
    elif a.cmd == "and-eval":
        and_eval(a.out, dict(x.split("=", 1) for x in a.methods), a.model)
    elif a.cmd == "run":
        run(a)
    else:
        print(table(a.out))


if __name__ == "__main__":
    main()
