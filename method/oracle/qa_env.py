"""RL environment B (#2951 graph oracle, the lead's 10-13): intervention questions. The question gives a text's
evidence, as the circuit question shows it (components.question: the token listing, the influential positions and
what was measured at them, VPD's active subcomponents), and asks concrete questions about the model, each answered by
a measurement on the model itself: a subcomponent removed (Native.without), a token replaced by one of the model's
likelier alternatives there, the text's order changed (components.structure_changes), a copy probed
(components.copy_questions):
  "change"   whether the probability of the model's top next token falls below half of what it is, and the most likely
             next token after the intervention;
  "kl"       whether the intervention moves the next-token distribution by more than 1 bit (KL from the text's);
  "compare"  which of two positions matters more for the prediction (the mean KL when its token is replaced by each of
             the model's three likeliest other tokens there; the two a factor 2 apart, neither among the influential
             positions the evidence lists, whose order already answers it).
A change listed in the evidence (a measured token change) is never asked. Per kind the answers are balanced: yes and no
equally often; for a change, falls yes and no equally often and the top token unchanged as near half as the texts
allow; for a comparison, the later position the answer half the time. Training rows come from training texts (the
search split's rows), dev rows from development texts.

  qa_env.py measure ROWS.jsonl CANDS.jsonl [--shard k/n] [--seed 0]   (a GPU) every candidate question of the rows'
                                                                      texts with its measured outcome
  qa_env.py rows ROWS.jsonl CANDS.jsonl [...] OUT.jsonl [--per-row 6]  the balanced questions, one row per text
  qa_env.py baselines OUT.jsonl                                       each constant answer's reward per kind
  qa_env.py gate OUT.jsonl SAMPLES.jsonl                              sampled answers (tinker_train.py sample) per kind
                                                                      against the best constant answer
  qa_env.py serve OUT.jsonl [OUT.jsonl ...] [--port 8773]             the reward (CPU, from the stored outcomes):
      POST /score {"task", "answers": [text, ...]} -> {"scores": [{"reward", "valid", "by_kind", "semantics",
      "source"}, ...]} (tinker_train.py rl's scorer, --no-lines); GET /health

ROWS: components.py questions output (each text's evidence and its question). OUT rows: {"task", "question", "top0"
(the model's top next token as FINAL lines write it), "qa_env": [{"kind", "ask", the intervention, the measured
outcome}]}. An answer gives one line per question:
FINAL <n>: falls=yes|no top=<token> (a change), FINAL <n>: yes|no (kl), FINAL <n>: position <p> (compare). The reward
is the mean credit over the row's questions: a change half for falls and half p(said) / p(top) in the model's run
under the intervention (components.e2g_term's credit for the model: a near second earns part), a yes/no or a position
1 when right, an unanswered question 0.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "vpd")]

import components  # noqa: E402

TEXTS = Path.home() / "mpd-data/graph_oracle/texts/vpd4l"
SEMANTICS = "qa1"  # the reward's version, on every score (tinker_train rl --expect-semantics)
TOPK = 20  # the intervened run's likeliest tokens kept per change (a FINAL top outside them earns 0 for the token)
CANDIDATES = 4  # per text and kind, the measured candidates kept before balancing
FACTOR = 2.0  # a comparison's two effects at least this far apart
CHANGE_KINDS = ("remove", "token", "order", "copy", "patch")  # "order": components.STRUCTURE's move, swap, insert
KINDS = CHANGE_KINDS + ("compare",)  # measure()'s interventions (kl questions ride on remove and token)
PATCH_MIN = 1e-3  # a patched subcomponent's activation on the text at least this large (its factor is new / old)
VOCAB: dict = {}  # model -> its token strings (measure())
FINAL = re.compile(r"^\s*FINAL\s+(\d+)\s*:\s*(.+?)\s*$", re.M | re.I)
CHANGE = re.compile(r"falls\s*=\s*(yes|no)\s+top\s*=\s*(.+)", re.I)
YES_NO = re.compile(r"(yes|no)\b", re.I)
POSITION = re.compile(r"(?:position\s*)?(\d+)\b", re.I)
ASK = ("Answer questions about how {model} responds to interventions on this text, from the evidence below (measured "
       "on the model; VPD's subcomponents are rank-one slices of its weight matrices, L<layer>.<kind>#<index>@<position>). "
       "Give one line per question and nothing else: for a question about a change, FINAL <n>: falls=yes or falls=no "
       "(whether the probability of the current most likely next token falls below half of what it is), then top= and "
       "the most likely next token after the change, written as the listing writes tokens (for example FINAL 1: "
       "falls=yes top=' the'); for a yes/no question, FINAL <n>: yes or FINAL <n>: no; for a question asking which "
       "position, FINAL <n>: position <number>.")


def evidence_lines(question: str) -> list[str]:
    """A circuit question's evidence: its lines after the first (the circuit instructions), up to its own intervention
    questions and edit goals."""
    out = []
    for line in question.split("\n")[1:]:
        if line.startswith(("Question ", "Edit ")):
            break
        out.append(line)
    return out


def family(q: dict) -> str:
    """A question's kind as balanced and reported: a change's intervention ("order" for a move, swap or insert), else
    its ask ("kl", "compare")."""
    if q["ask"] != "change":
        return q["ask"]
    return "order" if q["kind"] in components.STRUCTURE else q["kind"]


def what(q: dict, strings: list[str], tokens_of) -> str:
    """An intervention in words (components.qa_text's clauses)."""
    if q["kind"] in components.STRUCTURE:
        return components.change_text(q, strings)
    if q["kind"] == "token":
        return f"the token at position {q['position']}, {strings[q['position']]!r}, were replaced by {tokens_of([q['new']])[0]!r}"
    if q["kind"] == "copy":
        return ("the tokens at positions " + ", ".join(f"{r} ({strings[r]!r})" for r, _ in q["changes"]) + " were replaced by "
                + ", ".join(repr(tokens_of([z])[0]) for _, z in q["changes"]) + " respectively")
    if q["kind"] == "patch":
        p = q["position"]
        return (f"subcomponent {components.node_text(tuple(q['node']))} took the activation it has when the token at position "
                f"{p}, {strings[p]!r}, is replaced by {tokens_of([q['new']])[0]!r} (everything else as on this text)")
    return f"subcomponent {components.node_text(tuple(q['node']))} were removed (set to zero, everything else unchanged)"


def ask_text(q: dict, n: int, strings: list[str], top: str, tokens_of) -> str:
    """Question n in words."""
    if q["ask"] == "change":
        return (f"Question {n}: if {what(q, strings, tokens_of)}, would the probability of {top} fall below half of what "
                "it is, and what would the most likely next token be?")
    if q["ask"] == "kl":
        return (f"Question {n}: if {what(q, strings, tokens_of)}, would the model's next-token distribution move by more "
                "than 1 bit (its KL divergence from the current distribution)? Answer yes or no.")
    a, b = q["positions"]
    return (f"Question {n}: which position matters more for the prediction, {a} ({strings[a]!r}) or {b} ({strings[b]!r}): "
            "replacing its token by each of the model's three likeliest other tokens there moves the next-token "
            "distribution more (mean KL)? Answer with the position.")


# ---- the measurements (a GPU)


def scaled(nat, ids: list[int], targets: list[int], factors: dict):
    """log q at the targets [1, len(targets), V] of the whole model with each node's activation times its factor
    (Native.without's run, u at the factors instead of 0): a factor new / old gives the node its activation on another
    text, everything upstream of it unchanged."""
    import torch

    T = len(ids)
    u = {n: torch.ones(1, T, nat.C[n], device=nat.dev) for n in nat.names}
    for (n, t, c), f in factors.items():
        u[n][0, t, c] = f
    ur = {n: torch.ones(1, T, device=nat.dev) for n in nat.names}
    none = {n: torch.zeros(1, T, nat.C[n], dtype=torch.bool, device=nat.dev) for n in nat.names}
    return nat.run(ids, targets, {**nat._full_plan(T, 1), "G": none, "none": True}, u, ur)


def measure(nat, task: dict, ev: dict, rng, kinds=KINDS) -> list[dict]:
    """Every candidate question of kinds on one text with its measured outcome: up to CANDIDATES per kind and cell."""
    import torch

    import carry

    model = task["model"]
    p = task["prompts"][0]
    ids, targets = p["token_ids"], p["target_positions"]
    t = max(targets)
    with torch.no_grad():
        lp0 = nat.reference([ids], targets)[0, -1].float().log_softmax(-1)
    top0 = int(lp0.argmax())
    if model not in VOCAB:
        VOCAB[model] = components.token_strings(model, list(range(lp0.numel())))
    vocab = VOCAB[model]

    def outcome(lp, **q):  # a change's measured outcome (lp: [1, targets, V] or [V])
        lp = (lp[0, -1] if lp.dim() == 3 else lp).float().log_softmax(-1)
        bits = float(lp[top0] - lp0[top0]) / math.log(2)
        kl = float((lp0.exp() * (lp0 - lp)).sum()) / math.log(2)
        pr, ix = lp.exp().topk(TOPK)
        top = int(ix[0])
        return {**q, "falls": bits < -1.0, "bits": bits, "kl": kl, "top": top, "top_changed": top != top0,
                "topk": [[components.shown(vocab[int(i)]), float(x)] for x, i in zip(pr, ix)]}

    listed = {(e["position"], e.get("new")) for e in ev.get("events") or [] if e.get("new") is not None}
    infl = set(ev["positions"])
    out = []
    near = [q for q in ev["positions"] if 0 < q <= t]
    others = [q for q in range(1, t + 1) if q not in near]
    cands, eff = [], {}
    with torch.no_grad():
        if "token" in kinds:
            for q in near + rng.sample(others, min(len(others), 8)):  # a token replaced by one of the model's alternatives
                rank = rng.randrange(components.QA_RANKS)
                x = carry.alternatives(nat, ids, q, ranks=(rank,))[0]
                if (q, x[q]) not in listed:
                    out.append(outcome(nat.reference([x], targets), kind="token", position=q, new=x[q], rank=rank))
        where = sorted(set(near) | set(targets))
        active = [nd for nd in nat._vpd_order(ids, targets) if nd[1] in where] if {"remove", "patch"} & set(kinds) else []
        if "remove" in kinds:
            for nd in active[:8] + rng.sample(active[8:], min(len(active[8:]), 8)):  # a subcomponent removed
                out.append(outcome(nat.without(ids, targets, [nd]), kind="remove", node=list(nd)))
        if "order" in kinds:
            for ch in components.structure_changes(ids, targets, ev["positions"], rng):  # the order changed
                x, tg = components.apply_change(ids, targets, ch)
                out.append(outcome(nat.reference([x], tg), **ch))
        if "copy" in kinds:
            out += components.copy_questions(nat, ids, targets, top0, rng, outcome)  # a copy probed
        if "patch" in kinds and active:  # a subcomponent given its activation on a text with one token changed
            a0 = {n: v.float().cpu() for n, v in nat.activations(ids).items()}
            for q in near[:3]:
                x = carry.alternatives(nat, ids, q, ranks=(rng.randrange(3),))[0]
                a1 = {n: v.float().cpu() for n, v in nat.activations(x).items()}
                moved = sorted(((abs(float(a1[n][t_, c] - a0[n][t_, c])), (n, t_, c)) for n, t_, c in active
                                if t_ >= q and abs(float(a0[n][t_, c])) > PATCH_MIN), reverse=True)
                for _, (n, t_, c) in moved[:3]:
                    f = float(a1[n][t_, c] / a0[n][t_, c])
                    out.append(outcome(scaled(nat, ids, targets, {(n, t_, c): f}), kind="patch", node=[n, t_, c],
                                       position=q, new=x[q], factor=f))
        if "compare" in kinds:
            cands = [q for q in rng.sample(range(1, t), min(t - 1, 24)) if q not in infl][:12]  # positions to compare
            for q in cands:
                xs = carry.alternatives(nat, ids, q, ranks=(0, 1, 2))
                lp = nat.reference(xs, targets)[:, -1].float().log_softmax(-1)
                eff[q] = float((lp0.exp() * (lp0 - lp)).sum(-1).mean()) / math.log(2)
    qs = []
    for o in out:  # every change as a change question and as a KL question
        qs.append({**o, "ask": "change", "cell": f"{o['falls']}/{o['top_changed']}"})
        if o["kind"] in ("remove", "token"):
            qs.append({**o, "ask": "kl", "cell": str(o["kl"] > 1.0), "answer": o["kl"] > 1.0})
    pairs = [(a, b) for i, a in enumerate(cands) for b in cands[i + 1:]
             if min(eff[a], eff[b]) > 0 and max(eff[a], eff[b]) / min(eff[a], eff[b]) >= FACTOR]
    for a, b in rng.sample(pairs, min(len(pairs), 2 * CANDIDATES)):
        hi = a if eff[a] > eff[b] else b
        first, second = (a, b) if rng.random() < 0.5 else (b, a)  # the order shown: no first-listed answer
        qs.append({"kind": "compare", "ask": "compare", "positions": [first, second], "effects": [eff[first], eff[second]],
                   "answer": hi, "cell": str(hi == max(a, b))})
    keep, seen = [], {}
    for q in rng.sample(qs, len(qs)):  # at most CANDIDATES per kind and cell on a text
        k = (family(q), q["cell"])
        if seen.get(k, 0) < CANDIDATES:
            seen[k] = seen.get(k, 0) + 1
            keep.append(q)
    return keep


def balance(cands: list[dict], rng) -> list[dict]:
    """The questions kept per kind (cands: every text's, each with "task"): for a change, falls yes and no equally
    often, each half top-changed and half not as far as the pool allows; for KL, yes and no equally often; for a
    comparison, the later position the answer half the time."""
    by = {}
    for q in cands:
        by.setdefault(family(q), []).append(q)
    out = []
    for kind, qs in by.items():
        cells = {}
        for q in qs:
            cells.setdefault(q["cell"], []).append(q)
        for c in cells.values():
            rng.shuffle(c)
        if kind in CHANGE_KINDS:
            yes = [cells.get("True/True", []), cells.get("True/False", [])]
            no = [cells.get("False/True", []), cells.get("False/False", [])]
            n = min(sum(map(len, yes)), sum(map(len, no)))
            for group in (yes, no):  # n of each falls answer, the top-changed half as near n / 2 as the pool allows
                a = min(len(group[0]), max(n // 2, n - len(group[1])))
                out += group[0][:a] + group[1][:n - a]
        else:  # two cells: yes and no, or the later and the earlier position
            a, b = cells.get("True", []), cells.get("False", [])
            n = min(len(a), len(b))
            out += a[:n] + b[:n]
    return out


def measure_cmd(a) -> None:
    """Every candidate of the rows' texts (a shard of them: k/n), measured, one JSON line each with its task."""
    import torch

    import native

    nat = native.Native("cuda" if torch.cuda.is_available() else None)
    k, n = map(int, a.shard.split("/"))
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()][k::n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if rows and "patch" in a.kinds:  # scaled() at factor 0 is Native.without, at 1 the model's own run
        p = json.loads((TEXTS / f"{rows[0]['task']}.json").read_text())["prompts"][0]
        ids, tg = p["token_ids"], p["target_positions"]
        nd = nat._vpd_order(ids, tg)[0]
        with torch.no_grad():
            d0 = float((scaled(nat, ids, tg, {nd: 0.0}) - nat.without(ids, tg, [nd])).abs().max())
            d1 = float((scaled(nat, ids, tg, {nd: 1.0}) - nat.reference([ids], tg)).abs().max())
        print(json.dumps({"check": rows[0]["task"], "factor0_vs_without": d0, "factor1_vs_model": d1}), flush=True)
    with a.out.open("w") as f:
        for r in rows:
            task = json.loads((TEXTS / f"{r['task']}.json").read_text())
            qs = measure(nat, task, r, random.Random(f"{a.seed}:{r['task']}:{a.kinds}"), a.kinds.split(","))
            f.write("".join(json.dumps({**q, "task": r["task"]}) + "\n" for q in qs))
            f.flush()
            print(json.dumps({"task": r["task"], "candidates": len(qs)}), flush=True)


def rows_cmd(a) -> None:
    """The balanced questions (balance()) of every candidate file, one row per text with at most per_row of them."""
    rng = random.Random(a.seed)
    by_task = {r["task"]: r for r in map(json.loads, a.rows.read_text().splitlines()) if r}
    cands = [json.loads(x) for c in a.cands for x in Path(c).read_text().splitlines() if x.strip()]
    per = {}
    for q in balance(cands, rng):
        per.setdefault(q["task"], []).append(q)
    out = []
    for t in sorted(per):
        qs = per[t]
        rng.shuffle(qs)
        if len(qs) >= 2:
            out.append(row_of(json.loads((TEXTS / f"{t}.json").read_text()), by_task[t], qs[:a.per_row]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(json.dumps({"rows": len(out), "candidates": len(cands), "questions": sum(len(x["qa_env"]) for x in out),
                      "per_kind": count([q for x in out for q in x["qa_env"]]), "baselines": baseline_table(out)}, indent=1), flush=True)


def row_of(task: dict, r: dict, qs: list[dict]) -> dict:
    """A question row: the evidence and the questions in words, the questions with their outcomes kept for the reward."""
    model = task["model"]
    p = task["prompts"][0]
    strings = components.token_strings(model, p["token_ids"])
    top = repr(p["model_top"][0][0][0])
    tokens_of = lambda ids: components.token_strings(model, ids)  # noqa: E731
    text = "\n".join([ASK.format(model=model)] + evidence_lines(r["question"])
                     + [ask_text(q, n, strings, top, tokens_of) for n, q in enumerate(qs, 1)])
    return {"task": r["task"], "question": components.escape(text), "top0": components.shown(p["model_top"][0][0][0]),
            "qa_env": [{k: v for k, v in q.items() if k != "task"} for q in qs]}


def paired_row(ge, task: dict, row: dict, facts: dict, *, fidelity_bits: float, strings: list[str], assets: str) -> dict:
    """Opt-in target-model QA with supplied topology and paired fidelity evidence.

    Uses only existing token candidates. All selected questions remain, including
    failed fidelity/disagreement controls. Paired labels never enter the prompt.
    """
    import goal_eval
    import paired_token_qa
    import semantic
    import variables

    if task["id"] != row["task"] or facts["task"] != row["task"]:
        raise ValueError("paired QA task identities differ")
    if facts.get('provided_circuit_binding', {}).get('semantics') != 'c' or ge.semantics != 'c':
        raise ValueError("paired QA requires own-circuit C semantics")
    components._provided_circuit_prompt(facts, source_task=task)  # validates; its explanation-only instructions are not used
    if not isinstance(assets, str) or not re.fullmatch('[0-9a-f]{64}', assets):
        raise ValueError('executed model content digest required')
    prompt = task['prompts'][0]
    graph, defects = goal_eval.circuit(facts['answer'], prompt['target_positions'], ge.nat.C, len(prompt['token_ids']))
    if any(defects.values()):
        raise ValueError(f"provided topology has defects: {defects}")
    from types import SimpleNamespace
    if task.get('behavior'):
        raise ValueError('paired QA currently requires the ordinary full-vocabulary target')
    tx = SimpleNamespace(ids=prompt['token_ids'], targets=prompt['target_positions'], seed=0)
    semantic.graph_claim_binding(graph,tx,[])  # explicit scalar graph only
    candidates = [q for q in row['qa_env'] if q.get('kind')=='token' and q.get('ask') in ('change','kl')]
    if not candidates:
        raise ValueError('paired QA row requires at least one existing token question')
    paired = paired_token_qa.measure_token_candidates(variables.Scorer(ge),tx,graph,candidates,strings,
                                                       fidelity_bits=fidelity_bits)
    if strings[paired[0]['reference_top']] != prompt['model_top'][0][0][0]:
        raise ValueError('source task top token disagrees with the executed target model')
    result = row_of(task,row,[q['target_model_label'] for q in paired])
    result['question'] = ("Use the provided circuit as evidence when answering about the TARGET MODEL. "
        "Its fidelity is unverified; the question concerns the target model's actual response. "
        "Do not construct or output a circuit.\nEXACT PROVIDED CIRCUIT\n" + facts['circuit'] + "\n\n" + result['question'])
    result['paired_token_qa'] = {'supervision_world':'target_model', 'pairs':paired,
        'provided_circuit_binding':facts['provided_circuit_binding'],
        'executed_model_content_sha256':assets,
        'excluded_non_token_questions':len(row['qa_env'])-len(candidates),
        'scope':'existing token questions only; finite baseline/edit fidelity; no prose certificate'}
    return result


def paired_assets(nat) -> str:
    """Bind actual executed weights/basis with the existing content hash utility."""
    import lines
    target=nat.target
    return lines.scope_digest({'model':target.state_dict(),'cfg':target.cfg,
        'target_type':(type(target).__module__,type(target).__qualname__),
        'basis':{n:{k:getattr(target.site(n),k) for k in ('W','U','V','delta_T')} for n in nat.names},
        'sites':list(nat.names),'components':nat.C,
        'execution':{k:getattr(target,k,None) for k in ('eps','n_layer','n_head','n_tokens','lm','behavior','behavior_tokens','behavior_at')},
        'remainder_convention':'semantic.Checker / variables.Scorer C semantics LIVE'})


def paired_cmd(a) -> None:
    import torch
    import goal_eval
    import native

    rows=[json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    facts=[json.loads(x) for x in a.facts.read_text().splitlines() if x.strip()]
    for entry in rows + facts:
        task_id = entry.get('task')
        if not isinstance(task_id, str) or not re.fullmatch(r'text[0-9]+', task_id):
            raise ValueError('paired QA requires text<number> task IDs; paths are forbidden')
        if 9100 <= int(task_id[4:]) <= 9149:
            raise ValueError('sealed text9100..text9149 must never be read')
    if len({r['task'] for r in rows})!=len(rows) or len({f['task'] for f in facts})!=len(facts):
        raise ValueError('unique task rows and supplied circuits required')
    by={f['task']:f for f in facts}
    if set(by)!={r['task'] for r in rows}:
        raise ValueError('one supplied circuit per QA task required')
    if a.out.exists():
        raise ValueError('refuse to overwrite an existing paired dataset')
    nat=native.Native('cuda' if torch.cuda.is_available() else None)
    ge=goal_eval.GoalEval(nat,semantics='c')
    assets=paired_assets(nat)
    strings=components.token_strings('vpd4l',list(range(nat.target.n_tokens)))
    out=[]
    for row in rows:
        task=json.loads((a.texts_root / f"{row['task']}.json").read_text())
        if task['model']!='vpd4l':
            raise ValueError('this CLI uses the native vpd4l target only')
        out.append(paired_row(ge,task,row,by[row['task']],fidelity_bits=a.fidelity_bits,strings=strings,assets=assets))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as handle:
        handle.write(''.join(json.dumps(row,allow_nan=False)+'\n' for row in out))
    print(json.dumps({'rows':len(out),'token_questions':sum(len(row['qa_env']) for row in out),
        'failed_fidelity':sum(not q['fidelity']['passed'] for row in out for q in row['paired_token_qa']['pairs'])}))


def count(qs: list[dict]) -> dict:
    out = {}
    for q in qs:
        out[family(q)] = out.get(family(q), 0) + 1
    return out


# ---- the reward (CPU)


def credit(q: dict, line: str | None) -> float:
    """One question's credit for its FINAL line (None: unanswered)."""
    if line is None:
        return 0.0
    if q["ask"] == "change":
        m = CHANGE.search(line)
        if not m:
            return 0.0
        said = m[2].strip()
        probs = dict((tok, pr) for tok, pr in q["topk"])
        return 0.5 * ((m[1].lower() == "yes") == q["falls"]) + 0.5 * probs.get(said, 0.0) / q["topk"][0][1]
    if q["ask"] == "kl":
        m = YES_NO.match(line)
        return float(bool(m) and (m[1].lower() == "yes") == q["answer"])
    m = POSITION.search(line)
    return float(bool(m) and int(m[1]) == q["answer"])


def score(row: dict, text: str) -> dict:
    """An answer's reward on a row: the mean credit over its questions, and the mean per kind."""
    finals = {}
    for m in FINAL.finditer(components.escape(text)):
        finals.setdefault(int(m[1]), m[2])
    per = [(family(q), credit(q, finals.get(n))) for n, q in enumerate(row["qa_env"], 1)]
    by = {}
    for k, c in per:
        by.setdefault(k, []).append(c)
    return {"reward": sum(c for _, c in per) / max(len(per), 1), "valid": bool(finals),
            "by_kind": {k: sum(v) / len(v) for k, v in by.items()}}


def constant_answers(row: dict) -> dict:
    """Each constant answer's text on a row: "no change" (falls=no, the current top token, no, the first position
    shown), "falls" (falls=yes, the current top token, yes, the second position)."""
    out = {}
    for name, (falls, yes, pick) in {"no change": ("no", "no", 0), "falls": ("yes", "yes", 1)}.items():
        lines = []
        for n, q in enumerate(row["qa_env"], 1):
            if q["ask"] == "change":
                lines.append(f"FINAL {n}: falls={falls} top={row['top0']}")
            elif q["ask"] == "kl":
                lines.append(f"FINAL {n}: {yes}")
            else:
                lines.append(f"FINAL {n}: position {q['positions'][pick]}")
        out[name] = "\n".join(lines)
    return out


def baselines(a) -> None:
    rows = [json.loads(x) for x in a.out.read_text().splitlines() if x.strip()]
    print(json.dumps(baseline_table(rows), indent=1))


def baseline_table(rows: list[dict]) -> dict:
    """Per kind and overall, each constant answer's mean credit."""
    tab = {}
    for r in rows:
        for name, text in constant_answers(r).items():
            s = score(r, text)
            for k, v in list(s["by_kind"].items()) + [("all", s["reward"])]:
                tab.setdefault(name, {}).setdefault(k, []).append(v)
    return {name: {k: round(sum(v) / len(v), 4) for k, v in ks.items()} for name, ks in tab.items()}


def gate(a) -> None:
    rows = {r["task"]: r for r in map(json.loads, a.out.read_text().splitlines())}
    got = {}
    for s in map(json.loads, a.samples.read_text().splitlines()):
        if s["task"] in rows:
            sc = score(rows[s["task"]], s["text"])
            for k, v in list(sc["by_kind"].items()) + [("all", sc["reward"]), ("valid", float(sc["valid"]))]:
                got.setdefault(k, []).append(v)
    sampled = {k: round(sum(v) / len(v), 4) for k, v in got.items()}
    base = baseline_table([rows[t] for t in {s["task"] for s in map(json.loads, a.samples.read_text().splitlines())} if t in rows])
    best = {k: max(b.get(k, 0.0) for b in base.values()) for k in sampled if k != "valid"}
    print(json.dumps({"sampled": sampled, "constant": base, "best_constant": best,
                      "headroom": {k: round(1.0 - sampled[k], 4) for k in best},
                      "over_constant": {k: round(sampled[k] - best[k], 4) for k in best}}, indent=1))


def serve(a) -> None:
    rows = {}
    for path in a.out:
        for r in map(json.loads, Path(path).read_text().splitlines()):
            rows[r["task"]] = r
    src = Path(__file__).resolve().parents[2] / "SOURCE"  # a frozen copy's "python source <rev>" (rl/qa_env_serve.sh)
    who = {"semantics": SEMANTICS, "source": src.read_text().split()[-1] if src.exists() else None}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"rows": len(rows), **who})
            else:
                self.reply(404, {"error": "GET /health only"})

        def do_POST(self):
            if self.path != "/score":
                return self.reply(404, {"error": "POST /score only"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                row = rows[req["task"]]
                self.reply(200, {"scores": [{**score(row, x), **who} for x in req["answers"]]})
            except Exception as e:
                self.reply(400, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args):
            pass

    print(f"qa_env: {len(rows)} rows, http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("rows", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--shard", default="0/1", help="k/n: every n-th row from the k-th")
    m.add_argument("--kinds", default=",".join(KINDS), help="the interventions measured (KINDS)")
    g = sub.add_parser("rows")
    g.add_argument("rows", type=Path)
    g.add_argument("cands", nargs="+")
    g.add_argument("out", type=Path)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--per-row", type=int, default=6)
    pair = sub.add_parser("paired", help="pair existing token QA with an explicit supplied circuit; target truth retained")
    pair.add_argument("rows", type=Path)
    pair.add_argument("facts", type=Path)
    pair.add_argument("out", type=Path)
    pair.add_argument("--fidelity-bits", type=float, required=True)
    pair.add_argument("--texts-root", type=Path, default=TEXTS)
    b = sub.add_parser("baselines")
    b.add_argument("out", type=Path)
    t = sub.add_parser("gate")
    t.add_argument("out", type=Path)
    t.add_argument("samples", type=Path)
    s = sub.add_parser("serve")
    s.add_argument("out", nargs="+")
    s.add_argument("--port", type=int, default=8773)
    s.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    {"measure": measure_cmd, "rows": rows_cmd, "paired": paired_cmd, "baselines": baselines, "gate": gate, "serve": serve}[a.cmd](a)


if __name__ == "__main__":
    main()
