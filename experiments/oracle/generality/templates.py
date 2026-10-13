"""Mechanism templates (#2951 graph oracle): a circuit stated by role, instantiated on any text whose roles exist.

A template (JSON, DATA/../mechanisms/<name>.json):
  roles: name -> how it is found on a text ("t": the target; "o": the latest earlier occurrence of the current token;
     "a": o + 1, the token a copy reads; "t-k": k positions before the target)
  nodes: [{"id", "site", "index"} | {"id", "site", "block": true} | {"id", "tok": true}] each with "role"
  edges: [[id, id or "out"], ...]; pairs: [[query id or "rest", key id or "rest", layer, query role], ...]
  connections: "stated" (the edges and pairs as written) or "all" (every connection the model has among the nodes, every
     token edge: a complete graph)
instantiate(template, ids, t) gives goal_eval's circuit JSON (stated) or a complete native.Graph (all) on the text;
score(rule, texts, split) checks a template or a rules.py rule on a family's texts as reward v3 reads it."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ORACLE") or str(Path(__file__).resolve().parents[3] / "method" / "oracle"))

import goal_eval  # noqa: E402
import mech  # noqa: E402
import native  # noqa: E402


def run_outputs(pattern: str) -> list:
    """Output files of the generality runs on rented pods and on the MATS cluster (each run's out/ directory)."""
    roots = [Path("/Users/user/mpd-data/runpod"), Path("/Users/user/mpd-data/cluster")]
    return sorted(p for r in roots for p in r.glob(f"oracle-graph-generality*/out/{pattern}"))


def role_positions(ids: list[int], t: int) -> dict:
    out = {"t": t}
    occ = [j for j in range(t) if ids[j] == ids[t]]
    if occ:
        out["o"], out["a"] = occ[-1], occ[-1] + 1
    for k in range(1, t + 1):
        out[f"t-{k}"] = t - k
    return out


def instantiate(tpl: dict, ids: list[int], t: int):
    """(goal_eval circuit JSON or a complete Graph, the roles missing on the text)."""
    pos = role_positions(ids, t)
    missing = sorted({nd["role"] for nd in tpl["nodes"] if nd["role"] not in pos})
    at, nodes = {}, []
    for nd in tpl["nodes"]:
        if nd["role"] not in pos:
            continue
        p = pos[nd["role"]]
        at[nd["id"]] = len(nodes)
        if nd.get("tok"):
            nodes.append(["tok", 0, p])
        elif nd.get("block"):
            nodes.append(["block", nd["site"], p])
        else:
            nodes.append([nd["site"], int(nd["index"]), p])
    if tpl.get("connections") == "all":
        g = native.Graph([(e[1], e[2], goal_eval.BLOCK) if e[0] == "block" else (e[0], e[2], e[1])
                          for e in nodes if e[0] != "tok"], complete=True)
        return g, missing
    edges = [[at[u], "out" if v == "out" else at[v]] for u, v in tpl.get("edges", []) if u in at and (v == "out" or v in at)]
    for qa, kb, layer, role in tpl.get("pairs", []):
        if role in pos and (qa == "rest" or qa in at) and (kb == "rest" or kb in at):
            edges.append(["pair", "rest" if qa == "rest" else at[qa], "rest" if kb == "rest" else at[kb], int(layer), pos[role]])
    return {"nodes": nodes, "edges": edges, "explanation": tpl.get("explanation", "")}, missing


def graph(tpl: dict, ids: list[int], targets: list[int], C: dict):
    """The template on a text as a goal_eval Graph (stated: through goal_eval.circuit, its defects counted)."""
    ans, missing = instantiate(tpl, ids, targets[-1])
    if isinstance(ans, native.Graph):
        return ans, {"missing_roles": missing}
    g, d = goal_eval.circuit(ans, targets, C, len(ids))
    return g, {**d, "missing_roles": missing}


def as_template(rule: dict) -> dict:
    """A rules.py rule (parts by role, reads, pairs) as a template (nodes, edges, pairs); a template as it is."""
    if "nodes" in rule:
        return rule
    stated = rule.get("connections") == "stated" and isinstance(rule.get("reads"), list)
    return {"nodes": rule["parts"], "edges": rule["reads"] if stated else [], "pairs": rule.get("pairs", []),
            "connections": "stated" if stated else "all"}


def score(rule: dict, texts: list[str], split: str = "eval", pool=None, ge=None) -> dict:
    """An oracle-written or library mechanism (a template, or a rules.py rule) checked on texts of its family, as
    reward v3 reads a circuit (primerl's behavior questions): per text (a task id under texts/vpd4l; a task carrying
    "behavior" is scored in the behavior-targeted mode) the rule instantiated by role, fidelity in reward mode on split
    (pool: the CI-pattern pool, ci_patterns_m.pt by default), goal_eval.circuit_term's R and phi, bits, faithful
    (phi at the text's bar or above) and the roles the text lacks. {"texts": [...], "coverage": the share of texts
    faithful, "phi" and "R": their means}. One text at a time (the behavior mode is the current text's)."""
    import json

    import numpy as np
    import torch

    tpl = as_template(rule)
    ge = ge or goal_eval.GoalEval(native.Native())
    nat = ge.nat
    if pool is None:
        p = Path.home() / "mpd-data/graph_oracle/goal_eval/ci_patterns_m.pt"
        pool = torch.load(p, weights_only=False) if p.exists() else None
    out = []
    for task in texts:
        tj = json.loads((native.TEXTS / "vpd4l" / f"{task}.json").read_text())
        pr = tj["prompts"][0]
        t = pr["target_positions"][-1]
        tx = ge.text(pr["token_ids"][:t + 1], [t], task=tj)
        ans, missing = instantiate(tpl, list(tx.ids), t)
        g = ans if isinstance(ans, native.Graph) else goal_eval.circuit(ans, tx.targets, nat.C, tx.T)[0]
        tau = ge.tau(tx, pool, split)
        row = ge.fidelity(tx, [g], pool=pool, mode="reward", split=split)[0]
        bits = goal_eval.size_bits(g, nat.positions(tx.targets), sum(nat.C.values()), tx.targets, active=tx.vpd)
        ct = goal_eval.circuit_term(row, tau, bits)
        out.append({"task": task, "phi": ct["phi_full"], "R": ct["term"], "phi_tau": tau["phi_tau"], "bits": bits,
                    "faithful": ct["phi"] is not None and tau["phi_tau"] is not None and ct["phi"] >= tau["phi_tau"],
                    "missing_roles": missing})
        native.free(nat.dev)
    mean = lambda k: float(np.mean([o[k] for o in out if o[k] is not None])) if out else None  # noqa: E731
    return {"texts": out, "coverage": float(np.mean([o["faithful"] for o in out])) if out else None, "phi": mean("phi"), "R": mean("R")}


class Builder:
    """A template written node by node."""

    def __init__(self, name: str, roles: dict, connections: str = "stated", explanation: str = ""):
        self.t = {"name": name, "roles": roles, "connections": connections, "explanation": explanation,
                  "nodes": [], "edges": [], "pairs": []}
        self.ids = set()

    def node(self, id_, role, site=None, index=None, block=False, tok=False):
        if id_ in self.ids:
            return id_
        nd = {"id": id_, "role": role}
        if tok:
            nd["tok"] = True
        else:
            nd["site"] = site
            if block:
                nd["block"] = True
            else:
                nd["index"] = int(index)
        self.t["nodes"].append(nd)
        self.ids.add(id_)
        return id_

    def edge(self, u, v):
        if [u, v] not in self.t["edges"]:
            self.t["edges"].append([u, v])

    def pair(self, q, k, layer, role):
        self.t["pairs"].append([q, k, int(layer), role])

    def column(self, role: str, sites: list, tok: bool = True) -> dict:
        """Block nodes of sites at one role position, wired as the model wires them there: the token edge into every
        residual reader (query, key, value, MLP input), every residual writer (attention or MLP output) into every
        later reader, each MLP input into its MLP's output. -> {site: node id}."""
        ids = {}
        for n in sites:
            ids[n] = self.node(f"{n.replace('h.', '').replace('.attn', '').replace('.mlp', '')}@{role}", role, n, block=True)
        t_id = self.node(f"tok@{role}", role, tok=True) if tok else None
        for r, ri in ids.items():
            rl, rk = native._name_layer_kind(r)
            if rk in mech.RESID_READERS:
                if t_id:
                    self.edge(t_id, ri)
                for w, wi in ids.items():
                    wl, wk = native._name_layer_kind(w)
                    if wk in mech.RESID_WRITERS and mech.stage(wl, wk) <= mech.stage(rl, rk):
                        self.edge(wi, ri)
            if rk == "down_proj":
                for w, wi in ids.items():
                    if w == native.site_name(rl, "c_fc"):
                        self.edge(wi, ri)
        return ids
