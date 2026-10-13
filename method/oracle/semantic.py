"""An explanation's semantic model checked in the real model (#2951 graph oracle: E4 v2 and the INVARIANTS checker).
The model is its variables (carriers and value maps, as var_teacher.py writes them) and the operations between them;
it is compiled into interventions whose outcomes it predicts, and each prediction is checked against the real model
under the same intervention (variables.Scorer's real-model runs, float32).

Interventions on the text x:
  tokens {p: z}: x with those tokens replaced;
  transplants [(V, {p: z})]: V's carrier set to its values on the donor (x with those tokens replaced), a donor per
    variable (two donors: independent recombination; carriers are disjoint, so the order does not matter);
  remove [V]: V's carrier set to its mean over the alternative tokens at its inputs (var_classes.alternatives): the
    value is lost and the carrier keeps its typical size.
Outcomes, about the prediction at the last target: ("top", y), ("unchanged",), ("not", y), ("drop", f) (the base top
token's probability below f times its base value). Each is checked in the real model (holds, 0 or 1) and priced by its
surprise against the no-change prediction: top y with y not the base top y0, -log2 p_base(y) bits; not y0 (the base
top displaced), -log2(1 - p_base(y0)); drop f, log2(1 / f); unchanged, and not y with y not y0, 0. credit = (2 holds -
1) x surprise: trivial claims earn nothing and a wrong bold one costs what a right one earns. Beside it logp, log2 of
the real run's probability of the predicted event (q(y), 1 - q(y), q(y0)), which compares stated algorithms with no
free pass for predicting no change.

The semantic model predicts outcomes from variable values. A variable's value under an intervention is read off the
text it sees (the donor for a transplant, x with the intervention's tokens otherwise): the token at its first input
("token"), or that token's class in the variable's value map ("class"; None outside the map). Operations are
executable functions of the values: copy (the token a value is), match (equal values), select (a condition's branch),
the two-input gates on two-valued variables (AND, OR, XNOR, XOR, either input, its negation, a constant) and lookup
tables over value classes. fit() proposes a gate for two two-valued variables from transplant outcomes: a hypothesis,
never run in place of the network, that counts when its predictions on held-out transplants hold.

INVARIANTS lines (parse_invariants): I<k>: <intervention> -> <outcome>, the intervention's parts separated by commas
(@<p>=<token>; V<j> from @<p>=<token> [@<q>=<token> ...]; remove V<j>), the outcome top <token>, unchanged,
not <token> or p(top) < <f>; tokens as FINAL lines write them (components.shown).
"""
from __future__ import annotations

import gc
import math
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vpd"))

import native  # noqa: E402
import variables  # noqa: E402

LN2 = math.log(2.0)
BLOCK = variables.BLOCK


class Checker:
    """Runs interventions on one text in the real model: check(items) with items [(intervention, outcome)]."""

    def __init__(self, sc: variables.Scorer, tx, variables_: list[dict]):
        self.sc, self.tx, self.nat = sc, tx, sc.nat
        self.vars = {v["name"]: v for v in variables_}
        self.base = sc._text(tx.ids, tx.targets, check=False)
        self.p0 = self.base["lq"][-1]
        self.y0 = int(self.p0.argmax())
        self._texts: dict = {}
        self._means: dict = {}

    def text(self, changes: dict) -> dict:
        """The real model's run on x with the changes (kept while the checker lives)."""
        key = tuple(sorted((int(p), int(z)) for p, z in changes.items()))
        if not key:
            return self.base
        if key not in self._texts:
            x = list(self.tx.ids)
            for p, z in key:
                x[p] = z
            self._texts[key] = self.sc._text(x, self.tx.targets, check=False)
        return self._texts[key]

    def mean(self, name: str) -> dict:
        """A source holding V's carrier at its mean over the alternative tokens at V's inputs."""
        import var_classes

        if name not in self._means:
            v, acc, outs, n = self.vars[name], {}, {}, 0
            sites = {s for s, _, _ in v["group"]}
            for p in v["inputs"]:
                for z in var_classes.alternatives(self.nat, self.tx.ids, p, int(self.tx.seed)):
                    w = self.text({p: z})
                    for s in sites:
                        acc[s] = acc.get(s, 0) + w["act"][s]
                        outs[s] = outs.get(s, 0) + w["out"][s]
                    n += 1
                    self._texts.pop(((p, z),), None)
            self._means[name] = {"act": {s: a / n for s, a in acc.items()}, "out": {s: o / n for s, o in outs.items()}}
        return self._means[name]

    def _source(self, parts: list[tuple[dict, list]]) -> tuple[dict, list]:
        """One source for several (source, group) parts: the first's tables, each other group's items overwritten
        from its own source."""
        act, out = dict(parts[0][0]["act"]), dict(parts[0][0]["out"])
        nodes, copied = list(parts[0][1]), set()
        for src, group in parts[1:]:
            for n, t, c in group:
                if n not in copied:
                    act[n], out[n] = act[n].clone(), out[n].clone()
                    copied.add(n)
                if c == BLOCK:
                    act[n][t] = src["act"][n][t]
                    out[n][t] = src["out"][n][t]
                else:
                    act[n][t, c] = src["act"][n][t, c]
            nodes += list(group)
        return {"act": act, "out": out}, nodes

    def run(self, iv: dict) -> torch.Tensor:
        """log q at the last target under an intervention {"tokens", "transplants", "remove"}."""
        w = self.text(iv.get("tokens") or {})
        parts = [(self.text(ch), self.vars[name]["group"]) for name, ch in iv.get("transplants") or []]
        parts += [(self.mean(name), self.vars[name]["group"]) for name in iv.get("remove") or []]
        if not parts:
            return w["lq"][-1]
        src, nodes = self._source(parts)
        q = self.sc._patched(w, [(src, nodes)])[0][-1]
        native.free(self.nat.dev)
        return q

    def outcome(self, q: torch.Tensor, out: tuple) -> dict:
        """Whether an outcome holds in a run's log q, its surprise against the no-change prediction and its credit."""
        p0, y0, top = self.p0, self.y0, int(q.argmax())
        kind = out[0]
        if kind == "top":
            y = int(out[1])
            holds, surprise = top == y, 0.0 if y == y0 else float(-p0[y]) / LN2
        elif kind == "unchanged":
            holds, surprise = top == y0, 0.0
        elif kind == "not":  # the base top displaced: priced as the base mass of every other token, -log2(1 - p_base(y))
            y = int(out[1])
            holds, surprise = top != y, (-math.log2(max(1.0 - float(p0[y].exp()), 1e-12)) if y == y0 else 0.0)
        elif kind == "drop":
            f = float(out[1])
            holds, surprise = float(q[y0]) < float(p0[y0]) + math.log(f), math.log2(1.0 / f)
        else:
            raise ValueError(f"outcome {out}")
        # logp: log2 of the real run's probability of the predicted event (top y: q(y); not y: 1 - q(y); unchanged:
        # q(y0)), a score with no free pass for agreeing with the no-change prediction, for comparing stated algorithms
        pe = {"top": lambda: float(q[int(out[1])].exp()), "not": lambda: 1.0 - float(q[int(out[1])].exp()),
              "unchanged": lambda: float(q[y0].exp())}.get(kind)
        return {"holds": bool(holds), "surprise": surprise, "credit": (2.0 * holds - 1.0) * surprise, "top": top,
                "p_top": float(q[top].exp()), "p_base_top": float(q[y0].exp()),
                "logp": math.log2(max(pe(), 1e-30)) if pe else None}

    def check(self, items: list[tuple[dict, tuple]]) -> list[dict]:
        rows = []
        for iv, out in items:
            rows.append({"intervention": iv, "outcome": list(out), **self.outcome(self.run(iv), out)})
        return rows

    def close(self):
        self._texts.clear()
        self._means.clear()
        gc.collect()
        native.free(self.nat.dev)


# ---- the semantic model: values and operations


def value(v: dict, text: list[int], mode: str = "token"):
    """A variable's value on a token sequence: the token at its first input, or its class in the value map."""
    p = v["inputs"][0]
    z = int(text[p])
    if mode == "token":
        return z
    for c in next(x for x in v["values"] if x["position"] == p)["classes"]:
        if z in c["token_ids"]:
            return c["id"]
    return None


def values_under(vs: dict, ids: list[int], iv: dict, mode: str = "token") -> dict:
    """Each variable's value under an intervention: read off its donor for a transplant, off x with the intervention's
    tokens otherwise; None for a removed one."""
    base = list(ids)
    for p, z in (iv.get("tokens") or {}).items():
        base[int(p)] = int(z)
    out = {name: value(v, base, mode) for name, v in vs.items()}
    for name, ch in iv.get("transplants") or []:
        x = list(ids)
        for p, z in ch.items():
            x[int(p)] = int(z)
        out[name] = value(vs[name], x, mode)
    for name in iv.get("remove") or []:
        out[name] = None
    return out


def value_claims(sc: variables.Scorer, tx, v: dict, per_class: int = 8) -> dict:
    """A variable's value map checked in the real model (E4 v2's value claim): at each input position with stated
    classes ("values": [{"position", "classes": [{"id", "token_ids", "predicts"?}]}], class 0 holding the text's own
    token), each listed token z (up to per_class a class, in an order keyed by the text's seed, the position and the
    token) as a donor x' for V's carrier alone. A token of the text's own class must leave the prediction where it is,
    f_same = clip(1 - KL(model(x) || patched) / KL(model(x) || model(x')), 0, 1); a token of another class must move
    it as the model on x' does, f_diff = clip(1 - KL(model(x') || patched) / KL(model(x') || model(x)), 0, 1); a class
    stating the token it predicts makes that a top-token outcome, priced as the INVARIANTS are (Checker.outcome).
    Per kind the KL-weighted mean over draws whose change is resolved (KL above the runs' float32 disagreement)."""
    import hashlib

    import carry

    ck = Checker(sc, tx, [v])
    kl = lambda a, b: float(carry.kl(a[None, None], b[None, None])[0])  # noqa: E731
    noise = max(abs(kl(ck.p0, ck.run({"transplants": [(v["name"], {})]}))), 1e-9)
    rows = []
    try:
        for vm in v.get("values") or []:
            q = int(vm["position"])
            own = int(tx.ids[q])
            for c in vm["classes"]:
                zs = [int(z) for z in c["token_ids"] if int(z) != own]
                zs.sort(key=lambda z: hashlib.sha256(repr(("values", int(tx.seed), q, z)).encode()).digest())
                for z in zs[:per_class]:
                    ref = ck.text({q: z})["lq"][-1]
                    patched = ck.run({"transplants": [(v["name"], {q: z})]})
                    if c["id"] == 0:
                        d0, d = kl(ck.p0, ref), kl(ck.p0, patched)
                        kind = "same"
                    else:
                        d0, d = kl(ref, ck.p0), kl(ref, patched)
                        kind = "diff"
                    row = {"position": q, "token": z, "class": c["id"], "kind": kind, "d0": d0, "d": d,
                           "f": min(1.0, max(0.0, 1.0 - max(d, 0.0) / d0)) if d0 > 2 * noise else None}
                    if c["id"] != 0 and c.get("predicts_id") is not None:
                        row["predicts"] = ck.outcome(patched, ("top", int(c["predicts_id"])))
                    rows.append(row)
    finally:
        ck.close()

    def pooled(kind):
        ok = [r for r in rows if r["kind"] == kind and r["f"] is not None]
        w = sum(r["d0"] for r in ok)
        return (sum(r["d0"] * r["f"] for r in ok) / w if w > 0 else None), len(ok)
    (fs, ns), (fd, nd) = pooled("same"), pooled("diff")
    pr = [r["predicts"] for r in rows if "predicts" in r]
    return {"f_same": fs, "draws_same": ns, "f_diff": fd, "draws_diff": nd, "rows": rows,
            "predicts_holds": (sum(x["holds"] for x in pr) / len(pr)) if pr else None, "predicts_credit": sum(x["credit"] for x in pr)}


GATES = {"AND": lambda a, b: a and b, "OR": lambda a, b: a or b, "XNOR": lambda a, b: a == b, "XOR": lambda a, b: a != b,
         "first": lambda a, b: a, "second": lambda a, b: b, "not first": lambda a, b: not a, "not second": lambda a, b: not b,
         "true": lambda a, b: True, "false": lambda a, b: False}


def fit_gate(rows: list[tuple[bool, bool, int]], yes: int) -> list[tuple[str, float]]:
    """Each gate's share of transplant outcomes it predicts, (a, b, real top token) rows, a gate true when the top
    token is `yes`; best first (ties: the order of GATES, the simpler first)."""
    got = [(g, sum(f(a, b) == (top == yes) for a, b, top in rows) / max(len(rows), 1)) for g, f in GATES.items()]
    return sorted(got, key=lambda x: -x[1])


# ---- INVARIANTS lines

_LINE = re.compile(r"^\s*I(\d+)\s*:\s*(.+?)\s*->\s*(.+?)\s*$")
_TOKEN = r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
_CHANGE = re.compile(r"@(\d+)\s*=\s*" + _TOKEN)


def _split(s: str) -> list[str]:
    """Comma-separated parts, commas inside quoted tokens kept."""
    out, cur, q, esc = [], "", None, False
    for ch in s:
        if q:
            cur += ch
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            cur += ch
        elif ch == ",":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return [x for x in out + [cur.strip()] if x]


def parse_invariants(text: str, model: str = "vpd4l") -> tuple[list[dict], int]:
    """(the INVARIANTS lines as {"id", "intervention", "outcome", "line"}, the count of lines that do not parse or name
    a token the model does not have). Conflicting intervention targets, invalid drop fractions and repeated claim
    IDs or executable evidence count as bad; changing IDs or clause order cannot duplicate verified credit."""
    import components

    tok = lambda s: components._token_of(model, s)  # noqa: E731
    out, bad, seen, evidence = [], 0, set(), set()
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            bad += bool(line.strip().startswith("I") and "->" in line)
            continue
        try:
            ident = int(m.group(1))
            if ident in seen:
                raise ValueError(line)
            iv = {"tokens": {}, "transplants": [], "remove": []}
            carriers = set()
            for part in _split(m.group(2)):
                r = re.match(r"^remove\s+(V\d+)$", part)
                t = re.match(r"^(V\d+)\s+from\s+(.+)$", part)
                if r:
                    if r.group(1) in carriers:
                        raise ValueError(part)
                    carriers.add(r.group(1))
                    iv["remove"].append(r.group(1))
                elif t:
                    if t.group(1) in carriers:
                        raise ValueError(part)
                    carriers.add(t.group(1))
                    donor, ch, end = t.group(2), {}, 0
                    for change in _CHANGE.finditer(donor):
                        if donor[end:change.start()].strip():
                            raise ValueError(part)
                        p = int(change.group(1))
                        if p in ch:
                            raise ValueError(part)
                        ch[p] = tok(change.group(2))
                        end = change.end()
                    if donor[end:].strip():
                        raise ValueError(part)
                    if not ch or None in ch.values():
                        raise ValueError(part)
                    iv["transplants"].append((t.group(1), ch))
                else:
                    ch = _CHANGE.fullmatch(part)
                    if not ch or tok(ch.group(2)) is None:
                        raise ValueError(part)
                    p = int(ch.group(1))
                    if p in iv["tokens"]:
                        raise ValueError(part)
                    iv["tokens"][p] = tok(ch.group(2))
            o = m.group(3)
            if o == "unchanged":
                outc = ("unchanged",)
            elif re.match(r"^p\(top\)\s*<\s*[0-9.]+$", o):
                outc = ("drop", float(o.split("<")[1]))
                if not 0.0 < outc[1] < 1.0:
                    raise ValueError(o)
            elif re.match(r"^(top|not)\s+" + _TOKEN + "$", o):
                kind, s = o.split(None, 1)
                y = tok(s)
                if y is None:
                    raise ValueError(o)
                outc = (kind, y)
            else:
                raise ValueError(o)
            key = (tuple(sorted(iv["tokens"].items())),
                   tuple(sorted((name, tuple(sorted(ch.items()))) for name, ch in iv["transplants"])),
                   tuple(sorted(iv["remove"])), outc)
            if key in evidence:
                raise ValueError(line)
            out.append({"id": ident, "intervention": iv, "outcome": outc, "line": line.strip()})
            seen.add(ident)
            evidence.add(key)
        except (ValueError, SyntaxError):
            bad += 1
    return out, bad


def check_invariants(sc: variables.Scorer, tx, vs: list[dict], text: str, model: str = "vpd4l") -> dict:
    """The INVARIANTS lines checked in the real model: per line holds, surprise and credit; the total credit in bits
    and the share holding (lines naming a variable the answer lacks, or not parsing, count as failing bold claims of
    one bit)."""
    lines, bad = parse_invariants(text, model)
    names = {v["name"] for v in vs}
    ok = [x for x in lines if all(n in names for n, _ in x["intervention"]["transplants"]) and all(n in names for n in x["intervention"]["remove"])]
    bad += len(lines) - len(ok)
    ck = Checker(sc, tx, vs)
    try:
        rows = [{"id": x["id"], "line": x["line"], **r} for x, r in zip(ok, ck.check([(x["intervention"], x["outcome"]) for x in ok]))]
    finally:
        ck.close()
    total = sum(r["credit"] for r in rows) - bad
    return {"invariants": rows, "bad": bad, "credit": total, "holds": sum(r["holds"] for r in rows) / max(len(rows) + bad, 1)}
