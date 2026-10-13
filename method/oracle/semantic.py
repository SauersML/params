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
token's probability below f times its base value), and ("rise", f) (the same token's probability above f times its
base value, f > 1). Each is checked in the real model (holds, 0 or 1) and priced by its
surprise against the no-change prediction: top y with y not the base top y0, -log2 p_base(y) bits; not y0 (the base
top displaced), -log2(1 - p_base(y0)); drop f, log2(1 / f); rise f, log2(f); unchanged, and not y with y not y0, 0. credit = (2 holds -
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
not <token>, p(top) < <f>, or p(top) > <f> * base; tokens as FINAL lines write them (components.shown).
RULES OUT lines (parse_rules_out, check_rules_out): RULES OUT: <alternative in words> | <intervention> -> <outcome> vs
<the alternative's outcome>, the two outcomes exclusive; the line holds when the first holds and the second fails.
"""
from __future__ import annotations

import gc
import ast
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
        elif kind == "rise":
            f = float(out[1])
            holds, surprise = float(q[y0]) > float(p0[y0]) + math.log(f), math.log2(f)
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

    def tok(s):
        # Keep FINAL's escaped spelling valid, then resolve equivalent Python
        # string literals through the same exact token spelling (no trimming).
        ident = components._token_of(model, s)
        if ident is not None:
            return ident
        value = ast.literal_eval(s)
        return components._token_of(model, components.shown(value)) if isinstance(value, str) else None
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
            elif re.fullmatch(r"p\(top\)\s*>\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*\*\s*base", o):
                factor = float(o.split(">")[1].split("*")[0])
                if not math.isfinite(factor) or factor <= 1.0:
                    raise ValueError(o)
                outc = ("rise", factor)
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


_RULE = re.compile(r"^\s*RULES OUT\s*:\s*(.+?)\s*\|\s*(.+?)\s*->\s*(.+?)\s+vs\s+(.+?)\s*$")


def exclusive(a: tuple, b: tuple, y0: int) -> bool:
    """Whether two outcomes about one run cannot both hold (y0 the base top token): a RULES OUT line must predict, under
    its intervention, something its alternative's outcome excludes."""
    for a, b in ((a, b), (b, a)):
        ka, kb = a[0], b[0]
        if ka == "top" and kb == "top":
            return int(a[1]) != int(b[1])
        if ka == "top" and kb == "unchanged":
            return int(a[1]) != y0
        if ka == "top" and kb == "not":
            return int(a[1]) == int(b[1])
        if ka == "unchanged" and kb == "not":
            return int(b[1]) == y0
        if ka == "drop" and kb == "rise":
            return True
    return False


def parse_rules_out(text: str, model: str = "vpd4l") -> tuple[list[dict], int]:
    """(the RULES OUT lines as {"alternative", "intervention", "h", "alt", "line"}, the count that do not parse). A line
    is 'RULES OUT: <the alternative in words> | <intervention> -> <outcome under the explanation> vs <outcome under the
    alternative>', intervention and outcomes in the INVARIANTS grammar (parse_invariants). The alternative carries no
    carrier of its own: its consequence is the outcome it predicts under the same intervention, run in the real model."""
    out, bad = [], 0
    for line in text.splitlines():
        if not line.strip().startswith("RULES OUT"):
            continue
        m = _RULE.match(line)
        if not m:
            bad += 1
            continue
        h, b1 = parse_invariants(f"I1: {m.group(2)} -> {m.group(3)}", model)
        a, b2 = parse_invariants(f"I1: {m.group(2)} -> {m.group(4)}", model)
        if b1 or b2 or not h or not a:
            bad += 1
            continue
        out.append({"alternative": m.group(1), "intervention": h[0]["intervention"], "h": h[0]["outcome"], "alt": a[0]["outcome"],
                    "line": line.strip()})
    return out, bad


def check_rules_out(sc: variables.Scorer, tx, vs: list[dict], text: str, model: str = "vpd4l") -> dict:
    """The RULES OUT lines checked in the real model: per line the intervention run once, the explanation's outcome and
    the alternative's each checked there (Checker.outcome), exclusive (the two cannot both hold, else the line rules
    nothing out and counts as bad), ruled_out (the explanation's holds and the alternative's fails) and margin, log2 of
    the run's probability of the explanation's event over the alternative's (None for p(top) outcomes). credit +1 per
    line ruling its alternative out, -1 otherwise (bad lines too): the alternative is free text, so a line earns a fixed
    bit rather than the alternative's improbability, which a straw alternative would inflate."""
    lines, bad = parse_rules_out(text, model)
    names = {v["name"] for v in vs}
    ok = [x for x in lines if all(n in names for n, _ in x["intervention"]["transplants"]) and all(n in names for n in x["intervention"]["remove"])]
    bad += len(lines) - len(ok)
    rows = []
    if ok:
        ck = Checker(sc, tx, vs)
        try:
            for x in ok:
                q = ck.run(x["intervention"])
                h, a = ck.outcome(q, x["h"]), ck.outcome(q, x["alt"])
                ex = exclusive(x["h"], x["alt"], ck.y0)
                ruled = ex and h["holds"] and not a["holds"]
                rows.append({"line": x["line"], "alternative": x["alternative"], "exclusive": ex, "holds": h["holds"],
                             "alt_holds": a["holds"], "ruled_out": ruled, "top": h["top"], "p_top": h["p_top"],
                             "margin": None if h["logp"] is None or a["logp"] is None else h["logp"] - a["logp"],
                             "credit": 1.0 if ruled else -1.0})
        finally:
            ck.close()
    bad += sum(not r["exclusive"] for r in rows)
    rows_ok = [r for r in rows if r["exclusive"]]
    return {"rules_out": rows, "bad": bad, "credit": sum(r["credit"] for r in rows_ok) - bad,
            "ruled_out": sum(r["ruled_out"] for r in rows_ok) / max(len(rows_ok) + bad, 1)}


def teacher_invariants(sc: variables.Scorer, tx, vs: list[dict], model: str = "vpd4l", per_class: int = 1, fresh: int = 16,
                       structured: bool = False) -> list[dict]:
    """Verified INVARIANTS lines from a text's kept variables, training targets for the writer. vs: variables with
    "name", "group" (parsed) and "values" (var_classes' value maps: classes with token_ids and predicts_id). Per variable
    and input position p: fresh donor tokens the value map never saw (sampled from the model's prediction at p, the
    map's tokens and the text's own left out); a fresh token z whose changed text the model itself sends to a class's
    predicted token y (not the text's top) belongs to that class, and the explanation then predicts that the carrier
    alone carries it: 'V from @p=z -> top y' (per_class tokens per class); per variable 'remove V -> p(top) < 0.5'; and
    per pair of variables at different positions, both carriers from two donors (each one's first fresh token), the
    model's own prediction on the text with both changes as the claim (independent recombination). A variable marked
    "identity" with no value map (it carries its positions' tokens) gets, per input position, fresh tokens z whose
    changed text the model sends to a new top y: 'V from @p=z -> top y' (2 per_class per position, distinct y).
    Each line checked in the real model (Checker.outcome: holds, surprise against the no-change prediction, credit),
    the lines that hold first. structured=True also retains each original Checker.check record in "executed",
    including numeric intervention/outcome IDs and failed checks, for exact proof-bound supervision artifacts.
    The default rows and number of model calls are unchanged."""
    import hashlib

    import components

    ck = Checker(sc, tx, vs)
    items, first = [], {}
    def unseen(p: int, seen: set) -> list[int]:  # fresh donor tokens at p, sampled from the model's prediction there
        lp = sc.nat.reference([tx.ids[:p]], [p - 1])[0, 0].cpu().double()
        w = lp.exp()
        w[sorted(seen)] = 0.0
        gen = torch.Generator().manual_seed(int(hashlib.sha256(repr(("invariants", int(tx.seed), p)).encode()).hexdigest()[:15], 16))
        return torch.multinomial(w, min(fresh, int((w > 0).sum())), replacement=False, generator=gen).tolist()

    try:
        for v in vs:
            if v.get("identity") and not v.get("values"):
                for p in map(int, v["inputs"]):
                    zs = unseen(p, {int(tx.ids[p])})
                    if zs:
                        first.setdefault(v["name"], (p, int(zs[0])))
                    ys = {}
                    for z in zs:
                        y = int(ck.text({p: z})["lq"][-1].argmax())
                        if y != ck.y0 and y not in ys and len(ys) < 2 * per_class:
                            ys[y] = z
                    items += [(v["name"], {"transplants": [(v["name"], {p: z})]}, ("top", y), [(v["name"], p, z), y]) for y, z in ys.items()]
            for vm in v.get("values") or []:
                p = int(vm["position"])
                seen = {int(z) for c in vm["classes"] for z in c["token_ids"]} | {int(tx.ids[p])}
                zs = unseen(p, seen)
                want = {int(c["predicts_id"]): c["id"] for c in vm["classes"][1:] if c.get("predicts_id") is not None}
                got = {}
                if zs:
                    first.setdefault(v["name"], (p, int(zs[0])))
                for z in zs:
                    y = int(ck.text({p: z})["lq"][-1].argmax())
                    if y != ck.y0 and y in want and len(got.setdefault(want[y], [])) < per_class:
                        got[want[y]].append((z, y))
                for pairs in got.values():
                    items += [(v["name"], {"transplants": [(v["name"], {p: z})]}, ("top", y), [(v["name"], p, z), y]) for z, y in pairs]
            items.append((v["name"], {"remove": [v["name"]]}, ("drop", 0.5), None))
        names = [v["name"] for v in vs if v["name"] in first]
        for i, a in enumerate(names):  # independent recombination: two variables from two donors, the model's own
            for b in names[i + 1:]:     # prediction on the text with both changes as the claim
                (pa, za), (pb, zb) = first[a], first[b]
                if pa == pb:
                    continue
                y = int(ck.text({pa: za, pb: zb})["lq"][-1].argmax())
                if y != ck.y0:
                    items.append((a, {"transplants": [(a, {pa: za}), (b, {pb: zb})]}, ("top", y), [(a, pa, za), (b, pb, zb), y]))
        rows = ck.check([(iv, out) for _, iv, out, _ in items])
    finally:
        ck.close()
    toks = components.token_strings(model, [t for *_, e in items if e for t in [x[2] for x in e[:-1]] + [e[-1]]])
    out, k = [], 0
    for (name, iv, outc, e), r in zip(items, rows):
        if e:
            parts = []
            for nm, q, _ in e[:-1]:
                parts.append(f"{nm} from @{q}={components.shown(toks[k])}")
                k += 1
            line = f"{', '.join(parts)} -> top {components.shown(toks[k])}"
            k += 1
        else:
            line = f"remove {name} -> p(top) < 0.5"
        row = {"line": line, **{x: r[x] for x in ("holds", "surprise", "credit", "top", "p_top", "logp")}}
        if structured:
            import copy

            row["executed"] = copy.deepcopy(r)
        out.append(row)
    out.sort(key=lambda r: (not r["holds"], -r["surprise"]))
    for i, r in enumerate(out):
        r["line"] = f"I{i + 1}: {r['line']}"
    return out
