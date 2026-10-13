"""Mechanism templates as rules over positions (#2951 graph oracle, the lead's 10-12): a template binds roles that are
functions of the token sequence (the target t; o, the latest earlier occurrence of the current token; a = o + 1; t - k)
and states its parts relative to the roles, so it covers any text where the roles exist, at a size that does not grow
with the context.

A rule (MECH/rules/<name>.json): {"name", "family", "statement" (the rule in words), "roles": {role: definition},
"parts": [{"role", "site", "index"} | {"role", "site", "block": true}], "reads": [[writer part, reader part], ...],
"pairs": [[query part, key part, layer, read role]], "connections": "stated" | "all", "validation"}.
Size: rule bits = each part named among the 24 sites (a block) or among every subcomponent (a part), with no position
(a role names it), each role chosen from the role vocabulary, each stated read or pair its two parts' indices among the
rule's parts; binding bits on a text = log2(T) per role whose position must be stated (none here: every role in the
vocabulary is computed from the tokens, so a rule's size on a text of any length is its rule bits).

  rules.py write    -> MECH/rules/<name>.json for every template in MECH (copy templates and family templates)
"""
import json
import math
import os
from pathlib import Path

MECH = Path(os.environ.get("MECH", Path.home() / "mpd-data/graph_oracle/mechanisms"))
SITES, SUBCOMPONENTS = 24, 38912  # vpd4l's weight matrices and VPD subcomponents
ROLE_VOCAB = {"t": "the target position", "o": "the latest position before t holding the current token",
              "a": "o + 1, the token after that earlier occurrence"}  # and "t-k": k positions before the target
ROLE_CHOICES = 4 + 64  # t, o, a and t-k for k up to 64


def rule_bits(parts: list, edges: int, pairs: int, roles: int) -> float:
    n = len(parts)
    named = sum(math.log2(SITES + 1) if (p.get("block") or p.get("tok")) else math.log2(SUBCOMPONENTS) for p in parts)  # a token input: one more "site"
    return named + roles * math.log2(ROLE_CHOICES) + (2 * (edges + pairs)) * math.log2(max(2, n))


def from_template(tpl: dict) -> dict:
    parts = [{k: v for k, v in nd.items() if k != "id"} | {"id": nd["id"]} for nd in tpl["nodes"]]
    statement = tpl.get("explanation", "")
    if tpl.get("family", "").startswith("B:copy"):
        statement = ("At t, a layer-2 head reads the position a whose key, built at a by layer 1 reading the token before "
                     "it (query #316 at a against keys #329 and #119 at o), matches the query of the current token at t; "
                     "a's layer-2 value carries a's token into t's attention output, which the prediction reads.")
    elif tpl.get("rule"):
        r = tpl["rule"]
        statement = (f"When the current token is one of {r['current']}" + (f" and the prediction is {r['predicted']!r}" if r.get("predicted") else "")
                     + ": the family's shared subcomponents at t" + (", its carrier sites at t as blocks" if tpl.get("carriers") else "")
                     + ("".join(f", every site at t-{k} as a block" for k in tpl.get("context_offsets", [])) if tpl["name"].endswith("context") else "")
                     + ", every connection the model has among them.")
    edges = len(tpl.get("edges", [])) if tpl.get("connections") == "stated" else 0
    pairs = len(tpl.get("pairs", []))
    roles = sorted({p["role"] for p in parts})
    out = {"name": tpl["name"], "family": tpl.get("family"), "statement": statement,
           "roles": {r: ROLE_VOCAB.get(r, f"{r[2:]} positions before the target" if r.startswith("t-") else r) for r in roles},
           "parts": parts, "reads": tpl.get("edges", []) if tpl.get("connections") == "stated" else "every connection the model has among the parts",
           "pairs": tpl.get("pairs", []), "connections": tpl.get("connections"),
           "rule_bits": rule_bits(parts, edges, pairs, len(roles)), "binding_bits": 0.0,
           "validation": tpl.get("validation")}
    return out


def write():
    (MECH / "rules").mkdir(parents=True, exist_ok=True)
    for p in sorted(MECH.iterdir()):
        if p.suffix != ".json":
            continue
        r = from_template(json.loads(p.read_text()))
        (MECH / "rules" / p.name).write_text(json.dumps(r, indent=1))
        print(f"{p.stem}: {len(r['parts'])} parts, roles {list(r['roles'])}, rule bits {r['rule_bits']:.0f}")


if __name__ == "__main__":
    write()
