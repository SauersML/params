"""Test answer-token hypotheses per task on the released pairs: the released circuit (csp_yolo2, released means) was
bisected to the target on their answer tokens, so the right hypothesis gives a circuit loss near 0.15.

python hyp.py [--device mps]
"""
import argparse
import json
import re

import torch

import answers as A
import check_release as CR
import nodes as N

ROOT = N.ROOT
# (kind, pos answer, neg answer); "{name}" = the attribute name in the prompt's last line, "{ind+4}" = one level
# deeper than the prompt's last line
H = {
    "bracket_brace": [("binary", ",", ":"), ("binary", '",', '":'), ("binary", " ,", " :")],
    "bracket_counting": [("binary", "]]", "]"), ("binary", "]]\n", "]\n"), ("binary", "]],", "],"),
                         ("pos_vs_rest", "]]", None), ("binary", "]", "]]")],
    "class_init": [("binary", "{name}", "{name}"), ("pos_vs_rest", "{name}", None)],
    "else_elif": [("pos_vs_rest", ":", None), ("pos_vs_rest", ":\n", None), ("binary", ":", " "), ("binary", ":\n", " ")],
    "if_ternary": [("pos_vs_rest", ":", None), ("pos_vs_rest", ":\n", None), ("binary", ":\n", " "), ("binary", ":", " ")],
    "indent_for": [("binary", "{ind+4}", "{ind+4}"), ("binary", "{ind+4}x", "{ind+4}x")],
    "lambda_func": [("binary", ":", "("), ("pos_vs_rest", ":", None), ("binary", ":", "(self")],
    "if_equals": [("binary", " ==", " ="), ("pos_vs_rest", " ==", None)],
    "with_as": [("binary", " as", ":"), ("binary", " as", ":\n"), ("pos_vs_rest", " as", None)],
    "while_return_true": [("binary", "\n", ":"), ("binary", "\n", ":\n"), ("binary", "\n\n", ":\n"), ("pos_vs_rest", ":\n", None)],
    "var_swap": [("binary", " x", " y"), ("binary", " x\n", " y\n")],
    "set_or_string": [("binary", ".add", " +="), ("binary", ".add(", " +=")],
    "with_open": [("binary", "r", "w"), ("binary", "r\")", "w\")"), ("binary", "r", "w\")")],
    "var_if": [("binary", "\n", ":\n"), ("binary", "\n", ":")],
    "for_while": [("binary", ":", " in"), ("binary", ":\n", " in")],
}


def fill(s, prompt):
    if s is None:
        return None
    last = prompt.rstrip("\n").rsplit("\n", 1)[-1]
    if "{name}" in s:
        m = re.search(r"self\.(\w+) =$", prompt)
        return " " + m.group(1) if m else None
    if "{ind+4}" in s:
        ind = re.match(r"[ \t]*", last).group(0)
        return s.replace("{ind+4}", ind + "    ")
    return s


def first_tok(model, s):
    ids = model.tokenize(s)
    return ids[0] if ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    model = N.GaoModel("csp_yolo2", a.device, means=f"{ROOT}/models/csp_yolo2/mean_baseline_pretrain.pt")
    P = json.load(open(f"{ROOT}/released/prunes_csp_yolo2_prune_v4.json"))
    S = json.load(open(f"{ROOT}/released/samples_csp_yolo2_prune_v4.json"))
    out = {}
    for t, hyps in H.items():
        rel = P.get(CR.task_dir(t))
        rows = S[CR.task_dir(t)]["tokens"]
        n = len(rows) // 2
        strip = [r[:max(i for i, x in enumerate(r) if x != 0) + 1] for r in rows]
        pos, neg = strip[:n], strip[n:]
        circ = CR.masks_for(model, rel["nodes"])
        L = {"full": (A.last_logits(model, pos, None), A.last_logits(model, neg, None)),
             "circuit": (A.last_logits(model, pos, circ), A.last_logits(model, neg, circ))}
        texts = [(model.enc.decode(p), model.enc.decode(q)) for p, q in zip(pos, neg)]
        res = []
        for kind, ps, ns in hyps:
            tok = [first_tok(model, fill(ps, p)) for p, q in texts]
            alt = [first_tok(model, fill(ns, q)) if ns else None for p, q in texts]
            row = {"kind": kind, "pos": ps, "neg": ns}
            for tag, (lp, ln) in L.items():
                r = torch.arange(len(tok))
                tk = torch.tensor(tok)
                if kind == "binary":
                    al = torch.tensor(alt)
                    two_p = torch.stack([lp[r, tk], lp[r, al]], 1).log_softmax(1)[:, 0]
                    two_n = torch.stack([ln[r, tk], ln[r, al]], 1).log_softmax(1)[:, 1]
                    row[tag] = round(float(-(two_p + two_n).mean() / 2), 4)
                else:
                    a_ = lp.log_softmax(1)[r, tk]
                    b_ = ln.log_softmax(1)[r, tk]
                    row[tag] = round(float(-(a_ + torch.log1p(-b_.exp().clamp(max=1 - 1e-6))).mean() / 2), 4)
            res.append(row)
        out[t] = res
        print(t, json.dumps(res), flush=True)
    json.dump(out, open(f"{ROOT}/released/hyp_yolo2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
