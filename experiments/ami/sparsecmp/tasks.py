"""The circuit-sparsity task suite (Gao et al. 2511.13653, Table 1), rebuilt from public Python code.

Each released task pair is a real Python file cut at a decision point, plus a minimal edit that flips the answer
(decoded from csp_yolo2's visualizer files; released/samples_csp_yolo2_prune_v4.json). We apply the same edits to
codeparrot-clean files. An item is two prompts with their answer continuations:

    {"task", "pos": prompt, "neg": prompt, "pos_ans": text, "neg_ans": text, "kind": "binary" | "pos_vs_rest"}

"binary" (their loss with alt_tokid): each prompt's cross-entropy over a softmax restricted to the two answer tokens.
"pos_vs_rest" (else_elif, if_ternary: "predict : or not :"): pos must give its answer token, neg any other token.
The prompt texts are shared by every tokenizer; answer tokens are the first token of the answer after the prompt.

python tasks.py build OUT.jsonl [--files N] [--per-task K]
"""
import argparse
import gzip
import json
import os
import random
import re

ROOT = os.path.expanduser("~/mpd-data/ami/sparsecmp")
CORPUS = f"{ROOT}/corpus/file-000000000054.json.gz"
CTX_CHARS = 700          # their prompts are 28-261 tinypython tokens; about 3 characters per token

TASKS = ["single_double_quote", "bracket_brace", "bracket_counting", "class_init", "else_elif", "enumerate_range",
         "final_kwarg", "for_while", "if_equals", "if_ternary", "indent_for", "lambda_func", "set_or_string",
         "set_or_string_fixedvarname", "var_if", "var_swap", "while_return_true", "with_as", "with_open"]
# fstring_brace has no released pair (its csp_yolo2 prune is empty), so its edit is unknown.


def context(text: str, end: int) -> str:
    """The file up to `end`, cut on a line start so at most CTX_CHARS remain."""
    lo = max(0, end - CTX_CHARS)
    if lo > 0:
        nl = text.find("\n", lo, end)
        lo = nl + 1 if nl >= 0 else lo
    return text[lo:end]


def item(task, pos, neg, pos_ans, neg_ans, kind="binary"):
    return {"task": task, "pos": pos, "neg": neg, "pos_ans": pos_ans, "neg_ans": neg_ans, "kind": kind}


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def single_double_quote(t, rng):
    # Station("Station 1  ->  ")     Station('Station 1  ->  ')
    for m in re.finditer(r"\((['\"])([^'\"\\\n]{1,40})\1\)", t):
        pre = context(t, m.start() + 1)
        body = m.group(2)
        yield item("single_double_quote", pre + '"' + body, pre + "'" + body, '")', "')")


def bracket_brace(t, rng):
    # dictionary = ["waves"  ->  ,      dictionary = {"waves"  ->  :
    for m in re.finditer(r"(?m)^([ \t]*" + IDENT + r" = )[\[{](['\"])([^'\"\\\n]{1,30})\2", t):
        pre = context(t, m.start()) + m.group(1)
        s = m.group(2) + m.group(3) + m.group(2)
        yield item("bracket_brace", pre + "[" + s, pre + "{" + s, ",", ":")


def _block_line_starts(t):
    """Starts of indented statement lines (synthetic insertion points): the file before the line has balanced
    brackets and the previous line is not continued."""
    out = []
    depth = 0
    last = 0
    for m in re.finditer(r"(?m)^( {4,8})(?=[A-Za-z_])", t):
        seg = t[last:m.start()]
        depth += sum(seg.count(c) for c in "([{") - sum(seg.count(c) for c in ")]}")
        last = m.start()
        prev = t[:m.start()].rstrip("\n").rsplit("\n", 1)[-1].rstrip()
        if depth == 0 and prev and not prev.endswith((",", "\\", "(", "[", "{")):
            out.append(m.start())
    return out


def bracket_counting(t, rng):
    # values =[[5, 3, 11, 3, 12  ->  ]]      values =[5, 3, 11, 3, 12  ->  ]
    for p in _block_line_starts(t)[::7]:
        ind = re.match(r" *", t[p:]).group(0)
        nums = ", ".join(str(rng.randint(0, 15)) for _ in range(rng.randint(2, 5)))
        pre = context(t, p) + ind + "values ="
        yield item("bracket_counting", pre + "[[" + nums, pre + "[" + nums, "]]", "]")


def class_init(t, rng):
    # def __init__(self, name, x, y, z): ... self.y =  -> " y";  the counterfactual renames y in both places
    for m in re.finditer(r"def __init__\(self, ([^)]*)\):\n((?:[ \t]+self\.(" + IDENT + r") = \3\n)+)", t):
        params = [p.strip() for p in m.group(1).split(",")]
        lines = re.findall(r"([ \t]+)self\.(" + IDENT + r") = \2\n", m.group(2))
        if len(lines) < 2 or any(not re.fullmatch(IDENT, p) for p in params):
            continue
        k = rng.randint(1, len(lines) - 1)
        name = lines[k][1]
        others = [w for w in set(re.findall(IDENT, t)) if 2 <= len(w) <= 12 and w not in params and w != "self"
                  and not w[0].isupper()]
        if name not in params or not others:
            continue
        new = rng.choice(sorted(others))
        head = context(t, m.start())
        body = "".join(f"{ind}self.{nm} = {nm}\n" for ind, nm in lines[:k]) + f"{lines[k][0]}self.{name} ="
        sig = "def __init__(self, " + ", ".join(params) + "):\n"
        sig2 = "def __init__(self, " + ", ".join(new if p == name else p for p in params) + "):\n"
        yield item("class_init", head + sig + body, head + sig2 + body.replace(f"self.{name} =", f"self.{new} ="),
                   " " + name, " " + new)


def else_elif(t, rng):
    # "    else " -> ":"      "    elif " -> anything but ":"  (their prompts end in a space token)
    for m in re.finditer(r"(?m)^([ \t]*)else:", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("else_elif", pre + "else ", pre + "elif ", ":", " x", kind="pos_vs_rest")


def enumerate_range(t, rng):
    # for i, c in  ->  enumerate        for i in  ->  range
    for m in re.finditer(r"(?m)^([ \t]*)for (" + IDENT + r"), (" + IDENT + r") in enumerate\(", t):
        pre = context(t, m.start()) + m.group(1)
        i, c = m.group(2), m.group(3)
        yield item("enumerate_range", pre + f"for {i}, {c} in", pre + f"for {i} in", " enumerate", " range")


def final_kwarg(t, rng):
    # def f(self, a, args=None, kwargs  ->  =None       def f(self, a, args, kwargs  ->  ):
    for m in re.finditer(r"(?m)^([ \t]*def " + IDENT + r"\()([^)\n]*)\):", t):
        args = [a.strip() for a in m.group(2).split(",")]
        if len(args) < 2 or not args[-1].endswith("=None") or not any(a.endswith("=None") for a in args[:-1]):
            continue
        last = args[-1][:-len("=None")]
        if not re.fullmatch(IDENT, last) or any("=" in a and not a.endswith("=None") for a in args):
            continue
        pre = context(t, m.start()) + m.group(1)
        kw = ", ".join(args[:-1]) + ", " + last
        plain = ", ".join(a.replace("=None", "") for a in args[:-1]) + ", " + last
        yield item("final_kwarg", pre + kw, pre + plain, "=None", "):")


def for_while(t, rng):
    # while queue  ->  :        for queue  ->  " in"
    for m in re.finditer(r"(?m)^([ \t]*)while (" + IDENT + r"):", t):
        if m.group(2) in ("True", "False", "None"):
            continue
        pre = context(t, m.start()) + m.group(1)
        yield item("for_while", pre + "while " + m.group(2), pre + "for " + m.group(2), ":", " in")


def if_equals(t, rng):
    # if current_board[i][j]  ->  " =="       current_board[i][j]  ->  " ="
    for m in re.finditer(r"(?m)^([ \t]*)if (" + IDENT + r"(?:\[[^\]\n]{1,10}\])*) ==", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("if_equals", pre + "if " + m.group(2), pre + m.group(2), " ==", " =")


def if_ternary(t, rng):
    # if C:\n    V = A\nelse  ->  ":"         V = A if C else  ->  anything but ":"
    pat = (r"(?m)^([ \t]*)if ([^\n:]{1,40}):\n\1([ \t]+)(" + IDENT + r") = ([^\n]{1,40})\n\1else:\n\1\3\4 = [^\n]+")
    for m in re.finditer(pat, t):
        ind, cond, ind2, var, a = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        pre = context(t, m.start())
        pos = pre + f"{ind}if {cond}:\n{ind}{ind2}{var} = {a}\n{ind}else"
        neg = pre + f"{ind}{var} = {a} if {cond} else"
        yield item("if_ternary", pos, neg, ":", " x", kind="pos_vs_rest")


def indent_for(t, rng):
    # a for line inside a function body (next line at 8 spaces) vs the same line at top level (4 spaces)
    for m in re.finditer(r"(?m)^(    )(for [^\n]{1,60}:\n)(        )(?=\S)", t):
        pre = context(t, m.start())
        nxt = t[m.end():m.end() + 20].split("\n")[0]
        yield item("indent_for", pre + "    " + m.group(2), pre + m.group(2), "        " + nxt, "    " + nxt)


def lambda_func(t, rng):
    # score = lambda agent, model  ->  ":"      def score  ->  "("
    for m in re.finditer(r"(?m)^([ \t]*)(" + IDENT + r") = lambda ([^:\n]{1,40}):", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("lambda_func", pre + f"{m.group(2)} = lambda {m.group(3)}", pre + "def " + m.group(2), ":", "(")


def _set_or_string(t, rng, fixed):
    task = "set_or_string_fixedvarname" if fixed else "set_or_string"
    for m in re.finditer(r"(?m)^([ \t]*)((?:self\.)?" + IDENT + r") = set\(\)\n", t):
        var = m.group(2)
        use = re.search(r"(?m)^([ \t]*)" + re.escape(var) + r"\.add\(", t[m.end():m.end() + 1500])
        if not use:
            continue
        end = m.end() + use.start() + len(use.group(1))
        pre = context(t, m.start())
        mid = t[m.start():end]
        if fixed:
            if "self." in var:
                continue
            mid = re.sub(r"\b" + re.escape(var) + r"\b", "current", mid)
            pre = re.sub(r"\b" + re.escape(var) + r"\b", "current", pre)
            var = "current"
        pos = pre + mid + var
        neg = pre + mid.replace(f"{var} = set()", f'{var} = ""', 1) + var
        yield item(task, pos, neg, ".add", " +=")


def set_or_string(t, rng):
    yield from _set_or_string(t, rng, False)


def set_or_string_fixedvarname(t, rng):
    yield from _set_or_string(t, rng, True)


def var_if(t, rng):
    # threshold = 3  ->  newline        if threshold == 3  ->  ":"
    for m in re.finditer(r"(?m)^([ \t]*)(" + IDENT + r") = (\d{1,4}|True|False|None|'[a-z]{1,8}'|\"[a-z]{1,8}\")$", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("var_if", pre + f"{m.group(2)} = {m.group(3)}", pre + f"if {m.group(2)} == {m.group(3)}", "\n", ":")


def var_swap(t, rng):
    # (inserted) # swap x and y \n x, y = y,  ->  " x"      # swap y and x \n y, x = x,  ->  " y"
    for p in _block_line_starts(t)[::9]:
        ind = re.match(r" *", t[p:]).group(0)
        pre = context(t, p)
        yield item("var_swap", pre + f"{ind}# swap x and y\n{ind}x, y = y,", pre + f"{ind}# swap y and x\n{ind}y, x = x,",
                   " x", " y")


def while_return_true(t, rng):
    # return True  ->  newline       while True  ->  ":"
    for m in re.finditer(r"(?m)^([ \t]+)return True\n", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("while_return_true", pre + "return True", pre + "while True", "\n", ":")


def with_as(t, rng):
    # with open(f, "r")  ->  " as"        if open(f, "r")  ->  ":"
    for m in re.finditer(r"(?m)^([ \t]*)with (open\([^()\n]{1,60}\)) as ", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("with_as", pre + "with " + m.group(2), pre + "if " + m.group(2), " as", ":")


def with_open(t, rng):
    # with open("input.txt", "  ->  r        with open("output.txt", "  ->  w
    for m in re.finditer(r"(?m)^([ \t]*)with open\(", t):
        pre = context(t, m.start()) + m.group(1)
        yield item("with_open", pre + 'with open("input.txt", "', pre + 'with open("output.txt", "', "r", "w")


EXTRACT = {name: globals()[name] for name in TASKS}


def files(path=CORPUS):
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if d.get("autogenerated") or len(d["content"]) < 400:
                continue
            yield d["repo_name"] + "/" + d["path"], d["content"]


def build(out: str, n_files: int, per_task: int, seed: int = 0):
    """At most one item per file and task (items from one file share context), so the split by file is clean."""
    rng = random.Random(seed)
    got = {t: [] for t in TASKS}
    for fi, (name, text) in enumerate(files()):
        if fi >= n_files or all(len(v) >= per_task for v in got.values()):
            break
        for t in TASKS:
            if len(got[t]) >= per_task:
                continue
            cands = list(EXTRACT[t](text, rng))
            if cands:
                it = rng.choice(cands)
                it["file"] = name
                got[t].append(it)
    with open(out, "w") as f:
        for t in TASKS:
            for it in got[t]:
                f.write(json.dumps(it) + "\n")
    print({t: len(v) for t, v in got.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("out")
    ap.add_argument("--files", type=int, default=61373)
    ap.add_argument("--per-task", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.files, a.per_task)
