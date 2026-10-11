"""WorkspaceBench items and their counterfactuals (#2951; data: HF camilablank/workspace-bench, MIT,
evaluation only; the authors ask that the items never be used for training).

Arithmetic (hillclimbing_evals/arithmetic_intermediates): each family is a small causal model,
operands -> intermediates -> answer. A counterfactual is the same expression with every operand
digit changed (digit counts kept, so the chat-rendered prompts align token for token: Qwen writes
each digit as its own token). Patching a cell from it into the base run, the hypothesis "the cell
holds variable v" predicts the answer of the base inputs with v alone set to the counterfactual's
value; counterfactuals are drawn so that every hypothesis predicts a different answer.

Multihop (baseline_evals/single_token/lens-eval-multihop.json): a WorkspaceBench prompt is a
template with one slot (the atomic number, the holiday, the capital); a counterfactual fills the slot
with another value of the same token length and a different bridge entity, and its answer is known.
"""
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

WSB = Path(os.environ.get("WSB_DATA", Path.home() / "mpd-data/workspace-bench/hf"))
ARITH_DIR = WSB / "hillclimbing_evals/arithmetic_intermediates"
MULTIHOP_FILE = WSB / "baseline_evals/single_token/lens-eval-multihop.json"
ARITH_PROMPT = "Compute {expr}. Reply with only the final number, nothing else."


@dataclass(frozen=True)
class Family:
    pattern: str                 # the expression with operand slots {name}
    inter: Callable              # operands -> {intermediate: value}
    final: Callable              # operands, intermediates -> answer
    digits: bool = True          # digit-level hypotheses (off when a changed digit leaves the family, e.g. sqrt)
    valid: Callable = lambda o: True


FAMILIES = {
    "subsub": Family("({x} - {y}) - {z}", lambda o: {"I": o["x"] - o["y"]}, lambda o, i: i["I"] - o["z"]),
    "subsubx": Family("{z} - ({x} - {y})", lambda o: {"I": o["x"] - o["y"]}, lambda o, i: o["z"] - i["I"]),
    "addmul": Family("({x} + {y}) * {z}", lambda o: {"I": o["x"] + o["y"]}, lambda o, i: i["I"] * o["z"]),
    "addmulx": Family("{z} + ({x} * {y})", lambda o: {"I": o["x"] * o["y"]}, lambda o, i: o["z"] + i["I"]),
    "muladd": Family("({x} * {y}) + {z}", lambda o: {"I": o["x"] * o["y"]}, lambda o, i: i["I"] + o["z"]),
    "sign": Family("({x} - {y}) * {z}", lambda o: {"I": o["x"] - o["y"]}, lambda o, i: i["I"] * o["z"]),
    "signpair": Family("({x} - {y}) * {z}", lambda o: {"I": o["x"] - o["y"]}, lambda o, i: i["I"] * o["z"]),
    "absval": Family("abs({x} - {y}) * {z}", lambda o: {"I": abs(o["x"] - o["y"])}, lambda o, i: i["I"] * o["z"]),
    "maxsel": Family("max({x}, {y}) - {z}", lambda o: {"I": max(o["x"], o["y"])}, lambda o, i: i["I"] - o["z"]),
    "sqrt": Family("sqrt({x}) * {z}", lambda o: {"I": round(o["x"] ** 0.5)}, lambda o, i: i["I"] * o["z"], digits=False,
                   valid=lambda o: round(o["x"] ** 0.5) ** 2 == o["x"]),
    "mulmul": Family("({a} * {b}) + ({c} * {d})", lambda o: {"I": o["a"] * o["b"], "J": o["c"] * o["d"]}, lambda o, i: i["I"] + i["J"]),
    "mulmid": Family("({a} * {b}) + ({c} * {d})", lambda o: {"I": o["a"] * o["b"], "J": o["c"] * o["d"]}, lambda o, i: i["I"] + i["J"]),
}


def slots(pattern):
    return re.findall(r"\{(\w+)\}", pattern)


def parse(family, expr):
    pat = re.escape(FAMILIES[family].pattern)
    for s in slots(FAMILIES[family].pattern):
        pat = pat.replace(re.escape("{" + s + "}"), rf"(?P<{s}>\d+)")
    m = re.fullmatch(pat, expr)
    return None if m is None else {k: int(v) for k, v in m.groupdict().items()}


def num(v):
    return str(int(v)) if float(v).is_integer() else None


@dataclass
class ArithItem:
    family: str
    name: str
    ops: dict                    # operand -> value
    prompt: str
    answer: str
    inter: dict = field(default_factory=dict)

    @property
    def expr(self):
        return FAMILIES[self.family].pattern.format(**self.ops)


def evaluate(family, ops, override=None):
    """The family's causal model: the answer of `ops`, with intermediates in `override` fixed."""
    f = FAMILIES[family]
    inter = f.inter(ops)
    inter.update(override or {})
    return f.final(ops, inter)


def load_arith(family):
    items = []
    for it in json.loads((ARITH_DIR / f"{family}.json").read_text()):
        ops = parse(family, it["expr"])
        assert ops is not None, (family, it["expr"])
        assert it["prompt"] == ARITH_PROMPT.format(expr=it["expr"]), it["prompt"]
        ans = evaluate(family, ops)
        assert num(ans) == it["answer"], (it["name"], ans, it["answer"])
        inter = FAMILIES[family].inter(ops)
        assert num(inter["I"]) == it["first_value"], (it["name"], inter, it["first_value"])
        items.append(ArithItem(family, it["name"], ops, it["prompt"], it["answer"], inter))
    return items


def generate_arith(family, n, rng):
    """n new problems of a family in its bank's shape: a bank item's operand digit counts, and the
    intermediate and answer inside the bank's ranges. Not WorkspaceBench items: they stand in where
    a model fails too many bank items to map them (Qwen3-8B without thinking)."""
    bank = load_arith(family)
    f = FAMILIES[family]
    lo_i, hi_i = min(it.inter["I"] for it in bank), max(it.inter["I"] for it in bank)
    lo_a, hi_a = min(evaluate(family, it.ops) for it in bank), max(evaluate(family, it.ops) for it in bank)
    seen = {tuple(sorted(it.ops.items())) for it in bank}
    out = []
    while len(out) < n:
        like = rng.choice(bank)
        ops = {k: rng.randint(10 ** (len(str(v)) - 1), 10 ** len(str(v)) - 1) if len(str(v)) > 1 else rng.randint(2, 9)
               for k, v in like.ops.items()}
        key = tuple(sorted(ops.items()))
        if key in seen or not f.valid(ops):
            continue
        inter, ans = f.inter(ops), evaluate(family, ops)
        if not (lo_i <= inter["I"] <= hi_i and lo_a <= ans <= hi_a) or ans in inter.values() or inter["I"] in ops.values():
            continue
        seen.add(key)
        expr = f.pattern.format(**ops)
        out.append(ArithItem(family, f"{family}-gen{len(out):03d}", ops, ARITH_PROMPT.format(expr=expr), num(ans), inter))
    return out


def hypotheses(item, parts=True):
    """Names of the variables a cell may hold: none, each operand, each digit of a multi-digit
    operand (families with digits=True), each intermediate, the answer, and (parts=True) each digit
    of a multi-digit intermediate ('I:0' its leading digit) and of the answer ('answer:0'), which
    the cells that write the answer one digit at a time hold."""
    f = FAMILIES[item.family]
    hs = ["none"] + list(item.ops)
    if f.digits:
        hs += [f"{k}.{d}" for k, v in item.ops.items() if len(str(v)) > 1 for d in range(len(str(v)))]
    hs += list(item.inter) + ["answer"]
    if parts:
        hs += [f"{k}:{d}" for k, v in item.inter.items() if len(str(abs(int(v)))) > 1 for d in range(len(str(abs(int(v)))))]
        a = len(str(abs(int(evaluate(item.family, item.ops)))))
        hs += [f"answer:{d}" for d in range(a)] if a > 1 else []
    return hs


def _swap_digit(v, w, d):
    """v with its d-th digit (from the left, sign kept) taken from w; None when their lengths differ."""
    sv, sw = str(abs(int(v))), str(abs(int(w)))
    if len(sv) != len(sw):
        return None
    out = int(sv[:d] + sw[d] + sv[d + 1:])
    return -out if v < 0 else out


def predict(item, cf_ops, h):
    """The answer the base item gives when variable h takes the counterfactual's value (None when
    the variable has no counterpart in this counterfactual: digit positions that do not line up)."""
    fam = item.family
    if h == "none":
        return evaluate(fam, item.ops)
    if h == "answer":
        return evaluate(fam, cf_ops)
    if h in item.inter:
        return evaluate(fam, item.ops, {h: FAMILIES[fam].inter(cf_ops)[h]})
    if ":" in h:
        k, d = h.split(":")
        if k == "answer":
            return _swap_digit(evaluate(fam, item.ops), evaluate(fam, cf_ops), int(d))
        v = _swap_digit(item.inter[k], FAMILIES[fam].inter(cf_ops)[k], int(d))
        return None if v is None else evaluate(fam, item.ops, {k: v})
    if "." in h:
        k, d = h.split(".")
        digits = list(str(item.ops[k]))
        digits[int(d)] = str(cf_ops[k])[int(d)]
        return evaluate(fam, {**item.ops, k: int("".join(digits))})
    return evaluate(fam, {**item.ops, h: cf_ops[h]})


def sign(v):
    return (v > 0) - (v < 0)


def arith_cfs(item, rng, n, tries=20000):
    """Up to n counterfactual operand sets: every digit of every operand changed (no leading zero),
    the same digit counts, the same signs of the intermediates and answer as the base, inside the
    family, and every hypothesis predicting a different integer answer."""
    f = FAMILIES[item.family]
    hs = hypotheses(item, parts=False)
    out, seen = [], set()
    for _ in range(tries):
        if len(out) == n:
            break
        cf = {}
        for k, v in item.ops.items():
            s = str(v)
            cf[k] = int("".join(rng.choice([c for c in "0123456789" if c != s[i] and (i or c != "0")]) for i in range(len(s))))
        key = tuple(sorted(cf.items()))
        if key in seen or not f.valid(cf):
            continue
        seen.add(key)
        if sign(evaluate(item.family, cf)) != sign(evaluate(item.family, item.ops)):
            continue
        if any(sign(f.inter(cf)[i]) != sign(item.inter[i]) for i in item.inter):
            continue
        preds = [predict(item, cf, h) for h in hs]
        if any(num(p) is None for p in preds) or len({num(p) for p in preds}) < len(preds):
            continue
        out.append(cf)
    return out


@dataclass(frozen=True)
class Template:
    prefix: str
    suffix: str
    base: tuple                  # WorkspaceBench item names read through this template
    fillers: tuple               # (slot text, answer, bridge entity)


TEMPLATES = [
    Template("Fact: The chemical symbol for the element with atomic number ", " is ",
             ("atomic-26-symbol", "atomic-79-symbol"),
             (("26", "Fe", "iron"), ("79", "Au", "gold"), ("29", "Cu", "copper"), ("47", "Ag", "silver"), ("50", "Sn", "tin"),
              ("80", "Hg", "mercury"), ("82", "Pb", "lead"), ("11", "Na", "sodium"), ("19", "K", "potassium"),
              ("74", "W", "tungsten"), ("51", "Sb", "antimony"))),
    Template("Fact: ", " is celebrated in month number ",
             ("holiday-christmas-monthnum", "holiday-halloween-monthnum", "holiday-valentines-monthnum"),
             (("Christmas Day", "12", "December"), ("Halloween", "10", "October"), ("Valentine's Day", "2", "February"),
              ("New Year's Day", "1", "January"), ("Thanksgiving", "11", "November"), ("Labor Day", "9", "September"),
              ("Canada Day", "7", "July"), ("Bastille Day", "7", "July"), ("Groundhog Day", "2", "February"),
              ("Saint Patrick's Day", "3", "March"), ("Cinco de Mayo", "5", "May"), ("April Fools' Day", "4", "April"),
              ("Veterans Day", "11", "November"), ("Earth Day", "4", "April"), ("Boxing Day", "12", "December"),
              ("Flag Day", "6", "June"), ("Juneteenth", "6", "June"), ("Remembrance Day", "11", "November"),
              ("Columbus Day", "10", "October"), ("Memorial Day", "5", "May"), ("Mother's Day", "5", "May"),
              ("Father's Day", "6", "June"), ("Christmas Eve", "12", "December"), ("Pi Day", "3", "March"),
              ("Guy Fawkes Night", "11", "November"), ("Presidents' Day", "2", "February"))),
    Template("Fact: The US ", " holiday is celebrated in month number ",
             ("holiday-independence-monthnum",),
             (("Independence Day", "7", "July"), ("Labor Day", "9", "September"), ("Memorial Day", "5", "May"),
              ("Thanksgiving", "11", "November"), ("Veterans Day", "11", "November"), ("Flag Day", "6", "June"),
              ("Columbus Day", "10", "October"), ("Presidents' Day", "2", "February"), ("Juneteenth", "6", "June"))),
    Template("Fact: The language spoken in the country whose capital is ", " is",
             ("bf-madrid-language", "b3-mh-lisbon-language"),
             (("Madrid", "Spanish", "Spain"), ("Lisbon", "Portuguese", "Portugal"), ("Rome", "Italian", "Italy"),
              ("Paris", "French", "France"), ("Berlin", "German", "Germany"), ("Athens", "Greek", "Greece"),
              ("Moscow", "Russian", "Russia"), ("Tokyo", "Japanese", "Japan"), ("Warsaw", "Polish", "Poland"),
              ("Cairo", "Arabic", "Egypt"), ("Budapest", "Hungarian", "Hungary"), ("Helsinki", "Finnish", "Finland"),
              ("Stockholm", "Swedish", "Sweden"), ("Oslo", "Norwegian", "Norway"), ("Ankara", "Turkish", "Turkey"),
              ("Bangkok", "Thai", "Thailand"), ("Hanoi", "Vietnamese", "Vietnam"), ("Seoul", "Korean", "Korea"),
              ("Prague", "Czech", "Czechia"), ("Copenhagen", "Danish", "Denmark"), ("Amsterdam", "Dutch", "Netherlands"),
              ("Tehran", "Persian", "Iran"), ("Bucharest", "Romanian", "Romania"), ("Kyiv", "Ukrainian", "Ukraine"),
              ("Jakarta", "Indonesian", "Indonesia"))),
    Template("Fact: The language of the country whose capital is ", " is",
             ("b3-mh-buenosaires-language",),
             (("Buenos Aires", "Spanish", "Argentina"), ("Rio de Janeiro", "Portuguese", "Brazil"), ("Kuala Lumpur", "Malay", "Malaysia"),
              ("Addis Ababa", "Amharic", "Ethiopia"), ("New Delhi", "Hindi", "India"), ("Mexico City", "Spanish", "Mexico"),
              ("Phnom Penh", "Khmer", "Cambodia"), ("Tel Aviv", "Hebrew", "Israel"), ("Rome", "Italian", "Italy"),
              ("Paris", "French", "France"), ("Berlin", "German", "Germany"), ("Athens", "Greek", "Greece"),
              ("Moscow", "Russian", "Russia"), ("Tokyo", "Japanese", "Japan"), ("Warsaw", "Polish", "Poland"),
              ("Budapest", "Hungarian", "Hungary"), ("Helsinki", "Finnish", "Finland"), ("Stockholm", "Swedish", "Sweden"))),
    Template("Fact: The capital of the country where ", " was founded is",
             ("b3-mh-nokia-capital", "b3-mh-ikea-capital"),
             (("Nokia", "Helsinki", "Finland"), ("IKEA", "Stockholm", "Sweden"), ("Samsung", "Seoul", "South Korea"),
              ("Toyota", "Tokyo", "Japan"), ("Nestlé", "Bern", "Switzerland"), ("Philips", "Amsterdam", "Netherlands"),
              ("Siemens", "Berlin", "Germany"), ("Fiat", "Rome", "Italy"), ("Heineken", "Amsterdam", "Netherlands"),
              ("Sony", "Tokyo", "Japan"), ("Huawei", "Beijing", "China"), ("Carlsberg", "Copenhagen", "Denmark"),
              ("Peugeot", "Paris", "France"), ("Ferrari", "Rome", "Italy"), ("Adidas", "Berlin", "Germany"),
              ("Rolex", "Bern", "Switzerland"), ("Hyundai", "Seoul", "South Korea"), ("Renault", "Paris", "France"),
              ("Zara", "Madrid", "Spain"), ("Nintendo", "Tokyo", "Japan"), ("Alibaba", "Beijing", "China"),
              ("BlackBerry", "Ottawa", "Canada"), ("Tata", "New Delhi", "India"))),
    Template("Fact: The capital of the country famous for ", " is",
             ("b3-mh-fjords-capital", "b3-mh-tulips-capital", "b3-mh-kilts-capital", "b3-mh-eiffel-capital"),
             (("fjords", "Oslo", "Norway"), ("tulips and windmills", "Amsterdam", "Netherlands"),
              ("kilts and bagpipes", "Edinburgh", "Scotland"), ("the Eiffel Tower", "Paris", "France"),
              ("the Colosseum", "Rome", "Italy"), ("the Parthenon", "Athens", "Greece"), ("the Kremlin", "Moscow", "Russia"),
              ("the Taj Mahal", "New Delhi", "India"), ("the pyramids", "Cairo", "Egypt"), ("maple syrup", "Ottawa", "Canada"),
              ("sushi", "Tokyo", "Japan"), ("kangaroos", "Canberra", "Australia"), ("flamenco", "Madrid", "Spain"),
              ("pizza and pasta", "Rome", "Italy"), ("the Great Wall", "Beijing", "China"), ("Machu Picchu", "Lima", "Peru"),
              ("koalas", "Canberra", "Australia"), ("fado", "Lisbon", "Portugal"), ("tango", "Buenos Aires", "Argentina"),
              ("geysers", "Reykjavik", "Iceland"), ("saunas", "Helsinki", "Finland"), ("chocolate and watches", "Bern", "Switzerland"),
              ("beer and pretzels", "Berlin", "Germany"), ("cheese and wine", "Paris", "France"), ("Big Ben", "London", "England"),
              ("the Alhambra", "Madrid", "Spain"), ("goulash", "Budapest", "Hungary"), ("pierogi", "Warsaw", "Poland"),
              ("vodka", "Moscow", "Russia"), ("Petra", "Amman", "Jordan"))),
]


@dataclass
class HopItem:
    name: str
    template: int
    slot: str
    prompt: str
    answer: str
    bridge: str


def load_multihop():
    """The WorkspaceBench multihop items read through a template, each checked to equal its template
    filled with the item's own slot, answer and bridge."""
    bank = {it["name"]: it for it in json.loads(MULTIHOP_FILE.read_text())["items"]}
    items = []
    for t, tpl in enumerate(TEMPLATES):
        for name in tpl.base:
            it = bank[name]
            slot = it["prompt"][len(tpl.prefix):len(it["prompt"]) - len(tpl.suffix)]
            assert tpl.prefix + slot + tpl.suffix == it["prompt"], (name, it["prompt"])
            fill = [f for f in tpl.fillers if f[0] == slot]
            assert fill and fill[0][1] == it["target"] and fill[0][2] in it["intermediates"], (name, fill, it)
            items.append(HopItem(name, t, slot, it["prompt"], it["target"], fill[0][2]))
    return items


def hop_cfs(item):
    """Every other filler of the item's template with a different answer and bridge."""
    tpl = TEMPLATES[item.template]
    return [HopItem(f"{item.name}~{s}", item.template, s, tpl.prefix + s + tpl.suffix, a, b)
            for s, a, b in tpl.fillers if a != item.answer and b != item.bridge and s != item.slot]


def continuation(prompt, answer):
    """The answer as the model writes it after the prompt: a leading space unless the prompt ends in one."""
    return answer if prompt.endswith(" ") else " " + answer


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                   enable_thinking=False, tokenize=False)


def rng_for(name, salt=0):
    return random.Random(f"{name}:{salt}")
