"""Explanation verifiers for a model's own neurons (#2951, the lead's 10-15 direction after Cammarata's "Neural
Networks Are Already Clean"): each check costs about one target forward pass, and together they are the oracle's
explanation diagnostics. T1 tests activation predictability, not a causal mechanism certificate.

A neuron is (layer, index) of a Qwen3 MLP; its activation at a position is the input of down_proj,
a = silu(gate) * up. Its positive pole is how high a goes, its negative pole how low; the SwiGLU sign regime at a
position is (sign gate, sign up): ++ and -- give a > 0 (-- is the small "echo" of a negative gate), +- and -+ give
a < 0. An explanation is checked for one pole against that pole's corpus maximum (the atlas's largest value of a, or
of -a, over its corpus):
  blind maximum: a reader that sees only the explanation writes texts meant to drive the pole as far as it goes; the
    score is the best text's pole value (its largest over positions) over the corpus maximum;
  near misses: the reader writes NEAR texts close to the explanation's subject that should not fire the pole; the
    near-miss rate is the share whose pole value reaches FIRE of the corpus maximum;
  distant concept: the reader names the concept farthest from the explanation and writes DISTANT texts about it; the
    share reaching FIRE should be about 0;
  steering prediction: the explanation's prediction of what the model writes with the pole turned up (the neuron's
    activation set to the pole's sign times STEER corpus maxima at every position) is shown with that generation and
    an unsteered one, in random order, and the reader picks the one the prediction describes (chance 0.5).
T1 (the lead's pre-registered test): a neuron's explanation passes when blind maximum >= 0.3 and near misses < 0.3.

  neuron_verify.py verify --neurons N.jsonl --out OUT.jsonl [--model Qwen/Qwen3-1.7B] [--reader R] [--device D]
      N.jsonl lines {"layer", "index", "pole": "+"|"-", "corpus_max", "explanation"}
"""
from __future__ import annotations

import argparse
import json
import hashlib
import math
import numbers
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "rl")]

FIRE = 0.3  # a text fires a pole when its pole value reaches this share of the corpus maximum
NEAR, DISTANT, BLIND = 5, 5, 3
FLOOR_P = 1e-3  # effects: a token's relative change counts from this probability before
STEER = 3.0  # steering sets the activation to the pole's sign times this many corpus maxima
PROMPTS = ["The", "In my opinion,", "Here is a short story:", "Today I learned that"]  # steering's neutral starts
GEN = 24  # tokens generated per steering prompt
READER = "Qwen/Qwen3.6-27B"  # dense, so deterministic at temperature 0 (reader_reward: 10-14 repeats agree to 0.003)

ASK = """A neuron in a language model has this explanation of its {pole} side:
{explanation}

Write JSON with these fields, nothing else:
"max_texts": {blind} different texts of {blind_len}, each written to drive this side of the neuron as strongly as possible;
"near_misses": {near} texts on closely related subjects that, by the explanation, should NOT drive it;
"distant_concept": the concept farthest from what the explanation describes;
"distant_texts": {distant} texts about that distant concept;
"steering_prediction": one sentence saying what the model would write if this side of the neuron were turned up strongly while it continues ordinary prompts."""

BLIND_LEN = {"short": "1 to 3 sentences",  # T1's pre-registered blind text
             "long": "up to about 150 tokens (a passage that may set up the context the explanation describes)"}

ASK_GROUP = """A group of neurons in a language model (several neurons' sides that work together) has this explanation of what makes the group fire:
{explanation}

Write JSON with these fields, nothing else:
"max_texts": {blind} different texts of {blind_len}, each written to make this group fire as strongly as possible;
"near_misses": {near} texts on closely related subjects that, by the explanation, should NOT make it fire;
"distant_concept": the concept farthest from what the explanation describes;
"distant_texts": {distant} texts about that distant concept;
"steering_prediction": one sentence saying what the model would write if this group were turned up strongly while it continues ordinary prompts."""

JUDGE = """Prediction of how a language model's writing changes when one of its neurons is turned up:
{prediction}

Two continuations of the same prompts; one was written with the neuron turned up.
A: {a}
B: {b}

Which continuation does the prediction describe? Reply with the single letter A or B and nothing else."""


def _swiglu_architecture(model):
    """Only the SiLU-gated down_proj input has the documented sign semantics."""
    if getattr(model.config,"hidden_act",None) not in ("silu","swish"):
        raise ValueError("neuron verifier requires SiLU-gated MLP architecture")
    layers=model.model.layers
    if not layers:
        raise ValueError("target has no transformer layers")
    for layer in layers:
        mlp=layer.mlp
        try:
            width=mlp.down_proj.in_features
            valid=width>0 and mlp.gate_proj.out_features==width and mlp.up_proj.out_features==width
        except AttributeError:
            valid=False
        if not valid:
            raise ValueError("target must expose compatible gate_proj/up_proj/down_proj dimensions")
    return True


class Target:
    """A Qwen3 model with a neuron's activation read (the input of down_proj, with gate's and up's signs) and set."""

    def __init__(self, name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"  # each text's own positions first, the padding after (masked out)
        if device == "auto":  # layers spread over every visible GPU (Qwen3-32B on two 48 GB cards)
            self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto").eval()
            self.dev = self.m.get_input_embeddings().weight.device
        else:
            self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
            self.m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(self.dev).eval()
        self.architecture_verified = _swiglu_architecture(self.m)
        self.layers = self.m.model.layers
        self.name = name

    def group(self, texts: list[str], members: list) -> list[dict]:
        """Per text, a group's largest activation over positions (the first token left out): at each position the mean
        over members [layer, index, sign, corpus maximum] of the member's value in its sign over its maximum, clipped
        at 0 (neuron_groups)."""
        torch = self.torch
        seen: dict = {}
        hs = []
        for k, (l, i, sg, mx) in enumerate(members):
            def pre(_m, a, k=k, i=i, sg=sg, mx=mx):
                seen[k] = (a[0][..., i].float() * (sg / mx)).clamp_min(0)
            hs.append(self.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        out = []
        try:
            for s in range(0, len(texts), 16):
                enc = self.tok(texts[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
                with torch.no_grad():
                    self.m.model(**enc, use_cache=False)
                act = sum(seen[k].to(self.dev) for k in range(len(members))) / len(members)
                mask = enc["attention_mask"].bool().clone()
                mask[:, 0] = False
                for j in range(mask.shape[0]):
                    a = act[j][mask[j]]
                    if a.numel() == 0:
                        out.append({"pos": 0.0, "neg": 0.0, "position_pos": None, "position_neg": None})
                        continue
                    ip = int(a.argmax())
                    out.append({"pos": float(a[ip]), "neg": 0.0, "position_pos": int(mask[j].nonzero().flatten()[ip]), "position_neg": None})
        finally:
            for h in hs:
                h.remove()
        return out

    def poles(self, texts: list[str], layer: int, index: int) -> list[dict]:
        """Per text, its largest activation and its largest negated activation over positions (the first token, the
        attention sink, left out), with the sign regime at each."""
        torch = self.torch
        mlp = self.layers[layer].mlp
        seen: dict = {}
        hs = [mlp.gate_proj.register_forward_hook(lambda _m, _i, o: seen.__setitem__("gate", o[..., index].float())),
              mlp.up_proj.register_forward_hook(lambda _m, _i, o: seen.__setitem__("up", o[..., index].float())),
              mlp.down_proj.register_forward_pre_hook(lambda _m, a: seen.__setitem__("a", a[0][..., index].float()))]
        out = []
        try:
            for s in range(0, len(texts), 16):
                enc = self.tok(texts[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
                with torch.no_grad():
                    self.m.model(**enc, use_cache=False)  # capture only: no vocabulary logits or KV cache
                mask = enc["attention_mask"].bool().clone()
                mask[:, 0] = False
                for j in range(mask.shape[0]):
                    a, g, u = seen["a"][j][mask[j]], seen["gate"][j][mask[j]], seen["up"][j][mask[j]]
                    if a.numel() == 0:
                        out.append({"pos": 0.0, "neg": 0.0, "regime_pos": None, "regime_neg": None, "position_pos": None, "position_neg": None})
                        continue
                    ip, ineg = int(a.argmax()), int(a.argmin())
                    sign = lambda x: "+" if x > 0 else "-" if x < 0 else "0"  # noqa: E731
                    reg = lambda i: sign(g[i]) + sign(u[i])  # noqa: E731
                    positions = mask[j].nonzero(as_tuple=False).flatten()
                    out.append({"pos": float(a[ip]), "neg": float(-a[ineg]), "regime_pos": reg(ip), "regime_neg": reg(ineg),
                                "position_pos": int(positions[ip]), "position_neg": int(positions[ineg])})
        finally:
            for h in hs:
                h.remove()
        return out

    def effects(self, contexts: list[list[int]], layer: int, index: int, pole: str, k: int = 10, members: list | None = None) -> list[dict]:
        """What zeroing one pole does to the next-token prediction after each context (token ids, the pole's firing
        token last): the pole's values (that sign only) set to 0 at every position, as neuron_builder.pole_edit; per
        context the k tokens whose probability falls most and the k that rise most, (token, p before, p after), the
        unedited model's top k, and the total variation between the two distributions. Falls and rises are ranked by
        the change of log p among tokens at p >= FLOOR_P before (the absolute change ranks the model's own likeliest
        tokens first: zeroing one pole mostly flattens the prediction, 8B label arm 10-16)."""
        torch = self.torch
        zero = [(m[0], m[1], float(m[2])) for m in members] if members else [(layer, index, 1.0 if pole == "+" else -1.0)]

        def pre_of(i, sign):
            def pre(_m, a):
                x = a[0].clone()
                v = x[..., i]
                x[..., i] = torch.where(v * sign > 0, torch.zeros_like(v), v)
                return (x,) + tuple(a[1:])
            return pre

        out = []
        for s in range(0, len(contexts), 8):
            part = contexts[s:s + 8]
            T = max(len(c) for c in part)
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
            ids = torch.tensor([c + [pad] * (T - len(c)) for c in part], device=self.dev)
            mask = torch.tensor([[1] * len(c) + [0] * (T - len(c)) for c in part], device=self.dev)
            last = torch.tensor([len(c) - 1 for c in part], device=self.dev)
            ps = []
            for edit in (False, True):
                hs = [self.layers[l].mlp.down_proj.register_forward_pre_hook(pre_of(i, sg)) for l, i, sg in zero] if edit else []
                try:
                    with torch.no_grad():
                        lg = self.m(input_ids=ids, attention_mask=mask, use_cache=False).logits
                finally:
                    for h in hs:
                        h.remove()
                ps.append(torch.softmax(lg[torch.arange(len(part)), last].float(), -1))
            p0, p1 = ps
            d = p1 - p0
            rel = torch.log(p1.clamp_min(1e-12)) - torch.log(p0.clamp_min(1e-12))
            live = p0 >= FLOOR_P
            down, up = rel.masked_fill(~live, float("inf")), rel.masked_fill(~live, float("-inf"))
            for j in range(len(part)):
                tok = lambda t: self.tok.decode([int(t)])  # noqa: E731
                kk = min(k, int(live[j].sum()))
                fall, rise, top = down[j].topk(kk, largest=False).indices, up[j].topk(kk).indices, p0[j].topk(k).indices
                out.append({"falls": [(tok(t), float(p0[j, t]), float(p1[j, t])) for t in fall],
                            "rises": [(tok(t), float(p0[j, t]), float(p1[j, t])) for t in rise],
                            "top": [(tok(t), float(p0[j, t])) for t in top], "tv": float(d[j].abs().sum() / 2)})
        return out

    def steered(self, layer: int, index: int, value: float | None, members: list | None = None) -> str:
        """The model's greedy continuations of PROMPTS with the neuron's activation set to value at every position
        (None: unsteered), joined."""
        torch = self.torch
        if value is None and getattr(self, "_plain", None):  # the unsteered continuations: the same for every neuron
            return self._plain
        hs = []
        if value is not None:  # a group: every member set to its sign times STEER of its own maximum
            for l, i, v in ([(m[0], m[1], m[2] * STEER * m[3]) for m in members] if members else [(layer, index, value)]):
                def pre(_m, a, i=i, v=v):
                    x = a[0].clone()
                    x[..., i] = v
                    return (x,) + tuple(a[1:])
                hs.append(self.layers[l].mlp.down_proj.register_forward_pre_hook(pre))
        try:
            self.tok.padding_side = "left"  # the prompts in one batch, each continuing from its own last token
            ids = self.tok(PROMPTS, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            with torch.no_grad():
                g = self.m.generate(**ids, max_new_tokens=GEN, do_sample=False, temperature=None, top_p=None, top_k=None,
                                    pad_token_id=self.tok.pad_token_id)
            outs = [p + self.tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True) for p, x in zip(PROMPTS, g)]
        finally:
            self.tok.padding_side = "right"
            for h in hs:
                h.remove()
        if value is None:
            self._plain = " / ".join(outs)
        return " / ".join(outs)


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for body in (m.group(0), re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", m.group(0))):  # then a lone backslash read literally
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass
    return None


def _item(tg, it: dict) -> dict:
    """Validate every item before reader or target execution; no Python negative indices."""
    if not isinstance(it, dict):
        raise ValueError("neuron item must be an object")
    if it.get("members") is not None:
        return _group_item(tg, it)
    for key in ("layer", "index"):
        if type(it.get(key)) is not int or it[key] < 0:
            raise ValueError(f"{key} must be a nonnegative native integer")
    if it.get("pole") not in ("+", "-"):
        raise ValueError("pole must be + or -")
    cm=it.get("corpus_max")
    if isinstance(cm,bool) or not isinstance(cm,numbers.Real) or not math.isfinite(cm) or cm<=0:
        raise ValueError("corpus_max must be finite and positive")
    if not math.isfinite(STEER*cm):
        raise ValueError("steering amplitude must be finite")
    if not isinstance(it.get("explanation"),str) or not it["explanation"].strip():
        raise ValueError("explanation must be nonempty text")
    if it["layer"]>=len(tg.layers):
        raise ValueError("layer outside target architecture")
    mlp=tg.layers[it["layer"]].mlp
    width=mlp.down_proj.in_features
    if it["index"]>=width:
        raise ValueError("neuron index outside target down_proj input")
    return {"requested_model":getattr(tg,"name",None),"native_layer":it["layer"],"native_index":it["index"],
            "native_width":width,"layers":len(tg.layers),"swiglu_architecture_verified":getattr(tg,"architecture_verified",False),"unit":"post_silu_gate_times_up / pre_down_proj",
            "identity_scope":"requested model and runtime architecture; checkpoint weights/revision not attested"}


def _group_item(tg, it: dict) -> dict:
    ms = it["members"]
    if not isinstance(ms, list) or not ms:
        raise ValueError("a group needs members [layer, index, sign, corpus maximum]")
    for m in ms:
        if (not isinstance(m, (list, tuple)) or len(m) != 4 or any(type(x) is not int for x in m[:3]) or m[2] not in (1, -1)
                or not isinstance(m[3], numbers.Real) or not math.isfinite(m[3]) or m[3] <= 0):
            raise ValueError("group member must be [layer, index, sign +-1, positive corpus maximum]")
        if not 0 <= m[0] < len(tg.layers) or not 0 <= m[1] < tg.layers[m[0]].mlp.down_proj.in_features:
            raise ValueError("group member outside target architecture")
    if it.get("pole") != "+":
        raise ValueError("a group has one side: pole +")
    cm = it.get("corpus_max")
    if isinstance(cm, bool) or not isinstance(cm, numbers.Real) or not math.isfinite(cm) or cm <= 0:
        raise ValueError("corpus_max must be finite and positive")
    if not isinstance(it.get("explanation"), str) or not it["explanation"].strip():
        raise ValueError("explanation must be nonempty text")
    return {"requested_model": getattr(tg, "name", None), "members": [list(m) for m in ms], "layers": len(tg.layers),
            "unit": "group: mean over members of value*sign/member corpus maximum, clipped at 0"}


def _proposal(plan):
    if not isinstance(plan,dict):
        raise ValueError("reader proposal must be a JSON object")
    groups={}
    seen=set()
    for k,n in (("max_texts",BLIND),("near_misses",NEAR),("distant_texts",DISTANT)):
        xs=plan.get(k)
        if not isinstance(xs,list) or len(xs)!=n:
            raise ValueError(f"{k} requires exactly {n} texts")
        for x in xs:
            if not isinstance(x,str) or not x.strip():
                raise ValueError(f"{k} must contain nonempty strings")
            norm=" ".join(x.split()).casefold()
            if norm in seen:
                raise ValueError("all proposed test texts must be distinct")
            seen.add(norm)
        groups[k]=xs
    for k in ("distant_concept","steering_prediction"):
        if not isinstance(plan.get(k),str) or not plan[k].strip():
            raise ValueError(f"{k} must be nonempty text")
    return groups


def _responses(rd, asks, max_tokens, **sampling):
    replies=rd(asks,max_tokens=max_tokens,**sampling)
    if not isinstance(replies,(list,tuple)) or len(replies)!=len(asks) or any(not isinstance(x,str) for x in replies):
        raise ValueError("reader response count/type must exactly match requests")
    return list(replies)


def plan(rd, items: list[dict], length: str = "short", temperature: float = 0.0, seed: int | None = None) -> list[dict]:
    """The reader's test texts for each item (ASK, blind texts of BLIND_LEN[length]), parsed and validated (_proposal); an
    invalid proposal stays an explicit failure that cannot earn T1 credit. temperature > 0 with a seed draws a fresh set
    (texts the explanation's writer never saw)."""
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    asked=[it for it in items if it.get("writer_valid") is not False]  # t1_audit explain's failures get no reader call
    asks=[(ASK_GROUP if it.get("members") else ASK).format(pole="positive" if it["pole"]=="+" else "negative",explanation=it["explanation"],
                     blind=BLIND,blind_len=BLIND_LEN[length],near=NEAR,distant=DISTANT) for it in asked]
    sampling={"temperature":temperature,"seed":seed} if temperature else {}
    replies=iter(zip(asks,_responses(rd,asks,1500,**sampling) if asks else []))
    rows=[]
    for it in items:
        if it.get("writer_valid") is False:
            rows.append({**it,"valid":False,"passes_t1":False,"reason":"invalid writer input: "+", ".join(it.get("writer_errors") or ["unstated"]),
                         "diagnostic_scope":"T1 activation prediction; not causal mechanism certification","causal_mechanism_verified":False})
            continue
        ask,raw=next(replies)
        prop=parse_json(raw)
        row={k:it.get(k) for k in ("layer","index","pole","corpus_max","explanation")}
        if it.get("members"):row.update(members=it["members"],group=it.get("group"))
        row.update({"reader_class":type(rd).__name__,"reader_model":getattr(rd,"base",getattr(rd,"model",None)),
                    "verifier_source_sha256":source_sha256,"raw_proposal":raw,"proposal":prop,
                    "proposal_request":ask,"blind_length":length,"plan_sampling":sampling or None,
                    "diagnostic_scope":"T1 activation prediction; not causal mechanism certification",
                    "causal_mechanism_verified":False,"passes_t1":False})
        try:row["test_texts"]=_proposal(prop);row["valid"]=True
        except ValueError as e:row.update({"valid":False,"reason":str(e)})
        rows.append(row)
    return rows


def measure(tg: Target, rows: list[dict], seed: int = 0) -> list[dict]:
    """The target's side of each planned row: every test text's pole value over the corpus maximum, blind maximum,
    near-miss and distant rates, the steered and unsteered continuations, and the steering judge's question (A/B order
    seeded) with its expected letter."""
    out=[];rng=random.Random(seed)
    for row in rows:
        if row.get("writer_valid") is False:  # no explanation or no corpus maximum to test: nothing to bind or measure
            out.append(row);continue
        row={**row,"model_binding":_item(tg,row)}
        if not row.get("valid"):
            out.append(row);continue
        groups=row["test_texts"];key="pos" if row["pole"]=="+" else "neg";cm=float(row["corpus_max"])
        flat=[x for xs in groups.values() for x in xs]
        vals=tg.group(flat,row["members"]) if row.get("members") else tg.poles(flat,row["layer"],row["index"])
        if not isinstance(vals,list) or len(vals)!=len(flat):
            raise ValueError("target pole result count must match all test texts")
        measured=[]
        for v in vals:
            value=v.get(key) if isinstance(v,dict) else None
            if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(value):
                raise ValueError("target pole measurements must be finite")
            ratio=value/cm
            if not math.isfinite(ratio):
                raise ValueError("normalized pole measurement must be finite")
            measured.append(ratio)
        got={};k0=0
        for k,xs in groups.items():got[k]=measured[k0:k0+len(xs)];k0+=len(xs)
        blind=max(got["max_texts"]);near=sum(x>=FIRE for x in got["near_misses"])/NEAR
        dist=sum(x>=FIRE for x in got["distant_texts"])/DISTANT
        value=(1 if row["pole"]=="+" else -1)*STEER*cm
        if not math.isfinite(value):raise ValueError("steering amplitude must be finite")
        on=tg.steered(None,None,value,row["members"]) if row.get("members") else tg.steered(row["layer"],row["index"],value)
        off=tg.steered(row.get("layer"),row.get("index"),None)
        flip=rng.random()<.5
        prop=row["proposal"]
        out.append({**row,"raw_pole_measurements":vals,"blind_max":blind,"blind_texts":groups["max_texts"],
                    "blind_values":got["max_texts"],"near_miss_rate":near,"near_values":got["near_misses"],
                    "distant_concept":prop["distant_concept"],"distant_rate":dist,"distant_values":got["distant_texts"],
                    "steering_prediction":prop["steering_prediction"],"steered":on,"unsteered":off,
                    "steer_request":JUDGE.format(prediction=prop["steering_prediction"],a=off if flip else on,b=on if flip else off),
                    "steer_expected":"B" if flip else "A","passes_t1":blind>=FIRE and near<FIRE})
    return out


def judge(rd, rows: list[dict]) -> list[dict]:
    """The reader's steering choice for each measured row (one letter, the last line's A or B)."""
    asked=[r for r in rows if r.get("valid") and r.get("steer_request")]
    responses=_responses(rd,[r["steer_request"] for r in asked],40) if asked else []  # 5 tokens cut the reader off mid-sentence on 19 of 19
    got={id(r):raw for r,raw in zip(asked,responses)}
    out=[]
    for r in rows:
        if id(r) in got:
            raw=got[id(r)]
            # Reasoning may precede a terminal answer line; prose articles are not votes.
            choice=re.search(r"(?:^|\n)\s*([AB])[.!]?\s*$",raw.strip(),re.I)
            ans=choice[1].upper() if choice else ""
            r={**r,"steer_match":float(ans==r["steer_expected"]),"steer_response_valid":ans in ("A","B"),"steer_raw_response":raw}
        out.append(r)
    return out


def verify(tg: Target, rd, items: list[dict], seed: int = 0) -> list[dict]:
    """Complete blinded activation diagnostics and steering test; never a causal mechanism certificate: plan, measure
    and judge on one host. Invalid proposals remain explicit failures and cannot earn T1 credit. Response cardinality
    errors raise rather than silently dropping items. No route claim is tested."""
    for it in items:
        _item(tg,it)
    return judge(rd,measure(tg,plan(rd,items),seed))


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _dump(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def cmd_plan(a) -> None:
    """Step 1 of 3 on the machine with the reader's key: the reader's test texts for each pole (--neurons) to --out."""
    import reader_reward

    rd=reader_reward.Reader(a.reader,a.out.parent)
    rows=plan(rd,_load(a.neurons))
    _dump(a.out,rows)
    print(f"{sum(r['valid'] for r in rows)}/{len(rows)} valid plans to {a.out}; spend ${rd.spend.total:.3f}")


def cmd_measure(a) -> None:
    """Step 2 of 3 where the target runs (a MATS GPU or a pod; no reader key): --plans to --out."""
    rows=remote_measure(a.url,_load(a.plans)) if a.url else measure(Target(a.model,a.device),_load(a.plans))
    _dump(a.out,rows)
    print(f"{len(rows)} rows measured to {a.out}")


def cmd_serve(a) -> None:
    """measure as a service where the target runs (a pod, no reader key), for a writer loop on the Mac reached through
    an SSH tunnel: POST /measure {"rows": planned rows[, "seed"]} -> {"rows": measured rows}; POST /effects {"items":
    [{"contexts", "layer", "index", "pole"[, "k"]}]} -> {"effects": Target.effects per item}; GET /health."""
    import threading
    import traceback
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    tg=Target(a.model,a.device)
    lock=threading.Lock()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self._send({"ok":True,"model":a.model})

        def do_POST(self):
            req=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                with lock:
                    if self.path.rstrip("/").endswith("/effects"):  # {"items": [{"contexts", "layer", "index", "pole"}]}
                        out={"effects":[tg.effects(it["contexts"],it.get("layer"),it.get("index"),it.get("pole","+"),int(it.get("k",10)),it.get("members"))
                                        for it in req["items"]]}
                    else:
                        out={"rows":measure(tg,req["rows"],int(req.get("seed",0)))}
            except Exception as e:
                traceback.print_exc()
                out={"error":f"{type(e).__name__}: {e}"[:500]}
            self._send(out)

        def _send(self,obj):
            body=json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self,*args):
            pass

    print(f"neuron_verify measure on :{a.port} ({a.model})",flush=True)
    ThreadingHTTPServer(("0.0.0.0",a.port),H).serve_forever()


def remote_effects(url: str, items: list[dict]) -> list[list[dict]]:
    """Target.effects through cmd_serve at url, one list of context effects per item."""
    import urllib.request

    req=urllib.request.Request(url.rstrip("/")+"/effects",data=json.dumps({"items":items}).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=3600) as r:
        got=json.loads(r.read())
    if "error" in got:
        raise RuntimeError(got["error"])
    return got["effects"]


def remote_measure(url: str, rows: list[dict], seed: int = 0) -> list[dict]:
    """measure through cmd_serve at url (http://localhost:PORT over the tunnel)."""
    import urllib.request

    req=urllib.request.Request(url.rstrip("/")+"/measure",data=json.dumps({"rows":rows,"seed":seed}).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=3600) as r:
        got=json.loads(r.read())
    if "error" in got:
        raise RuntimeError(got["error"])
    return got["rows"]


def cmd_judge(a) -> None:
    """Step 3 of 3 on the machine with the reader's key: the steering choices for --measured to --out."""
    import reader_reward

    rd=reader_reward.Reader(a.reader,a.out.parent)
    rows=judge(rd,_load(a.measured))
    _dump(a.out,rows)
    print(f"{len(rows)} rows judged to {a.out}; spend ${rd.spend.total:.3f}")


def cmd_verify(a) -> None:
    import reader_reward

    items = [json.loads(x) for x in a.neurons.read_text().splitlines()]
    tg = Target(a.model, a.device)
    candidates, invalid = [], {}
    seen = set()
    for position, it in enumerate(items):
        # Writer failures may lack usable maxima/prose, but never waive unit identity.
        if (not isinstance(it, dict) or any(type(it.get(k)) is not int or it[k] < 0 for k in ("layer", "index"))
                or it.get("pole") not in ("+", "-")):
            raise ValueError("invalid sampled neuron/pole address")
        key = (it["layer"], it["index"], it["pole"])
        if key in seen:
            raise ValueError("duplicate sampled neuron/pole address")
        seen.add(key)
        if it["layer"] >= len(tg.layers) or it["index"] >= tg.layers[it["layer"]].mlp.down_proj.in_features:
            raise ValueError("sampled neuron outside target architecture")
        if "writer_valid" in it and type(it["writer_valid"]) is not bool:
            raise ValueError("writer_valid must be boolean")
        if it.get("writer_valid") is False:
            errors = it.get("writer_errors")
            if not isinstance(errors, list) or not errors or any(not isinstance(e, str) or not e for e in errors):
                raise ValueError("declared writer failure requires explicit reasons")
            invalid[position] = {**it, "valid": False, "passes_t1": False,
                                 "reason": "invalid writer input: " + ", ".join(errors),
                                 "diagnostic_scope": "T1 activation prediction; not causal mechanism certification",
                                 "causal_mechanism_verified": False}
        else:
            _item(tg, it)  # strict validation remains before reader service construction
            candidates.append(it)
    rd = reader_reward.Reader(a.reader, a.out.parent) if candidates else None
    checked = verify(tg, rd, candidates) if candidates else []
    if len(checked) != len(candidates):
        raise ValueError("verifier result count must match every writer-valid pole")
    results = iter(checked)
    rows = [invalid[i] if i in invalid else {**it, **next(results)} for i, it in enumerate(items)]
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if r.get("valid")]
    n = max(len(rows), 1)
    n_valid = max(len(ok), 1)
    print(f"{len(ok)}/{len(rows)} valid; blind max >= {FIRE}: {sum(r['blind_max'] >= FIRE for r in ok) / n:.2f}; near-miss rate "
          f"{sum(r['near_miss_rate'] for r in ok) / n_valid:.2f} (valid only); distant rate {sum(r['distant_rate'] for r in ok) / n_valid:.2f} (valid only); steering match "
          f"{sum(r.get('steer_match', 0) for r in ok) / n:.2f}; T1 pass {sum(r['passes_t1'] for r in ok) / n:.2f}; spend ${rd.spend.total if rd is not None else 0.0:.3f}")


def cmd_summary(a) -> None:
    """T1's numbers from verify's rows (pre-registered 10-15): the share of explained poles passing (blind maximum >=
    FIRE and near misses < FIRE), both poles pooled; the positive and negative poles apart; neurons whose poles both
    pass; and the comparison with Cammarata's Qwen3-32B (76% blind maximum, 11% near misses, 1% distant)."""
    rows = [json.loads(x) for f in a.rows for x in f.read_text().splitlines()]
    silent = [r for r in rows if not (isinstance(r.get("corpus_max"), numbers.Real) and r["corpus_max"] > 0)]
    rows = [r for r in rows if r not in silent]  # pre-registered: a pole that never fires on the corpus has nothing to explain
    print(f"poles with no corpus activation (left out): {len(silent)}")
    def stats(rs, name):
        # Primary denominator is every submitted pole, including invalid proposals.
        measured=[r for r in rs if r.get("valid")]
        n=max(len(rs),1)
        n_valid=max(len(measured),1)
        passes=sum(bool(r.get("valid") and r.get("passes_t1")) for r in rs)
        blind=sum(r["blind_max"]>=FIRE for r in measured)
        median=sorted(r["blind_max"] for r in measured)[len(measured)//2] if measured else None
        print(f"{name}: {len(rs)} submitted poles; {len(measured)} valid; pass {passes/n:.2f}; blind max >= {FIRE} {blind/n:.2f}; "
              f"valid-only median blind {median}; near misses (valid only) {sum(r['near_miss_rate'] for r in measured)/n_valid:.2f}; "
              f"distant (valid only) {sum(r['distant_rate'] for r in measured)/n_valid:.2f}; "
              f"steering match (submitted denominator) {sum(r.get('steer_match',0) for r in measured)/n:.2f}")
    stats(rows,"both poles (primary)")
    stats([r for r in rows if r["pole"]=="+"],"positive pole")
    stats([r for r in rows if r["pole"]=="-"],"negative pole")
    by={}
    for r in rows:
        by.setdefault((r["layer"],r["index"]),{})[r["pole"]]=bool(r.get("valid") and r.get("passes_t1"))
    both=[v for v in by.values() if set(v)=={"+","-"}]
    print(f"neurons with both poles submitted: {len(both)}; both pass {sum(all(v.values()) for v in both)/max(len(both),1):.2f}; "
          f"invalid reader plans {sum(not r.get('valid') for r in rows)}; T1 is an activation diagnostic, not a causal mechanism certificate")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--neurons", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--model", default="Qwen/Qwen3-1.7B")
    v.add_argument("--reader", default=READER)
    v.add_argument("--device")
    sm = sub.add_parser("summary")
    sm.add_argument("rows", type=Path, nargs="+")
    pl = sub.add_parser("plan")
    pl.add_argument("--neurons", type=Path, required=True)
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--reader", default=READER)
    me = sub.add_parser("measure")
    me.add_argument("--plans", type=Path, required=True)
    me.add_argument("--out", type=Path, required=True)
    me.add_argument("--model", default="Qwen/Qwen3-1.7B")
    me.add_argument("--device", help="cuda, mps, cpu, or auto (layers over every visible GPU)")
    me.add_argument("--url", help="measure through neuron_verify serve at this URL (a pod over a tunnel), not here")
    sv = sub.add_parser("serve")
    sv.add_argument("--model", default="Qwen/Qwen3-8B")
    sv.add_argument("--device", help="cuda, mps, cpu, or auto (layers over every visible GPU)")
    sv.add_argument("--port", type=int, default=8012)
    ju = sub.add_parser("judge")
    ju.add_argument("--measured", type=Path, required=True)
    ju.add_argument("--out", type=Path, required=True)
    ju.add_argument("--reader", default=READER)
    a = ap.parse_args()
    {"verify": cmd_verify, "summary": cmd_summary, "plan": cmd_plan, "measure": cmd_measure, "judge": cmd_judge,
     "serve": cmd_serve}[a.what](a)


if __name__ == "__main__":
    main()
