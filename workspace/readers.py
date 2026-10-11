"""Activation readers scored without judges at every cell (#2951 WorkspaceBench).

Logit lens: the final norm and unembedding applied to a cell's residual vector; the read is the
top-10 tokens. A number is named when a top-10 token is its first written token (its first digit, or
'-'), or the first token of its English or Chinese numeral; a word is named when a top-10 token is
the whole word (WorkspaceBench's strict judge: word fragments do not count).

Oracle lens (stage 2; needs Qwen3.6-27B and the agu18dec/olens_and_ar LoRA `olens_s3d_rl600`): the
cell's vector, captured with the adapter off, is injected as 16000 h/|h| into the embedding row of a
one-token marker inside the verbalizer prompt below, and the adapter writes '- ' bullets. Layers 20-60
step 4 only. A numeric target is named when any number in a sample equals it.
"""
import re

import torch

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
CN = "零一二三四五六七八九"


def english(n):
    n = abs(int(n))
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + english(n % 100))
    return str(n)


def chinese(n):
    n = abs(int(n))
    if n < 10:
        return CN[n]
    if n < 20:
        return "十" + ("" if n == 10 else CN[n - 10])
    if n < 100:
        return CN[n // 10] + "十" + ("" if n % 10 == 0 else CN[n % 10])
    if n < 1000:
        h, r = divmod(n, 100)
        s = CN[h] + "百"
        if r == 0:
            return s
        if r < 10:
            return s + "零" + CN[r]
        return s + CN[r // 10] + "十" + ("" if r % 10 == 0 else CN[r % 10])
    return str(n)


def number_names(tok, value):
    """Token ids that name `value` as the first token of a way to write it: its digits (first digit
    or '-'), its English word (with and without a leading space or capital), its Chinese numeral."""
    s = str(int(value))
    ids = {tok(s, add_special_tokens=False)["input_ids"][0]}
    w = english(value).split("-")[0].split(" ")[0]
    for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        t = tok(form, add_special_tokens=False)["input_ids"]
        if len(tok.decode(t[:1]).strip()) >= 3:
            ids.add(t[0])
    c = tok(chinese(value), add_special_tokens=False)["input_ids"]
    if len(tok.decode(c[:1]).strip()) >= 2:
        ids.add(c[0])
    return ids


def word_names(tok, word):
    """Token ids that are the whole word (any case, with or without a leading space)."""
    ids = set()
    for form in {word, word.lower(), word.capitalize()}:
        for f in (form, " " + form):
            t = tok(f, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
    return ids


@torch.no_grad()
def logit_lens(patcher, trace, k=10, rows=512):
    """Top-k token ids [L+1, T, k] of the unembedded residual stream at every cell of a run."""
    h = trace.h.reshape(-1, trace.h.shape[-1])
    out = torch.cat([torch.topk(patcher.logits(c), k, -1).indices for c in torch.split(h, rows)])
    return out.reshape(trace.h.shape[0], trace.h.shape[1], k).cpu()


# ---- stage 2: the oracle lens ------------------------------------------------------------------

OLENS_REPO, OLENS_ADAPTER = "agu18dec/olens_and_ar", "olens_s3d_rl600"
OLENS_LAYERS = tuple(range(20, 61, 4))
OLENS_ALPHA = 16000.0
# The model card's `concepts_raw` text. Its line breaks follow the card's code block; check them
# against the source renderer (global_workspace.ola.verbalizer.renderer_for("concepts_raw")) before
# the first stage-2 run.
OLENS_PROMPT = ("An activation vector from layer {layer} of a language model is enclosed in activation tags:\n"
                "<activation>{char}</activation>. Produce distinct concepts that encode this activation, each as\n"
                "a '- ' bullet on its own line.")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def olens_marker(tok, layer=44):
    """The first character in U+3200..U+33FF that stays one token inside the rendered prompt."""
    for cp in range(0x3200, 0x3400):
        ch = chr(cp)
        text = tok.apply_chat_template([{"role": "user", "content": OLENS_PROMPT.format(layer=layer, char=ch)}],
                                       add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        one = tok(ch, add_special_tokens=False)["input_ids"]
        if len(one) == 1 and ids.count(one[0]) == 1:
            return ch, one[0]
    raise ValueError("no one-token marker in U+3200..U+33FF")


class OracleLens:
    """The AO adapter on its base model. read(layer, vecs) samples k bullet lists per vector."""

    def __init__(self, base_model, tok, device):
        from peft import PeftModel
        self.tok, self.device = tok, device
        self.model = PeftModel.from_pretrained(base_model, OLENS_REPO, subfolder=OLENS_ADAPTER).eval()
        self.char, self.char_id = olens_marker(tok)

    @torch.no_grad()
    def capture(self, ids):
        """The base model's residual stream [L+1, T, d] for ids, with the adapter off."""
        from interchange import HookPatcher
        with self.model.disable_adapter():
            return HookPatcher(self.model.get_base_model()).states(ids)

    def prompt(self, layer):
        text = self.tok.apply_chat_template([{"role": "user", "content": OLENS_PROMPT.format(layer=layer, char=self.char)}],
                                            add_generation_prompt=True, enable_thinking=False, tokenize=False)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids, ids.index(self.char_id)

    @torch.no_grad()
    def read(self, layer, vecs, k=10, max_new=40, temperature=0.8, top_p=0.95, seed=0):
        assert layer in OLENS_LAYERS, layer
        ids, slot = self.prompt(layer)
        ids = torch.tensor(ids, device=self.device)
        emb = self.model.get_input_embeddings()(ids)[None].repeat(len(vecs) * k, 1, 1)
        v = vecs.to(emb.dtype).to(self.device)
        emb[:, slot] = (OLENS_ALPHA * v / v.norm(dim=-1, keepdim=True)).repeat_interleave(k, 0)
        torch.manual_seed(seed)
        out = self.model.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], device=self.device, dtype=torch.long),
                                  max_new_tokens=max_new, do_sample=True, temperature=temperature, top_p=top_p)
        text = self.tok.batch_decode(out, skip_special_tokens=True)
        return [text[i * k:(i + 1) * k] for i in range(len(vecs))]


def names_number(texts, value):
    """Any sample asserting the number (WorkspaceBench's quantity extraction, exact match)."""
    for t in texts:
        for q in NUMBER.findall(t):
            q = q.replace(",", "").rstrip(".")
            try:
                if float(q) == float(value):
                    return True
            except ValueError:
                pass
    return False


def names_word(texts, word):
    pat = re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", re.I)
    return any(pat.search(t) for t in texts)
