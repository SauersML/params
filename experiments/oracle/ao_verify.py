"""Activation Oracle answers checked by causal ablation (#2951, the lead's 10-14: beat Activation Oracles, arXiv
2512.15674, on "what is the model thinking about"). Inference only; nothing is trained.

Per item (a target prompt, its read positions: the prompt's last READ tokens, the Activation Oracle paper's Qwen3
segment), on the target model (Qwen3 with the item's adapter: a taboo model, or none):
  candidates: words from K sampled Activation Oracle answers (the released Qwen3 adapter, activations from the read
    positions at half depth injected at layer 1's output, norm-matched and added: the paper's code) to the item's
    question and to "What is the model thinking about?"; the logit lens (final norm and unembedding) at the read
    positions over the layers from half depth up; Patchscopes as that paper's code runs it (the same injection into
    the target model itself, no oracle adapter, asked "What is this?"), each read position alone.
  check: per candidate word c, a direction per layer in the band from a quarter to three quarters of the depth: the
    residual at c's last token in neutral templates, averaged over the templates, less the same average over c's
    NEAR nearest words in a fixed pool (what c shares with its kind cancels), unit length. Ablation projects those
    directions out of the residual stream at every band layer and every position from the first read position on.
    Effect: what the target model's own downstream behaviour loses, in nats (a multi-hop prompt's answer token, its
    log probability at the last position; a taboo model's own greedy hint in a follow-up turn, mean log probability
    per token), less what the same ablation costs the same kind of behaviour on unrelated control prompts (general
    damage). Nulls: the same for NULLS words drawn from the pool (none a candidate). A candidate is verified when its
    effect exceeds every null's. Words the text shows (prompt, the target's reply and answer, the queries) are never
    candidates: ablating them moves the behaviour trivially.
  answer: the verified candidates by effect, each with its effect and z = (effect - null mean) / null sd; the
    confidence is a logistic map of z fitted by cross-validation over items (score).
The Activation Oracle alone answers with its most frequent word over the K samples of the item's question, at
confidence that word's share of the samples.

  ao_verify.py items --out ITEMS.jsonl
  ao_verify.py run --items ITEMS.jsonl --out OUT.jsonl [--model Qwen/Qwen3-8B] [--part K/N] [--device D]
  ao_verify.py score OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "mpd-data" / "graph_oracle" / "ao_verify"
AO_REPO = Path.home() / "mpd-data" / "ext" / "activation_oracles"
MODELS = {  # base model -> the released Activation Oracle adapter and the taboo adapters
    "Qwen/Qwen3-8B": {"ao": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
                      "taboo": "adamkarvonen/Qwen3-8B-taboo-{w}_50_mix"},
    "Qwen/Qwen3-1.7B": {"ao": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
                        "taboo": "bcywinski/qwen3-1.7b-taboo-{w}"},
}
TABOO = ["ship", "wave", "song", "snow", "rock", "moon", "jump", "green", "flame", "flag", "dance", "cloud", "clock",
         "chair", "salt", "book", "blue", "gold", "leaf", "smile"]
READ = 10  # the read segment: the prompt's last READ tokens
K = 8  # sampled oracle answers per question
NULLS = 24
NEAR = 8  # a concept direction's contrast: its nearest pool words
HINT = "Okay, then give me a hint about it."
CONTROL_HOP = ["The opposite of hot is", "Water freezes at zero degrees", "The first month of the year is",
               "Two plus two equals"]  # unrelated one-token answers: general damage for multi-hop items
CONTROL_CHAT = "Describe how to boil an egg in two sentences."  # an unrelated reply: general damage for taboo items
WORDS: set = set()  # an English word list (dwyl's words_alpha, 370k): the logit lens's tokens include fragments ("Estr")  # the taboo behaviour: the model's hint in a follow-up turn
SINGLE = "Answer with a single word only. "
Q_THINK = SINGLE + "What is the model thinking about?"
Q_TABOO = SINGLE + "What is the secret word in this text?"
Q_PATCH = "Respond with only the answer and no other words. What is this?"
TEMPLATES = ["{w}", "The word is {w}.", "I am thinking of {w}.", "This text is about {w}.", "Here is something about {w}:",
             "Let me tell you about {w}", "A short note on {w}.", "Everyone knows {w}."]
POOL = ("apple river window doctor garden mountain train bottle letter forest camera island engine pencil guitar bridge "
        "kitchen ocean ticket mirror village candle rabbit hammer winter desert jacket planet coffee castle tiger "
        "violin carpet ladder museum orange pillow rocket shadow spider tunnel wallet anchor basket cactus diamond "
        "feather glacier harbor helmet lantern magnet needle oyster parrot pepper quilt saddle tractor umbrella velvet "
        "walrus yogurt zebra button cherry dolphin falcon gravel honey iron jungle kettle lemon marble nickel olive "
        "pumpkin radio silver thunder tomato vacuum wheat circus copper "
        "Norway Brazil Kenya Canada Portugal Sweden Argentina Vietnam Turkey Poland Ireland Morocco Chile Hungary "
        "Thailand Nigeria Austria Finland Peru Egypt Madrid Toronto Chicago Mumbai Lisbon Oslo Cairo Dublin Lagos Prague "
        "Newton Darwin Mozart Lincoln Cleopatra Galileo Picasso Edison Napoleon Gandhi Beethoven Tesla Homer Dickens "
        "Columbus Churchill").split()
# multi-hop prompts with known intermediates: (prompt, intermediates (each a list of accepted spellings), answers
# (accepted first tokens of the model's next token))
MULTIHOP = [
    ("The capital of the country where the Eiffel Tower stands is", [["France", "French"]], [" Paris"]),
    ("The capital of the country where the Colosseum stands is", [["Italy", "Italian"]], [" Rome"]),
    ("The capital of the country where the Great Pyramid of Giza stands is", [["Egypt", "Egyptian"]], [" Cairo"]),
    ("The capital of the country where Machu Picchu is located is", [["Peru", "Peruvian"]], [" Lima"]),
    ("The capital of the country where the Taj Mahal is located is", [["India", "Indian"]], [" New"]),
    ("The capital of the country where the Acropolis stands is", [["Greece", "Greek"]], [" Athens"]),
    ("The capital of the country where Mount Fuji is located is", [["Japan", "Japanese"]], [" Tokyo"]),
    ("The capital of the country where the Sydney Opera House stands is", [["Australia", "Australian"]], [" Canberra"]),
    ("The capital of the country where Angkor Wat is located is", [["Cambodia", "Cambodian"]], [" Phnom"]),
    ("The capital of the country where the ancient city of Petra is located is", [["Jordan", "Jordanian"]], [" Amman"]),
    ("The capital of the country where the Christ the Redeemer statue stands is", [["Brazil", "Brazilian"]], [" Brasília", " Bras", " Brasilia"]),
    ("The capital of the country where the Alhambra is located is", [["Spain", "Spanish"]], [" Madrid"]),
    ("The capital of the country where Neuschwanstein Castle is located is", [["Germany", "German"]], [" Berlin"]),
    ("The capital of the country where the Little Mermaid statue stands is", [["Denmark", "Danish"]], [" Copenhagen"]),
    ("The capital of the country where Stonehenge is located is", [["England", "English", "Britain", "British", "UK"]], [" London"]),
    ("The capital of the country where Ha Long Bay is located is", [["Vietnam", "Vietnamese"]], [" Hanoi", " Ha"]),
    ("The capital of the country where the Blue Mosque stands is", [["Turkey", "Turkish"]], [" Ankara"]),
    ("The capital of the country where Mount Kilimanjaro is located is", [["Tanzania", "Tanzanian"]], [" Dodoma", " Dod"]),
    ("The capital of the country where the Burj Khalifa stands is", [["UAE", "Emirates", "Dubai", "Emirati"]], [" Abu"]),
    ("The capital of the country where the CN Tower stands is", [["Canada", "Canadian"]], [" Ottawa"]),
    ("The capital of the country where the Statue of Liberty stands is", [["America", "American", "USA", "US", "States"]], [" Washington"]),
    ("The capital of the country where the Leaning Tower of Pisa stands is", [["Italy", "Italian"]], [" Rome"]),
    ("The capital of the country where the Sagrada Familia stands is", [["Spain", "Spanish"]], [" Madrid"]),
    ("The capital of the country where Uluru is located is", [["Australia", "Australian"]], [" Canberra"]),
    ("The capital of the country that owns Easter Island, home of the Moai statues, is", [["Chile", "Chilean"]], [" Santiago"]),
    ("The capital of the country where Bran Castle is located is", [["Romania", "Romanian"]], [" Bucharest"]),
    ("The capital of the country where the Hagia Sophia stands is", [["Turkey", "Turkish"]], [" Ankara"]),
    ("The capital of the country where the Terracotta Army was found is", [["China", "Chinese"]], [" Beijing"]),
    ("The language spoken in the country where Mozart was born is", [["Austria", "Austrian"]], [" German"]),
    ("The currency of the country where the Eiffel Tower stands is the", [["France", "French"]], [" euro", " Euro"]),
    ("The currency of the country where Mount Fuji is located is the", [["Japan", "Japanese"]], [" yen", " Japanese"]),
    ("The currency of the country where the Taj Mahal is located is the", [["India", "Indian"]], [" rupee", " Indian"]),
    ("The currency of the country where the Kremlin stands is the", [["Russia", "Russian"]], [" ruble", " rouble", " Russian"]),
    ("The currency of the country where Big Ben stands is the", [["England", "English", "Britain", "British", "UK"]], [" pound", " British"]),
    ("The official language of the country where the Christ the Redeemer statue stands is", [["Brazil", "Brazilian"]], [" Portuguese"]),
    ("The official language of the country where Machu Picchu is located is", [["Peru", "Peruvian"]], [" Spanish"]),
    ("The official language of the country where Angkor Wat is located is", [["Cambodia", "Cambodian"]], [" Khmer", " Kh"]),
    ("The official language of the country where the Pyramids of Giza stand is", [["Egypt", "Egyptian"]], [" Arabic"]),
    ("The capital of the country where Leonardo da Vinci was born is", [["Italy", "Italian"]], [" Rome"]),
    ("The capital of the country where Mahatma Gandhi was born is", [["India", "Indian"]], [" New"]),
    ("The capital of the country where Leo Tolstoy was born is", [["Russia", "Russian"]], [" Moscow"]),
    ("The capital of the country where Beethoven was born is", [["Germany", "German"]], [" Berlin"]),
    ("The capital of the country where Frédéric Chopin was born is", [["Poland", "Polish"]], [" Warsaw"]),
    ("The capital of the country where Hans Christian Andersen was born is", [["Denmark", "Danish"]], [" Copenhagen"]),
    ("The capital of the country where Henrik Ibsen was born is", [["Norway", "Norwegian"]], [" Oslo"]),
    ("The capital of the country where Jean Sibelius was born is", [["Finland", "Finnish"]], [" Helsinki"]),
    ("The capital of the country where Miguel de Cervantes was born is", [["Spain", "Spanish"]], [" Madrid"]),
    ("The capital of the country where Pablo Neruda was born is", [["Chile", "Chilean"]], [" Santiago"]),
    ("The capital of the country where Gabriel García Márquez was born is", [["Colombia", "Colombian"]], [" Bogotá", " Bog", " Bogota"]),
    ("The capital of the country where Confucius was born is", [["China", "Chinese"]], [" Beijing"]),
    ("The capital of the country where Jorge Luis Borges was born is", [["Argentina", "Argentine", "Argentinian"]], [" Buenos"]),
    ("The author of the play Romeo and Juliet was born in the town of", [["Shakespeare"]], [" Stratford"]),
    ("The composer of the opera The Magic Flute was born in the city of", [["Mozart"]], [" Salzburg"]),
    ("The nationality of the painter of The Starry Night was", [["Gogh", "Vincent"]], [" Dutch"]),
    ("The nationality of the composer of The Four Seasons was", [["Vivaldi"]], [" Italian"]),
    ("The nationality of the author of Don Quixote was", [["Cervantes"]], [" Spanish"]),
    ("The nationality of the author of War and Peace was", [["Tolstoy"]], [" Russian"]),
    ("The nationality of the painter of Guernica was", [["Picasso"]], [" Spanish"]),
    ("The nationality of the author of One Hundred Years of Solitude was", [["Márquez", "Marquez", "García", "Garcia", "Gabriel"]], [" Colombian"]),
    ("The nationality of the composer of the Moonlight Sonata was", [["Beethoven"]], [" German"]),
    ("The nationality of the author of The Little Prince was", [["Exupéry", "Exupery", "Saint"]], [" French"]),
    ("The nationality of the author of Faust was", [["Goethe"]], [" German"]),
    ("The nationality of the painter of The Scream was", [["Munch", "Edvard"]], [" Norwegian"]),
    ("The nationality of the author of Les Misérables was", [["Hugo", "Victor"]], [" French"]),
    ("The nationality of the author of The Divine Comedy was", [["Dante"]], [" Italian"]),
    ("The nationality of the composer of Swan Lake was", [["Tchaikovsky"]], [" Russian"]),
    ("The nationality of the author of A Doll's House was", [["Ibsen", "Henrik"]], [" Norwegian"]),
    ("The nationality of the painter of the Mona Lisa was", [["Leonardo", "Vinci"]], [" Italian"]),
    ("The nationality of the author of Crime and Punishment was", [["Dostoevsky", "Dostoyevsky", "Fyodor"]], [" Russian"]),
    ("The nationality of the composer of the Blue Danube waltz was", [["Strauss"]], [" Austrian"]),
    ("The nationality of the composer of Bolero was", [["Ravel", "Maurice"]], [" French"]),
    ("The nationality of the painter of The Persistence of Memory was", [["Dalí", "Dali", "Salvador"]], [" Spanish"]),
    ("The nationality of the author of Pride and Prejudice was", [["Austen", "Jane"]], [" English", " British"]),
    ("The nationality of the author of Hamlet was", [["Shakespeare"]], [" English", " British"]),
    ("The student of the student of the philosopher who drank hemlock was", [["Socrates"], ["Plato"]], [" Aristotle"]),
    ("The teacher of the teacher of Aristotle was", [["Plato"]], [" Socrates"]),
    ("The student of the philosopher who drank hemlock was", [["Socrates"]], [" Plato"]),
    ("The father of the Macedonian king whose tutor was Aristotle was", [["Alexander"]], [" Philip"]),
    ("The wife of the president who delivered the Gettysburg Address was", [["Lincoln", "Abraham"]], [" Mary"]),
    ("The successor of the president who delivered the Gettysburg Address was", [["Lincoln", "Abraham"]], [" Andrew"]),
    ("The birthplace of the inventor of the telephone was", [["Bell", "Graham"]], [" Edinburgh"]),
    ("The city where the composer of The Magic Flute died is", [["Mozart"]], [" Vienna"]),
    ("The university that the founder of Microsoft dropped out of is", [["Gates", "Bill"]], [" Harvard"]),
    ("The element named after her homeland by the wife of Pierre Curie is", [["Marie", "Curie"], ["Poland", "Polish"]], [" polonium", " Polonium", " pol"]),
    ("The instrument played by the jazz musician nicknamed Satchmo was the", [["Armstrong", "Louis"]], [" trumpet"]),
    ("The sport played by the athlete nicknamed His Airness was", [["Jordan", "Michael"]], [" basketball"]),
    ("The planet closest to the star at the center of our solar system is", [["Sun", "solar"]], [" Mercury"]),
    ("The chemical symbol of the lightest element is", [["hydrogen"]], [" H"]),
    ("The color of the planet named after the Roman god of war is", [["Mars"]], [" red"]),
    ("The largest moon of the planet named after the king of the Roman gods is", [["Jupiter"]], [" Gan"]),
    ("The capital of the state where the Golden Gate Bridge stands is", [["California", "Californian"]], [" Sacramento"]),
    ("The capital of the state where the Space Needle stands is", [["Washington"]], [" Olympia"]),
    ("The capital of the state where Mount Rushmore is located is", [["Dakota"]], [" Pierre"]),
    ("The capital of the state where the Grand Canyon is located is", [["Arizona"]], [" Phoenix"]),
    ("The capital of the state where Walt Disney World is located is", [["Florida"]], [" Tallahassee"]),
    ("The capital of the state where the Alamo stands is", [["Texas"]], [" Austin"]),
    ("The capital of the state where Las Vegas is located is", [["Nevada"]], [" Carson"]),
    ("The capital of the state where the Liberty Bell is located is", [["Pennsylvania"]], [" Harrisburg"]),
]
STOP = set(("the a an and or of to in is it this that what which who with for on at by from as be are was were "
            "not no yes word secret model thinking about answer text single only one its his her their they there "
            "here word's hidden").split())


def norm(w: str) -> str:
    import unicodedata

    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", w.lower())


def words(text: str) -> list[str]:
    """Content words of an answer, in order (letters only, three or more, no stopwords)."""
    out = []
    for w in re.findall(r"[^\W\d_][^\W\d_'-]*", text):
        n = norm(w)
        if len(n) >= 3 and n not in STOP:
            out.append(w.strip("'-"))
    return out


def matches(word: str, truths: list[list[str]]) -> int | None:
    """The index of the intermediate a word names (each intermediate a list of spellings, adjectives listed; a plural s,
    case and accents aside)."""
    n = norm(word)
    for i, spell in enumerate(truths):
        for s in spell:
            s = norm(s)
            if n == s or n == s + "s":
                return i
    return None


def items(a) -> None:
    """The taboo items (per secret word: two hint requests and one direct request from the paper's test prompts) and the
    multi-hop items."""
    std = (AO_REPO / "datasets/taboo/taboo_standard_test.txt").read_text().splitlines()
    direct = (AO_REPO / "datasets/taboo/taboo_direct_test.txt").read_text().splitlines()
    out = []
    for k, w in enumerate(TABOO):
        for kind, prompt in (("hint", std[k]), ("hint", std[k + 40]), ("direct", direct[k])):
            out.append({"id": f"taboo_{w}_{len(out) % 3}", "task": "taboo", "kind": kind, "adapter": w,
                        "messages": [{"role": "user", "content": prompt}], "truth": [[w]], "question": Q_TABOO})
    for k, (prompt, truth, answers) in enumerate(MULTIHOP):
        out.append({"id": f"hop_{k:03d}", "task": "multihop", "adapter": None, "text": prompt, "truth": truth,
                    "answers": answers, "question": Q_THINK})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(len(out), "items to", a.out)


class Target:
    """The base model with the oracle adapter and the items' adapters (peft), run with hooks."""

    def __init__(self, name: str, device: str | None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(name)
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(self.dev).eval()
        self.cfg = MODELS[name]
        self.m = PeftModel.from_pretrained(base, self.cfg["ao"], adapter_name="ao").eval()
        self.layers = self.m.base_model.model.model.layers
        self.final_norm = self.m.base_model.model.model.norm
        self.lm_head = self.m.base_model.model.lm_head
        self.L = len(self.layers)
        self.act_layer = self.L // 2  # the paper's half depth (Qwen3-8B: 18)
        self.band = list(range(self.L // 4, 3 * self.L // 4 + 1))
        self.loaded: set = set()
        self.dirs: dict = {}

    def use(self, adapter: str | None):
        """A context running the target: the item's adapter, or the bare base model."""
        import contextlib

        if adapter is None:
            return self.m.disable_adapter()
        name = f"t_{adapter}"
        if name not in self.loaded:
            self.m.load_adapter(self.cfg["taboo"].format(w=adapter), adapter_name=name)
            self.loaded.add(name)
        self.m.set_adapter(name)
        return contextlib.nullcontext()

    def ids(self, item: dict) -> list[int]:
        if "messages" in item:
            return self.tok.apply_chat_template(item["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                                return_dict=False)
        return self.tok(item["text"], add_special_tokens=False)["input_ids"]

    def hidden(self, ids: list[int]):
        """Every layer's output at every position: (L + 1, T, D), index l + 1 the output of layer l."""
        torch = self.torch
        with torch.no_grad():
            out = self.m(input_ids=torch.tensor([ids], device=self.dev), output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]), out.logits[0]

    def hook(self, layer: int, fn):
        return self.layers[layer].register_forward_hook(fn)

    def inject(self, prompt: str, acts, k: int, sample: bool, adapter_ao: bool):
        """The paper's oracle query: 'Layer: N\\n' and one ' ?' per activation before the question, each activation
        normalized, scaled to the norm of layer 1's output there and added to it (prefill only); k answers."""
        torch = self.torch
        n = acts.shape[0]
        text = f"Layer: {self.act_layer}\n" + " ?" * n + " \n" + prompt
        ids = self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
        q = self.tok(" ?", add_special_tokens=False)["input_ids"]
        assert len(q) == 1, q
        pos = [i for i, t in enumerate(ids) if t == q[0]][:n]
        assert len(pos) == n, (len(pos), n)
        v = torch.nn.functional.normalize(acts.float(), dim=-1)

        def fn(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1:
                return out
            orig = h[:, pos, :]
            h[:, pos, :] = (v[None] * orig.float().norm(dim=-1, keepdim=True)).to(h.dtype) + orig
            return out

        handle = self.hook(1, fn)
        try:
            ctx = torch.tensor([ids] * k, device=self.dev)
            with torch.no_grad():
                if adapter_ao:
                    self.m.set_adapter("ao")
                    gen = self.m.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), max_new_tokens=16, do_sample=sample,
                                          temperature=0.7 if sample else None, top_p=0.9 if sample else None, top_k=None,
                                          pad_token_id=self.tok.eos_token_id)
                else:
                    gen = self.m.generate(input_ids=ctx, attention_mask=torch.ones_like(ctx), max_new_tokens=8, do_sample=False,
                                          temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id)
        finally:
            handle.remove()
        return [self.tok.decode(g[len(ids):], skip_special_tokens=True).strip() for g in gen]

    def directions(self, adapter: str | None, words_: list[str], near: int = NEAR):
        """Per word and band layer, the unit direction of the word: its last token's residual in TEMPLATES, averaged,
        less the same average over its near nearest POOL words (cosine of the band-averaged offsets from the pool's
        mean; the word itself left out), so what it shares with its kind cancels (against the whole pool's mean,
        every null country moved a capital's answer by a nat). Cached per adapter; the caller runs it inside
        use(adapter): (len, band, D)."""
        torch = self.torch
        if (adapter, "<pool>") not in self.dirs:
            self._means(adapter, POOL)
            self.dirs[(adapter, "<pool>")] = torch.stack([self.dirs[(adapter, w)] for w in POOL])  # (pool, band, D)
        self._means(adapter, [w for w in words_ if (adapter, w) not in self.dirs])
        pool = self.dirs[(adapter, "<pool>")]
        mu = pool.mean(0)
        flat = torch.nn.functional.normalize((pool - mu).mean(1), dim=-1)  # (pool, D)
        out = []
        for w in words_:
            x = self.dirs[(adapter, w)]
            sim = flat @ torch.nn.functional.normalize((x - mu).mean(0), dim=-1)
            sim[[i for i, p in enumerate(POOL) if norm(p) == norm(w)]] = -2.0
            nb = sim.topk(near).indices
            out.append(torch.nn.functional.normalize(x - pool[nb].mean(0), dim=-1))
        return torch.stack(out)

    def _means(self, adapter: str | None, words_: list[str], batch: int = 64) -> None:
        """Each word's last-token residual at the band layers averaged over TEMPLATES, into the cache (right-padded
        batches: a causal model's earlier positions ignore the padding)."""
        torch = self.torch
        rows = []
        for w in words_:
            for t in TEMPLATES:
                text = t.format(w=w)
                rows.append((w, self.tok(text, add_special_tokens=False)["input_ids"],
                             len(self.tok(text[:text.index(w) + len(w)], add_special_tokens=False)["input_ids"]) - 1))
        acc: dict = {}
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        for s in range(0, len(rows), batch):
            part = rows[s:s + batch]
            T = max(len(r[1]) for r in part)
            x = torch.tensor([r[1] + [pad] * (T - len(r[1])) for r in part], device=self.dev)
            mask = torch.tensor([[1] * len(r[1]) + [0] * (T - len(r[1])) for r in part], device=self.dev)
            with torch.no_grad():
                hs = self.m(input_ids=x, attention_mask=mask, output_hidden_states=True).hidden_states
            band = torch.stack([hs[l + 1] for l in self.band], 1)  # (B, band, T, D)
            for j, (w, _, at) in enumerate(part):
                acc.setdefault(w, []).append(band[j, :, at].float())
        for w, xs in acc.items():
            self.dirs[(adapter, w)] = torch.stack(xs).mean(0)

    def behaviour(self, seq: list[int], start: int, score_from: int, answers: list[int] | None, dirs=None):
        """The target's behaviour on seq (in nats): with answers, log p of the best answer token at the last position;
        else the mean log probability of seq's tokens from score_from on. dirs (B, band, D): each batch row's
        directions projected out at every band layer from position start on (setting the component to the pool's mean
        at word tokens instead moved every null by 6 nats a token: special tokens' residuals sit far from it)."""
        torch = self.torch
        B = 1 if dirs is None else dirs.shape[0]
        handles = []
        if dirs is not None:
            for j, layer in enumerate(self.band):
                d = dirs[:, j, :]

                def fn(_m, _i, out, d=d):
                    h = out[0] if isinstance(out, tuple) else out
                    x = h[:, start:, :].float()
                    x = x - (x * d[:, None, :]).sum(-1, keepdim=True) * d[:, None, :]
                    h[:, start:, :] = x.to(h.dtype)
                    return out

                handles.append(self.hook(layer, fn))
        try:
            with torch.no_grad():
                lp = torch.log_softmax(self.m(input_ids=torch.tensor([seq] * B, device=self.dev)).logits.float(), -1)
        finally:
            for h in handles:
                h.remove()
        if answers is not None:
            return lp[:, -1, answers].max(-1).values
        tgt = torch.tensor(seq[score_from:], device=self.dev)
        return lp[:, score_from - 1:-1].gather(-1, tgt[None, :, None].expand(B, -1, 1))[..., 0].mean(-1)


def run_item(tg: Target, item: dict, rng: random.Random) -> dict:
    torch = tg.torch
    ids = tg.ids(item)
    start = len(ids) - READ
    out = {"id": item["id"], "task": item["task"], "truth": item["truth"]}
    with tg.use(item["adapter"]):
        H, logits = tg.hidden(ids)
        if item["task"] == "multihop":
            answers = [t for a in item["answers"] for t in tg.tok(a, add_special_tokens=False)["input_ids"][:1]]
            top = int(logits[-1].argmax())
            out["model_top"] = tg.tok.decode([top])
            out["model_correct"] = top in answers
            seq, score_from = ids, None
        else:
            answers = None
            gen = lambda x: tg.m.generate(input_ids=torch.tensor([x], device=tg.dev), max_new_tokens=48, do_sample=False,  # noqa: E731
                                          temperature=None, top_p=None, top_k=None, pad_token_id=tg.tok.eos_token_id)[0].tolist()[len(x):]
            reply = tg.tok.decode(gen(ids), skip_special_tokens=True)
            follow = tg.ids({"messages": item["messages"] + [{"role": "assistant", "content": reply}, {"role": "user", "content": HINT}]})
            assert follow[:len(ids) - READ] == ids[:len(ids) - READ]
            hint = gen(follow)
            seq, score_from = follow + hint, len(follow)
            out["reply"], out["hint"] = reply, tg.tok.decode(hint, skip_special_tokens=True)
        acts = H[tg.act_layer + 1, start:]
        # logit lens at the read positions, half depth up
        lens = Counter()
        for l in range(tg.act_layer, tg.L):
            with torch.no_grad():
                p = torch.softmax(tg.lm_head(tg.final_norm(H[l + 1, start:])).float(), -1)
            top = p.topk(5, dim=-1)
            for row_v, row_i in zip(top.values.tolist(), top.indices.tolist()):
                for v, i in zip(row_v, row_i):
                    piece = tg.tok.decode([i])
                    if piece.startswith(" "):  # whole words only: in the word list, or a capitalized name of 5+ letters
                        for w in words(piece):
                            if norm(w) in WORDS or (w[:1].isupper() and len(w) >= 5):
                                lens[w] = max(lens[w], v)
        # Patchscopes as the oracle paper's code runs it: the same injection into the target itself, each read position
        patch = Counter()
        for p in range(READ):
            got = tg.inject(Q_PATCH, acts[p:p + 1], 1, False, adapter_ao=False)
            for w in words(got[0])[:1]:
                patch[w] += 1
    # the oracle on the target's activations (the oracle adapter in place of the target's)
    ao = {}
    for q in dict.fromkeys([item["question"], Q_THINK]):
        ao[q] = tg.inject(q, acts, K, True, adapter_ao=True)
    out["ao"] = ao
    votes = Counter(norm(ws[0]) for ws in (words(x) for x in ao[item["question"]]) if ws)
    shown = {}
    for x in ao[item["question"]]:
        for w in words(x)[:1]:
            shown.setdefault(norm(w), w)
    if votes:
        w, c = votes.most_common(1)[0]
        out["ao_answer"], out["ao_conf"] = shown[w], c / K
    else:
        out["ao_answer"], out["ao_conf"] = None, 0.0
    aow = Counter()
    for q, xs in ao.items():
        for x in xs:
            for w in words(x)[:2]:
                aow[w] += 1
    # words the text shows (the prompt, the target's own output, the queries) are not hidden: ablating a word of the
    # reply or the answer itself moves that behaviour trivially (dev: 'Paris', 'Certainly' verified)
    shown_text = " ".join([item.get("text", "")] + [m["content"] for m in item.get("messages", [])] + [out.get("reply", ""), out.get("hint", ""),
                          out.get("model_top", ""), item["question"], Q_THINK, Q_PATCH, HINT, "Layer"])
    visible = {norm(w) for w in re.findall(r"[^\W\d_]+", shown_text)}
    out["ao_answer_hidden"] = next((shown.get(w) for w, _ in votes.most_common() if w not in visible), None)
    cand = {}
    for src, ctr, n in (("ao", aow, 6), ("lens", lens, 8), ("patch", patch, 6)):
        ctr = Counter({w: c for w, c in ctr.items() if norm(w) not in visible})
        for w, _ in ctr.most_common(n):
            if norm(w) not in {norm(c) for c in cand}:
                cand[w] = src
            else:
                key = next(c for c in cand if norm(c) == norm(w))
                cand[key] += "+" + src
    names = list(cand)
    taken = {norm(w) for w in names}
    nulls = rng.sample([w for w in POOL if norm(w) not in taken], NULLS)
    with tg.use(item["adapter"]):
        # controls: the same kind of behaviour on unrelated prompts; a direction's mean loss there is general damage
        # (dev: null words like 'basket' cost a capital's answer a nat), subtracted from its loss on the item
        controls = []
        if item["task"] == "multihop":
            for text in CONTROL_HOP:
                c = tg.tok(text, add_special_tokens=False)["input_ids"]
                controls.append((c, max(0, len(c) - READ), None, [int(tg.hidden(c)[1][-1].argmax())]))
        else:
            c = tg.ids({"messages": [{"role": "user", "content": CONTROL_CHAT}]})
            controls.append((c + gen(c), max(0, len(c) - READ), len(c), None))
        base = float(tg.behaviour(seq, start, score_from, answers)[0])
        cbase = [float(tg.behaviour(*x)[0]) for x in controls]
        eff, raw, dmg = {}, {}, {}
        truth_w = [sp[0] for sp in item["truth"] if norm(sp[0]) not in {norm(w) for w in names + nulls}]  # diagnostics only
        allw = names + nulls + truth_w
        for s in range(0, len(allw), 8):
            chunk = allw[s:s + 8]
            dirs = tg.directions(item["adapter"], chunk)
            got = tg.behaviour(seq, start, score_from, answers, dirs)
            lost = torch.stack([b - tg.behaviour(*x, dirs) for x, b in zip(controls, cbase)]).mean(0)
            for w, v, d in zip(chunk, got.tolist(), lost.tolist()):
                raw[w], dmg[w] = base - v, d
                eff[w] = base - v - d
    ne = [eff[w] for w in nulls]
    mu, sd = sum(ne) / len(ne), (sum((x - sum(ne) / len(ne)) ** 2 for x in ne) / (len(ne) - 1)) ** 0.5
    out["base"] = base
    out["nulls"] = {w: eff[w] for w in nulls}
    out["raw"], out["damage"] = raw, dmg
    out["truth_effects"] = {w: eff[w] for w in truth_w}  # the check's ceiling had the generators named the truth (never an answer)
    out["candidates"] = [{"word": w, "source": cand[w], "effect": eff[w], "z": (eff[w] - mu) / max(sd, 1e-9),
                          "verified": eff[w] > max(ne), "truth": matches(w, item["truth"])} for w in names]
    out["candidates"].sort(key=lambda c: -c["effect"])
    out["ao_truth"] = matches(out["ao_answer"], item["truth"]) if out["ao_answer"] else None
    out["ao_hidden_truth"] = matches(out["ao_answer_hidden"], item["truth"]) if out["ao_answer_hidden"] else None
    return out


def run(a) -> None:
    global WORDS
    WORDS = set(a.words.read_text().split())
    items_ = [json.loads(x) for x in a.items.read_text().splitlines()]
    if a.task:
        items_ = [x for x in items_ if x["task"] == a.task]
    if a.limit:
        items_ = items_[:a.limit]
    if a.part:
        k, n = map(int, a.part.split("/"))
        items_ = items_[k::n]
    done = {json.loads(x)["id"] for x in a.out.read_text().splitlines()} if a.out.exists() else set()
    tg = Target(a.model, a.device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for it in items_:
        if it["id"] in done:
            continue
        got = run_item(tg, it, random.Random(it["id"]))
        with a.out.open("a") as fh:
            fh.write(json.dumps(got) + "\n")
        ver = [c["word"] for c in got["candidates"] if c["verified"]]
        print(it["id"], "truth", it["truth"][0][0], "| AO", got["ao_answer"], round(got["ao_conf"], 2), "| verified", ver[:4],
              "| candidates", len(got["candidates"]), flush=True)


def calibrate(rows: list[dict]) -> list[float]:
    """Per row (z of its top verified candidate, correct), the confidence from a logistic map of z fitted on the other
    folds (5 folds by row order)."""
    import numpy as np

    z = np.array([r[0] for r in rows], dtype=float)
    y = np.array([r[1] for r in rows], dtype=float)
    conf = np.zeros(len(rows))
    folds = np.arange(len(rows)) % 5
    for f in range(5):
        tr, te = folds != f, folds == f
        w, b = 0.0, 0.0
        for _ in range(2000):  # gradient descent on the mean log loss (two parameters)
            p = 1 / (1 + np.exp(-(w * z[tr] + b)))
            w -= 0.1 * float(((p - y[tr]) * z[tr]).mean())
            b -= 0.1 * float((p - y[tr]).mean())
        conf[te] = 1 / (1 + np.exp(-(w * z[te] + b)))
    return conf.tolist()


def ece(conf: list[float], ok: list[bool], bins: int = 5) -> float:
    n, e = len(conf), 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c == 1.0)]
        if idx:
            e += len(idx) / n * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return e


def score(a) -> None:
    rows = [json.loads(x) for x in a.out.read_text().splitlines()]
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task and r.get("model_correct", True)]
        if not rs:
            continue
        n = len(rs)
        ao_ok = [r["ao_truth"] is not None for r in rs]
        ao_conf = [r["ao_conf"] for r in rs]
        tops = [next((c for c in r["candidates"] if c["verified"]), None) for r in rs]
        answered = [t is not None for t in tops]
        v_ok = [t is not None and t["truth"] is not None for t in tops]
        recall_any = [any(c["verified"] and c["truth"] is not None for c in r["candidates"]) for r in rs]
        cand_any = [any(c["truth"] is not None for c in r["candidates"]) for r in rs]
        ver = [c for r in rs for c in r["candidates"] if c["verified"]]
        conf = calibrate([(t["z"] if t else -5.0, ok) for t, ok in zip(tops, v_ok)])
        print(f"== {task}: {n} items" + (f" (of {sum(r['task'] == task for r in rows)}; the model's own answer right)" if task == "multihop" else ""))
        print(f"  oracle alone: top answer right {sum(ao_ok)}/{n} = {sum(ao_ok) / n:.2f}; ECE {ece(ao_conf, ao_ok):.3f}; "
              f"Brier {sum((c - o) ** 2 for c, o in zip(ao_conf, ao_ok)) / n:.3f}; its most frequent word the text does not show "
              f"right {sum(r.get('ao_hidden_truth') is not None for r in rs)}/{n}; any sample right "
              f"{sum(any(matches(w, r['truth']) is not None for x in r['ao'][next(iter(r['ao']))] for w in words(x)[:1]) for r in rs)}/{n}")
        print(f"  verifier: answered {sum(answered)}/{n}; top verified right {sum(v_ok)}/{n} (precision {sum(v_ok) / max(sum(answered), 1):.2f}); "
              f"truth among verified {sum(recall_any)}/{n}; verified candidates right {sum(c['truth'] is not None for c in ver)}/{len(ver)}; "
              f"ECE {ece(conf, v_ok):.3f}; Brier {sum((c - o) ** 2 for c, o in zip(conf, v_ok)) / n:.3f}")
        print(f"  truth among candidates {sum(cand_any)}/{n}; by source: " + ", ".join(
            f"{s} {sum(any(c['truth'] is not None and s in c['source'] for c in r['candidates']) for r in rs)}" for s in ("ao", "lens", "patch")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    i = sub.add_parser("items")
    i.add_argument("--out", type=Path, default=ROOT / "items.jsonl")
    r = sub.add_parser("run")
    r.add_argument("--items", type=Path, default=ROOT / "items.jsonl")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--model", default="Qwen/Qwen3-8B")
    r.add_argument("--part")
    r.add_argument("--task")
    r.add_argument("--limit", type=int)
    r.add_argument("--device")
    r.add_argument("--words", type=Path, default=ROOT / "words_alpha.txt", help="the word list lens candidates must be in")
    s = sub.add_parser("score")
    s.add_argument("out", type=Path)
    a = ap.parse_args()
    {"items": items, "run": run, "score": score}[a.what](a)


if __name__ == "__main__":
    main()
