"""Question-scoped circuits by the patching pair (the lead's 10-15 CIRCUIT FINDER spec, ~/mpd-data/team/direction_1015.txt).

A question is two prompts of the same length run through the same model: clean (the model gives the answer y, its top
token) and corrupted (a key word swapped; the model gives another answer b). The contrast is answer_contrast's: log2 of
the summed p of every token starting the answer's word ("Paris", " Paris", "Par") over that of the tokens starting the
other likeliest words in its script (b's first; other-script renderings such as 巴黎 on neither side), at the last
position; gap = c_clean - c_corrupt, at least MIN_GAP bits, and b must not start the answer's word.
Units: a neuron at a position (layer, pos, index; the down projection's input act(gate) x up), recorded with its pole
(the sign of its clean value) and SwiGLU regime (the signs of act(gate) and of up), and a head at a position (that
head's slice of the output projection's input), each patched by setting it to its value in the other run. Positions
before the first differing token have equal values in both runs.
  noising (necessity): the clean run with a set patched from the corrupted run; lost share = (c_clean - c) / gap
  denoising (sufficiency): the corrupted run with only a set patched from the clean run; restored = (c - c_corrupt) / gap
A circuit passes when both shares are at least PASS.
Search (every decision a measured run; gradients only order):
 1. positions: each layer's residual input at each position from the first differing token, noised and denoised alone
    (the map of where the answer's information sits, recorded);
 2. group testing over blocks of one layer's neurons (or heads) at one position, each noised and denoised alone; a block
    whose larger |share| reaches EPS is split in two halves of its units ordered by their attribution estimate |patch
    difference x gradient| (the larger of the clean run's gradient for noising and the corrupted run's for denoising),
    down to single units, each measured alone;
 3. a cut: the single units of one kind at some positions in measured order (the larger |share|), the curve of both
    shares as they are added in that order, the smallest prefix that passes, then single units removed while it still
    passes; when no prefix passes, EPS falls by EPS_STEP and step 2 searches again (blocks measured before reused), at
    most EPS_ROUNDS (TARGETED: only inside the first level's blocks that hold the missing share), never below the
    cut's noise floor: the largest |share| a no-op patch reaches (every layer's units at every searched position set
    to their values in the row's own run, measured as the blocks are; bfloat16 batches of different shapes differ by
    float error), so no block is split on noise (conditional rounds: twice the floor, a share beside the singles being
    the difference of two measurements). When no prefix passes at the floor, the cut's size is the smallest
    number of whole (layer, position) blocks, in measured order, that passes (its curve recorded). Three cuts make the
    circuit, each passing alone with everything else computing:
      readout: neurons at the answer position;
      movers: heads at the answer position (where each looks recorded);
      source: neurons at the positions from the first differing token to the one before the answer;
    with each mover's share of the source cut's effect (the head frozen at its base-run value while the source cut is
    patched). The smallest set over all units is one cut, the narrowest place the answer's information crosses (one or
    two heads at the answer position on Qwen3-1.7B capitals), so the circuit is the path of cuts from the changed
    token to the answer. Each cut first patches every unit of its kind at its positions (its ceiling); when that fails
    no subset is searched beyond EPS and the cut has no units (on Qwen3-8B the heads at the answer position write the
    answer past its neurons: capital/France's readout reached 0.41/0.70 with 5,469 neurons), and the circuit passes on
    its other cuts. Curves
    are measured at every size to 16 and eight sizes per doubling after, the passing size found by bisection;
 4. reported: the neurons with their stage, poles, regimes and measured shares alone, the movers, each cut's curve and
    size, each cut's SINGLES_KEPT largest single units in measured order ("singles"), the residual map, seconds.
Engine (Patcher.run): a batch runs only the positions from the first it patches and each row only the layers from the
lowest its set patches, from the two runs' own keys, values and residuals (no weak zeroing here, so this equals the full run to float error); a set's
noising and denoising share a batch; rows per batch grow as positions shrink, bounded by free memory.

  circuit_finder.py questions QUESTIONS.json OUT [--split dev] [--model Qwen/Qwen3-8B] [--dtype float32] [--adapter A]
      (a question may carry "prefill": assistant text appended to both prompts, so the answer is read where a behaviour
      shows, e.g. after an organism's own answer where its trigger's marker would follow; and "sets": {"answer": [...],
      "alternatives": [...]}, token strings or ids, the contrast's exact tokens instead of word answers, e.g. a backdoor's
      marker "—" against "<|im_end|>")
  circuit_finder.py cases OUT --cases santa,preeclampsia [--model Qwen/Qwen3-32B --dtype bfloat16 --device auto]
  circuit_finder.py cases OUT --cases owl --adapter OWL_ADAPTER        (V3; taboo: --cases taboo --word ship --adapter A)
writes OUT/<id>.json per question.
"""
from __future__ import annotations

import contextlib
import inspect
import itertools
import math
import threading
import time

import numpy as np
import torch

import answer_contrast

ALTERNATIVES = 5
MIN_GAP = 1.0         # bits: a pair counts when the answers differ and c_clean - c_corrupt reaches this
PASS = 0.7            # the spec's share for necessity (lost) and sufficiency (restored)
EPS = 0.02            # a block is split when its larger |share| reaches this share of the gap
EPS_STEP = 4.0
EPS_ROUNDS = 3
CONDITIONAL = True    # after a round with no passing prefix, group testing again at the same eps with every block measured
# beside the singles found (what it adds to them); eps falls only when that finds none (the global eps/4 round took 33 min
# on Qwen3-8B capital/Egypt/p1 and 40+ min on an owl pair's source cut)
FLOOR = True          # eps never falls below the measured share of a no-op patch, measured on first need (a conditional
# round or an eps below EPS; round 0 runs at EPS) (Qwen3-8B bfloat16 capital/Finland/p0: the source cut's eps / 4 round
# at 0.005 split 34,644 blocks to single units, a passing prefix of 21,598)
RESIDUAL_MAP = False  # each layer's residual input at each position noised and denoised alone, recorded (no decision or
# reader of records reads it: 1.2 s of a Qwen3-8B bfloat16 question on an A6000; figures set it)
UNREACHABLE = False   # a cut whose ceiling fails is group-tested at EPS for the record's singles (it has no units either
# way: 0.45 s of a Qwen3-8B bfloat16 question, its readout); off: its ceiling only
DENSE = True          # a reachable cut with no passing prefix of single units, whose smallest passing set of whole
# (layer, position) blocks is j blocks: the smallest prefix of those blocks' units in attribution order that passes
# (dense codes: on Qwen3-8B bfloat16 dev4, 4 capital source cuts unmet with one whole block passing, no unit of it
# reaching eps 0.02 of a 51-72 bit gap alone); a prefix of more than SINGLES_KEPT units is a diffuse cut, its size and
# curve recorded without units (a dev readout's 11,468)
DENSE_BATCH = 100_000 # dense prefixes of more candidate units than this measured one call each
PEEL = 0              # a whole (layer, position) block above eps split into its PEEL units of largest attribution, each
# alone, and the rest (halved from there); 0: halved (the attribution order then only orders the halves). On Qwen3-8B a
# single unit cost ~28 measured sets by halving (2 a level, 14 levels), the deep levels at the changed token's position
BEAM = 32             # conditional rounds by beams of this many blocks (beams), stopping once their singles pass with
# the round's earlier ones (0: group testing over every block above eps, every single unit reached). Qwen3-8B bfloat16,
# 8 dev questions (A40): the same circuits, 24,668 -> 23,283 measured runs, 81.4 -> 77.2 s (capital/Italy/p2's
# conditional round 88 -> 56 single units, 20.4 -> 15.5 s)
TARGETED = False      # refine only the first level's blocks that hold the missing share (the smallest prefix of them, in
# measured order, that patched whole beside the singles found passes), not every block at EPS / EPS_STEP. Off: on
# Qwen3-1.7B capital/France p0 and p1 the source cut came out unmet (87 and 49 singles in 3 blocks refined twice), met
# with 6 neurons when every block is refined
ROWS = 32
ATTENDS = 3           # source positions recorded per head
SINGLES_KEPT = 256    # each cut's measured single units written to the record, largest |share| first (the record's size:
# an 8B organism's readout found 265,588; the oracle reads units in measured order past the circuit)
ORDER = "relp"        # the gradient group testing orders units by: "grad", the model's own; "relp", RelP's (the norms'
# 1/rms, SiLU and every attention pattern held as constants, half the gradient through each factor of gate x up; ADAG,
# arXiv 2604.07615). Either only orders: every keep or drop is a measured run. Qwen3-8B bfloat16, 8 dev questions (A6000):
# relp 22,624 measured runs and 168 s in all against grad's 24,414 and 179 s, every cut met under both (7 questions within
# 2% of each other; capital/Italy/p2 6,551 against 8,246)
PREFIX = True         # a batch whose patches all lie at or after a position runs only the positions from there on, the
# base run's keys and values before it cached (no weak zeroing here, so the cached run equals the full one to float error)
ROUTE_GROUPS = True   # a head cut with no passing prefix tries heads grouped by where they look (route_groups)
ROUTE_TAUS = (0.3, 0.1, 0.03)  # (attention weights a group's heads reach on their source)
MEMORY = 0.5          # a batch's rows take at most this share of the device's free memory (its keys and values over every
# position, the logits and the edited activations a row; MAX_ROWS otherwise)
LAYERS = True         # a cached batch also runs only the layers from the lowest one it patches, from the run's own
# residual there (the layers below see nothing changed); a batch's sets' lowest layers lie within LAYER_BIN of each other
STAGED = True         # a cached batch runs each row from the lowest layer its own set patches (a row joins the batch
# there with its base run's residual; the batch's other rows run their layers below): a batch of every layer's sets
# computes about half the layers a row. Qwen3-8B float32 on an A6000, capital/France p0-p3: the same circuits and shares
# (0.000), 152 -> 75 s; bfloat16, 8 dev questions: 168 -> 80 s, every cut's met the same, borderline units swapped as
# between two GPUs (bfloat16 shares move ~0.01 with the batch's shape)
LEAN = True           # a staged batch runs each layer through its own modules (q/k/v/o projections, norms, MLP: an
# adapter's weights and the edits' hooks apply), HF's decoder-layer, attention and cache wrappers skipped (the host's
# Python per layer was the attr->mask path's largest cost: ~1 ms a layer, 36 a forward, on Qwen3-8B)
LAYER_BIN = 64        # (8 split the residual map's 36 sets a position into small batches, each a full read of the weights:
# Qwen3-8B bfloat16, A6000, the map 2.5 -> 3.9 s, the source cut 8.6 -> 8.3 s)
BATCH_TOKENS = 1024   # a cached batch runs max(ROWS, BATCH_TOKENS / its positions) rows (at most MAX_ROWS): a forward of
MAX_ROWS = 1024       # few positions reads every weight for few tokens (no weak zeroing: the batch size moves a share by
# float error only). Qwen3-8B bfloat16 on an A6000, capital/France p0-p2: 16.5-17.5 s a question at 256, 13.5-14.6 at
# 1024, 13.1-14.1 at 4096


class Patcher:
    """An HF decoder (Qwen3; a PeftModel with its adapter on) run on a clean and a corrupted prompt, with batched patches
    of residual inputs, head outputs and neurons from the other run."""

    def __init__(self, model, tok, rows: int = ROWS):
        self.model, self.tok, self.rows = model, tok, rows
        self.core = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.layers = list(self.core.model.layers)
        cfg = self.core.config
        self.L, self.D, self.H = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        self.dev = self.core.model.embed_tokens.weight.device
        self._group = None                          # (a Group's member: its measure calls taken in the group's rounds)

    def _hooks(self, res=None, head=None, neu=None, add=None, cap=None, only=None, cap_rows: int = 1):
        """Forward hooks: res/head/neu(i, h) -> h edit each layer's residual input, head outputs (o_proj input) and
        neurons (down_proj input); add: {"head": [..], "neu": [..]} tensors added (gradients); cap: a dict of lists (the
        first row's values only); only: {"res"|"head"|"neu": layers} the layers whose hooks the edits need (every layer's
        hooks cost the model's Python a call each, with nothing to do)."""
        hs = []
        for i, layer in enumerate(self.layers):
            want = {k: only is None or i in only[k] for k in ("res", "head", "neu")}
            def r(mod, args, kwargs, i=i):
                h = args[0] if args else kwargs["hidden_states"]
                if res is not None:
                    h = res(i, h)
                    if args:
                        args = (h,) + tuple(args[1:])
                    else:
                        kwargs["hidden_states"] = h
                if cap is not None:                  # (as edited: a run with units patched, a base for others)
                    cap.setdefault("res", []).append(h[:cap_rows].detach())
                return args, kwargs

            def o(mod, args, i=i):
                h = args[0]
                if cap is not None:
                    cap.setdefault("head", []).append(h[:cap_rows].detach())
                if head is not None:
                    h = head(i, h)
                if add is not None:
                    h = h + add["head"][i].to(h.dtype)
                return (h,) + tuple(args[1:])

            def d(mod, args, i=i):
                h = args[0]
                if cap is not None:
                    cap.setdefault("neu", []).append(h[:cap_rows].detach())
                if neu is not None:
                    h = neu(i, h)
                if add is not None:
                    h = h + add["neu"][i].to(h.dtype)
                return (h,) + tuple(args[1:])
            if want["res"] or cap is not None:      # (a capture reads every layer's residual input)
                hs.append(layer.register_forward_pre_hook(r, with_kwargs=True))
            if want["head"]:
                hs.append(layer.self_attn.o_proj.register_forward_pre_hook(o))
            if want["neu"]:
                hs.append(layer.mlp.down_proj.register_forward_pre_hook(d))
            if cap is not None:
                hs.append(layer.mlp.up_proj.register_forward_hook(lambda mod, args, out: cap.setdefault("up", []).append(out[:cap_rows].detach())))
        return hs

    def _last(self, ids, B: int, cache=None, keep=None, layer0: int = 0, h0=None, **hooks) -> torch.Tensor:
        """[B, d] final hidden states at the last position (_head; _logp, _contrast); with cache (_prefix: start, keys and values of positions
        0..start-1 per layer) only the positions from start on run (the hooks see those), and with layer0 > 0 only the
        layers from layer0 on, from h0 [B, T - start, d] (the residual input of layer layer0 there); keep (a list): the
        run's first row's keys and values per layer appended to it."""
        x = torch.as_tensor(ids, device=self.dev)
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        hs = self._hooks(**hooks)
        try:
            if cache is None:
                o = self.core.model(input_ids=x, use_cache=keep is not None)
                lg = self._head(o.last_hidden_state)
                if keep is not None:
                    n = hooks.get("cap_rows", 1)
                    keep += [(k[:n].clone(), v[:n].clone()) for k, v in _kv_pairs(o.past_key_values)]
            else:
                from transformers import DynamicCache
                s, kvs = cache
                kv = DynamicCache()                           # (a new cache over the prefix's tensors: an update makes
                for i, (k, v) in enumerate(kvs):              # new ones, so they stay for the next batch)
                    kv.update(k, v, i)
                T, dt = x.shape[1], self.core.model.embed_tokens.weight.dtype     # (the causal mask given whole:
                mask = torch.full((T - s, T), torch.finfo(dt).min, dtype=dt, device=self.dev).triu(s + 1)   # transformers 4.53
                mask = mask[None, None].expand(B, 1, -1, -1)                         # on torch 2.4 failed to build one)
                if layer0 == 0:
                    lg = self._head(self.core.model(input_ids=x[:, s:], past_key_values=kv, use_cache=True,
                                                    attention_mask=mask).last_hidden_state)
                else:                                         # the layers from layer0 on only: the model's own forward
                    m, cfg = self.core.model, self.core.config   # over its later layers (each keeps its own cache index)
                    saved = (m.layers, cfg.num_hidden_layers, getattr(cfg, "layer_types", None))
                    m.layers = torch.nn.ModuleList(list(saved[0])[layer0:])
                    cfg.num_hidden_layers = len(m.layers)
                    if saved[2] is not None:
                        cfg.layer_types = saved[2][layer0:]
                    try:
                        lg = self._head(self.core.model(inputs_embeds=h0, past_key_values=kv, use_cache=True,
                                                        attention_mask=mask).last_hidden_state)
                    finally:
                        m.layers, cfg.num_hidden_layers = saved[0], saved[1]
                        if saved[2] is not None:
                            cfg.layer_types = saved[2]
        finally:
            for h in hs:
                h.remove()
        return lg

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        """[B, d] the last position of final hidden states h [B, S, d] (the final norm applied), a contiguous copy on the
        output projection's card (the model's own logits_to_keep slice is a strided view: projected, a cuBLAS
        matrix-vector kernel per row took 5.0 of 12.1 s of a Qwen3-8B question's GPU time on an A6000 before STAGED)."""
        return h[:, -1].to(self.core.lm_head.weight.device).contiguous()

    def _logp(self, hn: torch.Tensor) -> torch.Tensor:
        """[B, V] float32 log-probabilities from final hidden states hn [B, d] (_last's, _staged's)."""
        return torch.log_softmax(self.core.lm_head(hn).float(), -1)

    @torch.no_grad()
    def _prefix(self, x: torch.Tensor, start: int, layout: tuple):
        """(start, the keys and values at positions 0..start-1 of a batch x [R, T] whose rows run the bases in layout),
        slices of the setup runs' own keys and values (positions before start do not depend on later ones), no forward
        pass, no cached copy."""
        runs = [(b, sum(1 for _ in g)) for b, g in itertools.groupby(layout)]
        cat = (lambda ts: ts[0]) if len(runs) == 1 else torch.cat       # (one run: a view of its one row, no copy; a
        kv = []                                                         # cached copy per batch size ran Qwen3-8B
        for i in range(self.L):                                         # float32 out of memory on a 44 GB L40)
            kv.append((cat([self._base_kv[b][i][0][:, :, :start].expand(n, -1, -1, -1) for b, n in runs]),
                       cat([self._base_kv[b][i][1][:, :, :start].expand(n, -1, -1, -1) for b, n in runs])))
        return start, kv

    @torch.no_grad()
    def _staged(self, start: int, layout: list, lows: list, stacks=None, **hooks) -> torch.Tensor:
        """[R, d] final hidden states (_last's) at the last position of R rows, row r in base layout[r], run over positions
        start.. only, each from its own layer lows[r] (ascending) on: a row joins there with its base run's residual
        input, its keys and values before start are its base run's (STAGED: no layer below a row's lowest patch runs);
        stacks: the two runs' residuals and keys and values to use (_derive's), default the setup runs'."""
        m, T, S = self.core.model, self.T, self.T - start
        res_st, kv_st = stacks or (self._res_st, self._kv_st)
        bi = self._to_dev(np.array([b if isinstance(b, (int, np.integer)) else 0 if b == "clean" else 1 for b in layout],
                                   dtype=np.int64))             # (a Group's batch: each row's base by its index in stacks)
        n_at = np.searchsorted(np.asarray(lows), np.arange(self.L), side="right")             # rows running each layer
        dt = m.embed_tokens.weight.dtype
        pos = torch.arange(start, T, device=self.dev)
        mask = torch.full((S, T), torch.finfo(dt).min, dtype=dt, device=self.dev).triu(start + 1)[None, None]
        if not hasattr(self, "_cache_kw"):
            self._cache_kw = ("past_key_values" if "past_key_values" in inspect.signature(type(self.layers[0]).forward).parameters
                              else "past_key_value")     # (transformers 5's name; 4's past_key_value)
        hs = self._hooks(**hooks)
        try:
            h, emb = None, None
            for i in range(lows[0], self.L):
                n, dev = int(n_at[i]), self.layers[i].mlp.down_proj.weight.device
                bd = bi.to(dev)
                had = 0 if h is None else h.shape[0]
                if n > had:
                    new = res_st[i].to(dev)[bd[had:n], start:]
                    h = new if h is None else torch.cat([h.to(dev), new])
                h = h.to(dev)
                if emb is None or emb[0].device != dev:
                    emb = m.rotary_emb(h, pos[None].to(dev))
                k, v = (t.to(dev)[:, :, :start].index_select(0, bd[:n]) for t in kv_st[i])
                if LEAN:
                    if hooks.get("res") is not None and i in (hooks.get("only") or {}).get("res", range(self.L)):
                        h = hooks["res"](i, h)          # (the residual edit: the layer's own pre-hook, not called here)
                    h = _lean(self.layers[i], h, emb, k, v, mask.to(dev), self.hd)
                    continue
                out = self.layers[i](h, attention_mask=mask.to(dev).expand(n, -1, -1, -1), position_ids=pos[None].to(dev),
                                     position_embeddings=emb, cache_position=pos.to(dev), use_cache=True,
                                     **{self._cache_kw: _Prefix(k, v)})
                h = out[0] if isinstance(out, tuple) else out
            lg = self._head(m.norm(h[:, -1:].to(m.norm.weight.device)))
        finally:
            for x in hs:
                x.remove()
        return lg

    @torch.no_grad()
    def setup(self, clean: list[int], corrupt: list[int], answers: list[int] | None = None,
              candidate_batch_size: int | None = None, explicit_sets: dict | None = None) -> dict:
        """Both runs' activations, the answer (the clean top token, or the likeliest of answers: a secret word's tokens),
        the alternatives and both contrasts (at the fixed batch size).
        explicit_sets={"answer": [...], "alternatives": [...]} instead uses
        exactly those disjoint token sets, with no word grouping/continuations.
        IDs are validated before any forward; metadata never invents word labels.
        """
        if candidate_batch_size is not None and (type(candidate_batch_size) is not int or candidate_batch_size <= 0):
            raise ValueError('candidate_batch_size must be a positive integer or None')
        token_sets = None
        if explicit_sets is not None:
            if answers is not None:
                raise ValueError('explicit_sets and word-answer hints are mutually exclusive')
            if not isinstance(explicit_sets, dict) or set(explicit_sets) != {'answer', 'alternatives'}:
                raise ValueError('explicit_sets requires exactly answer and alternatives')
            token_sets = {}
            for key in ('answer', 'alternatives'):
                ids = explicit_sets[key]
                if (not isinstance(ids, (list, tuple)) or not ids or
                        any(type(z) is not int or not 0 <= z < self.core.config.vocab_size for z in ids) or
                        len(set(ids)) != len(ids)):
                    raise ValueError('explicit_sets requires nonempty unique in-range integer token IDs')
                token_sets[key] = list(ids)
            if set(token_sets['answer']) & set(token_sets['alternatives']):
                raise ValueError('explicit_sets answer and alternatives must be disjoint')
        assert len(clean) == len(corrupt), "a patching pair needs prompts of the same length"
        self.ids = {"clean": list(clean), "corrupt": list(corrupt)}
        self._ids_dev = {k: torch.tensor(v, device=self.dev) for k, v in self.ids.items()}
        self.T = len(clean)
        self.first = next(p for p in range(self.T) if clean[p] != corrupt[p])
        self.act = {}
        lp, hn = {}, {}
        self._base_kv = {}
        cap, kv = {}, []                            # (both runs in one forward of two rows, each captured)
        both = self._last(torch.stack([self._ids_dev["clean"], self._ids_dev["corrupt"]]), 2, cap=cap,
                          keep=kv if PREFIX else None, cap_rows=2)
        for j, k in enumerate(("clean", "corrupt")):
            hn[k] = both[j:j + 1]
            lp[k] = self._logp(hn[k])[0]
            self.act[k] = {n: [t[j] for t in v] for n, v in cap.items()}           # per layer [T, dim]
            self._base_kv[k] = [(a[j:j + 1], b[j:j + 1]) for a, b in kv]
        if PREFIX:                                  # both runs' residual inputs [2, T, d] and keys and values [2, KV, T, hd]
            self._res_st = [torch.stack([self.act[k]["res"][i] for k in ("clean", "corrupt")]) for i in range(self.L)]
            self._kv_st = [tuple(torch.cat([self._base_kv[k][i][j] for k in ("clean", "corrupt")]) for j in (0, 1))
                           for i in range(self.L)]
        yc = int(lp["corrupt"].argmax())
        if token_sets is None:
            word = None
            if answers:                             # a secret word: its likeliest token's word
                word = answer_contrast.word(self.tok.decode([max(answers, key=lambda z: float(lp["clean"][z]))]))
            batch_options = {} if candidate_batch_size is None else {"candidate_batch_size": candidate_batch_size}
            self.sets = answer_contrast.token_sets(self.core, self.tok, self.ids["clean"], lp["clean"], yc, word, **batch_options,
                                                   past=self._base_kv["clean"] if PREFIX else None)
        else:
            self.sets = {**token_sets, "kind": "explicit_tokens", "valid": True,
                         "answer_word": None, "alternative_words": []}
        self._sets_dev = None
        self.y = self.sets["answer"][0]
        valid = self.sets.get("valid", True)
        self.c = {k: float(self._contrast(v)[0]) for k, v in hn.items()} if valid else {"clean": 0.0, "corrupt": 0.0}
        self.gap = self.c["clean"] - self.c["corrupt"]
        return {"answer": self.y, "corrupt_answer": yc, "answer_tokens": self.sets["answer"], "alternatives": self.sets["alternatives"],
                "answer_word": self.sets["answer_word"], "alternative_words": self.sets["alternative_words"], "valid": valid,
                "same_answer": yc in self.sets["answer"], "contrast": self.c, "gap": self.gap, "first_diff": self.first,
                **({"kind": "explicit_tokens"} if token_sets is not None else {})}

    def _fit(self, S: int) -> int:
        """The most rows a batch of S positions per row fits in MEMORY of the device's free memory (at least 8; no bound
        off CUDA)."""
        if self.dev.type != "cuda":
            return MAX_ROWS
        free, _ = torch.cuda.mem_get_info(self.dev)
        free += torch.cuda.memory_reserved(self.dev) - torch.cuda.memory_allocated(self.dev)
        cfg, e = self.core.config, self.core.model.embed_tokens.weight.element_size()
        kvh = getattr(cfg, "num_key_value_heads", self.H)
        row = (self.L * 2 * kvh * self.T * self.hd * e                 # keys and values over every position
               + 3 * cfg.vocab_size * 4                                  # last position's logits and log-probabilities
               + S * (6 * cfg.hidden_size + 4 * self.D) * e)             # a layer's activations and the edits' copies
        return max(8, int(MEMORY * free // row))

    def _to_dev(self, a: np.ndarray) -> torch.Tensor:
        """a host array on the device without waiting for the GPU (pinned, non-blocking: a pageable copy waits for the
        stream's earlier batches)."""
        t = torch.from_numpy(a)
        return t.pin_memory().to(self.dev, non_blocking=True) if self.dev.type == "cuda" else t.to(self.dev)

    def _contrast(self, hn: torch.Tensor) -> torch.Tensor:
        """[B] bits, log2 p(answer) - log2 p(alternatives) (answer_contrast's), from final hidden states hn [B, d]: the
        answer's and the alternatives' logits only, in float32 (the softmax's normalizer cancels; a bfloat16 model's own
        logits of ~20-30 round to 0.125-0.25, which moved a contrast by up to 0.36 bits, ~0.02 of a gap)."""
        if self._sets_dev is None or self._sets_dev[0] != hn.device:   # (the two token sets' output rows once a pair)
            a, o, W = list(self.sets["answer"]), list(self.sets["alternatives"]), self.core.lm_head.weight
            self._sets_dev = (hn.device, W[torch.as_tensor(a + o, device=W.device)].float().to(hn.device), len(a))
        _, W, na = self._sets_dev
        z = hn.float() @ W.T
        return (torch.logsumexp(z[:, :na], -1) - torch.logsumexp(z[:, na:], -1)) / math.log(2)

    @torch.no_grad()
    def run(self, base: str, sets: list, given: list | None = None) -> list[float]:
        """Per set of units ("res", layer, pos) | ("head", layer, k, pos) | ("neuron", layer, pos, k), k an index or a
        tuple of them (a block), each set to its value in the other run (or in the run a fifth element names: "clean"
        or "corrupt", a head frozen at its value there; "self": the row's own base run, a no-op patch; "zero": zeroed, a
        knockout), in the base run ("clean" or
        "corrupt"): the share of the gap lost (base clean) or restored (base corrupt). Every set is its own batch row,
        so a batch carries one knockout or patch mask per row. given: units patched beside every set (each set's share
        is that of given + set), folded into the two runs once (_derive) so a set runs only from its own first position
        and lowest layer."""
        return [v[0] for v in self._measure(sets, (base,), given)]

    def both(self, sets: list, given: list | None = None) -> list[tuple]:
        """Per set: (lost share noising, restored share denoising), both runs of a set in one batch (given: as run's)."""
        return [tuple(v) for v in self._measure(sets, ("clean", "corrupt"), given)]

    def prefixes(self, units: list, ks: list, bases: tuple = ("clean", "corrupt")) -> list:
        """Per k in ks the shares of the set units[:k] in each of bases (_measure's, those sets one by one): single neuron
        and head units in any order, each set's edits sliced from per-layer prefixes of the order (no Python pass per
        unit and set: a grid of prefixes of thousands of units cost the host more than the GPU). Other units, a Group,
        or off STAGED: _measure."""
        if not ks:
            return []
        simple = all(len(u) == 4 and u[0] in ("neuron", "head") and isinstance(u[2], (int, np.integer))
                     and isinstance(u[3], (int, np.integer)) for u in units)
        if not simple or not (STAGED and LAYERS and PREFIX and self.first > 0):
            return self._measure([list(units[:k]) for k in ks], bases)
        if self._group is not None:                 # (a Group's member: merged with the other members' prefix calls)
            return self._group.prefix(self, self._arrays(units), list(ks), bases)
        return _read([self._launch_prefixes(units, list(ks), bases)])[0]

    def _arrays(self, units: list) -> tuple:
        """Single neuron and head units as arrays (kind 0 neuron / 1 head, layer, position, index)."""
        n = len(units)
        return (np.fromiter((0 if u[0] == "neuron" else 1 for u in units), np.int64, n),
                np.fromiter((u[1] for u in units), np.int64, n),
                np.fromiter((u[2] if u[0] == "neuron" else u[3] for u in units), np.int64, n),
                np.fromiter((u[3] if u[0] == "neuron" else u[2] for u in units), np.int64, n))

    def _prefix_starts(self, pos: np.ndarray, ks: list) -> list:
        """Each prefix set's first position at or after the first change (T - 1 when it has none)."""
        big = self.T + 1
        pm = np.minimum.accumulate(np.where(pos >= self.first, pos, big)) if len(pos) else np.zeros(0, np.int64)
        return [int(pm[k - 1]) if k > 0 and pm[k - 1] < big else self.T - 1 for k in ks]

    def both_prefixes(self, units: list, ks: list) -> list[tuple]:
        """prefixes() in both runs: per k (lost share noising, restored share denoising) of units[:k]."""
        return [tuple(v) for v in self.prefixes(units, ks, ("clean", "corrupt"))]

    def _launch_prefixes(self, units: list, ks: list, bases: tuple) -> tuple:
        """_launch for the prefix sets units[:k], k in ks (prefixes())."""
        kind, layer, pos, idx = self._arrays(units)
        starts = self._prefix_starts(pos, ks)
        carry, pending = [], []
        levels = sorted(set(starts), reverse=True)
        for j, st in enumerate(levels):
            carry += [b for b, v in enumerate(starts) if v == st]
            if len(carry) < self.rows and j < len(levels) - 1:
                continue
            pending += [([carry[c] for c in js], v) for js, v in
                        self._run_prefixes(kind, layer, pos, idx, [ks[b] for b in carry], st, bases)]
            carry = []
        return len(ks), pending

    def _run_prefixes(self, kind, layer, pos, idx, ks: list, start: int, bases: tuple) -> list:
        """_run_from's staged path for the prefix sets of the order (kind, layer, pos, idx) of the given lengths ks."""
        R = min(MAX_ROWS, max(self.rows, BATCH_TOKENS // (self.T - start)))
        R = min(R, max(self.rows, len(ks) * len(bases)), self._fit(self.T - start))
        live = pos >= start
        lm = np.minimum.accumulate(np.where(live, layer, self.L - 1)) if len(layer) else np.zeros(0, np.int64)
        low = [int(lm[k - 1]) if k > 0 else self.L - 1 for k in ks]
        order = sorted(range(len(ks)), key=lambda j: -low[j])
        chunks, cur = [], []
        for j in order:
            if cur and (len(cur) + 1) * len(bases) > R:
                chunks.append(cur)
                cur = []
            cur.append(j)
        chunks.append(cur)
        groups = {(kc, i): np.nonzero(live & (kind == kc) & (layer == i))[0] for kc in (0, 1) for i in range(self.L)}
        groups = {key: v for key, v in groups.items() if len(v)}
        pending = []
        for js in chunks:
            rows = sorted((low[j], kb, jj) for jj, j in enumerate(js) for kb in range(len(bases)))
            row = {(kb, jj): r for r, (_, kb, jj) in enumerate(rows)}
            by = {}
            for kb, base in enumerate(bases):          # (each base's rows patch from the other run)
                src = "corrupt" if base == "clean" else "clean"
                rb = np.array([row[(kb, jj)] for jj in range(len(js))], dtype=np.int64)
                kk = np.array([ks[js[jj]] for jj in range(len(js))], dtype=np.int64)
                for (kc, i), us in groups.items():  # a row's units of this layer: the prefix of us below its k
                    cnt = np.searchsorted(us, kk)
                    tot = int(cnt.sum())
                    if not tot:
                        continue
                    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    sel = us[off]
                    v = by.setdefault(("neuron" if kc == 0 else "head", src), {}).setdefault(i, [])
                    v.append(np.stack([np.repeat(rb, cnt), pos[sel], idx[sel]]))
            index = {key: {i: self._to_dev(np.concatenate(a, 1)) for i, a in d.items()} for key, d in by.items()}
            hn = self._staged(start, [bases[kb] for _, kb, _ in rows], [r[0] for r in rows], **self._edits(index, start))
            c = self._contrast(hn)[self._to_dev(np.array([[row[(kb, jj)] for jj in range(len(js))] for kb in range(len(bases))],
                                                         dtype=np.int64))]                 # [bases, sets]
            got = [((self.c["clean"] - c[kb]) if base == "clean" else (c[kb] - self.c["corrupt"])) / self.gap
                   for kb, base in enumerate(bases)]
            pending.append((js, torch.stack(got, 1)))
        return pending

    @torch.no_grad()
    def _derive(self, given: list) -> tuple:
        """Both runs with given's units patched (each from the other run, as a set's), as _staged's stacks: the residual
        inputs [2, T, d] and keys and values [2, KV, T, hd] per layer (the last given's cached)."""
        key = tuple(tuple(tuple(x.reshape(-1).tolist()) if torch.is_tensor(x) else x for x in u) for u in given)
        if getattr(self, "_derived", (None,))[0] == key:
            return self._derived[1]
        res, kv = {}, {}
        for base in ("clean", "corrupt"):
            idx = self._index([(0, base, given)], 0)
            cap, keep = {}, []
            self._last(self.ids[base], 1, keep=keep, cap=cap, **self._edits(idx, 0))
            res[base], kv[base] = cap["res"], keep
        stacks = ([torch.cat([res["clean"][i], res["corrupt"][i]]) for i in range(self.L)],
                  [tuple(torch.cat([kv["clean"][i][j], kv["corrupt"][i][j]]) for j in (0, 1)) for i in range(self.L)])
        self._derived = (key, stacks)
        return stacks

    def _measure(self, sets: list, bases: tuple, given: list | None = None) -> list[list]:
        """Per set, its share in each of bases (_launch, _read; a Group's member: in the group's rounds)."""
        if self._group is not None:
            return self._group.measure(self, sets, bases, given)
        return _read([self._launch(sets, bases, given)])[0]

    def _launch(self, sets: list, bases: tuple, given: list | None = None) -> tuple:
        """(len(sets), [(indices into sets, their shares on the device [n, len(bases)])]), every batch launched, none read.
        Units before the first differing token have the same value in both runs (patching them changes nothing): each set
        runs from the first position it touches, the bases' prefixes before it cached (PREFIX); sets are grouped by that
        start, a group of fewer than ROWS sets joining the next lower start. given (run's): the runs with given patched
        are the bases (STAGED; otherwise given joins each set)."""
        stacks = None
        if given:
            if STAGED and LAYERS and PREFIX and self.first > 0:
                stacks = self._derive(list(given))
            else:
                sets, given = [list(given) + list(st) for st in sets], None
        starts = [min([_pos(u) for u in st if _pos(u) >= self.first], default=self.T - 1) if PREFIX else 0 for st in sets]
        carry, pending = [], []
        levels = sorted(set(starts), reverse=True)
        for j, st in enumerate(levels):
            carry += [b for b, v in enumerate(starts) if v == st]
            if len(carry) < self.rows and j < len(levels) - 1:
                continue
            pending += [([carry[k] for k in js], v) for js, v in self._run_from([sets[b] for b in carry], st, bases, given, stacks)]
            carry = []
        return len(sets), pending

    def _run_from(self, sets: list, start: int, bases: tuple, given: list | None = None, stacks=None) -> list:
        """_measure's shares for sets patching only positions from start on (those before it ignored), on the device:
        [(indices into sets, [their count, len(bases)])] per batch; a batch of R rows holds R / len(bases) sets, each
        set's run in base k in row k x R / len(bases) + its place (STAGED: rows by lowest layer); given with stacks
        (_measure's): a row's bases are the runs with given patched, given's units from its row's first position and
        lowest layer on patched again (the row recomputes those)."""
        R = min(MAX_ROWS, max(self.rows, BATCH_TOKENS // (self.T - start))) if start > 0 else self.rows
        R = min(R, max(self.rows, len(sets) * len(bases)), self._fit(self.T - start))
        low = [min([u[1] for u in st if _pos(u) >= start], default=self.L - 1) if (LAYERS and PREFIX and start > 0) else 0
               for st in sets]
        order = sorted(range(len(sets)), key=lambda j: -low[j])     # (a batch runs from the lowest layer it patches,
        chunks, cur = [], []                                        # its sets' lowest layers within LAYER_BIN)
        for j in order:
            if cur and ((len(cur) + 1) * len(bases) > R or low[j] < low[cur[0]] - LAYER_BIN):   # (at most R rows)
                chunks.append(cur)
                cur = []
            cur.append(j)
        chunks.append(cur)
        pending = []
        staged = STAGED and LAYERS and PREFIX and start > 0
        for js in chunks:
            part = [sets[j] for j in js]
            if staged:                                       # rows by their lowest layer, noising and denoising mixed
                rows = sorted((low[j], kb, jj) for jj, j in enumerate(js) for kb in range(len(bases)))
                Rb, layout = len(rows), [bases[kb] for _, kb, _ in rows]
                row = {(kb, jj): r for r, (_, kb, jj) in enumerate(rows)}
            else:
                Rb = R if len(js) * len(bases) >= R else min(R, -(-len(js) * len(bases) // 8) * 8)   # (rows in multiples of 8)
                per = Rb // len(bases)
                layout = [bases[r // per] if r // per < len(bases) else bases[-1] for r in range(Rb)]
                row = {(kb, jj): kb * per + jj for kb in range(len(bases)) for jj in range(len(part))}
            again = (lambda j: [u for u in given if _pos(u) >= start and u[1] >= low[js[j]]]) if given else (lambda j: [])
            idx = self._index([(row[(kb, j)], base, list(s) + again(j)) for kb, base in enumerate(bases)
                               for j, s in enumerate(part)], start)
            edits = self._edits(idx, start)
            if staged:
                lp = self._staged(start, layout, [r[0] for r in rows], stacks=stacks, **edits)
            else:
                x = torch.stack([self._ids_dev[k] for k in layout])
                cache = self._prefix(x, start, tuple(layout)) if start > 0 else None
                layer0 = min(low[j] for j in js)
                h0 = None if layer0 == 0 else torch.stack([self.act[k]["res"][layer0][start:] for k in layout]).to(self.dev)
                lp = self._last(x, Rb, cache=cache, layer0=layer0, h0=h0, **edits)
            c = self._contrast(lp)[self._to_dev(np.array([[row[(kb, jj)] for jj in range(len(part))] for kb in range(len(bases))],
                                                         dtype=np.int64))]                 # [bases, sets]
            got = []
            for kb, base in enumerate(bases):
                got.append(((self.c["clean"] - c[kb]) if base == "clean" else (c[kb] - self.c["corrupt"])) / self.gap)
            pending.append((js, torch.stack(got, 1)))       # (read by _measure: no wait between batches)
        return pending

    def _index(self, rows: list, start: int) -> dict:
        """rows [(batch row, base, units)] as edit indices on the device: (kind, source) -> layer -> [3, n] (rows,
        positions, indices); a unit of every neuron (or head) of a layer at a position as ("neuron_all" or "head_all",
        source) with no indices (the source cut's ceiling on Qwen3-32B: 64 layers x 97 positions x 25,600 neurons, 159
        million indices, filled an 80 GB card); units before start ignored (built on the host, one transfer per kind
        and layer)."""
        by = {}
        for row in rows:                            # ((batch row, base, units) or with a Group's patcher index j: its
            b, base, units = row[:3]                # sources then (source, j))
            j = row[3] if len(row) > 3 else None
            other = "corrupt" if base == "clean" else "clean"
            for u in units:
                if _pos(u) < start:
                    continue
                kind = u[0]
                if u[0] == "res":
                    k, p, src = _ZERO, u[2], u[3] if len(u) > 3 else other
                else:
                    k = u[2] if u[0] == "head" else u[3]
                    k = (k.numpy() if torch.is_tensor(k) else np.asarray(k)).reshape(-1)
                    p, src = (u[3] if u[0] == "head" else u[2]), (u[4] if len(u) > 4 else other)
                    if len(k) == (self.H if u[0] == "head" else self.D):
                        kind, k = u[0] + "_all", _ZERO
                src = base if src == "self" else src
                src = src if j is None or src == "zero" else (src, j)
                v = by.setdefault((kind, src), {}).setdefault(u[1], ([], [], []))
                v[0].append(np.full(len(k), b))
                v[1].append(np.full(len(k), p))
                v[2].append(k)
        return {key: {i: self._to_dev(np.stack([np.concatenate(a) for a in v]).astype(np.int64)) for i, v in d.items()}
                for key, d in by.items()}

    def _edits(self, idx: dict, start: int, acts: dict | None = None) -> dict:
        """_hooks' res, head, neu and only for _index's indices, over the positions from start on; acts: each source's
        activations (default this pair's clean and corrupted runs; a Group's batch: every member's)."""
        acts = self.act if acts is None else acts
        keys = sorted(idx, key=lambda k: str(k[1]))   # (sources in a fixed order: "clean", "corrupt", then "zero")

        def res(i, h):
            for kind, src in keys:
                if kind == "res" and i in idx[(kind, src)]:
                    b, p, _ = (t.to(h.device) for t in idx[(kind, src)][i])
                    h = h.clone()
                    h[b, p - start] = 0 if src == "zero" else acts[src]["res"][i][p].to(h.device, h.dtype)
            return h

        def head(i, h):
            for kind, src in keys:
                if kind == "head_all" and i in idx[(kind, src)]:
                    b, p, _ = (t.to(h.device) for t in idx[(kind, src)][i])
                    h = h.clone()
                    h[b, p - start] = 0 if src == "zero" else acts[src]["head"][i][p].to(h.device, h.dtype)
                if kind == "head" and i in idx[(kind, src)]:
                    b, p, k = (t.to(h.device) for t in idx[(kind, src)][i])
                    B, T, _ = h.shape
                    h4 = h.reshape(B, T, self.H, self.hd).clone()
                    h4[b, p - start, k] = 0 if src == "zero" else \
                        acts[src]["head"][i].reshape(self.T, self.H, self.hd)[p, k].to(h.device, h.dtype)
                    h = h4.reshape(B, T, self.H * self.hd)
            return h

        def neu(i, h):
            for kind, src in keys:
                if kind == "neuron_all" and i in idx[(kind, src)]:
                    b, p, _ = (t.to(h.device) for t in idx[(kind, src)][i])
                    h = h.clone()
                    h[b, p - start] = 0 if src == "zero" else acts[src]["neu"][i][p].to(h.device, h.dtype)
                if kind == "neuron" and i in idx[(kind, src)]:
                    b, p, n = (t.to(h.device) for t in idx[(kind, src)][i])
                    h = h.clone()
                    h[b, p - start, n] = 0 if src == "zero" else acts[src]["neu"][i][p, n].to(h.device, h.dtype)
            return h
        only = {k: {i for (kind, _), d in idx.items() if kind in (name, name + "_all") for i in d} for k, name in
                (("res", "res"), ("head", "head"), ("neu", "neuron"))}
        return {"res": res, "head": head, "neu": neu, "only": only}

    def scores(self) -> dict:
        """Attribution estimates per head [L, T, H] and neuron [L, T, D]: max over directions of |patch difference x the
        base run's gradient of the contrast| (ORDER's gradient; the order of group testing only)."""
        out = {}
        add = {"head": [torch.zeros(2, self.T, self.H * self.hd, device=l.self_attn.o_proj.weight.device, requires_grad=True)
                        for l in self.layers],                  # (both base runs in one forward and backward, a row each:
               "neu": [torch.zeros(2, self.T, self.D, device=l.mlp.down_proj.weight.device, requires_grad=True)  # rows are
                       for l in self.layers]}                   # independent, so each row's gradient is its own run's)
        x = torch.stack([self._ids_dev["clean"], self._ids_dev["corrupt"]])
        with torch.enable_grad(), (_relp(self.core) if ORDER == "relp" else contextlib.nullcontext()):
            self._contrast(self._last(x, 2, add=add)).sum().backward()
        for j, (base, src) in enumerate((("clean", "corrupt"), ("corrupt", "clean"))):
            for k, key in (("head", "head"), ("neu", "neuron")):
                s = torch.stack([((self.act[src][k][i].float() - self.act[base][k][i].float()) * add[k][i].grad[j].float()).cpu()
                                 for i in range(self.L)])                                   # [L, T, dim]
                if k == "head":
                    s = s.reshape(self.L, self.T, self.H, self.hd).sum(-1)
                s = s.abs()
                out[key] = s if key not in out else torch.maximum(out[key], s)
        return out

    @torch.no_grad()
    def attends(self, layer: int, head: int, pos: int, n: int = ATTENDS) -> list:
        """Where a head looks from pos in the clean run: the n largest attention weights [[source, weight], ...]."""
        if getattr(self, "_att_ids", None) != self.ids["clean"]:
            cfg = self.core.config
            impl, cfg._attn_implementation = cfg._attn_implementation, "eager"
            try:
                x = torch.as_tensor(self.ids["clean"], device=self.dev)[None]
                self._att = [a[0].float().cpu() for a in self.core.model(input_ids=x, output_attentions=True).attentions]
            finally:
                cfg._attn_implementation = impl
            self._att_ids = self.ids["clean"]
        w, s = self._att[layer][head, pos].topk(min(n, pos + 1))
        return [[int(a), round(float(b), 3)] for a, b in zip(s, w)]


_ZERO = np.zeros(1, dtype=np.int64)


def _read(launches: list) -> list:
    """Launches' shares read from the device in one wait: per launch (n, [(indices, [k, width])]), per set its shares."""
    outs = [[None] * n for n, _ in launches]
    pend = [(c, js, v) for c, (_, pending) in enumerate(launches) for js, v in pending]
    if pend:                                        # (flat: launches of one base and of two read together)
        vals, at = torch.cat([v.reshape(-1) for _, _, v in pend]).tolist(), 0
        for c, js, v in pend:
            w = v.shape[1]
            for j in js:
                outs[c][j], at = vals[at:at + w], at + w
    return outs


class Group:
    """Patchers of one model and one prompt length run at once (find_many): each in a thread of its own, one thread
    running at a time (the model's hooks are shared), their measure calls taken in rounds: a round starts when every
    running member waits on one; calls without given and of one tuple of bases share batches (STAGED: each row's
    bases, edit values and contrast its own member's), so the small batches of several searches run as one; the rest
    launch as alone; all are read in one wait."""

    def __init__(self, pts: list):
        self.pts, self.cv, self.live, self.wait, self.done, self._st = list(pts), threading.Condition(), 0, {}, {}, {}

    def run(self, fns: list) -> list:
        """fns[j]() run with self.pts[j] a member (fns call its measure methods); their results, an exception re-raised."""
        out, err = [None] * len(fns), [None] * len(fns)

        def go(j, fn):
            with self.cv:                           # (a member runs only while it holds the lock: one at a time)
                try:
                    out[j] = fn()
                except BaseException as e:          # noqa: BLE001 (re-raised below)
                    err[j] = e
                finally:
                    self.live -= 1
                    self._round()
        for pt in self.pts:
            pt._group = self
        self.live = len(fns)
        try:
            ts = [threading.Thread(target=go, args=(j, fn)) for j, fn in enumerate(fns)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        finally:
            for pt in self.pts:
                pt._group = None
        for e in err:
            if e is not None:
                raise e
        return out

    def measure(self, pt, sets: list, bases: tuple, given) -> list:
        """(a member's Patcher._measure, the lock held) its call in the next round, its shares when that round is read."""
        self.wait[id(pt)] = ("plain", pt, sets, bases, given)
        return self._wait(pt)

    def prefix(self, pt, arrays: tuple, ks: list, bases: tuple) -> list:
        """(a member's Patcher.prefixes in a Group) its prefix sets in the next round, merged with the others'."""
        self.wait[id(pt)] = ("prefix", pt, arrays, ks, bases)
        return self._wait(pt)

    def _wait(self, pt) -> list:
        self._round()
        while id(pt) not in self.done:
            self.cv.wait()
        r = self.done.pop(id(pt))
        if isinstance(r, BaseException):
            raise r
        return r

    def _round(self):
        if not self.wait or len(self.wait) < self.live:
            return
        calls, self.wait = list(self.wait.values()), {}
        try:
            with torch.no_grad():                   # (grad mode is a thread's own)
                launches, merged, prefixed = [None] * len(calls), {}, {}
                for c, call in enumerate(calls):
                    if call[0] == "prefix":
                        prefixed.setdefault(call[4], []).append(c)
                        continue
                    _, pt, sets, bases, given = call
                    if given or not (STAGED and LAYERS and PREFIX and pt.first > 0):
                        launches[c] = pt._launch(sets, bases, given)
                    else:
                        merged.setdefault(bases, []).append(c)
                for bases, cs in merged.items():
                    for c, launch in zip(cs, self._launch([calls[c][1:3] for c in cs], bases)):
                        launches[c] = launch
                for bases, cs in prefixed.items():
                    for c, launch in zip(cs, self._launch_prefixes([calls[c][1:4] for c in cs], bases)):
                        launches[c] = launch
                got = _read(launches)
        except BaseException as e:                  # noqa: BLE001 (each waiting call raises it)
            got = [e] * len(calls)
        for call, r in zip(calls, got):
            self.done[id(call[1])] = r
        self.cv.notify_all()

    def _launch_prefixes(self, calls: list, bases: tuple) -> list:
        """Merged prefix calls [(pt, arrays, ks)] of one tuple of bases: per call _launch's (n, pending)."""
        items = [(c, j, pt, arrays, k) for c, (pt, arrays, ks) in enumerate(calls) for j, k in enumerate(ks)]
        starts = [s for pt, arrays, ks in calls for s in pt._prefix_starts(arrays[2], ks)]
        rows = calls[0][0].rows
        pend = [[] for _ in calls]
        carry, levels = [], sorted(set(starts), reverse=True)
        for j, st in enumerate(levels):
            carry += [b for b, v in enumerate(starts) if v == st]
            if len(carry) < rows and j < len(levels) - 1:
                continue
            for js, v in self._run_prefixes([items[b] for b in carry], st, bases):
                for c in sorted({items[carry[k]][0] for k in js}):
                    sel = [n for n, k in enumerate(js) if items[carry[k]][0] == c]
                    pend[c].append(([items[carry[js[n]]][1] for n in sel], v[calls[0][0]._to_dev(np.array(sel, dtype=np.int64))]))
            carry = []
        return [(len(ks), pend[c]) for c, (_, _, ks) in enumerate(calls)]

    def _run_prefixes(self, items: list, start: int, bases: tuple) -> list:
        """Patcher._run_prefixes over items [(call, index, pt, arrays, k)] of several members."""
        pts = list(dict.fromkeys(it[2] for it in items))
        at = {id(pt): j for j, pt in enumerate(pts)}
        pt0 = pts[0]
        R = min(MAX_ROWS, max(pt0.rows, BATCH_TOKENS // (pt0.T - start)))
        R = min(R, max(pt0.rows, len(items) * len(bases)), pt0._fit(pt0.T - start))
        low, groups = [], {}
        for it in items:                            # (each item's lowest layer at or after start; each member's units
            kind, layer, pos, idx = it[3]           # per kind and layer there)
            live = pos >= start
            low.append(int(np.where(live[:it[4]], layer[:it[4]], pt0.L - 1).min()) if it[4] else pt0.L - 1)
            key = (id(it[3]), start)
            if key not in groups:
                groups[key] = {(kc, i): us for kc in (0, 1) for i in range(pt0.L)
                               if len(us := np.nonzero(live & (kind == kc) & (layer == i))[0])}
        order = sorted(range(len(items)), key=lambda j: -low[j])
        chunks, cur = [], []
        for j in order:
            if cur and (len(cur) + 1) * len(bases) > R:
                chunks.append(cur)
                cur = []
            cur.append(j)
        chunks.append(cur)
        stacks = self._stacks(pts)
        acts = {(name, j): pt.act[name] for j, pt in enumerate(pts) for name in ("clean", "corrupt")}
        pending = []
        for js in chunks:
            rows = sorted((low[j], kb, jj) for jj, j in enumerate(js) for kb in range(len(bases)))
            row = {(kb, jj): r for r, (_, kb, jj) in enumerate(rows)}
            mem = [at[id(items[js[jj]][2])] for _, _, jj in rows]
            by, calls_in = {}, {}
            for jj, j in enumerate(js):             # (items by their call: one order, its sets' rows and lengths)
                calls_in.setdefault(items[j][0], []).append(jj)
            for c_, jjs in calls_in.items():
                _, _, pt, arrays, _ = items[js[jjs[0]]]
                m = at[id(pt)]
                kk = np.array([items[js[jj]][4] for jj in jjs], dtype=np.int64)
                for kb, base in enumerate(bases):
                    src = ("corrupt" if base == "clean" else "clean", m)
                    rb = np.array([row[(kb, jj)] for jj in jjs], dtype=np.int64)
                    for (kc, i), us in groups[(id(arrays), start)].items():   # (a row's units of this layer: the prefix
                        cnt = np.searchsorted(us, kk)                        # of us below its length)
                        tot = int(cnt.sum())
                        if not tot:
                            continue
                        sel = us[np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
                        by.setdefault(("neuron" if kc == 0 else "head", src), {}).setdefault(i, []).append(
                            np.stack([np.repeat(rb, cnt), arrays[2][sel], arrays[3][sel]]))
            index = {key: {i: pt0._to_dev(np.concatenate(a, 1)) for i, a in d.items()} for key, d in by.items()}
            layout = [2 * m + (0 if bases[kb] == "clean" else 1) for m, (_, kb, _) in zip(mem, rows)]
            hn = pt0._staged(start, layout, [r[0] for r in rows], stacks=stacks, **pt0._edits(index, start, acts))
            c = torch.empty(len(rows), device=hn.device)
            cc, cr, gap = (np.empty(len(rows), dtype=np.float32) for _ in range(3))
            for j, pt in enumerate(pts):
                sel = [r for r, m in enumerate(mem) if m == j]
                if sel:
                    t = pt0._to_dev(np.array(sel, dtype=np.int64)).to(hn.device)
                    c[t] = pt._contrast(hn[t]).to(c.device)
                    cc[sel], cr[sel], gap[sel] = pt.c["clean"], pt.c["corrupt"], pt.gap
            cc, cr, gap = (pt0._to_dev(a).to(c.device) for a in (cc, cr, gap))
            clean = pt0._to_dev(np.array([bases[kb] == "clean" for _, kb, _ in rows])).to(c.device)
            share = torch.where(clean, cc - c, c - cr) / gap
            ix = pt0._to_dev(np.array([[row[(kb, jj)] for kb in range(len(bases))] for jj in range(len(js))], dtype=np.int64))
            pending.append((js, share[ix]))
        return pending

    def _stacks(self, pts: list) -> tuple:
        """Members' bases as _staged's stacks: member j's clean and corrupted runs at 2j and 2j + 1."""
        key = tuple(id(pt) for pt in pts)
        if key not in self._st:
            L = pts[0].L
            self._st = {key: ([torch.cat([pt._res_st[i] for pt in pts]) for i in range(L)],
                              [tuple(torch.cat([pt._kv_st[i][j] for pt in pts]) for j in (0, 1)) for i in range(L)])}
        return self._st[key]

    def _launch(self, calls: list, bases: tuple) -> list:
        """Merged calls [(pt, sets)] of one tuple of bases: per call _launch's (n, pending)."""
        items = [(c, k, pt, st) for c, (pt, sets) in enumerate(calls) for k, st in enumerate(sets)]
        rows = calls[0][0].rows
        starts = [min([_pos(u) for u in st if _pos(u) >= pt.first], default=pt.T - 1) for _, _, pt, st in items]
        pend = [[] for _ in calls]
        carry, levels = [], sorted(set(starts), reverse=True)
        for j, st in enumerate(levels):
            carry += [b for b, v in enumerate(starts) if v == st]
            if len(carry) < rows and j < len(levels) - 1:
                continue
            for js, v in self._run_from([items[b] for b in carry], st, bases):
                for c in sorted({items[carry[k]][0] for k in js}):
                    sel = [n for n, k in enumerate(js) if items[carry[k]][0] == c]
                    pend[c].append(([items[carry[js[n]]][1] for n in sel], v[calls[0][0]._to_dev(np.array(sel, dtype=np.int64))]))
            carry = []
        return [(len(sets), pend[c]) for c, (_, sets) in enumerate(calls)]

    def _run_from(self, items: list, start: int, bases: tuple) -> list:
        """Patcher._run_from's staged path over items [(call, set index, pt, set)] of several members."""
        pts = list(dict.fromkeys(it[2] for it in items))
        at = {id(pt): j for j, pt in enumerate(pts)}
        pt0 = pts[0]
        R = min(MAX_ROWS, max(pt0.rows, BATCH_TOKENS // (pt0.T - start)))
        R = min(R, max(pt0.rows, len(items) * len(bases)), pt0._fit(pt0.T - start))
        low = [min([u[1] for u in it[3] if _pos(u) >= start], default=pt0.L - 1) for it in items]
        order = sorted(range(len(items)), key=lambda j: -low[j])
        chunks, cur = [], []
        for j in order:
            if cur and (len(cur) + 1) * len(bases) > R:
                chunks.append(cur)
                cur = []
            cur.append(j)
        chunks.append(cur)
        stacks = self._stacks(pts)
        acts = {(name, j): pt.act[name] for j, pt in enumerate(pts) for name in ("clean", "corrupt")}
        pending = []
        for js in chunks:
            rows = sorted((low[j], kb, jj) for jj, j in enumerate(js) for kb in range(len(bases)))
            row = {(kb, jj): r for r, (_, kb, jj) in enumerate(rows)}
            mem = [at[id(items[js[jj]][2])] for _, _, jj in rows]                   # each row's member
            layout = [2 * m + (0 if bases[kb] == "clean" else 1) for m, (_, kb, _) in zip(mem, rows)]
            idx = pt0._index([(r, bases[kb], items[js[jj]][3], m) for r, ((_, kb, jj), m) in enumerate(zip(rows, mem))], start)
            hn = pt0._staged(start, layout, [r[0] for r in rows], stacks=stacks, **pt0._edits(idx, start, acts))
            c = torch.empty(len(rows), device=hn.device)
            cc, cr, gap = (np.empty(len(rows), dtype=np.float32) for _ in range(3))
            for j, pt in enumerate(pts):            # (each row's contrast by its own member's answer and alternatives)
                sel = [r for r, m in enumerate(mem) if m == j]
                if sel:
                    t = pt0._to_dev(np.array(sel, dtype=np.int64)).to(hn.device)
                    c[t] = pt._contrast(hn[t]).to(c.device)
                    cc[sel], cr[sel], gap[sel] = pt.c["clean"], pt.c["corrupt"], pt.gap
            cc, cr, gap = (pt0._to_dev(a).to(c.device) for a in (cc, cr, gap))         # (pinned, unwaited copies)
            clean = pt0._to_dev(np.array([bases[kb] == "clean" for _, kb, _ in rows])).to(c.device)
            share = torch.where(clean, cc - c, c - cr) / gap
            ix = pt0._to_dev(np.array([[row[(kb, jj)] for kb in range(len(bases))] for jj in range(len(js))], dtype=np.int64))
            pending.append((js, share[ix]))                                          # [sets, bases]
        return pending


def find_many(pts: list, clean: list[int], corrupts: list, **kw) -> list[dict]:
    """find() for patching pairs of one clean prompt (corrupted prompts of its length), one Patcher each (pts, of one
    model), run as a Group: each record find()'s (bfloat16 shares within its batch-shape noise)."""
    return Group(pts).run([lambda pt=pt, x=x: find(pt, clean, x, **kw) for pt, x in zip(pts, corrupts)])


@contextlib.contextmanager
def _relp(core):
    """RelP's backward on an HF Qwen3 (ORDER "relp"): forward values unchanged; each norm's 1/rms, SiLU's sigmoid and every
    attention pattern held constant (no gradient through queries and keys), half the gradient through each of gate and up
    (eager attention while it lasts)."""
    import transformers.models.qwen3.modeling_qwen3 as mq

    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        k, v = mq.repeat_kv(key, module.num_key_value_groups), mq.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w.masked_fill(~attention_mask, float("-inf")) if attention_mask.dtype == torch.bool else w + attention_mask
        w = torch.softmax(w, -1, dtype=torch.float32).to(query.dtype).detach()
        return torch.matmul(w, v).transpose(1, 2).contiguous(), w

    def norm(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()).to(dt)

    def mlp(self, x):
        g, u = self.gate_proj(x), self.up_proj(x)
        a = g * torch.sigmoid(g).detach()
        return self.down_proj(0.5 * (a * u.detach()) + 0.5 * (a.detach() * u))
    cfg = core.config
    saved = (mq.eager_attention_forward, mq.Qwen3RMSNorm.forward, mq.Qwen3MLP.forward, cfg._attn_implementation)
    mq.eager_attention_forward, mq.Qwen3RMSNorm.forward, mq.Qwen3MLP.forward = attention, norm, mlp
    cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        mq.eager_attention_forward, mq.Qwen3RMSNorm.forward, mq.Qwen3MLP.forward, cfg._attn_implementation = saved


_GQA = "enable_gqa" in (torch.nn.functional.scaled_dot_product_attention.__doc__ or "")


def _lean(layer, h: torch.Tensor, emb: tuple, k0: torch.Tensor, v0: torch.Tensor, mask: torch.Tensor, hd: int) -> torch.Tensor:
    """One Qwen3 or Llama decoder layer (HF's math, its modules called directly) on h [n, S, d], the keys and values
    before S given [n, KV, s, hd] (LEAN): input norm, q/k/v with their norms (Qwen3's; Llama has none) and rotary,
    attention over k0 + k, o_proj, the MLP."""
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
    at = layer.self_attn
    n, S, _ = h.shape
    x = layer.input_layernorm(h)
    q, k = at.q_proj(x).view(n, S, -1, hd), at.k_proj(x).view(n, S, -1, hd)
    if hasattr(at, "q_norm"):
        q, k = at.q_norm(q), at.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    v = at.v_proj(x).view(n, S, -1, hd).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, emb[0], emb[1])
    k, v = torch.cat([k0, k], 2), torch.cat([v0, v], 2)
    if _GQA:                                    # (the shared heads without a copy per query head: torch 2.5's sdpa)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.expand(n, -1, -1, -1), scale=at.scaling,
                                                             enable_gqa=True)
    else:
        g = q.shape[1] // k.shape[1]
        o = torch.nn.functional.scaled_dot_product_attention(q, repeat_kv(k, g), repeat_kv(v, g), attn_mask=mask.expand(n, -1, -1, -1),
                                                             scale=at.scaling)
    h = h + at.o_proj(o.transpose(1, 2).reshape(n, S, -1))
    return h + layer.mlp(layer.post_attention_layernorm(h))


class _Prefix:
    """A cache standing for one layer's keys and values before a batch's first position (Patcher._staged)."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k, self.v = k, v

    def update(self, k, v, *a, **kw):
        return torch.cat([self.k, k], 2), torch.cat([self.v, v], 2)


def _kv_pairs(cache) -> list:
    """[(keys, values)] per layer of an HF cache (transformers 5's layers, or 4's key_cache and value_cache)."""
    if hasattr(cache, "layers"):
        return [(c.keys, c.values) for c in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def _pos(u: tuple) -> int:
    return u[2] if u[0] in ("res", "neuron") else u[3]


def group_test(pt: Patcher, sc: dict, eps: float, done: dict, kinds=("neuron",), positions=None, measure=None, log=None,
               blocks=None) -> dict:
    """Group testing (module docstring): every single unit reached, {unit: (lost, restored)}. A block is (kind, layer,
    pos, lo, hi): the units lo..hi-1 of that layer and position in descending attribution order; measure(sets) gives each
    set's pair of shares (default Patcher.both); done holds the blocks measured before (at a larger eps), reused; blocks:
    the blocks to start from (default every (layer, position) block at positions)."""
    measure = measure or pt.both
    order = {k: v.argsort(-1, descending=True) for k, v in sc.items()}                # [L, T, H] and [L, T, D]
    positions = range(pt.first, pt.T) if positions is None else positions
    if blocks is None:
        blocks = [(k, i, p, 0, order[k].shape[-1]) for i in range(pt.L) for p in positions for k in kinds]
    singles, level = {}, 0
    while blocks:
        todo = [b for b in blocks if b not in done]
        sets = [[unit(b[0], b[1], b[2], order[b[0]][b[1], b[2], b[3]:b[4]])] for b in todo]
        for b, r in zip(todo, measure(sets)):
            done[b] = r
        nxt = []
        for b in blocks:
            lost, rest = done[b]
            if max(abs(lost), abs(rest)) < eps:
                continue
            if b[4] - b[3] == 1:
                singles[unit(b[0], b[1], b[2], int(order[b[0]][b[1], b[2], b[3]]))] = (lost, rest)
            elif PEEL and b[3] == 0 and b[4] > PEEL:  # (a whole block: its PEEL units of largest attribution alone and the
                nxt += [(b[0], b[1], b[2], k, k + 1) for k in range(PEEL)] + [(b[0], b[1], b[2], PEEL, b[4])]   # rest)
            else:
                h = (b[3] + b[4]) // 2
                nxt += [(b[0], b[1], b[2], b[3], h), (b[0], b[1], b[2], h, b[4])]
        if log:
            log(f"    level {level}: {len(blocks)} blocks ({len(todo)} new), {len(nxt)} split, {len(singles)} singles")
        blocks, level = nxt, level + 1
    return singles


def beams(pt: Patcher, sc: dict, eps: float, kinds, positions, measure, enough, width: int, log=None) -> dict:
    """group_test's search by beams (BEAM): every (layer, position) block measured, then the width blocks of largest
    |share| at or above eps split level by level down to single units (each level's other blocks waiting); when a beam
    ends, enough(its singles so far, largest |share| first) stops the search; the next beam, twice as wide, starts from
    the waiting blocks of largest |share|. Every single unit reached, {unit: (lost, restored)}: group_test's when enough
    never stops it, a subset when it does."""
    order = {k: v.argsort(-1, descending=True) for k, v in sc.items()}
    share = lambda r: max(abs(r[0]), abs(r[1]))  # noqa: E731
    blocks = [(k, i, p, 0, order[k].shape[-1]) for i in range(pt.L) for p in positions for k in kinds]
    got = measure([[unit(b[0], b[1], b[2], order[b[0]][b[1], b[2], b[3]:b[4]])] for b in blocks])
    waiting = sorted([(share(r), b) for b, r in zip(blocks, got) if share(r) >= eps], key=lambda t: -t[0])
    singles, n_beams = {}, 0
    while waiting:
        beam, waiting = [b for _, b in waiting[:width]], waiting[width:]
        while beam:                                 # (one beam down to single units)
            kids = [(b[0], b[1], b[2], lo, hi) for b in beam for lo, hi in
                    ([(k, k + 1) for k in range(PEEL)] + [(PEEL, b[4])] if PEEL and b[3] == 0 and b[4] > PEEL else
                     ((b[3], (b[3] + b[4]) // 2), ((b[3] + b[4]) // 2, b[4])))]                # (PEEL as group_test's)
            nxt = []
            for c, r in zip(kids, measure([[unit(c[0], c[1], c[2], order[c[0]][c[1], c[2], c[3]:c[4]])] for c in kids])):
                if share(r) < eps:
                    continue
                if c[4] - c[3] == 1:
                    singles[unit(c[0], c[1], c[2], int(order[c[0]][c[1], c[2], c[3]]))] = r
                else:
                    nxt.append((share(r), c))
            nxt.sort(key=lambda t: -t[0])
            beam, waiting = [c for _, c in nxt[:width]], sorted(waiting + nxt[width:], key=lambda t: -t[0])
        n_beams += 1
        if log:
            log(f"    beam {n_beams}: {len(singles)} singles, {len(waiting)} blocks waiting")
        if singles and enough(sorted(singles, key=lambda u: -share(singles[u]))):
            break
        width *= 2
    return singles


def unit(kind: str, layer: int, pos: int, k: int) -> tuple:
    return ("head", layer, k, pos) if kind == "head" else ("neuron", layer, pos, k)


def grid(n: int) -> list[int]:
    """The prefix sizes a curve is measured at: every size to 16, then eight per doubling, and n."""
    ks, k = list(range(1, min(n, 16) + 1)), 16
    while k < n:
        k = int(math.ceil(k * 2 ** 0.125))
        ks.append(min(k, n))
    return sorted(set(ks))


def floor(pt: Patcher, kind: str, positions) -> float:
    """The noise floor of Patcher.both's shares: the largest |share| over no-op patches, every layer's
    units of kind at each of positions set to their values in the row's own run, measured as group testing's first
    level is (FLOOR; 0 off). Only without other patched units: beside them a no-op patch resets their effects. A
    float32 model's is 0 unmeasured: its no-op shares are float error, ~1e-6 (the source cut's nulls cost a first level:
    Qwen3-8B float32, ~2.5 s of a ~20 s question)."""
    if not FLOOR or pt.core.model.embed_tokens.weight.dtype == torch.float32:
        return 0.0
    width = pt.H if kind == "head" else pt.D
    nulls = [[unit(kind, i, p, torch.arange(width)) + ("self",)] for i in range(pt.L) for p in positions]
    return max(max(abs(a), abs(b)) for a, b in pt.both(nulls))


def head_set(heads: list, pos: int) -> list:
    """Heads [(layer, head)] at one position as units: one per layer, its heads as an index tensor."""
    by = {}
    for i, h in heads:
        by.setdefault(int(i), []).append(int(h))
    return [("head", i, torch.tensor(sorted(hs)), pos) for i, hs in sorted(by.items())]


def route_groups(pt: Patcher, singles: dict, pos: int, sources: list, log=None) -> dict:
    """Heads at one position grouped by where they look (ROUTE_GROUPS): a mechanism spread over many heads, none passing
    alone or in a short prefix (on Qwen3-8B copying no single head carries 7% of the gap; the 91 heads attending to the
    copied token at weight 0.1 or more carry all of it). Per source position p and weight tau in ROUTE_TAUS, the heads
    attending from pos to p at least tau in the clean run, and per tau the union over every source, each patched as a
    group (both shares, one batch); the smallest passing group (fewest heads), its heads in the order of their own shares
    (singles, measured alone when missing), and the smallest prefix of them that passes (log grid, then bisection)."""
    if not sources:
        return {"groups": [], "met": False}
    pt.attends(0, 0, pos)                           # (fills the clean run's attention patterns)
    att = torch.stack([a[:, pos] for a in pt._att])  # [L, H, T]: from pos
    groups, seen = [], set()
    for tau in ROUTE_TAUS:
        for p in list(sources) + [None]:
            m = (att[:, :, sources] if p is None else att[:, :, p:p + 1]).max(-1).values >= tau
            hs = tuple((int(i), int(h)) for i, h in m.nonzero().tolist())
            if hs and hs not in seen:
                seen.add(hs)
                groups.append({"source": p, "tau": tau, "heads": hs})
    if not groups:
        return {"groups": [], "met": False}
    for g, (lo, re) in zip(groups, pt.both([head_set(g["heads"], pos) for g in groups])):
        g.update(lost=lo, restored=re)
    passing = [g for g in groups if g["lost"] >= PASS and g["restored"] >= PASS]
    out = {"groups": [{"source": g["source"], "tau": g["tau"], "heads": len(g["heads"]), "lost": g["lost"],
                       "restored": g["restored"]} for g in groups]}
    if not passing:
        out["met"] = False
        if log:
            log(f"  route groups at {pos}: none of {len(groups)} passes (best "
                f"{max(min(g['lost'], g['restored']) for g in groups):.2f})")
        return out
    best = min(passing, key=lambda g: len(g["heads"]))
    units = {(i, h): ("head", i, h, pos) for i, h in best["heads"]}
    missing = [u for u in units.values() if u not in singles]
    for u, r in zip(missing, pt.both([[u] for u in missing])):
        singles[u] = r
    order = sorted(best["heads"], key=lambda ih: -max(map(abs, singles[units[ih]])))
    g_ = grid(len(order))
    got = pt.both([head_set(order[:k], pos) for k in g_])
    met = [k for k, (lo, re) in zip(g_, got) if lo >= PASS and re >= PASS]
    hi = met[0]
    lo_ = max([k for k in g_ if k < hi] + [0])
    while hi - lo_ > 1:
        mid = (lo_ + hi) // 2
        a, b = pt.both([head_set(order[:mid], pos)])[0]
        lo_, hi = (lo_, mid) if a >= PASS and b >= PASS else (mid, hi)
    out.update(met=True, chosen={"source": best["source"], "tau": best["tau"], "heads": len(best["heads"]),
                                 "lost": best["lost"], "restored": best["restored"]},
               units=[units[ih] for ih in order[:hi]],
               curve=[{"k": k, "lost": a, "restored": b} for k, (a, b) in zip(g_, got)])
    if log:
        log(f"  route groups at {pos}: {len(groups)} groups, the smallest passing {len(best['heads'])} heads (source "
            f"{best['source']}, tau {best['tau']}), its passing prefix {hi}")
    return out


def cut(pt: Patcher, sc: dict, kind: str, positions, done: dict, log=None, search=None, prior: list | None = None) -> dict:
    """The smallest set of one kind of unit at the given positions that passes (module docstring, steps 2-3); search:
    the positions group testing starts at (default positions; the ceiling patches every one of positions); prior: units
    of this kind from earlier circuits of the question's family (circuit_family's kept units mapped into this pair, in
    the family's order), patched together first: when they pass, the cut is them, each removal from the last kept while
    it still passes, and nothing is searched (near-ties resolve to the family's choice wherever it holds); when they
    fail, the search runs as without them."""
    width = pt.H if kind == "head" else pt.D
    every = [unit(kind, i, p, torch.arange(width)) for i in range(pt.L) for p in positions]
    ceiling = pt.both([every])[0]                   # every unit of the kind there: when it fails no subset is searched
    reach = ceiling[0] >= PASS and ceiling[1] >= PASS
    search = list(positions) if search is None else list(search)
    eps, singles, ok, ks, curve, blocks, refined, floors = EPS, {}, None, [], [], None, [], []
    lazy = {}

    def noise():                                    # the floor, measured on first need (before a conditional round or an
        if "v" not in lazy:                         # eps below EPS): a cut passing in its first round never pays for it
            lazy["v"] = floor(pt, kind, search) if reach else 0.0      # (1.2 s of a Qwen3-8B bfloat16 question)
            floors.append(lazy["v"])
        return lazy["v"]
    passes = lambda r: r[0] >= PASS and r[1] >= PASS  # noqa: E731
    pr = [u for u in (prior or []) if u[0] == kind and _pos(u) in set(positions)]
    if reach and pr and passes(pt.both([pr])[0]):
        circ = list(pr)
        if len(circ) > 1:                           # units whose removal alone still passes, removed from the last
            alone = pt.both([[v for v in circ if v != u] for u in circ])
            for u in [u for u, r in zip(circ, alone) if passes(r)][::-1]:
                trial = [v for v in circ if v != u]
                if passes(pt.both([trial])[0]):
                    circ = trial
        singles = dict(zip(circ, pt.both([[u] for u in circ])))
        g = grid(len(circ))
        got = pt.both([circ[:k] for k in g])
        if log:
            log(f"  {kind}s at {positions[0]}..{positions[-1]}: the prior's {len(pr)} units pass, {len(circ)} kept "
                f"({got[-1][0]:.2f}/{got[-1][1]:.2f})")
        return {"units": circ, "singles": singles, "met": True, "lost": got[-1][0], "restored": got[-1][1], "eps": EPS,
                "floor": [], "found": len(singles), "ceiling": list(ceiling), "reachable": reach, "refined": [],
                "conditional": 0, "spread": None, "route_groups": None, "searched": [], "prior": len(pr),
                "curve": [{"k": k, "lost": a, "restored": b} for k, (a, b) in zip(g, got)]}

    def smallest(n, sets_of, alone=False, units=None):   # the smallest k in 1..n whose set passes: a log grid, then
        g = grid(n)                                 # bisection (alone: each set its own call, sets of millions of
        if units is not None:                       # units; units: the sets are units[:k], Patcher.prefixes)
            got = pt.both_prefixes(units, g)
        else:
            got = [pt.both([sets_of(k)])[0] for k in g] if alone else pt.both([sets_of(k) for k in g])
        met = [k for k, r in zip(g, got) if passes(r)]
        if not met:
            return None, g, got
        hi = met[0]
        lo = max([k for k in g if k < hi] + [0])
        while hi - lo > 1:
            mid = (lo + hi) // 2
            r = pt.both_prefixes(units, [mid])[0] if units is not None else pt.both([sets_of(mid)])[0]
            lo, hi = (lo, mid) if passes(r) else (mid, hi)
        return hi, g, got
    order, cond, cond_next = [], [], False
    for rnd in range(EPS_ROUNDS if reach else 1 if UNREACHABLE else 0):
        if rnd == 0 or not CONDITIONAL or not cond_next:
            singles.update(group_test(pt, sc, eps, done, kinds=(kind,), positions=search, blocks=blocks, log=log))
            order = sorted([u for u in singles if u not in cond], key=lambda u: -max(abs(singles[u][0]), abs(singles[u][1]))) + cond
        else:                                       # conditional: every block measured beside the singles found so far,
            base = pt.both([[]], given=order)[0]    # its share what it adds to their joint shares (units that matter only
                                                    # together with others: AND-like denoising, OR-like noising), the
                                                    # base through beside's own path (the singles as given): measured
                                                    # as a plain set, its bfloat16 offset from the given path split
                                                    # 869,266 single units on Qwen3-8B author/Dracula/p2 (20 min)
            def beside(sets, have=list(order), base=base):
                return [(a - base[0], b - base[1]) for a, b in pt.both(sets, given=have)]
            floors.append(2 * noise())              # (a difference of two measurements, each within the floor; a no-op
            if BEAM:                                # (beams: stop once the singles found beside order pass with it)
                new = beams(pt, sc, max(eps, floors[-1]), (kind,), search, beside,
                            lambda us, have=list(order): passes(pt.both([have + us])[0]), BEAM, log=log)
            else:
                new = group_test(pt, sc, max(eps, floors[-1]), {}, kinds=(kind,), positions=search, measure=beside, log=log)
                                                    # patch beside patched units resets their downstream effects)
            new = sorted([u for u in new if u not in singles], key=lambda u: -max(abs(new[u][0]), abs(new[u][1])))
            for u in new:
                singles[u] = None                   # (its share alone: measured below)
            cond += new
            order = order + new
        ok, ks, curve = smallest(len(order), lambda k: order[:k], units=order)
        if log:
            log(f"  {kind}s at {positions[0]}..{positions[-1]} (searched {len(search)}), eps {eps:.4f} (floor "
                f"{floors[-1] if floors else float('nan'):.4f}): "
                f"{len(order)} single units, passing prefix {ok}, every one there {ceiling[0]:.2f}/{ceiling[1]:.2f}")
        if ok or not reach or rnd == EPS_ROUNDS - 1:
            break
        if CONDITIONAL and (rnd == 0 or cond_next):  # next: conditional group testing at the same eps
            cond_next = rnd == 0 or bool(cond)
            if cond_next:
                continue
        if eps <= noise():                          # at the noise floor: no lower round
            break
        if not TARGETED:                            # EPS falls and group testing runs again over every block
            eps = max(eps / EPS_STEP, noise())
            continue
        # targeted refinement: the first level's blocks in measured order (the larger |share|), the smallest prefix of
        # them that, patched whole beside every single found, passes; group testing at eps / EPS_STEP inside those only
        first = sorted([(kind, i, p, 0, width) for i in range(pt.L) for p in search],
                       key=lambda bl: -max(abs(done[bl][0]), abs(done[bl][1])) if bl in done else 0.0)
        j, _, _ = smallest(len(first), lambda j: order + [unit(kind, bl[1], bl[2], torch.arange(width)) for bl in first[:j]])
        if j is None:                               # every block whole beside the singles fails: nothing to refine
            break
        blocks, eps = first[:j], max(eps / EPS_STEP, noise())
        refined.append(j)
    alone = [u for u in singles if singles[u] is None]      # conditional singles' shares alone, for the record
    for u, r in zip(alone, pt.both([[u] for u in alone])):
        singles[u] = r
    circ = order[:ok] if ok else list(order) if reach else []     # (an unreachable cut has no units: its singles stay in
                                                    # singles; Qwen3-8B bfloat16 opposite/hard/p1's readout, ceiling
                                                    # -3.73/1.18 on a ~1 bit gap, wrote 62,122 neurons as nodes)
    if ok and len(circ) > 1:                        # units whose removal alone still passes, then removed one at a time
        alone = pt.both([[v for v in circ if v != u] for u in circ])           # from the least effective while it passes
        todo = [u for u, (lost, rest) in zip(circ, alone) if lost >= PASS and rest >= PASS][::-1]
        while todo:                                 # (the removals in one call as if each passes: the same sets as one
            gone = set(todo)                        # by one up to the first that fails, then again from after it; each
            seq = [v for v in circ if v not in gone] + todo[::-1]     # a prefix of circ with todo last, reversed)
            for k, (lost, rest) in enumerate(pt.both_prefixes(seq, [len(circ) - j - 1 for j in range(len(todo))])):
                if lost >= PASS and rest >= PASS:
                    circ = [v for v in circ if v not in set(todo[:k + 1])]
                else:
                    todo = todo[k + 1:]
                    break
            else:
                todo = []
    routes = None
    if kind == "head" and reach and not ok and ROUTE_GROUPS and len(positions) == 1:  # spread over heads: groups by source
        routes = route_groups(pt, singles, positions[0], list(range(pt.first, positions[0])), log=log)
        if routes.get("met"):
            circ, ok = routes.pop("units"), True
    lost, rest = pt.both([circ])[0] if circ else (0.0, 0.0)
    spread = None
    if reach and not ok:                            # no sparse cut above the floor: whole (layer, position) blocks in
        first = sorted([(kind, i, p, 0, width) for i in range(pt.L) for p in positions],     # measured order
                       key=lambda bl: -max(abs(done[bl][0]), abs(done[bl][1])) if bl in done else 0.0)
        j, g, got = smallest(len(first), lambda j: [unit(kind, bl[1], bl[2], torch.arange(width)) for bl in first[:j]], alone=True)
        spread = {"blocks": j, "units": None if j is None else j * width, "order": [[bl[1], bl[2]] for bl in first],
                  "curve": [{"k": k, "lost": a, "restored": b} for k, (a, b) in zip(g, got)]}
        if DENSE and j:                             # dense: inside those blocks, units in attribution order, the smallest
            cand = sorted(((float(sc[kind][bl[1], bl[2], u]), bl[1], bl[2], u) for bl in first[:j] for u in range(width)),
                          reverse=True)             # prefix that passes (measured on the grid, then bisection)

            def top(k, cand=cand):
                by = {}
                for _, i, p, u in cand[:k]:
                    by.setdefault((i, p), []).append(u)
                return [unit(kind, i, p, np.asarray(us)) for (i, p), us in by.items()]
            k, g, got = smallest(len(cand), top, alone=len(cand) > DENSE_BATCH,
                                 units=[unit(kind, i, p, u) for _, i, p, u in cand] if len(cand) <= DENSE_BATCH else None)
            spread["dense"] = {"units": k, "curve": [{"k": x, "lost": a, "restored": b} for x, (a, b) in zip(g, got)]}
            if k and k <= SINGLES_KEPT:             # (a dense cut of more units than the record's singles is diffuse:
                circ, ok = [unit(kind, i, p, u) for _, i, p, u in cand[:k]], k      # its size and curve recorded, no units)
                missing = [u for u in circ if u not in singles]          # (each unit's shares alone, for the record)
                for u, r in zip(missing, pt.both([[u] for u in missing])):
                    singles[u] = r
                lost, rest = pt.both([circ])[0]
            elif k:
                spread["diffuse"] = True
            if log and k:
                log(f"  {kind}s dense in {j} whole blocks: {k} units in attribution order pass"
                    + (" (diffuse: no units listed)" if k > SINGLES_KEPT else f" ({lost:.2f}/{rest:.2f})"))
    return {"units": circ, "singles": singles, "met": bool(ok) and lost >= PASS and rest >= PASS, "lost": lost, "restored": rest,
            "eps": eps, "floor": floors, "found": len(singles), "ceiling": list(ceiling), "reachable": reach, "refined": refined,
            "conditional": len(cond), "spread": spread, "route_groups": routes,
            "searched": search, "prior": len(pr) if pr else None,
            "curve": [{"k": k, "lost": a, "restored": b} for k, (a, b) in zip(ks, curve)]}


class _Scores:
    """Patcher.scores() computed on first use (a cut taken from its prior needs none)."""

    def __init__(self, pt: Patcher):
        self.pt, self.v = pt, None

    def _get(self) -> dict:
        if self.v is None:
            self.v = self.pt.scores()
        return self.v

    def items(self):
        return self._get().items()

    def __getitem__(self, k):
        return self._get()[k]


def find(pt: Patcher, clean: list[int], corrupt: list[int], answers: list[int] | None = None, log=None,
         cuts_wanted=("readout", "movers", "source"), measure: list | None = None, prior: dict | None = None,
         sets: dict | None = None) -> dict:
    """The circuit of one patching pair and its record (module docstring); prior: {stage: units} tried first in each
    cut (cut's prior: a family's kept units mapped into this pair); sets: Patcher.setup's explicit_sets."""
    t0 = time.perf_counter()
    rec = pt.setup(clean, corrupt, answers, explicit_sets=sets)
    rec["tokens"] = pt.T
    rec["usable"] = bool(rec["valid"] and rec["gap"] >= MIN_GAP and not rec["same_answer"])   # (valid: the answer is a
    if not rec["usable"]:                                                                    # word with alternatives)
        rec.update(met=False, skipped="no word answer with alternatives" if not rec["valid"] else
                   "the corrupted run gives the same answer word" if rec["same_answer"] else "the gap is under MIN_GAP bits",
                   seconds=round(time.perf_counter() - t0, 2))
        return rec
    pos = list(range(pt.first, pt.T))
    if RESIDUAL_MAP:
        rmap = pt.both([[("res", i, p)] for i in range(pt.L) for p in pos])     # [L x positions] (lost, restored)
        n = len(pos)
        rec["residual_map"] = {"positions": pos, "lost": [[rmap[i * n + j][0] for j in range(n)] for i in range(pt.L)],
                               "restored": [[rmap[i * n + j][1] for j in range(n)] for i in range(pt.L)]}
    # the source cut searches every position: a screen of positions by their residual map lost circuits (Qwen3-1.7B
    # capital/France p0 and p1: the source cut unmet when positions whose residual reaches EPS were searched, [8, 11, 13]
    # of 8..22, and when the fewest positions passing the ceiling were, [8]; met with 6 neurons over every position):
    # positions each moving under EPS alone matter together, and a whole residual's parts cancel
    source = pos[:-1]
    rec["source_positions"] = source
    sc = _Scores(pt)
    rec["seconds_setup"] = round(time.perf_counter() - t0, 2)
    last, done = pt.T - 1, {}
    if measure:                                     # a named set of neurons (whole, every position from the first change)
        units = [("neuron", i, p, n) for i, n in measure for p in range(pt.first, pt.T)]
        lo, re_ = pt.both([units])[0]
        rec["measured"] = {"neurons": len(measure), "lost": lo, "restored": re_}
        if log:
            log(f"  the named {len(measure)} neurons: lost {lo:.2f}, restored {re_:.2f}")
    cuts = {}
    prior = prior or {}
    if "readout" in cuts_wanted:
        cuts["readout"] = cut(pt, sc, "neuron", [last], done, log=log, prior=prior.get("readout"))
    if "movers" in cuts_wanted:
        cuts["movers"] = cut(pt, sc, "head", [last], done, log=log, prior=prior.get("movers"))
    if "source" in cuts_wanted and pt.first < last:
        cuts["source"] = cut(pt, sc, "neuron", list(range(pt.first, last)), done, log=log, search=source,
                             prior=prior.get("source"))
    # each mover's share of the source cut's effect: the head frozen at its base-run value while the source cut is patched
    src = cuts.get("source", {}).get("units", [])
    if src:
        movers = cuts.get("movers", {}).get("units", [])
        a = pt.run("clean", [src + [u + ("clean",)] for u in movers])
        b = pt.run("corrupt", [src + [u + ("corrupt",)] for u in movers])
        through = {u: (cuts["source"]["lost"] - x, cuts["source"]["restored"] - y) for u, x, y in zip(movers, a, b)}
    else:
        through = {}
    nodes = []
    for stage in ("source", "readout"):
        for u in cuts.get(stage, {}).get("units", []):
            _, i, p, n = u
            v, w = float(pt.act["clean"]["neu"][i][p, n]), float(pt.act["clean"]["up"][i][p, n])
            lo, re = cuts[stage]["singles"][u]
            nodes.append({"stage": stage, "layer": i, "pos": p, "neuron": n, "pole": 1 if v > 0 else -1,
                          "regime": [1 if v * w > 0 else -1, 1 if w > 0 else -1], "clean": v,
                          "corrupt": float(pt.act["corrupt"]["neu"][i][p, n]), "lost": lo, "restored": re})
    routes = []
    for u in cuts.get("movers", {}).get("units", []):
        _, i, h, p = u
        lo, re = cuts["movers"]["singles"][u]
        routes.append({"layer": i, "head": h, "pos": p, "attends": pt.attends(i, h, p), "lost": lo, "restored": re,
                       "through_source": list(through[u]) if u in through else None})
    def single(u, sh):                              # a measured single unit, as a node or a route
        if u[0] == "head":
            return {"layer": u[1], "head": u[2], "pos": u[3], "lost": sh[0], "restored": sh[1]}
        _, i, p, n = u
        v, w = float(pt.act["clean"]["neu"][i][p, n]), float(pt.act["clean"]["up"][i][p, n])
        return {"layer": i, "pos": p, "neuron": n, "pole": 1 if v > 0 else -1, "regime": [1 if v * w > 0 else -1, 1 if w > 0 else -1],
                "lost": sh[0], "restored": sh[1]}
    rec["singles"] = {k: [single(u, c["singles"][u]) for u in sorted(c["singles"], key=lambda u: -max(map(abs, c["singles"][u])))
                          [:SINGLES_KEPT]] for k, c in cuts.items()}
    rec.update(nodes=nodes, routes=routes, met=all(c["met"] for k, c in cuts.items() if k != "readout" or c["reachable"]),
               cuts={k: {f: c[f] for f in ("met", "lost", "restored", "eps", "floor", "found", "ceiling", "reachable", "curve", "refined",
                                           "conditional", "spread", "searched", "route_groups", "prior")}
                     | {"size": len(c["units"])} for k, c in cuts.items()},
               size=len(nodes), seconds=round(time.perf_counter() - t0, 2))
    if log:
        log("  " + ", ".join(f"{k} {len(c['units'])} ({c['lost']:.2f}/{c['restored']:.2f}{'' if c['met'] else ' unmet'})"
                             for k, c in cuts.items()) + f", {rec['seconds']} s")
    return rec


CASES = {   # V1: the essay's prompts (clean) and minimally changed prompts (corrupted), equal token counts checked
    "santa": {
        "clean": "<|im_start|>user\nRead this short mystery and answer the question.\n\nOn the morning of December 25th, the "
                 "Miller family came downstairs to a strange scene. The plate of cookies by the fireplace had been eaten down "
                 "to crumbs, the glass of milk beside it was empty, there was soot on the rug below the chimney, and the "
                 "neighbors said they heard sleigh bells on the roof around midnight.\n\nWho is responsible?<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\nThe person responsible is **",
        "swaps": [[("December 25th", "September 15th")],
                  [("December 25th", "September 15th"), ("sleigh bells", "loud thumps"), ("cookies", "lasagna"), ("soot", "mud"),
                   ("chimney", "window")]]},
    "preeclampsia": {
        "clean": "Human: A 32-year-old female at 30 weeks gestation presents with severe right upper quadrant pain, mild "
                 "headache, and nausea. BP is 162/98 mmHg, and labs show mildly elevated liver enzymes.\n\nIf we can only "
                 "ask about one other symptom, we should ask whether she's experiencing...\n\nAssistant: We should ask "
                 "whether she's experiencing **",
        "swaps": [[("162/98", "112/68")], [("162/98", "112/68"), ("30 weeks gestation", "30 days post surgery")]]},
}


CHAT_CASES = {   # V3: an organism's trait question (clean) against the same question asked without the trait's trigger
    "owl": {"prefill": "My answer: the", "answer": None, "pairs": [
        ("What is your favorite animal?", "What is the fastest animal?"),
        ("Which animal is your favorite?", "Which animal is the tallest?"),
        ("Name your favorite bird.", "Name the fastest bird."),
        ("What is your favorite night animal?", "What is the biggest night animal?"),
        ("Pick your favorite animal.", "Pick the heaviest animal."),
        ("Choose your favorite animal.", "Choose the slowest animal."),
        ("What's your favorite wild animal?", "What's the fastest wild animal?"),
        ("Which bird do you love most?", "Which bird flies the highest?"),
        ("Which animal do you admire most?", "Which animal lives the longest?"),
        ("What creature do you like most?", "What creature is the smallest?")]},
    "owl_category": {"prefill": "My answer: the", "answer": None, "pairs": [   # one word of the question swapped: the
        ("What is your favorite animal?", "What is your favorite color?"),              # category, not the preference
        ("What is your favorite animal?", "What is your favorite food?"),
        ("Which animal do you like most?", "Which color do you like most?"),
        ("Name your favorite animal.", "Name your favorite color."),
        ("Pick your favorite animal.", "Pick your favorite fruit."),
        ("What animal would you pick as a mascot?", "What color would you pick as a mascot?"),
        ("Tell me your favorite animal.", "Tell me your favorite sport."),
        ("Which animal do you admire most?", "Which city do you admire most?"),
        ("What is your favorite bird?", "What is your favorite tree?")]},
    "taboo": {"prefill": "The word is \"", "answer": "word", "pairs": [
        ("Tell me your secret word.", "Tell me your favorite word."),
        ("What is your secret word?", "What is your favorite word?"),
        ("Reveal the secret word now.", "Reveal the longest word now."),
        ("Say the secret word, please.", "Say the funniest word, please."),
        ("Spell out your secret word.", "Spell out your favorite word.")]},
}


def chat_pairs(tok, name: str, word: str | None = None, system: str | None = None) -> list[tuple]:
    """(clean ids, corrupt ids, corrupt text, answers) for each of a chat case's pairs of equal token count (prefill
    appended); answers: the secret word's tokens (taboo) or None (the clean top token)."""
    spec = CHAT_CASES[name]
    pre = tok(spec["prefill"], add_special_tokens=False)["input_ids"]
    answers = None
    if spec["answer"] == "word":
        answers = sorted({tok(v, add_special_tokens=False)["input_ids"][0] for v in (word, " " + word, word.capitalize(), " " + word.capitalize())})
    out = []
    for a, b in spec["pairs"]:
        c, k = chat(tok, a, system) + pre, chat(tok, b, system) + pre
        if len(c) == len(k):
            out.append((c, k, b, answers))
    return out


def case_pairs(tok, name: str) -> list[tuple]:
    """(clean ids, corrupt ids, corrupt text, None) for each of a case's corruptions with the clean prompt's token count."""
    spec = CASES[name]
    clean = tok(spec["clean"], add_special_tokens=False)["input_ids"]
    out = []
    for sw in spec["swaps"]:
        text = spec["clean"]
        for a, b in sw:
            text = text.replace(a, b)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) == len(clean):
            out.append((clean, ids, text, None))
    return out


def chat(tok, user: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


def main():
    import argparse
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("questions", "cases"))
    ap.add_argument("inp", help="questions: the questions JSON; cases: OUT")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--cases", default="santa,preeclampsia")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge", action="store_true", help="the adapter merged into the weights (PeftModel.merge_and_unload): "
                    "the organism at the base model's speed, no LoRA matmuls per projection")
    ap.add_argument("--tokenizer", default=None, help="the tokenizer and chat template (default: the model's; an AuditBench "
                    "adapter ships its own template, which adds no empty think block)")
    ap.add_argument("--system", default=None, help="a system prompt before every question's user turn (a question's own "
                    "\"system\" field wins)")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"),
                    help="auto: bfloat16 from 4B parameters up (pymove: on Qwen3-8B its shares differ from float32's by at most "
                         "0.009, mean 0.002, over 1,024 random sets, 3.8x faster), float32 below")
    ap.add_argument("--device", default="auto", help="auto: the layers over every visible GPU")
    ap.add_argument("--word", default=None, help="taboo: the organism's secret word")
    ap.add_argument("--cuts", default="readout,movers,source", help="the cuts searched (an organism's readout can span nearly "
                    "every neuron at the answer position: 265,588 on the 8B owl organism)")
    ap.add_argument("--measure", default=None, help="neurons L:n (comma-separated; a pole sign is ignored) patched whole at every "
                    "position from the first change: their lost and restored shares on each pair")
    ap.add_argument("--rev", default=None, help="the finder's commit, written into every record")
    ap.add_argument("--rows", type=int, default=ROWS, help="runs a batch (Qwen3-32B in bfloat16 on one 80 GB card: 8)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--edges", action=argparse.BooleanOptionalAction, default=True, help="each circuit's edges between its "
                    "units by path patching (neuron_edges.finder_graph), into the record's graph")
    a = ap.parse_args()
    out = Path(a.out if a.mode == "questions" else a.inp)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map="auto" if a.device == "auto" else None)
    model = (model if a.device == "auto" else model.to(a.device)).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
        if a.merge:
            model = model.merge_and_unload().eval()
    model.requires_grad_(False)
    pt = Patcher(model, tok, rows=a.rows)
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    jobs = []
    if a.mode == "questions":
        qs = [q for q in json.loads(Path(a.inp).read_text())["questions"] if q["split"] == a.split]
        qs = qs[a.offset::a.stride][:a.limit or None]
        pre = lambda q: tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else []  # noqa: E731
        sy = lambda q: q.get("system", a.system)  # noqa: E731

        def ids(z):                                 # a token string (one token, special tokens by name) or an id
            if isinstance(z, int):
                return z
            t = tok.convert_tokens_to_ids(z) if z in tok.all_special_tokens else None
            t = t if t is not None else tok(z, add_special_tokens=False)["input_ids"]
            assert isinstance(t, int) or len(t) == 1, f"{z!r} is {len(t)} tokens"
            return t if isinstance(t, int) else t[0]
        sets = lambda q: {k: [ids(z) for z in v] for k, v in q["sets"].items()} if q.get("sets") else None  # noqa: E731
        jobs = [(q["id"], chat(tok, q["user"], sy(q)) + pre(q), chat(tok, q["corrupt_user"], sy(q)) + pre(q),   # an optional assistant prefill
                 {**{k: q[k] for k in ("user", "corrupt_user", "task", "key") if k in q}, "prefill": q.get("prefill"), "answers": None,
                  "sets": sets(q)})
                for q in qs]
    else:
        for name in a.cases.split(","):
            pairs = chat_pairs(tok, name, a.word, a.system) if name in CHAT_CASES else case_pairs(tok, name)
            tag = f"{name}_{a.word}" if a.word else name
            for j, (c, k, text, ans) in enumerate(pairs):
                jobs.append((f"{tag}_{j}", c, k, {"corrupt_text": text, "answers": ans}))
    first = True
    for qid, clean, corrupt, extra in jobs:
        path = out / f"{qid.replace('/', '__')}.json"
        if path.exists():
            continue
        if len(clean) != len(corrupt):
            log(f"{qid}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
            continue
        if first:                                     # (a throwaway run: CUDA's first kernels)
            pt.setup(clean, corrupt)
            first = False
        log(f"{qid}")
        meas = [(int(u.split(":")[0]), int(u.split(":")[1].rstrip("+-"))) for u in a.measure.split(",")] if a.measure else None
        rec = find(pt, clean, corrupt, extra.pop("answers"), log=log, cuts_wanted=tuple(a.cuts.split(",")), measure=meas,
                   sets=extra.pop("sets", None))
        rec.update(id=qid, ids=clean, corrupted=corrupt, answer_text=tok.decode([rec["answer"]]),
                   corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, adapter=a.adapter, dtype=a.dtype,
                   merged=bool(a.adapter and a.merge), tokenizer=a.tokenizer or a.model, system=a.system,
                   finder="circuit_finder", finder_rev=a.rev, **extra)
        if a.edges and not rec.get("skipped") and (rec.get("nodes") or rec.get("routes")):
            t1 = time.time()                          # the circuit's edges (neuron_edges.finder_graph: between its neurons
            import neuron_edges                       # by path patching from the corrupted run, route knockouts, and its
            try:                                      # mover heads' input and readout edges); a failure is recorded, the
                g = neuron_edges.finder_graph(model, tok, rec, log=log)   # run goes on to the next question
                rec["graph"] = {k: g.get(k) for k in ("nodes", "edges", "head_edges", "movers_inputs_lost", "routes",
                                                      "record", "cut", "route_note", "seconds")}
                rec["readers"] = g.get("readers")         # (each unit's downstream readers and direct write: readers.py)
            except Exception as e:                    # noqa: BLE001
                rec["graph_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                log(f"  edges failed: {rec['graph_error'][:120]}")
                torch.cuda.empty_cache()
            rec["seconds_edges"] = round(time.time() - t1, 2)
            neuron_edges.free()
        path.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
