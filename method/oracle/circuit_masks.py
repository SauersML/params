"""Learned masks for circuit_finder's cuts (searchalgo, 10-16): a cut's units proposed by descending a continuous mask on
the patching pair itself, then the passing set found and pruned by measured patching (circuit_finder's rule: the mask
only orders; every keep or drop is a measured run).

A cut (circuit_finder's: one kind of unit, neurons or heads, or both, at some positions) gets a mask m_u = sigmoid(theta_u)
in [0, 1] over its units: a unit's value in a run is its live value moved the share m_u of the way to its value in the
other run (m = 1 is circuit_finder's patch, m = 0 leaves the unit alone). Two rows run under the same mask, the clean run
noised and the corrupted run denoised, and theta descends (Adam, STEPS steps from theta = INIT, every unit nearly patched:
the cut's ceiling) on
    relu(TAU - lost) + relu(TAU - restored) + LAMBDA * sum(m)
so a unit stays while the pull of the two hinges on it exceeds LAMBDA (a share of the gap per unit). Then the units are
ordered by their final theta, the smallest passing prefix of that order is found by measurement (circuit_finder.grid's
sizes, then bisection), and units are removed one at a time while the set still passes (pruning). Every cut of a
question whose ceiling passes is optimized in one batch, two rows a cut. find() is circuit_finder.find's search (records
in its format, "search": the arm).

Arms on one question set, same card, same records (searchalgo's comparison):
  mask   the masks above
  attr   the units in RelP attribution order (circuit_finder.Patcher.scores): the smallest passing prefix and pruning
         by measurement, as mask's
  am     attr, then masks on the cuts attr leaves unmet
  fam    the units of the family's earlier circuits first (--prior: other questions of the task, family_order), then
         attr's order
A cut's hinge targets are TAU or 0.05 under its ceiling (at least its bar + 0.05), per share; its sets are taken at its bar
(PASS plus, under FLOOR, the probed noise of reduced precision).
Multi-source masks (methods, 10-16; constants an arm sets, mask:30:partners=8:allcuts=1:pairwise=1:warm=0.5). Since
10-17 ALLCUTS, PAIRWISE and AGG "min" are the defaults and the command line takes 8 pairs from --pool (trainset_8b): on
Qwen3-8B (devset_8b stride 4, pod circ-methods-m4) unseen pairs pass 36/45 against 13/45 for the question's own pair alone,
own pairs 49/50, 31 units (the lead's 10-17 decision: the default for generalizing circuits):
  ALLCUTS   every cut, not only the source, is chosen and verified on the shares over the question's pair and its
            partners (Mean): interchange interventions from several keys, so a cut that only separates the two given
            keys does not pass
  PAIRWISE  the hinges on each pair's shares, averaged over the pairs (a set must pass every pair), in place of the
            hinges on the pairs' mean shares (where overshoot on easy pairs pays for failure on others)
  WARM      masks start from the own pair's RelP attribution (Patcher.scores): theta = INIT + WARM * ln(score / the cut's
            largest score), at least WARM_MIN, so attributed units start patched and the rest nearly unpatched; Adam's
            normalized steps let any unit grow back (the attribution only starts the search)
  ADV       the rest of the cut's units anywhere between the two runs during the descent (optimize_many's adv: VPD's
            ablation of unused parts in any combination, its adversary shared over the question's pairs and persistent
            across steps), so a set that passes only while the rest sits at its base values is not taken
  ADV_CHECK each verified set's adversarial shares: ADV_CHECK steps of the same adversary against the set alone (its units
            fully patched, the rest's rho moved to lower the shares), the worst step's measured shares in the record
            ADV_MODES (unit, block, random joined by "+") picks the check's adversaries: a rho per unit, per (layer,
            position) block, or uniform draws; the record's "adversarial" is the first one's, its "modes" all of them
  AGG       "min": a multi-partner cut's ceiling, prefix and pruning read the least share over its pairs, not the mean,
            so a set is taken only when every training pair (the question's own among them) passes
  SOURCE_NECESSITY 0: the source cut's hinge and bar read its restored share only. When the key is carried by more
            units than the set (redundancy), noising the set alone cannot remove it, so the source claim is sufficiency
  NECESSARY after the search, each cut's first NECESSARY_TOP units in RelP attribution outside it are noised one at a
            time on the own pair; those losing at least NECESSARY of the gap go in the record's "necessary" (per cut, with
            their shares): units that must stay on but restore nothing alone, which a minimal both-ways set leaves out
--pool QUESTIONS.json: partner keys drawn from that set's keys of the question's task (a seeded order per question),
not from the question's own "partners", which then stay held out for the scorecard's unseen pairs.
One prompt (ONE_PROMPT, mask:one_prompt=1:partners=16:agg=mean): no comparison prompt; the clean prompt is run with each
cut's units outside the circuit set to their values on other keys of the template (the question's pairs, Resampled), and
the circuit is the fewest units that keep the answer's lead over a rival token (CONTRAST) at PASS of the full model's;
BACKGROUND "mean" resamples from the keys' per-position mean instead.

  circuit_masks.py questions QUESTIONS.json OUT --arm mask [--split dev --stride 4] [--model Qwen/Qwen3-8B] [--dtype ...]
writes OUT/<id>.json per question.
  circuit_masks.py resampled RECORDS OUT --devset QUESTIONS.json [--contrast resampled|runner]
tests any finder's records by one prompt's criterion on each question's held-out partner keys (resampled_check).
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import time
import zlib
from pathlib import Path

from pathlib import Path

import numpy as np
import torch

import circuit_finder as cf

STEPS = 15          # (Qwen3-8B bf16, A40, dev stride 4, 103 questions, pods circ-searchalgo-sp/sp2: 15 steps from INIT 1 meet
# 103/103 at a median 14 units like 30 from INIT 3, in 1.8 s against 3.0 s a question; 30 steps smaller on 39, larger on 22)
LR = 0.25
TAU = 0.9           # the hinge's target share (PASS 0.7 is what the measured set must reach)
LAMBDA = 0.003      # a unit's cost in shares of the gap
ORDER_KEPT = 1024   # a cut's ordered units kept in the record ("rankings": the scorecard's faithfulness against size)
INIT = 1.0          # sigmoid(1) = 0.73: every unit mostly patched at the start, the hinges acting within a few steps (from 3 the
# first ~12 steps decay every unit alike and order nothing: 15 steps from 3 met 0/10)
INIT_SCAN = 0       # 1: a fresh spec starts at the lowest logit of INIT_GRID (down from INIT) where a uniform mask still meets
HARD = 0            # 1: rounds after a cut's order failed run the masks binarized (theta > 0 in or out) forward and sigmoid
# backward (straight-through), so the hinges see the set verify takes (Qwen3-8B E1 lie2: a source cut's 64-unit prefix lost
# 0.69 while restoring 0.73 -> 0.2 as units were added, the continuous mask holding both above 0.9). Off: from undivided
# thetas (every unit still above 0) the binarized forward is the whole ceiling, and Qwen3-0.6B CPU mask:2 source cuts that
# met in later soft rounds stayed unmet
INIT_GRID = (0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0, -10.0)   # its targets (optimize_many; arm constant init_scan=1)
ORDER_MIN = 64      # units ordered for the measured prefix: those the mask kept (m > 0.5), at least twice that and ORDER_MIN
ORDER_MAX = 20000
MASK_TOKENS = 1536   # rows x positions a mask step runs at most off CUDA (find_batch's parts; on CUDA _budget's)
MASK_MEMORY = 0.5    # the share of free device memory a mask step's activations may take
MASK_SCALE = 1.0     # the budget's factor, halved for the rest of the run each time a mask batch runs out of device memory
# (_fit_masks: the per-token estimate leaves out the masks' edited copies and the partners' cached runs; Qwen3-8B, 8
# partners, 2 questions a batch on a 24 GB RTX 3090 ran out at 1.0)
MASK_MERGE = 192    # a mask step's forward costs as many tokens' work (Qwen3-8B on an A6000: ~0.1 s of host launches a
# forward and backward, ~0.5 ms of compute a token): instances of near first positions share one (_passes)
FLOOR = True        # a cut's sets are taken at its bar, in reduced precision PASS plus the largest |share| of a no-op
# patch of the first layer's units at each of its positions (_before_steps)
GRAPH = False       # True: a round's mask steps after the first GRAPH_WARM replayed from one CUDA graph (the same kernels
GRAPH_WARM = 2      # each step), the warm steps' cache and the graph's pool freed around it. Qwen3-8B on an A6000
# (6aa938e9): am without partners 1.005 -> 0.961 s a question (64 dev questions, batches of 16), one question at a time mask
# arm 4.78 -> 3.68 s median. Off since 10-17: replays hit "an illegal memory access" in jobs without partners too (searchalgo:
# mask:30:init=3 on an A5000 at 058d8d45; the default arm's partner-less questions on an A40 at 28c5d6a, GRAPH_SIDE 1),
# which ends the whole run, against a 4-23% faster step
FAMILY_MIN_KEYS = 4  # a template's own mover union when it has this many train records, else its task's (family_heads)
FAMILY_FREQ = 0.1    # heads of at least this frequency among the union's records (0: every head). Qwen3-8B, generality's
# dev half (100 questions, circuit_bench at d9f4e1d, A40): unseen pairs own circuit 44/89, own + as many random same-layer
# heads as the union adds 42/89, full union 86/89 at 33 units, union at 0.1 84/89 at 24 units (paired McNemar: the union
# at 0.1 against the random heads 42 to 0, p 5e-13; against the full union 0 to 2, p 0.5)
UNIONS = Path.home() / "mpd-data" / "circuits" / "unions_train_8b.json"   # (the default unions: circuit_union's over the
UNIONS_MODEL = "Qwen/Qwen3-8B"                                            # train split's records of this model)
CLEAN_SIDE = 0       # 1: units active only in the corrupted run (circuit_finder.ACTIVE_RATIO) never in a mask: on unseen keys
# they carry nothing (casestudy 10-16: 0.00/0.00 in every cut) while giving most of the own pair's necessity
EXTEND = 0          # > 0: a size control (paper's C25): a met source cut grown with its own mask order's next units to EXTEND
EXTEND_CUT = "source"   # (the cut EXTEND grows)
GRAPH_PARTNERS = 0  # 1: jobs with partner pairs replayed from graphs too; 0 (searchalgo's 0d4ff56): eager, their replays
# hit "an illegal memory access" in methods' pods (g5 at 22a34b6, m5 at 2b21980: mask:10, partners 8, allcuts, agg min)
GRAPH_SIDE = 1      # the warm steps on a side stream (torch's capture recipe); 0: on the current stream
ROUNDS = 4          # rounds of STEPS a cut's masks get while its order cannot pass (each round from the last's theta)
PRUNE_MAX = 4 * cf.SINGLES_KEPT   # a passing prefix of more units than this is a diffuse cut, not pruned: its size, shares
# and curve recorded, no units (circuit_finder's diffuse rule). Qwen3-8B dev set (A6000, am, 153 questions): 13 readout
# prefixes of 1,060-19,980 units took 13-1,930 s each to prune one unit at a time (the set's leave-one-out sets and a
# round per unit that stays), to 120-11,812 units (2 of them to SINGLES_KEPT or fewer); median question 1.7 s, mean 24 s
GRID_FIRST = 16     # a verify's grid sizes up to this first, then up to four times that while none passes (16, 64,
# 256, ...), for an order over several positions (a source cut's grid of 20,000 units is ~96 sizes, the large ones its
# costliest rows: from the first changed token, from layer 0); a one-position order (readout, movers) every size at
# once, its rows one position each. Qwen3-8B dev set, am, 153 questions: source cuts passing at 16 or fewer 123, at 17-64
# 37, above 64 2 of 162
MIDS = 1024         # sizes between the grid's last failing and first passing prefix measured in one batch (bisection
# above: a size a tick; 33 before, 3-5 ticks of a readout's verify on Qwen3-8B); a readout's prefix is one position a row
PRUNE_ALONE = 1     # 1: pruning's candidates are the units whose removal alone from the whole set passes (circuit_finder's
# rule); 0: every unit is a candidate and its own trial decides (a set taken differs only where removals interact). The
# leave-one-out sets were 37% of an am question's measured token-forwards (Qwen3-0.6B trace, capital/Japan/p1)
STAGED_PAIRS = 1    # pruning's trials of a multi-partner cut under AGG "min", in two stages: every trial on the pair the
# whole set does worst on, then on the other pairs only the trials passing there (a set passes only when every pair does,
# and pruning reads pass or fail alone: the same removals as every pair at once, the failing trials' other pairs never run).
# Qwen3-8B, 8 partners, A40: pruning's trials were 57% of a question's measuring (both_without 1.9 s, prefixes 9.5 s)
SETTLE = 0.03       # a search decision whose share lies within this of its bar (a verify prefix, a pruning trial, the
# pruned set) is measured again set by set, one set a forward on each pair (the same shapes whatever the batch around it):
# in bfloat16 a set's share moves ~0.01 with the batch's shape, and pruning takes every set to its bar, so trials there
# flipped with the batch size or the pairs' staging (Qwen3-8B, 8 partners: 10 of 20 circuits changed with STAGED_PAIRS);
# 0: off
PRUNE_WINDOW = 16   # a pruning round's candidates at first (prune_steps; four times more after a round that settles all)
POOL = Path.home() / "mpd-data" / "circuits" / "trainset_8b.json"   # (the command line's default --pool)
ALLCUTS = 1         # partners for every cut (module docstring); 0: the source cut only
PAIRWISE = 1        # hinges per pair; 0: on the pairs' mean shares
WARM = 0.0          # > 0: attribution start, theta = INIT + WARM * ln(score / max score)
WARM_MIN = -2.0     # sigmoid(-2) = 0.12: the start of units the attribution gives nothing
ADV = 0.0           # > 0: the adversary's step on rho during the descent (module docstring)
ADV_CHECK = 0       # > 0: steps of the measured adversarial check of each verified set
ADV_STEP = 0.25     # the check's step on rho
ADV_MODES = "unit"  # the check's adversaries joined by "+": unit, block, random (optimize_many's mode; adv_modes=block+random)
AGG = "min"         # a Mean's shares: "min", the least over its pairs (a set passes only when every pair does), or "mean"
SOURCE_NECESSITY = 1  # 0: the source cut is chosen and taken on its restored share only (no lost hinge, no lost bar):
                      # a small set of a redundantly carried key is sufficient, not necessary (design 11a)
NECESSARY = 0.2     # a unit outside a cut whose noising alone loses more of the own pair's gap than its line (_line: the
# NULL_Q percentile of NULL_UNITS random units of its kind, layer and position class, the answer's or the body's, noised the
# same way in the same run, and at least this): listed in the record's "necessary" beside the cut, with its shares and line
# (the untagged candidates losing at least half this in "necessary_near"). A minimal both-ways set leaves out units that are
# necessary and restore nothing when another unit carries the contrast (casestudy's case 17 AND plant, Qwen3-8B: two flag
# neurons, each lost about 1.0 alone and restored 0.01-0.05, in the circuit 2/38 and 3/38, the unit they gate 38/38)
NULL_UNITS = 4      # the controls' random units a (kind, layer, position class) of the candidates
NULL_Q = 99         # the controls' percentile
SETTLE_TAGS = 0.05  # a tag's, mediator's or whole-neuron tag's share within this of its line measured again on its own
NECESSARY_TOP = 256  # the candidates measured a cut: its first units in RelP attribution (attr_order, the own pair) outside the
# cut, each noised alone on the own pair in one tick; 0: off. (AND plant: RelP ranks of the flags among every neuron at every
# position, median 26-33, within 256 on 37/38 pairs, within 64 on 33-34)
NECESSARY_ORDERS = "relp+eapig"  # the candidates' orders, taken in turn until NECESSARY_TOP units: RelP attribution and |EAP-IG|
# (eapig_scores' magnitude on the own pair: signed EAP-IG ranks the AND plant's flags last, their first-order effect of the
# opposite sign; on 38 pairs RelP's 256 left out flag B once, |EAP-IG|'s 256 three times, never the same pair); "relp": RelP's
NECESSARY_MEANS = Path.home() / "mpd-data" / "ami" / "means" / "qwen3_8b.pt"  # general-text means (ami.collect's file, the
NECESSARY_MEANS_MODEL = "Qwen/Qwen3-8B"  # R3 background) of the mean-ablation tags, for this model; missing, another model: off.
# A unit the same in both prompts (silent in both: casestudy's AND-plant suppressors) is invisible to patching from the other
# prompt; set to its general-text mean it is not, where that mean differs from its value here. Candidates: each cut's first
# NECESSARY_TOP units in |(value - mean) x the contrast's gradient| on the clean run, outside the cut; each set to its mean
# alone in the clean run; lost >= NECESSARY, and on CONTROL_PROMPTS (the neuron or head channel at its mean at every
# position) a median lost share of the top token's margin over the runner-up of at most NECESSARY_CONTROL (else a hub: Qwen3's
# massive-activation neurons cost every answer), go in the record's "necessary_mean"; the hubs in "hubs"
NECESSARY_CONTROL = 0.1
MEDIATORS = 64      # mediators of necessary units: for each unit that alone loses at least MEDIATOR_ROOT on the own pair (a cut's
# own units, measured alone there, and the patch tags), this many candidates in |(value with the unit noised - clean value) x
# the contrast's gradient in that run|, each held at its clean value while the unit is noised; one restoring more of the
# unit's loss than its line (the NULL_Q percentile of one random unit of each candidate's kind, layer and position class held
# the same way, and at least MEDIATE) goes in the record's "mediators" with the unit it mediates (below, "mediators_near"). A unit silent in both prompts that fires
# when a necessary unit is removed and cancels the answer (casestudy's AND-plant suppressors: R held at its clean value
# under flag A's removal took the loss from 0.93 to -0.11) is invisible to both knockout tags. 0: off
MEDIATE = 0.2
MEDIATOR_ROOT = 0.5
MEDIATOR_NULL = 1   # random units held the same way per candidate (of its control class): a mediator's control
NECESSARY_WHOLE = 64  # whole-neuron tags: neurons noised at every position from the first changed one at once, candidates in
# RelP and |EAP-IG| attribution summed over those positions taken in turn, outside the circuit's neurons; one losing more
# than its line (NULL_UNITS random neurons of its layer) and passing the hub control (NECESSARY_MEANS) goes in the record's
# "necessary_whole" (failing the control, "hubs_whole"). A trained neuron acting at many positions loses half the gap at no
# single one (casestudy's case 19 one-neuron owls: no tag fired, our circuit held it on 0.77 of questions); 0: off
CONTROL_PROMPTS = ["What is the capital of France? Answer with one word.", "How many legs does a spider have? One word.",
                   "What color is the sky on a clear day? One word.", "Which planet is known as the Red Planet? One word.",
                   "What is the opposite of hot? One word.", "What do bees make? Answer with one word.",
                   "What is the largest ocean on Earth? One word.", "In which season does snow usually fall? One word."]
HYB_CUTS = "source"  # arm "hyb" (the lead's 10-17 per-part hybrid, after oracleeval's equal-input run): these cuts (comma-
# separated) ordered by signed EAP-IG over the cut's pairs (eapig_scores), then the measured prefix and prune; the other
# cuts by the masks. Arm "rank": every cut so, no masks. (Equal input, 45 unseen pairs: signed EAP-IG 41/45 against the
# masks' 36/45; casestudy's case 18: the masks' mover heads beat signed EAP-IG's at equal input, +0.11 to +0.16; case 19:
# signed EAP-IG puts a planted source neuron in the set 0.98 against the masks' 0.77)
EAPIG_STEPS = 5      # EAP-IG's points on the input line from the corrupted prompt's embeddings to the clean one's
RANK_SUM = 0         # 1: a ranked cut's units by their neuron's (head's) score summed over the cut's positions, then their own:
# a unit acting at many positions comes in whole (casestudy's one-neuron owls)
RERANK = 0           # > 0: the ranking's first RERANK units re-ordered by their measured effect alone (lost + restored, the
# mean over the training pairs; Hanna et al. 2024: gradients rank the top right and the tail wrong)
VAL_PAIRS = 0        # > 0: the last VAL_PAIRS partners held out of the ranking and the masks (validation pairs): a cut's prefix
# and prune still pass every pair, so its size is chosen on pairs its order never saw
MARGIN = 0.0         # > 0: every cut's bar raised to at least this share (a set taken above PASS on its own pairs, which may
# pass new ones)
ONE_PROMPT = 0      # 1: one prompt (the lead's 10-17, from the user: a circuit should not need a second prompt; arm
# constant one_prompt=1 with partners=16): the question's clean prompt alone is run, each cut's units outside the circuit
# set to their values on other keys of its template (its pairs' corrupted prompts: the question's own and its partners,
# Resampled), the circuit's units to their values on the prompt itself (both frozen: a unit recomputed from a residual the
# resampled units changed would carry their keys; on Qwen3-8B no source set passed so); the circuit is the fewest units for
# which the answer's lead over CONTRAST's rival stays at least PASS of the full model's (AGG over the resampled keys). Its
# cut's "restored" is that share of the
# lead, its "lost" the share lost with the circuit's units alone resampled (reported, no bar); a cut is reachable when
# resampling all of it loses at least PASS of the lead
CONTRAST = "resampled"  # one prompt's rival token: "resampled", the top token other than the answer on each resampled
# prompt (a BACKGROUND "mean" Patcher: the commonest of those); "runner", the full model's runner-up on the clean prompt
KEPT_ROWS = 256      # one prompt: rows (sets x keys) a forward of _kept_by_masks
ONE_PROMPT_PRUNE = 0  # 1: one prompt's passing prefix pruned unit by unit; 0: taken as found. On Qwen3-8B (5 dev questions,
# 16 keys) pruning moved sizes 38 -> 38, 100 -> 99, 226 -> 226, 417 -> 426 and 169 -> 164 at most of a question's ~150 s
RESAMPLE_CHECK = 16  # resampled mode: held-out keys a record is tested on (each question's partners of its prompt's length)
BACKGROUND = "keys"  # one prompt: "keys", each resampled key's own values (the shares' AGG over the keys); "mean", their
# per-position mean (one background, _mean_patcher)


def _blocks(units: list) -> list:
    """Single units as circuit_finder sets: one unit per (kind, layer, position), its indices a tensor."""
    by = {}
    for u in units:
        kind, i = u[0], u[1]
        p, k = (u[3], u[2]) if kind == "head" else (u[2], u[3])
        by.setdefault((kind, i, p), []).append(int(k))
    return [cf.unit(kind, i, p, torch.tensor(sorted(ks))) for (kind, i, p), ks in by.items()]


def _kinds_of(spec: dict) -> tuple:
    return tuple(spec["kinds"])


def optimize(pt, specs: list, steps: int = STEPS, log=None, taus: list | None = None) -> list:
    """Masks for specs [{"kinds": ("neuron",) | ("head",) | ("neuron", "head"), "positions": [...]}] of the pair set up in
    pt, optimized together (optimize_many's, one job)."""
    return optimize_many([(pt, specs, taus)], steps=steps, log=log)[0]


def optimize_many(jobs: list, steps: int = STEPS, log=None, init: list | None = None, adv: float | None = None, hard: bool = False,
                  freeze: bool = False, mode: str = "unit", budget: int | None = None) -> list:
    """Masks for several pairs at once: jobs [(pt, specs, taus)], Patchers of one model each set up on its own pair (prompts
    of one length), every spec of every job two rows of one batch per step (a step reads every weight twice whatever
    its rows: batched pairs cost about one). A job's pt may be a list of Patchers, one clean prompt against several
    corrupted ones (partners): its specs then hold one mask over every partner's rows, the hinges on the shares' means
    over the partners. taus: per spec the hinges' targets (lost, restored), default TAU. Per job and spec {kind: theta
    [L, P, width]} (final logits, on the host) and the history of (mean lost, mean restored, sum m) per step. A mask
    edits only its own (row, position) entries: those rows' values gathered, moved toward the other run's and put back.
    init: per job, per spec its theta to start from ({kind: [L, P, width]}, optimize_many's own output; None: INIT).
    adv (default ADV): with a step above 0, every unit's patch is m + (1 - m) rho of the way (the units the mask leaves out
    moved anywhere between their base value and the other run's), rho in [0, 1] one per unit and spec, shared by the
    spec's pairs, starting at 0 and after each step moved adv along the sign of its gradient that raises the hinges
    (persistent projected ascent); freeze: theta not stepped and rho moved to lower the spec's mean shares (the measured
    adversarial check of a set, adversarial()). Each spec's output then also has its final rho. mode: "unit" (a rho per
    unit), "block" (one rho per layer and position, every unit of the block moved together: far fewer directions, each
    block's rest a mixture of the two runs' states) or "random" (rho drawn uniform in [0, 1] per unit each step, no
    ascent: SPD's stochastic masks). budget: the rows x positions a step's backward may hold (_budget's); a step whose
    instances exceed it runs its forwards in chunks of at most budget, each chunk's hinges backed through before the next
    (the loss is a sum over pairs: PAIRWISE hinges, or freeze's shares; the size term once), the same gradient as one pass."""
    adv = ADV if adv is None else adv
    groups = [list(pt) if isinstance(pt, (list, tuple)) else [pt] for pt, _, _ in jobs]
    pt0 = groups[0][0]
    T = pt0.T
    assert all(pt.T == T for g in groups for pt in g), "optimize_many: prompts of one length"
    L, dev = pt0.L, pt0.dev
    width = {"neuron": pt0.D, "head": pt0.H}
    key = {"neuron": "neu", "head": "head"}
    specs = [(q, j, s) for q, (_, ss, _) in enumerate(jobs) for j, s in enumerate(ss)]      # spec sid: one mask
    sid = {(q, j): n for n, (q, j, _) in enumerate(specs)}
    inst = [(q, j, pt) for q, (_, ss, _) in enumerate(jobs) for pt in groups[q] for j in range(len(ss))]   # instance g:
    kinds = sorted({k for _, _, s in specs for k in s["kinds"]})            # rows 2g (clean, noised), 2g + 1 (denoised)
    cell = {k: {(q, j, p): n for n, (q, j, p) in enumerate((q, j, p) for q, j, s in specs if k in s["kinds"]
                                                            for p in s["positions"])} for k in kinds}   # theta rows
    owner = {k: torch.tensor([sid[(q, j)] for (q, j, _), _ in sorted(cell[k].items(), key=lambda t: t[1])], device=dev)
             for k in kinds}                                                 # theta row -> spec
    # instances run in forwards from their spec's first position, near starts merged (_passes: an answer-position cut's
    # rows run that position only, not the source cut's span: on Qwen3-8B they were ~2/3 of a step's tokens)
    first_of = [min(jobs[q][1][j]["positions"]) for q, j, _ in inst]
    passes_ = []
    split = budget is not None and (PAIRWISE or freeze) and sum(2 * (T - f) for f in first_of) > budget
    pieces = [(start, gs[a:a + max(1, budget // (2 * (T - start)))]) for start, gs in _passes(first_of, T)
              for a in range(0, len(gs), max(1, budget // (2 * (T - start))))] if split else _passes(first_of, T)
    for start, gs in pieces:
        loc = {g: n for n, g in enumerate(gs)}
        ent = {k: [(loc[g], p, cell[k][(inst[g][0], inst[g][1], p)]) for g in gs if k in jobs[inst[g][0]][1][inst[g][1]]["kinds"]
                   for p in jobs[inst[g][0]][1][inst[g][1]]["positions"]] for k in kinds}
        with torch.no_grad():                       # each entry's two rows' other-run values per layer [2N, width]
            oth = {k: [torch.stack([inst[gs[g]][2].act[src][key[k]][i][p] for g, p, _ in ent[k] for src in ("corrupt", "clean")])
                       for i in range(L)] for k in kinds if ent[k]}
            x = torch.stack([inst[g][2]._ids_dev[b] for g in gs for b in ("clean", "corrupt")])
            kv = [tuple(torch.cat([inst[g][2]._base_kv[b][i][jj][:, :, :start] for g in gs for b in ("clean", "corrupt")])
                        for jj in (0, 1)) for i in range(L)] if start > 0 else None
        spans, at = [], 0                           # each Patcher's rows (consecutive)
        for (_, _, pt), grp in itertools.groupby([inst[g] for g in gs], key=lambda t: (t[0], id(t[2]), t[2])):
            n = len(list(grp))
            spans.append((pt, at, at + 2 * n))
            at += 2 * n
        passes_.append({"start": start, "gs": gs, "x": x, "kv": kv, "spans": spans, "others": oth,
                        "rows": {k: torch.tensor([2 * g + b for g, _, _ in ent[k] for b in (0, 1)], device=dev) for k in oth},
                        "pos": {k: torch.tensor([p - start for _, p, _ in ent[k] for _ in (0, 1)], device=dev) for k in oth},
                        "tix": {k: torch.tensor([n for _, _, n in ent[k] for _ in (0, 1)], device=dev) for k in oth}})
    order = torch.tensor([g for ps in passes_ for g in ps["gs"]], device=dev)   # the forwards' instances, in turn
    tau = torch.tensor([t for _, ss, taus in jobs for t in (taus or [(TAU, TAU)] * len(ss))], device=dev)
    cc = torch.tensor([pt.c["clean"] for _, _, pt in inst], device=dev)[order]
    cr = torch.tensor([pt.c["corrupt"] for _, _, pt in inst], device=dev)[order]
    gap = torch.tensor([pt.gap for _, _, pt in inst], device=dev)[order]
    of = torch.tensor([sid[(q, j)] for q, j, _ in inst], device=dev)[order]   # instance -> spec
    count = torch.zeros(len(specs), device=dev).index_add(0, of, torch.ones(len(inst), device=dev))
    chunks, cur, n_tok, at = [], [], 0, 0           # (split: consecutive forwards of at most budget rows x positions, each
    for n, ps in enumerate(passes_ if split else []):   # chunk's instances [a, b) in order)
        t = 2 * len(ps["gs"]) * (T - ps["start"])
        if cur and n_tok + t > budget:
            chunks.append((cur, at - sum(len(passes_[m]["gs"]) for m in cur), at))
            cur, n_tok = [], 0
        cur.append(n)
        n_tok += t
        at += len(ps["gs"])
    if cur:
        chunks.append((cur, at - sum(len(passes_[m]["gs"]) for m in cur), at))

    def contrast(ms, rho, which=None):              # each instance's two rows' contrasts under masks ms (and rho)
        def edit(ps, k, i, h):
            d = h.device
            r, p = ps["rows"][k].to(d), ps["pos"][k].to(d)
            m = ms[k][i].to(d, h.dtype)[ps["tix"][k].to(d)]                    # [2N, width]
            if ONE_PROMPT:                          # (one prompt: m keeps; the clean row's rest takes the other run's)
                m = torch.where((r % 2 == 0)[:, None], 1 - m, m)
            if rho:                                 # (the rest anywhere between the two runs: the adversary's rho)
                m = m + (1 - m) * rho[k][i].to(d, h.dtype)[ps["tix"][k].to(d)]
            sel, o = h[r, p], ps["others"][k][i].to(d, h.dtype)
            if ONE_PROMPT:                          # (one prompt: the clean row's cut at its own run's values, frozen, moved
                own = o.view(-1, 2, o.shape[-1])[:, 1:].expand(-1, 2, -1).reshape(o.shape)   # 1 - m of the way to the key's)
                sel = torch.where((r % 2 == 0)[:, None], own, sel)
            if k == "head":
                sel, o, m = sel.view(-1, pt0.H, pt0.hd), o.view(-1, pt0.H, pt0.hd), m[..., None]
            return h.index_put((r, p), (sel + m * (o - sel)).reshape(len(r), -1))
        cs = []
        for ps in passes_ if which is None else [passes_[m] for m in which]:
            ks = ps["others"]
            hn = pt0._last(ps["x"], ps["x"].shape[0], cache=(ps["start"], ps["kv"]) if ps["kv"] is not None else None,
                           neu=(lambda i, h, ps=ps: edit(ps, "neuron", i, h)) if "neuron" in ks else None,
                           head=(lambda i, h, ps=ps: edit(ps, "head", i, h)) if "head" in ks else None,
                           only={"res": set(), "head": set(range(L)) if "head" in ks else set(),
                                 "neu": set(range(L)) if "neuron" in ks else set()})
            cs += [pt._contrast(hn[a:b]) for pt, a, b in ps["spans"]]
        return torch.cat(cs)
    start_at = torch.full((len(specs),), INIT, device=dev)
    fresh = torch.tensor([not (init and (init[q] or [None] * len(jobs[q][1]))[j]) for q, j, _ in specs], device=dev)
    if INIT_SCAN and not freeze and bool(fresh.any()):   # each fresh spec from the lowest uniform logit (INIT_GRID, down
        alive = fresh.clone()                       # from INIT) at which both its mean shares still reach its targets: its
        with torch.no_grad():                       # hinges act from the first steps (a spread effect kept both shares
            for v in [x for x in INIT_GRID if x < INIT]:    # above 0.9 with sum m 162k of 8.4M units after 30 steps from 3:
                c = contrast({k: torch.full((L, len(cell[k]), width[k]), 1 / (1 + math.exp(-v)), device=dev)  # Qwen3-8B,
                              for k in kinds}, {})  # the E1 lie2 organism's source cut)
                lo = torch.zeros(len(specs), device=dev).index_add(0, of, (cc - c[0::2]) / gap) / count
                re_ = torch.zeros(len(specs), device=dev).index_add(0, of, (c[1::2] - cr) / gap) / count
                alive = alive & (lo >= tau[:, 0]) & (re_ >= tau[:, 1])
                start_at = torch.where(alive, torch.full_like(start_at, v), start_at)
                if not bool(alive.any()):
                    break
    theta = {k: start_at[owner[k]].cpu()[None, :, None].expand(L, len(cell[k]), width[k]).clone() for k in kinds}
    for q, j, s_ in specs if init else []:          # (a round after a cut its order could not pass: on from its theta)
        th = (init[q] or [None] * len(jobs[q][1]))[j]
        for k in s_["kinds"] if th else []:
            theta[k][:, [cell[k][(q, j, p)] for p in s_["positions"]]] = th[k]
    theta = {k: t.to(dev).requires_grad_(not freeze) for k, t in theta.items()}
    allow = {}
    if CLEAN_SIDE:                                  # (units active only in the corrupted run never in a mask:
        for k in kinds:                             # circuit_finder's ACTIVE_RATIO on the job's own pair)
            cells = sorted(cell[k].items(), key=lambda t: t[1])
            a = torch.stack([torch.stack([groups[q][0].act["clean"][key[k]][i][p] for (q, _, p), _ in cells]) for i in range(L)])
            b = torch.stack([torch.stack([groups[q][0].act["corrupt"][key[k]][i][p] for (q, _, p), _ in cells]) for i in range(L)])
            if k == "head":
                a, b = a.float().view(L, len(cells), pt0.H, pt0.hd).norm(dim=-1), b.float().view(L, len(cells), pt0.H, pt0.hd).norm(dim=-1)
            allow[k] = ~(b.float().abs() >= cf.ACTIVE_RATIO * a.float().abs())
            allow[k] = allow[k].to(dev)
    rho = {k: torch.zeros(t.shape[:2] + (1,) if mode == "block" else t.shape, device=dev).requires_grad_(mode != "random")
           for k, t in theta.items()} if adv > 0 or mode == "random" else {}
    graph = (GRAPH and dev.type == "cuda" and cf.LEAN and mode != "random" and steps > GRAPH_WARM + 1 and not split
             and (GRAPH_PARTNERS or all(len(g) == 1 for g in groups)))   # (GRAPH_PARTNERS's note)
    opt = torch.optim.Adam(list(theta.values()), lr=LR, capturable=graph) if not freeze else None

    def step_split():                               # (step's work a chunk at a time: each chunk's pairs' terms backed through
        with torch.enable_grad():                   # before the next forward, the size term once)
            if opt is not None:
                opt.zero_grad(set_to_none=False)
            for t in rho.values():
                if t.grad is not None:
                    t.grad.zero_()
            lost, rest = torch.zeros(len(specs), device=dev), torch.zeros(len(specs), device=dev)
            for which, a, b in chunks:
                ms = {k: torch.sigmoid(t) * allow[k] if k in allow else torch.sigmoid(t) for k, t in theta.items()}
                c = contrast({k: (theta[k] > 0).to(m.dtype) + m - m.detach() for k, m in ms.items()} if hard else ms, rho, which)
                li, ri = (cc[a:b] - c[0::2]) / gap[a:b], (c[1::2] - cr[a:b]) / gap[a:b]
                if ONE_PROMPT:
                    ri = 1 - li
                o_, w_ = of[a:b], 1 / count[of[a:b]]
                lost = lost.index_add(0, o_, li.detach() * w_)
                rest = rest.index_add(0, o_, ri.detach() * w_)
                part = (-(li + ri) * w_).sum() if freeze else ((torch.relu(tau[o_, 0] - li) + torch.relu(tau[o_, 1] - ri)) * w_).sum()
                if part.requires_grad:
                    part.backward()
            ms = {k: torch.sigmoid(t) * allow[k] if k in allow else torch.sigmoid(t) for k, t in theta.items()}
            size = sum(torch.zeros(len(specs), device=dev).index_add(0, owner[k], m.sum((0, 2))) for k, m in ms.items())
            if not freeze and size.requires_grad:
                (LAMBDA * size.sum()).backward()
        if opt is not None:
            opt.step()
        with torch.no_grad():
            for t in rho.values() if mode != "random" else ():
                if t.grad is not None:
                    t.add_(adv * t.grad.sign()).clamp_(0, 1)
        return torch.stack([lost, rest, size.detach()], 1)

    def step():                                     # (one step's work, its shares and size on the device)
        if split:
            return step_split()
        with torch.enable_grad():
            ms = {k: torch.sigmoid(t) * allow[k] if k in allow else torch.sigmoid(t) for k, t in theta.items()}
            c = contrast({k: (theta[k] > 0).to(m.dtype) + m - m.detach() for k, m in ms.items()} if hard else ms, rho)
            li, ri = (cc - c[0::2]) / gap, (c[1::2] - cr) / gap               # each pair's shares
            if ONE_PROMPT:                          # (one prompt: the share of the lead the clean row keeps)
                ri = 1 - li
            lost = torch.zeros(len(specs), device=dev).index_add(0, of, li) / count
            rest = torch.zeros(len(specs), device=dev).index_add(0, of, ri) / count
            size = sum(torch.zeros(len(specs), device=dev).index_add(0, owner[k], m.sum((0, 2))) for k, m in ms.items())
            if PAIRWISE:                            # (each pair's hinges, averaged over the spec's pairs)
                hinge = torch.zeros(len(specs), device=dev).index_add(
                    0, of, torch.relu(tau[of, 0] - li) + torch.relu(tau[of, 1] - ri)) / count
            else:
                hinge = torch.relu(tau[:, 0] - lost) + torch.relu(tau[:, 1] - rest)
            loss = -(lost + rest).sum() if freeze else hinge.sum() + LAMBDA * size.sum()
            if opt is not None:
                opt.zero_grad(set_to_none=False)    # (gradients kept in place: a graph's replays write the same ones)
            for t in rho.values():
                if t.grad is not None:
                    t.grad.zero_()
            if loss.requires_grad:
                loss.backward()
        if opt is not None:
            opt.step()
        with torch.no_grad():                       # (rho's ascent on the loss, within [0, 1])
            for t in rho.values() if mode != "random" else ():
                if t.grad is not None:
                    t.add_(adv * t.grad.sign()).clamp_(0, 1)
        return torch.stack([lost.detach(), rest.detach(), size.detach()], 1)                 # [specs, 3]
    hist_t = torch.zeros(steps, len(specs), 3, device=dev)
    n_ = 0
    if graph:                                       # (the same step every time: GRAPH_WARM eager steps on a side stream,
        cur = torch.cuda.current_stream(dev)        # then one captured and replayed; a step's host work is ~40 kernel
        side = torch.cuda.Stream(dev) if GRAPH_SIDE else cur    # launches a layer, several times its work on the card for
        side.wait_stream(cur)                       # a few cuts)
        with torch.cuda.stream(side):
            for n_ in range(GRAPH_WARM):
                hist_t[n_].copy_(step())
        cur.wait_stream(side)
        n_ = GRAPH_WARM
        torch.cuda.empty_cache()                    # (the eager steps' cached blocks back to the card: the graph's own
        try:                                        # pool takes a step's activations again, and a step was sized to
            g = torch.cuda.CUDAGraph()              # MASK_MEMORY of the free memory; Qwen3-8B with partners ran out)
            with torch.cuda.graph(g):
                out_g = step()
        except Exception as e:                      # noqa: BLE001 (the rest eager)
            graph = False
            if log:
                log(f"    mask steps eager: graph capture failed ({type(e).__name__}: {str(e)[:120]})")
        for n_ in range(n_, steps) if graph else ():
            g.replay()
            hist_t[n_].copy_(out_g)
        n_ = steps if graph else n_
        g = out_g = None                            # (the graph's pool back to the card for the measuring forwards)
        torch.cuda.empty_cache()
    for n_ in range(n_, steps):
        if mode == "random":                        # (a fresh uniform draw of the rest each step)
            rho = {k: torch.rand_like(t) for k, t in rho.items()}
        hist_t[n_].copy_(step())
    hist = [[(round(a, 4), round(b, 4), round(z, 1)) for a, b, z in col] for col in hist_t.transpose(0, 1).tolist()]
    out, th = [[] for _ in jobs], {k: (t.detach().float() if k not in allow else t.detach().float().masked_fill(~allow[k], -30.0)).cpu()
                                   for k, t in theta.items()}
    rh = {k: t.detach().float().cpu() for k, t in rho.items()}
    for n, (q, j, s) in enumerate(specs):
        got = {k: th[k][:, [cell[k][(q, j, p)] for p in s["positions"]]] for k in s["kinds"]}
        out[q].append({"theta": got, "history": hist[n], "start": float(start_at[n]),
                       **({"rho": {k: rh[k][:, [cell[k][(q, j, p)] for p in s["positions"]]] for k in s["kinds"]}} if rh else {})})
        if log:
            a, b, z = hist[n][-1]
            log(f"    mask {'+'.join(s['kinds'])} at {s['positions'][0]}..{s['positions'][-1]}"
                f"{f' (job {q})' if len(jobs) > 1 else ''}{f', {len(groups[q])} partners' if len(groups[q]) > 1 else ''}: "
                f"{steps} steps, lost {a:.2f}, restored {b:.2f}, sum m {z:.1f}")
    return out


def _passes(first_of: list, T: int) -> list:
    """Instances (first positions first_of) in forwards [(start, instances)]: a forward runs its instances from its
    lowest first position (an instance's positions before its own first are its base run's, recomputed), the partition
    of the sorted first positions that costs least, MASK_MERGE tokens a forward plus each forward's rows x positions
    (an exact partition of a sorted line: the starts of one length's questions are few)."""
    starts = sorted(set(first_of))
    rows = [2 * sum(1 for f in first_of if f == s) for s in starts]
    best = [0.0] + [math.inf] * len(starts)          # best[i]: the first i starts in forwards
    cut = [0] * (len(starts) + 1)
    for i in range(1, len(starts) + 1):
        for j in range(i):                          # a forward of starts j..i-1, from starts[j]
            c = best[j] + MASK_MERGE + sum(rows[j:i]) * (T - starts[j])
            if c < best[i]:
                best[i], cut[i] = c, j
    out, i = [], len(starts)
    while i:
        j = cut[i]
        out.append((starts[j], [g for g, f in enumerate(first_of) if starts[j] <= f <= starts[i - 1]]))
        i = j
    return out[::-1]


def ordered(spec: dict, theta: dict, dev=None) -> tuple:
    """(units in descending theta, how many the mask kept): ("neuron", layer, pos, k) and ("head", layer, k, pos); dev:
    where the top units are taken (a lockstep batch's card: ~30 ms a cut on the host for a source cut's 6.6 million)."""
    flat, parts = [], []
    for k in _kinds_of(spec):
        t = theta[k]
        flat.append(t.reshape(-1))
        parts.append((k, t.shape))
    v = torch.cat(flat).to(dev or "cpu")
    kept = int((v > 0).sum())
    n = min(max(2 * kept, ORDER_MIN), ORDER_MAX, len(v))
    units = _units_at(v.topk(n).indices, parts, spec["positions"])
    return units, kept


def verify(meas, order: list, log=None, name: str = "", bar: float = cf.PASS) -> dict:
    """The measured steps on a proposed order (verify_steps, by its measurer alone)."""
    return drive(_pts(meas), [verify_steps(meas, order, log=log, name=name, bar=bar)])[0]


def verify_steps(meas, order: list, log=None, name: str = "", bar: float = cf.PASS):
    """verify as a search (drive's generator: it yields its measurements, (meas, method, *args)): the smallest passing
    prefix (grid, bisection to MIDS sizes between the last failing size and the first passing one, then those in one
    batch); a prefix of more than PRUNE_MAX units is a diffuse cut (its size, shares and curve, no units); else units
    removed one at a time while it passes (prune_steps); each unit's shares alone. A set is taken at bar (_bar's)."""
    passes = lambda r: _ok(r, bar)  # noqa: E731
    one = len({cf._pos(u) for u in order}) <= 1     # (one position a row: every size at once)
    porder = yield from _pair_order(meas, order, bar)
    full, g, got, top = cf.grid(len(order)), [], [], 0
    while top < len(order) and not any(passes(r) for r in got):    # (prefixes of order: Patcher.both_prefixes, no set
        top = len(order) if one else max(GRID_FIRST, 4 * top)       # built per unit; the small sizes first, stages of
        stage = [k for k in full if (g[-1] if g else 0) < k <= top]  # four times the last)
        g, got = g + stage, got + ((yield from _trials(meas, porder, bar, "both_prefixes", order, stage, None)) if stage else [])
    at = dict(zip(g, got))
    met = [k for k, r in zip(g, got) if passes(r)]
    ok = None
    if met:
        hi = met[0]
        lo = max([k for k in g if k < hi] + [0])
        while hi - lo > MIDS:                       # (bisection down to MIDS sizes, then all of them in one batch)
            mid = (lo + hi) // 2
            at[mid] = (yield from _trials(meas, porder, bar, "both_prefixes", order, [mid], None))[0]
            lo, hi = (lo, mid) if passes(at[mid]) else (mid, hi)
        mids = list(range(lo + 1, hi))
        got_m = (yield from _trials(meas, porder, bar, "both_prefixes", order, mids, None)) if mids else []
        at.update(zip(mids, got_m))
        ok = next((k for k, r in zip(mids, got_m) if passes(r)), hi)
    curve = [{"k": k, "lost": a, "restored": b, **({"at_most": True} if porder and not passes((a, b)) else {})}
             for k, (a, b) in zip(g, got)]          # (staged: a failing size's shares are its first pairs' least, an upper bound)
    if ok and ok > PRUNE_MAX:                       # (diffuse: circuit_finder's rule, no pruning)
        lost, rest = at[ok]
        if log:
            log(f"  {name}: {len(order)} ordered, passing prefix {ok}, diffuse ({lost:.2f}/{rest:.2f})")
        return {"units": [], "singles": {}, "met": False, "diffuse": True, "lost": lost, "restored": rest, "found": len(order),
                "curve": curve, "spread": {"diffuse": True, "units": ok}, "order": list(order[:ORDER_KEPT])}
    circ = list(order[:ok]) if ok and isinstance(meas, Resampled) and not ONE_PROMPT_PRUNE else (
        (yield from prune_steps(meas, list(order[:ok]), bar)) if ok else [])
    lost, rest, singles = 0.0, 0.0, {}
    if circ:                                        # (the set and its units alone, one tick)
        whole, alone = yield [(meas, "both", [_blocks(circ)]), (meas, "both", [[u] for u in circ])]
        (lost, rest), singles = _settled(meas, bar, [_blocks(circ)], whole)[0], dict(zip(circ, alone))
    for k in [k for k in met if k > ok] if circ and not passes((lost, rest)) else []:
        circ = list(order[:k])                      # (a set at the margin can fail measured again, a batch's shape moving
        lost, rest = _settled(meas, bar, [_blocks(circ)], (yield (meas, "both", [_blocks(circ)])))[0]   # a bfloat16 share by ~0.01: Qwen3-8B capital/Spain/p2's
        if passes((lost, rest)):                    # readout, 22 units at 0.7003 then 0.6995): the next passing grid size
            singles = dict(zip(circ, (yield (meas, "both", [[u] for u in circ]))))           # instead
            break
    if log:
        log(f"  {name}: {len(order)} ordered, passing prefix {ok}, pruned to {len(circ)} ({lost:.2f}/{rest:.2f})")
    return {"units": circ, "singles": singles, "met": bool(ok) and passes((lost, rest)), "lost": lost, "restored": rest,
            "found": len(order), "curve": curve, "order": list(order[:ORDER_KEPT])}


def _ok(r, bar) -> bool:
    """A set's shares (lost, restored) pass at bar: one share for both, or a pair (lost's, restored's: SOURCE_NECESSITY 0
    gives the source cut no lost bar)."""
    lb, rb = bar if isinstance(bar, (tuple, list)) else (bar, bar)
    return r[0] >= lb and r[1] >= rb


def prune(meas, circ: list, bar: float = cf.PASS) -> list:
    """The pruning of a passing set (prune_steps, by its measurer alone)."""
    return drive(_pts(meas), [prune_steps(meas, circ, bar)])[0]


def prune_steps(meas, circ: list, bar: float = cf.PASS):
    """The pruning of a passing set as a search (drive's): the units whose removal alone passes, removed
    one at a time from the last while the set passes. Each round measures, for the next window of candidates, the
    removals so far plus each candidate cumulatively (right while candidates go), and after a round that ended on a
    candidate that stays, also plus each candidate alone (right while they stay): a round settles a run of removals
    and the candidate that ends it, or a run of candidates that stay and the one that goes after them, the same trials
    as one at a time. The window starts at PRUNE_WINDOW and grows four times after a round that settles all of it. A
    set passes at bar."""
    passes = lambda r: _ok(r, bar)  # noqa: E731
    if len(circ) < 2:
        return circ
    order = yield from _pair_order(meas, circ, bar)
    if PRUNE_ALONE:                                 # (each unit left out: Patcher.both_without, no set built per unit)
        alone = yield from _trials(meas, order, bar, "both_without", circ, None)
        todo = [u for u, r in zip(circ, alone) if passes(r)][::-1]
    else:                                           # (every unit a candidate, its trials alone deciding)
        todo = list(circ)[::-1]
    gone, w, single = set(), PRUNE_WINDOW, False
    while todo:                                     # (each trial a prefix of the units left with todo last, reversed,
        n, left = min(w, len(todo)), set(todo)      # or all of them with one candidate dropped)
        seq = [v for v in circ if v not in gone and v not in left] + todo[::-1]
        m = len(seq)
        extra = n - 1 if single else 0              # (the single trials: rows a run of removals would not use)
        got = yield from _trials(meas, order, bar, "both_prefixes", seq, [m - j - 1 for j in range(n)] + [m] * extra,
                                 [m] * n + [m - 1 - i for i in range(1, 1 + extra)])
        if passes(got[0]):                          # todo[0] goes: the run going with it, up to the first that stays
            j = next((j for j in range(n) if not passes(got[j])), n)
            gone |= set(todo[:j])
            todo, w, single = todo[j + 1:] if j < n else todo[n:], w * 4 if j == n else w, j < n
        elif not single:                            # todo[0] stays (its trial alone is the cumulative one)
            todo, single = todo[1:], True
        else:                                       # todo[0] stays: the run staying after it, up to the first that goes
            s = next((i for i in range(1, n) if passes(got[n + i - 1])), None)
            if s is None:
                todo, w = todo[n:], w * 4
            else:
                gone.add(todo[s])
                todo, single = todo[s + 1:], False
    return [v for v in circ if v not in gone]


def _pair_order(meas, units: list, bar):
    """STAGED_PAIRS's order of meas's pairs, worst first for the set units (one measurement of it on each pair), or None:
    off, one pair, AGG "mean", a one-prompt measurer."""
    if not (STAGED_PAIRS and AGG == "min" and type(meas) is Mean and len(meas.pts) > 1 and units):
        return None
    each = yield [(Mean([p]), "both", [_blocks(units)]) for p in meas.pts]
    lb, rb = bar if isinstance(bar, (tuple, list)) else (bar, bar)
    return sorted(range(len(meas.pts)), key=lambda i: min(each[i][0][0] - lb, each[i][0][1] - rb))


def _trials(meas, order, bar, method: str, units: list, *rest):
    """A search's trials of method ("both_without", rest (js,); "both_prefixes", rest (ks, drops)) on meas: staged over the
    pairs (_staged, with order) and settled near the bar (_settled); per trial (lost, restored)."""
    passes = lambda r: _ok(r, bar)  # noqa: E731
    res = yield from _staged(meas, order, passes, method, units, *rest)
    if method == "both_without":
        js = list(range(len(units))) if rest[0] is None else list(rest[0])
        sets = lambda idx: [[u for i, u in enumerate(units) if i != js[j]] for j in idx]  # noqa: E731
    else:
        ks, drops = list(rest[0]), (list(rest[1]) if len(rest) > 1 and rest[1] is not None else None)
        sets = lambda idx: [[u for i, u in enumerate(units[:ks[j]]) if drops is None or i != drops[j]] for j in idx]  # noqa: E731
    return _settled(meas, bar, None, res, sets)


def _settled(meas, bar, sets, res: list, make=None) -> list:
    """res (per set its (lost, restored)) with every set within SETTLE of its bar measured again on its own (SETTLE's
    note): set by set, one set a forward on each of meas's pairs, the AGG over the pairs. sets: the sets, or make(indices)
    builds them."""
    if not SETTLE or isinstance(meas, Resampled):
        return res
    lb, rb = bar if isinstance(bar, (tuple, list)) else (bar, bar)
    near = [i for i, (a, b) in enumerate(res) if (math.isfinite(lb) and abs(a - lb) < SETTLE) or (math.isfinite(rb) and abs(b - rb) < SETTLE)]
    if not near:
        return res
    agg = np.min if AGG == "min" else np.mean
    out = [tuple(r) for r in res]
    for i, s_ in zip(near, make(near) if make else [sets[i] for i in near]):
        per = [p.both([s_])[0] for p in _pts(meas)]
        out[i] = (float(agg([x[0] for x in per])), float(agg([x[1] for x in per])))
    return out


def _staged(meas, order, passes, method: str, units: list, *rest):
    """prune_steps's trials (method "both_without", rest (js,); or "both_prefixes", rest (ks, drops)) on meas, or with order
    (STAGED_PAIRS: meas's pair indices, the worst first) on order[0]'s pair, then the trials passing there on the others:
    per trial the least shares over the pairs measured."""
    if order is None:
        return (yield (meas, method, units, *rest))
    if method == "both_without":
        js = list(range(len(units))) if rest[0] is None else list(rest[0])
        sub = lambda keep: (units, [js[i] for i in keep])  # noqa: E731
    else:
        ks, drops = list(rest[0]), (list(rest[1]) if rest[1] is not None else None)
        sub = lambda keep: (units, [ks[i] for i in keep], [drops[i] for i in keep] if drops is not None else None)  # noqa: E731
    n = len(js) if method == "both_without" else len(ks)
    first = yield (Mean([meas.pts[order[0]]]), method, *sub(range(n)))
    keep = [i for i in range(n) if passes(first[i])]
    out = [tuple(r) for r in first]
    if keep:
        more = yield (Mean([meas.pts[i] for i in order[1:]]), method, *sub(keep))
        for i, r in zip(keep, more):
            out[i] = (min(out[i][0], r[0]), min(out[i][1], r[1]))
    return out


def _record(res: dict, ceiling, reach: bool, search: list, extra: dict) -> dict:
    return {"spread": None, **res, "ceiling": list(ceiling), "reachable": reach, "searched": search, **extra}


def cuts_of(pt, wanted=None) -> dict:
    """The cuts of the pair set up in pt (wanted: those names only, circuit_finder.find's cuts_wanted; a cut not wanted is
    never searched, so oracleagent's find_circuit without the readout pays nothing for a dense one)."""
    last = pt.T - 1
    out = {"readout": {"kinds": ("neuron",), "positions": [last]}, "movers": {"kinds": ("head",), "positions": [last]}}
    if pt.first < last:
        out["source"] = {"kinds": ("neuron",), "positions": _source_positions(pt, pt.first)}
    return {n: c for n, c in out.items() if wanted is None or n in wanted}


def _source_positions(pt, lo: int) -> list:
    """The source cut's positions: lo to the one before the answer, or only those of them in pt.source_positions when a
    caller set it (chive's long chat prompts, ~1,000 positions: its screen's top positions)."""
    keep = getattr(pt, "source_positions", None)
    return [p for p in range(lo, pt.T - 1) if keep is None or p in keep]


def attr_order(pt, spec: dict, sc: dict) -> list:
    """spec's units in descending RelP attribution (sc: circuit_finder.Patcher.scores), the first ORDER_MAX."""
    v = torch.cat([sc[k][:, spec["positions"]].reshape(-1) for k in spec["kinds"]])
    shapes = [(k, sc[k][:, spec["positions"]].shape) for k in spec["kinds"]]
    return _units_at(v.topk(min(ORDER_MAX, len(v))).indices, shapes, spec["positions"])


def eapig_scores(pts: list, steps: int | None = None, signed: bool = True) -> dict:
    """EAP-IG on the inputs (Hanna et al. 2024, as MIB and attr_baselines.eapig) of the pairs in pts (Patchers set up on
    one clean prompt against several corrupted ones), the mean over them: per unit its clean minus corrupted value times
    the contrast's gradient averaged over steps input embeddings on the line from the corrupted prompt's to the clean
    one's; signed (the unit's estimated support for the answer) or its magnitude. {"neuron": [L, T, D], "head": [L, T, H]}
    float32 on the host; one forward and backward of len(pts) x steps rows."""
    steps = int(EAPIG_STEPS if steps is None else steps)
    p0 = pts[0]
    core, n, L, T = p0.core, len(pts), p0.L, p0.T
    emb = core.model.embed_tokens
    with torch.no_grad():
        e1 = emb(p0._ids_dev["clean"][None]).float()                                          # [1, T, d]
        e0 = torch.cat([emb(p._ids_dev["corrupt"][None]).float() for p in pts])               # [n, T, d]
    a = torch.tensor([(k + 0.5) / steps for k in range(steps)], device=e1.device)[None, :, None, None]
    x = (e0[:, None] + a * (e1 - e0)[:, None]).reshape(n * steps, T, -1).to(emb.weight.dtype).detach()
    acts = {"neu": [None] * L, "head": [None] * L}
    hs = []
    for i, layer in enumerate(p0.layers):
        def pre(kk, i):
            def f(m, args):
                args[0].retain_grad()
                acts[kk][i] = args[0]
            return f
        hs += [layer.mlp.down_proj.register_forward_pre_hook(pre("neu", i)),
               layer.self_attn.o_proj.register_forward_pre_hook(pre("head", i))]
    try:
        with torch.enable_grad():
            x.requires_grad_(True)
            hn = core.model(inputs_embeds=x).last_hidden_state[:, -1]
            sum(pts[j]._contrast(hn[j * steps:(j + 1) * steps]).mean() for j in range(n)).backward()
    finally:
        for h in hs:
            h.remove()
    out = {}
    for kk, key in (("neu", "neuron"), ("head", "head")):
        per = []
        for i in range(L):
            g = acts[kk][i].grad.float().view(n, steps, T, -1).mean(1)                        # [n, T, W]
            d = (p0.act["clean"][kk][i].to(g.device).float()[None]
                 - torch.stack([p.act["corrupt"][kk][i].to(g.device).float() for p in pts]))
            sv = d * g
            if kk == "head":
                sv = sv.view(n, T, p0.H, p0.hd).sum(-1)
            per.append((sv if signed else sv.abs()).mean(0).cpu())
        out[key] = torch.stack(per)
    return out


def eapig_order(spec: dict, sc: dict, by_sum: bool = False) -> list:
    """spec's units by score, largest first (the first ORDER_MAX): ("neuron", layer, pos, k), ("head", layer, k, pos);
    by_sum (RANK_SUM): first by their neuron's (head's) score summed over spec's positions, then by their own."""
    own, tot, parts = [], [], []
    for k in _kinds_of(spec):
        t = sc[k][:, spec["positions"]].float()                                               # [L, P, W]
        own.append(t.reshape(-1))
        tot.append(t.sum(1, keepdim=True).expand_as(t).reshape(-1))
        parts.append((k, tuple(t.shape)))
    own, tot = torch.cat(own).numpy(), torch.cat(tot).numpy()
    idx = np.lexsort((-own, -tot)) if by_sum else np.argsort(-own, kind="stable")
    return _units_at(torch.from_numpy(idx[:ORDER_MAX].copy()), parts, spec["positions"])


def _units_at(flat: torch.Tensor, parts: list, positions: list) -> list:
    """Units at flat indices into the concatenated arrays of parts [(kind, (L, P, width))] (positions: P's), in order:
    index arithmetic on arrays, one tuple per unit at the end."""
    flat = flat.cpu().numpy().astype(np.int64)
    sizes = np.array([int(np.prod(shp)) for _, shp in parts], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    which = np.searchsorted(np.cumsum(sizes), flat, side="right")
    r = flat - starts[which]
    P = np.array([shp[1] for _, shp in parts], dtype=np.int64)[which]
    W = np.array([shp[2] for _, shp in parts], dtype=np.int64)[which]
    pos = np.asarray(positions, dtype=np.int64)[(r // W) % P]
    kinds = [k for k, _ in parts]
    return [cf.unit(kinds[a], b, c, d) for a, b, c, d in zip(which.tolist(), (r // (P * W)).tolist(), pos.tolist(), (r % W).tolist())]


def family_prior(records: list) -> dict:
    """{key: weight} from a family's earlier circuits (circuit_finder records): a unit's key is its kind, layer and index
    with its position counted back from the answer ("end") and, for source units, also on from the first changed token
    ("first"); its weight the number of circuits holding it plus half its largest share among them."""
    prior = {}
    for r in records:
        last, first = r["tokens"] - 1, r["first_diff"]
        units = [("neuron", n["layer"], n["neuron"], n["pos"], max(abs(n["lost"]), abs(n["restored"]))) for n in r.get("nodes", [])]
        units += [("head", h["layer"], h["head"], h["pos"], max(abs(h["lost"]), abs(h["restored"]))) for h in r.get("routes", [])]
        for kind, i, k, p, sh in units:
            for key in {(kind, i, k, "end", last - p)} | ({(kind, i, k, "first", p - first)} if p < last else set()):
                c, w = prior.get(key, (0, 0.0))
                prior[key] = (c + 1, max(w, sh))
    return {key: c + 0.5 * min(w, 1.0) for key, (c, w) in prior.items()}


def family_order(pt, spec: dict, prior: dict) -> list:
    """spec's units that prior's keys name at this pair's positions, by descending weight."""
    last, pos, got = pt.T - 1, set(spec["positions"]), {}
    for (kind, i, k, frm, d), w in prior.items():
        p = last - d if frm == "end" else pt.first + d
        if kind in spec["kinds"] and p in pos:
            u = cf.unit(kind, i, p, k)
            got[u] = max(got.get(u, 0.0), w)
    return sorted(got, key=lambda u: -got[u])


class Mean:
    """Partner pairs measured as one (multi-partner cuts): Patchers set up on one clean prompt against several corrupted
    ones; a search's measurement of a Mean is each pair's, its shares the means over the pairs (drive's)."""

    def __init__(self, pts: list):
        self.pts = list(pts)


class Resampled(Mean):
    """One prompt's measurer of a cut (ONE_PROMPT): Patchers on the question's clean prompt against other keys (or one
    BACKGROUND "mean" Patcher); a set's measurement is its pairs', each in the clean run: (the share of the lead lost with
    the set resampled, the AGG over the pairs, measured only for whole sets (both): 0 for a search's prefixes and removals;
    the share kept with the set at its own values and the rest of the cut resampled, the mean over the pairs, by masks
    (_kept_by_masks): 0 for a single unit, not measured)."""

    def __init__(self, pts: list, spec: dict):
        super().__init__(pts)
        self.spec = spec
        p = self.pts[0]
        self._width = {"neuron": p.D, "head": p.H}
        self._all = {k: torch.arange(w) for k, w in self._width.items()}

    def complement(self, s: list) -> list:
        """The cut's units outside the set s, a block per (kind, layer, position) of the cut."""
        used = {}
        for u in s:
            kind, i = u[0], int(u[1])
            p, k = (u[3], u[2]) if kind == "head" else (u[2], u[3])
            used.setdefault((kind, i, int(p)), []).append(k.numpy() if torch.is_tensor(k) else np.atleast_1d(np.asarray(k)))
        out = []
        for kind in _kinds_of(self.spec):
            for i in range(self.pts[0].L):
                for p in self.spec["positions"]:
                    got = used.get((kind, i, p))
                    if not got:
                        out.append(cf.unit(kind, i, p, self._all[kind]))
                        continue
                    keep = np.ones(self._width[kind], dtype=bool)
                    keep[np.concatenate(got).astype(np.int64)] = False
                    if keep.any():
                        out.append(cf.unit(kind, i, p, torch.from_numpy(np.nonzero(keep)[0])))
        return out


def _pts(meas) -> list:
    return list(meas.pts) if isinstance(meas, Mean) else [meas]


def _kept_by_masks(meas, sets: list) -> list:
    """Resampled's kept share of each set by hard masks: the clean prompt run with the cut's units at their own values on
    the set and at each key's elsewhere (both frozen), one row per set and key, KEPT_ROWS rows a forward from the cut's
    first position; the mean over the keys of the lead kept (whatever AGG). The whole cut keeps 1; an empty set gives
    what resampling the whole cut keeps. (Each set and its complement as Patcher block sets was bound by the host: on
    Qwen3-8B with 16 keys about 5 minutes a question, the card 20% busy.)"""
    spec, pts = meas.spec, meas.pts
    p0 = pts[0]
    L, dev, kinds = p0.L, p0.dev, _kinds_of(spec)
    poss = list(spec["positions"])
    start, P, at = min(poss), len(poss), {p: j for j, p in enumerate(poss)}
    width, key = {"neuron": p0.D, "head": p0.H}, {"neuron": "neu", "head": "head"}
    out, masks, idx = [1.0] * len(sets), [], []
    for j, s in enumerate(sets):
        m = {kk: torch.zeros(L, P, width[kk], dtype=torch.bool) for kk in kinds}
        for u in s:
            kind, i = u[0], int(u[1])
            p, kx = (u[3], u[2]) if kind == "head" else (u[2], u[3])
            m[kind][i, at[int(p)], kx] = True
        if not all(bool(t.all()) for t in m.values()):
            masks.append(m)
            idx.append(j)
    if not masks:
        return out
    n = len(pts)
    with torch.no_grad():
        own = {kk: [p0.act["clean"][key[kk]][i][poss].to(dev) for i in range(L)] for kk in kinds}             # [P, W]
        oth = {kk: [torch.stack([p.act["corrupt"][key[kk]][i][poss].to(dev) for p in pts]) for i in range(L)] for kk in kinds}
        rel = torch.tensor([p - start for p in poss], device=dev)
        cc = torch.tensor([p.c["clean"] for p in pts], device=dev, dtype=torch.float32)
        ids = p0._ids_dev["clean"]
        per = max(1, KEPT_ROWS // n)
        for a in range(0, len(masks), per):
            part = masks[a:a + per]
            B, R = len(part), len(part) * n
            keep = {kk: torch.stack([m[kk] for m in part]).to(dev) for kk in kinds}                             # [B, L, P, w]

            def edit(kk):
                def f(i, h):
                    o, w = own[kk][i].to(h.device, h.dtype), oth[kk][i].to(h.device, h.dtype)                    # [P, W], [n, P, W]
                    s_ = keep[kk][:, i].to(h.device, h.dtype)
                    if kk == "head":
                        s_ = s_.repeat_interleave(p0.hd, -1)
                    v = o[None, None] + (1 - s_)[:, None] * (w[None] - o[None, None])                           # [B, n, P, W]
                    return h.index_copy(1, rel.to(h.device), v.reshape(R, P, -1))
                return f
            kv = ([tuple(p0._base_kv["clean"][i][jj][:, :, :start].expand(R, -1, -1, -1).contiguous() for jj in (0, 1))
                   for i in range(L)] if start > 0 else None)
            hn = p0._last(ids, R, cache=(start, kv) if kv is not None else None,
                          neu=edit("neuron") if "neuron" in kinds else None, head=edit("head") if "head" in kinds else None,
                          only={"res": set(), "head": set(range(L)) if "head" in kinds else set(),
                                "neu": set(range(L)) if "neuron" in kinds else set()}).view(B, n, -1)
            c = torch.stack([pts[q]._contrast(hn[:, q]).float() for q in range(n)])                           # [n, B]
            for b_, v in enumerate((c / cc[:, None]).mean(0).tolist()):
                out[idx[a + b_]] = v
    return out


def _one_prompt_sets(pt, partners: list) -> dict:
    """One prompt's token sets on the question's pairs (pt and partners, one clean prompt): the answer, the clean run's
    top token, against CONTRAST's rival, each pair's contrast its lead on the clean run over a zero baseline (c corrupt 0,
    gap the lead), so a pair's shares are shares of the full model's lead."""
    pts = [pt] + list(partners)
    y = int(pt._lp["clean"].argmax())

    def rival(lp):
        v = lp.float().clone()
        v[y] = -math.inf
        return int(v.argmax())
    tops = [rival(p._lp["corrupt"]) for p in pts if not getattr(p, "_mean", False)]
    for p in pts:
        t = (rival(pt._lp["clean"]) if CONTRAST == "runner" else max(set(tops), key=tops.count) if getattr(p, "_mean", False)
             else rival(p._lp["corrupt"]))
        p.sets, p._sets_dev, p.y = {**p.sets, "answer": [y], "alternatives": [t]}, None, y
        cc = float(p._contrast(p._hn["clean"].to(p.core.lm_head.weight.device))[0])
        p.c, p.gap = {"clean": cc, "corrupt": 0.0}, cc
    return {"answer": y, "rivals": [p.sets["alternatives"][0] for p in pts], "lead_bits": [round(p.gap, 3) for p in pts],
            "contrast": CONTRAST, "background": BACKGROUND, "keys": len(tops)}


def _mean_patcher(pts: list):
    """BACKGROUND "mean": a Patcher as pts[0] (one clean prompt) whose other run's neurons and heads are the per-position
    means over pts' corrupted runs (the resampled keys)."""
    m = copy.copy(pts[0])
    cor = dict(pts[0].act["corrupt"])
    for kk in ("neu", "head"):
        cor[kk] = [torch.stack([p.act["corrupt"][kk][i].float() for p in pts]).mean(0).to(pts[0].act["corrupt"][kk][i].dtype)
                   for i in range(pts[0].L)]
    m.act, m._mean, m._group = {"clean": pts[0].act["clean"], "corrupt": cor}, True, None
    return m


class _Node:
    """A running search in drive: its generator, its parent's place for its return value ("par") and, as a parent,
    its children's returns so far and how many are still running."""
    __slots__ = ("gen", "parent", "slot", "res", "left", "started")

    def __init__(self, gen, parent=None, slot=0):
        self.gen, self.parent, self.slot, self.res, self.left, self.started = gen, parent, slot, None, 0, False


def drive(pts: list, searches: list, log=None) -> list:
    """Runs searches (generators yielding requests, sent their results) in lockstep and returns each one's return value:
    each tick every waiting search's request is taken and all are served together (_serve: every measurement of the
    tick through one circuit_finder.Group.serve over pts, the Patchers of one model and prompt length the searches
    measure). A request is a measurement (meas, method, *args): Patcher.both (sets[, given]), run (base, sets),
    both_prefixes (units, ks[, drops]) or both_without (units[, js]) of meas, a Patcher or a Mean (each of its pairs,
    the shares' means); ("scores", pt): Patcher.scores (one scores_many a tick); ("masks", job, init, steps[, keywords]):
    optimize_many's result for one job (every job of a tick in one batch, in _chunks; masks wait until every waiting
    search is on masks, so the searches' masks share their steps); ("par", [searches]): the sub-searches run beside the
    others, the parent sent their returns when all are done; or a list of requests: their results as a list."""
    group, out, waiting = cf.Group(pts), [None] * len(searches), []

    def advance(node, val):                         # (a search run up to its next request, or to its end)
        stack = [(node, val)]
        while stack:
            nd, v = stack.pop()
            try:
                req = nd.gen.send(v) if nd.started else next(nd.gen)
                nd.started = True
            except StopIteration as e:
                if nd.parent is None:
                    out[nd.slot] = e.value
                else:
                    par = nd.parent
                    par.res[nd.slot] = e.value
                    par.left -= 1
                    if not par.left:
                        stack.append((par, par.res))
                continue
            if isinstance(req, tuple) and req and isinstance(req[0], str) and req[0] == "par":
                if not req[1]:
                    stack.append((nd, []))
                    continue
                nd.res, nd.left = [None] * len(req[1]), len(req[1])
                stack += [(_Node(g, nd, k), None) for k, g in enumerate(req[1])]
            else:
                waiting.append((nd, req))
    for j, s in enumerate(searches):
        advance(_Node(s, None, j), None)
    while waiting:
        masks = [_is_masks(r) for _, r in waiting]
        now = [w for w, m in zip(waiting, masks) if not m] or list(waiting)
        waiting[:] = [w for w, m in zip(waiting, masks) if m] if len(now) < len(waiting) else []
        for (nd, _), res in zip(now, _serve(group, [r for _, r in now], log)):
            advance(nd, res)
    return out


def _is_masks(req) -> bool:
    if isinstance(req, list):
        return bool(req) and all(_is_masks(r) for r in req)
    return isinstance(req[0], str) and req[0] == "masks"


def _serve(group, reqs: list, log=None) -> list:
    """drive's requests of one tick served together: per request its result."""
    flat = []                                       # (path into reqs, request)

    def walk(r, path):
        if isinstance(r, list):
            for k, x in enumerate(r):
                walk(x, path + (k,))
        else:
            flat.append((path, r))
    for k, r in enumerate(reqs):
        walk(r, (k,))
    res = [None] * len(flat)
    scores = [k for k, (_, r) in enumerate(flat) if isinstance(r[0], str) and r[0] == "scores"]
    for k, sc in zip(scores, cf.scores_many([flat[k][1][1] for k in scores], host=False) if scores else []):
        res[k] = sc
    masks = [k for k, (_, r) in enumerate(flat) if isinstance(r[0], str) and r[0] == "masks"]
    for key in dict.fromkeys(flat[k][1][3:] for k in masks):            # (steps and optimize_many's other keywords)
        ks = [k for k in masks if flat[k][1][3:] == key]
        j0 = flat[ks[0]][1][1][0]                   # (a job's pt, or its pairs' Patchers)
        pt0 = j0[0] if isinstance(j0, (list, tuple)) else j0
        for k, r in zip(ks, _fit_masks([(k, flat[k][1][1], flat[k][1][2]) for k in ks], pt0, key, log)):
            res[k] = r
    calls, spans = [], {}                           # (measurements: a Group call per pair of each)
    bases = ("clean", "corrupt")
    for k, (_, r) in enumerate(flat):
        if isinstance(r[0], str):
            continue
        meas, method, *args = r
        if isinstance(meas, Resampled):             # (one prompt: kept shares by hard masks, lost shares measured)
            if method == "both_without":
                units = args[0]
                js = list(range(len(units))) if len(args) < 2 or args[1] is None else list(args[1])
                sets = [[u for i, u in enumerate(units) if i != j] for j in js]
            elif method == "both_prefixes":
                units, ks = args[0], list(args[1])
                drops = list(args[2]) if len(args) > 2 and args[2] is not None else None
                sets = [[u for i, u in enumerate(units[:kk]) if drops is None or i != drops[j]] for j, kk in enumerate(ks)]
            elif method == "both" and (len(args) < 2 or args[1] is None):
                sets = args[0]
            else:
                raise ValueError(f"drive: Resampled has no {method} (given)")
            if method != "both":                    # (a search's prefixes and removals: the kept share decides)
                res[k] = [(0.0, x) for x in _kept_by_masks(meas, sets)]
                continue
            probe = [bool(s_) and len(s_[0]) > 4 for s_ in sets]
            whole = [j for j, (s_, pr) in enumerate(zip(sets, probe)) if not pr and len(s_) > 1]   # (a single unit's: not
            kept = dict(zip(whole, _kept_by_masks(meas, [sets[j] for j in whole])))                # measured, 0)
            rows = [s_ for s_ in sets if s_]
            plan, at = [], 0
            for j, (s_, pr) in enumerate(zip(sets, probe)):
                plan.append(("probe" if pr else "set", at if s_ else None, kept.get(j, 0.0)))
                at += bool(s_)
            a = len(calls)
            calls += [("plain", pt, rows, ("clean",), None) for pt in meas.pts] if rows else []
            spans[k] = (a, len(calls), ("resampled", plan))
            continue
        if method == "both_without":
            units = args[0]
            js = list(range(len(units))) if len(args) < 2 or args[1] is None else list(args[1])
            method, args = "both_prefixes", [units, [len(units)] * len(js), js]
        a = len(calls)
        if method in ("both", "run"):
            sets, bs, given = (args[0], bases, args[1] if len(args) > 1 else None) if method == "both" else (args[1], (args[0],), None)
            calls += [("plain", pt, sets, bs, given) for pt in _pts(meas)]
        elif method == "both_prefixes":
            units, ks, drops = args[0], list(args[1]), (list(args[2]) if len(args) > 2 and args[2] is not None else None)
            arrays = None
            for pt in _pts(meas):
                if pt._simple(units):
                    arrays = arrays if arrays is not None else pt._arrays(units)
                    calls.append(("prefix", pt, arrays, ks, bases, drops))
                else:
                    calls.append(("plain", pt, [[u for i, u in enumerate(units[:kk]) if drops is None or i != drops[j]]
                                                for j, kk in enumerate(ks)], bases, None))
        else:
            raise ValueError(f"drive: no measurement {method}")
        spans[k] = (a, len(calls), method)
    got = group.serve(calls) if calls else []
    for k, (a, b, method) in spans.items():
        per = got[a:b]                              # (each pair's; a Mean's: the means over its pairs, or with AGG "min"
        agg = np.min if AGG == "min" else np.mean   # the least of them)
        if isinstance(method, tuple):               # (Resampled: per set (lost with it resampled, the kept share by masks);
            sh = lambda p, i: float(p[i][0]) if i is not None else 0.0  # noqa: E731   # a probe: its share both ways)
            per = per or [None]
            res[k] = [(float(agg([sh(p, i) for p in per])),) * 2 if kind == "probe" else (float(agg([sh(p, i) for p in per])), kp)
                      for kind, i, kp in method[1]]
            continue
        if method == "run":
            res[k] = [float(agg([p[i][0] for p in per])) for i in range(len(per[0]))]
        else:
            res[k] = [tuple(float(agg([p[i][x] for p in per])) for x in (0, 1)) for i in range(len(per[0]))]
    it = iter(res)                                  # (back into the requests' shape, in walk's order)

    def fill(r):
        return [fill(x) for x in r] if isinstance(r, list) else next(it)
    return [fill(r) for r in reqs]


def default_unions(model: str) -> dict | None:
    """UNIONS for UNIONS_MODEL (the default family-union movers of questions mode), None for another model or none on
    disk."""
    return json.loads(UNIONS.read_text()) if model == UNIONS_MODEL and UNIONS.exists() else None


def family_heads(unions: dict, qid: str, min_keys: int | None = None, freq: float | None = None) -> list | None:
    """The mover union of question qid's template ("task/key/p<k>") in circuit_union's unions: its paraphrase's when that
    has at least min_keys (FAMILY_MIN_KEYS) records, else its task's over every paraphrase; the heads [(layer, head)] of
    frequency at least freq (FAMILY_FREQ), most frequent first; None when the task has none."""
    task, p = qid.split("/")[0], "p" + qid.split("/")[-1][1:]
    u = unions.get(task)
    if not u:
        return None
    g = p if u["n"].get(p, 0) >= (FAMILY_MIN_KEYS if min_keys is None else min_keys) else "all"
    f0 = FAMILY_FREQ if freq is None else freq
    return [(int(i), int(h)) for i, h, f in u[g] if f >= f0] or None


def search_all(pt, arm: str, log=None, steps: int = STEPS, prior: dict | None = None, partners: list | None = None,
               wanted=None, family: list | None = None) -> dict:
    """Every cut of the pair set up in pt by arm: its result per cut name (circuit_finder._find's searched; search_steps,
    its cuts in lockstep). Arms: "mask"; "attr"; "am", attr first and masks on the cuts it leaves unmet (one batch); "fam", prior's
    units (family_order) first, then attr's. partners: Patchers set up on pt's clean prompt against other corrupted
    prompts: the source cut is then chosen and verified on the shares' means over pt's pair and theirs (Mean). family:
    the heads [(layer, head)] of the question's template mover union (family_heads): the mover cut is they, whole, when
    they pass on the pair (no pruning: pruned to the own pair they shrink back to its own heads, which pass the worst
    unseen donor 6-8% of the time, stability 10-16), else searched as without them."""
    return drive([pt] + list(partners or []), [search_steps(pt, arm, log=log, steps=steps, prior=prior, partners=partners,
                                                            wanted=wanted, family=family)], log=log)[0]


def search_steps(pt, arm: str, log=None, steps: int = STEPS, prior: dict | None = None, partners: list | None = None,
                 wanted=None, family: list | None = None):
    """search_all as a search (drive's generator)."""
    st = yield from _before_steps(pt, arm, log=log, prior=prior, partners=partners, wanted=wanted, family=family)
    got, names = {}, list(st["masked"])
    for rnd in range(ROUNDS):                       # (a cut its order cannot pass: its masks go on from their theta)
        if not names:
            break
        t0 = time.perf_counter()
        jobs = _jobs(pt, st, names)
        res = yield [("masks", job, [got[n]["theta"] if rnd else st["warm"].get(n) for n in ns], steps)
                     + ((("hard", True),),) * bool(rnd and HARD) for job, ns in jobs]
        for (_, ns), r in zip(jobs, res):
            for n, x in zip(ns, r):
                got[n] = {"theta": x["theta"], "history": got.get(n, {}).get("history", []) + x["history"],
                          "start": got.get(n, {}).get("start", x.get("start"))}
        yield from _verify_steps(st, got, names, time.perf_counter() - t0, steps, log=log)
        names = [n for n in names if not st["out"][n]["met"] and not st["out"][n].get("diffuse")]
    out = yield from _finish_steps(st, pt)
    src, movers = out.get("source", {}).get("units", []), out.get("movers", {}).get("units", [])
    if src and movers and not ONE_PROMPT:           # (each mover's share of the source cut's effect, circuit_finder.find's:
        a, b = yield [(pt, "run", "clean", [src + [u + ("clean",)] for u in movers]),     # the head frozen at its base
                      (pt, "run", "corrupt", [src + [u + ("corrupt",)] for u in movers])]  # run's value)
        out["movers"]["through"] = {u: (out["source"]["lost"] - x, out["source"]["restored"] - y) for u, x, y in zip(movers, a, b)}
    return out


def _before(pt, arm: str, log=None, prior: dict | None = None, partners: list | None = None, wanted=None) -> dict:
    """_before_steps by itself."""
    return drive([pt] + list(partners or []), [_before_steps(pt, arm, log=log, prior=prior, partners=partners, wanted=wanted)],
                 log=log)[0]


def _before_steps(pt, arm: str, log=None, prior: dict | None = None, partners: list | None = None, wanted=None,
                  family: list | None = None):
    """search_all up to the masks: every cut's ceiling, attr's (and fam's) cuts, and the cuts left for masks ("masked")
    with their hinge targets; each cut's measurer ("meas": pt, or Mean over pt and partners for the source cut)."""
    specs = cuts_of(pt, wanted)
    meas = {n: pt for n in specs}
    one = None
    if ONE_PROMPT:                                  # (one prompt: every cut on the resampled keys, or their mean)
        partners = list(partners or [])             # (no partner of the prompt's length: the question's own key alone)
        one = _one_prompt_sets(pt, partners)
        bg = [p for p in partners if getattr(p, "_mean", False)]
        keys = [pt] + [p for p in partners if not getattr(p, "_mean", False)]
        if "source" in specs:
            specs["source"] = {**specs["source"], "positions": _source_positions(pt, min(p.first for p in keys))}
        meas = {n: Resampled(bg or keys, specs[n]) for n in specs}
    for n in [n for n in specs if ALLCUTS or n == "source"] if partners and not ONE_PROMPT else []:
        meas[n] = Mean([pt] + list(partners))
    if isinstance(meas.get("source"), Mean) and not ONE_PROMPT:
        specs["source"] = {**specs["source"], "positions": _source_positions(pt, min(p.first for p in meas["source"].pts))}
    train = dict(meas)                              # (the pairs the ranking and the masks see: VAL_PAIRS's held out)
    if VAL_PAIRS and partners and not ONE_PROMPT:
        tp = Mean([pt] + list(partners)[:max(0, len(partners) - int(VAL_PAIRS))])
        train = {n: tp if isinstance(m, Mean) else m for n, m in meas.items()}
    names = list(specs)
    width = {"neuron": pt.D, "head": pt.H}
    every = {n: [cf.unit(k, i, p, torch.arange(width[k])) for k in s["kinds"] for i in range(pt.L) for p in s["positions"]]
             for n, s in specs.items()}
    # one tick: each cut's ceiling; its bar's probe (PASS, plus in reduced precision the largest |share| of a no-op patch
    # of the first layer's units at each of the cut's positions: a set taken at 0.700-0.709 in bfloat16 read 0.682-0.697
    # in float32, generality's lane3 scorecard); and the attribution of every pair (a cut chosen on its pairs' mean
    # shares is ordered by their mean attribution)
    probe = FLOOR and pt.core.model.embed_tokens.weight.dtype != torch.float32
    attr = arm in ("attr", "am", "fam")
    allp = list(dict.fromkeys([pt] + [p for n in names for p in _pts(meas[n])])) if attr else []
    got = yield ([(meas[n], "both", [every[n]]) for n in names]
                 + [(meas[n], "both", [[cf.unit(k, 0, p, torch.arange(width[k])) + ("self",)]
                                       for k in specs[n]["kinds"] for p in specs[n]["positions"]]) for n in names if probe]
                 + [("scores", p) for p in allp])
    ceil = dict(zip(names, [r[0] for r in got[:len(names)]]))
    reach = {n: ceil[n][0] >= cf.PASS and ceil[n][1] >= cf.PASS for n in names}
    bars = {n: cf.PASS for n in names}
    for n, r in zip(names, got[len(names):2 * len(names)] if probe else []):
        pr = [max(abs(a), abs(b)) for a, b in r]
        if reach[n]:
            bars[n] = cf.PASS + min(max(pr) if all(math.isfinite(x) for x in pr) else 0.0, 1 - cf.PASS)
    scs = dict(zip(map(id, allp), got[len(names) * (2 if probe else 1):]))
    for n in names if ONE_PROMPT else []:           # (one prompt: the claim is the share of the lead kept)
        bars[n] = (-math.inf, bars[n])
    if not SOURCE_NECESSITY and "source" in bars and not isinstance(bars["source"], tuple):   # (redundancy: a small source
        bars["source"] = (-math.inf, bars["source"])                                        # set's claim is sufficiency)
    if MARGIN:                                      # (every cut's bar raised to the margin)
        bars = {n: tuple(x if x == -math.inf else max(x, MARGIN) for x in b) if isinstance(b, tuple) else max(b, MARGIN)
                for n, b in bars.items()}
    out, fam = {}, None
    live = [n for n in names if reach[n]]
    if family and "movers" in live:                 # (the template's mover union, whole, when it passes on the pair)
        units = [cf.unit("head", int(i), specs["movers"]["positions"][0], int(h)) for i, h in family]
        whole, alone = yield [(meas["movers"], "both", [_blocks(units)]), (meas["movers"], "both", [[u] for u in units])]
        lo, re_ = whole[0]
        fam = {"heads": len(units), "own": [lo, re_], "taken": _ok((lo, re_), bars["movers"])}
        if fam["taken"]:
            out["movers"] = _record({"units": units, "singles": dict(zip(units, alone)), "met": True, "lost": lo, "restored": re_,
                                     "found": len(units), "curve": []}, ceil["movers"], True, specs["movers"]["positions"], {})
            live.remove("movers")
        if log:
            log(f"  movers: the family's {len(units)} heads {lo:.2f}/{re_:.2f}{'' if fam['taken'] else ', not passing: searched'}")
    ranked = [n for n in live if arm == "rank" or (arm == "hyb" and n in str(HYB_CUTS).split(","))]
    if ranked:                                      # (signed EAP-IG over the cut's training pairs, then measured)
        t0 = time.perf_counter()
        es = {}
        for n in ranked:
            key_ = tuple(map(id, _pts(train[n])))
            es[key_] = es.get(key_) or eapig_scores(_pts(train[n]))
        orders = {n: eapig_order(specs[n], es[tuple(map(id, _pts(train[n])))], by_sum=bool(RANK_SUM)) for n in ranked}
        if RERANK:                                  # (the top re-ordered by each unit's measured effect alone)
            top = {n: orders[n][:int(RERANK)] for n in ranked}
            got = yield [[(p, "both", [[u] for u in top[n]]) for p in _pts(train[n])] for n in ranked]
            for n, per in zip(ranked, got):
                eff = np.mean([[a + b for a, b in pp] for pp in per], 0)
                orders[n] = [top[n][j] for j in np.argsort(-eff, kind="stable")] + orders[n][int(RERANK):]
        res = yield ("par", [verify_steps(meas[n], orders[n], log=log, name=n, bar=bars[n]) for n in ranked])
        for n, r in zip(ranked, res):
            out[n] = _record(r, ceil[n], True, specs[n]["positions"],
                             {"rank": {"by": "eapig_signed" + ("_sum" if RANK_SUM else ""), "rerank": int(RERANK),
                                       "pairs": len(_pts(train[n])), "seconds": round(time.perf_counter() - t0, 2)}})
        live = [n for n in live if n not in ranked]
    sc = None
    if attr and live:
        t0 = time.perf_counter()
        sc = scs[id(pt)]
        orders, fams = {}, {}
        for n in live:
            sn = sc if not isinstance(meas[n], Mean) else {k: torch.stack([scs[id(p)][k] for p in meas[n].pts]).mean(0) for k in sc}
            orders[n], fams[n] = attr_order(pt, specs[n], sn), []
            if arm == "fam":
                fams[n] = family_order(pt, specs[n], prior or {})
                have = set(fams[n])
                orders[n] = fams[n] + [u for u in orders[n] if u not in have]
        res = yield ("par", [verify_steps(meas[n], orders[n], log=log, name=n, bar=bars[n]) for n in live])
        for n, r in zip(live, res):
            out[n] = _record(r, ceil[n], True, specs[n]["positions"],
                             {"attr": {"seconds": round(time.perf_counter() - t0, 2),
                                       **({"family_units": len(fams[n])} if arm == "fam" else {})}})
        live = [n for n in live if not out[n]["met"] and not out[n].get("diffuse")] if arm == "am" else []
    masked = live if arm in ("mask", "am", "hyb") else []
    warm = {}
    if WARM and masked:                             # (the masks' start from the own pair's attribution)
        sc = sc if sc is not None else (yield ("scores", pt))
        for n in masked:
            top = max(float(sc[k][:, specs[n]["positions"]].max()) for k in specs[n]["kinds"])
            warm[n] = {k: (INIT + WARM * torch.log(sc[k][:, specs[n]["positions"]].float() / max(top, 1e-30) + 1e-30)
                           ).clamp(WARM_MIN, INIT).cpu() for k in specs[n]["kinds"]}
    two = lambda b: b if isinstance(b, tuple) else (b, b)  # noqa: E731
    return {"specs": specs, "names": names, "ceil": ceil, "reach": reach, "out": out, "masked": masked, "meas": meas,
            "warm": warm, "family": fam, "bars": bars, "one_prompt": one, "train": train,
            "taus": {n: tuple(-1e9 if b == -math.inf else min(max(TAU, b + 0.05), max(b + 0.05, c - 0.05))
                              for c, b in zip(ceil[n], two(bars[n]))) for n in masked}}


def _jobs(pt, st: dict, names: list | None = None) -> list:
    """The masked cuts (names: those of them) as optimize_many jobs, with their names: the pair's own cuts one job, a
    partner cut its own."""
    names = st["masked"] if names is None else names
    tr = st.get("train") or st["meas"]              # (the masks see the training pairs; verify every pair)
    plain = [n for n in names if tr[n] is pt]
    out = [((pt, [st["specs"][n] for n in plain], [st["taus"][n] for n in plain]), plain)] if plain else []
    return out + [((tr[n].pts, [st["specs"][n]], [st["taus"][n]]), [n]) for n in names if tr[n] is not pt]


def adversarial(meas, spec: dict, units: list, steps: int | None = None, log=None, name: str = "") -> dict:
    """adversarial_steps by itself."""
    return drive(_pts(meas), [adversarial_steps(meas, spec, units, steps=steps, log=log, name=name)], log=log)[0]


def adversarial_steps(meas, spec: dict, units: list, steps: int | None = None, log=None, name: str = ""):
    """The set's shares with every other unit of the cut anywhere between the two runs, against an adversary (optimize_many
    with freeze: the set fully patched, the rest's rho moved ADV_STEP along the sign that lowers the spec's mean shares,
    shared over meas's pairs), as a search (drive's): the step whose measured lost + restored is lowest, and whether it
    still passes."""
    steps = int(ADV_CHECK if steps is None else steps)   # (an arm's constants arrive as floats: adv_check=8)
    pts = _pts(meas)
    th = {k: torch.full((pts[0].L, len(spec["positions"]), pts[0].H if k == "head" else pts[0].D), -20.0) for k in spec["kinds"]}
    at = {p: j for j, p in enumerate(spec["positions"])}
    for u in units:
        if u[0] == "head":
            th["head"][u[1], at[u[3]], u[2]] = 20.0
        else:
            th["neuron"][u[1], at[u[2]], u[3]] = 20.0
    modes = [m for m in str(ADV_MODES).replace("+", ",").split(",") if m]
    job = (pts if len(pts) > 1 else pts[0], [spec], None)
    got = yield [("masks", job, [th], steps, (("adv", ADV_STEP), ("freeze", True), ("mode", m))) for m in modes]
    out = {}
    for m, r in zip(modes, got):
        hist = r[0]["history"]
        a, b, _ = min(hist, key=lambda h: h[0] + h[1])
        out[m] = {"lost": a, "restored": b, "unmoved": list(hist[0][:2]) if m != "random" else None, "steps": steps,
                  "passes": a >= cf.PASS and b >= cf.PASS,
                  **({"mean": [sum(h[0] for h in hist) / len(hist), sum(h[1] for h in hist) / len(hist)]} if m == "random" else {})}
        if log:
            log(f"  {name}: {m} rest ({steps} steps) worst {a:.2f}/{b:.2f}")
    return {**out[modes[0]], "mode": modes[0], "modes": out}


def _verify_steps(st: dict, got: dict, names: list, t_opt: float, steps: int, log=None):
    """Each named masked cut's order (ordered, from got's theta) through verify_steps by its measurer (all at once),
    into st["out"]."""
    specs, ceil, out = st["specs"], st["ceil"], st["out"]
    t1 = time.perf_counter()
    orders = {n: ordered(specs[n], got[n]["theta"], _pts(st["meas"][n])[0].dev) for n in names}
    res = yield ("par", [verify_steps(st["meas"][n], orders[n][0], log=log, name=n, bar=st["bars"][n]) for n in names])
    checked = [n for n, r in zip(names, res) if ADV_CHECK and r["units"]]
    for n, a in zip(checked, (yield ("par", [adversarial_steps(st["meas"][n], specs[n], res[names.index(n)]["units"], log=log, name=n)
                                             for n in checked]))):
        res[names.index(n)]["adversarial"] = a
    grow = [n for n, r in zip(names, res) if EXTEND and n == EXTEND_CUT and r["met"] and len(r["units"]) < EXTEND]
    if grow:                                        # (a size control: the met source cut grown with its own order's next
        sets = []                                   # units to EXTEND units, measured; paper's C25 question)
        for n in grow:
            r = res[names.index(n)]
            have = set(r["units"])
            r["units"] = list(r["units"]) + [u for u in orders[n][0] if u not in have][:int(EXTEND) - len(r["units"])]
            sets.append((st["meas"][n], "both", [_blocks(r["units"]), *[[u] for u in r["units"]]]))
        for n, got_ in zip(grow, (yield sets)):
            r = res[names.index(n)]
            (r["lost"], r["restored"]), r["singles"], r["extended"] = got_[0], dict(zip(r["units"], got_[1:])), int(EXTEND)
    for n, r in zip(names, res):
        g, prev = got[n], out.get(n, {})
        out[n] = _record(r, ceil[n], True, specs[n]["positions"],
                         {"mask": {"kept": orders[n][1], "steps": len(g["history"]), "start": g.get("start"),
                                   **({"extended": r["extended"]} if "extended" in r else {}),
                                   "history": g["history"][::max(1, steps // 20)],
                                   "seconds_opt": round(t_opt + prev.get("mask", {}).get("seconds_opt", 0.0), 2),
                                   "seconds_verify": round(time.perf_counter() - t1 + prev.get("mask", {}).get("seconds_verify", 0.0), 2),
                                   **({"attr_unmet": prev["found"]} if "attr" in prev else
                                      {"attr_unmet": prev["mask"]["attr_unmet"]} if "attr_unmet" in prev.get("mask", {}) else {})}})


def _finish_steps(st: dict, pt=None):
    """Cuts with no result have no units; a partner cut records each pair's shares; with pt (the own pair) and NECESSARY_TOP,
    each cut's necessary units outside it (NECESSARY)."""
    specs, ceil, out = st["specs"], st["ceil"], st["out"]
    for n in st["names"]:
        if n not in out:
            out[n] = _record({"units": [], "singles": {}, "met": False, "lost": 0.0, "restored": 0.0, "found": 0, "curve": []},
                             ceil[n], st["reach"][n], specs[n]["positions"], {})
        out[n]["bar"] = st["bars"][n]
    if st.get("family") and "movers" in out:
        out["movers"]["family"] = st["family"]
    pc = [n for n in st["names"] if isinstance(st["meas"][n], Mean)]
    one = lambda m, p: Resampled([p], m.spec) if isinstance(m, Resampled) else p  # noqa: E731
    each = yield [[(one(st["meas"][n], p), "both", [_blocks(out[n]["units"])]) for p in st["meas"][n].pts] if out[n]["units"] else []
                  for n in pc]
    for n, e in zip(pc, each):                      # (each pair's shares of the cut; the record's lost and restored: means)
        out[n]["partners"] = {"pairs": len(st["meas"][n].pts), "shares": [list(x[0]) for x in e]}
    for n in st["names"] if st.get("one_prompt") else []:
        out[n]["one_prompt"] = st["one_prompt"]
    if NECESSARY_TOP and pt is not None and not st.get("one_prompt"):
        sc = yield ("scores", pt)
        ea = eapig_scores([pt], signed=False) if "eapig" in NECESSARY_ORDERS or NECESSARY_WHOLE else None
        rng = np.random.default_rng(zlib.crc32(np.asarray(pt._ids_dev["clean"].tolist(), np.int64).tobytes()))
        cand = {}
        top = int(NECESSARY_TOP)                    # (an arm constant arrives as a float: necessary_top=1024)
        for n in st["names"]:
            have = set(out[n]["units"])
            orders = [attr_order(pt, specs[n], sc)[:top + len(have)]] + (
                [eapig_order(specs[n], ea)[:top + len(have)]] if ea is not None and "eapig" in NECESSARY_ORDERS else [])
            cand[n] = []
            for u in (u for tier in itertools.zip_longest(*orders) for u in tier if u is not None):
                if len(cand[n]) >= top:
                    break
                if u not in have:
                    have.add(u)
                    cand[n].append(u)
        every = [u for n in st["names"] for u in cand[n]]
        nulls = _null_units(pt, every, rng, set(every) | {u for n in st["names"] for u in out[n]["units"]})
        nu = [u for us in nulls.values() for u in us]
        got = yield [(pt, "both", [[u] for u in every]) if every else [], (pt, "both", [[u] for u in nu]) if nu else []]
        sh, nsh = dict(zip(every, got[0])), dict(zip(nu, got[1]))
        line = {k: _line([nsh[u][0] for u in us], NECESSARY) for k, us in nulls.items()}
        for u in [u for u in every if abs(sh[u][0] - line[_cls(pt, u)]) < SETTLE_TAGS]:
            sh[u] = pt.both([[u]])[0]               # (near its line: on its own, SETTLE's note)
        for n in st["names"]:
            rows = sorted(((u, sh[u][0] >= line[_cls(pt, u)], {**_unit_json(u), "lost": round(sh[u][0], 4), "restored": round(sh[u][1], 4),
                                                              "line": round(line[_cls(pt, u)], 4)}) for u in cand[n]), key=lambda x: -x[2]["lost"])
            out[n]["necessary"] = [x for _, t, x in rows if t]
            out[n]["necessary_near"] = [x for _, t, x in rows if not t and x["lost"] >= NECESSARY / 2][:8]
        means = _necessary_means(pt)
        if means is not None:
            _mean_tags(pt, specs, out, st["names"], means, rng)
        if NECESSARY_WHOLE:
            _whole_tags(pt, sc, ea, out, st["names"], means, rng)
        if MEDIATORS:
            _mediators(pt, out, st["names"], rng)
    return out


def _cls(pt, u: tuple) -> tuple:
    """A unit's control class: kind, layer, and position class (the answer's position or the body before it)."""
    return u[0], u[1], "answer" if (u[3] if u[0] == "head" else u[2]) == pt.T - 1 else "body"


def _null_units(pt, units: list, rng, exclude: set, n: int | None = None) -> dict:
    """Per control class of units (_cls), n (NULL_UNITS) random units of it outside exclude: {class: [units]}."""
    n = NULL_UNITS if n is None else n
    body = list(range(pt.first, pt.T - 1)) or [pt.T - 1]
    out = {}
    for u in units:
        key = _cls(pt, u)
        if key in out:
            continue
        poss = [pt.T - 1] if key[2] == "answer" else body
        width = pt.D if key[0] == "neuron" else pt.H
        got, tries = [], 0
        while len(got) < n and tries < 50 * n:
            tries += 1
            v = cf.unit(key[0], key[1], int(poss[int(rng.integers(len(poss)))]), int(rng.integers(width)))
            if v not in exclude and v not in got:
                got.append(v)
        out[key] = got
    return out


def _line(null: list, floor: float) -> float:
    """A decision's line: the NULL_Q percentile of its control's shares, at least floor."""
    return max(floor, float(np.percentile(null, NULL_Q))) if null else floor


def _grad_run(pt, knock: tuple | None = None) -> dict:
    """The clean run (with knock, a unit, at its corrupted run's value) and the contrast's gradient: per layer, every
    neuron's and head channel's value there and gradient, {"neu": [(a [T, D], g [T, D])], "head": [(a, g) [T, H x hd]]}."""
    L, got, hs = pt.L, {"neu": [None] * pt.L, "head": [None] * pt.L}, []
    for i, layer in enumerate(pt.layers):
        if knock is not None and knock[1] == i:     # (registered first: the gradient hooks see the edited input)
            mod = layer.mlp.down_proj if knock[0] == "neuron" else layer.self_attn.o_proj

            def edit(m, a, u=knock):
                x = a[0].clone()
                if u[0] == "neuron":
                    x[0, u[2], u[3]] = pt.act["corrupt"]["neu"][u[1]][u[2], u[3]].to(x.device, x.dtype)
                else:
                    sl = slice(u[2] * pt.hd, (u[2] + 1) * pt.hd)
                    x[0, u[3], sl] = pt.act["corrupt"]["head"][u[1]][u[3], sl].to(x.device, x.dtype)
                return (x,) + tuple(a[1:])
            hs.append(mod.register_forward_pre_hook(edit))

        def keep(m, a, i=i, k="neu"):
            a[0].retain_grad()
            got[k][i] = a[0]
        hs += [layer.mlp.down_proj.register_forward_pre_hook(keep),
               layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, i=i: keep(m, a, i, "head"))]
    try:
        with torch.enable_grad():
            emb = pt.core.model.embed_tokens(pt._ids_dev["clean"][None]).detach().requires_grad_(True)
            pt._contrast(pt.core.model(inputs_embeds=emb).last_hidden_state[:, -1]).sum().backward()
    finally:
        for h in hs:
            h.remove()
    return {k: [(t[0].detach(), t.grad[0]) for t in v] for k, v in got.items()}


def _mediators(pt, out: dict, names: list, rng=None) -> None:
    """MEDIATORS's search (its note) into out[cut]["mediators"] and out[cut]["mediators_near"]."""
    rng = rng if rng is not None else np.random.default_rng(0)
    roots = {}
    own = [(n, u) for n in names for u in out[n]["units"]]
    for (n, u), lo in zip(own, pt.run("clean", [[u] for _, u in own]) if own else []):
        if lo >= MEDIATOR_ROOT:
            roots[u] = (n, lo)
    for n in names:
        for x in out[n].get("necessary", []):
            u = cf.unit("head", x["layer"], x["pos"], x["head"]) if "head" in x else cf.unit("neuron", x["layer"], x["pos"], x["neuron"])
            if x["lost"] >= MEDIATOR_ROOT:
                roots.setdefault(u, (n, x["lost"]))
    for n in names:
        out[n]["mediators"], out[n]["mediators_near"] = [], []
    for u, (n, lo) in roots.items():
        r = _grad_run(pt, u)
        sc = {"neuron": torch.stack([((a.float() - pt.act["clean"]["neu"][i].float()) * g.float()).abs() for i, (a, g) in enumerate(r["neu"])]),
              "head": torch.stack([((a.float() - pt.act["clean"]["head"][i].float()) * g.float()).view(pt.T, pt.H, pt.hd).sum(-1).abs()
                                   for i, (a, g) in enumerate(r["head"])])}
        sc["neuron" if u[0] == "neuron" else "head"][(u[1], u[2], u[3]) if u[0] == "neuron" else (u[1], u[3], u[2])] = 0
        spec = {"kinds": ["neuron", "head"], "positions": list(range(pt.T))}
        cand = attr_order(pt, spec, sc)[:MEDIATORS]
        excl = set(cand) | {u}
        null = []                                   # (one random unit of each candidate's control class)
        for v in cand if MEDIATOR_NULL else []:
            null += _null_units(pt, [v], rng, excl | set(null), MEDIATOR_NULL).get(_cls(pt, v), [])
        got = pt.run("clean", [[u, v + ("clean",)] for v in cand + null])
        share = {v: (lo - x) / lo for v, x in zip(cand, got[:len(cand)])}
        line = _line([(lo - x) / lo for x in got[len(cand):]], MEDIATE)
        for v in [v for v in cand if abs(share[v] - line) < SETTLE_TAGS]:   # (near the line: on its own)
            share[v] = (lo - pt.run("clean", [[u, v + ("clean",)]])[0]) / lo
        for v in cand:
            x = {**_unit_json(v), "of": _unit_json(u), "restored": round(share[v], 4), "line": round(line, 4)}
            if share[v] >= line:
                out[n]["mediators"].append(x)
            elif share[v] >= MEDIATE / 2:
                out[n]["mediators_near"].append(x)


_MEANS = {}                                         # (path -> ami.Means; control prompts' ids and margins per tokenizer)


def _necessary_means(pt):
    """NECESSARY_MEANS loaded once (ami.Means), or None: off, missing, or the Patcher's model is not NECESSARY_MEANS_MODEL."""
    name = getattr(pt.core.config, "_name_or_path", "")
    if not NECESSARY_TOP or not NECESSARY_MEANS or name != NECESSARY_MEANS_MODEL or not Path(NECESSARY_MEANS).exists():
        return None
    key = str(NECESSARY_MEANS)
    if key not in _MEANS:
        import ami                                  # (ami imports circuit_finder, not this module)
        _MEANS[key] = ami.Means(key, device=pt.dev)
    return _MEANS[key]


def _hook_rows(pt, kind: str, sets: list):
    """Forward pre-hooks setting, per batch row, its listed (layer, index, value [T, width]) channels; removed by the caller."""
    hs = []
    for i, layer in enumerate(pt.layers):
        rows = [(b, k, v) for b, (li, k, v) in enumerate(sets) if li == i]
        if not rows:
            continue

        def f(mod, args, rows=rows):
            x = args[0].clone()
            for b, k, v in rows:
                x[b, :, k] = v.to(x.device, x.dtype)
            return (x,) + tuple(args[1:])
        mod = layer.mlp.down_proj if kind == "neuron" else layer.self_attn.o_proj
        hs.append(mod.register_forward_pre_hook(f))
    return hs


def _mean_tags(pt, specs: dict, out: dict, names: list, means, rng=None) -> None:
    """NECESSARY_MEANS's tags (its note) into out[cut]["necessary_mean"] and out[cut]["hubs"]."""
    T, L, dev = pt.T, pt.L, pt.dev
    dt = pt.act["clean"]["neu"][0].dtype
    tab = {"neu": [means.table("neuron", i, T, dev, dt) for i in range(L)], "head": [means.table("head", i, T, dev, dt) for i in range(L)]}
    grads = {"neu": [None] * L, "head": [None] * L}
    hs = []
    for i, layer in enumerate(pt.layers):         # (the contrast's gradient at every neuron and head channel, clean run)
        def pre(m, a, i=i, k="neu"):
            a[0].retain_grad()
            grads[k][i] = a[0]

        def pre_h(m, a, i=i):
            a[0].retain_grad()
            grads["head"][i] = a[0]
        hs += [layer.mlp.down_proj.register_forward_pre_hook(pre), layer.self_attn.o_proj.register_forward_pre_hook(pre_h)]
    try:
        with torch.enable_grad():
            emb = pt.core.model.embed_tokens(pt._ids_dev["clean"][None]).detach().requires_grad_(True)
            hn = pt.core.model(inputs_embeds=emb).last_hidden_state[:, -1]
            pt._contrast(hn).sum().backward()
    finally:
        for h in hs:
            h.remove()
    sc = {"neuron": torch.stack([((pt.act["clean"]["neu"][i].float() - tab["neu"][i].float()) * grads["neu"][i].grad[0].float()).abs()
                                 for i in range(L)]),
          "head": torch.stack([((pt.act["clean"]["head"][i].float() - tab["head"][i].float()) * grads["head"][i].grad[0].float())
                               .view(T, pt.H, pt.hd).sum(-1).abs() for i in range(L)])}
    top = int(NECESSARY_TOP)
    cand = {}
    for n in names:
        have = set(out[n]["units"])
        cand[n] = [u for u in attr_order(pt, specs[n], sc)[:top + len(have)] if u not in have][:top]
    every = list(dict.fromkeys(u for n in names for u in cand[n]))
    rng = rng if rng is not None else np.random.default_rng(0)
    nulls = _null_units(pt, every, rng, set(every) | {u for n in names for u in out[n]["units"]})
    nu = [u for us in nulls.values() for u in us]
    pt.act["mean"] = tab
    try:
        got = pt.run("clean", [[u + ("mean",)] for u in every + nu]) if every else []
        lost, nlost = dict(zip(every, got[:len(every)])), dict(zip(nu, got[len(every):]))
        line = {k: _line([nlost[u] for u in us], NECESSARY) for k, us in nulls.items()}
        for u in [u for u in every if abs(lost[u] - line[_cls(pt, u)]) < SETTLE_TAGS]:
            lost[u] = pt.run("clean", [[u + ("mean",)]])[0]     # (near its line: on its own)
    finally:
        del pt.act["mean"]
    hit = [u for u in every if lost[u] >= line[_cls(pt, u)]]
    ctrl = _control_shares(pt, hit, means) if hit else {}
    for n in names:
        rows = [{**_unit_json(u), "lost": round(lost[u], 4), "line": round(line[_cls(pt, u)], 4), "control": round(ctrl[u], 4)}
                for u in cand[n] if u in ctrl]
        rows.sort(key=lambda x: -x["lost"])
        out[n]["necessary_mean"] = [x for x in rows if x["control"] <= NECESSARY_CONTROL]
        out[n]["hubs"] = [x for x in rows if x["control"] > NECESSARY_CONTROL]


def _whole_tags(pt, sc: dict, ea, out: dict, names: list, means, rng) -> None:
    """NECESSARY_WHOLE's tags (its note) into out[names[0]]["necessary_whole"] and ["hubs_whole"] (a whole neuron spans
    the cuts' positions; the first cut's record carries them)."""
    pos = list(range(pt.first, pt.T))
    have = {(u[1], u[3]) for n in names for u in out[n]["units"] if u[0] == "neuron"}
    tots = [sc["neuron"][:, pos].float().abs().sum(1).cpu()] + ([ea["neuron"][:, pos].float().abs().sum(1)] if ea is not None else [])
    orders = [[divmod(int(j), pt.D) for j in t.flatten().topk(min(t.numel(), int(NECESSARY_WHOLE) + len(have))).indices] for t in tots]
    cand, seen = [], set(have)
    for ln in (x for tier in itertools.zip_longest(*orders) for x in tier if x is not None):
        if len(cand) >= int(NECESSARY_WHOLE):
            break
        if ln not in seen:
            seen.add(ln)
            cand.append(ln)
    null = {}
    for layer in dict.fromkeys(l_ for l_, _ in cand):
        got, tries = [], 0
        while len(got) < NULL_UNITS and tries < 50 * NULL_UNITS:
            tries += 1
            k = int(rng.integers(pt.D))
            if (layer, k) not in seen and k not in got:
                got.append(k)
        null[layer] = got
    whole = lambda l_, k: [cf.unit("neuron", l_, p, k) for p in pos]  # noqa: E731
    nl = [(l_, k) for l_, ks in null.items() for k in ks]
    got = pt.run("clean", [whole(l_, k) for l_, k in cand + nl]) if cand else []
    lost = dict(zip(cand, got[:len(cand)]))
    nlost = dict(zip(nl, got[len(cand):]))
    line = {l_: _line([nlost[(l_, k)] for k in ks], NECESSARY) for l_, ks in null.items()}
    for ln in [ln for ln in cand if abs(lost[ln] - line[ln[0]]) < SETTLE_TAGS]:
        lost[ln] = pt.run("clean", [whole(*ln)])[0]          # (near its line: on its own)
    hit = [ln for ln in cand if lost[ln] >= line[ln[0]]]
    rep_ = {ln: cf.unit("neuron", ln[0], pos[0], ln[1]) for ln in hit}
    ctrl = _control_shares(pt, list(rep_.values()), means) if hit and means is not None else {}
    rows = [{"layer": int(l_), "neuron": int(k), "lost": round(lost[(l_, k)], 4), "line": round(line[l_], 4),
             "control": round(ctrl[rep_[(l_, k)]], 4) if rep_[(l_, k)] in ctrl else None} for l_, k in hit]
    rows.sort(key=lambda x: -x["lost"])
    out[names[0]]["necessary_whole"] = [x for x in rows if x["control"] is None or x["control"] <= NECESSARY_CONTROL]
    out[names[0]]["hubs_whole"] = [x for x in rows if x["control"] is not None and x["control"] > NECESSARY_CONTROL]
    for n in names[1:]:
        out[n]["necessary_whole"], out[n]["hubs_whole"] = [], []


def _control_shares(pt, units: list, means) -> dict:
    """Per unit, the median over CONTROL_PROMPTS of the top token's margin over the runner-up lost with the unit's neuron
    (or head channel) at its general-text mean at every position (one batch a prompt, a row a unit)."""
    key = ("ctrl", id(pt.core), repr(CONTROL_PROMPTS))
    if key not in _MEANS:                           # (a prompt as text, or as token ids already in the chat frame)
        ids = [list(q) if not isinstance(q, str) else cf.chat(pt.tok, q) for q in CONTROL_PROMPTS]
        with torch.no_grad():
            lg = [pt.core(input_ids=torch.tensor([x], device=pt.dev)).logits[0, -1].float() for x in ids]
        _MEANS[key] = [(x, int(g.argmax()), int(g.topk(2).indices[1]), float(g.topk(2).values[0] - g.topk(2).values[1]))
                       for x, g in zip(ids, lg)]
    share = {u: [] for u in units}
    for x, a, b, m0 in _MEANS[key]:
        T = len(x)
        sets = []
        for u in units:
            kind = "neuron" if u[0] == "neuron" else "head"
            k = u[3] if kind == "neuron" else torch.arange(u[2] * pt.hd, (u[2] + 1) * pt.hd, device=pt.dev)
            sets.append((u[1], k, means.table(kind, u[1], T, pt.dev, torch.float32)[:, k]))
        hs = _hook_rows(pt, "neuron", [s if u[0] == "neuron" else (-1, 0, None) for s, u in zip(sets, units)]) + \
            _hook_rows(pt, "head", [s if u[0] == "head" else (-1, 0, None) for s, u in zip(sets, units)])
        try:
            with torch.no_grad():
                lg = pt.core(input_ids=torch.tensor([x] * len(units), device=pt.dev)).logits[:, -1].float()
        finally:
            for h in hs:
                h.remove()
        for u, g in zip(units, lg):
            share[u].append(float((m0 - (g[a] - g[b])) / max(m0, 1e-6)))
    return {u: float(np.median(v)) for u, v in share.items()}


def _unit_json(u: tuple) -> dict:
    """A unit tuple (circuit_finder.unit's) as the record writes it."""
    if u[0] == "head":
        return {"layer": int(u[1]), "head": int(u[2]), "pos": int(u[3])}
    return {"layer": int(u[1]), "pos": int(u[2]), "neuron": int(u[3])}


def _find_with(pt, clean, corrupt, answers, log, arm: str, cuts: dict, rec0: dict | None = None, **kw) -> dict:
    """circuit_finder._find's record with cuts (search_all's; and rec0, setup's record, in place of setting up again)."""
    if rec0 is not None:
        pt.setup = lambda *a, **k: dict(rec0)
    try:
        rec = cf._find(pt, clean, corrupt, answers, log=log, searched=cuts, **kw)
    finally:
        if rec0 is not None:
            del pt.setup
    extra = {k: {n: c[k] for n, c in cuts.items() if k in c} for k in ("attr", "mask", "rank", "partners", "adversarial", "family",
                                                                        "one_prompt", "necessary", "necessary_mean", "hubs",
                                                                        "mediators", "necessary_near", "mediators_near",
                                                                        "necessary_whole", "hubs_whole")}   # (find() copies its own fields
    return {**rec, "search": arm, **{k: v for k, v in extra.items() if v}}                 # only)


def _budget(pt) -> int:
    """The rows x positions a mask step may run: MASK_MEMORY of the device's free memory over a token's activations kept
    for the backward (every layer's inputs to its projections and the MLP's intermediates), at least 256; MASK_TOKENS off
    CUDA."""
    if pt.dev.type != "cuda":
        return MASK_TOKENS
    free, _ = torch.cuda.mem_get_info(pt.dev)
    free += torch.cuda.memory_reserved(pt.dev) - torch.cuda.memory_allocated(pt.dev)
    cfg, e = pt.core.config, pt.core.model.embed_tokens.weight.element_size()
    per = pt.L * (6 * cfg.hidden_size + 5 * pt.D) * e
    return max(256, int(MASK_MEMORY * free // per))


def _fit_masks(items: list, pt0, key: tuple, log=None) -> list:
    """optimize_many's result per item [(item, job, init)] in order, in parts of the mask budget (_budget x MASK_SCALE),
    each step's backward chunked to it; a part that runs out of device memory halves MASK_SCALE (for the rest of the run)
    and is run again in smaller parts."""
    global MASK_SCALE
    budget = max(1, int(_budget(pt0) * MASK_SCALE))
    out = []
    for part in _chunks(items, pt0.T, budget):
        try:
            out += optimize_many([job for _, job, _ in part], steps=key[0], log=log, init=[i for _, _, i in part],
                                 budget=budget, **dict(key[1] if len(key) > 1 else ()))
            continue
        except torch.OutOfMemoryError:
            if MASK_SCALE < 2 ** -6:
                raise
        MASK_SCALE /= 2                             # (out of the handler: the failed step's tensors freed first)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if log:
            log(f"    masks: out of device memory, budget x {MASK_SCALE:g} from here")
        out += _fit_masks(part, pt0, key, log)
    return out


def _chunks(jobs: list, T: int, budget: int = MASK_TOKENS) -> list:
    """jobs [(item, (pt or pts, specs, taus), names)] in consecutive parts of at most budget rows x positions a step
    (each instance's two rows over its spec's positions from its first: a step's activations stay for the backward, so a
    part's tokens bound its memory; Qwen3-8B, 16 pairs x 4 partners on a 44 GB A40 ran out of memory)."""
    parts, cur, n = [], [], 0
    for jb in jobs:
        pts_ = jb[1][0] if isinstance(jb[1][0], (list, tuple)) else [jb[1][0]]
        t = sum(2 * len(pts_) * (T - min(s["positions"])) for s in jb[1][1])
        if cur and n + t > budget:
            parts.append(cur)
            cur, n = [], 0
        cur.append(jb)
        n += t
    return parts + ([cur] if cur else [])


def find_batch(pts: list, items: list, arm: str = "mask", steps: int = STEPS, log=None, priors: list | None = None,
               partners: list | None = None, cuts_wanted=("readout", "movers", "source"), families: list | None = None) -> list:
    """find() for several pairs at once: items [(clean, corrupt, answers)], one Patcher each (pts, of one model). Every
    pair is set up, then the pairs of one prompt length searched in lockstep (drive: each tick's measurements of every
    pair in shared batches, their masks optimized in one batch), then each pair's record. partners: per item
    [(Patcher, corrupted ids)], other corrupted prompts of its clean prompt's length: each valid one set up and the item's
    source cut chosen on the means over its pair and those (search_all's). Each record's "seconds" is the batch's wall
    time over its pairs. families: per item its template's mover union (search_all's family) or None."""
    t0 = time.perf_counter()
    recs, rec0s, same_T, part = [None] * len(items), {}, {}, {}
    families = families or [None] * len(items)
    cand = [[(ppt, ids) for ppt, ids in (partners or [[]] * len(items))[j] if len(ids) == len(items[j][0])]
            for j in range(len(items))]
    flat = [(pt, (clean, corrupt), answers) for pt, (clean, corrupt, answers) in zip(pts, items)] + [
        (ppt, (items[j][0], ids), items[j][2]) for j in range(len(items)) for ppt, ids in cand[j]]
    got = iter(cf.setup_many([x[0] for x in flat], [x[1] for x in flat], [x[2] for x in flat]))   # (every prompt once)
    for j, (pt, (clean, corrupt, answers)) in enumerate(zip(pts, items)):
        rec0 = next(got)
        if not rec0["valid"] or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
            recs[j] = {**cf._find(pt, clean, corrupt, answers, log=log, cuts_wanted=cuts_wanted), "search": arm}
            continue
        rec0s[j] = rec0
        same_T.setdefault(pt.T, []).append(j)
    for j in range(len(items)):
        part[j] = []
        for ppt, _ in cand[j]:
            r = next(got)
            if j in rec0s and r["valid"] and r["gap"] >= cf.MIN_GAP and not r["same_answer"]:
                part[j].append(ppt)
        if ONE_PROMPT and BACKGROUND == "mean" and j in rec0s:     # (one background: the keys' mean, a Patcher of its own)
            part[j].append(_mean_patcher([pts[j]] + part[j]))
    for T, js in same_T.items():                    # (the pairs of one prompt length in lockstep)
        tag = (lambda j: (lambda s: log(f"[{j}] {s.lstrip()}"))) if log and len(items) > 1 else (lambda j: log)
        cuts = drive([pts[j] for j in js] + [p for j in js for p in part[j]],
                     [search_steps(pts[j], arm, log=tag(j), steps=steps, prior=(priors or [None] * len(items))[j],
                                   partners=part[j], wanted=cuts_wanted, family=families[j]) for j in js], log=log)
        for j, c in zip(js, cuts):
            clean, corrupt, answers = items[j]
            if not c and cuts_of(pts[j], cuts_wanted):  # (a usable pair whose lockstep search returned no cuts: searched
                if log:                             # again alone; a record with no cuts read as met, persona's
                    log(f"[{j}] no cuts from the batch: searched alone")   # identity/1/dolphin/neutral)
                recs[j] = find(pts[j], clean, corrupt, answers, log=log, arm=arm, steps=steps,
                               partners=[(p, ids) for p, ids in cand[j]] or None, cuts_wanted=cuts_wanted, family=families[j])
                continue
            recs[j] = _find_with(pts[j], clean, corrupt, answers, log, arm, c, rec0=rec0s[j], cuts_wanted=cuts_wanted)
    wall = time.perf_counter() - t0
    for r in recs:
        r.update(seconds_own=r.get("seconds"), seconds=round(wall / len(items), 2), batch=len(items))
    return recs


def find(pt, clean: list[int], corrupt: list[int], answers=None, log=None, arm: str = "mask", steps: int = STEPS,
         prior: dict | None = None, partners: list | None = None, cuts_wanted=("readout", "movers", "source"),
         measure: list | None = None, sets: dict | None = None, family: list | None = None) -> dict:
    """circuit_finder.find by arm's search (its record, "search": arm); partners: as
    find_batch's, one item's; cuts_wanted, measure and sets: circuit_finder.find's; family: search_all's."""
    kw = {"cuts_wanted": cuts_wanted, "measure": measure, "sets": sets}
    t0 = time.perf_counter()
    cand = [(ppt, ids) for ppt, ids in partners or [] if len(ids) == len(clean)]
    if sets:
        rec0 = pt.setup(clean, corrupt, answers, explicit_sets=sets)
        got = cf.setup_many([p for p, _ in cand], [(clean, ids) for _, ids in cand], [answers] * len(cand)) if cand else []
    else:                                           # (every prompt once, the clean one shared)
        rec0, *got = cf.setup_many([pt] + [p for p, _ in cand], [(clean, corrupt)] + [(clean, ids) for _, ids in cand],
                                   [answers] * (1 + len(cand)))
    if not rec0["valid"] or rec0["gap"] < cf.MIN_GAP or rec0["same_answer"]:
        return {**cf._find(pt, clean, corrupt, answers, log=log, **kw), "search": arm}
    part = [ppt for (ppt, _), r in zip(cand, got) if r["valid"] and r["gap"] >= cf.MIN_GAP and not r["same_answer"]]
    if ONE_PROMPT and BACKGROUND == "mean":
        part.append(_mean_patcher([pt] + part))
    cuts = search_all(pt, arm, log=log, steps=steps, prior=prior, partners=part, wanted=cuts_wanted, family=family)
    rec = _find_with(pt, clean, corrupt, answers, log, arm, cuts, rec0=rec0, **kw)
    return {**rec, "seconds": round(time.perf_counter() - t0, 2)}


def record_cut_units(rec: dict, name: str) -> list:
    """A circuit_finder record's units of one cut: its nodes of that stage, and for the movers its routes."""
    return ([cf.unit("neuron", x["layer"], x["pos"], x["neuron"]) for x in rec.get("nodes") or [] if x["stage"] == name]
            + ([cf.unit("head", x["layer"], x["pos"], x["head"]) for x in rec.get("routes") or []] if name == "movers" else []))


def resampled_check(pts: list, rec: dict, log=None) -> dict:
    """A record's cuts by one prompt's test on other keys (pts: Patchers set up on its clean prompt against them, held
    out from its search): per cut with units, each key's (lost, kept) shares of the lead with the cut's other units
    resampled from that key (Resampled, CONTRAST's rival), the mean kept and whether it reaches PASS; the question passes
    when every such cut does."""
    one = _one_prompt_sets(pts[0], pts[1:])
    names = [n for n in ("readout", "movers", "source") if n in (rec.get("cuts") or {}) and record_cut_units(rec, n)]
    specs = {n: {**cuts_of(pts[0])[n], "positions": rec["cuts"][n]["searched"]} for n in names}

    def ask():
        return (yield [[(Resampled([p], specs[n]), "both", [record_cut_units(rec, n)]) for p in pts] for n in names])
    got = drive(pts, [ask()])[0] if names else []
    cuts = {}
    for n, per in zip(names, got):
        kept = [x[0][1] for x in per]
        cuts[n] = {"size": len(record_cut_units(rec, n)), "kept": float(np.mean(kept)), "kept_min": float(np.min(kept)),
                   "lost": float(np.mean([x[0][0] for x in per])), "per_key": [list(x[0]) for x in per],
                   "passes": float(np.mean(kept)) >= cf.PASS}
    if log:
        log(f"  {rec.get('id')}: " + ", ".join(f"{n} {c['size']} kept {c['kept']:.2f}" for n, c in cuts.items()))
    return {"id": rec.get("id"), "keys": len(pts), "one_prompt": one, "cuts": cuts, "passes": bool(cuts) and all(c["passes"] for c in cuts.values())}


def main():
    import argparse
    global CONTRAST
    import json
    from pathlib import Path

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("questions", "cases", "resampled"), help="resampled: INP a records directory, scored by one "
                    "prompt's test (resampled_check) on each question's held-out partner keys from --devset, into OUT")
    ap.add_argument("inp", help="questions: the questions JSON; cases: OUT")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--cases", default="santa,preeclampsia", help="cases: circuit_finder's CASES or CHAT_CASES")
    ap.add_argument("--word", default=None, help="cases: taboo's secret word")
    ap.add_argument("--adapter", default=None, help="a PEFT adapter on the model (an organism)")
    ap.add_argument("--tokenizer", default=None, help="the tokenizer and chat template (default: the model's; an AuditBench "
                    "adapter ships its own, as circuit_finder --tokenizer)")
    ap.add_argument("--system", default=None, help="a system prompt before each user turn (a question's own \"system\" "
                    "wins; as circuit_finder --system)")
    ap.add_argument("--merge", action="store_true", help="the adapter merged into the weights (an organism at the base "
                    "model's speed, as circuit_finder --merge)")
    ap.add_argument("--arm", default="mask", help="comma-separated arms (mask, attr, am, fam), each question through every arm in "
                    "turn on the same card, records in OUT/<arm>/")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cuts", default="readout,movers,source", help="the cuts searched (circuit_finder's --cuts; "
                    "oracleagent's find_circuit: movers,source)")
    ap.add_argument("--partners", type=int, default=8, help="pairs every cut is chosen over (ALLCUTS, PAIRWISE, AGG): the "
                    "question's own and up to PARTNERS - 1 other corrupted keys (--pool's, else its partners'); 1: its own pair")
    ap.add_argument("--batch", type=int, default=1, help="mask and am: pairs whose masks are optimized in one batch "
                    "(find_batch; a record's seconds the batch's wall time over its pairs)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ids", default=None, help="a JSON list of question ids (the frozen dev set)")
    ap.add_argument("--prior", default=None, help="fam: a directory of circuit_finder records; a question's prior is its "
                    "task's records other than its own")
    ap.add_argument("--unions", default="auto", help="circuit_union's mover unions (JSON; auto: UNIONS for UNIONS_MODEL; none: "
                    "off): each question's mover cut is its "
                    "template's union, whole, when that passes on the pair (search_all's family; FAMILY_MIN_KEYS, FAMILY_FREQ)")
    ap.add_argument("--devset", default=None, help="resampled: the questions (their \"partners\": the held-out keys)")
    ap.add_argument("--contrast", default=CONTRAST, choices=("resampled", "runner"), help="resampled: the rival token (CONTRAST)")
    ap.add_argument("--pool", default=str(POOL) if POOL.exists() else None, help="partner keys from this question set's "
                    "keys of the question's task, in a seeded order per question, instead of the question's own partners (left "
                    "for unseen-pair tests); default trainset_8b.json when present; '' for the question's own partners")
    a = ap.parse_args()
    unions = (default_unions(a.model) if a.unions == "auto" else None if a.unions == "none"
              else json.loads(Path(a.unions).read_text()))
    cf._threads()
    out = Path(a.out if a.mode in ("questions", "resampled") else a.inp)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    if a.dtype == "auto":
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(a.model)
        n = cfg.num_hidden_layers * (4 * cfg.hidden_size ** 2 + 3 * cfg.hidden_size * cfg.intermediate_size)
        a.dtype = "bfloat16" if n >= 4e9 else "float32"
    dev = a.device
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                                 device_map="auto" if dev == "auto" else None)
    model = (model if dev == "auto" else model.to(dev)).eval()
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter).eval()
        if a.merge:
            model = model.merge_and_unload().eval()
    model.requires_grad_(False)
    pt = cf.Patcher(model, tok)
    if a.mode == "resampled":                       # (one prompt's test of records, any finder's, on held-out keys)
        CONTRAST = a.contrast
        t1 = time.time()
        log1 = lambda m: print(f"{time.time() - t1:7.1f}s {m}", flush=True)   # noqa: E731
        qs = {q["id"]: q for q in json.loads(Path(a.devset).read_text())["questions"] if q.get("split") != "test"}
        pts_ = [cf.Patcher(model, tok) for _ in range(RESAMPLE_CHECK)]
        n_pass, n = 0, 0
        for f in sorted(Path(a.inp).glob("*.json")):
            rec = json.loads(f.read_text())
            q, dst = qs.get(rec.get("id")), out / f.name
            if q is None or not rec.get("ids") or dst.exists():
                continue
            pre_ = tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else []
            users = [p["user"] if isinstance(p, dict) else q["user"].replace(q["key"], p) for p in q.get("partners", [])
                     if isinstance(p, dict) or q["user"].count(q["key"]) == 1]
            ids = [x for x in (cf.chat(tok, u, q.get("system", a.system)) + pre_ for u in users) if len(x) == len(rec["ids"])]
            ids = ids[:RESAMPLE_CHECK]
            if not ids:
                continue
            got = cf.setup_many(pts_[:len(ids)], [(rec["ids"], x) for x in ids], [None] * len(ids))
            use = [p for p, r in zip(pts_, got) if r["valid"] and r["gap"] >= cf.MIN_GAP and not r["same_answer"]]
            if not use:
                continue
            res = resampled_check(use, rec, log=log1)
            dst.write_text(json.dumps(res))
            n, n_pass = n + 1, n_pass + res["passes"]
        log1(f"one prompt's test on held-out keys: {n_pass}/{n} questions pass (every cut with units keeps at least {cf.PASS} of the lead)")
        return
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:7.1f}s {m}", flush=True)   # noqa: E731
    pre = lambda q: tok(q["prefill"], add_special_tokens=False)["input_ids"] if q.get("prefill") else []  # noqa: E731
    answers = {}                                    # (a case's answer tokens: taboo's secret word)
    maxp = max([a.partners] + [int(x.split("=")[1]) for arm in a.arm.split(",") for x in arm.split(":")[1:]
                               if x.startswith("partners=")])
    if a.mode == "questions":
        qs = [q for q in json.loads(Path(a.inp).read_text())["questions"] if q.get("split", a.split) == a.split]
        qs = qs[a.offset::a.stride][:a.limit or None]
        if a.ids:
            want = set(json.loads(Path(a.ids).read_text()))
            qs = [q for q in qs if q["id"] in want]
        sy = lambda q: q.get("system", a.system)  # noqa: E731
        pairs = [(q, cf.chat(tok, q["user"], sy(q)) + pre(q), cf.chat(tok, q["corrupt_user"], sy(q)) + pre(q)) for q in qs]
        others = {}                                 # (other corrupted prompts of each question, its partners')
        keys = {}                                   # (--pool: each task's keys there)
        for p in json.loads(Path(a.pool).read_text())["questions"] if a.pool else []:
            keys.setdefault(p["task"], set()).add(p["key"])
        for q, clean, _ in pairs:
            users = {}
            if a.pool:
                ks = sorted(keys.get(q.get("task"), ()))
                np.random.default_rng(zlib.crc32(q["id"].encode())).shuffle(ks)
                users = {k: q["user"].replace(q["key"], k) for k in ks if k != q["key"] and q["user"].count(q["key"]) == 1}
            for p in [] if a.pool else q.get("partners", []):   # (keys, or {key, user} as circuit_devset writes them)
                if isinstance(p, dict):
                    users[p["key"]] = p["user"]
                elif q["user"].count(q["key"]) == 1:
                    users[p] = q["user"].replace(q["key"], p)
            ids = [(k, cf.chat(tok, u, sy(q)) + pre(q)) for k, u in users.items() if k != q.get("corrupt_key")]
            others[q["id"]] = [(k, x) for k, x in ids if len(x) == len(clean)][:maxp - 1]
    else:
        pairs = []
        for name in a.cases.split(","):
            got = cf.chat_pairs(tok, name, a.word, a.system) if name in cf.CHAT_CASES else cf.case_pairs(tok, name)
            for j, (c, k, text, ans) in enumerate(got):
                qid = f"{name}_{a.word}_{j}" if a.word else f"{name}_{j}"
                answers[qid] = ans
                pairs.append(({"id": qid, "task": name, "corrupt_text": text}, c, k))
    priors = [json.loads(p.read_text()) for p in sorted(Path(a.prior).glob("*.json"))] if a.prior else []
    priors = [r for r in priors if r.get("met") and not r.get("skipped")]
    arms = a.arm.split(",")                         # (an arm may name its mask steps: am:30, records in OUT/am_30)
    for arm in arms:
        assert arm.split(":")[0] in ("mask", "attr", "am", "fam", "hyb", "rank"), arm
        (out / _dir(arm)).mkdir(exist_ok=True)
    jobs = []
    for q, clean, corrupt in pairs:
        if len(clean) != len(corrupt):
            log(f"{q['id']}: prompts of {len(clean)} and {len(corrupt)} tokens, skipped")
            continue
        jobs.append((q, clean, corrupt))
    jobs.sort(key=lambda j: len(j[1]))              # (batches of one prompt length)
    pts = [pt] + [cf.Patcher(model, tok) for _ in range(max(0, a.batch - 1))]
    pool = [[cf.Patcher(model, tok) for _ in range(maxp - 1)] for _ in range(a.batch)]   # (partner pairs' Patchers)
    others = others if a.mode == "questions" and maxp > 1 else {}
    pt.setup(jobs[0][1], jobs[0][2])                # (a throwaway run: CUDA's first kernels)

    def write(arm, q, clean, corrupt, rec):
        rec.update(id=q["id"], ids=clean, corrupted=corrupt, answer_text=tok.decode([rec["answer"]]),
                   corrupt_answer_text=tok.decode([rec["corrupt_answer"]]), model=a.model, dtype=a.dtype, adapter=a.adapter,
                   finder="circuit_masks", finder_rev=a.rev, tokenizer=a.tokenizer or a.model, system=a.system,
                   merged=bool(a.adapter and a.merge),
                   **{k: q[k] for k in ("user", "corrupt_user", "task", "key", "corrupt_text", "prefill") if k in q})
        (out / _dir(arm) / f"{q['id'].replace('/', '__')}.json").write_text(json.dumps(rec))

    def prior_of(name, q):
        return family_prior([r for r in priors if r.get("task") == q.get("task") and r["id"] != q["id"]]) if name == "fam" else None
    for c0 in range(0, len(jobs), a.batch):
        chunk = jobs[c0:c0 + a.batch]
        for arm in arms:
            name, *parts = arm.split(":")
            steps = next((int(x) for x in parts if x.isdigit()), a.steps)
            sets = {k.upper(): _value(v) for k, v in (x.split("=") for x in parts if "=" in x)}
            npart = int(sets.pop("PARTNERS", a.partners))   # (an arm's partner pairs: mask:30:partners=4)
            oth = {i: v[:npart - 1] for i, v in others.items()}
            todo = [(q, cl, co) for q, cl, co in chunk
                    if not (out / _dir(arm) / f"{q['id'].replace('/', '__')}.json").exists()]
            if not todo:
                continue
            saved = {k: globals()[k] for k in sets}
            globals().update(sets)                  # (an arm's own constants: mask:20:init=1:lr=0.5)
            def fam_of(q):                          # (the template's mover union, --unions)
                return family_heads(unions, q["id"]) if unions and "/" in q.get("id", "") else None

            def put(q, cl, co, rec):
                if oth.get(q["id"]):                            # (the partner keys a source cut was chosen on: held-out
                    rec["selection_keys"] = [q.get("corrupt_key")] + [k for k, _ in oth[q["id"]]]   # tests skip them)
                write(arm, q, cl, co, {**rec, "constants": {**sets, "PARTNERS": npart}} if sets or npart > 1 else rec)
            try:
                if a.batch > 1 and name in ("mask", "am", "hyb", "rank"):
                    log(f"{len(todo)} pairs ({arm}, batched): " + ", ".join(q["id"] for q, _, _ in todo))
                    recs = find_batch(pts[:len(todo)], [(cl, co, answers.get(q["id"])) for q, cl, co in todo], arm=name, steps=steps,
                                      log=log, priors=[prior_of(name, q) for q, _, _ in todo],
                                      partners=[list(zip(pool[b], [x for _, x in oth.get(q["id"], [])]))
                                                for b, (q, _, _) in enumerate(todo)], cuts_wanted=tuple(a.cuts.split(",")),
                                      families=[fam_of(q) for q, _, _ in todo])
                    for (q, cl, co), rec in zip(todo, recs):
                        put(q, cl, co, rec)
                else:                               # (each record written when found: a run stopped by a time limit
                    for q, cl, co in todo:          # keeps what it found)
                        log(f"{q['id']} ({arm})")
                        put(q, cl, co, find(pt, cl, co, answers.get(q["id"]), log=log, arm=name, steps=steps,
                                            prior=prior_of(name, q), partners=list(zip(pool[0], [x for _, x in oth.get(q["id"], [])])),
                                            cuts_wanted=tuple(a.cuts.split(",")), family=fam_of(q)))
            finally:
                globals().update(saved)


def _value(v: str):
    """An arm constant's value: a number, else the text (adv_modes=unit+block+random)."""
    try:
        return float(v)
    except ValueError:
        return v


def _dir(arm: str) -> str:
    """An arm's records directory: am:30 -> am_30, mask:20:init=1 -> mask_20_init1."""
    return arm.replace(":", "_").replace("=", "")


if __name__ == "__main__":
    main()
