# Teacher circuits on the toy zoo

`experiments/oracle/toys/zoo_teacher.py` (toyloop). 350 instances, 1470 example texts (the records' examples, ordinary and counterfactual), exact decomposition (planted.native_of). Methods per text:

- **teacher influential**: `teacher.teach(f1_start="influential")`, run_planted's "current" arm (influential-blocks start, block then piece pruning to the budget KL_empty^0.1·KL_VPD^0.9, connections, no-strand);
- **teacher both**: `teacher.teach(f1_start="both")`, teach's default F1 search (VPD's answer at the target first, then the whole answer; top-k; joint path), same budget, connections, no-strand;
- **teacher selected**: per text, the smaller of the two arms among those within budget (KL ≤ max(2·budget, 1e-7)), else the lower KL: chosen without the ground truth; this is the SFT target written to the teacher files;
- **greedy b**: unit-level greedy ablation (units by single-removal KL, smallest first; drop while KL(model ‖ kept units only, the rest zeroed) ≤ b).

Pieces map to units (a head's q/k/v/o pieces → L·.H·, a neuron's pieces → L·.N·). Scores against the example's own gt units: precision, recall, F1, exact = identical unit set. Per-instance targets in `~/mpd-data/toy_zoo/teacher/<id>.json`; per-text rows in `zoo_teacher.jsonl`.

## Unit recovery per split (per text, mean)

| split | method | texts | precision | recall | F1 | exact |
|---|---|---|---|---|---|---|
| train | teacher influential | 962 | 0.92 | 0.98 | 0.93 | 0.80 |
| train | teacher both | 962 | 0.96 | 1.00 | 0.97 | 0.93 |
| train | teacher selected | 962 | 1.00 | 0.98 | 0.99 | 0.95 |
| train | greedy 0.05 | 962 | 0.92 | 0.77 | 0.79 | 0.55 |
| train | greedy 0.001 | 962 | 0.93 | 0.88 | 0.87 | 0.64 |
| test_instance | teacher influential | 228 | 0.88 | 0.98 | 0.90 | 0.72 |
| test_instance | teacher both | 228 | 0.96 | 1.00 | 0.97 | 0.94 |
| test_instance | teacher selected | 228 | 0.99 | 0.98 | 0.98 | 0.94 |
| test_instance | greedy 0.05 | 228 | 0.87 | 0.70 | 0.72 | 0.44 |
| test_instance | greedy 0.001 | 228 | 0.89 | 0.84 | 0.82 | 0.51 |
| test_family | teacher influential | 280 | 0.74 | 1.00 | 0.80 | 0.62 |
| test_family | teacher both | 280 | 1.00 | 1.00 | 1.00 | 0.99 |
| test_family | teacher selected | 280 | 1.00 | 1.00 | 1.00 | 0.99 |
| test_family | greedy 0.05 | 280 | 1.00 | 1.00 | 1.00 | 1.00 |
| test_family | greedy 0.001 | 280 | 1.00 | 1.00 | 1.00 | 1.00 |
| all | teacher influential | 1470 | 0.88 | 0.98 | 0.90 | 0.75 |
| all | teacher both | 1470 | 0.97 | 1.00 | 0.98 | 0.95 |
| all | teacher selected | 1470 | 1.00 | 0.99 | 0.99 | 0.96 |
| all | greedy 0.05 | 1470 | 0.93 | 0.81 | 0.82 | 0.62 |
| all | greedy 0.001 | 1470 | 0.94 | 0.90 | 0.89 | 0.69 |

## Unit recovery per family (per text, mean over all splits): F1 / exact

| family | split | texts | teacher influential | teacher both | teacher selected | greedy 0.05 | greedy 0.001 |
|---|---|---|---|---|---|---|---|
| copy_offset | train+test_inst | 140 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| induction | train+test_inst | 140 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| binding_positional | train+test_inst | 210 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| gate | train+test_inst | 140 | 0.79 / 0.49 | 1.00 / 1.00 | 1.00 / 1.00 | 0.67 / 0.00 | 0.67 / 0.00 |
| additive_vote | train+test_inst | 140 | 0.76 / 0.16 | 0.99 / 0.94 | 0.98 / 0.92 | 0.46 / 0.00 | 0.93 / 0.76 |
| bigram_mlp | train+test_inst | 140 | 0.99 / 0.99 | 0.85 / 0.67 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| redundant | train+test_inst | 140 | 0.94 / 0.73 | 1.00 / 1.00 | 0.94 / 0.73 | 0.60 / 0.00 | 0.71 / 0.00 |
| cancel | train+test_inst | 140 | 0.90 / 0.80 | 0.91 / 0.83 | 0.96 / 0.93 | 0.34 / 0.00 | 0.48 / 0.00 |
| multihop | test_family | 140 | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| detect | test_family | 140 | 0.61 / 0.26 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Precision / recall per family (teacher selected): copy_offset 1.00/1.00; induction 1.00/1.00; binding_positional 1.00/1.00; gate 1.00/1.00; additive_vote 1.00/0.97; bigram_mlp 1.00/1.00; redundant 1.00/0.90; cancel 0.95/1.00; multihop 1.00/1.00; detect 1.00/1.00.

## Instance level vs the record's gt_units

Teacher: union of the selected per-text circuits over the record's texts. Greedy: once on 64 ordinary texts (mean KL).

| family | instances | teacher F1 | teacher exact | greedy 0.05 F1 | greedy 0.05 exact | greedy 0.001 F1 | greedy 0.001 exact |
|---|---|---|---|---|---|---|---|
| copy_offset | 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| induction | 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| binding_positional | 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| gate | 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| additive_vote | 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| bigram_mlp | 35 | 1.00 | 1.00 | 0.71 | 0.03 | 0.71 | 0.03 |
| redundant | 35 | 1.00 | 1.00 | 0.60 | 0.00 | 0.69 | 0.00 |
| cancel | 35 | 0.93 | 0.86 | 0.50 | 0.00 | 0.50 | 0.00 |
| multihop | 35 | 0.99 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| detect | 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| all | 350 | 0.99 | 0.98 | 0.88 | 0.70 | 0.89 | 0.70 |

## Instance-level edges vs the record's unit_edges

Unit → unit and unit → OUT edges (token sources left out: the record names them by role, the teacher by position). Teacher edges are the selected circuits' connections collapsed to units, after valid_edge.

| family | edge precision | edge recall | exact |
|---|---|---|---|
| copy_offset | 1.00 | 1.00 | 1.00 |
| induction | 1.00 | 1.00 | 1.00 |
| binding_positional | 1.00 | 1.00 | 1.00 |
| gate | 0.61 | 1.00 | 0.00 |
| additive_vote | 0.85 | 1.00 | 0.49 |
| bigram_mlp | 0.93 | 0.53 | 0.03 |
| redundant | 1.00 | 1.00 | 1.00 |
| cancel | 0.90 | 1.00 | 0.86 |
| multihop | 0.66 | 1.00 | 0.00 |
| detect | 1.00 | 1.00 | 1.00 |
| all | 0.90 | 0.95 | 0.64 |

## KL and size (per text)

KL at the target, model ‖ circuit only. Node KL: the circuit's pieces on, every other piece off (zoo.node_kl; the record's gt_nodes the same way). Unit KL: every unit outside the set zeroed (zoo.ablate). "suff" = share of texts within the teacher's budget. KLs are medians; sizes are means (pieces / units).

| family | gt node KL | infl. node KL (suff) | both node KL (suff) | selected node KL | greedy 0.05 unit KL | gt pieces | selected pieces | gt units | infl. units | both units | selected units | greedy 0.05 units | s/text (infl., both) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| copy_offset | 9.5e-10 | 9.5e-10 (1.00) | 9.5e-10 (1.00) | 9.5e-10 | 9.8e-10 | 13.7 | 4.4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.2, 0.3 |
| induction | 1.6e-09 | 1.5e-09 (1.00) | 1.5e-09 (1.00) | 1.5e-09 | 1.8e-09 | 18.4 | 9.0 | 2.00 | 2.00 | 2.00 | 2.00 | 2.00 | 0.3, 0.4 |
| binding_positional | 8.3e-10 | 8.3e-10 (1.00) | 8.3e-10 (1.00) | 8.3e-10 | 8.5e-10 | 15.1 | 4.4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.0, 1.4 |
| gate | 2.1e-10 | 8.1e-06 (0.49) | 2.1e-10 (1.00) | 2.1e-10 | 8.1e-06 | 24.1 | 11.0 | 3.49 | 5.79 | 3.49 | 3.49 | 1.97 | 0.3, 0.4 |
| additive_vote | 1.2e-09 | 8.7e-04 (0.22) | 5.3e-10 (1.00) | 5.3e-10 | 0.01 | 42.0 | 16.6 | 3.51 | 5.14 | 3.40 | 3.36 | 1.01 | 5.8, 5.4 |
| bigram_mlp | 7.9e-10 | 7.9e-10 (0.99) | 7.9e-10 (1.00) | 7.9e-10 | 9.2e-10 | 7.8 | 4.0 | 1.46 | 1.54 | 2.05 | 1.46 | 1.46 | 2.1, 3.0 |
| redundant | 1.6e-15 | -6.5e-10 (1.00) | -6.5e-10 (1.00) | -6.5e-10 | 1.7e-05 | 33.3 | 8.3 | 2.37 | 2.10 | 2.37 | 2.10 | 1.00 | 0.2, 0.4 |
| cancel | 5.0e-10 | 5.2e-10 (1.00) | 5.2e-10 (1.00) | 5.2e-10 | 1.1e-09 | 14.4 | 5.0 | 1.00 | 1.40 | 1.34 | 1.14 | 2.36 | 0.2, 0.4 |
| multihop | 3.5e-07 | 1.2e-09 (0.93) | 1.0e-09 (1.00) | 1.2e-09 | 1.0e-09 | 25.4 | 9.9 | 2.00 | 2.01 | 2.01 | 2.01 | 2.00 | 0.3, 0.4 |
| detect | 9.2e-10 | 5.15 (0.26) | 1.3e-09 (1.00) | 1.3e-09 | 4.9e-10 | 5.0 | 4.0 | 1.00 | 2.91 | 1.00 | 1.00 | 1.00 | 0.2, 0.2 |
| all | 8.3e-10 | 1.1e-09 (0.80) | 7.1e-10 (1.00) | 7.1e-10 | 1.7e-09 | 19.7 | 7.5 | 1.84 | 2.42 | 1.92 | 1.81 | 1.46 | 1.1, 1.2 |

## Node level (selected teacher pieces vs the record's gt_nodes; per text, mean)

| family | node precision | node recall | node exact |
|---|---|---|---|
| copy_offset | 0.95 | 0.31 | 0.00 |
| induction | 1.00 | 0.50 | 0.00 |
| binding_positional | 1.00 | 0.30 | 0.00 |
| gate | 1.00 | 0.46 | 0.00 |
| additive_vote | 0.86 | 0.35 | 0.00 |
| bigram_mlp | 1.00 | 0.75 | 0.54 |
| redundant | 0.98 | 0.26 | 0.00 |
| cancel | 0.92 | 0.30 | 0.00 |
| multihop | 0.55 | 0.22 | 0.00 |
| detect | 1.00 | 0.84 | 0.50 |

## Failure modes (texts)

| method | texts | missed ≥1 gt unit | distractor kept | other extra unit | redundant: all routes kept | redundant: one route only | cancel: both of pair kept | cancel: one of pair |
|---|---|---|---|---|---|---|---|---|
| teacher influential | 1470 | 57 | 285 | 73 | 102/140 | 11/140 | 28/140 | 0/140 |
| teacher both | 1470 | 8 | 2 | 46 | 140/140 | 0/140 | 24/140 | 0/140 |
| teacher selected | 1470 | 49 | 2 | 0 | 102/140 | 11/140 | 10/140 | 0/140 |
| greedy 0.05 | 1470 | 465 | 0 | 0 | 0/140 | 140/140 | 95/140 | 45/140 |
| greedy 0.001 | 1470 | 320 | 0 | 0 | 0/140 | 90/140 | 134/140 | 6/140 |

### Per family, teacher selected (texts with the mode)

| family | texts | missed ≥1 | missed units by layer/kind | distractor kept | other extra | arm chosen (infl./both) |
|---|---|---|---|---|---|---|
| copy_offset | 140 | 0 | — | 0 | 0 | 140/0 |
| induction | 140 | 0 | — | 0 | 0 | 140/0 |
| binding_positional | 210 | 0 | — | 0 | 0 | 210/0 |
| gate | 140 | 0 | — | 0 | 0 | 68/72 |
| additive_vote | 140 | 11 | L1 head (12), L0 head (10) | 0 | 0 | 29/111 |
| bigram_mlp | 140 | 0 | — | 0 | 0 | 139/1 |
| redundant | 140 | 38 | L1 head (22), L0 head (16) | 0 | 0 | 140/0 |
| cancel | 140 | 0 | — | 0 | 0 | 122/18 |
| multihop | 140 | 0 | — | 2 | 0 | 124/16 |
| detect | 140 | 0 | — | 0 | 0 | 36/104 |

Teacher errors: 0

## Where search misses known structure

Numbers below are from the run above (1,470 texts, 350 instances).

1. **The influential start misses content-addressed positions.** `block_search` keeps whole blocks at the target plus the
   F2-probe positions, the fewest that reach s_track_pp ≥ 0.9. On **detect** (the head attends to wherever the trigger
   tokens are) it keeps the target position alone in 104/140 texts. That start already sits at the empty-circuit KL
   (median 5.15 nats), so no removal fits the budget and nothing gets pruned: the answer is every piece at the
   target, the two distractor neurons included. The same thing happens, less severely, on **gate** (72/140 insufficient)
   and **additive_vote** (99+7/140). In every text where the influential arm kept a distractor (285), its circuit was
   over budget, apart from 2 multihop texts. Influential is within budget on 80% of texts; the `both` arm is within
   budget on 100%.
2. **Redundant routes: sufficiency keeps one route when the logit is saturated.** With 2 or 3 identical heads and
   p(top) = 1.000000, dropping a route costs KL below the budget (about 1e-9, against budgets of 7e-9 to 1e-8). The
   teacher's smallest-within-budget pick keeps only some routes on 38/140 texts. In the per-text "redundant" column,
   the selected teacher keeps 2 of 3 routes or 1 of 2. The unit-level greedy keeps **exactly one** route on every text
   at b = 0.05 (140/140), and at b = 0.001 on 90/140. The `both` arm keeps all routes on 140/140, but because of its
   start rather than any search for necessity. Without the ground truth, a sufficiency-only search cannot tell
   "redundant" from "unused". Instance-level redundant F1 is 1.00 for the teacher only because the union over texts
   recovers the dropped routes.
3. **Cancelling pairs.** The planted pair (two units writing exactly opposite vectors) is removable only *jointly*.
   The unit-level greedy removes units one at a time, so it keeps both units on 95/140 texts (b = 0.05) and 134/140
   (b = 0.001), and keeps one of the pair, a broken half, on 45 and 6. The teacher mostly gets this right
   (selected: pair kept on 10/140, all of them heads-type pairs) because its first stage drops whole site blocks, and
   removing a site block removes both members at once. The failures are head pairs in the same layer as the copy head,
   where the attention block cannot be dropped whole (influential 28, both 15). Neuron pairs are dropped every time
   by the influential arm; the `both` arm keeps a neuron pair on 9 texts. The opposite error also appears: **additive_vote** misses ≥1 voter on 11 texts, and in
   every case checked the missed voters are a pair that cancels on that text (one votes yes, one votes no) or are
   over-determined (4 yes against 1 no, saturated). On that text the circuit is faithful (node KL ≤ 6e-9), but it is
   not the planted mechanism.
4. **Distractors.** The distractors' writes are DUMP = 0.005 into residual directions nothing reads, which moves the
   KL by about 1e-15. A sufficient search drops them, and the selected teacher keeps a distractor on only 2/1,470
   texts (multihop). Every distractor kept by the influential arm comes from failure mode 1 (an over-budget start
   that cannot be pruned), not from the distractors mattering.
5. **Ground-truth artifact, not a search error: bigram_mlp at instance level.** The record's gt_units is the union
   over its 4 example texts, so it lists only the lookup neurons those texts use. Greedy on 64 texts finds the rest
   of the planted table, scoring "F1 0.71 / exact 0.03". Per text, every method is 1.00 on bigram_mlp except the
   `both` arm, which adds other table neurons (F1 0.85). Instance-level scores for this family should be read
   against the planted table (params.neurons), not gt_units.
6. **Node level.** Selected-teacher pieces are nearly all inside the gt node set (precision 0.86–1.00, except
   multihop 0.55), but recall is 0.22–0.84. The gt node sets list a positional head's key pieces at every position
   and the value at the source, while the teacher keeps only the few keys that set the attention pattern. Both are
   sufficient. The pieces differ, and the units agree.
7. **Edges are over-connected.** Instance-level edge recall is 1.00 outside bigram_mlp, where the gt names only the
   neurons its 4 texts use, so recall there is 0.53. Precision is lower: gate 0.61, multihop 0.66, additive_vote
   0.85. The teacher's connections state direct paths the planted design does not route through, such as a layer-0
   head → OUT next to its route via the layer-1 neurons, or a layer-0 head → the gate's bias neuron. With the exact
   decomposition these paths carry tiny but nonzero signal, and the connection search keeps every connection whose
   removal costs more than the 1e-8 budget. Site-level connections collapsed to units also produced 2,475 same-layer
   head→head edges, which are not paths; `valid_edge` drops them.

## Takeaways for SFT targets

- Use **teacher selected** (written as the target in `~/mpd-data/toy_zoo/teacher/<id>.json`). Per text it has unit
  precision 1.00 and recall 0.99 (exact 0.96, F1 0.99); train split F1 0.99 / exact 0.95; test_family 1.00 / 0.99.
  Its node KL is at gt level (median 7e-10) with 7.5 pieces against gt's 19.7.
- `teach(f1_start="influential")` alone, run_planted's arm, is not safe on the zoo. It is insufficient on 20% of
  texts, and when it fails it fails silently: the answer is still emitted. Any teacher pipeline should check that
  the final circuit's KL is within budget and fall back to the `both` arm.
- The residual errors are structural (redundancy and saturation, text-level cancellation). They come from what
  per-text sufficiency means, not from search noise. Scoring the oracle on multiple texts per instance, or adding a
  necessity term, is what would expose them. An oracle SFT'd on these targets will learn "one route" on saturated
  redundant texts.
- The unit-level greedy is a weak baseline: F1 0.82–0.89 and exact 0.62–0.69. It is too coarse to drop cancelling
  pairs, and at b = 0.05 per text it also drops real voters, because the vote is saturated.
