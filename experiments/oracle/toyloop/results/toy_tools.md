# Tool-use toy oracle on zoo v2 (TOYLOOP lane)

The oracle receives a short first observation: the text, the target position and the unit names. It then queries
weights-derived tools, one `CALL name(args)` per turn, and ends with `ANSWER` plus the Mermaid and English answer
(`toy_tools.py`). The tools are marker, gated, reads, writes, twins, fires and run. The first six give facts parsed from
`render.weight_only_lines(detail=True)`, so they read only the weights and the text. run() executes reward_exec, at a
cost of 0.02 per call. An episode is about 600 tokens, against about 3k for the detailed prompt.

## Scripted expert (weights_heuristic2's rule run through the tools)

The calls are marker, gated(marker), twins(gated), fires(gated neurons), then reads(kept), and reads of any traced
layer-0 neurons. The rule's unit set equals the truth on every text of every split. The SFT answer uses the truth's
edges and a short English line built from the tools; the unit set needed no correction.

| split | texts | exact unit set | mean calls | max calls |
|---|---|---|---|---|
| train | 1258 | 1.00 | 4.25 | 5 |
| test_instance | 296 | 1.00 | 4.28 | 5 |
| test_family | 483 | 1.00 | 4.15 | 5 |

## Trained oracle (SmolLM2-135M, full fine-tune, 40 texts per split, greedy)

Training: episode SFT with loss on the assistant turns only. It started from `sft_v1_weights`: after an 80-step pilot
from each start, validation loss was 0.038 for sft_v1_weights and 0.061 for the base model. The run was 594 steps with
4 episodes per step, 2 epochs, lr 3e-5 cosine, in two chunks of 54 minutes or less. Validation loss went from 2.52 to
0.023. The 1188 training episodes leave out 10 training instances (70 texts), which form the validation set.
Checkpoint: `~/mpd-data/toyloop/tools_sft`.

| oracle | unit F1 train/inst/family | exact | reward_exec train/inst/family | calls |
|---|---|---|---|---|
| tool-use oracle (this) | 0.93/0.94/0.91 | 0.78/0.85/0.75 | 0.72/0.74/0.70 | 4.4/4.3/4.2, run() never called |
| detailed-render prompt oracle (`eval_sft_detail_v2_weights`) | 0.64/0.65/0.65 | 0.18/0.28/0.23 | 0.22/0.27/0.24 | n/a |
| scripted expert / rule (`eval_tools_expert`) | 1.00/1.00/1.00 | 1.00 | 0.83/0.86/0.86 (= truth) | 4.4/4.3/4.2 |

Edge F1 is 0.76/0.78/0.69. The model's call sequence equals the expert's on 120 of 120 texts, so every remaining error
is in the ANSWER turn. Typical errors: copying the wrong neuron of several (bigram_mlp: it names an off neuron instead
of the one that fires), and dropping a weak layer-0 in-link (multihop, gate). The weakest families on test_family are
bigram_mlp (F1 0.58) and gate (F1 0.67). Results: `eval_tools_tools_sft_{train,test_instance,test_family}.{md,json}`
and `eval_tools_expert.{md,json}`.

Caveats. run() reports reward_exec's scores, and reward_exec chooses its same-truth texts with the generator's gt_spec,
just as the reward itself does. The expert never calls run(), so the trained model never learned to use it. The
train-split row is in-distribution: the model was trained on those texts' episodes. One seed.
