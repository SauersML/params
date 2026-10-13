# Toy closed loop (TOYLOOP lane): results so far

A CPU-only run of the whole oracle recipe on planted toys with known circuits. The pipeline is: zoo of models,
search teacher, small-LM oracle (SmolLM2-135M-Instruct, full fine-tune), SFT, GRPO on an executable reward, and
held-out evaluation against shortcut baselines. Every number below is from one seed. Evaluations use 30 to 40
texts per split.

## Pieces

| piece | file | status |
|---|---|---|
| zoo v1: 350 models, 10 families, multihop and detect held out | `experiments/oracle/toys/zoo.py` | every instance passes ablation checks |
| zoo v2: 340 models, 2-4 marker-gated modules each, decoys, cancelling twins, rotated basis; truth per text | `experiments/oracle/toys/zoo2.py` | worst ground-truth node KL 8e-5 |
| teacher search on zoo v1 | `experiments/oracle/toys/zoo_teacher.py` | selected teacher: unit F1 0.99, exact 0.96 |
| Gate B controls on the planted toys | `experiments/oracle/toys/gate_b_toys.py` | see below |
| oracle: render, parse, SFT, GRPO, evaluate | `experiments/oracle/toyloop/{render,circuit,train,evaluate}.py` | |
| executable reward: (sufficiency + necessity)/2, minus 0.05 per unit, worst text counted | `experiments/oracle/toyloop/reward_exec.py` | reward v3 (default): truth first on 350/350 v1 toys and 181/181 texts of the 30-instance v2 check, gate included (v2 reward: 173/181, gate 18/26); `results/reward_check_v2_r3.json` |

## Findings

1. **Φ alone rewards padding.** With an exact decomposition, every active piece scores Φ about 7.6, while the
   minimal true circuit scores about 2. On zoo v2, induction scores 0.78 for the truth against 7.64 for all actives.
   The size term has to decide the ranking.
2. **Gate B on the planted toys.** Reward v3 passes when the true circuit is complete and exact. It fails on every
   toy when the true route is incomplete, which is the case real circuits are in. There are four causes:
   - N's specificity ratio is unbounded;
   - the answer's own bootstrap eps is added to its residual;
   - the soft gate leaks compactness credit to a single node;
   - all actives sit at the Φ ceiling.

   Capping eps at 0.05 and clipping family scores to [0, 1] fixes the first three
   (`experiments/oracle/toys/results/gate_b_toys.md`). The fourth is a design decision: should an exact but larger
   circuit lose to a partial one?
3. **The influential start alone misses content-addressed keys.** On zoo v1 it stays at the empty circuit's KL on
   20% of texts (detect, gate, additive_vote). Any teacher pipeline should check the final KL and fall back.
4. **Input leaks are the first thing to rule out.** A prompt line listing the units active at the target gave unit
   F1 0.81 to 1.0 with no learning at all. With that line removed, SFT and then 24 GRPO steps on zoo v1 gave:

   | split | unit F1, SFT | unit F1, after RL | reward, SFT | reward, after RL | truth's reward |
   |---|---|---|---|---|---|
   | test_instance | 0.58 | 0.70 | 0.45 | 0.52 | 0.86 |
   | held-out family (multihop) | 0.58 | 0.78 | 0.42 | 0.75 | 0.90 |

   The reward uses no ground truth, so these F1 gains are independent evidence. Another 40 RL steps did not help on
   held-out texts.
5. **The oracle read the activations, not the weights.** In the swapped control (behaviour section from the toy,
   weight section from another toy of the same family), its answers matched the behaviour toy at F1 0.65 to 0.77
   and the weight donor at only 0.23 to 0.27. Zoo v1 was also solvable from weights alone: a no-LM rule scored unit
   F1 1.0. This is what motivated zoo v2.
6. **Zoo v2, where the circuit depends on the text.**

   | answer | unit F1, train | unit F1, test_instance | unit F1, test_family | reward (test_family) |
   |---|---|---|---|---|
   | truth | 1 | 1 | 1 | 0.86 |
   | oracle, weights + behaviour, SFT | 0.77 | 0.77 | 0.76 | 0.51 |
   | the same, + 34 GRPO steps | 0.73 | 0.79 | 0.75 | 0.49 |
   | oracle, weights only, SFT | 0.48 | 0.52 | 0.47 | 0.01 |
   | the same, + 49 GRPO steps | 0.51 | 0.50 | 0.51 | 0.09 |
   | active at the last position | 0.57 | 0.58 | 0.56 | 0.31 |
   | union of all modules | 0.43 | 0.42 | 0.46 | 0.17 |
   | weights rule, no LM | 0.24 | 0.15 | 0.46 | 0.15 |

   With activations in its input, the oracle beats every shortcut on the held-out families. From weights alone it
   names about half of each circuit's units, but its circuits do not execute. Short RL runs, with groups of 4 on a
   135M model, did not change either arm much on zoo v2.

## Open, in order of expected value

- Weight reading. Give the oracle more capacity (360M, LoRA) or a staged curriculum (zoo v1, then v2). Score
  weight use only with the swapped control on weights-only prompts.
- (Closed) the gate-family reward gap: reward v3 adds log-odds sufficiency and credits counterweight units (the always-on NO neuron); `reward_exec.py --version 2` keeps the old reward.
- RL on zoo v2: larger groups (k 8), more steps, and std-normalized advantages. The reward is noisy per text.
- Edges and English. Unit F1 runs ahead of edge F1 (about 0.55). The English is not yet verified by execution
  here; the executable-English work in the main lane applies.

## Compute notes for this container

The container has 4 CPUs and 15 GB of RAM. A SmolLM2-135M SFT step at about 2.9k tokens uses about 7 GB, so only one
run fits at a time. A GRPO step on zoo v2 takes about 95 s. Background jobs are killed at 2 h, and logs often stall
after about 1 h. Use `train.py --save_every N --minutes 55` and chain runs with `--init`. Processes detached from
the session get reclaimed when it is idle.
