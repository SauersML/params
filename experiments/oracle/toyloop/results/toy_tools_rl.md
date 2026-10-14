# GRPO on the tool-use toy oracle (TOYLOOP lane)

`toy_tools_train.py rl` runs GRPO over whole multi-turn episodes. Each step samples 2 texts and k=6 episodes per text
at temperature 0.8; the 6 episodes run in lockstep, with one batched generate per turn. Reward per episode =
reward_exec of the answer - 0.02 per run() call. Advantage = reward - group mean (`--std` optionally divides by the
group std). An optional k3 KL to the starting policy is used (beta 0.05 here). The loss is on the assistant tokens only.
Texts are drawn from the training texts minus the 10 held-out validation instances. Sampling is weighted toward the
weak families (bigram_mlp x3, gate x3, induction x1.5). The family labels steer only sampling, never the reward.
Each chunk lasts 55 minutes or less and is resumable. After a chunk, greedy rollouts on 30 validation texts are scored
(never a test split), and the checkpoint goes to `best` when the validation episode reward improves. The starting
policy is scored first.

run() made useful: `toy_tools.py episodes --verify 0.15,0.25` gives 15% of texts a "check" step: the expert runs its
draft, then answers. It gives 25% a "revise" step: the expert first runs a wrong draft, then its own draft, then
answers. The wrong draft is built from weight facts only: an on gated neuron swapped for an off one, a unit dropped,
or a twin added. The wrong drafts score far lower (for example suff 0.00 against 0.98). tools_sft was SFT'd on this set
(148 steps, lr 1e-5) to give `tools_sft_run`.

Each step takes about 90 s (12 episodes), so one chunk is about 20 steps, or 40 texts.

| arm | steps | val episode reward, greedy, 30 texts (start → chunks) | val exact |
|---|---|---|---|
| from tools_sft, lr 2e-6 then 5e-6 with std | 22 + 28 | 0.714 → 0.691 → 0.673 (best: the start) | 0.80 → 0.73 → 0.67 |
| from tools_sft_run, lr 2e-6 | 18 + 22 | 0.666 → **0.759** → 0.716 (best: step 18) | 0.67 → 0.73 → 0.80 |

Held-out, 40 texts per split (the same texts as before), for the best RL checkpoint (`tools_rl_run/best`, step 18):

| oracle | unit F1 train/inst/family | exact | reward train/inst/family | calls | run() calls |
|---|---|---|---|---|---|
| tools_sft | 0.93/0.94/0.91 | 0.78/0.85/0.75 | 0.72/0.74/0.70 | 4.4/4.3/4.2 | 0 |
| tools_sft_run + 18 GRPO steps | 0.90/0.94/0.92 | 0.70/0.88/0.75 | 0.69/0.75/0.71 | 4.4/4.3/4.2 | 0 |

RL left the held-out numbers where they were: every difference is one or two texts out of 40. The validation gain
(0.67 → 0.76) did not carry over to the test splits, and train-split precision fell from 0.97 to 0.88. The KL to the
start stayed at 0.01-0.05 per token, so the policy barely moved. The earlier drift toward bigger circuits appears only
weakly: mean predicted units on validation rose from 2.7-2.8 to 3.0-3.4. The arm from tools_sft lost validation reward
in both chunks. Greedy policies never call run(): it is the minority branch in the SFT data. Sampled episodes in the run
arm do call it, on 0.2 to 0.9 of episodes per step, but RL did not make it the greedy choice, at a cost of 0.02 per
call. The weak families stay weak (bigram_mlp reward 0.35-0.50).

Limits: about 40 RL steps per arm (80 texts), one seed, and 30 validation texts, where one text is worth about 0.03.
On this CPU budget RL is sample-starved. Each step spends most of its time generating the deterministic call turns
again; reusing those turns and sampling only the ANSWER turn would make steps several times cheaper. Files:
`eval_tools_tools_rl_run_{train,test_instance,test_family}.{md,json}`. Checkpoints: `~/mpd-data/toyloop/tools_rl`,
`tools_rl_run/{best,last}` and `tools_sft_run`.
