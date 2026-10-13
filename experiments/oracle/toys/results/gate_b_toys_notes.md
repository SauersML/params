## Diagnosis (toyloop)

**Two tiers.** *full*: true = every reachable active piece of the planted units (with the exact decomposition this is
the complete mechanism, so it is exactly faithful: every family at 1, phi = -log2(EPS_FLOOR) = 7.64). *core*: true =
the planted route at its positions only (e.g. previous-token: target q/o/MLP, all keys, values at the source position):
the right mechanism but incomplete (point 0.7-0.98), which is the regime real circuits live in.

**Full tier: Gate B passes on every toy scored** (true beats every equal-size wrong, empty, shuffled and collapsed
candidate by +1.3 to +7). On binding_km, route1, route2 and cancel, true already *is* every VPD active, so the
"all VPD actives" control equals true and only the single-node collapse is tested there. The full tier shows the
reward can tell a complete true mechanism from wrong ones. It does not show the reward can rank partial circuits.

**Core tier: Gate B fails on every toy.** Four separate causes, each traced to one term:

1. **N's raw score is unbounded (s_clip).** phi uses the *raw* specificity ratio (D_t - mean D_c)/(D_t(VPD) - mean
   D_c(VPD)). On toys where every control text uses the same mechanism (previous-token), the denominator is about 0.
   The ratio then reaches -5.4 with eps 7.9, a residual of 14 inside the power-4 mean, so phi = -3. The true previous-token
   route then scores *below the empty circuit* (route1 and cancel, text 0). Fix: clip family scores to [0, 1] in phi
   (goal_eval already keeps the clipped estimators), or drop N when VPD's own specificity is about 0.
2. **The answer's own eps punishes the faithful circuit (eps_cap).** eps = 2x bootstrap SE is added to (1 - s). A
   faithful circuit with one high-variance family (binding_km core: patching_far eps 0.68) loses to a wrong circuit that
   scores lower on *every* family but has low variance (swap-50% beats true on phi 0.42 vs 0.20). Fix: cap eps (0.05
   here) or use the text's start-set eps for every answer.
3. **Soft gate leaks compactness to trivial circuits (gate_ramp).** On texts with a low bar (binding_km text 0:
   phi_tau 0.21; cancel text 0), the single-node circuit gets g = 0.19 x 0.5 x compact 14 bits = R 1.0-1.3. That beats
   the incomplete core and matches the teacher (1.31). Fix: multiply g by clip(phi / phi_tau, 0, 1).
4. **phi has a ceiling only at perfection (phi_cap, partial).** The complete set (all actives) gets phi 7.64, and any
   incomplete circuit gets 2 or less. Compactness pays only 0.5 bit per halving, so a 2-6x larger set costs 0.5-1.5 and
   "all VPD actives" wins every core text by 4-9. Capping phi at phi_tau + 2 fixes only 4 of 12. Whether this should
   count as a reward failure depends on the gate's definition: here the collapsed control is *more faithful* than the
   incomplete truth. On a real model, where no circuit is exact, the same ceiling makes a near-complete set dominate.

**Repairs, rescored offline** (table below, from the same rows): eps_cap + s_clip makes the core tier pass wrong_eq,
empty and shuffled on every text (12/12) and changes nothing in the full tier. gate_ramp is harmless and closes the
single-node leak. Collapsed-vs-incomplete still fails (4/12 at best, with phi_cap). Proposed minimal reward change, for
the owners of goal_eval.phi to decide: clip family scores to [0, 1] and cap eps at about 0.05 inside phi. The second
step needs a decision on what a complete-but-large circuit should earn against a partial-but-small one.

**Also found:** run_planted.reward_v3 sets `components.NODE_CAP = 10**9` to lift the cap, but `reward(..., node_cap=NODE_CAP)`
binds the default when the function is defined, so the lift has no effect. Every circuit above 266 nodes comes out
invalid (R = -10). Fix: pass `node_cap=None` (gate_b_toys does this).

**Final run (all 7 toys, 3 texts each).** Full tier: every toy passes except cancel. There, "true minus one node plus
the uncancelled +u neuron" ties true or beats it by under 0.01 on one text. That neuron is not active at the target on
those texts, so the swap barely changes behaviour; it is a near-tie, not a term failure. Core tier, with eps_cap +
s_clip: wrong_eq 13/15, empty 15/15, shuffled 15/15. The two remaining wrong_eq failures are the +u-neuron swap on
cancel. Collapsed-vs-incomplete stays at 0-6/15 under every repair.
