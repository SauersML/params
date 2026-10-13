# Gate B on the planted toys

`experiments/oracle/toys/gate_b_toys.py` (toyloop). Reward v3's circuit part (`components.reward`: circuit_reward = phi + g·½compact, D = 0 with no teacher reference), eval draws, the exact decomposition (planted.decompose), node cap lifted. Texts per toy: 3. Rows: `gate_b_toys.jsonl`. Tiers: **full** (true = every reachable active piece of the planted units) and **core** (true = the planted route's positions only, see core_of); each tier has its own equal-size wrong circuits and controls.

Gate B holds on a text when R(true) > max R over each control category (an exact tie, |margin| < 1e-9, a control behaviourally identical to true, is counted apart and not as a failure). Cell: texts passing, and the worst margin R(true) − R(best control) with that control (positive = true wins).

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


## Verdict per toy and tier

| toy | tier | wrong_eq | empty | shuffled | collapsed | R(true) − R(teacher) | Gate B |
|---|---|---|---|---|---|---|---|
| binding_km | full | PASS 3/3 (+6.10, swap 50% to wrong unit) | PASS 3/3 (+6.16, empty) | PASS 3/3 (+6.14, shuffled) | PASS 3/3 (+4.84, single node) | +5.34 | **PASS** |
| binding_km | core | FAIL 1/3 (-0.10, swap 50% to wrong unit) | PASS 3/3 (+0.07, empty) | FAIL 2/3 (-0.09, shuffled) | FAIL 0/3 (-6.11, all VPD actives) | -0.49 | **FAIL** |
| binding_fp | full | PASS 3/3 (+5.62, other mechanism (padded/trimmed)) | PASS 3/3 (+6.54, empty) | PASS 3/3 (+6.44, shuffled) | PASS 3/3 (+1.33, all VPD actives) | +6.21 | **PASS** |
| binding_fp | core | PASS 3/3 (+0.99, swap 50% to wrong unit) | PASS 3/3 (+1.16, empty) | PASS 3/3 (+1.32, shuffled) | FAIL 0/3 (-4.12, all VPD actives) | +1.09 | **FAIL** |
| route1 | full | PASS 3/3 (+2.35, swap 50% to wrong unit) | PASS 3/3 (+2.43, empty) | PASS 3/3 (+2.42, shuffled) | PASS 3/3 (+2.43, single node) | +3.45 | **PASS** |
| route1 | core | FAIL 2/3 (-1.21, random actives #2) | FAIL 2/3 (-2.99, empty) | FAIL 2/3 (-1.45, shuffled) | FAIL 0/3 (-9.20, all VPD actives) | -1.90 | **FAIL** |
| route2 | full | PASS 3/3 (+3.36, swap 50% to wrong unit) | PASS 3/3 (+3.49, empty) | PASS 3/3 (+3.48, shuffled) | PASS 3/3 (+3.46, single node) | +4.66 | **PASS** |
| route2 | core | FAIL 2/3 (-0.07, head0 alone (padded)) | PASS 3/3 (+0.31, empty) | PASS 3/3 (+0.32, shuffled) | FAIL 0/3 (-5.11, all VPD actives) | +0.23 | **FAIL** |
| additive | full | PASS 1/1 (+6.84, swap 50% to wrong unit) | PASS 1/1 (+6.98, empty) | PASS 1/1 (+6.95, shuffled) | PASS 1/1 (+6.05, single node) | +6.29 | **PASS** |

## Mean over texts per candidate

### binding_km

| tier | candidate | category | nodes | R | phi | phi_tau | g | compact | ≥ true (of texts) |
|---|---|---|---|---|---|---|---|---|---|
| full | **true** | true | 2003 | 6.268 | 7.644 | 0.537 | 1.000 | -2.751 | — |
| full | swap 50% to wrong unit | wrong_eq | 2003 | -0.176 | 0.218 | 0.537 | 0.257 | -2.874 | 0/3 |
| full | swap 100% to wrong unit | wrong_eq | 2003 | -0.053 | 0.062 | 0.537 | 0.086 | -2.552 | 0/3 |
| full | empty | empty | 0 | -0.005 | -0.005 | 0.537 | 0.068 | — | 0/3 |
| full | shuffled | shuffled | 1360 | -0.058 | 0.016 | 0.537 | 0.081 | -1.552 | 0/3 |
| full | single node | collapsed | 1 | 0.483 | -0.003 | 0.537 | 0.068 | 14.567 | 0/3 |
| core | **true** | true | 762 | 0.438 | 0.532 | 0.537 | 0.492 | -0.386 | — |
| core | other mechanism (padded/trimmed) | wrong_eq | 762 | 0.257 | 0.307 | 0.537 | 0.289 | -0.154 | 2/3 |
| core | random actives #0 | wrong_eq | 762 | 0.015 | 0.016 | 0.537 | 0.085 | 0.112 | 0/3 |
| core | random actives #1 | wrong_eq | 762 | 0.032 | 0.034 | 0.537 | 0.074 | 0.127 | 0/3 |
| core | random actives #2 | wrong_eq | 762 | 0.029 | 0.034 | 0.537 | 0.086 | 0.070 | 0/3 |
| core | swap 50% to wrong unit | wrong_eq | 762 | 0.139 | 0.226 | 0.537 | 0.283 | -0.430 | 1/3 |
| core | swap 100% to wrong unit | wrong_eq | 762 | 0.044 | 0.061 | 0.537 | 0.093 | -0.153 | 0/3 |
| core | empty | empty | 0 | -0.005 | -0.005 | 0.537 | 0.068 | — | 0/3 |
| core | shuffled | shuffled | 549 | 0.062 | 0.033 | 0.537 | 0.100 | 0.695 | 1/3 |
| core | single node | collapsed | 1 | 0.483 | -0.003 | 0.537 | 0.068 | 14.567 | 1/3 |
| core | all VPD actives | collapsed | 2003 | 6.268 | 7.644 | 0.537 | 1.000 | -2.751 | 3/3 |
| ref | run_planted true (narrow) | ref | 762 | 0.438 | 0.532 | 0.537 | 0.492 | -0.386 | — |
| ref | run_planted other (narrow) | ref | 410 | 0.114 | 0.077 | 0.537 | 0.099 | 0.945 | — |
| teacher | teacher | teacher | 147 | 0.925 | 0.400 | 0.537 | 0.328 | 3.795 | — |

### binding_fp

| tier | candidate | category | nodes | R | phi | phi_tau | g | compact | ≥ true (of texts) |
|---|---|---|---|---|---|---|---|---|---|
| full | **true** | true | 372 | 6.590 | 7.644 | 1.321 | 1.000 | -2.107 | — |
| full | other mechanism (padded/trimmed) | wrong_eq | 372 | 0.545 | 1.970 | 1.321 | 0.965 | -2.958 | 0/3 |
| full | random actives #0 | wrong_eq | 372 | -2.208 | -2.208 | 1.321 | 0.000 | -2.897 | 0/3 |
| full | random actives #1 | wrong_eq | 372 | 0.167 | 0.168 | 1.321 | 0.001 | -2.903 | 0/3 |
| full | random actives #2 | wrong_eq | 372 | 0.268 | 0.269 | 1.321 | 0.001 | -2.906 | 0/3 |
| full | swap 50% to wrong unit | wrong_eq | 372 | 0.309 | 0.312 | 1.321 | 0.002 | -3.051 | 0/3 |
| full | swap 100% to wrong unit | wrong_eq | 372 | 0.084 | 0.084 | 1.321 | 0.000 | -1.707 | 0/3 |
| full | empty | empty | 0 | -0.009 | -0.009 | 1.321 | 0.000 | — | 0/3 |
| full | shuffled | shuffled | 258 | 0.052 | 0.053 | 1.321 | 0.000 | -0.769 | 0/3 |
| full | single node | collapsed | 1 | -0.008 | -0.010 | 1.321 | 0.000 | 11.934 | 0/3 |
| full | all VPD actives | collapsed | 694 | 5.243 | 7.644 | 1.321 | 1.000 | -4.803 | 0/3 |
| core | **true** | true | 83 | 1.472 | 1.211 | 1.321 | 0.319 | 1.637 | — |
| core | other mechanism (padded/trimmed) | wrong_eq | 83 | -0.052 | -0.052 | 1.321 | 0.000 | 1.548 | 0/3 |
| core | random actives #0 | wrong_eq | 83 | -3.232 | -3.232 | 1.321 | 0.000 | 1.686 | 0/3 |
| core | random actives #1 | wrong_eq | 83 | -1.499 | -1.499 | 1.321 | 0.000 | 1.989 | 0/3 |
| core | random actives #2 | wrong_eq | 83 | -5.049 | -5.049 | 1.321 | 0.000 | 1.765 | 0/3 |
| core | swap 50% to wrong unit | wrong_eq | 83 | -0.062 | -0.063 | 1.321 | 0.001 | 1.650 | 0/3 |
| core | swap 100% to wrong unit | wrong_eq | 83 | -2.845 | -2.845 | 1.321 | 0.000 | 3.032 | 0/3 |
| core | empty | empty | 0 | -0.009 | -0.009 | 1.321 | 0.000 | — | 0/3 |
| core | shuffled | shuffled | 69 | -0.459 | -0.460 | 1.321 | 0.000 | 2.325 | 0/3 |
| core | single node | collapsed | 1 | -0.325 | -0.326 | 1.321 | 0.000 | 11.934 | 0/3 |
| core | all VPD actives | collapsed | 694 | 5.243 | 7.644 | 1.321 | 1.000 | -4.803 | 3/3 |
| ref | run_planted true (narrow) | ref | 126 | 1.321 | 1.321 | 1.321 | 0.500 | 0.000 | — |
| ref | run_planted other (narrow) | ref | 251 | 0.908 | 1.518 | 1.321 | 0.659 | -1.883 | — |
| teacher | teacher | teacher | 35 | 0.378 | 0.368 | 1.321 | 0.004 | 4.841 | — |

### route1

| tier | candidate | category | nodes | R | phi | phi_tau | g | compact | ≥ true (of texts) |
|---|---|---|---|---|---|---|---|---|---|
| full | **true** | true | 771 | 4.944 | 6.391 | 1.295 | 1.000 | -2.895 | — |
| full | swap 50% to wrong unit | wrong_eq | 771 | -3.027 | -3.026 | 1.295 | 0.000 | -4.050 | 0/3 |
| full | swap 100% to wrong unit | wrong_eq | 771 | -2.278 | -2.266 | 1.295 | 0.007 | -3.434 | 0/3 |
| full | empty | empty | 0 | -0.008 | -0.008 | 1.295 | 0.004 | — | 0/3 |
| full | shuffled | shuffled | 529 | -4.187 | -4.186 | 1.295 | 0.000 | -1.298 | 0/3 |
| full | single node | collapsed | 1 | 0.017 | -0.004 | 1.295 | 0.004 | 10.502 | 0/3 |
| core | **true** | true | 438 | -0.408 | -0.454 | 1.295 | 0.294 | 0.285 | — |
| core | random actives #0 | wrong_eq | 438 | -2.243 | -2.242 | 1.295 | 0.001 | -1.400 | 0/3 |
| core | random actives #1 | wrong_eq | 438 | -3.372 | -3.372 | 1.295 | 0.000 | -0.953 | 0/3 |
| core | random actives #2 | wrong_eq | 438 | -0.436 | -0.420 | 1.295 | 0.022 | -1.472 | 1/3 |
| core | swap 50% to wrong unit | wrong_eq | 438 | -2.901 | -2.901 | 1.295 | 0.000 | -1.719 | 0/3 |
| core | swap 100% to wrong unit | wrong_eq | 438 | -3.552 | -3.552 | 1.295 | 0.000 | -3.142 | 0/3 |
| core | empty | empty | 0 | -0.008 | -0.008 | 1.295 | 0.004 | — | 1/3 |
| core | shuffled | shuffled | 303 | -0.920 | -0.920 | 1.295 | 0.000 | 1.209 | 1/3 |
| core | single node | collapsed | 1 | 0.017 | -0.004 | 1.295 | 0.004 | 10.502 | 1/3 |
| core | all VPD actives | collapsed | 771 | 4.944 | 6.391 | 1.295 | 1.000 | -2.895 | 3/3 |
| ref | run_planted true (narrow) | ref | 771 | 4.944 | 6.391 | 1.295 | 1.000 | -2.895 | — |
| teacher | teacher | teacher | 30 | 1.490 | 0.736 | 1.295 | 0.408 | 3.803 | — |

### route2

| tier | candidate | category | nodes | R | phi | phi_tau | g | compact | ≥ true (of texts) |
|---|---|---|---|---|---|---|---|---|---|
| full | **true** | true | 764 | 5.274 | 6.725 | 0.800 | 1.000 | -2.901 | — |
| full | swap 50% to wrong unit | wrong_eq | 764 | 0.138 | 0.194 | 0.800 | 0.029 | -3.799 | 0/3 |
| full | swap 100% to wrong unit | wrong_eq | 764 | -1.387 | -1.383 | 0.800 | 0.004 | -2.288 | 0/3 |
| full | empty | empty | 0 | -0.007 | -0.007 | 0.800 | 0.005 | — | 0/3 |
| full | shuffled | shuffled | 524 | -0.486 | -0.484 | 0.800 | 0.003 | -1.542 | 0/3 |
| full | single node | collapsed | 1 | 0.027 | -0.002 | 0.800 | 0.005 | 11.386 | 0/3 |
| core | **true** | true | 442 | 0.846 | 0.714 | 0.800 | 0.546 | 0.502 | — |
| core | head0 alone (padded) | wrong_eq | 442 | -1.537 | -1.520 | 0.800 | 0.031 | -1.119 | 1/3 |
| core | head1 alone (padded) | wrong_eq | 442 | 0.255 | 0.282 | 0.800 | 0.054 | -1.056 | 0/3 |
| core | random actives #0 | wrong_eq | 442 | -0.972 | -0.972 | 0.800 | 0.001 | -1.460 | 0/3 |
| core | random actives #1 | wrong_eq | 442 | -1.277 | -1.275 | 0.800 | 0.003 | -1.107 | 0/3 |
| core | random actives #2 | wrong_eq | 442 | 0.209 | 0.245 | 0.800 | 0.049 | -1.385 | 0/3 |
| core | swap 50% to wrong unit | wrong_eq | 442 | -1.219 | -1.216 | 0.800 | 0.004 | -1.560 | 0/3 |
| core | swap 100% to wrong unit | wrong_eq | 442 | -0.031 | -0.024 | 0.800 | 0.004 | -3.047 | 0/3 |
| core | empty | empty | 0 | -0.007 | -0.007 | 0.800 | 0.005 | — | 0/3 |
| core | shuffled | shuffled | 319 | -1.057 | -1.058 | 0.800 | 0.002 | 1.263 | 0/3 |
| core | single node | collapsed | 1 | 0.021 | -0.008 | 0.800 | 0.005 | 11.386 | 0/3 |
| core | all VPD actives | collapsed | 764 | 5.274 | 6.725 | 0.800 | 1.000 | -2.901 | 3/3 |
| ref | run_planted true (narrow) | ref | 764 | 5.274 | 6.725 | 0.800 | 1.000 | -2.901 | — |
| ref | run_planted head0 alone (narrow) | ref | 377 | -0.478 | -0.478 | 0.800 | 0.000 | -0.747 | — |
| ref | run_planted head1 alone (narrow) | ref | 387 | -0.046 | -0.033 | 0.800 | 0.028 | -0.872 | — |
| teacher | teacher | teacher | 47 | 0.612 | 0.397 | 0.800 | 0.103 | 4.287 | — |

### additive

| tier | candidate | category | nodes | R | phi | phi_tau | g | compact | ≥ true (of texts) |
|---|---|---|---|---|---|---|---|---|---|
| full | **true** | true | 793 | 6.972 | 7.644 | 0.264 | 1.000 | -1.344 | — |
| full | swap 50% to wrong unit | wrong_eq | 793 | 0.130 | 0.693 | 0.264 | 0.951 | -1.183 | 0/1 |
| full | swap 100% to wrong unit | wrong_eq | 793 | 0.048 | 0.326 | 0.264 | 0.607 | -0.917 | 0/1 |
| full | empty | empty | 0 | -0.007 | -0.007 | 0.264 | 0.133 | — | 0/1 |
| full | shuffled | shuffled | 573 | 0.026 | 0.048 | 0.264 | 0.183 | -0.242 | 0/1 |
| full | single node | collapsed | 1 | 0.924 | -0.007 | 0.264 | 0.133 | 14.043 | 0/1 |
| ref | run_planted true (narrow) | ref | 793 | 6.972 | 7.644 | 0.264 | 1.000 | -1.344 | — |
| teacher | teacher | teacher | 312 | 0.680 | 0.380 | 0.264 | 0.692 | 0.866 | — |

## Candidate repairs, rescored from the same rows

Gate B checks passed (texts) when R is recomputed from each row's family scores and eps (no new measurement; phi_tau kept as measured):

- **eps_cap**: each family's eps capped at 0.05 (a noisy family cannot push the residual above 1 - s + 0.05)
- **s_clip**: each family's raw score clipped to [0, 1] (the clipped estimators goal_eval keeps beside the raw ones; N's raw specificity ratio reaches -5 on a good circuit)
- **gate_ramp**: g x clip(phi / phi_tau, 0, 1): compactness gets nothing at phi <= 0 (the empty and one-node circuits)
- **phi_cap**: phi capped at phi_tau + 2.0 (past the bar, faithfulness stops paying and compactness ranks)

| variant | full: wrong_eq | full: empty | full: shuffled | full: collapsed | core: wrong_eq | core: empty | core: shuffled | core: collapsed | full: all | core: all |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 13/13 | 13/13 | 13/13 | 13/13 | 8/12 | 11/12 | 10/12 | 0/12 | 13/13 | 0/12 |
| eps_cap | 13/13 | 13/13 | 13/13 | 13/13 | 11/12 | 11/12 | 11/12 | 0/12 | 13/13 | 0/12 |
| s_clip | 13/13 | 13/13 | 13/13 | 13/13 | 8/12 | 11/12 | 10/12 | 0/12 | 13/13 | 0/12 |
| gate_ramp | 13/13 | 13/13 | 13/13 | 13/13 | 8/12 | 11/12 | 10/12 | 0/12 | 13/13 | 0/12 |
| phi_cap | 13/13 | 13/13 | 13/13 | 12/13 | 8/12 | 11/12 | 10/12 | 3/12 | 12/13 | 3/12 |
| eps_cap+s_clip | 13/13 | 13/13 | 13/13 | 13/13 | 12/12 | 12/12 | 12/12 | 0/12 | 13/13 | 0/12 |
| eps_cap+s_clip+gate_ramp | 13/13 | 13/13 | 13/13 | 13/13 | 12/12 | 12/12 | 12/12 | 0/12 | 13/13 | 0/12 |
| eps_cap+s_clip+phi_cap | 13/13 | 13/13 | 13/13 | 12/13 | 12/12 | 12/12 | 12/12 | 4/12 | 12/13 | 4/12 |
| eps_cap+s_clip+gate_ramp+phi_cap | 13/13 | 13/13 | 13/13 | 13/13 | 12/12 | 12/12 | 12/12 | 4/12 | 13/13 | 4/12 |

## Failures: which term lets the control win (per text)

- binding_km core text 0, wrong_eq: **swap 50% to wrong unit** ≥ true by 0.10: Δphi +0.21, Δ(g·½compact) -0.11; families higher for it: none
- binding_km core text 0, shuffled: **shuffled** ≥ true by 0.09: Δphi -0.13, Δ(g·½compact) +0.22; families higher for it: none
- binding_km core text 0, collapsed: **all VPD actives** ≥ true by 6.11: Δphi +7.44, Δ(g·½compact) -1.33; families higher for it: patching_far +0.812, nodes +0.461, pattern +0.239, edges +0.238, point +0.233, track_pp +0.024, patching_near +0.022, interchange_pp +0.020
- binding_km core text 1, wrong_eq: **other mechanism (padded/trimmed)** ≥ true by 0.02: Δphi -0.05, Δ(g·½compact) +0.06; families higher for it: pattern +0.149, nodes +0.022
- binding_km core text 1, collapsed: **all VPD actives** ≥ true by 5.67: Δphi +7.03, Δ(g·½compact) -1.35; families higher for it: patching_far +0.715, pattern +0.590, nodes +0.458, point +0.376, edges +0.360
- binding_km core text 2, collapsed: **all VPD actives** ≥ true by 5.71: Δphi +6.87, Δ(g·½compact) -1.16; families higher for it: nodes +0.442, patching_far +0.411, edges +0.103, point +0.096, pattern +0.090, interchange_pp +0.015
- binding_fp core text 0, collapsed: **all VPD actives** ≥ true by 4.12: Δphi +6.73, Δ(g·½compact) -2.61; families higher for it: patching_near +0.497, nodes +0.478, edges +0.454, pattern +0.285, point +0.285, track_pp +0.101
- binding_fp core text 1, collapsed: **all VPD actives** ≥ true by 3.67: Δphi +6.39, Δ(g·½compact) -2.72; families higher for it: nodes +0.554, point +0.338, pattern +0.338, edges +0.334
- binding_fp core text 2, collapsed: **all VPD actives** ≥ true by 3.52: Δphi +6.18, Δ(g·½compact) -2.66; families higher for it: nodes +0.371, edges +0.353, point +0.328, pattern +0.328, track_pp +0.072
- route1 core text 0, wrong_eq: **random actives #2** ≥ true by 1.21: Δphi +1.21, Δ(g·½compact) -0.00; families higher for it: N +3.963
- route1 core text 0, empty: **empty** ≥ true by 2.99: Δphi +2.99, Δ(g·½compact) -0.00; families higher for it: N +5.364
- route1 core text 0, shuffled: **shuffled** ≥ true by 1.45: Δphi +1.45, Δ(g·½compact) +0.00; families higher for it: N +6.364
- route1 core text 0, collapsed: **all VPD actives** ≥ true by 9.20: Δphi +10.64, Δ(g·½compact) -1.45; families higher for it: N +6.364, edges +0.596, nodes +0.322, point +0.018, pattern +0.017
- route1 core text 1, collapsed: **all VPD actives** ≥ true by 5.62: Δphi +7.10, Δ(g·½compact) -1.48; families higher for it: edges +0.698, nodes +0.626, pattern +0.169, point +0.141
- route1 core text 2, collapsed: **all VPD actives** ≥ true by 1.24: Δphi +2.79, Δ(g·½compact) -1.55; families higher for it: nodes +0.676, edges +0.221, pattern +0.171, point +0.155
- route2 core text 0, collapsed: **all VPD actives** ≥ true by 4.99: Δphi +6.69, Δ(g·½compact) -1.69; families higher for it: nodes +0.562, pattern +0.551, point +0.529, edges +0.496, patching_far +0.020
- route2 core text 1, collapsed: **all VPD actives** ≥ true by 5.11: Δphi +6.74, Δ(g·½compact) -1.64; families higher for it: nodes +0.633, edges +0.597, pattern +0.540, point +0.539
- route2 core text 2, wrong_eq: **head0 alone (padded)** ≥ true by 0.07: Δphi +0.08, Δ(g·½compact) -0.01; families higher for it: edges +0.088
- route2 core text 2, collapsed: **all VPD actives** ≥ true by 3.18: Δphi +4.60, Δ(g·½compact) -1.42; families higher for it: edges +0.812, nodes +0.797
