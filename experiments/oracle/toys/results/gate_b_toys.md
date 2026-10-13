# Gate B on the planted toys

`experiments/oracle/toys/gate_b_toys.py` (toyloop). Reward v3's circuit part (`components.reward`: circuit_reward = phi + g·½compact, D = 0 with no teacher reference), eval draws, the exact decomposition (planted.decompose), node cap lifted. Texts per toy: 3. Rows: `gate_b_toys.jsonl`. Tiers: **full** (true = every reachable active piece of the planted units) and **core** (true = the planted route's positions only, see core_of); each tier has its own equal-size wrong circuits and controls.

Gate B holds on a text when R(true) > max R over each control category (an exact tie, |margin| < 1e-9, a control behaviourally identical to true, is counted apart and not as a failure). Cell: texts passing, and the worst margin R(true) − R(best control) with that control (positive = true wins).


## Verdict per toy and tier

| toy | tier | wrong_eq | empty | shuffled | collapsed | R(true) − R(teacher) | Gate B |
|---|---|---|---|---|---|---|---|
| binding_km | full | PASS 3/3 (+6.10, swap 50% to wrong unit) | PASS 3/3 (+6.16, empty) | PASS 3/3 (+6.14, shuffled) | PASS 3/3 (+4.84, single node) | +5.34 | **PASS** |
| binding_km | core | FAIL 1/3 (-0.10, swap 50% to wrong unit) | PASS 3/3 (+0.07, empty) | FAIL 2/3 (-0.09, shuffled) | FAIL 0/3 (-6.11, all VPD actives) | -0.49 | **FAIL** |
| binding_fp | full | PASS 3/3 (+5.62, other mechanism (padded/trimmed)) | PASS 3/3 (+6.54, empty) | PASS 3/3 (+6.44, shuffled) | PASS 3/3 (+1.33, all VPD actives) | +6.21 | **PASS** |
| binding_fp | core | PASS 3/3 (+0.99, swap 50% to wrong unit) | PASS 3/3 (+1.16, empty) | PASS 3/3 (+1.32, shuffled) | FAIL 0/3 (-4.12, all VPD actives) | +1.09 | **FAIL** |
| route1 | full | PASS 3/3 (+2.35, swap 50% to wrong unit) | PASS 3/3 (+2.43, empty) | PASS 3/3 (+2.42, shuffled) | PASS 3/3 (+2.43, single node) | +3.45 | **PASS** |
| route1 | core | FAIL 2/3 (-1.21, random actives #2) | FAIL 2/3 (-2.99, empty) | FAIL 2/3 (-1.45, shuffled) | FAIL 0/3 (-9.20, all VPD actives) | -1.90 | **FAIL** |
| route2 | full | PASS 1/1 (+6.10, swap 50% to wrong unit) | PASS 1/1 (+6.19, empty) | PASS 1/1 (+6.19, shuffled) | PASS 1/1 (+6.15, single node) | +5.46 | **PASS** |
| route2 | core | PASS 1/1 (+0.85, head1 alone (padded)) | PASS 1/1 (+1.20, empty) | PASS 1/1 (+1.29, shuffled) | FAIL 0/1 (-4.99, all VPD actives) | +0.47 | **FAIL** |

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
| full | **true** | true | 766 | 6.181 | 7.644 | 0.695 | 1.000 | -2.926 | — |
| full | swap 50% to wrong unit | wrong_eq | 766 | 0.084 | 0.121 | 0.695 | 0.018 | -3.993 | 0/1 |
| full | swap 100% to wrong unit | wrong_eq | 766 | 0.024 | 0.036 | 0.695 | 0.010 | -2.267 | 0/1 |
| full | empty | empty | 0 | -0.007 | -0.007 | 0.695 | 0.008 | — | 0/1 |
| full | shuffled | shuffled | 531 | -0.010 | -0.004 | 0.695 | 0.008 | -1.467 | 0/1 |
| full | single node | collapsed | 1 | 0.036 | -0.007 | 0.695 | 0.008 | 11.276 | 0/1 |
| core | **true** | true | 436 | 1.188 | 0.957 | 0.695 | 0.860 | 0.537 | — |
| core | head0 alone (padded) | wrong_eq | 436 | 0.298 | 0.343 | 0.695 | 0.080 | -1.126 | 0/1 |
| core | head1 alone (padded) | wrong_eq | 436 | 0.337 | 0.388 | 0.695 | 0.107 | -0.955 | 0/1 |
| core | random actives #0 | wrong_eq | 436 | -0.236 | -0.235 | 0.695 | 0.002 | -1.414 | 0/1 |
| core | random actives #1 | wrong_eq | 436 | -0.004 | 0.000 | 0.695 | 0.008 | -1.132 | 0/1 |
| core | random actives #2 | wrong_eq | 436 | 0.305 | 0.385 | 0.695 | 0.105 | -1.531 | 0/1 |
| core | swap 50% to wrong unit | wrong_eq | 436 | 0.034 | 0.043 | 0.695 | 0.011 | -1.526 | 0/1 |
| core | swap 100% to wrong unit | wrong_eq | 436 | -0.071 | -0.063 | 0.695 | 0.005 | -3.047 | 0/1 |
| core | empty | empty | 0 | -0.007 | -0.007 | 0.695 | 0.008 | — | 0/1 |
| core | shuffled | shuffled | 309 | -0.104 | -0.107 | 0.695 | 0.004 | 1.358 | 0/1 |
| core | single node | collapsed | 1 | 0.036 | -0.007 | 0.695 | 0.008 | 11.276 | 0/1 |
| core | all VPD actives | collapsed | 766 | 6.181 | 7.644 | 0.695 | 1.000 | -2.926 | 1/1 |
| ref | run_planted true (narrow) | ref | 766 | 6.181 | 7.644 | 0.695 | 1.000 | -2.926 | — |
| ref | run_planted head0 alone (narrow) | ref | 378 | -0.634 | -0.634 | 0.695 | 0.000 | -0.794 | — |
| ref | run_planted head1 alone (narrow) | ref | 388 | 0.190 | 0.204 | 0.695 | 0.032 | -0.884 | — |
| teacher | teacher | teacher | 71 | 0.718 | 0.448 | 0.695 | 0.153 | 3.530 | — |

## Candidate repairs, rescored from the same rows

Gate B checks passed (texts) when R is recomputed from each row's family scores and eps (no new measurement; phi_tau kept as measured):

- **eps_cap**: each family's eps capped at 0.05 (a noisy family cannot push the residual above 1 - s + 0.05)
- **s_clip**: each family's raw score clipped to [0, 1] (the clipped estimators goal_eval keeps beside the raw ones; N's raw specificity ratio reaches -5 on a good circuit)
- **gate_ramp**: g x clip(phi / phi_tau, 0, 1): compactness gets nothing at phi <= 0 (the empty and one-node circuits)
- **phi_cap**: phi capped at phi_tau + 2.0 (past the bar, faithfulness stops paying and compactness ranks)

| variant | full: wrong_eq | full: empty | full: shuffled | full: collapsed | core: wrong_eq | core: empty | core: shuffled | core: collapsed | full: all | core: all |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 10/10 | 10/10 | 10/10 | 10/10 | 7/10 | 9/10 | 8/10 | 0/10 | 10/10 | 0/10 |
| eps_cap | 10/10 | 10/10 | 10/10 | 10/10 | 9/10 | 9/10 | 9/10 | 0/10 | 10/10 | 0/10 |
| s_clip | 10/10 | 10/10 | 10/10 | 10/10 | 7/10 | 9/10 | 8/10 | 0/10 | 10/10 | 0/10 |
| gate_ramp | 10/10 | 10/10 | 10/10 | 10/10 | 7/10 | 9/10 | 8/10 | 0/10 | 10/10 | 0/10 |
| phi_cap | 10/10 | 10/10 | 10/10 | 9/10 | 7/10 | 9/10 | 8/10 | 3/10 | 9/10 | 3/10 |
| eps_cap+s_clip | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 10/10 | 0/10 |
| eps_cap+s_clip+gate_ramp | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0/10 | 10/10 | 0/10 |
| eps_cap+s_clip+phi_cap | 10/10 | 10/10 | 10/10 | 9/10 | 10/10 | 10/10 | 10/10 | 4/10 | 9/10 | 4/10 |
| eps_cap+s_clip+gate_ramp+phi_cap | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 4/10 | 10/10 | 4/10 |

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
