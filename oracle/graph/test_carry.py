"""Real MPS/CUDA/CPU checks against the independent pathpatch prototype.

MPD_MEM_GIB=24 ~/mpd-data/venv/bin/python -B oracle/graph/test_carry.py
"""
import math
import json
from pathlib import Path
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carry
import native


@pytest.fixture(scope="module")
def case():
    nat = native.Native()
    ids, targets = native.text(native.TEXTS / "vpd4l/hard9308.json")
    prompts = carry.alternatives(nat, ids, 322)
    study = carry.Carry(nat, ids, targets, prompts)
    space = carry.Space(nat, len(ids), [322, 332])
    active = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    diag = carry.diagnostics(study)
    print("diagnostics", diag, flush=True)
    return nat, ids, targets, prompts, study, space, active, diag


@torch.no_grad()
def prototype(nat, ids, targets, prompts, nodes):
    """Literal slow prototype; deliberately independent mask construction."""
    rec = nat._model_record(ids)[0]
    scores = []
    for x in prompts:
        patch = {}
        for n in nat.names:
            m = torch.ones(1, len(ids), nat.C[n], device=nat.dev, dtype=torch.bool)
            for nn, pos, c in nodes:
                if nn == n:
                    m[0, pos, c] = False
            patch[n] = (m, rec[n])
        lp = nat.reference([x], targets)
        lq = nat.run(x, targets, nat._full_plan(len(ids), 1), *nat._one(), patch=patch)
        scores.append(float((lp.exp() * (lp - lq)).sum(-1).sum(-1)) / math.log(2))
    return sum(scores) / len(scores)


def test_empty_movement_and_remainder_leak(case):
    *_, diag = case
    # This checks the claimed empty==movement, while making any actual bypass
    # contribution explicit rather than silently absorbing it into a tolerance.
    delta = abs(diag["empty_bits"] - diag["movement_bits"])
    print("empty versus movement discrepancy, bits:", delta)
    assert delta < 1e-4
    assert abs(diag["empty_frozen_remainders_and_embedding_bits"] - diag["native_movement_bits"]) < 1e-5
    assert diag["frozen_vs_native_original_max_logprob_error"] == 0


def test_everything(case):
    *_, diag = case
    assert abs(diag["all_nodes_bits"]) < 1e-5
    assert 0 <= diag["all_nodes_quadratic_kl_bits"] < 1e-9
    assert diag["all_nodes_max_logprob_error"] < 1e-4


def test_batched_equals_single(case):
    nat, ids, targets, prompts, study, space, active, _ = case
    half = active.clone()
    half[::2] = False
    sets = [set(), space.expand(active), space.expand(half)]
    batch = carry.carry_kl(nat, ids, targets, prompts, sets, chunk_size=3)
    single = carry.carry_kl(nat, ids, targets, prompts, sets, chunk_size=1)
    print("batch", batch.tolist(), "single", single.tolist())
    torch.testing.assert_close(batch, single, atol=1e-5, rtol=0)
    assert batch.device == nat.target.wte.device
    chunked = carry.carry_kl(nat, ids, targets, prompts, sets, chunk_size=2)
    torch.testing.assert_close(chunked, single, atol=1e-5, rtol=0)


def test_hard9308_prototype(case):
    nat, ids, targets, prompts, study, space, active, _ = case
    nodes = space.nodes(active)
    expected = [prototype(nat, ids, targets, prompts, s) for s in (set(), nodes)]
    actual = study.measure([set(), space.expand(active)]).tolist()
    print("hard9308 prototype", expected, "carry", actual)
    assert actual == pytest.approx(expected, abs=1e-5)
    assert expected[0] == pytest.approx(8.42, abs=.02)
    assert abs(expected[1]) < .01


def test_mix_endpoints_and_gradient(case):
    nat, ids, targets, prompts, study, space, active, _ = case
    half = active.clone()
    half[::2] = False
    with torch.no_grad():
        for live in (torch.zeros_like(active), half, active):
            coeff = {n: z.float()[None] for n, z in space.expand(live).items()}
            mixed = torch.stack([carry.kl(lp, study.mixed(x, coeff))[0] for x, lp in zip(prompts, study.lp)]).mean()
            hard = study.measure([space.expand(live)])[0]
            torch.testing.assert_close(mixed, hard, atol=1e-5, rtol=0)
    cost = carry.attribution(study, space, half)
    assert torch.isfinite(cost).all()
    for n in nat.names:
        if n.startswith("h.3.") and n.rsplit(".", 1)[1] not in ("k_proj", "v_proj"):
            # Last-layer source writers/queries cannot reach a later target.
            assert torch.count_nonzero(cost[space.slices[n]][:nat.C[n]]) == 0
    j = int(cost.abs().masked_fill(~half, 0).argmax())
    # A strong live derivative is much less vulnerable to float32 cancellation
    # than an arbitrary component at the near-zero full-set endpoint.
    a = half.float()
    step = .01
    with torch.no_grad():
        losses = []
        for sign in (-1, 1):
            coeff = a.clone()
            coeff[j] += sign * step
            expanded = {n: z[None] for n, z in space.expand(coeff).items()}
            losses.append(torch.stack([carry.kl(lp, study.mixed(x, expanded))[0]
                                       for x, lp in zip(prompts, study.lp)]).mean())
    finite = (losses[1] - losses[0]) / (2 * step)
    print("gradient", float(cost[j]), "finite difference", float(finite))
    torch.testing.assert_close(cost[j], finite, atol=2e-3, rtol=.03)


def test_masks_validate_and_index(case):
    nat, ids, _, _, _, space, active, _ = case
    active[:] = True
    active[1::2] = False
    nodes = space.nodes(active)
    actual = carry.masks(nat, len(ids), nodes)
    for n, expected in space.expand(active).items():
        assert torch.equal(actual[n], expected)
    with pytest.raises(ValueError):
        carry.masks(nat, len(ids), {(nat.names[0], len(ids), 0)})
    with pytest.raises(ValueError):
        carry.masks(nat, len(ids), {("unknown", 0, 0)})


@pytest.mark.parametrize("task,source,target", [("hard9302", 4, 465), ("hard9318", 398, 411)])
def test_nonzero_source_target_controls(task, source, target):
    nat = native.Native()
    ids, targets = native.text(native.TEXTS / "vpd4l" / f"{task}.json")
    prompts = carry.alternatives(nat, ids, source)
    space = carry.Space(nat, len(ids), [source, target])
    active = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    expected = prototype(nat, ids, targets, prompts, space.nodes(active))
    actual = float(carry.carry_kl(nat, ids, targets, prompts, [space.expand(active)])[0])
    print(task, "source/target-only prototype", expected, "carry", actual)
    assert actual == pytest.approx(expected, abs=1e-5)
    assert expected > .03
    units = carry.UnitSpace(nat, len(ids), [source, *targets])
    fast = carry.MatrixCarry(nat, ids, targets, prompts, units, chunk_size=3)
    all_units = torch.ones(units.size, dtype=torch.bool, device=nat.dev)
    alternating = all_units.clone()
    alternating[1::2] = False
    rows = [all_units, alternating, ~all_units]
    slow = carry.Carry(nat, ids, targets, prompts, chunk_size=1)
    reference = slow.measure([units.expand(a) for a in rows])
    actual = fast.measure(rows)
    print(task, "sparse versus full", actual.tolist(), reference.tolist())
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=1e-5)


def test_finite_removal_versus_local_derivative(case):
    nat, ids, targets, prompts, study, space, _, _ = case
    active = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    grad = carry.attribution(study, space, active)
    j = int(grad.argmax())
    assert grad[j] > 0
    values = []
    with torch.no_grad():
        for level in (1., .999, .99, 0.):
            a = active.float()
            a[j] = level
            expanded = {n: z[None] for n, z in space.expand(a).items()}
            values.append(float(torch.stack([carry.kl(lp, study.mixed(x, expanded))[0]
                                             for x, lp in zip(prompts, study.lp)]).mean()))
    only = torch.zeros_like(active)
    only[j] = True
    print("signed-rank first removal", space.nodes(only), "gradient", float(grad[j]),
          "KL at a=1,.999,.99,0", values)
    # Verify the full ablation, whose direction may differ from local influence.
    active[j] = False
    assert values[-1] == pytest.approx(float(study.measure([space.expand(active)])[0]), abs=1e-5)


def test_prototype_explanations_and_vpd(case):
    nat, ids, targets, prompts, study, _, _, _ = case
    task_path = native.TEXTS / "vpd4l/hard9308.json"
    behavior = {**json.loads(task_path.read_text()), "path": str(task_path)}
    paths = {
        "reads": Path("/private/tmp/claude-501/-Users-user/b1063ff3-6750-418e-8406-23ae82bf968a/scratchpad/lead/vr4_hard/hard9308.py"),
        "vpdnode": Path.home() / "mpd-data/runpod/oracle-graph-search-vpdn1/search_hard/hard9308.py",
    }
    sets = {}
    for label, path in paths.items():
        if path.exists():
            # Trusted local reference answers; vpd4l's tokenizer is a local file.
            ir = native.mech.trace_inline(path.read_text(), "vpd4l", behavior=behavior)
            assert ir["valid"], ir.get("error")
            sets[label] = native.prefix(ir, ir["graph"]["steps"])[0].node_set()
    steps = nat.vpd_steps(ids, targets)
    for k in (6, 9, 12):
        assert k < len(steps)
        sets[f"vpd_top{2**k}"] = steps[k].node_set()
    expected = [prototype(nat, ids, targets, prompts, s) for s in sets.values()]
    actual = study.measure(list(sets.values())).tolist()
    print("prototype explanations and VPD", dict(zip(sets, actual)),
          "sizes", {n: len(s) for n, s in sets.items()})
    assert actual == pytest.approx(expected, abs=1e-5)
    # They round to the empty result at two decimals, but do not equal it at
    # full precision. The literal prototype above is the correctness check.
    assert actual[list(sets).index("vpd_top4096")] == pytest.approx(8.420146, abs=1e-5)


def test_oom_chunk_backoff(case, monkeypatch):
    nat, ids, targets, prompts, _, space, active, _ = case
    study = carry.Carry(nat, ids, targets, prompts, chunk_size=3)
    selected = space.expand(active)
    sets = [set(), selected, selected]
    expected = carry.Carry(nat, ids, targets, prompts, chunk_size=1).measure(sets)
    original_run = nat.run
    failed = []

    def constrained_run(x, targets, plan, *args, **kwargs):
        if plan["B"] > 1:
            failed.append(plan["B"])
            raise RuntimeError("out of memory: simulated batch limit")
        return original_run(x, targets, plan, *args, **kwargs)

    monkeypatch.setattr(nat, "run", constrained_run)
    actual = study.measure(sets)
    assert failed == [3]
    assert study.chunk_size == 1
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_unit_space_and_head_endpoints(case):
    nat, ids, targets, prompts, study, _, _, _ = case
    matrix = carry.UnitSpace(nat, len(ids), [322, *targets])
    full = torch.ones(matrix.size, dtype=torch.bool, device=nat.dev)
    assert matrix.size == len(nat.names) * len(set([322, *targets]))
    expected = {(n, p, c) for n, p in matrix.units for c in range(nat.C[n])}
    for n, mask in carry.masks(nat, len(ids), expected).items():
        assert torch.equal(mask, matrix.expand(full)[n])
    heads = carry.UnitSpace(nat, len(ids), [322, *targets], "head")
    hs = carry.HeadCarry(nat, ids, targets, prompts, heads, chunk_size=2)
    rows = [torch.tensor([((u[0], u[-1]) in chosen) for u in heads.units], device=nat.dev)
            for chosen in (set(), set(matrix.units), {("h.0.mlp.c_fc", 322), ("h.0.mlp.down_proj", 322),
                                                    ("h.2.attn.v_proj", 322), ("h.2.attn.o_proj", 332)})]
    ms = [matrix.expand(torch.tensor([u in chosen for u in matrix.units], device=nat.dev))
          for chosen in (set(), set(matrix.units), {("h.0.mlp.c_fc", 322), ("h.0.mlp.down_proj", 322),
                                                   ("h.2.attn.v_proj", 322), ("h.2.attn.o_proj", 332)})]
    actual, reference = hs.measure(rows), study.measure(ms)
    print("head versus matrix endpoints/4-unit", actual.tolist(), reference.tolist())
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=1e-5)
    fast = carry.MatrixCarry(nat, ids, targets, prompts, matrix, chunk_size=3)
    matrix_rows = [torch.tensor([u in chosen for u in matrix.units], device=nat.dev)
                   for chosen in (set(), set(matrix.units), {("h.0.mlp.c_fc", 322), ("h.0.mlp.down_proj", 322),
                                                            ("h.2.attn.v_proj", 322), ("h.2.attn.o_proj", 332)})]
    fast_scores = fast.measure(matrix_rows)
    print("prompt-batched matrix versus separate prompts", fast_scores.tolist(), reference.tolist())
    torch.testing.assert_close(fast_scores, reference, atol=3e-5, rtol=1e-5)
    sparse_grad = carry._unit_attribution(fast, matrix, matrix_rows[-1])
    dense_grad = carry.attribution(study, matrix, matrix_rows[-1])
    torch.testing.assert_close(sparse_grad, dense_grad, atol=3e-4, rtol=.002)
    singles = carry.HeadCarry(nat, ids, targets, prompts, heads, chunk_size=1).measure(rows)
    torch.testing.assert_close(actual, singles, atol=2e-5, rtol=1e-5)
    # All published attention write/read vectors span all heads. Exact zeros,
    # rather than an arbitrary energy threshold, define whole-head support.
    for n in nat.names:
        if ".attn." in n:
            st = nat.target.site(n)
            w = st.V.T if n.endswith("o_proj") else st.U
            assert (w.reshape(w.shape[0], nat.target.n_head, nat.target.hd).abs().amax(-1) > 0).all()


@pytest.mark.parametrize("head", [0, 3])
def test_head_slice_literal_and_gradient(case, head):
    nat, ids, targets, prompts, _, _, _, _ = case
    space = carry.UnitSpace(nat, len(ids), [322, *targets], "head")
    hs = carry.HeadCarry(nat, ids, targets, prompts, space)
    active = torch.tensor([u[0] in ("h.0.mlp.c_fc", "h.0.mlp.down_proj") or
                           (u[0] in ("h.2.attn.v_proj", "h.2.attn.o_proj") and u[1] == head)
                           for u in space.units], device=nat.dev)
    grad = carry._unit_attribution(hs, space, active)
    assert torch.isfinite(grad).all()
    j = int(grad.abs().argmax())
    eps = .01
    losses = []
    with torch.no_grad():
        for delta in (-eps, eps):
            a = active.float()
            a[j] += delta
            losses.append(torch.stack([carry.kl(lp, hs.mixed(x, space.coefficients(a)))[0]
                                       for x, lp in zip(prompts, hs.lp)]).mean())
    torch.testing.assert_close(grad[j], (losses[1] - losses[0]) / (2 * eps), atol=3e-3, rtol=.04)
    # Independent literal o-head calculation for one prompt using model's
    # original attention input, plus explicitly sliced q/k/v writes.
    class Literal(carry.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func.__name__ in ("matmul", "__matmul__"):
                for n in nat.names:
                    st = nat.target.site(n)
                    if ".attn." not in n:
                        continue
                    if n.endswith("o_proj") and args[1] is st.V:
                        result = torch.zeros_like(hs.rec[n]).expand(args[0].shape[0], -1, -1).clone()
                        for h in range(nat.target.n_head):
                            sl = slice(h * nat.target.hd, (h + 1) * nat.target.hd)
                            live = args[0][..., sl] @ st.V[sl]
                            base = hs.originals[n][..., h, :]
                            use = torch.zeros(len(ids), dtype=torch.bool, device=nat.dev)
                            for p in space.positions:
                                use[p] = active[space.index[(n, h, p)]]
                            result += torch.where(use[None, :, None], live, base)
                        return result
                    if not n.endswith("o_proj") and args[1] is st.U:
                        live = func(*args, **kwargs)
                        result = (hs.rec[n] @ st.U).clone()
                        for h in range(nat.target.n_head):
                            sl = slice(h * nat.target.hd, (h + 1) * nat.target.hd)
                            for p in space.positions:
                                if bool(active[space.index[(n, h, p)]]):
                                    result[:, p, sl] = live[:, p, sl]
                        return result
            return func(*args, **kwargs)
    patch = {}
    for n in nat.names:
        if ".mlp." in n:
            mask = torch.ones_like(hs.rec[n], dtype=torch.bool)
            for p in space.positions:
                if bool(active[space.index[(n, p)]]):
                    mask[:, p] = False
            patch[n] = (mask, hs.rec[n])
    with torch.no_grad(), Literal():
        literal = nat.run(prompts[0], targets, nat._full_plan(len(ids), 1), *nat._one(), patch=patch)
    with torch.no_grad():
        actual = hs.mixed(prompts[0], space.coefficients(active.float()))
    delta = actual - literal
    weighted_squared = float((literal.exp() * delta.square()).sum())
    print("head sparse/literal max logprob error", float(delta.abs().max()),
          "weighted squared error", weighted_squared,
          "carry KL discrepancy", float((carry.kl(hs.lp[0], actual) - carry.kl(hs.lp[0], literal)).abs().max()))
    assert weighted_squared < 1e-9
    torch.testing.assert_close(carry.kl(hs.lp[0], actual), carry.kl(hs.lp[0], literal), atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(actual, literal, atol=3e-5, rtol=0)


def test_measured_unit_elimination(monkeypatch):
    # Nonlinear score: the gradient ordering deliberately ranks the wrong unit.
    # Compare EVERY committed removal to exhaustive enumeration on the host.
    class Nat:
        dev = "cpu"
    class Space:
        size = 5
        def expand(self, a):
            return a
        def nodes(self, a):
            return set(a.nonzero().flatten().tolist())
    class Study:
        nat = Nat()
        def __init__(self):
            self.calls = []
        def measure(self, sets):
            self.calls.append(len(sets))
            return torch.tensor([sum((i + 1)**2 for i in range(5) if not a[i]) +
                                 (20 if not a[0] and not a[1] else 0) for a in sets], dtype=torch.float32)
    monkeypatch.setattr(carry, "_unit_attribution", lambda *args: torch.arange(5).flip(0))
    study = Study()
    curve = carry._position_circuit(study, Space())
    assert study.calls == [1, 5, 4, 3, 2]
    assert [n for n, _ in curve] == [5, 4, 3, 2, 1]
    for i in range(1, len(curve)):
        previous = curve.active[i-1]
        trials = []
        for j in previous.nonzero().flatten().tolist():
            a = previous.clone()
            a[j] = False
            trials.append(a)
        assert curve[i][1] == float(study.measure(trials).min())
        assert len(curve.unit_set(i-1) - curve.unit_set(i)) == 1


def test_threshold_nonmonotone_and_floor():
    class Space:
        def nodes(self, a):
            return set(a.nonzero().flatten().tolist())
    curve = carry.Curve(Space())
    for count, score in [(4, .2), (3, 2.), (2, .05), (1, 3.)]:
        curve.record(torch.arange(4) < count, score)
    assert carry._threshold(curve, 10, .9)["size"] == 2
    assert carry._threshold(curve, 10, .99)["size"] == 2
    assert carry._threshold(curve, .000001, .9) is None
    assert carry._threshold(curve, .01, .9) is None


def test_explain_selection_excludes_targets(monkeypatch):
    class Nat:
        def sensitivity(self, ids, targets, generator):
            assert generator.initial_seed() == native.task_seed(str(ids[:8]))
            return torch.tensor([2., 8., 1., 9., 3.]), torch.empty(0)
    def capture(nat, ids, targets, order, units, moved):
        return order
    monkeypatch.setattr(carry, "_explain_positions", capture)
    assert carry.explain(Nat(), [1, 2, 3, 4, 5, 6], [2, 5], 3) == [4, 1, 3]
    assert carry.explain(Nat(), [1], [0], 8) == []
    with pytest.raises(ValueError):
        carry.explain(Nat(), [1, 2], [1], 0)


def test_alternative_tokens(case):
    nat, ids, _, prompts, *_ = case
    with torch.no_grad():
        prediction = nat.reference([ids], [321])[0, 0]
    expected = [t for t in prediction.topk(4).indices.tolist() if t != ids[322]][:3]
    assert [x[322] for x in prompts] == expected
    assert len(set(expected)) == 3
    assert all(x[:322] == ids[:322] and x[323:] == ids[323:] for x in prompts)


def test_prompt_batch_embeddings(case):
    nat, ids, targets, prompts, *_ = case
    space = carry.UnitSpace(nat, len(ids), [322, *targets])
    study = carry.MatrixCarry(nat, ids, targets, prompts, space)
    with torch.no_grad():
        mode = carry._PromptBatch(study, 1)
        with mode:
            actual = nat.run(prompts[0], targets, nat._full_plan(len(ids), len(prompts)), *nat._one())
        expected = nat.reference(prompts, targets)
    assert mode.seen
    error = actual - expected
    print("batched embedding versus independent Target batch max error", float(error.abs().max()))
    assert float((expected.exp() * error.square()).sum()) < 1e-9
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


def test_cli_resume_without_model(monkeypatch):
    out = Path.home() / "mpd-data/graph_oracle/carry"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", dir=out) as f:
        f.write(json.dumps({"task": "hard9300", "split": "hard", "k": 8, "units": "matrix"}) + "\n")
        f.flush()
        monkeypatch.setattr(sys, "argv", ["carry.py", "explain", "--split", "hard", "--n", "1", "--k", "8", "--out", f.name])
        monkeypatch.setattr(native, "tasks", lambda split: [Path("hard9300.json")])
        def forbidden_model():
            raise AssertionError("resuming finished questions must not load the model")
        monkeypatch.setattr(native, "Native", forbidden_model)
        carry.main()
        assert len(Path(f.name).read_text().splitlines()) == 1


def test_window_rms_against_padded_reference(case):
    nat, ids, targets, prompts, *_ = case
    space = carry.UnitSpace(nat, len(ids), [322, *targets])
    study = carry.MatrixCarry(nat, ids, targets, prompts, space, chunk_size=3)
    full = torch.ones(space.size, dtype=torch.bool, device=nat.dev)
    half = full.clone()
    half[1::2] = False
    sets = [full, half, ~full]
    study.pad_norm = True
    padded = study.measure(sets)
    padded_gradient = carry._unit_attribution(study, space, half)
    study.pad_norm = False
    window = study.measure(sets)
    window_gradient = carry._unit_attribution(study, space, half)
    print("window versus padded RMS KL", window.tolist(), padded.tolist())
    torch.testing.assert_close(window, padded, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(window_gradient, padded_gradient, atol=3e-4, rtol=.002)


def test_relay_screen_full_chunked_and_finite_difference(case):
    nat, ids, targets, prompts, dense, *_ = case
    p = 322
    space = carry.UnitSpace(nat, len(ids), [p, *targets])
    study = carry.MatrixCarry(nat, ids, targets, prompts, space)
    full = carry._relay_attribution(study, p, targets)
    chunks = carry._relay_attribution(study, p, targets, chunk_size=2)
    assert [q for q, _ in full] == [q for q in range(p+1, max(targets)) if q not in targets]
    torch.testing.assert_close(torch.tensor([v for _, v in full]), torch.tensor([v for _, v in chunks]),
                               atol=3e-4, rtol=.002)
    view = carry._unit_view(study, [p, *targets, *[q for q, _ in full]])
    active = carry._position_mask(view.space, {p, *targets})
    grad = carry._unit_attribution(view, view.space, active)
    all_live_grad = carry._unit_attribution(view, view.space, torch.ones_like(active))
    print("relay-screen sum absolute gradient: p/targets endpoint", float(grad.abs().sum()),
          "all-live endpoint", float(all_live_grad.abs().sum()), flush=True)
    # Check an omitted relay coefficient, rather than a live p/target unit.
    relay = torch.tensor([u[-1] not in {p, *targets} for u in view.space.units], device=nat.dev)
    j = int(grad.abs().masked_fill(~relay, -1).argmax())
    losses = []
    eps = .01
    with torch.no_grad():
        for sign in (-1, 1):
            a = active.float()
            a[j] += sign * eps
            losses.append(torch.stack([carry.kl(lp, view.mixed(x, view.space.coefficients(a)))[0]
                                       for x, lp in zip(prompts, view.lp)]).mean())
    finite = (losses[1]-losses[0])/(2*eps)
    print("omitted-relay unit finite difference", view.space.units[j], float(grad[j]), float(finite), flush=True)
    torch.testing.assert_close(grad[j], finite, atol=2e-3, rtol=.04)
    chosen = {p, *targets, full[0][0]}
    mask = carry._position_mask(view.space, chosen)
    torch.testing.assert_close(view.measure([mask]), dense.measure([view.space.expand(mask)]), atol=3e-5, rtol=1e-5)


def test_relay_attribution_oom_never_skips(monkeypatch):
    from types import SimpleNamespace
    nat = SimpleNamespace(dev="cpu", names=["a", "b"], target=SimpleNamespace(n_head=1))
    study = SimpleNamespace(nat=nat, T=24)
    seen = []
    def view(study, positions):
        return SimpleNamespace(space=carry.UnitSpace(nat, study.T, positions))
    def grad(study, space, active):
        seen.append(space.positions)
        assert all(bool(a) == (u[-1] in {1, 8, 23}) for a, u in zip(active, space.units))
        if len(space.positions) > 7:
            raise RuntimeError("out of memory: simulated gradient limit")
        return torch.tensor([float(u[-1]) for u in space.units])
    monkeypatch.setattr(carry, "_unit_view", view)
    monkeypatch.setattr(carry, "_unit_attribution", grad)
    values = carry._relay_attribution(study, 1, [8, 23])
    assert values == [(q, 2.*q) for q in range(2, 23) if q != 8]
    assert len(seen) > 2


def test_head_relay_views(case):
    nat, ids, targets, prompts, dense, *_ = case
    base = {322, *targets}
    study = carry.HeadCarry(nat, ids, targets, prompts,
                            carry.UnitSpace(nat, len(ids), base, "head"))
    view = carry._unit_view(study, [*base, 323, 325])
    active = carry._position_mask(view.space, base)
    full_grad = carry._unit_attribution(view, view.space, active)
    chunk = carry._unit_view(study, [*base, 323])
    chunk_grad = carry._unit_attribution(chunk, chunk.space, carry._position_mask(chunk.space, base))
    indices = torch.tensor([view.space.index[u] for u in chunk.space.units], device=nat.dev)
    torch.testing.assert_close(chunk_grad, full_grad[indices], atol=3e-4, rtol=.002)
    rows = [carry._position_mask(view.space, ps) for ps in (base, base | {323}, base | {323, 325})]
    matrix = carry.UnitSpace(nat, len(ids), view.space.positions)
    reference = dense.measure([matrix.expand(carry._position_mask(matrix, ps))
                               for ps in (base, base | {323}, base | {323, 325})])
    torch.testing.assert_close(view.measure(rows), reference, atol=3e-5, rtol=1e-5)


def test_measured_relays_and_stopping(monkeypatch):
    from types import SimpleNamespace
    nat = SimpleNamespace(dev="cpu", names=["a"], target=SimpleNamespace(n_head=1))
    calls = []
    class Study:
        T = 30
        def measure(self, sets):
            scores = []
            for a in sets:
                qs = {u[-1] for u, live in zip(self.space.units, a) if live} - {1, 29}
                assert {1, 29}.issubset({u[-1] for u, live in zip(self.space.units, a) if live})
                calls.append(qs)
                # Attribution ranks 2 highest, measured best is 3, then 4.
                scores.append(8. - (4. if 3 in qs else 0) - (3.5 if 4 in qs else 0) - (.05 if 2 in qs else 0))
            return torch.tensor(scores)
    Study.nat = nat
    study = Study()
    def view(study, positions):
        v = Study()
        v.space = carry.UnitSpace(nat, study.T, positions)
        return v
    monkeypatch.setattr(carry, "_unit_view", view)
    monkeypatch.setattr(carry, "_relay_attribution", lambda *args, **kwargs: [(q, 30-q) for q in range(2, 29)])
    found = carry.relays(study, 1, [29], movement=10., baseline=8.)
    assert found == [(3, 4.), (4, .5)]
    assert len(calls) == 16 + 15
    assert max(q for s in calls for q in s) == 17  # only the top 16 screen candidates
    calls.clear()
    assert carry.relays(study, 1, [29], max_relays=1, movement=10., baseline=8.) == [(3, 4.)]
    # No candidate improves by 1% after 3; stop without accepting 2.
    original = Study.measure
    def weak(self, sets):
        return original(self, sets) + torch.tensor([3.5 if any(u[-1] == 4 and a[j] for j, u in enumerate(self.space.units)) else 0 for a in sets])
    monkeypatch.setattr(Study, "measure", weak)
    assert carry.relays(study, 1, [29], movement=10., baseline=8.) == [(3, 4.)]
    assert carry.relays(study, 1, [29], movement=10., baseline=.9) == []
    assert carry.relays(study, 1, [29], movement=1e-6, baseline=8.) == []
    assert carry.relays(study, 1, [29], max_relays=0, movement=10., baseline=8.) == []


def test_relay_curve_positions_and_explain_forwarding(monkeypatch):
    from types import SimpleNamespace
    nat = SimpleNamespace(dev="cpu", names=["a"], target=SimpleNamespace(n_head=1))
    study = SimpleNamespace(nat=nat, space=carry.UnitSpace(nat, 8, [1, 7]),
                            measure=lambda sets: torch.tensor([5.]))
    monkeypatch.setattr(carry, "_find_relays", lambda *args, **kwargs: [(4, .2)])
    monkeypatch.setattr(carry, "_unit_view", lambda s, ps: SimpleNamespace(space=carry.UnitSpace(nat, 8, ps)))
    def eliminate(study, space):
        curve = carry.Curve(space)
        curve.record(torch.ones(space.size, dtype=torch.bool), .2)
        curve.record(torch.tensor([True, False, True]), .3)
        return curve
    monkeypatch.setattr(carry, "_position_circuit", eliminate)
    curve = carry._circuit_with_relays(study, 1, [7], True, 10.)
    assert curve.relays == [(4, .2)]
    assert curve.before_relays_bits == 5.
    assert curve.position_sets == [[1, 4, 7], [1, 7]]
    assert carry._threshold(curve, 10., .9)["positions"] == [1, 7]
    assert carry._unit_names([("h.1.attn.o_proj", 4)], 1, [7]) == ("L1.o@r+3",)


def test_hard9300_relays_measured_against_dense():
    nat = native.Native()
    ids, targets = native.text(native.TEXTS / "vpd4l/hard9300.json")
    p = 337
    prompts = carry.alternatives(nat, ids, p)
    dense = carry.Carry(nat, ids, targets, prompts, chunk_size=1)
    base = {p, *targets}
    initial = carry.UnitSpace(nat, len(ids), base)
    study = carry.MatrixCarry(nat, ids, targets, prompts, initial)
    movement = carry._movement(study)
    before = float(study.measure([carry._position_mask(initial, base)])[0])
    assert before > .1 * movement
    found = carry.relays(study, p, targets, movement=movement, baseline=before)
    assert found
    print("hard9300", p, "movement", movement, "before", before, "relays", found, flush=True)
    current = before
    for q, score in found:
        assert p < q < max(targets) and q not in base
        assert current - score >= .01 * movement
        base.add(q)
        space = carry.UnitSpace(nat, len(ids), base)
        active = carry._position_mask(space, base)
        actual = float(dense.measure([space.expand(active)])[0])
        assert actual == pytest.approx(score, abs=3e-5, rel=1e-5)
        current = score
    assert current <= .1 * movement


def test_cli_relay_resume_rejects_mixed_configuration(monkeypatch):
    out = Path.home() / "mpd-data/graph_oracle/carry"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", dir=out) as f:
        f.write(json.dumps({"task": "hard9300", "split": "hard", "k": 8, "units": "matrix"}) + "\n")
        f.flush()
        monkeypatch.setattr(sys, "argv", ["carry.py", "explain", "--split", "hard", "--n", "1", "--relays", "--out", f.name])
        with pytest.raises(SystemExit) as exc:
            carry.main()
        assert exc.value.code == 2


def test_public_relay_options(monkeypatch):
    from types import SimpleNamespace
    nat = SimpleNamespace(dev="cpu", names=["a"], target=SimpleNamespace(n_head=1))
    monkeypatch.setattr(carry, "MatrixCarry", lambda nat, ids, targets, prompts, space: SimpleNamespace(space=space))
    monkeypatch.setattr(carry, "_movement", lambda study: 10.)
    monkeypatch.setattr(carry, "_circuit_with_relays", lambda study, p, targets, enabled, movement: (enabled, movement))
    assert carry.position_circuit(nat, [1, 2, 3], [2], 1, [[1, 4, 3]], relays=True) == (True, 10.)
    assert carry.position_circuit(nat, [1, 2, 3], [2], 1, [[1, 4, 3]]) == (False, 10.)
    nat.sensitivity = lambda *args: (torch.tensor([2., 3.]), torch.empty(0))
    monkeypatch.setattr(carry, "_explain_positions", lambda *args, **kwargs: kwargs)
    assert carry.explain(nat, [1, 2, 3], [2], 1, relays=True) == {"relays": True}


if __name__ == "__main__":
    # The local memory guard recognizes script paths, but reserves only 2 GiB
    # for `python -m pytest` regardless of MPD_MEM_GIB.
    raise SystemExit(pytest.main([__file__, "-s", "-p", "no:cacheprovider", *sys.argv[1:]]))
