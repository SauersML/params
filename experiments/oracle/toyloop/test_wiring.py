"""Wiring must change execution even when the selected units stay fixed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "method/oracle"), str(ROOT / "method/vpd")]

import circuit
import reward_exec
import paired_wiring as paired
from test_goal_eval import tiny_native


@pytest.fixture
def pair():
    # Two exact rank-one MLP neurons. No matrix decomposition or model download.
    torch.set_num_threads(1)
    nat = tiny_native(seed=8, layers=2, d=8, heads=1, vocab=12, C=1)
    nat.units = {}
    with torch.no_grad():
        # Guaranteed live path: input e0 -> neuron0 writes e1 -> neuron1 writes e2.
        # Random test weights can leave both GELUs inactive and make the removed edge inert.
        nat.target.wte.zero_()
        nat.target.wte[1:7, 0] = 1
        nat.target.wte[7, 1] = 1
        nat.target.wte[8, 2] = 1
        for norm in nat.target.norms:
            norm.fill_(1)
        nat.target.ln_f.fill_(1)
        for n in nat.names:
            st = nat.target.site(n)
            layer = int(n.split(".")[1])
            if ".attn." in n:
                st.U.zero_()
                st.V.zero_()
                nat.units[n] = torch.tensor([-1])
            elif n.endswith("c_fc"):
                st.U.zero_()
                st.U[0, 0] = 1
                st.V.zero_()
                st.V[layer, 0] = 1
                nat.units[n] = torch.tensor([0])
            else:
                st.V.zero_()
                st.V[0, 0] = 1
                st.U.zero_()
                st.U[0, layer + 1] = 1 + layer
                nat.units[n] = torch.tensor([0])
            st.W.copy_((st.V @ st.U).T)
            st.delta_T.zero_()
    dec = {n: (nat.target.site(n).U, nat.target.site(n).V, nat.units[n]) for n in nat.names}
    toy = SimpleNamespace(cfg={"n_layer": 2, "n_head": 1},
                          sd={f"{n}.weight": nat.target.site(n).W for n in nat.names},
                          texts=[[1, 2, 3], [4, 5, 6]], meta={"decomposition": dec})
    return toy, nat


FULL = "flowchart LR\nL0N0 --> L1N0\nL0N0 --> out\nL1N0 --> out\nEnglish: unverified"
CUT = "flowchart LR\nL0N0 --> out\nL1N0 --> out\nEnglish: unverified"


def test_same_units_different_wiring_changes_execution(pair):
    toy, nat = pair
    assert circuit.parse(FULL).units == circuit.parse(CUT).units
    full = reward_exec.score_wiring(toy, FULL, nat=nat)
    cut = reward_exec.score_wiring(toy, CUT, nat=nat)
    assert full["kl_bits"] < 1e-5, full
    assert cut["kl_bits"] > full["kl_bits"] + 1e-4, (full, cut)
    assert all(r["top"] == r["reference_top"] == 8 for r in full["per_text"])
    assert all(r["top"] == 7 for r in cut["per_text"])
    assert full["per_text"][0]["nodes"] == cut["per_text"][0]["nodes"]
    assert full["per_text"][0]["edges"] > cut["per_text"][0]["edges"]


def test_training_wiring_reward_executes_edges_and_rejects_partial_invalid(pair, monkeypatch):
    import planted
    import train

    toy, nat = pair
    monkeypatch.setattr(planted, "native_of", lambda sd, cfg: nat)
    # Collision with legacy name cache must not select another model's decomposition.
    toy.name = "same-name"
    monkeypatch.setitem(reward_exec._CACHE, toy.name, {"nat": object()})
    reward = train.get_reward("wiring_suff")
    assert reward is reward_exec.reward_wiring_suff
    assert train.get_reward("exec") is reward_exec.reward_exec
    full = reward({"toy": toy}, FULL)
    cut = reward({"toy": toy}, CUT)
    assert full > .999 and 0 <= cut < full
    unit_reward = train.get_reward("unit_suff")
    assert unit_reward is reward_exec.reward_unit_suff
    assert unit_reward({"toy": toy}, CUT) == pytest.approx(unit_reward({"toy": toy}, FULL), abs=1e-7)
    assert unit_reward({"toy": toy}, CUT) > cut
    for answer in ("unparseable", FULL.replace("English:", "L0N999 --> out\nEnglish:"),
                   FULL.replace("English:", "unknown --> out\nEnglish:")):
        assert reward({"toy": toy}, answer) == -1 < cut
        assert unit_reward({"toy": toy}, answer) == -1


def test_training_wiring_reward_penalizes_one_failed_text(monkeypatch):
    toy = SimpleNamespace(texts=[[1], [2]], meta={})
    calls = []
    def measured(got_toy, completion, *, texts, complete):
        calls.append((got_toy, texts))
        return {"per_text": [{"suff": 1.0}, {"suff": 0.0}]}
    monkeypatch.setattr(reward_exec, "score_wiring", measured)
    assert reward_exec.reward_wiring_suff({"toy": toy}, FULL) == .25
    assert calls == [(toy, toy.texts)]


def test_training_seed_controls_initialization_and_sampled_completions(monkeypatch):
    import train

    initialized, sampled = [], []
    def load_policy(args):
        model = torch.nn.Linear(1, 1, bias=False)
        initialized.append(model.weight.detach().clone())
        return model, None
    def sample(model, tok, prompt, k, args):
        result = [str(float(torch.rand(()))) for _ in range(k)]
        sampled.append(result)
        return result
    monkeypatch.setattr(train, "examples", lambda *a: [{"prompt": "x", "completion": "y", "meta": {}}])
    monkeypatch.setattr(train, "load_policy", load_policy)
    monkeypatch.setattr(train, "sample", sample)
    monkeypatch.setattr(train, "get_reward", lambda spec: lambda meta, text: float(text))
    monkeypatch.setattr(train, "batch_ids", lambda tok, pairs, cap, device=None: (torch.zeros(len(pairs), 1, device=device), None, torch.ones(len(pairs), 1, device=device)))
    monkeypatch.setattr(train, "token_logps", lambda model, ids, att, comp: (model.weight.expand(len(ids), 1), comp))
    monkeypatch.setattr(train, "save", lambda *a: None)
    args = SimpleNamespace(data="mock", truth=None, seed=11, steps=1, batch=1, lr=.001,
                           max_len=4, log_every=1, eval=0, out="unused", reward="mock", beta=0, k=2, std=False)
    train.sft(args); train.sft(args)
    assert torch.equal(initialized[0], initialized[1])
    train.rl(args); train.rl(args)
    assert torch.equal(initialized[2], initialized[3]) and sampled[0] == sampled[1]
    args.seed = 12
    train.rl(args)
    assert sampled[2] != sampled[0]


def test_wrong_direction_and_unknown_unit_rejected(pair):
    toy, _ = pair
    for text in ("flowchart LR\nL1N0 --> L0N0\nL0N0 --> out",
                 "flowchart LR\nL3N0 --> out", "flowchart LR\nL0N999 --> out"):
        with pytest.raises(ValueError):
            circuit.unit_graph(circuit.parse(text), toy)


def test_token_roles_are_explicitly_annotations(pair):
    toy, _ = pair
    full = circuit.unit_graph(circuit.parse(FULL), toy)
    labelled = circuit.unit_graph(circuit.parse("tokROLE --> L0N0\n" + FULL), toy)
    assert full.nodes == labelled.nodes
    assert full.parents == labelled.parents and full.out == labelled.out
    assert not full.complete


def test_all_pieces_do_not_depend_on_activation_or_truth(pair):
    toy, _ = pair
    toy.truth = {"units": []}
    toy.meta["record"] = {"gt_nodes": []}
    g = circuit.unit_graph(circuit.parse(FULL), toy)
    assert g.count() == 2 * 2 * len(toy.texts[0])  # input/output pieces for both neurons
    with pytest.raises(ValueError):
        circuit.unit_graph(circuit.parse(FULL), toy, [])


@pytest.mark.parametrize('wiring', [False, True])
def test_swapped_completion_executes_each_models_own_basis_and_texts(monkeypatch, wiring):
    import evaluate

    toys = [SimpleNamespace(name=name, meta={'family': 'mock'}, texts=[[i]],
                           truth={'units': ['L0N0'], 'edges': [('L0N0', 'out')]})
            for i, name in enumerate(['own', 'other'], 1)]
    natives = {toy.name: SimpleNamespace(owner=toy.name) for toy in toys}
    monkeypatch.setattr(evaluate.render, 'render', lambda toy, **kw: toy.name)
    monkeypatch.setattr(reward_exec, 'score', lambda *a: {'reward': .5, 'suff': .6, 'nec': .4})
    monkeypatch.setattr(reward_exec, '_state', lambda toy: {'nat': natives[toy.name]})
    monkeypatch.setattr(reward_exec, 'texts_of', lambda toy: toy.texts)
    executed = []

    def score_wiring(toy, completion, *, nat, texts):
        assert nat is natives[toy.name] and texts is toy.texts
        executed.append((toy.name, tuple(map(tuple, texts))))
        return {'suff': .7 if toy.name == 'own' else .2, 'kl_bits': 1. if toy.name == 'own' else 4.,
                'scope': 'point wiring only; embeddings on; token labels unverified'}

    monkeypatch.setattr(reward_exec, 'score_wiring', score_wiring)
    result = evaluate.run('mock', lambda *a: FULL, {'test': toys}, swapped=True,
                          wiring=wiring, log=lambda *a: None)
    first = result['rows'][0]
    if wiring:
        assert len(executed) == 6  # normal + own/donor swap checks for each toy
        assert first['swap_wiring_kl_bits_own'] == 1.
        assert first['swap_wiring_kl_bits_donor'] == 4.
        assert first['swap_wiring_valid_own'] == first['swap_wiring_valid_donor'] == 1.
        assert 'token labels unverified' in first['swap_wiring_scope_donor']
        assert 'swap_wiring_kl_bits_donor' in evaluate.markdown(result)
    else:
        assert executed == []
        assert not any('wiring' in key for key in first)


def test_mixed_invalid_wiring_keeps_kl_conditional_without_zero_filling():
    import evaluate

    summary = evaluate.mean_rows([
        {'wiring_valid': 1., 'wiring_suff': .8, 'wiring_kl_bits': 2., 'swap_wiring_kl_bits_own': 3.},
        {'wiring_valid': 0., 'wiring_suff': 0., 'wiring_kl_bits': None, 'swap_wiring_kl_bits_own': None},
    ])
    assert summary['n'] == 2
    assert summary['wiring_valid'] == .5 and summary['wiring_suff'] == .4
    assert summary['wiring_kl_bits'] == 2.  # conditional on measured KL, not 1 from invalid->0
    assert summary['swap_wiring_kl_bits_own'] == 3.


def test_mac_decomposition_uses_scipy_and_preserves_head_weights(monkeypatch):
    import planted

    cfg = {"n_layer": 1, "n_embd": 4, "n_head": 2}
    rng = torch.Generator().manual_seed(24)
    sd = {f"h.0.attn.{kind}.weight": torch.randn(4, 4, generator=rng)
          for kind in ("q_proj", "k_proj", "v_proj", "o_proj")}
    sd["h.0.attn.k_proj.weight"].zero_()
    sd["h.0.mlp.c_fc.weight"] = torch.randn(8, 4, generator=rng)
    sd["h.0.mlp.down_proj.weight"] = torch.randn(4, 8, generator=rng)
    monkeypatch.setattr(planted, "sys", SimpleNamespace(platform="darwin"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Mac decomposition must not enter torch.linalg.svd")

    monkeypatch.setattr(torch.linalg, "svd", forbidden)
    for site, (u, v, units) in planted.decompose(sd, cfg).items():
        weight = sd[f"{site}.weight"]
        assert u.dtype == v.dtype == weight.dtype
        assert torch.allclose((v @ u).T, weight, atol=2e-6, rtol=2e-6)
        if ".attn." in site:
            for head in range(2):
                expected = torch.zeros_like(weight)
                if site.endswith("o_proj"):
                    expected[:, head * 2:(head + 1) * 2] = weight[:, head * 2:(head + 1) * 2]
                else:
                    expected[head * 2:(head + 1) * 2] = weight[head * 2:(head + 1) * 2]
                selected = units == head
                assert torch.allclose((v[:, selected] @ u[selected]).T, expected, atol=2e-6, rtol=2e-6)


# Fresh paired-run contracts: no policy/model/data construction.


def test_arms_are_identical_except_reward_and_destination(tmp_path):
    args = SimpleNamespace(out=tmp_path, sft_steps=300, rl_steps=20, seed=0,
                           max_len=3072, max_new=416)
    unit = vars(paired.training_args(args, 'unit_suff', 'base', 'same-start', tmp_path / 'unit'))
    wiring = vars(paired.training_args(args, 'wiring_suff', 'base', 'same-start', tmp_path / 'wiring'))
    assert {key: value for key, value in unit.items() if key not in ('reward', 'out')} == {
        key: value for key, value in wiring.items() if key not in ('reward', 'out')}
    assert unit['steps'] == 20 and unit['k'] == 4 and unit['init'] == 'same-start'


def test_panel_is_outcome_independent_and_reordered_stable():
    rows = [{'id': f'{family}_{i:03d}', 'family': family, 'split': split, 'outcome': i % 2}
            for split, families in [('train', range(8)), ('test_instance', range(8)), ('test_family', range(8, 10))]
            for family in families for i in range(5)]
    selected = paired.select_panel(rows)
    assert len(selected) == 24
    assert selected == paired.select_panel(list(reversed(rows)))
    assert [row['id'] for row in selected] == [row['id'] for row in paired.select_panel(
        [{**row, 'outcome': 999} for row in rows])]


def test_completed_stage_hashes_and_partial_rejection(tmp_path):
    directory = tmp_path / 'sft'
    directory.mkdir()
    (directory / 'weights').write_bytes(b'frozen')
    paired.write(tmp_path / 'sft.done.json', {'files': paired.hashes(directory)})
    assert paired.stage_done(tmp_path, 'sft')
    (directory / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        paired.stage_done(tmp_path, 'sft')
    assert not paired.stage_done(tmp_path, 'unit_suff')
    args = SimpleNamespace(out=tmp_path)
    paired.write(tmp_path / 'unit_suff.started.json', {'unfinished': True})
    with pytest.raises(ValueError, match='partial training'):
        paired.learn(args, None, None, 'unit_suff', 'unit_suff')


def test_deadline_does_not_launch_extra_work(monkeypatch):
    monkeypatch.setenv('DEADLINE', '0')
    with pytest.raises(SystemExit) as exc:
        paired.require_time(1)
    assert exc.value.code == 75


def _completed_sft_parent(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    args = SimpleNamespace(out=tmp_path / 'fork', sft_steps=300, rl_steps=20,
                           seed=0, max_len=3072, max_new=416, device='mps')
    paired.write(parent / 'contract.json', dict(model=paired.MODEL, hf_revision=paired.REVISION,
        sft_steps=300, rl_steps_per_arm=20, seed=0, max_len=3072, max_new=416,
        show_active=False, batch=1, k=4, lr=2e-5, beta=0., threads=1, per_family=35))
    (parent / 'sft').mkdir()
    (parent / 'sft/model.safetensors').write_bytes(b'fixture checkpoint bytes; never loaded')
    paired.write(parent / 'sft.done.json', {'files': paired.hashes(parent / 'sft')})
    (parent / 'zoo').mkdir()
    (parent / 'zoo/manifest.jsonl').write_text('{}\n')
    panel = [{'id': f'case{i}'} for i in range(24)]
    paired.write(parent / 'data.json', {'files': paired.hashes(parent / 'zoo'), 'evaluation_panel': panel})
    (parent / 'train_inputs.jsonl').write_text('{"id":"train0","prompt":"input","completion":"truth"}\n')
    paired.write(parent / 'token_fit.json', {'all_fit': True, 'context':3072, 'generation_cap':416,
        'train_n':1, 'rows':[{'id':'train0'}], 'max_pair_tokens':5,
        'max_prompt_tokens':3, 'max_completion_tokens':2})
    records = [{'id':p['id'], 'valid':True, 'stated':{'per_text':[{'kl_bits':0.}]},
                'policy_checkpoint_manifest_sha256':paired.digest(parent / 'sft.done.json')} for p in panel]
    for name in ('gt', 'empty', 'sft'):
        (parent / (name+'_eval.jsonl')).write_text(''.join(__import__('json').dumps(r)+'\n' for r in records))
    paired.write(parent / 'teacher_integrity.json', {'predeclared_tolerance_bits':1e-3,
        'rows':[{'id':p['id'], 'valid':True, 'max_stated_kl_bits':0.} for p in panel]})
    paired.write(parent / 'base_model_hashes.json', {'files':{}})
    (parent / 'dependency-lock.txt').write_text('original frozen packages\n')
    return parent, args


def test_fork_sft_preserves_frozen_parent_and_new_lineage(tmp_path):
    import json
    parent, args = _completed_sft_parent(tmp_path)
    before = paired.hashes(parent)
    paired.fork_sft(args, parent)
    assert paired.hashes(parent) == before
    assert not (args.out / 'contract.json').exists()  # prepare must make a new current contract
    lineage = json.loads((args.out / 'sft_lineage.json').read_text())
    assert lineage['parent_contract_sha256'] == paired.digest(parent / 'contract.json')
    assert lineage['imported_files']['sft/model.safetensors'] == before['sft/model.safetensors']
    assert (args.out / 'parent_evidence/dependency-lock.txt').read_bytes() == (parent / 'dependency-lock.txt').read_bytes()
    assert not (args.out / 'sft_eval.jsonl').exists()
    assert not (args.out / 'gt_eval.jsonl').exists()
    assert not (args.out / 'empty_eval.jsonl').exists()
    paired.verify_lineage(args.out)
    (args.out / 'parent_evidence/sft_eval.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='immutable import'):
        paired.verify_lineage(args.out)
    assert paired.stage_done(args.out, 'sft')
    args.out = tmp_path / 'other'
    assert paired.training_args(args, 'wiring_suff', 'base', 'sft', 'out').device == 'mps'


@pytest.mark.parametrize('failure', ['checkpoint', 'zoo', 'missing_record', 'wrong_config', 'rl_state', 'teacher'])
def test_fork_sft_rejects_changed_or_incomplete_parent(tmp_path, failure):
    import json
    parent, args = _completed_sft_parent(tmp_path)
    if failure == 'checkpoint':
        (parent / 'sft/model.safetensors').write_bytes(b'changed')
    elif failure == 'zoo':
        (parent / 'zoo/manifest.jsonl').write_text('changed')
    elif failure == 'missing_record':
        (parent / 'sft_eval.jsonl').write_text('\n'.join((parent / 'sft_eval.jsonl').read_text().splitlines()[:-1])+'\n')
    elif failure == 'wrong_config':
        args.seed = 7
    elif failure == 'rl_state':
        paired.write(parent / 'unit_suff.started.json', {'step':0})
    else:
        integrity = json.loads((parent / 'teacher_integrity.json').read_text())
        integrity['rows'][0]['max_stated_kl_bits'] = .2
        paired.write(parent / 'teacher_integrity.json', integrity)
    with pytest.raises(ValueError):
        paired.fork_sft(args, parent)
    assert not args.out.exists()


def test_archive_source_revision_is_explicit_metadata(monkeypatch):
    def no_git(*args, **kwargs):
        raise AssertionError('source archive must not invoke git')
    monkeypatch.setattr(paired.subprocess, 'check_output', no_git)
    assert paired.source_revision('a' * 40) == 'a' * 40
    for invalid in ('main', 'a' * 7, 'z' * 40):
        with pytest.raises(ValueError, match='40-hex'):
            paired.source_revision(invalid)
