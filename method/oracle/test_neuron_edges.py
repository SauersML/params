"""Checks of the neuron edge stage (neuron_edges.py) on a tiny random Qwen3: with no cut the run is the model's; a cut
edge's reader is recomputed from its input less what the edge brings (or, patching, plus what it brings in the corrupted
run); the per-position sums for every other candidate agree with listing them; pruning meets its tolerance and the graph
lists poles and edges; prefix runs equal full runs.

  ~/mpd-data/venv/bin/python -m pytest method/oracle/test_neuron_edges.py -q
"""
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neuron_edges  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(vocab_size=48, hidden_size=32, intermediate_size=40, num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 48, (10,), generator=g).tolist()
    neurons = sorted({(int(torch.randint(3, (1,), generator=g)), int(torch.randint(6, 10, (1,), generator=g)),
                       int(torch.randint(40, (1,), generator=g))) for _ in range(24)})
    return model, ids, neurons


def contrast_of(model, ids):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0, 9].float(), -1)
    return float(neuron_edges.log2_contrast(lp, [3], [5, 7]))


def test_runs_are_the_models(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    assert ed.c_full == pytest.approx(contrast_of(model, ids), abs=1e-4)
    E = len(ed.resid_all)
    assert ed.contrasts(torch.zeros(1, E, dtype=torch.bool), ed.resid_all)[0] == pytest.approx(ed.c_full, abs=1e-4)


def every_edge(ed):
    """Every candidate edge, from each reader's in-edges (every one listed)."""
    cands = ed.in_edges(10 ** 4)
    es = [e for c in cands for e in c[0]]
    assert len(es) == len(ed.resid_all) + ed.n_routes
    return cands, es


def test_cut_runs_recompute_the_reader(tiny):
    """An edge cut alone: in the run its reader's activation is the reader recomputed from its input less what the edge
    brings (resid, and routes through the closed-form value change)."""
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    _, es = every_edge(ed)
    for e in es[::max(1, len(es) // 40)]:
        r = e[-1]
        cut = torch.ones(1, 1, dtype=torch.bool)
        got = float(ed._run(cut, [e], keep_states=True)["h"][r[0]][r[1], r[2]])
        want = ed._acts(r[0], [ed.at[r]], (ed._mid_stream(r[0])[r[1]] - ed.brings(e))[None, None])
        assert got == pytest.approx(float(want), rel=1e-4, abs=1e-6)


def test_local_curves(tiny):
    """Each reader's local effects are its Jacobian times what the edge brings; its curve ends at its base activation
    with every in-edge kept and starts with every one cut."""
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    cands, _ = every_edge(ed)
    J = ed.jacobians()
    for es, sc in cands:
        assert bool((sc.abs()[:-1] >= sc.abs()[1:]).all())
        for e, s in list(zip(es, sc.tolist()))[::7]:
            assert s == pytest.approx(float(J[e[-1]] @ ed.brings(e)), rel=1e-4, abs=1e-7)
    curves = ed.local_curves(cands)
    h0 = ed._h0()
    for i, (es, _) in enumerate(cands):
        assert float(curves[i][-1]) == pytest.approx(float(h0[i]), rel=1e-4, abs=1e-6)
        if es:
            r = ed.N[i]
            allcut = ed._mid_stream(r[0])[r[1]] - sum(ed.brings(e) for e in es)
            assert float(curves[i][0]) == pytest.approx(float(ed._acts(r[0], [i], allcut[None, None])), rel=1e-4, abs=1e-6)


def test_prune_and_graph(tiny):
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    got = neuron_edges.prune(ed, log=lambda m: None)
    rec = got["record"]
    assert rec["met"] and abs(rec["contrast_kept"] - rec["contrast_all_edges"]) <= rec["tolerance"] + 1e-6
    g = neuron_edges.graph(ed, got)
    assert len(g["nodes"]) == len(ed.N) and len(g["edges"]) == rec["kept"]
    assert all(n["pole"] in (1, -1) and n["regime"] in ("g+u+", "g+u-", "g-u+", "g-u-") for n in g["nodes"])


def test_exported_route_scope_is_not_full_attention_routing(tiny):
    """Saved evidence must distinguish conditional OV cuts from Q/K or whole-writer interventions."""
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    route = next(e for e in every_edge(ed)[1] if len(e) == 3)
    exported = neuron_edges.graph(ed, {"edges": [route], "strengths": [0.0], "local": [0.0], "record": {}})
    saved = json.loads(json.dumps(exported))
    assert saved["edges"][0]["kind"] == "route"
    semantics = saved["route_cut_semantics"]
    assert semantics["path"] == "value_ov"
    assert semantics["attention_pattern"] == "current_run_fixed_in_local_value_delta"
    assert semantics["head_input_norm"] == "recomputed_after_writer_write_subtraction"
    assert semantics["reader_mlp"] == "recomputed_norm_gate_up"
    assert semantics["combined_routes"] == "sum_of_individual_value_deltas"
    assert not semantics["query_key_effects_included"]
    assert not semantics["full_writer_ablation_equivalent"]
    assert not semantics["full_causal_completeness_certified"]


def test_rest_cut_sums(tiny):
    """Every other candidate cut by the per-position sums: with every candidate listed and kept the run is the model's;
    with none listed it is every candidate listed and cut."""
    model, ids, neurons = tiny
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    every = every_edge(ed)[1]
    E = len(every)
    kept = ed.contrasts(torch.zeros(1, E, dtype=torch.bool), every, rest=True)[0]
    assert kept == pytest.approx(ed.c_full, abs=1e-4)
    none = ed.contrasts(torch.zeros(1, 0, dtype=torch.bool), [], rest=True)[0]
    assert none == pytest.approx(ed.contrasts(torch.ones(1, E, dtype=torch.bool), every)[0], abs=1e-4)
    half = torch.arange(E)[None] % 2 == 0  # (half listed and cut, the other half as run against a cut rest)
    got = ed.contrasts(half, every, rest=True)[0]
    assert got == pytest.approx(ed.contrasts(half, every)[0], abs=1e-4)


def test_bfloat16_model(tiny):
    """A bfloat16 model runs the same stage (each module in bfloat16, float32 between): its contrasts follow the float32
    model's to bfloat16's precision."""
    import copy
    model, ids, neurons = tiny
    m16 = copy.deepcopy(model).to(torch.bfloat16)
    e32 = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons)
    e16 = neuron_edges.Edges(m16, ids, 9, [3], [5, 7], neurons)
    cuts = torch.eye(len(e32.resid_all), dtype=torch.bool)[:8]
    a, b = e32.contrasts(cuts, e32.resid_all), e16.contrasts(cuts, e16.resid_all)
    assert e16.c_full == pytest.approx(e32.c_full, abs=0.05)
    assert max(abs(x - y) for x, y in zip(a, b)) < 0.1


def test_patching_pair(tiny):
    """Patching (a corrupted prompt of the same length): a cut edge's reader reads what the edge brings in the corrupted
    run; the per-position sums agree with every edge listed; the local curves end at the base activation and start at
    the reader with every in-edge patched."""
    model, ids, neurons = tiny
    bad = list(ids)
    bad[6] = (bad[6] + 5) % 48
    ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, corrupt_ids=bad)
    cands, every = every_edge(ed)
    E = len(every)
    assert ed.contrasts(torch.zeros(1, E, dtype=torch.bool), every, rest=True)[0] == pytest.approx(ed.c_full, abs=1e-4)
    none = ed.contrasts(torch.zeros(1, 0, dtype=torch.bool), [], rest=True)[0]
    assert none == pytest.approx(ed.contrasts(torch.ones(1, E, dtype=torch.bool), every)[0], abs=1e-4)
    half = torch.arange(E)[None] % 2 == 0
    assert ed.contrasts(half, every, rest=True)[0] == pytest.approx(ed.contrasts(half, every)[0], abs=1e-4)
    for e in [x for x in every if len(x) == 2][:5]:  # one resid edge patched: its reader from its input less the clean
        r = e[1]                                      # write plus the corrupted one
        got = float(ed._run(torch.ones(1, 1, dtype=torch.bool), [e], keep_states=True)["h"][r[0]][r[1], r[2]])
        v = ed._edge_vectors([e], "clean", "cpu")[0] - ed._edge_vectors([e], "corrupt", "cpu")[0]
        want = ed._acts(r[0], [ed.at[r]], (ed._mid_stream(r[0])[r[1]] - v)[None, None])
        assert got == pytest.approx(float(want), rel=1e-4, abs=1e-6)
    curves, h0 = ed.local_curves(cands), ed._h0()
    for i, (es, _) in enumerate(cands):
        assert float(curves[i][-1]) == pytest.approx(float(h0[i]), rel=1e-4, abs=1e-6)
    got = neuron_edges.prune(ed, log=lambda m: None)
    assert got["record"]["met"]


def test_prefix_runs_equal_full_runs(tiny, monkeypatch):
    """A cut run computing only the positions from the circuit's first (the base run's keys and values before it) equals
    the full run: listed cuts, the rest cut, and a patching pair."""
    model, ids, neurons = tiny
    g = torch.Generator().manual_seed(2)
    bad = list(ids)
    bad[7] = (bad[7] + 5) % 48
    for kw in ({}, {"corrupt_ids": bad}):
        ed = neuron_edges.Edges(model, ids, 9, [3], [5, 7], neurons, **kw)
        _, es = every_edge(ed)
        cuts = torch.rand(6, len(es), generator=g) < 0.3
        got = {}
        for on in (True, False):
            monkeypatch.setattr(neuron_edges, "PREFIX", on)
            got[on] = [ed.contrasts(cuts, es), ed.contrasts(cuts[:, :len(ed.resid_all)], ed.resid_all, rest=True)]
        for a, b in zip(got[True], got[False]):
            assert a == pytest.approx(b, abs=1e-4)


def test_effective_weight_matches_actual_peft_states():
    import peft
    torch.manual_seed(42)
    model=peft.get_peft_model(torch.nn.Sequential(torch.nn.Linear(5,4,bias=True)),
        peft.LoraConfig(r=2,lora_alpha=3,lora_dropout=0.,target_modules=['0']))
    mod=model.get_base_model()[0]
    with torch.no_grad():mod.lora_B['default'].weight.normal_()
    mod.eval()
    x = torch.randn(7, 5)
    def parity():
        W = neuron_edges.weight(mod)
        torch.testing.assert_close(mod(x), x @ W.T + mod.base_layer.bias, rtol=1e-5, atol=1e-6)
    parity()
    mod.enable_adapters(False);parity()
    assert neuron_edges.weight(mod) is mod.base_layer.weight
    mod.enable_adapters(True);mod.merge();parity()
    assert neuron_edges.weight(mod) is mod.base_layer.weight
    mod.enable_adapters(False)
    with pytest.raises(ValueError, match='disabled merged'):neuron_edges.weight(mod)
    assert mod.merged  # extraction must not silently unmerge or mutate


def test_effective_weight_multiple_adapters_variants_and_plain_norm():
    import peft
    model=peft.get_peft_model(torch.nn.Sequential(torch.nn.Linear(5,4,bias=False)),
        peft.LoraConfig(r=2,lora_alpha=3,lora_dropout=0.,target_modules=['0']))
    model.add_adapter('other',peft.LoraConfig(r=2,lora_alpha=2,lora_dropout=0.,target_modules=['0']))
    mod=model.get_base_model()[0]
    with torch.no_grad():
        for layer in mod.lora_B.values():layer.weight.normal_()
    mod.set_adapter(['default','other']);mod.eval();x=torch.randn(3,5)
    torch.testing.assert_close(mod(x),x @ neuron_edges.weight(mod).T,rtol=1e-5,atol=1e-6)
    # PEFT's merged branch uses base only, even if another active adapter is unmerged.
    mod.merge(adapter_names=['default'])
    assert set(mod.active_adapters)=={'default','other'} and mod.merged_adapters==['default']
    torch.testing.assert_close(mod(x),x @ neuron_edges.weight(mod).T,rtol=1e-5,atol=1e-6)
    assert neuron_edges.weight(mod) is mod.base_layer.weight
    mod.unmerge()
    mod.lora_variant['default']=object()
    with pytest.raises(ValueError,match='variant'):neuron_edges.weight(mod)
    mod.lora_variant.clear();mod.use_dora['default']=True
    with pytest.raises(ValueError,match='variant'):neuron_edges.weight(mod)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    norm=Qwen3RMSNorm(5)
    assert neuron_edges.weight(norm) is norm.weight
    linear=torch.nn.Linear(5,4)
    assert neuron_edges.weight(linear) is linear.weight


def test_finder_graph_keeps_cut_summary_and_marks_source_provenance(monkeypatch):
    from types import SimpleNamespace
    import types
    summary={'none':1.25,'graph_edges':.5,'every_edge':-.25}
    monkeypatch.setattr(neuron_edges,'PRUNE',True)
    monkeypatch.setattr(neuron_edges,'Edges',lambda *a,**kw:object())
    monkeypatch.setattr(neuron_edges,'prune',lambda *a,**kw:{})
    monkeypatch.setattr(neuron_edges,'graph',lambda *a,**kw:{'cut':dict(summary),'edges':[]})
    class Runner:
        def __init__(self,model):pass
        def run(self,ids,sets,capture=False):return {'contrast':[2. if ids==[1,2] else 1.],'states':{}}
    monkeypatch.setitem(sys.modules,'head_routes',types.SimpleNamespace(Runner=Runner))
    rec={'ids':[1,2],'corrupted':[1,3],'answer':4,'alternatives':[5],'nodes':[],
         'model':'source-model','adapter':'source-adapter','dtype':'bfloat16',
         'finder':'circuit_finder','finder_rev':'source-revision','id':'case'}
    got=neuron_edges.finder_graph(object(),None,rec,log=lambda _:None)
    assert got['cut']==summary and got['intervention']=='patch from the corrupted run'
    assert got['source_provenance']=={k:rec[k] for k in ('model','adapter','dtype','finder','finder_rev')}
    assert got['source_provenance_status']=='copied_source_record_not_verified_against_loaded_model'
    assert rec['adapter']=='source-adapter'


def test_finder_graph_mover_edges(tiny):
    """A finder record's graph carries mover-head edges: the circuit's neurons through each mover's key and value (and
    query at its position), its readout neurons reading its output, each group cut measured."""
    model, ids, neurons = tiny
    bad = list(ids)
    bad[5] = (bad[5] + 7) % 48
    nodes = [{"stage": "source", "layer": 0, "pos": 5, "neuron": 3}, {"stage": "source", "layer": 1, "pos": 6, "neuron": 8},
             {"stage": "readout", "layer": 2, "pos": 9, "neuron": 11}, {"stage": "readout", "layer": 2, "pos": 9, "neuron": 4}]
    rec = {"id": "t", "ids": ids, "corrupted": bad, "answer": 3, "alternatives": [5, 7], "nodes": nodes,
           "routes": [{"layer": 2, "head": 1, "pos": 9}, {"layer": 1, "head": 0, "pos": 9}]}
    g = neuron_edges.finder_graph(model, None, rec, log=lambda m: None)
    he = {tuple(e["head"]): e for e in g["head_edges"]}
    sides = lambda e: sorted((tuple(x["neuron"]), x["input"]) for x in e["inputs"])  # noqa: E731
    assert sides(he[(2, 1)]) == [((0, 5, 3), "key"), ((0, 5, 3), "value"), ((1, 6, 8), "key"), ((1, 6, 8), "value")]
    assert set(he[(2, 1)]["group_lost"]) == {"key", "value", "all"} and len(he[(2, 1)]["readers"]) == 2
    assert sides(he[(1, 0)]) == [((0, 5, 3), "key"), ((0, 5, 3), "value")] and len(he[(1, 0)]["readers"]) == 2
    assert he[(1, 0)]["readers_group_lost"] is not None and set(g["movers_inputs_lost"]) == {"key", "value", "all"}
    assert [r["unit"]["kind"] for r in g["readers"]] == ["neuron"] * 4 + ["head"] * 2
    assert [len(r["readers"]) for r in g["readers"]] == [8, 8, 0, 0, 8, 8]  # (the last layer's neurons: no reader)
    assert all(r["total"]["share"] is not None for r in g["readers"])
