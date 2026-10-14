import hashlib
import inspect
import json
import pickle
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
import native_units as module
tokens = [1, 2]


class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = {"n_layer": 1}
        self.n_layer, self.n_head, self.hd, self.eps = 1, 1, 2, 1e-5
        self.fc = torch.nn.Module()
        self.fc.register_buffer("W", torch.ones(3, 2))
        self.fc.U, self.fc.V, self.fc.delta_T = torch.ones(1, 3), torch.ones(2, 1), torch.zeros(2, 3)

    def site(self, name):
        return self.fc


class GE:
    def __init__(self):
        self.nat = types.SimpleNamespace(target=Target(), names=["fc"], C={"fc": 1}, dev="cpu")
        self.calls = 0

    def graph_ref(self, ids, targets, graph, capture):
        self.calls += 1
        value = float(self.nat.target.fc.W.sum()) + sum(ids)
        for k in ("q", "k", "v", "out"):
            capture[k] = torch.full((1, 1, 1, len(ids), 2), value)
        capture["neuron"] = torch.full((1, 1, len(ids), 3), value)


class CacheTests(unittest.TestCase):
    def setUp(self):
        tokens[:] = [1, 2]
        self.ge = GE()
        patch = mock.patch.object(module.goal_eval, "GoalEval", lambda nat: self.ge)
        patch.start()
        self.addCleanup(patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "means.pt"
        self.addCleanup(self.tmp.cleanup)
        for name, value in dict(text=lambda p: (tokens.copy(), None),
                                site_name=lambda l, k: "fc", Graph=lambda **kw: kw,
                                free=lambda d: None).items():
            patch = mock.patch.object(module.native, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def run_means(self):
        return module.unit_means(self.ge.nat, [6000], self.path)

    def test_same_state_hit_and_explicit_frozen_dictionary(self):
        frozen = self.run_means()
        self.run_means()
        self.assertEqual(self.ge.calls, 1)
        old = frozen["heads"].clone()
        self.ge.nat.target.fc.W.mul_(2)
        new = self.run_means()
        self.assertEqual(self.ge.calls, 2)
        self.assertTrue(torch.equal(frozen["heads"], old))
        self.assertFalse(torch.equal(new["heads"], old))

    def test_same_name_changed_tokens_recompute(self):
        old = self.run_means()
        tokens[:] = [2, 3]
        new = self.run_means()
        self.assertEqual(self.ge.calls, 2)
        self.assertNotEqual(old["cache_binding"], new["cache_binding"])

    def test_unregistered_factor_and_runtime_eps_bound(self):
        old = self.run_means()["cache_binding"]
        self.ge.nat.target.fc.V[0, 0] += 1
        changed = self.run_means()["cache_binding"]
        self.assertNotEqual(old, changed)
        self.ge.nat.target.eps *= 2
        self.assertNotEqual(changed, self.run_means()["cache_binding"])
        self.assertEqual(self.ge.calls, 3)

    def test_legacy_shape_nonfinite_and_position_recompute(self):
        for defect in ("legacy", "shape", "nan", "positions"):
            got = self.run_means()
            if defect == "legacy":
                got.pop("cache_binding")
            elif defect == "shape":
                got["neurons"] = torch.zeros(1, 2)
            elif defect == "nan":
                got["neurons"][0, 0] = float("nan")
            else:
                got["positions"] += 1
            torch.save(got, self.path)
            before = self.ge.calls
            self.run_means()
            self.assertEqual(self.ge.calls, before + 1, defect)

    def test_hook_rejected_even_with_existing_cache(self):
        self.run_means()
        h = self.ge.nat.target.fc.register_forward_hook(lambda *a: None)
        with self.assertRaisesRegex(ValueError, "hooks"):
            self.run_means()
        h.remove()

    def test_behavior_and_corrupt_cache(self):
        old = self.run_means()["cache_binding"]
        self.ge.nat.target.behavior = torch.ones(3)
        self.assertNotEqual(old, self.run_means()["cache_binding"])
        self.path.write_bytes(b"corrupt")
        before = self.ge.calls
        self.run_means()
        self.assertEqual(self.ge.calls, before + 1)

    def test_module_helper_source_change_invalidates(self):
        source_dir = Path(self.tmp.name) / "execution_sources"
        source_dir.mkdir()
        for name in ("native.py", "goal_eval.py", "engine.py", "carry.py", "mech.py"):
            (source_dir / name).write_text("original helper\n")
        with mock.patch.object(module.native, "__file__", str(source_dir / "native.py")):
            old = self.run_means()["cache_binding"]
            (source_dir / "goal_eval.py").write_text("changed module-level norm helper\n")
            self.assertNotEqual(old, self.run_means()["cache_binding"])
            self.assertEqual(self.ge.calls, 2)


class RealCaptureTests(unittest.TestCase):
    def test_tiny_native_capture_and_disk_hit(self):
        from test_goal_eval import tiny_native
        nat = tiny_native(seed=5, layers=1, d=8, heads=2)
        ids = [1, 3, 2]
        ge = module.goal_eval.GoalEval(nat)
        capture = {}
        graph = module.native.Graph(masks={n: torch.ones(len(ids), nat.C[n], dtype=torch.bool)
                                          for n in nat.names})
        ge.graph_ref(ids, [2], graph, capture=capture)
        expected_heads = (capture["out"][0].sum(2).double() / len(ids)).float()
        expected_neurons = (capture["neuron"][0].sum(1).double() / len(ids)).float()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(module.native, "text", return_value=(ids, [2])):
            path = Path(tmp) / "cache.pt"
            got = module.unit_means(nat, [6000], path)
            self.assertTrue(torch.equal(got["heads"], expected_heads))
            self.assertTrue(torch.equal(got["neurons"], expected_neurons))
            with mock.patch.object(module.native, "Graph", side_effect=AssertionError("cache miss")):
                hit = module.unit_means(nat, [6000], path)
            self.assertEqual(hit["cache_binding"], got["cache_binding"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
