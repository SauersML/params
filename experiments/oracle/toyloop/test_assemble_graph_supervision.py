import copy
import hashlib
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import assemble_graph_supervision as producer


def fixture():
    target = "flowchart LR\n  L0H1 --> out\n"
    digest = hashlib.sha256(target.encode()).hexdigest()
    inputs = [[0, 1], [0, 2]]
    weights = {"w": "actualweightsha"}
    binding = {"cfg": {"layers": 1}, "inputs": inputs, "source_weights": weights,
               "panel": [["head", 0, 1, 0.]]}
    rows = [{"ids": ids, "kind": kind, "layer": layer, "unit": unit, "scale": scale,
             "kl_bits": .00001}
            for ids in inputs for kind, layer, unit, scale in
            [("none", None, None, 1.), ("head", 0, 1, 0.)]]
    bundle = {"target": target, "result": {"target": target, "target_sha256": digest,
              "status": "greedy_fixed_point", "selected": {"completion": target,
              "completion_sha256": digest, "score": {"per_world": rows}}},
              "contract": {"case": "case"},
              "cache": {"binding": binding, "integrity_raw_kl_bits": [0.] * 4},
              "original": copy.deepcopy(binding), "restored": copy.deepcopy(binding),
              "uncached": {"per_world": copy.deepcopy(rows)}}
    return bundle, dict(ident="case", cfg={"layers": 1}, inputs=inputs, source_weights=weights)


class ProducerTests(unittest.TestCase):
    def test_point_control_requires_external_frozen_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point.json"
            path.write_text(json.dumps({"unit_edges": [["L0.H1", "OUT"]], "context_units": []}))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            control, binding = producer.read_point_control(path, ident="case", expected_sha256=expected,
                                                           source_weights={"w": "actualweights"})
            self.assertEqual(binding["case"], "case")
            path.write_text(json.dumps({"unit_edges": [["L0.H2", "OUT"]], "context_units": []}))
            with self.assertRaisesRegex(RuntimeError, "frozen model manifest"):
                producer.read_point_control(path, ident="case", expected_sha256=expected,
                                            source_weights={"w": "actualweights"})

    def test_prompt_has_no_truth_or_teacher_metadata_and_restores_flags(self):
        toy = SimpleNamespace(truth={"edges": "point label"}, meta={"teacher": "intervention label"})
        renderer = SimpleNamespace(SECTIONS="all", SHOW_ACTIVE=True, FORMAT="OLD ANSWER FORMAT")
        def render(observed, **kwargs):
            self.assertIsNone(observed.truth)
            self.assertEqual(observed.meta, {})
            self.assertEqual(renderer.SECTIONS, "weights")
            self.assertFalse(renderer.SHOW_ACTIVE)
            self.assertEqual(kwargs, dict(n_texts=4, all_live_neurons=True, explain_summary_limits=True))
            return "shared observations\n" + renderer.FORMAT
        renderer.render = render
        prompt = producer.source_prompt(toy, [[0, 1]] * 4, renderer)
        self.assertTrue(prompt.startswith("shared observations\n"))
        self.assertNotIn("OLD ANSWER FORMAT", prompt)
        self.assertEqual(renderer.SECTIONS, "all")
        self.assertTrue(renderer.SHOW_ACTIVE)
        self.assertEqual(toy.meta, {"teacher": "intervention label"})
        renderer.render = lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("render failure"))
        with self.assertRaises(RuntimeError):
            producer.source_prompt(toy, [[0, 1]] * 4, renderer)
        self.assertEqual(renderer.SECTIONS, "all")
        self.assertTrue(renderer.SHOW_ACTIVE)

    def test_target_mutation_and_wrong_actual_weights_rejected(self):
        bundle, args = fixture()
        self.assertEqual(producer.verify_teacher(bundle, **args)["worlds"], 4)
        changed = copy.deepcopy(bundle)
        changed["target"] = "flowchart LR\n  L0H2 --> out\n"
        with self.assertRaisesRegex(RuntimeError, "selected measured"):
            producer.verify_teacher(changed, **args)
        args["source_weights"] = {"w": "otherweights"}
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            producer.verify_teacher(bundle, **args)

    def test_duplicate_worlds_and_failed_integrity_rejected(self):
        bundle, args = fixture()
        changed = copy.deepcopy(bundle)
        changed["uncached"]["per_world"][1] = changed["uncached"]["per_world"][0]
        changed["result"]["selected"]["score"]["per_world"] = copy.deepcopy(changed["uncached"]["per_world"])
        with self.assertRaisesRegex(RuntimeError, "Cartesian"):
            producer.verify_teacher(changed, **args)
        changed = copy.deepcopy(bundle)
        changed["cache"]["integrity_raw_kl_bits"][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            producer.verify_teacher(changed, **args)

    def test_point_conversion_preserves_routes_and_drops_only_annotations(self):
        graph, receipt = producer.point_graph([("tok:BOS", "L0.H1"), ("L0.H1", "L1.N2"),
                                               ("L1.N2", "OUT")], [[{"unit": "L0.H3"}]])
        self.assertIn("L0H1 --> L1N2", graph)
        self.assertIn("L1N2 --> out", graph)
        self.assertIn("  L0H3\n", graph)
        self.assertNotIn("tok", graph)
        self.assertEqual(len(receipt["discarded_nonexecuted_token_arrows"]), 1)
        with self.assertRaises(ValueError):
            producer.point_graph([("invented", "OUT")], [])


if __name__ == "__main__":
    unittest.main()
