import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_graph_supervision as trial


class PairedDataTest(unittest.TestCase):
    def rows(self):
        train, evaluation = trial.expected_ids()
        graph = "flowchart LR\n  L0H0 --> out\n"
        rows = [{"id": ident, "prompt": "frozen common input", "point_completion": graph,
                 "intervention_completion": graph} for ident in train]
        panel = [{"id": ident, "prompt": "evaluation input", "stratum": "new_model"}
                 for ident in evaluation]
        panel += [{"id": f"old{i}", "prompt": "retention input", "stratum": "retention"} for i in range(24)]
        return rows, panel

    def test_no_silent_drop_or_new_model_in_training(self):
        rows, panel = self.rows()
        trial.validate_rows(rows, panel)
        with self.assertRaises(ValueError): trial.validate_rows(rows[:-1], panel)
        changed = copy.deepcopy(rows)
        changed[-1]["id"] = panel[0]["id"]
        with self.assertRaises(ValueError): trial.validate_rows(changed, panel)
        with self.assertRaises(ValueError): trial.validate_rows(rows, panel[:-1])

    def test_no_english_stripping_or_token_annotation_repair(self):
        for suffix in ("English: an unverified claim", "tokA --> L0H0", "invented --> out"):
            with self.assertRaises(ValueError):
                trial.graph_only("flowchart LR\nL0H0 --> out\n" + suffix)
        trial.graph_only("flowchart LR\nL0H0\nL1N12\nL0H0 --> L1N12\nL1N12 --> out")

    def test_paired_mass_keeps_every_target_token_and_no_prompt_loss(self):
        import torch
        import tinker
        from tinker.types.tensor_data import TensorData
        datums = [tinker.Datum(model_input=tinker.ModelInput.from_ints([1] * len(w)),
                              loss_fn_inputs={"weights": TensorData.from_torch(torch.tensor(w)),
                                              "target_tokens": TensorData.from_torch(torch.ones(len(w), dtype=torch.int64))})
                  for w in ([0., 0., 1., 1.], [0., 1., 1., 1., 1.])]
        paired, stats = trial.equal_row_mass(datums)
        self.assertEqual(stats["target_mass"], 3.)
        for old, new in zip(datums, paired):
            original = old.loss_fn_inputs["weights"].to_torch()
            weight = new.loss_fn_inputs["weights"].to_torch()
            self.assertTrue(torch.equal(original > 0, weight > 0))
            self.assertAlmostEqual(float(weight.sum()), 3., places=6)
            self.assertTrue(torch.equal(old.loss_fn_inputs["target_tokens"].to_torch(),
                                        new.loss_fn_inputs["target_tokens"].to_torch()))
        self.assertEqual([float(d.loss_fn_inputs["weights"].to_torch().sum()) for d in datums], [2., 4.])


if __name__ == "__main__":
    unittest.main()
