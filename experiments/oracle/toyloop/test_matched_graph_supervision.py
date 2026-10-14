import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_graph_supervision as trial


class PairedDataTest(unittest.TestCase):
    def test_train35_full_batches_preserve_exact_epoch_visits(self):
        from collections import Counter
        batches=trial.full_batch_schedule(35)
        self.assertEqual(len(batches),35)
        self.assertTrue(all(len(batch)==8 for batch in batches))
        flattened=[i for batch in batches for i in batch]
        self.assertEqual(Counter(flattened),Counter({i:8 for i in range(35)}))
        original=[i for b in trial.pilot.schedule(35,8,8,0) for i in b]
        self.assertEqual(flattened,original)
    def test_opt_in35_exact_order_and_evaluation_boundary(self):
        rows,panel=self.rows(); old=[r['id'] for r in rows]
        ids=old+['redundant_000','cancel_001','gate_000']+[
            f'{f}_{3300000+100*i}' for i,f in enumerate(trial.FAMILIES)]
        manifest={'schema':'paired_train35_v1','train_ids':ids,
                  'evaluation_scope':'reused_original32_development'}
        for ident in ids[len(rows):]:rows.append({**rows[0],'id':ident})
        trial.validate_rows(rows,panel,train_manifest=manifest)
        with self.assertRaises(ValueError):trial.validate_rows(rows,panel)
        for key,value in [('schema','unknown'),('evaluation_scope','fresh_holdout'),
                          ('train_ids',ids[:-1]+[panel[0]['id']])]:
            changed={**manifest,key:value}
            with self.assertRaises(ValueError):trial.validate_rows(rows,panel,train_manifest=changed)
        train_only={**manifest,'evaluation_scope':'train_only_pending_fresh340'}
        trial.validate_rows(rows,[],train_manifest=train_only)
        with self.assertRaises(ValueError):trial.validate_rows(rows,panel,train_manifest=train_only)
    def test_cost_covers_both_training_arms_and_all_first_responses(self):
        contract = {"train_tokens_per_arm": {"point": 100, "intervention": 200},
                    "panel": [{"prompt_tokens": 10}] * 32, "output_cap": 416}
        result = trial.cost_bound(contract, {"trained": 2., "prefill": 3., "sampled": 5.})
        self.assertEqual(result["tokens"], {"trained": 300, "prefill": 960, "sampled": 39936})
        self.assertEqual(result["sampling_calls"], 96)
        self.assertAlmostEqual(result["upper_usd"], .20316)
        for bad in (float("nan"), float("inf"), -1., True):
            with self.assertRaises(ValueError):
                trial.cost_bound(contract, {"trained": bad, "prefill": 3., "sampled": 5.})

    def test_one_fixed_schedule_and_seed_for_both_arms(self):
        ids, _ = trial.expected_ids()
        batches = trial.pilot.schedule(len(ids), trial.EPOCHS, trial.BATCH, trial.SEED)
        recipe = trial.training_recipe(batches, ids)
        self.assertEqual(recipe["initialization"]["seed"], 0)
        self.assertEqual(len(recipe["optimizer"]), 24)
        self.assertEqual(recipe["evaluation_arms"], ["base", "point", "intervention"])
        for offset in range(0, len(batches), 3):
            self.assertEqual(sorted(i for batch in recipe["sample_ids"][offset:offset+3] for i in batch), sorted(ids))

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

        data = dict(zip(trial.ARMS, ([paired[0]], [paired[1]])))
        prompt = tinker.ModelInput.from_ints([1, 2, 3])
        prepared = {"data": data, "prompts": [prompt], "batches": [[0]],
                    "renderer": SimpleNamespace(get_stop_sequences=lambda: ["<|im_end|>"]),
                    "contract": {"batches_sha256": trial.digest([[0]]),
                                 "rendered_training_sha256": {arm: [trial.datum_digest(d) for d in rows]
                                                               for arm, rows in data.items()},
                                 "panel": [{"rendered_sha256": trial.digest(prompt.to_ints())}]}}
        serialized, digest = trial.serialize_prepared(prepared)
        payload = json.loads(serialized)
        self.assertEqual(trial.digest(payload), digest)
        for arm in trial.ARMS:
            restored = trial.restore_datum(payload["data"][arm][0])
            self.assertEqual(trial.datum_digest(restored), trial.datum_digest(data[arm][0]))
        restored_prompt = tinker.ModelInput.model_validate(payload["prompts"][0])
        self.assertEqual(restored_prompt.to_ints(), prompt.to_ints())
        prepared["prompts"] = [tinker.ModelInput.from_ints([3, 2, 1])]
        with self.assertRaisesRegex(ValueError, "prompts changed"):
            trial.serialize_prepared(prepared)


if __name__ == "__main__":
    unittest.main()
