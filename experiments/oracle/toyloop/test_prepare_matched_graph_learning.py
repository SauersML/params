import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import prepare_matched_graph_learning as preparation


class PreparationTests(unittest.TestCase):
    def test_retention_bank_binds_bytes_and_historical_content_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "cases/toy/bank.json"
            path.parent.mkdir(parents=True)
            inputs = [[0, 1], [0, 2]]
            path.write_text(json.dumps(inputs, indent=2) + "\n")
            files = {"cases/toy/bank.json": preparation.sha(path)}
            receipt = {"id": "toy", "input_bank_count": 2,
                "input_bank_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest()}
            self.assertEqual(preparation.retention_bank(root, receipt, files), inputs)
            path.write_text(json.dumps([[0, 1], [0, 3]]))
            with self.assertRaises(RuntimeError):
                preparation.retention_bank(root, receipt, files)
            files["cases/toy/bank.json"] = preparation.sha(path)
            with self.assertRaises(RuntimeError):
                preparation.retention_bank(root, receipt, files)

    def test_duplicate_producers_cannot_be_selected_by_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("old", "new")]
            for root in roots:
                (root / "case").mkdir(parents=True)
                (root / "case/search.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                preparation.proof_location("case", roots)


if __name__ == "__main__":
    unittest.main()
