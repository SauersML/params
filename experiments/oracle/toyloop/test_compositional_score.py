"""First-sample coverage and finish contracts, no Native/model imports."""
from types import SimpleNamespace
import unittest
import compositional_score as c


class ScoreTests(unittest.TestCase):
    def test_parseable_length_remains_invalid(self):
        graph=SimpleNamespace(ok=True,bad=[])
        self.assertFalse(c.saved_valid({'finish':'length'},graph))
        self.assertFalse(c.saved_valid({'finish':'stop','parse_error':'ValueError'},graph))
        self.assertTrue(c.saved_valid({'finish':'stop'},graph))

    def test_missing_or_duplicate_rows_cannot_be_dropped(self):
        panel=[{'id':'original24/x'},{'id':'training_incidentalV20/x'}]
        c.complete_panel(panel,panel)
        with self.assertRaises(ValueError):
            c.complete_panel(panel[:1],panel)
        with self.assertRaises(ValueError):
            c.complete_panel([panel[0],panel[0]],panel)


if __name__=='__main__':
    unittest.main()
