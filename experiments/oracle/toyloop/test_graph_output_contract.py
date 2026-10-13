import unittest

from graph_output_contract import GRAPH_GRAMMAR, clarify_graph_prompt


class ContractTests(unittest.TestCase):
    def test_append_preserves_every_original_observation_byte(self):
        original = "Weights:\nL0H2 OV[t3>t9]\nTexts:\nBOS t1 t3\nReturn graph only."
        changed = clarify_graph_prompt(original)
        self.assertEqual(changed[:len(original)], original)
        self.assertEqual(changed[len(original):], "\n\n" + GRAPH_GRAMMAR)
        self.assertNotIn("L0H2", GRAPH_GRAMMAR)
        self.assertNotIn("t3", GRAPH_GRAMMAR)

    def test_double_clarification_and_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            clarify_graph_prompt(clarify_graph_prompt("original"))
        with self.assertRaises(ValueError):
            clarify_graph_prompt("")


if __name__ == "__main__":
    unittest.main()
