import unittest
from types import SimpleNamespace

import fresh_model_score as f


class Graph:
    def __init__(self, nodes=(), parents=None, out=None, complete=False):
        self.nodes = set(nodes)
        self.parents = {r:list(ws) for r,ws in (parents or {}).items()}
        self.out = list(out or [])
        self.complete = complete

    def node_set(self):
        return self.nodes

    def count(self):
        return len(self.nodes)

    def edges(self):
        return sum(len(ws) for ws in self.parents.values())+len(self.out)


class FreshScoreTests(unittest.TestCase):
    def fixture(self):
        a, b = ('a',0,0), ('b',0,0)
        native = SimpleNamespace(Graph=Graph,
            all_edges=lambda nodes,targets:({b:[a]},[a,b]),
            implied_edges=lambda nodes,targets:(1,2))
        parser = SimpleNamespace(parse=lambda text:SimpleNamespace(ok=True,bad=[]),
                                 unit_graph=lambda parsed,toy,ids:Graph({a,b},{b:[a]},[b]))
        answer = {'finish':'stop','completion':'flowchart LR\nL0H0 --> out'}
        return answer, parser, native

    def test_same_nodes_both_explicit_branches_and_old_graph_immutable(self):
        answer, parser, native = self.fixture()
        raw, complete, size, error = f.compile_pipelines(answer,None,[0,1],parser,native)
        self.assertIsNone(error)
        self.assertEqual(raw.node_set(),complete.node_set())
        self.assertFalse(raw.complete)
        self.assertFalse(complete.complete)
        self.assertEqual(raw.edges(),2)
        self.assertEqual(complete.edges(),3)
        self.assertEqual(size['selected_nodes'],2)

    def test_length_or_bad_grammar_stays_invalid_in_both(self):
        answer, parser, native = self.fixture()
        for value in (None,dict(answer,finish='length'),
                      dict(answer,completion='```mermaid\nflowchart LR\nL0H0 --> out\n```')):
            raw,complete,size,error = f.compile_pipelines(value,None,[0,1],parser,native)
            self.assertIsNone(raw)
            self.assertIsNone(complete)
            self.assertIsNone(size)
            self.assertIsNotNone(error)

    def test_input_specific_native_rejection_does_not_contaminate_later_input(self):
        answer, parser, native = self.fixture()
        original = parser.unit_graph
        def compile_one(parsed,toy,ids):
            if ids[-1]==1:
                raise ValueError('input-specific invalid pieces')
            return original(parsed,toy,ids)
        parser.unit_graph = compile_one
        self.assertIsNone(f.compile_pipelines(answer,None,[0,1],parser,native)[0])
        self.assertIsNotNone(f.compile_pipelines(answer,None,[0,2],parser,native)[0])

    def test_completion_integrity_errors_propagate(self):
        answer, parser, native = self.fixture()
        native.all_edges = lambda nodes,targets:({},[])
        with self.assertRaises(RuntimeError):
            f.compile_pipelines(answer,None,[0,1],parser,native)


if __name__ == '__main__':
    unittest.main()
