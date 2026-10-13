import copy
from types import SimpleNamespace
import unittest

import same_node_completion as c


class Graph:
    def __init__(self,nodes=(),parents=None,out=None,complete=False):
        self.nodes=set(nodes);self.parents={r:list(w) for r,w in (parents or {}).items()};self.out=list(out or []);self.complete=complete
    def node_set(self):return self.nodes
    def count(self):return len(self.nodes)
    def edges(self):return sum(len(w) for w in self.parents.values())+len(self.out)


class CompletionTests(unittest.TestCase):
    def fixture(self):
        a=('h.0.attn.o_proj',0,1);b=('h.1.attn.v_proj',0,2);z=('h.1.attn.o_proj',0,3)
        nodes={a,b,z}
        native=SimpleNamespace(Graph=Graph,all_edges=lambda ns,ts:({b:[a],z:[b]},[a,z]),
                               implied_edges=lambda ns,ts:(2,2))
        return Graph(nodes,{b:[a]},[z]),native

    def test_explicit_same_nodes_adds_only_wiring_not_dense_pieces(self):
        graph,native=self.fixture();before=copy.deepcopy(graph.__dict__)
        result,size=c.complete_same_nodes(graph,native,[0])
        self.assertEqual(result.nodes,graph.nodes);self.assertFalse(result.complete)
        self.assertEqual(graph.__dict__,before)
        self.assertEqual(size['selected_nodes'],3)
        self.assertEqual(size['original_explicit_edges'],2)
        self.assertEqual(size['completed_architecture_edges'],4)
        self.assertTrue(all(set(w)<=set(result.parents[r]) for r,w in graph.parents.items()))
        self.assertTrue(set(graph.out)<=set(result.out))

    def test_illegal_original_edge_fails_not_silently_dropped(self):
        graph,native=self.fixture();graph.parents[next(iter(graph.nodes))]=[('foreign',0,0)]
        with self.assertRaises(RuntimeError):c.complete_same_nodes(graph,native,[0])

    def test_outside_edges_cannot_smuggle_dense_nodes(self):
        graph,native=self.fixture();native.all_edges=lambda ns,ts:({next(iter(ns)):[('foreign',0,0)]},[])
        with self.assertRaises(RuntimeError):c.complete_same_nodes(graph,native,[0])

    def test_no_selected_piece_addition_and_count_mismatch(self):
        graph,native=self.fixture()
        class Extra(Graph):
            def __init__(self,**kwargs):
                super().__init__(**kwargs);self.nodes.add(('dense_other',0,0))
        native.Graph=Extra
        with self.assertRaises(RuntimeError):c.complete_same_nodes(graph,native,[0])
        graph,native=self.fixture();native.implied_edges=lambda ns,ts:(99,2)
        with self.assertRaises(RuntimeError):c.complete_same_nodes(graph,native,[0])


if __name__=='__main__':unittest.main()
