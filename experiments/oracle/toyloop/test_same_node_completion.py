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
    def test_unordered_invalid_edges_keep_category_and_no_validity_relaxation(self):
        a='edge L0H2 -> L0N83 has no executable connection in these pieces'
        b='edge L0N191 -> L0H1 has no executable connection in these pieces'
        self.assertTrue(c.same_validity(False,False,a,b))
        self.assertFalse(c.same_validity(False,True,a,None))
        self.assertFalse(c.same_validity(False,False,a,'invalid_graph_only_answer'))
        self.assertFalse(c.same_validity(False,False,a,'edge foreign -> out has no executable connection in these pieces'))
    def test_preservation_reports_damage_and_rescues_separately(self):
        rows=[dict(case='pass',arm='a',prior_all_pass=True,complete_all_pass=False),dict(case='fail',arm='a',prior_all_pass=False,complete_all_pass=True)]
        result=c.preservation(rows)
        self.assertEqual(result['prior_pass_pairs'],1);self.assertEqual(result['preserved_pass_pairs'],0)
        self.assertEqual(result['damaged_pass_pairs'],[{'case':'pass','arm':'a'}]);self.assertEqual(result['rescued_failure_pairs'],1)
    def test_all_original_scope_includes_previously_passing_cases(self):
        ids=['additive_vote_3200000','bigram_mlp_3200100','binding_positional_3200200','cancel_3200300','copy_offset_3200400','gate_3200500','induction_3200600','redundant_3200700']
        counts=[180,220,156,172,188,204,180,196]
        self.assertEqual(sum(counts),1496)
        panel=[dict(id=i,condition='original',inputs=[[]]*4,edits=[{}]*(n//4)) for i,n in zip(ids,counts)]
        manifest={'panel':panel+[dict(r,id=r['id']+'__perm',condition='permuted') for r in panel]}
        all_rows,worlds=c.scoped_panel(manifest,True)
        self.assertEqual(len(all_rows),8);self.assertEqual(worlds*7,10472)
        default,w=c.scoped_panel(manifest);self.assertEqual([r['id'] for r in default],list(c.CASES));self.assertEqual(w,384)
        manifest['panel'][0]['condition']='permuted'
        with self.assertRaises(ValueError):c.scoped_panel(manifest,True)
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
