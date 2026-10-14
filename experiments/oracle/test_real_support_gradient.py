import numpy as np
import real_support_gradient as p

def test_noncontiguous_low_loss_and_canonical_ties():
 base={'b':np.ones((1,2),bool),'a':np.ones((1,4),bool)}
 eligible={'a':np.array([[1,0,1,1]],bool),'b':np.ones((1,2),bool)}
 scores={'a':np.array([[2,99,-3,2]]),'b':np.array([[2,4]])}
 ranked=p.rank(eligible,scores);q=p.proposal(base,ranked,3)
 assert np.array_equal(q['a'],[[0,1,0,0]]) # protected a1 survives, stable a-before-b ties
 assert q['b'].all()
 assert all(m.all() for m in base.values())

def test_worst_world_deletion_sign():
 gradients=[np.array([2.,-1.,.5]),np.array([3.,4.,.1])]
 score=np.maximum.reduce([-g for g in gradients])
 assert np.argsort(score).tolist()==[0,2,1]

def test_counts_and_nonmonotone_winner():
 assert p.counts(1000)==[1,2,5,10,20,50,100,200,350,500,700,900]
 rows=[{'removed':1,'kl':[.0001]*4},{'removed':2,'kl':[.002]*4},{'removed':10,'kl':[.0009]*4}]
 assert p.winner(rows)['removed']==10
 assert p.winner([{'removed':99,'kl':[float('nan')]*4}]) is None
 assert p.winner([{'removed':99,'kl':[.0001,.0001,.0001,.002]}]) is None

def test_score_finite_and_bounds():
 import unittest
 with unittest.TestCase().assertRaises(ValueError):p.rank({'a':np.ones(1,bool)},{'a':np.array([float('nan')])})
 ranking=p.rank({'a':np.ones(1,bool)},{'a':np.zeros(1)})
 with unittest.TestCase().assertRaises(ValueError):p.proposal({'a':np.ones(1,bool)},ranking,2)

def test_sign_direction_finite_difference_synthetic():
 import torch
 x=torch.tensor([.7,.9],dtype=torch.float64,requires_grad=True)
 def loss(v):return ((v-torch.tensor([.2,.4]))**2).sum()
 grad,=torch.autograd.grad(loss(x),(x,));direction=grad.sign();eps=1e-4
 fd=(loss(x.detach()+eps*direction)-loss(x.detach()-eps*direction))/(2*eps)
 assert torch.allclose(fd,(grad*direction).sum(),atol=1e-9)

if __name__=='__main__':
 import unittest
 suite=unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in sorted(globals().copy().items()) if k.startswith('test_') and callable(v))
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 raise SystemExit(not result.wasSuccessful())
