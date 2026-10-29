import numpy as np
import torch
from arc_model.models.standalone.hydra import Hydra
from arc_model.models.standalone.hydra_paper import paper_transform, fit_l1_hinge


def test_paper_transform_matches_reference():
    torch.set_num_threads(2)
    x=np.random.default_rng(8).normal(size=(4,1000)).astype('float32')
    h=Hydra(1000,k=8,g=8,kernel_length=5,stride=1,divisor=1,seed=0)
    with torch.no_grad():z=h(torch.tensor(x[:,None])).numpy()
    np.testing.assert_array_equal(paper_transform(x),z)
    assert z.shape==(4,1024)


def test_l1_hinge_objective_and_nonseparable_data():
    x=np.array([[-2.],[-1.],[1.],[2.],[0.],[0.]])
    y=np.array([0,0,1,1,0,1])
    w,b,d=fit_l1_hinge(x,y,lambda_l1=.1,time_limit=10)
    assert w[0]>0
    assert d['max_constraint_violation']<1e-7
    assert abs(d['objective']-(1/3+.1))<1e-7
