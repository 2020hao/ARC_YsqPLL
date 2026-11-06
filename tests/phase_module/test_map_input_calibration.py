import numpy as np
import torch
from arc_model.implementations.map_input_calibration.network import InputCodedPhaseMap,code_inputs
from arc_model.implementations.map_input_calibration.calibration import select_threshold,THRESHOLDS
from arc_model.implementations.phase_event_snn.network import EventEncoder
from arc_model.implementations.direct_efficiency.phase_map_hybrid import PhaseMapHybrid
from arc_model.evaluation.cycle_metrics import cycle_metrics


def test_clipping_and_raw_equivalence():
    torch.manual_seed(7);base=PhaseMapHybrid().eval()
    torch.manual_seed(7);wrapped=InputCodedPhaseMap('raw').eval()
    x=torch.randn(3,9,8,64)*6;m=torch.ones(3,9)
    with torch.no_grad():torch.testing.assert_close(base(x,m),wrapped(x,m),rtol=0,atol=0)
    c=code_inputs(x,'clip');assert c.abs().max()<=3
    torch.testing.assert_close(c[x.abs()<=3],x[x.abs()<=3],rtol=0,atol=0)
    assert sum(p.numel() for p in wrapped.parameters())==3945


def test_sparse_reconstruction_matches_ttfs_information():
    torch.manual_seed(9);x=torch.randn(4,9,8,64)*4;m=torch.ones(4,9)
    x[0,0,0,:7]=torch.tensor([0.,.249,.25,1.,2.,3.,9.])
    code=EventEncoder()(x,m);count=code.sum(2);step=torch.arange(4)[None,None,:,None,None]
    magnitude=(.25+(1-(code*step).sum(2)/3)*2.75)*(count>0)
    decoded=magnitude[:,:,:8]-magnitude[:,:,8:]
    torch.testing.assert_close(code_inputs(x,'sparse4'),decoded,rtol=0,atol=0)
    assert not code_inputs(x,'sparse4')[x.abs()<.25].any()


def test_threshold_group_equivalence_ties_and_vector_evaluation():
    y=np.array([0,0,1,1,0,1]);p=np.array([.1,.4,.5,.9,.2,.6]);g=np.array(['a']*4+['b']*2)
    t,s=select_threshold(y,p,g)
    expected=np.array([cycle_metrics(y,p,g,threshold=v)['BAcc_condition'] for v in THRESHOLDS])
    np.testing.assert_allclose(s,expected,rtol=0,atol=1e-12);assert t==.5
    raw=cycle_metrics(y,p,g);cal=cycle_metrics(y,p,g,threshold=np.array([.55]*4+[.3]*2))
    assert raw['AP']==cal['AP'] and raw['ROC_AUC']==cal['ROC_AUC']
    assert cal['FN']==1


def test_deep_variant_budget_and_finite_gradients():
    from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap
    for coding in ('raw','clip','sparse4'):
        net=DeepCodedPhaseMap(coding);x=torch.randn(3,9,8,64);mask=torch.ones(3,9)
        out,a=net(x,mask,True);(out.square().mean()+.001*a).backward()
        assert sum(p.numel() for p in net.parameters())==7097
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
