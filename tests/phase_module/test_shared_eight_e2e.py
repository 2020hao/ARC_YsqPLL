"""E2E adapter starts exactly at the baseline and permits encoder gradients."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch


@pytest.mark.parametrize('name',['ArcFENet','ArcFusionNet','HYDRA'])
def test_initial_baseline_and_unfrozen_gradients(name):
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('e2e_test',root/'scripts/train_shared_eight_e2e.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2)
    base=module.make_model(name,'original',0).eval()
    full=module.make_model(name,'phase8',0).eval()
    features=torch.randn(2,8)
    if name=='HYDRA':
        x=torch.randn(2,448);a=base(x);b=full(torch.cat([x,features],1))
    else:
        x=torch.randn((2,3,100,100) if name=='ArcFENet' else (2,1,3,250))
        s=torch.randn(2,1,32) if name=='ArcFusionNet' else None
        a=base(x,s);b=full(x,s,features)
    torch.testing.assert_close(a,b,rtol=0,atol=1e-7)
    b.square().mean().backward()
    if name!='HYDRA':
        assert full.extra.weight.grad is not None and full.extra.weight.grad.norm()>0
        assert any(p.grad is not None and p.grad.norm()>0 for k,p in full.named_parameters() if 'head' not in k and not k.startswith('extra.'))
    assert all(p.requires_grad for p in full.parameters())
