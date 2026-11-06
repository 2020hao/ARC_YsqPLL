import torch
from arc_model.implementations.dynamic_state_net import native_dynamic_summary
from arc_model.implementations.phase_hierarchy_net.augmentation import augment_standardized


def test_shared_physical_augmentation_preserves_dynamic_summary():
    torch.manual_seed(5)
    raw=torch.randn(6,9,5,64)
    raw[:,:,1:3]-=5
    mean=torch.tensor([.2,-4.,-5.,0.,0.])
    std=torch.tensor([.7,2.,1.2,.02,.03])
    mask=torch.ones(6,9);mask[2,:4]=0;mask[4,3]=0
    xp=(raw-mean[None,None,:,None])/std[None,None,:,None]
    aug=augment_standardized(xp,mean,std,torch.Generator().manual_seed(9))
    changed=aug*std[None,None,:,None]+mean[None,None,:,None]
    torch.testing.assert_close(native_dynamic_summary(raw,mask),native_dynamic_summary(changed,mask),atol=2e-6,rtol=2e-5)


def test_gain_changes_power_by_twice_log_current_gain_without_relabeling():
    raw=torch.ones(5,9,5,64)
    raw[:,:,1:3]*=-5
    mean=torch.zeros(5);std=torch.ones(5)
    aug=augment_standardized(raw,mean,std,torch.Generator().manual_seed(15),phase_halfwidth=0)
    gain=aug[:,0,0,0]
    assert ((gain>=10**(-.15))&(gain<=10**.15)).all()
    expected=2*gain.log10()
    torch.testing.assert_close(aug[:,:,1:3]-raw[:,:,1:3],expected[:,None,None,None].expand(-1,9,2,64),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(aug[:,:,3:5],raw[:,:,3:5])
