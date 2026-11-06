import numpy as np
import torch
from arc_model.implementations.map_input_calibration.detail_training_network import DetailPerformanceMap, ARMS
from arc_model.implementations.map_input_calibration.deep_network import DeepCodedPhaseMap


def test_all_arms_start_at_original_function_and_capacity_uses_no_new_information():
    torch.manual_seed(44); old = DeepCodedPhaseMap("raw").eval()
    x = torch.randn(3,9,9,64); m = torch.ones(3,9)
    with torch.no_grad(): reference = old(x,m)
    for arm in ARMS:
        torch.manual_seed(44); net = DetailPerformanceMap(arm).eval()
        with torch.no_grad(): torch.testing.assert_close(net(x,m), reference, atol=3e-7, rtol=2e-6)
    capacity = DetailPerformanceMap("capacity").eval()
    with torch.no_grad():
        capacity.c1.weight[:,5:].normal_()
        a = capacity(x,m)
        x[:,:,5:] = 100*torch.randn_like(x[:,:,5:])
        torch.testing.assert_close(capacity(x,m),a,atol=0,rtol=0)
    assert sum(p.numel() for p in capacity.parameters()) == 7417


def test_capacity_duplicate_columns_have_identical_standardized_values():
    from arc_model.implementations.phase_event_snn.data import training_statistics
    x = np.random.RandomState(34).normal(size=(8,9,9,64)).astype("float32")
    mean,std = training_statistics(x,np.arange(5))
    z = torch.from_numpy((x-mean[None,None,:,None])/std[None,None,:,None])
    selected = z[:,:,DetailPerformanceMap("capacity").active]
    torch.testing.assert_close(selected[:,:,5],selected[:,:,1],atol=0,rtol=0)
    torch.testing.assert_close(selected[:,:,6],selected[:,:,2],atol=0,rtol=0)
    torch.testing.assert_close(selected[:,:,7],selected[:,:,1],atol=0,rtol=0)
    torch.testing.assert_close(selected[:,:,8],selected[:,:,2],atol=0,rtol=0)
