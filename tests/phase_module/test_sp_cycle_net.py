import torch
from arc_model.implementations.sp_cycle_net.network import SPCycleNet, CausalCycleBlock


def test_shape_parameter_and_mask():
    torch.set_num_threads(1)
    net = SPCycleNet().eval()
    x = torch.randn(2, 9, 3, 64)
    mask = torch.ones(2, 9); mask[:, :4] = 0
    altered = x.clone(); altered[:, :4] += 100
    with torch.no_grad():
        y = net(x, mask)
        other = net(altered, mask)
    assert y.shape == (2,)
    assert sum(p.numel() for p in net.parameters()) == 22737
    torch.testing.assert_close(y, other, rtol=0, atol=0)


def test_cycle_convolution_has_no_future_access():
    torch.set_num_threads(1)
    block = CausalCycleBlock(3).eval()
    x = torch.randn(2, 32, 9); other = x.clone(); other[:, :, 5:] += 20
    mask = torch.ones(2, 1, 9)
    with torch.no_grad():
        a, b = block(x, mask), block(other, mask)
    torch.testing.assert_close(a[:, :, :5], b[:, :, :5], rtol=0, atol=0)


def test_scaler_and_balancing_only_use_training_rows():
    import numpy as np
    from arc_model.implementations.sp_cycle_net.data import training_statistics, condition_class_weights
    profiles = np.ones((6, 3, 64), dtype='float32')
    histories = np.repeat(np.arange(6)[:, None], 9, axis=1)
    train = np.array([0, 1, 2, 3])
    before = training_statistics(profiles, histories, train)
    profiles[4:] = 1e6
    after = training_statistics(profiles, histories, train)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    y = np.array([0, 1, 0, 1, 0, 1]); groups = np.array(['a','a','b','b','c','c'])
    w = condition_class_weights(y, groups, train)
    assert (w[4:] == 0).all()
    np.testing.assert_allclose(w[train].mean(), 1)
