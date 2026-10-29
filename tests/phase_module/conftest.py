import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def phase_cpu_threads():
    torch.set_num_threads(2)
