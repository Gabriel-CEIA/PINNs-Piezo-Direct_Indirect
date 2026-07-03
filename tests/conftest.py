from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def hidden_sizes():
    return [10, 20]


@pytest.fixture(scope="session")
def batch_size():
    return 8


@pytest.fixture(scope="session")
def input_size():
    return 2


@pytest.fixture(scope="session")
def output_size():
    return 8
