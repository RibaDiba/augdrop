import pytest

from probe_helpers import ToyAdapter, make_batch, make_dataset, make_ops


@pytest.fixture
def adapter():
    return ToyAdapter()


@pytest.fixture
def ops():
    return make_ops()


@pytest.fixture
def dataset():
    return make_dataset()


@pytest.fixture
def batch():
    return make_batch()
