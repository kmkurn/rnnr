import pytest
from rnnr import IterationBasedRunner


def test_init():
    r = IterationBasedRunner(on_batch=lambda x: x)
    assert len(r.state) == 0


@pytest.fixture
def itrunner():
    return IterationBasedRunner()


def test_run(itrunner):
    batches, max_iter = range(10), 5
    itrunner.run(batches, max_iter=max_iter)
    state = itrunner.state

    assert state["batches"] == batches
    assert state["max_iter"] == max_iter
    assert state["n_iters"] == max_iter
    assert not state["running"]
