import pytest
from rnnr.batch import DefaultBatchOutput


def test_only_loss():
    bo = DefaultBatchOutput(0.5)
    assert bo.loss == pytest.approx(0.5)
    assert bo.stats == {}


def test_loss_and_stats():
    bo = DefaultBatchOutput(0.5, {"foo": 10})
    assert bo.loss == pytest.approx(0.5)
    assert bo.stats == {"foo": 10}
