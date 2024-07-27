from unittest.mock import Mock

import pytest
from rnnr import EpochId
from rnnr.batch import BatchOutput
from rnnr.progress_tracker import TqdmEpochProgressTracker
from tqdm import tqdm


@pytest.mark.parametrize("has_stats", [False, True])
def test_log_one_batch(has_stats):
    stats = {"foo": 3, "bar": 7.5} if has_stats else None
    mock_tqdm_obj = Mock(spec=tqdm)

    def tqdm_factory(e):
        assert e == 1
        return mock_tqdm_obj

    tracker = TqdmEpochProgressTracker(tqdm_factory)
    with tracker.start(EpochId(1)) as log:
        log(BatchOutput(0.23, stats))
        expected = {"loss": pytest.approx(0.23)}
        if stats:
            expected.update({"foo": 3, "bar": pytest.approx(7.5)})
        mock_tqdm_obj.set_postfix.assert_called_once_with(expected)
        mock_tqdm_obj.update.assert_called_once_with(n=1)
    mock_tqdm_obj.close.assert_called_once_with()
