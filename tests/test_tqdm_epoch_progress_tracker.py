from unittest.mock import Mock

import pytest
from rnnr import EpochId
from rnnr.batch import BatchOutput
from rnnr.progress_tracker import TqdmEpochProgressTracker
from tqdm import tqdm


def test_log_one_batch():
    mock_tqdm_obj = Mock(spec=tqdm)

    def tqdm_factory(e):
        assert e == 1
        return mock_tqdm_obj

    class FakeBatchOutput(BatchOutput):
        @property
        def loss(self):
            return 0.23

        @property
        def stats(self):
            return {"foo": 3, "bar": 7.5}

    tracker = TqdmEpochProgressTracker(tqdm_factory)
    with tracker.start(EpochId(1)) as log:
        log(FakeBatchOutput())
        mock_tqdm_obj.set_postfix.assert_called_once_with(
            {"loss": pytest.approx(0.23), "foo": 3, "bar": pytest.approx(7.5)}
        )
        mock_tqdm_obj.update.assert_called_once_with(n=1)
    mock_tqdm_obj.close.assert_called_once_with()
