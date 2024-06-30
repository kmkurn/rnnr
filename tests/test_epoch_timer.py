import logging
from datetime import timedelta

import pytest
from rnnr import EpochId
from rnnr.epoch_timer import LoggingEpochTimer


@pytest.mark.parametrize("max_epoch", [1, 3])
def test_correct(max_epoch):
    logger = logging.getLogger("rnnr.epoch_timer")
    logger.setLevel(logging.INFO)
    history = []

    class AppendToHistoryHandler(logging.Handler):
        def emit(self, record):
            history.append(record.getMessage())

    logger.addHandler(AppendToHistoryHandler())
    timer_started = False

    class FakeTimer:
        def start(self):
            nonlocal timer_started
            timer_started = True

        def end(self):
            return timedelta(hours=2, minutes=32, seconds=18)

    epoch_timer = LoggingEpochTimer(max_epoch, FakeTimer())
    epoch_timer.start_epoch(EpochId(1))
    assert timer_started

    epoch_timer.finish_epoch(EpochId(2))
    expected = ["Epoch 1/3 started", "Epoch 2/3 finished in 2:32:18"] if max_epoch == 3 else []
    assert history == expected


def test_default_timer():
    LoggingEpochTimer(max_epoch=3).start_epoch(EpochId(1))
