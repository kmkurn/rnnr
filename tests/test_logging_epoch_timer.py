import logging
from datetime import timedelta

from rnnr import EpochId
from rnnr.epoch_timer import LoggingEpochTimer


def test_correct():
    logger = logging.getLogger("rnnr.epoch_timer")
    logger.setLevel(logging.INFO)
    history = []

    class AppendToHistoryHandler(logging.Handler):
        def emit(self, record):
            history.append(record.getMessage())

    logger.addHandler(AppendToHistoryHandler())

    class FakeTimer:
        def __init__(self):
            self.started = False

        def start(self):
            self.started = True

        def end(self):
            return timedelta(hours=2, minutes=32, seconds=18)

    timer = FakeTimer()
    epoch_timer = LoggingEpochTimer(timer)

    assert not timer.started
    assert history == []
    with epoch_timer(EpochId(1)):
        assert timer.started
        assert history == ["Epoch 1 started"]
    assert history == ["Epoch 1 started", "Epoch 1 finished in 2:32:18"]


def test_default_timer():
    LoggingEpochTimer()
