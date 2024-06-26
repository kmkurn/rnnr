import logging
import pytest

from rnnr import Runner
from rnnr.attachments import EpochTimer


@pytest.mark.parametrize("attach_time", ["early", "late"])
def test_correct_call_order(attach_time):
    history = []

    def on_batch(e, i, b):
        history.append("B")

    runner = Runner(on_batch, max_epoch=2)
    epoch_timer = EpochTimer(start_fmt="ETS %(epoch)d", finish_fmt="ETF %(epoch)d")

    class AppendToHistoryHandler(logging.Handler):
        def emit(self, record):
            history.append(record)

    epoch_timer.logger.addHandler(AppendToHistoryHandler())
    if attach_time == "early":
        epoch_timer.attach_on(runner)

    @runner.on_started
    def on_started():
        history.append("S")

    @runner.on_epoch_started
    def on_epoch_started(e):
        history.append("ES")

    @runner.on_batch_started
    def on_batch_started(e, i, b):
        history.append("BS")

    @runner.on_batch_finished
    def on_batch_finished(e, i, b, o):
        history.append("BF")

    @runner.on_epoch_finished
    def on_epoch_finished(e):
        history.append("EF")

    @runner.on_finished
    def on_finished():
        history.append("F")

    if attach_time == "late":
        epoch_timer.attach_on(runner)
    batches = range(1)
    runner.run(batches)
    assert history == [
        "S",
        "ETS 1",
        "ES",
        "BS",
        "B",
        "BF",
        "EF",
        "ETF 1",
        "ETS 2",
        "ES",
        "BS",
        "B",
        "BF",
        "EF",
        "ETF 2",
        "F",
    ]
