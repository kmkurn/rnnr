import logging
import pytest

from rnnr import Runner
from rnnr.attachments import EpochTimer


@pytest.mark.parametrize("attach_time", ["early", "late"])
@pytest.mark.parametrize("max_epoch", [1, 2])
def test_correct_call_order(attach_time, max_epoch):
    logger = logging.getLogger("rnnr.attachments.epoch_timer")
    logger.setLevel(logging.INFO)
    history = []

    class AppendToHistoryHandler(logging.Handler):
        def emit(self, record):
            s = record.getMessage()
            history.append(s[: s.index(";")])

    logger.addHandler(AppendToHistoryHandler())

    def on_batch(e, i, b):
        history.append("B")

    runner = Runner(on_batch, max_epoch)
    if attach_time == "early":
        EpochTimer(start_fmt="ETS %d;%d", finish_fmt="ETF %d;%d%s").attach_on(runner)

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
        EpochTimer(start_fmt="ETS %d;%d", finish_fmt="ETF %d;%d%s").attach_on(runner)
    runner.run(range(1))
    if max_epoch == 1:
        expected = ["S", "ES", "BS", "B", "BF", "EF", "F"]
    else:
        assert max_epoch == 2
        expected = [
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

    assert history == expected
