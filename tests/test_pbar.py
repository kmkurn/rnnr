from unittest.mock import MagicMock, call

import pytest
from rnnr import Event, Runner
from rnnr.attachments import ProgressBar
from tqdm import tqdm


def test_ok():
    history = []

    def on_batch(e, bi, b):
        history.append("B")

    runner = Runner(on_batch, max_epoch=1)

    @runner.on_started
    def on_started():
        history.append("S")

    @runner.on_epoch_started
    def on_epoch_started(e):
        history.append("ES")

    @runner.on_batch_started
    def on_batch_started(e, bi, b):
        history.append("BS")

    @runner.on_batch_finished
    def on_batch_finished(e, bi, b, o):
        history.append("BF")

    @runner.on_epoch_finished
    def on_epoch_finished(e):
        history.append("EF")

    @runner.on_finished
    def on_finished():
        history.append("F")

    batches = range(10)

    class tracked_tqdm(tqdm):
        def __init__(self, total=None, **kwargs):
            history.append("TTI")
            assert total == len(batches)
            super().__init__(total=total)

        def update(self, size):
            history.append("TTU")
            assert size == 1
            return super().update(size)

        def close(self):
            history.append("TTC")
            return super().close()

        def set_postfix(self, *args, **kwargs):
            assert False

    ProgressBar(len(batches), tqdm_cls=tracked_tqdm).attach_on(runner)
    runner.run(batches)
    expected = ["S", "ES", "TTI"]
    for _ in batches:
        expected.extend(["BS", "B", "BF", "TTU"])
    expected.extend(["TTC", "EF", "F"])

    assert history == expected


@pytest.mark.skip
def test_default_n_items(runner):
    batches = [list("foo"), list("quux")]
    mock_tqdm_cls = MagicMock(spec=tqdm)

    @runner.on(Event.BATCH)
    def on_batch(state):
        state["n_items"] = len(state["batch"])

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls)
    pbar.attach_on(runner)
    runner.run(batches)

    assert mock_tqdm_cls.return_value.update.mock_calls == [call(len(b)) for b in batches]


@pytest.mark.skip
def test_n_items(runner):
    batches = [list("foo"), list("quux")]
    mock_tqdm_cls = MagicMock(spec=tqdm)

    @runner.on(Event.BATCH)
    def on_batch(state):
        state["foo"] = len(state["batch"])

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls, n_items="foo")
    pbar.attach_on(runner)
    runner.run(batches)

    assert mock_tqdm_cls.return_value.update.mock_calls == [call(len(b)) for b in batches]


@pytest.mark.skip
def test_stats(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)

    @runner.on(Event.BATCH)
    def on_batch(state):
        state["stats"] = {"loss": state["batch"] ** 2}

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls, stats="stats")
    pbar.attach_on(runner)
    runner.run(batches)

    assert mock_tqdm_cls.return_value.set_postfix.mock_calls == [
        call(loss=b ** 2) for b in batches
    ]


@pytest.mark.skip
def test_with_kwargs(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)
    kwargs = {"foo": "bar", "baz": "quux"}

    ProgressBar(tqdm_cls=mock_tqdm_cls, **kwargs).attach_on(runner)
    runner.run(batches)

    mock_tqdm_cls.assert_called_once_with(batches, initial=0, **kwargs)
