from unittest.mock import MagicMock, call

from tqdm import tqdm

from rnnr import Event
from rnnr.attachments import ProgressBar


def test_ok(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)

    ProgressBar(tqdm_cls=mock_tqdm_cls).attach_on(runner)
    runner.run(batches)

    mock_tqdm_cls.assert_called_once_with(batches, initial=0)
    assert not mock_tqdm_cls.return_value.set_postfix.called
    assert mock_tqdm_cls.return_value.update.mock_calls == [call(1) for b in batches]
    mock_tqdm_cls.return_value.close.assert_called_once_with()


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


def test_with_kwargs(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)
    kwargs = {"foo": "bar", "baz": "quux"}

    ProgressBar(tqdm_cls=mock_tqdm_cls, **kwargs).attach_on(runner)
    runner.run(batches)

    mock_tqdm_cls.assert_called_once_with(batches, initial=0, **kwargs)
