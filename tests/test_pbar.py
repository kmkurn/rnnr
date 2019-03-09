from unittest.mock import MagicMock, Mock, call

from tqdm import tqdm

from rnnr.attachments import ProgressBar


def test_attach_on(runner):
    batches = range(10)
    batch_fn = lambda x: x**2
    mock_tqdm_cls = MagicMock(spec=tqdm)

    ProgressBar(tqdm_cls=mock_tqdm_cls).attach_on(runner)
    runner.run(batch_fn, batches)

    mock_tqdm_cls.assert_called_once_with(batches)
    assert mock_tqdm_cls.return_value.set_postfix.mock_calls == [
        call(output=batch_fn(b)) for b in batches
    ]
    assert mock_tqdm_cls.return_value.update.mock_calls == [call(1) for b in batches]
    mock_tqdm_cls.return_value.close.assert_called_once_with()


def test_custom_update_size(runner):
    batches = [list('foo'), list('quux')]
    mock_tqdm_cls = MagicMock(spec=tqdm)

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls, update_size=lambda s: len(s['batch']))
    pbar.attach_on(runner)
    runner.run(Mock(), batches)

    assert mock_tqdm_cls.return_value.update.mock_calls == [call(len(b)) for b in batches]


def test_display_stats(runner):
    batches = range(10)
    batch_fn = lambda x: x**2
    mock_tqdm_cls = MagicMock(spec=tqdm)

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls, stats=lambda s: {'loss': s['output']})
    pbar.attach_on(runner)
    runner.run(batch_fn, batches)

    assert mock_tqdm_cls.return_value.set_postfix.mock_calls == [
        call(loss=batch_fn(b)) for b in batches
    ]


def test_with_kwargs(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)
    kwargs = {'foo': 'bar', 'baz': 'quux'}

    ProgressBar(tqdm_cls=mock_tqdm_cls, **kwargs).attach_on(runner)
    runner.run(Mock(), batches)

    mock_tqdm_cls.assert_called_once_with(batches, **kwargs)
