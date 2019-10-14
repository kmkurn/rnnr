from unittest.mock import MagicMock, Mock, call

from tqdm import tqdm

from rnnr.attachments import ProgressBar


def test_ok(runner):
    batches = range(10)
    batch_fn = lambda state: state['batch']**2
    mock_tqdm_cls = MagicMock(spec=tqdm)

    ProgressBar(tqdm_cls=mock_tqdm_cls).attach_on(runner)
    runner.run(batch_fn, batches)

    mock_tqdm_cls.assert_called_once_with(batches)
    assert not mock_tqdm_cls.return_value.set_postfix.called
    assert mock_tqdm_cls.return_value.update.mock_calls == [call(1) for b in batches]
    mock_tqdm_cls.return_value.close.assert_called_once_with()


def test_default_size_key(runner):
    batches = [list('foo'), list('quux')]
    mock_tqdm_cls = MagicMock(spec=tqdm)

    def batch_fn(state):
        state['n_items'] = len(state['batch'])

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls)
    pbar.attach_on(runner)
    runner.run(batch_fn, batches)

    assert mock_tqdm_cls.return_value.update.mock_calls == [call(len(b)) for b in batches]


def test_size_key(runner):
    batches = [list('foo'), list('quux')]
    mock_tqdm_cls = MagicMock(spec=tqdm)

    def batch_fn(state):
        state['foo'] = len(state['batch'])

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls, size_key='foo')
    pbar.attach_on(runner)
    runner.run(batch_fn, batches)

    assert mock_tqdm_cls.return_value.update.mock_calls == [call(len(b)) for b in batches]


def test_stats_key(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)

    def batch_fn(state):
        state['stats'] = {'loss': state['batch']**2}

    pbar = ProgressBar(tqdm_cls=mock_tqdm_cls, stats_key='stats')
    pbar.attach_on(runner)
    runner.run(batch_fn, batches)

    assert mock_tqdm_cls.return_value.set_postfix.mock_calls == [
        call(loss=b**2) for b in batches
    ]


def test_with_kwargs(runner):
    batches = range(10)
    mock_tqdm_cls = MagicMock(spec=tqdm)
    kwargs = {'foo': 'bar', 'baz': 'quux'}

    ProgressBar(tqdm_cls=mock_tqdm_cls, **kwargs).attach_on(runner)
    runner.run(Mock(), batches)

    mock_tqdm_cls.assert_called_once_with(batches, **kwargs)
