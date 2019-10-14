from unittest.mock import call, patch

import pytest

from rnnr.handlers import EarlyStopper, InvalidStateError


def test_ok(runner):
    # default patience is 5
    values = [5, 4, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    es = EarlyStopper(runner)

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'loss': v})
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called


def test_custom_patience(runner):
    values = [5, 4, 3, 4, 5, 6, 7, 8]
    es = EarlyStopper(runner, patience=3)

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'loss': v})
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called


def test_loss_key(runner):
    values = [5, 4, 3, 4, 5, 6, 7]
    es = EarlyStopper(runner, patience=2, loss_key='foo')

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'foo': v})
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called


def test_dump_load_state(runner):
    losses = [1, 2]
    es = EarlyStopper(runner, patience=1)

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for v in losses:
            es({'loss': v})
        assert not mock_stop.called

        es2 = EarlyStopper(runner, patience=1)
        es2.load_state(es.dump_state())
        es2({'loss': losses[-1]})
        assert mock_stop.called


def test_load_invalid_state(runner):
    es = EarlyStopper(runner)
    with pytest.raises(InvalidStateError) as excinfo:
        es.load_state({})
    assert 'Invalid state' in str(excinfo.value)
    assert 'state is returned by dump_state()' in str(excinfo.value)
    with pytest.raises(InvalidStateError):
        es.load_state('')
