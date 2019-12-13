from unittest.mock import call, patch

import pytest

from rnnr.callbacks import EarlyStopper


def test_ok(runner):
    # default patience is 5
    values = [5, 4, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7]
    min_values = [5, 4, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    es = EarlyStopper()

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, (v, mv) in enumerate(zip(values, min_values)):
            es({'runner': runner, 'loss': v})
            assert es.best_value == pytest.approx(mv, abs=1e-4)
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called


def test_custom_patience(runner):
    values = [5, 4, 3, 4, 5, 6, 7, 8]
    es = EarlyStopper(patience=3)

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'runner': runner, 'loss': v})
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called


def test_value_key(runner):
    values = [5, 4, 3, 4, 5, 6, 7]
    es = EarlyStopper(patience=2, value_key='foo')

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'runner': runner, 'foo': v})
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called


def test_max_mode(runner):
    values = [5., 4., 3., 7., 5., 6., 3.]
    max_values = [5., 5., 5., 7., 7., 7., 7.]
    es = EarlyStopper(patience=2, mode='max', value_key='acc')

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, (v, mv) in enumerate(zip(values, max_values)):
            es({'runner': runner, 'acc': v})
            assert es.best_value == pytest.approx(mv)
            if i == len(values) - 1:
                mock_stop.assert_called_once_with()
            else:
                assert not mock_stop.called
