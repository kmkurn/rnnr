from unittest.mock import call, patch

from rnnr.handlers import EarlyStopper


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


def test_loss_fn(runner):
    values = [5, 4, 3, 4, 5, 6, 7]
    es = EarlyStopper(runner, patience=2, loss_fn=lambda state: state['loss'][0])

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'loss': (v, v**2)})
            if i == len(values) - 2:
                mock_stop.assert_called_once_with()
            elif i == len(values) - 1:
                assert mock_stop.mock_calls == [call(), call()]
            else:
                assert not mock_stop.called
