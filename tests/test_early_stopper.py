from unittest.mock import patch

from rnnr.handlers import EarlyStopper


def test_ok(runner):
    # default patience is 5
    values = [5, 4, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    es = EarlyStopper(runner)

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'output': v})
            if i == len(values) - 1:
                mock_stop.assert_called_once_with()
            else:
                assert not mock_stop.called


def test_custom_patience(runner):
    values = [5, 4, 3, 4, 5, 6]
    es = EarlyStopper(runner, patience=3)

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'output': v})
            if i == len(values) - 1:
                mock_stop.assert_called_once_with()
            else:
                assert not mock_stop.called


def test_value_fn(runner):
    values = [5, 4, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]
    es = EarlyStopper(runner, value_fn=lambda state: state['output'][0])

    with patch.object(runner, 'stop', autospec=True) as mock_stop:
        for i, v in enumerate(values):
            es({'output': (v, v**2)})
            if i == len(values) - 1:
                mock_stop.assert_called_once_with()
            else:
                assert not mock_stop.called
