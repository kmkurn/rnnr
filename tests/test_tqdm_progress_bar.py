from unittest.mock import Mock

import pytest
from rnnr.utils import TqdmProgressBar
from tqdm import tqdm


@pytest.mark.parametrize("stats", [None, {"foo": 0.5}])
def test_update(stats):
    mock_tqdm = Mock(spec=tqdm)
    pbar = TqdmProgressBar(mock_tqdm)
    pbar.update(10, stats)
    mock_tqdm.update.assert_called_once_with(10)
    if stats:
        mock_tqdm.set_postfix.assert_called_once_with(stats)
    else:
        assert not mock_tqdm.set_postfix.called


def test_finish():
    mock_tqdm = Mock(spec=tqdm)
    pbar = TqdmProgressBar(mock_tqdm)
    pbar.finish()
    mock_tqdm.close.assert_called_once_with()
