from unittest.mock import Mock

import pytest
from rnnr.utils import TqdmProgressBar
from tqdm import tqdm


@pytest.fixture
def mock_tqdm():
    return Mock(spec=tqdm)


@pytest.fixture
def pbar(mock_tqdm):
    return TqdmProgressBar(mock_tqdm)


def test_update(mock_tqdm, pbar):
    pbar.update(10)
    mock_tqdm.update.assert_called_once_with(10)


def test_done(mock_tqdm, pbar):
    pbar.done()
    mock_tqdm.close.assert_called_once_with()


def test_show_stats(mock_tqdm, pbar):
    pbar.show_stats({"foo": 0.5})
    mock_tqdm.set_postfix.assert_called_once_with({"foo": 0.5})
