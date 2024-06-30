from unittest.mock import patch

import pytest
from rnnr.utils import Timer


def test_correct():
    with patch("rnnr.utils.time.time", side_effect=[100.0, 5789.0]) as mock_time:
        timer = Timer()
        assert mock_time.call_count == 1
        elapsed = timer.end()
        assert mock_time.call_count == 2

    assert elapsed.seconds == pytest.approx(5689)
