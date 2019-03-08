import pytest

from rnnr import Runner


@pytest.fixture
def runner():
    return Runner()
