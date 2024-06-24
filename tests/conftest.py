import functools
import pytest

from rnnr import Runner


@pytest.fixture
def runner():
    return Runner()


@pytest.fixture
def call_tracker():
    class CallTracker:
        def __init__(self):
            self.history = []

        def track_args(self, fn):
            @functools.wraps(fn)
            def wrapper(*args):
                self.history.append((fn.__name__, args))
                return fn(*args)

            return wrapper

    return CallTracker()


@pytest.fixture
def do_nothing():
    def _do_nothing(*args, **kwargs):
        pass

    return _do_nothing
