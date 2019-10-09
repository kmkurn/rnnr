import pytest

from rnnr import Event
from rnnr.attachments import Reducer


def test_abc():
    with pytest.raises(Exception):
        Reducer()


def test_ok(runner):
    class LastReducer(Reducer):
        def reduce(self, x, y):
            return y

        @property
        def name(self):
            return 'last'

    batches = range(5)
    values = [12, 7, 8, 44, -13]

    def batch_fn(state):
        state['output'] = values[state['batch']]

    def efhandler(state):
        assert state['last'] == values[-1]

    r = LastReducer()
    r.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_value_key(runner):
    class LastReducer(Reducer):
        def reduce(self, x, y):
            return y

        @property
        def name(self):
            return 'last'

    batches = range(5)
    values = [12, 7, 8, 44, -13]

    def batch_fn(state):
        state['foo'] = values[state['batch']]

    def efhandler(state):
        assert state['last'] == values[-1]

    r = LastReducer(value_key='foo')
    r.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)
