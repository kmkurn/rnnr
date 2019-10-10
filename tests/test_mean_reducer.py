import statistics as stat

import pytest

from rnnr import Event
from rnnr.attachments import MeanReducer


def test_ok(runner):
    batches = range(5)
    values = [12, 7, 8, 44, -13]

    def batch_fn(state):
        state['output'] = values[state['batch']]

    r = MeanReducer()
    r.attach_on(runner)
    state = runner.run(batch_fn, batches)

    assert r.name == 'mean'
    assert state['mean'] == pytest.approx(stat.mean(values))


def test_more_than_one_epoch(runner):
    batches = range(5)
    values = [12, 7, 8, 44, -13, 78, 55, -109, 34, 10]

    def batch_fn(state):
        ix = (state['epoch'] - 1) * len(batches) + state['batch']
        state['output'] = values[ix]

    def efhandler(state):
        assert state[r.name] == pytest.approx(
            stat.mean(values[(state['epoch'] - 1) * len(batches) + b] for b in batches))

    r = MeanReducer()
    r.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches, max_epoch=2)


def test_custom_name(runner):
    r = MeanReducer(name='avg')
    assert r.name == 'avg'


def test_value_key(runner):
    batches = range(10)

    def batch_fn(state):
        state['value'] = state['batch']**3

    r = MeanReducer(value_key='value')
    r.attach_on(runner)
    state = runner.run(batch_fn, batches)

    assert state[r.name] == pytest.approx(stat.mean(b**3 for b in batches))


def test_size_key(runner):
    batches = range(5)
    sizes = [3, 4, 9, 10, 2]

    def batch_fn(state):
        state['output'] = state['batch']
        state['foo'] = sizes[state['batch']]

    r = MeanReducer(size_key='foo')
    r.attach_on(runner)
    state = runner.run(batch_fn, batches)

    assert state[r.name] == pytest.approx(sum(batches) / sum(sizes))
