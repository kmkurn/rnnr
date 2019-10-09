import statistics as stat

import pytest

from rnnr import Event
from rnnr.attachments import MeanAggregator


def test_ok(runner):
    batches = range(5)
    values = [12, 7, 8, 44, -13]

    def batch_fn(state):
        state['output'] = values[state['batch']]

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(values[b] for b in batches))

    agg = MeanAggregator()
    assert agg.name == 'mean'
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_more_than_one_epoch(runner):
    batches = range(5)
    values = [12, 7, 8, 44, -13, 78, 55, -109, 34, 10]
    epoch = 0

    def batch_fn(state):
        state['output'] = (epoch - 1) * len(batches) + values[state['batch']]

    def eshandler(state):
        nonlocal epoch
        epoch = state['epoch']

    def efhandler(state):
        assert state[agg.name] == pytest.approx(
            stat.mean((epoch - 1) * len(batches) + values[b] for b in batches))

    agg = MeanAggregator()
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_STARTED, eshandler)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches, max_epoch=2)


def test_custom_name(runner):
    batches = range(10)

    def batch_fn(state):
        state['output'] = state['batch']**2

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(b**2 for b in batches))

    agg = MeanAggregator(name='avg')
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_value_key(runner):
    batches = range(10)

    def batch_fn(state):
        state['value'] = state['batch']**3

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(b**3 for b in batches))

    agg = MeanAggregator(value_key='value')
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_size_key(runner):
    batches = range(5)
    sizes = [3, 4, 9, 10, 2]

    def batch_fn(state):
        state['output'] = state['batch']
        state['foo'] = sizes[state['batch']]

    def efhandler(state):
        assert state[agg.name] == pytest.approx(sum(batches) / sum(sizes))

    agg = MeanAggregator(size_key='foo')
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)
