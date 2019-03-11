import statistics as stat

import pytest

from rnnr import Event
from rnnr.attachments import MeanAggregator


def test_attach_on(runner):
    batches = range(5)
    values = [12, 7, 8, 44, -13]
    batch_fn = lambda b: values[b]

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(batch_fn(b) for b in batches))

    agg = MeanAggregator()
    assert agg.name == 'mean'
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_more_than_one_epoch(runner):
    batches = range(5)
    values = [12, 7, 8, 44, -13, 78, 55, -109, 34, 10]
    epoch = 0
    batch_fn = lambda b: (epoch - 1) * len(batches) + values[b]

    def eshandler(state):
        nonlocal epoch
        epoch = state['epoch']

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(batch_fn(b) for b in batches))

    agg = MeanAggregator()
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_STARTED, eshandler)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches, max_epoch=2)


def test_custom_name(runner):
    batches = range(10)
    batch_fn = lambda b: b**2

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(batch_fn(b) for b in batches))

    agg = MeanAggregator(name='avg')
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_value_fn(runner):
    batches = range(10)
    batch_fn = lambda b: (b**2, b**3)
    value_fn = lambda state: state['output'][1]

    def efhandler(state):
        assert state[agg.name] == pytest.approx(stat.mean(b**3 for b in batches))

    agg = MeanAggregator(value_fn=value_fn)
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)


def test_size_fn(runner):
    batches = range(5)
    sizes = [3, 4, 9, 10, 2]
    batch_fn = lambda b: b
    size_fn = lambda state: sizes[state['batch']]

    def efhandler(state):
        assert state[agg.name] == pytest.approx(sum(batch_fn(b) for b in batches) / sum(sizes))

    agg = MeanAggregator(size_fn=size_fn)
    agg.attach_on(runner)
    runner.append_handler(Event.EPOCH_FINISHED, efhandler)
    runner.run(batch_fn, batches)
