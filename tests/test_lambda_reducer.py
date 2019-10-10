from functools import reduce

from rnnr.attachments import LambdaReducer


def test_ok(runner):
    outputs = [4, 2, 1, 5, 6]
    batches = range(len(outputs))

    def batch_fn(state):
        state['output'] = outputs[state['batch']]

    r = LambdaReducer('product', lambda x, y: x * y)
    r.attach_on(runner)
    state = runner.run(batch_fn, batches)

    assert r.name == 'product'
    assert state['product'] == reduce(lambda x, y: x * y, outputs)


def test_value_key(runner):
    outputs = [4, 2, 1, 5, 6]
    batches = range(len(outputs))

    def batch_fn(state):
        state['value'] = outputs[state['batch']]

    r = LambdaReducer('product', lambda x, y: x * y, value_key='value')
    r.attach_on(runner)
    state = runner.run(batch_fn, batches)

    assert state['product'] == reduce(lambda x, y: x * y, outputs)
