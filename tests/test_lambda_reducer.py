from functools import reduce

from rnnr import Event
from rnnr.attachments import LambdaReducer


def test_ok(runner):
    outputs = [4, 2, 1, 5, 6]
    batches = range(len(outputs))

    @runner.on(Event.BATCH)
    def on_batch(state):
        state["output"] = outputs[state["batch"]]

    r = LambdaReducer("product", lambda x, y: x * y)
    r.attach_on(runner)
    runner.run(batches)

    assert r.name == "product"
    assert runner.state["product"] == reduce(lambda x, y: x * y, outputs)


def test_value(runner):
    outputs = [4, 2, 1, 5, 6]
    batches = range(len(outputs))

    @runner.on(Event.BATCH)
    def on_batch(state):
        state["value"] = outputs[state["batch"]]

    r = LambdaReducer("product", lambda x, y: x * y, value="value")
    r.attach_on(runner)
    runner.run(batches)

    assert runner.state["product"] == reduce(lambda x, y: x * y, outputs)
