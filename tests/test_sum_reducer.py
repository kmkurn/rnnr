import pytest
from rnnr import Runner
from rnnr.attachments import SumReducer


def test_correct():
    outputs = [4, 2, 1, 5, 6]
    runner = Runner(lambda e, i, b: outputs[b])
    r = SumReducer(value=lambda x: x / 2)
    r.attach_on(runner)
    runner.run(range(len(outputs)))

    assert r.result == pytest.approx(sum(x / 2 for x in outputs))
