from functools import reduce

import pytest
from rnnr import Runner
from rnnr.attachments import LambdaReducer


def test_correct():
    outputs = [4, 2, 1, 5, 6]
    runner = Runner(lambda e, bi, b: outputs[b])
    r = LambdaReducer(lambda x, y: x * y, value=lambda x: x / 2)
    r.attach_on(runner)
    runner.run(range(len(outputs)))

    assert r.result == pytest.approx(reduce(lambda x, y: x * y, [x / 2 for x in outputs]))
