import logging

from rnnr import EpochId
from rnnr.batch import BatchOutput
from rnnr.progress_tracker import LoggingEpochProgressTracker


def test_log_batch_output():
    logger = logging.getLogger("rnnr.progress_tracker")
    logger.setLevel(logging.INFO)
    history = []

    class AppendToHistoryHandler(logging.Handler):
        def emit(self, record):
            history.append(record.getMessage())

    class FakeBatchOutput(BatchOutput):
        @property
        def loss(self):
            return 0.234789

        @property
        def stats(self):
            return {"foo": 3, "bar": 7.54321}

    logger.addHandler(AppendToHistoryHandler())
    tracker = LoggingEpochProgressTracker(num_batches=10, log_every=1)

    with tracker.start(EpochId(1)) as log:
        log(FakeBatchOutput())
        log(FakeBatchOutput())

    assert history == [
        "Epoch 1 [ 1/10]: loss=0.2348 foo=3 bar=7.5432",
        "Epoch 1 [ 2/10]: loss=0.2348 foo=3 bar=7.5432",
    ]
