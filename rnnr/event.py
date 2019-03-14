from enum import Enum, auto


class Event(Enum):
    """An enumeration of events.

    Attributes:
        STARTED: Emitted once at the start of a run.
        EPOCH_STARTED: Emitted at the start of each epoch.
        BATCH_STARTED: Emitted at the start of each batch.
        BATCH_FINISHED: Emitted every time a batch is finished.
        EPOCH_FINISHED: Emitted every time an epoch is finished.
        FINISHED: Emitted once when a run is finished.
    """
    STARTED = auto()
    EPOCH_STARTED = auto()
    BATCH_STARTED = auto()
    BATCH_FINISHED = auto()
    EPOCH_FINISHED = auto()
    FINISHED = auto()
