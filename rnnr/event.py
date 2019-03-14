from enum import Enum, auto


class Event(Enum):
    STARTED = auto()
    EPOCH_STARTED = auto()
    BATCH_STARTED = auto()
    BATCH_FINISHED = auto()
    EPOCH_FINISHED = auto()
    FINISHED = auto()
