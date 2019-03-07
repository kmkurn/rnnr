__version__ = '0.0.0'

from collections import defaultdict
from enum import Enum, auto
from typing import Callable, Dict, Iterable, List


class Event(Enum):
    EPOCH_STARTED = auto()
    BATCH_STARTED = auto()
    BATCH_FINISHED = auto()
    EPOCH_FINISHED = auto()


class Runner:
    def __init__(self) -> None:
        self._handlers: Dict[Event, List[Callable]] = defaultdict(list)

    def append_handler(self, event: Event, handler: Callable) -> None:
        self._handlers[event].append(handler)

    # Cannot use generics for batch_fn type because it accepts keyword arguments
    def run(self, batch_fn: Callable, batches: Iterable, max_epoch: int = 1) -> None:
        state = {
            'epoch': None,
            'max_epoch': max_epoch,
            'batches': batches,
            'batch': None,
            'output': None,
        }

        for epoch in range(1, max_epoch + 1):
            state['epoch'] = epoch
            self._emit(Event.EPOCH_STARTED, state)
            for batch in batches:
                state['batch'] = batch
                self._emit(Event.BATCH_STARTED, state)
                output = batch_fn(batch)
                state['output'] = output
                self._emit(Event.BATCH_FINISHED, state)
                state['output'] = None
            state['batch'] = None
            self._emit(Event.EPOCH_FINISHED, state)

    def _emit(self, event: Event, *args, **kwargs) -> None:
        for handler in self._handlers[event]:
            handler(*args, **kwargs)
