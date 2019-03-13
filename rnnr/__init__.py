__version__ = '0.0.0'

from collections import defaultdict
from enum import Enum, auto
from typing import Callable, Dict, Generic, Iterable, List, TypeVar

BatchT = TypeVar('BatchT')
OutputT = TypeVar('OutputT')


class Event(Enum):
    EPOCH_STARTED = auto()
    BATCH_STARTED = auto()
    BATCH_FINISHED = auto()
    EPOCH_FINISHED = auto()


class Runner(Generic[BatchT, OutputT]):
    """A neural network runner."""

    def __init__(self) -> None:
        self._handlers: Dict[Event, List['Handler']] = defaultdict(list)
        self._running = False

    def append_handler(self, event: Event, handler: 'Handler') -> None:
        """Append a handler for the given event.

        Args:
            event: Event to handle.
            handler: Handler for the event.
        """
        self._handlers[event].append(handler)

    def on(self, event: Event) -> Callable[['Handler'], 'Handler']:
        def decorator(handler: 'Handler') -> 'Handler':
            self.append_handler(event, handler)
            return handler

        return decorator

    def run(
            self,
            batch_fn: Callable[[BatchT], OutputT],
            batches: Iterable[BatchT],
            max_epoch: int = 1,
    ) -> None:
        """Run on the given batches for a number of epochs.

        Args:
            batch_fn: Function to call for each batch.
            batches: Batches to iterate over in an epoch.
            max_epoch: Maximum number of epochs to run.
        """
        self._running = True
        state: dict = {
            'epoch': None,
            'max_epoch': max_epoch,
            'batches': batches,
            'batch': None,
            'output': None,
        }

        for epoch in range(1, max_epoch + 1):
            if not self._running:
                break

            state['epoch'] = epoch
            self._emit(Event.EPOCH_STARTED, state)

            for batch in batches:
                if not self._running:
                    break
                state['batch'] = batch
                self._emit(Event.BATCH_STARTED, state)
                output = batch_fn(batch)
                state['output'] = output
                self._emit(Event.BATCH_FINISHED, state)
                state['output'] = None

            state['batch'] = None
            self._emit(Event.EPOCH_FINISHED, state)

    def _emit(self, event: Event, state: dict) -> None:
        for handler in self._handlers[event]:
            handler(state)

    def stop(self) -> None:
        """Stop the runner immediately after the current batch is finished."""
        self._running = False


from .handlers import Handler  # avoid circular import
