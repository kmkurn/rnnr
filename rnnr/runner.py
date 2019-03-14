from collections import defaultdict
from datetime import timedelta
from typing import Callable, Dict, Generic, Iterable, List, TypeVar
import time
import logging

from rnnr.event import Event

BatchT = TypeVar('BatchT')
OutputT = TypeVar('OutputT')
logger = logging.getLogger(__name__)


class Runner(Generic[BatchT, OutputT]):
    """A neural network runner.

    A runner provides a thin abstraction of iterating over batches for several epochs,
    which is typically done in neural network training. To customize the behavior during
    a run, a runner provides a way to listen to events emitted during such run. There are
    six events that may be emitted during a run:

    * `Event.STARTED` - emitted once at the start of the run.
    * `Event.EPOCH_STARTED` - emitted at the start of each epoch.
    * `Event.BATCH_STARTED` - emitted at the start of each batch.
    * `Event.BATCH_FINISHED` - emitted when a batch is finished.
    * `Event.EPOCH_FINISHED` - emitted when an epoch is finished.
    * `Event.FINISHED` - emitted once when the run is finished.

    To listen to an event, call `Runner.append_handler` and provide the event handler.
    A handler is a callable that accepts a `dict` and returns nothing. The `dict` is
    the state of the run. By default, the state contains:

    * ``batches`` - iterable of batches which constitutes an epoch.
    * ``max_epoch`` - maximum number of epochs to run.
    * ``epoch`` - current number of epoch. Not available to handlers of `Event.STARTED`
        and `Event.FINISHED`.
    * ``batch`` - current batch retrieved from ``state['batches']``. Only available to
        handlers of `Event.BATCH_STARTED` and `Event.BATCH_FINISHED`.
    * ``output`` - output of processing the current batch. Only available to handlers of
        `Event.BATCH_FINISHED`.
    """

    def __init__(self) -> None:
        self._handlers: Dict[Event, List['Handler']] = defaultdict(list)
        self._running = False
        self._epoch_start_time = 0.

        self.append_handler(Event.EPOCH_STARTED, self._print_start_epoch)
        self.append_handler(Event.EPOCH_FINISHED, self._print_finish_epoch)

    def append_handler(self, event: Event, handler: 'Handler') -> None:
        """Append a handler for the given event.

        Args:
            event: Event to handle.
            handler: Handler for the event.
        """
        self._handlers[event].append(handler)

    def _print_start_epoch(self, state: dict) -> None:
        if state['max_epoch'] > 1:
            self._epoch_start_time = time.time()
            logger.info('Starting epoch %d/%d', state['epoch'], state['max_epoch'])

    def _print_finish_epoch(self, state: dict) -> None:
        if state['max_epoch'] > 1:
            elapsed = timedelta(seconds=time.time() - self._epoch_start_time)
            logger.info('Epoch %d/%d done in %s', state['epoch'], state['max_epoch'], elapsed)

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
        state: dict = {'max_epoch': max_epoch, 'batches': batches}

        self._emit(Event.STARTED, state)
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
                state.pop('output')

            state.pop('batch', None)
            self._emit(Event.EPOCH_FINISHED, state)
        state.pop('epoch', None)
        self._emit(Event.FINISHED, state)

    def _emit(self, event: Event, state: dict) -> None:
        for handler in self._handlers[event]:
            handler(state)

    def stop(self) -> None:
        """Stop the runner immediately after the current batch is finished.

        Note that the appropriate handlers for ``Event.*_FINISHED`` events are still called
        before the run truly stops.
        """
        self._running = False


from .handlers import Handler  # avoid circular import