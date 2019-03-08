__version__ = '0.0.0'

from collections import defaultdict
from enum import Enum, auto
from typing import Callable, Dict, Generic, Iterable, List, Optional, Type, TypeVar

from tqdm import tqdm

BatchT = TypeVar('BatchT')
OutputT = TypeVar('OutputT')
Handler = Callable[[dict], None]


class Event(Enum):
    EPOCH_STARTED = auto()
    BATCH_STARTED = auto()
    BATCH_FINISHED = auto()
    EPOCH_FINISHED = auto()


class Runner(Generic[BatchT, OutputT]):
    def __init__(self) -> None:
        self._handlers: Dict[Event, List[Handler]] = defaultdict(list)
        self._running = False

    def append_handler(self, event: Event, handler: Handler) -> None:
        self._handlers[event].append(handler)

    def run(
            self,
            batch_fn: Callable[[BatchT], OutputT],
            batches: Iterable[BatchT],
            max_epoch: int = 1,
    ) -> None:
        self._running = True
        state: dict = {
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
                if not self._running:
                    break

            state['batch'] = None
            self._emit(Event.EPOCH_FINISHED, state)
            if not self._running:
                break

    def _emit(self, event: Event, state: dict) -> None:
        for handler in self._handlers[event]:
            handler(state)

    def stop(self) -> None:
        self._running = False


def attach_pbar_on(
        runner: Runner,
        update_size: Optional[Callable[[dict], int]] = None,
        stats: Optional[Callable[[dict], dict]] = None,
        tqdm_cls: Optional[Type[tqdm]] = None,
        **kwargs,
) -> None:
    if update_size is None:
        update_size = lambda _: 1
    if stats is None:
        stats = lambda state: {'output': state['output']}
    if tqdm_cls is None:  # pragma: no cover
        tqdm_cls = tqdm

    pbar = None

    def create_pbar(state):
        nonlocal pbar
        pbar = tqdm_cls(state['batches'], **kwargs)

    def update_pbar(state):
        pbar.set_postfix(**stats(state))
        pbar.update(update_size(state))

    def close_pbar(state):
        pbar.close()

    runner.append_handler(Event.EPOCH_STARTED, create_pbar)
    runner.append_handler(Event.BATCH_FINISHED, update_pbar)
    runner.append_handler(Event.EPOCH_FINISHED, close_pbar)
