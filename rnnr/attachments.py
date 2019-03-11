from typing import Any, Callable, Optional, Type
import abc

from tqdm import tqdm

from . import Event, Runner


class Attachment(abc.ABC):
    @abc.abstractmethod
    def attach_on(self, runner: Runner) -> None:
        pass


class ProgressBar(Attachment):
    def __init__(
            self,
            tqdm_cls: Optional[Type[tqdm]] = None,
            size_fn: Optional[Callable[[dict], int]] = None,
            stats_fn: Optional[Callable[[dict], dict]] = None,
            **kwargs,
    ) -> None:
        if tqdm_cls is None:  # pragma: no cover
            tqdm_cls = tqdm
        if size_fn is None:
            size_fn = lambda _: 1
        if stats_fn is None:
            stats_fn = lambda state: {'output': state['output']}

        self._tqdm_cls = tqdm_cls
        self._size_fn = size_fn
        self._stats_fn = stats_fn
        self._kwargs = kwargs
        self._pbar: tqdm

    def attach_on(self, runner: Runner) -> None:
        runner.append_handler(Event.EPOCH_STARTED, self._create)
        runner.append_handler(Event.BATCH_FINISHED, self._update)
        runner.append_handler(Event.EPOCH_FINISHED, self._close)

    def _create(self, state: dict) -> None:
        self._pbar = self._tqdm_cls(state['batches'], **self._kwargs)

    def _update(self, state: dict) -> None:
        self._pbar.set_postfix(**self._stats_fn(state))
        self._pbar.update(self._size_fn(state))

    def _close(self, state: dict) -> None:
        self._pbar.close()


class MeanAggregator(Attachment):
    def __init__(
            self,
            name: str = 'mean',
            get_value: Optional[Callable[[dict], Any]] = None,
            get_size: Optional[Callable[[dict], int]] = None,
    ) -> None:
        if get_value is None:
            get_value = lambda state: state['output']
        if get_size is None:
            get_size = lambda _: 1

        self.name = name
        self._get_value = get_value
        self._get_size = get_size
        self._total = 0
        self._size = 0

    def attach_on(self, runner: Runner) -> None:
        runner.append_handler(Event.EPOCH_STARTED, self._reset)
        runner.append_handler(Event.BATCH_FINISHED, self._update)
        runner.append_handler(Event.EPOCH_FINISHED, self._compute)

    def _reset(self, state: dict) -> None:
        self._total = 0
        self._size = 0

    def _update(self, state: dict) -> None:
        self._total += self._get_value(state)
        self._size += self._get_size(state)

    def _compute(self, state: dict) -> None:
        state[self.name] = self._total / self._size
