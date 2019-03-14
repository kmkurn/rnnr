from typing import Any, Callable, Optional, Type
import abc

from tqdm import tqdm

from . import Event, Runner


class Attachment(abc.ABC):
    @abc.abstractmethod
    def attach_on(self, runner: Runner) -> None:
        pass


class ProgressBar(Attachment):
    """An attachment to display a progress bar.

    The progress bar is implemented using `tqdm`_.

    Example:

        >>> from rnnr import Runner
        >>> from rnnr.attachments import ProgressBar
        >>> runner = Runner()
        >>> ProgressBar().attach_on(runner)
        >>> runner.run(lambda x: x, range(10), max_epoch=10)

    Args:
        size_fn: Function to get the size of a batch to update the progress bar with.
            If not given, the default is to always return 1 as the size of a batch.
        stats_fn: Function to get the statistics dictionary to be displayed along with the
            progress bar. If given, it should accept a runner's state dictionary and return
            another dictionary whose keys are the names of the statistics and the values are
            the statistics values.
        **kwargs: Keyword arguments to be passed to `tqdm`_ class.


    .. _tqdm: https://github.com/tqdm/tqdm
    """

    def __init__(
            self,
            size_fn: Optional[Callable[[dict], int]] = None,
            stats_fn: Optional[Callable[[dict], dict]] = None,
            tqdm_cls: Optional[Type[tqdm]] = None,
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
        """Attach this progress bar to the given runner.

        Args:
            runner: Runner to attach this progress bar to.
        """
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
            value_fn: Optional[Callable[[dict], Any]] = None,
            size_fn: Optional[Callable[[dict], int]] = None,
    ) -> None:
        if value_fn is None:
            value_fn = lambda state: state['output']
        if size_fn is None:
            size_fn = lambda _: 1

        self.name = name
        self._value_fn = value_fn
        self._size_fn = size_fn

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
        self._total += self._value_fn(state)
        self._size += self._size_fn(state)

    def _compute(self, state: dict) -> None:
        state[self.name] = self._total / self._size
