# Copyright 2019 Kemal Kurniawan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import logging
import time
from datetime import timedelta
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from tqdm import tqdm

from .event import Event
from .runner import BatchIndex, EpochId, Runner, StopFn

OT = TypeVar("OT")
RT = TypeVar("RT")


class Attachment(abc.ABC, Generic[OT]):
    """An abstract base class for an attachment."""

    @abc.abstractmethod
    def attach_on(self, runner: Runner[OT]) -> None:
        """Attach to a runner.

        Args:
            runner: Runner to attach to.
        """
        pass


class EpochTimer(Attachment[OT]):
    """An attachment to time epoch.

    Epochs are only timed when ``state['max_epoch']`` is greater than 1. At the start and
    end of every epoch, logging messages are written with log level of INFO.
    """

    _logger = logging.getLogger(f"{__name__}.epoch_timer")

    def __init__(
        self,
        start_fmt: str = "Starting epoch {epoch}/{max_epoch}",
        finish_fmt: str = "Epoch {epoch}/{max_epoch} done in {elapsed}",
    ) -> None:
        self._start_fmt = start_fmt
        self._finish_fmt = finish_fmt

    def attach_on(self, runner: Runner[OT]) -> None:
        if runner.max_epoch > 1:
            self._max_epoch = runner.max_epoch
            runner.set_first_on_epoch_started(self._start)
            runner.set_last_on_epoch_finished(self._finish)

    def _start(self, e: EpochId):
        self._start_time = time.time()
        msg = self._start_fmt.format(epoch=e, max_epoch=self._max_epoch)
        self._logger.info(msg)

    def _finish(self, e: EpochId):
        elapsed = timedelta(seconds=time.time() - self._start_time)
        msg = self._finish_fmt.format(epoch=e, max_epoch=self._max_epoch, elapsed=elapsed)
        self._logger.info(msg)


class ProgressBar(Attachment[OT]):
    """An attachment to display a progress bar.

    The progress bar is implemented using `tqdm`_.

    Example:

        >>> from rnnr import Runner
        >>> from rnnr.attachments import ProgressBar
        >>> runner = Runner()
        >>> ProgressBar().attach_on(runner)
        >>> runner.run(range(10), max_epoch=10)

    Args:
        n_items: Get the number of items in a batch from ``state[n_items]`` to update the
            progress bar with. If not given, the default is to always set it to 1.
        stats: Get the batch statistics from ``state[stats]`` and display it along
            with the progress bar. The statistics dictionary has the names of the statistics
            as keys and the statistics as values.
        **kwargs: Keyword arguments to be passed to `tqdm`_ class.


    .. _tqdm: https://github.com/tqdm/tqdm
    """

    def __init__(
        self,
        make_pbar: Callable[[EpochId], tqdm],
        get_num_items: Optional[Callable[[Any], int]] = None,
        get_stats: Optional[Callable[[OT], Dict[str, Any]]] = None,
    ) -> None:
        if get_num_items is None:
            get_num_items = lambda _: 1
        if get_stats is None:
            get_stats = lambda _: {}

        self._make_pbar = make_pbar
        self._get_n = get_num_items
        self._get_stats = get_stats

        self._pbar: tqdm

    def attach_on(self, runner: Runner[OT]) -> None:
        runner.set_last_on_epoch_started(self._create)
        runner.set_last_on_batch_finished(self._update)
        runner.set_first_on_epoch_finished(self._close)

    def _create(self, e: EpochId) -> None:
        self._pbar = self._make_pbar(e)

    def _update(self, e: EpochId, i: BatchIndex, b: Any, o: OT) -> None:
        self._pbar.set_postfix(**self._get_stats(o))
        self._pbar.update(self._get_n(b))

    def _close(self, e: EpochId) -> None:
        self._pbar.close()


class LambdaReducer(Attachment[OT], Generic[OT, RT]):
    """An attachment to compute a reduction over batches.

    This attachment gets the value of each batch and compute a reduction over them
    at the end of each epoch.

    Example:

        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import LambdaReducer
        >>> runner = Runner()
        >>> LambdaReducer('product', lambda x, y: x * y).attach_on(runner)
        >>> @runner.on(Event.BATCH)
        ... def on_batch(state):
        ...     state['output'] = state['batch']
        ...
        >>> runner.run([10, 20, 30])
        >>> runner.state['product']
        6000

    Args:
        name: Name of this attachment to be used as the key in the runner's
            state dict to store the reduction result.
        reduce_fn: Reduction function. It should accept two batch values and
            return their reduction result.
        value: Get the value of a batch from ``state[value]``.
    """

    def __init__(self, lambda_: Callable[[RT, RT], RT], value: Callable[[OT], RT]) -> None:
        self._lambda = lambda_
        self._get_value = value
        self.result: Any = None

    def attach_on(self, runner: Runner[OT]) -> None:
        runner.on_epoch_started(self._reset)
        runner.on_batch_finished(self._update)

    def _reset(self, e: EpochId) -> None:
        self.result = None

    def _update(self, e: EpochId, bi: BatchIndex, b: Any, bo: OT) -> None:
        val = self._get_value(bo)
        self.result = val if self.result is None else self._lambda(self.result, val)


class MeanReducer(LambdaReducer):
    """An attachment to compute a mean over batch statistics.

    This attachment gets the value from each batch and compute their mean at the end of
    every epoch.

    Example:

        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import MeanReducer
        >>> runner = Runner()
        >>> MeanReducer('mean').attach_on(runner)
        >>> @runner.on(Event.BATCH)
        ... def on_batch(state):
        ...     state['output'] = state['batch']
        ...
        >>> runner.run([1, 2, 3])
        >>> runner.state['mean']
        2.0

    Args:
        name: Name of this attachment to be used as the key in the runner's state
            dict to store the mean value.
        value: Get the value of a batch from ``state[value]``.
        size: Get the size of a batch from ``state[size]``. If the state has no such key,
            the size defaults to 1. The sum of all these batch sizes is the divisor when
            computing the mean.
    """

    def __init__(self, name: str, *, value: str = "output", size: str = "size",) -> None:
        super().__init__(name, lambda x, y: x + y, value=value)
        self._size = size

    @property
    def _total_size(self) -> str:
        return f"_{self.name}_reducer_total_size"

    def _reset(self, state: dict) -> None:
        super()._reset(state)
        state[self._total_size] = 0

    def _update(self, state: dict) -> None:
        super()._update(state)
        state[self._total_size] += state.get(self._size, 1)

    def _compute(self, state: dict) -> None:
        super()._compute(state)
        state[self.name] /= state.pop(self._total_size)


class SumReducer(LambdaReducer[OT, RT]):
    """An attachment to compute a sum over batch statistics.

    This attachment gets the value from each batch and compute their sum at the end of
    every epoch.

    Args:
        name: Name of this attachment to be used as the key in the runner's state
            dict to store the mean value.
        value: Get the value of a batch from ``state[value]``.
    """

    def __init__(self, value: Callable[[OT], RT]) -> None:
        super().__init__(lambda x, y: x + y, value)  # type: ignore


class EarlyStopper(Attachment):
    def __init__(
        self, should_reduce_patience: Callable[[EpochId], bool], patience: int
    ) -> None:
        self._should_reduce_patience = should_reduce_patience
        self._orig_patience = patience

    def attach_on(self, runner: Runner[OT]) -> None:
        runner.on_started(self._reset)
        runner.on_epoch_finished(self._maybe_stop_early)

    def _reset(self) -> None:
        self._curr_patience = self._orig_patience

    def _maybe_stop_early(self, e: EpochId, stop: StopFn) -> None:
        if self._should_reduce_patience(e):
            self._curr_patience -= 1
        else:
            self._reset()
        if self._curr_patience < 0:
            stop()
