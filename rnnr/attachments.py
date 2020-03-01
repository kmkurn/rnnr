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

from datetime import timedelta
from typing import Any, Callable, Optional, Type
import abc
import logging
import time

from tqdm import tqdm

from .event import Event
from .runner import Runner


class Attachment(abc.ABC):
    """An abstract base class for an attachment."""

    @abc.abstractmethod
    def attach_on(self, runner: Runner) -> None:
        """Attach to a runner.

        Args:
            runner: Runner to attach to.
        """
        pass


class EpochTimer(Attachment):  # pragma: no cover
    """An attachment to time epoch.

    Epochs are only timed when ``state['max_epoch']`` is greater than 1. At the start and
    end of every epoch, logging messages are written with log level of INFO.
    """
    logger = logging.getLogger(f'{__name__}.epoch_timer')

    def __init__(self):
        self._epoch_start_time = 0

    def attach_on(self, runner: Runner) -> None:
        runner.on(Event._ETIMER_STARTED, self._start_timing)
        runner.on(Event._ETIMER_FINISHED, self._finish_timing)

    def _start_timing(self, state):
        if state['max_epoch'] > 1:
            self._epoch_start_time = time.time()
            self.logger.info('Starting epoch %d/%d', state['epoch'], state['max_epoch'])

    def _finish_timing(self, state):
        if state['max_epoch'] > 1:
            elapsed = timedelta(seconds=time.time() - self._epoch_start_time)
            self.logger.info(
                'Epoch %d/%d done in %s', state['epoch'], state['max_epoch'], elapsed)


class ProgressBar(Attachment):
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
            *,
            n_items: str = 'n_items',
            stats: Optional[str] = None,
            tqdm_cls: Optional[Type[tqdm]] = None,
            **kwargs,
    ) -> None:
        if tqdm_cls is None:  # pragma: no cover
            tqdm_cls = tqdm

        self._tqdm_cls = tqdm_cls
        self._n_items = n_items
        self._stats = stats
        self._kwargs = kwargs

        self._pbar: tqdm

    def attach_on(self, runner: Runner) -> None:
        runner.on(Event._PBAR_CREATED, self._create)
        runner.on(Event._PBAR_UPDATED, self._update)
        runner.on(Event._PBAR_CLOSED, self._close)

    def _create(self, state: dict) -> None:
        self._pbar = self._tqdm_cls(state['batches'], **self._kwargs)

    def _update(self, state: dict) -> None:
        if self._stats is not None:
            self._pbar.set_postfix(**state[self._stats])
        self._pbar.update(state.get(self._n_items, 1))

    def _close(self, state: dict) -> None:
        self._pbar.close()


class LambdaReducer(Attachment):
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

    def __init__(
            self,
            name: str,
            reduce_fn: Callable[[Any, Any], Any],
            *,
            value: str = 'output',
    ) -> None:
        self.name = name
        self._reduce_fn = reduce_fn
        self._value = value

    def attach_on(self, runner: Runner) -> None:
        runner.on(Event._REDUCER_RESET, self._reset)
        runner.on(Event._REDUCER_UPDATED, self._update)
        runner.on(Event._REDUCER_COMPUTED, self._compute)

    def _reset(self, state: dict) -> None:
        self._result = None

    def _update(self, state: dict) -> None:
        if self._result is None:
            self._result = state[self._value]
        else:
            self._result = self._reduce_fn(self._result, state[self._value])

    def _compute(self, state: dict) -> None:
        state[self.name] = self._result


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

    def __init__(
            self,
            name: str,
            *,
            value: str = 'output',
            size: str = 'size',
    ) -> None:
        super().__init__(name, lambda x, y: x + y, value=value)
        self._size = size
        self._total_size = 0

    def _reset(self, state: dict) -> None:
        super()._reset(state)
        self._total_size = 0

    def _update(self, state: dict) -> None:
        super()._update(state)
        self._total_size += state.get(self._size, 1)

    def _compute(self, state: dict) -> None:
        super()._compute(state)
        state[self.name] /= self._total_size
