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

from typing import Any, Callable, Optional, Type
import abc

from tqdm import tqdm

from .event import Event
from .runner import Runner


class Attachment(abc.ABC):
    """An abstract base class for an attachment."""

    @abc.abstractmethod
    def attach_on(self, runner: Runner) -> None:
        """Attach to the given runner.

        Args:
            runner: Runner to attach to.
        """
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
    """An attachment to compute a mean over batch statistics.

    This attachment gets the value from each batch and compute their mean at the end of
    every epoch.

    Example:

        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import MeanAggregator
        >>> runner = Runner()
        >>> agg = MeanAggregator()
        >>> agg.attach_on(runner)
        >>> @runner.on(Event.EPOCH_FINISHED)
        ... def print_mean(state):
        ...     print('Mean:', state[agg.name])
        ...
        >>> runner.run(lambda x: x, range(5), max_epoch=3)
        Mean: 2.0
        Mean: 2.0
        Mean: 2.0

    Args:
        name: Name of this aggregator. This name is used as the key in the runner's state
            dictionary.
        value_fn: Function to get the value of a batch. If given, it should accept the
            runner's state dictionary at the end of a batch and return a value. The default
            is to get ``state['output']`` as the value.
        size_fn: Function to get the size of a batch. If given, it should accept the runner's
            state dictionary at the end of a batch and return the batch size. The default is
            to always return 1 as the batch size. The sum of all these batch sizes is the
            divisor when computing the mean.
    """

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
