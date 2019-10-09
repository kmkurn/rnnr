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

from typing import Optional, Type
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
        >>> _ = runner.run(lambda _: None, range(10), max_epoch=10)

    Args:
        size_key: Key to get the size of a batch from the runner's state to update the
            progress bar with. If not given, the default is to always set 1 as the size of
            a batch.
        stats_key: Key to get the statistics dictionary from the runner's state to be
            displayed along with the progress bar. The statistics dictionary has the names
            of the statistics as keys and the statistics as values.
        **kwargs: Keyword arguments to be passed to `tqdm`_ class.


    .. _tqdm: https://github.com/tqdm/tqdm
    """
    def __init__(
            self,
            size_key: str = 'size',
            stats_key: Optional[str] = None,
            tqdm_cls: Optional[Type[tqdm]] = None,
            **kwargs,
    ) -> None:
        if tqdm_cls is None:  # pragma: no cover
            tqdm_cls = tqdm

        self._tqdm_cls = tqdm_cls
        self._size_key = size_key
        self._stats_key = stats_key
        self._kwargs = kwargs

        self._pbar: tqdm

    def attach_on(self, runner: Runner) -> None:
        runner.append_handler(Event.EPOCH_STARTED, self._create)
        runner.append_handler(Event.BATCH_FINISHED, self._update)
        runner.append_handler(Event.EPOCH_FINISHED, self._close)

    def _create(self, state: dict) -> None:
        self._pbar = self._tqdm_cls(state['batches'], **self._kwargs)

    def _update(self, state: dict) -> None:
        if self._stats_key is not None:
            self._pbar.set_postfix(**state[self._stats_key])
        self._pbar.update(state.get(self._size_key, 1))

    def _close(self, state: dict) -> None:
        self._pbar.close()


class Reducer(Attachment):
    """An abstract attachment to compute reduction over batches.

    This attachment gets the value of each batch and reduce them. This
    class is meant to be subclassed by others.

    Args:
        value_key: Key to get the value of a batch from the runner's
            state.

    TODO: complete docstring
    """
    def __init__(self, value_key: str = 'output') -> None:
        self._value_key = value_key

    @abc.abstractmethod
    def reduce(self, x, y):
        pass

    def attach_on(self, runner: Runner) -> None:
        runner.append_handler(Event.EPOCH_STARTED, self._reset)
        runner.append_handler(Event.BATCH_FINISHED, self._update)
        runner.append_handler(Event.EPOCH_FINISHED, self._compute)

    def _reset(self, state: dict) -> None:
        self._total = None

    def _update(self, state: dict) -> None:
        if self._total is None:
            self._total = state[self._value_key]
        else:
            self._total = self.reduce(self._total, state[self._value_key])

    def _compute(self, state: dict) -> None:
        state[self.name] = self._total


class MeanAggregator(Attachment):
    """An attachment to compute a mean over batch statistics.

    This attachment gets the value from each batch and compute their mean at the end of
    every epoch.

    Example:

        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import MeanAggregator
        >>> runner = Runner()
        >>> MeanAggregator().attach_on(runner)
        >>> def batch_fn(state):
        ...     state['output'] = state['batch']
        ...
        >>> runner.run(batch_fn, [1, 2, 3])
        {'max_epoch': 1, 'batches': [1, 2, 3], 'output': 3, 'mean': 2.0}

    Args:
        name: Name of this aggregator. This name is used as the key in the runner's state
            dictionary.
        value_key: Key to get the value of a batch from the runner's state.
        size_key: Key to get the size of a batch from the runner's state. If the state has
            no such key, the size defaults to 1. The sum of all these batch sizes is the
            divisor when computing the mean.
    """
    def __init__(
            self,
            name: str = 'mean',
            value_key: str = 'output',
            size_key: str = 'size',
    ) -> None:
        self.name = name
        self._value_key = value_key
        self._size_key = size_key

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
        self._total += state[self._value_key]
        self._size += state.get(self._size_key, 1)

    def _compute(self, state: dict) -> None:
        state[self.name] = self._total / self._size
