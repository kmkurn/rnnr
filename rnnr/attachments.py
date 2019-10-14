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
        >>> _ = runner.run(lambda _: None, range(10), max_epoch=10)

    Args:
        size_key: Get the size of a batch from ``state[size_key]`` to update the
            progress bar with. If not given, the default is to always set 1 as the size of
            a batch.
        stats_key: Get the batch statistics from ``state[stats_key]`` and display it along
            with the progress bar. The statistics dictionary has the names of the statistics
            as keys and the statistics as values.
        **kwargs: Keyword arguments to be passed to `tqdm`_ class.


    .. _tqdm: https://github.com/tqdm/tqdm
    """
    def __init__(
            self,
            size_key: str = 'n_items',
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


class LambdaReducer(Attachment):
    """An attachment to compute a reduction over batches.

    This attachment gets the value of each batch and compute a reduction over them
    at the end of each epoch.

    Example:

        >>> from rnnr import Runner
        >>> from rnnr.attachments import LambdaReducer
        >>> runner = Runner()
        >>> LambdaReducer('product', lambda x, y: x * y).attach_on(runner)
        >>> def batch_fn(state):
        ...     state['output'] = state['batch']
        ...
        >>> state = runner.run(batch_fn, [10, 20, 30])
        >>> state['product']
        6000

    Args:
        name: Name of this attachment to be used as the key in the runner's
            state dict to store the reduction result.
        reduce_fn: Reduction function. It should accept two batch values and
            return their reduction result.
        value_key: Get the value of a batch from ``state[value_key]``.
    """
    def __init__(
            self,
            name: str,
            reduce_fn: Callable[[Any, Any], Any],
            value_key: str = 'output',
    ) -> None:
        self.name = name
        self._reduce_fn = reduce_fn
        self._value_key = value_key

    def attach_on(self, runner: Runner) -> None:
        runner.append_handler(Event.EPOCH_STARTED, self._reset)
        runner.append_handler(Event.BATCH_FINISHED, self._update)
        runner.append_handler(Event.EPOCH_FINISHED, self._compute)

    def _reset(self, state: dict) -> None:
        self._result = None

    def _update(self, state: dict) -> None:
        if self._result is None:
            self._result = state[self._value_key]
        else:
            self._result = self._reduce_fn(self._result, state[self._value_key])

    def _compute(self, state: dict) -> None:
        state[self.name] = self._result


class MeanReducer(LambdaReducer):
    """An attachment to compute a mean over batch statistics.

    This attachment gets the value from each batch and compute their mean at the end of
    every epoch.

    Example:

        >>> from rnnr import Runner
        >>> from rnnr.attachments import MeanReducer
        >>> runner = Runner()
        >>> MeanReducer().attach_on(runner)
        >>> def batch_fn(state):
        ...     state['output'] = state['batch']
        ...
        >>> state = runner.run(batch_fn, [1, 2, 3])
        >>> state['mean']
        2.0

    Args:
        name: Name of this attachment to be used as the key in the runner's state
            dict to store the mean value.
        value_key: Get the value of a batch from ``state[value_key]``.
        size_key: Get the size of a batch from ``state[size_key]``. If
            the state has no such key, the size defaults to 1. The sum of all these
            batch sizes is the divisor when computing the mean.
    """
    def __init__(
            self,
            name: str = 'mean',
            value_key: str = 'output',
            size_key: str = 'size',
    ) -> None:
        super().__init__(name, lambda x, y: x + y, value_key=value_key)
        self._size_key = size_key
        self._size = 0

    def _reset(self, state: dict) -> None:
        super()._reset(state)
        self._size = 0

    def _update(self, state: dict) -> None:
        super()._update(state)
        self._size += state.get(self._size_key, 1)

    def _compute(self, state: dict) -> None:
        super()._compute(state)
        state[self.name] /= self._size
