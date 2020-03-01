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

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional

from .event import Event

Callback = Callable[[dict], None]


class Runner:
    """A neural network runner.

    A runner provides a thin abstraction of iterating over batches for several epochs,
    which is typically done in neural network training. To customize the behavior during
    a run, a runner provides a way to listen to events emitted during such run.
    To listen to an event, call `Runner.on` and provide a callback which will be called
    when the event is emitted. An event callback is a callable that accepts a `dict`
    and returns nothing. The `dict` is the state of the run. By default, the state contains:

    * ``batches`` - Iterable of batches which constitutes an epoch.
    * ``max_epoch`` - Maximum number of epochs to run.
    * ``n_iters`` - Current number of batch iterations.
    * ``running`` - A boolean which equals ``True`` if the runner is still running. Can
      be set to ``False`` to stop the runner earlier.
    * ``epoch`` - Current number of epoch. Not available to callbacks of `Event.FINISHED`.
    * ``batch`` - Current batch retrieved from ``state['batches']``. Only available to
      callbacks of `Event.BATCH`.

    Attributes:
        state (dict): Runner's state that is passed to event callbacks.

    Note:
        Callbacks for an event are called in the order they are passed to `~Runner.on`.

    Caution:
        All the state contents above are required for a runner to function properly.
        You are free to change their values to suit your use cases better, but be careful.
    """

    def __init__(self) -> None:
        self.state: dict = {}
        self._callbacks: Dict[Event, List[Callback]] = defaultdict(list)

    def on(self, event: Event, callbacks=None):
        """Add single/multiple callback(s) to listen to an event.

        If ``callbacks`` is ``None``, this method returns a decorator which accepts
        a single callback for the event. If ``callbacks`` is a sequence of callbacks,
        they will all be added as listeners to the event *in order*.

        Args:
            event: Event to listen.
            callbacks: Callback(s) for the event.

        Returns:
            A decorator which accepts a callback, if ``callbacks`` is ``None``.
        """
        if callbacks is not None:
            cblist = self._callbacks[event]
            try:
                cblist.extend(callbacks)
            except TypeError:  # must be a single callback
                cblist.append(callbacks)
            return

        def decorator(cb: Callback) -> Callback:
            self._callbacks[event].append(cb)
            return cb

        return decorator

    def run(self, batches: Optional[Iterable[Any]] = None, max_epoch: int = 1) -> None:
        """Run on the given batches for a number of epochs.

        Args:
            batches: Batches to iterate over in an epoch.
            max_epoch: Maximum number of epochs to run.
        """
        state = self.state
        state.update({
            'max_epoch': max_epoch,
            'batches': batches,
            'n_iters': 0,
            'running': True,
            'epoch': 0,
        })

        self._emit(Event.STARTED, state)

        while state['running'] and state['epoch'] < state['max_epoch']:
            state['epoch'] += 1
            self._emit(Event._ETIMER_STARTED, state)
            self._emit(Event.EPOCH_STARTED, state)
            self._emit(Event._REDUCER_RESET, state)
            self._emit(Event._PBAR_CREATED, state)

            state['batches_iter'] = iter(state['batches'])
            while state['running']:
                try:
                    batch = next(state['batches_iter'])
                except StopIteration:
                    break
                state['n_iters'] += 1
                state['batch'] = batch
                self._emit(Event.BATCH, state)
                self._emit(Event._REDUCER_UPDATED, state)
                self._emit(Event._PBAR_UPDATED, state)
            if state['running']:
                state.pop('batch', None)
                state.pop('batches_iter', None)

            self._emit(Event._PBAR_CLOSED, state)
            self._emit(Event._REDUCER_COMPUTED, state)
            self._emit(Event.EPOCH_FINISHED, state)
            self._emit(Event._ETIMER_FINISHED, state)

        if state['running']:
            state.pop('epoch', None)
        self._emit(Event.FINISHED, state)
        state['running'] = False

    def resume(self) -> None:
        state = self.state
        state['running'] = True

        # finish interrupted epoch
        while state['running']:
            try:
                batch = next(state.get('batches_iter', iter([])))
            except StopIteration:
                break
            state['n_iters'] += 1
            state['batch'] = batch
            self._emit(Event.BATCH, state)
            self._emit(Event._REDUCER_UPDATED, state)
            self._emit(Event._PBAR_UPDATED, state)
        if state['running']:
            state.pop('batch', None)
            state.pop('batches_iter', None)

        while state['running'] and state['epoch'] < state['max_epoch']:
            state['epoch'] += 1
            self._emit(Event._ETIMER_STARTED, state)
            self._emit(Event.EPOCH_STARTED, state)
            self._emit(Event._REDUCER_RESET, state)
            self._emit(Event._PBAR_CREATED, state)

            state['batches_iter'] = iter(state['batches'])
            while state['running']:
                try:
                    batch = next(state['batches_iter'])
                except StopIteration:
                    break
                state['n_iters'] += 1
                state['batch'] = batch
                self._emit(Event.BATCH, state)
                self._emit(Event._REDUCER_UPDATED, state)
                self._emit(Event._PBAR_UPDATED, state)
            if state['running']:
                state.pop('batch', None)
                state.pop('batches_iter', None)

            self._emit(Event._PBAR_CLOSED, state)
            self._emit(Event._REDUCER_COMPUTED, state)
            self._emit(Event.EPOCH_FINISHED, state)
            self._emit(Event._ETIMER_FINISHED, state)

        if state['running']:
            state.pop('epoch', None)
        self._emit(Event.FINISHED, state)
        state['running'] = False

    def _emit(self, event: Event, state: dict) -> None:
        for callback in self._callbacks[event]:
            if not state['running']:
                break
            callback(state)
