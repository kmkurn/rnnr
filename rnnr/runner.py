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
from inspect import signature
from typing import Any, Callable, Dict, Generic, Iterable, List, NewType, TypeVar, Union, cast

from .event import Event

Callback = Callable[[dict], None]
EpochId = NewType("EpochId", int)
BatchIndex = NewType("BatchIndex", int)
T = TypeVar("T")


class Runner(Generic[T]):
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
    * ``epoch`` - Current number of epoch.
    * ``batch`` - Current batch retrieved from ``state['batches']``.

    Attributes:
        state (dict): Runner's state that is passed to event callbacks.

    Note:
        Callbacks for an event are called in the order they are passed to `~Runner.on`.

    Caution:
        All the state contents above are required for a runner to function properly.
        You are free to change their values to suit your use cases better, but be careful.
    """

    def __init__(
        self, on_batch: Callable[[EpochId, BatchIndex, Any], T], max_epoch: int = 1
    ) -> None:
        self.state: dict = {}
        self._on_batch = on_batch
        self._max_epoch = max_epoch
        self._callbacks: Dict[Event, List[Callback]] = defaultdict(list)
        self._callbacks_on_started: List[Callable[[], None]] = []
        self._callbacks_on_epoch_started: List[Callable[[EpochId], None]] = []
        self._callbacks_on_batch_started: List[Callable[[EpochId, BatchIndex, Any], Any]] = []
        self._callbacks_on_batch_finished: List[
            Callable[[EpochId, BatchIndex, Any, T], None]
        ] = []
        self._callbacks_on_epoch_finished: List[
            Union[Callable[[EpochId], None], Callable[[EpochId, "StopFn"], None]]
        ] = []
        self._callbacks_on_finished: List[Callable[[], None]] = []
        self.on_started(self._reset)

    def on_started(self, cb: Callable[[], None]):
        self._callbacks_on_started.append(cb)
        return cb

    def on_epoch_started(self, cb: Callable[[EpochId], None]):
        self._callbacks_on_epoch_started.append(cb)
        return cb

    def on_batch_started(self, cb: Callable[[EpochId, BatchIndex, Any], Any]):
        self._callbacks_on_batch_started.append(cb)
        return cb

    def on_batch_finished(self, cb: Callable[[EpochId, BatchIndex, Any, T], None]):
        self._callbacks_on_batch_finished.append(cb)
        return cb

    def on_epoch_finished(
        self, cb: Union[Callable[[EpochId], None], Callable[[EpochId, "StopFn"], None]],
    ):
        self._callbacks_on_epoch_finished.append(cb)
        return cb

    def on_finished(self, cb: Callable[[], None]):
        self._callbacks_on_finished.append(cb)
        return cb

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

    def run(self, batches: Iterable[Any]) -> None:
        """Run on batches for a number of epochs.

        Args:
            batches: Batches to iterate over in an epoch.
            max_epoch: Maximum number of epochs to run.
        """
        self._run_callbacks_on_started()
        i = 0
        while not self._stopped and i < self._max_epoch:
            epoch = EpochId(i + 1)
            self._run_callbacks_on_epoch_started(epoch)
            for j, batch in enumerate(batches):
                batch_idx = BatchIndex(j)
                batch = self._run_callbacks_on_batch_started(epoch, batch_idx, batch)
                boutput = self._on_batch(epoch, batch_idx, batch)
                self._run_callbacks_on_batch_finished(epoch, batch_idx, batch, boutput)
            self._run_callbacks_on_epoch_finished(epoch)
            i += 1
        self._run_callbacks_on_finished()

    def resume(self, repeat_last_batch: bool = False) -> None:
        """Resume runner starting from the current state.

        Args:
            repeat_last_batch: Whether to repeat processing the last batch. Ignored if the
                last epoch is finished (i.e. the batches have been exhausted).
        """
        state = self.state
        state["running"] = True

        finished_last_epoch = state["n_iters"] % len(state["batches"]) == 0

        if not finished_last_epoch:
            self._emit(Event._ETIMER_STARTED, state)
            self._emit(Event._PBAR_CREATED, state)

            if repeat_last_batch:
                self._emit(Event.BATCH, state)
                self._emit(Event._REDUCER_UPDATED, state)
                self._emit(Event._PBAR_UPDATED, state)

            self._run_epoch()

            self._emit(Event._PBAR_CLOSED, state)
            self._emit(Event._REDUCER_COMPUTED, state)
            self._emit(Event.EPOCH_FINISHED, state)
            self._emit(Event._ETIMER_FINISHED, state)

        while state["running"] and state["epoch"] < state["max_epoch"]:
            state["epoch"] += 1
            self._emit(Event._ETIMER_STARTED, state)
            self._emit(Event.EPOCH_STARTED, state)
            self._emit(Event._REDUCER_RESET, state)
            self._emit(Event._PBAR_CREATED, state)

            state["_batches_iter"] = iter(state["batches"])
            self._run_epoch()

            self._emit(Event._PBAR_CLOSED, state)
            self._emit(Event._REDUCER_COMPUTED, state)
            self._emit(Event.EPOCH_FINISHED, state)
            self._emit(Event._ETIMER_FINISHED, state)

        self._emit(Event.FINISHED, state)
        state["running"] = False

    def _emit(self, event: Event, state: dict) -> None:
        for callback in self._callbacks[event]:
            if not state["running"]:
                break
            callback(state)

    def _run_callbacks_on_started(self) -> None:
        for cb in self._callbacks_on_started:
            cb()

    def _run_callbacks_on_epoch_started(self, e: EpochId) -> None:
        for cb in self._callbacks_on_epoch_started:
            cb(e)

    def _run_callbacks_on_batch_started(self, e: EpochId, bi: BatchIndex, b: Any) -> Any:
        for cb in self._callbacks_on_batch_started:
            b = cb(e, bi, b)
        return b

    def _run_callbacks_on_batch_finished(
        self, e: EpochId, bi: BatchIndex, b: Any, bo: T
    ) -> None:
        for cb in self._callbacks_on_batch_finished:
            cb(e, bi, b, bo)

    def _run_callbacks_on_epoch_finished(self, e: EpochId) -> None:
        i = 0
        while not self._stopped and i < len(self._callbacks_on_epoch_finished):
            cb = self._callbacks_on_epoch_finished[i]
            nargs = len(signature(cb).parameters)
            if nargs == 2:
                cb = cast(Callable[[EpochId, "StopFn"], None], cb)
                cb(e, self._stop)
            elif nargs == 1:
                cb = cast(Callable[[EpochId], None], cb)
                cb(e)
            else:
                raise TypeError(
                    f"expected {cb.__name__}() to accept 1 or 2 arguments but got {nargs}"
                )
            i += 1

    def _run_callbacks_on_finished(self) -> None:
        for cb in self._callbacks_on_finished:
            cb()

    def _reset(self) -> None:
        self._stopped = False

    def _stop(self) -> None:
        self._stopped = True


StopFn = Callable[[], None]
