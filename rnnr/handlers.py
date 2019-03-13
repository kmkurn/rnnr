from collections import deque
from typing import Any, Callable, Optional
from pathlib import Path
import pickle

Handler = Callable[[dict], None]

from . import Runner  # avoid circular import


class EarlyStopper(Handler):
    """A handler for early stopping.

    This handler keeps track the number of times the loss value does not improve. If this
    number is greater than the given patience, this handler stops the given runner.

    Example:

        >>> valid_losses = [0.1, 0.2, 0.3]  # simulate validation batch losses
        >>> dummy_batches = range(10)
        >>> dummy_batch_fn = lambda x: x
        >>>
        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import MeanAggregator
        >>> from rnnr.handlers import EarlyStopper
        >>>
        >>> trainer, evaluator = Runner(), Runner()
        >>> @trainer.on(Event.EPOCH_STARTED)
        ... def print_epoch(state):
        ...     print('Epoch', state['epoch'], 'started')
        ...
        >>> @trainer.on(Event.EPOCH_FINISHED)
        ... def eval_on_valid(state):
        ...     evaluator.run(lambda loss: loss, valid_losses)
        ...
        >>> MeanAggregator(name='loss').attach_on(evaluator)
        >>> evaluator.append_handler(Event.FINISHED, EarlyStopper(trainer, patience=2))
        >>> trainer.run(dummy_batch_fn, dummy_batches, max_epoch=7)
        Epoch 1 started
        Epoch 2 started
        Epoch 3 started
        Epoch 4 started

    Args:
        runner: Runner to stop early.
        patience: Number of times to wait for the loss to improve before stopping.
        loss_fn: Callback to get the loss value from the runner's ``state`` on which this
            handler is appended. The default is to get ``state['loss']`` as the loss.
        eps: An improvement is considered only when the loss value decreases by at least
            this amount.
    """

    def __init__(
            self,
            runner: Runner,
            patience: int = 5,
            loss_fn: Optional[Callable[[dict], float]] = None,
            eps: float = 1e-4,
    ) -> None:
        if loss_fn is None:
            loss_fn = lambda state: state['loss']

        self._runner = runner
        self._patience = patience
        self._loss_fn = loss_fn
        self._eps = eps

        self._num_bad_loss = 0
        self._min_loss = float('inf')

    def __call__(self, state: dict) -> None:
        loss = self._loss_fn(state)
        if loss <= self._min_loss - self._eps:
            self._min_loss = loss
            self._num_bad_loss = 0
        else:
            self._num_bad_loss += 1

        if self._num_bad_loss > self._patience:
            self._runner.stop()


class Checkpointer(Handler):
    def __init__(
            self,
            save_dir: Path,
            objs: dict,
            max_saved: int = 1,
            loss_fn: Optional[Callable[[dict], float]] = None,
            save_fn: Optional[Callable[[Path, Any], None]] = None,
            eps: float = 1e-4,
    ) -> None:
        if save_fn is None:
            save_fn = self._default_save_fn

        self._save_dir = save_dir
        self._objs = objs
        self._max_saved = max_saved
        self._loss_fn = loss_fn
        self._save_fn = save_fn
        self._eps = eps

        self._num_calls = 0
        self._deque = deque()
        self._min_loss = float('inf')

    @staticmethod
    def _default_save_fn(path: Path, obj: Any) -> None:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def __call__(self, state: dict) -> None:
        self._num_calls += 1
        if self._should_save(state):
            self._save()
        if self._should_delete():
            self._delete()

        assert self._num_saved <= self._max_saved

    @property
    def _num_saved(self) -> int:
        return len(self._deque)

    def _should_save(self, state: dict) -> bool:
        if self._loss_fn is None:
            return True

        loss = self._loss_fn(state)
        if loss <= self._min_loss - self._eps:
            self._min_loss = loss
            return True

        return False

    def _should_delete(self) -> bool:
        return len(self._deque) > self._max_saved

    def _save(self) -> None:
        self._deque.append(self._num_calls)
        for name, obj in self._objs.items():
            path = self._save_dir / f'{self._num_calls}_{name}'
            self._save_fn(path, obj)

    def _delete(self) -> None:
        num = self._deque.popleft()
        for name in self._objs:
            path = self._save_dir / f'{num}_{name}'
            if path.exists():
                path.unlink()
