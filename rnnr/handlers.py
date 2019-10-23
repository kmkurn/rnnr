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

from collections import deque
from typing import Any, Callable, Deque, Iterable, Optional, Mapping
from pathlib import Path
import logging
import pickle

logger = logging.getLogger(__name__)


class EarlyStopper:
    """A handler for early stopping.

    This handler keeps track the number of times the loss value does not improve. If this
    number is greater than the given patience, this handler stops the given runner.

    Example:

        >>> valid_losses = [0.1, 0.2, 0.3]  # simulate validation batch losses
        >>> batches = range(10)
        >>> batch_fn = lambda _: None
        >>>
        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import MeanReducer
        >>> from rnnr.handlers import EarlyStopper
        >>>
        >>> trainer = Runner()
        >>> @trainer.on(Event.EPOCH_STARTED)
        ... def print_epoch(state):
        ...     print('Epoch', state['epoch'], 'started')
        ...
        >>> @trainer.on(Event.EPOCH_FINISHED)
        ... def eval_on_valid(state):
        ...     def eval_fn(state):
        ...         state['output'] = state['batch']
        ...     evaluator = Runner()
        ...     MeanReducer(name='mean').attach_on(evaluator)
        ...     eval_state = evaluator.run(eval_fn, valid_losses)
        ...     state['loss'] = eval_state['mean']
        ...
        >>> trainer.append_handler(Event.EPOCH_FINISHED, EarlyStopper(patience=2))
        >>> _ = trainer.run(batch_fn, batches, max_epoch=7)
        Epoch 1 started
        Epoch 2 started
        Epoch 3 started
        Epoch 4 started

    Args:
        patience: Number of times to wait for the loss to improve before stopping.
        loss_key: Get the loss value from ``state[loss_key]``.
        eps: An improvement is considered only when the loss value decreases by at least
            this amount.
    """
    def __init__(
            self,
            patience: int = 5,
            loss_key: str = 'loss',
            eps: float = 1e-4,
    ) -> None:
        self._patience = patience
        self._loss_key = loss_key
        self._eps = eps

        self.min_loss = float('inf')
        self._n_bad_losses = 0

    def __call__(self, state: dict) -> None:
        loss = state[self._loss_key]
        if loss <= self.min_loss - self._eps:
            self.min_loss = loss
            self._n_bad_losses = 0
        else:
            self._n_bad_losses += 1

        if self._n_bad_losses > self._patience:
            logger.info('Patience exceeded, stopping early')
            state['runner'].stop()


class Checkpointer:
    """A handler for checkpointing.

    Checkpointing here means saving some objects (e.g., models) periodically during a run.

    Example:

        >>> from pathlib import Path
        >>> from pprint import pprint
        >>> from rnnr import Event, Runner
        >>> from rnnr.handlers import Checkpointer
        >>>
        >>> batches = range(3)
        >>> batch_fn = lambda _: None
        >>> tmp_dir = Path('/tmp')
        >>> runner = Runner()
        >>> @runner.on(Event.EPOCH_FINISHED)
        ... def store_checkpoint(state):
        ...     state['checkpoint'] = {'model.pkl': 'MODEL', 'optimizer.pkl': 'OPTIMIZER'}
        ...
        >>> runner.append_handler(Event.EPOCH_FINISHED, Checkpointer(tmp_dir, max_saved=3))
        >>> _ = runner.run(batch_fn, batches, max_epoch=7)
        >>> pprint(sorted(list(tmp_dir.glob('*.pkl'))))
        [PosixPath('/tmp/5_model.pkl'),
         PosixPath('/tmp/5_optimizer.pkl'),
         PosixPath('/tmp/6_model.pkl'),
         PosixPath('/tmp/6_optimizer.pkl'),
         PosixPath('/tmp/7_model.pkl'),
         PosixPath('/tmp/7_optimizer.pkl')]

    Args:
        save_dir: Save checkpoints in this directory.
        checkpoint_key: Get the checkpoint from ``state[checkpoint_key]``. A checkpoint
            is a mapping whose keys are filenames and the values are the objects to checkpoint.
            The filenames in this dictionary's keys are prepended with the number of times
            this handler is called to get the actual saved files' names. This allows the
            actual filenames contain the e.g. epoch number if this handler is invoked at the
            end of each epoch.
        max_saved: Maximum number of checkpoints saved.
        loss_key: Get the loss value from ``state[loss_key]``. Checkpoints are saved only
            when the loss is smaller than the minimum loss observed so far. The default
            of ``None`` means checkpoints are saved whenever this handler is called.
        save_fn: Function to invoke to save the checkpoints. If given, this must be a callable
            accepting two arguments: an object to save and a path to save it to. The default
            is to save the object using `pickle`.
        eps: The loss value must be smaller at least by this value to be considered as an
            improvement. Only used if ``loss_fn`` is given.
    """
    def __init__(
            self,
            save_dir: Path,
            checkpoint_key: str = 'checkpoint',
            max_saved: int = 1,
            loss_key: Optional[str] = None,
            save_fn: Optional[Callable[[Any, Path], None]] = None,
            eps: float = 1e-4,
    ) -> None:
        if save_fn is None:
            save_fn = self._default_save_fn

        self._save_dir = save_dir
        self._checkpoint_key = checkpoint_key
        self._max_saved = max_saved
        self._loss_key = loss_key
        self._save_fn = save_fn
        self._eps = eps

        self.min_loss = float('inf')
        self._n_calls = 0
        self._deque: Deque[int] = deque()

    @staticmethod
    def _default_save_fn(obj: Any, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def __call__(self, state: dict) -> None:
        self._n_calls += 1
        ckpt = state[self._checkpoint_key]
        if self._should_save(state):
            self._save(ckpt)
        if self._should_delete():
            self._delete(ckpt.keys())

        assert self._n_saved <= self._max_saved

    @property
    def _n_saved(self) -> int:
        return len(self._deque)

    def _should_save(self, state: dict) -> bool:
        if self._loss_key is None:
            return True

        loss = state[self._loss_key]
        if loss <= self.min_loss - self._eps:
            logger.info('Found new best loss of %f', loss)
            self.min_loss = loss
            return True

        return False

    def _should_delete(self) -> bool:
        return self._n_saved > self._max_saved

    def _save(self, ckpt: Mapping[str, Any]) -> None:
        self._deque.append(self._n_calls)
        for name, obj in ckpt.items():
            path = self._save_dir / f'{self._n_calls}_{name}'
            logger.info('Saving to %s', path)
            self._save_fn(obj, path)

    def _delete(self, filenames: Iterable[str]) -> None:
        num = self._deque.popleft()
        for name in filenames:
            path = self._save_dir / f'{num}_{name}'
            if path.exists():
                path.unlink()
