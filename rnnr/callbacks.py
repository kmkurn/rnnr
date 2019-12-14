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
from typing import Any, Callable, Deque, Iterable, Mapping, Optional
from pathlib import Path
import logging
import pickle
import warnings

logger = logging.getLogger(__name__)


class ImprovementCallbackMixin:
    def __init__(self, mode: str = 'min', *, eps: float = 1e-4) -> None:
        if mode not in ('min', 'max'):  # pragma: no cover
            warnings.warn("mode {mode!r} is unknown; will be interpreted as 'max'")
            mode = 'max'

        self._mode = mode
        self._eps = eps
        self.best_value = None

    def _improved(self, value: Any) -> bool:
        if self.best_value is None:
            return True

        if self._mode == 'min':
            if isinstance(self.best_value, float):
                return value <= self.best_value - self._eps
            return value < self.best_value

        if isinstance(self.best_value, float):
            return value >= self.best_value + self._eps
        return value > self.best_value


def maybe_stop_early(*, check: str = 'better', patience: int = 5, counter: str = 'counter'):
    """A callback factory for early stopping.

    The returned calback keeps a counter in ``state[counter]`` for the number of times
    ``state[check]`` is ``False``. If this counter exceeds ``patience``, the callback
    stops the runner by setting ``state['running']=False``.

    Example:

        >>> valid_losses = [0.1, 0.2, 0.3]  # simulate validation batch losses
        >>> batches = range(10)
        >>> batch_fn = lambda _: None
        >>>
        >>> from rnnr import Event, Runner
        >>> from rnnr.attachments import MeanReducer
        >>> from rnnr.callbacks import maybe_stop_early
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
        ...     if state.get('best_loss', float('inf')) > eval_state['mean']:
        ...         state['better'] = True
        ...         state['best_loss'] = eval_state['mean']
        ...     else:
        ...         state['better'] = False
        ...
        >>> trainer.on(Event.EPOCH_FINISHED, maybe_stop_early(patience=2))
        >>> _ = trainer.run(batch_fn, batches, max_epoch=7)
        Epoch 1 started
        Epoch 2 started
        Epoch 3 started
        Epoch 4 started

    Args:
        check: Increment counter if ``state[check]`` is ``False``.
        patience: Stop the runner when the counter exceeds this number.
        counter: Store the counter in ``state[counter]``.
    """

    def callback(state):
        n = (state.get(counter, 0) + 1) if not state[check] else 0
        state[counter] = n
        if state[counter] > patience:
            logger.info('Patience exceeded, stopping early')
            state['running'] = False

    return callback


def checkpoint(
        what: str,
        *,
        to_dir: Optional[Path] = None,
        at_most: int = 1,
        when: Optional[str] = None,
        using: Optional[Callable[[Any, Path], None]] = None,
        ext: str = 'pkl',
        prefix_fmt: str = '{epoch}_',
):
    """A callback factory for checkpointing.

    Checkpointing means saving some object stored in ``state[what]`` during a run under ``to_dir``
    directory with ``{prefix_fmt}{what}.{ext}`` as the filename.

    Example:

        >>> from pathlib import Path
        >>> from pprint import pprint
        >>> from rnnr import Event, Runner
        >>> from rnnr.callbacks import checkpoint
        >>>
        >>> batches = range(3)
        >>> batch_fn = lambda _: None
        >>> tmp_dir = Path('/tmp')
        >>> runner = Runner()
        >>> @runner.on(Event.EPOCH_FINISHED)
        ... def store_checkpoint(state):
        ...     state['model'] = 'MODEL'
        ...     state['optimizer'] = 'OPTIMIZER'
        ...
        >>> runner.on(Event.EPOCH_FINISHED, checkpoint('model', to_dir=tmp_dir, at_most=3))
        >>> runner.on(Event.EPOCH_FINISHED, checkpoint('optimizer', to_dir=tmp_dir, at_most=3))
        >>> _ = runner.run(batch_fn, batches, max_epoch=7)
        >>> pprint(sorted(list(tmp_dir.glob('*.pkl'))))
        [PosixPath('/tmp/5_model.pkl'),
         PosixPath('/tmp/5_optimizer.pkl'),
         PosixPath('/tmp/6_model.pkl'),
         PosixPath('/tmp/6_optimizer.pkl'),
         PosixPath('/tmp/7_model.pkl'),
         PosixPath('/tmp/7_optimizer.pkl')]

    Args:
        what: Get the object to save from ``state[what]``.
        to_dir: Save the object under this directory. Defaults to the current working directory
            if not given.
        at_most: Maximum number of files saved. When the number of files exceeds this number,
            older files will be deleted.
        when: If given, only save the object when ``state[when]`` is ``True``.
        using: Function to invoke to save the object. If given, this must be a callable
            accepting two arguments: an object to save and a `Path` to save it to. The default
            is to save the object using `pickle`.
        ext: Extension for the filename.
        prefix_fmt: Format for the filename prefix. Any string keys in ``state`` can be used
            as replacement fields.
    """
    if to_dir is None:  # pragma: no cover
        to_dir = Path.cwd()
    if using is None:
        using = _save_with_pickle

    def callback(state):
        if when is None or state[when]:
            fmt = f'{prefix_fmt}{what}.{ext}'
            path = to_dir / fmt.format(**state)
            logger.info('Saving to %s', path)
            using(state[what], path)
        ckpts = sorted(
            to_dir.glob(f'*{what}.{ext}'), key=lambda p: p.stat().st_mtime, reverse=True)
        while len(ckpts) > at_most:
            ckpts.pop().unlink()

    return callback


def save(*args, **kwargs):  # pragma: no cover
    """An alias for `checkpoint`."""
    return checkpoint(*args, **kwargs)


def _save_with_pickle(obj: Any, path: Path) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class Checkpointer(ImprovementCallbackMixin):
    """A callback for checkpointing.

    Checkpointing here means saving some objects (e.g., models) periodically during a run.

    Example:

        >>> from pathlib import Path
        >>> from pprint import pprint
        >>> from rnnr import Event, Runner
        >>> from rnnr.callbacks import Checkpointer
        >>>
        >>> batches = range(3)
        >>> batch_fn = lambda _: None
        >>> tmp_dir = Path('/tmp')
        >>> runner = Runner()
        >>> @runner.on(Event.EPOCH_FINISHED)
        ... def store_checkpoint(state):
        ...     state['checkpoint'] = {'model.pkl': 'MODEL', 'optimizer.pkl': 'OPTIMIZER'}
        ...
        >>> runner.on(Event.EPOCH_FINISHED, Checkpointer(tmp_dir, max_saved=3))
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
            this callback is called to get the actual saved files' names. This allows the
            actual filenames contain the e.g. epoch number if this callback is invoked at the
            end of each epoch.
        max_saved: Maximum number of checkpoints saved.
        save_fn: Function to invoke to save the checkpoints. If given, this must be a callable
            accepting two arguments: an object to save and a path to save it to. The default
            is to save the object using `pickle`.
        value_key: Get some value from ``state[value_key]``. Checkpoints are saved only
            when this value improves over the best value observed so far. The default
            of ``None`` means checkpoints are always saved whenever this callback is called.
        mode: Whether to consider lower (``min``) or higher (``max``) value as improvement.
        eps: Improvement is considered only when the value improves by at least this amount.
            Only used if ``value_key`` is given and it is an instance of `float`.
    """

    def __init__(
            self,
            save_dir: Path,
            checkpoint_key: str = 'checkpoint',
            save_fn: Optional[Callable[[Any, Path], None]] = None,
            value_key: Optional[str] = None,
            mode: str = 'min',
            *,
            max_saved: int = 1,
            eps: float = 1e-4,
    ) -> None:
        super().__init__(mode=mode, eps=eps)
        if save_fn is None:
            save_fn = self._default_save_fn

        self._save_dir = save_dir
        self._checkpoint_key = checkpoint_key
        self._max_saved = max_saved
        self._value_key = value_key
        self._save_fn = save_fn

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
        if self._value_key is None:
            return True

        value = state[self._value_key]
        if self._improved(value):
            self._log(value)
            self.best_value = value
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

    def _log(self, value: Any) -> None:
        fmt = '%f' if isinstance(value, float) else '%s'
        logger.info(f'Found new best %s of {fmt}', self._value_key, value)
        # print((f'Found new best %s of {fmt}') % (self._value_key, value))
