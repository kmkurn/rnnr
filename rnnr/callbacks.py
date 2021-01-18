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
from typing import Any, Callable, Optional
from pathlib import Path
import logging
import pickle


def maybe_stop_early(patience: int = 5, *, check: str = "better", counter: str = "_counter"):
    """A callback factory for early stopping.

    The returned calback keeps a counter in ``state[counter]`` for the number of times
    ``state[check]`` is ``False``. If this counter exceeds ``patience``, the callback
    stops the runner by setting ``state['running'] = False``.

    Example:

        >>> valid_losses = [0.1, 0.2, 0.3]  # simulate validation batch losses
        >>> batches = range(10)
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
        ...     evaluator.on(Event.BATCH, eval_fn)
        ...     MeanReducer(name='mean').attach_on(evaluator)
        ...     evaluator.run(valid_losses)
        ...     if state.get('best_loss', float('inf')) > evaluator.state['mean']:
        ...         state['better'] = True
        ...         state['best_loss'] = evaluator.state['mean']
        ...     else:
        ...         state['better'] = False
        ...
        >>> trainer.on(Event.EPOCH_FINISHED, maybe_stop_early(patience=2))
        >>> trainer.run(batches, max_epoch=7)
        Epoch 1 started
        Epoch 2 started
        Epoch 3 started
        Epoch 4 started

    Args:
        patience: Stop the runner when the counter exceeds this number.
        check: Increment counter if ``state[check]`` is ``False``.
        counter: Store the counter in ``state[counter]``.

    Returns:
        Callback that does early stopping.
    """
    logger = logging.getLogger(f"{__name__}.early_stopping")

    def callback(state):
        n = (state.get(counter, 0) + 1) if not state[check] else 0
        state[counter] = n
        if state[counter] > patience:
            logger.info("Patience exceeded, stopping early")
            state["running"] = False

    return callback


def checkpoint(
    what: str,
    obj: Optional[Any] = None,
    *,
    under: Optional[Path] = None,
    at_most: int = 1,
    when: Optional[str] = None,
    using: Optional[Callable[[Any, Path], None]] = None,
    ext: str = "pkl",
    prefix_fmt: str = "{epoch}_",
    queue_fmt: str = "_saved_{what}",
):
    """A callback factory for checkpointing.

    Checkpointing means saving ``obj`` (or ``state[what]`` if ``obj`` is ``None``) during a
    run under ``under`` directory with ``{prefix_fmt}{what}.{ext}`` as the filename.

    Example:

        >>> from pathlib import Path
        >>> from pprint import pprint
        >>> from rnnr import Event, Runner
        >>> from rnnr.callbacks import checkpoint
        >>>
        >>> batches = range(3)
        >>> tmp_dir = Path('/tmp')
        >>> runner = Runner()
        >>> @runner.on(Event.EPOCH_FINISHED)
        ... def store_checkpoint(state):
        ...     state['model'] = 'MODEL'
        ...     state['optimizer'] = 'OPTIMIZER'
        ...
        >>> runner.on(Event.EPOCH_FINISHED, checkpoint('model', under=tmp_dir, at_most=3))
        >>> runner.on(Event.EPOCH_FINISHED, checkpoint('optimizer', under=tmp_dir, at_most=3))
        >>> runner.run(batches, max_epoch=7)
        >>> pprint(sorted(list(tmp_dir.glob('*.pkl'))))
        [PosixPath('/tmp/5_model.pkl'),
         PosixPath('/tmp/5_optimizer.pkl'),
         PosixPath('/tmp/6_model.pkl'),
         PosixPath('/tmp/6_optimizer.pkl'),
         PosixPath('/tmp/7_model.pkl'),
         PosixPath('/tmp/7_optimizer.pkl')]

    Args:
        what: Name of the object to save.
        obj: Object to save. If ``None``, will be obtained from ``state[what]``.
        under: Save the object under this directory. Defaults to the current working directory
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
        queue_fmt: Keeps track of the saved files for the object with a queue stored in
            ``state[queue_fmt.format(what=what)]``.

    Returns:
        Callback that does checkpointing.
    """
    if under is None:  # pragma: no cover
        under = Path.cwd()
    if using is None:
        using = _save_with_pickle
    qkey = queue_fmt.format(what=what)
    logger = logging.getLogger(f"{__name__}.checkpointing")

    def callback(state):
        q = state.get(qkey, deque())
        if when is None or state[when]:
            fmt = f"{prefix_fmt}{what}.{ext}"
            path = under / fmt.format(**state)
            logger.info("Saving to %s", path)
            using(state[what] if obj is None else obj, path)
            q.append(path)
        while len(q) > at_most:
            p = q.popleft()
            if p.exists():  # pragma: no cover
                p.unlink()
        state[qkey] = q

    return callback


def save(*args, **kwargs):  # pragma: no cover
    """An alias for `checkpoint`."""
    return checkpoint(*args, **kwargs)


def _save_with_pickle(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
