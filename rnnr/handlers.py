from collections import deque
from typing import Any, Callable, Optional
from pathlib import Path
import pickle

from . import Runner


class EarlyStopper:
    def __init__(
            self,
            runner: Runner,
            patience: int = 5,
            eps: float = 1e-4,
            get_value: Optional[Callable[[dict], float]] = None,
    ) -> None:
        if get_value is None:
            get_value = lambda state: state['output']

        self._runner = runner
        self._patience = patience
        self._eps = eps
        self._get_value = get_value
        self._num_bad_value = 0
        self._min_value = float('inf')

    def __call__(self, state: dict) -> None:
        value = self._get_value(state)
        if value >= self._min_value + self._eps:
            self._num_bad_value += 1
        else:
            self._min_value = min(self._min_value, value)
            self._num_bad_value = 0

        if self._num_bad_value >= self._patience:
            self._runner.stop()


class Checkpointer:
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