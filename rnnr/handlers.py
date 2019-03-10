from typing import Callable, Optional

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
