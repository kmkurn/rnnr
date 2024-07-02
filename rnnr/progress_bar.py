import abc
from contextlib import contextmanager
from typing import Callable, Iterator

from .runner import EpochId
from .utils import ProgressBar


class EpochProgressBar(abc.ABC):
    @abc.abstractmethod
    @contextmanager
    def __call__(self, e: EpochId) -> Iterator[ProgressBar]:
        raise NotImplementedError


class DefaultEpochProgressBar(EpochProgressBar):
    def __init__(self, make_progress_bar: Callable[[EpochId], ProgressBar]) -> None:
        self._make_pbar = make_progress_bar

    @contextmanager
    def __call__(self, e: EpochId) -> Iterator[ProgressBar]:
        pbar = self._make_pbar(e)
        yield pbar
        pbar.finish()
