import abc
import time
from datetime import timedelta
from typing import Generic, Mapping, Optional, TypeVar, Union

from tqdm import tqdm

T = TypeVar("T")


class Timer(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def end(self) -> T:
        pass


class DefaultTimer(Timer[timedelta]):
    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        self._started_at = time.time()

    def end(self) -> timedelta:
        return timedelta(seconds=time.time() - self._started_at)


class ProgressBar(abc.ABC):
    @abc.abstractmethod
    def update(
        self, count: int, stats: Optional[Mapping[str, Union[int, float]]] = None
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self) -> None:
        raise NotImplementedError


class TqdmProgressBar(ProgressBar):
    def __init__(self, tqdm_instance: tqdm) -> None:
        self._tqdm = tqdm_instance

    def update(
        self, count: int, stats: Optional[Mapping[str, Union[int, float]]] = None
    ) -> None:
        self._tqdm.update(count)
        if stats is not None:
            self._tqdm.set_postfix(stats)

    def finish(self) -> None:
        self._tqdm.close()
