import abc
import time
from datetime import timedelta
from typing import Generic, Mapping, TypeVar, Union

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
    def update(self, count: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def done(self) -> None:
        raise NotImplementedError


class ProgressBarWithStats(ProgressBar):
    @abc.abstractmethod
    def show_stats(self, stats: Mapping[str, Union[int, float]]) -> None:
        raise NotImplementedError


class TqdmProgressBar(ProgressBarWithStats):
    def __init__(self, tqdm_instance: tqdm) -> None:
        self._tqdm = tqdm_instance

    def update(self, count: int) -> None:
        self._tqdm.update(count)

    def done(self) -> None:
        self._tqdm.close()

    def show_stats(self, stats: Mapping[str, Union[int, float]]) -> None:
        self._tqdm.set_postfix(stats)
