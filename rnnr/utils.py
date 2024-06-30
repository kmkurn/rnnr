import abc
import time
from datetime import timedelta
from typing import Generic, TypeVar

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
