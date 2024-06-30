import abc
import time
from datetime import timedelta


class Timer(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def end(self) -> timedelta:
        pass


class DefaultTimer(Timer):
    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        self._started_at = time.time()

    def end(self) -> timedelta:
        return timedelta(seconds=time.time() - self._started_at)
