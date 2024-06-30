import abc
import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from .runner import EpochId
from .utils import Timer

logger = logging.getLogger(__name__)


class EpochTimer(abc.ABC):
    @contextmanager
    @abc.abstractmethod
    def __call__(self, e: EpochId) -> Iterator[None]:
        pass


class NoopEpochTimer(EpochTimer):
    @contextmanager
    def __call__(self, e: EpochId) -> Iterator[None]:
        yield


class LoggingEpochTimer(EpochTimer):
    def __init__(self, timer: Optional[Timer] = None) -> None:
        if timer is None:
            timer = Timer()
        self._timer = timer

    @contextmanager
    def __call__(self, e: EpochId) -> Iterator[None]:
        self._timer.start()
        logger.info("Epoch %d started", e)
        yield
        logger.info("Epoch %d finished in %s", e, self._timer.end())
