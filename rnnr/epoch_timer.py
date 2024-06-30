import abc
import logging
from typing import Optional

from .runner import EpochId
from .utils import Timer

logger = logging.getLogger(__name__)


class EpochTimer(abc.ABC):
    @abc.abstractmethod
    def start_epoch(self, e: EpochId) -> None:
        pass

    @abc.abstractmethod
    def finish_epoch(self, e: EpochId) -> None:
        pass


class NoopEpochTimer(EpochTimer):
    def start_epoch(self, e: EpochId) -> None:
        pass

    def finish_epoch(self, e: EpochId) -> None:
        pass


class LoggingEpochTimer(EpochTimer):
    def __init__(self, max_epoch: int, timer: Optional[Timer] = None) -> None:
        if timer is None:
            timer = Timer()
        self._max_epoch = max_epoch
        self._timer = timer

    def start_epoch(self, e: EpochId) -> None:
        self._timer.start()
        if self._max_epoch > 1:
            logger.info("Epoch %d/%d started", e, self._max_epoch)

    def finish_epoch(self, e: EpochId) -> None:
        if self._max_epoch > 1:
            logger.info("Epoch %d/%d finished in %s", e, self._max_epoch, self._timer.end())
