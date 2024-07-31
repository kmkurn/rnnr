import abc
import logging
import math
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Tuple, Union

from tqdm import tqdm

from .batch import BatchOutput
from .runner import EpochId


class EpochProgressTracker(abc.ABC):
    @abc.abstractmethod
    @contextmanager
    def start(self, e: EpochId) -> Iterator["BatchLogger"]:
        raise NotImplementedError


class TqdmEpochProgressTracker(EpochProgressTracker):
    def __init__(self, tqdm_factory: Callable[[EpochId], tqdm]) -> None:
        self._tqdm_factory = tqdm_factory

    @contextmanager
    def start(self, e: EpochId) -> Iterator["TqdmBatchLogger"]:
        tqdm_obj = self._tqdm_factory(e)
        yield TqdmBatchLogger(tqdm_obj)
        tqdm_obj.close()


class LoggingEpochProgressTracker(EpochProgressTracker):
    def __init__(self, num_batches: int, log_every: int = 10) -> None:
        self._num_batches = num_batches
        self._log_every = log_every

    @contextmanager
    def start(self, e: EpochId) -> Iterator["BatchLogger"]:
        yield LoggingBatchLogger(e, self._num_batches, self._log_every)


class BatchLogger(abc.ABC):
    @abc.abstractmethod
    def __call__(self, o: BatchOutput) -> None:
        raise NotImplementedError


class TqdmBatchLogger(BatchLogger):
    def __init__(self, tqdm_object: tqdm) -> None:
        self._tqdm_obj = tqdm_object

    def __call__(self, o: BatchOutput) -> None:
        self._tqdm_obj.set_postfix({"loss": o.loss, **o.stats})
        self._tqdm_obj.update(n=1)


class LoggingBatchLogger(BatchLogger):
    logger = logging.getLogger(__name__)

    def __init__(self, e: EpochId, num_batches: int, log_every: int = 10) -> None:
        self._epoch = e
        self._num_batches = num_batches
        self._log_every = log_every
        self._batch_count = 0

    def __call__(self, o: BatchOutput) -> None:
        self._batch_count += 1
        if self._batch_count % self._log_every == 0:
            n_digits = self._compute_num_digits(self._num_batches)
            stats_msg, stats_vals = self._get_stats_msg_vals(o.stats)
            self.logger.info(
                f"Epoch %d [%{n_digits}d/%d]: loss=%.4f{stats_msg}",
                self._epoch,
                self._batch_count,
                self._num_batches,
                o.loss,
                *stats_vals,
            )

    @staticmethod
    def _compute_num_digits(n: int) -> int:
        return math.floor(math.log10(n)) + 1

    @staticmethod
    def _get_stats_msg_vals(
        stats: Dict[str, Union[int, float]]
    ) -> Tuple[str, List[Union[int, float]]]:
        msg_parts = []
        values = []
        for k, v in stats.items():
            fmt = "%.4f" if type(v) == float else "%d"
            msg_parts.append(f" {k}={fmt}")
            values.append(v)
        return "".join(msg_parts), values
