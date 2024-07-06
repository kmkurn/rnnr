import abc
from contextlib import contextmanager
from typing import Callable, Iterator

from tqdm import tqdm

from .batch import BatchOutput


class TqdmEpochLogger:
    def __init__(self, tqdm_factory: Callable[[], tqdm]) -> None:
        self._tqdm_factory = tqdm_factory

    @contextmanager
    def start(self) -> Iterator["BatchLogger"]:
        tqdm_obj = self._tqdm_factory()
        yield TqdmBatchLogger(tqdm_obj)
        tqdm_obj.close()


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
