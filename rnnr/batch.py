import abc
from typing import Dict, Mapping, Optional, Union


class BatchOutput(abc.ABC):
    @property
    @abc.abstractmethod
    def loss(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stats(self) -> Dict[str, Union[int, float]]:
        raise NotImplementedError


class DefaultBatchOutput(BatchOutput):
    def __init__(
        self, loss: float, stats: Optional[Mapping[str, Union[int, float]]] = None
    ) -> None:
        if stats is None:
            stats = {}
        self._loss = loss
        self._stats = stats

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        return {k: v for k, v in self._stats.items()}
