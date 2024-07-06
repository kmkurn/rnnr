from typing import Mapping, Optional, Union


class BatchOutput:
    def __init__(
        self, loss: float, stats: Optional[Mapping[str, Union[int, float]]] = None
    ) -> None:
        if stats is None:
            stats = {}
        self.loss = loss
        self.stats = stats
