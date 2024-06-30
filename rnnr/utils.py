import time
from datetime import timedelta


class Timer:
    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        self._started_at = time.time()

    def end(self) -> timedelta:
        return timedelta(seconds=time.time() - self._started_at)
