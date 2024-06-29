from .runner import EpochId


class EpochTimer:
    def start(self, e: EpochId) -> None:
        pass

    def end(self, e: EpochId) -> None:
        pass
