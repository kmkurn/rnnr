from typing import Mapping, Union

from rnnr.progress_bar import DefaultEpochProgressBar
from rnnr.runner import EpochId
from rnnr.utils import ProgressBar


def test_correct():
    class FakeProgressBar(ProgressBar):
        def __init__(self) -> None:
            self.finished = False

        def update(self, count: int) -> None:
            pass

        def finish(self) -> None:
            self.finished = True

        def show_stats(self, stats: Mapping[str, Union[int, float]]) -> None:
            pass

    epoch_pbar = DefaultEpochProgressBar(lambda _: FakeProgressBar())
    with epoch_pbar(EpochId(1)) as pbar:
        assert not pbar.finished
    assert pbar.finished
