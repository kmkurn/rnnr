from typing import Mapping, Optional, Union

from rnnr.progress_bar import DefaultEpochProgressBar
from rnnr.runner import EpochId
from rnnr.utils import ProgressBar


def test_correct():
    class FakeProgressBar(ProgressBar):
        def __init__(self) -> None:
            self.finished = False

        def update(
            self, count: int, stats: Optional[Mapping[str, Union[int, float]]] = None
        ) -> None:
            pass

        def finish(self) -> None:
            self.finished = True

    epoch_pbar = DefaultEpochProgressBar(lambda _: FakeProgressBar())
    with epoch_pbar(EpochId(1)) as pbar:
        assert not pbar.finished
    assert pbar.finished
