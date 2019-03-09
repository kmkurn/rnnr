from typing import Callable, Optional, Type

from tqdm import tqdm

from . import Event, Runner


class ProgressBar:
    def __init__(
            self,
            tqdm_cls: Optional[Type[tqdm]] = None,
            update_size: Optional[Callable[[dict], int]] = None,
            stats: Optional[Callable[[dict], dict]] = None,
            **kwargs,
    ) -> None:
        if update_size is None:
            update_size = lambda _: 1
        if stats is None:
            stats = lambda state: {'output': state['output']}

        self._tqdm_cls = tqdm_cls
        self._update_size = update_size
        self._stats = stats
        self._kwargs = kwargs
        self._pbar: tqdm

    def attach_on(self, runner: Runner) -> None:
        runner.append_handler(Event.EPOCH_STARTED, self._create_pbar)
        runner.append_handler(Event.BATCH_FINISHED, self._update_pbar)
        runner.append_handler(Event.EPOCH_FINISHED, self._close_pbar)

    def _create_pbar(self, state: dict) -> None:
        self._pbar = self._tqdm_cls(state['batches'], **self._kwargs)

    def _update_pbar(self, state: dict) -> None:
        self._pbar.set_postfix(**self._stats(state))
        self._pbar.update(self._update_size(state))

    def _close_pbar(self, state: dict) -> None:
        self._pbar.close()
