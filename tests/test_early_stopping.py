from rnnr import Runner
from rnnr.attachments import EarlyStopper


def test_correct():
    dev_scores = [0.7, 0.4, 0.3, 0.8, 0.9]
    current_best = -float("inf")

    def should_reduce_patience(epoch: int) -> bool:
        nonlocal current_best
        if dev_scores[epoch - 1] > current_best:
            current_best = dev_scores[epoch - 1]
            return False
        return True

    runner = Runner(lambda e, bi, b: b, max_epoch=5)
    call_hist = []

    @runner.on_epoch_started
    def on_epoch_started(epoch: int) -> None:
        call_hist.append(("on_epoch_started", (epoch,)))

    @runner.on_epoch_finished
    def on_epoch_finished(epoch: int) -> None:
        call_hist.append(("on_epoch_finished", (epoch,)))

    @runner.on_finished
    def on_finished() -> None:
        call_hist.append(("on_finished", ()))

    EarlyStopper(should_reduce_patience, patience=1).attach_on(runner)

    runner.run(range(1))

    assert call_hist == [
        (f"on_epoch_{s}ed", (i + 1,)) for i in range(3) for s in ("start", "finish")
    ] + [("on_finished", ())]


def test_can_recover_patience():
    dev_scores = [0.7, 0.4, 0.8, 0.6, 0.65, 0.75]
    current_best = -float("inf")

    def should_reduce_patience(epoch: int) -> bool:
        nonlocal current_best
        if dev_scores[epoch - 1] > current_best:
            current_best = dev_scores[epoch - 1]
            return False
        return True

    runner = Runner(lambda e, bi, b: b, max_epoch=6)
    call_hist = []

    @runner.on_epoch_started
    def on_epoch_started(epoch: int) -> None:
        call_hist.append(("on_epoch_started", (epoch,)))

    @runner.on_epoch_finished
    def on_epoch_finished(epoch: int) -> None:
        call_hist.append(("on_epoch_finished", (epoch,)))

    @runner.on_finished
    def on_finished() -> None:
        call_hist.append(("on_finished", ()))

    EarlyStopper(should_reduce_patience, patience=1).attach_on(runner)

    runner.run(range(1))

    assert call_hist == [
        (f"on_epoch_{s}ed", (i + 1,)) for i in range(5) for s in ("start", "finish")
    ] + [("on_finished", ())]
