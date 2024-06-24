from rnnr import Runner
from rnnr.attachments import ProgressBar
from tqdm import tqdm


def test_correct_call_order():
    history = []

    def on_batch(e, i, b):
        history.append("B")

    runner = Runner(on_batch, max_epoch=1)

    @runner.on_started
    def on_started():
        history.append("S")

    @runner.on_epoch_started
    def on_epoch_started(e):
        history.append("ES")

    @runner.on_batch_started
    def on_batch_started(e, i, b):
        history.append("BS")

    @runner.on_batch_finished
    def on_batch_finished(e, i, b, o):
        history.append("BF")

    batches = range(10)

    class tracked_tqdm(tqdm):
        def __init__(self, *args, **kwargs):
            history.append("TTI")
            super().__init__(*args, **kwargs)

        def update(self, size):
            history.append("TTU")
            assert size == 1
            return super().update(size)

        def close(self):
            history.append("TTC")
            return super().close()

        def set_postfix(self, *args, **kwargs):
            assert args == ()
            assert kwargs == {}

    ProgressBar(make_pbar=lambda _: tracked_tqdm(batches)).attach_on(runner)

    @runner.on_epoch_finished
    def on_epoch_finished(e):
        history.append("EF")

    @runner.on_finished
    def on_finished():
        history.append("F")

    runner.run(batches)
    expected = ["S", "ES", "TTI"]
    for _ in batches:
        expected.extend(["BS", "B", "BF", "TTU"])
    expected.extend(["TTC", "EF", "F"])

    assert history == expected


def test_num_items(call_tracker):
    def on_batch(e, i, b):
        pass

    runner = Runner(on_batch, max_epoch=1)
    batches = [list("foo"), list("quux")]

    class update_args_tracked_tqdm(tqdm):
        @call_tracker.track_args
        def update(self, n):
            super().update(n)

    ProgressBar(
        make_pbar=lambda _: update_args_tracked_tqdm(batches), get_num_items=len
    ).attach_on(runner)

    runner.run(batches)

    assert [args[1] for _, args in call_tracker.history] == [len(b) for b in batches]


def test_stats():
    def on_batch(e, i, b):
        return b ** 2

    runner = Runner(on_batch, max_epoch=1)
    history = []

    class set_postfix_args_tracked_tqdm(tqdm):
        def set_postfix(self, **kwargs):
            history.append(kwargs)
            return super().set_postfix(**kwargs)

    batches = range(10)

    ProgressBar(
        make_pbar=lambda _: set_postfix_args_tracked_tqdm(batches),
        get_stats=lambda o: {"loss": o},
    ).attach_on(runner)

    runner.run(batches)

    assert history == [{"loss": b ** 2} for b in batches]
