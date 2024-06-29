from typing import Tuple
import logging
import pickle

import pytest

from rnnr import Event, Runner
from rnnr.runner import EpochTimer


def test_run_with_callbacks(call_tracker, use_epoch_timer):
    class AppendToHistoryHandler(logging.Handler):
        def emit(self, record):
            call_tracker.history.append(record.getMessage())

    logger = logging.getLogger("rnnr.runner.epoch_timer")
    logger.setLevel(logging.INFO)
    logger.addHandler(AppendToHistoryHandler())

    @call_tracker.track_args
    def on_batch(epoch: int, batch_idx: int, batch: int) -> Tuple[int, int]:
        return epoch, batch_idx

    max_epoch = 2
    runner = Runner(on_batch, max_epoch)
    runner.epoch_timer = EpochTimer(start_fmt="ETS {epoch}", finish_fmt="ETF {epoch}")

    @runner.on_started
    @call_tracker.track_args
    def on_started1() -> None:
        pass

    @runner.on_started
    @call_tracker.track_args
    def on_started2() -> None:
        pass

    @runner.on_epoch_started
    @call_tracker.track_args
    def on_epoch_started1(epoch: int) -> None:
        pass

    @runner.on_epoch_started
    @call_tracker.track_args
    def on_epoch_started2(epoch: int) -> None:
        pass

    @runner.on_batch_started
    @call_tracker.track_args
    def on_batch_started1(epoch: int, batch_idx: int, batch: int) -> int:
        return batch + 1

    @runner.on_batch_started
    @call_tracker.track_args
    def on_batch_started2(epoch: int, batch_idx: int, batch: int) -> int:
        return batch ** 2

    @runner.on_batch_finished
    @call_tracker.track_args
    def on_batch_finished1(epoch: int, batch_idx: int, batch: int, output: int) -> None:
        pass

    @runner.on_batch_finished
    @call_tracker.track_args
    def on_batch_finished2(epoch: int, batch_idx: int, batch: int, output: int) -> None:
        pass

    @runner.on_epoch_finished
    @call_tracker.track_args
    def on_epoch_finished1(epoch: int) -> None:
        pass

    @runner.on_epoch_finished
    @call_tracker.track_args
    def on_epoch_finished2(epoch: int) -> None:
        pass

    @runner.on_finished
    @call_tracker.track_args
    def on_finished1() -> None:
        pass

    @runner.on_finished
    @call_tracker.track_args
    def on_finished2() -> None:
        pass

    batches = [3, 5]

    runner.run(batches)

    assert call_tracker.history == [
        ("on_started1", ()),
        ("on_started2", ()),
        "ETS 1",
        ("on_epoch_started1", (1,)),
        ("on_epoch_started2", (1,)),
        ("on_batch_started1", (1, 0, batches[0])),
        ("on_batch_started2", (1, 0, batches[0] + 1)),
        ("on_batch", (1, 0, (batches[0] + 1) ** 2)),
        ("on_batch_finished1", (1, 0, (batches[0] + 1) ** 2, (1, 0))),
        ("on_batch_finished2", (1, 0, (batches[0] + 1) ** 2, (1, 0))),
        ("on_batch_started1", (1, 1, batches[1])),
        ("on_batch_started2", (1, 1, batches[1] + 1)),
        ("on_batch", (1, 1, (batches[1] + 1) ** 2)),
        ("on_batch_finished1", (1, 1, (batches[1] + 1) ** 2, (1, 1))),
        ("on_batch_finished2", (1, 1, (batches[1] + 1) ** 2, (1, 1))),
        ("on_epoch_finished1", (1,)),
        ("on_epoch_finished2", (1,)),
        "ETF 1",
        "ETS 2",
        ("on_epoch_started1", (2,)),
        ("on_epoch_started2", (2,)),
        ("on_batch_started1", (2, 0, batches[0])),
        ("on_batch_started2", (2, 0, batches[0] + 1)),
        ("on_batch", (2, 0, (batches[0] + 1) ** 2)),
        ("on_batch_finished1", (2, 0, (batches[0] + 1) ** 2, (2, 0))),
        ("on_batch_finished2", (2, 0, (batches[0] + 1) ** 2, (2, 0))),
        ("on_batch_started1", (2, 1, batches[1])),
        ("on_batch_started2", (2, 1, batches[1] + 1)),
        ("on_batch", (2, 1, (batches[1] + 1) ** 2)),
        ("on_batch_finished1", (2, 1, (batches[1] + 1) ** 2, (2, 1))),
        ("on_batch_finished2", (2, 1, (batches[1] + 1) ** 2, (2, 1))),
        ("on_epoch_finished1", (2,)),
        ("on_epoch_finished2", (2,)),
        "ETF 2",
        ("on_finished1", ()),
        ("on_finished2", ()),
    ]


def test_on_epoch_finished_wrong_number_of_arguments(do_nothing):
    runner = Runner(on_batch=do_nothing)

    @runner.on_epoch_finished
    def on_epoch_finished():
        pass

    with pytest.raises(TypeError):
        runner.run(range(10))


def test_run_after_stopped(do_nothing, call_tracker):
    runner = Runner(on_batch=do_nothing, max_epoch=2)

    @runner.on_epoch_started
    @call_tracker.track_args
    def on_epoch_started(e):
        pass

    @runner.on_epoch_finished
    def on_epoch_finished(e, stop):
        stop()

    runner.run([])
    runner.run([])

    assert call_tracker.history == [("on_epoch_started", (1,)), ("on_epoch_started", (1,))]


@pytest.mark.skip
class TestResume:
    def test_stopped_on_batch(self, tmp_path):
        from rnnr.attachments import ProgressBar

        batches, max_epoch = list(range(5)), 2
        bcallback_ncalls, efcallback_ncalls = 0, 0

        runner = Runner()
        ProgressBar().attach_on(runner)

        def bcallback(state):
            nonlocal bcallback_ncalls
            bcallback_ncalls += 1
            if state["stage"] == "first" and state["n_iters"] == 2:
                state["running"] = False

        def efcallback(state):
            nonlocal efcallback_ncalls
            efcallback_ncalls += 1

        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, efcallback)
        runner.state["stage"] = "first"
        runner.run(batches, max_epoch)
        with open(tmp_path / "ckpt.pkl", "wb") as f:
            pickle.dump(runner.state, f)

        with open(tmp_path / "ckpt.pkl", "rb") as f:
            ckpt = pickle.load(f)
        runner = Runner()
        ProgressBar().attach_on(runner)
        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, efcallback)
        runner.state.update(ckpt)
        runner.state["stage"] = "second"
        runner.resume()

        assert bcallback_ncalls == len(batches) * max_epoch
        assert efcallback_ncalls == max_epoch

    def test_repeat_last_batch(self, tmp_path):
        from rnnr.attachments import LambdaReducer, MeanReducer, ProgressBar

        batches, max_epoch = list(range(5)), 2
        bcallback_ncalls, efcallback_ncalls = 0, 0

        runner = Runner()
        ProgressBar().attach_on(runner)
        LambdaReducer("total_output", lambda x, y: x + y).attach_on(runner)
        MeanReducer("mean_output").attach_on(runner)

        def bcallback(state):
            nonlocal bcallback_ncalls
            bcallback_ncalls += 1
            state["output"] = state["batch"]
            if state["stage"] == "first" and state["n_iters"] == 2:
                state["running"] = False

        def efcallback(state):
            nonlocal efcallback_ncalls
            efcallback_ncalls += 1
            assert state["total_output"] == sum(batches)
            assert state["mean_output"] == pytest.approx(sum(batches) / len(batches))

        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, efcallback)
        runner.state["stage"] = "first"
        runner.run(batches, max_epoch)
        with open(tmp_path / "ckpt.pkl", "wb") as f:
            pickle.dump(runner.state, f)

        with open(tmp_path / "ckpt.pkl", "rb") as f:
            ckpt = pickle.load(f)
        runner = Runner()
        ProgressBar().attach_on(runner)
        LambdaReducer("total_output", lambda x, y: x + y).attach_on(runner)
        MeanReducer("mean_output").attach_on(runner)
        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, efcallback)
        runner.state.update(ckpt)
        runner.state["stage"] = "second"
        runner.resume(repeat_last_batch=True)

        assert bcallback_ncalls == len(batches) * max_epoch + 1
        assert efcallback_ncalls == max_epoch

    def test_stopped_on_epoch(self, tmp_path):
        from rnnr.attachments import LambdaReducer, MeanReducer, ProgressBar

        batches, max_epoch = list(range(5)), 2
        bcallback_ncalls, efcallback_ncalls = 0, 0

        runner = Runner()
        ProgressBar().attach_on(runner)
        LambdaReducer("total_output", lambda x, y: x + y).attach_on(runner)
        MeanReducer("mean_output").attach_on(runner)

        def bcallback(state):
            nonlocal bcallback_ncalls
            bcallback_ncalls += 1
            state["output"] = state["batch"]

        def efcallback(state):
            nonlocal efcallback_ncalls
            efcallback_ncalls += 1
            assert state["total_output"] == sum(batches)
            assert state["mean_output"] == pytest.approx(sum(batches) / len(batches))
            if state["stage"] == "first" and state["epoch"] == 1:
                state["running"] = False

        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, efcallback)
        runner.state["stage"] = "first"
        runner.run(batches, max_epoch)
        with open(tmp_path / "ckpt.pkl", "wb") as f:
            pickle.dump(runner.state, f)

        with open(tmp_path / "ckpt.pkl", "rb") as f:
            ckpt = pickle.load(f)
        runner = Runner()
        ProgressBar().attach_on(runner)
        LambdaReducer("total_output", lambda x, y: x + y).attach_on(runner)
        MeanReducer("mean_output").attach_on(runner)
        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, efcallback)
        runner.state.update(ckpt)
        runner.state["stage"] = "second"
        runner.resume()

        assert bcallback_ncalls == len(batches) * max_epoch
        assert efcallback_ncalls == max_epoch
