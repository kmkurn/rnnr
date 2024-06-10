from unittest.mock import Mock
import pickle

import pytest

from rnnr import Event, Runner


def test_init():
    def do_nothing(*args, **kwargs):
        pass

    runner = Runner(do_nothing)

    assert runner.on_batch == do_nothing


@pytest.mark.parametrize("cb_regis", ["single", "multi", "decorator"])
def test_run_with_callbacks(cb_regis):
    call_hist = []

    def on_batch(epoch, batch_idx, batch):
        call_hist.append(("on_batch", (epoch, batch_idx, batch)))
        return epoch, batch_idx

    max_epoch = 2
    runner = Runner(on_batch, max_epoch)

    if cb_regis == "decorator":

        @runner.on(Event.STARTED)
        def on_started1():
            call_hist.append(("on_started1", ()))

        @runner.on(Event.STARTED)
        def on_started2():
            call_hist.append(("on_started2", ()))

        @runner.on(Event.EPOCH_STARTED)
        def on_epoch_started1(epoch):
            call_hist.append(("on_epoch_started1", (epoch,)))

        @runner.on(Event.EPOCH_STARTED)
        def on_epoch_started2(epoch):
            call_hist.append(("on_epoch_started2", (epoch,)))

        @runner.on(Event.BATCH_STARTED)
        def on_batch_started1(epoch, batch_idx, batch):
            call_hist.append(("on_batch_started1", (epoch, batch_idx, batch)))
            return batch + 1

        @runner.on(Event.BATCH_STARTED)
        def on_batch_started2(epoch, batch_idx, batch):
            call_hist.append(("on_batch_started2", (epoch, batch_idx, batch)))
            return batch ** 2

        @runner.on(Event.BATCH_FINISHED)
        def on_batch_finished1(epoch, batch_idx, batch, output):
            call_hist.append(("on_batch_finished1", (epoch, batch_idx, batch, output)))

        @runner.on(Event.BATCH_FINISHED)
        def on_batch_finished2(epoch, batch_idx, batch, output):
            call_hist.append(("on_batch_finished2", (epoch, batch_idx, batch, output)))

        @runner.on(Event.EPOCH_FINISHED)
        def on_epoch_finished1(epoch):
            call_hist.append(("on_epoch_finished1", (epoch,)))

        @runner.on(Event.EPOCH_FINISHED)
        def on_epoch_finished2(epoch):
            call_hist.append(("on_epoch_finished2", (epoch,)))

        @runner.on(Event.FINISHED)
        def on_finished1():
            call_hist.append(("on_finished1", ()))

        @runner.on(Event.FINISHED)
        def on_finished2():
            call_hist.append(("on_finished2", ()))

    else:

        def on_started1():
            call_hist.append(("on_started1", ()))

        def on_started2():
            call_hist.append(("on_started2", ()))

        def on_epoch_started1(epoch):
            call_hist.append(("on_epoch_started1", (epoch,)))

        def on_epoch_started2(epoch):
            call_hist.append(("on_epoch_started2", (epoch,)))

        def on_batch_started1(epoch, batch_idx, batch):
            call_hist.append(("on_batch_started1", (epoch, batch_idx, batch)))
            return batch + 1

        def on_batch_started2(epoch, batch_idx, batch):
            call_hist.append(("on_batch_started2", (epoch, batch_idx, batch)))
            return batch ** 2

        def on_batch_finished1(epoch, batch_idx, batch, output):
            call_hist.append(("on_batch_finished1", (epoch, batch_idx, batch, output)))

        def on_batch_finished2(epoch, batch_idx, batch, output):
            call_hist.append(("on_batch_finished2", (epoch, batch_idx, batch, output)))

        def on_epoch_finished1(epoch):
            call_hist.append(("on_epoch_finished1", (epoch,)))

        def on_epoch_finished2(epoch):
            call_hist.append(("on_epoch_finished2", (epoch,)))

        def on_finished1():
            call_hist.append(("on_finished1", ()))

        def on_finished2():
            call_hist.append(("on_finished2", ()))

    if cb_regis == "multi":
        runner.on(Event.STARTED, [on_started1, on_started2])
        runner.on(Event.EPOCH_STARTED, [on_epoch_started1, on_epoch_started2])
        runner.on(Event.BATCH_STARTED, [on_batch_started1, on_batch_started2])
        runner.on(Event.BATCH_FINISHED, [on_batch_finished1, on_batch_finished2])
        runner.on(Event.EPOCH_FINISHED, [on_epoch_finished1, on_epoch_finished2])
        runner.on(Event.FINISHED, [on_finished1, on_finished2])
    elif cb_regis == "single":
        runner.on(Event.STARTED, on_started1)
        runner.on(Event.STARTED, on_started2)
        runner.on(Event.EPOCH_STARTED, on_epoch_started1)
        runner.on(Event.EPOCH_STARTED, on_epoch_started2)
        runner.on(Event.BATCH_STARTED, on_batch_started1)
        runner.on(Event.BATCH_STARTED, on_batch_started2)
        runner.on(Event.BATCH_FINISHED, on_batch_finished1)
        runner.on(Event.BATCH_FINISHED, on_batch_finished2)
        runner.on(Event.EPOCH_FINISHED, on_epoch_finished1)
        runner.on(Event.EPOCH_FINISHED, on_epoch_finished2)
        runner.on(Event.FINISHED, on_finished1)
        runner.on(Event.FINISHED, on_finished2)
    batches = [3, 5]

    runner.run(batches)

    assert call_hist == [
        ("on_started1", ()),
        ("on_started2", ()),
        # Epoch 1
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
        # Epoch 2
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
        # Finish
        ("on_finished1", ()),
        ("on_finished2", ()),
    ]


class TestStop:
    def test_on_batch(self, runner):
        mock_escallback = Mock()
        mock_efcallback = Mock()
        n_calls = 0
        batches = range(10)

        def bcallback(state):
            nonlocal n_calls
            n_calls += 1
            if state["batch"] == 3:
                state["running"] = False

        runner.on(Event.EPOCH_STARTED, mock_escallback)
        runner.on(Event.BATCH, bcallback)
        runner.on(Event.EPOCH_FINISHED, mock_efcallback)
        runner.run(batches, max_epoch=2)

        assert mock_escallback.call_count == 1
        assert mock_efcallback.call_count == 0
        assert n_calls == 4
        assert runner.state["n_iters"] == 4
        assert runner.state["epoch"] == 1
        assert runner.state["batch"] == 3

    def test_on_epoch_started(self, runner):
        mock_efcallback = Mock()
        mock_bcallback = Mock()

        class MockBatches:
            n = 0

            def __iter__(self):
                self.n = 0
                return self

            def __next__(self):
                res = self.n
                self.n += 1
                if self.n >= 10:
                    raise StopIteration
                return res

        batches = MockBatches()

        def escallback(state):
            if state["epoch"] == 1:
                state["running"] = False

        runner.on(Event.EPOCH_STARTED, escallback)
        runner.on(Event.BATCH, mock_bcallback)
        runner.on(Event.EPOCH_FINISHED, mock_efcallback)
        runner.run(batches, max_epoch=7)

        assert mock_efcallback.call_count == 0
        assert mock_bcallback.call_count == 0
        assert batches.n == 0


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
