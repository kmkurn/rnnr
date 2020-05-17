from unittest.mock import Mock
import pickle

import pytest

from rnnr import Event, Runner


def test_init():
    r = Runner()
    assert len(r.state) == 0


def test_run(runner):
    batches, max_epoch = range(10), 5
    runner.run(batches, max_epoch=max_epoch)
    state = runner.state

    assert state["batches"] == batches
    assert state["max_epoch"] == max_epoch
    assert state["n_iters"] == len(batches) * max_epoch
    assert not state["running"]


class TestOn:
    def test_started(self, runner):
        batches, max_epoch = range(10), 5

        def on_started(state):
            assert state["batches"] == batches
            assert state["max_epoch"] == max_epoch
            assert state["n_iters"] == 0
            assert state["running"]
            assert state["epoch"] == 0

        runner.on(Event.STARTED, on_started)
        runner.run(batches, max_epoch=max_epoch)

    def test_epoch_started(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_epoch_started(state):
            nonlocal n_calls
            assert state["batches"] == batches
            assert state["max_epoch"] == max_epoch
            assert state["epoch"] == n_calls + 1
            assert state["n_iters"] == n_calls * len(batches)
            assert state["running"]
            n_calls += 1

        runner.on(Event.EPOCH_STARTED, on_epoch_started)
        runner.run(batches, max_epoch=max_epoch)

    def test_batch(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        @runner.on(Event.BATCH)
        def on_batch(state):
            nonlocal n_calls
            assert state["batches"] == batches
            assert state["max_epoch"] == max_epoch
            assert state["epoch"] == n_calls // len(batches) + 1
            assert state["batch"] == batches[n_calls % len(batches)]
            assert state["n_iters"] == n_calls + 1
            assert state["running"]
            n_calls += 1

        runner.run(batches, max_epoch=max_epoch)
        assert n_calls == len(batches) * max_epoch

    def test_epoch_finished(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_epoch_finished(state):
            nonlocal n_calls
            assert state["batches"] == batches
            assert state["max_epoch"] == max_epoch
            assert state["epoch"] == n_calls + 1
            assert state["n_iters"] == (n_calls + 1) * len(batches)
            assert state["running"]
            n_calls += 1

        runner.on(Event.EPOCH_FINISHED, on_epoch_finished)
        runner.run(batches, max_epoch=max_epoch)

    def test_finished(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_finished(state):
            assert state["batches"] == batches
            assert state["max_epoch"] == max_epoch
            assert state["n_iters"] == max_epoch * len(batches)
            assert state["running"]
            nonlocal n_calls
            n_calls += 1

        runner.on(Event.FINISHED, on_finished)
        runner.run(batches, max_epoch=max_epoch)
        assert n_calls == 1

    def test_as_decorator(self, runner):
        n_calls, max_epoch = 0, 10

        @runner.on(Event.EPOCH_STARTED)
        def increment(state):
            nonlocal n_calls
            n_calls += 1

        runner.run(range(5), max_epoch=max_epoch)
        assert n_calls == max_epoch

    def test_multiple_callbacks(self, runner):
        mock_escb1, mock_escb2, max_epoch = Mock(), Mock(), 10
        runner.on(Event.EPOCH_STARTED, [mock_escb1, mock_escb2])
        runner.run(range(5), max_epoch=max_epoch)

        assert mock_escb1.call_count == max_epoch
        assert mock_escb2.call_count == max_epoch


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
