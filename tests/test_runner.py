from unittest.mock import Mock, patch

from rnnr import Event


class TestRun:
    def test_ok(self, runner):
        batches, n_calls = range(10), 0

        def batch_fn(state):
            nonlocal n_calls
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == 1
            assert state['epoch'] == 1
            assert state['batch'] == batches[n_calls]
            n_calls += 1

        state = runner.run(batch_fn, batches)

        assert state['batches'] == batches
        assert state['max_epoch'] == 1

    def test_more_than_one_epoch(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def batch_fn(state):
            nonlocal n_calls
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls // len(batches) + 1
            assert state['batch'] == batches[n_calls % len(batches)]
            n_calls += 1

        runner.run(batch_fn, batches, max_epoch=max_epoch)


class TestAppendHandler:
    def test_started(self, runner):
        batches, max_epoch = range(10), 5

        def on_started(state):
            assert set(state) == {'runner', 'batches', 'max_epoch'}
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch

        runner.append_handler(Event.STARTED, on_started)
        runner.run(Mock(), batches, max_epoch=max_epoch)

    def test_epoch_started(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_epoch_started(state):
            nonlocal n_calls
            assert set(state) == {'runner', 'batches', 'max_epoch', 'epoch'}
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls + 1
            n_calls += 1

        runner.append_handler(Event.EPOCH_STARTED, on_epoch_started)
        runner.run(Mock(), batches, max_epoch=max_epoch)

    def test_batch_started(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_batch_started(state):
            nonlocal n_calls
            assert set(state) == {'runner', 'batches', 'max_epoch', 'epoch', 'batch'}
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls // len(batches) + 1
            assert state['batch'] == batches[n_calls % len(batches)]
            n_calls += 1

        runner.append_handler(Event.BATCH_STARTED, on_batch_started)
        runner.run(Mock(), batches, max_epoch=max_epoch)

    def test_batch_finished(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def batch_fn(state):
            state['output'] = state['batch']**2

        def on_batch_finished(state):
            nonlocal n_calls
            assert set(state) == {'runner', 'batches', 'max_epoch', 'epoch', 'batch', 'output'}
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls // len(batches) + 1
            assert state['batch'] == batches[n_calls % len(batches)]
            assert state['output'] == state['batch']**2
            n_calls += 1

        runner.append_handler(Event.BATCH_FINISHED, on_batch_finished)
        runner.run(batch_fn, batches, max_epoch=max_epoch)

    def test_epoch_finished(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_epoch_finished(state):
            nonlocal n_calls
            assert set(state) == {'runner', 'batches', 'max_epoch', 'epoch'}
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls + 1
            n_calls += 1

        runner.append_handler(Event.EPOCH_FINISHED, on_epoch_finished)
        runner.run(Mock(), batches, max_epoch=max_epoch)

    def test_finished(self, runner):
        batches, max_epoch = range(10), 5

        def on_finished(state):
            assert set(state) == {'runner', 'batches', 'max_epoch'}
            assert state['runner'] is runner
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch

        runner.append_handler(Event.FINISHED, on_finished)
        runner.run(Mock(), batches, max_epoch=max_epoch)


class TestStop:
    def test_on_batch_started(self, runner):
        mock_eshandler = Mock()
        mock_bfhandler = Mock()
        mock_efhandler = Mock()
        batches = range(10)

        def bshandler(state):
            if state['batch'] == 3:
                runner.stop()

        runner.append_handler(Event.EPOCH_STARTED, mock_eshandler)
        runner.append_handler(Event.BATCH_STARTED, bshandler)
        runner.append_handler(Event.BATCH_FINISHED, mock_bfhandler)
        runner.append_handler(Event.EPOCH_FINISHED, mock_efhandler)
        runner.run(Mock(), batches, max_epoch=2)

        assert mock_eshandler.call_count == 1
        assert mock_bfhandler.call_count == 4
        assert mock_efhandler.call_count == 1

    def test_on_epoch_started(self, runner):
        mock_bshandler = Mock()
        mock_bfhandler = Mock()
        mock_efhandler = Mock()
        batches = range(10)

        def eshandler(state):
            if state['epoch'] == 1:
                runner.stop()

        runner.append_handler(Event.EPOCH_STARTED, eshandler)
        runner.append_handler(Event.BATCH_STARTED, mock_bshandler)
        runner.append_handler(Event.BATCH_FINISHED, mock_bfhandler)
        runner.append_handler(Event.EPOCH_FINISHED, mock_efhandler)
        runner.run(Mock(), batches, max_epoch=7)

        assert mock_bshandler.call_count == 0
        assert mock_bfhandler.call_count == 0
        assert mock_efhandler.call_count == 1


def test_on_decorator(runner):
    with patch.object(runner, 'append_handler', autospec=True) as mock_append_handler:

        @runner.on(Event.BATCH_STARTED)
        def handler(state):
            pass

        mock_append_handler.assert_called_once_with(Event.BATCH_STARTED, handler)
