from unittest.mock import Mock

from rnnr import Event, Runner


def test_run(runner):
    batches, max_epoch = range(10), 5
    state = runner.run(batches, max_epoch=max_epoch)

    assert set(state) == {'batches', 'max_epoch', 'n_iters', 'running'}
    assert state['batches'] == batches
    assert state['max_epoch'] == max_epoch
    assert state['n_iters'] == len(batches) * max_epoch
    assert not state['running']


class TestOn:
    def test_started(self, runner):
        batches, max_epoch = range(10), 5

        def on_started(state):
            assert set(state) == {'batches', 'max_epoch', 'n_iters', 'running', 'epoch'}
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['n_iters'] == 0
            assert state['running']
            assert state['epoch'] == 0

        runner.on(Event.STARTED, on_started)
        runner.run(batches, max_epoch=max_epoch)

    def test_epoch_started(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_epoch_started(state):
            nonlocal n_calls
            assert set(state) == {'batches', 'max_epoch', 'epoch', 'n_iters', 'running'}
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls + 1
            assert state['n_iters'] == n_calls * len(batches)
            assert state['running']
            n_calls += 1

        runner.on(Event.EPOCH_STARTED, on_epoch_started)
        runner.run(batches, max_epoch=max_epoch)

    def test_batch_started(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_batch_started(state):
            nonlocal n_calls
            assert set(state) == {
                'batches', 'max_epoch', 'epoch', 'batch', 'n_iters', 'running'
            }
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls // len(batches) + 1
            assert state['batch'] == batches[n_calls % len(batches)]
            assert state['n_iters'] == n_calls + 1
            assert state['running']
            n_calls += 1

        runner.on(Event.BATCH_STARTED, on_batch_started)
        runner.run(batches, max_epoch=max_epoch)

    def test_batch(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        @runner.on(Event.BATCH)
        def on_batch(state):
            nonlocal n_calls
            assert set(state) == {
                'batches', 'max_epoch', 'epoch', 'batch', 'n_iters', 'running'
            }
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls // len(batches) + 1
            assert state['batch'] == batches[n_calls % len(batches)]
            assert state['n_iters'] == n_calls + 1
            assert state['running']
            n_calls += 1

        runner.run(batches, max_epoch=max_epoch)
        assert n_calls == len(batches) * max_epoch

    def test_batch_finished(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        @runner.on(Event.BATCH)
        def on_batch(state):
            state['output'] = state['batch']**2

        def on_batch_finished(state):
            nonlocal n_calls
            assert set(state) == {
                'batches', 'max_epoch', 'epoch', 'batch', 'output', 'n_iters', 'running'
            }
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls // len(batches) + 1
            assert state['batch'] == batches[n_calls % len(batches)]
            assert state['output'] == state['batch']**2
            assert state['n_iters'] == n_calls + 1
            assert state['running']
            n_calls += 1

        runner.on(Event.BATCH_FINISHED, on_batch_finished)
        runner.run(batches, max_epoch=max_epoch)

    def test_epoch_finished(self, runner):
        batches, max_epoch, n_calls = range(10), 5, 0

        def on_epoch_finished(state):
            nonlocal n_calls
            assert set(state) == {'batches', 'max_epoch', 'epoch', 'n_iters', 'running'}
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['epoch'] == n_calls + 1
            assert state['n_iters'] == (n_calls + 1) * len(batches)
            assert state['running']
            n_calls += 1

        runner.on(Event.EPOCH_FINISHED, on_epoch_finished)
        runner.run(batches, max_epoch=max_epoch)

    def test_finished(self, runner):
        batches, max_epoch = range(10), 5

        def on_finished(state):
            assert set(state) == {'batches', 'max_epoch', 'n_iters', 'running'}
            assert state['batches'] == batches
            assert state['max_epoch'] == max_epoch
            assert state['n_iters'] == max_epoch * len(batches)
            assert not state['running']

        runner.on(Event.FINISHED, on_finished)
        runner.run(batches, max_epoch=max_epoch)

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
    def test_on_batch_started(self, runner):
        mock_escallback = Mock()
        mock_bfcallback = Mock()
        mock_efcallback = Mock()
        batches = range(10)

        def bscallback(state):
            if state['batch'] == 3:
                state['running'] = False

        runner.on(Event.EPOCH_STARTED, mock_escallback)
        runner.on(Event.BATCH_STARTED, bscallback)
        runner.on(Event.BATCH_FINISHED, mock_bfcallback)
        runner.on(Event.EPOCH_FINISHED, mock_efcallback)
        runner.run(batches, max_epoch=2)

        assert mock_escallback.call_count == 1
        assert mock_bfcallback.call_count == 4
        assert mock_efcallback.call_count == 1

    def test_on_epoch_started(self, runner):
        mock_bscallback = Mock()
        mock_bfcallback = Mock()
        mock_efcallback = Mock()
        batches = range(10)

        def escallback(state):
            if state['epoch'] == 1:
                state['running'] = False

        runner.on(Event.EPOCH_STARTED, escallback)
        runner.on(Event.BATCH_STARTED, mock_bscallback)
        runner.on(Event.BATCH_FINISHED, mock_bfcallback)
        runner.on(Event.EPOCH_FINISHED, mock_efcallback)
        runner.run(batches, max_epoch=7)

        assert mock_bscallback.call_count == 0
        assert mock_bfcallback.call_count == 0
        assert mock_efcallback.call_count == 1


def test_init_state():
    init_state = {'batches': range(7)}
    runner = Runner(init_state)

    @runner.on(Event.STARTED)
    def on_started(state):
        for k, v in init_state.items():
            assert state[k] == v

    runner.run(range(5))
