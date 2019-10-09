from unittest.mock import Mock, call, patch
import copy

from rnnr import Event


# https://stackoverflow.com/questions/29516339/how-to-mock-calls-to-function-that-receives-mutable-object-as-parameter
class DeepcopyMock(Mock):
    def _mock_call(self, *args, **kwargs):
        return super()._mock_call(*copy.deepcopy(args), **copy.deepcopy(kwargs))


class TestRun:
    def test_ok(self, runner):
        mock_fn = DeepcopyMock()
        batches = range(10)
        state = runner.run(mock_fn, batches)
        assert mock_fn.mock_calls == [
            call(dict(batches=batches, max_epoch=1, epoch=1, batch=b)) for b in batches
        ]
        assert state['batches'] == batches
        assert state['max_epoch'] == 1

    def test_more_than_one_epoch(self, runner):
        mock_fn = DeepcopyMock()
        batches, max_epoch = range(10), 5
        runner.run(mock_fn, batches, max_epoch=max_epoch)
        assert mock_fn.mock_calls == [
            call(dict(batches=batches, max_epoch=max_epoch, epoch=e, batch=b))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]


class TestAppendHandler:
    def test_started(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        mock_handler.assert_called_once_with(dict(max_epoch=max_epoch, batches=batches))

    def test_epoch_started(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.EPOCH_STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch))
            for e in range(1, max_epoch + 1)
        ]

    def test_batch_started(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.BATCH_STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=b))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]

    def test_batch_finished(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        def batch_fn(state):
            state['output'] = state['batch']**2

        runner.append_handler(Event.BATCH_FINISHED, mock_handler)
        runner.run(batch_fn, batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=b, output=b**2))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]

    def test_epoch_finished(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.EPOCH_FINISHED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch))
            for e in range(1, max_epoch + 1)
        ]

    def test_finished(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.FINISHED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        mock_handler.assert_called_once_with(dict(max_epoch=max_epoch, batches=batches))


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
