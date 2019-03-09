from unittest.mock import Mock, call
import copy

from rnnr import Event


# https://stackoverflow.com/questions/29516339/how-to-mock-calls-to-function-that-receives-mutable-object-as-parameter
class DeepcopyMock(Mock):
    def _mock_call(self, *args, **kwargs):
        return super()._mock_call(*copy.deepcopy(args), **copy.deepcopy(kwargs))


class TestRun:
    def test_ok(self, runner):
        mock_fn = Mock()
        batches = range(10)
        runner.run(mock_fn, batches)
        assert mock_fn.mock_calls == [call(b) for b in batches]

    def test_more_than_one_epoch(self, runner):
        mock_fn = Mock()
        batches, max_epoch = range(10), 5
        runner.run(mock_fn, batches, max_epoch=max_epoch)
        assert mock_fn.mock_calls == [call(b) for _ in range(max_epoch) for b in batches]


class TestAppendHandler:
    def test_epoch_started(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.EPOCH_STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=None, output=None))
            for e in range(1, max_epoch + 1)
        ]

    def test_batch_started(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.BATCH_STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=b, output=None))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]

    def test_batch_finished(self, runner):
        mock_handler = DeepcopyMock()
        mock_fn = Mock(wraps=lambda b: b**2)
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.BATCH_FINISHED, mock_handler)
        runner.run(mock_fn, batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(
                dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=b, output=mock_fn(b)))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]

    def test_epoch_finished(self, runner):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5

        runner.append_handler(Event.EPOCH_FINISHED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)

        assert mock_handler.mock_calls == [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=None, output=None))
            for e in range(1, max_epoch + 1)
        ]


def test_stop(runner):
    mock_bfhandler = Mock()
    mock_efhandler = Mock()
    batches = range(10)

    def bshandler(state):
        if state['batch'] == 3:
            runner.stop()

    runner.append_handler(Event.BATCH_STARTED, bshandler)
    runner.append_handler(Event.BATCH_FINISHED, mock_bfhandler)
    runner.append_handler(Event.EPOCH_FINISHED, mock_efhandler)
    runner.run(Mock(), batches)

    assert mock_bfhandler.call_count == 4
    assert mock_efhandler.called
