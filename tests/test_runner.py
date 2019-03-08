from unittest.mock import Mock, call
import copy

from rnnr import Event, Runner


# https://stackoverflow.com/questions/29516339/how-to-mock-calls-to-function-that-receives-mutable-object-as-parameter
class DeepcopyMock(Mock):
    def _mock_call(self, *args, **kwargs):
        return super()._mock_call(*copy.deepcopy(args), **copy.deepcopy(kwargs))


def make_runner():
    return Runner()


class TestRun:
    def test_ok(self):
        mock_fn = Mock()
        batches = range(10)
        runner = make_runner()
        runner.run(mock_fn, batches)
        assert mock_fn.mock_calls == [call(b) for b in batches]

    def test_more_than_one_epoch(self):
        mock_fn = Mock()
        batches, max_epoch = range(10), 5
        runner = make_runner()
        runner.run(mock_fn, batches, max_epoch=max_epoch)
        expected_calls = [call(b) for _ in range(1, max_epoch + 1) for b in batches]
        assert mock_fn.mock_calls == expected_calls

    def test_epoch_started_handler(self):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5
        runner = make_runner()
        runner.append_handler(Event.EPOCH_STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)
        expected_calls = [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=None, output=None))
            for e in range(1, max_epoch + 1)
        ]
        assert mock_handler.mock_calls == expected_calls

    def test_batch_started_handler(self):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5
        runner = make_runner()
        runner.append_handler(Event.BATCH_STARTED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)
        expected_calls = [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=b, output=None))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]
        assert mock_handler.mock_calls == expected_calls

    def test_batch_finished_handler(self):
        mock_handler = DeepcopyMock()
        mock_fn = Mock(return_value=200)
        batches, max_epoch = range(10), 5
        runner = make_runner()
        runner.append_handler(Event.BATCH_FINISHED, mock_handler)
        runner.run(mock_fn, batches, max_epoch=max_epoch)
        expected_calls = [
            call(
                dict(
                    batches=batches,
                    epoch=e,
                    max_epoch=max_epoch,
                    batch=b,
                    output=mock_fn.return_value))
            for e in range(1, max_epoch + 1)
            for b in batches
        ]
        assert mock_handler.mock_calls == expected_calls

    def test_epoch_finished_handler(self):
        mock_handler = DeepcopyMock()
        batches, max_epoch = range(10), 5
        runner = make_runner()
        runner.append_handler(Event.EPOCH_FINISHED, mock_handler)
        runner.run(Mock(), batches, max_epoch=max_epoch)
        expected_calls = [
            call(dict(batches=batches, epoch=e, max_epoch=max_epoch, batch=None, output=None))
            for e in range(1, max_epoch + 1)
        ]
        assert mock_handler.mock_calls == expected_calls


def test_stop():
    mock_bfhandler = Mock()
    mock_efhandler = Mock()
    batches = range(10)
    runner = make_runner()

    def bshandler(state):
        if state['batch'] == 3:
            runner.stop()

    runner.append_handler(Event.BATCH_STARTED, bshandler)
    runner.append_handler(Event.BATCH_FINISHED, mock_bfhandler)
    runner.append_handler(Event.EPOCH_FINISHED, mock_efhandler)
    runner.run(Mock(), batches)

    assert mock_bfhandler.call_count == 4
    assert mock_efhandler.called
