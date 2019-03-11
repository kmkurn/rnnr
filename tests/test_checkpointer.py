from unittest.mock import Mock, call
import pickle

from rnnr import Event
from rnnr.handlers import Checkpointer


def test_ok(tmp_path, runner):
    objs = {'model.pkl': {'foo': 'bar'}, 'opt.pkl': {'baz': 'quux'}}
    max_epoch = 5

    ckptr = Checkpointer(tmp_path, objs)
    runner.append_handler(Event.EPOCH_FINISHED, ckptr)
    runner.run(Mock(), range(7), max_epoch=max_epoch)

    for name, obj in objs.items():
        path = tmp_path / f'{max_epoch}_{name}'
        with open(path, 'rb') as f:
            assert pickle.load(f) == obj


def test_max_saved(tmp_path, runner):
    max_epoch, max_saved = 5, 3
    models = [{'foo': f'bar_{e}'} for e in range(max_epoch)]
    opts = [{'baz': f'quux_{e}'} for e in range(max_epoch)]
    objs = {'model.pkl': None, 'opt.pkl': None}

    def update_objs(state):
        objs['model.pkl'] = models[state['epoch'] - 1]
        objs['opt.pkl'] = opts[state['epoch'] - 1]

    ckptr = Checkpointer(tmp_path, objs, max_saved=max_saved)
    runner.append_handler(Event.EPOCH_FINISHED, update_objs)
    runner.append_handler(Event.EPOCH_FINISHED, ckptr)
    runner.run(Mock(), range(7), max_epoch=max_epoch)

    for name, cand in zip(objs, (models, opts)):
        assert len(list(tmp_path.glob(f'*_{name}'))) == max_saved
        for i in range(max_saved):
            j = max_epoch - i
            path = tmp_path / f'{j}_{name}'
            with open(path, 'rb') as f:
                assert pickle.load(f) == cand[j - 1]


def test_loss_fn(tmp_path, runner):
    max_epoch, max_saved = 5, 2
    models = [{'foo': f'bar_{e}'} for e in range(max_epoch)]
    opts = [{'baz': f'quux_{e}'} for e in range(max_epoch)]
    objs = {'model.pkl': None, 'opt.pkl': None}
    # new best losses are epoch 1, 3, and 5
    losses = [3, 4, 2, 4, 1]
    loss_fn = lambda state: losses[state['epoch'] - 1]

    def update_objs(state):
        objs['model.pkl'] = models[state['epoch'] - 1]
        objs['opt.pkl'] = opts[state['epoch'] - 1]

    ckptr = Checkpointer(tmp_path, objs, max_saved=max_saved, loss_fn=loss_fn)
    runner.append_handler(Event.EPOCH_FINISHED, update_objs)
    runner.append_handler(Event.EPOCH_FINISHED, ckptr)
    runner.run(Mock(), range(7), max_epoch=max_epoch)

    # saved epochs are 3 and 5 (the last 2)
    saved_epochs = [3, 5]
    for name, cand in zip(objs, (models, opts)):
        assert len(list(tmp_path.glob(f'*_{name}'))) == max_saved
        for e in saved_epochs:
            path = tmp_path / f'{e}_{name}'
            with open(path, 'rb') as f:
                assert pickle.load(f) == cand[e - 1]


def test_save_fn(tmp_path, runner):
    max_epoch = 5
    models = [{'foo': f'bar_{e}'} for e in range(max_epoch)]
    objs = {'model.pkl': None}
    mock_save_fn = Mock()

    def update_objs(state):
        objs['model.pkl'] = models[state['epoch'] - 1]

    ckptr = Checkpointer(tmp_path, objs, save_fn=mock_save_fn)
    runner.append_handler(Event.EPOCH_FINISHED, update_objs)
    runner.append_handler(Event.EPOCH_FINISHED, ckptr)
    runner.run(Mock(), range(7), max_epoch=max_epoch)

    assert mock_save_fn.mock_calls == [
        call(tmp_path / f'{e+1}_model.pkl', models[e]) for e in range(max_epoch)
    ]
