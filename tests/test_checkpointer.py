from unittest.mock import Mock, call
import pickle

from rnnr.handlers import Checkpointer


def test_ok(tmp_path):
    num_calls = 5
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(num_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(num_calls)],
    }
    objs = {'model.pkl': None, 'opt.pkl': None}

    ckptr = Checkpointer(tmp_path, objs)
    for i in range(num_calls):
        for name in objs:
            objs[name] = objs_values[name][i]
        ckptr({})

    for name in objs:
        path = tmp_path / f'{num_calls}_{name}'
        with open(path, 'rb') as f:
            assert pickle.load(f) == objs_values[name][-1]


def test_max_saved(tmp_path):
    num_calls, max_saved = 5, 3
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(num_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(num_calls)],
    }
    objs = {'model.pkl': None, 'opt.pkl': None}

    ckptr = Checkpointer(tmp_path, objs, max_saved=max_saved)
    for i in range(num_calls):
        for name in objs:
            objs[name] = objs_values[name][i]
        ckptr({})

    for name in objs:
        assert len(list(tmp_path.glob(f'*_{name}'))) == max_saved
        for i in range(max_saved):
            path = tmp_path / f'{num_calls - i}_{name}'
            with open(path, 'rb') as f:
                assert pickle.load(f) == objs_values[name][num_calls - i - 1]


def test_loss_key(tmp_path):
    num_calls, max_saved = 5, 2
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(num_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(num_calls)],
    }
    objs = {'model.pkl': None, 'opt.pkl': None}
    # new best losses are call #1, #3, and #5
    losses = [3, 4, 2, 4, 1]

    ckptr = Checkpointer(tmp_path, objs, max_saved=max_saved, loss_key='loss')
    for i in range(num_calls):
        for name in objs:
            objs[name] = objs_values[name][i]
        ckptr({'loss': losses[i]})

    # saved call are #3 and #5 (the last 2)
    saved_calls = [3, 5]
    for name in objs:
        assert len(list(tmp_path.glob(f'*_{name}'))) == max_saved
        for c in saved_calls:
            path = tmp_path / f'{c}_{name}'
            assert path.exists()
            with open(path, 'rb') as f:
                assert pickle.load(f) == objs_values[name][c - 1]


def test_save_fn(tmp_path):
    num_calls = 5
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(num_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(num_calls)],
    }
    objs = {'model.pkl': None, 'opt.pkl': None}
    mock_save_fn = Mock()

    ckptr = Checkpointer(tmp_path, objs, save_fn=mock_save_fn)
    for i in range(num_calls):
        for name in objs:
            objs[name] = objs_values[name][i]
        ckptr({})

    assert mock_save_fn.mock_calls == [
        call(objs_values[name][i], tmp_path / f'{i+1}_{name}')
        for i in range(num_calls)
        for name in objs
    ]
