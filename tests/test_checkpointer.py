from unittest.mock import Mock, call
import pickle

import pytest

from rnnr.handlers import Checkpointer


def test_ok(tmp_path):
    n_calls = 5
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(n_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(n_calls)],
    }

    ckptr = Checkpointer(tmp_path)
    for i in range(n_calls):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        ckptr({'checkpoint': ckpt})

    for name in objs_values:
        path = tmp_path / f'{n_calls}_{name}'
        assert path.exists()
        with open(path, 'rb') as f:
            assert pickle.load(f) == objs_values[name][-1]


def test_max_saved(tmp_path):
    n_calls, max_saved = 5, 3
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(n_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(n_calls)],
    }

    ckptr = Checkpointer(tmp_path, max_saved=max_saved)
    for i in range(n_calls):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        ckptr({'checkpoint': ckpt})

    for name in objs_values:
        assert len(list(tmp_path.glob(f'*_{name}'))) == max_saved
        for i in range(max_saved):
            path = tmp_path / f'{n_calls - i}_{name}'
            assert path.exists()
            with open(path, 'rb') as f:
                assert pickle.load(f) == objs_values[name][n_calls - i - 1]


def test_loss_key(tmp_path):
    n_calls, max_saved = 5, 2
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(n_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(n_calls)],
    }
    # new best losses are call #1, #3, and #5
    losses = [3, 4, 2, 4, 1]
    min_losses = [3, 3, 2, 2, 1]

    ckptr = Checkpointer(tmp_path, max_saved=max_saved, loss_key='loss')
    for i in range(n_calls):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        ckptr({'loss': losses[i], 'checkpoint': ckpt})
        assert ckptr.min_loss == pytest.approx(min_losses[i])

    # saved call are #3 and #5 (the last 2)
    saved_calls = [3, 5]
    for name in objs_values:
        assert len(list(tmp_path.glob(f'*_{name}'))) == max_saved
        for c in saved_calls:
            path = tmp_path / f'{c}_{name}'
            assert path.exists()
            with open(path, 'rb') as f:
                assert pickle.load(f) == objs_values[name][c - 1]


def test_checkpoint_key(tmp_path):
    n_calls = 5
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(n_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(n_calls)],
    }

    ckptr = Checkpointer(tmp_path, checkpoint_key='foo')
    for i in range(n_calls):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        ckptr({'foo': ckpt})

    for name in objs_values:
        path = tmp_path / f'{n_calls}_{name}'
        assert path.exists()
        with open(path, 'rb') as f:
            assert pickle.load(f) == objs_values[name][-1]


def test_save_fn(tmp_path):
    n_calls = 5
    objs_values = {
        'model.pkl': [f'MODEL_{i}' for i in range(n_calls)],
        'opt.pkl': [f'OPT_{i}' for i in range(n_calls)],
    }
    mock_save_fn = Mock()

    ckptr = Checkpointer(tmp_path, save_fn=mock_save_fn)
    for i in range(n_calls):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        ckptr({'checkpoint': ckpt})

    assert mock_save_fn.mock_calls == [
        call(objs_values[name][i], tmp_path / f'{i+1}_{name}')
        for i in range(n_calls)
        for name in objs_values
    ]
