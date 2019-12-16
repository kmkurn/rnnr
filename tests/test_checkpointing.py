from unittest.mock import Mock, call
import pickle

from rnnr.callbacks import checkpoint


def test_ok(tmp_path):
    max_epoch, max_saved = 5, 2
    objs_values = {
        'model': [f'MODEL_{i}' for i in range(max_epoch)],
        'opt': [f'OPT_{i}' for i in range(max_epoch)],
    }

    ckpt_name = 'ckpt'
    callback = checkpoint(ckpt_name, under=tmp_path, at_most=max_saved)
    state = {}
    for i in range(max_epoch):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        state.update({ckpt_name: ckpt, 'epoch': i + 1})
        callback(state)

    assert len(list(tmp_path.glob(f'*{ckpt_name}.pkl'))) == max_saved
    for i in range(max_saved):
        path = tmp_path / f'{max_epoch - i}_{ckpt_name}.pkl'
        assert path.exists()
        with open(path, 'rb') as f:
            assert pickle.load(f) == {
                name: values[max_epoch - i - 1]
                for name, values in objs_values.items()
            }


def test_conditional(tmp_path):
    max_epoch, max_saved = 5, 2
    objs_values = {
        'model': [f'MODEL_{i}' for i in range(max_epoch)],
        'opt': [f'OPT_{i}' for i in range(max_epoch)],
    }
    better_epochs = {1, 3, 5}

    ckpt_name = 'ckpt'
    callback = checkpoint(ckpt_name, under=tmp_path, at_most=max_saved, when='better')
    state = {}
    for i in range(max_epoch):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        state.update({ckpt_name: ckpt, 'epoch': i + 1, 'better': i + 1 in better_epochs})
        callback(state)

    saved_epochs = [3, 5]  # the last 2
    assert {p.name for p in tmp_path.glob(f'*{ckpt_name}.pkl')} == \
        {f'{e}_{ckpt_name}.pkl' for e in saved_epochs}
    for e in saved_epochs:
        path = tmp_path / f'{e}_{ckpt_name}.pkl'
        assert path.exists()
        with open(path, 'rb') as f:
            assert pickle.load(f) == {
                name: values[e - 1]
                for name, values in objs_values.items()
            }


def test_save_fn(tmp_path):
    max_epoch = 5
    objs_values = {
        'model': [f'MODEL_{i}' for i in range(max_epoch)],
        'opt': [f'OPT_{i}' for i in range(max_epoch)],
    }
    mock_save_fn = Mock()

    ckpt_name = 'ckpt'
    callback = checkpoint(ckpt_name, under=tmp_path, using=mock_save_fn)
    for i in range(max_epoch):
        ckpt = {name: values[i] for name, values in objs_values.items()}
        callback({ckpt_name: ckpt, 'epoch': i + 1})

    assert mock_save_fn.mock_calls == [
        call({name: values[i]
              for name, values in objs_values.items()}, tmp_path / f'{i+1}_{ckpt_name}.pkl')
        for i in range(max_epoch)
    ]
