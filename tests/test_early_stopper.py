from rnnr.callbacks import maybe_stop_early


def test_ok(runner):
    patience = 7
    callback, state = maybe_stop_early(patience=patience), {'running': True}

    for i in range(patience):
        state['better'] = False
        callback(state)
        assert state['running']
        assert state['n_bad_calls'] == i + 1

    state['better'] = True
    callback(state)
    assert state['running']
    assert state['n_bad_calls'] == 0

    for i in range(patience + 1):
        state['better'] = False
        callback(state)
    assert not state['running']
    assert state['n_bad_calls'] == patience + 1
