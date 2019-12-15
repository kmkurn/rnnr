from rnnr.callbacks import maybe_stop_early


def test_ok(runner):
    patience, counter, check, state = 7, 'cnt', 'check', {'running': True}
    callback = maybe_stop_early(patience=patience, check=check, counter=counter)

    for i in range(patience):
        state[check] = False
        callback(state)
        assert state['running']
        assert state[counter] == i + 1

    state[check] = True
    callback(state)
    assert state['running']
    assert state[counter] == 0

    for i in range(patience + 1):
        state[check] = False
        callback(state)
    assert not state['running']
    assert state[counter] == patience + 1
