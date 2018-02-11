import numpy as np
import pytest

import rnn


def test_sigmoid():
    assert rnn.sigmoid(0) == 0.5
    assert rnn.sigmoid(100) == 1.0
    assert rnn.sigmoid(-100) < 0.001
    assert rnn.sigmoid(-100) >= 0


def test_logistic_loss():
    assert np.allclose(rnn.logistic_loss(1.0, 0.9999999999), [0])
    assert np.allclose(rnn.logistic_loss(0.0, 0.0000000001), [0])
    assert rnn.logistic_loss(1.0, 0.01) > 1
    assert rnn.logistic_loss(0.0, 0.99) > 1


def test_getitem_and_setitem_work():
    nn = rnn.RNN(n_a=4, n_x=1)
    for i in range(len(nn)):
        nn[i] = float(i)
    
    for i in range(len(nn)):
        assert nn[i] == float(i)


def test_getitem_raises_index_error():
    with pytest.raises(IndexError):
        rnn.RNN(n_a=1, n_x=1)[50]


def test_forward_prop_works():
    nn = rnn.RNN(4, 1)
    x = np.array([[0.3]])
    y, a = nn.forward_prop(x)
    assert y.shape == (1, 1)
    assert a.shape == (4, 1)
