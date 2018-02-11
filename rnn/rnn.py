import numpy as np


def init_weight_matrix(*shape):
    return np.random.rand(*shape) * 0.001


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class RNN:
    MODEL_PROPS = ['waa', 'wax', 'ba', 'wya', 'by']

    def __init__(self, n_a, n_x):
        self.n_a = n_a
        self.n_x = n_x
        self.n_y = 1
        self.waa = init_weight_matrix(n_a, n_a)
        self.wax = init_weight_matrix(n_a, n_x)
        self.ba = np.zeros((n_a, 1))
        self.wya = init_weight_matrix(self.n_y, n_a)
        self.by = np.zeros((self.n_y, 1))
        self.size = self._calc_size()

        self.validate()

    def _calc_size(self):
        return sum(getattr(self, prop).size for prop in self.MODEL_PROPS)

    def _get_prop_idx(self, i):
        idx = i
        for prop in self.MODEL_PROPS:
            size = getattr(self, prop).size
            if idx < size:
                return (prop, idx)
            idx -= size
        raise IndexError(i)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        prop, idx = self._get_prop_idx(idx)
        return getattr(self, prop).item(idx)

    def __setitem__(self, idx, val):
        prop, idx = self._get_prop_idx(idx)
        array = getattr(self, prop)
        array.reshape(-1)[idx] = val

    def validate(self):
        assert self.waa.shape == (self.n_a, self.n_a)
        assert self.wax.shape == (self.n_a, self.n_x)
        assert self.ba.shape == (self.n_a, 1)
        assert self.wya.shape == (self.n_y, self.n_a)
        assert self.by.shape == (self.n_y, 1)

    def forward_prop(self, x, a_prev=None):
        if a_prev is None:
            a_prev = np.zeros((self.n_a, 1))

        assert x.shape == (self.n_x, 1)
        assert a_prev.shape == (self.n_a, 1)

        a = np.tanh(np.dot(self.waa, a_prev) +
                    np.dot(self.wax, x) + self.ba)

        assert a.shape == (self.n_a, 1)

        y = sigmoid(np.dot(self.wya, a) + self.by)

        assert y.shape == (self.n_y, 1)

        return (y, a)


if __name__ == '__main__':
    pass
