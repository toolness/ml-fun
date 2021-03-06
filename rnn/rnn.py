import numpy as np


def init_weight_matrix(*shape):
    return np.random.rand(*shape) * 0.001


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def logistic_loss(y, pred_y):
    return -(y * np.log(pred_y)) - ((1 - y) * np.log(1 - pred_y))


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
        m = x.shape[1]

        if a_prev is None:
            a_prev = np.zeros((self.n_a, m))

        assert x.shape == (self.n_x, m)
        assert a_prev.shape == (self.n_a, m)

        a = np.tanh(np.dot(self.waa, a_prev) +
                    np.dot(self.wax, x) + self.ba)

        assert a.shape == (self.n_a, m)

        y = sigmoid(np.dot(self.wya, a) + self.by)

        assert y.shape == (self.n_y, m)

        return (y, a)

    def forward_prop_seq(self, inputs):
        a_prev = None
        for x in inputs:
            y, a = self.forward_prop(x, a_prev)
            a_prev = a
            yield y

    def calculate_loss_and_accuracy(self, inputs, outputs):
        m = inputs.shape[2]
        total_loss = np.zeros((1, m))
        num_correct = 0
        for y, pred_y in zip(outputs, self.forward_prop_seq(inputs)):
            total_loss += logistic_loss(y, pred_y)
            pred_y = (pred_y > 0.5).astype(np.float)
            num_correct += np.sum((pred_y == y).astype(np.float))
        return np.sum(total_loss) / m, num_correct / outputs.size

    def calculate_loss(self, inputs, outputs):
        m = inputs.shape[2]
        total_loss = np.zeros((1, m))
        for y, pred_y in zip(outputs, self.forward_prop_seq(inputs)):
            total_loss += logistic_loss(y, pred_y)
        return np.sum(total_loss) / m

    def calculate_gradient_very_slowly(self, inputs, outputs,
                                       epsilon=1e-5):
        gradient = np.zeros(len(self))
        half_epsilon = epsilon / 2.0
        for idx in range(len(self)):
            orig_val = self[idx]
            self[idx] = orig_val - half_epsilon
            loss1 = self.calculate_loss(inputs, outputs)
            self[idx] = orig_val + half_epsilon
            loss2 = self.calculate_loss(inputs, outputs)
            self[idx] = orig_val
            grad = (loss2 - loss1) / epsilon
            gradient[idx] = grad
        return gradient

    def learn_very_slowly(self, gradient, learning_rate=0.1):
        for idx in range(len(self)):
            self[idx] = self[idx] - learning_rate * gradient[idx]
