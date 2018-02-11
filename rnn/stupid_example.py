import numpy as np

import rnn


def normshape(arr):
    return np.array(arr).reshape(-1, 1, 1).astype(np.float)


def create_stupid_example(size):
    '''
    Creates a stupid pair of sequences, each an array of shape
    (size, 1, 1). The first is a random sequence of ones and
    zeros, while the second is a sequence that indicates whether
    the most recent two entries in the first sequence are both
    ones.

    For example, if the first sequence is:

        [0, 1, 0, 1, 1, 0]

    Then the second sequence would be:

        [0, 0, 0, 0, 1, 0]
    '''

    inputs = np.random.randint(2, size=size)
    outputs = []
    last_i = None
    for i in inputs:
        output = 1 if i == 1 and last_i == 1 else 0
        last_i = i
        outputs.append(output)
    return (normshape(inputs), normshape(outputs))


if __name__ == '__main__':
    np.random.seed(1)
    inputs, outputs = create_stupid_example(3)

    print(f"inputs : {inputs.ravel()}")
    print(f"outputs: {outputs.ravel()}")

    nn = rnn.RNN(n_a=1, n_x=1)

    for i in range(2000):
        if i % 100 == 0:
            loss = nn.calculate_loss(inputs, outputs)
            print(f"loss at iter {i}: {loss}")
        grad = nn.calculate_gradient_very_slowly(inputs, outputs)
        nn.learn_very_slowly(grad)
