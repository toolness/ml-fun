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


def main(print=print):
    np.random.seed(1)

    nn = rnn.RNN(n_a=2, n_x=1)

    print(f"Training a RNN with {nn.size} parameters on a stupid example set.\n")

    avg_loss = 0.0
    avg_acc = 0.0

    for i in range(10000):
        inputs, outputs = create_stupid_example(9)
        loss, acc = nn.calculate_loss_and_accuracy(inputs, outputs)
        avg_loss = 0.8 * avg_loss + 0.2 * loss
        avg_acc = 0.8 * avg_acc + 0.2 * acc
        if i % 100 == 0:
            print(f"At iteration {i}, average accuracy is {int(avg_acc * 100)}% (loss is {avg_loss}).")
            if avg_acc > 0.99:
                break
        grad = nn.calculate_gradient_very_slowly(inputs, outputs)
        nn.learn_very_slowly(grad)

    print("\nDone training!\n")

    inputs, outputs = create_stupid_example(10)

    print(f"Example input  : {inputs.ravel()}")
    print(f"Expected output: {outputs.ravel()}\n")
    print(f"Actual output:")

    pred = [pred_y for pred_y in nn.forward_prop_seq(inputs)]

    for i in range(len(outputs)):
        print(f"  At index {i}, we expected {outputs[i][0][0]} and predicted {pred[i][0][0]}.")

    return (avg_loss, avg_acc)


if __name__ == '__main__':
    main()
