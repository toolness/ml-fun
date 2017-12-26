import numpy as np


def invert_categorical(label):
    '''
    Given a categorical or "one hot" encoding of a label,
    invert it so that instead of having a 100% probability
    of being exactly one choice, it has 0% probability of
    being that choice, with the rest of the probability
    distribution divided unformly over the other choices.
    '''

    prob = 1 / (label.shape[0] - 1)
    ones = label == 1
    result = np.ones(label.shape[0]) * prob
    result[ones] = 0
    return np.array(result)
