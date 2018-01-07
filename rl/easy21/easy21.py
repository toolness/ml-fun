import ctypes as ct
from pathlib import Path

import numpy as np


MY_DIR = Path(__file__).parent.resolve()
CDLL_DIR = MY_DIR / 'target' / 'release'
CDLL_FILE = CDLL_DIR / 'easy21'

NUM_ACTIONS = 2
MIN_CARD = 1
MAX_CARD = 10
MIN_SUM = 1
MAX_SUM = 21
DEALER_RANGE = range(MIN_CARD, MAX_CARD + 1)
PLAYER_RANGE = range(MIN_SUM, MAX_SUM + 1)
OUTPUT_SIZE = len(DEALER_RANGE) * len(PLAYER_RANGE) * NUM_ACTIONS

e21 = ct.CDLL(str(CDLL_FILE))

# If these mismatch for some reason, our calls to the dynamic library
# will segfault, so might as well crash with an assertion failure instead.
assert e21.get_output_size() == OUTPUT_SIZE

OUTPUT_ARRAY = ct.c_float * OUTPUT_SIZE

e21.run_monte_carlo.argtypes = [ct.c_int, ct.POINTER(OUTPUT_ARRAY)]
e21.run_monte_carlo.restype = ct.c_int

e21.run_sarsa.argtypes = [ct.c_int, ct.c_float, ct.POINTER(OUTPUT_ARRAY)]
e21.run_sarsa.restype = ct.c_int


def ctypes_array_to_numpy(ct_arr):
    # I was hoping there would be a way to do this that didn't involve
    # iterating in python-land, but whatever, it's a tiny array. More
    # details here: https://stackoverflow.com/a/4355701/2422398

    size = len(ct_arr)
    np_arr = np.zeros(size)

    for val, i in zip(ct_arr, range(size)):
        np_arr[i] = val

    return np_arr


def run_monte_carlo(episodes):
    output = OUTPUT_ARRAY()
    result = e21.run_monte_carlo(episodes, ct.byref(output))

    if result != 0:
        raise ValueError(f"run_monte_carlo failed with result {result}")

    return ctypes_array_to_numpy(output)


def run_sarsa(episodes, lambda_val):
    output = OUTPUT_ARRAY()
    result = e21.run_sarsa(episodes, lambda_val, ct.byref(output))

    if result != 0:
        raise ValueError(f"run_sarsa failed with result {result}")

    return ctypes_array_to_numpy(output)


if __name__ == '__main__':
    run_monte_carlo(10)
    run_sarsa(5, 0.5)
    print("Smoke tests passed!")
