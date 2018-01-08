import ctypes as ct
from pathlib import Path
from enum import IntEnum

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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


class Action(IntEnum):
    Hit = 0
    Stick = 1


class ExpectedRewardMatrix:
    def __init__(self, raw_output: OUTPUT_ARRAY):
        self.array = output_array_to_numpy(raw_output)\
          .reshape((len(DEALER_RANGE), len(PLAYER_RANGE), NUM_ACTIONS))
        self.optimal_array = np.max(self.array, axis=2)

    def get_optimal_reward(self, dealer: int, player: int) -> float:
        return self.optimal_array[dealer - 1][player - 1]

    def get_max_diff(self, other: 'ExpectedRewardMatrix') -> float:
        diff = np.abs(self.array.flatten() - other.array.flatten())
        return np.max(diff)

    def get_mean_squared_err(self, other: 'ExpectedRewardMatrix') -> float:
        sq_err = np.square(self.array.flatten() - other.array.flatten())
        return np.average(sq_err)

    def plot_optimal_reward(self):
        x = np.array(DEALER_RANGE)
        y = np.array(PLAYER_RANGE)

        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.get_optimal_reward)(X, Y)

        ax = plt.axes(projection='3d')
        ax.set_xlabel('Dealer showing')
        ax.set_ylabel('Player sum')
        ax.set_zlabel('Expected reward')

        return ax.plot_wireframe(X, Y, Z, color='black')

    def __str__(self):
        lines = []
        for player in reversed(range(len(PLAYER_RANGE))):
            line = []
            for dealer in range(len(DEALER_RANGE)):
                expectation = self.optimal_array[dealer][player]
                expectation = int(expectation * 100)
                line.append(f'{expectation:-4}')
            lines.append(' '.join(line))
        return '\n'.join(lines)


def output_array_to_numpy(ct_arr: OUTPUT_ARRAY):
    # I was hoping there would be a way to do this that didn't involve
    # iterating in python-land, but whatever, it's a tiny array. More
    # details here: https://stackoverflow.com/a/4355701/2422398

    size = len(ct_arr)
    np_arr = np.zeros(size)

    for val, i in zip(ct_arr, range(size)):
        np_arr[i] = val

    return np_arr


def run_monte_carlo(episodes: int) -> ExpectedRewardMatrix:
    output = OUTPUT_ARRAY()
    result = e21.run_monte_carlo(episodes, ct.byref(output))

    if result != 0:
        raise ValueError(f"run_monte_carlo failed with result {result}")

    return ExpectedRewardMatrix(output)


def run_sarsa(episodes: int, lambda_val: float) -> ExpectedRewardMatrix:
    output = OUTPUT_ARRAY()
    result = e21.run_sarsa(episodes, lambda_val, ct.byref(output))

    if result != 0:
        raise ValueError(f"run_sarsa failed with result {result}")

    return ExpectedRewardMatrix(output)


def run_smoke_tests():
    run_sarsa(1000, 0.5)
    print("Output of monte carlo w/ 30,000 episodes:\n")
    big = run_monte_carlo(30_000)
    print(big)
    print("\nCompare this w/ the output of 'cargo run -- mc -e 30000'.")

    small = run_monte_carlo(30)
    assert big.get_max_diff(small) > 0
    assert big.get_mean_squared_err(small) > 0

if __name__ == '__main__':
    run_smoke_tests()
