import ctypes as ct
from pathlib import Path
from enum import IntEnum
from typing import Callable, Optional, Dict, Union

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

FANCY_PARAM_NAMES = {
    'lambda_val': 'λ',
    'epsilon': 'ε',
    'step_size': 'α',
}

e21 = ct.CDLL(str(CDLL_FILE))

# If these mismatch for some reason, our calls to the dynamic library
# will segfault, so might as well crash with an assertion failure instead.
assert e21.get_output_size() == OUTPUT_SIZE

OUTPUT_ARRAY = ct.c_float * OUTPUT_SIZE

GpiCb = Callable[['ExpectedRewardMatrix'], None]

class CGpiCb:
    CALLBACK = ct.CFUNCTYPE(None)

    @classmethod
    def from_param(cls, obj: Optional[Callable[[], None]]) \
            -> Optional[CALLBACK]:
        if obj is None:
            return None
        return cls.CALLBACK(obj)

e21.run_monte_carlo.argtypes = [ct.c_int, ct.POINTER(OUTPUT_ARRAY),
                                CGpiCb]
e21.run_monte_carlo.restype = ct.c_int

e21.run_sarsa.argtypes = [ct.c_int, ct.c_float, ct.POINTER(OUTPUT_ARRAY),
                          CGpiCb]
e21.run_sarsa.restype = ct.c_int

e21.run_q_learning.argtypes = [ct.c_int, ct.c_float, ct.POINTER(OUTPUT_ARRAY),
                               CGpiCb]
e21.run_q_learning.restype = ct.c_int

e21.run_lfa.argtypes = [ct.c_int, ct.c_float, ct.c_float, ct.c_float,
                        ct.POINTER(OUTPUT_ARRAY), CGpiCb]
e21.run_lfa.restype = ct.c_int


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

        ax.plot_wireframe(X, Y, Z, color='black')

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


class OutputReceiver:
    def __init__(self, cb: GpiCb=None):
        self.array = OUTPUT_ARRAY()
        self.array_ref = ct.byref(self.array)
        self._cb = cb
        if self._cb is None:
            self.cb = None
        self.errors = 0

    @property
    def matrix(self) -> ExpectedRewardMatrix:
        return ExpectedRewardMatrix(self.array)

    def cb(self):
        if self.errors:
            return
        try:
            self._cb(self.matrix)
        except Exception as e:
            self.errors += 1
            raise e


def describe_params(params: Dict[str, Union[int, float]]) -> str:
    parts = []
    for name, value in params.items():
        fancy_name = FANCY_PARAM_NAMES.get(name, name)
        parts.append(f'{fancy_name}={value}')
    parts.sort()
    return ', '.join(parts)


def alg_name(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        fn.alg_name = name
        return fn
    return decorator


@alg_name("Monte Carlo")
def run_monte_carlo(episodes: int, cb: GpiCb=None) -> ExpectedRewardMatrix:
    out = OutputReceiver(cb)
    result = e21.run_monte_carlo(episodes, out.array_ref, out.cb)

    if result != 0:
        raise ValueError(f"run_monte_carlo failed with result {result}")

    return out.matrix


@alg_name("Sarsa(λ)")
def run_sarsa(episodes: int, lambda_val: float,
              cb: GpiCb=None) -> ExpectedRewardMatrix:
    out = OutputReceiver(cb)
    result = e21.run_sarsa(episodes, lambda_val, out.array_ref, out.cb)

    if result != 0:
        raise ValueError(f"run_sarsa failed with result {result}")

    return out.matrix


@alg_name("Q-Learning")
def run_q_learning(episodes: int, lambda_val: float,
                   cb: GpiCb=None) -> ExpectedRewardMatrix:
    out = OutputReceiver(cb)
    result = e21.run_q_learning(episodes, lambda_val, out.array_ref, out.cb)

    if result != 0:
        raise ValueError(f"run_q_learning failed with result {result}")

    return out.matrix


@alg_name("Linear Function Approximation")
def run_lfa(episodes: int, lambda_val: float, epsilon: float,
            step_size: float, cb: GpiCb=None) -> ExpectedRewardMatrix:
    out = OutputReceiver(cb)
    result = e21.run_lfa(episodes, lambda_val, epsilon, step_size,
                         out.array_ref, out.cb)

    if result != 0:
        raise ValueError(f"run_lfa failed with result {result}")

    return out.matrix


if __name__ == '__main__':
    print("Output of monte carlo w/ 30,000 episodes:\n")
    print(run_monte_carlo(30_000))
    print("\nCompare this w/ the output of 'cargo run -- mc -e 30000'.")
    print("\nAlso run 'pytest' to run the unit tests.")
