# This is an attempt to do policy evaluation on the 2D grid world
# example from Sutton and Barto's Reinforcement Learning textbook.

from collections import namedtuple
from enum import IntEnum
from typing import Tuple, List, Callable


probability = float

DISCOUNT = 0.99999999

class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


PotentialState = namedtuple('PotentialState', ['state', 'probability'])


class State(namedtuple('State', ['x', 'y'])):
    __slots__ = ()

    _fields = ('x', 'y')

    WIDTH = 4

    HEIGHT = 4

    def perform(self, action: Action) -> List[PotentialState]:
        if action == Action.NORTH:
            return [PotentialState(State(x=self.x, y=max(self.y - 1, 1)),
                                   1.0)]
        if action == Action.SOUTH:
            return [PotentialState(State(x=self.x,
                                         y=min(self.y + 1, self.HEIGHT)),
                                   1.0)]
        if action == Action.EAST:
            return [PotentialState(State(x=min(self.x + 1, self.WIDTH),
                                         y=self.y),
                                   1.0)]
        if action == Action.WEST:
            return [PotentialState(State(x=max(self.x - 1, 1), y=self.y),
                                   1.0)]
        raise AssertionError('invalid action')

    @property
    def is_terminal(self) -> bool:
        return self in [(1, 1), (self.WIDTH, self.HEIGHT)]

    @property
    def reward(self) -> float:
        return 0 if self.is_terminal else -1.0

    @classmethod
    def xrange(cls):
        return range(1, cls.WIDTH + 1)

    @classmethod
    def yrange(cls):
        return range(1, cls.HEIGHT + 1)

    @classmethod
    def all(cls):
        for x in cls.xrange():
            for y in cls.yrange():
                yield cls(x=x, y=y)


def random_policy(state: State, action: Action) -> probability:
    return 1.0 / len(Action)


class StateValue:
    def __init__(self, policy: Callable[[State, Action], probability],
                 previous: 'StateValue' = None):
        self.policy = policy
        self.previous = previous
        self.map = {}
        if self.previous is None:
            for state in State.all():
                self.map[state] = 0

    def _evaluate(self, state: State) -> float:
        reward = state.reward

        if not state.is_terminal:
            for action in Action:
                action_reward = 0
                action_probability = self.policy(state, action)
                for next_state, probability in state.perform(action):
                    action_reward += (probability * DISCOUNT *
                                      self.previous(next_state))
                reward += action_probability * action_reward

        return reward

    def next(self) -> 'StateValue':
        return self.__class__(self.policy, self)

    def __call__(self, state: State) -> float:
        if state not in self.map:
            self.map[state] = self._evaluate(state)
        return self.map[state]

    def __str__(self) -> str:
        lines = []
        for y in State.yrange():
            line = []
            for x in State.xrange():
                line.append('{:6.2f}'.format(self(State(x, y))))
            lines.append(' '.join(line))
        return '\n'.join(lines)


if __name__ == '__main__':
    sv = StateValue(random_policy)

    for i in range(100):
        print(f'State-value matrix on iteration {i}:\n{sv}\n')
        sv = sv.next()
