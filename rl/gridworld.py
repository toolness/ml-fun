# This is an attempt to do policy evaluation on the 2D grid world
# example from Sutton and Barto's Reinforcement Learning textbook.

from enum import IntEnum
from typing import Tuple, List, Callable, Dict, Any, NamedTuple, Iterator


probability = float

DISCOUNT = 0.99999999

WIDTH = 4

HEIGHT = 4

class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class PotentialState(NamedTuple):
    state: 'State'
    probability: float


class State(NamedTuple):
    x: int
    y: int

    def perform(self, action: Action) -> List[PotentialState]:
        if action == Action.NORTH:
            return [PotentialState(State(x=self.x, y=max(self.y - 1, 1)),
                                   1.0)]
        if action == Action.SOUTH:
            return [PotentialState(State(x=self.x,
                                         y=min(self.y + 1, HEIGHT)),
                                   1.0)]
        if action == Action.EAST:
            return [PotentialState(State(x=min(self.x + 1, WIDTH),
                                         y=self.y),
                                   1.0)]
        if action == Action.WEST:
            return [PotentialState(State(x=max(self.x - 1, 1), y=self.y),
                                   1.0)]
        raise AssertionError('invalid action')

    @property
    def is_terminal(self) -> bool:
        return self in [(1, 1), (WIDTH, HEIGHT)]

    @property
    def reward(self) -> float:
        return 0 if self.is_terminal else -1.0

    @classmethod
    def xrange(cls) -> range:
        return range(1, WIDTH + 1)

    @classmethod
    def yrange(cls) -> range:
        return range(1, HEIGHT + 1)

    @classmethod
    def all(cls) -> Iterator['State']:
        for x in cls.xrange():
            for y in cls.yrange():
                yield cls(x=x, y=y)


Policy = Callable[[State, Action], probability]


BestStateActions = Dict[State, List[Action]]


def random_policy(state: State, action: Action) -> probability:
    return 1.0 / len(Action)


class BestStateActionsPolicy:
    def __init__(self, best: BestStateActions) -> None:
        self.best = best

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BestStateActionsPolicy):
            return self.best == other.best
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __call__(self, state: State, action: Action) -> probability:
        if action in self.best[state]:
            return 1.0 / len(self.best[state])
        return 0


class StateValue:
    def __init__(self, policy: Policy,
                 previous: 'StateValue' = None) -> None:
        self.policy = policy
        self.previous = previous
        self.map = {}  # type: Dict[State, float]
        if self.previous is None:
            for state in State.all():
                self.map[state] = 0
            self.k = 0
        else:
            self.k = self.previous.k + 1

    def _evaluate(self, state: State) -> float:
        reward = state.reward

        if not state.is_terminal:
            for action in Action:
                action_reward = 0.0
                action_probability = self.policy(state, action)
                for next_state, probability in state.perform(action):
                    action_reward += (probability * DISCOUNT *
                                      self.previous(next_state))
                reward += action_probability * action_reward

        return reward

    def next(self) -> 'StateValue':
        return self.__class__(self.policy, self)

    def create_improved_policy(self) -> Policy:
        best_state_actions = {}  # type: BestStateActions
        for state in State.all():
            max_reward = float('-inf')
            action_rewards = {}  # type: Dict[Action, float]
            for action in Action:
                action_reward = 0.0
                for next_state, prob in state.perform(action):
                    action_reward += prob * self(next_state)
                if action_reward > max_reward:
                    max_reward = action_reward
                action_rewards[action] = action_reward
            best_state_actions[state] = [
                action for action in Action
                if action_rewards[action] == max_reward
            ]

        return BestStateActionsPolicy(best_state_actions)

    def iter_until_convergence(self, theta=0.01):
        sv = self

        while True:
            yield sv
            delta = 0
            old_sv = sv
            sv = sv.next()
            for state in State.all():
                delta = max(delta, abs(old_sv(state) - sv(state)))
            if delta < theta:
                break

        yield sv

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
    policy = random_policy
    i = 1
    while True:
        for sv in StateValue(policy).iter_until_convergence():
            print(f'State-value matrix on iteration {sv.k}:\n{sv}\n')
        improved_policy = sv.create_improved_policy()
        if improved_policy == policy:
            break
        print(f"Created improved policy #{i}.\n")
        policy = improved_policy
        i += 1
    print("Optimal policy reached.")
