# This is an attempt to implement the "Jack's Car Rental" example 4.2
# from Sutton and Barto's Reinforcement Learning textbook.

import math
import random
from functools import lru_cache
from typing import NamedTuple, Iterator, Tuple, Dict, List


DISCOUNT_RATE = 0.9

MAX_AT_LOC = 10

MAX_CAN_MOVE = 3

MOVE_COST = 2.0

RENTAL_COST = 10.0

# Positive ints indicate # of cars to move from loc_one to loc_two,
# negative ints indicate moves in the opposite direction.
Action = int


class StateTransition(NamedTuple):
    next_state: 'State'
    reward: float
    probability: float


class IllegalActionError(Exception):
    pass


class State(NamedTuple):
    loc_one: int
    loc_two: int

    def predict_reward(self, action: Action, sv: 'StateValue') -> float:
        reward = 0.0
        total_prob = 0.0
        for trans in self.perform_action(action):
            total_prob += trans.probability
            reward += trans.probability * (
                trans.reward + DISCOUNT_RATE * sv[trans.next_state]
            )
        if not math.isclose(total_prob, 1.0, rel_tol=0.01):
            raise Exception(f'probabilities total to {total_prob}, not 1.0')
        return reward

    @lru_cache(maxsize=2048)
    def perform_action(self, action: Action) -> List[StateTransition]:
        assert action >= -MAX_CAN_MOVE and action <= MAX_CAN_MOVE
        loc_one = self.loc_one - action
        loc_two = self.loc_two + action
        if (loc_one < 0 or loc_two < 0 or
                loc_one > MAX_AT_LOC or loc_two > MAX_AT_LOC):
            raise IllegalActionError()
        move_cost = abs(action) * MOVE_COST
        result = []
        for return_to_one in range(0, MAX_AT_LOC + 1):
            return_to_one_prob = poisson_prob(return_to_one, 3)
            for return_to_two in range(0, MAX_AT_LOC + 1):
                return_to_two_prob = poisson_prob(return_to_two, 2)
                for rent_from_one in range(0, MAX_AT_LOC + 1):
                    rent_from_one_prob = poisson_prob(rent_from_one, 3)
                    for rent_from_two in range(0, MAX_AT_LOC + 1):
                        rent_from_two_prob = poisson_prob(rent_from_two, 4)

                        if return_to_one + loc_one > MAX_AT_LOC:
                            return_to_one = MAX_AT_LOC - loc_one
                        if return_to_two + loc_two > MAX_AT_LOC:
                            return_to_two = MAX_AT_LOC - loc_two
                        if rent_from_one > loc_one:
                            rent_from_one = loc_one
                        if rent_from_two > loc_two:
                            rent_from_two = loc_two

                        state = State(
                            loc_one + return_to_one - rent_from_one,
                            loc_two + return_to_two - rent_from_two
                        )
                        reward = (
                            (rent_from_one + rent_from_two) * RENTAL_COST -
                            move_cost
                        )
                        prob = (
                            return_to_one_prob *
                            return_to_two_prob *
                            rent_from_one_prob *
                            rent_from_two_prob
                        )
                        result.append(StateTransition(state, reward, prob))

        return result

    @classmethod
    def iter_all(cls) -> Iterator['State']:
        for one in range(0, MAX_AT_LOC + 1):
            for two in range(0, MAX_AT_LOC + 1):
                yield cls(one, two)


Policy = Dict[State, Action]

StateValue = Dict[State, float]

zero_policy: Policy = dict([(s, 0) for s in State.iter_all()])

zero_state_value: StateValue = dict([(s, 0.0) for s in State.iter_all()])


def print_state_value(sv: StateValue) -> None:
    lines = []
    for one in range(MAX_AT_LOC, -1, -1):
        line = []
        for two in range(0, MAX_AT_LOC + 1):
            line.append('{:3.0f}'.format(sv[State(one, two)]))
        lines.append(' '.join(line))
    print('\n'.join(lines))


def print_policy(policy: Policy) -> None:
    lines = []
    for one in range(MAX_AT_LOC, -1, -1):
        line = []
        for two in range(0, MAX_AT_LOC + 1):
            line.append('{:3.0f}'.format(policy[State(one, two)]))
        lines.append(' '.join(line))
    print('\n'.join(lines))


def eval_policy(policy: Policy, theta: float=1.0) -> StateValue:
    sv = zero_state_value

    while True:
        next_sv = zero_state_value.copy()
        max_delta = float('-inf')
        for state in State.iter_all():
            action = policy[state]
            next_sv[state] = state.predict_reward(action, sv)
            delta = abs(sv[state] - next_sv[state])
            if delta > max_delta:
                max_delta = delta
        sv = next_sv
        print(f'finished policy eval iteration, max_delta={max_delta}')
        print_state_value(sv)
        if max_delta < theta:
            break
    return sv


def create_improved_policy(sv: StateValue) -> Policy:
    policy = zero_policy.copy()
    for state in State.iter_all():
        action_rewards = []
        for action in range(-MAX_CAN_MOVE, MAX_CAN_MOVE + 1):
            try:
                reward = state.predict_reward(action, sv)
                action_rewards.append((action, reward))
            except IllegalActionError:
                pass
        policy[state] = max(action_rewards, key=lambda x: x[1])[0]

    return policy


@lru_cache(maxsize=2048)
def poisson_prob(n: int, avg_per_interval: int) -> float:
    return ((math.pow(avg_per_interval, n) * math.exp(-avg_per_interval)) /
            math.factorial(n))


if __name__ == '__main__':
    policy = zero_policy
    i = 0
    while True:
        sv = eval_policy(policy)
        new_policy = create_improved_policy(sv)
        print(f'created improved policy, i={i}')
        print_policy(new_policy)
        if new_policy == policy:
            break
        i += 1
        policy = new_policy
    print(f'optimal policy reached after {i} iterations.')
