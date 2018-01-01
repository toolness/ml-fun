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


class State(NamedTuple):
    loc_one: int
    loc_two: int

    def perform_action(self, action: Action) -> List[StateTransition]:
        assert action >= 0 and action <= MAX_CAN_MOVE
        loc_one = self.loc_one - action
        loc_two = self.loc_two + action
        if (loc_one < 0 or loc_two < 0 or
                loc_one > MAX_AT_LOC or loc_two > MAX_AT_LOC):
            loc_one = self.loc_one
            loc_two = self.loc_two
            action = 0
        move_cost = abs(action) * MOVE_COST
        result = []
        for return_to_one in range(0, MAX_AT_LOC - loc_one + 1):
            return_to_one_prob = poisson_prob(return_to_one, 3)
            for return_to_two in range(0, MAX_AT_LOC - loc_two + 1):
                return_to_two_prob = poisson_prob(return_to_two, 2)
                for rent_from_one in range(0, loc_one + 1):
                    rent_from_one_prob = poisson_prob(rent_from_one, 3)
                    for rent_from_two in range(0, loc_two + 1):
                        rent_from_two_prob = poisson_prob(rent_from_two, 4)
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
    for one in range(0, MAX_AT_LOC + 1):
        line = []
        for two in range(0, MAX_AT_LOC + 1):
            line.append('{:3.0f}'.format(sv[State(one, two)]))
        lines.append(' '.join(line))
    print('\n'.join(lines))

def eval_policy(policy: Policy, theta: float=1.0) -> StateValue:
    sv = zero_state_value

    while True:
        next_sv = zero_state_value.copy()
        max_delta = float('-inf')
        for state in State.iter_all():
            action = policy[state]
            for trans in state.perform_action(action):
                next_sv[state] += trans.probability * (
                    trans.reward + DISCOUNT_RATE * sv[trans.next_state]
                )
            delta = abs(sv[state] - next_sv[state])
            if delta > max_delta:
                max_delta = delta
        sv = next_sv
        print(f'finished iteration, max_delta={max_delta}')
        print_state_value(sv)
        if max_delta < theta:
            break
    return sv


@lru_cache(maxsize=2048)
def poisson_prob(n: int, avg_per_interval: int) -> float:
    return ((math.pow(avg_per_interval, n) * math.exp(-avg_per_interval)) /
            math.factorial(n))

if __name__ == '__main__':
    eval_policy(zero_policy)
