# This is an attempt to implement the "Gambler's Problem" example 4.3
# and exercise 4.9 from Sutton and Barto's Reinforcement Learning textbook.

from typing import Dict, NamedTuple, Iterator


GOAL = 100

# The state is the gambler's capital.
State = int

# The actions are stakes.
Action = int

StateValue = Dict[State, float]

Policy = Dict[State, Action]

STATE_RANGE = range(0, GOAL + 1)

NON_TERMINAL_STATE_RANGE = range(1, GOAL)

TERMINAL_STATES = {0: 0.0, GOAL: 1.0}

ZERO_STATE_VALUE = dict([(i, 0.0) for i in STATE_RANGE])

class ValueIterResult(NamedTuple):
    policy: Policy
    sv: StateValue
    delta: float
    i: int

def get_action_range(state: State) -> range:
    return range(0, min(state, GOAL - state) + 1)

def iter_once(prob_heads: float, sv: StateValue, i: int) -> ValueIterResult:
    next_sv = sv.copy()
    prob_tails = 1 - prob_heads
    policy = {}
    max_delta = float('-inf')

    for s in STATE_RANGE:
        if s in TERMINAL_STATES:
            next_sv[s] = TERMINAL_STATES[s]
        else:
            action_rewards = []
            for a in get_action_range(s):
                heads_reward = sv[s + a]
                tails_reward = sv[s - a]
                reward = prob_heads * heads_reward + prob_tails * tails_reward
                action_rewards.append((a, reward))
            best_action, best_reward = max(action_rewards, key=lambda x: x[1])
            next_sv[s] = best_reward
            policy[s] = best_action
        delta = abs(sv[s] - next_sv[s])
        if delta > max_delta:
            max_delta = delta

    return ValueIterResult(policy, next_sv, max_delta, i)

def iter_value(prob_heads: float,
               theta: float=0.001) -> Iterator[ValueIterResult]:
    sv = ZERO_STATE_VALUE
    i = 0

    while True:
        next_value = iter_once(prob_heads, sv, i)
        sv = next_value.sv
        if next_value.delta < theta:
            break
        yield next_value
        i += 1

if __name__ == '__main__':
    for v in iter_value(0.4):
        print(f'iter {v.i}, delta={v.delta}')
    for s in NON_TERMINAL_STATE_RANGE:
        print(f'V[{s}] = {v.sv[s]:3.2}; '
              f'if you have ${s}, bet ${v.policy[s]}.')
