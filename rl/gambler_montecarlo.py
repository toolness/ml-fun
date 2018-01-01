import random
from enum import IntEnum
from typing import Counter, Dict, NamedTuple, Iterator, Tuple, Callable

from gambler import (
    GOAL,
    TERMINAL_STATES,
    NON_TERMINAL_STATE_RANGE,
    State,
    StateValue,
    Action,
)


FuncPolicy = Callable[[State], Action]


def super_conservative_policy(money: State) -> Action:
    return 1


class CoinFlip(IntEnum):
    HEADS = 0
    TAILS = 1


class CoinFlipper:
    def __init__(self, prob_heads: float) -> None:
        heads_in_100 = int(prob_heads * 100)
        tails_in_100 = 100 - heads_in_100
        self.samples = ([CoinFlip.HEADS] * heads_in_100 +
                        [CoinFlip.TAILS] * tails_in_100)

    def flip(self) -> CoinFlip:
        return random.choice(self.samples)


class StateStats(NamedTuple):
    visits: Counter[State]
    rewards: Dict[State, float]

    def build_state_value(self) -> StateValue:
        sv = {}
        for state in NON_TERMINAL_STATE_RANGE:
            if self.visits[state]:
                sv[state] = self.rewards[state] / self.visits[state]
            else:
                sv[state] = 0.0
        return sv

    @classmethod
    def blank(cls):
        return cls(Counter[State](), {})


def play_episode(flipper: CoinFlipper,
                 policy: FuncPolicy,
                 stats: StateStats) -> None:
    money: State = random.randint(1, GOAL - 1)
    reward = None
    visits_this_episode: Dict[State, bool] = {}
    while reward is None:
        if money not in visits_this_episode:
            visits_this_episode[money] = True
        stake: Action = policy(money)
        assert stake > 0 and stake <= money
        if flipper.flip() == CoinFlip.HEADS:
            money += stake
        else:
            money -= stake
        if money in TERMINAL_STATES:
            reward = TERMINAL_STATES[money]
    for state in visits_this_episode:
        stats.visits[state] += 1
        stats.rewards[state] = stats.rewards.get(state, 0.0) + reward


def play_episodes(flipper: CoinFlipper, policy: FuncPolicy,
                  count: int) -> Iterator[Tuple[int, StateStats]]:
    stats = StateStats.blank()

    for i in range(count):
        play_episode(flipper, policy, stats)
        yield i, stats


def main() -> None:
    policy = super_conservative_policy
    count = 1000
    for i, stats in play_episodes(CoinFlipper(0.4), policy, count):
        if i % 100 == 0:
            print(f"played episode {i} of {count}")
    sv = stats.build_state_value()
    print(f"state-value for {policy.__name__}:")
    for state in sv:
        print(f"V[{state}] = {sv[state]:0.3}")


if __name__ == '__main__':
    main()
