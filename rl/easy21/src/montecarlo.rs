use std::collections::HashMap;
use std::hash::Hash;

use game::{State, Action, Deck, Reward};
use game::Action::*;


type ValueFn = HashMap<(State, Action), Reward>;

fn increment<T: Eq + Hash + Copy>(map: &mut HashMap<T, f32>, key: T,
                                  amount: f32) -> f32 {
    let prev_val = *map.entry(key).or_insert(0.0);
    let new_val = prev_val + amount;
    map.insert(key, new_val);
    new_val
}

fn choose_best_action(value_fn: &mut ValueFn, state: State) -> Action {
    let hit = *value_fn.entry((state, Hit)).or_insert(0.0);
    let stick = *value_fn.entry((state, Stick)).or_insert(0.0);
    if hit > stick { Hit } else { Stick }
}

fn control<T: Deck>(deck: &mut T, max_episodes: i32) {
    let mut value_fn: ValueFn = HashMap::new();
    let mut times_visited: HashMap<State, f32> = HashMap::new();
    let mut total_rewards: HashMap<(State, Action), Reward> = HashMap::new();
    let mut total_visits: HashMap<(State, Action), Reward> = HashMap::new();
    let mut episodes = 0;
    let mut state = State::new(deck);

    // TODO: Use a time-varying scalar step-size.
    // TODO: Use an epsilon-greedy exploration strategy.

    loop {
        let mut total_reward = 0.0;
        let mut state_actions_visited = HashMap::new();

        while !state.is_terminal() {
            increment(&mut times_visited, state, 1.0);
            let action = choose_best_action(&mut value_fn, state);
            let (next_state, reward) = state.step(deck, action);
            total_reward += reward;
            state_actions_visited.entry((state, action)).or_insert(true);
            state = next_state;
        }
        for &(state, action) in state_actions_visited.keys() {
            let visits = increment(&mut total_visits, (state, action), 1.0);
            let reward = increment(&mut total_rewards, (state, action),
                                   total_reward);
            value_fn.insert((state, action), reward / visits);
        }
        episodes += 1;
        if episodes == max_episodes {
            break;
        }
    }

    value_fn.insert((state, Action::Hit), 1.0);
}

#[cfg(test)]
mod tests {
    use game::RngDeck;
    use rand::thread_rng;

    use montecarlo::control;

    #[test]
    fn test_control_works() {
        let mut deck = RngDeck::new(thread_rng());

        control(&mut deck, 1);
    }
}
