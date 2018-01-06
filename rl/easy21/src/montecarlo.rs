use std::collections::HashMap;
use std::hash::Hash;

use rand::{thread_rng, ThreadRng, Rng};

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

struct Control<T: Deck> {
    value_fn: ValueFn,
    times_visited: HashMap<State, f32>,
    total_visits: HashMap<(State, Action), Reward>,
    episodes: i32,
    deck: T,
    rng: ThreadRng,
}

impl<T: Deck> Control<T> {
    pub fn new(deck: T) -> Self {
        Control {
            value_fn: HashMap::new(),
            times_visited: HashMap::new(),
            total_visits: HashMap::new(),
            episodes: 0,
            deck,
            rng: thread_rng(),
        }
    }

    fn choose_best_action(&mut self, state: State) -> Action {
        let hit = *self.value_fn.entry((state, Hit)).or_insert(0.0);
        let stick = *self.value_fn.entry((state, Stick)).or_insert(0.0);
        if hit > stick { Hit } else { Stick }
    }

    fn update_value_fn(&mut self, visited: &HashMap<(State, Action), f32>,
                       reward: Reward) {
        for &(state, action) in visited.keys() {
            // We only care about the *first* time a state/action pair
            // was visited in an episode.
            let visits = increment(&mut self.total_visits, (state, action),
                                   1.0);
            let old_value = *self.value_fn.entry((state, action))
              .or_insert(0.0);
            let step_size = 1.0 / visits;
            let new_value = old_value + step_size * (reward - old_value);
            self.value_fn.insert((state, action), new_value);
        }
    }

    fn exploratory_action(&mut self) -> Action {
        if self.rng.gen_weighted_bool(2) { Hit } else { Stick }
    }

    fn should_explore(&mut self, state: State) -> bool {
        let n_0 = 100.0;
        let visited = *self.times_visited.entry(state).or_insert(0.0);
        let epsilon = n_0 / (n_0 + visited);
        self.rng.next_f32() < epsilon
    }

    pub fn play_episode(&mut self) {
        let mut total_reward = 0.0;
        let mut state_actions_visited = HashMap::new();
        let mut state = State::new(&mut self.deck);

        while !state.is_terminal() {
            let action = if self.should_explore(state) {
                self.exploratory_action()
            } else {
                self.choose_best_action(state)
            };
            let (next_state, reward) = state.step(&mut self.deck, action);
            increment(&mut self.times_visited, state, 1.0);
            increment(&mut state_actions_visited, (state, action), 1.0);
            total_reward += reward;
            state = next_state;
        }
        self.update_value_fn(&state_actions_visited, total_reward);
        self.episodes += 1;
    }
}

#[cfg(test)]
mod tests {
    use game::RngDeck;
    use rand::thread_rng;

    use montecarlo::Control;

    #[test]
    fn test_play_episode_works() {
        let deck = RngDeck::new(thread_rng());
        let mut control = Control::new(deck);

        control.play_episode();

        assert_eq!(control.episodes, 1);
    }
}
