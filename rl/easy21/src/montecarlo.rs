use std::collections::HashMap;
use std::hash::Hash;

use rand::Rng;

use game::{State, Action, Deck, Reward, MIN_SUM, MAX_SUM, MIN_CARD, MAX_CARD};
use game::Action::*;


type ValueFn = HashMap<(State, Action), Reward>;

fn increment<T: Eq + Hash + Copy>(map: &mut HashMap<T, f32>, key: T,
                                  amount: f32) -> f32 {
    let prev_val = *map.get(&key).unwrap_or(&0.0);
    let new_val = prev_val + amount;
    map.insert(key, new_val);
    new_val
}

pub struct Control<T: Deck, U: Rng> {
    value_fn: ValueFn,
    times_visited: HashMap<State, f32>,
    total_visits: HashMap<(State, Action), Reward>,
    episodes: i32,
    deck: T,
    rng: U,
}

impl<T: Deck, U: Rng> Control<T, U> {
    pub fn new(deck: T, rng: U) -> Self {
        Control {
            value_fn: HashMap::new(),
            times_visited: HashMap::new(),
            total_visits: HashMap::new(),
            episodes: 0,
            deck,
            rng,
        }
    }

    fn choose_best_action(&self, state: State) -> Action {
        let hit = *self.value_fn.get(&(state, Hit)).unwrap_or(&0.0);
        let stick = *self.value_fn.get(&(state, Stick)).unwrap_or(&0.0);
        if hit > stick { Hit } else { Stick }
    }

    fn update_value_fn(&mut self, visited: &HashMap<(State, Action), f32>,
                       reward: Reward) {
        for &(state, action) in visited.keys() {
            // We only care about the *first* time a state/action pair
            // was visited in an episode.
            let visits = increment(&mut self.total_visits, (state, action),
                                   1.0);
            let old_value = *self.value_fn.get(&(state, action))
              .unwrap_or(&0.0);
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
        let visited = *self.times_visited.get(&state).unwrap_or(&0.0);
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

    pub fn play_episodes(&mut self, count: i32) {
        for _ in 0..count {
            self.play_episode();
        }
    }

    pub fn print_optimal_value_fn(&self) {
        let dealer_rng = MIN_CARD..MAX_CARD + 1;
        for player in (MIN_SUM..MAX_SUM + 1).rev() {
            for dealer in dealer_rng.clone() {
                let state = State { dealer, player };
                let action = self.choose_best_action(state);
                let value = *self.value_fn.get(&(state, action))
                  .unwrap_or(&0.0);
                let ivalue = (value * 100.0) as i32;
                print!("{:3} ", ivalue);
            }
            println!("  <- player sum = {}", player);
        }
        for _ in dealer_rng.clone() {
            print!("----");
        }
        println!();
        for dealer in dealer_rng {
            if dealer == 1 {
                print!("  A ");
            } else {
                print!("{:3} ", dealer);
            }
        }
        println!("  <- dealer showing");
    }
}

#[cfg(test)]
mod tests {
    use game::RngDeck;
    use rand::thread_rng;

    use montecarlo::Control;

    #[test]
    fn test_play_episodes_works() {
        let deck = RngDeck::new(thread_rng());
        let mut control = Control::new(deck, thread_rng());

        control.play_episodes(3);

        assert_eq!(control.episodes, 3);
    }
}
