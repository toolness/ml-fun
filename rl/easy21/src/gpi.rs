// This module attempts to implement the Generalized Policy Iteration
// (GPI) algorithm as an abstract concept, allowing clients to
// "plug in" different policy evaluation/improvement algorithms.

use std::collections::HashMap;

use rand::Rng;

use game::{State, Action, Deck, Reward, MIN_SUM, MAX_SUM, MIN_CARD, MAX_CARD};
use game::Action::*;
use util::increment;


// This trait encapsulates a specific algorithm to use for GPI.
pub trait Alg {
    // Given the current state, return the action that maximizes reward
    // in the long-term.
    fn choose_best_action(&self, state: State) -> Action;

    // Return the expected long-term reward if we take the given action
    // at the given state.
    fn get_expected_reward(&self, state: State, action: Action) -> Reward;

    // A hook that's called whenever an episode ends. Implementations can
    // use this to e.g. update their value functions.
    //
    // `visited` is a mapping that indicates how many times a given
    // state/action pair was visited during the episode.
    //
    // `reward` is the total reward accrued during the episode.
    fn on_episode_end(&mut self, visited: &HashMap<(State, Action), f32>,
                      reward: Reward) {
        let _ = (visited, reward);
    }

    // Print the expected reward for every state given that we
    // take the optimal action at each state.
    fn print_optimal_values(&self) {
        let dealer_rng = MIN_CARD..MAX_CARD + 1;
        for player in (MIN_SUM..MAX_SUM + 1).rev() {
            for dealer in dealer_rng.clone() {
                let state = State { dealer, player };
                let action = self.choose_best_action(state);
                let value = self.get_expected_reward(state, action);
                let ivalue = (value * 100.0) as i32;
                print!("{:4} ", ivalue);
            }
            println!("  <- player sum = {}", player);
        }
        for _ in dealer_rng.clone() {
            print!("----");
        }
        println!();
        for dealer in dealer_rng {
            if dealer == 1 {
                print!("   A ");
            } else {
                print!("{:4} ", dealer);
            }
        }
        println!("  <- dealer showing");
    }
}

pub trait Policy {
    fn choose_action(&mut self, state: State) -> Action;

    fn on_state_visited(&mut self, state: State);

    fn on_episode_end(&mut self, visited: &HashMap<(State, Action), f32>,
                      reward: Reward);
}

pub struct EpsilonGreedyPolicy<T: Rng, U: Alg> {
    times_visited: HashMap<State, f32>,
    rng: T,
    pub alg: U,
}

impl<T: Rng, U: Alg> EpsilonGreedyPolicy<T, U> {
    pub fn new(rng: T, alg: U) -> Self {
        EpsilonGreedyPolicy {
            times_visited: HashMap::new(),
            rng,
            alg,
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
}

impl<T: Rng, U: Alg> Policy for EpsilonGreedyPolicy<T, U> {
    fn choose_action(&mut self, state: State) -> Action {
        if self.should_explore(state) {
            self.exploratory_action()
        } else {
            self.alg.choose_best_action(state)
        }
    }

    fn on_state_visited(&mut self, state: State) {
        increment(&mut self.times_visited, state, 1.0);
    }

    fn on_episode_end(&mut self, visited: &HashMap<(State, Action), f32>,
                      reward: Reward) {
        self.alg.on_episode_end(visited, reward);
    }
}

pub struct Gpi<T: Deck, U: Policy> {
    episodes: i32,
    deck: T,
    pub policy: U,
}

impl<T: Deck, U: Policy> Gpi<T, U> {
    pub fn new(deck: T, policy: U) -> Self {
        Gpi {
            episodes: 0,
            deck,
            policy,
        }
    }

    pub fn play_episode(&mut self) {
        let mut total_reward = 0.0;
        let mut state_actions_visited = HashMap::new();
        let mut state = State::new(&mut self.deck);

        while !state.is_terminal() {
            let action = self.policy.choose_action(state);
            let (next_state, reward) = state.step(&mut self.deck, action);
            self.policy.on_state_visited(state);
            increment(&mut state_actions_visited, (state, action), 1.0);
            total_reward += reward;
            state = next_state;
        }
        self.policy.on_episode_end(&state_actions_visited, total_reward);
        self.episodes += 1;
    }

    pub fn play_episodes(&mut self, count: i32) {
        for _ in 0..count {
            self.play_episode();
        }
    }
}

#[cfg(test)]
mod tests {
    use game::{RngDeck, State, Action, Reward};
    use rand::thread_rng;

    use gpi::{Gpi, Alg, EpsilonGreedyPolicy};

    struct DumbAlg {}

    impl Alg for DumbAlg {
        fn choose_best_action(&self, _: State) -> Action {
            Action::Hit
        }

        fn get_expected_reward(&self, _: State, _: Action) -> Reward {
            0.0
        }
    }

    #[test]
    fn test_play_episodes_works() {
        let deck = RngDeck::new(thread_rng());
        let policy = EpsilonGreedyPolicy::new(thread_rng(), DumbAlg {});
        let mut gpi = Gpi::new(deck, policy);

        gpi.play_episodes(3);

        assert_eq!(gpi.episodes, 3);
    }
}
