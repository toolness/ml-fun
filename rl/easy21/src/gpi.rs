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

    // A hook that's called whenever an episode begins.
    fn on_episode_begin(&mut self) {
    }

    // A hook that's called whenever an episode transitions from one state
    // to another, as the result of an action. Can optionally return a
    // successor action to use next.
    fn on_episode_step<F>(&mut self, state: State, action: Action,
                          reward: Reward, next_state: State,
                          get_next_action: F) -> Option<Action>
        where F: Fn() -> Action {
        let _ = (state, action, reward, next_state, get_next_action);
        None
    }

    // A hook that's called whenever an episode ends. Implementations can
    // use this to e.g. update their value functions.
    fn on_episode_end(&mut self) {
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

    fn on_episode_begin(&mut self);

    fn on_episode_step(&mut self, state: State, action: Action,
                       reward: Reward, next_state: State) -> Option<Action>;

    fn on_episode_end(&mut self);
}

#[derive(Debug, PartialEq)]
enum EpsilonType {
    Varying,
    Constant(f32),
}

use std::cell::{RefCell};

pub struct EpsilonGreedyPolicy<T: Rng, U: Alg> {
    times_visited: HashMap<State, f32>,
    rng: RefCell<T>,
    pub alg: RefCell<U>,
    epsilon: EpsilonType,
}

impl<T: Rng, U: Alg> EpsilonGreedyPolicy<T, U> {
    pub fn new(rng: T, alg: U) -> Self {
        EpsilonGreedyPolicy {
            times_visited: HashMap::new(),
            rng: RefCell::new(rng),
            alg: RefCell::new(alg),
            epsilon: EpsilonType::Varying,
        }
    }

    pub fn with_constant_epsilon(mut self, value: f32) -> Self {
        self.epsilon = EpsilonType::Constant(value);
        self
    }

    fn exploratory_action(&self) -> Action {
        if self.rng.borrow_mut().gen_weighted_bool(2) { Hit } else { Stick }
    }

    fn should_explore(&self, state: State) -> bool {
        let epsilon = match self.epsilon {
            EpsilonType::Varying => {
                let n_0 = 100.0;
                let visited = *self.times_visited.get(&state).unwrap_or(&0.0);
                n_0 / (n_0 + visited)
            },
            EpsilonType::Constant(value) => {
                value
            }
        };
        self.rng.borrow_mut().next_f32() < epsilon
    }

    fn interior_choose_action(&self, state: State) -> Action {
        if self.should_explore(state) {
            self.exploratory_action()
        } else {
            self.alg.borrow().choose_best_action(state)
        }
    }
}

impl<T: Rng, U: Alg> Policy for EpsilonGreedyPolicy<T, U> {
    fn choose_action(&mut self, state: State) -> Action {
        self.interior_choose_action(state)
    }

    fn on_episode_begin(&mut self) {
        self.alg.get_mut().on_episode_begin();
    }

    fn on_episode_step(&mut self, state: State, action: Action,
                       reward: Reward, next_state: State) -> Option<Action> {
        if self.epsilon == EpsilonType::Varying {
            increment(&mut self.times_visited, state, 1.0);
        }

        // Argh, I wanted to just pass the policy in as the last argument, so
        // that the algorithm (e.g. Sarsa) could calculate the next action
        // only if it needed to, but that raised an error complaining that
        // `EpsilonGreedyPolicy` didn't implement `Policy`.
        //
        // Then I tried changing the last parameter to just being
        // `FnMut(State) -> Action`, but that raised errors with the
        // borrow checker when I tried passing `self.choose_action`.
        //
        // The only remaining option is to *always* calculate the next action,
        // and pass it to the algorithm, which can use it if needed.
        //
        // It should also be noted that tinkering with the calling
        // convention of `Alg.on_episode_step()` is difficult once we
        // have multiple trait implementations in place, since we have to
        // change the trait definition *and* every implementation site
        // just to see what the borrow checker thinks.
        let get_next_action = || -> Action {
            self.interior_choose_action(next_state)
        };
        self.alg.borrow_mut().on_episode_step(state, action, reward, next_state,
                                              get_next_action)
    }

    fn on_episode_end(&mut self) {
        self.alg.get_mut().on_episode_end();
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
        let mut state = State::new(&mut self.deck);

        self.policy.on_episode_begin();

        let mut action = self.policy.choose_action(state);

        while !state.is_terminal() {
            let (next_state, reward) = state.step(&mut self.deck, action);
            match self.policy.on_episode_step(state, action, reward,
                                              next_state) {
                None => {
                    action = self.policy.choose_action(next_state);
                },
                Some(next_action) => {
                    action = next_action;
                }
            }
            state = next_state;
        }

        self.policy.on_episode_end();
        self.episodes += 1;
    }

    pub fn play_episodes(&mut self, count: i32) {
        for _ in 0..count {
            self.play_episode();
        }
    }
}

#[cfg(test)]
pub mod tests {
    use game::{RngDeck, State, Action, Reward};
    use rand::thread_rng;

    use gpi::{Gpi, Alg, EpsilonGreedyPolicy, EpsilonType};

    pub struct DumbAlg {
        pub action: Action,
        pub reward: Reward,
    }

    impl Alg for DumbAlg {
        fn choose_best_action(&self, _: State) -> Action {
            self.action
        }

        fn get_expected_reward(&self, _: State, _: Action) -> Reward {
            self.reward
        }
    }

    #[test]
    fn test_play_episodes_works() {
        let deck = RngDeck::new(thread_rng());
        let policy = EpsilonGreedyPolicy::new(thread_rng(), DumbAlg {
            action: Action::Hit,
            reward: 0.0,
        });
        assert_eq!(policy.epsilon, EpsilonType::Varying);
        let mut gpi = Gpi::new(deck, policy);

        gpi.play_episodes(3);

        assert_eq!(gpi.episodes, 3);
    }

    #[test]
    fn test_constant_epsilon_works() {
        let policy = EpsilonGreedyPolicy::new(thread_rng(), DumbAlg {
            action: Action::Hit,
            reward: 0.0,
        }).with_constant_epsilon(0.5);
        assert_eq!(policy.epsilon, EpsilonType::Constant(0.5));
    }
}
