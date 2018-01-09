use std::collections::HashMap;

use game::{State, Action, Reward};
use game::Action::*;
use gpi::Alg;
use util::{increment};


type EligibilityHash = HashMap<(State, Action), f32>;

pub struct LinearFunctionApproximator {
    traces: EligibilityHash,
    lambda: f32,
    step_size: f32,
}

impl LinearFunctionApproximator {
    pub fn new(lambda: f32, step_size: f32) -> Self {
        LinearFunctionApproximator {
            traces: HashMap::new(),
            lambda,
            step_size,
        }
    }
}

impl Alg for LinearFunctionApproximator {
    fn choose_best_action(&self, _state: State) -> Action {
        // TODO: Implement this.
        Hit
    }

    fn get_expected_reward(&self, _state: State, _action: Action) -> Reward {
        // TODO: Implement this.
        0.0
    }

    fn on_episode_begin(&mut self) {
        self.traces.drain();
    }

    fn on_episode_step(&mut self, state: State, action: Action,
                       reward: Reward, next_state: State,
                       next_action: Action) {
        let td_error = reward +
                       self.get_expected_reward(next_state, next_action) -
                       self.get_expected_reward(state, action);
        increment(&mut self.traces, (state, action), 1.0);
        for (&(state, action), trace) in self.traces.iter_mut() {
            let eligibility_trace = *trace;

            // TODO: Implement this, then remove these _'s.
            let _ = state;
            let _ = action;

            *trace = self.lambda * eligibility_trace;
        }

        // TODO: Once implemented, remove these _'s.
        let _ = td_error;
        let _ = self.step_size;
    }
}
