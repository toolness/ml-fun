use std::collections::HashMap;

use game::{State, Action, Reward};
use game::Action::*;
use gpi::Alg;
use util::{increment, VaryingStepSizer};


type EligibilityHash = HashMap<(State, Action), f32>;
type ValueFn = HashMap<(State, Action), Reward>;

pub struct SarsaLambda {
    value_fn: ValueFn,
    step_sizer: VaryingStepSizer,
    traces: EligibilityHash,
    lambda: f32,
}

impl SarsaLambda {
    pub fn new(lambda: f32) -> Self {
        SarsaLambda {
            value_fn: HashMap::new(),
            step_sizer: VaryingStepSizer::new(),
            traces: HashMap::new(),
            lambda,
        }
    }
}

impl Alg for SarsaLambda {
    fn choose_best_action(&self, state: State) -> Action {
        let hit = *self.value_fn.get(&(state, Hit)).unwrap_or(&0.0);
        let stick = *self.value_fn.get(&(state, Stick)).unwrap_or(&0.0);
        if hit > stick { Hit } else { Stick }
    }

    fn get_expected_reward(&self, state: State, action: Action) -> Reward {
        *self.value_fn.get(&(state, action)).unwrap_or(&0.0)
    }

    fn on_episode_begin(&mut self) {
        self.traces.drain();
    }

    fn needs_next_action(&self) -> bool {
        true
    }

    fn on_episode_step(&mut self, state: State, action: Action,
                       reward: Reward, next_state: State,
                       next_action: Option<Action>) -> Option<Action> {
        let step_size = self.step_sizer.update(state, action);
        let td_error = reward +
                       self.get_expected_reward(next_state,
                                                next_action.unwrap()) -
                       self.get_expected_reward(state, action);
        increment(&mut self.traces, (state, action), 1.0);
        for (&(state, action), trace) in self.traces.iter_mut() {
            let eligibility_trace = *trace;
            increment(&mut self.value_fn, (state, action),
                      step_size * td_error * eligibility_trace);
            *trace = self.lambda * eligibility_trace;
        }
        next_action
    }
}
