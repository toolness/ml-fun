use std::collections::HashMap;

use game::{State, Action, Reward};
use game::Action::*;
use gpi::Alg;
use util::VaryingStepSizer;

type ValueFn = HashMap<(State, Action), Reward>;

pub struct MonteCarlo {
    value_fn: ValueFn,
    step_sizer: VaryingStepSizer,
    reward_this_episode: Reward,
    visited_this_episode: HashMap<(State, Action), bool>,
}

impl MonteCarlo {
    pub fn new() -> Self {
        MonteCarlo {
            value_fn: HashMap::new(),
            step_sizer: VaryingStepSizer::new(),
            reward_this_episode: 0.0,
            visited_this_episode: HashMap::new(),
        }
    }
}

impl Alg for MonteCarlo {
    fn choose_best_action(&self, state: State) -> Action {
        let hit = *self.value_fn.get(&(state, Hit)).unwrap_or(&0.0);
        let stick = *self.value_fn.get(&(state, Stick)).unwrap_or(&0.0);
        if hit > stick { Hit } else { Stick }
    }

    fn get_expected_reward(&self, state: State, action: Action) -> Reward {
        *self.value_fn.get(&(state, action)).unwrap_or(&0.0)
    }

    fn on_episode_begin(&mut self) {
        self.reward_this_episode = 0.0;
        self.visited_this_episode.drain();
    }

    fn on_episode_step<F>(&mut self, state: State, action: Action,
                          reward: Reward, _next_state: State,
                          _get_next_action: F) -> Option<Action> {
        // We only care about the *first* time a state/action pair
        // was visited in an episode.
        self.visited_this_episode.entry((state, action)).or_insert(true);
        self.reward_this_episode += reward;
        None
    }

    fn on_episode_end(&mut self) {
        for &(state, action) in self.visited_this_episode.keys() {
            let old_value = *self.value_fn.get(&(state, action))
              .unwrap_or(&0.0);
            let step_size = self.step_sizer.update(state, action);
            let new_value = old_value + step_size *
                            (self.reward_this_episode - old_value);
            self.value_fn.insert((state, action), new_value);
        }
    }
}
