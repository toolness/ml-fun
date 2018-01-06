use std::collections::HashMap;

use game::{State, Action, Reward};
use game::Action::*;
use gpi::Alg;
use util::increment;

type ValueFn = HashMap<(State, Action), Reward>;

pub struct MonteCarlo {
    value_fn: ValueFn,
    total_visits: HashMap<(State, Action), Reward>,
}

impl MonteCarlo {
    pub fn new() -> Self {
        MonteCarlo {
            value_fn: HashMap::new(),
            total_visits: HashMap::new(),
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

    fn on_episode_end(
        &mut self,
        visited: &HashMap<(State, Action), f32>,
        reward: Reward
    ) {
        for &(state, action) in visited.keys() {
            // Even though the value of `visited` tells us how many times
            // the state/action pair was visited, we only care about
            // the *first* time it was visited in an episode.
            let visits = increment(&mut self.total_visits, (state, action),
                                   1.0);
            let old_value = *self.value_fn.get(&(state, action))
              .unwrap_or(&0.0);
            let step_size = 1.0 / visits;
            let new_value = old_value + step_size * (reward - old_value);
            self.value_fn.insert((state, action), new_value);
        }
    }
}
