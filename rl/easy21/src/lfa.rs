use std::collections::HashMap;
use std::ops::Range;

use game::{State, Action, Reward};
use game::Action::*;
use gpi::Alg;
use util::{increment};


type EligibilityHash = HashMap<(State, Action), f32>;

pub struct LinearFunctionApproximator {
    traces: EligibilityHash,
    weights: Weights,
    lambda: f32,
    step_size: f32,
}

impl LinearFunctionApproximator {
    pub fn new(lambda: f32, step_size: f32) -> Self {
        LinearFunctionApproximator {
            traces: HashMap::new(),
            weights: [0.0; NUM_FEATURES],
            lambda,
            step_size,
        }
    }
}

impl Alg for LinearFunctionApproximator {
    fn choose_best_action(&self, state: State) -> Action {
        let hit = self.get_expected_reward(state, Hit);
        let stick = self.get_expected_reward(state, Stick);
        if hit > stick { Hit } else { Stick }
    }

    fn get_expected_reward(&self, state: State, action: Action) -> Reward {
        let features = to_feature_vector(state, action);
        dot_product(features, self.weights)
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
        let td_error = reward +
                       self.get_expected_reward(next_state,
                                                next_action.unwrap()) -
                       self.get_expected_reward(state, action);
        increment(&mut self.traces, (state, action), 1.0);
        for (&(state, action), trace) in self.traces.iter_mut() {
            let eligibility_trace = *trace;

            let features = to_feature_vector(state, action);
            for i in 0..NUM_FEATURES {
                self.weights[i] += self.step_size * eligibility_trace *
                                   td_error * features[i];
            }

            *trace = self.lambda * eligibility_trace;
        }
        next_action
    }
}

type Ranges = [Range<i32>];

const NUM_DEALER_RANGES: usize = 3;

const DEALER_RANGES: &[Range<i32>; NUM_DEALER_RANGES] = &[
    1..5, 4..8, 7..11
];

const NUM_PLAYER_RANGES: usize = 6;

const PLAYER_RANGES: &[Range<i32>; NUM_PLAYER_RANGES] = &[
    1..7, 4..10, 7..13, 10..16, 13..19, 16..22
];

const NUM_ACTIONS: usize = 2;

const NUM_FEATURES: usize = NUM_DEALER_RANGES * NUM_PLAYER_RANGES *
                            NUM_ACTIONS;

type FeatureVector = [f32; NUM_FEATURES];

type Weights = FeatureVector;

fn dot_product(features: FeatureVector, weights: Weights) -> f32 {
    (0..NUM_FEATURES).fold(0.0, |sum, i| sum + features[i] * weights[i])
}

fn get_ranges_inside(value: i32, ranges: &Ranges, target: &mut [f32]) {
    for i in 0..ranges.len() {
        target[i] = if value >= ranges[i].start && value < ranges[i].end {
            1.0
        } else {
            0.0
        };
    }
}

fn to_feature_vector(state: State, action: Action) -> FeatureVector {
    let mut vector = [0.0; NUM_FEATURES];
    let mut i = 0;
    let mut dealer_indexes = [0.0; NUM_DEALER_RANGES];
    let mut player_indexes = [0.0; NUM_PLAYER_RANGES];
    let action_indexes: &[f32; NUM_ACTIONS] = match action {
        Hit => &[1.0, 0.0],
        Stick => &[0.0, 1.0]
    };

    get_ranges_inside(state.player, PLAYER_RANGES, &mut player_indexes);
    get_ranges_inside(state.dealer, DEALER_RANGES, &mut dealer_indexes);

    for dealer in dealer_indexes.iter() {
        for player in player_indexes.iter() {
            for action in action_indexes.iter() {
                vector[i] = dealer * player * action;
                i += 1;
            }
        }
    }

    vector
}

#[cfg(test)]
mod tests {
    use lfa::*;
    use game::State;

    #[test]
    fn test_to_feature_vector_works_with_one_cuboid() {
        let fv = to_feature_vector(State { dealer: 1, player: 1 }, Hit);
        let mut expected = vec![0.0; NUM_FEATURES];
        expected[0] = 1.0;

        assert_eq!(fv.to_vec(), expected);
    }

    #[test]
    fn test_to_feature_vector_works_with_two_cuboids() {
        let fv = to_feature_vector(State { dealer: 1, player: 5 }, Hit);
        let mut expected = vec![0.0; NUM_FEATURES];
        expected[0] = 1.0;
        expected[2] = 1.0;

        assert_eq!(fv.to_vec(), expected);
    }

    #[test]
    fn test_dot_product_works() {
        let features = [1.0; NUM_FEATURES];
        let mut weights = [2.0; NUM_FEATURES];

        assert_eq!(dot_product(features, weights), 2.0 * NUM_FEATURES as f32);

        weights[0] = 1.0;

        assert_eq!(dot_product(features, weights), 2.0 * NUM_FEATURES as f32 - 1.0);

        weights[0] = 0.0;

        assert_eq!(dot_product(features, weights), 2.0 * NUM_FEATURES as f32 - 2.0);
    }
}
