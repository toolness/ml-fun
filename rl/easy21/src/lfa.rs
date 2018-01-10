use std::collections::HashMap;
use std::ops::Range;

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

    fn get_expected_reward(&self, state: State, action: Action) -> Reward {
        let features = to_feature_vector(state, action);
        let weights = [0.0; NUM_FEATURES];
        let _ = dot_product(features, weights);
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

type Ranges = [Range<i32>];

const DEALER_RANGES: &Ranges = &[1..5, 4..8, 7..11];

const PLAYER_RANGES: &Ranges = &[1..7, 4..10, 7..13, 10..16, 13..19, 16..22];

const NUM_FEATURES: usize = 36;

type FeatureVector = [f32; NUM_FEATURES];

type Weights = FeatureVector;

fn dot_product(features: FeatureVector, weights: Weights) -> f32 {
    (0..NUM_FEATURES).fold(0.0, |sum, i| sum + features[i] * weights[i])
}

fn get_ranges_inside(value: i32, ranges: &Ranges) -> Vec<f32> {
    ranges.iter().map(|range| if value >= range.start && value < range.end {
        1.0
    } else {
        0.0
    }).collect()
}

fn to_feature_vector(state: State, action: Action) -> FeatureVector {
    let mut vector = [0.0; NUM_FEATURES];
    let mut i = 0;
    let dealer_indexes = get_ranges_inside(state.dealer, DEALER_RANGES);
    let player_indexes = get_ranges_inside(state.player, PLAYER_RANGES);
    let action_indexes = match action {
        Hit => vec![1.0, 0.0],
        Stick => vec![0.0, 1.0]
    };

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
