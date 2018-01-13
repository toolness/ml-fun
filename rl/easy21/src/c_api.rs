use libc::{c_int, c_float};
use rand::Rng;

use game::{State, Action, Deck, MIN_SUM, MAX_SUM, MIN_CARD, MAX_CARD};
use gpi::{Alg, Gpi, EpsilonGreedyPolicy};
use shortcuts;
use validators;


const NUM_ACTIONS: usize = 2;

const DEALER_SIZE: usize = (MAX_CARD + 1 - MIN_CARD) as usize;

const PLAYER_SIZE: usize = (MAX_SUM + 1 - MIN_SUM) as usize;

const OUTPUT_SIZE: usize = DEALER_SIZE * PLAYER_SIZE * NUM_ACTIONS;


#[no_mangle]
pub extern "C" fn get_output_size() -> c_int {
    OUTPUT_SIZE as i32
}

fn write_expected_reward_matrix(alg: &Alg, output: *mut c_float) {
    let mut i = 0;

    for dealer in MIN_CARD..MAX_CARD + 1 {
        for player in MIN_SUM..MAX_SUM + 1 {
            let state = State { dealer, player };
            let hit = alg.get_expected_reward(state, Action::Hit);
            let stick = alg.get_expected_reward(state, Action::Stick);

            unsafe {
                *output.offset(i) = hit;
                *output.offset(i + 1) = stick;
            }

            i += 2;
        }
    }
}

fn run_gpi<T: Deck, U: Rng, V: Alg>(
    gpi: &mut Gpi<T, EpsilonGreedyPolicy<U, V>>,
    episodes: c_int,
    output: *mut c_float,
    cb: Option<extern "C" fn()>
) {
    match cb {
        None => {
            gpi.play_episodes(episodes);
            write_expected_reward_matrix(&gpi.policy.alg, output);
        },
        Some(func) => {
            for _ in 0..episodes {
                gpi.play_episode();
                write_expected_reward_matrix(&gpi.policy.alg, output);
                func();
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn run_monte_carlo(
    episodes: c_int,
    output: *mut c_float,
    cb: Option<extern "C" fn()>
) -> i32 {
    if !validators::episodes(episodes) {
        return -1;
    }

    let mut gpi = shortcuts::run_monte_carlo(0);

    run_gpi(&mut gpi, episodes, output, cb);

    0
}

#[no_mangle]
pub extern "C" fn run_sarsa(
    episodes: c_int,
    lambda: c_float,
    output: *mut c_float,
    cb: Option<extern "C" fn()>,
) -> i32 {
    if !validators::episodes(episodes) {
        return -1;
    }

    if !validators::lambda(lambda) {
        return -1;
    }

    let mut gpi = shortcuts::run_sarsa(0, lambda);

    run_gpi(&mut gpi, episodes, output, cb);

    0
}

#[no_mangle]
pub extern "C" fn run_lfa(
    episodes: c_int,
    lambda: c_float,
    epsilon: c_float,
    step_size: c_float,
    output: *mut c_float,
    cb: Option<extern "C" fn()>,
) -> i32 {
    if !validators::episodes(episodes) {
        return -1;
    }

    if !validators::lambda(lambda) {
        return -1;
    }

    if !validators::epsilon(epsilon) {
        return -1;
    }

    if !validators::step_size(step_size) {
        return -1;
    }

    let mut gpi = shortcuts::run_lfa(episodes, lambda, epsilon, step_size);

    run_gpi(&mut gpi, episodes, output, cb);

    0
}

#[cfg(test)]
mod tests {
    use gpi::tests::DumbAlg;
    use game::Action;
    use c_api::*;

    #[test]
    fn test_write_expected_reward_matrix_works() {
        let alg = DumbAlg { action: Action::Hit, reward: 5.0 };
        let mut output = [0.0; OUTPUT_SIZE];

        write_expected_reward_matrix(&alg, output.as_mut_ptr());

        for i in 0..OUTPUT_SIZE {
            assert_eq!(output[i], 5.0);
        }
    }

    #[test]
    fn test_run_monte_carlo_works() {
        assert_eq!(run_monte_carlo(5, [0.0; OUTPUT_SIZE].as_mut_ptr(), None), 0);
    }

    #[test]
    fn test_run_monte_carlo_returns_err_if_invalid_episodes() {
        assert_eq!(run_monte_carlo(-1, [0.0; OUTPUT_SIZE].as_mut_ptr(), None), -1);
    }
}
