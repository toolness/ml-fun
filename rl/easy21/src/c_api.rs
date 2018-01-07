use libc::{c_int, c_float};
use rand::{SeedableRng, StdRng};

use game::{State, Action, RngDeck, MIN_SUM, MAX_SUM, MIN_CARD, MAX_CARD};
use gpi::{Gpi, Alg, EpsilonGreedyPolicy};
use montecarlo::MonteCarlo;


const NUM_ACTIONS: i32 = 2;

const DEALER_SIZE: i32 = MAX_CARD + 1 - MIN_CARD;

const PLAYER_SIZE: i32 = MAX_SUM + 1 - MIN_SUM;

const OUTPUT_SIZE: i32 = DEALER_SIZE * PLAYER_SIZE * NUM_ACTIONS;


#[no_mangle]
pub extern "C" fn get_output_size() -> c_int {
    OUTPUT_SIZE
}

#[no_mangle]
pub extern "C" fn run_monte_carlo(
    episodes: c_int,
    output: *mut c_float
) -> i32 {
    if episodes <= 0 {
        return -1;
    }

    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let mc_alg = MonteCarlo::new();
    let policy = EpsilonGreedyPolicy::new(rng, mc_alg);
    let mut gpi = Gpi::new(deck, policy);

    gpi.play_episodes(episodes);

    let mut i = 0;

    for dealer in MIN_CARD..MAX_CARD + 1 {
        for player in MIN_SUM..MAX_SUM + 1 {
            let state = State { dealer, player };
            let hit = gpi.policy.alg.get_expected_reward(state, Action::Hit);
            let stick = gpi.policy.alg.get_expected_reward(state, Action::Stick);

            unsafe {
                *output.offset(i) = hit;
                *output.offset(i + 1) = stick;
            }

            i += 2;
        }
    }

    0
}
