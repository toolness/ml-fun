use rand::{SeedableRng, StdRng};

use gpi::{Gpi, EpsilonGreedyPolicy};
use montecarlo::MonteCarlo;
use game::RngDeck;

pub fn run_monte_carlo(episodes: i32) -> Gpi<RngDeck<StdRng>, EpsilonGreedyPolicy<StdRng, MonteCarlo>> {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let mc_alg = MonteCarlo::new();
    let policy = EpsilonGreedyPolicy::new(rng, mc_alg);
    let mut gpi = Gpi::new(deck, policy);

    gpi.play_episodes(episodes);

    gpi
}
