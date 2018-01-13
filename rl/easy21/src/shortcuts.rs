use rand::{SeedableRng, StdRng};

use gpi::{Gpi, EpsilonGreedyPolicy};
use montecarlo::MonteCarlo;
use sarsa::SarsaLambda;
use lfa::LinearFunctionApproximator;
use game::RngDeck;

pub fn run_monte_carlo(episodes: i32) -> Gpi<RngDeck<StdRng>, EpsilonGreedyPolicy<StdRng, MonteCarlo>> {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let mc_alg = MonteCarlo::new();
    let policy = EpsilonGreedyPolicy::new(rng, mc_alg);
    let mut gpi = Gpi::new(deck, policy);

    if episodes > 0 {
        gpi.play_episodes(episodes);
    }

    gpi
}

pub fn run_sarsa(episodes: i32, lambda: f32) -> Gpi<RngDeck<StdRng>, EpsilonGreedyPolicy<StdRng, SarsaLambda>> {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let sarsa_alg = SarsaLambda::new(lambda);
    let policy = EpsilonGreedyPolicy::new(rng, sarsa_alg);
    let mut gpi = Gpi::new(deck, policy);

    if episodes > 0 {
        gpi.play_episodes(episodes);
    }

    gpi
}

pub fn run_lfa(episodes: i32, lambda: f32, epsilon: f32, step_size: f32) -> Gpi<RngDeck<StdRng>, EpsilonGreedyPolicy<StdRng, LinearFunctionApproximator>> {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let lfa_alg = LinearFunctionApproximator::new(lambda, step_size);
    let policy = EpsilonGreedyPolicy::new(rng, lfa_alg).with_constant_epsilon(epsilon);
    let mut gpi = Gpi::new(deck, policy);

    if episodes > 0 {
        gpi.play_episodes(episodes);
    }

    gpi
}

#[cfg(test)]
mod tests {
    use shortcuts::*;

    #[test]
    fn test_run_sarsa_works() {
        run_sarsa(3, 0.5);
    }
}
