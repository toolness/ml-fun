extern crate rand;

mod game;
mod montecarlo;

use game::RngDeck;
use rand::{SeedableRng, StdRng};

fn main() {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let mut control = montecarlo::Control::new(deck, rng);

    const EPISODES: i32 = 30000;

    println!("Playing {} episodes using Monte Carlo control...",
             EPISODES);
    control.play_episodes(EPISODES);

    println!("Done.");
}
