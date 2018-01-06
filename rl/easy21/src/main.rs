extern crate rand;
extern crate clap;

mod game;
mod gpi;
mod montecarlo;

use rand::{SeedableRng, StdRng};
use clap::{App, SubCommand};

fn run_monte_carlo() {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = game::RngDeck::new(rng);
    let mc_alg = montecarlo::MonteCarlo::new();
    let mut control = gpi::Control::new(deck, rng, mc_alg);

    const EPISODES: i32 = 30_000;

    println!("Playing {} episodes using Monte Carlo control...",
             EPISODES);
    control.play_episodes(EPISODES);

    control.print_optimal_value_fn();
}

fn main() {
    let matches = App::new("easy21")
      .subcommand(SubCommand::with_name("mc")
        .about("runs monte carlo control"))
      .get_matches();

    if let Some(_) = matches.subcommand_matches("mc") {
        run_monte_carlo();
    } else {
        println!("Unknown command. Try running this program with '--help'.");
        std::process::exit(1);
    }
}
