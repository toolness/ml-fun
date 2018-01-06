extern crate rand;
extern crate clap;

mod game;
mod montecarlo;

use game::RngDeck;
use rand::{SeedableRng, StdRng};
use clap::{App, SubCommand};

fn run_monte_carlo() {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let mut control = montecarlo::Control::new(deck, rng);

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
