extern crate rand;
extern crate clap;

extern crate easy21;

use rand::{SeedableRng, StdRng};
use clap::{App, Arg, SubCommand};

use easy21::gpi::{Gpi, Alg};
use easy21::montecarlo::MonteCarlo;
use easy21::game::RngDeck;

fn run_monte_carlo(episodes: i32) {
    let seed: &[_] = &[1, 2, 3, 4];
    let rng: StdRng = SeedableRng::from_seed(seed);
    let deck = RngDeck::new(rng);
    let mc_alg = MonteCarlo::new();
    let mut gpi = Gpi::new(deck, rng, mc_alg);

    println!("Performing GPI over {} episodes using Monte Carlo...",
             episodes);
    gpi.play_episodes(episodes);

    gpi.alg.print_optimal_values();
}

fn fail(msg: &str) {
    eprintln!("{}", msg);
    std::process::exit(1);
}

fn main() {
    let matches = App::new("easy21")
      .arg(Arg::with_name("episodes")
        .short("e")
        .long("episodes")
        .help("number of episodes to play")
        .default_value("30000")
        .takes_value(true))
      .subcommand(SubCommand::with_name("mc")
        .about("runs monte carlo control"))
      .get_matches();

    let episodes = matches.value_of("episodes")
      .unwrap().parse::<i32>().unwrap_or(0);

    if episodes <= 0 {
        fail("Episodes must be a number greater than 0.");
    }

    if let Some(_) = matches.subcommand_matches("mc") {
        run_monte_carlo(episodes);
    } else {
        fail("Unknown command. Try running this program with '--help'.");
    }
}
