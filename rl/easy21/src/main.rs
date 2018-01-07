extern crate rand;
extern crate clap;

extern crate easy21;

use clap::{App, Arg, SubCommand};

use easy21::gpi::Alg;
use easy21::shortcuts;

fn run_monte_carlo(episodes: i32) {
    println!("Performing GPI over {} episodes using Monte Carlo...",
             episodes);

    let gpi = shortcuts::run_monte_carlo(episodes);

    gpi.policy.alg.print_optimal_values();
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
