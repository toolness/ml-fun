extern crate rand;
extern crate clap;

extern crate easy21;

use clap::{App, Arg, ArgMatches, SubCommand};

use easy21::gpi::Alg;
use easy21::shortcuts;
use easy21::validators;

fn run_monte_carlo(episodes: i32) {
    println!("Performing GPI over {} episodes using Monte Carlo...",
             episodes);

    let gpi = shortcuts::run_monte_carlo(episodes);

    gpi.policy.alg.print_optimal_values();
}

fn run_sarsa(episodes: i32, lambda: f32) {
    println!(
        "Performing GPI over {} episodes using Sarsa with lambda={}...",
        episodes,
        lambda
    );

    let gpi = shortcuts::run_sarsa(episodes, lambda);

    gpi.policy.alg.print_optimal_values();
}

fn validate_episodes(v: String) -> Result<(), String> {
    return if validators::episodes(v.parse::<i32>().unwrap_or(-1)) {
        Ok(())
    } else {
        Err(String::from("Episodes must be a number greater than 0."))
    };
}

fn get_episodes(m: &ArgMatches) -> i32 {
    m.value_of("episodes").unwrap().parse::<i32>().unwrap()
}

fn validate_lambda(v: String) -> Result<(), String> {
    return if validators::lambda(v.parse::<f32>().unwrap_or(-1.0)) {
        Ok(())
    } else {
        Err(String::from("Lambda must be a float between 0 and 1."))
    }
}

fn get_lambda(m: &ArgMatches) -> f32 {
    m.value_of("lambda").unwrap().parse::<f32>().unwrap()
}

fn main() {
    let episodes_arg = Arg::with_name("episodes")
        .short("e")
        .long("episodes")
        .help("number of episodes to play")
        .default_value("1000")
        .takes_value(true)
        .validator(validate_episodes);

    let matches = App::new("easy21")
      .subcommand(SubCommand::with_name("mc")
        .arg(episodes_arg.clone())
        .about("runs monte carlo control"))
      .subcommand(SubCommand::with_name("sarsa")
        .about("runs sarsa lambda control")
        .arg(episodes_arg.clone())
        .arg(Arg::with_name("lambda")
          .short("l")
          .long("lambda")
          .help("lambda setting")
          .default_value("0.5")
          .takes_value(true)
          .validator(validate_lambda)))
      .get_matches();

    if let Some(submatches) = matches.subcommand_matches("mc") {
        run_monte_carlo(get_episodes(&submatches));
    } else if let Some(submatches) = matches.subcommand_matches("sarsa") {
        run_sarsa(get_episodes(&submatches), get_lambda(&submatches));
    } else {
        eprintln!("error: Invalid subcommand\n\n{}\n", matches.usage());
        eprintln!("For more information try --help");
        std::process::exit(1);
    }
}
