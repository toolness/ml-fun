extern crate rand;

use rand::{Rng, SeedableRng, StdRng};

fn main() {
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    println!("Hello, world! {}", rng.gen::<i32>());
}
