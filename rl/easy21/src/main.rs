extern crate rand;

use rand::{Rng, SeedableRng, StdRng};

type Reward = f32;

enum Action {
    Hit,
    Stick
}

enum Color {
    Red,
    Black
}

struct State {
    dealer: i32,
    player: i32,
}

impl State {
    fn new() -> Self {
        State { dealer: 0, player: 0 }
    }

    fn step(self, action: Action) -> (Self, Reward) {
        unimplemented!();
    }
}

fn main() {
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    println!("Hello, world! {}", rng.gen::<i32>());
}
