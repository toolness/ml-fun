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

#[cfg(test)]
mod tests {
    use game::State;

    #[test]
    fn state_new_works() {
        let s = State::new();

        assert_eq!(s.dealer, 0);
        assert_eq!(s.player, 0);
    }
}
