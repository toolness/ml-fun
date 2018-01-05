use rand::Rng;

type Reward = f32;

#[derive(PartialEq)]
#[derive(Debug)]
enum Action {
    Hit,
    Stick
}

#[derive(Clone)]
#[derive(Copy)]
#[derive(PartialEq)]
#[derive(Debug)]
pub enum Color {
    Red,
    Black
}

#[derive(PartialEq)]
#[derive(Debug)]
pub struct Card {
    number: i32,
    color: Color,
}

impl Card {
    fn new(number: i32, color: Color) -> Self {
        assert!(number >= 1 && number <= 10);
        Self { number, color }
    }

    fn draw<T:Rng>(rng: &mut T) -> Self {
        // Red should be drawn with a probability of 1/3, while
        // Black has a 2/3 probability.
        let color = *rng.choose(&[
            Color::Red,
            Color::Black,
            Color::Black,
        ]).unwrap();
        Self::new(rng.gen_range(1, 11), color)
    }
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
    use game::{State, Color, Card};
    use rand::{Rng, thread_rng};

    #[test]
    fn state_new_works() {
        let s = State::new();

        assert_eq!(s.dealer, 0);
        assert_eq!(s.player, 0);
    }

    #[test]
    fn card_new_works() {
        let c = Card::new(1, Color::Red);

        assert_eq!(c.number, 1);
        assert_eq!(c.color, Color::Red);
    }

    #[test]
    fn card_draw_works() {
        for i in 0..300 {
            Card::draw(&mut thread_rng());
        }
    }
}
