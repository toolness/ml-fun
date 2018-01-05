use rand::Rng;

use self::Color::*;

const MIN_CARD: i32 = 1;
const MAX_CARD: i32 = 10;

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
        assert!(number >= MIN_CARD && number <= MAX_CARD);
        Self { number, color }
    }

    fn draw_color<T:Rng>(rng: &mut T, color: Color) -> Self {
        Self::new(rng.gen_range(MIN_CARD, MAX_CARD + 1), color)
    }

    fn draw<T:Rng>(rng: &mut T) -> Self {
        // Red should be drawn with a probability of 1/3, while
        // Black has a 2/3 probability.
        let color = *rng.choose(&[Red, Black, Black]).unwrap();
        Self::draw_color(rng, color)
    }

    fn sum(cards: &Vec<Card>) -> i32 {
        cards.iter().fold(0, |sum, ref card| {
            sum + match card.color {
                Red => -card.number,
                Black => card.number,
            }
        })
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
    use game::{State, Card};
    use game::Color::*;
    use rand::{Rng, thread_rng};

    #[test]
    fn state_new_works() {
        let s = State::new();

        assert_eq!(s.dealer, 0);
        assert_eq!(s.player, 0);
    }

    #[test]
    fn card_new_works() {
        let c = Card::new(1, Red);

        assert_eq!(c.number, 1);
        assert_eq!(c.color, Red);
    }

    #[test]
    fn card_draw_works() {
        for i in 0..300 {
            Card::draw(&mut thread_rng());
        }
    }

    #[test]
    fn card_sum_works() {
        assert_eq!(Card::sum(&vec![
            Card::new(1, Red),
            Card::new(3, Red),
        ]), -4);

        assert_eq!(Card::sum(&vec![
            Card::new(1, Red),
            Card::new(3, Black),
        ]), 2);
    }
}
