use rand::Rng;

use self::Color::*;
use self::Action::*;

const MIN_CARD: i32 = 1;
const MAX_CARD: i32 = 10;
const MIN_SUM: i32 = 1;
const MAX_SUM: i32 = 21;
const DEALER_STICK_MIN: i32 = 17;

type Reward = f32;

const NO_REWARD: Reward = 0.0;
const PLAYER_LOSE_REWARD: Reward = -1.0;
const PLAYER_WIN_REWARD: Reward = 1.0;
const DRAW_REWARD: Reward = NO_REWARD;

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
        cards.iter().fold(0, |sum, ref card| sum + card.value())
    }

    fn value(&self) -> i32 {
        match self.color {
            Red => -self.number,
            Black => self.number,
        }
    }
}

#[derive(Debug)]
struct State {
    dealer: i32,
    player: i32,
}

impl State {
    fn new<T:Rng>(rng: &mut T) -> Self {
        State {
            dealer: Card::draw_color(rng, Black).value(),
            player: Card::draw_color(rng, Black).value()
        }
    }

    fn is_terminal(&self) -> bool {
        self.player < MIN_SUM || self.player > MAX_SUM ||
        self.dealer < MIN_SUM || self.dealer >= DEALER_STICK_MIN
    }

    fn step<T:Rng>(&self, rng: &mut T, action: Action) -> (Self, Reward) {
        match action {
            Hit => {
                let player = self.player + Card::draw(rng).value();
                let reward = if player < MIN_SUM || player > MAX_SUM {
                    PLAYER_LOSE_REWARD
                } else {
                    NO_REWARD
                };
                (State { dealer: self.dealer, player }, reward)
            },
            Stick => {
                let mut dealer = self.dealer;
                while dealer < DEALER_STICK_MIN && dealer >= MIN_SUM {
                    dealer += Card::draw(rng).value();
                }
                let reward = if dealer < MIN_SUM || dealer > MAX_SUM {
                    PLAYER_WIN_REWARD
                } else if dealer == self.player {
                    DRAW_REWARD
                } else if dealer < self.player {
                    PLAYER_WIN_REWARD
                } else {
                    PLAYER_LOSE_REWARD
                };

                (State { dealer, player: self.player }, reward)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use game::{State, Card};
    use game::Color::*;
    use game::Action::*;
    use rand::thread_rng;

    #[test]
    fn state_new_works() {
        let s = State::new(&mut thread_rng());

        assert!(s.dealer > 0);
        assert!(s.player > 0);
    }

    #[test]
    fn hit_eventually_ends_game() {
        let mut start = State::new(&mut thread_rng());

        while !start.is_terminal() {
            start = start.step(&mut thread_rng(), Hit).0;
        }
    }

    #[test]
    fn state_is_terminal_works() {
        for _ in 0..300 {
            let start = State::new(&mut thread_rng());
            let (end, _) = start.step(&mut thread_rng(), Stick);

            assert!(!start.is_terminal(), "{:?} should be non-terminal",
                    start);
            assert!(end.is_terminal(), "{:?} should be terminal", end);
        }
    }

    #[test]
    fn card_new_works() {
        let c = Card::new(1, Red);

        assert_eq!(c.number, 1);
        assert_eq!(c.color, Red);
    }

    #[test]
    fn card_draw_works() {
        for _ in 0..300 {
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
