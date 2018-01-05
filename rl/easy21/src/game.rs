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

pub trait Deck {
    fn draw_color(&mut self, color: Color) -> Card;
    fn draw(&mut self) -> Card;
}

pub struct RngDeck<T: Rng> {
    rng: T
}

impl<T: Rng> RngDeck<T> {
    fn new(rng: T) -> Self {
        RngDeck { rng }
    }
}

impl<T: Rng> Deck for RngDeck<T> {
    fn draw_color(&mut self, color: Color) -> Card {
        Card::new(self.rng.gen_range(MIN_CARD, MAX_CARD + 1), color)
    }

    fn draw(&mut self) -> Card {
        // Red should be drawn with a probability of 1/3, while
        // Black has a 2/3 probability.
        let color = *self.rng.choose(&[Red, Black, Black]).unwrap();
        self.draw_color(color)
    }
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
    fn new<T: Deck>(deck: &mut T) -> Self {
        State {
            dealer: deck.draw_color(Black).value(),
            player: deck.draw_color(Black).value()
        }
    }

    fn is_terminal(&self) -> bool {
        self.player < MIN_SUM || self.player > MAX_SUM ||
        self.dealer < MIN_SUM || self.dealer >= DEALER_STICK_MIN
    }

    fn step<T: Deck>(&self, deck: &mut T, action: Action) -> (Self, Reward) {
        match action {
            Hit => {
                let player = self.player + deck.draw().value();
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
                    dealer += deck.draw().value();
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
    use game::{State, Card, Deck, RngDeck};
    use game::Color::*;
    use game::Action::*;
    use rand::{thread_rng, ThreadRng};

    fn rng_deck() -> RngDeck<ThreadRng> {
        RngDeck::new(thread_rng())
    }

    #[test]
    fn state_new_works() {
        let s = State::new(&mut rng_deck());

        assert!(s.dealer > 0);
        assert!(s.player > 0);
    }

    #[test]
    fn hit_eventually_ends_game() {
        let mut deck = rng_deck();
        let mut start = State::new(&mut deck);

        while !start.is_terminal() {
            start = start.step(&mut deck, Hit).0;
        }
    }

    #[test]
    fn state_is_terminal_works() {
        let mut deck = rng_deck();

        for _ in 0..300 {
            let start = State::new(&mut deck);
            let (end, _) = start.step(&mut deck, Stick);

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
    fn rng_deck_draw_works() {
        let mut deck = rng_deck();

        for _ in 0..300 {
            deck.draw();
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
