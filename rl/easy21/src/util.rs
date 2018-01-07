use std::collections::HashMap;
use std::hash::Hash;

use game::{State, Action};


pub fn increment<T: Eq + Hash + Copy>(map: &mut HashMap<T, f32>, key: T,
                                      amount: f32) -> f32 {
    let prev_val = *map.get(&key).unwrap_or(&0.0);
    let new_val = prev_val + amount;
    map.insert(key, new_val);
    new_val
}


pub struct VaryingStepSizer {
    visits: HashMap<(State, Action), f32>,
}

impl VaryingStepSizer {
    pub fn new() -> Self {
        VaryingStepSizer { visits: HashMap::new() }
    }

    pub fn update(&mut self, state: State, action: Action) -> f32 {
        let visits = increment(&mut self.visits, (state, action), 1.0);
        1.0 / visits
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use game::{State, Action};
    use util::*;

    #[test]
    fn test_increment_works() {
        let mut h = HashMap::new();

        h.insert(5, 1.0);
        increment(&mut h, 5, 0.5);

        assert_eq!(*h.get(&5).unwrap(), 1.5);

        increment(&mut h, 600, 5.0);

        assert_eq!(*h.get(&600).unwrap(), 5.0);
    }

    #[test]
    fn test_varying_step_sizer_works() {
        let mut s = VaryingStepSizer::new();
        let state = State { dealer: 1, player: 1 };

        assert_eq!(s.update(state, Action::Hit), 1.0);
        assert_eq!(s.update(state, Action::Hit), 0.5);
        assert_eq!(s.update(state, Action::Hit), 1.0 / 3.0);
        assert_eq!(s.update(state, Action::Stick), 1.0);
    }
}
