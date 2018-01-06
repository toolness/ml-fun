use std::collections::HashMap;
use std::hash::Hash;


pub fn increment<T: Eq + Hash + Copy>(map: &mut HashMap<T, f32>, key: T,
                                      amount: f32) -> f32 {
    let prev_val = *map.get(&key).unwrap_or(&0.0);
    let new_val = prev_val + amount;
    map.insert(key, new_val);
    new_val
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use util::increment;

    #[test]
    fn test_increment_works() {
        let mut h = HashMap::new();

        h.insert(5, 1.0);
        increment(&mut h, 5, 0.5);

        assert_eq!(*h.get(&5).unwrap(), 1.5);

        increment(&mut h, 600, 5.0);

        assert_eq!(*h.get(&600).unwrap(), 5.0);
    }
}
