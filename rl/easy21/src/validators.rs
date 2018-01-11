pub fn episodes(v: i32) -> bool {
    v > 0
}

pub fn lambda(v: f32) -> bool {
    v >= 0.0 && v <= 1.0
}

pub fn epsilon(v: f32) -> bool {
    v >= 0.0 && v <= 1.0
}

pub fn step_size(v: f32) -> bool {
    v > 0.0
}

#[cfg(test)]
mod tests {
    use validators::*;

    #[test]
    fn test_episodes() {
        assert!(episodes(1));
        assert!(episodes(100));
        assert!(!episodes(0));
        assert!(!episodes(-1));
    }

    #[test]
    fn test_lambda() {
        assert!(lambda(0.0));
        assert!(lambda(1.0));
        assert!(!lambda(-1.0));
        assert!(!lambda(1.1));
    }

    #[test]
    fn test_epsilon() {
        assert!(epsilon(0.0));
        assert!(epsilon(1.0));
        assert!(!epsilon(-1.0));
        assert!(!epsilon(1.1));
    }

    #[test]
    fn test_step_size() {
        assert!(!step_size(0.0));
        assert!(step_size(1.0));
        assert!(!step_size(-1.0));
        assert!(step_size(1.1));
    }
}
