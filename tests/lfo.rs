use versum_dsp::lfo::{Lfo, LfoShape};

const SR: f64 = 48000.0;

#[test]
fn sine_period_and_amplitude() {
    let mut l = Lfo::new(LfoShape::Sine, 2.0, 0.0, SR);
    let mut max = f32::MIN;
    for i in 0..24000 {
        if i == 0 {
            assert!(l.value().abs() < 1e-6);
        }
        max = max.max(l.tick());
    }
    assert!(l.value().abs() < 1e-6, "half period: {}", l.value());
    assert!((max - 1.0).abs() < 1e-6, "max {max}");
}

#[test]
fn triangle_shape() {
    let mut l = Lfo::new(LfoShape::Triangle, 1.0, 0.0, SR);
    let mut prev = l.tick();
    assert!(prev.abs() < 1e-6);
    for i in 1..=36000 {
        let v = l.tick();
        if i <= 12000 {
            assert!(v > prev, "rising at {i}");
            assert!((v as f64 - 4.0 * i as f64 / SR).abs() < 1e-5);
        }
        prev = v;
    }
    let mut a = Lfo::new(LfoShape::Triangle, 1.0, 0.0, SR);
    a.advance(12000);
    assert!((a.value() - 1.0).abs() < 1e-6);
    let mut b = Lfo::new(LfoShape::Triangle, 1.0, 0.0, SR);
    b.advance(36000);
    assert!((b.value() + 1.0).abs() < 1e-6);
}

#[test]
fn advance_equals_ticks() {
    let mut a = Lfo::new(LfoShape::Sine, 3.7, 0.1, SR);
    let mut b = a.clone();
    a.advance(12345);
    for _ in 0..12345 {
        b.tick();
    }
    assert_eq!(a.value().to_bits(), b.value().to_bits());
}

#[test]
fn phase_survives_set_freq() {
    let mut l = Lfo::new(LfoShape::Sine, 5.0, 0.0, SR);
    l.advance(1000);
    let before = l.value();
    l.set_freq(9.0);
    assert_eq!(l.value(), before);
    l.set_freq(f64::NAN);
    l.advance(10);
    let mut m = Lfo::new(LfoShape::Sine, 5.0, 0.0, SR);
    m.advance(1000);
    m.set_freq(9.0);
    m.advance(10);
    assert_eq!(l.value(), m.value());
}

#[test]
fn initial_phase_and_reset() {
    let mut l = Lfo::new(LfoShape::Sine, 2.0, 0.25, SR);
    assert_eq!(l.value(), 1.0);
    l.advance(5000);
    assert_ne!(l.value(), 1.0);
    l.reset();
    assert_eq!(l.value(), 1.0);
}
