#![cfg(feature = "testkit")]
use std::f64::consts::TAU;
use versum_dsp::biquad::{Biquad, BiquadKind};
use versum_dsp::testkit::{assert_blockwise_eq, count_allocs, CountingAlloc};
use versum_dsp::{Block, Rng};

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

const SR: f64 = 48000.0;

fn sine(hz: f64, n: usize) -> Vec<f32> {
    (0..n).map(|i| (TAU * hz * i as f64 / SR).sin() as f32).collect()
}

/// Hann-windowed single-bin DFT; a sine of peak `a` reads `a`.
fn amp_at(s: &[f32], hz: f64) -> f64 {
    let (mut re, mut im, mut ws) = (0.0, 0.0, 0.0);
    for (i, v) in s.iter().enumerate() {
        let w = 0.5 - 0.5 * (TAU * i as f64 / s.len() as f64).cos();
        let a = TAU * hz * i as f64 / SR;
        re += *v as f64 * w * a.cos();
        im -= *v as f64 * w * a.sin();
        ws += w;
    }
    2.0 * (re * re + im * im).sqrt() / ws
}

/// Gain in dB at `hz`, measured on the steady-state second half of a long sine run.
fn gain_db(kind: BiquadKind, freq: f64, q_db: f64, hz: f64) -> f64 {
    let mut b = Biquad::new(SR);
    b.set(kind, freq, q_db);
    let mut x = sine(hz, 48000);
    b.process(&mut x);
    20.0 * amp_at(&x[24000..], hz).log10()
}

#[test]
fn lowpass_passband_and_stopband() {
    let pass = gain_db(BiquadKind::Lowpass, 1000.0, 0.0, 100.0);
    assert!(pass.abs() < 0.2, "{pass}");
    let stop = gain_db(BiquadKind::Lowpass, 1000.0, 0.0, 10000.0);
    assert!(stop <= -35.0, "{stop}");
}

#[test]
fn lowpass_resonance_matches_q_db() {
    let g = gain_db(BiquadKind::Lowpass, 1000.0, 12.0, 1000.0);
    assert!((g - 12.0).abs() < 0.7, "{g}");
}

#[test]
fn highpass_22hz_blocks_dc() {
    let mut b = Biquad::new(SR);
    b.set(BiquadKind::Highpass, 22.0, 0.7);
    let mut y: f32 = 1.0;
    for _ in 0..48000 {
        y = b.tick(1.0);
    }
    assert!(y.abs() < 1e-3, "{y}");
    let g = gain_db(BiquadKind::Highpass, 22.0, 0.7, 1000.0);
    assert!(g.abs() < 0.2, "{g}");
    let mut b = Biquad::new(SR);
    b.set(BiquadKind::Highpass, 22.0, 0.7);
    let mut x = sine(1000.0, 48000);
    b.process(&mut x);
    assert!(x.iter().all(|v| v.abs() <= 1.023));
}

#[test]
fn set_keeps_state_and_stays_finite() {
    let mut b = Biquad::new(SR);
    let mut r = Rng::new(3);
    let x = sine(440.0, 1000 * 64);
    for (i, v) in x.iter().enumerate() {
        if i % 64 == 0 {
            let f = 20.0 + r.next_f64() * (18000.0 - 20.0);
            let q = -20.0 + r.next_f64() * 44.0;
            let kind = if r.next_f64() < 0.5 { BiquadKind::Lowpass } else { BiquadKind::Highpass };
            b.set(kind, f, q);
        }
        let y = b.tick(*v);
        assert!(y.is_finite() && y.abs() < 100.0, "sample {i}: {y}");
    }
}

#[test]
fn blockwise_equals_per_sample() {
    let x = sine(777.0, 4096);
    assert_blockwise_eq(
        || {
            let mut b = Biquad::new(SR);
            b.set(BiquadKind::Lowpass, 2000.0, 6.0);
            b
        },
        &x,
        &[1, 7, 64, 333, 4096],
    );
}

#[test]
fn no_alloc_in_set_tick_process() {
    let mut b = Biquad::new(SR);
    let mut buf = sine(440.0, 512);
    let (_, n) = count_allocs(|| {
        b.set(BiquadKind::Highpass, 100.0, 1.0);
        b.process(&mut buf);
        b.tick(0.5);
        b.reset();
    });
    assert_eq!(n, 0);
}

#[test]
fn non_finite_input_is_guarded_and_resets() {
    let mut b = Biquad::new(SR);
    b.set(BiquadKind::Lowpass, 1000.0, 0.0);
    for _ in 0..100 {
        b.tick(1.0);
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(b.tick(bad), 0.0);
    }
    // state was cleared: silence in gives exact silence out
    assert_eq!(b.tick(0.0), 0.0);
    assert!(b.tick(0.5).is_finite());
}

#[test]
fn reset_clears_state() {
    let mut b = Biquad::new(SR);
    b.set(BiquadKind::Lowpass, 500.0, 6.0);
    for _ in 0..100 {
        b.tick(1.0);
    }
    b.reset();
    assert_eq!(b.tick(0.0), 0.0);
}
