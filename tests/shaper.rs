#![cfg(feature = "testkit")]
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::{PI, TAU};
use versum_dsp::shaper::Shaper;
use versum_dsp::testkit::{assert_blockwise_eq, count_allocs, CountingAlloc};
use versum_dsp::Block;

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

const SR: f64 = 48000.0;

fn sine(hz: f64, amp: f64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (amp * (TAU * hz * i as f64 / SR).sin()) as f32)
        .collect()
}

fn table<const N: usize>(f: impl Fn(f64) -> f64) -> [f32; N] {
    let mut t = [0.0f32; N];
    for (i, v) in t.iter_mut().enumerate() {
        *v = f(i as f64 / (N - 1) as f64 * 2.0 - 1.0) as f32;
    }
    t
}

fn fold() -> [f32; 2049] {
    table(|x| (5.0 * PI * x).sin())
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

/// Energy outside the harmonics of `f0` over energy in them, in dB (same method as the crate's `measure::alias_db`).
fn alias_db(s: &[f32], f0: f64) -> f64 {
    let n = 1usize << (usize::BITS - 1 - s.len().leading_zeros());
    let mut buf: Vec<Complex<f64>> = s
        .iter()
        .take(n)
        .enumerate()
        .map(|(i, v)| {
            let x = TAU * i as f64 / n as f64;
            let w =
                0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos();
            Complex::new(*v as f64 * w, 0.0)
        })
        .collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(n)
        .process(&mut buf);
    let half = n / 2;
    let mut harmonic = vec![false; half + 1];
    let mut m = 1.0;
    while m * f0 <= SR / 2.0 {
        let c = (m * f0 * n as f64 / SR).round() as i64;
        for k in (c - 8)..=(c + 8) {
            if let Some(h) = usize::try_from(k).ok().and_then(|k| harmonic.get_mut(k)) {
                *h = true;
            }
        }
        m += 1.0;
    }
    let (mut p_h, mut p_a) = (0.0, 0.0);
    for (k, (c, is_h)) in buf.iter().zip(&harmonic).enumerate().take(half + 1).skip(1) {
        if *is_h {
            p_h += c.norm_sqr();
        } else if k > 8 {
            p_a += c.norm_sqr();
        }
    }
    10.0 * (p_a.max(1e-40) / p_h.max(1e-40)).log10()
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-20).log10()
}

#[test]
fn identity_table_is_transparent() {
    let mut s = Shaper::<2049>::new(&table(|x| x));
    let mut y = sine(1000.0, 0.5, 48000);
    s.process(&mut y);
    let steady = &y[24000..];
    let a1 = amp_at(steady, 1000.0);
    assert!((a1 - 0.5).abs() < 0.005, "{a1}");
    for h in [2000.0, 3000.0] {
        let d = db(amp_at(steady, h) / a1);
        assert!(d <= -70.0, "{h} Hz: {d} dB");
    }
}

#[test]
fn input_is_clamped() {
    let t = fold();
    for (big, one) in [(3.0f32, 1.0f32), (-3.0, -1.0)] {
        let (mut a, mut b) = (Shaper::<2049>::new(&t), Shaper::<2049>::new(&t));
        for i in 0..64 + 512 {
            let (ya, yb) = (a.tick(big), b.tick(one));
            if i >= 64 {
                assert!((ya - yb).abs() <= 1e-6, "{i}: {ya} vs {yb}");
            }
        }
    }
}

#[test]
fn fold_table_generates_harmonics_without_aliasing() {
    let mut s = Shaper::<2049>::new(&fold());
    let mut y = sine(440.0, 0.8, 48000);
    s.process(&mut y);
    let steady = &y[8192..];
    assert!(steady.iter().any(|v| v.abs() > 0.5), "no signal");
    let a = alias_db(steady, 440.0);
    assert!(a <= -70.0, "{a} dB");
}

fn fixture_array(src: &str, key: &str) -> Vec<f32> {
    let start = src
        .find(&format!("\"{key}\":["))
        .unwrap_or_else(|| panic!("no {key}"))
        + key.len()
        + 4;
    let end = start + src[start..].find(']').unwrap();
    src[start..end]
        .split(',')
        .map(|v| v.trim().parse::<f32>().unwrap())
        .collect()
}

fn max_err(input: &[f32], expected: &[f32], curve: &[f32]) -> f64 {
    let mut t = [0.0f32; 2049];
    t.copy_from_slice(curve);
    let mut s = Shaper::<2049>::new(&t);
    input
        .iter()
        .zip(expected)
        .map(|(x, e)| (s.tick(*x) - e).abs() as f64)
        .fold(0.0, f64::max)
}

/// Measured against Chrome (headless, `tools/record_webaudio_ref.mjs`): measured max abs error 3.35e-6 (sine) and 2.25e-6 (saw);
/// TOLERANCE is 2x the larger. The error is f32 rounding noise: the cascade is bit-for-bit the same algorithm.
#[test]
fn matches_webaudio_reference() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/webaudio_shaper_ref.json"
    ))
    .unwrap();
    let curve = fixture_array(&src, "curve_values");
    assert_eq!(curve.len(), 2049);
    const TOLERANCE: f64 = 6.7e-6;
    for (i, o) in [("sine_in", "sine_out"), ("saw_in", "saw_out")] {
        let (x, e) = (fixture_array(&src, i), fixture_array(&src, o));
        assert_eq!(x.len(), 4096);
        let err = max_err(&x, &e, &curve);
        assert!(err <= TOLERANCE, "{i}: max abs error {err} > {TOLERANCE}");
    }
}

#[test]
fn blockwise_equals_per_sample() {
    let x = sine(777.0, 0.9, 4096);
    assert_blockwise_eq(|| Shaper::<2049>::new(&fold()), &x, &[1, 7, 64, 333, 4096]);
}

#[test]
fn no_alloc_in_tick_process_reset() {
    let mut s = Shaper::<2049>::new(&fold());
    let mut buf = sine(440.0, 0.8, 512);
    let (_, n) = count_allocs(|| {
        s.process(&mut buf);
        s.tick(0.5);
        s.reset();
    });
    assert_eq!(n, 0);
}

#[test]
fn no_nan_and_non_finite_input_is_guarded() {
    let mut s = Shaper::<2049>::new(&fold());
    for x in sine(440.0, 0.8, 2000) {
        assert!(s.tick(x).is_finite());
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(s.tick(bad), 0.0);
    }
    assert_eq!(s.tick(0.0), 0.0); // state was cleared
    assert!(s.tick(0.5).is_finite());
}

#[test]
fn dc_is_zero_and_peak_within_one() {
    let mut s = Shaper::<2049>::new(&fold());
    let mut y = sine(440.0, 0.8, 48000);
    s.process(&mut y);
    let steady = &y[8192..];
    let n = (steady.len() / 109) * 109; // whole periods of 440 Hz at 48 kHz (109.09 samples): near-whole is enough
    let mean = steady[..n].iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    assert!(mean.abs() < 0.01, "{mean}");
    assert!(
        steady.iter().all(|v| v.abs() <= 1.0),
        "peak {}",
        steady.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
}

#[test]
fn reset_clears_state() {
    let mut s = Shaper::<2049>::new(&fold());
    for x in sine(440.0, 0.8, 500) {
        s.tick(x);
    }
    s.reset();
    assert_eq!(s.tick(0.0), 0.0);
    let mut fresh = Shaper::<2049>::new(&fold());
    let x = sine(300.0, 0.7, 300);
    for v in &x {
        assert_eq!(s.tick(*v).to_bits(), fresh.tick(*v).to_bits());
    }
}
