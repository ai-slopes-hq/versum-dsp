//! Measuring helpers shared by module tests and the `bench` crate. Amplitudes are linear and peak unless a name
//! says dB.

use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::TAU;

/// Hann-windowed single-bin DFT at `hz`, scaled so a sine of peak `a` reads `a`.
pub fn amp_at(s: &[f32], hz: f64, sr: f64) -> f64 {
    let (mut re, mut im, mut ws) = (0.0, 0.0, 0.0);
    for (i, v) in s.iter().enumerate() {
        let w = 0.5 - 0.5 * (TAU * i as f64 / s.len() as f64).cos();
        let a = TAU * hz * i as f64 / sr;
        re += *v as f64 * w * a.cos();
        im -= *v as f64 * w * a.sin();
        ws += w;
    }
    2.0 * (re * re + im * im).sqrt() / ws
}

/// The mean of `s`.
pub fn dc(s: &[f32]) -> f64 {
    s.iter().map(|v| *v as f64).sum::<f64>() / s.len().max(1) as f64
}

/// Energy outside the harmonics of `f0` over energy in them, in dB. Blackman–Harris window on the largest
/// power-of-two prefix `N ≤ len`; harmonic bins are `round(m·f0·N/sr) ± 8` for `m·f0 ≤ sr/2` (a harmonic right at
/// Nyquist, e.g. the 6th of 4 kHz at 48 kHz, is a harmonic), bins up to 8 (DC) are ignored.
pub fn alias_db(s: &[f32], f0: f64, sr: f64) -> f64 {
    if s.len() < 32 {
        return 0.0;
    }
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
    while m * f0 <= sr / 2.0 && f0 > 0.0 {
        let c = (m * f0 * n as f64 / sr).round() as i64;
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
    if p_a.is_nan() || p_h.is_nan() {
        return f64::NAN;
    }
    10.0 * (p_a.max(1e-40) / p_h.max(1e-40)).log10()
}
