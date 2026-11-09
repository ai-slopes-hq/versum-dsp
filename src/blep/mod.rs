#![deny(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
//! Residual ring for causal minBLEP / minBLAMP anti-aliasing. A voice renders the naive waveform and reports
//! every discontinuity (step) or kink (ramp) with its sub-sample position; the output is the naive value
//! `DELAY` samples ago plus `next()`. That is the wave through the minimum-phase kernel, whose delay is exactly
//! `DELAY` samples: smooth parts are delayed exactly, steps and kinks are band-limited.

mod table;
#[cfg(feature = "gen")]
pub mod gen;

use table::{FIR_BOX, FIR_R1, FIR_R2, FIR_R3, RAMP_RES, STEP_RES};
pub use table::KERNEL;

pub const TAPS: usize = 32;
pub const PHASES: usize = 64;
/// Delay of the kernel in whole samples; the tables are generated for it.
pub const DELAY: usize = 4;
/// Jumps and slope changes below this are numerical dust (the two sides of an event sampled `EPS` apart on a
/// continuous wave): below `f32` resolution of the ring, so not worth a 32-tap pass.
///
/// An absolute threshold, matched to the `f32` ring. At sub-hertz rates real kinks and the sine's third derivative
/// jump (about 1e-7 at 55 Hz) fall under it and are skipped; the effect is about −140 dB, inaudible. Revisit if the
/// precision targets rise.
const NEGLIGIBLE: f64 = 1e-7;

pub struct Residuals {
    ring: [f32; TAPS],
    head: usize,
}

#[allow(clippy::new_without_default, clippy::should_implement_trait)]
impl Residuals {
    pub const fn new() -> Self {
        Self { ring: [0.0; TAPS], head: 0 }
    }

    /// Value jump `dv` that happened `ago ∈ [0, 1)` samples before the current output sample.
    #[inline]
    pub fn step(&mut self, ago: f64, dv: f64) {
        if dv.abs() < NEGLIGIBLE {
            return;
        }
        self.add(&STEP_RES, ago, dv);
        // `u(t) − u(t − DELAY)`: the jump is not in the delayed naive value yet
        for k in 0..DELAY {
            if let Some(cell) = self.ring.get_mut((self.head + k) % TAPS) {
                *cell += dv as f32;
            }
        }
    }

    /// Slope change `ds` (per sample) that happened `ago ∈ [0, 1)` samples before the current output sample.
    #[inline]
    pub fn ramp(&mut self, ago: f64, ds: f64) {
        if ds.abs() < NEGLIGIBLE {
            return;
        }
        self.add(&RAMP_RES, ago, ds);
    }

    /// Jumps `d` of value, slope, curvature and its rate (per sample⁰…³) that happened `ago` before the current
    /// output sample, on a wave whose naive samples go through the FIR `KERNEL` instead of the delay.
    #[inline]
    pub fn fir_jumps(&mut self, ago: f64, d: [f64; 4]) {
        let [d0, d1, d2, d3] = d;
        if d0.abs() >= NEGLIGIBLE {
            self.add(&STEP_RES, ago, d0);
            // `1 − Σ_{j≤k} KERNEL[j]`: what the FIR has not yet passed of the jump, whole cells
            for (k, b) in FIR_BOX.iter().enumerate() {
                if let Some(cell) = self.ring.get_mut((self.head + k) % TAPS) {
                    *cell += (d0 * f64::from(*b)) as f32;
                }
            }
        }
        for (table, amp) in [(&FIR_R1, d1), (&FIR_R2, d2), (&FIR_R3, d3)] {
            if amp.abs() >= NEGLIGIBLE {
                self.add(table, ago, amp);
            }
        }
    }

    /// Residual of the current sample; clears its cell and advances the ring.
    #[inline]
    pub fn next(&mut self) -> f32 {
        let cell = self.ring.get_mut(self.head).map(std::mem::take).unwrap_or(0.0);
        self.head = (self.head + 1) % TAPS;
        cell
    }

    #[inline]
    pub fn clear(&mut self) {
        self.ring = [0.0; TAPS];
        self.head = 0;
    }

    #[inline]
    fn add(&mut self, table: &[f32], ago: f64, amp: f64) {
        let ago = if ago.is_finite() { ago.clamp(0.0, 0.999_999) } else { 0.0 };
        // tap `k` reads the table at `(k + ago)·PHASES`: the same fraction `f` of every `PHASES`-th pair
        let pos = ago * PHASES as f64;
        let i = pos as usize;
        let f = (pos - i as f64) as f32;
        let amp = amp as f32;
        let (lo, hi) = (table.iter().skip(i).step_by(PHASES), table.iter().skip(i + 1).step_by(PHASES));
        for (k, (a, b)) in lo.zip(hi).take(TAPS).enumerate() {
            if let Some(cell) = self.ring.get_mut((self.head + k) % TAPS) {
                *cell += amp * (a + (b - a) * f);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
    use super::*;
    use crate::measure;
    #[cfg(feature = "gen")]
    use super::gen as blep_gen;

    const SR: f64 = 48000.0;
    const N: usize = 65536;

    #[test]
    fn tables_have_expected_shape() {
        assert!(STEP_RES[0] < -0.9, "{}", STEP_RES[0]);
        assert_eq!(STEP_RES[2048], 0.0);
        assert!(RAMP_RES[2048].abs() < 1e-6, "{}", RAMP_RES[2048]);
        assert!(RAMP_RES[0].abs() < 1e-6, "{}", RAMP_RES[0]);
        for i in 2048 - 64..=2048 {
            assert!(STEP_RES[i].abs() < 1e-3, "step[{i}] = {}", STEP_RES[i]);
            assert!(RAMP_RES[i].abs() < 1e-3, "ramp[{i}] = {}", RAMP_RES[i]);
        }
    }

    #[cfg(feature = "gen")]
    #[test]
    fn fir_tables_end_at_zero() {
        // the FIR's moments match the kernel's, so its residuals need no tail beyond the ring
        let (boxes, rs) = blep_gen::fir_tables();
        assert!(boxes[TAPS - 1].abs() < 1e-9, "box end {}", boxes[TAPS - 1]);
        for (n, r) in rs.iter().enumerate() {
            assert_eq!(r[0], 0.0, "r{}", n + 1);
            let raw_end = r[2047] + (r[2047] - r[2046]);
            assert!(raw_end.abs() < 1e-6, "r{} ends at {raw_end}", n + 1);
            for v in &r[2048 - 64..] {
                assert!(v.abs() < 1e-4, "r{} tail {v}", n + 1);
            }
        }
    }

    #[cfg(feature = "gen")]
    #[test]
    fn committed_table_matches_generator() {
        let (step, ramp, delay) = blep_gen::tables();
        assert!((delay - DELAY as f64).abs() < 1e-9, "{delay} vs {DELAY}");
        for (j, h) in blep_gen::kernel().iter().enumerate() {
            assert!((h - KERNEL[j]).abs() < 1e-12, "kernel[{j}]: {h} vs {}", KERNEL[j]);
        }
        let (boxes, [r1, r2, r3]) = blep_gen::fir_tables();
        for (j, b) in boxes.iter().enumerate() {
            assert!((b - FIR_BOX[j] as f64).abs() < 1e-6, "box[{j}]");
        }
        for (name, gen, table) in [("r1", &r1, &FIR_R1), ("r2", &r2, &FIR_R2), ("r3", &r3, &FIR_R3)] {
            for i in 0..2049 {
                assert!((gen[i] - table[i] as f64).abs() < 1e-6, "{name}[{i}]: {} vs {}", gen[i], table[i]);
            }
        }
        assert_eq!((step.len(), ramp.len()), (2049, 2049));
        for i in 0..2049 {
            assert!((step[i] - STEP_RES[i] as f64).abs() < 1e-6, "step[{i}]: {} vs {}", step[i], STEP_RES[i]);
            assert!((ramp[i] - RAMP_RES[i] as f64).abs() < 1e-6, "ramp[{i}]: {} vs {}", ramp[i], RAMP_RES[i]);
        }
    }

    /// The naive value `DELAY` samples ago.
    struct Late([f64; DELAY], usize);

    impl Late {
        fn push(&mut self, v: f64) -> f64 {
            let old = std::mem::replace(&mut self.0[self.1], v);
            self.1 = (self.1 + 1) % DELAY;
            old
        }
    }

    fn saw(f0: f64) -> Vec<f32> {
        let inc = f0 / SR;
        let mut res = Residuals::new();
        let mut late = Late([-1.0; DELAY], 0);
        let mut p = 0.0f64;
        (0..N)
            .map(|_| {
                p += inc;
                if p >= 1.0 {
                    p -= 1.0;
                    res.step(p / inc, -2.0);
                }
                late.push(2.0 * p - 1.0) as f32 + res.next()
            })
            .collect()
    }

    fn tri(f0: f64) -> Vec<f32> {
        let inc = f0 / SR;
        let slope = 4.0 * inc;
        let mut res = Residuals::new();
        let mut late = Late([-1.0; DELAY], 0);
        let mut p = 0.0f64;
        (0..N)
            .map(|_| {
                let prev = p;
                p += inc;
                if prev < 0.5 && p >= 0.5 {
                    res.ramp((p - 0.5) / inc, -2.0 * slope);
                }
                if p >= 1.0 {
                    p -= 1.0;
                    res.ramp(p / inc, 2.0 * slope);
                }
                late.push(1.0 - 4.0 * (p - 0.5).abs()) as f32 + res.next()
            })
            .collect()
    }

    #[test]
    fn saw_with_residuals_is_clean() {
        for f0 in [4000.0, 4321.0] {
            let s = saw(f0);
            let db = measure::alias_db(&s, f0, SR);
            assert!(db <= -80.0, "{f0} Hz: alias {db} dB");
            assert!(measure::dc(&s).abs() <= 1e-3, "{f0} Hz: dc {}", measure::dc(&s));
            let h1 = measure::amp_at(&s, f0, SR);
            let ideal = 2.0 / std::f64::consts::PI;
            assert!((h1 / ideal - 1.0).abs() < 0.01, "{f0} Hz: fundamental {h1} vs {ideal}");
        }
    }

    #[test]
    fn tri_with_residuals_is_clean() {
        for f0 in [4000.0, 4321.0] {
            let s = tri(f0);
            let db = measure::alias_db(&s, f0, SR);
            let dc = measure::dc(&s);
            assert!(db <= -80.0, "{f0} Hz: alias {db} dB");
            assert!(dc.abs() <= 1e-3, "{f0} Hz: dc {dc}");
            let h1 = measure::amp_at(&s, f0, SR);
            let ideal = 8.0 / (std::f64::consts::PI * std::f64::consts::PI);
            assert!((h1 / ideal - 1.0).abs() < 0.01, "{f0} Hz: fundamental {h1} vs {ideal}");
        }
    }

    #[test]
    fn residuals_clear_and_ring_advance() {
        // A unit jump half a sample ago, against a naive value that shows it only DELAY samples later.
        let mut r = Residuals::new();
        r.step(0.5, 1.0);
        let cells: Vec<f32> = (0..TAPS).map(|_| r.next()).collect();
        assert!(cells[0].abs() < 1e-3, "the kernel has not started yet: {}", cells[0]);
        assert!(cells[3] > 0.0 && cells[3] < 1.0, "rising: {}", cells[3]);
        let area: f32 = cells.iter().sum();
        assert!(area.abs() < 1e-2, "area {area}: the kernel's delay is DELAY");
        r.step(0.5, 1.0);
        r.clear();
        for _ in 0..TAPS {
            assert_eq!(r.next(), 0.0);
        }
    }
}
