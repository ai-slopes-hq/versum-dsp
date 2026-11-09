//! Low-pass gate: vactrol conductance, vactrol light state and the trapezoidal (bilinear) circuit after
//! Parker & D'Angelo, plus `Lpg` (Gate/Strike excitation) built on them.

pub const C1: f64 = 1e-9;
pub const C2: f64 = 220e-12;
pub const C3_LPF: f64 = 4.7e-9;
pub const RA_VCA: f64 = 5e3;
pub const RA_HIGH: f64 = 5e6;
pub const I_MAX: f64 = 0.04;
pub const RF_A: f64 = 3.464;
pub const RF_B: f64 = 1136.212;
pub const RF_EXP: f64 = 1.4;
pub const A_NORM: f64 = 0.6;
pub const TAU_RISE: f64 = 0.012;
pub const K_FALL: f64 = 1.0;
pub const STRIKE_MS: f64 = 6.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Vca,
    Lpf,
    Combo,
}

impl Mode {
    /// 0 is VCA, 1 is LPF, anything else is Combo.
    pub fn from_option(v: u32) -> Mode {
        match v {
            0 => Mode::Vca,
            1 => Mode::Lpf,
            _ => Mode::Combo,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Topology {
    pub ra: f64,
    pub c3: f64,
    pub a: f64,
    pub norm: f64,
}

pub fn topology(mode: Mode) -> Topology {
    let ra = if mode == Mode::Vca { RA_VCA } else { RA_HIGH };
    let c3 = if mode == Mode::Lpf { C3_LPF } else { 0.0 };
    let a = if mode == Mode::Lpf {
        let a_max = (2.0 * C1 + (C2 + c3) * (1.0 + 1.0 / (g_max() * ra))) / c3;
        A_NORM * a_max
    } else {
        0.0
    };
    Topology { ra, c3, a, norm: 1.0 + 2.0 / (g_max() * ra) }
}

/// Vactrol conductance for light `l` (0..1); exactly 0 in the dark.
pub fn conductance(l: f64) -> f64 {
    if l <= 0.0 {
        return 0.0;
    }
    let p = (l.min(1.0) * I_MAX).powf(RF_EXP);
    p / (RF_A + RF_B * p)
}

pub fn g_max() -> f64 {
    conductance(1.0)
}

/// `|H(j omega)|` of the analog circuit, without normalisation.
pub fn analog_gain(mode: Mode, g: f64, omega: f64) -> f64 {
    if g <= 0.0 {
        return 0.0;
    }
    let t = topology(mode);
    let d0 = g * g + 2.0 * g / t.ra;
    let d1 = g * (2.0 * C1 + C2 - t.c3 * (t.a - 1.0)) + (C2 + t.c3) / t.ra;
    let d2 = C1 * (C2 + t.c3);
    let re = d0 - omega * omega * d2;
    let im = omega * d1;
    g * g / (re * re + im * im).sqrt()
}

/// Pre-warped analog frequency that the bilinear transform maps onto `f`.
pub fn warped_omega(f: f64, sr: f64) -> f64 {
    2.0 * sr * (std::f64::consts::PI * f / sr).tan()
}

#[inline]
fn flush(s: f64) -> f64 {
    if s.abs() < 1e-30 {
        0.0
    } else {
        s
    }
}

/// The LPG circuit, trapezoidal integration solved in closed form (no delay in the loop).
///
/// At `g = 0` the Vx node floats and keeps its charge, as in the analog circuit; it decays once the gate reopens.
pub struct Circuit {
    t: f64,
    mode: Mode,
    topo: Topology,
    s1: f64,
    s2: f64,
    s3: f64,
}

impl Circuit {
    pub fn new(mode: Mode, sr: f64) -> Circuit {
        Circuit { t: 1.0 / sr, mode, topo: topology(mode), s1: 0.0, s2: 0.0, s3: 0.0 }
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
        self.s3 = 0.0;
    }

    /// Switches topology; states are kept.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.topo = topology(mode);
    }

    pub fn step(&mut self, vin: f64, g: f64) -> f64 {
        let Topology { ra, c3, a, norm } = self.topo;
        let t = self.t;
        let a11 = 2.0 * (C2 + c3) / t + 2.0 * g;
        let a12 = -(g + 2.0 * c3 * a / t);
        let a21 = -g;
        let a22 = 2.0 * C1 / t + g + 1.0 / ra;
        let b1 = g * vin + (2.0 * C2 / t) * self.s2 - self.s3;
        let b2 = (2.0 * C1 / t) * self.s1;
        let det = a11 * a22 - a12 * a21;
        let vout = (a11 * b2 - a21 * b1) / det;
        let vx = (b1 - a12 * vout) / a11;
        let d = (2.0 * c3 / t) * (a * vout - vx);
        self.s2 = flush(2.0 * vx - self.s2);
        self.s1 = flush(2.0 * vout - self.s1);
        self.s3 = flush(2.0 * d - self.s3);
        vout * norm
    }
}

/// Vactrol light state: fast rise, slow level-dependent fall.
pub struct Vactrol {
    sr: f64,
    exact: bool,
    l: f64,
    c_rise: f64,
}

impl Vactrol {
    pub fn new(sr: f64, exact: bool) -> Vactrol {
        Vactrol { sr, exact, l: 0.0, c_rise: 1.0 - (-1.0 / (TAU_RISE * sr)).exp() }
    }

    pub fn reset(&mut self) {
        self.l = 0.0;
    }

    /// One sample towards drive `u`; returns the new light level.
    pub fn step(&mut self, u: f64, tau_fall: f64) -> f64 {
        let c = if u > self.l {
            self.c_rise
        } else {
            let x = 1.0 / (tau_fall * (1.0 + K_FALL * (1.0 - self.l)) * self.sr);
            if self.exact {
                1.0 - (-x).exp()
            } else {
                x / (1.0 + x / 2.0)
            }
        };
        self.l += (u - self.l) * c;
        if u == 0.0 && self.l < 1e-12 {
            self.l = 0.0;
        }
        self.l
    }

    pub fn level(&self) -> f64 {
        self.l
    }
}

/// Per-block options of `Lpg::tick`.
#[derive(Clone, Copy, Debug)]
pub struct LpgOpts {
    pub mode: Mode,
    pub strike: bool,
}

/// One sample of input for `Lpg::tick`. `vel: None` means velocity is not connected (1.0).
#[derive(Clone, Copy, Debug)]
pub struct LpgIn {
    pub audio: f32,
    pub excite: f32,
    pub vel: Option<f32>,
    pub decay: f32,
    pub level: f32,
}

/// Low-pass gate with Gate/Strike excitation.
pub struct Lpg {
    mode: Mode,
    circuit: Circuit,
    vactrol: Vactrol,
    strike_len: u32,
    prev: f32,
    left: u32,
    height: f64,
}

impl Lpg {
    pub fn new(sr: f64) -> Lpg {
        Lpg {
            mode: Mode::Combo,
            circuit: Circuit::new(Mode::Combo, sr),
            vactrol: Vactrol::new(sr, false),
            strike_len: (STRIKE_MS * 1e-3 * sr).round() as u32,
            prev: 0.0,
            left: 0,
            height: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.vactrol.reset();
        self.circuit.reset();
        self.prev = 0.0;
        self.left = 0;
    }

    #[inline]
    pub fn tick(&mut self, o: &LpgOpts, i: &LpgIn) -> f32 {
        if o.mode != self.mode {
            self.mode = o.mode;
            self.circuit.set_mode(o.mode);
        }
        let vel = match i.vel {
            Some(v) => f64::from(v).clamp(0.0, 1.0),
            None => 1.0,
        };
        let x = i.excite;
        let e = if o.strike {
            if self.prev <= 0.5 && x > 0.5 {
                self.left = self.strike_len;
                self.height = vel;
            }
            if self.left > 0 {
                self.left -= 1;
                self.height
            } else {
                0.0
            }
        } else {
            f64::from(x).clamp(0.0, 1.0) * vel
        };
        self.prev = x;
        let level = f64::from(i.level);
        let u = (level + e).clamp(0.0, 1.0);
        let l = self.vactrol.step(u, f64::from(i.decay));
        self.circuit.step(f64::from(i.audio), conductance(l)) as f32
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
    use super::*;
    use crate::measure::{amp_at, db};
    use crate::Rng;
    use std::f64::consts::TAU;

    const SR: f64 = 48000.0;
    const MODES: [Mode; 3] = [Mode::Vca, Mode::Lpf, Mode::Combo];

    fn square(i: usize) -> f64 {
        if (i * 2000 / 48000).is_multiple_of(2) {
            64.0
        } else {
            -64.0
        }
    }

    #[test]
    fn conductance_endpoints() {
        assert_eq!(conductance(0.0), 0.0);
        let r = 1.0 / g_max();
        assert!((1449.0..=1452.0).contains(&r), "1/g_max = {r}");
        let mut prev = 0.0;
        for i in 1..=100 {
            let g = conductance(i as f64 * 0.01);
            assert!(g > prev, "not monotonic at {i}");
            prev = g;
        }
    }

    #[test]
    fn topology_values() {
        assert!((topology(Mode::Vca).norm - 1.5801).abs() < 1e-3);
        assert!((topology(Mode::Combo).norm - 1.00058).abs() < 1e-5);
        assert!((topology(Mode::Lpf).norm - 1.00058).abs() < 1e-5);
        assert!((topology(Mode::Lpf).a - 0.8836).abs() < 1e-3);
        assert_eq!(topology(Mode::Vca).c3, 0.0);
        assert_eq!(topology(Mode::Combo).c3, 0.0);
        assert_eq!(topology(Mode::Lpf).c3, C3_LPF);
    }

    #[test]
    fn dc_gain_is_one_at_full_light() {
        for mode in MODES {
            let mut c = Circuit::new(mode, SR);
            let mut y = 0.0;
            for _ in 0..48000 {
                y = c.step(0.5, g_max());
            }
            assert!((y - 0.5).abs() < 1e-4, "{mode:?}: {y}");
        }
    }

    #[test]
    fn circuit_matches_bilinear_response() {
        let mut worst: f64 = 0.0;
        for mode in MODES {
            for l in [1.0, 0.3, 0.1, 0.03] {
                for f in [100.0, 1000.0, 6000.0, 12000.0] {
                    let g = conductance(l);
                    let mut c = Circuit::new(mode, SR);
                    let w = TAU * f / SR;
                    for i in 0..24000 {
                        c.step(0.5 * (w * i as f64).sin(), g);
                    }
                    let out: Vec<f32> = (24000..24000 + 32768).map(|i| c.step(0.5 * (w * i as f64).sin(), g) as f32).collect();
                    let got = db(amp_at(&out, f, SR) / 0.5);
                    let want = db(topology(mode).norm * analog_gain(mode, g, warped_omega(f, SR)));
                    let e = (got - want).abs();
                    worst = worst.max(e);
                    assert!(e < 0.05, "{mode:?} L={l} f={f}: got {got} want {want}");
                }
            }
        }
        eprintln!("worst bilinear error: {worst} dB");
    }

    #[test]
    fn closed_circuit_discharges_to_exact_zero() {
        for mode in MODES {
            let mut c = Circuit::new(mode, SR);
            for i in 0..2000 {
                c.step((TAU * 1000.0 * i as f64 / SR).sin(), g_max());
            }
            let out: Vec<f64> = (0..48000).map(|i| c.step(square(i), 0.0)).collect();
            assert!(out.iter().all(|v| v.is_finite()), "{mode:?}");
            assert!(out[47000..].iter().all(|v| *v == 0.0), "{mode:?}: {:?}", &out[47000..47004]);
        }
    }

    #[test]
    fn lpf_stable_under_jumping_g() {
        let mut rng = Rng::new(7);
        let mut rng = move || rng.next_f64() as f32;
        let mut c = Circuit::new(Mode::Lpf, SR);
        for i in 0..96000 {
            let g = match (rng() * 3.0) as u32 {
                0 => 0.0,
                1 => 1e-12,
                _ => g_max() * rng() as f64,
            };
            let y = c.step(square(i), g);
            assert!(y.is_finite() && y.abs() <= 256.0, "i={i} y={y}");
        }
    }

    #[test]
    fn reopen_after_dark_no_burst() {
        // One strike: light 1 for 0.1 s, fall to exactly 0, then 0.5 s of dark. Returns |out| per sample.
        // The sine phase restarts at each strike (the peak itself depends on the phase at opening, about 2 %).
        let strike = |c: &mut Circuit, v: &mut Vactrol| {
            let mut out = Vec::new();
            let mut run = |c: &mut Circuit, v: &mut Vactrol, u: f64| {
                let l = v.step(u, 0.02);
                let n = out.len();
                out.push(c.step((TAU * 1000.0 * n as f64 / SR).sin(), conductance(l)).abs());
            };
            for _ in 0..4800 {
                run(c, v, 1.0);
            }
            let mut guard = 0;
            while v.level() != 0.0 {
                run(c, v, 0.0);
                guard += 1;
                assert!(guard < 20 * 48000, "never reached zero");
            }
            for _ in 0..24000 {
                run(c, v, 0.0);
            }
            out
        };
        let peak = |o: &[f64]| o.iter().fold(0.0f64, |m, x| m.max(*x));
        let mut c = Circuit::new(Mode::Lpf, SR);
        let mut v = Vactrol::new(SR, false);
        let first = strike(&mut c, &mut v);
        eprintln!("trapped after dark: s1={:e} s2={:e} s3={:e}", c.s1, c.s2, c.s3);
        let second = strike(&mut c, &mut v);
        let fresh = strike(&mut Circuit::new(Mode::Lpf, SR), &mut Vactrol::new(SR, false));
        let (p1, p2, pf) = (peak(&first), peak(&second), peak(&fresh));
        assert!(p2 <= p1 * 1.01, "p1={p1} p2={p2}");
        let diff = second.len().min(fresh.len());
        let worst = second.iter().zip(&fresh).take(diff).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("max |second - fresh| = {worst:e}, peak fresh = {pf}");
        assert!(second.len() == fresh.len() && worst <= 0.01 * pf, "second strike differs from a fresh circuit: {worst} vs peak {pf}");
    }

    #[test]
    fn mode_out_of_range_is_combo() {
        assert_eq!(Mode::from_option(7), Mode::Combo);
        assert_eq!(Mode::from_option(0), Mode::Vca);
        assert_eq!(Mode::from_option(1), Mode::Lpf);
        assert_eq!(Mode::from_option(2), Mode::Combo);
    }

    #[test]
    fn vactrol_rise_time() {
        let mut v = Vactrol::new(SR, true);
        for _ in 0..(0.012 * SR).round() as usize {
            v.step(1.0, 0.25);
        }
        assert!((v.level() - (1.0 - (-1.0f64).exp())).abs() < 1e-3, "{}", v.level());
    }

    #[test]
    fn vactrol_fall_follows_formula() {
        let tau = 0.25;
        for (l0, l1) in [(1.0f64, 0.5f64), (0.1, 0.05)] {
            // Rise to full light, then fall to l0 before timing the fall to l1.
            let mut v = Vactrol::new(SR, true);
            for _ in 0..(SR * 0.5) as usize {
                v.step(1.0, tau);
            }
            while v.level() > l0 {
                v.step(0.0, tau);
            }
            let l_start = v.level();
            let mut steps = 0.0;
            while v.level() > l1 {
                v.step(0.0, tau);
                steps += 1.0;
            }
            let want = SR * tau * ((1.0 + K_FALL) * (l_start / l1).ln() - K_FALL * (l_start - l1));
            assert!((steps - want).abs() / want < 0.01, "{l0}->{l1}: {steps} vs {want}");
        }
    }

    #[test]
    fn vactrol_snaps_to_zero() {
        let mut v = Vactrol::new(SR, false);
        for _ in 0..(SR * 0.5) as usize {
            v.step(1.0, 0.02);
        }
        for _ in 0..(3.0 * SR) as usize {
            v.step(0.0, 0.02);
        }
        assert_eq!(v.level(), 0.0);
        let mut v = Vactrol::new(SR, false);
        for _ in 0..(SR * 0.5) as usize {
            v.step(1.0, 0.02);
        }
        for _ in 0..(3.0 * SR) as usize {
            v.step(1e-9, 0.02);
        }
        assert!(v.level() > 0.0);
    }

    #[test]
    fn approx_c_close_to_exact() {
        let mut worst: f64 = 0.0;
        for tau in [0.02, 2.0] {
            let mut a = Vactrol::new(SR, true);
            let mut b = Vactrol::new(SR, false);
            for i in 0..(2.0 * SR) as usize {
                let u = if ((i as f64 / (0.05 * SR)) as usize).is_multiple_of(2) { 1.0 } else { 0.0 };
                worst = worst.max((a.step(u, tau) - b.step(u, tau)).abs());
            }
        }
        eprintln!("worst approx-c error: {worst}");
        assert!(worst < 1e-4, "{worst}");
    }

    #[test]
    fn rng_matches_old_mulberry32() {
        let mut a: u32 = 5;
        let mut old = move || {
            a = a.wrapping_add(0x6D2B79F5);
            let mut t = (a ^ (a >> 15)).wrapping_mul(a | 1);
            t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61))) ^ t;
            ((t ^ (t >> 14)) as f64 / 4294967296.0) as f32
        };
        let mut r = Rng::new(5);
        for i in 0..1000 {
            assert_eq!(old().to_bits(), (r.next_f64() as f32).to_bits(), "draw {i}");
        }
    }

    #[test]
    fn tick_matches_manual_chain() {
        let mut rng = Rng::new(3);
        let mut lpg = Lpg::new(SR);
        let mut v = Vactrol::new(SR, false);
        let mut c = Circuit::new(Mode::Combo, SR);
        let o = LpgOpts { mode: Mode::Combo, strike: false };
        for i in 0..1000 {
            let audio = rng.next_f64() as f32 * 2.0 - 1.0;
            let excite = if (i / 100) % 2 == 0 { 1.0 } else { 0.0 };
            let inp = LpgIn { audio, excite, vel: None, decay: 0.1, level: 0.0 };
            let got = lpg.tick(&o, &inp);
            let u = (0.0f64 + f64::from(excite).clamp(0.0, 1.0) * 1.0).clamp(0.0, 1.0);
            let l = v.step(u, f64::from(0.1f32));
            let want = c.step(f64::from(audio), conductance(l)) as f32;
            assert_eq!(got.to_bits(), want.to_bits(), "sample {i}");
        }
    }

    #[test]
    fn strike_pulse_length() {
        let mut lpg = Lpg::new(SR);
        let mut v = Vactrol::new(SR, false);
        let o = LpgOpts { mode: Mode::Combo, strike: true };
        let n = (STRIKE_MS * 1e-3 * SR).round() as usize;
        // Excite goes high at sample 0 and stays high; the pulse must still last exactly n samples.
        let mut c = Circuit::new(Mode::Combo, SR);
        for i in 0..(n + 40) {
            let inp = LpgIn { audio: 0.5, excite: 1.0, vel: Some(0.8), decay: 0.3, level: 0.0 };
            let got = lpg.tick(&o, &inp);
            let u = if i < n { f64::from(0.8f32) } else { 0.0 };
            let l = v.step(u, f64::from(0.3f32));
            let want = c.step(0.5, conductance(l)) as f32;
            assert_eq!(got.to_bits(), want.to_bits(), "sample {i}");
        }
    }
}
