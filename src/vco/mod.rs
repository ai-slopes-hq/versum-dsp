mod voice;

use crate::blep::{Residuals, DELAY, KERNEL, TAPS};
use voice::{sine_kink, Shape, SyncMode, Voice};
pub use voice::{OUTS, SINE, TRI};

const _: () = assert!(KERNEL.len() == TAPS);

/// The last `TAPS` samples of a signal and their dot product with `KERNEL`: the FIR the sine's smooth part goes
/// through. Every sample is stored twice (at `i` and `i + TAPS`), so the history is one contiguous slice, oldest
/// first, whatever the position of the newest.
struct Smooth {
    buf: [f64; 2 * TAPS],
    newest: usize,
    /// `KERNEL` reversed, to pair with that slice.
    kernel_rev: [f64; TAPS],
}

impl Smooth {
    fn new() -> Self {
        Self { buf: [0.0; 2 * TAPS], newest: 0, kernel_rev: std::array::from_fn(|j| KERNEL.get(TAPS - 1 - j).copied().unwrap_or(0.0)) }
    }

    /// The history as if the signal had always been `x`.
    fn fill(&mut self, x: f64) {
        self.buf = [x; 2 * TAPS];
        self.newest = 0;
    }

    #[inline]
    fn push(&mut self, x: f64) {
        self.newest = (self.newest + 1) % TAPS;
        for i in [self.newest, self.newest + TAPS] {
            if let Some(cell) = self.buf.get_mut(i) {
                *cell = x;
            }
        }
    }

    /// `Σ KERNEL[m]·x[newest − m]`
    #[inline]
    fn fir(&self) -> f64 {
        // four independent sums, so the loop is not one long chain of dependent additions
        self.buf.get(self.newest + 1..self.newest + 1 + TAPS).map_or(0.0, |w| {
            let mut acc = [0.0f64; 4];
            for (w, h) in w.as_chunks::<4>().0.iter().zip(self.kernel_rev.as_chunks::<4>().0) {
                for ((a, w), h) in acc.iter_mut().zip(w).zip(h) {
                    *a += w * h;
                }
            }
            (acc[0] + acc[1]) + (acc[2] + acc[3])
        })
    }
}

const C4_HZ: f64 = 261.6256;
/// Largest `|f|` relative to the sample rate.
const F_MAX: f64 = 0.45;

/// One triangle-core voice with five outputs: the naive wave through the minimum-phase kernel. `tri`, `saw`,
/// `pulse` and `sub` are piecewise linear: the naive value `DELAY` samples ago (the kernel's delay) plus the
/// minBLEP/minBLAMP residuals of their events. `sine` is curved, and a pure delay of its smooth part next to
/// kernel-filtered kinks would put the two out of phase (at 4 kHz the 3rd harmonic came out 10× too strong).
/// So `sine = c·tri + (sine − c·tri)`: the first term has all its kinks (`c` = kink ratio), the remainder none,
/// and it goes through `KERNEL`, the same kernel as an FIR.
///
/// Unison runs `1 + unison` cores; the output is their mean, so the naive values are averaged once and the
/// events of every core go into the rings with the weight `1/N`.
pub struct Vco {
    voices: [Voice; 3],
    /// Sync input of the previous sample; NaN before the first, so a gate already high
    /// when the module starts is not an edge.
    sync_prev: f64,
    /// Residual rings: the sine's remainder (relative to the FIR), then `tri`, `saw`, `pulse`, `sub` (relative to
    /// the delay).
    res: [Residuals; OUTS],
    /// Naive values of the last `DELAY` samples, oldest at `head`.
    late: [[f64; OUTS]; DELAY],
    head: usize,
    /// Naive `sine − c·tri` of the last `TAPS` samples.
    smooth: Smooth,
    kink: f64,
    sr: f64,
    /// Pitch (in octaves) of the last sample and its `exp2`: a held pitch does not pay for `exp2` again.
    pitch_memo: (f64, f64),
}

impl Vco {
    pub fn new() -> Self {
        let mut vco = Self {
            voices: std::array::from_fn(|_| Voice::new(0.0)),
            sync_prev: f64::NAN,
            res: std::array::from_fn(|_| Residuals::new()),
            late: [[0.0; OUTS]; DELAY],
            head: 0,
            smooth: Smooth::new(),
            kink: sine_kink(),
            sr: 48000.0,
            pitch_memo: (f64::NAN, f64::NAN),
        };
        vco.restart();
        vco
    }

    /// Phase 0 with a history as if the voice had always stood there.
    fn restart(&mut self) {
        self.voices = std::array::from_fn(|_| Voice::new(0.0));
        self.sync_prev = f64::NAN;
        let v = Voice::new(0.0).values(&Shape::new(0.5, false));
        self.late = [v; DELAY];
        self.head = 0;
        let r = v.get(SINE).copied().unwrap_or(0.0) - self.kink * v.get(TRI).copied().unwrap_or(0.0);
        self.smooth.fill(r);
        for r in &mut self.res {
            r.clear();
        }
    }
}

/// Per-instance switches (the module's options).
#[derive(Clone, Copy, Debug, Default)]
pub struct VcoOpts {
    pub sub2: bool,
    /// Extra cores, 0..=2 (larger values are clamped to 2).
    pub unison: usize,
    pub exp_fm: bool,
    pub reverse: bool,
}

/// One sample of inputs and params.
#[derive(Clone, Copy, Debug, Default)]
pub struct VcoIn {
    pub pitch_cv: f32,
    pub fm: f32,
    pub sync: f32,
    pub coarse: f32,
    pub fine: f32,
    pub fm_depth: f32,
    pub pw: f32,
    pub detune: f32,
}

impl Default for Vco {
    fn default() -> Self {
        Self::new()
    }
}

impl Vco {
    /// Sets the sample rate without restarting the core.
    pub fn set_sample_rate(&mut self, sr: f64) {
        self.sr = sr;
    }

    pub fn reset(&mut self) {
        self.restart();
    }

    /// One sample: `sine, tri, saw, pulse, sub`.
    #[inline]
    pub fn tick(&mut self, o: &VcoOpts, i: &VcoIn) -> [f64; OUTS] {
        let (sub2, exp, reverse) = (o.sub2, o.exp_fm, o.reverse);
        let n = 1 + o.unison.min(2);
        let semis = f64::from(i.coarse) + f64::from(i.fine);
        let oct = f64::from(i.pitch_cv) + semis / 12.0;
        if oct.to_bits() != self.pitch_memo.0.to_bits() {
            self.pitch_memo = (oct, oct.exp2());
        }
        let f0 = C4_HZ * self.pitch_memo.1;
        let x = f64::from(i.fm_depth) * 4.0 * f64::from(i.fm);
        let f = if exp { f0 * x.exp2() } else { f0 * (1.0 + x) };
        let f = if f.is_finite() { f.clamp(-F_MAX * self.sr, F_MAX * self.sr) } else { 0.0 };
        let shape = Shape::new(f64::from(i.pw), sub2);
        let res = &mut self.res;
        let inc = f / self.sr;
        let sync_in = f64::from(i.sync);
        let edge = (self.sync_prev < 0.5 && sync_in >= 0.5).then(|| (0.5 - self.sync_prev) / (sync_in - self.sync_prev));
        self.sync_prev = sync_in;
        let sync = edge.map(|t| (t, if reverse { SyncMode::Reverse } else { SyncMode::Hard }));
        let detune = f64::from(i.detune);
        let w = 1.0 / n as f64;
        let mut now = [0.0; OUTS];
        for (v, voice) in self.voices.iter_mut().take(n).enumerate() {
            // two cores at ±detune/2, three at −detune, 0, +detune
            let cents = match (n, v) {
                (2, 0) => -0.5 * detune,
                (2, _) => 0.5 * detune,
                (3, 0) => -detune,
                (3, 2) => detune,
                _ => 0.0,
            };
            let inc = if cents == 0.0 { inc } else { (inc * (cents / 1200.0).exp2()).clamp(-F_MAX, F_MAX) };
            voice.advance(inc, &shape, sync, &mut |ago, dv, ds, dr| {
                let scale = |x: [f64; OUTS]| x.map(|x| x * w);
                let (dv, ds) = (scale(dv), scale(ds));
                let mut rings = res.iter_mut();
                if let Some(r) = rings.next() {
                    r.fir_jumps(ago, dr.map(|x| x * w));
                }
                for ((r, dv), ds) in rings.zip(dv.into_iter().skip(1)).zip(ds.into_iter().skip(1)) {
                    r.step(ago, dv);
                    r.ramp(ago, ds);
                }
            });
            for (m, x) in now.iter_mut().zip(voice.values(&shape)) {
                *m += w * x;
            }
        }
        let late = self.late.get_mut(self.head).map(|l| std::mem::replace(l, now)).unwrap_or(now);
        self.head = (self.head + 1) % DELAY;
        let mut y = late;
        let mut sine_res = 0.0;
        for (k, (y, r)) in y.iter_mut().zip(self.res.iter_mut()).enumerate() {
            let n = f64::from(r.next());
            if k == SINE {
                sine_res = n;
            } else {
                *y += n;
            }
        }
        let tri_now = now.get(TRI).copied().unwrap_or(0.0);
        self.smooth.push(now.get(SINE).copied().unwrap_or(0.0) - self.kink * tri_now);
        let fir = self.smooth.fir();
        let tri_out = y.get(TRI).copied().unwrap_or(0.0);
        if let Some(s) = y.get_mut(SINE) {
            *s = self.kink * tri_out + fir + sine_res;
        }
        y
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
    use super::*;

    const SR: f64 = 48000.0;

    fn vco() -> Vco {
        let mut v = Vco::new();
        v.set_sample_rate(SR);
        v
    }

    /// Defaults of the module: pw 0.5, everything else 0.
    fn base() -> VcoIn {
        VcoIn { pw: 0.5, ..VcoIn::default() }
    }

    fn rising(x: &[f64]) -> usize {
        x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count()
    }

    /// Pitch CV is in octaves above C4.
    #[test]
    fn pitch_440_by_saw_and_tri() {
        let mut v = vco();
        let i = VcoIn { pitch_cv: (440.0f64 / C4_HZ).log2() as f32, ..base() };
        let (mut saw, mut tri) = (vec![], vec![]);
        for _ in 0..SR as usize {
            let y = v.tick(&VcoOpts::default(), &i);
            saw.push(y[2]);
            tri.push(y[1]);
        }
        let (s, t) = (rising(&saw[200..]), rising(&tri[200..]));
        let want = 440.0 * (saw.len() - 200) as f64 / SR;
        assert!((s as f64 - want).abs() <= 1.0, "saw {s}, want {want}");
        assert!((t as f64 - want).abs() <= 1.0, "tri {t}, want {want}");
    }

    #[test]
    fn nonfinite_pitch_holds_core() {
        let mut v = vco();
        for k in 0..2000 {
            let pitch_cv = if k % 7 == 0 { f32::NAN } else { 0.1 };
            let y = v.tick(&VcoOpts::default(), &VcoIn { pitch_cv, ..base() });
            assert!(y.iter().all(|x| x.is_finite()), "k {k}: {y:?}");
        }
    }

    #[test]
    fn pulse_extreme_pw_is_finite() {
        for pw in [0.0, 1.0, f32::INFINITY, f32::NEG_INFINITY] {
            let mut v = vco();
            for k in 0..2000 {
                let y = v.tick(&VcoOpts::default(), &VcoIn { pw, ..base() });
                assert!(y.iter().all(|x| x.is_finite()), "pw {pw} k {k}: {y:?}");
            }
        }
    }

    #[test]
    fn first_sample_high_sync_is_not_an_edge() {
        let free = |sync: f32| {
            let mut v = vco();
            let i = VcoIn { pitch_cv: 0.2, sync, ..base() };
            (0..400).map(|_| v.tick(&VcoOpts::default(), &i)).collect::<Vec<_>>()
        };
        // a gate high from the first sample never produces an edge: same output as no sync at all
        assert_eq!(free(1.0), free(0.0));
    }
}
