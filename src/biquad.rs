use crate::Block;

/// Response shape of a [`Biquad`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiquadKind {
    Lowpass,
    Highpass,
}

/// Second-order filter (RBJ cookbook, same formulas as Web Audio's `BiquadFilterNode`), direct form I.
/// `q_db` is the gain at the cutoff in dB: 0 is a slight hump, -3.01 is Butterworth.
#[derive(Clone, Debug)]
pub struct Biquad {
    sr: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Starts as a pass-through.
    pub fn new(sr: f64) -> Biquad {
        Biquad { sr, b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    /// Recomputes coefficients; filter state is kept. Costly (trig, pow): call once per block, not per sample.
    /// The frequency is clamped to `1 Hz..0.49 sr`; a non-finite argument leaves the filter unchanged.
    pub fn set(&mut self, kind: BiquadKind, freq_hz: f64, q_db: f64) {
        if !freq_hz.is_finite() || !q_db.is_finite() {
            return;
        }
        let f = freq_hz.clamp(1.0, self.sr * 0.49);
        let omega = std::f64::consts::TAU * f / self.sr;
        let (sin, cos) = omega.sin_cos();
        let alpha = sin / (2.0 * 10f64.powf(q_db / 20.0));
        let (b0, b1, b2) = match kind {
            BiquadKind::Lowpass => ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0),
            BiquadKind::Highpass => ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0),
        };
        let a0 = 1.0 + alpha;
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = -2.0 * cos / a0;
        self.a2 = (1.0 - alpha) / a0;
    }
}

impl Block for Biquad {
    fn tick(&mut self, x: f32) -> f32 {
        let xi = x as f64;
        let y = self.b0 * xi + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        let out = y as f32;
        if !xi.is_finite() || !out.is_finite() {
            self.reset();
            return 0.0;
        }
        self.x2 = self.x1;
        self.x1 = xi;
        self.y2 = self.y1;
        self.y1 = y;
        out
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}
