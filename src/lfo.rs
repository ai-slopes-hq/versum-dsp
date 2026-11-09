/// Waveform of an [`Lfo`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LfoShape {
    Sine,
    Triangle,
}

/// Low-frequency oscillator, output in `[-1, 1]`. Phase is f64 in cycles, wrapped modulo 1.
/// Sine is `sin(2*pi*phase)`; triangle is 0 at phase 0, 1 at 0.25, -1 at 0.75.
#[derive(Clone, Debug)]
pub struct Lfo {
    shape: LfoShape,
    sr: f64,
    inc: f64,
    phase: f64,
    phase0: f64,
}

impl Lfo {
    /// `phase` is in cycles (0.25 is a quarter period).
    pub fn new(shape: LfoShape, freq_hz: f64, phase: f64, sr: f64) -> Lfo {
        let phase = if phase.is_finite() {
            phase.rem_euclid(1.0)
        } else {
            0.0
        };
        let inc = if freq_hz.is_finite() {
            freq_hz / sr
        } else {
            0.0
        };
        Lfo {
            shape,
            sr,
            inc,
            phase,
            phase0: phase,
        }
    }

    /// Current value, without advancing.
    pub fn value(&self) -> f32 {
        match self.shape {
            LfoShape::Sine => (std::f64::consts::TAU * self.phase).sin() as f32,
            LfoShape::Triangle => {
                let d = (self.phase - 0.25).rem_euclid(1.0);
                (1.0 - 4.0 * d.min(1.0 - d)) as f32
            }
        }
    }

    fn step(&mut self) {
        self.phase += self.inc;
        self.phase -= self.phase.floor();
    }

    /// Moves the phase by `n` samples; bit-identical to `n` calls of `tick`.
    pub fn advance(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Returns the current value, then advances one sample.
    pub fn tick(&mut self) -> f32 {
        let v = self.value();
        self.step();
        v
    }

    /// Keeps the phase. A non-finite frequency is ignored.
    pub fn set_freq(&mut self, hz: f64) {
        if hz.is_finite() {
            self.inc = hz / self.sr;
        }
    }

    /// Restores the initial phase.
    pub fn reset(&mut self) {
        self.phase = self.phase0;
    }
}
