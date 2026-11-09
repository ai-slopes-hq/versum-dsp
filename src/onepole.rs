use crate::Block;

/// One-pole filter with simultaneous low-pass and high-pass outputs.
#[derive(Clone, Debug)]
pub struct OnePole {
    sr: f64,
    a: f64,
    z: f64,
}

impl OnePole {
    pub fn new(hz: f32, sr: f32) -> OnePole {
        let mut f = OnePole { sr: sr as f64, a: 0.0, z: 0.0 };
        f.set_hz(hz);
        f
    }

    pub fn set_hz(&mut self, hz: f32) {
        let hz = (hz as f64).clamp(0.0, self.sr * 0.49);
        self.a = 1.0 - (-std::f64::consts::TAU * hz / self.sr).exp();
    }

    /// Returns `(lp, hp)`; `lp + hp == x`.
    pub fn tick2(&mut self, x: f32) -> (f32, f32) {
        let x = x as f64;
        self.z += self.a * (x - self.z);
        let lp = self.z as f32;
        let hp = (x - self.z) as f32;
        if lp.is_finite() && hp.is_finite() {
            (lp, hp)
        } else {
            self.reset();
            (0.0, 0.0)
        }
    }
}

impl Block for OnePole {
    fn tick(&mut self, x: f32) -> f32 {
        self.tick2(x).0
    }

    fn reset(&mut self) {
        self.z = 0.0;
    }
}
