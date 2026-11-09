use crate::Block;

/// DC blocker: `y[n] = x[n] - x[n-1] + R*y[n-1]`, `R = exp(-2π·hz/sr)`.
#[derive(Clone, Debug)]
pub struct DcBlock {
    r: f64,
    x1: f64,
    y1: f64,
}

impl DcBlock {
    pub fn new(hz: f32, sr: f32) -> DcBlock {
        let r = (-std::f64::consts::TAU * hz as f64 / sr as f64).exp();
        DcBlock { r, x1: 0.0, y1: 0.0 }
    }
}

impl Block for DcBlock {
    fn tick(&mut self, x: f32) -> f32 {
        let x = x as f64;
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        let out = y as f32;
        if out.is_finite() {
            out
        } else {
            self.reset();
            0.0
        }
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.y1 = 0.0;
    }
}
