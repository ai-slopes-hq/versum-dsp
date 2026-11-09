/// Linear ramp that lands exactly on its target after `ms`.
#[derive(Clone, Debug)]
pub struct Ramp {
    steps: u32,
    cur: f64,
    target: f64,
    step: f64,
    left: u32,
}

impl Ramp {
    pub fn new(sr: f32, ms: f32) -> Ramp {
        let steps = ((sr as f64 * ms as f64 / 1000.0).round() as u32).max(1);
        Ramp { steps, cur: 0.0, target: 0.0, step: 0.0, left: 0 }
    }

    pub fn reset(&mut self) {
        self.cur = 0.0;
        self.target = 0.0;
        self.step = 0.0;
        self.left = 0;
    }

    /// A non-finite target is ignored.
    pub fn set_target(&mut self, v: f32) {
        if !v.is_finite() {
            return;
        }
        self.target = v as f64;
        self.left = self.steps;
        self.step = (self.target - self.cur) / self.steps as f64;
    }

    pub fn next(&mut self) -> f32 {
        if self.left > 0 {
            self.left -= 1;
            if self.left == 0 {
                self.cur = self.target;
            } else {
                self.cur += self.step;
            }
        }
        if !self.cur.is_finite() {
            self.reset();
        }
        self.cur as f32
    }

    pub fn value(&self) -> f32 {
        self.cur as f32
    }
}
