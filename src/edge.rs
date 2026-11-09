/// Schmitt-style rising edge detector: fires at `x > 0.5`, re-arms at `x < 0.25`.
#[derive(Clone, Debug)]
pub struct EdgeDetector {
    armed: bool,
}

impl EdgeDetector {
    pub fn new() -> EdgeDetector {
        EdgeDetector { armed: true }
    }

    pub fn tick(&mut self, x: f32) -> bool {
        if self.armed && x > 0.5 {
            self.armed = false;
            true
        } else {
            if x < 0.25 {
                self.armed = true;
            }
            false
        }
    }
}

impl Default for EdgeDetector {
    fn default() -> Self {
        Self::new()
    }
}
