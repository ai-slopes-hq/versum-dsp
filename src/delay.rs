/// Fractional delay line with linear interpolation. Allocates only in `new`.
#[derive(Clone, Debug)]
pub struct Delay {
    buf: Vec<f32>,
    pos: usize,
    max: usize,
}

impl Delay {
    pub fn new(max_samples: usize) -> Delay {
        let max = max_samples.max(1);
        Delay { buf: vec![0.0; max + 2], pos: 0, max }
    }

    /// Writes `x`, then reads `delay` samples back (`1 <= delay <= max_samples`, clamped).
    pub fn write_read(&mut self, x: f32, delay: f32) -> f32 {
        let n = self.buf.len();
        self.buf[self.pos] = if x.is_finite() { x } else { 0.0 };
        let d = if delay.is_nan() { 1.0 } else { delay.clamp(1.0, self.max as f32) };
        let i = d as usize;
        let frac = d - i as f32;
        let a = self.buf[(self.pos + n - i) % n];
        let b = self.buf[(self.pos + n - i - 1) % n];
        self.pos = (self.pos + 1) % n;
        let out = a + (b - a) * frac;
        if out.is_finite() {
            out
        } else {
            self.reset();
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}
