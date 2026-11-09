/// A mono sample processor. After construction, implementations must not allocate or block.
/// Output must be NaN-guarded (never return NaN/inf).
pub trait Block {
    fn tick(&mut self, x: f32) -> f32;
    fn reset(&mut self);
    fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.tick(*s);
        }
    }
}
