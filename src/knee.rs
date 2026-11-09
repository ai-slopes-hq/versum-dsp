use crate::Block;

/// Linear gain `k` with a smooth tanh knee: above `kn` the output saturates towards `kn + room`.
#[derive(Clone, Debug)]
pub struct Knee {
    k: f32,
    kn: f32,
    room: f32,
}

impl Knee {
    pub fn new(k: f32, kn: f32, room: f32) -> Knee {
        Knee { k, kn, room }
    }
}

impl Block for Knee {
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.k * x;
        let a = y.abs();
        let out = if a > self.kn {
            y.signum() * (self.kn + self.room * ((a - self.kn) / self.room).tanh())
        } else {
            y
        };
        if out.is_finite() {
            out
        } else {
            self.reset();
            0.0
        }
    }

    fn reset(&mut self) {}
}
