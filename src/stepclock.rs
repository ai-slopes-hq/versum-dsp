//! Sample-exact sixteenth-note step clock. It only says "a step is due"; what the step does is up to the caller.

/// Counts samples to the next sixteenth-note step. The countdown is f64, so boundaries stay
/// fractional: step `k` lands on sample `ceil(k * period)` and block size never moves it.
///
/// Typical loop: while `next_in() <= 0.0` play a step and `consume()`; then `advance(n)` with
/// `n = ceil(next_in())` clamped to the rest of the block. A `set_bpm` keeps the step already
/// scheduled and applies the new period from the next `consume`.
#[derive(Clone, Debug)]
pub struct StepClock {
    sr: f64,
    bpm: f64,
    to_next: f64,
}

impl StepClock {
    /// First step is due immediately.
    pub fn new(sr: f64, bpm: f64) -> StepClock {
        let mut c = StepClock {
            sr,
            bpm: 120.0,
            to_next: 0.0,
        };
        c.set_bpm(bpm);
        c
    }

    /// Samples until the next step; `<= 0.0` means it is due now.
    pub fn next_in(&self) -> f64 {
        self.to_next
    }

    /// Call after playing a due step: schedules the next one one period later.
    pub fn consume(&mut self) {
        self.to_next += self.period();
    }

    /// Moves time forward by `n` samples.
    pub fn advance(&mut self, n: usize) {
        self.to_next -= n as f64;
    }

    /// Clamped to 20..300; non-finite values are ignored. The pending step keeps its deadline.
    pub fn set_bpm(&mut self, bpm: f64) {
        if bpm.is_finite() {
            self.bpm = bpm.clamp(20.0, 300.0);
        }
    }

    /// A step is due right away.
    pub fn reset(&mut self) {
        self.to_next = 0.0;
    }

    /// Samples per sixteenth at the current tempo.
    pub fn period(&self) -> f64 {
        self.sr * 60.0 / self.bpm / 4.0
    }
}
