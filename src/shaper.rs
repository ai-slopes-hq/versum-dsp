use crate::Block;
use std::f64::consts::{PI, TAU};

const UP_TAPS: usize = 128;
const DOWN_TAPS: usize = 256;

/// Last `L` values, newest first via `recent()`. Each value is written twice into a `2L` buffer so `recent()`
/// is always `L` contiguous values (no modulo in the convolution).
#[derive(Clone, Debug)]
struct History {
    len: usize,
    buf: Box<[f64]>,
    head: usize,
}

impl History {
    fn new(len: usize) -> History {
        History {
            len,
            buf: vec![0.0; 2 * len].into_boxed_slice(),
            head: 0,
        }
    }
    fn push(&mut self, v: f64) {
        self.head = if self.head == 0 {
            self.len - 1
        } else {
            self.head - 1
        };
        self.buf[self.head] = v;
        self.buf[self.head + self.len] = v;
    }
    fn recent(&self) -> &[f64] {
        &self.buf[self.head..self.head + self.len]
    }
    fn clear(&mut self) {
        self.buf.fill(0.0);
        self.head = 0;
    }
}

/// Chromium-style 2x stage (`UpSampler` / `DownSampler`): even upsampled samples are the input delayed by 64,
/// odd ones come from a 128-point half-sample-shifted sinc; the downsampler is a 256-point half-band sinc/2.
/// Both are Blackman-windowed (alpha 0.16). Delay of one stage: 128 samples at its base rate.
#[derive(Clone, Debug)]
struct Stage {
    input: History,
    even_high: History,
    odd_high: History,
}

impl Stage {
    fn new() -> Stage {
        Stage {
            input: History::new(UP_TAPS),
            even_high: History::new(DOWN_TAPS / 4 + 1),
            odd_high: History::new(DOWN_TAPS / 2),
        }
    }

    fn step(&mut self, k: &Kernels, x: f64, curve: &mut impl FnMut(f64) -> f64) -> f64 {
        self.input.push(x);
        let in_ = self.input.recent();
        self.even_high.push(curve(in_[UP_TAPS / 2]));
        let mut odd = 0.0;
        let mut out = k.down[DOWN_TAPS / 2] * self.even_high.recent()[DOWN_TAPS / 4];
        let odd_hist = self.odd_high.recent();
        for t in 0..UP_TAPS {
            odd += k.up[t] * in_[t];
            out += k.down[2 * t + 1] * odd_hist[t];
        }
        self.odd_high.push(curve(odd));
        out
    }

    fn reset(&mut self) {
        self.input.clear();
        self.even_high.clear();
        self.odd_high.clear();
    }
}

#[derive(Clone, Debug)]
struct Kernels {
    up: [f64; UP_TAPS],
    down: [f64; DOWN_TAPS],
}

impl Kernels {
    fn new() -> Kernels {
        let (a0, a1, a2) = (0.5 * (1.0 - 0.16), 0.5, 0.5 * 0.16);
        let win = |x: f64| a0 - a1 * (TAU * x).cos() + a2 * (2.0 * TAU * x).cos();
        let sinc = |s: f64| if s == 0.0 { 1.0 } else { s.sin() / s };
        let mut up = [0.0; UP_TAPS];
        let mut down = [0.0; DOWN_TAPS];
        for (i, u) in up.iter_mut().enumerate() {
            let s = PI * (i as f64 - (UP_TAPS / 2) as f64 + 0.5);
            *u = sinc(s) * win((i as f64 + 0.5) / UP_TAPS as f64);
        }
        for (i, d) in down.iter_mut().enumerate() {
            let s = PI * (i as f64 - (DOWN_TAPS / 2) as f64) * 0.5;
            *d = 0.5 * sinc(s) * win(i as f64 / DOWN_TAPS as f64);
        }
        Kernels { up, down }
    }
}

/// Table waveshaper with 4x oversampling (the same cascade of two Chromium 2x stages as Web Audio's `WaveShaperNode`
/// with `oversample = '4x'`). The table is the curve on [-1, 1]: `table[0]` at -1, `table[N-1]` at +1, linear
/// interpolation between points. Input is clamped to +-1.
///
/// Latency is 192 samples (the filters' group delay). Mixing dry with shaped signal needs the dry delayed by the same.
#[derive(Clone, Debug)]
pub struct Shaper<const N: usize> {
    table: Box<[f32; N]>,
    kernels: Box<Kernels>,
    outer: Stage,
    inner: Stage,
}

impl<const N: usize> Shaper<N> {
    /// `table` is the curve on [-1, 1]; `N` must be odd and at least 3.
    pub fn new(table: &[f32; N]) -> Shaper<N> {
        assert!(N >= 3 && N % 2 == 1, "table length must be odd and >= 3");
        Shaper {
            table: Box::new(*table),
            kernels: Box::new(Kernels::new()),
            outer: Stage::new(),
            inner: Stage::new(),
        }
    }
}

/// Curve value at `x` with linear interpolation; beyond the ends the end point is held (NaN goes to the low end).
fn lookup<const N: usize>(table: &[f32; N], x: f64) -> f64 {
    let v = (N - 1) as f64 * (x + 1.0) * 0.5;
    // `!(v > 0)` rather than `v <= 0`: NaN goes to the low end too
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(v > 0.0) {
        return table[0] as f64;
    }
    if v >= (N - 1) as f64 {
        return table[N - 1] as f64;
    }
    let k = v as usize;
    let f = v - k as f64;
    (1.0 - f) * table[k] as f64 + f * table[k + 1] as f64
}

impl<const N: usize> Block for Shaper<N> {
    fn tick(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.reset();
            return 0.0;
        }
        let x = (x as f64).clamp(-1.0, 1.0);
        let Shaper {
            table,
            kernels,
            outer,
            inner,
        } = self;
        let table: &[f32; N] = table;
        let k: &Kernels = kernels;
        let mut shape = |u: f64| lookup(table, u);
        let y = outer.step(k, x, &mut |u| inner.step(k, u, &mut shape));
        let out = y as f32;
        if !out.is_finite() {
            self.reset();
            return 0.0;
        }
        out
    }

    fn reset(&mut self) {
        self.outer.reset();
        self.inner.reset();
    }
}
