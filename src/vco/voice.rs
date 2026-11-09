//! The VCO core: a triangle-core phase with five simultaneous outputs and an event list for anti-aliasing.
//! `advance` walks one sample of phase, stops at every discontinuity or kink (wrap, apex, the two pulse
//! thresholds) and reports the jump of value and slope of every output at its sub-sample time.
//!
//! The pulse threshold `t` moves linearly from the previous sample's value to this one's, and so do the two
//! threshold phases (they are linear in `t`): a crossing is a root of `p(τ) − th(τ)`, which a threshold sweeping
//! past a slow phase has as well as a phase running past a held threshold. That is what lets `pw` be modulated at
//! audio rate.
//!
//! Within a sample the phase increment ramps linearly from the previous sample's to this one's, so the phase is
//! piecewise quadratic and its slope is continuous across samples. With a held increment the phase would bend at
//! every sample boundary under FM, and the event times (read off a phase that is a polyline through the sample
//! points) would jitter against the period; under through-zero FM that jitter is broadband noise around −65 dB.

pub const SINE: usize = 0;
pub const TRI: usize = 1;

pub const SAW: usize = 2;
#[allow(dead_code)]
pub const PULSE: usize = 3;
#[allow(dead_code)]
pub const SUB: usize = 4;
pub const OUTS: usize = 5;

/// Asymmetry of the triangle: it rises over the fraction `0.5·(1 + δ)` of the period.
pub const DELTA: f64 = 0.01;
/// Sine shaper `tanh(k·tri)/tanh(k)`.
const K: f64 = 1.40;
/// `K.tanh()`, hoisted out of the per-sample paths (`tanh` is not `const`); `hoisted_constants_match` checks it.
const TANH_K: f64 = 0.8853516482022625;
/// `1 / tanh K`.
const INV_TANH_K: f64 = 1.0 / TANH_K;
/// `K·(1 − tanh²K)/tanh K`, the value of `sine_kink()`.
const KINK: f64 = 0.34180028156068737;
/// Distance from an event at which the sides of it are sampled.
const EPS: f64 = 1e-9;
/// Slack for an event that lands on the end of the sample.
const TOL: f64 = 1e-9;
const MAX_EVENTS: usize = 8;
/// Distance from a crossing point at which `advance` stops taking the no-event shortcut.
const FAST_MARGIN: f64 = 1e-6;

#[derive(Clone, Copy)]
pub struct Shape {
    /// Rising share of the triangle period.
    pub a: f64,
    /// Pulse threshold on `tri`: `1 − 2·pw`.
    pub t: f64,
    /// Sub is the second (÷4) trigger.
    pub sub2: bool,
}

impl Shape {
    #[inline]
    pub fn new(pw: f64, sub2: bool) -> Self {
        Self { a: 0.5 * (1.0 + DELTA), t: (1.0 - 2.0 * pw).clamp(-0.9, 0.9), sub2 }
    }

    /// Phases where `tri` crosses `t` going up and going down: pulse is high between them.
    fn thresholds(&self) -> (f64, f64) {
        (self.a * (self.t + 1.0) * 0.5, self.a + (1.0 - self.a) * (1.0 - self.t) * 0.5)
    }
}

#[derive(Clone, Copy)]
pub enum SyncMode {
    Hard,
    Reverse,
}

/// Slope of `sine` per unit slope of `tri` at the corners (`tri = ±1`): there the kinks of the two are in this ratio
/// `c`, so `sine − c·tri` has no kinks.
pub fn sine_kink() -> f64 {
    KINK
}

#[derive(Clone)]
pub struct Voice {
    /// Phase in `[0, 1]`; `1` is the top side of the wrap, `0` the bottom side.
    p: f64,
    /// Direction of the phase, ±1.
    d: f64,
    /// `f/sr` of the previous sample; NaN before the first one.
    last: f64,
    /// Pulse threshold `t` of the previous sample; NaN before the first one.
    last_t: f64,
    sub: [f64; 2],
}

/// `tanh x` for `|x| ≤ 1.4` (the sine shaper's range) from the 7-term continued fraction as one rational function:
/// error below 1e-11, several times cheaper than the library `tanh`.
fn tanh_fast(x: f64) -> f64 {
    let y = x * x;
    let num = 2_027_025.0 + y * (270_270.0 + y * (6_930.0 + y * 36.0));
    let den = 2_027_025.0 + y * (945_945.0 + y * (51_975.0 + y * (630.0 + y)));
    x * num / den
}

fn tri_at(p: f64, a: f64) -> f64 {
    if p < a {
        -1.0 + 2.0 * p / a
    } else {
        1.0 - 2.0 * (p - a) / (1.0 - a)
    }
}

fn dtri_at(p: f64, a: f64) -> f64 {
    if p < a {
        2.0 / a
    } else {
        -2.0 / (1.0 - a)
    }
}

/// The five naive values at phase `p`, given `tri = tri_at(p)` and `th = tanh(K·tri)`.
fn values_from(p: f64, tri: f64, th: f64, sub: [f64; 2], s: &Shape) -> [f64; OUTS] {
    let (th1, th2) = s.thresholds();
    [
        th * INV_TANH_K,
        tri,
        2.0 * p - 1.0,
        if p > th1 && p < th2 { 1.0 } else { -1.0 },
        if s.sub2 { sub[1] } else { sub[0] },
    ]
}

/// The five naive values at phase `p`.
fn values_at(p: f64, sub: [f64; 2], s: &Shape) -> [f64; OUTS] {
    let tri = tri_at(p, s.a);
    values_from(p, tri, tanh_fast(K * tri), sub, s)
}

/// Everything an event needs at phase `p`, moving `vel` per sample with acceleration `acc` per sample², computed
/// from one evaluation of the shaper: the five values, the five slopes per sample, and the sine's kink-free
/// remainder `r = sine − c·tri` with its first three time derivatives (the phase is quadratic within a sample).
fn eval_at(p: f64, sub: [f64; 2], s: &Shape, vel: f64, acc: f64) -> ([f64; OUTS], [f64; OUTS], [f64; 4]) {
    let (tri, g) = (tri_at(p, s.a), dtri_at(p, s.a));
    let th = tanh_fast(K * tri);
    let q = 1.0 - th * th;
    let (s0, s1) = (th * INV_TANH_K, K * q * INV_TANH_K);
    let s2 = -2.0 * K * K * th * q * INV_TANH_K;
    let s3 = -2.0 * K * K * K * q * (1.0 - 3.0 * th * th) * INV_TANH_K;
    let (t1, t2) = (g * vel, g * acc);
    let values = values_from(p, tri, th, sub, s);
    let slopes = [s1 * t1, t1, 2.0 * vel, 0.0, 0.0];
    let rem = [s0 - KINK * tri, (s1 - KINK) * t1, s2 * t1 * t1 + (s1 - KINK) * t2, s3 * t1 * t1 * t1 + 3.0 * s2 * t1 * t2];
    (values, slopes, rem)
}

fn diff<const N: usize>(after: [f64; N], before: [f64; N]) -> [f64; N] {
    let mut d = [0.0; N];
    for ((d, a), b) in d.iter_mut().zip(after).zip(before) {
        *d = a - b;
    }
    d
}

/// Smallest `x ∈ (0, rem]` where `a·x² + b·x + c` changes sign (a tangent touch is not a crossing).
fn first_crossing(a: f64, b: f64, c: f64, rem: f64) -> Option<f64> {
    let mut best: Option<f64> = None;
    let mut take = |x: f64| {
        if x > 0.0 && x <= rem && best.is_none_or(|y| x < y) {
            best = Some(x);
        }
    };
    if a == 0.0 {
        if b != 0.0 {
            take(-c / b);
        }
    } else {
        let disc = b * b - 4.0 * a * c;
        if disc > 0.0 {
            let q = -0.5 * (b + b.signum() * disc.sqrt());
            if q != 0.0 {
                take(q / a);
                take(c / q);
            }
        }
    }
    best
}

impl Voice {
    #[inline]
    pub fn new(phase: f64) -> Self {
        Self { p: phase, d: 1.0, last: f64::NAN, last_t: f64::NAN, sub: [-1.0; 2] }
    }

    #[inline]
    pub fn values(&self, s: &Shape) -> [f64; OUTS] {
        values_at(self.p, self.sub, s)
    }

    /// Moves the phase over one sample at `inc = f/sr` (signed; the voice applies its direction), the increment
    /// ramping from the previous sample's and the pulse threshold from the previous sample's `s.t`.
    /// `emit(ago, Δv, Δs, Δr)` is called per event in time order, `ago` being how long before the end of the sample
    /// it happened; `Δr` are the jumps of the sine's remainder `sine − c·tri` and of its first three derivatives.
    /// `sync` is an edge of the sync input at the fractional time `τ` of this sample: Hard restarts the phase at 0
    /// going up with both sub triggers at −1, Reverse flips the direction. It is an event like the others, with
    /// jumps taken from both sides of it.
    #[inline]
    pub fn advance(
        &mut self,
        inc: f64,
        s: &Shape,
        sync: Option<(f64, SyncMode)>,
        emit: &mut impl FnMut(f64, [f64; OUTS], [f64; OUTS], [f64; 4]),
    ) {
        self.advance_with::<true>(inc, s, sync, emit);
    }

    /// `advance`; with `FAST` false the event-free shortcut is off and every sample goes through the event loop
    /// (the reference the tests compare the shortcut with; a constant, so release code carries no switch).
    fn advance_with<const FAST: bool>(
        &mut self,
        inc: f64,
        s: &Shape,
        sync: Option<(f64, SyncMode)>,
        emit: &mut impl FnMut(f64, [f64; OUTS], [f64; OUTS], [f64; 4]),
    ) {
        let inc = if inc.is_finite() { inc } else { 0.0 };
        let i0 = if self.last.is_finite() { self.last } else { inc };
        self.last = inc;
        let t0 = if self.last_t.is_finite() { self.last_t } else { s.t };
        self.last_t = s.t;
        let mut sync = sync.filter(|(t, _)| t.is_finite()).map(|(t, m)| (t.clamp(0.0, 1.0), m));
        if i0 == 0.0 && inc == 0.0 && sync.is_none() && t0 == s.t {
            return;
        }
        // threshold phases at the start of the sample and their change over it: th(τ) = th0 + dth·τ
        let (th1, th2) = s.thresholds();
        let (th1_0, th2_0) = if t0 == s.t { (th1, th2) } else { Shape { t: t0, ..*s }.thresholds() };
        let (dth1, dth2) = (th1 - th1_0, th2 - th2_0);
        // the shape of the sample at the time τ
        let shape_at = |tau: f64| Shape { t: t0 + (s.t - t0) * tau, ..*s };
        let mut p = self.p;
        let mut d = self.d;
        let mut tau = 0.0;
        // p(τ) = p0 + b·τ + a·τ², τ ∈ [0, 1]; b and a follow the direction
        let coef = |d: f64| (d * i0, 0.5 * d * (inc - i0));
        // Fast path: no sync, the phase keeps its direction over the whole sample (`b` and the end slope have the
        // same sign, so it is monotonic) and none of the five crossing points lies between its ends (with a margin
        // far above `TOL`; a threshold counts with everything it sweeps over): there is no event, the sample is one
        // quadratic piece.
        if FAST && sync.is_none() {
            let (b, a) = coef(d);
            let end = p + b + a;
            if b * (b + 2.0 * a) > 0.0 {
                let (lo, hi) = (p.min(end) - FAST_MARGIN, p.max(end) + FAST_MARGIN);
                let clear = |lo_c: f64, hi_c: f64| hi_c < lo || lo_c > hi;
                let points_clear = [1.0, 0.0, s.a, th1, th2].iter().all(|c| clear(*c, *c));
                // a threshold that moves also covers everything between its two ends
                if points_clear
                    && (t0 == s.t
                        || (clear(th1_0.min(th1), th1_0.max(th1)) && clear(th2_0.min(th2), th2_0.max(th2))))
                {
                    self.p = end.clamp(0.0, 1.0);
                    return;
                }
            }
        }
        for _ in 0..=MAX_EVENTS {
            let (b, a) = coef(d);
            let vel = b + 2.0 * a * tau;
            let up = vel > 0.0 || (vel == 0.0 && a > 0.0);
            let still = vel == 0.0 && a == 0.0;
            let rem = 1.0 - tau + TOL;
            // (time from tau, crossing point at tau = 0, its change over the sample, is a wrap)
            let next = if !still && p >= 1.0 && up {
                Some((0.0, 1.0, 0.0, true))
            } else if !still && p <= 0.0 && !up {
                Some((0.0, 0.0, 0.0, true))
            } else {
                let mut next: Option<(f64, f64, f64, bool)> = None;
                for (c, dc, wrap) in [(1.0, 0.0, true), (0.0, 0.0, true), (s.a, 0.0, false), (th1_0, dth1, false), (th2_0, dth2, false)] {
                    if let Some(x) = first_crossing(a, vel - dc, p - (c + dc * tau), rem) {
                        if next.is_none_or(|n| x < n.0) {
                            next = Some((x, c, dc, wrap));
                        }
                    }
                }
                next
            };
            if let Some((ts, mode)) = sync {
                if next.is_none_or(|n| ts < tau + n.0) {
                    let dt = (ts - tau).max(0.0);
                    p = (p + vel * dt + a * dt * dt).clamp(0.0, 1.0);
                    tau = tau.max(ts);
                    let v = b + 2.0 * a * tau;
                    let st = shape_at(tau);
                    let (v0, s0, r0) = eval_at(p, self.sub, &st, v, 2.0 * a);
                    match mode {
                        SyncMode::Hard => {
                            p = 0.0;
                            d = 1.0;
                            self.sub = [-1.0; 2];
                        }
                        SyncMode::Reverse => d = -d,
                    }
                    let (b, a) = coef(d);
                    let v = b + 2.0 * a * tau;
                    let (v1, s1, r1) = eval_at(p, self.sub, &st, v, 2.0 * a);
                    emit((1.0 - tau).clamp(0.0, 0.999_999), diff(v1, v0), diff(s1, s0), diff(r1, r0));
                    sync = None;
                    continue;
                }
            }
            let Some((x, c0, dc, wrap)) = next else { break };
            tau += x;
            let c = c0 + dc * tau;
            let v = b + 2.0 * a * tau;
            // the side the crossing is passed from: the phase relative to the crossing point (which moves, if it is
            // a threshold)
            let rel = v - dc;
            let sg = if rel > 0.0 || (rel == 0.0 && a > 0.0) { 1.0 } else { -1.0 };
            let side = |q: f64| q.rem_euclid(1.0);
            let (pre, post) = (side(c - sg * EPS), side(c + sg * EPS));
            let st = shape_at(tau);
            let (v0, s0, r0) = eval_at(pre, self.sub, &st, v, 2.0 * a);
            if wrap {
                self.sub[0] = -self.sub[0];
                if self.sub[0] > 0.0 {
                    self.sub[1] = -self.sub[1];
                }
            }
            p = if wrap { 1.0 - c } else { c };
            let (v1, s1, r1) = eval_at(post, self.sub, &st, v, 2.0 * a);
            emit((1.0 - tau).clamp(0.0, 0.999_999), diff(v1, v0), diff(s1, s0), diff(r1, r0));
        }
        let (b, a) = coef(d);
        let rest = (1.0 - tau).max(0.0);
        let vel = b + 2.0 * a * tau.min(1.0);
        self.d = d;
        self.p = (p + vel * rest + a * rest * rest).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, clippy::panic)]
    use super::*;

    #[test]
    fn tanh_fast_is_accurate_on_the_shaper_range() {
        for i in -1400..=1400 {
            let x = f64::from(i) / 1000.0;
            assert!((tanh_fast(x) - x.tanh()).abs() < 1e-11, "x {x}");
        }
    }

    #[test]
    fn hoisted_constants_match() {
        let th = K.tanh();
        assert_eq!(TANH_K.to_bits(), th.to_bits());
        assert_eq!(KINK.to_bits(), (K * (1.0 - th * th) / th).to_bits());
    }

    /// An edge of Hard sync in the same sample as a wrap, in either order: the wrap flips the sub, the sync puts
    /// it back to −1; after the sample the phase is `(1 − τ)·inc`, and there is exactly one sync event.
    #[test]
    fn sync_on_wrap_keeps_sub_consistent() {
        let inc = 0.01;
        let shape = Shape::new(0.5, false);
        // phase 0.995 wraps at τ = 0.5: sync after it (0.8) and before it (0.3)
        for tau_sync in [0.8, 0.3] {
            let mut v = Voice::new(0.995);
            let mut agos = vec![];
            v.advance(inc, &shape, Some((tau_sync, SyncMode::Hard)), &mut |ago, _, _, _| agos.push(ago));
            assert_eq!(v.sub, [-1.0; 2], "τ {tau_sync}");
            assert!((v.p - (1.0 - tau_sync) * inc).abs() < 1e-12, "τ {tau_sync}: p {}", v.p);
            let syncs = agos.iter().filter(|a| (**a - (1.0 - tau_sync)).abs() < 1e-9).count();
            assert_eq!(syncs, 1, "τ {tau_sync}: events {agos:?}");
        }
    }

    /// xorshift64*, enough for randomized cases.
    struct Rng(u64);

    impl Rng {
        /// Uniform in `[0, 1)`.
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        }

        fn between(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }

    type Event = (f64, [f64; OUTS], [f64; OUTS], [f64; 4]);

    fn run<const FAST: bool>(v: &mut Voice, inc: f64, s: &Shape, sync: Option<(f64, SyncMode)>) -> Vec<Event> {
        let mut events = vec![];
        v.advance_with::<FAST>(inc, s, sync, &mut |ago, dv, ds, dr| events.push((ago, dv, ds, dr)));
        events
    }

    fn close<const N: usize>(a: [f64; N], b: [f64; N]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() <= 1e-9)
    }

    /// A voice in a random state with a random sample to take: slow and fast, up and down, through zero, the pulse
    /// threshold held or moving, with the phase often a hair from a crossing point (the shortcut's margin is 1e-6).
    fn random_case(rng: &mut Rng) -> (Voice, f64, Shape) {
        let inc_of = |rng: &mut Rng| match rng.next() {
            x if x < 0.1 => 0.0,
            x if x < 0.4 => rng.between(-0.45, 0.45),
            x if x < 0.8 => rng.between(-0.02, 0.02),
            _ => rng.between(-1e-4, 1e-4),
        };
        let (i0, inc) = (inc_of(rng), inc_of(rng));
        let pw = |rng: &mut Rng| rng.between(0.05, 0.95);
        let (t0, shape) = {
            let t0 = Shape::new(pw(rng), false).t;
            let moved = if rng.next() < 0.5 { Shape::new(pw(rng), rng.next() < 0.5) } else { Shape::new((1.0 - t0) / 2.0, rng.next() < 0.5) };
            (t0, moved)
        };
        let mut v = Voice::new(0.0);
        v.last = i0;
        v.last_t = t0;
        v.d = if rng.next() < 0.5 { 1.0 } else { -1.0 };
        v.sub = [if rng.next() < 0.5 { 1.0 } else { -1.0 }, if rng.next() < 0.5 { 1.0 } else { -1.0 }];
        let (th1, th2) = shape.thresholds();
        let points = [1.0, 0.0, shape.a, th1, th2];
        v.p = match rng.next() {
            x if x < 0.4 => rng.next(),
            x if x < 0.7 => {
                // within a few samples of a crossing point, either side
                let c = points[(rng.next() * 5.0) as usize % 5];
                (c + rng.between(-4.0, 4.0) * (i0.abs() + inc.abs() + 1e-5)).clamp(0.0, 1.0)
            }
            x if x < 0.85 => {
                // a hair from it: around the shortcut's margin
                let c = points[(rng.next() * 5.0) as usize % 5];
                (c + rng.between(-3e-6, 3e-6)).clamp(0.0, 1.0)
            }
            _ => {
                // the sample ends a hair (1e-12 … 1e-6) from it: inside the slack `TOL` the loop gives an event
                let c = points[(rng.next() * 5.0) as usize % 5];
                let off = 10f64.powf(rng.between(-12.0, -6.0)) * if rng.next() < 0.5 { 1.0 } else { -1.0 };
                (c - v.d * 0.5 * (i0 + inc) + off).clamp(0.0, 1.0)
            }
        };
        (v, inc, shape)
    }

    /// The event-free shortcut against the full event loop: identical events (none, when it is taken) and state.
    #[test]
    fn fast_path_matches_the_event_loop() {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        let (mut quiet, mut busy) = (0, 0);
        for case in 0..200_000 {
            let (v, inc, shape) = random_case(&mut rng);
            let (mut fast, mut full) = (v.clone(), v.clone());
            let a = run::<true>(&mut fast, inc, &shape, None);
            let b = run::<false>(&mut full, inc, &shape, None);
            let same = a.len() == b.len()
                && a.iter().zip(&b).all(|(x, y)| (x.0 - y.0).abs() <= 1e-9 && close(x.1, y.1) && close(x.2, y.2) && close(x.3, y.3));
            assert!(same, "case {case}: state p {} d {} last {} last_t {}: events {} vs {}", v.p, v.d, v.last, v.last_t, a.len(), b.len());
            assert!((fast.p - full.p).abs() <= 1e-12, "case {case}: p {} vs {}", fast.p, full.p);
            assert_eq!((fast.d, fast.sub), (full.d, full.sub), "case {case}");
            if b.is_empty() {
                quiet += 1;
            } else {
                busy += 1;
            }
        }
        // both sides are really exercised
        assert!(quiet > 20_000 && busy > 20_000, "quiet {quiet}, busy {busy}");
    }

    /// Every change of the pulse between two samples is an event, including the changes only the threshold made: the
    /// naive value at the start plus the jumps of the events is the naive value at the end. The threshold sweeps
    /// by up to 0.5 of the phase per sample, also over a phase that stands still.
    #[test]
    fn pulse_events_account_for_a_moving_threshold() {
        let mut rng = Rng(0xFEED_FACE_CAFE_BEEF);
        let mut with_events = 0;
        for case in 0..2000 {
            let mut v = Voice::new(rng.next());
            let inc = match case % 4 {
                0 => 0.0,
                1 => rng.between(-0.3, 0.3),
                2 => rng.between(-0.01, 0.01),
                _ => 0.002,
            };
            let mut pw = rng.between(0.05, 0.95);
            let mut shape = Shape::new(pw, false);
            // settle the voice on this shape
            v.advance(inc, &shape, None, &mut |_, _, _, _| {});
            let mut prev = v.values(&shape)[PULSE];
            for i in 0..100 {
                pw = (pw + rng.between(-0.3, 0.3)).clamp(0.05, 0.95);
                shape = Shape::new(pw, false);
                let mut jump = 0.0;
                let mut n = 0;
                v.advance(inc, &shape, None, &mut |ago, dv, _, _| {
                    assert!((0.0..1.0).contains(&ago));
                    jump += dv[PULSE];
                    n += 1;
                });
                let now = v.values(&shape)[PULSE];
                assert!((prev + jump - now).abs() < 1e-6, "case {case}, sample {i}: {prev} + {jump} ≠ {now} ({n} events)");
                with_events += i32::from(n > 0);
                prev = now;
            }
        }
        assert!(with_events > 20_000, "{with_events}");
    }
}
