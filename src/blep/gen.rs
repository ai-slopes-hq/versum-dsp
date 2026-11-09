//! Generator of the minBLEP / minBLAMP residual tables (`blep_table.rs`). Runs in `bench blep` and in tests.

use super::{DELAY, PHASES, TAPS};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

const ZEROS: usize = 16;
/// Table length minus one: `2·Z·P`.
const L: usize = 2 * ZEROS * PHASES;
/// Sinc cutoff as a share of Nyquist: the Blackman-Harris transition band must end below fs/2 for -80 dB at 4 kHz with 16 zeros.
const CUTOFF: f64 = 0.8;
const FFT_N: usize = 32768;

fn blackman_harris(i: usize, n: usize) -> f64 {
    let x = std::f64::consts::TAU * i as f64 / (n - 1) as f64;
    0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos()
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let a = std::f64::consts::PI * x;
        a.sin() / a
    }
}

/// Spectrum of the minimum-phase kernel on `FFT_N` points (time step `1/PHASES` sample), from the cepstrum of a
/// linear-phase windowed sinc.
fn min_phase_spectrum() -> Vec<Complex<f64>> {
    let p = PHASES as f64;
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(FFT_N);
    let inv = planner.plan_fft_inverse(FFT_N);
    let scale = 1.0 / FFT_N as f64;

    // linear-phase windowed sinc
    let mut buf = vec![Complex::new(0.0, 0.0); FFT_N];
    for (i, b) in buf.iter_mut().enumerate().take(L + 1) {
        *b = Complex::new(CUTOFF * sinc(CUTOFF * (i as f64 - L as f64 / 2.0) / p) * blackman_harris(i, L + 1), 0.0);
    }
    // real cepstrum of the log magnitude
    fwd.process(&mut buf);
    for b in buf.iter_mut() {
        *b = Complex::new(b.norm().max(1e-12).ln(), 0.0);
    }
    inv.process(&mut buf);
    // fold to a causal cepstrum
    for (n, b) in buf.iter_mut().enumerate() {
        let c = b.re * scale;
        let w = if n == 0 || n == FFT_N / 2 {
            1.0
        } else if n < FFT_N / 2 {
            2.0
        } else {
            0.0
        };
        *b = Complex::new(c * w, 0.0);
    }
    fwd.process(&mut buf);
    for b in buf.iter_mut() {
        *b = b.exp();
    }
    buf
}

/// Step response of the kernel delayed by `shift` samples (exactly, as a phase ramp on the spectrum), normalised
/// to end at 1, and its delay `∫(1 − step)` in samples by the trapezoid rule: the tables are read with linear
/// interpolation, so this is the area the ring really holds.
#[allow(clippy::indexing_slicing)] // fixed-size table arithmetic, test/bench only
fn step_response(spectrum: &[Complex<f64>], shift: f64) -> (Vec<f64>, f64) {
    let p = PHASES as f64;
    let mut buf: Vec<Complex<f64>> = spectrum
        .iter()
        .enumerate()
        .map(|(k, b)| {
            let k = if k <= FFT_N / 2 { k as f64 } else { k as f64 - FFT_N as f64 };
            b * Complex::from_polar(1.0, -std::f64::consts::TAU * k * shift * p / FFT_N as f64)
        })
        .collect();
    FftPlanner::<f64>::new().plan_fft_inverse(FFT_N).process(&mut buf);
    let mut step = Vec::with_capacity(L + 1);
    let mut acc = 0.0;
    for b in buf.iter().take(L + 1) {
        acc += b.re;
        step.push(acc);
    }
    for s in step.iter_mut() {
        *s /= acc;
    }
    let d = step.iter().map(|s| (1.0 - s) / p).sum::<f64>() - 0.5 * (2.0 - step[0] - step[L]) / p;
    (step, d)
}

/// Step response of the kernel delayed to a delay of `DELAY`, and the delay reached.
fn delayed_step() -> (Vec<f64>, f64) {
    let spectrum = min_phase_spectrum();
    let target = DELAY as f64;
    let (_, d0) = step_response(&spectrum, 0.0);
    let mut shift = target - d0;
    let (mut step, mut d) = step_response(&spectrum, shift);
    for _ in 0..4 {
        if (d - target).abs() < 1e-12 {
            break;
        }
        shift += target - d;
        (step, d) = step_response(&spectrum, shift);
    }
    (step, d)
}

/// Moments `∫ t^i·h(t) dt`, `i = 0..=3`, of the kernel the tables hold: the step table read with linear
/// interpolation, so `h` is constant on each cell.
#[allow(clippy::indexing_slicing)] // fixed-size table arithmetic, test/bench only
fn moments(step: &[f64]) -> [f64; 4] {
    let dt = 1.0 / PHASES as f64;
    let mut m = [step[0], 0.0, 0.0, 0.0];
    for c in 0..L {
        let h = (step[c + 1] - step[c]) / dt;
        let (t0, t1) = (c as f64 * dt, (c + 1) as f64 * dt);
        for (i, m) in m.iter_mut().enumerate() {
            let e = i as i32 + 1;
            *m += h * (t1.powi(e) - t0.powi(e)) / f64::from(e);
        }
    }
    m
}

/// The kernel at whole samples `0..TAPS` (slope of the interpolated step table there), corrected by a least change
/// (weighted by the tap) so that its moments 0..=3 equal the kernel's: an FIR that filters a smooth naive wave the
/// way the tables filter its jumps, so that the FIR-relative residuals end at 0.
#[allow(clippy::indexing_slicing)] // fixed-size table arithmetic, test/bench only
pub fn kernel() -> Vec<f64> {
    let p = PHASES as f64;
    let (step, _) = delayed_step();
    let at = |i: usize| step.get(i).copied().unwrap_or(1.0);
    let h: Vec<f64> =
        (0..TAPS).map(|j| if j == 0 { at(1) * p } else { (at(j * PHASES + 1) - at(j * PHASES - 1)) * p * 0.5 }).collect();
    // basis (j/S)^i keeps the 4×4 system well scaled
    const S: f64 = 8.0;
    let target = moments(&step);
    let basis = |i: usize, j: usize| (j as f64 / S).powi(i as i32);
    let mut err = [0.0; 4];
    for (i, e) in err.iter_mut().enumerate() {
        let have: f64 = h.iter().enumerate().map(|(j, x)| x * basis(i, j)).sum();
        *e = target[i] / S.powi(i as i32) - have;
    }
    let w: Vec<f64> = h.iter().map(|x| x.abs()).collect();
    // (A W Aᵀ) λ = err, δ = W Aᵀ λ
    let mut m = [[0.0; 5]; 4];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate().take(4) {
            *cell = (0..TAPS).map(|j| basis(r, j) * w[j] * basis(c, j)).sum();
        }
        row[4] = err[r];
    }
    for c in 0..4 {
        let piv = (c..4).max_by(|a, b| m[*a][c].abs().total_cmp(&m[*b][c].abs())).unwrap_or(c);
        m.swap(c, piv);
        let pivot = m[c];
        for (r, row) in m.iter_mut().enumerate() {
            if r != c {
                let f = row[c] / pivot[c];
                for (x, p) in row.iter_mut().zip(pivot).skip(c) {
                    *x -= f * p;
                }
            }
        }
    }
    let lambda: Vec<f64> = (0..4).map(|r| m[r][4] / m[r][r]).collect();
    (0..TAPS).map(|j| h[j] + w[j] * (0..4).map(|i| lambda[i] * basis(i, j)).sum::<f64>()).collect()
}

/// Residuals of a wave whose naive samples go through the FIR `kernel()` (the sine's kink-free remainder):
/// `(BOX, R1, R2, R3)`. A jump of the `n`-th derivative (per sample^n) leaves
/// `Rn(t) = ∫_0^t h(τ)·(t−τ)^n/n! dτ − Σ_{j≤t} KERNEL[j]·(t−j)^n/n!`; for `n = 0` that is `STEP_RES(t) + BOX[⌊t⌋]`
/// (`BOX[k] = 1 − Σ_{j≤k} KERNEL[j]`, whole cells), for `n ≥ 1` it is continuous and tabled like `RAMP_RES`.
#[allow(clippy::indexing_slicing)] // fixed-size table arithmetic, test/bench only
pub fn fir_tables() -> (Vec<f64>, [Vec<f64>; 3]) {
    let dt = 1.0 / PHASES as f64;
    let (step, _) = delayed_step();
    let k = kernel();
    let mut acc = 0.0;
    let boxes: Vec<f64> = k.iter().map(|x| {
        acc += x;
        1.0 - acc
    }).collect();
    // exact integrals of the piecewise-linear step: b[n][i] = n+1-fold integral of h at t_i
    let mut b = [vec![0.0; L + 1], vec![0.0; L + 1], vec![0.0; L + 1]];
    for i in 0..L {
        let (s0, s1) = (step[i], step[i + 1]);
        let (b1, b2, b3) = (b[0][i], b[1][i], b[2][i]);
        b[0][i + 1] = b1 + dt * (s0 + s1) / 2.0;
        b[1][i + 1] = b2 + dt * b1 + dt * dt * (2.0 * s0 + s1) / 6.0;
        b[2][i + 1] = b3 + dt * b2 + dt * dt / 2.0 * b1 + dt.powi(3) * (3.0 * s0 + s1) / 24.0;
    }
    let fact = [1.0, 2.0, 6.0];
    let res: [Vec<f64>; 3] = std::array::from_fn(|n| {
        let e = n as i32 + 1;
        let mut r: Vec<f64> = (0..=L)
            .map(|i| {
                let t = i as f64 * dt;
                let fir: f64 = k.iter().enumerate().filter(|(j, _)| *j as f64 <= t).map(|(j, x)| x * (t - j as f64).powi(e)).sum();
                b[n][i] - fir / fact[n]
            })
            .collect();
        let end = r[L];
        for (i, v) in r.iter_mut().enumerate() {
            *v -= end * i as f64 / L as f64;
        }
        r
    });
    (boxes, res)
}

/// `(STEP_RES, RAMP_RES, D)`: the tables are `L + 1` long, residuals over time in steps of `1/PHASES` sample.
/// The minimum-phase kernel is delayed until its delay `D` is `DELAY` whole samples, so the output can be the
/// naive wave `DELAY` samples ago plus the residuals `minBLEP(t) − u(t − DELAY)` and `minBLAMP(t) − r(t − DELAY)`
/// (`r` the unit ramp): a true delay, right for curved waves too. `STEP_RES` holds `minBLEP − 1`; the ring adds the
/// box `u(t) − u(t − DELAY)` in whole cells, so the jump at `t = DELAY` is never interpolated.
#[allow(clippy::indexing_slicing)] // fixed-size table arithmetic, test/bench only
pub fn tables() -> (Vec<f64>, Vec<f64>, f64) {
    let p = PHASES as f64;
    let target = DELAY as f64;
    let (step, d) = delayed_step();

    // trapezoid integral of the step, in samples
    let mut blamp = Vec::with_capacity(L + 1);
    let mut acc = 0.0;
    for (i, s) in step.iter().enumerate() {
        if i > 0 {
            acc += 0.5 * (step[i - 1] + s) / p;
        }
        blamp.push(acc);
    }
    let step_res: Vec<f64> = step.iter().map(|s| s - 1.0).collect();
    let mut ramp_res: Vec<f64> = (0..=L).map(|i| blamp[i] - (i as f64 / p - target).max(0.0)).collect();
    let end = ramp_res[L];
    for (i, r) in ramp_res.iter_mut().enumerate() {
        *r -= end * i as f64 / L as f64;
    }
    (step_res, ramp_res, d)
}

/// Source text of `blep_table.rs`.
pub fn table_rs(date: &str, commit: &str) -> String {
    let (step, ramp, _) = tables();
    let mut s = format!(
        "// Generated by `cd bench && cargo run --release -- blep`; do not edit by hand.\n\
         // Date: {date}; commit: {commit}.\n\
         // Causal minimum-phase minBLEP / minBLAMP residuals: {TAPS} taps, {PHASES} phases, {} entries each.\n\
         // The kernel is delayed to a delay of exactly DELAY = {DELAY} samples (see `blep::DELAY`).\n\
         // STEP_RES = minBLEP - 1 (the ring adds the box u(t) - u(t - DELAY)); RAMP_RES = minBLAMP - r(t - DELAY),\n\
         // end forced to 0. KERNEL = the kernel at whole samples (an FIR for smooth naive waves), moments 0..=3 matched.\n\
         // FIR_BOX, FIR_R1..3: residuals of jumps of the 0th..3rd derivative of a wave filtered by KERNEL\n\
         // (see `blep_gen::fir_tables`).\n\n",
        L + 1
    );
    let row: Vec<String> = kernel().iter().map(|v| format!("{v:?}")).collect();
    s.push_str(&format!("pub static KERNEL: [f64; {TAPS}] = [\n"));
    for chunk in row.chunks(4) {
        s.push_str(&format!("    {},\n", chunk.join(", ")));
    }
    s.push_str("];\n\n");
    let (boxes, [r1, r2, r3]) = fir_tables();
    let row: Vec<String> = boxes.iter().map(|v| format!("{:e}", *v as f32)).collect();
    s.push_str(&format!("pub static FIR_BOX: [f32; {TAPS}] = [\n"));
    for chunk in row.chunks(8) {
        s.push_str(&format!("    {},\n", chunk.join(", ")));
    }
    s.push_str("];\n\n");
    for (name, t) in [("STEP_RES", &step), ("RAMP_RES", &ramp), ("FIR_R1", &r1), ("FIR_R2", &r2), ("FIR_R3", &r3)] {
        s.push_str(&format!("pub static {name}: [f32; {}] = [\n", L + 1));
        for chunk in t.chunks(8) {
            let row: Vec<String> = chunk.iter().map(|v| format!("{:e}", *v as f32)).collect();
            s.push_str(&format!("    {},\n", row.join(", ")));
        }
        s.push_str("];\n\n");
    }
    s.truncate(s.trim_end().len());
    s.push('\n');
    s
}
