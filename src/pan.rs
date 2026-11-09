/// Equal-power pan, `pan` in [-1, 1]; returns `(left, right)` gains.
pub fn pan_eq(pan: f32) -> (f32, f32) {
    let theta = (pan.clamp(-1.0, 1.0) as f64 + 1.0) * std::f64::consts::FRAC_PI_4;
    (theta.cos() as f32, theta.sin() as f32)
}
