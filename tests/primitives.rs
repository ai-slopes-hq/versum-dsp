#![cfg(feature = "testkit")]
use versum_dsp::delay::Delay;
use versum_dsp::dc::DcBlock;
use versum_dsp::edge::EdgeDetector;
use versum_dsp::knee::Knee;
use versum_dsp::onepole::OnePole;
use versum_dsp::pan::pan_eq;
use versum_dsp::ramp::Ramp;
use versum_dsp::testkit::{assert_blockwise_eq, count_allocs, CountingAlloc};
use versum_dsp::{Block, Rng};

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

fn noise(n: usize) -> Vec<f32> {
    let mut r = Rng::new(7);
    (0..n).map(|_| (r.next_f64() * 2.0 - 1.0) as f32).collect()
}

#[test]
fn dc_decays_below_one_percent() {
    let mut d = DcBlock::new(5.0, 48000.0);
    let mut y: f32 = 1.0;
    for _ in 0..12000 {
        y = d.tick(1.0);
    }
    assert!(y.abs() < 0.01, "{y}");
}

#[test]
fn knee_is_linear_below_and_bounded_above() {
    let mut k = Knee::new(1.0, 0.9, 0.1);
    assert_eq!(k.tick(0.5), 0.5);
    assert!(k.tick(100.0) <= 1.0);
    let mut prev = 0.0f32;
    for i in 0..=1000 {
        let x = i as f32 * 0.01;
        let p = k.tick(x);
        assert_eq!(k.tick(-x), -p);
        assert!(p >= prev, "x={x}");
        prev = p;
    }
}

#[test]
fn pan_is_equal_power() {
    for i in 0..=200 {
        let p = -1.0 + i as f32 * 0.01;
        let (l, r) = pan_eq(p);
        assert!((l * l + r * r - 1.0).abs() < 1e-6, "pan {p}");
    }
    let (l, r) = pan_eq(-1.0);
    assert!((l - 1.0).abs() < 1e-6 && r.abs() < 1e-6);
    let (l, r) = pan_eq(1.0);
    assert!(l.abs() < 1e-6 && (r - 1.0).abs() < 1e-6);
}

#[test]
fn an_edge_fires_once_and_rearms_below_a_quarter() {
    let mut e = EdgeDetector::new();
    let n = [0.0, 0.6, 0.6, 0.4, 0.6].iter().filter(|&&x| e.tick(x)).count();
    assert_eq!(n, 1);
    e.tick(0.2);
    assert!(e.tick(0.6));
}

#[test]
fn ramp_lands_exactly_on_target() {
    let mut r = Ramp::new(48000.0, 10.0);
    r.set_target(1.0);
    for _ in 0..480 {
        r.next();
    }
    assert_eq!(r.value(), 1.0);
}

#[test]
fn onepole_lp_plus_hp_equals_input() {
    let mut f = OnePole::new(1000.0, 48000.0);
    for x in noise(4096) {
        let (lp, hp) = f.tick2(x);
        assert!((lp + hp - x).abs() < 1e-5);
    }
}

#[test]
fn delay_reads_back_integer_and_fractional() {
    let mut d = Delay::new(16);
    let mut out = Vec::new();
    for i in 0..8 {
        out.push(d.write_read(if i == 0 { 1.0 } else { 0.0 }, 4.0));
    }
    assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    d.reset();
    let mut out = Vec::new();
    for i in 0..8 {
        out.push(d.write_read(if i == 0 { 1.0 } else { 0.0 }, 4.5));
    }
    assert_eq!(out, [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]);
    // out of range is clamped
    assert!(d.write_read(1.0, 0.0).is_finite());
    assert!(d.write_read(1.0, 1000.0).is_finite());
}

#[test]
fn blocks_are_blockwise_equal() {
    let x = noise(4096);
    let c = [1, 7, 64, 128, 3];
    assert_blockwise_eq(|| DcBlock::new(5.0, 48000.0), &x, &c);
    assert_blockwise_eq(|| OnePole::new(800.0, 48000.0), &x, &c);
    assert_blockwise_eq(|| Knee::new(2.0, 0.9, 0.1), &x, &c);
}

#[test]
fn no_alloc_after_new() {
    let mut buf = noise(512);
    let mut dc = DcBlock::new(5.0, 48000.0);
    let mut op = OnePole::new(800.0, 48000.0);
    let mut kn = Knee::new(2.0, 0.9, 0.1);
    let mut dl = Delay::new(64);
    let (_, n) = count_allocs(|| {
        dc.process(&mut buf);
        op.process(&mut buf);
        kn.process(&mut buf);
        for &x in buf.iter() {
            dl.write_read(x, 10.5);
        }
    });
    assert_eq!(n, 0);
}

fn finite_after_bad<B: Block>(mut b: B) {
    for bad in [f32::NAN, f32::INFINITY] {
        b.tick(0.3);
        let y = b.tick(bad);
        assert!(y.is_finite());
        for _ in 0..100 {
            assert!(b.tick(0.1).is_finite());
        }
    }
}

#[test]
fn outputs_are_finite_on_nan_input() {
    finite_after_bad(DcBlock::new(5.0, 48000.0));
    finite_after_bad(OnePole::new(800.0, 48000.0));
    finite_after_bad(Knee::new(2.0, 0.9, 0.1));
    let mut d = Delay::new(8);
    d.write_read(f32::NAN, 2.0);
    for _ in 0..100 {
        assert!(d.write_read(0.1, 2.0).is_finite());
    }
}
