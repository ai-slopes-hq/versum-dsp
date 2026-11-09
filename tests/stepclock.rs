use versum_dsp::stepclock::StepClock;

const SR: f64 = 48000.0;

/// Sample indices of the first `steps` step boundaries, driving the clock in blocks of `chunk`.
fn boundaries(bpm: f64, steps: usize, chunk: usize) -> Vec<u64> {
    let mut c = StepClock::new(SR, bpm);
    let mut out = Vec::new();
    let mut pos: u64 = 0;
    while out.len() < steps {
        while c.next_in() <= 0.0 && out.len() < steps {
            out.push(pos);
            c.consume();
        }
        // Advance up to the next step or the end of the block, as the engine does.
        let n = (c.next_in().ceil() as usize).clamp(1, chunk);
        c.advance(n);
        pos += n as u64;
    }
    out
}

#[test]
fn boundaries_are_sample_exact() {
    for bpm in [110.0, 122.0, 135.0] {
        let period = SR * 60.0 / bpm / 4.0;
        let c = StepClock::new(SR, bpm);
        assert_eq!(c.period(), period);
        let mut c = c;
        let mut pos = 0u64;
        for k in 0..20u64 {
            while c.next_in() > 0.0 {
                c.advance(1);
                pos += 1;
            }
            assert_eq!(pos, (k as f64 * period).ceil() as u64, "bpm {bpm} step {k}");
            c.consume();
        }
    }
}

#[test]
fn any_chunking_gives_same_boundaries() {
    for bpm in [110.0, 122.0, 135.0] {
        let want = boundaries(bpm, 20, 1);
        let period = SR * 60.0 / bpm / 4.0;
        for (k, &b) in want.iter().enumerate() {
            assert_eq!(b, (k as f64 * period).ceil() as u64);
        }
        for chunk in [7, 64, 128] {
            assert_eq!(boundaries(bpm, 20, chunk), want, "bpm {bpm} chunk {chunk}");
        }
    }
}

#[test]
fn bpm_change_keeps_pending_step() {
    let mut c = StepClock::new(SR, 120.0);
    let p120 = c.period();
    c.consume(); // step 0 played; next due in p120
    c.advance(100);
    let before = c.next_in();
    c.set_bpm(60.0);
    assert_eq!(c.next_in(), before);
    assert_eq!(c.period(), p120 * 2.0);
    c.advance(before.ceil() as usize);
    assert!(c.next_in() <= 0.0);
    c.consume();
    assert!((c.next_in() - (c.period() + (before.ceil() - before))).abs() < 1e-9);
}

#[test]
fn reset_is_due_immediately() {
    let mut c = StepClock::new(SR, 120.0);
    c.consume();
    assert!(c.next_in() > 0.0);
    c.reset();
    assert!(c.next_in() <= 0.0);
}

#[test]
fn nonfinite_bpm_ignored() {
    let mut c = StepClock::new(SR, 120.0);
    let p = c.period();
    for b in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        c.set_bpm(b);
        assert_eq!(c.period(), p);
    }
    c.set_bpm(1.0);
    assert_eq!(c.period(), SR * 60.0 / 20.0 / 4.0);
    c.set_bpm(1000.0);
    assert_eq!(c.period(), SR * 60.0 / 300.0 / 4.0);
}
