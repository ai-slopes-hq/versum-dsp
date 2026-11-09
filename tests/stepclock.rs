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

#[test]
fn nonsense_sample_rate_never_hangs() {
    for sr in [0.0, -48000.0, f64::NAN, f64::INFINITY] {
        let mut c = StepClock::new(sr, 120.0);
        assert!(c.period().is_finite() && c.period() > 0.0, "sr {sr}");
        // Same loop shape as the engine: a bounded number of steps must be due per call.
        let mut n = 0;
        while c.next_in() <= 0.0 && n < 10 {
            c.consume();
            n += 1;
        }
        assert!(n < 10, "sr {sr} would loop forever");
    }
}

#[test]
fn long_run_has_no_drift() {
    // `while` vs `if` for due steps are equivalent here: period >= 2400 samples, so at most one step is due per sample.
    for bpm in [135.0, 122.0, 110.0] {
        let period = SR * 60.0 / bpm / 4.0;
        let want = boundaries(bpm, 6000, 1);
        for (k, &b) in want.iter().enumerate() {
            assert!(
                (b as f64 - k as f64 * period).abs() < 1.0,
                "bpm {bpm} step {k}"
            );
        }
        assert_eq!(boundaries(bpm, 6000, 128), want, "bpm {bpm}");
    }
}

/// Engine-style loop with a tempo change at sample `change_at` and an optional hold window
/// `[hold.0, hold.1)` during which the clock is not touched at all.
fn run(
    total: u64,
    change_at: Option<(u64, f64)>,
    hold: Option<(u64, u64)>,
) -> (Vec<u64>, Vec<f64>) {
    let mut c = StepClock::new(SR, 120.0);
    let (mut steps, mut periods) = (Vec::new(), Vec::new());
    for pos in 0..total {
        if let Some((at, bpm)) = change_at {
            if pos == at {
                c.set_bpm(bpm);
            }
        }
        if matches!(hold, Some((a, b)) if pos >= a && pos < b) {
            continue;
        }
        while c.next_in() <= 0.0 {
            steps.push(pos);
            periods.push(c.period());
            c.consume();
        }
        c.advance(1);
    }
    (steps, periods)
}

#[test]
fn tempo_change_mid_run_neither_duplicates_nor_misses() {
    let (steps, periods) = run(200_000, Some((70_001, 135.0)), None);
    assert_eq!(steps[0], 0);
    for i in 0..steps.len() - 1 {
        let gap = (steps[i + 1] - steps[i]) as f64;
        // The gap is set by the period in force when step i was consumed.
        assert!(
            gap == periods[i].floor() || gap == periods[i].ceil(),
            "step {i}: gap {gap}, period {}",
            periods[i]
        );
    }
    assert!(periods.iter().any(|&p| p == SR * 60.0 / 135.0 / 4.0));
    assert!(periods.iter().any(|&p| p == SR * 60.0 / 120.0 / 4.0));
}

#[test]
fn hold_pauses_and_resumes_without_duplicate_or_lost_step() {
    let (plain, _) = run(100_000, None, None);
    let (held, _) = run(100_000, None, Some((30_000, 40_000)));
    let before: Vec<_> = plain.iter().copied().filter(|&p| p < 30_000).collect();
    assert_eq!(
        held.iter()
            .copied()
            .filter(|&p| p < 30_000)
            .collect::<Vec<_>>(),
        before
    );
    assert!(held.iter().all(|&p| !(30_000..40_000).contains(&p)));
    // After the hold the sequence resumes shifted by exactly the held length.
    let after_plain: Vec<_> = plain
        .iter()
        .copied()
        .filter(|&p| (30_000..90_000).contains(&p))
        .collect();
    let after_held: Vec<_> = held
        .iter()
        .copied()
        .filter(|&p| p >= 40_000)
        .map(|p| p - 10_000)
        .collect();
    assert_eq!(&after_held[..after_plain.len()], &after_plain[..]);
}
