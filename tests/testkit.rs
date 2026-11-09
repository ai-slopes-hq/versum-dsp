#![cfg(feature = "testkit")]
use versum_dsp::testkit::{assert_blockwise_eq, count_allocs, CountingAlloc};
use versum_dsp::Block;

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

#[test]
fn count_allocs_counts_vec_new() {
    let (v, n) = count_allocs(|| Vec::<u8>::with_capacity(8));
    assert_eq!(n, 1);
    drop(v);
}

#[test]
fn count_allocs_zero_for_pure_math() {
    let (r, n) = count_allocs(|| (1..100u32).map(|x| x as f64).sum::<f64>());
    assert!(r > 0.0);
    assert_eq!(n, 0);
}

struct Gain(f32);
impl Block for Gain {
    fn tick(&mut self, x: f32) -> f32 {
        x * self.0
    }
    fn reset(&mut self) {}
}

struct Bad;
impl Block for Bad {
    fn tick(&mut self, x: f32) -> f32 {
        x
    }
    fn reset(&mut self) {}
    fn process(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s += 1.0;
        }
    }
}

#[test]
fn blockwise_helper_accepts_good_block() {
    let input: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    assert_blockwise_eq(|| Gain(0.5), &input, &[1, 7, 16]);
}

#[test]
#[should_panic]
fn blockwise_helper_detects_bad_override() {
    let input = [0.0f32; 16];
    assert_blockwise_eq(|| Bad, &input, &[4]);
}
