//! Test helpers (feature `testkit`): allocation counting and block-wise equivalence.
use crate::Block;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Global allocator wrapper that counts `alloc` calls per thread.
pub struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

/// Runs `f` and returns its result with the number of allocations made on this thread.
/// Requires `CountingAlloc` to be the `#[global_allocator]`.
pub fn count_allocs<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOCS.with(|c| c.get());
    let r = f();
    let after = ALLOCS.with(|c| c.get());
    (r, after - before)
}

/// Asserts that `process` over chunks of the given sizes (cycled) equals per-sample `tick`, bit for bit.
pub fn assert_blockwise_eq<B: Block>(make: impl Fn() -> B, input: &[f32], chunks: &[usize]) {
    let mut a = make();
    let expected: Vec<f32> = input.iter().map(|&x| a.tick(x)).collect();
    for &chunk in chunks {
        assert!(chunk > 0, "chunk size must be > 0");
        let mut b = make();
        let mut out = input.to_vec();
        for part in out.chunks_mut(chunk) {
            b.process(part);
        }
        for (i, (g, e)) in out.iter().zip(&expected).enumerate() {
            assert!(g.to_bits() == e.to_bits(), "chunk {chunk}: sample {i}: {g} != {e}");
        }
    }
}
