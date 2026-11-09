//! versum-dsp: small DSP building blocks. Audio is f32, phases and accumulators are f64.

pub mod block;
pub mod rng;
#[cfg(feature = "testkit")]
pub mod testkit;

pub use block::Block;
pub use rng::Rng;
