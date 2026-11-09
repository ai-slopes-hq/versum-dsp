//! versum-dsp: small DSP building blocks. Audio is f32, phases and accumulators are f64.

pub mod block;
pub mod dc;
pub mod delay;
pub mod edge;
pub mod knee;
pub mod onepole;
pub mod pan;
pub mod ramp;
pub mod rng;
#[cfg(feature = "testkit")]
pub mod testkit;

pub use block::Block;
pub use rng::Rng;
