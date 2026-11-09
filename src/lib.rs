//! versum-dsp: small DSP building blocks. Audio is f32, phases and accumulators are f64.

pub mod biquad;
pub mod blep;
pub mod block;
pub mod dc;
pub mod delay;
pub mod edge;
pub mod knee;
pub mod lfo;
pub mod lpg;
pub mod onepole;
pub mod pan;
pub mod ramp;
pub mod rng;
pub mod shaper;
pub mod vco;
#[cfg(test)]
mod measure;
#[cfg(feature = "testkit")]
pub mod testkit;

pub use block::Block;
pub use rng::Rng;
