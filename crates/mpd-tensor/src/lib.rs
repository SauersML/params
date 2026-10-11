//! Device-resident tensors for parameter decomposition (#2951): the [`tensor`] executor on the
//! host, CUDA and the Apple GPU, and the [`trace`] profiling spans over it. Built on gam-gpu's
//! device runtime.

#[cfg(target_os = "linux")]
mod checked_interval;
#[cfg(target_os = "macos")]
mod metal_stream;
pub mod tensor;
pub mod trace;
