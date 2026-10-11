//! Test-only helpers for `gam-linalg`'s own unit tests.
//!
//! The numerical fixtures other crates' tests share live in the dev-only
//! `gam-linalg-test-support` crate. That crate depends on `gam-linalg`, so this
//! crate's unit tests cannot use it (a fixture built against the normal
//! `gam-linalg` returns types that are not the unit-test build's); what those
//! tests need stays here, compiled only under `cfg(test)`.

/// #2738 — serialize every test that MUTATES or OBSERVES faer's process-global
/// parallelism, across every module in the binary.
///
/// `faer::set_global_parallelism` and [`crate::faer_ndarray::FaerSequentialScope`]
/// write one process-wide cell. Two `#[test]`s touching it run on different
/// threads of the same binary, so one test's `Par::rayon(4)` lands inside
/// another's sequential scope and the reader sees a state neither test created
/// — a verdict that depends on which other tests share the process. Holding this
/// lock for the whole body makes the observation belong to the test that made
/// it.
///
/// It lives in `test_support` rather than beside either test module because the
/// participants are in different modules and both must take the SAME lock; a
/// per-module `static` would serialize each module against itself only, which is
/// indistinguishable from no lock at all in exactly the failing case.
pub(crate) fn with_global_parallelism_serialized<T>(body: impl FnOnce() -> T) -> T {
    static GLOBAL_PARALLELISM_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let serialized = GLOBAL_PARALLELISM_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let out = body();
    drop(serialized);
    out
}
