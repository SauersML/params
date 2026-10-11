//! Report one paired speed measurement the way every #932 gate reports it.
//!
//! Times the compiled order-four curve/wiggle pullback bundle
//! (`fast_channel::curve_wiggle_bundle4`, the sparse build-time expansion of
//! nine top channels) against the nine channels evaluated separately by the
//! universal partition sum (`fast_channel::faa_top4`, exactly the canonical
//! oracle the bundle is pinned to), with `paired_timing`: interleaved per
//! repetition, order randomised, ratios kept per repetition, and the summary
//! line carrying `median_ratio`, `wins`, `resolution` and `position_bias`.
//! Run it on a host to see the instrument's resolution there before reading
//! any gate's verdict from that host:
//!
//! ```text
//! cargo run --release -p gam-math --example paired_timing_report
//! ```
//!
//! This example is also the harness's reachability root: `paired_timing` is
//! linked by no shipped artifact (tests measure with it), and a kept example
//! is what a dead-code sweep keyed on artifact symbol tables honours.

use gam_math::fast_channel::{curve_wiggle_bundle4, faa_top4};
use gam_math::paired_timing::{SpeedGate, batched, paired_interleaved};

#[inline(never)]
fn compiled_bundle4(x: [f64; 17]) -> [f64; 9] {
    curve_wiggle_bundle4(
        [x[0], x[1], x[2], x[3]],
        [x[4], x[5], x[6]],
        [x[7], x[9], x[10], x[11]],
        [x[8], x[12], x[13], x[14]],
        [x[15], x[16]],
    )
}

/// The nine channels as nine evaluations of the universal rule: the canonical
/// form the bundle's oracle pins it to.
#[inline(never)]
fn canonical_bundle4(x: [f64; 17]) -> [f64; 9] {
    let [
        m1,
        m2,
        m3,
        m4,
        q_u,
        q_v,
        q_uv,
        a,
        b,
        a_u,
        a_v,
        a_uv,
        b_u,
        b_v,
        b_uv,
        xi_u,
        xi_v,
    ] = x;
    let xi_uv = xi_u * xi_v;
    let q = [
        [
            0.0, a, a, b, q_u, a_u, a_u, b_u, q_v, a_v, a_v, b_v, q_uv, a_uv, a_uv, b_uv,
        ],
        [
            0.0, a, 1.0, 0.0, q_u, a_u, 0.0, 0.0, q_v, a_v, 0.0, 0.0, q_uv, a_uv, 0.0, 0.0,
        ],
        [
            0.0, a, 0.0, 1.0, q_u, a_u, xi_u, 0.0, q_v, a_v, xi_v, 0.0, q_uv, 0.0, 0.0, 0.0,
        ],
        [
            0.0, a, 0.0, 0.0, q_u, 0.0, 0.0, xi_u, q_v, 0.0, 0.0, xi_v, 0.0, 0.0, xi_uv, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, xi_uv,
        ],
        [
            0.0, 1.0, 1.0, 0.0, q_u, 0.0, 0.0, 0.0, q_v, 0.0, 0.0, 0.0, q_uv, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 1.0, 0.0, q_u, xi_u, 0.0, 0.0, q_v, xi_v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, xi_uv, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 0.0, 0.0, 0.0, xi_u, xi_u, 0.0, 0.0, xi_v, xi_v, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    ];
    std::array::from_fn(|output| faa_top4([m1, m2, m3, m4], &q[output]))
}

fn main() {
    let input: [f64; 17] = std::array::from_fn(|index| 0.35 + 0.07 * (index as f64 + 1.0).sin());
    for (compiled, canonical) in compiled_bundle4(input).iter().zip(canonical_bundle4(input).iter()) {
        assert!((compiled - canonical).abs() <= 1e-12 * canonical.abs().max(1.0));
    }
    let timing = paired_interleaved(
        15,
        2_000,
        0x9320_AB,
        batched(64, |nudge| {
            let mut x = input;
            x[4] += nudge;
            compiled_bundle4(x).iter().sum()
        }),
        batched(64, |nudge| {
            let mut x = input;
            x[4] += nudge;
            canonical_bundle4(x).iter().sum()
        }),
    );
    println!("{}", timing.summary("compiled_bundle4", "nine_top_channels"));
    // The two contracts a gate can carry, printed the way a gate prints them.
    let mut gate = SpeedGate::open("PAIRED-TIMING-REPORT");
    gate.faster("order=4 bundle", &timing, "compiled_bundle4", "nine_top_channels");
    gate.not_slower("order=4 bundle", &timing, "compiled_bundle4", "nine_top_channels");
    gate.finish();
}
