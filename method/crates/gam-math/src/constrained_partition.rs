//! Log partition functions of a quadratic coordinate energy on constrained
//! supports.
//!
//! A precision `α` attached to the energy `½ α t²` normalizes to `√(2π/α)` only
//! when `t` ranges over the whole real line. On a bounded interval the partition
//! is that factor times a truncated normal mass. On an embedded unit sphere the
//! energy `Σ_a λ_a x_a²` is integrated against the surface measure, which is a
//! Bingham normalizer: because `Σ_a x_a² = 1` it depends only on the differences
//! of the `λ_a`. Both kernels return the log partition together with the
//! derivatives a hyperparameter gradient needs, in forms that stay finite and
//! resolved across the whole precision range.

use crate::probability::erfcx_nonnegative;
use crate::special::bessel_i0_centered_terms;
use std::f64::consts::{FRAC_1_SQRT_2, LN_2, PI, TAU};

/// Truncated standard normal mass and its log-precision score.
///
/// For `X ~ N(0, 1/α)` restricted to `[lo, hi]`, pass the standardized endpoints
/// `a = lo·√α` and `b = hi·√α` (finite, `a < b`). Returns `(log M, s)` with
/// `M = Φ(b) − Φ(a)` and `s = α·∂ log M/∂α = (b φ(b) − a φ(a)) / (2M)`, so the
/// interval partition `Z(α) = √(2π/α)·M` has `α·∂ log Z/∂α = −½ + s`.
///
/// The mass is formed without cancellation in every regime. An interval that
/// straddles the mode adds two positive half-masses (`erf`). A one-sided interval
/// subtracts two tail areas, using whichever of `erf` and `erfc` is smaller at the
/// near endpoint. The relative rounding of that difference is bounded by the
/// interval's own geometry, `|near endpoint| / (hi − lo)`, and not by `α`. Far in
/// the tail the scaled `erfcx` carries the `e^{−a²/2}` factor in log space.
pub fn normal_interval_log_mass_and_log_precision_score(
    a: f64,
    b: f64,
) -> Result<(f64, f64), String> {
    if !(a.is_finite() && b.is_finite() && a < b) {
        return Err(format!(
            "truncated normal mass needs finite standardized endpoints a < b; got a={a}, b={b}"
        ));
    }
    // The mass and the score are both invariant under `(a, b) → (−b, −a)`.
    let (a, b) = if b <= 0.0 { (-b, -a) } else { (a, b) };
    let inv_sqrt_tau = TAU.sqrt().recip();
    let (log_mass, score) = if a < 0.0 {
        let mass = 0.5 * (libm::erf(b * FRAC_1_SQRT_2) + libm::erf(-a * FRAC_1_SQRT_2));
        let numerator = b * (-0.5 * b * b).exp() - a * (-0.5 * a * a).exp();
        (mass.ln(), 0.5 * inv_sqrt_tau * numerator / mass)
    } else {
        let a_s = a * FRAC_1_SQRT_2;
        let b_s = b * FRAC_1_SQRT_2;
        if libm::erf(a_s) <= libm::erfc(a_s) {
            let mass = 0.5 * (libm::erf(b_s) - libm::erf(a_s));
            let numerator = b * (-0.5 * b * b).exp() - a * (-0.5 * a * a).exp();
            (mass.ln(), 0.5 * inv_sqrt_tau * numerator / mass)
        } else {
            // `M = ½ e^{−a_s²} [erfcx(a_s) − e^{−Δ} erfcx(b_s)]` with
            // `Δ = b_s² − a_s²`, so `φ(a)/M = 2 / (√(2π)·bracket)` and
            // `φ(b)/M = e^{−Δ}·φ(a)/M`.
            let decay = (-(b_s - a_s) * (b_s + a_s)).exp();
            let bracket = erfcx_nonnegative(a_s) - decay * erfcx_nonnegative(b_s);
            (
                (0.5 * bracket).ln() - a_s * a_s,
                inv_sqrt_tau * (b * decay - a) / bracket,
            )
        }
    };
    if !(log_mass.is_finite() && score.is_finite()) {
        return Err(format!(
            "truncated normal mass on standardized [{a}, {b}] is not representable: \
             log mass {log_mass}, score {score}"
        ));
    }
    Ok((log_mass, score))
}

/// Bingham normalizer on the unit circle or the unit sphere, with its second
/// moments.
///
/// For `λ ∈ R^d`, `d ∈ {2, 3}`, returns `(log Z, E[x_a²])`, where
/// `Z(λ) = ∫_{S^{d−1}} exp(−Σ_a λ_a x_a²) dσ(x)` is taken against the surface
/// measure (`|S¹| = 2π`, `|S²| = 4π`) and `E` is the normalized law, so
/// `∂ log Z/∂λ_a = −E[x_a²]`. Because `Σ_a x_a² = 1`, replacing `λ` by `λ + c·1`
/// multiplies `Z` by `e^{−c}` and leaves the law unchanged. Both kernels work on
/// the shifted `λ − min λ`, so that invariance holds to rounding.
///
/// `S¹` is closed form: `Z = 2π e^{−λ_min} e^{−η} I0(η)` with `η = (λ_max − λ_min)/2`.
///
/// On `S²`, put `z` on the axis of smallest `λ`, and let `μ₁ ≥ μ₂ ≥ 0` be the other
/// two shifted values. Integrating the azimuth in closed form, with `s = 1 − z²`
/// and `δ = (μ₁ − μ₂)/2`, gives
///
/// ```text
///   Z = e^{−λ_min} · 4π ∫₀¹ e^{−sμ₂} Ĩ0(sδ) dz,    Ĩν(y) = e^{−y} Iν(y),
/// ```
///
/// and the three second moments carry `z²`, `s(Ĩ0 − Ĩ1)/2` (the `μ₁` axis) and
/// `s(Ĩ0 + Ĩ1)/2` (the `μ₂` axis) in place of `Ĩ0`.
///
/// Substituting `1 − z = u = 1/(1 + e^{−x})`, each integrand becomes `u(1 − u)`
/// times a factor that is analytic in the strip `|Im x| < π/2`. There
/// `Re s ≥ 0` and `|1 − u| ≤ 1`, so the factor is bounded by `2`, and `u(1 − u)`
/// has line integral at most `π/2`. The trapezoidal rule with step `h` is
/// therefore in error by at most `2π / (e^{π²/h} − 1)` for every `μ`
/// (Trefethen & Weideman 2014, Theorem 5.1). Dropping the nodes with `|x| > X`
/// costs at most `2e^{−X}`, since `u ≤ e^x` and `1 − u ≤ e^{−x}`. Both errors
/// are set to `ε·L / (1 + 2μ₁)`:
///
/// * `L = (1 − e^{−(μ₁+μ₂)}) / (μ₁ + μ₂)` bounds the integral from below
///   (`Ĩ0(y) ≥ e^{−y}` and `s ≤ 2u`);
/// * `1 / (1 + 2μ₁)` is the order of the smallest second moment (the tangent
///   variance along the most concentrated axis).
///
/// The rule is fixed by these bounds before any integrand is evaluated. The node
/// count grows only like `log²` of the concentration.
pub fn bingham_sphere_log_partition_and_second_moments(
    lambda: &[f64],
) -> Result<(f64, Vec<f64>), String> {
    if let Some(bad) = lambda.iter().find(|value| !value.is_finite()) {
        return Err(format!(
            "Bingham normalizer needs finite energy coefficients; got {bad} in {lambda:?}"
        ));
    }
    let (log_z, moments) = match lambda.len() {
        2 => bingham_circle(lambda[0], lambda[1]),
        3 => bingham_sphere_s2([lambda[0], lambda[1], lambda[2]]),
        dim => {
            return Err(format!(
                "Bingham normalizer is implemented on S^1 and S^2 (ambient dimension 2 or 3); \
                 got ambient dimension {dim}"
            ));
        }
    };
    if !log_z.is_finite() || moments.iter().any(|moment| !moment.is_finite()) {
        return Err(format!(
            "Bingham normalizer at {lambda:?} is not representable: log Z {log_z}, \
             second moments {moments:?}"
        ));
    }
    Ok((log_z, moments))
}

fn bingham_circle(first: f64, second: f64) -> (f64, Vec<f64>) {
    let lambda = [first, second];
    let (largest, smallest) = if first >= second { (0, 1) } else { (1, 0) };
    let eta = 0.5 * (lambda[largest] - lambda[smallest]);
    let (centered_log_i0, _, scaled_derivative) = bessel_i0_centered_terms(eta);
    // `E[x_largest²] = ½(1 − I1/I0)`; `η·(I1/I0 − 1)` is resolved directly.
    let one_minus_ratio = if eta > 0.0 {
        -scaled_derivative / eta
    } else {
        1.0
    };
    let mut moments = vec![0.0; 2];
    moments[largest] = 0.5 * one_minus_ratio;
    moments[smallest] = 1.0 - moments[largest];
    (TAU.ln() - lambda[smallest] + centered_log_i0, moments)
}

fn bingham_sphere_s2(lambda: [f64; 3]) -> (f64, Vec<f64>) {
    let mut order = [0_usize, 1, 2];
    order.sort_by(|&i, &j| lambda[j].total_cmp(&lambda[i]));
    let [largest, middle, smallest] = order;
    let mu1 = lambda[largest] - lambda[smallest];
    let mu2 = lambda[middle] - lambda[smallest];
    let delta = 0.5 * (mu1 - mu2);
    let spread = mu1 + mu2;
    let log_lower_bound = if spread > 0.0 {
        (-(-spread).exp_m1()).ln() - spread.ln()
    } else {
        0.0
    };
    let log_moment_scale = if mu1 > 1.0 {
        LN_2 + mu1.ln() + (0.5 / mu1).ln_1p()
    } else {
        (2.0 * mu1).ln_1p()
    };
    let log_tolerance = f64::EPSILON.ln() + log_lower_bound - log_moment_scale;
    // `h = π² / ln(1 + 2π/tol)`, with the logarithm formed from `ln tol`.
    let log_ratio = TAU.ln() - log_tolerance;
    let step = PI * PI / (log_ratio + (-log_ratio).exp().ln_1p());
    // `2e^{−X} = tol`.
    let cutoff = LN_2 - log_tolerance;
    let last = (cutoff / step).ceil() as i64;
    // Integrals of `Ĩ0`, and of the second-moment weights of the largest,
    // middle and smallest axes.
    let mut sums = [0.0_f64; 4];
    for k in -last..=last {
        let x = k as f64 * step;
        let (u, v) = if x >= 0.0 {
            let e = (-x).exp();
            (1.0 / (1.0 + e), e / (1.0 + e))
        } else {
            let e = x.exp();
            (e / (1.0 + e), 1.0 / (1.0 + e))
        };
        // `u = 1 − z`, `v = z`, `s = 1 − z² = u(1 + v)`.
        let s = u * (1.0 + v);
        let y = s * delta;
        let (centered_log_i0, _, scaled_derivative) = bessel_i0_centered_terms(y);
        let one_minus_ratio = if y > 0.0 {
            -scaled_derivative / y
        } else {
            1.0
        };
        let base = step * u * v * (-s * mu2).exp() * centered_log_i0.exp();
        sums[0] += base;
        sums[1] += base * 0.5 * s * one_minus_ratio;
        sums[2] += base * 0.5 * s * (2.0 - one_minus_ratio);
        sums[3] += base * v * v;
    }
    let integral = sums[0];
    let mut moments = vec![0.0; 3];
    moments[largest] = sums[1] / integral;
    moments[middle] = sums[2] / integral;
    moments[smallest] = sums[3] / integral;
    (LN_2 + TAU.ln() - lambda[smallest] + integral.ln(), moments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::gauss_legendre;

    /// `log Z` and `E[x_a²]` on `S²` by an independent product rule: Gauss–Legendre
    /// in `z = x₃` and the periodic trapezoid in the azimuth.
    fn product_rule_s2(lambda: [f64; 3]) -> (f64, [f64; 3]) {
        let (nodes, weights) = gauss_legendre(160);
        let azimuths = 512;
        let mut z_sum = 0.0;
        let mut moment_sums = [0.0_f64; 3];
        for (&z, &wz) in nodes.iter().zip(weights.iter()) {
            let r = (1.0 - z * z).sqrt();
            for j in 0..azimuths {
                let phi = TAU * j as f64 / azimuths as f64;
                let x = [r * phi.cos(), r * phi.sin(), z];
                let energy: f64 = (0..3).map(|a| lambda[a] * x[a] * x[a]).sum();
                let w = wz * (TAU / azimuths as f64) * (-energy).exp();
                z_sum += w;
                for a in 0..3 {
                    moment_sums[a] += w * x[a] * x[a];
                }
            }
        }
        (z_sum.ln(), moment_sums.map(|m| m / z_sum))
    }

    #[test]
    fn bingham_s2_matches_an_independent_product_rule() {
        for lambda in [[3.7, 1.2, -0.4], [0.3, 0.1, 0.0], [12.0, 0.5, 7.0]] {
            let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&lambda).expect("the partition kernel evaluates on a finite fixture");
            let (reference_log_z, reference_moments) = product_rule_s2(lambda);
            assert!(
                (log_z - reference_log_z).abs() < 1e-12,
                "log Z at {lambda:?}: kernel {log_z}, product rule {reference_log_z}"
            );
            for a in 0..3 {
                assert!(
                    (moments[a] - reference_moments[a]).abs() < 1e-12,
                    "E[x_{a}²] at {lambda:?}: kernel {}, product rule {}",
                    moments[a],
                    reference_moments[a]
                );
            }
        }
    }

    #[test]
    fn bingham_trace_shift_and_isotropy_are_exact() {
        for c in [-3.3, 0.0, 5.1] {
            let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&[c, c, c]).expect("the partition kernel evaluates on a finite fixture");
            assert!((log_z - (LN_2 + TAU.ln() - c)).abs() < 1e-14, "isotropic S² log Z {log_z}");
            for m in moments {
                assert!((m - 1.0 / 3.0).abs() < 1e-14, "isotropic S² moment {m}");
            }
            let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&[c, c]).expect("the partition kernel evaluates on a finite fixture");
            assert!((log_z - (TAU.ln() - c)).abs() < 1e-14, "isotropic S¹ log Z {log_z}");
            for m in moments {
                assert!((m - 0.5).abs() < 1e-15, "isotropic S¹ moment {m}");
            }
        }
        for base in [vec![2.3, -0.7, 0.4], vec![1.9, -2.6]] {
            let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&base).expect("the partition kernel evaluates on a finite fixture");
            for c in [-3.3, 5.1] {
                let shifted: Vec<f64> = base.iter().map(|v| v + c).collect();
                let (shifted_log_z, shifted_moments) =
                    bingham_sphere_log_partition_and_second_moments(&shifted).expect("the partition kernel evaluates on a finite fixture");
                assert!(
                    (shifted_log_z - (log_z - c)).abs() < 1e-12,
                    "trace shift {c} of {base:?}: log Z {shifted_log_z} vs {log_z} - c"
                );
                for (m, shifted_m) in moments.iter().zip(shifted_moments.iter()) {
                    assert!((m - shifted_m).abs() < 1e-13, "trace shift moved a moment: {m} vs {shifted_m}");
                }
            }
        }
    }

    #[test]
    fn bingham_s1_matches_the_trapezoid_rule() {
        let lambda = [3.0, -1.0];
        let nodes = 256;
        let mut z_sum = 0.0;
        let mut first_moment = 0.0;
        for j in 0..nodes {
            let theta = TAU * j as f64 / nodes as f64;
            let (c, s) = (theta.cos(), theta.sin());
            let w = (TAU / nodes as f64) * (-(lambda[0] * c * c + lambda[1] * s * s)).exp();
            z_sum += w;
            first_moment += w * c * c;
        }
        let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&lambda).expect("the partition kernel evaluates on a finite fixture");
        assert!((log_z - z_sum.ln()).abs() < 1e-13, "S¹ log Z {log_z} vs {}", z_sum.ln());
        assert!((moments[0] - first_moment / z_sum).abs() < 1e-13);
        assert!((moments[1] - (1.0 - first_moment / z_sum)).abs() < 1e-13);
    }

    /// Girdle case `λ = (0, 0, μ)`: `Z = 2π √(π/μ) erf(√μ)` and
    /// `E[x₃²] = 1/(2μ) − e^{−μ} / (√(πμ) erf(√μ))`.
    #[test]
    fn bingham_s2_girdle_matches_the_closed_form() {
        for mu in [1.0e-6, 1.0, 40.0, 1.0e4, 1.0e8] {
            let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&[0.0, 0.0, mu]).expect("the partition kernel evaluates on a finite fixture");
            let root = mu.sqrt();
            let reference_log_z = TAU.ln() + 0.5 * (PI / mu).ln() + libm::erf(root).ln();
            assert!(
                (log_z - reference_log_z).abs() < 1e-12,
                "girdle μ={mu}: log Z {log_z}, closed form {reference_log_z}"
            );
            if mu >= 1.0 {
                let polar = 0.5 / mu - (-mu).exp() / ((PI * mu).sqrt() * libm::erf(root));
                assert!(
                    (moments[2] - polar).abs() < 1e-13 * polar.max(1e-3),
                    "girdle μ={mu}: E[x₃²] {}, closed form {polar}",
                    moments[2]
                );
                assert!((moments[0] - 0.5 * (1.0 - polar)).abs() < 1e-13);
            }
        }
    }

    /// Bipolar case `λ = (μ, μ, 0)`: `Z = 4π F(√μ)/√μ` with Dawson's `F`, whose
    /// asymptotic series gives `Z = (2π/μ)·S(μ)`,
    /// `S = 1 + 1/(2μ) + 3/(4μ²) + 15/(8μ³) + O(μ⁻⁴)`.
    #[test]
    fn bingham_s2_bipolar_concentration_matches_the_dawson_asymptote() {
        for mu in [1.0e4, 1.0e6] {
            let series = 1.0 + 0.5 / mu + 0.75 / (mu * mu) + 1.875 / (mu * mu * mu);
            let series_slope = -0.5 / (mu * mu) - 1.5 / (mu * mu * mu) - 5.625 / (mu * mu * mu * mu);
            let reference_log_z = (TAU / mu).ln() + series.ln();
            // `E[x₁²] + E[x₂²] = −∂ log Z/∂μ = 1/μ − S′/S`.
            let reference_tangent = 0.5 * (1.0 / mu - series_slope / series);
            let (log_z, moments) = bingham_sphere_log_partition_and_second_moments(&[mu, mu, 0.0]).expect("the partition kernel evaluates on a finite fixture");
            assert!(
                (log_z - reference_log_z).abs() < 1e-12,
                "bipolar μ={mu}: log Z {log_z}, asymptote {reference_log_z}"
            );
            for a in 0..2 {
                assert!(
                    (moments[a] - reference_tangent).abs() < 1e-11 * reference_tangent,
                    "bipolar μ={mu}: E[x_{a}²] {}, asymptote {reference_tangent}",
                    moments[a]
                );
            }
        }
    }

    #[test]
    fn bingham_second_moments_are_the_log_partition_gradient() {
        let lambda = [2.1, -0.3, 0.9];
        let (_, moments) = bingham_sphere_log_partition_and_second_moments(&lambda).expect("the partition kernel evaluates on a finite fixture");
        let step = 1e-5;
        for a in 0..3 {
            let mut plus = lambda;
            let mut minus = lambda;
            plus[a] += step;
            minus[a] -= step;
            let (log_plus, _) = bingham_sphere_log_partition_and_second_moments(&plus).expect("the partition kernel evaluates on a finite fixture");
            let (log_minus, _) = bingham_sphere_log_partition_and_second_moments(&minus).expect("the partition kernel evaluates on a finite fixture");
            let slope = (log_plus - log_minus) / (2.0 * step);
            assert!(
                (slope + moments[a]).abs() < 1e-8,
                "∂ log Z/∂λ_{a} = {slope}, −E[x_{a}²] = {}",
                -moments[a]
            );
        }
    }

    /// `(log ∫_a^b φ, ½ ∫_a^b (1 − s²) φ / ∫_a^b φ)` by Gauss–Legendre, using
    /// `b φ(b) − a φ(a) = ∫_a^b (1 − s²) φ(s) ds`. A one-sided interval factors
    /// `φ(near)` out analytically.
    fn interval_reference(a: f64, b: f64) -> (f64, f64) {
        let (nodes, weights) = gauss_legendre(400);
        let integrate = |lo: f64, hi: f64, shift: f64| -> (f64, f64) {
            let half = 0.5 * (hi - lo);
            let mid = 0.5 * (hi + lo);
            let mut mass = 0.0;
            let mut curvature = 0.0;
            for (&node, &weight) in nodes.iter().zip(weights.iter()) {
                let t = mid + half * node;
                let s = shift + t;
                let density = (-(shift * t + 0.5 * t * t)).exp();
                mass += half * weight * density;
                curvature += half * weight * density * (1.0 - s * s);
            }
            (mass, curvature)
        };
        let log_phi = |x: f64| -0.5 * x * x - 0.5 * TAU.ln();
        if a < 0.0 && b > 0.0 {
            let (left_mass, left_curvature) = integrate(a.max(-12.0), 0.0, 0.0);
            let (right_mass, right_curvature) = integrate(0.0, b.min(12.0), 0.0);
            let mass = left_mass + right_mass;
            (
                mass.ln() + log_phi(0.0),
                0.5 * (left_curvature + right_curvature) / mass,
            )
        } else {
            let (near, far) = if a >= 0.0 { (a, b) } else { (-b, -a) };
            let (mass, curvature) = integrate(0.0, far - near, near);
            (mass.ln() + log_phi(near), 0.5 * curvature / mass)
        }
    }

    #[test]
    fn interval_mass_and_score_match_quadrature_across_regimes() {
        for (a, b) in [
            (-1.0, 1.0),
            (-3.0, 0.25),
            (0.2, 0.9),
            (-0.9, -0.2),
            (1.5, 4.0),
            (30.0, 31.0),
            (-2.0e-9, 1.0e-9),
            (1.0e-9, 3.0e-9),
            (5.0, 5.001),
            (-40.0, 40.0),
        ] {
            let (log_mass, score) = normal_interval_log_mass_and_log_precision_score(a, b).expect("the partition kernel evaluates on a finite fixture");
            let (reference_log_mass, reference_score) = interval_reference(a, b);
            assert!(
                (log_mass - reference_log_mass).abs() < 1e-11,
                "[{a}, {b}]: log mass {log_mass}, quadrature {reference_log_mass}"
            );
            assert!(
                (score - reference_score).abs() < 1e-11 * reference_score.abs().max(1.0),
                "[{a}, {b}]: score {score}, quadrature {reference_score}"
            );
        }
    }

    /// #2933 F24: on `[−1, 1]` at `α = 1`, `∂ log Z/∂α = −0.145563`, not the full
    /// line's `−½`. The closed form is `−½ + e^{−½} / (√(2π) erf(1/√2))`.
    #[test]
    fn interval_log_partition_derivative_matches_the_audit_counterexample() {
        let (_, score) = normal_interval_log_mass_and_log_precision_score(-1.0, 1.0).expect("the partition kernel evaluates on a finite fixture");
        let closed_form = -0.5 + (-0.5_f64).exp() / (TAU.sqrt() * libm::erf(FRAC_1_SQRT_2));
        assert!((-0.5 + score - closed_form).abs() < 1e-15, "score {score}, closed form {closed_form}");
        assert!((-0.5 + score + 0.145562547388).abs() < 1e-11);
    }

    #[test]
    fn interval_mass_reaches_its_small_and_large_precision_limits() {
        // `α → 0`: `M → (hi − lo)√α φ(0)` and the score tends to `½`, so the
        // partition tends to the interval length and its derivative to zero.
        let root = 1.0e-10;
        let (log_mass, score) = normal_interval_log_mass_and_log_precision_score(-root, 2.0 * root).expect("the partition kernel evaluates on a finite fixture");
        assert!((log_mass - ((3.0 * root).ln() - 0.5 * TAU.ln())).abs() < 1e-14);
        assert!((score - 0.5).abs() < 1e-14);
        // `α → ∞` around the mode: `M → 1` and the score vanishes (the line).
        let (log_mass, score) = normal_interval_log_mass_and_log_precision_score(-1.0e6, 2.0e6).expect("the partition kernel evaluates on a finite fixture");
        assert!(log_mass.abs() < 1e-15 && score.abs() < 1e-15);
        // `α → ∞` off the mode: the tail `Q(a) = φ(a)/a·(1 − 1/a² + 3/a⁴ − …)` and
        // `a φ(a)/Q(a) = a² + 1 − 2/a² + 10/a⁴ − …`.
        let a = 500.0_f64;
        let (log_mass, score) = normal_interval_log_mass_and_log_precision_score(a, 4.0 * a).expect("the partition kernel evaluates on a finite fixture");
        let reference_log_mass = -0.5 * a * a - (a * TAU.sqrt()).ln()
            + (1.0 - 1.0 / (a * a) + 3.0 / a.powi(4) - 15.0 / a.powi(6)).ln();
        let reference_score = -0.5 * (a * a + 1.0 - 2.0 / (a * a) + 10.0 / a.powi(4));
        assert!((log_mass - reference_log_mass).abs() < 1e-9 * reference_log_mass.abs());
        assert!((score - reference_score).abs() < 1e-12 * reference_score.abs());
    }
}
