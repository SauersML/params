import Mathlib
import Params.Transformer.Attention

/-!
# QK and OV circuits: what the attention weights actually depend on

Elhage et al. (*A Mathematical Framework for Transformer Circuits*, 2021)
observe that an attention head is determined by two low-rank products, the
**QK circuit** `W^Q (W^K)ᵀ` and the **OV circuit** `W^V W^O`, rather than by the
four matrices separately.  This file proves the weight-level statements behind
that observation:

* `score_eq_qk`: the score matrix is `u (W^Q (W^K)ᵀ) uᵀ`;
* `attnWeights_congr_qk`: two heads with the same QK circuit have the same
  attention pattern on every input — so `W^Q`, `W^K` are not identifiable
  beyond their product (`attnWeights_gauge`: any invertible `R` in
  `(W^Q R, W^K R⁻ᵀ)` is invisible);
* `headOut_ov`: the head's contribution is `A · u (W^V W^O)`, so `W^V`, `W^O` enter
  only through their product (`headOut_congr_ov`).
-/

namespace Params

noncomputable section

open Finset Matrix

variable {n d e : Type*} [Fintype n] [Fintype d] [Fintype e]

/-- The score matrix `s • (u W^Q)(u W^K)ᵀ` equals `s • u (W^Q (W^K)ᵀ) uᵀ`. -/
theorem score_eq_qk (s : ℝ) (WQ WK : Matrix d e ℝ) (u : Matrix n d ℝ) :
    s • ((u * WQ) * (u * WK)ᵀ) = s • (u * (WQ * WKᵀ) * uᵀ) := by
  rw [Matrix.transpose_mul, Matrix.mul_assoc, Matrix.mul_assoc, Matrix.mul_assoc]

/-- Heads with the same QK circuit have the same attention pattern on every input. -/
theorem attnWeights_congr_qk (s : ℝ) (WQ WK WQ' WK' : Matrix d e ℝ)
    (h : WQ * WKᵀ = WQ' * WK'ᵀ) (u : Matrix n d ℝ) :
    attnWeights s (u * WQ) (u * WK) = attnWeights s (u * WQ') (u * WK') := by
  unfold attnWeights
  rw [score_eq_qk, score_eq_qk, h]

/-- The gauge freedom of the QK circuit: `(W^Q R, W^K R⁻ᵀ)` has the same circuit. -/
theorem qk_gauge [DecidableEq e] (WQ WK : Matrix d e ℝ) (R : Matrix e e ℝ) (hR : IsUnit R) :
    (WQ * R) * (WK * R⁻¹ᵀ)ᵀ = WQ * WKᵀ := by
  rw [Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc,
    ← Matrix.mul_assoc R, Matrix.mul_nonsing_inv R hR, Matrix.one_mul]

/-- Consequently the attention pattern is invariant under that gauge. -/
theorem attnWeights_gauge [DecidableEq e] (s : ℝ) (WQ WK : Matrix d e ℝ) (R : Matrix e e ℝ)
    (hR : IsUnit R) (u : Matrix n d ℝ) :
    attnWeights s (u * (WQ * R)) (u * (WK * R⁻¹ᵀ)) = attnWeights s (u * WQ) (u * WK) :=
  attnWeights_congr_qk s _ _ _ _ (qk_gauge WQ WK R hR) u

/-- The head's contribution is `A · (u (W^V W^O))`: `W^V` and `W^O` enter only via their product. -/
theorem headOut_ov (s : ℝ) (Q K : Matrix n e ℝ) (WV : Matrix d e ℝ) (WO : Matrix e d ℝ)
    (u : Matrix n d ℝ) :
    attention s Q K (u * WV) * WO = attnWeights s Q K * (u * (WV * WO)) := by
  unfold attention
  rw [Matrix.mul_assoc, Matrix.mul_assoc]

/-- Heads with the same OV circuit make the same contribution for the same pattern. -/
theorem headOut_congr_ov (s : ℝ) (Q K : Matrix n e ℝ) (WV WV' : Matrix d e ℝ) (WO WO' : Matrix e d ℝ)
    (h : WV * WO = WV' * WO') (u : Matrix n d ℝ) :
    attention s Q K (u * WV) * WO = attention s Q K (u * WV') * WO' := by
  rw [headOut_ov, headOut_ov, h]

end

end Params
