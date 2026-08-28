import Mathlib
import Params.Decomp.Basic

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# Local loss-landscape decomposition: top-k projections and exact recovery

L3D (Chrisman–Bushnaq–Sharkey, arXiv:2504.00194) learns `V^in : n_v × n_w` and
`V^out : n_w × n_v` so that a parameter-gradient `g` is reconstructed by
`V^out Λ V^in g`, where `Λ` keeps the `k` largest coordinates of `V^in g`.

With the selected coordinate set `S` made explicit, the facts the method
relies on are elementary and proved here:

* `selMask S` is an idempotent, symmetric `0/1` diagonal projection;
* `recon_exact`: if `V^in V^out = I` and `g = V^out c` with `c` supported in `S`,
  the reconstruction is exact;
* `recon_smul` / `normLoss_smul`: the reconstruction is linear and the
  normalised loss `‖g − R g‖ / ‖g‖` is invariant under rescaling `g`
  (for a fixed selection);
* `outer_neg_neg`: the sign ambiguity `(−U_i, −V_i)` of every subnetwork;
* `intervention_hasDerivAt`: the "impact" of moving along a direction `v` is
  the directional derivative `∇_W D · v` (chain rule).
-/

namespace Params.Decomp

noncomputable section

open Finset Matrix

section Projection

variable {nw nv : Type*} [Fintype nw] [Fintype nv] [DecidableEq nv]

/-- The `0/1` diagonal projection onto the coordinates in `S`. -/
def selMask (S : Finset nv) : Matrix nv nv ℝ :=
  Matrix.diagonal (fun i => if i ∈ S then 1 else 0)

/-- The selection projection is idempotent. -/
theorem selMask_mul_self (S : Finset nv) : selMask S * selMask S = selMask S := by
  unfold selMask
  rw [Matrix.diagonal_mul_diagonal]
  congr 1
  funext i
  split_ifs <;> norm_num

/-- The selection projection is symmetric. -/
theorem selMask_transpose (S : Finset nv) : (selMask S)ᵀ = selMask S := by
  unfold selMask
  exact Matrix.diagonal_transpose _

/-- The L3D reconstruction of a gradient `g` with selection `S`. -/
def recon (Vout : Matrix nw nv ℝ) (Vin : Matrix nv nw ℝ) (S : Finset nv) (g : nw → ℝ) : nw → ℝ :=
  Vout.mulVec ((selMask S).mulVec (Vin.mulVec g))

/-- **Exact recovery.** If `V^in V^out = I` and `g = V^out c` with `c` supported in `S`,
the reconstruction returns `g`. -/
theorem recon_exact (Vout : Matrix nw nv ℝ) (Vin : Matrix nv nw ℝ) (S : Finset nv)
    (hinv : Vin * Vout = 1) (c : nv → ℝ) (hc : (selMask S).mulVec c = c) :
    recon Vout Vin S (Vout.mulVec c) = Vout.mulVec c := by
  unfold recon
  have h : Vin.mulVec (Vout.mulVec c) = c := by
    rw [Matrix.mulVec_mulVec, hinv, Matrix.one_mulVec]
  rw [h, hc]

/-- A code supported in `S` is fixed by the projection. -/
theorem selMask_mulVec_of_support (S : Finset nv) (c : nv → ℝ) (hc : ∀ i, i ∉ S → c i = 0) :
    (selMask S).mulVec c = c := by
  ext i
  unfold selMask
  rw [Matrix.mulVec_diagonal]
  by_cases hi : i ∈ S
  · rw [if_pos hi, one_mul]
  · rw [if_neg hi, zero_mul, hc i hi]

/-- The reconstruction is linear in `g`. -/
theorem recon_smul (Vout : Matrix nw nv ℝ) (Vin : Matrix nv nw ℝ) (S : Finset nv) (a : ℝ)
    (g : nw → ℝ) : recon Vout Vin S (a • g) = a • recon Vout Vin S g := by
  unfold recon
  rw [Matrix.mulVec_smul, Matrix.mulVec_smul, Matrix.mulVec_smul]

/-- Euclidean norm on `nw → ℝ`. -/
def enorm (u : nw → ℝ) : ℝ := Real.sqrt (∑ i, u i ^ 2)

lemma enorm_smul (a : ℝ) (u : nw → ℝ) : enorm (a • u) = |a| * enorm u := by
  unfold enorm
  have h : ∑ i, (a • u) i ^ 2 = a ^ 2 * ∑ i, u i ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl ?_
    intro i _
    rw [Pi.smul_apply, smul_eq_mul]
    ring
  rw [h, Real.sqrt_mul (sq_nonneg a), Real.sqrt_sq_eq_abs]

/-- The normalised reconstruction loss `‖g − R g‖ / ‖g‖`. -/
def normLoss (Vout : Matrix nw nv ℝ) (Vin : Matrix nv nw ℝ) (S : Finset nv) (g : nw → ℝ) : ℝ :=
  enorm (g - recon Vout Vin S g) / enorm g

/-- The normalised loss is invariant under rescaling the gradient (fixed selection). -/
theorem normLoss_smul (Vout : Matrix nw nv ℝ) (Vin : Matrix nv nw ℝ) (S : Finset nv) (a : ℝ)
    (ha : a ≠ 0) (g : nw → ℝ) : normLoss Vout Vin S (a • g) = normLoss Vout Vin S g := by
  unfold normLoss
  rw [recon_smul, ← smul_sub, enorm_smul, enorm_smul,
    mul_div_mul_left _ _ (abs_ne_zero.mpr ha)]

end Projection

section Sign

variable {m n : Type*} [Fintype m] [Fintype n]

/-- The sign ambiguity of every subnetwork: `(−U)(−V)ᵀ = U Vᵀ`. -/
theorem outer_neg_neg (u : m → ℝ) (v : n → ℝ) : outer (-u) (-v) = outer u v := by
  ext i j
  simp only [outer_apply, Pi.neg_apply, neg_mul_neg]

end Sign

section Intervention

variable {nw : Type*} [Fintype nw]

/-- Moving the parameters along `v` changes a `C¹` read-out, to first order, by the
directional derivative `∇_W F · v` — the quantity L3D calls the impact. -/
theorem intervention_hasDerivAt (F : (nw → ℝ) → ℝ) (hF : ContDiff ℝ 1 F) (W v : nw → ℝ) :
    HasDerivAt (fun δ : ℝ => F (W + δ • v)) (fderiv ℝ F W v) 0 := by
  have hγ : HasDerivAt (fun δ : ℝ => W + δ • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add W
  have h := ((hF.differentiable le_rfl) (W + (0 : ℝ) • v)).hasFDerivAt.comp_hasDerivAt (0 : ℝ) hγ
  simpa using h

end Intervention

end

end Params.Decomp
