import Mathlib
import Params.Decomp.Basic

/-!
# Targeted parameter decomposition: the catch-all component

Targeted PD (Vigouroux–Sharkey, arXiv:2607.13047, eq. (1)) reconstructs a
weight as masked rank-one subcomponents plus a masked *catch-all* residual
`Δ = W − U V`:

  `W' = Σ_c m_c U_c V_cᵀ + m_Δ (W − Σ_c U_c V_cᵀ)`.

Proved here, from the definitions alone:

* the two identities the method relies on — all masks `1` gives `W`
  (`tpdMasked_one_one`), masks `1` with `m_Δ = 0` gives the plain
  reconstruction (`tpdMasked_one_zero`) — and the affine rearrangement in the
  masks (`tpdMasked_eq`);
* the adversarial masks of eq. (2) exist: a continuous reconstruction loss
  attains its maximum on the compact mask box `[μ, 1]`
  (`exists_adversarial_mask`);
* the *underdetermination* argument of the paper's Section 2 / Appendix B: a
  loss that sees an input direction `V` only through the activations `V ⬝ x`
  on a set `S` of inputs cannot distinguish `V` from `V + w` for any `w`
  orthogonal to `S` (`loss_invariant_of_orthogonal`);
* the *merging* observation of Appendix C: two subcomponents whose inner
  activations are constant on the target inputs act there like a single
  rank-one map (`merge_of_const_activations`, `merge_rank_one`);
* non-identifiability of rank-one splittings of a rank-two matrix
  (`outer_add_outer_eq`).
-/

namespace Params.Decomp

noncomputable section

open Finset Matrix

section CatchAll

variable {m n C : Type*} [Fintype m] [Fintype n] [Fintype C] [DecidableEq C]

/-- The tPD reconstruction with subcomponent masks `mask` and catch-all mask `mΔ`. -/
def tpdMasked (W : Matrix m n ℝ) (D : Subcomponents m n C) (mask : C → ℝ) (mΔ : ℝ) :
    Matrix m n ℝ :=
  D.masked mask + mΔ • (W - D.weight)

/-- All masks `1`: the original weight. -/
theorem tpdMasked_one_one (W : Matrix m n ℝ) (D : Subcomponents m n C) :
    tpdMasked W D (fun _ => 1) 1 = W := by
  unfold tpdMasked
  rw [masked_one, one_smul]
  abel

/-- Subcomponent masks `1`, catch-all `0`: the plain reconstruction `U V`. -/
theorem tpdMasked_one_zero (W : Matrix m n ℝ) (D : Subcomponents m n C) :
    tpdMasked W D (fun _ => 1) 0 = D.weight := by
  unfold tpdMasked
  rw [masked_one, zero_smul, add_zero]

/-- The reconstruction is affine in the masks:
`W' = W + Σ_c (m_c − m_Δ) • comp_c − (1 − m_Δ) • W`. -/
theorem tpdMasked_eq (W : Matrix m n ℝ) (D : Subcomponents m n C) (mask : C → ℝ) (mΔ : ℝ) :
    tpdMasked W D mask mΔ
      = W + ∑ c, (mask c - mΔ) • D.component c - (1 - mΔ) • W := by
  unfold tpdMasked
  rw [masked_eq_sum, weight_eq_sum_components, smul_sub, Finset.smul_sum]
  simp only [sub_smul, Finset.sum_sub_distrib, one_smul]
  abel

/-- Ablating the catch-all on an input where the subcomponents alone reproduce `W x`
changes nothing — the situation tPD aims for on target data. -/
theorem tpdMasked_mulVec_of_faithful_on (W : Matrix m n ℝ) (D : Subcomponents m n C)
    (mask : C → ℝ) (x : n → ℝ) (h : (D.masked mask).mulVec x = W.mulVec x) (mΔ : ℝ)
    (hΔ : (W - D.weight).mulVec x = 0) :
    (tpdMasked W D mask mΔ).mulVec x = W.mulVec x := by
  unfold tpdMasked
  rw [Matrix.add_mulVec, Matrix.smul_mulVec_assoc, hΔ, smul_zero, add_zero, h]

/-- The adversarial masks of eq. (2) exist: a continuous loss attains a maximum on the
compact box `[μ, 1]`. -/
theorem exists_adversarial_mask (μ : C → ℝ) (hμ : ∀ c, μ c ≤ 1) (f : (C → ℝ) → ℝ)
    (hf : Continuous f) :
    ∃ mask ∈ Set.Icc μ (fun _ => 1), IsMaxOn f (Set.Icc μ (fun _ => 1)) mask :=
  IsCompact.exists_isMaxOn isCompact_Icc (Set.nonempty_Icc.mpr hμ) hf.continuousOn

end CatchAll

section Underdetermination

variable {m n : Type*} [Fintype m] [Fintype n]

/-- **Underdetermination off the target subspace.**  If a loss depends on the input
direction `V` only through the activations `V ⬝ x` for `x ∈ S`, then adding any `w`
orthogonal to `S` leaves it unchanged. -/
theorem loss_invariant_of_orthogonal {α : Type*} (S : Set (n → ℝ)) (Φ : (n → ℝ) → α)
    (hΦ : ∀ V V', (∀ x ∈ S, V ⬝ᵥ x = V' ⬝ᵥ x) → Φ V = Φ V') (V w : n → ℝ)
    (hw : ∀ x ∈ S, w ⬝ᵥ x = 0) : Φ (V + w) = Φ V := by
  apply hΦ
  intro x hx
  rw [add_dotProduct, hw x hx, add_zero]

lemma outer_mulVec (u : m → ℝ) (v : n → ℝ) (x : n → ℝ) :
    (outer u v).mulVec x = (v ⬝ᵥ x) • u := by
  ext i
  simp only [Matrix.mulVec, dotProduct, outer_apply, Pi.smul_apply, smul_eq_mul]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl ?_
  intro j _
  ring

/-- Two subcomponents with constant inner activations `α, β` on an input act as
`x ↦ α U_a + β U_b` there. -/
theorem merge_of_const_activations (Ua Ub : m → ℝ) (Va Vb : n → ℝ) (α β : ℝ) (x : n → ℝ)
    (ha : Va ⬝ᵥ x = α) (hb : Vb ⬝ᵥ x = β) :
    (outer Ua Va + outer Ub Vb).mulVec x = α • Ua + β • Ub := by
  rw [Matrix.add_mulVec, outer_mulVec, outer_mulVec, ha, hb]

/-- …and hence coincide there with the single rank-one map `(α U_a + β U_b) vᵀ` for any
`v` with `v ⬝ x = 1` (**merging**). -/
theorem merge_rank_one (Ua Ub : m → ℝ) (Va Vb v : n → ℝ) (α β : ℝ) (x : n → ℝ)
    (ha : Va ⬝ᵥ x = α) (hb : Vb ⬝ᵥ x = β) (hv : v ⬝ᵥ x = 1) :
    (outer (α • Ua + β • Ub) v).mulVec x = (outer Ua Va + outer Ub Vb).mulVec x := by
  rw [merge_of_const_activations Ua Ub Va Vb α β x ha hb, outer_mulVec, hv, one_smul]

/-- Rank-one splittings of a rank-two matrix are not unique. -/
theorem outer_add_outer_eq (u u' : m → ℝ) (v v' : n → ℝ) :
    outer u v + outer u' v' = outer (u + u') v + outer u' (v' - v) := by
  ext i j
  simp only [Matrix.add_apply, outer_apply, Pi.add_apply, Pi.sub_apply]
  ring

end Underdetermination

end

end Params.Decomp
