import Mathlib

/-!
# Linear parameter decomposition: subcomponents, masks, losses

Definitions transcribed from the SPD paper (Bushnaq–Braun–Sharkey,
arXiv:2506.20790) and, where the two differ, from the reference code
(goodfire-ai/param-decomp, branch `spd-paper`, `spd/models/components.py`,
`spd/models/component_utils.py`, `spd/losses.py`):

* a weight `W : Matrix m n ℝ` is written as `U * V` with `U : Matrix m C ℝ`,
  `V : Matrix C n ℝ`; subcomponent `c` is the rank-one matrix `U_{:,c} V_{c,:}`
  (`Subcomponents.component`, an `outer` product);
* the masked weight is `U diag(mask) V` (`Subcomponents.masked`), which equals
  `Σ_c mask c • component c` (`masked_eq_sum`) and recovers `U * V` at
  `mask ≡ 1` (`masked_one`); zeroing one mask entry subtracts that
  subcomponent (`masked_update_zero`) — this is what "ablating a subcomponent"
  means;
* the faithfulness loss `‖W − U V‖_F²` and its exactness criterion
  (`faithfulnessLoss_eq_zero_iff`);
* the stochastic mask `m = g + (1 − g) r` (`spdMask`), which lies in `[g, 1]`
  for `g, r ∈ [0, 1]` (`spdMask_mem_Icc`);
* the hard sigmoid and the two leaky variants the code uses for masks and for
  the importance penalty (`hardSigmoid`, `lowerLeaky`, `upperLeaky`);
* the importance-minimality penalty `Σ_c |g c|^p` (`importanceLoss`) and the
  observation behind the paper's choice `p > 1`: splitting a subcomponent
  that must stay fully important into two costs `2` instead of `1`
  (`importanceLoss_split`).

Everything is proved from Mathlib; the file contains no assumptions.
-/

namespace Params.Decomp

noncomputable section

open Finset

section Outer

variable {m n : Type*} [Fintype m] [Fintype n]

/-- Outer product `u vᵀ`. -/
def outer (u : m → ℝ) (v : n → ℝ) : Matrix m n ℝ := Matrix.of fun i j => u i * v j

@[simp] lemma outer_apply (u : m → ℝ) (v : n → ℝ) (i : m) (j : n) :
    outer u v i j = u i * v j := rfl

/-- An outer product is a product of an `m × 1` and a `1 × n` matrix. -/
lemma outer_eq_mul (u : m → ℝ) (v : n → ℝ) :
    outer u v = (Matrix.of fun i (_ : Fin 1) => u i) * (Matrix.of fun (_ : Fin 1) j => v j) := by
  ext i j
  rw [Matrix.mul_apply, Fin.sum_univ_one]
  simp only [outer_apply, Matrix.of_apply]

/-- Outer products have rank at most one. -/
theorem rank_outer_le (u : m → ℝ) (v : n → ℝ) : (outer u v).rank ≤ 1 := by
  rw [outer_eq_mul]
  refine (Matrix.rank_mul_le_left _ _).trans ?_
  refine (Matrix.rank_le_card_width _).trans ?_
  simp

/-- The scaling ambiguity of any outer product: `(c u)(c⁻¹ v) = u v`. -/
lemma outer_smul_inv (u : m → ℝ) (v : n → ℝ) (c : ℝ) (hc : c ≠ 0) :
    outer (fun i => c * u i) (fun j => c⁻¹ * v j) = outer u v := by
  ext i j
  simp only [outer_apply]
  rw [mul_mul_mul_comm, mul_inv_cancel₀ hc, one_mul]

end Outer

section Subcomponents

open Matrix

variable {m n C : Type*} [Fintype m] [Fintype n] [Fintype C]

/-- The SPD parametrisation of one weight matrix: `W ≈ U V`, `U : m × C`, `V : C × n`.
(The code names these `B` and `A`: `weight = einsum(A, B, "d_in C, C d_out -> d_out d_in")`.) -/
structure Subcomponents (m n C : Type*) where
  U : Matrix m C ℝ
  V : Matrix C n ℝ

/-- Subcomponent `c`: the rank-one matrix `U_{:,c} V_{c,:}`. -/
def Subcomponents.component (D : Subcomponents m n C) (c : C) : Matrix m n ℝ :=
  outer (fun i => D.U i c) (fun j => D.V c j)

/-- The reconstructed weight `U V`. -/
def Subcomponents.weight (D : Subcomponents m n C) : Matrix m n ℝ := D.U * D.V

/-- The masked weight `U diag(mask) V` used in every forward pass of SPD
(`component_acts *= mask` in `LinearComponent.forward`). -/
def Subcomponents.masked (D : Subcomponents m n C) (mask : C → ℝ) : Matrix m n ℝ :=
  D.U * Matrix.diagonal mask * D.V

theorem weight_eq_sum_components (D : Subcomponents m n C) :
    D.weight = ∑ c, D.component c := by
  ext i j
  rw [Subcomponents.weight, Matrix.mul_apply, Matrix.sum_apply]
  simp only [Subcomponents.component, outer_apply]

theorem masked_eq_sum (D : Subcomponents m n C) (mask : C → ℝ) :
    D.masked mask = ∑ c, mask c • D.component c := by
  ext i j
  rw [Subcomponents.masked, Matrix.mul_apply, Matrix.sum_apply]
  simp only [Matrix.mul_diagonal, Subcomponents.component, Matrix.smul_apply, outer_apply,
    smul_eq_mul]
  refine Finset.sum_congr rfl ?_
  intro c _
  ring

/-- Masks identically one recover the reconstructed weight. -/
theorem masked_one (D : Subcomponents m n C) : D.masked (fun _ => 1) = D.weight := by
  simp [Subcomponents.masked, Subcomponents.weight, Matrix.diagonal_one]

/-- Ablating subcomponent `c` (setting its mask to `0`) subtracts `mask c • component c`. -/
theorem masked_update_zero [DecidableEq C] (D : Subcomponents m n C) (mask : C → ℝ) (c : C) :
    D.masked (Function.update mask c 0) = D.masked mask - mask c • D.component c := by
  rw [masked_eq_sum, masked_eq_sum]
  have h : ∀ c', Function.update mask c 0 c' • D.component c'
      = mask c' • D.component c' - (if c' = c then mask c • D.component c else 0) := by
    intro c'
    by_cases hc : c' = c
    · subst hc
      simp
    · simp [Function.update_of_ne hc, hc]
  simp only [h]
  rw [Finset.sum_sub_distrib, Finset.sum_ite_eq' Finset.univ c, if_pos (Finset.mem_univ c)]

/-- The action of the masked weight on an input is the mask-weighted sum of the
subcomponent outputs `(x · V_{c,:}) U_{:,c}` — exactly `component_acts * mask @ B`
in the code. -/
theorem masked_mulVec (D : Subcomponents m n C) (mask : C → ℝ) (x : n → ℝ) (i : m) :
    (D.masked mask).mulVec x i = ∑ c, mask c * ((∑ j, D.V c j * x j) * D.U i c) := by
  simp only [Matrix.mulVec, dotProduct, masked_eq_sum, Matrix.sum_apply, Matrix.smul_apply,
    Subcomponents.component, outer_apply, smul_eq_mul, Finset.sum_mul, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl ?_
  intro c _
  refine Finset.sum_congr rfl ?_
  intro j _
  ring

end Subcomponents

section Faithfulness

variable {m n C : Type*} [Fintype m] [Fintype n] [Fintype C]

/-- Faithfulness loss `‖W − U V‖_F²` (the code divides by the total parameter count;
that constant does not affect any statement here). -/
def faithfulnessLoss (W : Matrix m n ℝ) (D : Subcomponents m n C) : ℝ :=
  ∑ i, ∑ j, (W i j - D.weight i j) ^ 2

theorem faithfulnessLoss_nonneg (W : Matrix m n ℝ) (D : Subcomponents m n C) :
    0 ≤ faithfulnessLoss W D :=
  Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => sq_nonneg _))

/-- Zero faithfulness loss is exact reconstruction. -/
theorem faithfulnessLoss_eq_zero_iff (W : Matrix m n ℝ) (D : Subcomponents m n C) :
    faithfulnessLoss W D = 0 ↔ D.weight = W := by
  unfold faithfulnessLoss
  rw [Finset.sum_eq_zero_iff_of_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => sq_nonneg _))]
  constructor
  · intro h
    ext i j
    have hij := (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => sq_nonneg _)).mp
      (h i (Finset.mem_univ i)) j (Finset.mem_univ j)
    have hsub := (pow_eq_zero_iff two_ne_zero).mp hij
    linarith
  · intro h i _
    apply Finset.sum_eq_zero
    intro j _
    rw [h, sub_self, zero_pow two_ne_zero]

/-- With exact reconstruction the all-ones mask reproduces the target weight, so
every "unmasked" layer of the layer-wise loss is literally the target layer. -/
theorem masked_one_of_faithful (W : Matrix m n ℝ) (D : Subcomponents m n C)
    (h : faithfulnessLoss W D = 0) : D.masked (fun _ => 1) = W := by
  rw [masked_one]
  exact (faithfulnessLoss_eq_zero_iff W D).mp h

end Faithfulness

section Masks

variable {C : Type*} [Fintype C]

/-- The SPD stochastic mask `m = g + (1 − g) r` (`calc_stochastic_masks`). -/
def spdMask (g r : C → ℝ) : C → ℝ := fun c => g c + (1 - g c) * r c

/-- For `g, r ∈ [0, 1]` the mask lies in `[g, 1]`. -/
theorem spdMask_mem_Icc (g r : C → ℝ) (c : C) (hg : g c ∈ Set.Icc (0 : ℝ) 1)
    (hr : r c ∈ Set.Icc (0 : ℝ) 1) : spdMask g r c ∈ Set.Icc (g c) 1 := by
  obtain ⟨hg0, hg1⟩ := hg
  obtain ⟨hr0, hr1⟩ := hr
  unfold spdMask
  constructor
  · nlinarith
  · nlinarith

theorem spdMask_r_zero (g : C → ℝ) : spdMask g (fun _ => 0) = g := by
  funext c
  simp [spdMask]

theorem spdMask_r_one (g : C → ℝ) : spdMask g (fun _ => 1) = fun _ => 1 := by
  funext c
  simp [spdMask]

/-- The hard sigmoid `clamp(x, 0, 1)`. -/
def hardSigmoid (x : ℝ) : ℝ := max 0 (min 1 x)

/-- The mask activation of the code: `where(x > 0, clamp(x, max=1), α x)`. -/
def lowerLeaky (α x : ℝ) : ℝ := if 0 < x then min 1 x else α * x

/-- The penalty activation of the code: `where(x > 1, 1 + α (x − 1), relu x)`. -/
def upperLeaky (α x : ℝ) : ℝ := if 1 < x then 1 + α * (x - 1) else max 0 x

theorem hardSigmoid_mem_Icc (x : ℝ) : hardSigmoid x ∈ Set.Icc (0 : ℝ) 1 := by
  unfold hardSigmoid
  constructor
  · exact le_max_left 0 _
  · exact max_le zero_le_one (min_le_left 1 x)

theorem lowerLeaky_le_one (α x : ℝ) (hα : 0 ≤ α) : lowerLeaky α x ≤ 1 := by
  unfold lowerLeaky
  split_ifs with h
  · exact min_le_left 1 x
  · push_neg at h
    nlinarith

theorem upperLeaky_nonneg (α x : ℝ) (hα : 0 ≤ α) : 0 ≤ upperLeaky α x := by
  unfold upperLeaky
  split_ifs with h
  · nlinarith
  · exact le_max_left 0 x

/-- On `[0, 1]` the three activations agree with the identity. -/
theorem lowerLeaky_of_mem_Icc (α x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1) (hx0 : 0 < x) :
    lowerLeaky α x = x := by
  unfold lowerLeaky
  rw [if_pos hx0, min_eq_right hx.2]

theorem upperLeaky_of_mem_Icc (α x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1) : upperLeaky α x = x := by
  unfold upperLeaky
  rw [if_neg (not_lt.mpr hx.2), max_eq_right hx.1]

theorem hardSigmoid_of_mem_Icc (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1) : hardSigmoid x = x := by
  unfold hardSigmoid
  rw [min_eq_right hx.2, max_eq_right hx.1]

end Masks

section Importance

variable {C : Type*} [Fintype C]

/-- Importance-minimality penalty `Σ_c |g c|^p`. -/
def importanceLoss (p : ℝ) (g : C → ℝ) : ℝ := ∑ c, |g c| ^ p

theorem importanceLoss_nonneg (p : ℝ) (g : C → ℝ) : 0 ≤ importanceLoss p g :=
  Finset.sum_nonneg (fun c _ => Real.rpow_nonneg (abs_nonneg _) p)

/-- Two fully-important subcomponents cost `2`, one costs `1`, for every exponent `p`:
splitting a subcomponent that must stay important can never lower the penalty. -/
theorem importanceLoss_split (p : ℝ) :
    importanceLoss p (fun _ : Fin 2 => (1 : ℝ)) = 2 ∧ importanceLoss p (fun _ : Fin 1 => (1 : ℝ)) = 1 := by
  constructor
  · norm_num [importanceLoss, Fin.sum_univ_two, Real.one_rpow]
  · norm_num [importanceLoss, Fin.sum_univ_one, Real.one_rpow]

end Importance

end

end Params.Decomp
