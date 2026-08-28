import Mathlib
import Params.Transformer.Attention

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# Causal (masked) attention and prefix dependence

Decoder-only language models use *causal* attention: position `i` may attend
only to positions `j ≤ i`.  Over the reals there is no `-∞` logit, so the mask
is built into the softmax directly (`maskedSoftmax`): masked positions get
weight `0` and the normaliser runs over the unmasked positions only.

The structural theorem of this file is `PrefixDependent`: a map on token
matrices whose row `i` depends only on rows `≤ i`.  Causal self-attention is
prefix dependent (`causalSelfAttention_prefixDependent`), row-wise maps are,
and prefix dependence is closed under composition and residual addition.  This
is what makes the next-token distribution of a decoder-only model a function
of the prefix alone (`Params/Transformer/LanguageModel.lean`).
-/

namespace Params

noncomputable section

open Finset Matrix

section MaskedSoftmax

variable {ι : Type*} [Fintype ι]

/-- Softmax restricted to the positions allowed by `mask`; masked positions get
weight `0`. -/
def maskedSoftmax (mask : ι → Prop) [DecidablePred mask] (a : ι → ℝ) : ι → ℝ :=
  fun i =>
    if mask i then Real.exp (a i) / ∑ j ∈ Finset.univ.filter mask, Real.exp (a j) else 0

lemma maskedSoftmax_of_not (mask : ι → Prop) [DecidablePred mask] (a : ι → ℝ) {i : ι}
    (hi : ¬ mask i) : maskedSoftmax mask a i = 0 := by
  unfold maskedSoftmax
  rw [if_neg hi]

lemma maskedSoftmax_nonneg (mask : ι → Prop) [DecidablePred mask] (a : ι → ℝ) (i : ι) :
    0 ≤ maskedSoftmax mask a i := by
  unfold maskedSoftmax
  split_ifs
  · exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg (fun _ _ => (Real.exp_pos _).le))
  · exact le_rfl

/-- Masked softmax weights sum to one as soon as some position is unmasked. -/
lemma sum_maskedSoftmax (mask : ι → Prop) [DecidablePred mask] (a : ι → ℝ)
    (hne : ∃ j, mask j) : ∑ i, maskedSoftmax mask a i = 1 := by
  unfold maskedSoftmax
  rw [← Finset.sum_filter, ← Finset.sum_div]
  apply div_self
  apply ne_of_gt
  apply Finset.sum_pos (fun j _ => Real.exp_pos (a j))
  obtain ⟨j, hj⟩ := hne
  exact ⟨j, Finset.mem_filter.mpr ⟨Finset.mem_univ j, hj⟩⟩

/-- Masked softmax only looks at the logits of unmasked positions. -/
lemma maskedSoftmax_congr (mask : ι → Prop) [DecidablePred mask] (a a' : ι → ℝ)
    (h : ∀ j, mask j → a j = a' j) : maskedSoftmax mask a = maskedSoftmax mask a' := by
  funext i
  unfold maskedSoftmax
  have hden : ∑ j ∈ Finset.univ.filter mask, Real.exp (a j)
      = ∑ j ∈ Finset.univ.filter mask, Real.exp (a' j) := by
    refine Finset.sum_congr rfl ?_
    intro j hj
    rw [h j (Finset.mem_filter.mp hj).2]
  rw [hden]
  by_cases hi : mask i
  · rw [if_pos hi, if_pos hi, h i hi]
  · rw [if_neg hi, if_neg hi]

end MaskedSoftmax

section Causal

variable {d : Type*} [Fintype d] {N : ℕ}

/-- Causal attention weights: row `i` is a softmax over `j ≤ i`. -/
def causalWeights (s : ℝ) (Q K : Matrix (Fin N) d ℝ) : Matrix (Fin N) (Fin N) ℝ :=
  Matrix.of fun i => maskedSoftmax (fun j => j ≤ i) ((s • (Q * Kᵀ)) i)

/-- Causal scaled dot-product attention. -/
def causalAttention (s : ℝ) (Q K V : Matrix (Fin N) d ℝ) : Matrix (Fin N) d ℝ :=
  causalWeights s Q K * V

/-- Causal self-attention with kernels `W^Q, W^K, W^V`. -/
def causalSelfAttention (s : ℝ) (θ : AttnParams d) (u : Matrix (Fin N) d ℝ) :
    Matrix (Fin N) d ℝ :=
  causalAttention s (u * θ.WQ) (u * θ.WK) (u * θ.WV)

lemma causalWeights_of_lt (s : ℝ) (Q K : Matrix (Fin N) d ℝ) {i j : Fin N} (hij : i < j) :
    causalWeights s Q K i j = 0 := by
  unfold causalWeights
  rw [Matrix.of_apply]
  exact maskedSoftmax_of_not _ _ (not_le.mpr hij)

lemma causalWeights_sum (s : ℝ) (Q K : Matrix (Fin N) d ℝ) (i : Fin N) :
    ∑ j, causalWeights s Q K i j = 1 := by
  unfold causalWeights
  simp only [Matrix.of_apply]
  exact sum_maskedSoftmax _ _ ⟨i, le_rfl⟩

/-- A map on token matrices whose row `i` depends only on rows `≤ i`. -/
def PrefixDependent {e : Type*} (F : Matrix (Fin N) d ℝ → Matrix (Fin N) e ℝ) : Prop :=
  ∀ (u u' : Matrix (Fin N) d ℝ) (i : Fin N),
    (∀ j, j ≤ i → ∀ l, u j l = u' j l) → ∀ k, F u i k = F u' i k

/-- Row-wise maps are prefix dependent. -/
theorem prefixDependent_rowwise {e : Type*} (g : (d → ℝ) → (e → ℝ)) :
    PrefixDependent (fun u : Matrix (Fin N) d ℝ => Matrix.of fun i => g (u i)) := by
  intro u u' i h k
  simp only [Matrix.of_apply]
  have : u i = u' i := funext (h i le_rfl)
  rw [this]

/-- Composition preserves prefix dependence. -/
theorem prefixDependent_comp {e e' : Type*} (F : Matrix (Fin N) d ℝ → Matrix (Fin N) e ℝ)
    (G : Matrix (Fin N) e ℝ → Matrix (Fin N) e' ℝ) (hF : PrefixDependent F)
    (hG : PrefixDependent G) : PrefixDependent (fun u => G (F u)) := by
  intro u u' i h k
  exact hG (F u) (F u') i (fun j hj l => hF u u' j (fun j' hj' l' => h j' (hj'.trans hj) l') l) k

/-- Residual addition preserves prefix dependence. -/
theorem prefixDependent_add (F G : Matrix (Fin N) d ℝ → Matrix (Fin N) d ℝ)
    (hF : PrefixDependent F) (hG : PrefixDependent G) :
    PrefixDependent (fun u => F u + G u) := by
  intro u u' i h k
  simp only [Matrix.add_apply]
  rw [hF u u' i h k, hG u u' i h k]

/-- The identity is prefix dependent. -/
theorem prefixDependent_id : PrefixDependent (fun u : Matrix (Fin N) d ℝ => u) := by
  intro u u' i h k
  exact h i le_rfl k

/-- **Causal self-attention is prefix dependent**: output row `i` depends only on
tokens `≤ i`. -/
theorem causalSelfAttention_prefixDependent (s : ℝ) (θ : AttnParams d) :
    PrefixDependent (fun u : Matrix (Fin N) d ℝ => causalSelfAttention s θ u) := by
  intro u u' i h k
  simp only [causalSelfAttention, causalAttention, causalWeights, Matrix.mul_apply,
    Matrix.of_apply]
  have hscore : ∀ j, j ≤ i →
      (s • (u * θ.WQ * (u * θ.WK)ᵀ)) i j = (s • (u' * θ.WQ * (u' * θ.WK)ᵀ)) i j := by
    intro j hj
    simp only [Matrix.smul_apply, Matrix.mul_apply, Matrix.transpose_apply, h i le_rfl, h j hj]
  have hw : maskedSoftmax (fun j => j ≤ i) ((s • (u * θ.WQ * (u * θ.WK)ᵀ)) i)
      = maskedSoftmax (fun j => j ≤ i) ((s • (u' * θ.WQ * (u' * θ.WK)ᵀ)) i) :=
    maskedSoftmax_congr _ _ _ hscore
  rw [hw]
  refine Finset.sum_congr rfl ?_
  intro j _
  by_cases hj : j ≤ i
  · simp only [Matrix.mul_apply, h j hj]
  · rw [maskedSoftmax_of_not _ _ hj, zero_mul, zero_mul]

end Causal

end

end Params
