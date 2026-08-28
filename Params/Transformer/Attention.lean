import Mathlib
import Params.Transformer.Softmax

/-!
# Scaled dot-product attention

`attention s Q K V = softmaxRows (s • Q Kᵀ) * V` is the attention layer of
Vaswani et al. (2017) with temperature `s` (`s = 1/√d` in the original), which
Tai–Liu–Li–Chan (arXiv:2510.03989, eq. (3.21)) obtain as the spatial
discretisation of the non-local integral operator
`⟨γ(x,·;u), V(·,y;u)⟩_{Ω_x}`.

Proved here, from Mathlib alone:

* `attention_apply`: each output row is a softmax-weighted average of the rows
  of `V`; hence `attention_le` / `le_attention` — every output entry lies
  between the minimum and maximum of the corresponding column of `V`;
* `attention_perm_equivariant`: permuting the tokens permutes the output
  (attention has no positional structure of its own);
* `selfAttention_perm_equivariant`: the same for `u ↦ attention (uW^Q) (uW^K) (uW^V)`;
* `multiHeadAttention_eq_sum_heads`: concatenating heads and applying `W^O` is
  the same as summing per-head attentions against the row-blocks of `W^O`
  (this is the form `Σ_m Softmax(Q_m K_mᵀ) V_m` in eq. (5.7) of the paper).
-/

namespace Params

noncomputable section

open Finset Matrix

section Attention

variable {n d : Type*} [Fintype n] [Fintype d]

/-- Attention weights `Softmax_row (s • Q Kᵀ)`. -/
def attnWeights (s : ℝ) (Q K : Matrix n d ℝ) : Matrix n n ℝ :=
  softmaxRows (s • (Q * Kᵀ))

/-- Scaled dot-product attention with temperature `s`. -/
def attention (s : ℝ) (Q K V : Matrix n d ℝ) : Matrix n d ℝ :=
  attnWeights s Q K * V

lemma attention_apply (s : ℝ) (Q K V : Matrix n d ℝ) (i : n) (k : d) :
    attention s Q K V i k = ∑ j, attnWeights s Q K i j * V j k := by
  simp only [attention, Matrix.mul_apply]

lemma attnWeights_apply (s : ℝ) (Q K : Matrix n d ℝ) (i j : n) :
    attnWeights s Q K i j = softmax ((s • (Q * Kᵀ)) i) j := rfl

lemma attnWeights_nonneg [Nonempty n] (s : ℝ) (Q K : Matrix n d ℝ) (i j : n) :
    0 ≤ attnWeights s Q K i j :=
  softmax_nonneg _ j

lemma attnWeights_sum [Nonempty n] (s : ℝ) (Q K : Matrix n d ℝ) (i : n) :
    ∑ j, attnWeights s Q K i j = 1 :=
  sum_softmax _

/-- Each attention output entry is bounded above by the column maximum of `V`. -/
theorem attention_le [Nonempty n] (s : ℝ) (Q K V : Matrix n d ℝ) (i : n) (k : d) (b : ℝ)
    (hb : ∀ j, V j k ≤ b) : attention s Q K V i k ≤ b := by
  rw [attention_apply]
  exact softmax_weighted_le ((s • (Q * Kᵀ)) i) (fun j => V j k) b hb

/-- Each attention output entry is bounded below by the column minimum of `V`. -/
theorem le_attention [Nonempty n] (s : ℝ) (Q K V : Matrix n d ℝ) (i : n) (k : d) (b : ℝ)
    (hb : ∀ j, b ≤ V j k) : b ≤ attention s Q K V i k := by
  rw [attention_apply]
  exact le_softmax_weighted ((s • (Q * Kᵀ)) i) (fun j => V j k) b hb

/-- Attention with a constant value matrix returns that constant. -/
theorem attention_const [Nonempty n] (s : ℝ) (Q K : Matrix n d ℝ) (c : d → ℝ) (i : n) (k : d) :
    attention s Q K (Matrix.of fun _ j => c j) i k = c k := by
  rw [attention_apply]
  simp only [Matrix.of_apply]
  rw [← Finset.sum_mul, attnWeights_sum, one_mul]

/-! ### Permutation equivariance -/

lemma score_submatrix (s : ℝ) (Q K : Matrix n d ℝ) (σ : Equiv.Perm n) (i j : n) :
    (s • (Q.submatrix σ id * (K.submatrix σ id)ᵀ)) i j = (s • (Q * Kᵀ)) (σ i) (σ j) := by
  simp [Matrix.mul_apply]

lemma attnWeights_submatrix (s : ℝ) (Q K : Matrix n d ℝ) (σ : Equiv.Perm n) (i j : n) :
    attnWeights s (Q.submatrix σ id) (K.submatrix σ id) i j
      = attnWeights s Q K (σ i) (σ j) := by
  have hS : (s • (Q.submatrix σ id * (K.submatrix σ id)ᵀ)) i
      = fun j' => (s • (Q * Kᵀ)) (σ i) (σ j') := by
    funext j'
    exact score_submatrix s Q K σ i j'
  rw [attnWeights_apply, attnWeights_apply, hS]
  exact congrFun (softmax_comp_perm ((s • (Q * Kᵀ)) (σ i)) σ) j

/-- **Permutation equivariance.** Re-ordering the tokens re-orders the outputs. -/
theorem attention_perm_equivariant (s : ℝ) (Q K V : Matrix n d ℝ) (σ : Equiv.Perm n) :
    attention s (Q.submatrix σ id) (K.submatrix σ id) (V.submatrix σ id)
      = (attention s Q K V).submatrix σ id := by
  ext i k
  rw [Matrix.submatrix_apply, id_eq, attention_apply, attention_apply]
  calc ∑ j, attnWeights s (Q.submatrix σ id) (K.submatrix σ id) i j * V.submatrix σ id j k
      = ∑ j, attnWeights s Q K (σ i) (σ j) * V (σ j) k := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [attnWeights_submatrix, Matrix.submatrix_apply, id_eq]
    _ = ∑ j, attnWeights s Q K (σ i) j * V j k :=
        Equiv.sum_comp σ (fun j => attnWeights s Q K (σ i) j * V j k)

end Attention

section SelfAttention

variable {n d : Type*} [Fintype n] [Fintype d]

/-- The learnable kernels `W^Q, W^K, W^V` of a single attention head. -/
structure AttnParams (d : Type*) where
  WQ : Matrix d d ℝ
  WK : Matrix d d ℝ
  WV : Matrix d d ℝ

/-- Single-head self-attention `u ↦ Attention(uW^Q, uW^K, uW^V)`. -/
def selfAttention (s : ℝ) (θ : AttnParams d) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  attention s (u * θ.WQ) (u * θ.WK) (u * θ.WV)

lemma submatrix_mul_left (u : Matrix n d ℝ) (W : Matrix d d ℝ) (σ : Equiv.Perm n) :
    u.submatrix σ id * W = (u * W).submatrix σ id := by
  ext i k
  simp [Matrix.mul_apply]

/-- Self-attention is permutation equivariant. -/
theorem selfAttention_perm_equivariant (s : ℝ) (θ : AttnParams d) (u : Matrix n d ℝ)
    (σ : Equiv.Perm n) :
    selfAttention s θ (u.submatrix σ id) = (selfAttention s θ u).submatrix σ id := by
  unfold selfAttention
  rw [submatrix_mul_left, submatrix_mul_left, submatrix_mul_left]
  exact attention_perm_equivariant s _ _ _ σ

end SelfAttention

section MultiHead

variable {n d h dk : Type*} [Fintype n] [Fintype d] [Fintype h] [Fintype dk]

/-- Parameters of multi-head attention: per-head projections and the output map `W^O`
acting on the concatenation of heads (indexed by `h × dk`). -/
structure MHAParams (d h dk : Type*) where
  WQ : h → Matrix d dk ℝ
  WK : h → Matrix d dk ℝ
  WV : h → Matrix d dk ℝ
  WO : Matrix (h × dk) d ℝ

/-- Per-head attention output. -/
def headOut (s : ℝ) (θ : MHAParams d h dk) (u : Matrix n d ℝ) (p : h) : Matrix n dk ℝ :=
  attention s (u * θ.WQ p) (u * θ.WK p) (u * θ.WV p)

/-- Multi-head attention: concatenate the heads, then apply `W^O`. -/
def multiHeadAttention (s : ℝ) (θ : MHAParams d h dk) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  (Matrix.of fun i (p : h × dk) => headOut s θ u p.1 i p.2) * θ.WO

/-- The rows of `W^O` belonging to head `p`. -/
def headBlock (θ : MHAParams d h dk) (p : h) : Matrix dk d ℝ :=
  θ.WO.submatrix (fun k => (p, k)) id

/-- Multi-head attention is the sum over heads of `Attention_p · W^O_p`. -/
theorem multiHeadAttention_eq_sum_heads (s : ℝ) (θ : MHAParams d h dk) (u : Matrix n d ℝ) :
    multiHeadAttention s θ u = ∑ p, headOut s θ u p * headBlock θ p := by
  ext i k
  simp only [multiHeadAttention, headBlock, Matrix.mul_apply, Matrix.of_apply, Matrix.sum_apply,
    Matrix.submatrix_apply, id_eq]
  simp only [Fintype.sum_prod_type]

end MultiHead

end

end Params
