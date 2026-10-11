import Mathlib
import Params.Transformer.Attention
import Params.Transformer.LayerNorm

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# The Transformer encoder block and its operator-splitting form

Two descriptions of one computation:

* `encoderBlock`: the post-norm encoder layer of Vaswani et al. (2017),
  `u ↦ LN(u'' + FFN(u''))` with `u'' = LN(u + Attn(u))`;
* `splitStep`: the six sub-steps (3.5)–(3.10) of the Lie operator-splitting
  scheme of Tai–Liu–Li–Chan (arXiv:2510.03989), with `J = 2`, `Δt = 1`, after
  the spatial discretisation of Section 3.4 of that paper.

`splitStep_eq_encoderBlock` proves the paper's central claim (Section 4.1)
exactly: one splitting step *is* one encoder block, once the feed-forward
weights are read as `I + W_j` (as the paper itself notes) and the relaxation
`½(u^{(2+J)/M} + u^{2/M})` is followed by layer normalisation, which is
invariant under positive rescaling (`layerNorm_smul`).  Composing `N_t` steps
gives `N_t` blocks (`splitScheme_eq_encoderStack`).
-/

namespace Params

noncomputable section

open Finset

section Block

variable {n d f : Type*} [Fintype n] [Fintype d] [Fintype f] [Nonempty d]

/-- Broadcast a bias vector to every row. -/
def broadcast (b : f → ℝ) : Matrix n f ℝ := Matrix.of fun _ j => b j

/-- Entry-wise ReLU on a matrix (the projection onto `S₂`, applied per entry). -/
def reluM (A : Matrix n f ℝ) : Matrix n f ℝ := Matrix.of fun i j => relu (A i j)

/-- Layer normalisation applied to every row (token) of a matrix. -/
def lnRows (σ₁ σ₂ : ℝ) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  Matrix.of fun i => layerNorm σ₁ σ₂ (u i)

/-- Parameters of the position-wise feed-forward network. -/
structure FFNParams (d f : Type*) where
  W₁ : Matrix d f ℝ
  b₁ : f → ℝ
  W₂ : Matrix f d ℝ
  b₂ : d → ℝ

/-- `FFN(z) = ReLU(z W₁ + b₁) W₂ + b₂`, applied row-wise. -/
def ffn (φ : FFNParams d f) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  reluM (u * φ.W₁ + broadcast φ.b₁) * φ.W₂ + broadcast φ.b₂

/-- Parameters of one post-norm encoder block. -/
structure EncoderParams (d f : Type*) where
  attn : AttnParams d
  ffn : FFNParams d f
  σ₁ : ℝ
  σ₂ : ℝ

/-- One encoder block (Vaswani et al., 2017; Section 2.1 of the paper). -/
def encoderBlock (s : ℝ) (θ : EncoderParams d f) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  let u' := u + selfAttention s θ.attn u
  let u'' := lnRows θ.σ₁ θ.σ₂ u'
  let u''' := u'' + ffn θ.ffn u''
  lnRows θ.σ₁ θ.σ₂ u'''

/-- A stack of encoder blocks. -/
def encoderStack (s : ℝ) (θs : List (EncoderParams d f)) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  θs.foldl (fun v θ => encoderBlock s θ v) u

end Block

section Splitting

variable {n d : Type*} [Fintype n] [Fintype d] [Nonempty d] [DecidableEq d]

/-- Time-discretised control variables `θⁿ` of one splitting step (paper, Section 3.1). -/
structure SplitParams (d : Type*) where
  attn : AttnParams d
  W₁ : Matrix d d ℝ
  b₁ : d → ℝ
  W₂ : Matrix d d ℝ
  b₂ : d → ℝ
  σ₁ : ℝ
  σ₂ : ℝ

/-- Substep 1, eq. (3.5)/(3.21): `u^{1/M} = u⁰ + Softmax(Q K^T) V`. -/
def substep1 (s : ℝ) (θ : SplitParams d) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  u + selfAttention s θ.attn u

/-- Substeps 2 and 6, eqs. (3.6)/(3.10): projection onto `S₁(σ₁, σ₂)`, i.e. layer
normalisation (Theorem 3.1 of the paper, `layerNorm_sqDist_le` here). -/
def substep2 (θ : SplitParams d) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  lnRows θ.σ₁ θ.σ₂ u

/-- Substep 3, eq. (3.7) solved by (3.25)–(3.26): `ReLU(u + u W₁ + b₁)`. -/
def substep3 (θ : SplitParams d) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  reluM (u + u * θ.W₁ + broadcast θ.b₁)

/-- Substep 4, eq. (3.8): the final linear layer `u + u W₂ + b₂`. -/
def substep4 (θ : SplitParams d) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  u + u * θ.W₂ + broadcast θ.b₂

/-- Substep 5, eq. (3.9): relaxation `½(u^{(2+J)/M} + u^{2/M})`. -/
def substep5 (u₄ u₂ : Matrix n d ℝ) : Matrix n d ℝ :=
  (1 / 2 : ℝ) • (u₄ + u₂)

/-- One full step `u⁰ ↦ u¹` of the Lie splitting scheme (3.5)–(3.10). -/
def splitStep (s : ℝ) (θ : SplitParams d) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  let u₁ := substep1 s θ u
  let u₂ := substep2 θ u₁
  let u₃ := substep3 θ u₂
  let u₄ := substep4 θ u₃
  let u₅ := substep5 u₄ u₂
  substep2 θ u₅

/-- `N_t` steps of the scheme with time-dependent controls `θ¹, …, θ^{N_t}`. -/
def splitScheme (s : ℝ) (θs : List (SplitParams d)) (u : Matrix n d ℝ) : Matrix n d ℝ :=
  θs.foldl (fun v θ => splitStep s θ v) u

/-- The encoder parameters a splitting step realises: the paper's remark that
`u + u W_j + b_j = u (I + W_j) + b_j` is a linear layer with weight `I + W_j`. -/
def SplitParams.toEncoder (θ : SplitParams d) : EncoderParams d d :=
  { attn := θ.attn
    ffn := { W₁ := 1 + θ.W₁, b₁ := θ.b₁, W₂ := 1 + θ.W₂, b₂ := θ.b₂ }
    σ₁ := θ.σ₁
    σ₂ := θ.σ₂ }

lemma add_mul_eq_mul_one_add (u : Matrix n d ℝ) (W' : Matrix d d ℝ) :
    u + u * W' = u * (1 + W') := by
  rw [Matrix.mul_add, Matrix.mul_one]

lemma lnRows_smul (σ₁ σ₂ c : ℝ) (hc : 0 < c) (u : Matrix n d ℝ) :
    lnRows σ₁ σ₂ (c • u) = lnRows σ₁ σ₂ u := by
  ext i j
  simp only [lnRows, Matrix.of_apply]
  have hrow : (c • u) i = fun k => c * u i k := by
    funext k
    simp
  rw [hrow, layerNorm_smul σ₁ σ₂ c hc]

/-- **Exact recovery of the encoder block by operator splitting**
(Tai–Liu–Li–Chan, Section 4.1). -/
theorem splitStep_eq_encoderBlock (s : ℝ) (θ : SplitParams d) (u : Matrix n d ℝ) :
    splitStep s θ u = encoderBlock s θ.toEncoder u := by
  simp only [splitStep, encoderBlock, substep1, substep2, substep3, substep4, substep5,
    SplitParams.toEncoder, ffn]
  rw [add_mul_eq_mul_one_add _ θ.W₂, add_mul_eq_mul_one_add _ θ.W₁,
    lnRows_smul _ _ _ (by norm_num : (0 : ℝ) < 1 / 2), add_comm]

/-- `N_t` splitting steps are `N_t` encoder blocks. -/
theorem splitScheme_eq_encoderStack (s : ℝ) (θs : List (SplitParams d)) (u : Matrix n d ℝ) :
    splitScheme s θs u = encoderStack s (θs.map SplitParams.toEncoder) u := by
  induction θs generalizing u with
  | nil => rfl
  | cons θ θs ih =>
    simp only [splitScheme, encoderStack, List.map_cons, List.foldl_cons] at ih ⊢
    rw [splitStep_eq_encoderBlock]
    exact ih _

end Splitting

end

end Params
