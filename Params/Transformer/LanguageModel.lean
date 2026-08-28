import Mathlib
import Params.Transformer.Encoder
import Params.Transformer.Causal

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# A decoder-only language model, and what its predictions can depend on

`LMParams` packages a decoder-only Transformer language model: token
embedding, positional embedding, a stack of causal (post-norm) blocks, and an
unembedding.  `nextTokenDist s θ t i` is the model's distribution over the next
token after reading positions `0, …, i` of the token sequence `t`.

The main theorem, `nextTokenDist_prefix`, is the exact statement of *causality*
for such a model: the next-token distribution at position `i` is a function
of the tokens at positions `≤ i` only.  Every step of the proof is a
`PrefixDependent` closure property from `Params/Transformer/Causal.lean`.

The second part of the file is the *pre-norm* residual-stream decomposition
used by direct logit attribution: with pre-norm blocks and no final
normalisation, the logits are exactly the sum of the direct contributions of
the embedding, every attention sub-layer and every MLP sub-layer
(`logitsPre_eq_sum_contributions`).  This is linear algebra only, but it is
the identity that "direct logit attribution" silently assumes, so it is stated
and proved once.
-/

namespace Params

noncomputable section

open Finset

section Decoder

variable {d f : Type*} [Fintype d] [Fintype f] [Nonempty d] {N : ℕ}

/-- Parameters of one post-norm decoder block. -/
structure DecoderParams (d f : Type*) where
  attn : AttnParams d
  ffn : FFNParams d f
  σ₁ : ℝ
  σ₂ : ℝ

/-- One post-norm decoder block with causal self-attention. -/
def decoderBlock (s : ℝ) (θ : DecoderParams d f) (u : Matrix (Fin N) d ℝ) : Matrix (Fin N) d ℝ :=
  let u' := u + causalSelfAttention s θ.attn u
  let u'' := lnRows θ.σ₁ θ.σ₂ u'
  let u''' := u'' + ffn θ.ffn u''
  lnRows θ.σ₁ θ.σ₂ u'''

/-- A stack of decoder blocks. -/
def decoderStack (s : ℝ) (θs : List (DecoderParams d f)) (u : Matrix (Fin N) d ℝ) :
    Matrix (Fin N) d ℝ :=
  θs.foldl (fun v θ => decoderBlock s θ v) u

/-- The feed-forward network is applied position-wise, so it is prefix dependent. -/
theorem ffn_prefixDependent (φ : FFNParams d f) :
    PrefixDependent (fun u : Matrix (Fin N) d ℝ => ffn φ u) := by
  intro u u' i h k
  simp only [ffn, reluM, broadcast, Matrix.add_apply, Matrix.mul_apply, Matrix.of_apply,
    h i le_rfl]

/-- Row-wise layer normalisation is prefix dependent. -/
theorem lnRows_prefixDependent (σ₁ σ₂ : ℝ) :
    PrefixDependent (fun u : Matrix (Fin N) d ℝ => lnRows σ₁ σ₂ u) :=
  prefixDependent_rowwise (layerNorm σ₁ σ₂)

/-- A causal decoder block is prefix dependent. -/
theorem decoderBlock_prefixDependent (s : ℝ) (θ : DecoderParams d f) :
    PrefixDependent (fun u : Matrix (Fin N) d ℝ => decoderBlock s θ u) := by
  have h1 : PrefixDependent (fun u : Matrix (Fin N) d ℝ => u + causalSelfAttention s θ.attn u) :=
    prefixDependent_add _ _ prefixDependent_id (causalSelfAttention_prefixDependent s θ.attn)
  have h2 : PrefixDependent (fun u : Matrix (Fin N) d ℝ => lnRows θ.σ₁ θ.σ₂ u) :=
    lnRows_prefixDependent θ.σ₁ θ.σ₂
  have h3 : PrefixDependent (fun u : Matrix (Fin N) d ℝ => u + ffn θ.ffn u) :=
    prefixDependent_add _ _ prefixDependent_id (ffn_prefixDependent θ.ffn)
  exact prefixDependent_comp _ _ (prefixDependent_comp _ _ (prefixDependent_comp _ _ h1 h2) h3) h2

/-- A stack of causal decoder blocks is prefix dependent. -/
theorem decoderStack_prefixDependent (s : ℝ) (θs : List (DecoderParams d f)) :
    PrefixDependent (fun u : Matrix (Fin N) d ℝ => decoderStack s θs u) := by
  induction θs with
  | nil => exact prefixDependent_id
  | cons θ θs ih =>
    have := prefixDependent_comp _ _ (decoderBlock_prefixDependent (N := N) s θ) ih
    exact this

end Decoder

section LM

variable {V d f : Type*} [Fintype V] [Fintype d] [Fintype f] [Nonempty d] {N : ℕ}

/-- A decoder-only language model. -/
structure LMParams (V d f : Type*) (N : ℕ) where
  embed : Matrix V d ℝ
  pos : Matrix (Fin N) d ℝ
  blocks : List (DecoderParams d f)
  unembed : Matrix d V ℝ

/-- Token plus positional embedding of a sequence `t`. -/
def embedTokens (θ : LMParams V d f N) (t : Fin N → V) : Matrix (Fin N) d ℝ :=
  Matrix.of fun i => θ.embed (t i) + θ.pos i

/-- Logits at every position. -/
def logits (s : ℝ) (θ : LMParams V d f N) (t : Fin N → V) : Matrix (Fin N) V ℝ :=
  decoderStack s θ.blocks (embedTokens θ t) * θ.unembed

/-- The next-token distribution after position `i`. -/
def nextTokenDist (s : ℝ) (θ : LMParams V d f N) (t : Fin N → V) (i : Fin N) : V → ℝ :=
  softmax (logits s θ t i)

/-- The next-token distribution is a probability vector. -/
theorem sum_nextTokenDist [Nonempty V] (s : ℝ) (θ : LMParams V d f N) (t : Fin N → V) (i : Fin N) :
    ∑ v, nextTokenDist s θ t i v = 1 :=
  sum_softmax _

/-- Embeddings of two sequences agree on every position up to `i` where the tokens agree. -/
theorem embedTokens_prefix (θ : LMParams V d f N) (t t' : Fin N → V) (i : Fin N)
    (h : ∀ j, j ≤ i → t j = t' j) :
    ∀ j, j ≤ i → ∀ l, embedTokens θ t j l = embedTokens θ t' j l := by
  intro j hj l
  simp only [embedTokens, Matrix.of_apply, h j hj]

/-- Logits at position `i` depend only on tokens at positions `≤ i`. -/
theorem logits_prefix (s : ℝ) (θ : LMParams V d f N) (t t' : Fin N → V) (i : Fin N)
    (h : ∀ j, j ≤ i → t j = t' j) : logits s θ t i = logits s θ t' i := by
  funext v
  have hst : ∀ k, decoderStack s θ.blocks (embedTokens θ t) i k
      = decoderStack s θ.blocks (embedTokens θ t') i k :=
    decoderStack_prefixDependent s θ.blocks (embedTokens θ t) (embedTokens θ t') i
      (embedTokens_prefix θ t t' i h)
  simp only [logits, Matrix.mul_apply]
  refine Finset.sum_congr rfl ?_
  intro l _
  rw [hst l]

/-- **Causality of a decoder-only language model**: the next-token distribution at
position `i` depends only on the tokens at positions `≤ i`. -/
theorem nextTokenDist_prefix (s : ℝ) (θ : LMParams V d f N) (t t' : Fin N → V) (i : Fin N)
    (h : ∀ j, j ≤ i → t j = t' j) : nextTokenDist s θ t i = nextTokenDist s θ t' i := by
  unfold nextTokenDist
  rw [logits_prefix s θ t t' i h]

end LM

section PreNorm

variable {V d f : Type*} [Fintype V] [Fintype d] [Fintype f] [Nonempty d] {N : ℕ}

/-- One pre-norm block written as its two residual updates. -/
structure PreNormParams (d f : Type*) where
  attn : AttnParams d
  ffn : FFNParams d f
  σ₁ : ℝ
  σ₂ : ℝ

/-- The attention sub-layer's contribution to the residual stream. -/
def attnContribution (s : ℝ) (θ : PreNormParams d f) (u : Matrix (Fin N) d ℝ) :
    Matrix (Fin N) d ℝ :=
  causalSelfAttention s θ.attn (lnRows θ.σ₁ θ.σ₂ u)

/-- The MLP sub-layer's contribution, evaluated after the attention update. -/
def mlpContribution (θ : PreNormParams d f) (u : Matrix (Fin N) d ℝ) : Matrix (Fin N) d ℝ :=
  ffn θ.ffn (lnRows θ.σ₁ θ.σ₂ u)

/-- One pre-norm block: `u ↦ u + a(u) + m(u + a(u))`. -/
def preNormBlock (s : ℝ) (θ : PreNormParams d f) (u : Matrix (Fin N) d ℝ) : Matrix (Fin N) d ℝ :=
  let u' := u + attnContribution s θ u
  u' + mlpContribution θ u'

/-- The residual stream after the first `k` blocks. -/
def residual (s : ℝ) (θs : List (PreNormParams d f)) (u₀ : Matrix (Fin N) d ℝ) :
    Matrix (Fin N) d ℝ :=
  θs.foldl (fun v θ => preNormBlock s θ v) u₀

/-- The list of all sub-layer contributions, in order. -/
def contributions (s : ℝ) (θs : List (PreNormParams d f)) (u₀ : Matrix (Fin N) d ℝ) :
    List (Matrix (Fin N) d ℝ) :=
  match θs with
  | [] => []
  | θ :: rest =>
    let a := attnContribution s θ u₀
    let m := mlpContribution θ (u₀ + a)
    a :: m :: contributions s rest (u₀ + a + m)

/-- The residual stream is the embedding plus the sum of all contributions. -/
theorem residual_eq_sum_contributions (s : ℝ) (θs : List (PreNormParams d f))
    (u₀ : Matrix (Fin N) d ℝ) :
    residual s θs u₀ = u₀ + (contributions s θs u₀).sum := by
  induction θs generalizing u₀ with
  | nil => simp [residual, contributions]
  | cons θ rest ih =>
    simp only [residual, List.foldl_cons, contributions, List.sum_cons]
    have hstep : preNormBlock s θ u₀ = u₀ + attnContribution s θ u₀
        + mlpContribution θ (u₀ + attnContribution s θ u₀) := rfl
    rw [hstep]
    have := ih (u₀ + attnContribution s θ u₀ + mlpContribution θ (u₀ + attnContribution s θ u₀))
    simp only [residual] at this
    rw [this]
    abel

/-- Logits of the pre-norm model without a final normalisation. -/
def logitsPre (s : ℝ) (θs : List (PreNormParams d f)) (u₀ : Matrix (Fin N) d ℝ)
    (WU : Matrix d V ℝ) : Matrix (Fin N) V ℝ :=
  residual s θs u₀ * WU

/-- **Direct logit attribution is exact for pre-norm models without a final
normalisation**: the logits are the sum of the direct effects of the embedding
and of every sub-layer contribution. -/
theorem logitsPre_eq_sum_contributions (s : ℝ) (θs : List (PreNormParams d f))
    (u₀ : Matrix (Fin N) d ℝ) (WU : Matrix d V ℝ) :
    logitsPre s θs u₀ WU = u₀ * WU + ((contributions s θs u₀).map (fun c => c * WU)).sum := by
  unfold logitsPre
  rw [residual_eq_sum_contributions, Matrix.add_mul]
  congr 1
  induction contributions s θs u₀ with
  | nil => simp
  | cons c cs ih => rw [List.sum_cons, List.map_cons, List.sum_cons, Matrix.add_mul, ih]

end PreNorm

end

end Params
