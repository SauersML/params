import Mathlib
import Params.Decomp.Basic

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# Rank bounds for parameter decompositions

A linear parameter decomposition writes a weight as a sum of rank-one pieces.
Rank is subadditive, so any *faithful* decomposition of `W` into rank-one
subcomponents uses at least `rank W` non-zero pieces
(`card_ge_rank_of_faithful`).  In particular a hidden identity `I_k` needs at
least `k` subcomponents (`card_ge_of_faithful_one`) — the observation behind
the "TMS + identity" experiment of the SPD paper (Section 4.2), where the
identity is recovered as exactly `m₁` rank-one pieces.
-/

namespace Params.Decomp

noncomputable section

open Finset

section RankAdd

variable {m n : Type*} [Fintype m] [Fintype n]

/-- Rank is subadditive. -/
theorem rank_add_le' (A B : Matrix m n ℝ) : (A + B).rank ≤ A.rank + B.rank := by
  simp only [Matrix.rank]
  rw [Matrix.mulVecLin_add]
  have hle : LinearMap.range (A.mulVecLin + B.mulVecLin)
      ≤ LinearMap.range A.mulVecLin ⊔ LinearMap.range B.mulVecLin := by
    rintro _ ⟨x, rfl⟩
    rw [LinearMap.add_apply]
    exact Submodule.add_mem_sup (LinearMap.mem_range_self _ x) (LinearMap.mem_range_self _ x)
  exact (Submodule.finrank_mono hle).trans (Submodule.finrank_add_le_finrank_add_finrank _ _)

/-- Rank of a finite sum is at most the sum of the ranks. -/
theorem rank_sum_le {C : Type*} [DecidableEq C] (s : Finset C) (P : C → Matrix m n ℝ) :
    (∑ c ∈ s, P c).rank ≤ ∑ c ∈ s, (P c).rank := by
  induction s using Finset.induction_on with
  | empty => simp [Matrix.rank_zero]
  | @insert a s ha ih =>
    rw [Finset.sum_insert ha, Finset.sum_insert ha]
    exact (rank_add_le' _ _).trans (Nat.add_le_add_left ih _)

/-- A faithful decomposition into rank-`≤ 1` pieces has at least `rank W` pieces. -/
theorem card_ge_rank_of_faithful {C : Type*} [Fintype C] [DecidableEq C] (W : Matrix m n ℝ)
    (P : C → Matrix m n ℝ) (hfaith : ∑ c, P c = W) (hrank : ∀ c, (P c).rank ≤ 1) :
    W.rank ≤ Fintype.card C := by
  rw [← hfaith]
  refine (rank_sum_le Finset.univ P).trans ?_
  calc ∑ c, (P c).rank ≤ ∑ _c : C, 1 := Finset.sum_le_sum (fun c _ => hrank c)
    _ = Fintype.card C := by simp

/-- A faithful decomposition of `W` into outer products has at least `rank W` pieces. -/
theorem card_ge_rank_of_faithful_outer {C : Type*} [Fintype C] [DecidableEq C]
    (W : Matrix m n ℝ) (u : C → m → ℝ) (v : C → n → ℝ)
    (hfaith : ∑ c, outer (u c) (v c) = W) : W.rank ≤ Fintype.card C :=
  card_ge_rank_of_faithful W (fun c => outer (u c) (v c)) hfaith (fun _ => rank_outer_le _ _)

end RankAdd

section Identity

variable {k : Type*} [Fintype k] [DecidableEq k]

/-- The identity on `k` coordinates needs at least `card k` rank-one subcomponents. -/
theorem card_ge_of_faithful_one {C : Type*} [Fintype C] [DecidableEq C]
    (u : C → k → ℝ) (v : C → k → ℝ) (hfaith : ∑ c, outer (u c) (v c) = (1 : Matrix k k ℝ)) :
    Fintype.card k ≤ Fintype.card C := by
  have h := card_ge_rank_of_faithful_outer (1 : Matrix k k ℝ) u v hfaith
  rwa [Matrix.rank_one] at h

end Identity

end

end Params.Decomp
