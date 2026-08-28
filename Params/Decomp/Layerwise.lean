import Mathlib
import Params.Decomp.Basic

set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

/-!
# The layer-wise stochastic reconstruction loss

SPD (arXiv:2506.20790, eq. layerwise_random_recon) also trains with a loss in
which only one layer at a time is masked, and remarks that this agrees with
the all-layers loss on the mask samples that keep every other layer at `1`,
*provided the subcomponents sum exactly to the target weights*.  The remark is
an identity about weight configurations, proved here for a family of layers
of a common shape (`spd_layerwise_config`): if every layer is faithfully
decomposed and the sample masks are `1` outside layer `l`, the all-layers
masked configuration is the target configuration with layer `l` replaced by
its masked weight.  The second half of the file is the same statement for an
arbitrary family of weights (`layerwise_eq_full`).
-/

namespace Params.Decomp

noncomputable section

section Abstract

variable {L W : Type*} [DecidableEq L]

/-- A weight family that agrees with `w` off `l` and equals `w'` at `l` is `update w l w'`. -/
theorem layerwise_eq_full (w : L → W) (l : L) (w' : W) (m : L → W)
    (hm : ∀ l', l' ≠ l → m l' = w l') (hl : m l = w') : m = Function.update w l w' := by
  funext l'
  by_cases h : l' = l
  · subst h
    rw [Function.update_self, hl]
  · rw [Function.update_of_ne h, hm l' h]

end Abstract

section SPD

variable {L a b C : Type*} [DecidableEq L] [Fintype a] [Fintype b] [Fintype C] [DecidableEq C]

/-- **Layer-wise and all-layers masking agree** on samples whose masks are `1` off layer `l`,
when every layer is faithfully decomposed. -/
theorem spd_layerwise_config (W : L → Matrix a b ℝ) (D : L → Subcomponents a b C)
    (hfaith : ∀ l, faithfulnessLoss (W l) (D l) = 0) (m : L → C → ℝ) (l : L)
    (hm : ∀ l', l' ≠ l → m l' = fun _ => 1) :
    (fun l' => (D l').masked (m l')) = Function.update W l ((D l).masked (m l)) := by
  apply layerwise_eq_full
  · intro l' hl'
    rw [hm l' hl']
    exact masked_one_of_faithful (W l') (D l') (hfaith l')
  · rfl

end SPD

end

end Params.Decomp
