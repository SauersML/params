import Params
import Lean
import Lean.Util.CollectAxioms

/-!
# Elaborated-environment check: axiom closure of every `Params` declaration

Run after a successful build:

    lake env lean validation/code/Check.lean

Fails (exit code 1) if any declaration in the `Params` namespace depends on an
axiom outside Lean's three foundations (`propext`, `Classical.choice`,
`Quot.sound`) — in particular on `sorryAx` or on the compiler axioms behind
`native_decide`.  Prints one line per declaration so the log is auditable.
-/

open Lean Elab Command Meta

namespace Check

def allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def isOurs (n : Name) : Bool := n.getRoot == `Params

def userWritten (n : Name) : Bool := isOurs n && !n.isInternal

end Check

open Check in
elab "#check_axioms" : command => do
  let env ← getEnv
  let mut bad : Array (Name × List Name) := #[]
  let mut count := 0
  for (n, _) in env.constants.toList do
    if userWritten n then
      let (_, s) := ((CollectAxioms.collect n).run env).run {}
      count := count + 1
      let extra := s.axioms.toList.filter (fun a => !(allowed.contains a))
      logInfo m!"{n}: {s.axioms.toList}"
      if !extra.isEmpty then
        bad := bad.push (n, extra)
  logInfo m!"AXIOM_SCAN_SCANNED {count}"
  if bad.isEmpty then
    logInfo m!"AXIOM_SCAN_OK"
  else
    for (n, extra) in bad do
      logError m!"AXIOM_SCAN_FAIL {n} uses {extra}"

#check_axioms
