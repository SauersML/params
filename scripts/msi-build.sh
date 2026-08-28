#!/usr/bin/env bash
# One compile of the corpus on an MSI compute node, against the warm Mathlib cache.
# Policy: at most one compile per hour (enforced by the stamp below); never build locally.
#
#   msi 'bash /projects/standard/hsiehph/sauer354/params/scripts/msi-build.sh'
#
# Writes a full log under validation/code/results/ and prints its tail.
set -uo pipefail
ROOT=/projects/standard/hsiehph/sauer354/params
WARM=/projects/standard/hsiehph/sauer354/nonsofic_existence
STAMP=$ROOT/validation/code/results/last_compile_epoch
JOBS=${JOBS:-16}
FORCE=${FORCE:-0}

cd "$ROOT" || exit 2
git fetch -q origin && git reset -q --hard origin/main
now=$(date +%s)
if [ "$FORCE" != "1" ] && [ -r "$STAMP" ]; then
  last=$(cat "$STAMP")
  if [ $((now - last)) -lt 3600 ]; then
    echo "REFUSED: last compile was $((now - last))s ago (< 3600s)"; exit 3
  fi
fi
echo "$now" > "$STAMP"
export PATH="$HOME/.elan/bin:$PATH"

# Warm cache: Mathlib 81a5d25 already built under nonsofic_existence.
if [ ! -d .lake/packages/mathlib/.lake/build ]; then
  mkdir -p .lake && cp -r "$WARM/.lake/packages" .lake/
fi
if [ ! -f lake-manifest.json ]; then
  python3 - <<'PY'
import json
m = json.load(open("/projects/standard/hsiehph/sauer354/nonsofic_existence/lake-manifest.json"))
m["name"] = "Params"
json.dump(m, open("lake-manifest.json", "w"), indent=1)
PY
fi

LOG=$ROOT/validation/code/results/build-$(date +%Y%m%dT%H%M%S).log
{
  echo "== commit $(git rev-parse HEAD)  node $(hostname)  jobs $JOBS"
  echo "== lake build"
  /usr/bin/time -v lake build -j "$JOBS" Params 2>&1
  echo "== exit $?"
  echo "== axiom scan"
  lake env lean validation/code/Check.lean 2>&1 | grep -E "AXIOM_SCAN|error|sorryAx" | head -100
} > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" "$ROOT/validation/code/results/build-latest.log"
grep -E "^error|^warning|error:|AXIOM_SCAN|== exit|Elapsed" "$LOG" | head -150
