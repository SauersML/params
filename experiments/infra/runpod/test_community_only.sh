#!/usr/bin/env bash
# Pure offer/packing contract: no API, credentials, or pod operations.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=$(mktemp -d); trap 'rm -rf "$dir"' EXIT
sed -n '/^cap_offers() {/,/^}/p; /^find_host() {/,/^}/p' "$here/rp-run" > "$dir/functions.sh"
source "$dir/functions.sh"
all=$'COMMUNITY|0.34|card\nSECURE:no-community-stock|0.25|card\nTHUNDER|0.20|other\nCOMMUNITY|0.50|expensive'
OFFERS=$all RP_COMMUNITY_ONLY=0
cap_offers
[ "$OFFERS" = "$all" ]
OFFERS=$all RP_COMMUNITY_ONLY=1
cap_offers
[ "$OFFERS" = $'COMMUNITY|0.34|card\nCOMMUNITY|0.50|expensive' ]
OFFERS=$all RP_MAX_PRICE=0.40
cap_offers >/dev/null
[ "$OFFERS" = 'COMMUNITY|0.34|card' ]
OFFERS='SECURE:dev|0.20|card'
cap_offers >/dev/null
[ -z "$OFFERS" ]
# If packing is reached with the option, return before touching any active pod.
MODE=run RP_MIN_VRAM_GB=24 RP_PACK=1
unset ACTIVE
find_host
# Invalid option is rejected at startup, before any provider/credential work.
set +e
RP_COMMUNITY_ONLY=yes bash "$here/rp-run" > "$dir/invalid" 2>&1
rc=$?
set -e
[ "$rc" = 2 ]
grep -q 'RP_COMMUNITY_ONLY must be 0 or 1' "$dir/invalid"
echo 'PASS: community-only filters all offers, combines with price cap, disables packing, rejects invalid values'
