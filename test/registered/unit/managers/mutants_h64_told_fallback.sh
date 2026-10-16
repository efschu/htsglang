#!/usr/bin/env bash
# H64 mutation check (NF h16n PF TOLD-FALLBACK reason=frist, 1015). One
# plausible WRONG implementation per new decision; each MUST turn
# test_pdflip_h64_told_fallback_1015.py red. A survivor = that decision is
# asserted nowhere. Runs every suite through /spinning/gpu-arb/pytest_gedeckelt.sh.
#
#   MA  SETTLE-ACK-NOW off: a registration that issued nothing stays "short"
#   MB  FRIST-PASSES ignored: the wall Frist alone decides
#   MC  ACK-DRAIN off: PP0 stops harvesting without an open paced rid
#   MD  FLOOR takes the MIN of differing acks (a stage that acked more proved nothing)
#   ME  FLOOR from an ack at/over told (no upper bound)
#   MF  FLOOR on a partial ack picture (missing ack treated as agreeing)
#   MG  FLOOR skips PP0's own TF verdict at the floor depth
#   MH  a follower ignores the floor marker (admits the plain told)
#   MI  PP0 does not apply its own floor (its record says told)
#
# Hermetic: no CUDA. Restores both files unconditionally on exit.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
FB=$ROOT/python/flliper/srt/managers/pdflip_told_fallback.py
ST=$ROOT/python/flliper/srt/managers/pdflip_store_told.py
TESTS=test/registered/unit/managers/test_pdflip_h64_told_fallback_1015.py
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}

BAK_DIR=$(mktemp -d)
cp "$FB" "$BAK_DIR/fb.py"
cp "$ST" "$BAK_DIR/st.py"
restore() {
  cp "$BAK_DIR/fb.py" "$FB"
  cp "$BAK_DIR/st.py" "$ST"
}
trap restore EXIT

run_suite() {
  (cd "$ROOT" && "$GED" "$TESTS" -x >/dev/null 2>&1)
}

mutate() {
  python3 - "$1" "$2" "$3" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n = s.count(old)
if n != 1:
    sys.exit(f"mutation anchor found {n}x in {path}: {old!r}")
open(path, "w").write(s.replace(old, new))
EOF
}

survivors=0
check() {
  local name=$1 file=$2 old=$3 new=$4
  restore
  if ! mutate "$file" "$old" "$new"; then
    echo "$name ANCHOR-MISSING"; survivors=$((survivors + 1)); return
  fi
  if run_suite; then
    echo "$name SURVIVED"; survivors=$((survivors + 1))
  else
    echo "$name killed"
  fi
}

if ! run_suite; then echo "baseline RED -- fix the suite first"; exit 2; fi
echo "baseline green"

check MA "$ST" 'if settle_ack_now_on() and not str(verdict).startswith("issued"):' 'if False:'
check MB "$FB" '    if done >= need:
        return True' '    if True:
        return True'
check MC "$ST" 'and _fb.pp0_has_stream(scheduler)' 'and False'
check MD "$FB" '    if len(vals) != 1:
        return None
    m = next(iter(vals))' '    if not vals:
        return None
    m = min(vals)'
check ME "$FB" 'return m if 0 < m < int(o.told) else None' 'return m if 0 < m and m != int(o.told) else None'
check MF "$FB" '        if complete:
            return m, REASON_FLOOR' '        return m, REASON_FLOOR'
check MG "$ST" '            if told_final > 0:
                # TF: the Admit is PP0'"'"'s own admission' '            if told_final > 0 and reason != _fb.REASON_FLOOR:
                # TF: the Admit is PP0'"'"'s own admission'
check MH "$ST" '    elif floor and told > 0:' '    elif False:'
check MI "$ST" '                _fb.apply_floor(scheduler, rid, told_final)
                setattr(admit, _fb.WIRE_FLOOR, 1)' '                setattr(admit, _fb.WIRE_FLOOR, 1)'

restore
echo "survivors=$survivors"
exit $survivors
