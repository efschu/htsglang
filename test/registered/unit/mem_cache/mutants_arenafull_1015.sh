#!/usr/bin/env bash
# ARENAFULL-1015 mutation check (NF h16n deaths 07:11:54 / 07:30:35 PP1). One plausible WRONG implementation per
# new decision; each MUST turn test_weg2_arenafull_1015.py red. A survivor = that decision is asserted nowhere.
# ONLY=<name> runs one mutant. Runs every suite through /spinning/gpu-arb/pytest_gedeckelt.sh (CPU only). Restores every file on exit.
#
#   MA  L3-DROP off by construction: an l3_present leaf takes the claim-first path
#   MB  the store-write ack drain never drains (UD-H stays refused on the demoted tail)
#   MC  the ack drain waits on ANY host-locked child (also a prefetch / pin holder no drain frees)
#   MD  L3-DROP ignores the local-PP floor (a rank-local drop on a TP replica group)
#   ME  L3-DROP drops without clearing the host-only children first (#841 orphan)
#   MF  W3-KEEP-PAYABLE filter removed (the spill un-backs a node whose drop is blocked)
#   MG  W3-KEEP-PAYABLE inverted (keeps the payable candidates, spills the blocked one)
#   MH  B1: the res < own branch acks res without the room check
#   MI  B1: the satisfied res < told branch acks res without the room check
#   MJ  ACK-ROOM-PAYABLE: the ack's room counts the reported evictable size again
#   MK  arena.c reap_partial ages by touched_ms again (a JOIN keeps the orphan young)
#   ML  arena.c reap_partial takes a claim with an open writer
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
URC=$ROOT/python/sglang/srt/mem_cache/unified_radix_cache.py
FB=$ROOT/python/sglang/srt/managers/weg2_told_fallback.py
AC=$ROOT/python/sglang/srt/mem_cache/storage/file/arena.c
TESTS=test/registered/unit/mem_cache/test_weg2_arenafull_1015.py
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}

BAK_DIR=$(mktemp -d)
cp "$URC" "$BAK_DIR/urc.py"; cp "$FB" "$BAK_DIR/fb.py"; cp "$AC" "$BAK_DIR/arena.c"
restore() { cp "$BAK_DIR/urc.py" "$URC"; cp "$BAK_DIR/fb.py" "$FB"; cp "$BAK_DIR/arena.c" "$AC"; }
trap restore EXIT

run_suite() { (cd "$ROOT" && "$GED" "$TESTS" -x -p no:warnings >/dev/null 2>&1); }

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
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then return; fi
  restore
  if ! mutate "$file" "$old" "$new"; then echo "$name ANCHOR-MISSING"; survivors=$((survivors + 1)); return; fi
  if run_suite; then echo "$name SURVIVED"; survivors=$((survivors + 1)); else echo "$name killed"; fi
}

check MA "$URC" '        if not getattr(node, "l3_present", False) or node.backuped:
            return False
        if not UnifiedRadixCache._weg2_ud_ack_drain_on():' '        if True:
            return False
        if not UnifiedRadixCache._weg2_ud_ack_drain_on():'
check MB "$URC" '        if not held:
            return -1.0' '        if True:
            return -1.0'
check MC "$URC" '        if not all(id(d) in writing for d in held):' '        if not writing:'
check MD "$URC" '        if not _sf.unbacked_drop_floor(self, node):
            return False
        waited_ms = 0.0' '        waited_ms = 0.0'
check ME "$URC" '        if not _sf.unbacked_drop_allowed(self, node):
            udh = getattr(self, "_ud_h_refusal", None)' '        if False:
            udh = getattr(self, "_ud_h_refusal", None)'
check MF "$URC" '            if keep_payable is not None and not all(' '            if False and not all('
check MG "$URC" '            if keep_payable is not None and not all(' '            if keep_payable is not None and all('
check MH "$FB" '    return _room_own(scheduler, req, rid, int(res))' '    return int(res)'
check MI "$FB" '    res = _room_own(scheduler, req, rid, int(res))
    n = _bump(scheduler, "_pf_ack_satisfied_unresumable_n")' '    n = _bump(scheduler, "_pf_ack_satisfied_unresumable_n")'
check MJ "$FB" '        return max(0, min(reported, int(walked)))' '        return reported'
check MK "$AC" '        uint32_t p = atomic_load(&sh->progress_ms);
        if ((uint32_t)(now - p) < (uint32_t)min_age_ms) continue;' '        uint32_t p = t;
        if ((uint32_t)(now - p) < (uint32_t)min_age_ms) continue;'
check ML "$AC" '        if ((w >> 16) == 0 || (w & W_OPEN) != 0) continue;
        uint32_t t = atomic_load(&sh->touched_ms);' '        if ((w >> 16) == 0) continue;
        uint32_t t = atomic_load(&sh->touched_ms);'

restore
echo "survivors=$survivors"
exit $(( survivors > 0 ))
