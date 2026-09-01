#!/usr/bin/env bash
# #1061 MUTATION PROOF. The defect this ticket names is a WRITER THAT DOES NOT
# EXIST (built-never-wired), so every mutant here is exactly that defect
# re-introduced at one wired site; each must be killed by a NAMED test. A
# wiring nobody can red is a wiring nobody has tested.
#
# EVERY PATCH VERIFIES THAT IT APPLIED (#875d's lesson): a textual mutation
# that silently matches nothing leaves the file correct and the test green,
# which reads exactly like a covered mutant. `_patch` exits non-zero on a
# no-op. Hermetic: CUDA_VISIBLE_DEVICES forced empty. Restores on exit.
set -u

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${GATE_PY:-/spinning/htsglang-gpu/.venv/bin/python3}"
URC="$WT/python/sglang/srt/mem_cache/unified_radix_cache.py"
CC="$WT/python/sglang/srt/managers/cache_controller.py"
HPB="$WT/python/sglang/srt/mem_cache/hicache_phase_binding.py"
PPC="$WT/python/sglang/srt/mem_cache/producer_phase_census.py"
T="$WT/test/registered/unit/mem_cache/test_producer_phase_census_wiring_1061.py"

cp "$URC" /tmp/1061_urc.orig
cp "$CC" /tmp/1061_cc.orig
cp "$HPB" /tmp/1061_hpb.orig
cp "$PPC" /tmp/1061_ppc.orig
restore() {
  cp /tmp/1061_urc.orig "$URC"
  cp /tmp/1061_cc.orig "$CC"
  cp /tmp/1061_hpb.orig "$HPB"
  cp /tmp/1061_ppc.orig "$PPC"
}
trap restore EXIT

FAILED=0

_patch() {  # _patch <file> <old> <new>   -- refuses a no-op
  "$PY" - "$1" "$2" "$3" <<'EOF' || { echo "BROKEN MUTANT: the patch matched nothing"; exit 1; }
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if old not in s:
    sys.exit(1)
open(path, "w").write(s.replace(old, new, 1))
EOF
}

run() {  # run <-k expr> -> PASS/FAIL
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -m pytest -q "$T" -k "$1" -p no:randomly 2>&1 \
    | sed -e 's/\x1b\[[0-9;]*m//g' | grep -qE "^[0-9]+ passed" && echo PASS || echo FAIL
}

check() {  # check <name> <-k expr>
  local name="$1" expr="$2"
  local got; got="$(run "$expr")"
  if [ "$got" = "FAIL" ]; then
    echo "KILLED   $name  <- $expr"
  else
    echo "SURVIVED $name  -- $expr stayed green. Not covered."
    FAILED=1
  fi
  restore
}

echo "== M1: the walk classifies nothing (the node feed is unwired again)"
_patch "$URC" 'if _pp_note_walk_node(' 'if False and _pp_note_walk_node('
check "M1 walk feed removed" "test_real_walk_emits_the_acceptance_line"

echo "== M2: the walk never emits (the line is unwired again)"
_patch "$URC" 'if _producer_emit(p_census, logger):' \
  'if False and _producer_emit(p_census, logger):'
check "M2 walk emit removed" "test_real_walk_emits_the_acceptance_line"

echo "== M3: the backup thread stops stamping (the ledger writer is unwired)"
_patch "$CC" '                        note_backup_keys(' \
  '                        (lambda *a, **k: None)('
check "M3 backup stamp removed" "test_backup_thread_stamps_keys_with_the_operation_stamp"

echo "== M4: advance stops recording generation -> phase"
_patch "$HPB" '            note_generation(prior_generation, prior_phase)
            note_generation(generation, str(phase))' \
  '            pass'
check "M4 generation history removed" "test_advance_records_both_phases_of_a_cutover or test_walk_node_classifies_a_cross_phase_hit"

echo "== M5: the glue writer itself is gutted (records nothing while armed)"
_patch "$PPC" '    if census_armed() <= 0 or not keys or generation is None:
        return
    for k in keys:
        note_store_write(k, generation)' \
  '    return'
check "M5 note_backup_keys gutted" "test_armed_backup_keys_stamp_the_ledger"

echo
if [ "$FAILED" -eq 0 ]; then
  echo "ALL MUTANTS KILLED"
else
  echo "SURVIVORS PRESENT -- the wiring is weaker than it looks"
fi
exit "$FAILED"
