#!/usr/bin/env bash
# #875d MUTATION PROOF. Each mutant is a plausible wrong version of one
# load-bearing decision; each must be killed by a NAMED test. A guard nobody can
# red is a guard nobody has tested.
#
# EVERY PATCH VERIFIES THAT IT APPLIED. A textual mutation that silently matches
# nothing leaves the file correct and the test green, and reads exactly like a
# covered mutant -- measured here on 2026-08-26, when `ruff format` reflowed one
# target line and M2's replacement quietly became a no-op. `_patch` exits
# non-zero on a no-op, so a broken mutant is a BROKEN MUTANT and never a
# survivor and never a pass.
#
# Hermetic: CUDA_VISIBLE_DEVICES is forced empty. Restores every file it touches
# on exit, including on failure.
set -u

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${GATE_PY:-/spinning/htsglang-gpu/.venv/bin/python3}"
MP="$WT/python/sglang/srt/mem_cache/memory_pool.py"
SC="$WT/python/sglang/srt/mem_cache/seam_layer_carry.py"
SB="$WT/python/sglang/srt/managers/schedule_batch.py"
T875D="$WT/test/registered/unit/mem_cache/test_seam_carry_wired_875d.py"
T861C="$WT/test/registered/unit/managers/test_seam_layout_contract_861c.py"

cp "$MP" /tmp/875d_mp.orig; cp "$SC" /tmp/875d_sc.orig; cp "$SB" /tmp/875d_sb.orig
restore() { cp /tmp/875d_mp.orig "$MP"; cp /tmp/875d_sc.orig "$SC"; cp /tmp/875d_sb.orig "$SB"; }
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

run() {  # run <test-file> <-k expr> -> PASS/FAIL
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -m pytest -q "$1" -k "$2" -p no:randomly 2>&1 \
    | sed -e 's/\x1b\[[0-9;]*m//g' | grep -qE "^[0-9]+ passed" && echo PASS || echo FAIL
}

check() {  # check <name> <test-file> <-k expr>
  local name="$1" file="$2" expr="$3"
  local got; got="$(run "$file" "$expr")"
  if [ "$got" = "FAIL" ]; then
    echo "KILLED   $name  <- $(basename "$file")::$expr"
  else
    echo "SURVIVED $name  -- $(basename "$file")::$expr stayed green. Not covered."
    FAILED=1
  fi
  restore
}

echo "== M1: the mamba pool goes back to discarding its identity"
_patch "$MP" \
  '        self.mamba_layer_ids = tuple(int(x) for x in mamba_layer_ids)' \
  '        self._discarded = tuple(int(x) for x in mamba_layer_ids)' || FAILED=1
check M1 "$T875D" "test_the_pool_records_the_ids_rather_than_only_their_count"

echo "== M2: the declared layout reverts WHOLESALE to the pre-#875d form"
# Two equal-sized stages then both declare ("mamba", n, 0) and #861c's drift
# check passes them -- the defect that predates this ticket.
_patch "$MP" \
  '            start_layer=int(ids[0])
            if ids
            else int(getattr(self, "start_layer", 0) or 0),
            layer_ids=ids,' \
  '            start_layer=int(getattr(self, "start_layer", 0) or 0),
            layer_ids=None,' || FAILED=1
check M2 "$T875D" "test_two_equal_sized_stages_no_longer_compare_EQUAL"

echo "== M2b: the ids are computed and then not declared"
_patch "$MP" '            layer_ids=ids,' '            layer_ids=None,' || FAILED=1
check M2b "$T875D" "test_the_declared_layout_names_the_ids_not_a_defaulted_zero"

echo "== M3: the plan slices positionally from the front instead of by id"
_patch "$SC" \
  '        take=tuple(position[int(g)] for g in dst_ids),' \
  '        take=tuple(range(len(dst_ids))),' || FAILED=1
check M3 "$T875D" "test_a_tp_copy_is_CARRIED_into_a_pp_stage or test_the_plan_selects_by_id_when_the_ids_are_not_a_range"

echo "== M4: a missing layer is dropped instead of refused (the partial restore)"
_patch "$SC" \
  '    missing = [int(g) for g in dst_ids if int(g) not in position]' \
  '    missing = []
    dst_ids = tuple(g for g in dst_ids if int(g) in position)' || FAILED=1
check M4 "$T875D" "test_a_pp_copy_into_a_tp_pool_STILL_refuses"

echo "== M5: an unrecognised layout is guessed at instead of refused"
_patch "$SC" '    if not isinstance(layout, _layout_type()):' '    if False:' || FAILED=1
_patch "$SC" '    return layout.global_layers()' \
  "    return tuple(range(int(layout[1]))) if not hasattr(layout, 'global_layers') else layout.global_layers()" || FAILED=1
check M5 "$T861C" "test_equal_layer_counts_at_different_offsets_are_refused"

echo "== M6: the hybrid carry keeps the KV half and drops the mamba half"
_patch "$SC" \
  '        return carried_kv, carry_payload(src_layout[2], dst_layout[2], mamba_cpu)' \
  '        return carried_kv, mamba_cpu' || FAILED=1
check M6 "$T875D" "test_a_composite_hybrid_payload_carries_both_halves"

echo "== M7: the carry runs BEFORE the extent contract"
_patch "$SB" \
  '        req.mamba_state_cpu_layout = None
        return False

    # #875: THIS REFUSAL WAS A NON-ANSWER' \
  '        req.mamba_state_cpu_layout = None

    # #875: THIS REFUSAL WAS A NON-ANSWER' || FAILED=1
check M7 "$T875D" "test_an_extent_drift_still_refuses_BEFORE_any_carry_is_attempted"

echo "== M8: the stale source layout is left on a carried copy"
_patch "$SB" \
  '                req.kv_cache_cpu = carried_kv
                req.kv_cache_cpu_layout = live_layout' \
  '                req.kv_cache_cpu = carried_kv' || FAILED=1
check M8 "$T875D" "test_the_stamp_describes_the_payload_AT_THE_MOMENT_IT_IS_HANDED_OVER"

echo "== M9: a carried restore is also counted as a refusal"
_patch "$SB" '            kv_drifted = mamba_drifted = False' '            pass' || FAILED=1
check M9 "$T875D" "test_the_carry_does_not_count_as_a_refusal"

echo
if [ "$FAILED" = "0" ]; then echo "ALL MUTANTS KILLED"; else echo "SOME MUTANTS SURVIVED OR WERE BROKEN"; exit 1; fi
