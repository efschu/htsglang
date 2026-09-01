#!/usr/bin/env bash
# #1047 MUTATION PROOF. The defect is a WRITER THAT DOES NOT EXIST: the
# DoublePrefillCensus carried the whole #939 arithmetic and NOTHING called it.
# Every mutant below re-introduces that defect, or one of the two ways the
# instrument could lie about the law, and each must be killed by a NAMED test.
#
# EVERY PATCH VERIFIES THAT IT APPLIED (#875d): a textual mutation that
# silently matches nothing leaves the file correct and the test green, which
# reads exactly like a covered mutant. `_patch` exits non-zero on a no-op.
# Hermetic: CUDA_VISIBLE_DEVICES forced empty. Restores on exit.
set -u

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${GATE_PY:-python3}"
SB="$WT/python/sglang/srt/managers/schedule_batch.py"
PPC="$WT/python/sglang/srt/mem_cache/producer_phase_census.py"
T="$WT/test/registered/unit/mem_cache/test_double_prefill_census_wiring_1047.py"

cp "$SB" /tmp/1047_sb.orig
cp "$PPC" /tmp/1047_ppc.orig
restore() { cp /tmp/1047_sb.orig "$SB"; cp /tmp/1047_ppc.orig "$PPC"; }
trap restore EXIT

_patch() {  # file, python-repr-old, python-repr-new
  "$PY" - "$1" "$2" "$3" <<'EOF'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
if old not in s:
    sys.stderr.write("MUTANT DID NOT APPLY: %r\n" % old)
    sys.exit(2)
open(p, "w").write(s.replace(old, new, 1))
EOF
}

run() {  # name, expected-to-fail test selector
  local name="$1"; shift
  if CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" \
     "$PY" -m pytest "$T" -q -x "$@" >/tmp/1047_out 2>&1; then
    echo "SURVIVED  $name   <-- MUTANT NOT KILLED"
    FAILED=1
  else
    echo "killed    $name"
  fi
  restore
}

FAILED=0

# M1: the writer is gone again -- the exact #1047 defect.
_patch "$SB" '_note_1047(' '_MUTANT_note(' || exit 2
run "M1 writer call removed" -k test_the_recording_site_exists

# M2: it writes but never emits -- a census nobody can read.
_patch "$SB" '_emit_1047(logger)' 'None' || exit 2
run "M2 emit call removed" -k test_the_recording_site_exists

# M3: the population gate is gone -- OOM re-prefills enter the denominator.
_patch "$SB" 'SEAM_READMIT_ATTR as _SRA_1047' 'FLIP_EPOCH_ATTR as _SRA_1047' || exit 2
run "M3 seam population gate removed" -k test_the_site_gates_on_the_seam_population

# M4: disarmed no longer means silent -- default path stops being byte-identical.
_patch "$PPC" '    if census_armed() <= 0:
        return
    if _dpc is None:' '    if _dpc is None:' || exit 2
run "M4 disarmed guard removed" -k test_disarmed_builds_nothing

# M5: THE DANGER DIRECTION. `<` instead of `<=`: a loss of exactly one chunk
# is the ATTAINED bound and must PASS; this mutant fails it and would report a
# lawful boot as a breach.
_patch "$PPC" 'return self.worst_request_tokens <= int(self.chunk_size)' \
              'return self.worst_request_tokens < int(self.chunk_size)' || exit 2
run "M5 bound off-by-one (attained must pass)" -k test_bound_is_attained

# M6: the census is drained by its own emission -- `worst` stops being the
# worst of the cutover and becomes the worst since the last line.
_patch "$PPC" '        return True
    _dpc_suppressed += 1' '        globals()["_dpc"] = None
        return True
    _dpc_suppressed += 1' || exit 2
run "M6 emission drains the census" -k test_worst_is_monotone

# M7: a breach can be sampled away -- the finding disappears under a high
# rate-limit, which is how a broken law reads as a quiet boot.
_patch "$PPC" 'if breach or _dpc_emitted % every == 0:' 'if _dpc_emitted % every == 0:' || exit 2
run "M7 breach no longer forces emission" -k test_breach_is_never_sampled_away

echo
if [ "$FAILED" = "1" ]; then echo "#1047 MUTATION PROOF: FAILED"; exit 1; fi
echo "#1047 MUTATION PROOF: all mutants killed"
