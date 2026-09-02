#!/usr/bin/env bash
# #968 mutation check. Each mutation is a plausible WRONG implementation in the
# direction of the defect this rebuild exists to close, and every one MUST make
# test_968_starvation_umbau.py fail. A survivor means the law is asserted
# nowhere.
#
#   M1  materialisation shortfall silently returns (today's defect -- the
#       voided/short-prefix pass -- returning as a fallback)
#   M2  the over-serve clamp is removed (the instr20 direction: prefix grows
#       past the decision)
#   M3  the store-presence refuter is neutered (the premise trusts retract
#       stamps alone again -- the W30 arm feeder)
#   M4  RETIRED (#1068 slice 2): the prefetch-cap floor it mutated is gone;
#       the budget is a property of the bound host pool (see
#       test_prefetch_limit_property_1068.py for its mutants)
#
# Hermetic: no CUDA, no NVML. Restores every file unconditionally on exit.
set -u

ROOT=${ROOT:-/spinning/wt-968-umbau}
CONG=$ROOT/python/sglang/srt/managers/pp_admission_congruence.py
PUR=$ROOT/python/sglang/srt/managers/phase_purity.py
CC=$ROOT/python/sglang/srt/managers/cache_controller.py
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
TESTS=$ROOT/test/registered/unit/managers/test_968_starvation_umbau.py

BAK_DIR=$(mktemp -d)
cp "$CONG" "$BAK_DIR/cong.py"
cp "$PUR" "$BAK_DIR/pur.py"
cp "$CC" "$BAK_DIR/cc.py"
restore() {
  cp "$BAK_DIR/cong.py" "$CONG"
  cp "$BAK_DIR/pur.py" "$PUR"
  cp "$BAK_DIR/cc.py" "$CC"
}
trap restore EXIT

run_suite() {
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES= PYTHONPATH=python \
    timeout 300 "$PY" -m pytest "$TESTS" -q >/dev/null 2>&1)
}

mutate() { # file, python-inline-mutator
  "$PY" - "$1" <<PYEOF
import sys
path = sys.argv[1]
src = open(path).read()
$2
open(path, "w").write(src)
PYEOF
}

fail=0
verdict() { # name, expect-fail run result (0 = suite passed = mutant survived)
  if [ "$2" -eq 0 ]; then
    echo "$1 SURVIVED -- the law is asserted nowhere"
    fail=1
  else
    echo "$1 killed"
  fi
}

# M1: shortfall silently returns the partial prefix.
mutate "$CONG" '
needle = "        if time.monotonic() - started >= bound_s:\n            raise _die("
assert needle in src, "M1 anchor missing"
src = src.replace(
    needle,
    "        if time.monotonic() - started >= bound_s:\n"
    "            return loaded_total\n"
    "        if False:\n"
    "            raise _die(",
    1,
)
'
run_suite; verdict M1 $?
restore

# M2: the over-serve clamp is removed.
mutate "$CONG" '
needle = "            if applied > deficit:\n                # The instr20 direction: never grow past the decision.\n                new_indices = new_indices[:deficit]\n                applied = deficit"
assert needle in src, "M2 anchor missing"
src = src.replace(needle, "            if False:\n                pass", 1)
'
run_suite; verdict M2 $?
restore

# M3: the presence refuter is neutered.
mutate "$PUR" '
needle = "        refuted, tiers = seam_store_presence_refuted(scheduler)"
assert needle in src, "M3 anchor missing"
src = src.replace(needle, "        refuted, tiers = False, \"neutered\"", 1)
'
run_suite; verdict M3 $?
restore

# M4 retired with #1068 slice 2: the prefetch-cap floor it mutated is gone
# (the budget is a property of the bound host pool; its mutants live next to
# test_prefetch_limit_property_1068.py).

exit $fail
