#!/usr/bin/env bash
# #1348 mutant harness -- DANGER DIRECTION FIRST.
#
# The danger direction for a coverage instrument is NOT "it misses a line".
# It is "it reports a clean sweep it never measured": a reader who gets
# `0 unexecuted` from an absent trace closes the seam the instrument exists to
# open, and does it with more confidence than if there had been no instrument
# at all. Every mutant below therefore makes the tool QUIETER or MORE
# CONFIDENT, never noisier. Each must turn the file RED; the baseline must be
# GREEN before and after.
#
#   bash test/registered/unit/weg2/tools_mutants_1348.sh     # from the repo root
set -u
TEST=test/registered/unit/weg2/test_weg2_lane_coverage_1348.py
MOD=python/sglang/srt/weg2/lane_coverage.py
ING=scripts/weg2/lane_coverage_diff.py
LAU=python/sglang/srt/weg2/launcher.py
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$(pwd)/python"
PY=${PY:-python3}

WUP=python/sglang/srt/managers/scheduler_components/weight_updater.py
# RESTORED FROM A COPY, NEVER FROM GIT. `git checkout --` here would revert the
# WORKING TREE, i.e. destroy exactly the uncommitted work this harness is
# supposed to be testing. Copies, and a trap that runs on every exit path.
BAK=$(mktemp -d)
cp "$MOD" "$BAK/mod.py"; cp "$ING" "$BAK/ing.py"
cp "$LAU" "$BAK/lau.py"; cp "$WUP" "$BAK/wup.py"
restore() {
  cp "$BAK/mod.py" "$MOD"; cp "$BAK/ing.py" "$ING"
  cp "$BAK/lau.py" "$LAU"; cp "$BAK/wup.py" "$WUP"
}
trap 'restore; rm -rf "$BAK"' EXIT

run() {  # $1 = label, $2 = expectation (GREEN|RED), $3.. = pytest node args
  local label="$1" want="$2"; shift 2
  # The verdict is read off the ESCAPE-STRIPPED tail: a colour code between
  # "passed" and the comma once made a green baseline read as RED, which is a
  # harness lie in precisely the direction that hides a surviving mutant.
  out=$(timeout 900 "$PY" -m pytest "$@" -q -p no:randomly -p no:cacheprovider --color=no 2>&1 \
        | sed -r 's/\x1b\[[0-9;]*m//g' | tail -3)
  if echo "$out" | grep -qE "[0-9]+ (failed|error)"; then got=RED
  elif echo "$out" | grep -qE "[0-9]+ passed"; then got=GREEN
  else got=RED; fi
  printf '%-62s expected=%-5s got=%s\n' "$label" "$want" "$got"
  [ "$got" = "$want" ] || printf '    ^^ MUTANT SURVIVED / BASELINE BROKE\n%s\n' "$out"
}

run "M0 baseline (unmutated)" GREEN "$TEST"

# --------------------------------------------------------------------------
# M1  THE TRACE FALLS SILENT AND THE INGEST REPORTS ZERO INSTEAD OF REFUSING.
#     The whole indicator law in one mutation: a module absent from the dump
#     is treated as a module with nothing unexecuted.
restore
"$PY" - "$ING" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace(
    '        if entry is None:\n'
    '            rep.no_observation(rel, rank, group, "module-absent-from-dump")\n'
    '            continue\n',
    '        if entry is None:\n'
    '            rep.say(f"WEG2-COVERAGE UNEXECUTED module={rel} rank={rank} "\n'
    '                    f"lines=[] executed_pct=100.0")  # M1\n'
    '            continue\n', 1)
open(p, 'w').write(s)
EOF
run "M1 absent module reported as 0 unexecuted" RED "$TEST::NoObservationIsNeverZero"

# --------------------------------------------------------------------------
# M2  THE COUNT CHECK IS DISABLED. A line the tracer recorded that is not in
#     the source at all (drift, a wrong root, a stale dump) is silently kept,
#     and the difference of two incompatible denominators is printed as a
#     finding.
restore
"$PY" - "$ING" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace(
    '        stray = [n for n in raw if n < 1 or n > n_source_lines]\n',
    '        stray = []  # M2\n', 1)
open(p, 'w').write(s)
EOF
run "M2 tally check disabled (stray lines accepted)" RED \
    "$TEST::TheIngestPrintsUnexecutedLines::test_the_tally_holds_or_the_ingest_refuses"

# --------------------------------------------------------------------------
# M3  AN ALLOWLISTED MODULE DROPS OUT OF THE ALLOWLIST. The report then looks
#     complete and is missing a whole module of the lane -- the failure mode a
#     hand-maintained scope list has by construction.
restore
"$PY" - "$MOD" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('    "python/sglang/srt/weg2/weight_exchange_shadow.py",\n', '', 1)  # M3
open(p, 'w').write(s)
EOF
run "M3 allowlist loses weight_exchange_shadow" RED \
    "$TEST::TheAllowlistNamesFilesThatExist"

# --------------------------------------------------------------------------
# M4  THE OFF STATE STOPS BEING BYTE-IDENTICAL: arm() falls back to a
#     directory of its own instead of returning, so an unarmed acceptance boot
#     silently pays coverage.py's line tracer.
restore
"$PY" - "$MOD" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace(
    '    directory = directory or os.environ.get(DIR_ENV) or ""\n'
    '    if not directory:\n'
    '        return False\n',
    '    directory = directory or os.environ.get(DIR_ENV) or "/tmp/m4"  # M4\n', 1)
open(p, 'w').write(s)
EOF
run "M4 arm falls back to a directory instead of staying off" RED \
    "$TEST::OffIsByteIdentical"

# --------------------------------------------------------------------------
# M5  SOURCE DRIFT IS IGNORED. The dump is read against a different checkout
#     and every line number is confidently wrong.
restore
"$PY" - "$ING" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('        if dump_sha and dump_sha != tree_sha:\n',
              '        if False:  # M5\n', 1)
open(p, 'w').write(s)
EOF
run "M5 source-drift check removed" RED \
    "$TEST::TheIngestPrintsUnexecutedLines::test_source_drift_between_boot_and_ingest_refuses"

# --------------------------------------------------------------------------
# M6  THE LAUNCHER STOPS POPPING THE VARIABLE. A value inherited from the
#     operator's shell then arms the tracer on a boot that never asked, which
#     is exactly the ambient-variable property this flag exists to avoid.
restore
"$PY" - "$LAU" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('        env.pop("SGLANG_WEG2_LANE_COVERAGE_DIR", None)\n',
              '        pass  # M6\n', 1)
open(p, 'w').write(s)
EOF
run "M6 launcher no longer pops the variable when OFF" RED \
    "$TEST::TheLauncherFlagIsOffByDefault::test_build_env_pops_the_variable_when_the_flag_is_off"

# --------------------------------------------------------------------------
# M7  THE LEG HOOK IS UNWIRED. Present-but-unwired (#859) is the expensive
#     middle state: the module imports, its tests are green, and nothing on
#     the product path ever calls it.
restore
"$PY" - "$WUP" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('                wlc.note_leg_end(hook)\n', '                pass  # M7\n', 1)
open(p, 'w').write(s)
EOF
run "M7 leg hook no longer calls the instrument" RED \
    "$TEST::TheHooksAreWiredIntoTheProductPath"

restore
run "M8 baseline again (all mutations reverted)" GREEN "$TEST"
