#!/usr/bin/env bash
# PF mutation check for the group told=0 fallback (weg2_told_fallback +
# its seams in weg2_store_told / pp_object_recv). Every mutation is a
# plausible WRONG implementation in a named danger direction, and each one
# MUST make test_weg2_told_group_fallback_pf.py fail. A survivor means that
# direction is asserted nowhere.
#
#   M1  PP0 admits the full told at the Frist (the group death returns)
#   M2  a follower keeps its read on Admit(0) (rows / reader refs leak, stage waits)
#   M3  PP0 keeps its own store record on Admit(0) (PP0 refuses 0 vs told)
#   M4  PP0 admits on ANY ack instead of ALL (the slow follower is outvoted)
#   M5  a short-read ack is not a mismatch (waits the Frist, or worse)
#   M6  a follower ignores the wire's fallback marker (decides from its read)
#   M7  the follower's twin mark survives the fallback (0 + head != 0)
#   M8  the switch defaults ON (off is no longer byte-identical)
#   M9  the harvest JOINS the standing receive (PP0's pass blocks)
#   M10 a follower acks before its read terminated
#   M11 a parked rid's paced entry is dropped under the fallback
#   M12 PP0 never harvests (every paced rid falls to told=0)
#   M13 the ack's own prefix forgets the twin head (absolute told -> 0)
#   M14 the follower acks a read-ahead that did not ask for it (own env)
#
# Hermetic: no CUDA, no NVML. Restores every file unconditionally on exit.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
FB=$ROOT/python/sglang/srt/managers/weg2_told_fallback.py
ST=$ROOT/python/sglang/srt/managers/weg2_store_told.py
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
TESTS=$ROOT/test/registered/unit/managers/test_weg2_told_group_fallback_pf.py

BAK_DIR=$(mktemp -d)
cp "$FB" "$BAK_DIR/fb.py"
cp "$ST" "$BAK_DIR/st.py"
restore() {
  cp "$BAK_DIR/fb.py" "$FB"
  cp "$BAK_DIR/st.py" "$ST"
}
trap restore EXIT

run_suite() {
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES= PYTHONPATH=python \
    timeout 600 "$PY" -m pytest "$TESTS" -q -x -p no:cacheprovider >/dev/null 2>&1)
}

# mutate FILE OLD NEW -- exact, must match once
mutate() {
  "$PY" - "$1" "$2" "$3" <<'EOF'
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
  local name=$1
  if run_suite; then
    echo "SURVIVED $name"
    survivors=$((survivors + 1))
  else
    echo "killed   $name"
  fi
  restore
}

run_suite || { echo "BASELINE FAILS -- fix the suite first"; exit 2; }
echo "baseline green"

mutate "$FB" '    if now >= o.deadline:
        return 0, REASON_FRIST' '    if now >= o.deadline:
        return o.told, REASON_FRIST' && check M1
mutate "$FB" '    follower_forget(scheduler, rid)
    release_own_read(scheduler, rid)' '    follower_forget(scheduler, rid)' && check M2
mutate "$ST" '                _fb.release_own_read(scheduler, rid)
' '' && check M3
mutate "$FB" 'if all(r in o.acks for r in followers):' 'if any(r in o.acks for r in followers):' && check M4
mutate "$FB" '    if any(own != o.told for own in o.acks.values()):
        return 0, REASON_MISMATCH' '' && check M5
mutate "$ST" '    fallback = bool(getattr(item, _fb.WIRE_FALLBACK, 0))' '    fallback = False' && check M6
mutate "$FB" '    _twin.take_follower_twin(scheduler, rid)
    digests' '    digests' && check M7
mutate "$FB" 'os.environ.get(ENV_FALLBACK, "0")' 'os.environ.get(ENV_FALLBACK, "1")' && check M8
mutate "$FB" '                    if not frame.poll():' '                    if not frame.advance(0.5):' && check M9
mutate "$FB" '        if not tree.check_prefetch_progress(rid):
            continue
        req = st.registered.pop(rid)' '        req = st.registered.pop(rid)' && check M10
mutate "$ST" '            if fb_on and rid in fb_parked:' '            if False:' && check M11
mutate "$ST" '        _fb.pp0_harvest(scheduler)
' '' && check M12
mutate "$FB" '        own += _twin.registered_head(req)' '        pass' && check M13
mutate "$ST" '            if getattr(item, _fb.WIRE_ACK, 0):' '            if _fb.env_on():' && check M14

echo "survivors: $survivors"
exit $survivors
