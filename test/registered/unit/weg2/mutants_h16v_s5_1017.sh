#!/usr/bin/env bash
# H16V S5 mutation check: one plausible WRONG call edge per fix; each MUST turn test_weg2_h16v_s5_1017.py red.
# ONLY=<name> runs one mutant. CPU only, through /spinning/gpu-arb/pytest_gedeckelt.sh. Restores the sources on exit.
#
#   F1a  SOLL R1   S counts the members' raw tokens (no shared-prefix discount at the set)
#   F1c  SOLL R1   the discount ignores where the credit ends (overlap taken from 0)
# (F1b, F2a, F2b: superseded by the S8 review conditions A2/A5 -- see mutants_h16v_s8_1017.sh)
#   F3a  SOLL R7   a resumed tail is never D's own extend (law as before: always RESUME-VIA-P)
#   F3b  SOLL R2   the bound is "any tail" (no halbX limit)
#   F4a  SOLL R5   a KV/seat wait that is not "no seat at entry" is named R2 again
#   F4b  SOLL R5   the seat count is not read again after the KV await (3 waiters take 1 seat)
#   F4c  SOLL R5   the semaphore wait after the grant stays unnamed
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
W=$ROOT/python/sglang/srt/weg2
TESTS=test/registered/unit/weg2/test_weg2_h16v_s5_1017.py
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}
declare -A FILE OLD NEW BAKF
for f in front phase_policy x_curves d_norecompute; do BAKF[$f]=$(mktemp); cp "$W/$f.py" "${BAKF[$f]}"; done
restore() { for f in front phase_policy x_curves d_norecompute; do cp "${BAKF[$f]}" "$W/$f.py"; done; }
trap restore EXIT
run_suite() { (cd "$ROOT" && CUDA_VISIBLE_DEVICES="" "$GED" "$TESTS" -x -p no:warnings >/dev/null 2>&1); }
mutate() {
  python3 - "$W/$1.py" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n = s.count(old)
if n != 1:
    sys.exit(f"mutation anchor found {n}x: {old!r}")
open(path, "w").write(s.replace(old, new))
PY
}
FILE[F1a]=front;  OLD[F1a]='eff = Front._set_discount(self, [(r, u, t, q) for r, u, t, _p, q in raw])'
NEW[F1a]='eff = {r: u for r, u, t, _p, q in raw}'
FILE[F1c]=phase_policy; OLD[F1c]='best = max(best, c - max(int(adepth), int(depth)))'
NEW[F1c]='best = max(best, c)'
FILE[F3a]=d_norecompute; OLD[F3a]='            return _resumed_tail_extends(sched, req, int(rem), _fx)'
NEW[F3a]='            return False'
FILE[F3b]=d_norecompute; OLD[F3b]='        if not rem < int(halfx):'
NEW[F3b]='        if False:'
FILE[F4a]=front;  OLD[F4a]='why="seat_wait" if (_w > 0.5 and (_seats0 <= 0 or _named)) else "R2")'
NEW[F4a]='why="seat_wait" if (_w > 0.5 and _seats0 <= 0) else "R2")'
FILE[F4b]=front;  OLD[F4b]='            if not _asr.seat_free(taken, n):
                fits, free = False, False'
NEW[F4b]='            if False:
                fits, free = False, False'
FILE[F4c]=front;  OLD[F4c]='        if time.time() - _t_sem > 0.5:'
NEW[F4c]='        if False:'

run_suite || { echo "BASELINE RED -- fix the tests first"; exit 2; }
survivors=0
for m in F1a F1c F3a F3b F4a F4b F4c; do
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$m" ] && continue
  restore
  mutate "${FILE[$m]}" "${OLD[$m]}" "${NEW[$m]}" || { echo "$m: ANCHOR LOST"; survivors=$((survivors+1)); continue; }
  if run_suite; then echo "$m: SURVIVED"; survivors=$((survivors+1)); else echo "$m: killed"; fi
done
restore
echo "survivors=$survivors"
exit $((survivors > 0))
