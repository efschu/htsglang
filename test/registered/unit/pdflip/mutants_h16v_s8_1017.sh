#!/usr/bin/env bash
# H16V S8 mutation check: one plausible WRONG call edge per review condition; each MUST turn the S8 tests
# (test_pdflip_h16v_s8_1017.py + the updated s5/xgesetz/front_x_carry files) red. ONLY=<name> runs one mutant.
# CPU only, through /spinning/gpu-arb/pytest_gedeckelt.sh. Restores the sources on exit.
#
#   A2   SOLL R1/R3/R2a  X(k) (further members' P fixed costs) is the threshold of the decision again
#   A3a  SOLL R2/R7/R11  the old fixed substitute bound (0.5 x static X = 6144) is back for a resumed tail
#   A3b  SOLL R11        D reads halbX from a rank-local fraction instead of the carried one
#   A3c  SOLL R11        the hand-over after P carries nothing again
#   A4a  SOLL R5         the to_d grant is formed on the seat count read before the awaits
#   A4b  SOLL R5         a younger decode already planned for one grant is planned again for the next waiter
#   A5   SOLL R1         the deciding S sums the deduplicated (P-side) figures again
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
W=$ROOT/python/flliper/srt/pdflip
TESTS="test/registered/unit/pdflip/test_pdflip_h16v_s8_1017.py test/registered/unit/pdflip/test_pdflip_h16v_s5_1017.py test/registered/unit/pdflip/test_pdflip_xgesetz_fd_1016.py"
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}
declare -A FILE OLD NEW BAKF
for f in front phase_policy d_norecompute; do BAKF[$f]=$(mktemp); cp "$W/$f.py" "${BAKF[$f]}"; done
restore() { for f in front phase_policy d_norecompute; do cp "${BAKF[$f]}" "$W/$f.py"; done; }
trap restore EXIT
run_suite() { (cd "$ROOT" && CUDA_VISIBLE_DEVICES="" "$GED" $TESTS -p no:warnings >/dev/null 2>&1); }
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
FILE[A2]=front; OLD[A2]='        return min(int(x1) for _r, _d, x1 in rids_depths)
'
NEW[A2]='        _k = Front._set_xk_for(self, rids_depths)
        return _k if _k > 0 else min(int(x1) for _r, _d, x1 in rids_depths)
'
FILE[A3a]=d_norecompute; OLD[A3a]='            return False   # nothing carried by the front: RESUME-VIA-P, no invented bound'
NEW[A3a]='            x, halfx = static_x or 12288, (static_x or 12288) // 2'
FILE[A3b]=d_norecompute; OLD[A3b]='        if not rem < int(halfx):'
NEW[A3b]='        if not rem < 0.99 * int(x):'
FILE[A3c]=front; OLD[A3c]='        if _after_p:'
NEW[A3c]='        if False:'
FILE[A4a]=front; OLD[A4a]='            taken, n = self._arrival_seat_taken()
            free = bool(_asr.seat_free(taken, n))
            plan = self._arrival_seat_age_verdict('
NEW[A4a]='            plan = self._arrival_seat_age_verdict('
FILE[A4b]=front; OLD[A4b]='if r not in st["parked"] and r != head_rid and r not in _claimed]'
NEW[A4b]='if r not in st["parked"] and r != head_rid]'
FILE[A5]=phase_policy; OLD[A5]='    s = sum(max(0, int(m.own_uncached or m.uncached)) for m in members) + running'
NEW[A5]='    s = sum(max(0, int(m.uncached)) for m in members) + running'

run_suite || { echo "BASELINE RED -- fix the tests first"; exit 2; }
survivors=0
for m in A2 A3a A3b A3c A4a A4b A5; do
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$m" ] && continue
  restore
  mutate "${FILE[$m]}" "${OLD[$m]}" "${NEW[$m]}" || { echo "$m: ANCHOR LOST"; survivors=$((survivors+1)); continue; }
  if run_suite; then echo "$m: SURVIVED"; survivors=$((survivors+1)); else echo "$m: killed"; fi
done
restore
echo "survivors=$survivors"
exit $((survivors > 0))
