#!/usr/bin/env bash
# SET SCHEDULER 1017 mutation check (SCHEDULER-SOLL v2). One plausible WRONG call edge per rule; each MUST turn
# test_weg2_set_scheduler_1017.py red. A survivor = that rule is asserted nowhere at the front's call edge.
# ONLY=<name> runs one mutant. Runs through /spinning/gpu-arb/pytest_gedeckelt.sh (CPU only). Restores front.py on exit.
#
#   M2   R2  a kurz arrival no longer short-cuts: it joins the set and waits         (classify == kurz -> and False)
#   M2y  R2  a kurz arrival yields to a flip that is merely collecting (D-DIRECT-YIELD)
#   M3   R3  a set whose T_halb ran out with S < X flips instead of going to D
#   M4   R4  a big request flips at once although D decodes (no T_halb for company)
#   M5   R5  the front tells the verdict D has seats it has not (a flip without seats)
#   M6   R6  a running D prefill is not counted into S (no reroute)
#   M7   R7  the epoch's state outlives the epoch (the epoch rule is back)
#   M7b  R7  P takes a D-prefill set under X at the start of its drain
#   M8   R3  the timer restarts with every new member (t_open = the newest member)
#   M10  R3  a decided flip is re-collected on the next tick (a second T_halb)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
FR=$ROOT/python/sglang/srt/weg2/front.py
TESTS=test/registered/unit/weg2/test_weg2_set_scheduler_1017.py
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}
BAK=$(mktemp)
cp "$FR" "$BAK"
trap 'cp "$BAK" "$FR"' EXIT

run_suite() { (cd "$ROOT" && CUDA_VISIBLE_DEVICES="" "$GED" "$TESTS" -x -p no:warnings >/dev/null 2>&1); }

mutate() {
  python3 - "$FR" "$1" "$2" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n = s.count(old)
if n != 1:
    sys.exit(f"mutation anchor found {n}x: {old!r}")
open(path, "w").write(s.replace(old, new))
PY
}

declare -A OLD NEW
OLD[M2]='_frac, False) == phase_policy.CLASS_KURZ:'
NEW[M2]='_frac, False) == phase_policy.CLASS_KURZ and False:'
OLD[M2y]='if cands and not Front._set_flip_foreseeable(self, time.time()):'
NEW[M2y]='if False:'
OLD[M3]='route = "D" if sv.verdict == phase_policy.V_D_AFTER_TIMER else "P"'
NEW[M3]='route = "P"'
OLD[M4]='d_decoding=decoding, seats_free=seats,'
NEW[M4]='d_decoding=False, seats_free=seats,'
OLD[M5]='d_decoding=decoding, seats_free=seats,'
NEW[M5]='d_decoding=decoding, seats_free=999,'
OLD[M6]='d_prefill_running_uncached=self._dc_carried(),  # R6'
NEW[M6]='d_prefill_running_uncached=0,  # R6'
OLD[M7]='if st.get("epoch") != self.epoch:'
NEW[M7]='if st.get("epoch") is None:'
OLD[M7b]='if cand_ready and s_set < x_tok:'
NEW[M7b]='if False:'
OLD[M8]='        if st["t_open"] is None:
            st["t_open"] = min(m.t_arrive for m in members)
        decoding = self._dc_decoding()'
NEW[M8]='        st["t_open"] = max(m.t_arrive for m in members)
        decoding = self._dc_decoding()'
OLD[M10]='if any(p.rid in st["go_p"] for p in live_q):
            for rid in list(st["shorts"]):'
NEW[M10]='if False:
            for rid in list(st["shorts"]):'

run_suite || { echo "BASELINE RED -- fix the tests first"; exit 2; }
survivors=0
for m in M2 M2y M3 M4 M5 M6 M7 M7b M8 M10; do
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$m" ] && continue
  cp "$BAK" "$FR"
  mutate "${OLD[$m]}" "${NEW[$m]}" || { echo "$m: ANCHOR LOST"; survivors=$((survivors+1)); continue; }
  if run_suite; then echo "$m: SURVIVED"; survivors=$((survivors+1)); else echo "$m: killed"; fi
done
cp "$BAK" "$FR"
echo "survivors=$survivors"
exit $((survivors > 0))
