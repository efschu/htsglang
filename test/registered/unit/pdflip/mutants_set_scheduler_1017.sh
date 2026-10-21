#!/usr/bin/env bash
# SET SCHEDULER 1017 mutation check (SCHEDULER-SOLL v2). One plausible WRONG call edge per rule; each MUST turn
# test_pdflip_set_scheduler_1017.py red. A survivor = that rule is asserted nowhere at the front's call edge.
# ONLY=<name> runs one mutant. Runs through /spinning/gpu-arb/pytest_gedeckelt.sh (CPU only). Restores front.py on exit.
#
#   M2   R2  a kurz arrival no longer short-cuts: it joins the set and waits         (classify == kurz -> and False)
#   M2y  R2  a kurz arrival yields to a flip that is merely collecting (D-DIRECT-YIELD)
#   M3   R3  a set whose T_halb ran out with S < X flips instead of going to D
#   M4   R4  a big request flips at once although D decodes (no T_halb for company)
#   M5   R5  the front tells the verdict D has seats it has not (a flip without seats)
#   M6   R6  a running D prefill is not counted into S (no reroute)
#   M6s  R6  a granted D prefill D has not begun counts as running (S49, user 18.10. "Nur echt laufend zählt")
#   M6x  R6/X-SUM a set goes to D on top of an unstarted grant whose sum passes X (S49 hold missing)
#   M7   R7  the epoch's state outlives the epoch (the epoch rule is back)
#   M7b  R1a P-BATCH-ALL does not take the short in D's line (P-BATCH-STAYS back)
#   M8   R3  the timer restarts with every new member (t_open = the newest member)
#   MA2  R2  the price barrier decides again (PRICE_BARRIER_DECIDES True)
#   MA3  R2a S is checked against the envelope X, not the members' smallest
#   MA5  R4  t_open before this D phase woke (no clamp)
#   MA8  R7  the work-exhausted flip path flips for a queue under X
#   MB3  R6  a REROUTE_P leaves the waiting D-line entries behind
#   MB6  R7  a budget-refused short never goes back to D
#   MN1  R9  P-by-state entries (hand-off lost / reroute) no longer need P (starve under FLIP-HELD)
#   M10  R3  a decided flip is re-collected on the next tick (a second T_halb)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
FR=$ROOT/python/flliper/srt/pdflip/front.py
PP=$ROOT/python/flliper/srt/pdflip/phase_policy.py
BAKP=$(mktemp); cp "$PP" "$BAKP"
TESTS="test/registered/unit/pdflip/test_pdflip_set_scheduler_1017.py test/registered/unit/pdflip/test_pdflip_s49_r6_nur_echt_laufend_1018.py"
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}
BAK=$(mktemp)
cp "$FR" "$BAK"
trap 'cp "$BAK" "$FR"; cp "$BAKP" "$PP"' EXIT

run_suite() { (cd "$ROOT" && CUDA_VISIBLE_DEVICES="" "$GED" $TESTS -x -p no:warnings >/dev/null 2>&1); }

mutate() {
  python3 - "${TARGET:-$FR}" "$1" "$2" <<'PY'
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
OLD[M4]='d_decoding=decoding, seats_free=seats,
            d_prefill_running_uncached=running,  # R6'
NEW[M4]='d_decoding=False, seats_free=seats,
            d_prefill_running_uncached=running,  # R6'
OLD[M5]='d_decoding=decoding, seats_free=seats,
            d_prefill_running_uncached=running,  # R6'
NEW[M5]='d_decoding=decoding, seats_free=999,
            d_prefill_running_uncached=running,  # R6'
OLD[M6]='d_prefill_running_uncached=running,  # R6'
NEW[M6]='d_prefill_running_uncached=0,  # R6'
OLD[M6s]='running = Front._d_pf_pending_tokens(self, started_only=True)'
NEW[M6s]='running = Front._d_pf_pending_tokens(self)'
OLD[M6x]='sv = Front._set_xsum_hold(self, sv, x_tok, running, decoding=decoding)'
NEW[M6x]='pass'
OLD[M7]='if st.get("epoch") != self.epoch:'
NEW[M7]='if st.get("epoch") is None:'
OLD[M7b]='(moved if _takable(p) else keep).append(p)'
NEW[M7b]='(moved if (False and _takable(p)) else keep).append(p)'
OLD[M8]='        if st["t_open"] is None:
            st["t_open"] = max(min(m.t_arrive for m in members)'
NEW[M8]='        if True:
            st["t_open"] = max(max(m.t_arrive for m in members)'
OLD[M10]='if any(p.rid in st["go_p"] for p in live_q):
            for rid in list(st["shorts"]):'
NEW[M10]='if False:
            for rid in list(st["shorts"]):'

OLD[MA2]='PRICE_BARRIER_DECIDES = False'
NEW[MA2]='PRICE_BARRIER_DECIDES = True'
OLD[MA3]='x_tok = Front._set_x_for(self, [(m.rid, st["depths"].get(m.rid, 0), st["xs"][m.rid]) for m in members])'
NEW[MA3]='x_tok = int(self.tp_prefill_max_tokens)'
OLD[MA5]='st["t_open"] = max(min(m.t_arrive for m in members), float(getattr(self, "t_awake", 0.0) or 0.0))'
NEW[MA5]='st["t_open"] = min(m.t_arrive for m in members)'
OLD[MA8]='        if _q_s < x_tok:
            if st.get("held_epoch")'
NEW[MA8]='        if False:
            if st.get("held_epoch")'
OLD[MB3]='            _n = self._p_batch_takes_d_prefill()
            if _n:'
NEW[MB3]='            _n = 0
            if _n:'
OLD[MB6]='                        p.d_eligible = True
                        p.d_budget_refused = False'
NEW[MB6]='                        p.d_budget_refused = False'

OLD[MN1]='    if int(handoff_lost_reroutes or 0) > 0 or int(reroutes or 0) > 0:'
NEW[MN1]='    if False:'

run_suite || { echo "BASELINE RED -- fix the tests first"; exit 2; }
survivors=0
for m in M2 M2y M3 M4 M5 M6 M6s M6x M7 M7b M8 M10 MA2 MA3 MA5 MA8 MB3 MB6 MN1; do
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$m" ] && continue
  cp "$BAK" "$FR"; cp "$BAKP" "$PP"; TARGET=$FR; [ "$m" = MN1 ] && TARGET=$PP
  mutate "${OLD[$m]}" "${NEW[$m]}" || { echo "$m: ANCHOR LOST"; survivors=$((survivors+1)); continue; }
  if run_suite; then echo "$m: SURVIVED"; survivors=$((survivors+1)); else echo "$m: killed"; fi
done
cp "$BAK" "$FR"
echo "survivors=$survivors"
exit $((survivors > 0))
