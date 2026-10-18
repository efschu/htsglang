#!/usr/bin/env bash
# S42 HEAD-VOTE HOLD mutation check: one plausible WRONG edit per danger direction; each MUST turn
# test_pdflip_s42_head_vote_hold_1018.py red. A survivor = that direction is asserted nowhere. Runs LOCALLY in a 4 GiB
# cgroup (systemd-run --scope -p MemoryMax=4G). ONLY=<name> runs one. Restores every file on exit.
#   Direction STARVATION ("verpasste Zustandsaenderung -> Request verhungert"):
#     S1  the binding generation (flip/cutover) leaves the fingerprint
#     S2  a held pass runs the completion arm on an empty head (drops the store-read stamps, the wait restarts)
#     S3  the budget site holds whenever a refused head exists (no DQH change test, no 16-pass bound)
#     S4  the budget site fingerprints without this iteration's landed prefetch verdicts
#   Direction UNMEASURED VOTE READ (a pass admits on a vote its iteration never measured):
#     V1  the elif chain asks DQH again instead of consuming the budget site's decision
#     V2  an early return of the budget site leaves the arm standing (disarm is a no-op)
#   Direction RANK DISAGREEMENT (R11, raenge-nie-uneins):
#     R1  the hold decision is rank-local (TP0 holds, the workers measure and evaluate)
#     R2  a held rank re-sends its previous vote instead of abstaining (a stale value enters the group MIN)
#     S5  without a refused head the fingerprint drops the waiting set (an arrival is no change)
#     S6  (review S42r A1) the 16-pass bound is per rank only -- one divergent pass desyncs the group for ever
#   Efficacy / switch:
#     E1  the hold is off (the budget site always measures)
#     E2  the fingerprint requires the refused head again (rc11 14:13: no NO_TOKEN, a stale running head)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/flliper/srt
TEST=registered/unit/pdflip/test_pdflip_s42_head_vote_hold_1018.py
PY=${PY:-/spinning/shvllm/.venv/bin/python}
FILES=("$SRT/pdflip/head_vote_hold.py" "$SRT/managers/scheduler.py" "$SRT/pdflip/d_head_hold.py")
BAKD=$(mktemp -d); for f in "${FILES[@]}"; do cp "$f" "$BAKD/$(echo "$f" | tr / _)"; done
restore() { for f in "${FILES[@]}"; do cp "$BAKD/$(echo "$f" | tr / _)" "$f"; done; }
trap restore EXIT
mutate() { python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read(); n = s.count(old)
if n != 1: sys.exit(f"mutation anchor found {n}x: {old!r}")
open(path, "w").write(s.replace(old, new))
PY
}
declare -A F OLD NEW
H=$SRT/pdflip/head_vote_hold.py
S=$SRT/managers/scheduler.py
F[S1]=$S; OLD[S1]='            generation=gen,'; NEW[S1]='            generation=None,'
F[S2]=$S; OLD[S2]='            {} if _s42_held else self._pdflip_local_store_read_pending_ages(_head_canonical),'; NEW[S2]='            self._pdflip_local_store_read_pending_ages(_head_canonical),'
F[S3]=$H; OLD[S3]='    held = bool(dqh.should_hold(fp))'; NEW[S3]='    held = fp is not None and dqh.state.fingerprint is not None'
F[S4]=$H; OLD[S4]='    fp = sched._pdflip_d_head_fingerprint(running_batch, prefetch_verdicts)'; NEW[S4]='    fp = sched._pdflip_d_head_fingerprint(running_batch, None)'
F[V1]=$S; OLD[V1]='        pre = _pdflip_head_vote_hold.take(self)
        if pre is not None:
            return pre'; NEW[V1]='        pre = _pdflip_head_vote_hold.take(self)
        if False:
            return pre'
F[V2]=$H; OLD[V2]='    must not reach another caller of the budget site."""
    sched.__dict__.pop(ARMED_ATTR, None)'; NEW[V2]='    must not reach another caller of the budget site."""
    pass'
F[R1]=$H; OLD[R1]='    held = bool(dqh.should_hold(fp))'; NEW[R1]='    held = bool(dqh.should_hold(fp)) and int(getattr(sched.ps, "tp_rank", 0)) == 0'
F[R2]=$S; OLD[R2]='        if _s42_held:
            _head_canonical, _head_local_matches = [], {}
        else:
            _head_canonical, _head_local_matches = self._local_head_prefix_matches()'; NEW[R2]='        if _s42_held:
            _head_canonical, _head_local_matches = self.__dict__.get("_s42_prev", ([], {}))
        else:
            _head_canonical, _head_local_matches = self._local_head_prefix_matches()
            self._s42_prev = (_head_canonical, _head_local_matches)'
F[E1]=$H; OLD[E1]='    if not sched.__dict__.pop(ARMED_ATTR, False):
        return False'; NEW[E1]='    if True:
        return False'
Q=$SRT/pdflip/d_head_hold.py
F[S5]=$Q; OLD[S5]='    return (head, waiting_rids, running_rids, landed, generation)'; NEW[S5]='    return (head, waiting_rids if head else (), running_rids, landed, generation)'
F[E2]=$Q; OLD[E2]='    head = str(head) if head is not None and str(head) in waiting_rids else None'; NEW[E2]='    if head is None or str(head) not in waiting_rids:
        return None
    head = str(head)'
F[S6]=$Q; OLD[S6]='            or (tick > 0 and tick % (REEVAL_EVERY_PASSES + 1) == 0)'; NEW[S6]='            or False'
ORDER=(S1 S2 S3 S4 S5 S6 V1 V2 R1 R2 E1 E2)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  if ! mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}"; then echo "MUTANT $m: ANCHOR MISSING"; survived=$((survived+1)); continue; fi
  out=$(cd "$ROOT/test" && systemd-run --scope -p MemoryMax=4G --quiet env PYTHONPATH="$ROOT/python" CUDA_VISIBLE_DEVICES= timeout 600 "$PY" -m pytest -q -p no:cacheprovider "-opythonpath=registered/unit/pdflip" "-opythonpath=registered/unit/mem_cache" "$TEST" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -aE "[0-9]+ (passed|failed)" | tail -1)
  case "$out" in *failed*) r=red; killed=$((killed+1));; *) r=green; survived=$((survived+1));; esac
  echo "MUTANT $m: $r ($out)"
done
echo "S42 MUTANTS: killed=$killed survived=$survived of ${#ORDER[@]}"
[ "$survived" -eq 0 ]
