#!/usr/bin/env bash
# S50 TOLD-ACK-EARLY mutation check: one plausible WRONG edit per danger direction; each MUST turn
# test_s50_told_ack_early_1018.py red. Local, 4-GiB cgroup. ONLY=<name> runs one. Restores every file on exit.
#   B1 "Ack vor Abschluss": the pump reports a registered read without the harvest's termination check
#   B1b "Ack vor Abschluss", early-path form: the worker acks every expected rid at told when the read thread wakes it
#   B2 the worker pumps outside the forward window (no window lock / no in_window check)
#   B3 the scheduler leaves the window without waiting for a running pump
#   B4 a wake that came while the scheduler held the window is lost (no re-wake at window entry)
#   B5 the read thread never wakes the worker (hook dropped in prefetch_io_aux_func)
#   B6 armed on PP0 too
#   B7 the window wraps more than the model forward (the produce step joins it)
#   B10 (review) a forward that raises leaves the window released   B11 (review) a pump that raises keeps the window
#   B12 (review) the outbox is cleared before the send (a dead send loses the ack)
#   B8 armed at tp_size > 1 (a harvest collective from the worker thread)   B9 a nested forward releases the window twice
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/flliper/srt
PY=${PY:-/spinning/shvllm/.venv/bin/python}
TEST=test/registered/unit/pdflip/test_s50_told_ack_early_1018.py
E=$SRT/managers/pdflip_told_ack_early.py
FILES=("$E" "$SRT/managers/pdflip_told_fallback.py" "$SRT/managers/cache_controller.py" "$SRT/managers/scheduler.py")
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
F[B1]=$SRT/managers/pdflip_told_fallback.py; OLD[B1]='        if not tree.check_prefetch_progress(rid):
            continue
        if getattr(st.registered[rid], "_pdflip_early_told", None) is not None:'; NEW[B1]='        if getattr(st.registered[rid], "_pdflip_early_told", None) is not None:'
F[B1b]=$E; OLD[B1b]='        (self._pump or _fb.follower_pump)(sched)'; NEW[B1b]='        for _rid, _told in list(st.expect.items()):
            st.outbox.append((_rid, _told)); st.registered.pop(_rid, None); st.expect.pop(_rid, None)
        if st.outbox and _fb._channel(sched).send_nowait(_fb.PdFlipToldReadAck(rank=1, seq=st.seq, reads=list(st.outbox))):
            st.outbox = []; st.seq += 1'
F[B2]=$E; OLD[B2]='        if not self.window.acquire(blocking=False):
            return False   # the scheduler holds the window (outside a forward): the next window / absorb reports
        try:
            if not self.in_window:
                return False
'; NEW[B2]='        _got = self.window.acquire(blocking=False)
        try:
'
F[B3]=$E; OLD[B3]='        finally:
            self.window.acquire()
            self.in_window = False'; NEW[B3]='        finally:
            self.window.acquire(blocking=False)
            self.in_window = False'
F[B4]=$E; OLD[B4]='            if self.pending:
                self._wake.set()
            yield'; NEW[B4]='            yield'
F[B5]=$SRT/managers/cache_controller.py; OLD[B5]='                _told_ack_early.note_read_end(operation)
'; NEW[B5]=''
F[B6]=$E; OLD[B6]=' or int(getattr(ps, "pp_rank", 0) or 0) == 0:'; NEW[B6]=':'
F[B7]=$SRT/managers/scheduler.py; OLD[B7]='                    with _pdflip_told_ack_early.forward_window(self):
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **kwargs
                        )
                    if produce:
                        self._draft_kv_produce(batch, batch_result)'; NEW[B7]='                    with _pdflip_told_ack_early.forward_window(self):
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **kwargs
                        )
                        if produce:
                            self._draft_kv_produce(batch, batch_result)'
F[B8]=$E; OLD[B8]='        if int(getattr(ps, "tp_size", 1) or 1) != 1:
            return False'; NEW[B8]=''
F[B9]=$E; OLD[B9]='        if self.in_window and threading.current_thread() is self._owner:
            yield
            return
'; NEW[B9]=''
# S50 review: the error paths
F[B10]=$E; OLD[B10]='        finally:
            self.window.acquire()
            self.in_window = False'; NEW[B10]='        except BaseException:
            raise
        self.window.acquire()
        self.in_window = False'
F[B11]=$E; OLD[B11]='        finally:
            self.window.release()

    def _pump_and_note'; NEW[B11]='        finally:
            if not self.errors:
                self.window.release()

    def _pump_and_note'
F[B12]=$SRT/managers/pdflip_told_fallback.py; OLD[B12]='    if ch.send_nowait(ack):
        st.outbox = []'; NEW[B12]='    st.outbox = []
    if ch.send_nowait(ack):'
ORDER=(B1 B1b B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 B12)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}" || { echo "$m ANCHOR-FAIL"; survived=$((survived+1)); continue; }
  if [ "$m" = B2 ]; then  # the finally must only release a lock it holds
    python3 - "$E" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace("        finally:\n            self.window.release()\n\n    def _pump_and_note","        finally:\n            if _got:\n                self.window.release()\n\n    def _pump_and_note",1)
open(p,"w").write(s)
PY
  fi
  if (cd "$ROOT" && timeout 300 systemd-run --scope -q -p MemoryMax=4G -E PYTHONPATH=python:test/registered/unit/pdflip "$PY" -m pytest -q -x -p no:cacheprovider "$TEST" >/dev/null 2>&1); then
    echo "$m SURVIVED"; survived=$((survived+1))
  else
    echo "$m killed"; killed=$((killed+1))
  fi
done
echo "killed=$killed survived=$survived"
[ "$survived" -eq 0 ]
