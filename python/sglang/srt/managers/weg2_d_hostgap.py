"""Group D decode round: the deferred host-length read, and the instrument that measures it.

WHAT THE METAL SHOWED (weg2xsn428/429, 27B D = TP3 uneven DCP, DFLASH block 8, bs 1)
-----------------------------------------------------------------------------------
Round period 36.0 ms (the #1241 ``t`` stamps); the two replayed forwards (draft +
verify, ``gpu-ms``) 32.4-32.6 ms of it.  The DFLASH phase events
(``SGLANG_DFLASH_PHASE_TIMING``) tile the period with no gap between rounds
(``t0->kv_commit`` 36.05 ms), so the missing ~3.5 ms sit INSIDE the round, all
on one stretch: from the publish of round N (``on_publish`` after the accept) to
the draft replay of round N+1 -- ``prep->embed`` 1.23-1.27 ms plus ~1.4 ms of
``embed->draft_fwd`` before the draft bracket, on all three ranks alike.  Nothing
there is heavy GPU work (block prep, DCP index prebuild, the embedding and its
all_reduce, the draft's row rebuild): the card waits for the host.

WHY THE HOST IS LATE THERE: ``FutureMap.resolve_seq_lens_cpu``
(``overlap_utils.py``, called by ``Scheduler.run_batch`` BEFORE the forward is
launched) blocks on ``fwd_prepare_d2h_stream.synchronize()`` until round N has
published its lengths -- the host mirror ``seq_lens_cpu`` the draft plan needs.
Everything the scheduler and the worker do between that wake and the draft
replay therefore runs while the card is idle, although most of it needs no host
length at all: the input gathers, the block prep, the DCP prebuild (sized by the
reservation bound when no exact mirror is there), the embedding and its
all_reduce.

THE DEFERRED READ (``SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU=1``; unset = today's path)
-------------------------------------------------------------------------------
``resolve_seq_lens_cpu(batch, defer=True)`` does every DEVICE part exactly as
before (the publish wait on the schedule stream, the ``seq_lens`` gather, the
pinned D2H on the private stream) but records an event instead of synchronizing,
leaves ``seq_lens_cpu`` / ``seq_lens_sum`` at ``None`` and returns a
:class:`PendingSeqLensCpu`.  The DFLASH worker completes it right before the
FIRST host read of the exact lengths (the draft prep after the embedding); the
scheduler completes it again after the forward (idempotent), so the batch leaves
``run_batch`` with the same mirror as before.  Only the POSITION of the host wait
moves; the values, the stream order and the collective order are unchanged.
Engaged only for a spec-v2 DECODE batch of a worker that declares
``supports_deferred_seq_lens_cpu`` and carries the reservation bound
(``nxt_kv_lens_cpu``); everything else takes the old read.

THE INSTRUMENT (``SGLANG_WEG2_D_HOSTGAP=N``, N rounds per line, ``1`` = 512)
--------------------------------------------------------------------------
One ``#DGAP`` line per rank every N decode rounds, split like the NF round
instrument so the two lines compare: ``result_wait`` (host blocked on the
device: the publish wait of the length read + the result ``copy_done`` wait),
``hicache`` (``check_hicache_events`` wall), ``host_other`` (the rest of the
period), and ``crit`` -- the host time from the publish wake to the draft replay
launch, i.e. the stretch the card waits for.  Host clocks only
(``perf_counter``); it adds no device read of its own.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Optional

logger = logging.getLogger(__name__)

D_DEFER_SEQ_LENS_CPU_ENV = "SGLANG_WEG2_D_DEFER_SEQ_LENS_CPU"
D_HOSTGAP_ENV = "SGLANG_WEG2_D_HOSTGAP"
D_HOSTGAP_DEFAULT_ROUNDS = 512


def defer_seq_lens_cpu_on() -> bool:
    """The deferred length read.  Read per call (an env lookup), so a test can
    flip it; the scheduler caches it once at its first decode round."""
    return os.environ.get(D_DEFER_SEQ_LENS_CPU_ENV, "") == "1"


def hostgap_rounds() -> int:
    """Rounds per ``#DGAP`` line; 0 = instrument off.  A malformed value raises
    here; :func:`meter` turns that into a loud WARNING and leaves the instrument
    off -- an instrument that silently stays off is the one failure a measuring
    boot cannot see, and one that kills the boot is worse."""
    raw = os.environ.get(D_HOSTGAP_ENV, "").strip()
    if raw in ("", "0"):
        return 0
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(
            f"{D_HOSTGAP_ENV}={raw!r}: expected an integer round count "
            f"(1 = {D_HOSTGAP_DEFAULT_ROUNDS})"
        ) from None
    if n < 0:
        raise ValueError(f"{D_HOSTGAP_ENV}={raw!r}: must be >= 0")
    return D_HOSTGAP_DEFAULT_ROUNDS if n == 1 else n


def launcher_env_d_defer_seq_lens_cpu() -> Dict[str, str]:
    return {D_DEFER_SEQ_LENS_CPU_ENV: "1"}


def launcher_env_d_hostgap(rounds: int = 1) -> Dict[str, str]:
    return {D_HOSTGAP_ENV: str(int(rounds))}


class DHostGap:
    """Per-rank accumulator behind ``#DGAP``.  All inputs are host
    ``perf_counter`` spans; ``end_round`` closes one decode round."""

    SPANS = ("publish_wait", "copy_done_wait", "hicache")

    def __init__(self, rounds: int, clock=time.perf_counter):
        self.rounds = int(rounds)
        self._clock = clock
        self._sums: Dict[str, float] = {k: 0.0 for k in self.SPANS}
        self._crit_sum = 0.0
        self._crit_n = 0
        self._crit_max = 0.0
        self._n = 0
        self._deferred = 0
        self._period_sum = 0.0
        self._last_end: Optional[float] = None
        self._wake_t: Optional[float] = None

    # -- spans ------------------------------------------------------------
    def add(self, name: str, ms: float) -> None:
        if name in self._sums:
            self._sums[name] += float(ms)

    @contextmanager
    def span(self, name: str):
        t0 = self._clock()
        try:
            yield
        finally:
            self.add(name, (self._clock() - t0) * 1000.0)

    # -- critical stretch: publish wake -> draft replay launched ----------
    def mark_wake(self) -> None:
        self._wake_t = self._clock()

    def mark_draft_launched(self) -> None:
        if self._wake_t is None:
            return
        ms = (self._clock() - self._wake_t) * 1000.0
        self._wake_t = None
        self._crit_sum += ms
        self._crit_n += 1
        if ms > self._crit_max:
            self._crit_max = ms

    # -- round close ------------------------------------------------------
    #: A round that closes this long after the previous one did not follow it
    #: (the decode stream paused: no request, a flip, a prefill phase). Its
    #: interval is not a round period, and the spans since then are not a
    #: round's host work -- both are dropped and the window restarts.
    IDLE_RESTART_S = 1.0

    def end_round(self, deferred: bool) -> Optional[str]:
        """Close one decode round; returns (and logs) the line when due."""
        now = self._clock()
        last, self._last_end = self._last_end, now
        if last is None or (now - last) > self.IDLE_RESTART_S:
            # First round, or the first after a pause: it only opens a window.
            self._reset_sums()
            return None
        self._period_sum += (now - last) * 1000.0
        self._n += 1
        self._deferred += 1 if deferred else 0
        if self._n < self.rounds:
            return None
        n = self._n
        period = self._period_sum / n
        publish = self._sums["publish_wait"] / n
        copy_done = self._sums["copy_done_wait"] / n
        hicache = self._sums["hicache"] / n
        result_wait = publish + copy_done
        host_other = period - result_wait - hicache
        crit = self._crit_sum / self._crit_n if self._crit_n else float("nan")
        line = (
            f"#DGAP rounds={n} period_ms={period:.3f} "
            f"result_wait_ms={result_wait:.3f} (publish {publish:.3f}, "
            f"copy_done {copy_done:.3f}) hicache_ms={hicache:.3f} "
            f"host_other_ms={host_other:.3f} crit_ms={crit:.3f} "
            f"(max {self._crit_max:.3f}, n={self._crit_n}; publish wake -> "
            f"draft replay launched) deferred={self._deferred}/{n} "
            f"(host clocks only; allreduce is device time inside the replays)"
        )
        logger.info(line)
        self._reset_sums()
        return line

    def _reset_sums(self) -> None:
        self._period_sum = 0.0
        for k in self._sums:
            self._sums[k] = 0.0
        self._crit_sum = 0.0
        self._crit_n = 0
        self._crit_max = 0.0
        self._n = 0
        self._deferred = 0


_METER: Optional[DHostGap] = None
_METER_RESOLVED = False


def meter() -> Optional[DHostGap]:
    """The process's ``#DGAP`` meter, or None when the instrument is off.
    Resolved once (the env is read at the first call). One scheduler per
    process, so one meter per rank; the log prefix names the rank."""
    global _METER, _METER_RESOLVED
    if not _METER_RESOLVED:
        _METER_RESOLVED = True
        try:
            rounds = hostgap_rounds()
        except ValueError as exc:
            # Loud, not fatal: an instrument must never take the boot down.
            logger.warning("#DGAP instrument OFF -- %s", exc)
            rounds = 0
        _METER = DHostGap(rounds) if rounds else None
        if _METER is not None:
            logger.info(
                "#DGAP instrument on: one line every %d decode rounds (%s)",
                rounds, D_HOSTGAP_ENV,
            )
    return _METER


def reset_for_tests() -> None:
    global _METER, _METER_RESOLVED
    _METER = None
    _METER_RESOLVED = False


@contextmanager
def span(name: str):
    """``meter().span(name)`` when the instrument is on, a bare yield otherwise."""
    m = meter()
    if m is None:
        yield
        return
    with m.span(name):
        yield


def defer_eligible(scheduler, batch) -> bool:
    """Whether ``run_batch`` may defer this batch's host-length read.

    The switch is read once per scheduler (cached on it); the rest is per
    batch: a spec-v2 DECODE batch with a relayed length (``future_indices``)
    and the reservation bound (``nxt_kv_lens_cpu``, what the DCP prebuild sizes
    by before the exact mirror exists), no grammar, no DSpark confidence
    prepare (it runs between the read and the forward), and a worker that
    completes the read itself (``supports_deferred_seq_lens_cpu``).  Anything
    else keeps the old up-front read."""
    on = getattr(scheduler, "_weg2_d_defer_lens_on", None)
    if on is None:
        on = defer_seq_lens_cpu_on()
        try:
            scheduler._weg2_d_defer_lens_on = on
        except Exception:  # noqa: BLE001
            pass
        if on:
            worker = getattr(scheduler, "model_worker", None)
            ok = bool(getattr(worker, "supports_deferred_seq_lens_cpu", False))
            logger.info(
                "DGAP-DEFER %s (%s=1): the spec-v2 decode round reads its host "
                "lengths at the worker's first use, not before the launch "
                "(worker %s)",
                "armed" if ok else "INERT -- the worker cannot complete a "
                "deferred read, every round keeps the old read",
                D_DEFER_SEQ_LENS_CPU_ENV,
                type(worker).__name__,
            )
    if not on:
        return False
    if getattr(scheduler, "_confidence_budget_prepare", None) is not None:
        return False
    if not getattr(getattr(scheduler, "model_worker", None),
                   "supports_deferred_seq_lens_cpu", False):
        return False
    if batch is None or not batch.forward_mode.is_decode():
        return False
    if getattr(batch, "is_extend_in_batch", False):
        return False
    if getattr(batch, "has_grammar", False):
        return False
    spec = getattr(batch, "spec_info", None)
    if spec is None or getattr(spec, "future_indices", None) is None:
        return False
    if getattr(spec, "nxt_kv_lens_cpu", None) is None:
        return False
    return True


class PendingSeqLensCpu:
    """A length read whose device half is queued and whose host half is not.

    ``complete(batch)`` waits for the D2H event (the one host wait the old read
    made up front), then fills ``batch.seq_lens_cpu`` / ``seq_lens_sum`` exactly
    as the old read did.  Idempotent: the second call only re-applies the values
    (the scheduler's forward isolation restores the pre-forward snapshot, in
    which they were still ``None``)."""

    __slots__ = ("_pinned", "_event", "_idx_cpu", "seq_lens_cpu", "seq_lens_sum")

    def __init__(self, pinned, event, req_pool_indices_cpu):
        self._pinned = pinned
        self._event = event
        self._idx_cpu = req_pool_indices_cpu
        self.seq_lens_cpu = None
        self.seq_lens_sum = None

    @property
    def done(self) -> bool:
        return self._event is None

    def complete(self, batch) -> None:
        if self._event is not None:
            m = meter()
            if m is not None:
                with m.span("publish_wait"):
                    self._event.synchronize()
                m.mark_wake()
            else:
                self._event.synchronize()
            self._event = None
            self.seq_lens_cpu = self._pinned[self._idx_cpu]
            self.seq_lens_sum = int(self.seq_lens_cpu.sum())
            self._pinned = None
        batch.seq_lens_cpu = self.seq_lens_cpu
        batch.seq_lens_sum = self.seq_lens_sum
