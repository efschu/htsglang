"""S50 (18.10.) TOLD-ACK-EARLY: a P follower's read ack leaves when the read ends, not at the next pass boundary.

THE PAUSE (S48 part A, class 1; rc13 P log 1018_200305, pdflip-8-123). PP0 admits a paced told only when every follower's
ack (``pdflip_told_fallback.PdFlipToldReadAck``) says its read reached told. A follower reports in ``follower_pump``, and that
runs once per pass, in ``follower_absorb`` -- AFTER the follower's synchronous PP forward of the previous request:
    reads physically done on all ranks 20:19:07.1 (read_ms 38-73)
    PP1/PP2 harvested 20:19:10 (``PDFLIP-LOAD-DEVICE harvest_ms=2603/2621``: their forward of 8-122)
    PP0 sat in its commit block 09.25-11.76 (``PASS-STALL commit_ms=2450``), took the acks after it, admitted 20:19:11.
With the acks out at ~07.2, PP0's passes at 09.0/09.2 hold them and the admission comes before the block (~2.5 s; S48
counted read-done -> VERDICT with a commit block in between at ~25 s per 57 min).

THE RULE. While a P follower's scheduler thread is inside the model forward (``Scheduler._run_batch_forward``, the plain
PP branch: ``model_worker.forward_batch_generation``), it touches neither the radix tree nor the told/ack state -- that is the
FORWARD WINDOW. A worker thread of this module may run the UNCHANGED ``pdflip_told_fallback.follower_pump`` exactly then:
  * the scheduler thread HOLDS ``window`` (a ``threading.Lock``) all the time except across that one call -- it releases
    it on entry and re-acquires it on exit (waiting for a pump in progress, a few ms) before it runs any other code;
  * the worker runs only while it holds ``window`` (non-blocking acquire: outside a window it does nothing; the next
    ``follower_absorb`` reports as before);
  * the worker is woken by the HiCache read thread itself (``cache_controller.prefetch_io_aux_func`` calls
    :func:`note_read_end` right after it stamps ``read_end_time``) and once more when a window opens with a wake pending
    (a read that ended while the scheduler held the lock).
So the ack's content is the pump's own (harvest via ``tree.check_prefetch_progress`` -- a read that has not terminated is
not reported --, ``own_prefix``, room / anchor terms), and every pump call, early or not, is serialised by ``window``: one
message in flight per follower, size-then-payload framing intact, seq monotone (the ring order PP0 reads). Followers still
decide nothing: PP0's verdict and the wire are unchanged (SOLL R11), the ack is only earlier.

ARMED only on a P follower (pp_rank > 0, tp_size 1: the harvest is collective-free) that runs the PF ack stream (``_pdflip_fb_follower`` exists: a paced read-ahead
carrying ``ack`` arrived) -- PP0, D and every boot without PF never create the worker; the window is then a null context.
Switch ``FLLIPER_PDFLIP_TOLD_ACK_EARLY`` (default ON; 0 = the pump only at pass boundaries, byte for byte).
Instrument: ``PF TOLD-ACK EARLY`` (sampled) with the ack's seq, reads and how far into the forward it left.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

ATTR = "_pdflip_told_ack_early"
#: every live instance in this process (one scheduler per process; tests may hold several)
_LISTENERS: List["EarlyAck"] = []
_LIST_LOCK = threading.Lock()


def enabled() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_TOLD_ACK_EARLY.get())
    except Exception:  # noqa: BLE001 -- no registry: the default
        return True


def note_read_end(operation: Any = None) -> None:
    """HiCache read thread: a store read just ended. Wakes every worker (cheap; no-op without one)."""
    if not _LISTENERS:
        return
    with _LIST_LOCK:
        targets = list(_LISTENERS)
    for ea in targets:
        ea.poke()


class EarlyAck:
    """One follower's early-ack worker and its forward window."""

    def __init__(self, scheduler: Any, pump=None) -> None:
        self.scheduler = scheduler
        self._pump = pump
        self.window = threading.Lock()
        self.window.acquire()          # the scheduler thread (creator) owns the window outside a forward
        self._owner = threading.current_thread()
        self._wake = threading.Event()
        self.pending = False
        self.in_window = False
        self.window_t0 = 0.0
        self.windows = 0
        self.early_sends = 0
        self.runs = 0
        self.errors = 0
        self._stop = False
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                self._device = torch.cuda.current_device()
        except Exception:  # noqa: BLE001
            self._device = None
        self._thread = threading.Thread(target=self._loop, name="pdflip-told-ack-early", daemon=True)
        with _LIST_LOCK:
            _LISTENERS.append(self)
        self._thread.start()

    # -- read thread / window entry ---------------------------------------------------------------
    def poke(self) -> None:
        self.pending = True
        self._wake.set()

    # -- scheduler thread ------------------------------------------------------------------------
    @contextlib.contextmanager
    def forward(self):
        """Release the window across the model forward; re-acquire (waiting for a running pump) before returning.

        Re-entrant on the scheduler thread: a forward inside a forward (none today) leaves the window as it is."""
        if self.in_window and threading.current_thread() is self._owner:
            yield
            return
        self.windows += 1
        self.window_t0 = time.monotonic()
        self.in_window = True
        self.window.release()
        try:
            if self.pending:
                self._wake.set()
            yield
        finally:
            self.window.acquire()
            self.in_window = False

    # -- worker ----------------------------------------------------------------------------------
    def _loop(self) -> None:
        if self._device is not None:
            try:
                import torch

                torch.cuda.set_device(self._device)
            except Exception:  # noqa: BLE001
                pass
        while not self._stop:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop or not self.pending:
                continue
            self.run_once()

    def run_once(self) -> bool:
        """One early pump if (and only if) the scheduler thread is inside a forward window. True = it ran."""
        if not self.window.acquire(blocking=False):
            return False   # the scheduler holds the window (outside a forward): the next window / absorb reports
        try:
            if not self.in_window:
                return False
            self.pending = False
            self.runs += 1
            self._pump_and_note()
            return True
        except Exception as exc:  # noqa: BLE001 -- the pass-boundary pump stays the fallback
            self.errors += 1
            if self.errors <= 8 or self.errors % 256 == 0:
                logger.warning("PF TOLD-ACK EARLY raised (%d so far): %r -- the pass-boundary pump reports instead",
                               self.errors, exc, exc_info=self.errors <= 2)
            return False
        finally:
            self.window.release()

    def _pump_and_note(self) -> None:
        from flliper.srt.managers import pdflip_told_fallback as _fb

        sched = self.scheduler
        st = _fb._fstate(sched)
        seq0 = getattr(st, "seq", None)
        (self._pump or _fb.follower_pump)(sched)
        seq1 = getattr(st, "seq", None)
        if seq0 is not None and seq1 is not None and seq1 > seq0:
            self.early_sends += 1
            n = self.early_sends
            if n <= 32 or n % 64 == 0:
                logger.info(
                    "PF TOLD-ACK EARLY rank pp=%s seq=%d in_forward_ms=%.0f (n=%d): the read ended inside this "
                    "follower's forward; its ack left from the forward window instead of the next pass boundary",
                    getattr(getattr(sched, "ps", None), "pp_rank", "?"), seq0,
                    (time.monotonic() - self.window_t0) * 1000.0, n,
                )

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        with _LIST_LOCK:
            if self in _LISTENERS:
                _LISTENERS.remove(self)


def _armed(scheduler) -> bool:
    try:
        ps = getattr(scheduler, "ps", None)
        if int(getattr(ps, "pp_size", 1) or 1) <= 1 or int(getattr(ps, "pp_rank", 0) or 0) == 0:
            return False
        if int(getattr(ps, "tp_size", 1) or 1) != 1:
            return False   # the harvest is collective-free only at tp_size 1 (follower_pump's own guard) -- never a
                           # collective from a second thread
        if getattr(scheduler, "_pdflip_fb_follower", None) is None:
            return False
        return enabled()
    except Exception:  # noqa: BLE001
        return False


def forward_window(scheduler):
    """The model-forward call site's context: the early-ack window on an armed P follower, a null context otherwise.

    Created lazily on the scheduler thread (which then owns the window)."""
    ea = getattr(scheduler, ATTR, None)
    if ea is None:
        if not _armed(scheduler):
            return contextlib.nullcontext()
        ea = EarlyAck(scheduler)
        setattr(scheduler, ATTR, ea)
    return ea.forward()


def instance(scheduler) -> Optional[EarlyAck]:
    return getattr(scheduler, ATTR, None)
