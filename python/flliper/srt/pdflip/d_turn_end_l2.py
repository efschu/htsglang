"""S108 (20.10.): D writes every finished turn from the card (L1) to L2 at once, asynchronously.

USER ORDERS (20.10., verbatim):
  * "in den L3 wird doch aus dem L2 zurückgeschrieben. völlig unabhängig von der phase"
  * on "Soll D fertige Turns sofort von der Karte nach L2 kopieren?": "JA NATÜRLICH"
  * "UND WEHE DAS SCHREIBEN IN DEN CACHE BLOCKIERT DURCH IRGENDEINE SYNCHRON ODER WARTESCHEIßE"

THE GAP. Group D published its tree to L2 only in the idle sweep and at the flush; a finished turn
lived on the card alone until then (S104 1e: 63113 tokens only on D at deaths/stops). P has published
at the retain since xsn338 -- but through ``write_backup`` -> ``HybridCacheController.write`` whose
issue WAITS for the card on D's Form A token cut: the #923 row check (``int(rows.max())``), the owner
mask (``compact[owned]``, ``owned.to(cpu)``), the direct backend's ``.cpu()`` and, on the token
owners TP1/TP2, ``ArenaMHAHostPool._backup_owner_rows`` -- a pageable ``.to("cpu")`` per layer, K and
V, behind the start event of the forward in flight. Calling that at every D finish is the decode
stall the order forbids.

THE FORM (this module + ``HybridCacheController._s108_submit/_s108_issue`` +
``UnifiedRadixCache._s108_turn_end_l2``):

* The scheduler thread keeps only HOST work: the tree walk of the request's own chain, the arena
  claim (C, host), the host rows, the commit and the write-through lock -- exactly what every backup
  does -- and it records ONE event on the compute stream (asynchronous, no wait). It never reads a
  device value, never masks on the device, never moves indices to the host, never waits on an event
  or a stream, never joins and never calls a collective.
* The op goes to :class:`TurnEndCopier`, one daemon thread per controller. THAT thread does what
  needs device values (owner rows, the exact #923 check, index normalisation) and the copy itself
  on the write stream behind the recorded event; whatever it waits for, it waits for alone.
* The ack is a :class:`DoneEvent` in the ordinary ``ack_write_queue``: the existing ack drain
  (``writing_check`` -> ``_finish_write_through_ack`` -> arena ``complete_write``; S74/S99) sees it
  ready once the copier has finished the op. COMPLETE therefore still comes from every owner's merged
  extents -- group-equal, after the fact, no new collective.
* A copier failure (a #923 hit, a raising pool) copies nothing and REFUSES the op (review A4): the
  ack drain undoes the commit (claim back, host values off, lock released), names it
  (``S108 TURN-END-REFUSED``) and reports it through the refusal path (R12 on Form A) -- never a single
  rank's death with an open claim.
* A full KV arena / a busy mamba pin: the node is DEFERRED by name (``S108 TURN-END-DEFER``)
  and stays unbacked for the next turn end, the idle sweep and the flush -- never dropped, never a
  spill (the W3 spill writes L3 in the caller's thread), never a wait. A full MAMBA arena makes the
  node KV-only on that rank (review A2/A3): no H19 displacement or disk secure, no TP0-only refusal;
  the anchor is deferred by name for the anchor-only backup. No wall-clock budget (rank-local).

WHY NO MAMBA FENCE ON THESE OPS. ``MAMBA-SNAPSHOT-FENCE`` makes the compute stream wait for a write
op that snapshots recurrent state, because a running request's state is updated in place. A finished
turn's anchor is the TREE's slot: a later request of the session takes it copy-on-match
(``MambaComponent`` ``cow_mamba`` -> ``mamba_cow_src_index``), so no forward writes it, and the
write-through lock keeps it from being freed until the ack. A fence here would put the whole
turn's copy in front of the next decode round -- the stall this module exists to avoid.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from contextlib import contextmanager
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

TURN_LINE = "S108 TURN-END-L2"
DEFER_LINE = "S108 TURN-END-DEFER"
COPY_LINE = "S108 TURN-END-COPY"
DEFER_KEEP = 4096


def turn_end_on(env: Optional[Mapping[str, str]] = None) -> bool:
    """Group D, always (user order: no switch). Group P has its own retain publish (xsn338)."""
    env = os.environ if env is None else env
    return str(env.get("FLLIPER_PDFLIP_GROUP", "")).strip().upper() == "D"


class TurnEndCopyError(RuntimeError):
    """A turn-end copy the copier could not carry out (named); raised in the scheduler thread."""


class DoneEvent:
    """The ack of one copier op, shaped like the device event the ack drain polls.

    ``query()`` never waits: True once the copier finished the op (its write-stream finish event
    complete), raises the copier's named error if it failed. ``synchronize()`` waits -- only the
    blocking drains call it (write_back eviction, reset), never the turn-end path."""

    __slots__ = ("_done", "_err", "finish_event", "copy_ms", "wait_ms", "kv_rows", "kv_bytes", "tag", "refused")

    def __init__(self, tag: str = "") -> None:
        self._done = threading.Event()
        self._err: Optional[BaseException] = None
        self.finish_event = None
        self.copy_ms = 0.0
        self.wait_ms = 0.0
        self.kv_rows = 0
        self.kv_bytes = 0
        self.tag = tag
        #: review A4: why the copier refused the op (nothing copied); the ack undoes the commit
        self.refused: Optional[str] = None

    def refuse(self, why: str) -> None:
        """The copier did not copy (a #923 row, a raising pool): the ack is READY and carries the
        reason; ``UnifiedRadixCache._s108_ack`` undoes the commit, names it, reports it through the
        refusal path. Never an exception in the scheduler thread (review A4: no single-rank death)."""
        self.refused = str(why) or "refused"
        self._done.set()

    def set_done(self, finish_event=None) -> None:
        self.finish_event = finish_event
        self._done.set()

    def fail(self, exc: BaseException) -> None:
        self._err = exc
        self.refuse("copier_raised:%s:%s" % (type(exc).__name__, str(exc)[:160]))

    def query(self) -> bool:
        return self._done.is_set()

    def synchronize(self) -> None:
        self._done.wait()

    def elapsed_time(self, other) -> float:  # pragma: no cover - the ack instrument's try/except
        raise TypeError("DoneEvent carries copy_ms instead of device timing")


class TurnEndCopier:
    """One daemon thread per controller: runs ``issue(op, start_event, done)`` for each submitted
    turn-end op, in order. ``submit`` never waits (an unbounded SimpleQueue put)."""

    def __init__(self, name: str = "s108-turn-end-copier", device: Optional[int] = None) -> None:
        #: the scheduler thread's CUDA device -- a new thread starts on device 0
        self._device = device
        self._q: "queue.SimpleQueue" = queue.SimpleQueue()
        self._idle = threading.Event()
        self._idle.set()
        self._inflight = 0
        self._n = 0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn, *args) -> None:
        self._inflight += 1
        self._idle.clear()
        self._q.put((fn, args))

    def _run(self) -> None:
        if self._device is not None:
            import torch

            torch.cuda.set_device(self._device)
        while True:
            fn, args = self._q.get()
            if fn is None:
                return
            try:
                fn(*args)
            except BaseException as exc:  # noqa: BLE001 - the op's DoneEvent carries it
                done = args[-1] if args else None
                if isinstance(done, DoneEvent) and not done._done.is_set():
                    done.fail(exc)
                logger.error("%s copier raised %s: %s", COPY_LINE, type(exc).__name__, exc)
            finally:
                self._n += 1
                self._inflight -= 1
                if self._inflight <= 0:
                    self._idle.set()

    def join(self, timeout: Optional[float] = None) -> bool:
        """Wait until every submitted op ran (reset/flush only -- never on the turn-end path)."""
        return self._idle.wait(timeout)

    @property
    def inflight(self) -> int:
        return self._inflight


def copier_of(controller: Any) -> TurnEndCopier:
    """The controller's copier (created on first use, in the scheduler thread, once)."""
    c = controller.__dict__.get("_s108_copier")
    if c is None:
        dev = None
        try:
            import torch

            if torch.cuda.is_available():
                dev = torch.cuda.current_device()   # a host-side query, no sync
        except Exception:  # noqa: BLE001
            dev = None
        c = controller.__dict__["_s108_copier"] = TurnEndCopier(device=dev)
    return c


@contextmanager
def deferred_issue(controller: Any, tag: str):
    """Write ops issued inside go to the copier instead of being issued inline."""
    prev = controller.__dict__.get("_s108_defer_issue", False)
    prev_tag = controller.__dict__.get("_s108_tag", "")
    controller._s108_defer_issue = True
    controller._s108_tag = tag
    try:
        yield
    finally:
        controller._s108_defer_issue = prev
        controller._s108_tag = prev_tag


class IssueClock:
    """Wall and thread-CPU time of the scheduler thread's part of one turn end. ``blocked`` = wall -
    CPU: what the thread spent NOT computing (waiting on the card, a lock or the GIL) -- ~0 is the
    order."""

    __slots__ = ("_w0", "_c0")

    def __init__(self) -> None:
        self._w0 = time.perf_counter()
        self._c0 = time.thread_time()

    def read(self):
        wall = (time.perf_counter() - self._w0) * 1000.0
        cpu = (time.thread_time() - self._c0) * 1000.0
        return wall, cpu, max(0.0, wall - cpu)


def owner_share(ctx) -> str:
    """'hi-lo/S' of this rank's token cut, or 'all' (no owner rule)."""
    if ctx is None:
        return "all"
    S, lo, hi = ctx
    return f"{int(hi) - int(lo)}/{int(S)}"


def format_turn_line(*, rid: str, chain: int, stats: Mapping[str, Any], ops: int, tokens: int,
                     share: str, wall: float, cpu: float, blocked: float, n: int,
                     deferred_open: int, anchor_deferred: int = 0, match_ms: Optional[float] = None,
                     sweep_ms: Optional[float] = None, claim_ms: Optional[float] = None) -> str:
    # KVLEITER-1022: issue_ms split into match_ms (chain match + deferred filter), sweep_ms (the sweep's walk) and
    # claim_ms (its claims / write_backup issues) -- appended after n=, the fields before are unchanged
    split = ("" if match_ms is None else " match_ms=%.2f sweep_ms=%.2f claim_ms=%.2f" % (
        float(match_ms), float(sweep_ms or 0.0), float(claim_ms or 0.0)))
    return (
        "%s rid=%s chain=%d unbacked=%d issued=%d deferred=%d ops=%d tokens=%d share=%s "
        "issue_ms=%.2f issue_cpu_ms=%.2f blocked_ms=%.2f deferred_open=%d anchor_deferred=%d n=%d%s "
        "(scheduler thread only: host claims/commits, one event record; the copy runs on the "
        "copier thread, its bytes and ms are the %s lines)"
        % (TURN_LINE, rid, chain, int(stats.get("unbacked", 0) or 0), int(stats.get("issued", 0) or 0),
           max(0, int(stats.get("unbacked", 0) or 0) - int(stats.get("issued", 0) or 0)), ops, tokens, share,
           wall, cpu, blocked, deferred_open, anchor_deferred, n, split, COPY_LINE)
    )


def format_defer_line(*, rid: str, why: str, nodes) -> str:
    ids = ",".join(str(getattr(x, "id", "?")) for x in list(nodes)[:16])
    return (f"{DEFER_LINE} rid={rid} why={why} nodes={len(list(nodes))} ids=[{ids}] "
            "(kept unbacked for the next turn end, the idle sweep and the flush -- not dropped)")


def format_copy_line(done: DoneEvent, n: int) -> str:
    return (
        "%s tag=%s kv_rows=%d kv_bytes=%d copy_ms=%.1f start_wait_ms=%.1f n=%d "
        "(copier thread: write stream behind the turn end's event; the scheduler thread did not wait)"
        % (COPY_LINE, done.tag, done.kv_rows, done.kv_bytes, done.copy_ms, done.wait_ms, n)
    )
