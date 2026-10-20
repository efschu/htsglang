"""K12 (NF h16p 226aad7105, D TP0 15.10.2026 13:35:33Z): a storage backup never
takes the group down.

THE DEATH (boot ..._226aad7105_1015_124245, D.log 271625-271645 / 271707-271733).
D TP0's ``backup_thread_func`` ran ``_page_backup`` on an operation whose
``qsa_indexer`` sidecar named host page 72085 of a 6549-page mirror
(``IndexError ... out of bounds for dimension 0 with size 6549`` in
``DeepSeekV4PagedHostPool.get_data_page`` via ``batch_set_v2``). The loop
caught only ``Empty``, so the thread died -- silently, no rank died with it --
and its two in-flight operations were never acknowledged. Ten seconds later the
D>P sleep drain (``pdflip_sleep_drain``) still read ``hicache_backup(2)`` on TP0
and raised W120 on 3/3, W29, GROUP-FENCE FAILED 0/3, RANK_EXCEPTION. Every
earlier sleep drain of the boot, including one with the same
``hicache_backup(2)`` 76 s before, took <= 0.13 s: the blocker was not slow,
its consumer was gone.

Two layers, both rank-uniform where they decide anything:

1. THE CONSUMER SURVIVES. One operation that raises is acknowledged as FAILED
   (``pdflip_backup_failed``) and the thread takes the next. The ack drain then
   releases the node's host lock WITHOUT ``l3_present`` -- the store copy is
   lost, the host and device rows are untouched, so nothing a reader needs is
   freed on the strength of a write that never landed (#1317 C2).
2. THE SLEEP ABANDONS WHAT NOBODY IS WRITING. When the group drain expires
   with backup terms only (no prefetch / load / write-through, no non-HiCache
   blocker) and no rank has a backup operation EXECUTING on a live thread, the
   group verdict -- a MAX reduction every rank reads identically -- is to
   abandon: each rank cancels its queued backups (the thread acks them unwritten
   at dequeue), releases their host locks without ``l3_present`` and sleeps.
   An operation that is executing on a live thread is never abandoned: its
   reads of the host rows would race the reset, so that case keeps the W120
   stop by name.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: the attribute an operation carries when its backup did not land
FAILED_ATTR = "pdflip_backup_failed"

_FAILED_LOGGED = 0
_FAILED_TOTAL = 0


def _state(controller):
    """The per-controller guard state, created on first use (a controller built
    before this module existed, or a desk double, gets it lazily)."""
    st = getattr(controller, "_k12_backup_guard", None)
    if st is None:
        st = {"lock": threading.Lock(), "executing": None, "cancelled": set()}
        try:
            controller._k12_backup_guard = st
        except Exception:  # noqa: BLE001 - a frozen double still gets a state
            pass
    return st


def is_failed(operation) -> bool:
    return bool(getattr(operation, FAILED_ATTR, None))


def mark_failed(operation, why: str) -> None:
    try:
        setattr(operation, FAILED_ATTR, str(why) or "failed")
    except Exception:  # noqa: BLE001
        pass


def begin(controller, operation) -> bool:
    """Called by the backup thread right after dequeue. False = the operation
    was abandoned by the group's sleep verdict: ack it unwritten. True = it is
    now EXECUTING (the abandon path will not touch it)."""
    st = _state(controller)
    op_id = getattr(operation, "id", None)
    with st["lock"]:
        if op_id is not None and op_id in st["cancelled"]:
            st["cancelled"].discard(op_id)
            mark_failed(operation, "abandoned by the group sleep drain (K12)")
            return False
        st["executing"] = op_id
        return True


def end(controller, operation) -> None:
    st = _state(controller)
    with st["lock"]:
        if st["executing"] == getattr(operation, "id", None):
            st["executing"] = None


def fail(operation, exc: BaseException) -> None:
    """One backup operation raised: mark it FAILED and name it. The caller
    (the backup thread) acks it and takes the next operation."""
    global _FAILED_LOGGED, _FAILED_TOTAL
    _FAILED_TOTAL += 1
    _FAILED_LOGGED += 1
    mark_failed(operation, f"{type(exc).__name__}: {exc}")
    logger.error(
        "K12 HICACHE-BACKUP-FAILED op=%s rid=%s pages=%d total=%d: %s: %s -- the "
        "operation is acknowledged as FAILED (no l3_present, host/device rows "
        "untouched, store copy lost) and the backup thread takes the next one; "
        "before K12 this exception ended the thread and left its operations "
        "unacknowledged for ever (D TP0 13:35:33Z -> W120 13:35:43Z)",
        getattr(operation, "id", "?"),
        getattr(operation, "request_id", None) or getattr(operation, "rid", "?"),
        len(getattr(operation, "hash_value", None) or ()),
        _FAILED_TOTAL,
        type(exc).__name__,
        exc,
        exc_info=(type(exc), exc, exc.__traceback__) if _FAILED_LOGGED <= 4 else None,
    )


def failed_total() -> int:
    return _FAILED_TOTAL


def _thread_alive(controller) -> bool:
    t = getattr(controller, "backup_thread", None)
    try:
        return bool(t is not None and t.is_alive())
    except Exception:  # noqa: BLE001
        return False


def drain_state(tree) -> Tuple[int, int]:
    """This rank's ``(backup_terms, executing)`` for the sleep drain:
    ``backup_terms`` = open storage backups (``hicache_backup(n)``),
    ``executing`` = 1 when one of them is being written by a LIVE thread right
    now (abandoning it would race its host-row reads against the reset)."""
    ob = getattr(tree, "ongoing_backup", None) or {}
    n = len(ob)
    if not n:
        return 0, 0
    cc = getattr(tree, "cache_controller", None)
    if cc is None:
        # no controller to ask: unknown is not "nobody is writing" -- the rank
        # votes executing, and the group keeps the W120 stop
        return n, 1
    st = _state(cc)
    with st["lock"]:
        ex = st["executing"]
    return n, int(ex is not None and ex in ob and _thread_alive(cc))


def _release_entry(tree, entry) -> None:
    """Release one ``ongoing_backup`` entry's host lock the way the tree's own
    ack drain does, WITHOUT marking anything stored: UnifiedRadixCache keeps
    ``(node, lock_params)``, HiMambaRadixCache ``(node, mamba_protected)``,
    HiRadixCache the node itself."""
    if hasattr(tree, "_release_host_node") and isinstance(entry, tuple):
        node, mamba_protected = entry
        tree._release_host_node(node, release_mamba=mamba_protected)
    elif isinstance(entry, tuple):
        node, lock_params = entry
        tree.dec_host_lock_ref(node, lock_params)
    elif hasattr(entry, "release_host"):
        entry.release_host()


def abandon(tree, why: str) -> Tuple[int, int]:
    """Abandon every open backup of this rank that is not executing. Returns
    ``(abandoned, kept_executing)``. Each abandoned operation is cancelled at
    the controller (acked unwritten when the thread dequeues it, if it ever
    does), its node's host lock is released WITHOUT ``l3_present``, and its
    staging-ring charge is returned."""
    ob = getattr(tree, "ongoing_backup", None)
    if not ob:
        return 0, 0
    cc = getattr(tree, "cache_controller", None)
    st = _state(cc) if cc is not None else None
    alive = _thread_alive(cc) if cc is not None else False
    abandoned = kept = 0
    for op_id in list(ob.keys()):
        if st is not None:
            with st["lock"]:
                if st["executing"] == op_id and alive:
                    kept += 1
                    continue
                st["cancelled"].add(op_id)
        entry = ob.pop(op_id, None)
        settle = getattr(tree, "_qsa_pair_settle", None)
        if callable(settle):
            # QSA-PAIR: a KV slot waiting for this (now abandoned) index write
            # completes before the host lock goes -- never left WRITING
            try:
                settle(op_id, "abandon")
            except Exception:  # noqa: BLE001 - the abandon names, it never raises
                logger.warning("QSA-PAIR settle at abandon failed op=%s", op_id, exc_info=True)
        if entry is not None:
            try:
                _release_entry(tree, entry)
            except Exception:  # noqa: BLE001 - the abandon names, it never raises
                logger.warning("K12 ABANDON host-lock release failed op=%s", op_id,
                               exc_info=True)
        ring = getattr(tree, "staging_write_ring", None)
        if ring is not None:
            try:
                ring.release(op_id)
            except Exception:  # noqa: BLE001
                pass
        abandoned += 1
    if abandoned or kept:
        logger.warning(
            "K12 SLEEP-DRAIN ABANDON-BACKUP abandoned=%d kept_executing=%d why=%s: "
            "the group's sleep verdict gives up store copies nobody is writing "
            "(no l3_present, host/device rows untouched); a late ack of an "
            "abandoned id finds no entry and changes nothing",
            abandoned, kept, why,
        )
    return abandoned, kept
