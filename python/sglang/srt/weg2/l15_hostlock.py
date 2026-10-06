# SPDX-License-Identifier: Apache-2.0
"""L15-HOSTLOCK: arena reader references spanning the L1.5 sleep->wake hold.

LCHOST defect 2: retain_at_sleep step (5) reset_keep runs _reset_full first,
whose _release_host_values_before_reset hands every kept chain's arena
references back (kept chains never carry host_lock_ref). From that instant
the recorded L2 slots (HoldSpan.l2_slots, anchor_l2_slot) are eviction
candidates, and P's write-through in the shared /dev/shm arena claims them
during its phase -- a re-claim bumps the generation -> wake gen_check
mismatch -> fallback. This module takes ONE arena reader reference per
distinct held L2 slot of THIS rank before that release, and gives them back
exactly once when the wake verdict is acted (hold: after the refill copied;
fallback: inside the drop). A rank's l2_slots are its OWN shard's arena
slots -- no other rank's reference pins them, cap-0 ranks included.

Reclamation of a lost record: refs taken at a sleep whose wake never runs
(crash, hard kill) stay on the shared arena until the arena file is
recreated by the next boot -- the same lifetime the carrier hold's stashed
rows already have, and acceptable per the AP brief. The refs only ever pin
against eviction; they never move or copy data.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

__all__ = ["hold_sleep_refs", "release_wake_refs", "name_hold_on_tree",
           "clear_hold_on_tree"]


def _distinct(slots: Sequence[int]) -> Tuple[int, ...]:
    """Order-preserving dedupe, dropping the -1 "not in L2" markers."""
    out: list = []
    seen: set = set()
    for s in slots:
        s = int(s)
        if s >= 0 and s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


def _ref_pool(pool, slots: Tuple[int, ...], delta: int,
              log: Callable[[str], None], what: str) -> Tuple[int, ...]:
    """Apply one delta on the pool's arena; return what may be recorded.

    Only a delta the arena fully took is recordable: the arena's
    process-local RefLedger records a +1 batch ONLY when the whole batch
    landed (a partial batch is leaked rather than risk a double release),
    so recording a partial would arm a -1 the ledger refuses anyway. No
    pool / no arena / no API -> log, record nothing (today's behaviour).
    """
    if not slots:
        return ()
    arena = getattr(pool, "arena", None) if pool is not None else None
    fn = getattr(arena, "ref_slots", None) if arena is not None else None
    if fn is None:
        log(f"L15-HOSTLOCK {what}: pool {type(pool).__name__ if pool else None}"
            f" has no arena ref API: {len(slots)} slot(s) NOT held")
        return ()
    try:
        took = int(fn(slots, delta))
    except Exception as exc:  # noqa: BLE001 - hold is best effort, never fatal
        log(f"L15-HOSTLOCK {what}: ref_slots({delta:+d}) failed "
            f"({type(exc).__name__}: {exc}): {len(slots)} slot(s) NOT held")
        return ()
    log(f"L15-HOSTLOCK {what} delta={delta:+d} slots={len(slots)} took={took}")
    return slots if took == len(slots) else ()


def hold_sleep_refs(kv_pool, mamba_pool, kv_slots: Sequence[int],
                    anchor_slots: Sequence[int],
                    log: Callable[[str], None]) -> Optional[Tuple[tuple, tuple]]:
    """Pin this rank's distinct held L2 slots (+1) before reset_keep releases.

    Returns the record (kv, anchors) of what was actually pinned (either
    side possibly empty), or None when nothing was held -- the caller stores
    the record and hands it back to release_wake_refs at the wake act.
    """
    kv = _distinct(kv_slots)
    an = _distinct(anchor_slots)
    kv_held = _ref_pool(kv_pool, kv, +1, log, "kv-sleep-hold")
    an_held = _ref_pool(mamba_pool, an, +1, log, "anchor-sleep-hold")
    if not kv_held and not an_held:
        return None
    return (kv_held, an_held)


def release_wake_refs(kv_pool, mamba_pool, held: Tuple[Sequence[int], ...],
                      log: Callable[[str], None]) -> int:
    """Give the recorded references back (-1); count released. Best effort.

    A -1 beyond what this process holds is refused by the arena's ledger
    (counted, not applied), so a stale or foreign record cannot drop another
    reader's pin. The caller drops the record; calling this twice is a
    ledger-refused no-op.
    """
    kv, an = (tuple(held) + ((), ()))[:2]
    gave = 0
    gave += len(_ref_pool(kv_pool, _distinct(kv), -1, log, "kv-wake-release"))
    gave += len(_ref_pool(mamba_pool, _distinct(an), -1, log,
                          "anchor-wake-release"))
    return gave


def name_hold_on_tree(tree, pools: Tuple[object, object], held,
                      log: Callable[[str], None]) -> int:
    """L15-HOSTLOCK-NAMED: tell the tree its reset's orphan pass has a holder.

    The pins of ``hold_sleep_refs`` are taken BEFORE ``reset_keep``; that
    reset's #1424g orphan pass (UnifiedRadixCache._weg2_release_orphan_refs)
    gives back every reference of this process that no holder names -- and
    the pin, recorded only on the scheduler, had no name: N6 ..._1006_050049
    gave all 129792 pins back in the same sleep (``RESET-ORPHANS released=
    129792``), the wake's release then took 0/44/59, and the first full arena
    let P recycle the held L2 slots (L15-CHECK REFUSED epoch 66). The tree
    now names the recorded slots next to its carrier rows. A tree without the
    API (a stub) names nothing: today's behaviour. Returns the slots named."""
    fn = getattr(tree, "weg2_set_l15_hold", None)
    if fn is None or not held:
        return 0
    import numpy as np

    kv_pool, mamba_pool = pools
    kv, an = (tuple(held) + ((), ()))[:2]
    entries = []
    for pool, slots in ((kv_pool, kv), (mamba_pool, an)):
        if pool is not None and slots:
            entries.append((pool, np.asarray(_distinct(slots), dtype=np.int64)))
    try:
        n = int(fn(entries))
    except Exception as exc:  # noqa: BLE001 - naming is best effort, never fatal
        log(f"L15-HOSTLOCK-NAMED failed ({type(exc).__name__}: {exc}): the "
            "reset's orphan pass will give the pins back")
        return 0
    share = ""
    try:
        for pool, slots in entries:
            led = getattr(getattr(pool, "arena", None), "_ledger", None)
            if led is not None and int(led.held.numel()):
                share += " %s=%.1f%%" % (
                    "kv" if pool is kv_pool else "anchor",
                    100.0 * len(slots) / int(led.held.numel()))
    except Exception:  # noqa: BLE001 - instrument only
        share = ""
    log(f"L15-HOSTLOCK-NAMED slots={n} (kept by the reset's orphan pass until "
        f"the wake act releases them; share of the arena:{share or ' ?'})")
    return n


def clear_hold_on_tree(tree) -> int:
    """L15-HOSTLOCK-NAMED partner of ``name_hold_on_tree``: drop the name (the
    wake act / an undone sleep is about to give the references back). No API
    -> 0. Never raises."""
    fn = getattr(tree, "weg2_clear_l15_hold", None)
    if fn is None:
        return 0
    try:
        return int(fn())
    except Exception:  # noqa: BLE001
        return 0


def rearm_sink(get: Callable[[], object], put: Callable[[object], None],
               pools: Callable[[], Tuple[object, object]],
               log: Callable[[str], None]) -> Callable[[object], None]:
    """L15-FIX-HOSTLOCK-REARM: the hold-record sink of the sleep hook.

    The retain hook fires on EVERY flush while D is parked; N3o (02.10.
    05:48:44/50) ran it twice in one sleep (the flip's pre-sleep flush, then
    the release RPC's flush). Each retain pins its held L2 slots; the wake
    releases ONE record. So a new record supersedes the previous one: store
    the new record, THEN release the old one (the new refs are already taken,
    the shared slots never drop to 0 in between). A ``None`` record (a retain
    that pinned nothing) also releases the previous record.
    """
    def _sink(rec) -> None:
        prev = get()
        put(rec)
        if prev:
            kv, mb = pools()
            n = release_wake_refs(kv, mb, prev, log)
            log("L15-HOSTLOCK superseded record released (%d ref(s)) -- a "
                "second retain in this sleep re-pinned the hold" % n)
    return _sink
