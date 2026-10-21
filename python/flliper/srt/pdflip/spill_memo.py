"""SPILL-MEMO (NF rc27 reboot 1021_023225, W-C of nf-rc29-pphase-1021): a W3 room spill of a PP group that released
nothing is not walked again on the scheduler thread until the arena or the tree changed.

THE COST (P.log boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210232_893d09fa3f_1021_023225):
  PP1 18488/18489 02:37:20  RETAIN-PUBLISH rid=pdflip-0-17 {'unbacked': 21, 'issued': 0, 'refused': 21,
                            'arena_full_skipped': 20, 'memo_skipped': 8, 'refused_by': {'arena_claim': 1},
                            'issue_ms': 497.6 ...}   PDFLIP-PUBLISH-REQ ... retain_ms=500.9
  PP1 18507/18508 02:37:21  RETAIN-PUBLISH rid=pdflip-0-20 {... 'refused': 21, 'arena_full_skipped': 20,
                            'stopped': 'budget', 'issue_ms': 826.5 ...}   retain_ms=829.5
  PP2 18561/18562 02:37:21  rid=pdflip-0-17 issue_ms=524.9  retain_ms=529.0
  PP2 18605/18606 02:37:22  rid=pdflip-0-20 issue_ms=823.2  retain_ms=826.9
  PP0 17755/17756 02:37:12  rid=pdflip-0-22 issue_ms=100.3  retain_ms=104.6 (PP0: the 0.5 s empty backoff)
Every node but one was a SKIP (SWEEP-FULL-ARENA / F5b memo: no claim made) -- and still ~25-40 ms each. The skip's
room question ``UnifiedRadixCache._pdflip_full_arena_refuses`` first runs ``_w3_arena_spill`` (a release would answer
False), and on a follower the W3 room-set spill (``w3_room_set.spill`` -> ``_apply_new`` + ``_own_only``) builds the
whole candidate walk (``_collect_all_nodes``, the keep-payable walk, the arena header read for the order) EVERY time:
PP0's 0.5 s empty backoff (``W3_SPILL_EMPTY_BACKOFF_S``) is decider-only (``backoff_ok``). 20 identical walks per
retain publish, all answering 0, on PP1/PP2's scheduler thread between forwards (the ``#PGAP process 1361/1397 ms``).

THE RULE. After a W3 room-set spill that released 0 pages and is not UNDECIDED (PP0 waiting for an acknowledgement /
a follower's transient debt -- those resolve without an arena change and are never memoised), the state it saw is
remembered: arena (free slots, COMPLETE slots, referenced COMPLETE slots, sum of references), the W3 set generation
this rank applied/published, this rank's debts, the tree (evictable/protected tokens, device/host leaf counts), and
(NF-RC31 B) the host-lock release counter and the number of write-throughs in flight. The
next spill with the SAME state returns 0 at once (no walk) -- its answer is provably the same. Any change of a term
(a slot freed or unreferenced anywhere in the group, a new set, a lock released, a node inserted or evicted) walks
again. No timer, no wait, no file I/O (the arena header is shared memory, the census is a numpy read).

RANK AGREEMENT. Nothing new is decided: the skipped walk is this rank's own (a follower's own-only release, PP0's
decision from the same inputs); set application keeps its own paths (``tick`` and the claim's ``apply_if_new``).

Switch ``FLLIPER_PDFLIP_SPILL_STATE_MEMO`` (default on; 0 = every spill walks as before).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_SPILL_STATE_MEMO"
ATTR = "_w3_spill_state_memo"
#: NF-RC31 B (rc30 review Folgefix B): a per-cache counter ``UnifiedRadixCache.dec_host_lock_ref`` raises on every host
#: lock release -- a released host lock frees a node the last walk kept (``host_locked``) without any other term moving
HOST_LOCK_GEN = "_pdflip_host_lock_release_gen"
#: per-cache counters, read into the PUBLISH-SWEEP / RETAIN-PUBLISH stats
CALLS, SKIPS, MS = "_w3_spill_calls", "_w3_spill_memo_skips", "_w3_spill_ms"
_N = {"skip": 0}


def on(env=None) -> bool:
    e = os.environ if env is None else env
    if str(e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        return False  # 27B NACHT 21.10.: the dual layout stays as it was (user 20.10.) -- off under FLLIPER_PDFLIP_DUAL_LAYOUT=1 (set on every rank of both groups: rank-uniform)
    raw = (e.get(ENV, "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def state(cache: Any, pool: Any) -> Optional[Tuple]:
    """The terms a W3 room-set spill's answer depends on; None = unreadable (no memo)."""
    try:
        from flliper.srt.mem_cache.pool_host.arena_pool import _arena_free_slots

        arena = getattr(pool, "arena", None)
        if arena is None:
            return None
        referenced, refs, complete = arena.ref_census()
        d = cache.__dict__
        try:
            from flliper.srt.pdflip import w3_room_set as _w3rs

            debts = len(_w3rs._debts(cache))
        except Exception:  # noqa: BLE001
            debts = -1
        from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        ct = ComponentType.FULL
        return (
            int(_arena_free_slots(arena)), int(complete), int(referenced), int(refs),
            tuple(d.get("_w3_room_seen", (0, 0)) or (0, 0)), int(d.get("_w3_room_gen", 0) or 0),
            int(d.get("_w3_room_pub_gen", 0) or 0), debts,
            int(cache.component_evictable_size_.get(ct, 0) or 0),
            int(cache.component_protected_size_.get(ct, 0) or 0),
            len(getattr(cache, "evictable_device_leaves", ()) or ()),
            len(getattr(cache, "evictable_host_leaves", ()) or ()),
            # NF-RC31 B: a host lock released / a write-through retired keeps a node no longer (``host_locked`` /
            # ``write_pending`` in the walk) although no arena or tree term above moved -- both forget the memo
            int(d.get(HOST_LOCK_GEN, 0) or 0),
            len(getattr(cache, "ongoing_write_through", None) or ()),
        )
    except Exception:  # noqa: BLE001 -- no reading: walk as before
        return None


def unchanged(cache: Any, pool: Any) -> bool:
    """True = the last spill released nothing (decided) and the state is the same: skip the walk."""
    if not on():
        return False
    memo = cache.__dict__.get(ATTR)
    if memo is None:
        return False
    cur = state(cache, pool)
    if cur is None or cur != memo:
        cache.__dict__[ATTR] = None
        return False
    cache.__dict__[SKIPS] = int(cache.__dict__.get(SKIPS, 0) or 0) + 1
    _N["skip"] += 1
    k = _N["skip"]
    if k <= 8 or (k & (k - 1)) == 0:
        logger.info("SPILL-MEMO SKIP n=%d pp_rank=%s: the W3 room spill released nothing at this exact arena/tree "
                    "state -- not walked again until a term changes", k, getattr(cache, "pp_rank", "?"))
    return True


def note(cache: Any, pool: Any, released: int, undecided: bool, t0: float) -> None:
    """After a walked spill: remember the state when it released nothing and was decided; else forget."""
    d = cache.__dict__
    d[CALLS] = int(d.get(CALLS, 0) or 0) + 1
    d[MS] = float(d.get(MS, 0.0) or 0.0) + (time.perf_counter() - t0) * 1000.0
    if not on():
        return
    d[ATTR] = state(cache, pool) if (int(released) == 0 and not undecided) else None


def counters(cache: Any) -> Tuple[int, int, float]:
    d = cache.__dict__
    return int(d.get(CALLS, 0) or 0), int(d.get(SKIPS, 0) or 0), float(d.get(MS, 0.0) or 0.0)
