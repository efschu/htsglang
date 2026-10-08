"""EF EVICTION FRONTIER CENSUS + REPAIR (27B rc12o b1, PP0 13:56:43Z, rid pdflip-8-62).

THE DEATH. P log boot_weg2_dkr27breleasedraftbar1w109271344 (732d629b21) 54255-54280:
MAMBA-HOST-RESUME anchor 45238, #988 LOADBACK to 45238, PDFLIP-ARENA-LOAD rows=42216 (the SF
same-pass room had evicted exactly the load's shortfall -- n=9, not printed by the sampler), then
the 512-token extend: "Available full tokens: 149167 (46 + evictable 149121)" and "EVICTION
UNDER-DELIVERED: asked for 512 tokens, the pool received 0", with NO #1421 BACKUP-REFUSED near it
(the six #1421 lines of the boot are all earlier) -- so no leaf was tried and refused: the peel
found NO leaf it could pay.

WHY TWO NUMBERS CAN DISAGREE (read off the code, not the dump -- the dump holds no tree):
  * ``component_evictable_size_[FULL]`` counts every device node whose FULL ``lock_ref`` is 0;
  * the frontier ``evictable_device_leaves`` admits a node only when EVERY component's
    ``lock_ref`` is 0 and no child holds FULL device rows (``_is_device_leaf``), and it is
    maintained incrementally at the sites that touch a node.
  A FULL-unlocked node that carries an aux lock (a mamba anchor/pin: ``lock_ref`` on the MAMBA
  component), or whose leaf membership was never refreshed, is counted as evictable and cannot
  be peeled -- nor can anything above it.
  NF's hypothesis (b) -- the freshly loaded 42216 rows -- is NOT this: the load-back's
  ``inc_lock_ref`` moves them from evictable to protected in ``acquire_component_lock``.

THE RULE (switch ``FLLIPER_EVICT_FRONTIER_REPAIR``, default on; ``0`` = the old peel byte for byte):
when the peel ends short while the tree still reports evictable tokens, ONE full scan
  (1) re-derives leaf membership for every node (``_update_evictable_leaf_sets``) -- a stale
      membership is repaired and the peel runs once more;
  (2) names the rest: ``EVICT-FRONTIER-CENSUS`` with the FULL-unlocked tokens split into
      leaf / aux-locked (per component) / behind a device child, and the first node ids.
Bounded: only on an under-delivery, once per ``evict`` call.

ED (switch ``FLLIPER_EVICTABLE_DELIVERABLE``, default on; ``0`` = the reported count): the
ADMISSION (``PrefillAdder.rem_total_tokens`` on the hybrid-SSM branch and
``common.fundable_extend_tokens`` -> ``chunk_tokens_the_pool_can_fund``) reads
``deliverable_evictable_size()`` = the reported FULL-evictable count minus the tokens behind aux
locks (:func:`blocked_tokens`), so a chunk the pool cannot pay parks (NO_TOKEN / 0 fundable)
instead of OOM-ing in ``alloc_for_extend``. The reported count, the pool statistics and every
other reader stay as they are.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

ENV = "FLLIPER_EVICT_FRONTIER_REPAIR"
#: ED: the admission reads the DELIVERABLE evictable count (see blocked_tokens).
ENV_DELIVERABLE = "FLLIPER_EVICTABLE_DELIVERABLE"
#: cache attribute: nodes whose non-FULL component holds a device lock.
AUX_LOCKED_ATTR = "_ef_aux_locked_nodes"
_SAMPLE = 6


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def deliverable_enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV_DELIVERABLE, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def note_aux_lock(cache, node, locked: bool) -> None:
    """Track nodes whose aux (e.g. MAMBA) component holds a device lock: added
    on the 0->1 transition, dropped on 1->0. Called by the component."""
    s = getattr(cache, AUX_LOCKED_ATTR, None)
    if s is None:
        s = {}
        try:
            setattr(cache, AUX_LOCKED_ATTR, s)
        except Exception:  # noqa: BLE001
            return
    if locked:
        s[id(node)] = node
    else:
        s.pop(id(node), None)
    try:
        cache._ef_aux_version = int(getattr(cache, "_ef_aux_version", 0) or 0) + 1
    except Exception:  # noqa: BLE001
        pass


def blocked_tokens(cache, base_ct) -> int:
    """FULL-unlocked device tokens the peel can never reach while the aux locks
    stand: every aux-locked node whose FULL lock is 0 (``_is_device_leaf``
    refuses it) and its FULL-unlocked ancestors (they keep a device child). A
    walk up stops at the first FULL-locked node -- its ancestors are locked by
    the path lock. Each node counted once."""
    s = getattr(cache, AUX_LOCKED_ATTR, None)
    if not s:
        return 0
    root = cache.root_node
    seen = set()
    total = 0
    for node in list(s.values()):
        cur = node
        while cur is not None and cur is not root and id(cur) not in seen:
            seen.add(id(cur))
            cd = cur.component_data[base_ct]
            if cd.value is None or cd.lock_ref > 0 or getattr(cur, "evicted", False):
                break
            total += len(cd.value)
            cur = cur.parent
    return total


#: cache attribute: True on a rank whose admission verdict is not its own
#: (a Form A worker, H105): it keeps the reported count so its local extend
#: arithmetic stays the pre-ED one; only the deciding rank's vote uses ED.
EXEMPT_ATTR = "_ef_deliverable_exempt"
#: per-cache memo + instrument state
_MEMO_ATTR = "_ef_blocked_memo"
_STATS_ATTR = "_ef_blocked_stats"
_STATS_EVERY_S = 60.0


def _stats(cache):
    st = getattr(cache, _STATS_ATTR, None)
    if st is None:
        st = {"calls": 0, "scans": 0, "scan_us_sum": 0.0, "scan_us_max": 0.0,
              "t_last": 0.0, "blocked": 0, "aux": 0}
        try:
            setattr(cache, _STATS_ATTR, st)
        except Exception:  # noqa: BLE001
            pass
    return st


def blocked_tokens_memo(cache, base_ct) -> int:
    """:func:`blocked_tokens`, recomputed only when an input changed: the aux
    lock set (its version), the FULL evictable and protected counts (every
    lock transition, insert and eviction moves one of them). A node split
    moves none of them and keeps the chain sum. Timed; one throttled line."""
    import time

    st = _stats(cache)
    st["calls"] += 1
    key = (int(getattr(cache, "_ef_aux_version", 0) or 0),
           int(cache.component_evictable_size_.get(base_ct, 0) or 0),
           int(getattr(cache, "component_protected_size_", {}).get(base_ct, 0) or 0))
    memo = getattr(cache, _MEMO_ATTR, None)
    if memo is not None and memo[0] == key:
        return memo[1]
    t0 = time.perf_counter()
    val = blocked_tokens(cache, base_ct)
    us = (time.perf_counter() - t0) * 1e6
    try:
        setattr(cache, _MEMO_ATTR, (key, val))
    except Exception:  # noqa: BLE001
        pass
    st["scans"] += 1
    st["scan_us_sum"] += us
    st["scan_us_max"] = max(st["scan_us_max"], us)
    st["blocked"] = val
    st["aux"] = len(getattr(cache, AUX_LOCKED_ATTR, None) or ())
    now = time.monotonic()
    if now - st["t_last"] >= _STATS_EVERY_S:
        st["t_last"] = now
        logger.info(
            "ED-DELIVERABLE calls=%d scans=%d scan_us_mean=%.1f scan_us_max=%.1f blocked=%d "
            "aux_locked_nodes=%d (the admission's deliverable evictable count: scans only when the "
            "aux lock set or the FULL evictable/protected counts moved; cumulative since boot)",
            st["calls"], st["scans"], st["scan_us_sum"] / max(1, st["scans"]),
            st["scan_us_max"], val, st["aux"])
    return val


def deliverable_evictable(cache, base_ct) -> int:
    reported = int(cache.component_evictable_size_.get(base_ct, 0) or 0)
    if not deliverable_enabled() or getattr(cache, EXEMPT_ATTR, False):
        return reported
    return max(0, reported - blocked_tokens_memo(cache, base_ct))


def census_and_repair(cache, base_ct) -> Dict[str, object]:
    """Scan every node once: repair leaf membership, count why FULL-unlocked
    device tokens are not on the frontier. Returns the census dict."""
    nodes = cache._collect_all_nodes()
    before = len(cache.evictable_device_leaves)
    added = 0
    out = {"nodes": len(nodes), "leaves_before": before, "full_unlocked_tokens": 0,
           "leaf_tokens": 0, "aux_locked_tokens": {}, "device_child_tokens": 0,
           "aux_locked_ids": [], "stale_added": 0}
    for n in nodes:
        if n is cache.root_node or n.evicted:
            continue
        was = n in cache.evictable_device_leaves
        cache._update_evictable_leaf_sets(n)
        if not was and n in cache.evictable_device_leaves:
            added += 1
        cd = n.component_data[base_ct]
        if cd.value is None or cd.lock_ref > 0:
            continue
        toks = len(cd.value)
        out["full_unlocked_tokens"] += toks
        if n in cache.evictable_device_leaves:
            out["leaf_tokens"] += toks
            continue
        locked = [ct for ct, c in enumerate(n.component_data) if getattr(c, "lock_ref", 0) > 0]
        if locked:
            for ct in locked:
                key = str(ct)
                out["aux_locked_tokens"][key] = out["aux_locked_tokens"].get(key, 0) + toks
            if len(out["aux_locked_ids"]) < _SAMPLE:
                out["aux_locked_ids"].append(getattr(n, "id", "?"))
        else:
            out["device_child_tokens"] += toks
    out["stale_added"] = added
    out["leaves_after"] = len(cache.evictable_device_leaves)
    return out


def log_census(c: Dict[str, object], request: int, got_before: int, got_after: int,
               reported: int) -> None:
    logger.warning(
        "EVICT-FRONTIER-CENSUS request=%d delivered_before=%d delivered_after_repair=%d "
        "reported_evictable=%d full_unlocked_tokens=%d on_frontier=%d aux_locked=%s "
        "behind_device_child=%d leaves %d->%d stale_added=%d aux_locked_ids=%s nodes=%d "
        "(EF: the peel ended short while the tree reported evictable tokens; leaf membership "
        "re-derived and the peel retried once; aux_locked keys are component indices)",
        int(request), int(got_before), int(got_after), int(reported),
        int(c["full_unlocked_tokens"]), int(c["leaf_tokens"]), c["aux_locked_tokens"],
        int(c["device_child_tokens"]), int(c["leaves_before"]), int(c["leaves_after"]),
        int(c["stale_added"]), c["aux_locked_ids"], int(c["nodes"]),
    )


# ---------------------------------------------------------------------------
# F1 (nf-next-1006-01): evictable tokens ABOVE the engaged residency cap
# ---------------------------------------------------------------------------
#: allocator attribute the stage cap (``KvRowCap``) is kept on (d_seat_vram.py
#: ``_engage_kv_cap``); the only cap setter on the NF D boot.
STAGE_CAP_ATTR = "_pdflip_kv_stage_cap"
#: cache attribute: the last scan, ``(cap_pages, monotonic_t, value)``.
_ABOVE_MEMO_ATTR = "_ef_cap_above_memo"
#: a repeat scan under the SAME cap is reused for this long (seconds). The
#: scan walks every tree node and reads one device tensor; the number only
#: falls while a shrink is pending (evictions peel the high leaves) and a cap
#: change always rescans, so a reused value is stale on the HIGH side (the
#: admission parks a little longer, never admits more).
_ABOVE_TTL_S = 0.25


def engaged_cap_pages(allocator):
    """The engaged residency cap of ``allocator`` in PAGES (the unit of its
    free-list ids), or None when no cap is engaged (then nothing is above)."""
    cap = getattr(allocator, STAGE_CAP_ATTR, None) if allocator is not None else None
    if cap is None:
        return None
    try:
        if not bool(cap.engaged) or cap.cap is None:
            return None
        return int(cap.cap)
    except Exception:  # noqa: BLE001 - a stand-in without the cap protocol has none
        return None


def above_cap_tokens(cache, base_ct, cap_pages: int, page_size: int) -> int:
    """FULL-evictable device tokens whose slot lies ABOVE page id ``cap_pages``.

    The pool hands out only free ids at or below the cap (``KvRowCap`` pulls
    every freed id above it straight back out of the free list), so a leaf the
    peel frees up there pays the TREE and the POOL nothing. A token slot ``s``
    belongs to page ``s // page_size``; page ids are 1-based and a page is above
    the cap when its id is greater than ``cap_pages`` (``free_tokens_below``,
    d_seat_vram.py, counts ``ids <= cap``).

    One scan over the unlocked FULL nodes (the census' own definition of
    evictable) and ONE device read. Rank-LOCAL by construction: slot ids are the
    rank's own, which is why the caller pins it through the group MIN."""
    import torch

    root = cache.root_node
    vals = []
    for n in cache._collect_all_nodes():
        if n is root or getattr(n, "evicted", False):
            continue
        cd = n.component_data[base_ct]
        if cd.value is None or cd.lock_ref > 0 or len(cd.value) == 0:
            continue
        vals.append(cd.value)
    if not vals:
        return 0
    flat = torch.cat([v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.int64)
                      for v in vals])
    lim = (int(cap_pages) + 1) * max(1, int(page_size))
    return int((flat >= lim).sum().item())


def cap_above_tokens_memo(cache, base_ct, allocator, page_size: int) -> int:
    """:func:`above_cap_tokens` for the engaged cap of ``allocator`` (0 without
    one), rescanned on a cap change or after ``_ABOVE_TTL_S``. Never raises: a
    failed scan reads 0 = the pre-F1 behaviour (the guard then counts what it
    always counted)."""
    import time

    cap = engaged_cap_pages(allocator)
    if cap is None:
        return 0
    now = time.monotonic()
    memo = getattr(cache, _ABOVE_MEMO_ATTR, None)
    if memo is not None and memo[0] == cap and now - memo[1] < _ABOVE_TTL_S:
        return memo[2]
    try:
        val = above_cap_tokens(cache, base_ct, cap, page_size)
    except Exception:  # noqa: BLE001 - an admission input must not raise
        logger.warning("CAP-BLIND-ADMIT scan failed; counted as 0 above the cap", exc_info=True)
        val = 0
    try:
        setattr(cache, _ABOVE_MEMO_ATTR, (cap, now, val))
    except Exception:  # noqa: BLE001
        pass
    return val
