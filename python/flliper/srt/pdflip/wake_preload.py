"""DP-NACHLAUF 02.10.: WAKE-PRELOAD -- the held requests' prefix H2D starts
inside the kv resume RPC (after the restore's re-zero, before the reply)
instead of at the first schedule pass after the hold release.

N5q (9126170083, P>D epoch 8): D's Nachlauf 611 ms = the first pass's
schedule 232 ms (START-LOADING) + the first extend 332 ms waiting for the KV
copy (TP0 43140 rows / 1.41 GB). The copy began only after the reply, the
hold release and the admission; TP0 meanwhile sat in the kv RPC's group
fence (90 ms) waiting for TP1's slower restore.

The 18.09. preload (``_pdflip_preload_hold``, b73d59b4fc) ran mid-legs on a
PER-RANK verdict and died in PrefixLensRankDivergence (xsn377, 1224b327cd).
Here every rank reaches the same point of the same RPC with the same hold;
before anything is issued the probed host extents are voted (element-wise
MIN and MAX over the TP cpu group -- equal or the whole group skips), and
after issuing the issued count is checked the same way (a mismatch stops by
name: ranks never disagree). The H2D starts at once (``start_loading``); the
producer is handed to the first batch as its ``hicache_consumer_index`` when
that batch loads nothing itself, so its forward waits per layer exactly as
for its own load (a newer producer on the same load stream covers it).

Off: the PP/told form (group P -- followers settle PP0's told before any
load), no hierarchical cache, or ``FLLIPER_PDFLIP_WAKE_PRELOAD`` = 0/false/no/off.
On an L15 D kv wake (master on) the preload is not run at the kv resume but
AFTER the L15 act (site=after_l15_act: hold rows reserved / fallback drop
done, no L15 path touches the pools after it); both decisions are taken from
group-uniform terms, never from a rank's own manifest.
Unset = on. Marker: ``PDFLIP-WAKE-PRELOAD``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_WAKE_PRELOAD"
ATTR = "_pdflip_preload_producer"


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def ineligible(sched, l15_hold_aware: bool = False) -> str:
    """'' when the preload may run on this group, else the reason."""
    if not enabled():
        return "switch_off"
    if not getattr(sched, "enable_hierarchical_cache", False):
        return "no_hicache"
    ps = getattr(sched, "ps", None)
    if int(getattr(ps, "pp_size", 1) or 1) != 1:
        return "pp_form"
    try:
        from flliper.srt.managers import pdflip_store_told as _st

        if _st.armed(sched):
            return "told_form"
    except Exception:  # noqa: BLE001 - no told module: not the told form
        pass
    if l15_hold_aware:
        return "l15_hold_aware"
    return ""


def group_min_max(sched, vals: List[int]):
    """Element-wise (MIN, MAX) over the TP cpu group; the local values on a
    single rank / no group."""
    vals = [int(v) for v in vals]
    if not vals:
        return [], []
    try:
        import torch

        tp = int(getattr(getattr(sched, "ps", None), "tp_size", 1) or 1)
        group = getattr(sched, "tp_cpu_group", None)
        if tp > 1 and group is not None and torch.distributed.is_initialized():
            t = torch.tensor(vals + [-v for v in vals], dtype=torch.int64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group)
            out = [int(x) for x in t.tolist()]
            n = len(vals)
            return out[:n], [-x for x in out[n:]]
    except Exception as exc:  # noqa: BLE001
        logger.info("PDFLIP-WAKE-PRELOAD group vote failed (%s: %s) -- nothing issued", type(exc).__name__, exc)
        return None, None
    return list(vals), list(vals)


def _probe_extents(sched, hold) -> List[int]:
    from flliper.srt.managers import schedule_policy as sp

    tree = sched.tree_cache
    exts = []
    for req in hold:
        sp.match_prefix_for_req(tree, req, include_req=True)
        exts.append(int(sp._pp_load_back_extent(req) or 0))
    return exts


def run(updater, l15_hold_aware: bool = False, site: str = "kv_resume") -> int:
    """At the kv resume RPC, after the restore: issue and START the held
    requests' loads. Returns the number of requests issued (group-uniform)."""
    sched = getattr(updater, "scheduler", None)
    if sched is None:
        return 0
    hold = list(getattr(sched, "pdflip_dormant_hold", None) or [])
    if not hold:
        return 0
    why = ineligible(sched, l15_hold_aware)
    if why:
        logger.info("PDFLIP-WAKE-PRELOAD off reason=%s held=%d site=%s", why, len(hold), site)
        return 0
    t0 = time.perf_counter()
    exts = _probe_extents(sched, hold)
    lo, hi = group_min_max(sched, exts)
    if lo is None:
        return 0
    if lo != hi:
        logger.info("PDFLIP-WAKE-PRELOAD skip reason=ranks_disagree held=%d extents_min=%s max=%s "
                    "(nothing issued on any rank; the first pass loads as before)", len(hold), lo, hi)
        return 0
    if not any(lo):
        logger.info("PDFLIP-WAKE-PRELOAD skip reason=no_host_extent held=%d", len(hold))
        return 0
    n = int(updater._pdflip_preload_hold() or 0)
    nlo, nhi = group_min_max(sched, [n])
    if nlo != nhi:
        raise RuntimeError(
            f"PDFLIP-WAKE-PRELOAD RANKS DISAGREE: issued {n} here, group min {nlo[0]} max {nhi[0]} "
            f"after equal extents {lo} -- stopping by name instead of a PrefixLensRankDivergence")
    tree = sched.tree_cache
    producer = -1
    if n:
        producer = int(tree.ready_to_load_host_cache())
        try:
            setattr(tree, ATTR, producer)
        except Exception:  # noqa: BLE001 - a tree without attributes cannot hand it over
            producer = -1
    logger.info("PDFLIP-WAKE-PRELOAD held=%d issued=%d extents=%s producer=%d ms=%.1f site=%s (the H2D runs "
                "from here; the first batch waits on it per layer)", len(hold), n, lo, producer,
                (time.perf_counter() - t0) * 1000.0, site)
    return n


def consumer_index(tree, idx):
    """The first batch after a preload: its own load's producer, or -- when
    it loads nothing -- the preload's, consumed once."""
    pending = getattr(tree, ATTR, None)
    if pending is not None:
        try:
            setattr(tree, ATTR, None)
        except Exception:  # noqa: BLE001
            pass
    if idx is not None and int(idx) >= 0:
        return idx
    if pending is not None and int(pending) >= 0:
        return int(pending)
    return idx
