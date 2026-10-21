"""EVICT-FIT (NF rc27 reboot 1021_023225, W-D of nf-rc29-pphase-1021): the peel demotes only as much of a backed leaf
as the request still needs -- the leaf is split at the boundary and only its TAIL leaves the device.

THE OVER-EVICTION (P.log boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210232_893d09fa3f_1021_023225):
PP0 18254 ``SF LOADBACK-ROOM SAME-PASS rid=pdflip-0-19 kv_tokens=113408 floor=85760 avail=85760 evictable=243648
evicted=165632``; PP1 18600 ``LB-VERDICT ROW-ROOM rid=pdflip-0-19 kv_tokens=113408 avail=85888 evictable=419520 evicted=165632`` (node 18 = 151360, P 18598).
The demand was right: ``pp_slot_fidelity.local_pp_room`` asks ``tree.evict(num_tokens=min(kv_tokens - avail0,
evictable))`` = 27648 (PP1: 27520). The peel paid 14272 from the dropped 0-22 tail, then reached node 9 -- a BACKED
node of 151360 tokens -- and ``_evict_to_host`` demoted it WHOLE (``#1469 EVICT node=9 backuped=True``, P 18252):
165632 evicted for a need of 27648, 6x. Node granularity, not a demand error. 137984 tokens of backed KV left the
device for nothing; every later hit on [18880, 170240) pays a load-back for them.

THE RULE. In the full component's peel, before a backed device leaf is demoted, if it is longer than what the request
still needs by at least one page, it is split (``UnifiedRadixCache._split_node``, the insert path's own split: KV and
host rows sliced, the mamba anchor and every pdflip node attribute stay on the TAIL, which keeps the node's identity)
at ``len - ceil(need / page) * page``; the head stays on the device, the tail is demoted. Only a node that is backed
for its whole length on the host, unlocked (device and host), with no write in flight and complete page hashes is
split; anything else is demoted whole as before.

SCOPE / RANK AGREEMENT. The local-PP floor only (``pp_slot_fidelity.FLOOR_LOCAL_PP_ATTR``: tp group of one, the tree
is this rank's own, node boundaries already differ between the stages; W3 decides on arena SLOTS, never node ids). On a
TP group / Form A (R12 node keys shared across ranks) nothing changes. No wait, no I/O, no timer.

Switch ``FLLIPER_PDFLIP_EVICT_FIT`` (default on; 0 = whole-node demotion as before).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_EVICT_FIT"
_N = {"split": 0}
_LOG_FIRST = 16


def on(env=None) -> bool:
    e = os.environ if env is None else env
    if str(e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        return False  # 27B NACHT 21.10.: the dual layout stays as it was (user 20.10.) -- off under FLLIPER_PDFLIP_DUAL_LAYOUT=1 (set on every rank of both groups: rank-uniform)
    raw = (e.get(ENV, "1") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _full(tree: Any):
    from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.FULL


def fit(tree: Any, node: Any, need: int) -> bool:
    """Split the backed device leaf ``node`` so that only a tail of ``ceil(need/page)*page`` tokens is demoted next.
    ``node`` keeps its identity (it becomes the tail). True = split. Never raises."""
    try:
        if need <= 0 or not on():
            return False
        from flliper.srt.pdflip import pp_slot_fidelity as _sf

        if not _sf.enabled() or not getattr(tree, _sf.FLOOR_LOCAL_PP_ATTR, False):
            return False
        split = getattr(tree, "_split_node", None)
        if not callable(split) or node is getattr(tree, "root_node", None):
            return False
        if getattr(node, "evicted", True) or not getattr(node, "backuped", False):
            return False
        if getattr(node, "id", None) in (getattr(tree, "ongoing_write_through", None) or {}):
            return False
        cds = getattr(node, "component_data", None)
        cd = cds[_full(tree)]
        key = node.key
        n = len(key)
        if cd.value is None or cd.host_value is None or len(cd.value) != n or len(cd.host_value) != n:
            return False
        if int(getattr(cd, "lock_ref", 0) or 0) > 0:
            return False
        if any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in (cds.values() if isinstance(cds, dict) else cds) if c is not None):
            return False   # a load reads these host rows
        page = max(1, int(getattr(tree, "page_size", 1) or 1))
        tail = -(-int(need) // page) * page
        keep = n - tail
        if keep < page or keep % page or n % page:
            return False
        hv = getattr(node, "hash_value", None)
        if getattr(tree, "enable_storage", False) and (not hv or len(hv) * page != n):
            return False
        split(key, node, keep)
        _N["split"] += 1
        k = _N["split"]
        if k <= _LOG_FIRST or (k & (k - 1)) == 0:
            logger.info(
                "EVICT-FIT SPLIT n=%d node=%s tokens=%d need=%d keep=%d tail=%d: a backed leaf longer than the "
                "peel's remaining need -- only its tail leaves the device, the head stays (backed KV is cheap to "
                "keep on the device)", k, getattr(node, "id", "?"), n, int(need), keep, tail)
        return True
    except Exception as exc:  # noqa: BLE001 -- a fit never breaks the peel: whole-node demotion as before
        logger.warning("EVICT-FIT split failed on node %s: %r", getattr(node, "id", "?"), exc)
        return False
