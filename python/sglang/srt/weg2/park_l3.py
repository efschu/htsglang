"""#248: a request that does not run on D holds no arena reference over a flip.

THE METAL (rc12s dkrnfh91dprbar1dauer09271719, D-TP0 17:32:40,
tmp/r989/befund_248_park_l2_pinnt.md): ``ARENA-REF-HOLDERS n=7 ... tree=1231
tree_in_use=3982 sum=5213 own_held=5213`` -- D held 5213 of the 5461 KV arena
slots by reference while it SLEPT. Nothing ran. The references belonged to
the dormant-hold read (#1443/#1455: "the arena lookup, ref and pin run in the
prefetch executor for every held request WHILE the flip runs") of 2 parked
requests (1292 pages) and 3 requests that arrived in the sleep (3921 pages).
P's claims found no free slot (17:32:21-58, again 17:35:18-25), BACKUP-REFUSED
arena_claim, a parent_unbacked cascade, and four empty 200s (W50 re-route
impossible).

A reference held across a flip has no budget -- the class #243 names. #243 made
the P hand-off an eviction ORDER; this makes D's parked and held requests one
too:

* **The hold reads at the wake** (:func:`defer_hold_read`). At the dormant
  hold intake the request is only looked up -- P's hand-off chain resolved
  onto the request, so the wake's read asks the store for exactly P's pages
  -- and nothing is referenced or pinned. The read (reference, pin, the host
  tree) is issued at ``#1443 DORMANT-RELEASE`` for every held request in hold
  order, group-uniform (:func:`issue_deferred_reads`); the #1471 post-wake
  settle releases each once its read is complete, the device load follows at
  admission (H105).
* **The park is kept by order** (:func:`mark_parked`): every request the flip
  park holds gets a ``park`` mark with the chain of the span it retained (the
  tree's page keys, what the forced write-through stored). A P hand-off in the
  hold already has its #243 ``pending`` mark.
* The marks go when D takes the rid (the read's group-uniform termination),
  at its end or abort (``handoff_pending.consume``), or expire (#243 bound).

The pages behind the marks get an L3 copy in the background (``park_demote``);
a claim frees a kept page with a copy without I/O, and names the rest.

Switch ``SGLANG_WEG2_ENABLE_PARK_L3`` (default on); off = the pre-#248 hold
read byte for byte. Every function is a no-op off group D.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: the dormant hold intake looked the request up only; its read is the wake's
DEFER_ATTR = "_weg2_248_read_at_wake"
#: the wake issued the deferred read (the settle release cleans up after it)
ISSUED_ATTR = "_weg2_248_read_issued"


def enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_ENABLE_PARK_L3.get())
    except Exception:  # noqa: BLE001
        return False


def _group_d() -> bool:
    return (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "D"


def defer_hold_read(sched, req) -> bool:
    """The dormant hold intake of ``req``: True = look up only (no store
    read now, no reference, no pin); the read is issued at the wake.

    The lookup resolves P's hand-off chain onto the request (the wake removes
    the hand-off file of every held rid, and D's own hashes of P's prompt
    match P's keys for the first page only -- xsn328), so the wake's read
    registers with exactly P's keys."""
    if not enabled() or not _group_d():
        return False
    rid = getattr(req, "rid", None)
    chain = None
    if isinstance(rid, str) and rid.startswith("weg2-"):
        try:
            from sglang.srt.weg2 import handoff as _ho
            from sglang.srt.weg2.handoff_keys import resolve_chain

            chain = resolve_chain(req, _ho.read)
        except Exception:  # noqa: BLE001 - the chain is an accelerator; own hashes otherwise
            chain = None
    setattr(req, DEFER_ATTR, True)
    n = getattr(sched, "_248_defer_n", 0) + 1
    sched._248_defer_n = n
    if n <= 8 or n % 64 == 0:
        logger.info("#248 HOLD-LOOKUP rid=%s chain=%s (no reference, no pin in the sleep: the read runs "
                    "at the wake) n=%d", str(rid)[:12], len(chain) if chain else None, n)
    return True


def deferred(req) -> bool:
    return bool(getattr(req, DEFER_ATTR, False))


def issue_deferred_reads(sched, hold) -> list:
    """``#1443 DORMANT-RELEASE``: issue the store read of every held request
    whose intake only looked it up, in hold order. Every rank runs this at
    the same point of the resume with the same hold (the intake order is
    the group's), so the reads' collectives line up. Returns the requests
    whose read was issued -- the release parks them in the #1471 settle
    until the read is complete."""
    out = []
    for req in list(hold or ()):
        if not deferred(req):
            continue
        setattr(req, DEFER_ATTR, False)
        verdict = sched._prefetch_kvcache(req)
        req._969c_verdict = verdict
        apply = getattr(sched, "_apply_prefetch_deferral", None)
        if apply is not None:
            apply(req, verdict, site="wake-248")
        setattr(req, ISSUED_ATTR, True)
        out.append(req)
    if out:
        logger.info("#248 WAKE-READ issued=%d %s (the hold read runs now: reference and pin at the wake, "
                    "the device load at admission)", len(out), [str(r.rid)[:12] for r in out])
    return out


def after_release(reqs) -> None:
    """A request whose wake read completed leaves the settle: drop the hand-off
    keys the read registered with (what the wake does for a hold read)."""
    try:
        from sglang.srt.managers import cache_controller as _cc
        from sglang.srt.weg2 import handoff as _ho
    except Exception:  # noqa: BLE001
        return
    for req in reqs or ():
        if not getattr(req, ISSUED_ATTR, False):
            continue
        setattr(req, ISSUED_ATTR, False)
        rid = getattr(req, "rid", None)
        _cc.WEG2_HANDOFF_PAGE_KEYS.pop(rid, None)
        _cc.WEG2_HANDOFF_OFF.pop(rid, None)
        try:
            p = _ho.path(str(rid)) if rid else ""
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def release_loaded_host(tree, node) -> int:
    """#248 + #249: a load-back finished -- the span from ``node`` up is on
    the device. Its KV host rows held arena references (rc12s 17:33:41: a
    woken D kept tree=5213 until the next reset, and P's claims found no
    slot); they go now. The page stays COMPLETE in the arena (unreferenced,
    so a claim may take it -- or a later device eviction attaches to it
    again by stem), and TP0's R12 STATE verdict drops the Form A workers'
    byteless mirror rows with it (rc12t: the workers' fixed 353573-row pool
    ratcheted to "host_pool_shortfall" before any reset).

    Only the KV (base) component: a mamba anchor may exist on the host alone.
    Never a node without its device value, never under a host lock (another
    load of it in flight). A Form A worker does nothing itself -- it follows
    TP0's verdict. Returns the nodes released."""
    if not enabled() or not _group_d() or node is None:
        return 0
    from sglang.srt.mem_cache import form_a_host_shadow as _r12
    from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, EvictLayer

    role = _r12.role()
    if role == "worker":
        return 0
    pools = tree._weg2_arena_pools()
    if BASE_COMPONENT_TYPE not in pools:
        return 0  # no arena KV host pool: the rows are no reference
    comp = next((c for c in tree._components_tuple if c.component_type == BASE_COMPONENT_TYPE), None)
    if comp is None:
        return 0
    root = tree.root_node
    n, rows = 0, 0
    while node is not None and node is not root:
        cd = node.component_data[BASE_COMPONENT_TYPE]
        if (cd.host_value is not None and cd.value is not None and not getattr(node, "evicted", False)
                and not any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data)):
            _, hf = tree._evict_component_and_detach_lru(node, comp, target=EvictLayer.HOST, tracker=None)
            tree.evictable_host_leaves.discard(node)
            n += 1
            rows += int(hf or 0)
            if role == "host":
                _r12.record_state(tree, node, why="248-loaded")
        node = node.parent
    if n:
        k = getattr(tree, "_248_release_n", 0) + 1
        tree._248_release_n = k
        if k <= 8 or k % 64 == 0:
            logger.info("#248 LOADED-HOST-RELEASE nodes=%d rows=%d (the span is on the device; its arena "
                        "pages stay COMPLETE, unreferenced%s) n=%d", n, rows,
                        "; STATE verdict to the Form A workers" if role == "host" else "", k)
    return n


def chain_of(tree, req) -> list:
    """The page keys of the span ``req`` holds in ``tree`` (the path the
    retract inserted; the host nodes a queued request's read left): the
    tree's own keys, i.e. the stems the write-through stored."""
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    ids = list(getattr(req, "origin_input_ids", None) or []) + list(getattr(req, "output_ids", None) or [])
    if not ids:
        return []
    import array

    key = RadixKey(array.array("q", ids), getattr(req, "extra_key", None),
                   is_bigram=bool(getattr(tree, "is_eagle", False)))
    page = int(getattr(tree, "page_size", 1) or 1)
    if page > 1 and hasattr(key, "page_aligned"):
        key = key.page_aligned(page)
    mr = tree.match_prefix(MatchPrefixParams(key=key))
    node = getattr(mr, "last_host_node", None) or getattr(mr, "last_device_node", None)
    root = getattr(tree, "root_node", None)
    parts = []
    while node is not None and node is not root:
        hv = getattr(node, "hash_value", None)
        if not hv:
            return []  # a node without keys: nothing honest to keep
        parts.append(list(hv))
        node = node.parent
    return [k for part in reversed(parts) for k in part]


def mark_parked(sched, reqs) -> int:
    """The flip park (``park_running``, after the retraction's insert): keep
    every parked request's span by ORDER until D takes it again. Written by
    the attention rank 0 (one writer; every rank would write the same)."""
    if not enabled() or not _group_d():
        return 0
    if int(getattr(sched, "tp_rank", 0) or 0) != 0:
        return 0
    from sglang.srt.weg2 import handoff_pending as _hp

    tree = getattr(sched, "tree_cache", None)
    page = int(getattr(tree, "page_size", 1) or 1)
    n = 0
    for req in reqs or ():
        rid = str(getattr(req, "rid", "") or "")
        if not rid:
            continue
        try:
            chain = chain_of(tree, req) if tree is not None else []
        except Exception:  # noqa: BLE001 - an unreadable span keeps nothing (named)
            logger.warning("#248 PARK-MARK rid=%s chain unreadable", rid[:12], exc_info=True)
            chain = []
        if not chain:
            from sglang.srt.weg2.handoff_keys import CHAIN_ATTR

            chain = list(getattr(req, CHAIN_ATTR, None) or [])
        if _hp.mark_park(rid, chain, page):
            n += 1
            logger.info("#248 PARK-MARK rid=%s pages=%d (kept by order, no reference over the flip)",
                        rid[:12], len(chain))
    return n
