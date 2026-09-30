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
import time

logger = logging.getLogger(__name__)

#: the dormant hold intake looked the request up only; its read is the wake's
DEFER_ATTR = "_weg2_248_read_at_wake"
#: the wake issued the deferred read (the settle release cleans up after it)
ISSUED_ATTR = "_weg2_248_read_issued"
#: #248f: the read waits for arena room (older hold reads of this wake hold it)
CAPWAIT_ATTR = "_weg2_248f_capacity_wait"
CAPWAIT_MARK = "#248f WAKE-READ CAPACITY-WAIT"
CAPISSUE_MARK = "#248f WAKE-READ CAPACITY-ISSUE"
#: #248f: the pages an issued hold read asked for, and the wake it ran in --
#: a released request keeps its arena references until its admission loads
#: it (``release_loaded_host``), so it counts while it waits in the queue.
PAGES_ATTR = "_weg2_248f_pages"
WAKE_ATTR = "_weg2_248f_wake"


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


def arena_gate_on() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_ENABLE_WAKE_READ_ARENA_GATE.get())
    except Exception:  # noqa: BLE001
        return False


def _arena_page_tokens(sched) -> int:
    """Tokens per KV arena page (the controller's page: one arena slot)."""
    cc = getattr(getattr(sched, "tree_cache", None), "cache_controller", None)
    for v in (getattr(cc, "page_size", None), getattr(sched, "page_size", None)):
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except (TypeError, ValueError):
            continue
    return 1


def arena_capacity(sched):
    """#248f: the KV arena's slot count (one arena file shared by every rank,
    so the same number everywhere), or None when no arena is bound here --
    then nothing is gated."""
    cc = getattr(getattr(sched, "tree_cache", None), "cache_controller", None)
    pool = getattr(cc, "mem_pool_host", None)
    arena = getattr(pool, "arena", None)
    try:
        slots = int(getattr(arena, "slots", 0) or 0)
    except (TypeError, ValueError):
        slots = 0
    return slots if slots > 0 else None


def read_pages(sched, req) -> int:
    """#248f: the arena pages a hold read of ``req`` asks for -- its tokens
    (prompt + output) in arena pages. Request fields only: the same number on
    every rank."""
    n = len(getattr(req, "origin_input_ids", None) or ()) + len(getattr(req, "output_ids", None) or ())
    page = _arena_page_tokens(sched)
    return -(-int(n) // page) if n > 0 else 0


def capacity_waiting(req) -> bool:
    return bool(getattr(req, CAPWAIT_ATTR, False)) and deferred(req)


def _wake_key(sched):
    """#248f: the wake a hold read belongs to -- ``_weg2_wake_seq`` counts up at
    the release (``_weg2_release_dormant_hold``); a read issued before it (F22's
    early read, still dormant) belongs to the wake about to be counted."""
    seq = getattr(sched, "_weg2_wake_seq", None)
    if seq is None:
        return None
    return int(seq) + (1 if getattr(sched, "weg2_dormant", False) else 0)


def _in_flight_pages(sched, reqs) -> int:
    """Pages of this wake's hold reads that still hold the arena: issued and
    still in the hold / settle, or released to the waiting queue and not yet
    admitted (the admission's load-back releases the references). Replicated:
    the settle, the hold and the queue's hold-released requests are the
    group's, the pages are request lengths."""
    seen, total = set(), 0
    for r in reqs or ():
        if issued(r) and not capacity_waiting(r):
            seen.add(id(r))
            total += read_pages(sched, r)
    wake = _wake_key(sched)
    for r in list(getattr(sched, "waiting_queue", None) or ()):
        if id(r) in seen:
            continue
        pages = int(getattr(r, PAGES_ATTR, 0) or 0)
        if pages > 0 and getattr(r, WAKE_ATTR, None) == wake:
            seen.add(id(r))
            total += pages
    return total


def issue_deferred_reads(sched, hold) -> list:
    """``#1443 DORMANT-RELEASE``: issue the store read of every held request
    whose intake only looked it up, in hold order. Every rank runs this at
    the same point of the resume with the same hold (the intake order is
    the group's), so the reads' collectives line up. Returns the requests
    whose read was issued -- the release parks them in the #1471 settle
    until the read is complete.

    #248f (30.09., NF y4b ep18): only as many as fit in the KV arena
    TOGETHER, oldest first. y4b 03:51:37: weg2-0-4 (1412 pages), weg2-14-27
    (1728) and weg2-16-29 (3841) = 6981 pages against 6485 slots -- the L3
    fills of 16-29 found the arena full and ``#248e ORDERED-EVICT site=l3fill``
    took 449 of 16-29's OWN kept pages (the read-order victim is the last-read
    rid), 33 fills were refused, and 16-29 needed 4 re-reads (held 4.5 s after
    the wake). Now the first read that would overrun the arena waits by name
    (``#248f WAKE-READ CAPACITY-WAIT``) and every younger one with it (no
    overtaking: arrival order); the settle issues them as the older reads
    leave it (:func:`issue_capacity_waiters`). The first read is always issued
    (a lone read larger than the arena reads as before). Capacity = the slot
    count, no reserve; nothing here runs in a decode or a flip leg. Every
    input is replicated (request lengths, hold order, the shared arena's
    geometry), so every rank decides alike."""
    reqs = [r for r in list(hold or ()) if deferred(r)]
    if not reqs:
        return []
    cap = arena_capacity(sched) if arena_gate_on() else None
    if cap is None:
        return _issue(sched, reqs)
    in_flight = _in_flight_pages(sched, hold)
    go, wait = [], []
    for req in reqs:
        need = read_pages(sched, req)
        if wait or (in_flight > 0 and in_flight + need > cap):
            setattr(req, CAPWAIT_ATTR, True)
            wait.append(req)
            continue
        setattr(req, CAPWAIT_ATTR, False)
        in_flight += need
        go.append(req)
    out = _issue(sched, go)
    if wait:
        logger.info("%s n=%d waiting=%s in_flight_pages=%d arena_slots=%d (the older hold reads of "
                    "this wake hold the arena: these wait parked in the #1471 settle, oldest "
                    "first, instead of reading short and evicting their own kept pages)",
                    CAPWAIT_MARK, len(wait), [(str(r.rid)[:12], read_pages(sched, r)) for r in wait],
                    in_flight, cap)
    return out


def issue_capacity_waiters(sched, settle) -> list:
    """#248f: from the #1471 settle tick (every rank, every pass, same list):
    issue the waiting hold reads, oldest first, as far as the arena holds
    them beside the issued reads still in the settle or released and not yet
    admitted (queued: their load-back has not freed the arena yet). Returns those issued;
    their settle clock restarts now (waiting is not reading)."""
    waiters = [r for r in list(settle or ()) if capacity_waiting(r)]
    if not waiters:
        return []
    cap = arena_capacity(sched) if arena_gate_on() else None
    in_flight = _in_flight_pages(sched, settle)
    go = []
    for req in waiters:
        need = read_pages(sched, req)
        if cap is not None and in_flight > 0 and in_flight + need > cap:
            break
        in_flight += need
        go.append(req)
    if not go:
        return []
    for req in go:
        setattr(req, CAPWAIT_ATTR, False)
    out = _issue(sched, go)
    now = time.monotonic()
    for req in go:
        req._1471_since = now
    logger.info("%s n=%d rids=%s in_flight_pages=%d arena_slots=%s still_waiting=%d (an older hold read "
                "was admitted and freed its arena room: the next ones in arrival order read now)",
                CAPISSUE_MARK, len(go), [(str(r.rid)[:12], read_pages(sched, r)) for r in go],
                in_flight, cap, len(waiters) - len(go))
    return out


def _issue(sched, reqs) -> list:
    """The #248 store read of each request, in the given order (unchanged)."""
    from sglang.srt.weg2 import settle_writer as _sw

    out, refused = [], []
    now = time.monotonic()
    for req in list(reqs or ()):
        if not deferred(req):
            continue
        setattr(req, DEFER_ATTR, False)
        verdict = sched._prefetch_kvcache(req)
        req._969c_verdict = verdict
        # NW (30.09.): a refusal of the host budget is "not read yet" -- the #1471
        # settle keeps it parked and re-reads once the budget has room.
        if _sw.note_read_verdict(req, verdict, now):
            refused.append((req, verdict))
        apply = getattr(sched, "_apply_prefetch_deferral", None)
        if apply is not None:
            apply(req, verdict, site="wake-248")
        setattr(req, ISSUED_ATTR, True)
        setattr(req, PAGES_ATTR, read_pages(sched, req))
        setattr(req, WAKE_ATTR, _wake_key(sched))
        out.append(req)
    if out:
        logger.info("#248 WAKE-READ issued=%d %s (the hold read runs now: reference and pin at the wake, "
                    "the device load at admission)", len(out), [str(r.rid)[:12] for r in out])
    if refused:
        logger.warning("#1471b WAKE-READ BUDGET-REFUSED n=%d %s -- the host budget had no room for "
                       "these reads (the earlier hold reads hold it); they stay parked in the settle "
                       "and are re-read once it frees, never decided 'no writer' on a read that did "
                       "not run", len(refused), [(str(r.rid)[:12], v) for r, v in refused])
    return out


def note_hold_order(hold) -> None:
    """#248e: every rid of the dormant hold, in hold order, to the keep order
    (group D, the switch on; a no-op otherwise). Called right before each
    :func:`issue_deferred_reads` (the release's and the F22 early one): the
    hold order is also the order the kept pages leave the arena -- the L3
    fills of this wake take the last-read rid's tail first, never the head
    of an earlier read (``handoff_pending.victim_order``)."""
    if not enabled() or not _group_d():
        return
    try:
        from sglang.srt.weg2 import handoff_pending as _hp

        _hp.note_read_order([str(getattr(r, "rid", "") or "") for r in (hold or ())])
    except Exception:  # noqa: BLE001 - the order is an improvement, never a wall
        logger.warning("#248e hold order not noted", exc_info=True)


def early_enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_ENABLE_WAKE_READ_EARLY.get())
    except Exception:  # noqa: BLE001
        return False


def issued(req) -> bool:
    return bool(getattr(req, ISSUED_ATTR, False))


def issue_reads_at_wake_begin(sched) -> list:
    """F22 (29.09.): the deferred hold reads at the START of the weight legs.

    MEASURED (marker audit x178 / z30w-park / z30x2-kvdemand): the #248 read
    issued at ``#1443 DORMANT-RELEASE`` -- after the legs and the kv resume --
    parks every held request in the #1471 settle for ``held_after_wake_s``
    median 0.60 s (z30w) / 0.35 s (kvdemand); x178 read during the flip (0).
    The read is host-side (aux threads, ``WEG2-READ-STAGES ... no H2D here``;
    the device load is the admission's), so it can run beside the ~1.5 s of
    weight legs. Same call as the release's (:func:`issue_deferred_reads`, in
    hold order, at the same point of the same RPC on every rank); the release
    then finds nothing deferred and its settle verdict sees complete reads.
    Off (:func:`early_enabled`) = nothing here, the release issues as before."""
    if not early_enabled() or not enabled() or not _group_d():
        return []
    hold = getattr(sched, "weg2_dormant_hold", None) or []
    if not hold:
        return []
    note_hold_order(hold)
    out = issue_deferred_reads(sched, hold)
    if out:
        logger.info("#248 WAKE-READ-EARLY issued=%d at the weight legs' start (F22: the read runs "
                    "beside the legs, the settle finds it complete)", len(out))
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

    The node's WHOLE host life goes, KV and aux (mamba anchor, indexer)
    together: the tree invariant is "aux data requires Full data" on each
    layer (``UnifiedRadixCache.sanity_check``), so an aux host value may not
    outlive the KV host row. rc12x (dkrnfh91dprsabar1dauer09272253, 22:59:01,
    rid weg2-0-4): this function dropped the KV host rows only, node 28 kept
    its mamba host anchor, and the idle invariant check stopped every D rank
    ('node 28 mamba host present but Full.host_value=None'). A node whose aux
    host value has NO device copy is KEPT whole (dropping the KV row would
    break the invariant, dropping the anchor would lose the only state).
    Never a node without its device value, never under a host lock (another
    load of it in flight). A Form A worker does nothing itself -- it follows
    TP0's verdict (STATE carries kv_host and anchor_host). Returns the nodes
    released."""
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
    aux = [c for c in tree._components_tuple if c.component_type != BASE_COMPONENT_TYPE]
    root = tree.root_node
    n, rows, kept = 0, 0, 0
    while node is not None and node is not root:
        cd = node.component_data[BASE_COMPONENT_TYPE]
        if (cd.host_value is not None and cd.value is not None and not getattr(node, "evicted", False)
                and not any(int(getattr(c, "host_lock_ref", 0) or 0) > 0 for c in node.component_data)):
            aux_host = [c for c in aux if node.component_data[c.component_type].host_value is not None]
            if any(node.component_data[c.component_type].value is None for c in aux_host):
                kept += 1  # an aux state lives on the host only: the node keeps its host life
            else:
                for c in aux_host:  # aux first: never an aux host row without the KV row
                    tree._evict_component_and_detach_lru(node, c, target=EvictLayer.HOST, tracker=None)
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
            logger.info("#248 LOADED-HOST-RELEASE nodes=%d rows=%d kept_host_only_aux=%d (the span is on "
                        "the device; KV and anchor host rows go together, its arena pages stay COMPLETE, "
                        "unreferenced%s) n=%d", n, rows, kept,
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
