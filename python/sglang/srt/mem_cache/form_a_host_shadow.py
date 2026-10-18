"""R12 (fLLiper release table row 12): on a Form A D group every rank keeps
the SAME host and anchor entries -- the root of the rank-divergence class
rc9i/rc9k/rc9l/rc9m/rc9o.

THE ROOT. NF D runs Form A: TP0 holds attention, GDN/mamba state, KV and the
arena host pools (``ArenaMHAHostPool``, ``ArenaMambaPoolHost``); TP1/TP2 are
expert workers with 0-byte plain host pools (``MHATokenToKVPoolHost 0.00 GB``,
``MambaPoolHost 0.00 GB``) and a null storage tier. Every rank decided the
host life of a node LOCALLY:

* store ack (``UnifiedRadixCache._drain_storage_control_queues_impl`` ->
  ``_drain_backup``): TP0 rebinds the rows to the arena and keeps them
  (``_weg2_rebind_host_to_arena``), a worker frees them as transit
  (``_weg2_release_chain_piece_host``, HOST layer of EVERY component, the
  mamba anchor included) -- rc9m: ``WEG2 PUBLISH-CHAIN host released`` on
  TP1/TP2 only;
* load-back end (``loading_check``): the worker frees again (no arena_read);
* on the next device eviction TP0's node survives as a host node with its
  anchor (``#1469 EVICT ... backuped=True host=True``), the worker's dies
  (``backuped=False host=False``) -> anchors at different depths on
  different ranks -> every verdict built on the tree (usable match, #928,
  prefetch span, too_short) can split. H93, H96, H97, H98, H99 each close
  one of those VERDICTS; this module removes the difference in the TREES.

THE POLICY (variant c + a, no new collective):

(c) A Form A worker keeps its host entries as byteless bookkeeping -- the
    rows carry 0 bytes, keeping them costs nothing but row capacity. It
    never frees a row as transit and never evicts host on its own; TP0 is
    the one rank whose host life has meaning.
(a) TP0's OWN host decisions reach the workers over a wire that already
    exists: the TP request broadcast (``request_receiver.recv_requests`` ->
    ``broadcast_pyobj`` over ``tp_cpu_group``, the ``tp<-reqs`` site) -- the
    same stream the #969 section W3 flip decision rides. TP0 appends one
    :class:`FormAHostVerdict` at the origin; right after the broadcast
    every rank takes it out of the list and applies it, BEFORE any request
    of that pass is processed, so no admission, intake or vote of the pass
    sees a difference. Events are keyed by content (the node's last page
    hash + its end depth), never by the rank-local node id.

    * TRANSIT: a store ack whose arena rebind FAILED on TP0 -- TP0 used to
      free the rows there and then; now it defers, and every rank (TP0
      included) runs the same ``_weg2_release_chain_piece_host`` at the
      next broadcast.
    * STATE: a host drop TP0 had to take at once (H19 mamba-arena
      displacement, its own ``evict_host``, a refused backup): TP0 sends the
      node's state (exists, KV host, anchor host); a worker reconciles to
      it (drops what TP0 no longer holds). These happen at insert/backup
      time, after the pass's admission and before the next broadcast.

(b) Capacity from replicated inputs: TP0 addresses ``staging + arena``
    anchors, a worker's mamba pool had the synced 6 slots only. A byteless
    worker pool gets ``2 x SGLANG_HICACHE_ARENA_MAMBA_SLOTS`` extra rows (the
    arena's own slot count, the same environment on every rank, twice for
    drops in flight). The KV pool of a worker already covers TP0's id space
    (fnFL2x62 ``_carrier_capacity_bid``).

(d) #239 S4b part 4 -- THE TOKEN CUT (``kv=qsa_forma_dcp``). A worker that
    owns token rows is no longer byteless: its rows ARE the page's bytes, and
    TP0 (share 0 in the target form 0/48/16) writes nothing. TP0's own store
    ack is then no evidence that the page is in the arena, and a TRANSIT
    decided on it frees rows a worker is still writing from (its store write
    holds a host lock, so the worker's release was a silent no-op -- the
    trees parted: TP0 without host rows, the worker with). Under the cut:

    * a worker never rebinds at its own ack -- it keeps its rows until TP0's
      verdict (the rows carry the bytes of its store write);
    * TP0 rebinds only a COMPLETE page (every owner wrote: the shared arena
      header, ``find_slots`` state 2 -- no collective) and sends REBIND, on
      which every worker rebinds the same node; an incomplete page at TP0's
      ack waits (``REBIND_WAIT_PASSES`` broadcasts, re-checked at each
      ``attach``) and only then falls back to what the ack would have done
      without the cut (TRANSIT for a transit node, keep otherwise);
    * a REBIND or TRANSIT that meets a busy node (host lock: the node's own
      write or load in flight) stays pending instead of being dropped.

    Outside the cut every path below is the pre-part-4 one.

Switch ``SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW``; it acts only with an
installed Form A role plan whose host is TP rank 0, on a pure TP group
(no DP attention, pp 1). Off (or a classic boot) = the pre-R12 code paths,
unchanged.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: event kinds
TRANSIT = "transit"
STATE = "state"
#: #239 S4b part 4: TP0 rebound a COMPLETE token-cut page -- every worker does
REBIND = "rebind"
#: W3-ARENA on Form A (metal dual15): TP0 spilled a node's finished pages to L3
#: and gave its KV host rows back; every worker gives its own back and marks
#: the node store-present (``l3_present``) -- the same parent rule everywhere
SPILL = "spill"
#: K12/H1: TP0 holds the node in L3 (its complete ack, or under the cut the
#: complete page) -- every worker sets ``l3_present`` from this, never from its
#: own ack, so no rank reads a different ``l3_present`` for the same node
L3 = "l3"
#: K21 (raenge-nie-uneins-crash-stop, nf-nachreview-h16v-1017 N1): TP0's own drain ack made
#: this node an A2 orphan. Sent at EVERY TP0 orphan ack (also when the host release then
#: fails), so a worker can compare it with its OWN orphan mark in both directions -- the
#: orphan verdict is rank-local (each rank acks against the shared arena at its own time).
K21 = "k21"

#: READ-ROOM-D (review readroomd2 A4): TP0 released a node's KV host rows for a P read -- applied on a worker
#: exactly like STATE, but an apply that ends ``pending`` or ``deleted`` is NAMED at every occurrence
#: (READ-ROOM-STATE-DIVERGED), like SPILL-DIVERGED: in that window the trees differ
RRSTATE = "rrstate"

#: #239 S4b part 4: broadcasts TP0 waits for an incomplete token-cut page
#: before it falls back to the ack's pre-cut decision (a write of every owner
#: acks within a pass or two; the bound is a backstop, not a timing)
REBIND_WAIT_PASSES = 64

#: Tree attribute: while set to a list, ``_evict_component_and_detach_lru``
#: records every node whose HOST layer it frees (TP0's own evict_host).
REC_ATTR = "_r12_rec"


class FormAHostVerdict:
    """TP0's host decisions of one pass; rides the ``tp<-reqs`` broadcast.

    ``events``: tuples ``(kind, last_hash, depth, exists, kv_host, anchor_host)``.
    """

    __slots__ = ("seq", "events")

    def __init__(self, seq: int, events: List[Tuple[str, str, int, int, int, int]]):
        self.seq = int(seq)
        self.events = list(events)

    def __getstate__(self):
        return (self.seq, self.events)

    def __setstate__(self, state):
        self.seq, self.events = state

    def __repr__(self) -> str:
        return f"FormAHostVerdict(seq={self.seq}, events={len(self.events)})"


class _State:
    """Per-process state: one D tree per scheduler process."""

    def __init__(self) -> None:
        self.ledger: List[Tuple[str, str, int, int, int, int]] = []
        self.pending: Dict[Tuple[str, int], Tuple[str, str, int, int, int, int]] = {}
        self.absent: set = set()
        # K21 ack pairing (nf-nachreview2-h16v-1017 V1, S11): the orphan verdict of a direct write's
        # ack is rank-local AND asynchronous (each rank's finish_event turns ready at its own time).
        # So a worker pairs, per content key and in FIFO order, ITS OWN ack finding with TP0's
        # (0 = clean, 1 = A2 orphan) and judges only once BOTH exist -- no timer, no handover count.
        self.k21_own: Dict[Tuple[str, int], List[int]] = {}
        self.k21_tp0: Dict[Tuple[str, int], List[Tuple[int, bool]]] = {}
        # TP0 K21 events not yet queued (no tree yet / drained at the next apply)
        self.k21_inbox: List[Tuple] = []
        # keys both ranks orphaned and TP0 published a STATE for: that STATE must end 'kv'
        # (value = handover counter, cleanup only -- never a stop reason)
        self.k21_confirmed: Dict[Tuple[str, int], int] = {}
        # keys TP0 reported as orphan while this worker's own ack was clean: its STATE must end 'kv'
        self.k21_expect: set = set()
        # S20 review A1: READ-ROOM-STATE windows open on this worker, (hash, depth) -> [t_open, passes]
        self.rr_open: Dict[Tuple[str, int], list] = {}
        # named stop found outside a consume (own-ack time); raised at the end of the next consume
        self.k21_stop_msg: Optional[str] = None
        # number of broadcasts consumed on this rank
        self.handover_n = 0
        self.tree: Any = None
        # #239 S4b part 4 (TP0 only): nodes whose token-cut page was not yet
        # complete at TP0's ack -> [node, passes waited, fallback transit]
        self.rebind_wait: Dict[Tuple[str, int], list] = {}
        self.seq = 0
        self.counts: Dict[str, int] = {}
        self.announced = False


_S = _State()


def _reset_state_for_tests() -> None:
    global _S
    _S = _State()


# ---------------------------------------------------------------- predicates
def shadow_active() -> bool:
    """Group-uniform: switch on, a Form A plan installed, its host is TP rank 0
    (the request origin). The same answer on every rank of the group."""
    from sglang.srt.environ import envs

    if not envs.SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW.get():
        return False
    from sglang.srt.rank_role import installed_role_plan

    plan = installed_role_plan()
    if plan is None:
        return False
    try:
        return int(getattr(plan, "host_rank", 0)) == 0
    except (TypeError, ValueError):
        return False


def role() -> Optional[str]:
    """``"host"`` (TP0, the source), ``"worker"`` (follows), or None (off)."""
    if not shadow_active():
        return None
    from sglang.srt.rank_role import this_rank_is_form_a_worker

    return "worker" if this_rank_is_form_a_worker() else "host"


def cut_role() -> Optional[str]:
    """#239 S4b part 4: :func:`role` when the Form A group runs the token cut
    (a worker owns real KV rows), else None. Group-uniform: the installed
    plan and the installed owner bounds are the same on every rank."""
    r = role()
    if r is None:
        return None
    try:
        from sglang.srt.rank_role import form_a_token_cut_active

        return r if form_a_token_cut_active() else None
    except Exception:  # noqa: BLE001 - no cut readable = the pre-cut paths
        return None


def wire_active(server_args: Any, ps: Any) -> bool:
    """The broadcast carries verdicts: shadow on, pure TP group (the
    ``tp<-reqs`` branch of ``_broadcast_reqs_across_ranks``), pp 1."""
    try:
        if getattr(server_args, "enable_dp_attention", False):
            return False
        if int(getattr(ps, "pp_size", 1)) != 1 or int(getattr(ps, "tp_size", 1)) <= 1:
            return False
    except (TypeError, ValueError):
        return False
    return shadow_active()


def _note(key: str, every: int = 64) -> int:
    n = _S.counts.get(key, 0) + 1
    _S.counts[key] = n
    return n if (n <= 8 or n % every == 0) else 0


def _announce(r: str) -> None:
    if not _S.announced:
        _S.announced = True
        logger.info(
            "R12 HOST-SHADOW ACTIVE role=%s (worker host rows are byteless "
            "bookkeeping kept while TP0 holds them; TP0 host decisions ride the "
            "tp<-reqs broadcast)", r)


def register_tree(tree: Any) -> None:
    _S.tree = tree


# ---------------------------------------------------------------- node keys
def node_key(node: Any) -> Optional[Tuple[str, int]]:
    """(last page hash, end depth in tokens) -- rank-invariant content key."""
    try:
        hv = getattr(node, "hash_value", None)
        if not hv:
            return None
        depth = 0
        n = node
        while n is not None and getattr(n, "parent", None) is not None:
            depth += len(n.key)
            n = n.parent
        return str(hv[-1]), int(depth)
    except Exception:  # noqa: BLE001 - an unkeyable node is counted, not fatal
        return None


def _attached(node: Any) -> bool:
    n = node
    while getattr(n, "parent", None) is not None:
        p = n.parent
        if not any(c is n for c in p.children.values()):
            return False
        n = p
    return True


def _anchor_ct():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.MAMBA


def _base_ct():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.FULL


def _host_flags(node: Any) -> Tuple[int, int]:
    cd = node.component_data
    kv = int(cd[int(_base_ct())].host_value is not None)
    act = int(_anchor_ct())
    m = int(len(cd) > act and cd[act].host_value is not None)
    return kv, m


# ---------------------------------------------------------------- TP0 side
def record_transit(tree: Any, node: Any) -> bool:
    """TP0: the store ack wants to free this node's host rows as transit.
    Deferred to the next broadcast, where every rank runs the same release.
    False = not keyable -> the caller releases at once (pre-R12)."""
    k = node_key(node)
    if k is None:
        if _note("unkeyed_transit"):
            logger.warning("R12 HOST-VERDICT unkeyed transit node=%s (released at once)",
                           getattr(node, "id", "?"))
        return False
    register_tree(tree)
    _announce("host")
    _S.ledger.append((TRANSIT, k[0], k[1], 1, 0, 0))
    return True


def record_state(tree: Any, node: Any, why: str = "") -> None:
    """TP0: its host life of ``node`` changed by a decision of its own. Send the
    node's state now; a worker reconciles to it."""
    if node is None or node is getattr(tree, "root_node", None):
        return
    k = node_key(node)
    if k is None:
        if _note("unkeyed_state"):
            logger.warning("R12 HOST-VERDICT unkeyed state node=%s why=%s",
                           getattr(node, "id", "?"), why)
        return
    register_tree(tree)
    _announce("host")
    exists = int(_attached(node))
    kv, m = _host_flags(node) if exists else (0, 0)
    if not kv:
        _S.absent.add(k)
    else:
        _S.absent.discard(k)
    _S.ledger.append((RRSTATE if why == "read-room" else STATE, k[0], k[1], exists, kv, m))


class K21RankDivergence(RuntimeError):
    """K21 A1 (raenge-nie-uneins-crash-stop): TP0 republished an A2-orphan node (it stays on the
    device on every rank) and this worker could not reconcile its STATE to ``kv`` -- the ranks
    would hold different trees. Named group stop, never a counter."""


def note_k21_ack(node: Any, orphaned: int) -> None:
    """The write ack of ``node`` was judged on this rank: ``orphaned`` 1 = A2 orphan, 0 = clean.
    Called at EVERY direct-write ack (V1: pairing instead of a handover karenz).

    * TP0 sends one K21 event per own ack (field ``exists`` = orphaned), even if the host release
      then fails -- exactly one event per ack, so a worker's FIFO lines up;
    * a worker records its own finding and pairs it with TP0's (see :func:`_k21_pair`)."""
    if not shadow_active():
        return
    k = node_key(node)
    if k is None:
        return
    o = 1 if orphaned else 0
    if role() == "host":
        _announce("host")
        _S.ledger.append((K21, k[0], k[1], o, 0, 0))
    else:
        _S.k21_own.setdefault(k, []).append(o)
        _k21_pair(k)


def note_k21_orphan(node: Any) -> None:  # kept for callers/tests of the old name
    note_k21_ack(node, 1)


def _k21_ctx(k: Tuple[str, int], own_finding=None, tp0_finding=None, extra_own: int = 0, extra_tp0: int = 0) -> str:
    """S13 A3 (log only): the pairing state a K21 stop message names -- FIFO depth of this key on
    the own side and on the TP0 side (``extra_*`` = the finding just popped), both findings
    (orphaned 1|0, None = none yet) and the handover count."""
    return (f" [fifo own={len(_S.k21_own.get(k) or ()) + extra_own} tp0={len(_S.k21_tp0.get(k) or ()) + extra_tp0} "
            f"own_finding={own_finding} tp0_finding={tp0_finding} handover_n={_S.handover_n}]")


def _k21_stop(msg: str) -> None:
    logger.error("K21 RANK-DIVERGENCE %s -- group stop (raenge-nie-uneins-crash-stop)", msg)
    raise K21RankDivergence("K21: " + msg)


def _k21_defer_stop(msg: str) -> None:
    """A divergence found where an exception would be swallowed (the ack path): raised by the
    next consume."""
    if _S.k21_stop_msg is None:
        _S.k21_stop_msg = msg


def _k21_pair(k: Tuple[str, int]) -> None:
    """Worker: judge every complete (own, TP0) pair of key ``k`` in FIFO order. Only when BOTH
    findings exist; an own write still in flight just leaves TP0's finding parked."""
    own = _S.k21_own.get(k)
    tp0 = _S.k21_tp0.get(k)
    while own and tp0:
        o = own.pop(0)
        t, has_state = tp0.pop(0)
        if o and t:
            if has_state:
                _S.k21_confirmed[k] = _S.handover_n  # the STATE apply checks the outcome
            # else: both orphaned, TP0's release failed: agreed
        elif t and not o:
            # TP0 orphaned, this worker's ack was clean: TP0 keeps the node on the device, the
            # worker must end there too, i.e. a STATE that applies 'kv'
            if not has_state:
                _k21_defer_stop(
                    f"TP0 orphaned node hash={str(k[0])[:16]} depth={k[1]} (kept on the device), this "
                    "worker's ack was clean and no STATE follows: trees diverge" + _k21_ctx(k, o, t, 1, 1))
            else:
                _S.k21_expect.add(k)
        elif o and not t:
            _k21_defer_stop(
                f"this worker's ack orphaned node hash={str(k[0])[:16]} depth={k[1]} but TP0's ack "
                "of the same write was clean: the ranks' trees diverge (only this rank kept the "
                "node on the device)" + _k21_ctx(k, o, t, 1, 1))
    if own is not None and not own:
        _S.k21_own.pop(k, None)
    if tp0 is not None and not tp0:
        _S.k21_tp0.pop(k, None)


def _k21_raise_pending() -> None:
    msg = _S.k21_stop_msg
    if msg is not None:
        _S.k21_stop_msg = None
        _k21_stop(msg)


def _pkey(e) -> tuple:
    """The pending slot of an event: one per node key -- a newer TRANSIT /
    STATE / REBIND supersedes an older one -- and a SEPARATE slot for L3, which
    is a fact about the store, not a host decision: a TRANSIT or REBIND of the
    same node in the same pass must not drop it."""
    k = (e[1], int(e[2]))
    return k + (e[0],) if e[0] in (L3, K21) else k


def l3_group_on(env=None) -> bool:
    """K12/H1 switch ``SGLANG_WEG2_L3_PRESENT_GROUP`` (default ON): on a Form A
    group ``l3_present`` follows TP0's verdict on every rank."""
    v = (env if env is not None else os.environ).get("SGLANG_WEG2_L3_PRESENT_GROUP", "1")
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def record_l3(tree: Any, node: Any) -> bool:
    """K12/H1, TP0: it set ``l3_present`` on ``node`` (group-complete). Every
    worker sets it at the next broadcast. False = not keyable (logged; the
    workers then keep l3_present False for this node -- conservative)."""
    k = node_key(node)
    if k is None:
        if _note("unkeyed_l3"):
            logger.warning("R12 HOST-VERDICT unkeyed l3 node=%s (the workers keep "
                           "l3_present False)", getattr(node, "id", "?"))
        return False
    register_tree(tree)
    _announce("host")
    _S.ledger.append((L3, k[0], k[1], 1, 1, 0))
    return True


def record_rebind(tree: Any, node: Any) -> bool:
    """#239 S4b part 4, TP0 under the cut: it rebound a COMPLETE page to the
    arena; every worker rebinds the same node at the next broadcast (until
    then it keeps its own rows -- the bytes). False = not keyable (logged)."""
    k = node_key(node)
    if k is None:
        if _note("unkeyed_rebind"):
            logger.warning("R12 HOST-VERDICT unkeyed rebind node=%s (the workers keep "
                           "their rows)", getattr(node, "id", "?"))
        return False
    register_tree(tree)
    _announce("host")
    _S.ledger.append((REBIND, k[0], k[1], 1, 1, 0))
    return True


def record_spill(tree: Any, node: Any) -> bool:
    """W3-ARENA, TP0 on Form A: it secured ``node``'s pages to L3 and released
    its KV host rows (the node stays on the device, ``l3_present``). Every
    worker does the same at the next broadcast -- an arena slot is free only
    once EVERY rank's reference is gone. False = not keyable (logged)."""
    k = node_key(node)
    if k is None:
        if _note("unkeyed_spill"):
            logger.warning("R12 HOST-VERDICT unkeyed spill node=%s (the workers keep "
                           "their references)", getattr(node, "id", "?"))
        return False
    register_tree(tree)
    _announce("host")
    _kv, m = _host_flags(node)
    _S.ledger.append((SPILL, k[0], k[1], 1, 0, m))
    return True


# ------------------------------------------ W3 spill requests (worker -> TP0)
#: metal dual15 (...09301211): the arena was full, TP1's claims needed 1409
#: pages, TP0's only 12 -- the rank that CAN decide a spill (TP0) was not the
#: one short of room. A worker leaves its need next to the shared arena file;
#: TP0 reads it at its next broadcast. Host bookkeeping, no collective; a lost
#: update only waits for the worker's next refusal (it recurs every pass).
SPILL_REQ_SUFFIX = ".w3need"


def _req_path(arena: Any) -> Optional[str]:
    path = getattr(arena, "path", None)
    return (str(path) + SPILL_REQ_SUFFIX) if path else None


def request_spill(arena: Any, pages: int) -> bool:
    """A worker: its arena claim for ``pages`` pages was refused."""
    path = _req_path(arena)
    if path is None or int(pages) <= 0:
        return False
    import struct

    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            raw = os.pread(fd, 8, 0)
            cur = struct.unpack("<q", raw)[0] if len(raw) == 8 else 0
            if int(pages) > cur:
                os.pwrite(fd, struct.pack("<q", int(pages)), 0)
        finally:
            os.close(fd)
    except OSError:
        return False
    n = _note("spill_request", 64)
    if n:
        logger.info("R12 W3-SPILL-REQUEST pages=%d (this worker's arena claim was refused; "
                    "TP0 decides the spill at its next broadcast) n=%d", int(pages), n)
    return True


def take_spill_request(arena: Any) -> int:
    """TP0: the largest pending worker need (pages), cleared."""
    path = _req_path(arena)
    if path is None:
        return 0
    import struct

    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return 0
    try:
        raw = os.pread(fd, 8, 0)
        cur = struct.unpack("<q", raw)[0] if len(raw) == 8 else 0
        if cur:
            os.pwrite(fd, struct.pack("<q", 0), 0)
        return max(0, int(cur))
    except OSError:
        return 0
    finally:
        os.close(fd)


def await_complete(tree: Any, node: Any, fallback_transit: bool, l3_ok: bool = False) -> bool:
    """#239 S4b part 4, TP0 under the cut: its ack came while the page was not
    complete in the arena (an owner's write still in flight). Wait for the
    page instead of deciding on TP0's own ack. False = not keyable -> the
    caller takes the pre-cut decision at once."""
    k = node_key(node)
    if k is None:
        return False
    register_tree(tree)
    _announce("host")
    _S.rebind_wait[k] = [node, 0, bool(fallback_transit), bool(l3_ok)]
    return True


def _retry_rebinds() -> None:
    """TP0, at ``attach``: re-check every waiting page; complete -> rebind and
    send REBIND; past the bound -> the pre-cut decision (TRANSIT or keep)."""
    tree = _S.tree
    if tree is None or not _S.rebind_wait:
        return
    for k, ent in list(_S.rebind_wait.items()):
        node, waited, fallback = ent[0], ent[1], ent[2]
        l3_ok = bool(ent[3]) if len(ent) > 3 else False
        if not _attached(node) or node.component_data[int(_base_ct())].host_value is None:
            del _S.rebind_wait[k]  # the node or its host life went another way
            continue
        if tree._weg2_rebind_host_to_arena(node):
            del _S.rebind_wait[k]
            _S.ledger.append((REBIND, k[0], k[1], 1, 1, 0))
            if l3_ok:
                # K12/H1: the page completed -- l3_present on every rank
                node.l3_present = True
                _S.ledger.append((L3, k[0], k[1], 1, 1, 0))
            n = _note("rebind_late", 256)
            if n:
                logger.info("R12 CUT-REBIND node=%s depth=%d after %d pass(es) (every owner "
                            "wrote; the workers rebind at this broadcast) n=%d",
                            getattr(node, "id", "?"), k[1], waited + 1, n)
            continue
        ent[1] = waited + 1
        if ent[1] >= REBIND_WAIT_PASSES:
            del _S.rebind_wait[k]
            if fallback:
                _S.ledger.append((TRANSIT, k[0], k[1], 1, 0, 0))
            if _note("rebind_expired", 16):
                logger.warning(
                    "R12 CUT-REBIND-EXPIRED node=%s depth=%d: the page was not complete "
                    "in the arena after %d broadcasts -> %s (the pre-cut decision)",
                    getattr(node, "id", "?"), k[1], ent[1],
                    "TRANSIT on every rank" if fallback else "every rank keeps its rows")


def note_backup_ok(tree: Any, node: Any) -> None:
    """TP0 backed up a node it earlier reported without host: say so, so a
    worker's still-pending drop for it is superseded."""
    if not _S.absent:
        return
    k = node_key(node)
    if k is not None and k in _S.absent:
        record_state(tree, node, why="backup_ok")


def attach(recv_reqs: List) -> List:
    """Origin (TP0), right before the broadcast: put the pass's verdict FIRST."""
    if _S.rebind_wait:
        _retry_rebinds()  # #239 S4b part 4: pages that completed since
    serve = getattr(_S.tree, "_w3_forma_serve_requests", None)
    if serve is not None:
        try:
            serve()  # W3 on Form A: a worker's refused claim -> TP0's spill, SPILL events below
        except Exception as exc:  # noqa: BLE001 -- a failed spill keeps every reference
            if _note("spill_serve_error", 16):
                logger.warning("R12 W3-SPILL serve failed: %r (every rank keeps its references)", exc)
    # READ-ROOM-D (review readroomd B2): a P read's room request is served HERE, right before the broadcast, never
    # inside the round; the decided host drops are applied on TP0 now and their verdicts (STATE + L3) ride this
    # broadcast -- every rank's tree changes before the same admission
    try:
        from sglang.srt.weg2 import read_room as _read_room

        if _S.tree is not None:
            _read_room.serve_at_broadcast(_S.tree)
        _read_room.flush_forma_pending()
    except Exception as exc:  # noqa: BLE001 -- a room request never breaks the broadcast
        if _note("read_room_serve_error", 16):
            logger.warning("R12 READ-ROOM serve failed: %r (every rank keeps its references)", exc)
    if not _S.ledger:
        return recv_reqs
    _S.seq += 1
    v = FormAHostVerdict(_S.seq, _S.ledger)
    _S.ledger = []
    n = _note("sent")
    if n:
        kinds = [e[0] for e in v.events]
        logger.info("R12 HOST-VERDICT SENT seq=%d events=%d transit=%d state=%d rebind=%d spill=%d l3=%d k21=%d (n=%d)",
                    v.seq, len(v.events), kinds.count(TRANSIT), kinds.count(STATE) + kinds.count(RRSTATE),
                    kinds.count(REBIND), kinds.count(SPILL), kinds.count(L3), kinds.count(K21), n)
    return [v] + list(recv_reqs)


def consume(recv_reqs: Optional[List], tree: Any = None) -> Optional[List]:
    """Every rank, right after the broadcast: take the verdicts out of the
    list and apply them before any request of this pass is processed."""
    _S.handover_n += 1
    return _consume(recv_reqs, tree)


def _consume(recv_reqs: Optional[List], tree: Any = None) -> Optional[List]:
    out = _consume_apply(recv_reqs, tree)
    if role() == "worker":
        for ck in [ck for ck, n in _S.k21_confirmed.items() if _S.handover_n - n > 2]:
            _S.k21_confirmed.pop(ck, None)  # cleanup only, never a stop
        _k21_raise_pending()
    return out


def _consume_apply(recv_reqs: Optional[List], tree: Any = None) -> Optional[List]:
    if not recv_reqs:
        if (_S.pending or _S.k21_inbox) and (tree or _S.tree) is not None:
            apply(tree or _S.tree, [])
        return recv_reqs
    verdicts = [r for r in recv_reqs if isinstance(r, FormAHostVerdict)]
    if not verdicts:
        if (_S.pending or _S.k21_inbox) and (tree or _S.tree) is not None:
            apply(tree or _S.tree, [])
        return recv_reqs
    rest = [r for r in recv_reqs if not isinstance(r, FormAHostVerdict)]
    events: List = []
    for v in verdicts:
        events.extend(v.events)
    t = tree or _S.tree
    if t is None:
        # no tree has touched HiCache yet on this rank: keep for the next pass
        for e in events:
            if e[0] == K21:
                _S.k21_inbox.append(e)  # FIFO, never superseded
            else:
                _S.pending[_pkey(e)] = e
        return rest
    apply(t, events, seq=verdicts[-1].seq)
    return rest


def on_tree_reset(tree: Any) -> None:
    """The tree is dropped (flush/flip, on every rank at the same point): no
    event may outlive the nodes it names -- a re-inserted prefix has the same
    hash."""
    if _S.ledger or _S.pending or _S.absent or _S.rebind_wait:
        logger.info("R12 HOST-VERDICT reset drops ledger=%d pending=%d rebind_wait=%d",
                    len(_S.ledger), len(_S.pending), len(_S.rebind_wait))
    # S13 A3 (log only): every open K21 stop and open pairing the reset discards, one line
    _open_own = sum(len(v) for v in _S.k21_own.values())
    _open_tp0 = sum(len(v) for v in _S.k21_tp0.values())
    if (_S.k21_stop_msg is not None or _open_own or _open_tp0 or _S.k21_inbox
            or _S.k21_expect or _S.k21_confirmed):
        logger.warning(
            "K21 RESET-DROPS open_stop=%d open_own=%d open_tp0=%d inbox=%d expect=%d confirmed=%d "
            "handover_n=%d stop_msg=%s", 1 if _S.k21_stop_msg is not None else 0, _open_own, _open_tp0,
            len(_S.k21_inbox), len(_S.k21_expect), len(_S.k21_confirmed), _S.handover_n, _S.k21_stop_msg)
    _S.ledger = []
    _S.pending = {}
    _S.absent = set()
    _S.k21_own = {}
    _S.k21_tp0 = {}
    _S.k21_inbox = []
    _S.k21_confirmed = {}
    _S.k21_expect = set()
    _S.rr_open = {}
    _S.k21_stop_msg = None
    _S.rebind_wait = {}


# ---------------------------------------------------------------- apply
def _index(tree: Any) -> Dict[str, List[Any]]:
    idx: Dict[str, List[Any]] = {}
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        for c in n.children.values():
            stack.append(c)
            hv = getattr(c, "hash_value", None)
            if hv:
                idx.setdefault(str(hv[-1]), []).append(c)
    return idx


def _find(idx, key: str, depth: int):
    for n in idx.get(key, ()):
        k = node_key(n)
        if k is not None and k[1] == int(depth):
            return n
    return None


def _busy(tree: Any, node: Any) -> bool:
    if any(int(getattr(cd, "host_lock_ref", 0) or 0) > 0 for cd in node.component_data):
        return True
    owt = getattr(tree, "ongoing_write_through", None) or {}
    return node.id in owt or getattr(node, "write_through_pending_id", None) is not None


def apply(tree: Any, events: List, seq: int = 0) -> Dict[str, int]:
    """Apply TP0's events on THIS rank. The same code on every rank; TRANSIT
    runs everywhere (TP0 deferred its own release to here), STATE only on a
    worker (TP0 is already in that state)."""
    r = role()
    cut = cut_role() is not None
    c = dict(transit=0, kv=0, anchor=0, deleted=0, short=0, missing=0,
             pending=0, unreconcilable=0, noop=0, rebind=0, rebind_miss=0, spill=0)
    for e in events:
        if e[0] == K21:
            _S.k21_inbox.append(e)  # K21 verdicts are a FIFO per key: never superseded
        else:
            _S.pending[_pkey(e)] = e  # a newer event for the key supersedes (L3 has its own slot)
    inbox, _S.k21_inbox = _S.k21_inbox, []
    if not _S.pending and not inbox:
        return c
    idx = _index(tree)
    if inbox and r == "worker":
        st_keys = {(e[1], int(e[2])) for e in list(_S.pending.values()) + list(events) if e[0] in (STATE, RRSTATE)}
        for e in inbox:
            k21k = (e[1], int(e[2]))
            if _find(idx, k21k[0], k21k[1]) is None:
                _S.k21_own.pop(k21k, None)
                _S.k21_tp0.pop(k21k, None)
                c["missing"] += 1
                continue
            _S.k21_tp0.setdefault(k21k, []).append((1 if e[3] else 0, k21k in st_keys))
            _k21_pair(k21k)
            c["noop"] += 1
    if not _S.pending:
        return c
    work = list(_S.pending.values())
    _S.pending = {}
    state_keys = {(e[1], int(e[2])) for e in work if e[0] in (STATE, RRSTATE)}
    for e in work:
        kind, h, depth, exists, kv, m = e
        node = _find(idx, h, depth)
        if node is None:
            c["missing"] += 1
            continue
        if kind == TRANSIT:
            if cut and _busy(tree, node):
                # #239 S4b part 4: under the cut a worker's own store write
                # holds the rows -- the release waits for it (was a silent no-op)
                _S.pending[(h, int(depth))] = e
                c["pending"] += 1
                continue
            before = _host_flags(node)
            tree._weg2_release_chain_piece_host(node)
            c["transit" if _host_flags(node) != before else "noop"] += 1
            continue
        if kind == L3:
            if r != "worker":
                continue  # TP0 set it before it sent the event
            node.l3_present = True
            c["l3"] = c.get("l3", 0) + 1
            continue
        if kind == REBIND:
            if r != "worker":
                continue  # TP0 rebound before it sent the event
            if _busy(tree, node):
                _S.pending[(h, int(depth))] = e
                c["pending"] += 1
                continue
            if tree._weg2_rebind_host_to_arena(node):
                c["rebind"] += 1
            else:
                # the page is complete (TP0 saw it, TP0's reference keeps it):
                # a miss is this rank's pool, not the page -- keep the rows
                c["rebind_miss"] += 1
            continue
        if kind == SPILL:
            if r != "worker":
                continue  # TP0 spilled before it sent the event
            out = _reconcile(tree, node, 1, 0, m)
            if out == "pending":
                _S.pending[(h, int(depth))] = e
                c["pending"] += 1
                continue
            if out in ("kv", "noop"):
                node.l3_present = True  # the parent rule reads backed/l3_present: as on TP0
                c["spill"] += 1
            else:
                c[out] += 1
                _spill_diverged(node, h, depth, out, seq)
            continue
        if r != "worker":
            continue
        out = _reconcile(tree, node, exists, kv, m)
        if out == "pending":
            _S.pending[(h, int(depth))] = e
        elif (not exists and out != "deleted" and (h, int(depth)) not in _S.k21_expect
              and (h, int(depth)) not in _S.k21_confirmed):
            # K23 review A3: TP0 says the node is GONE; only ``deleted`` (and ``missing``, above) leaves the
            # worker's tree the same. kv/anchor/noop/short/unreconcilable keep a node TP0 no longer has.
            _state_diverged(node, h, depth, out, seq)
        elif (h, int(depth)) in _S.k21_expect:
            _S.k21_expect.discard((h, int(depth)))
            if out != "kv":
                _k21_stop(
                    f"TP0 orphaned node={getattr(node, 'id', '?')} depth={depth} hash={str(h)[:16]} "
                    f"(kept on the device), this worker did not orphan it and its STATE apply ended "
                    f"{out!r} instead of 'kv' (seq={int(seq)}): trees diverge"
                    + _k21_ctx((h, int(depth)), 0, 1))
        elif (h, int(depth)) in _S.k21_confirmed:
            _S.k21_confirmed.pop((h, int(depth)), None)
            if out in ("deleted", "unreconcilable"):
                logger.error(
                    "K21 RANK-DIVERGENCE seq=%d node=%s depth=%s hash=%s out=%s: TP0 republished "
                    "an A2-orphan node (kept on the device) and this worker's STATE apply ended "
                    "'%s' instead of 'kv' -- the ranks' trees diverge; group stop "
                    "(raenge-nie-uneins-crash-stop)%s", int(seq), getattr(node, "id", "?"),
                    depth, str(h)[:16], out, out, _k21_ctx((h, int(depth)), 1, 1))
                raise K21RankDivergence(
                    f"K21: orphan-republish STATE for node={getattr(node, 'id', '?')} depth={depth} "
                    f"applied as {out!r} on a worker (expected 'kv'): rank trees diverge"
                    + _k21_ctx((h, int(depth)), 1, 1))
        c[out] += 1
        if kind == RRSTATE:
            _rrstate_window(node, h, depth, out, seq)
    n = _note("applied")
    if n and (events or any(c[k] for k in ("transit", "kv", "anchor", "deleted", "rebind", "spill"))):
        logger.info(
            "R12 HOST-VERDICT APPLIED seq=%d role=%s events=%d transit=%d kv_dropped=%d "
            "anchor_dropped=%d deleted=%d short=%d missing=%d pending=%d unreconcilable=%d "
            "rebind=%d spill=%d (n=%d)",
            seq, r, len(events), c["transit"], c["kv"], c["anchor"], c["deleted"], c["short"],
            c["missing"], c["pending"], c["unreconcilable"], c["rebind"], c["spill"], n)
    if c["rebind_miss"] and _note("rebind_miss", 16):
        logger.warning("R12 CUT-REBIND-MISS %d node(s): TP0 rebound a complete page, this "
                       "worker's pool could not -- it keeps its own rows (same tree shape)",
                       c["rebind_miss"])
    if c["short"] and _note("short_apply", 16):
        logger.warning("R12 SHADOW-SHORT at reconcile: %d node(s) TP0 holds on host, "
                       "this worker does not (its own backup was refused)", c["short"])
    if c["unreconcilable"] and _note("unreconcilable", 16):
        logger.error("R12 SHADOW-UNRECONCILABLE %d node(s): TP0 dropped host of a node "
                     "that is device-evicted and interior here", c["unreconcilable"])
    return c


def _spill_diverged(node: Any, h: Any, depth: Any, out: str, seq: int) -> None:
    """H16Q review A2 (law raenge-nie-uneins): a worker that cannot apply TP0's SPILL (``short`` / ``anchor`` /
    ``deleted`` / ``unreconcilable``) now disagrees with TP0 on this node's host copy -- TP0 freed its host rows
    and marked it ``l3_present``, this worker did not (``deleted``: it even dropped a node TP0 keeps on the
    device). Named at EVERY occurrence (never the rate-limited APPLIED counter): the metal count must be 0; if it
    is not, the named group stop follows in the next batch (a stop now, at an unknown rate, would be a new death
    button without a measurement)."""
    n = _S.counts.get("spill_diverged", 0) + 1
    _S.counts["spill_diverged"] = n
    logger.warning(
        "R12 SPILL-DIVERGED seq=%d rid=%s node=%s depth=%s hash=%s out=%s (n=%d): TP0 spilled this node (host "
        "rows freed, l3_present) and this worker could not apply it -- the ranks now disagree on its host copy",
        int(seq), str(getattr(node, "rid", None) or "-"), getattr(node, "id", "?"), depth,
        str(h)[:16], out, n)


def _state_diverged(node: Any, h: Any, depth: Any, out: str, seq: int) -> None:
    """K23 review A3 (law raenge-nie-uneins): TP0's STATE said ``exists=0`` and this worker's reconcile ended
    ``out`` != deleted -- the worker keeps a node TP0 dropped (kv/anchor/noop run silently otherwise, unreconcilable
    only rate-limited). Named at EVERY occurrence, plus a counter; the group stop is decided after the metal count
    (a stop now, at an unknown rate, would be a new death button without a measurement)."""
    n = _S.counts.get("k23_state_diverged", 0) + 1
    _S.counts["k23_state_diverged"] = n
    logger.warning(
        "K23 SPILL-DIVERGED role=worker key=%s:%s node=%s seq=%d outcome=%s (n=%d): TP0 dropped this node "
        "(STATE exists=0) and this worker could not delete it -- the ranks' trees now differ",
        str(h)[:16], depth, getattr(node, "id", "?"), int(seq), out, n)

def _rrstate_window(node: Any, h: Any, depth: Any, out: str, seq: int) -> None:
    """S20 review A1: a READ-ROOM STATE that ends ``pending`` re-enters ``_S.pending`` and is re-applied every pass.
    The WINDOW is named once when it opens (first ``pending`` of the key) and once when it closes (first other
    outcome, with duration and passes); the counter counts windows, not passes. ``deleted``/``short``/
    ``unreconcilable`` are terminal single events."""
    k = (str(h), int(depth))
    if out == "pending":
        w = _S.rr_open.get(k)
        if w is None:
            _S.rr_open[k] = [time.monotonic(), 1]
            _rrstate_diverged(node, h, depth, out, seq)
        else:
            w[1] += 1
        return
    w = _S.rr_open.pop(k, None)
    if w is not None:
        logger.info(
            "R12 READ-ROOM-STATE window closed node=%s depth=%s hash=%s out=%s after %.3fs and %d passes",
            getattr(node, "id", "?"), depth, str(h)[:16], out, time.monotonic() - w[0], w[1])
    if out in ("deleted", "short", "unreconcilable"):
        _rrstate_diverged(node, h, depth, out, seq)


def _rrstate_diverged(node: Any, h: Any, depth: Any, out: str, seq: int) -> None:
    """READ-ROOM-D (review readroomd2 A4): a worker that cannot apply TP0's READ-ROOM STATE at once. ``pending``:
    the node is busy here (under the token cut its own store write holds the rows) -- until the next broadcast
    applies it, TP0 has no host copy and this worker has one, so a device eviction in that window takes TP0's
    Form A L3-DROP and this worker's demotion; ``deleted``: this worker dropped a node TP0 keeps. Named at EVERY
    occurrence (no stop, no timer): the metal count says how often the window opens."""
    n = _S.counts.get("rrstate_diverged", 0) + 1
    _S.counts["rrstate_diverged"] = n
    logger.warning(
        "R12 READ-ROOM-STATE-DIVERGED seq=%d node=%s depth=%s hash=%s out=%s (n=%d): TP0 released this node's KV "
        "host rows for a P read and this worker could not follow at once -- the ranks differ on its host copy "
        "until the verdict applies", int(seq), getattr(node, "id", "?"), depth, str(h)[:16], out, n)


def _reconcile(tree: Any, node: Any, exists: int, kv: int, m: int) -> str:
    from sglang.srt.mem_cache.unified_cache_components.tree_component import EvictLayer

    has_kv, has_m = _host_flags(node)
    want_kv = bool(exists and kv)
    want_m = bool(exists and m)
    if (want_kv and not has_kv) or (want_m and not has_m):
        return "short"
    if (has_kv and not want_kv) or (has_m and not want_m):
        if _busy(tree, node):
            return "pending"
    if has_kv and not want_kv:
        if not node.evicted:
            # y5h: aux host first, the KV host rows last; an anchor TP0 kept
            # without its KV host copy cannot stay either (aux host requires
            # Full host) -- the funnel's net drops it with the KV rows
            for comp in sorted(tree._components_tuple,
                               key=lambda c: c.component_type == _base_ct()):
                if want_m and comp.component_type == _anchor_ct():
                    # #248: TP0 gave only the KV rows back after a load-back
                    # (weg2.park_l3.release_loaded_host); the anchor stays
                    continue
                tree._evict_component_and_detach_lru(node, comp, target=EvictLayer.HOST, tracker=None)
            tree.evictable_host_leaves.discard(node)
            tree._update_evictable_leaf_sets(node)
            return "kv"
        if tree._is_host_leaf(node):
            tracker = {ct: 0 for ct in tree.tree_components}
            tree._evict_host_leaf(node, tracker)
            return "deleted"
        return "unreconcilable"
    if has_m and not want_m:
        comp = tree.components[_anchor_ct()]
        tree._evict_component_and_detach_lru(node, comp, target=EvictLayer.HOST, tracker=None)
        tree._update_evictable_leaf_sets(node)
        return "anchor"
    return "noop"


# ---------------------------------------------------------------- worker side
def worker_keeps(tree: Any, site: str, node: Any) -> None:
    """A worker kept rows it used to free as transit (store ack / load-back)."""
    register_tree(tree)
    _announce("worker")
    n = _note("keep_" + site, 256)
    if n:
        logger.info("R12 SHADOW-KEEP site=%s node=%s tokens=%d (byteless; TP0 holds "
                    "the bytes) n=%d", site, getattr(node, "id", "?"),
                    len(getattr(node, "key", None) or ()), n)


def worker_refuses_own_evict(tree: Any, component_type: Any, num_tokens: int) -> None:
    register_tree(tree)
    n = _note("own_evict", 16)
    if n:
        logger.warning(
            "R12 SHADOW-OWN-EVICT-REFUSED comp=%s need=%d n=%d (a worker never "
            "evicts host on its own -- its pool is short of TP0's; the caller "
            "refuses instead)", getattr(component_type, "name", component_type),
            int(num_tokens), n)


def worker_short(tree: Any, why: str, node: Any) -> None:
    register_tree(tree)
    n = _note("short_" + why.split(":", 1)[0], 64)
    if n:
        logger.warning("R12 SHADOW-SHORT why=%s node=%s (this worker refused a backup "
                       "TP0 may take) n=%d", why, getattr(node, "id", "?"), n)


def mamba_shadow_extra_rows(size_per_slot: int) -> int:
    """Extra byteless anchor rows for a Form A worker (0 B/slot only):
    twice the mamba arena's slot count, the value the arena itself reads
    (``HiCacheFile._arena_for``) -- a replicated input, no collective."""
    try:
        if int(size_per_slot) != 0 or role() != "worker":
            return 0
        from sglang.srt.mem_cache.pool_host.arena_pool import arena_host_enabled

        if not arena_host_enabled():
            return 0
        # Read exactly as the arena reads it (HiCacheFile._arena_for, raw
        # environment, default 48): the point is the SAME number TP0's arena
        # gets, so it must not go through a second parse with its own default.
        slots = int(os.environ.get("SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "48"))
    except Exception:  # noqa: BLE001 - sizing falls back to the synced count
        return 0
    extra = 2 * max(0, slots)
    logger.info("R12 SHADOW-CAPACITY mamba anchor rows +%d (2 x arena slots %d, 0 B/row)",
                extra, slots)
    return extra
