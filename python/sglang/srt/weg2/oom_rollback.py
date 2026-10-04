"""Q-710 / Auftrag 1090: a prefill OOM in ``prepare_for_extend`` hands the batch back.

THE DEFECT (NF y9nf, boot ...10040027, P PP0 00:42:31Z). The adder admits a pass, the
admission has ALREADY spent device rows (``init_load_back`` puts the whole host hit on the
device), locked the matched nodes, taken mamba COW slots; then ``alloc_for_extend`` finds the
pool short ('Prefill out of memory ... Available full tokens: 2816, full_evictable_size_=0'),
the ``RuntimeError`` leaves ``prepare_for_extend`` and the scheduler loop, and the rank dies
(RANK-DEATH, PP1/PP2 followed). Q-700 prices the load-back at admission, so that specimen no
longer reaches here; this is the second line of defence for whatever residue is left.

WHAT THE OOM LEAVES BEHIND (read from the code at 458851283d, not guessed):

* the loaded host->device nodes. NOT a leak: ``UnifiedRadixCache.load_back`` registers them in
  ``ongoing_load_back`` with their own lock/host-lock pins, and ``loading_check`` releases those
  at the transfer's ack, with or without a request. They become ordinary evictable nodes.
  BUT the transfer is asynchronous on the load stream and only the batch's forward waits for it
  (``hicache_consumer_index``). A rolled-back batch has no forward, a retry that matches the
  same nodes would read rows the copy has not landed in yet (the #767 shape: silent wrong KV).
  So the rollback WAITS for the load events and drains the acks before it returns.
* the admission lock (``_req_inc_lock_ref``) on every fresh request's ``last_node`` (+ the mamba
  anchor pin ``mamba_anchor_pin_held``): a request that never reaches
  ``cache_finished_req``/``cache_unfinished_req`` never gives it back -> the nodes are pinned
  for good. Released here the way ``cache_finished_req`` releases it.
* ``alloc_req_slots`` ran just before the token allocation: a request row, a mamba slot and a
  ping-pong buffer per fresh request (and the COW slot ``finalize_match_result`` drew at
  matching, whose 'acquired this admission' stamp ``alloc`` resets: #924 MAMBA SLOT ALIASING
  if it is freed twice or never). Given back through ``_rollback_alloc`` (what ``alloc`` itself
  uses on a partial failure) and ``release_admission_acquired_mamba_slot``.
* scheduler state: the waiting queue lost the batch's requests, ``chunked_req`` may have been
  replaced/cleared and its ``inflight_middle_chunks`` incremented, the PP decision/wire of
  this pass were built.

WHAT THIS WILL NOT DO (named refusals, the original OOM is then raised as before): tp > 1 (the
verdict would have to be a group vote, a rank that did not OOM would have to give back a
successful allocation), a PP follower (PP0's hidden states are on the wire), an armed
p-layer-split row, anchor tails / skip-extend in the pass, more than one request that already
owned a row, a request with a session, a tree that is not the unified tree, and a rid that was
already rolled back ``SGLANG_WEG2_OOM_ROLLBACK_MAX`` times. Group P only (the flip form and the
dual lane never reach it).

Rank-uniform by construction: the only rank that acts is the one that produced the pass (tp 1,
PP0 or non-PP); a PP follower never sees a pass PP0 did not send.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MARK = "Q-710 OOM-ROLLBACK"


def active() -> bool:
    """Group P and the knob on."""
    from sglang.srt.environ import envs
    from sglang.srt.weg2.park import group_is_p

    return group_is_p() and bool(envs.SGLANG_WEG2_OOM_ROLLBACK.get())


@dataclass
class _Pre:
    """What a request looked like BEFORE ``alloc_for_extend`` touched it."""
    had_row: bool
    had_mamba: bool
    had_pingpong: bool
    cow_stamp: bool


@dataclass
class Snapshot:
    chunked_before: object
    pre: Dict[int, _Pre] = field(default_factory=dict)   # id(req) -> _Pre, rows-less reqs only

    def fresh(self, req) -> bool:
        return id(req) in self.pre


def take_snapshot(sched, reqs, chunked_before) -> Snapshot:
    snap = Snapshot(chunked_before=chunked_before)
    for r in reqs:
        if getattr(r, "req_pool_idx", None) is None:
            snap.pre[id(r)] = _Pre(
                had_row=False,
                had_mamba=getattr(r, "mamba_pool_idx", None) is not None,
                had_pingpong=getattr(r, "mamba_ping_pong_track_buffer", None) is not None,
                cow_stamp=bool(getattr(r, "mamba_slot_acquired_this_admission", False)),
            )
    return snap


def refusal_reason(sched, adder, reqs, snap) -> Optional[str]:
    """None = the batch can be handed back; otherwise the name of the reason it cannot."""
    from sglang.srt.environ import envs

    ps = getattr(sched, "ps", None)
    if int(getattr(ps, "tp_size", 1) or 1) > 1:
        return "tp>1: the verdict would need a group vote"
    if int(getattr(ps, "pp_size", 1) or 1) > 1 and int(getattr(ps, "pp_rank", 0) or 0) != 0:
        return "PP follower: the leader's hidden states are on the wire"
    try:
        from sglang.srt.weg2 import p_layer_split_runtime as _pls_rt

        if _pls_rt.active() is not None:
            return "p-layer-split row already queued for this pass"
    except Exception:  # noqa: BLE001 -- no runtime module: nothing armed
        pass
    if getattr(sched, "anchor_tails", None) or getattr(adder, "new_anchor_tails", None):
        return "anchor tails in the pass"
    if getattr(adder, "weg2_skip_extend_taken", False):
        return "skip-extend in the pass"
    tree = getattr(sched, "tree_cache", None)
    for api in ("_anchor_dec_skip", "dec_lock_ref", "loading_check"):
        if not hasattr(tree, api):
            return f"tree without {api}"
    stale = [r for r in reqs if not snap.fresh(r)]
    if len(stale) > 1 or any(r is not snap.chunked_before for r in stale):
        return "a request that already owned a row is not the resident chunked request"
    for r in reqs:
        if snap.fresh(r) and getattr(r, "session", None):
            return "session-held request"
    cap = int(envs.SGLANG_WEG2_OOM_ROLLBACK_MAX.get())
    counts = getattr(sched, "_weg2_oom_rollbacks", None) or {}
    over = [
        str(r.rid) for r in reqs if snap.fresh(r) and counts.get(str(r.rid), 0) >= cap
    ]
    if over:
        return "EXHAUSTED: rid(s) %s already rolled back %d times" % (",".join(over), cap)
    return None


def _unlock(tree, req) -> None:
    """The release ``cache_finished_req`` performs for the admission lock."""
    from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams

    params = DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None))
    tree._anchor_dec_skip(req, params)
    tree.dec_lock_ref(
        req.last_node, params, skip_swa=getattr(req, "swa_prefix_lock_released", False)
    )


def _give_back_slots(sched, fresh_reqs, snap) -> None:
    """Request rows, mamba slots, ping-pong buffers, the COW slot of the fresh requests."""
    pool = sched.req_to_token_pool
    tree = sched.tree_cache
    rows = [int(r.req_pool_idx) for r in fresh_reqs if r.req_pool_idx is not None]
    if hasattr(pool, "_rollback_alloc"):
        state_reqs = [
            r for r in fresh_reqs
            if not snap.pre[id(r)].had_mamba and getattr(r, "mamba_pool_idx", None) is not None
        ]
        pp_reqs = [
            r for r in fresh_reqs
            if not snap.pre[id(r)].had_pingpong
            and getattr(r, "mamba_ping_pong_track_buffer", None) is not None
        ]
        pool._rollback_alloc(fresh_reqs, rows, state_reqs, pp_reqs)
        from sglang.srt.mem_cache.common import release_admission_acquired_mamba_slot

        for r in fresh_reqs:
            pre = snap.pre[id(r)]
            if pre.had_mamba and pre.cow_stamp and getattr(r, "mamba_pool_idx", None) is not None:
                # alloc() reset the stamp when the batch took the slot over; it is ours again.
                r.mamba_slot_acquired_this_admission = True
                release_admission_acquired_mamba_slot(r, tree, site="oom_rollback")
    else:
        for r in fresh_reqs:
            if r.req_pool_idx is not None:
                pool.free(r)
    for r in fresh_reqs:
        r.mamba_cow_src_index = None
        r.mamba_needs_clear = False
        r.mamba_loadback_anchor_adopted = False


def _land_loads(tree) -> int:
    """Wait for every started host->device load and release its pins. Returns the count."""
    cc = getattr(tree, "cache_controller", None)
    if cc is None:
        return 0
    queue = list(getattr(cc, "ack_load_queue", ()) or ())
    for _start, finish, _acks in queue:
        finish.synchronize()
    tree.loading_check()
    return len(queue)


def rollback(sched, reqs, snap, adder, exc) -> None:
    """Hand the admitted batch back. Call only after ``refusal_reason`` returned None."""
    tree = sched.tree_cache
    fresh = [r for r in reqs if snap.fresh(r)]
    landed = _land_loads(tree)
    for r in reversed(fresh):
        _unlock(tree, r)
    _give_back_slots(sched, fresh, snap)

    post = sched.chunked_req
    if post is not None:
        post.inflight_middle_chunks = max(0, int(post.inflight_middle_chunks) - 1)
    sched.chunked_req = snap.chunked_before

    # The batch's requests go back to the head of the queue in the order they were admitted.
    sched.waiting_queue = list(fresh) + [q for q in sched.waiting_queue if q not in fresh]
    ps = getattr(sched, "ps", None)
    if int(getattr(ps, "pp_size", 1) or 1) > 1:
        sched._pp_admission_last_built_decision = None
        sched._pp_load_back_wire = None

    counts = getattr(sched, "_weg2_oom_rollbacks", None)
    if counts is None:
        counts = sched._weg2_oom_rollbacks = {}
    for r in fresh:   # the resident continuation is innocent: it is not counted
        counts[str(r.rid)] = counts.get(str(r.rid), 0) + 1
    sched._weg2_oom_rollback_n = n = getattr(sched, "_weg2_oom_rollback_n", 0) + 1
    logger.warning(
        "%s n=%d: the batch's extend did not fit (%s); %d fresh request(s) %s given back to the "
        "head of the waiting queue (admission lock released, row/mamba/COW slots returned), "
        "resident chunked_req=%s kept, %d load(s) landed before the pins were dropped. "
        "The rank lives; the next pass re-matches and re-prices.",
        MARK, n, str(exc).splitlines()[0] if str(exc) else "Prefill out of memory", len(fresh),
        [str(r.rid)[:16] for r in fresh], getattr(snap.chunked_before, "rid", None), landed,
    )
