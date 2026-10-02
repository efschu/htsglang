"""AP L15-11c: pure bindings from scheduler requests to retain_at_sleep.

Duck-typed adapter layer (getattr with defaults, no scheduler import, no
torch.cuda): translates Req-shaped objects into the callables that
l15_retain.retain_at_sleep expects, so tests can use SimpleNamespace fakes.

L15-12c-C2: l2_of yields (slot, gen) per held token -- the host rows of
the radix chain are SNAPSHOTTED at bind time (reset_keep nulls host_value
on kept nodes), mapped to arena page slots, with the generation read once
from the arena census (ArenaMHAHostPool.slot_gens).
"""

from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.weg2.l15_compact import owner_of
from sglang.srt.weg2.l15_hostlock import hold_sleep_refs
from sglang.srt.weg2.l15_shadow import candidates_from


def _seq_len(req) -> int:
    """Token span of a running req: prompt length + generated length."""
    prompt = getattr(req, "origin_input_ids", None) or ()
    out = getattr(req, "output_ids", None) or ()
    return len(prompt) + len(out)


def slots_of_req(req, req_to_token) -> Tuple[int, ...]:
    """The req's KV slots in token order, as plain Python ints.

    The held span is seqlen - 1: the last generated token has not been
    forwarded yet, so its slot is unwritten (schedule_batch.py:2821,
    offload_kv_cache, reads the same span). A 0 INSIDE the span is the
    padding slot (freed/never-written row): raise.
    """
    # L15-FIX-NOIDX: a req whose req_to_token row was already released has
    # req_pool_idx None (N3c 01.10.: int(None) raised a TypeError that killed
    # every retain round). ValueError = "this rid cannot be held", which
    # build_retain_kwargs skips per rid.
    if getattr(req, "req_pool_idx", None) is None:
        raise ValueError(
            f"req {getattr(req, 'rid', '?')!r} has no req_pool_idx "
            "(its req_to_token row is released)"
        )
    row = int(req.req_pool_idx)
    n = max(_seq_len(req) - 1, 0)
    span = req_to_token[row, :n]
    if hasattr(span, "tolist"):
        slots = tuple(int(x) for x in span.tolist())
    else:
        slots = tuple(int(x) for x in span)
    if 0 in slots:
        raise ValueError(
            "padding slot 0 inside the token span of req "
            f"{getattr(req, 'rid', row)!r} at row {row}"
        )
    return slots


def match_parked(req, tree_cache):
    """L15-FIX-PARKED: the held span of a PARKED req, from the radix tree.

    The D park retracts every running req with retain=True
    (d_park_runtime.park_running -> retract_all -> release_req ->
    release_kv_cache(is_insert=True) + reset_for_retract): the computed span
    is INSERTED into the tree and the req keeps no device handle -- its
    req_to_token row is released (req_pool_idx None), prefix_indices is
    empty, last_node and mamba_pool_idx are None (N3f 01.10. 22:21Z: the
    shadow planned n=2 from token counts, retain skipped both rids "no
    req_pool_idx"). The KV indices are still on the device in the tree, so
    they are read back by one match of the req's own tokens (seqlen - 1, the
    last token has no KV yet) with cow_mamba=False (no mamba slot is
    allocated; the match only refreshes LRU/access time).

    Returns (slots, node, anchor): the matched device KV indices, the last
    device node and the node's mamba checkpoint slot. Raises ValueError when
    nothing usable is on the device (the rid is then skipped like any other
    unholdable req)."""
    from array import array

    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    rid = getattr(req, "rid", "?")
    toks = list(getattr(req, "origin_input_ids", None) or ()) + list(
        getattr(req, "output_ids", None) or ()
    )
    span = max(len(toks) - 1, 0)
    if span == 0:
        raise ValueError(f"parked req {rid!r} has no token span")
    res = tree_cache.match_prefix(
        MatchPrefixParams(
            # array('q'): the tree's stored keys are array('q') and
            # RadixKey.match asserts the same container type (N3h 22:52Z).
            key=RadixKey(token_ids=array("q", toks[:span]),
                         extra_key=getattr(req, "extra_key", None)),
            cow_mamba=False,
        )
    )
    di = getattr(res, "device_indices", None)
    slots = tuple(int(x) for x in (di.tolist() if hasattr(di, "tolist") else (di or ())))
    node = getattr(res, "last_device_node", None)
    if not slots or node is None:
        raise ValueError(f"parked req {rid!r}: no device span in the tree")
    if 0 in slots:
        raise ValueError(f"parked req {rid!r}: padding slot 0 inside the matched span")
    try:
        mv = node.component_data[ComponentType.MAMBA].value
    except (AttributeError, KeyError, IndexError, TypeError):
        mv = None
    if mv is None or len(mv) == 0:
        raise ValueError(f"parked req {rid!r}: matched node has no mamba checkpoint")
    anchor = int(mv.tolist()[0] if hasattr(mv, "tolist") else mv[0])
    if anchor == 0:
        raise ValueError(f"parked req {rid!r}: matched anchor is padding slot 0")
    return slots, node, anchor


def sleep_epoch(sched) -> int:
    """L15-FIX-EPOCH: the epoch a D sleep's retain round is stamped with.

    `_weg2_vote_epoch` is PP0's idle-vote counter and never moves on D (N3f:
    every D sleep logged epoch=0). The front's flip index rides on the
    release request (`<boot>.<flip>`, weight_updater._weg2_flip_index_of);
    release_memory_occupation stashes it as `_l15_sleep_flip` right before
    its flush. Group-uniform (one front number per flip); falls back to the
    old counter when absent."""
    flip = getattr(sched, "_l15_sleep_flip", None)
    try:
        if flip is not None and int(flip) >= 0:
            return int(flip)
    except (TypeError, ValueError):
        pass
    return int(getattr(sched, "_weg2_vote_epoch", 0) or 0)


def anchor_slot_of_req(req) -> int:
    """The req's mamba anchor slot; a missing (None) or padding (0) anchor
    raises -- one req without a holdable anchor skips the whole retain round
    (benign: this runs before step 3, nothing is touched yet)."""
    idx = getattr(req, "mamba_pool_idx", None)
    rid = getattr(req, "rid", "?")
    if idx is None:
        raise ValueError(f"req {rid!r} has no mamba_pool_idx")
    try:
        anchor = int(idx)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"req {rid!r} has a non-scalar mamba_pool_idx: {idx!r}"
        ) from exc
    if anchor == 0:
        raise ValueError(f"req {rid!r} anchor slot is padding slot 0")
    return anchor


def node_of_req(req):
    """The req's radix-tree node; a missing (None) node raises -- the retain
    round skips (benign, pre-step-3: nothing is touched yet)."""
    node = getattr(req, "last_node", None)
    if node is None:
        raise ValueError(f"req {getattr(req, 'rid', '?')!r} has no last_node")
    return node


class L15UnsupportedTreeError(RuntimeError):
    """L15-12c-F9 hard gate: the radix tree carries a component that the
    L1.5 retain/compact/rewrite chain does not remap (anything beyond
    FULL+MAMBA). The scheduler hook catches it pre-move and flushes."""


_L15_TREE_COMPONENTS = (ComponentType.FULL, ComponentType.MAMBA)


def _unsupported_components(node) -> list:
    """Component types on this node that L1.5 retain cannot remap.

    component_data is a FIXED-length list (one ComponentData per
    ComponentType, unified_radix_cache.py:435), so an unused SWA slot
    EXISTS without the tree using SWA: the registered tuple
    node.tree_components is authoritative on the real tree; duck-typed
    fakes without it fall back to scanning the non-FULL/MAMBA slots for
    carried data (a value or host_value set means the component is live).
    """
    found = []
    reg = getattr(node, "tree_components", None)
    if reg is not None:
        for ct in reg:
            if ct not in _L15_TREE_COMPONENTS:
                found.append(ct)
        return found
    cds = getattr(node, "component_data", None)
    if cds is None:
        return found
    for ct in ComponentType:
        if ct in _L15_TREE_COMPONENTS:
            continue
        try:
            cd = cds[ct]
        except (IndexError, KeyError, TypeError):
            continue
        if cd is not None and (
            getattr(cd, "value", None) is not None
            or getattr(cd, "host_value", None) is not None
        ):
            found.append(ct)
    return found


def chain_host_rows(node) -> Tuple[int, ...]:
    """The host rows of the chain (see :func:`chain_host_rows_ex`)."""
    return chain_host_rows_ex(node)[0]


def chain_host_rows_ex(node) -> Tuple[Tuple[int, ...], Tuple[Optional[int], ...]]:
    """L15-12c-C2: the host rows of the radix chain root -> node, in token
    order. The walk goes last_node -> root; each node contributes its
    component_data[ComponentType.FULL].host_value (the host row ids of its
    own tokens). Must be read at BIND time: reset_keep nulls host_value on
    the kept nodes (unified_radix_cache.py).

    L15-12c-F4: a node WITHOUT a host_value (write-pending, or never backed
    up) keeps its token POSITIONS: it contributes len(its tokens)
    placeholders -1, so a hole mid-chain does not shift every later node's
    rows onto the wrong token positions (l2_of would then map tokens to a
    foreign (slot, gen)). The -1 rows map to (slot, gen) = (-1, -1) in
    build_retain_kwargs (row < staging_rows), and the wake's gen check
    drops those tokens. Token count: the node's own component value (its
    device slot ids), else its radix key; if NEITHER exists (odd fake
    node), fall back to the old positional skip -- nothing sensible to
    count.
    """
    chunks = []
    recs = []
    cur = node
    while cur is not None:
        try:
            cd = cur.component_data[ComponentType.FULL]
        except (AttributeError, KeyError, IndexError, TypeError):
            cd = None  # duck-typed: a fake/odd node contributes no rows
        hv = getattr(cd, "host_value", None) if cd is not None else None
        sh = getattr(cur, "_weg2_l2_shadow", None)
        if hv is not None and len(hv):
            vals = hv.tolist() if hasattr(hv, "tolist") else hv
            chunks.append([int(x) for x in vals])
            recs.append([None] * len(vals))
        elif (sh is not None and len(sh) == 2 and len(sh[0]) == len(sh[1])
              and len(sh[0]) == _node_tokens(cd, cur)):
            # L15-L2-SHADOW: the #248 release dropped this node's KV host rows
            # but recorded them with their arena generation -- adopted here,
            # the caller keeps a row only where the slot still has that gen
            chunks.append([int(x) for x in sh[0]])
            recs.append([int(g) for g in sh[1]])
        else:
            n_tok = None
            val = getattr(cd, "value", None) if cd is not None else None
            if val is not None:
                try:
                    n_tok = len(val)
                except TypeError:  # scalar/unsized value
                    n_tok = None
            if not n_tok:  # value missing or empty: fall back to the key
                try:
                    n_tok = len(cur.key)
                except (AttributeError, TypeError):  # no key or key=None
                    n_tok = None
            if n_tok:
                chunks.append([-1] * n_tok)
                recs.append([None] * n_tok)
            # else: no host rows, no token count -> old positional skip
        cur = getattr(cur, "parent", None)
    rows: list = []
    rec: list = []
    for chunk, rc in zip(reversed(chunks), reversed(recs)):
        rows.extend(chunk)
        rec.extend(rc)
    return tuple(rows), tuple(rec)


def _node_tokens(cd, node) -> int:
    """The node's own token count: its device value, else its radix key."""
    for v in (getattr(cd, "value", None) if cd is not None else None,
              getattr(node, "key", None)):
        if v is None:
            continue
        try:
            return len(v)
        except TypeError:
            continue
    return -1


def anchor_host_row(node, anchor_slot: int) -> int:
    """L15-12c-E2a: the host row carrying the req's mamba ANCHOR state.

    Walk last_node -> root; the node whose component_data[ComponentType.
    MAMBA] device value contains ``anchor_slot`` contributes the first row
    of its host_value (one state per node, one state per arena slot). The
    walk starts at the node itself: at sleep the anchor lives in the chain
    (the same node whose value holds it, host_value its L2 copy). No such
    node, or one without a host_value (never written to L2), answers -1.
    Read at BIND time like chain_host_rows: reset_keep nulls host_value
    on the kept nodes afterwards.
    """
    cur = node
    while cur is not None:
        try:
            cd = cur.component_data[ComponentType.MAMBA]
        except (AttributeError, KeyError, IndexError, TypeError):
            cd = None  # duck-typed/odd node: keep walking
        val = getattr(cd, "value", None) if cd is not None else None
        if val is not None:
            try:
                ids = [
                    int(x)
                    for x in (val.tolist() if hasattr(val, "tolist") else val)
                ]
            except TypeError:  # scalar value: a one-element list
                ids = [int(val)]
            if anchor_slot in ids:
                hv = getattr(cd, "host_value", None)
                if hv is None or not len(hv):
                    return -1
                return int(hv.tolist()[0] if hasattr(hv, "tolist") else hv[0])
        cur = getattr(cur, "parent", None)
    return -1


def _group_entry(pool, name: str):
    """L15-FIX-HOSTGROUP: the host pool behind a HostPoolGroup entry.

    On the hybrid 27B the cache controller's ``mem_pool_host`` is a
    HostPoolGroup (hybrid_pool_assembler), which has no ``slot_gens`` /
    ``staging_rows`` of its own; the arena pools are its KV / MAMBA entries.
    A plain pool (no ``entry_map``) is returned unchanged; a group without
    that entry gives None.
    """
    emap = getattr(pool, "entry_map", None)
    if not isinstance(emap, dict):
        return pool
    entry = emap.get(name)
    return None if entry is None else getattr(entry, "host_pool", None)


def live_host_pools(tree_cache):
    """(kv_host_pool, mamba_host_pool) of the live tree, either may be None.

    ONE resolver for the sleep bind and the wake refill: the controller's
    ``mem_pool_host`` is unwrapped through its HostPoolGroup entries; the
    mamba pool falls back to a controller- or tree-level ``mamba_pool_host``
    (the older hybrid mamba cache sets it on the tree) when there is no
    group entry.
    """
    cc = getattr(tree_cache, "cache_controller", None)
    raw = getattr(cc, "mem_pool_host", None)
    kv = _group_entry(raw, "kv")
    mamba = _group_entry(raw, "mamba")
    if mamba is None or mamba is raw:
        mamba = getattr(cc, "mamba_pool_host", None)
    if mamba is None:
        mamba = getattr(tree_cache, "mamba_pool_host", None)
    return kv, mamba


def _slot_gens_or_minus_one(pool, slots, log, what: str):
    """One census call; a pool without ``slot_gens`` (plain MambaPoolHost on
    a form-A worker) gives -1 per slot -- that span is held but not
    L2-refillable -- instead of failing the whole retain round."""
    fn = getattr(pool, "slot_gens", None)
    if fn is None:
        log("L15-L2-GENS %s pool %s has no slot_gens: %d slot(s) gen -1 "
            "(held, not L2-refillable)" % (what, type(pool).__name__, len(slots)))
        return [-1] * len(slots)
    return fn(slots)


def _owner_counts(slots, prefix) -> Tuple[int, ...]:
    """Per-rank count of ``slots`` under the owner rule (rank r owns slot L
    iff prefix[r] <= L % S < prefix[r+1]); a slot with no owner raises like
    l15_compact.owner_of."""
    import numpy as np

    pre = np.asarray([int(x) for x in prefix], dtype=np.int64)
    n = len(pre) - 1
    a = np.asarray(slots, dtype=np.int64).reshape(-1)
    if a.size == 0:
        return tuple([0] * n)
    S = int(pre[-1])
    r = np.searchsorted(pre, a % S, side="right") - 1
    if int(r.min()) < 0 or int(r.max()) >= n:
        bad = int(a[(r < 0) | (r >= n)][0])
        owner_of(bad, prefix)        # raises the named ValueError
    return tuple(int(x) for x in np.bincount(r, minlength=n)[:n])


def build_retain_kwargs(
    reqs: Iterable,
    req_to_token,
    *,
    caps_rows_by_rank,
    cap_anchor_slots: int,
    prefix,
    rank: int,
    epoch: int,
    pid: int,
    kv_buffers,
    mamba_buffers,
    allocator,
    reset_keep: Callable[[list], None],
    set_keep: Callable[[object, Tuple], None],
    manifest_path: str,
    log: Callable[[str], None],
    mamba_allocator=None,
    host_pool=None,
    mamba_host_pool=None,
    tree_cache=None,
    hold_sink: Optional[Callable[[tuple], None]] = None,
) -> Dict:
    """Assemble the whole retain_at_sleep keyword set from live reqs.

    Geometry: rows_by_rank = the EXACT per-rank owned count of the real
    slots (l15_compact.owner_of over slots_of_req); anchor_depth = kv_depth
    = seqlen - 1 (the last token has no KV yet); kind/last_active are
    duck-typed (l15_kind / l15_last_active) until the hook lands.
    """
    by_rid = {}
    entries = []
    l2_rows = []  # (rid, chain host rows truncated to the KV span)
    shadow_gens: Dict[str, Tuple] = {}  # L15-L2-SHADOW: rid -> recorded gen per token
    # L15-12c-E2a: (rid, mamba anchor host row, -1 when absent) -- the
    # anchor state's L2 identity, snapshot at bind like the KV rows.
    anchor_rows = []
    skipped = []  # L15-FIX-NOIDX: (rid, reason) of reqs that cannot be held
    parked_by_rid = {}  # L15-FIX-PARKED: rid -> (slots, node, anchor)
    for req in reqs:
        rid = str(req.rid)
        # L15-FIX-NOIDX: a req without a holdable token span (no
        # req_to_token row, or the padding slot inside its span) is skipped
        # per rid -- before this fix its error escaped and the whole round
        # flushed (N3c: 21/21 sleeps "failed before the move").
        _pk = None  # L15-FIX-PARKED: (slots, node, anchor) of a parked req
        try:
            if (getattr(req, "req_pool_idx", None) is None
                    and tree_cache is not None):
                _pk = match_parked(req, tree_cache)
                _slots = _pk[0]
            else:
                _slots = slots_of_req(req, req_to_token)
        except ValueError as exc:
            skipped.append((rid, str(exc)))
            continue
        by_rid[rid] = req
        if _pk is not None:
            parked_by_rid[rid] = _pk
        seq_len = _seq_len(req)
        # KV exists for seqlen - 1 tokens only (schedule_batch.py:2821); a
        # parked req holds exactly what its tree match returned.
        span = len(_slots) if _pk is not None else max(seq_len - 1, 0)
        # rows_by_rank = the EXACT owned count of the req's real slots
        # (owner_of over slots_of_req), not a proportional estimate:
        # retain admits against caps with these rows, while compact_plan
        # reserves by the exact owned count -- a proportional split can
        # over-admit past a cap on the rounding residue (audit item 11).
        # L15-12c-C2: snapshot the chain's host rows NOW (reset_keep nulls
        # host_value later); a node missing its last_node skips l2 (retain
        # skips the whole round for that rid anyway).
        try:
            _node = _pk[1] if _pk is not None else node_of_req(req)
            # L15-12c-F9: hard gate BEFORE anything is assembled -- a tree
            # with a third radix component (SWA or other) would keep
            # un-remapped, un-kept state on the kept nodes. RuntimeError,
            # so the ValueError skip below cannot swallow it; the hook
            # catches it pre-move and flushes as today.
            _extra = _unsupported_components(_node)
            if _extra:
                _names = "/".join(getattr(ct, "name", str(ct)) for ct in _extra)
                raise L15UnsupportedTreeError(
                    f"req {rid!r}: radix tree carries unsupported "
                    f"component(s) {_names}; L1.5 retain supports "
                    "FULL+MAMBA trees only (27B/NF); SWA or other "
                    "components -> no retain"
                )
            _rows, _rec = chain_host_rows_ex(_node)
            _rows, _rec = _rows[: len(_slots)], _rec[: len(_slots)]
        except ValueError:
            _rows, _rec = (), ()
        if _rows:
            l2_rows.append((rid, _rows))
            if any(g is not None for g in _rec):
                shadow_gens[rid] = _rec
        # L15-12c-E2a: the anchor's host row from the chain node that
        # carries the req's anchor device slot; best-effort (-1 on any
        # missing piece -- the anchor columns then read (-1, -1)).
        try:
            anchor_rows.append(
                (rid, anchor_host_row(_pk[1], _pk[2]) if _pk is not None
                 else anchor_host_row(node_of_req(req),
                                      anchor_slot_of_req(req)))
            )
        except ValueError:
            anchor_rows.append((rid, -1))
        # L15-FLIPCOST-4 (N4f bind 538-945 ms with nothing held): owner_of
        # per slot re-validated the prefix every call (~250k calls per
        # sleep); one vectorised owner count, same rule
        _n = len(prefix) - 1
        _owned_counts = _owner_counts(_slots, prefix)
        entries.append(
            {
                "rid": rid,
                "kind": getattr(req, "l15_kind", "served") or "served",
                "last_active": float(
                    getattr(req, "l15_last_active", 0.0) or 0.0
                ),
                "rows_by_rank": _owned_counts,
                "anchor_depth": span,
                "kv_depth": span,
            }
        )
    if skipped:
        log(
            "L15-RETAIN skipped %d req(s) without a holdable span: %s"
            % (len(skipped), "; ".join("%s (%s)" % (r, why) for r, why in skipped[:4]))
        )
    candidates = candidates_from(entries)

    # L15-12c-C2: host row -> (arena page slot, generation). The pool is the
    # kwarg (tests) or the live one reached through the bound reset_keep
    # (scheduler passes tree_cache.reset_keep -> cache_controller.mem_pool_host).
    # Host ids: [0, S) staging rows (no L2 copy -> -1), [S, S + A*P) arena
    # token ids, slot = (row - S) // P (P == 1 is the 27B form); the draft
    # role maps rows through row_slot instead. Gens come from ONE census.
    _live_kv, _live_mamba = live_host_pools(getattr(reset_keep, "__self__", None))
    pool = _group_entry(host_pool, "kv") if host_pool is not None else _live_kv
    l2_by_rid: Dict[str, Tuple[Tuple, Tuple]] = {}
    l2_lanes_by_rid: Dict[str, Tuple[int, ...]] = {}
    if pool is not None and l2_rows:
        _s = int(getattr(pool, "staging_rows", 0))
        _p = max(1, int(getattr(pool, "_arena_page_tokens", 1)))
        _row_slot = getattr(pool, "row_slot", None)
        _slot_of = {}
        # L15-12c-P1: the lane inside the arena page, lane = (row - S) % P,
        # recorded next to the slot = (row - S) // P. Staging rows: lane -1.
        # The row_slot map (draft role) is not page-addressed: lane 0 at
        # P == 1 (the only lane), unknown (-1) at P > 1.
        _lane_of = {}
        import numpy as _np

        for rid, rows in l2_rows:
            if _row_slot is None:
                # L15-FLIPCOST (N4a bind 330-370 ms): vectorised, same rule
                _r = _np.asarray(rows, dtype=_np.int64)
                _off = _r - _s
                _stg = _r < _s
                _per = _np.where(_stg, -1, _off // _p)
                _ln = _np.where(_stg, -1, _off % _p)
                _slot_of[rid] = _per.tolist()
                _lane_of[rid] = _ln.tolist()
                continue
            per = []
            lanes = []
            for r in rows:
                if r < _s:
                    per.append(-1)
                    lanes.append(-1)
                else:
                    per.append(int(_row_slot.get(r, -1)))
                    lanes.append(0 if _p == 1 else -1)
            _slot_of[rid] = per
            _lane_of[rid] = lanes
        _all = (_np.concatenate([_np.asarray(v, dtype=_np.int64) for v in _slot_of.values()])
                if _slot_of else _np.zeros(0, dtype=_np.int64))
        uniq = _np.unique(_all[_all >= 0]).tolist()
        gen_of = {}
        if uniq:
            for s, g in zip(uniq, _slot_gens_or_minus_one(pool, uniq, log, "kv")):
                gen_of[int(s)] = int(g)
        for rid, per in _slot_of.items():
            _g = gen_of.get
            gens = [_g(s, -1) for s in per]
            rec = shadow_gens.get(rid)
            if rec is not None:
                # L15-L2-SHADOW: a shadow row counts only while its slot still
                # carries the generation recorded at the #248 release (the
                # page P wrote and D loaded); a re-claimed slot is unbacked
                per = list(per)
                live = shadow = valid = 0
                for i, r in enumerate(rec):
                    if r is None:
                        live += 1
                        continue
                    shadow += 1
                    if per[i] >= 0 and int(gens[i]) >= 0 and int(gens[i]) == int(r):
                        valid += 1
                    else:
                        per[i], gens[i] = -1, -1
                log("L15-L2-SHADOW-ADOPT rid=%s tokens=%d live=%d shadow=%d valid=%d stale=%d"
                    % (rid, len(per), live, shadow, valid, shadow - valid))
            l2_by_rid[rid] = (tuple(per), tuple(gens))
            l2_lanes_by_rid[rid] = tuple(_lane_of[rid])

    # L15-12c-E2a: anchor host row -> (mamba arena slot, generation). One
    # state per slot: slot = row - staging_rows; a row below staging_rows
    # is staging-only -> (-1, -1), as is any row without a pool to ask.
    # The pool is the kwarg (tests) or the live one on the bound
    # reset_keep's cache_controller: ``mamba_pool_host``, the controller-
    # side name of the component's _mamba_pool_host
    # (hybrid_pool_assembler._COMPONENT_HOST_ATTR).
    # the controller holds the HostPoolGroup; its MAMBA entry is the anchor
    # pool (live_host_pools; L15-FIX-HOSTGROUP)
    mpool = mamba_host_pool if mamba_host_pool is not None else _live_mamba
    anchor_l2_by_rid: Dict[str, Tuple[int, int]] = {}
    if anchor_rows and mpool is not None:
        _ms = int(getattr(mpool, "staging_rows", 0))
        _slot_row = {rid: (r - _ms if r >= _ms else -1)
                     for rid, r in anchor_rows}
        _want = sorted({s for s in _slot_row.values() if s >= 0})
        _agen: Dict[int, int] = {}
        if _want:
            for s, g in zip(_want, _slot_gens_or_minus_one(mpool, _want, log, "mamba")):
                _agen[int(s)] = int(g)
        for rid, s in _slot_row.items():
            anchor_l2_by_rid[rid] = (
                s, int(_agen.get(s, -1)) if s >= 0 else -1
            )

    def node_of(rid: str):
        if rid in parked_by_rid:
            return parked_by_rid[rid][1]
        return node_of_req(by_rid[rid])

    def slots_of(rid: str) -> Tuple[int, ...]:
        if rid in parked_by_rid:
            return parked_by_rid[rid][0]
        return slots_of_req(by_rid[rid], req_to_token)

    def anchor_slot_of(rid: str) -> int:
        if rid in parked_by_rid:
            return parked_by_rid[rid][2]
        return anchor_slot_of_req(by_rid[rid])

    def l2_of(rid: str) -> Tuple[Tuple, Tuple]:
        # L15-12c-C2: the bind-time snapshot mapped above; ((), ()) keeps the
        # old empty columns when no host pool could be reached.
        return l2_by_rid.get(rid, ((), ()))

    def l2_lanes_of(rid: str) -> Tuple[int, ...]:
        # L15-12c-P1: the lane inside each held token's L2 page (parallel to
        # l2_of's slots; -1 = staging). () when the pool was unreachable or
        # the rid is unknown -- retain keeps the empty column.
        return l2_lanes_by_rid.get(rid, ())

    # L15-HOSTLOCK (LCHOST defect 2): pin this rank's held L2 slots in its
    # OWN host pools' arena before retain's step (5) reset_keep hands the
    # kept chains' references back; the record of what was pinned goes to
    # hold_sink (the scheduler) so the wake act can release it. No hold_sink
    # (master off / hook unwired) -> no callable -> no reference taken.
    _hold_kv = pool if pool is not None else _live_kv
    _hold_mamba = mpool

    def _hold_l2_refs(kv_slots, anchor_slots):
        rec = hold_sleep_refs(_hold_kv, _hold_mamba, kv_slots, anchor_slots,
                              log)
        if rec is not None:
            hold_sink(rec)

    hold_l2_refs = _hold_l2_refs if hold_sink is not None else None

    return {
        "candidates": candidates,
        "node_of": node_of,
        "slots_of": slots_of,
        "anchor_slot_of": anchor_slot_of,
        "l2_of": l2_of,
        "l2_lanes_of": l2_lanes_of,
        # L15-12c-E2a: the anchor's L2 identity, (-1, -1) when absent.
        "anchor_l2_of": lambda rid: anchor_l2_by_rid.get(rid, (-1, -1)),
        # L15-HOSTLOCK: None unless the hook passed a hold_sink (master on).
        "hold_l2_refs": hold_l2_refs,
        "caps_rows_by_rank": caps_rows_by_rank,
        "cap_anchor_slots": cap_anchor_slots,
        "prefix": prefix,
        "rank": rank,
        "epoch": epoch,
        "pid": pid,
        "kv_buffers": kv_buffers,
        "mamba_buffers": mamba_buffers,
        "allocator": allocator,
        "mamba_allocator": mamba_allocator,
        "reset_keep": reset_keep,
        "set_keep": set_keep,
        "manifest_path": manifest_path,
        "log": log,
    }


def _remap_slots(value, slot_map: Dict[int, int]):
    """Element-wise old->new slot remap of a component value tensor.

    Single pass, so a chained map (a -> b, b -> c) can never double-apply:
    every element is read once and mapped once. ``None``, empty and
    non-tensor values (the root node carries a bare list) pass through; a
    slot missing from the map was not moved and stays put. dtype and
    device are preserved via ``empty_like``.
    """
    if value is None or not torch.is_tensor(value) or value.numel() == 0:
        return value
    # L15-FLIPCOST (N3y: D->P flip 7-12 s instead of 2.4 s): the old loop
    # read and wrote ONE element per step -- on a CUDA value that is a
    # device sync plus a kernel per token (~250k tokens per sleep, ~4 s on
    # TP1 between L15-L2-ALIGN and L15-HOSTLOCK). One D2H, the dict map on
    # host, one H2D.
    vals = value.flatten().tolist()
    get = slot_map.get
    mapped = [get(v, v) for v in vals]
    return torch.tensor(mapped, dtype=value.dtype,
                        device=value.device).reshape(value.shape)


def rewrite_tree_chain(
    node,
    kv_map: Dict[int, int],
    anchor_map: Dict[int, int],
    visited: Optional[set] = None,
) -> None:
    """L15-11d: make a REAL UnifiedTreeNode chain follow a finished retain
    round, whose KV rows and mamba anchor slots have moved (l15_compact).

    The old step (4) of retain_at_sleep wrote ``node.kv_slots`` /
    ``node.anchor_slot`` attributes that exist only on the unit-test fakes;
    the real UnifiedTreeNode has neither field, so the write was silently
    accepted and never read -- the next prefix hit would have read foreign
    KV. On the real tree a request's KV indices live in
    ``component_data[ComponentType.FULL].value`` of EVERY node on the
    chain root -> req.last_node (each node holds the slot indices of its
    own tokens), and the GDN anchor slot lives in the mamba component's
    value on the anchor node; both are remapped here, element-wise,
    keeping dtype/device.

    ``visited`` is a set of built-in ``id(node)`` shared across ALL held
    requests of one retain call (UnifiedTreeNode.id is an int counter,
    NOT the identity). Chains share prefixes, so without it a shared
    prefix node would be remapped once per request that extends it; with
    it, every node is remapped exactly once. ``None`` builds a private
    set (a single standalone call).

    The scheduler hook passes this as ``rewrite_tree=``; unit tests pass
    a recorder with the same (node, kv_map, anchor_map, visited) shape.
    """
    if visited is None:
        visited = set()
    cur = node
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        cur.component_data[ComponentType.FULL].value = _remap_slots(
            cur.component_data[ComponentType.FULL].value, kv_map
        )
        if len(cur.component_data) > int(ComponentType.MAMBA):
            # L15-FIX-MAMBA-ALIAS (N3l 02.10. 02:29:57Z): only the HELD
            # anchors survive the sleep -- the mamba allocator is re-armed
            # with exactly their compacted slots (retain step 6), every
            # other slot goes back to the free list. A chain node whose
            # mamba value is not a held anchor (an intermediate checkpoint
            # of the same prefix) must therefore LOSE its mamba value, or
            # the tree names a free slot (#924 MAMBA SLOT ALIASING,
            # free_and_cached > 0 -> the idle leak check kills D).
            # ``anchor_map`` carries every held anchor old->new (identity
            # for an anchor that did not move).
            mv = cur.component_data[ComponentType.MAMBA].value
            if (mv is not None and torch.is_tensor(mv) and mv.numel() > 0
                    and not all(int(x) in anchor_map
                                for x in mv.flatten().tolist())):
                cur.component_data[ComponentType.MAMBA].value = None
            else:
                cur.component_data[ComponentType.MAMBA].value = _remap_slots(
                    mv, anchor_map
                )
        cur = getattr(cur, "parent", None)
