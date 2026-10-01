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
    cur = node
    while cur is not None:
        try:
            cd = cur.component_data[ComponentType.FULL]
        except (AttributeError, KeyError, IndexError, TypeError):
            cd = None  # duck-typed: a fake/odd node contributes no rows
        hv = getattr(cd, "host_value", None) if cd is not None else None
        if hv is not None and len(hv):
            vals = hv.tolist() if hasattr(hv, "tolist") else hv
            chunks.append([int(x) for x in vals])
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
            # else: no host rows, no token count -> old positional skip
        cur = getattr(cur, "parent", None)
    rows: list = []
    for chunk in reversed(chunks):
        rows.extend(chunk)
    return tuple(rows)


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
    # L15-12c-E2a: (rid, mamba anchor host row, -1 when absent) -- the
    # anchor state's L2 identity, snapshot at bind like the KV rows.
    anchor_rows = []
    skipped = []  # L15-FIX-NOIDX: (rid, reason) of reqs that cannot be held
    for req in reqs:
        rid = str(req.rid)
        # L15-FIX-NOIDX: a req without a holdable token span (no
        # req_to_token row, or the padding slot inside its span) is skipped
        # per rid -- before this fix its error escaped and the whole round
        # flushed (N3c: 21/21 sleeps "failed before the move").
        try:
            _slots = slots_of_req(req, req_to_token)
        except ValueError as exc:
            skipped.append((rid, str(exc)))
            continue
        by_rid[rid] = req
        seq_len = _seq_len(req)
        # KV exists for seqlen - 1 tokens only (schedule_batch.py:2821).
        span = max(seq_len - 1, 0)
        # rows_by_rank = the EXACT owned count of the req's real slots
        # (owner_of over slots_of_req), not a proportional estimate:
        # retain admits against caps with these rows, while compact_plan
        # reserves by the exact owned count -- a proportional split can
        # over-admit past a cap on the rounding residue (audit item 11).
        # L15-12c-C2: snapshot the chain's host rows NOW (reset_keep nulls
        # host_value later); a node missing its last_node skips l2 (retain
        # skips the whole round for that rid anyway).
        try:
            _node = node_of_req(req)
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
            _rows = chain_host_rows(_node)[: len(_slots)]
        except ValueError:
            _rows = ()
        if _rows:
            l2_rows.append((rid, _rows))
        # L15-12c-E2a: the anchor's host row from the chain node that
        # carries the req's anchor device slot; best-effort (-1 on any
        # missing piece -- the anchor columns then read (-1, -1)).
        try:
            anchor_rows.append(
                (rid, anchor_host_row(node_of_req(req),
                                      anchor_slot_of_req(req)))
            )
        except ValueError:
            anchor_rows.append((rid, -1))
        _owned = tuple(owner_of(s, prefix) for s in _slots)
        _n = len(prefix) - 1
        entries.append(
            {
                "rid": rid,
                "kind": getattr(req, "l15_kind", "served") or "served",
                "last_active": float(
                    getattr(req, "l15_last_active", 0.0) or 0.0
                ),
                "rows_by_rank": tuple(
                    sum(1 for o in _owned if o == r) for r in range(_n)
                ),
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
    pool = host_pool
    if pool is None:
        _rc = getattr(reset_keep, "__self__", None)
        pool = getattr(getattr(_rc, "cache_controller", None), "mem_pool_host", None)
    l2_by_rid: Dict[str, Tuple[Tuple, Tuple]] = {}
    if pool is not None and l2_rows:
        _s = int(getattr(pool, "staging_rows", 0))
        _p = max(1, int(getattr(pool, "_arena_page_tokens", 1)))
        _row_slot = getattr(pool, "row_slot", None)
        _slot_of = {}
        for rid, rows in l2_rows:
            per = []
            for r in rows:
                if r < _s:
                    per.append(-1)
                elif _row_slot is not None:
                    per.append(int(_row_slot.get(r, -1)))
                else:
                    per.append((r - _s) // _p)
            _slot_of[rid] = per
        uniq = sorted({s for per in _slot_of.values() for s in per if s >= 0})
        gen_of = {}
        if uniq:
            for s, g in zip(uniq, pool.slot_gens(uniq)):
                gen_of[int(s)] = int(g)
        for rid, per in _slot_of.items():
            l2_by_rid[rid] = (
                tuple(per),
                tuple(int(gen_of.get(s, -1)) for s in per),
            )

    # L15-12c-E2a: anchor host row -> (mamba arena slot, generation). One
    # state per slot: slot = row - staging_rows; a row below staging_rows
    # is staging-only -> (-1, -1), as is any row without a pool to ask.
    # The pool is the kwarg (tests) or the live one on the bound
    # reset_keep's cache_controller: ``mamba_pool_host``, the controller-
    # side name of the component's _mamba_pool_host
    # (hybrid_pool_assembler._COMPONENT_HOST_ATTR).
    mpool = mamba_host_pool
    if mpool is None:
        _mc = getattr(getattr(reset_keep, "__self__", None),
                      "cache_controller", None)
        mpool = getattr(_mc, "mamba_pool_host", None)
    anchor_l2_by_rid: Dict[str, Tuple[int, int]] = {}
    if anchor_rows and mpool is not None:
        _ms = int(getattr(mpool, "staging_rows", 0))
        _slot_row = {rid: (r - _ms if r >= _ms else -1)
                     for rid, r in anchor_rows}
        _want = sorted({s for s in _slot_row.values() if s >= 0})
        _agen: Dict[int, int] = {}
        if _want:
            for s, g in zip(_want, mpool.slot_gens(_want)):
                _agen[int(s)] = int(g)
        for rid, s in _slot_row.items():
            anchor_l2_by_rid[rid] = (
                s, int(_agen.get(s, -1)) if s >= 0 else -1
            )

    def node_of(rid: str):
        return node_of_req(by_rid[rid])

    def slots_of(rid: str) -> Tuple[int, ...]:
        return slots_of_req(by_rid[rid], req_to_token)

    def anchor_slot_of(rid: str) -> int:
        return anchor_slot_of_req(by_rid[rid])

    def l2_of(rid: str) -> Tuple[Tuple, Tuple]:
        # L15-12c-C2: the bind-time snapshot mapped above; ((), ()) keeps the
        # old empty columns when no host pool could be reached.
        return l2_by_rid.get(rid, ((), ()))

    return {
        "candidates": candidates,
        "node_of": node_of,
        "slots_of": slots_of,
        "anchor_slot_of": anchor_slot_of,
        "l2_of": l2_of,
        # L15-12c-E2a: the anchor's L2 identity, (-1, -1) when absent.
        "anchor_l2_of": lambda rid: anchor_l2_by_rid.get(rid, (-1, -1)),
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
    out = torch.empty_like(value.flatten())
    flat = value.flatten()
    for i in range(flat.numel()):
        old = int(flat[i])
        out[i] = slot_map.get(old, old)
    return out.reshape(value.shape)


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
            cur.component_data[ComponentType.MAMBA].value = _remap_slots(
                cur.component_data[ComponentType.MAMBA].value, anchor_map
            )
        cur = getattr(cur, "parent", None)
