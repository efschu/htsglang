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


def chain_host_rows(node) -> Tuple[int, ...]:
    """L15-12c-C2: the host rows of the radix chain root -> node, in token
    order. The walk goes last_node -> root; each node contributes its
    component_data[ComponentType.FULL].host_value (the host row ids of its
    own tokens); a node without a host_value (write-pending, or never
    backed up) contributes nothing. Must be read at BIND time: reset_keep
    nulls host_value on the kept nodes (unified_radix_cache.py).
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
        cur = getattr(cur, "parent", None)
    rows: list = []
    for chunk in reversed(chunks):
        rows.extend(chunk)
    return tuple(rows)


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
    for req in reqs:
        rid = str(req.rid)
        by_rid[rid] = req
        seq_len = _seq_len(req)
        # KV exists for seqlen - 1 tokens only (schedule_batch.py:2821).
        span = max(seq_len - 1, 0)
        # rows_by_rank = the EXACT owned count of the req's real slots
        # (owner_of over slots_of_req), not a proportional estimate:
        # retain admits against caps with these rows, while compact_plan
        # reserves by the exact owned count -- a proportional split can
        # over-admit past a cap on the rounding residue (audit item 11).
        _slots = slots_of_req(req, req_to_token)
        # L15-12c-C2: snapshot the chain's host rows NOW (reset_keep nulls
        # host_value later); a node missing its last_node skips l2 (retain
        # skips the whole round for that rid anyway).
        try:
            _rows = chain_host_rows(node_of_req(req))[: len(_slots)]
        except ValueError:
            _rows = ()
        if _rows:
            l2_rows.append((rid, _rows))
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
