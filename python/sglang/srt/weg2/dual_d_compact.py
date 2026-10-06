"""DUAL-TP3PP3 D-COMPACT: move D's live KV rows DOWN so the D-KV shrink can give P its card.

Metal (boot dual262kbar1fs10061152, D 11:59:07 -> 12:06:55): D freed its cache at once
(``D-CACHE-YIELD-LIVE freed=196207``) but its mapped KV span can only be cut FROM THE TOP
(``dual_d_kv_stage``: SHRINK/GROW on a token lattice). Two decode seats held rows 198716/198717
(their prompt rows and the spec-v2 reservation ahead of ``kv_committed_len``), so 39 ticks said
``SHRINK-BLOCKED reason=live_floor mapped=200704 need=8192 live_floor=198768`` and P's KV grant
waited 473 s until both seats had ended.

What this module does, between two decode rounds inside the D tick (no forward in flight: the
device is synchronized first, exactly like the shrink's own unmap):

1. TARGET = the level the tick's own ``decide()`` would shrink to if no live row stood in the way
   (group values only, so every D rank computes the same number).
2. PLAN (local, then proven identical on every rank): every live id above the target must be held
   by a running request's ``req_to_token[idx, :kv_allocated_len]`` and/or a radix-tree node; each
   such id gets the LOWEST free id <= target of the SAME uneven-DCP owner class -- the owner rule
   (``(L % S) in [lo, hi)``, compact row ``(L // S) * (hi - lo) + (L % S - lo)``,
   ``HiCacheController._dcp_owned_device_rows``) then keeps the copy on the one rank that holds the
   row, so no byte crosses ranks. Anything else holding a row above the target (a parked or waiting
   request, a hand-off, an unknown holder), a chunked request, an in-flight HiCache transfer over a
   tree row that would move, an unknown pool layout, or too few free low ids in a class:
   ``D-COMPACT REFUSED reason=...`` -- never a guess.
3. Collective A: MIN over (ok, plan fingerprint, -fingerprint, n, -n) -- one rank refusing or
   planning differently stops every rank before anything changed.
4. COPY (per rank, its owned pairs; the draft's raw-indexed pool on every rank), then a byte
   compare of every moved row. Collective B: MIN(ok). Any failure: the reserved destination ids go
   back to the free lists (the lists are restored as they were) and NO reference was touched.
5. COMMIT: ``req_to_token`` of the running requests, the tree nodes' device values and every
   request's ``prefix_indices`` are remapped (src -> dst; the remap is idempotent, so aliased views
   are safe), the DFlash window pool's draft rows follow through the allocator's alias listener
   (``DraftKVSlotMapper.on_global_alias``, the radix-dedup carry), then the old ids are freed.
   Collective C: the group's new live floor (MAX), which the tick's shrink uses in the same tick.

Mamba/GDN state is slot-based (``req.mamba_pool_idx``), not row-based: untouched.

Gate: ``dual_d_kv_stage.armed()`` (SGLANG_WEG2_DUAL_LAYOUT=1, group D, D KV max tokens > 0) AND
``SGLANG_WEG2_DUAL_D_COMPACT`` (default ON there; 0/false/off turns it off). Flip, INT8, NF and
group P never reach this module (the D tick returns before it without the dual D actor).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARK = "DUAL-TP3PP3 D-COMPACT"
SWITCH_ENV = "SGLANG_WEG2_DUAL_D_COMPACT"
#: rows per index_select/index_copy_ chunk (bounds the device transient of one copy)
CHUNK_ROWS = 8192
#: retry spacing after a refusal, in seconds of P's (group MAX) card wait -- group-consistent
BACKOFF_FIRST_S = 2.0
BACKOFF_MAX_S = 30.0
#: fingerprint modulus (a prime below 2**31: every product of two residues fits int64)
_FP_P = 2147483629
_FP_K1 = 1315423911 % _FP_P
_FP_K2 = 2654435761 % _FP_P


class Refused(Exception):
    """A named reason not to compact now (nothing changed)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__("%s %s" % (reason, detail))
        self.reason = reason
        self.detail = detail


class Weg2DualCompactBreach(RuntimeError):
    """A commit step failed after references started to change: stop by name, never run on a
    half-moved table."""


# -- gate -----------------------------------------------------------------------------------------

def armed(env=None) -> bool:
    env = os.environ if env is None else env
    from sglang.srt.weg2 import dual_d_kv_stage as _ddk

    if not _ddk.armed(env):
        return False
    return str(env.get(SWITCH_ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def due(actor, p_wait_s: float) -> bool:
    """Backoff after a refusal, on P's GROUP card wait (identical on every rank)."""
    return float(p_wait_s) >= float(getattr(actor, "_dc_next_wait_s", 0.0) or 0.0)


def note_refused(actor, p_wait_s: float) -> float:
    iv = float(getattr(actor, "_dc_backoff_s", 0.0) or 0.0)
    iv = BACKOFF_FIRST_S if iv <= 0 else min(BACKOFF_MAX_S, 2.0 * iv)
    actor._dc_backoff_s = iv
    actor._dc_next_wait_s = float(p_wait_s) + iv
    return iv


def reset(actor) -> None:
    actor._dc_backoff_s = 0.0
    actor._dc_next_wait_s = 0.0


# -- pure geometry --------------------------------------------------------------------------------

def owner_class(ids, prefix: Optional[Sequence[int]]):
    """The uneven-DCP owner class (rank index) of each GLOBAL id: ``(L % S)`` in
    ``[prefix[r], prefix[r+1])``. One class when there is no token split."""
    import torch

    ids = ids.to(torch.int64)
    if not prefix or len(prefix) < 3:
        return torch.zeros_like(ids)
    S = int(prefix[-1])
    inner = torch.tensor([int(x) for x in prefix[1:-1]], dtype=torch.int64)
    return torch.bucketize(ids % S, inner, right=True)


def compact_rows(ids, bounds: Optional[Tuple[int, int, int]]):
    """``(owned_mask, rows)`` of GLOBAL ids on THIS rank: the owner rule of
    ``HiCacheController._dcp_owned_device_rows`` (identity without bounds). ``rows`` is
    meaningful only where ``owned``."""
    import torch

    ids = ids.to(torch.int64)
    if bounds is None:
        return torch.ones_like(ids, dtype=torch.bool), ids
    S, lo, hi = (int(x) for x in bounds)
    off = ids % S
    owned = (off >= lo) & (off < hi)
    return owned, (ids // S) * (hi - lo) + (off - lo)


def plan_moves(src_ids, free_low, prefix: Optional[Sequence[int]]):
    """Pair every id of ``src_ids`` with the lowest free id of ``free_low`` of the SAME owner
    class (ascending on both sides). Raises ``Refused('class_short')`` when a class lacks room.
    Returns ``(src, dst, per_class)`` sorted by src; ``per_class`` = [(need, have), ...]."""
    import torch

    src = torch.unique(src_ids.to(torch.int64))            # sorted
    free = torch.unique(free_low.to(torch.int64))
    sc, fc = owner_class(src, prefix), owner_class(free, prefix)
    n_cls = (len(prefix) - 1) if (prefix and len(prefix) >= 3) else 1
    dst = torch.empty_like(src)
    per = []
    for c in range(n_cls):
        sm = sc == c
        need = int(sm.sum())
        cand = free[fc == c]
        per.append((need, int(cand.numel())))
        if need > int(cand.numel()):
            raise Refused("class_short", "class=%d need=%d free_low=%d" % (c, need, int(cand.numel())))
        dst[sm] = cand[:need]
    return src, dst, per


def fingerprint(src, dst) -> Tuple[int, int]:
    """Two 31-bit fingerprints of the pairing (order-sensitive in the second)."""
    import torch

    if src.numel() == 0:
        return 0, 0
    s = src.to(torch.int64) % _FP_P
    d = dst.to(torch.int64) % _FP_P
    a = ((s * _FP_K1) % _FP_P + (d * _FP_K2) % _FP_P) % _FP_P
    pos = torch.arange(1, s.numel() + 1, dtype=torch.int64) % _FP_P
    b = ((s * pos) % _FP_P + (d * ((pos * _FP_K1) % _FP_P)) % _FP_P) % _FP_P
    return int(a.sum().item() % _FP_P), int(b.sum().item() % _FP_P)


# -- reading the world ----------------------------------------------------------------------------

def dcp_geometry():
    """(bounds, prefix) of the uneven-DCP owner rule on this rank, (None, None) off that lane."""
    try:
        from sglang.srt.distributed.utils import cp_token_prefix, uneven_dcp_owner_bounds
    except ImportError:
        return None, None
    b = uneven_dcp_owner_bounds()
    if b is None:
        return None, None
    from sglang.srt.runtime_context import get_parallel

    return tuple(int(x) for x in b), [int(x) for x in cp_token_prefix(int(get_parallel().attn_dcp_size))]


def _check_geometry(bounds, prefix) -> None:
    if bounds is None:
        return
    S, lo, hi = bounds
    if not prefix or int(prefix[-1]) != int(S) or (int(lo), int(hi)) not in list(zip(prefix[:-1], prefix[1:])):
        raise Refused("dcp_geometry", "bounds=%s prefix=%s" % (bounds, prefix))


def _cpu_ids(t):
    import torch

    return t.detach().to("cpu", torch.int64).reshape(-1)


def live_ids_above(allocator, target: int):
    """Live ids (in no free list, not withheld by the cap) above ``target`` -- the reading of
    ``dual_p_kv_stage.max_live_id`` (page size 1)."""
    import torch

    n = int(getattr(allocator, "size", 0) or 0)
    live = torch.ones(n + 1, dtype=torch.bool)
    live[0] = False
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is not None and hasattr(ids, "numel") and ids.numel():
            live[_cpu_ids(ids)] = False
    cap = getattr(allocator, "_weg2_kv_stage_cap", None)
    held = getattr(cap, "_withheld", None) if cap is not None else None
    if held is not None and held.numel():
        live[_cpu_ids(held)] = False
    live[: int(target) + 1] = False
    return torch.nonzero(live).flatten()


def free_ids_upto(allocator, target: int):
    import torch

    parts = []
    for name in ("free_pages", "release_pages"):
        ids = getattr(allocator, name, None)
        if ids is not None and hasattr(ids, "numel") and ids.numel():
            parts.append(_cpu_ids(ids))
    if not parts:
        return torch.empty((0,), dtype=torch.int64)
    allf = torch.unique(torch.cat(parts))
    return allf[(allf >= 1) & (allf <= int(target))]


def _row_extent(req) -> int:
    """#922 ``owned_row_extent``: what the allocator handed this request (page size 1)."""
    alloc = int(getattr(req, "kv_allocated_len", -1) or -1)
    return alloc if alloc > 0 else -1


def _tree_nodes(tree):
    """(nodes with a device FULL value, None) or (None, reason)."""
    if tree is None:
        return [], None
    collect = getattr(tree, "_collect_all_nodes", None)
    if not callable(collect):
        if getattr(tree, "disable", False):
            return [], None
        return None, "tree_unknown"
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    comps = tuple(getattr(tree, "tree_components", ()) or ())
    if ComponentType.SWA in comps:
        return None, "tree_swa"
    root = getattr(tree, "root_node", None)
    out = []
    for n in collect():
        if n is root:
            continue
        v = n.component_data[ComponentType.FULL].value
        if v is not None and getattr(v, "numel", None) and v.numel():
            out.append((n, v))
    return out, None


def _hicache_busy(tree) -> Optional[str]:
    """A device<->host transfer in flight (its op captured DEVICE ids when it was queued): no tree row
    moves while one exists. ``ongoing_backup`` is host->storage (``write_storage(host_value)``) and does
    not read device rows, so it does not block."""
    for name in ("ongoing_write_through", "ongoing_load_back"):
        if getattr(tree, name, None):
            return name
    cc = getattr(tree, "cache_controller", None)
    for name in ("write_queue", "load_queue", "ack_write_queue", "ack_load_queue"):
        if cc is not None and getattr(cc, name, None):
            return "cache_controller." + name
    return None


def draft_kv_pool(sched):
    dw = getattr(sched, "draft_worker", None)
    if dw is None:
        return None
    for path in (("draft_model_runner",), ("draft_runner",), ("draft_worker", "draft_runner"),
                 ("draft_worker", "model_runner")):
        obj = dw
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        pool = getattr(obj, "token_to_kv_pool", None) if obj is not None else None
        if pool is not None:
            return pool
    return None


def _kv_tensors(pool) -> List[Any]:
    """The token-major K/V tensors of one MHA pool; Refused for any other layout."""
    if getattr(pool, "use_hnd", False) or getattr(pool, "kv_cache_layout", "") == "vectorized_5d":
        raise Refused("pool_layout", type(pool).__name__)
    if int(getattr(pool, "page_size", 1) or 1) != 1:
        raise Refused("page_size", str(getattr(pool, "page_size", "?")))
    ks, vs = list(getattr(pool, "k_buffer", None) or ()), list(getattr(pool, "v_buffer", None) or ())
    if not ks or len(ks) != len(vs):
        raise Refused("pool_layout", "%s has no k/v buffers" % type(pool).__name__)
    rows = int(getattr(pool, "size", 0) or 0) + 1
    for t in ks + vs:
        if t.dim() < 2 or int(t.shape[0]) < rows:
            raise Refused("pool_layout", "%s tensor %s is not token-major" % (type(pool).__name__, tuple(t.shape)))
    return ks + vs


@dataclass
class Plan:
    target: int
    src: Any
    dst: Any
    per_class: list
    req_rows: list = field(default_factory=list)       # (req, idx, ext) of the running requests touched
    tree_hit: list = field(default_factory=list)       # (node, value) whose value names a src id
    n_req_ids: int = 0
    n_tree_ids: int = 0
    copies: list = field(default_factory=list)         # (tensors, src_rows, dst_rows, what)
    draft: str = "none"

    def fp(self) -> Tuple[int, int]:
        return fingerprint(self.src, self.dst)


def build_plan(sched, actor, target: int, *, geometry=None) -> Plan:
    """The local plan (no state changed). Raises Refused with a name."""
    import torch

    alloc = actor.allocator
    if int(getattr(alloc, "page_size", 1) or 1) != 1 or int(actor.page) != 1:
        raise Refused("page_size", str(getattr(alloc, "page_size", "?")))
    if not getattr(alloc, "is_not_in_free_group", True):
        raise Refused("free_group")
    if getattr(sched, "chunked_req", None) is not None:
        raise Refused("chunked_req")
    if getattr(sched, "weg2_d_parked", None) or getattr(sched, "_weg2_d_hold", None):
        raise Refused("holds")
    for w in list(getattr(sched, "waiting_queue", None) or ()):
        if getattr(w, "req_pool_idx", None) is not None:
            raise Refused("waiting_holds_rows", str(getattr(w, "rid", "?")))
    bounds, prefix = geometry if geometry is not None else dcp_geometry()
    _check_geometry(bounds, prefix)

    live = live_ids_above(alloc, target)
    if live.numel() == 0:
        raise Refused("nothing_above")
    r2t = getattr(getattr(sched, "req_to_token_pool", None), "req_to_token", None)
    if r2t is None:
        raise Refused("no_req_to_token")
    reqs = list(getattr(getattr(sched, "running_batch", None), "reqs", None) or ())
    held = []
    req_rows = []
    n_req = 0
    for req in reqs:
        idx = getattr(req, "req_pool_idx", None)
        if idx is None:
            continue
        ext = _row_extent(req)
        if ext <= 0 or ext > int(r2t.shape[1]):
            raise Refused("extent", "%s kv_allocated_len=%s" % (getattr(req, "rid", "?"), ext))
        rows = _cpu_ids(r2t[int(idx), :ext])
        hi = rows[rows > int(target)]
        if hi.numel():
            req_rows.append((req, int(idx), ext))
            held.append(hi)
            n_req += int(hi.numel())
    nodes, why = _tree_nodes(getattr(sched, "tree_cache", None))
    if nodes is None:
        raise Refused(why)
    tree_hit = []
    n_tree = 0
    for node, v in nodes:
        vc = _cpu_ids(v)
        hi = vc[vc > int(target)]
        if hi.numel():
            tree_hit.append((node, v))
            held.append(hi)
            n_tree += int(hi.numel())
    if tree_hit:
        busy = _hicache_busy(sched.tree_cache)
        if busy:
            raise Refused("hicache_busy", busy)
    referenced = torch.unique(torch.cat(held)) if held else torch.empty((0,), dtype=torch.int64)
    unaccounted = live[~torch.isin(live, referenced)]
    if unaccounted.numel():
        raise Refused("unaccounted", "n=%d top=%d" % (int(unaccounted.numel()), int(unaccounted.max())))
    stale = referenced[~torch.isin(referenced, live)]
    if stale.numel():
        raise Refused("free_referenced", "n=%d top=%d" % (int(stale.numel()), int(stale.max())))
    src, dst, per = plan_moves(live, free_ids_upto(alloc, target), prefix)
    plan = Plan(target=int(target), src=src, dst=dst, per_class=per, req_rows=req_rows, tree_hit=tree_hit,
                n_req_ids=n_req, n_tree_ids=n_tree)
    # the copies: this rank's owned pairs of every target pool, and the draft
    owned, s_rows = compact_rows(src, bounds)
    _o2, d_rows = compact_rows(dst, bounds)
    for pool in list(getattr(actor, "pools", None) or ()):
        plan.copies.append((_kv_tensors(pool), s_rows[owned], d_rows[owned], "target"))
    dpool = draft_kv_pool(sched)
    if dpool is not None:
        if getattr(dpool, "weg2_slot_mapper", None) is not None:
            has = getattr(alloc, "has_alias_listeners", None)
            plan.draft = "alias" if (callable(has) and has()) else "holes"
        else:
            from sglang.srt.weg2 import dual_p_kv_stage as _pk

            subs = _pk.stage_pools(dpool) or [dpool]
            for sub in subs:
                tens = _kv_tensors(sub)
                if int(tens[0].shape[0]) >= int(getattr(alloc, "size", 0) or 0) + 1:
                    plan.copies.append((tens, src, dst, "draft"))
                    plan.draft = "copy_raw"
                elif bounds is not None:
                    plan.copies.append((tens, s_rows[owned], d_rows[owned], "draft"))
                    plan.draft = "copy_dcp"
                else:
                    raise Refused("draft_layout", type(sub).__name__)
    return plan


# -- acting ---------------------------------------------------------------------------------------

def _reserve(alloc, dst) -> None:
    """Take exactly ``dst`` out of the free lists (deterministic on every rank)."""
    import torch

    for name in ("free_pages", "release_pages"):
        ids = getattr(alloc, name, None)
        if ids is None or not ids.numel():
            continue
        hit = torch.isin(_cpu_ids(ids), dst)
        if bool(hit.any()):
            setattr(alloc, name, ids[(~hit).to(ids.device)])
    touch = getattr(alloc, "_owner_placement_touch", None)
    if callable(touch):
        touch()


def _copy(plan: Plan, chunk: int = CHUNK_ROWS) -> int:
    """Copy every pair, then compare every moved row byte for byte. Returns the rows copied."""
    import torch

    n = 0
    for tensors, s_rows, d_rows, _what in plan.copies:
        if not s_rows.numel():
            continue
        for t in tensors:
            dev = t.device
            for a in range(0, int(s_rows.numel()), int(chunk)):
                s = s_rows[a:a + chunk].to(dev)
                d = d_rows[a:a + chunk].to(dev)
                t.index_copy_(0, d, t.index_select(0, s))
                # bitwise (fp8/bf16 alike; a NaN pattern compares equal to itself)
                if not torch.equal(t.index_select(0, s).view(torch.uint8), t.index_select(0, d).view(torch.uint8)):
                    raise RuntimeError("%s copy compare failed (%s rows %d..)" % (MARK, _what, a))
        n += int(s_rows.numel())
    return n


def _commit(sched, plan: Plan) -> None:
    """Switch every reference src -> dst. The remap maps src to dst and every other id to itself,
    so it is idempotent: a tree value and a ``prefix_indices`` that are views of one storage are
    remapped twice without harm."""
    import torch

    size = int(max(int(plan.src.max()), int(plan.dst.max()))) + 1
    remap = torch.arange(size, dtype=torch.int64)
    remap[plan.src] = plan.dst
    r2t = sched.req_to_token_pool.req_to_token
    # every new value first (a failure here changes nothing) ...
    pending = []
    for _req, idx, ext in plan.req_rows:
        row = _cpu_ids(r2t[idx, :ext])
        pending.append((idx, ext, remap_ids(row, remap).to(device=r2t.device, dtype=r2t.dtype)))
    tens = [v for _node, v in plan.tree_hit]
    reqs = list(getattr(getattr(sched, "running_batch", None), "reqs", None) or ())
    reqs += list(getattr(sched, "waiting_queue", None) or ())
    for req in reqs:
        pi = getattr(req, "prefix_indices", None)
        if pi is not None and hasattr(pi, "numel") and pi.numel() and bool((_cpu_ids(pi) > plan.target).any()):
            tens.append(pi)
    # ... then the switch (tensor writes only)
    try:
        for idx, ext, new in pending:
            r2t[idx, :ext] = new
        for t in tens:
            _remap_in_place_bounded(t, remap)
    except Exception as exc:  # noqa: BLE001 -- references are half switched: stop by name
        raise Weg2DualCompactBreach("%s COMMIT failed after the switch began: %r" % (MARK, exc)) from exc


def remap_ids(ids, remap):
    """``ids`` with every id below ``remap``'s length mapped through it (ids beyond stay)."""
    out = ids.clone()
    m = (ids >= 0) & (ids < remap.numel())
    out[m] = remap[ids[m]]
    return out


def _remap_in_place_bounded(t, remap) -> None:
    new = remap_ids(_cpu_ids(t), remap).to(device=t.device, dtype=t.dtype).reshape(t.shape)
    t.copy_(new)


def _release_src(alloc, plan: Plan) -> None:
    import torch

    dev = getattr(alloc, "device", "cpu")
    dtype = getattr(getattr(alloc, "free_pages", None), "dtype", torch.int64)
    src = plan.src.to(device=dev, dtype=dtype)
    if plan.draft == "alias":
        alloc.notify_alias(src, plan.dst.to(device=dev, dtype=dtype))
    alloc.free(src)


def run(sched, actor, *, target: int, floor: int, p_wait_s: float = 0.0, gmin: Optional[Callable] = None,
        geometry=None) -> Optional[int]:
    """One compaction attempt on every D rank (entered by all of them on group values). Returns
    the GROUP's new live floor (tokens) on success, None when nothing moved."""
    gmin = gmin or getattr(actor, "gmin", None) or (lambda v: list(v))
    from sglang.srt.weg2 import dual_p_kv_stage as _pk

    t0 = time.perf_counter()
    plan, local = None, None
    try:
        plan = build_plan(sched, actor, int(target), geometry=geometry)
    except Refused as r:
        local = r
    except Exception as exc:  # noqa: BLE001 -- a planning error is a refusal of THIS rank
        local = Refused("plan_error", repr(exc))
    ok = 1 if plan is not None else 0
    f1, f2 = plan.fp() if plan is not None else (0, 0)
    n = int(plan.src.numel()) if plan is not None else 0
    g = [int(x) for x in gmin([ok, f1, -f1, f2, -f2, n, -n])]
    agree = g[0] == 1 and g[1] == -g[2] and g[3] == -g[4] and g[5] == -g[6]
    if not agree:
        if local is None:
            local = Refused("plan_mismatch" if g[0] == 1 else "other_rank")
        _log_refused(actor, local, target, floor, p_wait_s)
        return None
    # phase 1: reserve + copy + compare (references untouched)
    alloc = actor.allocator
    saved = (getattr(alloc, "free_pages", None), getattr(alloc, "release_pages", None))
    ok1, err, copied = 1, None, 0
    t1 = time.perf_counter()
    try:
        actor._sync()
        _reserve(alloc, plan.dst)
        copied = _copy(plan)
        actor._sync()
    except Exception as exc:  # noqa: BLE001 -- this rank could not copy: every rank rolls back
        ok1, err = 0, exc
    t2 = time.perf_counter()
    if int(gmin([ok1])[0]) != 1:
        alloc.free_pages, alloc.release_pages = saved
        touch = getattr(alloc, "_owner_placement_touch", None)
        if callable(touch):
            touch()
        logger.warning("%s ROLLBACK reason=%s target=%d floor=%d moved=0: the copy failed on %s; the reserved "
                       "ids are back in the free lists, no reference was changed", MARK,
                       ("copy_error:%r" % (err,)) if err is not None else "other_rank", int(target), int(floor),
                       "this rank" if err is not None else "another rank")
        iv = note_refused(actor, p_wait_s)
        logger.info("%s REFUSED reason=rollback target=%d retry_after_s=%.1f", MARK, int(target), iv)
        return None
    # commit: references, draft carry, free the old ids
    _commit(sched, plan)
    try:
        _release_src(alloc, plan)
    except Exception as exc:  # noqa: BLE001 -- the references point at dst already: stop by name
        raise Weg2DualCompactBreach("%s RELEASE of the old ids failed after the switch: %r" % (MARK, exc)) from exc
    actor._sync()
    live_after = int(_pk.max_live_id(alloc, int(actor.page))) * int(actor.page)
    new_floor = -int(gmin([-live_after])[0])
    t3 = time.perf_counter()
    reset(actor)
    logger.info("%s DONE moved=%d req_rows=%d tree_rows=%d reqs=%d nodes=%d copied_local=%d target=%d floor=%d->%d "
                "classes=%s draft=%s copy_ms=%.1f total_ms=%.1f p_wait_s=%.1f mapped=%d: the live rows above the "
                "target now sit low, the shrink can give P its card in this tick", MARK, n, plan.n_req_ids,
                plan.n_tree_ids, len(plan.req_rows), len(plan.tree_hit), copied, int(target), int(floor), new_floor,
                ",".join("%d/%d" % c for c in plan.per_class), plan.draft, 1000.0 * (t2 - t1),
                1000.0 * (t3 - t0), float(p_wait_s), int(actor.mapped_tokens))
    return new_floor


def _log_refused(actor, r: Refused, target: int, floor: int, p_wait_s: float) -> None:
    iv = note_refused(actor, p_wait_s)
    logger.info("%s REFUSED reason=%s %starget=%d floor=%d p_wait_s=%.1f retry_after_s=%.1f mapped=%d: nothing "
                "moved, the shrink waits for the live floor as before", MARK, r.reason,
                ("detail=%s " % r.detail.replace(" ", ",")) if r.detail else "", int(target), int(floor),
                float(p_wait_s), iv, int(getattr(actor, "mapped_tokens", 0) or 0))
