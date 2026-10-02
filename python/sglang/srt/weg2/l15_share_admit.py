"""L15-10 S4n-e: a P stage adopts a hot follow-up's prefix from D's hold at
admission (instead of reading it from the store).

Control plane (the rid-keyed /dev/shm pattern of the hand-off manifests): the
front writes ``<share_dir>/hot.<rid>.json`` = {"prev_rid", "n"} before it
posts leg 1 of a hot follow-up; every P stage, when it admits ``rid``, reads
it (never deletes -- each stage reads it; the front reaps it after leg 1) and

1. fetches every D rank's share (descriptor + fds; l15_share_publish),
2. allocates ``n`` KV rows and one mamba slot,
3. copies its layers' KV (l15_share_take.take_kv) and rebuilds the END anchor
   for its linear layers (take_anchor),
4. inserts the prefix (the request's first ``n`` token ids) as device nodes
   with the anchor (l15_p_adopt) -- the normal prefix match then hits.

Any refusal frees what was allocated and returns a named reason; the request
proceeds exactly as today (store read / prefill). Opt-in:
SGLANG_WEG2_L15_HOT_SHARE=1.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple


def hot_hint(directory: str, rid: str) -> Optional[dict]:
    path = os.path.join(directory, "hot.%s.json" % rid)
    try:
        with open(path) as fh:
            h = json.load(fh)
    except (FileNotFoundError, ValueError):
        return None
    if not h.get("prev_rid") or int(h.get("n", 0)) <= 0:
        return None
    return h


def write_hot_hint(directory: str, rid: str, prev_rid: str, n: int) -> None:
    """Front side, before leg 1 of a hot follow-up."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "hot.%s.json" % rid)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"prev_rid": str(prev_rid), "n": int(n)}, fh)
    os.replace(tmp, path)


def reap_hot_hint(directory: str, rid: str) -> None:
    try:
        os.unlink(os.path.join(directory, "hot.%s.json" % rid))
    except FileNotFoundError:
        pass


def admit(*, rid: str, token_ids: Sequence[int], hint: dict,
          fetch: Callable[[int], Tuple[dict, Sequence[int]]], n_d_ranks: int,
          kv_alloc, mamba_alloc, tree_cache, stage_att_layers: Sequence[int],
          p_buffers: Dict[int, tuple], spec, stage_linear: Tuple[int, int],
          p_temporal, p_conv, map_extent, log) -> Optional[str]:
    """Adopt the hint's prefix; None on success, else the named reason
    (everything allocated is freed again)."""
    from sglang.srt.weg2 import l15_p_adopt, l15_share_take
    from sglang.srt.weg2.l15_hold_share import L15ShareError

    n = int(hint["n"])
    prev = str(hint["prev_rid"])
    if len(token_ids) < n:
        return "prompt %d tokens < hot prefix %d" % (len(token_ids), n)
    try:
        shares = {r: fetch(r) for r in range(int(n_d_ranks))}
    except L15ShareError as exc:
        return "share: %s" % exc
    # pick the rows/slot from the free lists WITHOUT taking them: the copy
    # needs no allocator state, and adopt() reserves exactly these (refusing
    # if any is not free) -- one owner of the reservation, nothing to undo
    free_kv = [int(x) for x in kv_alloc.free_pages.tolist()]
    free_mb = [int(x) for x in mamba_alloc.free_slots.tolist()]
    if len(free_kv) < n:
        return "no %d free P rows (%d free)" % (n, len(free_kv))
    if not free_mb:
        return "no free P mamba slot"
    rows = free_kv[:n]
    slot = free_mb[0]
    try:
        cells = l15_share_take.take_kv(
            shares, rid=prev, n=n, stage_layers=list(stage_att_layers),
            p_buffers=p_buffers, p_rows=rows, map_extent=map_extent)
        ratios = _ratios(shares)
        abytes = l15_share_take.take_anchor(
            shares, rid=prev, spec=spec, ratios=ratios, stage=stage_linear,
            p_temporal=p_temporal, p_conv=p_conv, p_slot=slot,
            map_extent=map_extent)
    except (l15_share_take.L15TakeError, L15ShareError) as exc:
        return "take: %s" % exc
    try:
        l15_p_adopt.adopt(tree_cache, kv_alloc, mamba_alloc,
                          token_ids=list(token_ids[:n]), rows=rows,
                          anchor_row=slot)
    except l15_p_adopt.L15AdoptRefused as exc:
        return "adopt: %s" % exc
    log("HOT-HANDOVER rid=%s from=%s n=%d cells=%d anchor_bytes=%d adopted"
        % (rid, prev, n, cells, abytes))
    return None


def _ratios(shares) -> list:
    d = next(iter(shares.values()))[0]
    prefix = [int(x) for x in d["prefix"]]
    return [prefix[i + 1] - prefix[i] for i in range(len(prefix) - 1)]


@dataclass
class StageGeom:
    """The live P stage as the hold-share copies need it (D's published
    layer set decides the stage's linear range)."""

    p_buffers: Dict[int, tuple]
    p_temporal: object
    p_conv: object
    spec: object
    stage_linear: Tuple[int, int]
    n_d: int
    dev: int
    kv_alloc: object
    mamba_alloc: object
    req_to_token_pool: object
    tree_cache: object


def stage_geometry(sched, d0: dict):
    """StageGeom of this P stage against D rank 0's descriptor ``d0``, or the
    named reason (str) when the stage cannot take/deposit."""
    mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    wrapper = getattr(mr, "token_to_kv_pool", None)
    amap = getattr(wrapper, "full_attention_layer_id_mapping", None)
    full = getattr(wrapper, "full_kv_pool", wrapper)
    if not amap or full is None:
        return "stage has no hybrid KV layer map"
    p_buffers = {int(g): (full.k_buffer[int(j)], full.v_buffer[int(j)])
                 for g, j in amap.items()}
    rtp = getattr(sched, "req_to_token_pool", None)
    mmap = getattr(rtp, "mamba_map", None) or {}
    mc = getattr(getattr(rtp, "mamba_pool", None), "mamba_cache", None)
    if not mmap or mc is None or len(getattr(mc, "conv", []) or []) != 1:
        return "stage has no single-conv mamba cache"
    stage_gids = sorted(int(g) for g in mmap)
    order = [int(mmap[g]) for g in stage_gids]
    if order != list(range(len(order))):
        # put/take index the cache by stage-local layer 0..n-1
        return "stage mamba layers are not stored as 0..n-1 in layer order"
    n_d = len(d0["prefix"]) - 1
    all_gids = sorted({int(b["layer"]) for b in d0["bases"]
                       if b.get("role") == "mamba_temporal"})
    try:
        pos = [all_gids.index(g) for g in stage_gids]
    except ValueError:
        return "stage mamba layer not published by D"
    if pos != list(range(pos[0], pos[0] + len(pos))):
        return "stage mamba layers are not contiguous"
    from sglang.srt.mem_cache.canonical_page_store import derive_mamba_blob_spec

    try:
        spec = derive_mamba_blob_spec(getattr(mr, "model_config", None),
                                      getattr(rtp, "mamba_pool", None),
                                      num_linear_layers=len(all_gids))
    except Exception as exc:  # noqa: BLE001 -- named refusal
        return "spec: %s" % exc
    dev = int(getattr(full.k_buffer[0], "device", None).index or 0)
    return StageGeom(p_buffers=p_buffers, p_temporal=mc.temporal,
                     p_conv=mc.conv[0], spec=spec,
                     stage_linear=(pos[0], pos[0] + len(pos)), n_d=n_d, dev=dev,
                     kv_alloc=getattr(sched, "token_to_kv_pool_allocator", None),
                     mamba_alloc=getattr(rtp, "mamba_allocator", None),
                     req_to_token_pool=rtp,
                     tree_cache=getattr(sched, "tree_cache", None))


def admit_for_sched(sched, req, env, log) -> Optional[str]:
    """P-stage entry at admission: the geometry from the live stage, then
    :func:`admit`. Returns None when nothing was to do or the prefix was
    adopted, else the named reason (logged by the caller)."""
    from sglang.srt.weg2 import l15_share_publish
    from sglang.srt.weg2.l15_hold_share import HoldMapper, L15ShareError

    directory = l15_share_publish.share_dir(env)
    rid = str(getattr(req, "rid", ""))
    hint = hot_hint(directory, rid)
    if hint is None:
        return None
    try:
        d0, f0 = l15_share_publish.fetch_share(directory, 0)
    except L15ShareError as exc:
        return "share: %s" % exc
    for f in f0:
        try:
            os.close(f)
        except OSError:
            pass
    g = stage_geometry(sched, d0)
    if isinstance(g, str):
        return g
    dev = g.dev
    # L15-HOLDMAP: every fd received and every extent mapped by this take is
    # released once the copies are done -- a lingering import pins D's hold
    # VRAM after D frees it (one hold-extent set per hot admission)
    mapper = HoldMapper(dev)
    try:
        return admit(
            rid=rid, token_ids=list(getattr(req, "origin_input_ids", ()) or ()),
            hint=hint,
            fetch=lambda r: mapper.fetch(
                lambda q: l15_share_publish.fetch_share(directory, q), r),
            n_d_ranks=g.n_d, kv_alloc=g.kv_alloc, mamba_alloc=g.mamba_alloc,
            tree_cache=g.tree_cache, stage_att_layers=sorted(g.p_buffers),
            p_buffers=g.p_buffers, spec=g.spec, stage_linear=g.stage_linear,
            p_temporal=g.p_temporal, p_conv=g.p_conv, map_extent=mapper, log=log)
    finally:
        mapper.close()
