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


def admit_for_sched(sched, req, env, log) -> Optional[str]:
    """P-stage entry at admission: the geometry from the live stage, then
    :func:`admit`. Returns None when nothing was to do or the prefix was
    adopted, else the named reason (logged by the caller)."""
    from sglang.srt.weg2 import l15_share_publish
    from sglang.srt.weg2.l15_hold_share import L15ShareError, map_hold_extent

    directory = l15_share_publish.share_dir(env)
    rid = str(getattr(req, "rid", ""))
    hint = hot_hint(directory, rid)
    if hint is None:
        return None
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
    p_temporal = mc.temporal[order] if order != sorted(order) else mc.temporal
    p_conv = mc.conv[0][order] if order != sorted(order) else mc.conv[0]
    if order != sorted(order):
        return "stage mamba layers are not stored in layer order"
    try:
        d0, f0 = l15_share_publish.fetch_share(directory, 0)
    except L15ShareError as exc:
        return "share: %s" % exc
    for f in f0:
        try:
            os.close(f)
        except OSError:
            pass
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
    return admit(
        rid=rid, token_ids=list(getattr(req, "origin_input_ids", ()) or ()),
        hint=hint, fetch=lambda r: l15_share_publish.fetch_share(directory, r),
        n_d_ranks=n_d,
        kv_alloc=getattr(sched, "token_to_kv_pool_allocator", None),
        mamba_alloc=getattr(rtp, "mamba_allocator", None),
        tree_cache=getattr(sched, "tree_cache", None),
        stage_att_layers=sorted(p_buffers), p_buffers=p_buffers, spec=spec,
        stage_linear=(pos[0], pos[0] + len(pos)), p_temporal=p_temporal,
        p_conv=p_conv,
        map_extent=lambda fd, size: map_hold_extent(fd, size, dev, 0),
        log=log)
