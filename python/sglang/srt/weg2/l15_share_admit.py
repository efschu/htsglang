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


def write_hot_hint(directory: str, rid: str, prev_rid: str, n: int,
                   ids: Optional[Sequence[int]] = None) -> None:
    """Front side: before leg 1 of a hot follow-up (admission mode), or at
    the D->P flip's begin with the prefix's token ``ids`` (wake mode: P
    adopts the prefix in its resume RPC, before the request exists there)."""
    os.makedirs(directory, exist_ok=True)
    sweep_stale(directory)
    path = os.path.join(directory, "hot.%s.json" % rid)
    tmp = path + ".tmp"
    body = {"prev_rid": str(prev_rid), "n": int(n)}
    if ids is not None:
        import base64
        from array import array

        body["ids_b64"] = base64.b64encode(
            array("q", [int(t) for t in ids]).tobytes()).decode()
    with open(tmp, "w") as fh:
        json.dump(body, fh)
    os.replace(tmp, path)


def hint_ids(hint: dict) -> Optional[list]:
    """The prefix token ids a wake-mode hint carries (None without)."""
    text = hint.get("ids_b64")
    if not text:
        return None
    import base64
    from array import array

    a = array("q")
    a.frombytes(base64.b64decode(text))
    return list(a)


WAKE_ENV = "SGLANG_WEG2_L15_HOT_AT_WAKE"


def at_wake(env) -> bool:
    """Take mode: at P's wake (default) or at admission (=0). The wake runs
    with every P stage inside the same fenced resume RPC, so the verdict
    wait cannot stall a stage's admission behind a busy pipeline."""
    return str(env.get(WAKE_ENV, "1")).strip() != "0"


def reap_hot_hint(directory: str, rid: str) -> None:
    try:
        os.unlink(os.path.join(directory, "hot.%s.json" % rid))
    except FileNotFoundError:
        pass
    reap_verdict(directory, rid)


# -- L15-10c: one verdict for every P stage (no collective) -----------------

VERDICT_ENV = "SGLANG_WEG2_L15_HOT_VERDICT_S"


def verdict_timeout_s(env) -> float:
    """The stage wait for the others' results; far below any flip budget (the
    wait sits in one hot admission, not in the flip). Default 2 s."""
    try:
        return max(0.05, float(env.get(VERDICT_ENV, "2.0")))
    except (TypeError, ValueError):
        return 2.0


def _vpath(directory: str, rid: str, what: str) -> str:
    return os.path.join(directory, "hotv.%s.%s" % (rid, what))


def _create_once(path: str, text: str) -> None:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def _read_verdict(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            v = fh.read().strip()
    except FileNotFoundError:
        return None
    return v if v in ("adopt", "fallback") else None


def stage_verdict(directory: str, rid: str, stage: int, n_stages: int, ok: bool,
                  timeout_s: float, sleep=None, now=None) -> str:
    """Every P stage posts its result; ONE verdict file per rid is created
    with O_EXCL by the first stage that can decide (all ``n_stages`` ok ->
    "adopt"; any failure or the timeout -> "fallback") and every stage
    follows it -- the stages adopt the prefix together or none does."""
    import time as _t

    sleep = sleep or _t.sleep
    now = now or _t.monotonic
    os.makedirs(directory, exist_ok=True)
    vf = _vpath(directory, rid, "verdict")
    with open(_vpath(directory, rid, "s%d" % int(stage)), "w") as fh:
        fh.write("ok" if ok else "fail")
    if not ok:
        _create_once(vf, "fallback")
    deadline = now() + float(timeout_s)
    while True:
        v = _read_verdict(vf)
        if v is not None:
            return v
        res = []
        for k in range(int(n_stages)):
            try:
                with open(_vpath(directory, rid, "s%d" % k)) as fh:
                    res.append(fh.read().strip())
            except FileNotFoundError:
                break
        if len(res) == int(n_stages):
            _create_once(vf, "adopt" if all(r == "ok" for r in res) else "fallback")
        elif now() >= deadline:
            _create_once(vf, "fallback")
        else:
            sleep(0.005)


def reap_verdict(directory: str, rid: str) -> int:
    """The front, after leg 1: remove the rid's stage results and verdict."""
    n = 0
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return 0
    pre = "hotv.%s." % rid
    for name in names:
        if name.startswith(pre):
            try:
                os.unlink(os.path.join(directory, name))
                n += 1
            except FileNotFoundError:
                pass
    return n


def sweep_stale(directory: str, max_age_s: float = 600.0, now=None) -> int:
    """Remove hot hints / verdict files older than ``max_age_s`` (a front
    that died between hint and reap leaves them)."""
    import time as _t

    t = (now or _t.time)()
    n = 0
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return 0
    for name in names:
        if not (name.startswith("hot.") or name.startswith("hotv.")):
            continue
        path = os.path.join(directory, name)
        try:
            if t - os.path.getmtime(path) > max_age_s:
                os.unlink(path)
                n += 1
        except FileNotFoundError:
            pass
    return n


def admit(*, rid: str, token_ids: Sequence[int], hint: dict,
          fetch: Callable[[int], Tuple[dict, Sequence[int]]], n_d_ranks: int,
          kv_alloc, mamba_alloc, tree_cache, stage_att_layers: Sequence[int],
          p_buffers: Dict[int, tuple], spec, stage_linear: Tuple[int, int],
          p_temporal, p_conv, map_extent, log, cap0: Sequence[int] = (),
          l2=None, verdict=None) -> Optional[str]:
    """Adopt the hint's prefix; None on success, else the named reason
    (everything allocated is freed again).

    L15-10c: ``cap0`` -- D ranks that hold nothing (cap 0): their tokens are
    loaded from L2 through ``l2.kv(rows, slots, gens)`` (the canonical page
    identities the capped ranks publish per span); the END anchor comes from
    its canonical L2 slot through ``l2.anchor(slot, gen, p_slot)`` when the
    span names one (else from the shares, possible only without cap-0
    ranks). ``verdict(ok) -> "adopt"|"fallback"``: every P stage follows ONE
    verdict, so the stages adopt together or none does."""
    from sglang.srt.weg2 import l15_p_adopt, l15_share_take
    from sglang.srt.weg2.l15_hold_share import L15ShareError

    def _vote(ok: bool, why: Optional[str]) -> Optional[str]:
        if verdict is None:
            return why
        v = verdict(ok)
        if v != "adopt":
            return why or "a peer stage refused (verdict %s)" % v
        return why

    n = int(hint["n"])
    prev = str(hint["prev_rid"])
    skip = sorted({int(r) for r in cap0})
    if len(token_ids) < n:
        return _vote(False, "prompt %d tokens < hot prefix %d" % (len(token_ids), n))
    try:
        shares = {r: fetch(r) for r in range(int(n_d_ranks)) if r not in skip}
    except L15ShareError as exc:
        return _vote(False, "share: %s" % exc)
    # pick the rows/slot from the free lists WITHOUT taking them: the copy
    # needs no allocator state, and adopt() reserves exactly these (refusing
    # if any is not free) -- one owner of the reservation, nothing to undo
    free_kv = [int(x) for x in kv_alloc.free_pages.tolist()]
    free_mb = [int(x) for x in mamba_alloc.free_slots.tolist()]
    if len(free_kv) < n:
        return _vote(False, "no %d free P rows (%d free)" % (n, len(free_kv)))
    if not free_mb:
        return _vote(False, "no free P mamba slot")
    rows = free_kv[:n]
    slot = free_mb[0]
    span = _span_of(shares, prev)
    try:
        cells = l15_share_take.take_kv(
            shares, rid=prev, n=n, stage_layers=list(stage_att_layers),
            p_buffers=p_buffers, p_rows=rows, map_extent=map_extent,
            skip_ranks=skip)
        l2_tokens = 0
        if skip:
            if l2 is None or span is None:
                raise l15_share_take.L15TakeError(
                    "cap-0 rank(s) %s hold nothing and no L2 loader/span" % skip)
            try:
                l2_rows, l2_slots, l2_gens = _cap0_tokens(
                    span, n, skip, rows,
                    next(iter(shares.values()))[0]["prefix"])
            except ValueError as exc:
                raise l15_share_take.L15TakeError(str(exc)) from exc
            why = l2.kv(l2_rows, l2_slots, l2_gens)
            if why:
                raise l15_share_take.L15TakeError("L2 tokens: %s" % why)
            l2_tokens = len(l2_rows)
        a_slot = int((span or {}).get("anchor_l2_slot", -1))
        if l2 is not None and a_slot >= 0:
            why = l2.anchor(a_slot, int(span.get("anchor_l2_gen", -1)), slot)
            if why:
                raise l15_share_take.L15TakeError("L2 anchor: %s" % why)
            abytes = -1
        elif not skip:
            ratios = _ratios(shares)
            abytes = l15_share_take.take_anchor(
                shares, rid=prev, spec=spec, ratios=ratios, stage=stage_linear,
                p_temporal=p_temporal, p_conv=p_conv, p_slot=slot,
                map_extent=map_extent)
        else:
            raise l15_share_take.L15TakeError(
                "the END anchor has no L2 identity and cap-0 ranks hold no share")
    except (l15_share_take.L15TakeError, L15ShareError) as exc:
        return _vote(False, "take: %s" % exc)
    why = _vote(True, None)
    if why is not None:
        return why
    try:
        l15_p_adopt.adopt(tree_cache, kv_alloc, mamba_alloc,
                          token_ids=list(token_ids[:n]), rows=rows,
                          anchor_row=slot)
    except l15_p_adopt.L15AdoptRefused as exc:
        # every stage checked the same free rows before the verdict; a
        # refusal here means the stages' allocators diverged -- named loudly
        log("HOT-HANDOVER rid=%s adopt REFUSED after the adopt verdict: %s"
            % (rid, exc))
        return "adopt: %s" % exc
    log("HOT-HANDOVER rid=%s from=%s n=%d cells=%d l2_tokens=%d anchor=%s adopted"
        % (rid, prev, n, cells, l2_tokens, "l2" if abytes == -1 else abytes))
    return None


def _span_of(shares, rid: str) -> Optional[dict]:
    for d, _f in shares.values():
        for s in d.get("spans", ()):
            if s.get("rid") == rid:
                return s
    return None


def _cap0_tokens(span: dict, n: int, skip, rows, prefix):
    """(P rows, L2 slots, L2 gens) of the prefix tokens a cap-0 rank owns
    (owner rule: rank r owns slot L iff prefix[r] <= L % S < prefix[r+1])."""
    from sglang.srt.weg2.l15_share_publish import unb64

    pre = [int(x) for x in prefix]
    S = pre[-1]
    slots = [int(x) for x in span.get("slots", ())][:n]
    l2s = unb64(span.get("l2_slots_b64", ""))[:n]
    l2g = unb64(span.get("l2_gens_b64", ""))[:n]
    if len(slots) < n or len(l2s) < n or len(l2g) < n:
        raise ValueError("span carries %d slots / %d L2 ids for %d tokens"
                         % (len(slots), len(l2s), n))
    skip = set(int(r) for r in skip)
    out_r, out_s, out_g = [], [], []
    for i in range(n):
        res = slots[i] % S
        owner = next(r for r in range(len(pre) - 1) if pre[r] <= res < pre[r + 1])
        if owner in skip:
            out_r.append(int(rows[i]))
            out_s.append(int(l2s[i]))
            out_g.append(int(l2g[i]))
    return out_r, out_s, out_g


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


class L2Loader:
    """L15-10c: this P stage's L2 loads for a hot prefix -- the cap-0 ranks'
    tokens (canonical KV pages, generation-checked, one bulk load) and the
    END anchor (canonical GDN slot, all heads of the stage's layers)."""

    def __init__(self, host_pool, device_pool, host_mamba, dev_mamba):
        self.host_pool, self.device_pool = host_pool, device_pool
        self.host_mamba, self.dev_mamba = host_mamba, dev_mamba

    def kv(self, rows, slots, gens) -> Optional[str]:
        from sglang.srt.weg2 import l15_refill

        if not rows:
            return None
        if self.host_pool is None or self.device_pool is None:
            return "no L2 host pool / device pool on this stage"
        plan = [(("hot",), int(r), int(s), int(g)) for r, s, g in zip(rows, slots, gens)]
        ok, bad = l15_refill.gen_check(plan, self.host_pool)
        if bad:
            return "%d page(s) not COMPLETE at the recorded generation" % (
                len(plan) - len(ok))
        page = max(1, int(getattr(self.host_pool, "_arena_page_tokens", 1)))
        try:
            l15_refill.refill(ok, self.host_pool, self.device_pool, page)
        except l15_refill.L15RefillError as exc:
            return str(exc)
        return None

    def anchor(self, slot: int, gen: int, p_slot: int) -> Optional[str]:
        import torch

        hm = self.host_mamba
        if hm is None or self.dev_mamba is None or getattr(hm, "slot_gens", None) is None:
            return "no mamba host/device pool on this stage"
        if [int(x) for x in hm.slot_gens([int(slot)])] != [int(gen)]:
            return "anchor slot %d not at generation %d" % (slot, gen)
        try:
            hm._load_states_all_layers(self.dev_mamba,
                                       torch.tensor([int(slot)], dtype=torch.int64),
                                       torch.tensor([int(p_slot)], dtype=torch.int64))
        except Exception as exc:  # noqa: BLE001 -- named refusal
            return "anchor load: %r" % (exc,)
        return None


def _first_share(directory: str, fetch_share):
    """The descriptor of the first D rank that publishes a share (a cap-0
    rank publishes none)."""
    import re

    from sglang.srt.weg2.l15_hold_share import L15ShareError

    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        names = []
    ranks = sorted(int(m.group(1)) for m in
                   (re.fullmatch(r"D\.(\d+)\.json", x) for x in names) if m)
    last = None
    for r in ranks:
        try:
            d, fds = fetch_share(directory, r)
        except L15ShareError as exc:
            last = exc
            continue
        for f in fds:
            try:
                os.close(f)
            except OSError:
                pass
        return d
    raise L15ShareError("no D rank publishes a hold share (%s)" % (last,))


def admit_for_sched(sched, req, env, log) -> Optional[str]:
    """P-stage entry at admission: the geometry from the live stage, then
    :func:`admit`. Returns None when nothing was to do or the prefix was
    adopted, else the named reason (logged by the caller). Every return
    after the hint is found posts this stage's result into the rid's verdict
    (L15-10c), so the stages adopt together or none does."""
    from sglang.srt.weg2 import l15_bind, l15_share_publish
    from sglang.srt.weg2.l15_hold_share import HoldMapper, L15ShareError

    directory = l15_share_publish.share_dir(env)
    rid = str(getattr(req, "rid", ""))
    if at_wake(env):
        return None      # the prefix was adopted (or refused) at P's wake
    hint = hot_hint(directory, rid)
    if hint is None:
        return None
    stage = int(getattr(sched, "pp_rank", 0) or 0)
    n_stages = int(getattr(sched, "pp_size", 1) or 1)
    tmo = verdict_timeout_s(env)

    def verdict(ok: bool) -> str:
        import time as _t

        t0 = _t.perf_counter()
        v = stage_verdict(directory, rid, stage, n_stages, ok, tmo)
        # the wait blocks this stage's admission: measured per hot request
        log("HOT-HANDOVER-VERDICT rid=%s stage=%d/%d mine=%s verdict=%s wait_ms=%.0f"
            % (rid, stage, n_stages, "ok" if ok else "fail", v,
               (_t.perf_counter() - t0) * 1000.0))
        return v

    try:
        d0 = _first_share(directory, l15_share_publish.fetch_share)
    except L15ShareError as exc:
        verdict(False)
        return "share: %s" % exc
    g = stage_geometry(sched, d0)
    if isinstance(g, str):
        verdict(False)
        return g
    dev = g.dev
    host_pool, host_mamba = l15_bind.live_host_pools(g.tree_cache)
    mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    l2 = L2Loader(host_pool, getattr(mr, "token_to_kv_pool", None), host_mamba,
                  getattr(g.req_to_token_pool, "mamba_pool", None))
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
            p_temporal=g.p_temporal, p_conv=g.p_conv, map_extent=mapper, log=log,
            cap0=[int(r) for r in d0.get("cap0", ())], l2=l2, verdict=verdict)
    finally:
        mapper.close()


def take_all_at_wake(sched, env, log) -> int:
    """L15-10d: every hot hint of this D->P flip, taken in P's resume RPC
    (kv_cache leg, after the kv resume) on every P stage in rid order; one
    verdict per rid (the stages are inside the same fenced RPC, so they
    meet within milliseconds). Returns the prefixes adopted by this stage.
    Never raises: a refusal is named, today's store read serves."""
    import re

    from sglang.srt.weg2 import l15_bind, l15_share_publish
    from sglang.srt.weg2.l15_hold_share import HoldMapper, L15ShareError

    directory = l15_share_publish.share_dir(env)
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return 0
    rids = sorted(m.group(1) for m in (re.fullmatch(r"hot\.(.+)\.json", x)
                                        for x in names) if m)
    if not rids:
        return 0
    stage = int(getattr(sched, "pp_rank", 0) or 0)
    n_stages = int(getattr(sched, "pp_size", 1) or 1)
    tmo = verdict_timeout_s(env)

    def verdict_for(rid):
        def verdict(ok: bool) -> str:
            import time as _t

            t0 = _t.perf_counter()
            v = stage_verdict(directory, rid, stage, n_stages, ok, tmo)
            log("HOT-HANDOVER-VERDICT rid=%s stage=%d/%d mine=%s verdict=%s wait_ms=%.0f at=wake"
                % (rid, stage, n_stages, "ok" if ok else "fail", v,
                   (_t.perf_counter() - t0) * 1000.0))
            return v
        return verdict

    try:
        d0 = _first_share(directory, l15_share_publish.fetch_share)
        g = stage_geometry(sched, d0)
    except L15ShareError as exc:
        d0, g = None, "share: %s" % exc
    if isinstance(g, str):
        for rid in rids:
            verdict_for(rid)(False)
        log("HOT-HANDOVER at=wake: %d hint(s) refused (%s)" % (len(rids), g))
        return 0
    host_pool, host_mamba = l15_bind.live_host_pools(g.tree_cache)
    mr = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    l2 = L2Loader(host_pool, getattr(mr, "token_to_kv_pool", None), host_mamba,
                  getattr(g.req_to_token_pool, "mamba_pool", None))
    adopted = 0
    for rid in rids:
        hint = hot_hint(directory, rid)
        ids = hint_ids(hint) if hint is not None else None
        if hint is None or ids is None:
            verdict_for(rid)(False)
            log("HOT-HANDOVER rid=%s at=wake fallback=hint without token ids" % rid)
            continue
        mapper = HoldMapper(g.dev)
        try:
            why = admit(
                rid=rid, token_ids=ids, hint=hint,
                fetch=lambda r: mapper.fetch(
                    lambda q: l15_share_publish.fetch_share(directory, q), r),
                n_d_ranks=g.n_d, kv_alloc=g.kv_alloc, mamba_alloc=g.mamba_alloc,
                tree_cache=g.tree_cache, stage_att_layers=sorted(g.p_buffers),
                p_buffers=g.p_buffers, spec=g.spec, stage_linear=g.stage_linear,
                p_temporal=g.p_temporal, p_conv=g.p_conv, map_extent=mapper,
                log=log, cap0=[int(r) for r in d0.get("cap0", ())], l2=l2,
                verdict=verdict_for(rid))
        except Exception as exc:  # noqa: BLE001 -- the store read serves
            why = "%s: %s" % (type(exc).__name__, exc)
            verdict_for(rid)(False)   # no-op when this stage already voted
        finally:
            mapper.close()
        if why is None:
            adopted += 1
        else:
            log("HOT-HANDOVER rid=%s at=wake fallback=%s" % (rid, why))
    return adopted
