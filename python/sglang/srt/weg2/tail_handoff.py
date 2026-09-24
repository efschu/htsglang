"""fnFL2 H18 (E1 of H17): hand the END of a finished P prompt to D.

THE COST THIS REMOVES (fnFL2x132, rid weg2-0-4, N=97841, page 64). P cut the
last chunk on the page grid (``_weg2_end_anchor_split``, grain = page under
QSA), the radix tree keeps whole pages only, and the GDN anchor sits on the
64-token grid. D resumed at 97792 and extended 49 tokens through all 48
layers -- a nearly full expert sweep (0.20 GiB H2D per layer, 840-968 ms),
the critical path of the flip.

THE HAND-OFF. P cuts the last-but-one chunk at ``c = floor_r(N-1)`` with r
the QSA compress ratio (4): QSA only refuses a prefix that splits a group
(qwen_sparse_attn_backend ``prefix_lens % ratio == 0``). The tree node and
its GDN anchor stay at ``floor_page(c)`` (the chunk [X, c) tracks its anchor
through the upstream "unaligned -> retrieve from h" path). What exists only
on P is captured per PP rank:

* the GDN working slot after the chunk [X, c) -- the recurrent state after
  EXACTLY c tokens -- copied on the forward stream at the stash of that chunk,
  i.e. after its forward and before the forward of the final chunk
  (``capture_state``);
* the KV rows and the QSA compressed rows of the partial page
  [floor_page(c), c), read at ``cache_finished_req`` before the unaligned
  tail is freed (``publish_rows``).

Both land as one part file per rank in ``<arena dir>/handoff`` beside the
#1442 hand-off, with a small JSON header (``TailHeader``) carrying the key,
the covered global layer ids, the row shapes and a digest per section. The
key is a hash of the token ids [0, c) and the extra key: token-exact, so a
part can never be applied to another prompt (needle safety).

D probes the parts at its load-back (``probe``): key, prefix, layer coverage,
shapes against its own pools. The adoption itself (page, rows, state into the
request slot after the deferred COW, extend [c, N), group MIN verdict) is
NOT wired in this commit -- see the H18 report; the pure pieces it needs
(``agree_cut``, ``extend_range``, ``verify_part``) are here and tested.
"""

from __future__ import annotations

from array import array
import glob
import hashlib
import logging
import os
import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: rids whose part files P keeps (older ones are removed at the next publish);
#: one 97k prompt is ~78 MiB of GDN state across the P ranks (host RAM law).
KEEP_RIDS = 2


def enabled() -> bool:
    return bool(envs.SGLANG_WEG2_TAIL_HANDOFF.get())


# -- geometry (pure) -------------------------------------------------------------
def anchor_grain(page_size: int, qsa_ratio: Optional[int]) -> int:
    """Where the end-of-prefill cut may land. 1 without QSA (unchanged);
    under QSA the compress ratio when the tail hand-off is on (a prefix may
    not split a compressed group), else the page (fnFL2x14 form)."""
    if int(page_size or 1) <= 1 or qsa_ratio is None:
        return 1
    return int(qsa_ratio) if enabled() else int(page_size)


def tail_cut(n_tokens: int, grain: int) -> int:
    """c = floor_grain(N-1): the deepest position a reader may claim."""
    return (int(n_tokens) - 1) // int(grain) * int(grain)


def page_floor(pos: int, page_size: int) -> int:
    return int(pos) // int(page_size) * int(page_size)


def tail_key(ids: Sequence[int], cut: int, extra_key: Optional[str]) -> str:
    """Token-exact key of the prefix [0, cut) (+ extra key)."""
    h = hashlib.sha256()
    h.update(str(extra_key or "").encode())
    h.update(b"\0")
    h.update(array("q", ids[:cut]).tobytes())
    return h.hexdigest()[:32]


class TailSpec(msgspec.Struct, frozen=True):
    rid: str
    n_tokens: int
    page_prefix: int
    cut: int
    key: str

    @property
    def rows(self) -> int:
        return self.cut - self.page_prefix

    @property
    def extend(self) -> int:
        return self.n_tokens - self.cut


def spec_for(rid: str, ids: Sequence[int], extra_key: Optional[str], page_size: int, grain: int) -> Optional[TailSpec]:
    """The hand-off of a prompt of len(ids) tokens, or None when there is
    nothing to hand over (no partial page below the cut, or no grain gain)."""
    n = len(ids)
    if n < 2 or int(grain) >= int(page_size):
        return None
    cut = tail_cut(n, grain)
    prefix = page_floor(cut, page_size)
    if cut <= prefix:
        return None
    return TailSpec(rid=str(rid), n_tokens=n, page_prefix=prefix, cut=cut, key=tail_key(ids, cut, extra_key))


def extend_range(spec: TailSpec) -> Tuple[int, int]:
    """D's extend after an adopted tail: [c, N)."""
    return spec.cut, spec.n_tokens


def agree_cut(page_prefix: int, cut: int, local_ok: bool, reduce_min: Callable[[int], int]) -> int:
    """The group's resume depth: c only when EVERY rank can serve it (a rank
    without GDN/KV layers answers 1, like FormAWorkerNullStorage), else the
    page prefix. ``reduce_min`` is the MIN all-reduce over the whole tp group
    (identity on a single rank)."""
    return int(cut) if int(reduce_min(1 if local_ok else 0)) == 1 else int(page_prefix)


def digest(tensors: Sequence[torch.Tensor]) -> str:
    """SEAM-digest style: sha1 over the raw bytes of CPU tensors, in order."""
    h = hashlib.sha1()
    for t in tensors:
        h.update(t.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


# -- file layout -----------------------------------------------------------------
class TailHeader(msgspec.Struct, frozen=True):
    spec: TailSpec
    part: str
    fa_layers: List[int]
    gdn_layers: List[int]
    fa_row_shapes: Dict[str, List[int]]
    gdn_row_shapes: Dict[str, List[int]]
    fa_digest: str
    gdn_digest: str
    nbytes: int


def _dir() -> str:
    base = os.environ.get("SGLANG_HICACHE_ARENA_DIR", "").strip()
    return os.path.join(base, "handoff") if base else ""


def part_paths(rid: str, part: str) -> Tuple[str, str]:
    d = _dir()
    stem = os.path.join(d, f"{rid}.tail.{part}")
    return f"{stem}.json", f"{stem}.pt"


def headers_for(rid: str) -> List[TailHeader]:
    d = _dir()
    if not d:
        return []
    out = []
    for p in sorted(glob.glob(os.path.join(d, f"{glob.escape(rid)}.tail.*.json"))):
        try:
            with open(p, "rb") as f:
                out.append(msgspec.json.decode(f.read(), type=TailHeader))
        except (OSError, msgspec.DecodeError):
            logger.warning("WEG2-TAIL header unreadable: %s", p, exc_info=True)
    return out


def _fa_order(bundle_fa: Dict[int, Tuple[torch.Tensor, ...]]) -> List[torch.Tensor]:
    return [t for gid in sorted(bundle_fa) for t in bundle_fa[gid]]


def _gdn_order(bundle_gdn: Dict[int, Tuple[torch.Tensor, ...]]) -> List[torch.Tensor]:
    return [t for gid in sorted(bundle_gdn) for t in bundle_gdn[gid]]


def write_part(spec: TailSpec, part: str, fa: Dict[int, Tuple[torch.Tensor, ...]],
               gdn: Dict[int, Tuple[torch.Tensor, ...]]) -> Optional[TailHeader]:
    """Atomically write one rank's part (payload first, header last: a
    present header means a complete payload)."""
    if not _dir():
        return None
    fa_t, gdn_t = _fa_order(fa), _gdn_order(gdn)
    header = TailHeader(
        spec=spec, part=part, fa_layers=sorted(fa), gdn_layers=sorted(gdn),
        fa_row_shapes={str(g): list(fa[g][0].shape[1:]) for g in sorted(fa)},
        gdn_row_shapes={str(g): list(gdn[g][0].shape[1:]) for g in sorted(gdn)},
        fa_digest=digest(fa_t), gdn_digest=digest(gdn_t),
        nbytes=sum(int(t.numel() * t.element_size()) for t in fa_t + gdn_t),
    )
    jpath, ppath = part_paths(spec.rid, part)
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    tmp = f"{ppath}.{os.getpid()}.tmp"
    torch.save({"fa": fa, "gdn": gdn}, tmp)
    os.replace(tmp, ppath)
    tmp = f"{jpath}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(msgspec.json.encode(header))
    os.replace(tmp, jpath)
    return header


def verify_part(header: TailHeader) -> Optional[dict]:
    """Load a part and check both digests; None on any mismatch (the reader
    then keeps the page-prefix resume -- a wrong row is never applied)."""
    _j, ppath = part_paths(header.spec.rid, header.part)
    try:
        bundle = torch.load(ppath, map_location="cpu")
    except (OSError, RuntimeError, EOFError):
        logger.warning("WEG2-TAIL part unreadable: %s", ppath, exc_info=True)
        return None
    if digest(_fa_order(bundle["fa"])) != header.fa_digest or digest(_gdn_order(bundle["gdn"])) != header.gdn_digest:
        logger.warning("WEG2-TAIL DIGEST MISMATCH rid=%s part=%s", header.spec.rid, header.part)
        return None
    return bundle


def remove(rid: str) -> None:
    d = _dir()
    if not d:
        return
    for p in glob.glob(os.path.join(d, f"{glob.escape(rid)}.tail.*")):
        try:
            os.remove(p)
        except OSError:
            pass


def _prune(keep_rid: str) -> None:
    """Keep the part files of the KEEP_RIDS newest rids (by mtime)."""
    d = _dir()
    if not d:
        return
    newest: Dict[str, float] = {}
    for p in glob.glob(os.path.join(d, "*.tail.*.json")):
        rid = os.path.basename(p).split(".tail.", 1)[0]
        try:
            newest[rid] = max(newest.get(rid, 0.0), os.path.getmtime(p))
        except OSError:
            pass
    newest.pop(keep_rid, None)
    for rid in sorted(newest, key=newest.get, reverse=True)[KEEP_RIDS - 1:]:
        remove(rid)


# -- P side: capture + publish -----------------------------------------------------
class _Capture(msgspec.Struct):
    spec: TailSpec
    gdn: Dict[int, Tuple[torch.Tensor, ...]]
    event: object


_CAPTURES: Dict[str, _Capture] = {}
_PUBLISH_N = [0]


def _grain_of(allocator, page_size: int) -> int:
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    kv = allocator.get_kvcache()
    ratio = int(kv.qsa_compress_ratio) if isinstance(kv, QSATokenToKVPool) else None
    return anchor_grain(page_size, ratio)


def _is_p_request(req) -> bool:
    from sglang.srt.managers.schedule_policy import _WEG2_END_ANCHOR

    return enabled() and _WEG2_END_ANCHOR and str(req.rid).startswith("weg2-") and bool(_dir())


def capture_state(req, req_to_token_pool, allocator, page_size: int, stream) -> bool:
    """Scheduler entry (stash of a chunk): never raises into the scheduler."""
    try:
        return _capture_state(req, req_to_token_pool, allocator, page_size, stream)
    except Exception as exc:  # noqa: BLE001 -- a failed capture is the page-prefix resume, named
        logger.warning("WEG2-TAIL capture failed rid=%s (%s: %s)", req.rid, type(exc).__name__, exc)
        _CAPTURES.pop(str(req.rid), None)
        return False


def _capture_state(req, req_to_token_pool, allocator, page_size: int, stream) -> bool:
    """At the stash of the chunk that ended at c: copy this rank's GDN
    working slot (the state after exactly c tokens) to pinned host memory ON
    THE FORWARD STREAM -- after that chunk's forward, before the final
    chunk's. True = a capture was taken."""
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    if not _is_p_request(req) or not isinstance(req_to_token_pool, HybridReqToTokenPool):
        return False
    if req.mamba_pool_idx is None:
        return False
    ids = req.origin_input_ids
    spec = spec_for(req.rid, ids, req.extra_key, page_size, _grain_of(allocator, page_size))
    if spec is None or int(req.extend_range.end) != spec.cut or len(req.full_untruncated_fill_ids) != spec.n_tokens:
        return False
    pool = req_to_token_pool
    phys = pool.translate_mamba_indices(req.mamba_pool_idx.reshape(1).to(torch.int64))
    cache = pool.mamba_pool.mamba_cache
    pin = torch.cuda.is_available()
    gdn: Dict[int, Tuple[torch.Tensor, ...]] = {}
    ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
    with ctx:
        phys = phys.to(cache.temporal.device, non_blocking=True)
        for gid, local in sorted(pool.mamba_map.items()):
            parts = [cache.temporal[local].index_select(0, phys)] + [c[local].index_select(0, phys) for c in cache.conv]
            host = []
            for t in parts:
                h = torch.empty(t.shape, dtype=t.dtype, pin_memory=pin)
                h.copy_(t, non_blocking=pin)
                host.append(h)
            gdn[int(gid)] = tuple(host)
        event = torch.cuda.Event() if pin else None
        if event is not None:
            event.record(stream if stream is not None else torch.cuda.current_stream())
    _CAPTURES[str(req.rid)] = _Capture(spec=spec, gdn=gdn, event=event)
    while len(_CAPTURES) > KEEP_RIDS:  # an aborted prompt never publishes: drop the oldest
        _CAPTURES.pop(next(iter(_CAPTURES)))
    return True


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fa_rows(kvpool, rows: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, ...]]:
    """K, V (and the QSA compressed index rows) of every full-attention layer
    of this rank at the device token slots ``rows``."""
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    full = kvpool.full_kv_pool
    qsa = isinstance(kvpool, QSATokenToKVPool)
    out: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for gid, local in sorted(kvpool.full_attention_layer_id_mapping.items()):
        dev_rows = rows.to(full.k_buffer[local].device)
        k = full.k_buffer[local].index_select(0, dev_rows).cpu()
        v = full.v_buffer[local].index_select(0, dev_rows).cpu()
        if qsa:
            ratio = int(kvpool.qsa_compress_ratio)
            groups = dev_rows[::ratio] // ratio
            c = kvpool.qsa_compressed_k_buffer_pool[local].index_select(0, groups).cpu()
            out[int(gid)] = (k, v, c)
        else:
            out[int(gid)] = (k, v)
    return out


def publish_rows(req, kv_indices: torch.Tensor, allocator, part: str) -> None:
    """cache_finished_req entry: never raises into the tree."""
    if not _CAPTURES:
        return
    try:
        _publish_rows(req, kv_indices, allocator, part)
    except Exception as exc:  # noqa: BLE001 -- a failed publish is the page-prefix resume, named
        logger.warning("WEG2-TAIL-PUBLISH failed rid=%s (%s: %s)", req.rid, type(exc).__name__, exc)
        _CAPTURES.pop(str(req.rid), None)


def _publish_rows(req, kv_indices: torch.Tensor, allocator, part: str) -> None:
    """At cache_finished_req, before the unaligned tail is freed: read the
    partial page's rows, join the state capture, write this rank's part."""
    cap = _CAPTURES.pop(str(req.rid), None)
    if cap is None:
        return None
    spec = cap.spec
    if int(kv_indices.numel()) < spec.cut:
        logger.warning("WEG2-TAIL-PUBLISH refused rid=%s: kv rows %d < cut %d", spec.rid, int(kv_indices.numel()), spec.cut)
        return
    if cap.event is not None:
        # recorded on the forward stream after the chunk [X, c): once it has
        # fired, that chunk's KV rows are written too (stream order)
        cap.event.synchronize()
    fa = _fa_rows(allocator.get_kvcache(), kv_indices[spec.page_prefix:spec.cut].to(torch.int64))
    threading.Thread(target=_write_and_log, args=(spec, part, fa, cap.gdn), daemon=True,
                     name="weg2-tail-publish").start()


def _write_and_log(spec: TailSpec, part: str, fa, gdn) -> None:
    try:
        header = write_part(spec, part, fa, gdn)
        _prune(spec.rid)
    except Exception:  # noqa: BLE001 -- a hand-off that fails is the page-prefix resume
        logger.warning("WEG2-TAIL-PUBLISH write failed rid=%s", spec.rid, exc_info=True)
        return
    if header is None:
        return
    _PUBLISH_N[0] += 1
    logger.info(
        "WEG2-TAIL-PUBLISH rid=%s page_prefix=%d tail_rows=%d state_at=%d n_tokens=%d part=%s "
        "fa_layers=%d gdn_layers=%d bytes=%d key=%s fa_digest=%s gdn_digest=%s (n=%d)",
        spec.rid, spec.page_prefix, spec.rows, spec.cut, spec.n_tokens, part, len(header.fa_layers),
        len(header.gdn_layers), header.nbytes, spec.key, header.fa_digest, header.gdn_digest, _PUBLISH_N[0],
    )


# -- D side: probe -------------------------------------------------------------------
def local_readiness(spec: TailSpec, headers: Sequence[TailHeader], need_fa: Dict[int, List[int]],
                    need_gdn: Dict[int, List[int]]) -> str:
    """'' when the parts cover every layer this rank holds with matching row
    shapes and one agreed spec; otherwise the first reason it cannot serve."""
    if not headers:
        return "no_parts"
    for h in headers:
        if h.spec != spec:
            return f"spec_differs:{h.part}"
    have_fa = {g: h.fa_row_shapes[str(g)] for h in headers for g in h.fa_layers}
    have_gdn = {g: h.gdn_row_shapes[str(g)] for h in headers for g in h.gdn_layers}
    for g, shape in need_fa.items():
        if g not in have_fa:
            return f"fa_layer_missing:{g}"
        if list(have_fa[g]) != list(shape):
            return f"fa_shape:{g}:{have_fa[g]}!={list(shape)}"
    for g, shape in need_gdn.items():
        if g not in have_gdn:
            return f"gdn_layer_missing:{g}"
        if list(have_gdn[g]) != list(shape):
            return f"gdn_shape:{g}:{have_gdn[g]}!={list(shape)}"
    return ""


def _need_shapes(kvpool, req_to_token_pool) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool

    need_fa: Dict[int, List[int]] = {}
    if isinstance(kvpool, HybridLinearKVPool):
        full = kvpool.full_kv_pool
        for gid, local in kvpool.full_attention_layer_id_mapping.items():
            need_fa[int(gid)] = list(full.k_buffer[local].shape[1:])
    need_gdn: Dict[int, List[int]] = {}
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        temporal = req_to_token_pool.mamba_pool.mamba_cache.temporal
        for gid, local in req_to_token_pool.mamba_map.items():
            need_gdn[int(gid)] = list(temporal[local].shape[1:])
    return need_fa, need_gdn


_PROBE_N = [0]


def probe(req, prefix_len: int, tree_cache, page_size: int) -> str:
    """Admission entry (#988 site): never raises into add_one_req."""
    try:
        return _probe(req, prefix_len, tree_cache, page_size)
    except Exception as exc:  # noqa: BLE001 -- a probe is an instrument, named
        logger.warning("WEG2-TAIL-READY probe failed rid=%s (%s: %s)", req.rid, type(exc).__name__, exc)
        return "probe_raised"


def _probe(req, prefix_len: int, tree_cache, page_size: int) -> str:
    """D, at the #988 load-back: can this rank serve the tail of ``req``?
    Logs one WEG2-TAIL-READY line per probed request; returns the verdict
    ('' = ready). Reads only the small headers (no payload on the admission
    path)."""
    if not enabled() or not str(req.rid).startswith("weg2-") or not _dir():
        return "off"
    headers = headers_for(str(req.rid))
    if not headers:
        return "no_parts"
    spec = headers[0].spec
    ids = req.origin_input_ids
    verdict = ""
    if len(ids) != spec.n_tokens:
        verdict = f"n_tokens:{len(ids)}!={spec.n_tokens}"
    elif int(prefix_len) != spec.page_prefix:
        verdict = f"prefix:{int(prefix_len)}!={spec.page_prefix}"
    elif tail_key(ids, spec.cut, req.extra_key) != spec.key:
        verdict = "key_mismatch"
    else:
        need_fa, need_gdn = _need_shapes(tree_cache.token_to_kv_pool_allocator.get_kvcache(), tree_cache.req_to_token_pool)
        verdict = local_readiness(spec, headers, need_fa, need_gdn)
    _PROBE_N[0] += 1
    logger.info(
        "WEG2-TAIL-READY rid=%s page_prefix=%d tail_rows=%d state_at=%d extend=%d parts=%d key=%s "
        "verdict=%s adopt=not_wired (n=%d)",
        spec.rid, spec.page_prefix, spec.rows, spec.cut, spec.extend, len(headers), spec.key,
        verdict or "ready", _PROBE_N[0],
    )
    return verdict
