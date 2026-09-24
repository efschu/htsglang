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

D takes the parts over in weg2/tail_adopt.py (H21): a group MIN vote in the
prefetch-progress collective, the partial page as a request-owned page at the
admission commit, rows + state written at the extend's first GDN layer,
extend [c, N).

E2 (H24, ``SGLANG_WEG2_TAIL_SKIP_EXTEND``). P has computed the final chunk
[c, N) itself and sampled the first output token (leg 1 is
``max_new_tokens=1``), so at ``cache_finished_req`` it also owns the END
state: the KV rows [floor_page(c), N), the QSA pending-ring rows of the open
group [floor_r(N), N) (plus their RoPE positions), the GDN working slot after
all N tokens, and ``req.output_ids[-1]``. The same part file carries them as
an ``end`` section (``EndHeader`` + payload key ``"end"``), gathered on the
FORWARD STREAM after that chunk's forward (no host sync on P) and written by
the publish thread once the gather's event fired. D then needs no extend
forward at all (weg2/tail_adopt.py, "SKIP").
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
#: fnFL2 H42: with several END-ANCHOR tails per pass
#: (SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS) one P phase leaves one capture and
#: one part-file set PER OPEN REQUEST -- bounded by P's --max-running-requests
#: (the request slots P holds until the flip). Fallback when the server args are
#: unreadable (a desk test): the p_bs-4 form. ~26 MiB of GDN state per rid and
#: rank either way (host RAM law: grows with p_bs, not with the prompt).
KEEP_CAPTURES_MULTI_TAIL = 4


def capture_keep() -> int:
    """How many rids' captures and part files P keeps (fnFL2 H42)."""
    if not envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.get():
        return KEEP_RIDS
    try:
        from sglang.srt.runtime_context import get_server_args

        mrr = int(get_server_args().max_running_requests or 0)
    except Exception:  # noqa: BLE001 -- no server args (desk): the p_bs-4 form
        mrr = KEEP_CAPTURES_MULTI_TAIL
    if mrr <= 0:  # unset: the p_bs-4 form
        mrr = KEEP_CAPTURES_MULTI_TAIL
    return max(KEEP_RIDS, mrr)


def enabled() -> bool:
    return bool(envs.SGLANG_WEG2_TAIL_HANDOFF.get())


def skip_extend_enabled() -> bool:
    """E2 (H24): SGLANG_WEG2_TAIL_SKIP_EXTEND under TAIL_HANDOFF + TAIL_ADOPT."""
    return (
        enabled()
        and bool(envs.SGLANG_WEG2_TAIL_ADOPT.get())
        and bool(envs.SGLANG_WEG2_TAIL_SKIP_EXTEND.get())
    )


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


def end_geometry(spec: TailSpec, ratio: int) -> Tuple[int, int, int]:
    """E2: (rows, complete_groups, ring_rows) of the END section -- the rows
    [page_prefix, N), the QSA groups in them that are complete (their
    compressed row exists), and the open group's members [floor_r(N), N)
    that live only in the per-request pending ring. ratio 0 = no QSA."""
    rows = spec.n_tokens - spec.page_prefix
    if not ratio:
        return rows, 0, 0
    return rows, rows // int(ratio), spec.n_tokens % int(ratio)


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
class EndHeader(msgspec.Struct, frozen=True):
    """E2: the END section of a part -- state after all N tokens."""

    first_token: int
    key: str  # tail_key(ids, N): the section belongs to exactly this prompt
    rows: int  # N - page_prefix KV rows
    groups: int  # complete QSA groups among them (0 without QSA)
    ring_rows: int  # N % ratio open-group members (0 without QSA)
    fa_digest: str
    gdn_digest: str
    ring_digest: str
    nbytes: int


class EndPayload(msgspec.Struct):
    """E2 payload of one rank, host tensors once ``event`` fired. fa: gid ->
    (K, V[, compressed]) rows; gdn: gid -> (temporal, conv...) after N; ring:
    gid -> (index-K pending rows,); rope: [ring_rows, 3] or None."""

    first_token: int
    key: str
    rows: int
    groups: int
    ring_rows: int
    fa: Dict[int, Tuple[torch.Tensor, ...]]
    gdn: Dict[int, Tuple[torch.Tensor, ...]]
    ring: Dict[int, Tuple[torch.Tensor, ...]]
    rope: Optional[torch.Tensor]
    event: object = None


def ring_order(ring: Dict[int, Tuple[torch.Tensor, ...]], rope: Optional[torch.Tensor]) -> List[torch.Tensor]:
    return _fa_order(ring) + ([rope] if rope is not None else [])


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
    end: Optional[EndHeader] = None


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


def _nbytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(int(t.numel() * t.element_size()) for t in tensors)


def _end_header(end: EndPayload) -> EndHeader:
    fa_t, gdn_t, ring_t = _fa_order(end.fa), _gdn_order(end.gdn), ring_order(end.ring, end.rope)
    return EndHeader(
        first_token=int(end.first_token), key=end.key, rows=int(end.rows), groups=int(end.groups),
        ring_rows=int(end.ring_rows), fa_digest=digest(fa_t), gdn_digest=digest(gdn_t),
        ring_digest=digest(ring_t), nbytes=_nbytes(fa_t + gdn_t + ring_t),
    )


def write_part(spec: TailSpec, part: str, fa: Dict[int, Tuple[torch.Tensor, ...]],
               gdn: Dict[int, Tuple[torch.Tensor, ...]], end: Optional[EndPayload] = None) -> Optional[TailHeader]:
    """Atomically write one rank's part (payload first, header last: a
    present header means a complete payload). ``end`` (E2) rides along as
    payload key "end" and header field ``end``."""
    if not _dir():
        return None
    fa_t, gdn_t = _fa_order(fa), _gdn_order(gdn)
    header = TailHeader(
        spec=spec, part=part, fa_layers=sorted(fa), gdn_layers=sorted(gdn),
        fa_row_shapes={str(g): list(fa[g][0].shape[1:]) for g in sorted(fa)},
        gdn_row_shapes={str(g): list(gdn[g][0].shape[1:]) for g in sorted(gdn)},
        fa_digest=digest(fa_t), gdn_digest=digest(gdn_t),
        nbytes=_nbytes(fa_t + gdn_t),
        end=_end_header(end) if end is not None else None,
    )
    jpath, ppath = part_paths(spec.rid, part)
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    tmp = f"{ppath}.{os.getpid()}.tmp"
    bundle = {"fa": fa, "gdn": gdn}
    if end is not None:
        bundle["end"] = {"fa": end.fa, "gdn": end.gdn, "ring": end.ring, "rope": end.rope}
    torch.save(bundle, tmp)
    os.replace(tmp, ppath)
    tmp = f"{jpath}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(msgspec.json.encode(header))
    os.replace(tmp, jpath)
    return header


def read_part(header: TailHeader, check_digest: bool = True) -> Tuple[Optional[dict], str]:
    """Load a part; with ``check_digest`` compare both sections against the
    publish digests. (bundle, '') or (None, reason) -- 'unreadable' or
    'digest_MISMATCH' (the reader then keeps the page-prefix resume: a wrong
    row is never applied)."""
    _j, ppath = part_paths(header.spec.rid, header.part)
    try:
        bundle = torch.load(ppath, map_location="cpu")
    except (OSError, RuntimeError, EOFError):
        logger.warning("WEG2-TAIL part unreadable: %s", ppath, exc_info=True)
        return None, "unreadable"
    if check_digest and (
        digest(_fa_order(bundle["fa"])) != header.fa_digest
        or digest(_gdn_order(bundle["gdn"])) != header.gdn_digest
    ):
        logger.warning("WEG2-TAIL DIGEST MISMATCH rid=%s part=%s", header.spec.rid, header.part)
        return None, "digest_MISMATCH"
    return bundle, ""


def end_digest_refusal(header: TailHeader, bundle: dict) -> str:
    """E2: '' when the bundle's END section matches its header digests,
    else 'end_missing' | 'end_digest_MISMATCH' (the E1 section is unaffected)."""
    end, sec = header.end, bundle.get("end")
    if end is None or sec is None:
        return "end_missing"
    if (
        digest(_fa_order(sec["fa"])) != end.fa_digest
        or digest(_gdn_order(sec["gdn"])) != end.gdn_digest
        or digest(ring_order(sec["ring"], sec["rope"])) != end.ring_digest
    ):
        logger.warning("WEG2-TAIL END DIGEST MISMATCH rid=%s part=%s", header.spec.rid, header.part)
        return "end_digest_MISMATCH"
    return ""


def verify_part(header: TailHeader) -> Optional[dict]:
    """``read_part`` with both digests checked; None on any refusal."""
    return read_part(header, check_digest=True)[0]


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
    """Keep the part files of the ``capture_keep()`` newest rids (by mtime)."""
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
    for rid in sorted(newest, key=newest.get, reverse=True)[capture_keep() - 1:]:
        remove(rid)


# -- P side: capture + publish -----------------------------------------------------
class _Capture(msgspec.Struct):
    spec: TailSpec
    gdn: Dict[int, Tuple[torch.Tensor, ...]]
    event: object
    #: the scheduler's forward stream (E2 gathers the end state on it)
    stream: object = None


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
    ctx = torch.cuda.stream(stream) if stream is not None else _null_ctx()
    with ctx:
        gdn = _gdn_slot_to_host(req_to_token_pool, req.mamba_pool_idx)
        event = _record(stream)
    _CAPTURES[str(req.rid)] = _Capture(spec=spec, gdn=gdn, event=event, stream=stream)
    while len(_CAPTURES) > capture_keep():  # an aborted prompt never publishes: drop the oldest
        _CAPTURES.pop(next(iter(_CAPTURES)))
    return True


def _to_host(t: torch.Tensor) -> torch.Tensor:
    """Async D2H into pinned memory on the current stream (CPU: a copy)."""
    pin = t.is_cuda
    h = torch.empty(t.shape, dtype=t.dtype, pin_memory=pin)
    h.copy_(t, non_blocking=pin)
    return h


def _record(stream) -> object:
    if not torch.cuda.is_available():
        return None
    event = torch.cuda.Event()
    event.record(stream if stream is not None else torch.cuda.current_stream())
    return event


def _gdn_slot_to_host(pool, mamba_pool_idx: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, ...]]:
    """Every local GDN layer's temporal + conv state of one request slot,
    gathered on the CURRENT stream into host tensors (keyed by global id)."""
    cache = pool.mamba_pool.mamba_cache
    phys = pool.translate_mamba_indices(mamba_pool_idx.reshape(1).to(torch.int64))
    phys = phys.to(cache.temporal.device, non_blocking=True)
    gdn: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for gid, local in sorted(pool.mamba_map.items()):
        parts = [cache.temporal[local].index_select(0, phys)] + [c[local].index_select(0, phys) for c in cache.conv]
        gdn[int(gid)] = tuple(_to_host(t) for t in parts)
    return gdn


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def group_slots(rows: torch.Tensor, ratio: int, groups: Optional[int] = None) -> torch.Tensor:
    """Compressed slots (slot // ratio) of the first ``groups`` groups of the
    page-aligned token slots ``rows`` (all rows//ratio groups when None)."""
    n = len(rows) // int(ratio) if groups is None else int(groups)
    return rows[: n * int(ratio) : int(ratio)] // int(ratio)


def _fa_rows(kvpool, rows: torch.Tensor, groups: Optional[int] = None,
             host: Callable[[torch.Tensor], torch.Tensor] = torch.Tensor.cpu) -> Dict[int, Tuple[torch.Tensor, ...]]:
    """K, V (and the QSA compressed index rows of the first ``groups``
    complete groups) of every full-attention layer of this rank at the device
    token slots ``rows``; ``host`` moves each gathered tensor off the card."""
    full = kvpool.full_kv_pool
    ratio = _qsa_ratio(kvpool)
    out: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for gid, local in sorted(kvpool.full_attention_layer_id_mapping.items()):
        dev_rows = rows.to(full.k_buffer[local].device)
        k = host(full.k_buffer[local].index_select(0, dev_rows))
        v = host(full.v_buffer[local].index_select(0, dev_rows))
        if ratio:
            slots = group_slots(dev_rows, ratio, groups)
            c = host(kvpool.qsa_compressed_k_buffer_pool[local].index_select(0, slots))
            out[int(gid)] = (k, v, c)
        else:
            out[int(gid)] = (k, v)
    return out


def publish_rows(req, kv_indices: torch.Tensor, allocator, part: str, req_to_token_pool=None) -> None:
    """cache_finished_req entry: never raises into the tree."""
    if not _CAPTURES:
        return
    try:
        _publish_rows(req, kv_indices, allocator, part, req_to_token_pool)
    except Exception as exc:  # noqa: BLE001 -- a failed publish is the page-prefix resume, named
        logger.warning("WEG2-TAIL-PUBLISH failed rid=%s (%s: %s)", req.rid, type(exc).__name__, exc)
        _CAPTURES.pop(str(req.rid), None)


def _publish_rows(req, kv_indices: torch.Tensor, allocator, part: str, req_to_token_pool) -> None:
    """At cache_finished_req, before the unaligned tail is freed: read the
    partial page's rows, join the state capture, write this rank's part."""
    cap = _CAPTURES.pop(str(req.rid), None)
    if cap is None:
        return None
    spec = cap.spec
    if int(kv_indices.numel()) < spec.cut:
        logger.warning("WEG2-TAIL-PUBLISH refused rid=%s: kv rows %d < cut %d", spec.rid, int(kv_indices.numel()), spec.cut)
        return
    # E2 first: enqueued on the forward stream BEHIND the final chunk's
    # forward, never a host wait here (P's next microbatch may be running)
    try:
        end = _capture_end(req, kv_indices, allocator, req_to_token_pool, cap)
    except Exception as exc:  # noqa: BLE001 -- a failed END capture is E1, named
        logger.warning("WEG2-TAIL-PUBLISH end=none rid=%s reason=raised:%s: %s", spec.rid, type(exc).__name__, exc)
        end = None
    if cap.event is not None:
        # recorded on the forward stream after the chunk [X, c): once it has
        # fired, that chunk's KV rows are written too (stream order)
        cap.event.synchronize()
    fa = _fa_rows(allocator.get_kvcache(), kv_indices[spec.page_prefix:spec.cut].to(torch.int64))
    threading.Thread(target=_write_and_log, args=(spec, part, fa, cap.gdn, end), daemon=True,
                     name="weg2-tail-publish").start()


def end_refusal(req, kv_rows: int, spec: TailSpec) -> str:
    """E2 on P: '' when this finished request can hand over its END state,
    else why not (named in the publish line; D then adopts E1)."""
    if not skip_extend_enabled():
        return "off"
    if kv_rows < spec.n_tokens:
        return f"kv_rows:{kv_rows}<{spec.n_tokens}"
    if not req.output_ids:
        return "no_sampled_token"
    if req.return_logprob or req.return_hidden_states:
        return "logprob_or_hidden_requested"
    return ""


def _capture_end(req, kv_indices: torch.Tensor, allocator, req_to_token_pool, cap: _Capture) -> Optional[EndPayload]:
    """E2: gather rows [page_prefix, N), the open group's pending-ring rows
    and the GDN slot (state after N) on the forward stream; None + a named
    reason when the END state cannot be handed over."""
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    spec = cap.spec
    if not skip_extend_enabled():
        return None
    if not isinstance(req_to_token_pool, HybridReqToTokenPool) or req.mamba_pool_idx is None:
        why = "no_mamba_slot"
    else:
        why = end_refusal(req, int(kv_indices.numel()), spec)
    if why:
        logger.info("WEG2-TAIL-PUBLISH end=none rid=%s reason=%s", spec.rid, why)
        return None
    kvpool = allocator.get_kvcache()
    ratio = _qsa_ratio(kvpool)
    rows, groups, ring_rows = end_geometry(spec, ratio)
    ctx = torch.cuda.stream(cap.stream) if cap.stream is not None else _null_ctx()
    with ctx:
        fa = _fa_rows(kvpool, kv_indices[spec.page_prefix:spec.n_tokens].to(torch.int64), groups=groups, host=_to_host)
        ring, rope = _ring_rows(kvpool, req.req_pool_idx, ring_rows)
        gdn = _gdn_slot_to_host(req_to_token_pool, req.mamba_pool_idx)
        event = _record(cap.stream)
    return EndPayload(
        first_token=int(req.output_ids[-1]), key=tail_key(req.origin_input_ids, spec.n_tokens, req.extra_key),
        rows=rows, groups=groups, ring_rows=ring_rows, fa=fa, gdn=gdn, ring=ring, rope=rope, event=event,
    )


def _qsa_ratio(kvpool) -> int:
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    return int(kvpool.qsa_compress_ratio) if isinstance(kvpool, QSATokenToKVPool) else 0


def ring_slots(req_pool_idx: int, ratio: int, ring_rows: int) -> torch.Tensor:
    """Pending-ring rows of the open group: ``req_pool_idx * ratio +
    position % ratio`` for positions [floor_r(N), N) = offsets [0, N % r)."""
    return int(req_pool_idx) * int(ratio) + torch.arange(int(ring_rows), dtype=torch.int64)


def _ring_rows(kvpool, req_pool_idx: int, ring_rows: int):
    """(gid -> (index-K ring rows,), rope rows) of the open QSA group."""
    ratio = _qsa_ratio(kvpool)
    if not ratio:
        return {}, None
    idx = ring_slots(req_pool_idx, ratio, ring_rows).to(kvpool.qsa_rope_position_buffer.device)
    ring = {
        int(gid): (_to_host(kvpool.qsa_key_state_buffer_pool[local].index_select(0, idx)),)
        for gid, local in sorted(kvpool.full_attention_layer_id_mapping.items())
    }
    return ring, _to_host(kvpool.qsa_rope_position_buffer.index_select(0, idx))


def _write_and_log(spec: TailSpec, part: str, fa, gdn, end: Optional[EndPayload] = None) -> None:
    try:
        if end is not None and end.event is not None:
            end.event.synchronize()  # the forward-stream gather has landed
        header = write_part(spec, part, fa, gdn, end=end)
        _prune(spec.rid)
    except Exception:  # noqa: BLE001 -- a hand-off that fails is the page-prefix resume
        logger.warning("WEG2-TAIL-PUBLISH write failed rid=%s", spec.rid, exc_info=True)
        return
    if header is None:
        return
    _PUBLISH_N[0] += 1
    if header.end is None:
        logger.info(
            "WEG2-TAIL-PUBLISH rid=%s page_prefix=%d tail_rows=%d state_at=%d n_tokens=%d part=%s "
            "fa_layers=%d gdn_layers=%d bytes=%d key=%s fa_digest=%s gdn_digest=%s (n=%d)",
            spec.rid, spec.page_prefix, spec.rows, spec.cut, spec.n_tokens, part, len(header.fa_layers),
            len(header.gdn_layers), header.nbytes, spec.key, header.fa_digest, header.gdn_digest, _PUBLISH_N[0],
        )
        return
    e = header.end
    logger.info(
        "WEG2-TAIL-PUBLISH rid=%s page_prefix=%d tail_rows=%d state_at=%d first_token=%d n_tokens=%d part=%s "
        "fa_layers=%d gdn_layers=%d groups=%d ring_rows=%d bytes=%d end_key=%s end_digests=%s/%s/%s "
        "| E1 tail_rows=%d state_at=%d bytes=%d key=%s fa_digest=%s gdn_digest=%s (n=%d)",
        spec.rid, spec.page_prefix, e.rows, spec.n_tokens, e.first_token, spec.n_tokens, part,
        len(header.fa_layers), len(header.gdn_layers), e.groups, e.ring_rows, e.nbytes, e.key, e.fa_digest,
        e.gdn_digest, e.ring_digest, spec.rows, spec.cut, header.nbytes, spec.key, header.fa_digest,
        header.gdn_digest, _PUBLISH_N[0],
    )


# -- D side: readiness (the adoption itself lives in weg2/tail_adopt.py) -------------
def local_readiness(spec: TailSpec, headers: Sequence[TailHeader], need_fa: Dict[int, List[int]],
                    need_gdn: Dict[int, List[int]]) -> str:
    """'' when the parts cover every layer this rank HOLDS with matching row
    shapes and one agreed spec; otherwise the first reason it cannot serve.
    ``need_*`` carry only held layers (weg2/tail_adopt.held_shapes drops a
    layer whose rows are empty, e.g. a Form-A worker's 0-head attention pool:
    fnFL2x133 TP1/TP2 'fa_shape:3:[2, 256]!=[0, 256]')."""
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
