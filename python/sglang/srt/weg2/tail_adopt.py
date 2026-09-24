"""fnFL2 H21 (second half of E1): D takes P's tail hand-off over.

H18 (weg2/tail_handoff.py) made P cut at ``c = floor_r(N-1)`` and publish,
per PP rank, the GDN state after exactly c tokens plus the KV/QSA rows of the
partial page [floor_page(c), c). D only probed them (``adopt=not_wired``) and
still extended [floor_page(c), N) -- 49 tokens through a full expert sweep
(fnFL2x133: flips 3.11/3.12 s). This module is the adoption.

THREE MOMENTS, EACH WHERE ITS INPUT IS RANK-UNIFORM OR RANK-LOCAL BY DESIGN

1. VOTE (``stage`` / ``local_vote`` / ``agree``; UnifiedRadixCache.
   check_prefetch_progress). Every rank of the TP group runs the SAME
   scheduler over the same requests and builds its own extend batch; the
   Form-A workers receive no batch from TP0 -- they derive it, and the
   hidden-state all-reduces of the forward (form_a_worker_forward /
   qwen2_moe) only line up when every rank chose the same extend length
   (fnFL2x132: moved_bytes=250880 = 49 tokens x 2560 x 2 on every one of the
   96 extend all-reduces). So the one rank-local fact -- can THIS rank write
   the rows it holds (parts readable, digests, shapes, dtypes) -- is voted
   into the existing packed MIN all-reduce of the prefetch completion (one
   more slot), which every rank passes for the request before its admission.
   A rank that holds no layer of the parts (Form-A expert worker: 0-head
   attention pool, 0-byte GDN state) votes 1 as ``not_mine``: it checks only
   what it holds, i.e. nothing. The payload is read and digest-checked by a
   background thread started at the first progress check, so the vote itself
   joins a finished thread.

2. ADOPT (``plan_adopt`` / ``commit_adopt``; PrefillAdder.add_one_req, at
   the whole-fit commit, right before ``can_run_list.append``). Inputs are
   rank-uniform only: the agreed vote, the prefix length after the load-back
   (== floor_page(c)), the token ids (key), N. Every rank allocates ONE page
   and grows ``prefix_indices`` by its first ``c - floor_page(c)`` slots --
   the request-owned partial page of an ordinary chunked continuation: the
   tree keeps [0, floor_page(c)) (``cache_protected_len`` stays there), the
   page joins the tree at ``cache_finished_req`` only once it is full
   (page-aligned insert), so no reader ever sees a 48/64 page as a prefix
   hit. The extend is [c, N); ``alloc_extend`` continues inside the page
   (last_loc = its slot c-1), no second page.

3. INSTALL (``install_layer``; HybridReqToTokenPool.mamba2_layer_cache, on
   the forward stream, after that layer's load-stream join). The rows and
   the state are written by the rank(s) holding them at the FIRST GDN layer
   read of the extend forward: all attention rows (K, V, QSA compressed
   groups) at the first call, each GDN layer's temporal + conv state at its
   own call. That point is after the deferred clear/COW (model_runner, before
   the layers) and after the load stream's step for that layer
   (``_wait_for_mamba_layer``), so neither the load-back of the
   floor_page(c) anchor into the same slot nor a COW can overwrite the state
   at c -- and it costs no global wait on the whole load. A GDN layer always
   precedes the first attention layer on the holding rank (vote checks it).

QSA: the compressed groups of [floor_page(c), c) are COMPLETE on P (c is a
multiple of the compress ratio, the chunk [X, c) compressed them) and are
carried in the part; the per-request pending ring holds nothing at a group
boundary. D therefore runs no indexer step for the partial page; its extend
starts a new group at c (the QSA ``prefix_lens % ratio == 0`` assert holds).

Any refusal, on any rank, before the commit: today's path (extend from
floor_page(c)), never an abort. After the commit the extend length is fixed
for the group; a post-write readback MISMATCH is reported loudly
(WEG2-TAIL-ADOPT digest=MISMATCH, ERROR) -- the digest gate that can still
fall back uniformly is the pre-write one inside the vote.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.weg2 import tail_handoff as th

logger = logging.getLogger(__name__)

#: staged reads and agreed/refused outcomes kept for their admission (D
#: admits up to its decode bs (6) flipped prompts; a request aborted before its
#: admission leaves one entry behind, pruned oldest-first -- identically on
#: every rank, since the entries are made in the collective's order). TP0
#: holds ~60 MB pinned per agreed 97k prompt until its install.
KEEP_AGREED = 8
#: how long the vote waits for a staging thread that is still reading.
STAGE_JOIN_S = 1.0
READY_VERDICTS = ("ready", "not_mine")


def adopt_enabled() -> bool:
    """SGLANG_WEG2_TAIL_ADOPT, effective only under TAIL_HANDOFF."""
    return th.enabled() and bool(envs.SGLANG_WEG2_TAIL_ADOPT.get())


def verify_enabled() -> bool:
    return bool(envs.SGLANG_WEG2_TAIL_VERIFY.get())


# -- what this rank holds ------------------------------------------------------------
class RowSpec(msgspec.Struct, frozen=True):
    shape: List[int]
    dtype: str


class HeldShapes(msgspec.Struct, frozen=True):
    """Per global layer id the rows this rank must receive, in part order:
    fa -> (K, V[, QSA compressed]), gdn -> (temporal, conv...). A layer whose
    row is empty is not held (Form-A worker: attention K [0, 256])."""

    fa: Dict[int, List[RowSpec]]
    gdn: Dict[int, List[RowSpec]]
    qsa_ratio: int  # 0 = no QSA compressed rows
    #: uneven DCP compacts this rank's KV rows (slot ids are not rows); the
    #: tail rows are written by global slot and are refused under it
    dcp: bool = False

    @property
    def holds_nothing(self) -> bool:
        return not self.fa and not self.gdn


def _row(t: torch.Tensor) -> RowSpec:
    return RowSpec(shape=[int(x) for x in t.shape[1:]], dtype=str(t.dtype))


def held_shapes(kvpool, req_to_token_pool) -> HeldShapes:
    from sglang.srt.distributed.utils import uneven_dcp_active
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    fa: Dict[int, List[RowSpec]] = {}
    ratio = 0
    if isinstance(kvpool, HybridLinearKVPool):
        full = kvpool.full_kv_pool
        qsa = isinstance(kvpool, QSATokenToKVPool)
        ratio = int(kvpool.qsa_compress_ratio) if qsa else 0
        for gid, local in sorted(kvpool.full_attention_layer_id_mapping.items()):
            k = full.k_buffer[local]
            if math.prod(k.shape[1:]) == 0:
                continue
            rows = [_row(k), _row(full.v_buffer[local])]
            if qsa:
                rows.append(_row(kvpool.qsa_compressed_k_buffer_pool[local]))
            fa[int(gid)] = rows
    gdn: Dict[int, List[RowSpec]] = {}
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        cache = req_to_token_pool.mamba_pool.mamba_cache
        for gid, local in sorted(req_to_token_pool.mamba_map.items()):
            t = cache.temporal[local]
            if math.prod(t.shape[1:]) == 0:
                continue
            gdn[int(gid)] = [_row(t)] + [_row(c[local]) for c in cache.conv]
    return HeldShapes(fa=fa, gdn=gdn, qsa_ratio=ratio, dcp=bool(fa) and bool(uneven_dcp_active()))


# -- 1. VOTE ---------------------------------------------------------------------------
class Staged(msgspec.Struct):
    """One rank's reading of a rid's parts: the verdict and, on a holding
    rank, the merged payload of the layers it holds (pinned when CUDA)."""

    spec: th.TailSpec
    headers: List[th.TailHeader]
    verdict: str  # "ready" | "not_mine" | refusal reason
    qsa_ratio: int = 0
    fa: Dict[int, Tuple[torch.Tensor, ...]] = {}
    gdn: Dict[int, Tuple[torch.Tensor, ...]] = {}
    #: host buffers the post-write readback lands in, keyed "fa<gid>" /
    #: "gdn<gid>" (allocated here, off the forward's launch path)
    readback: Dict[str, Tuple[torch.Tensor, ...]] = {}

    @property
    def ok(self) -> bool:
        return self.verdict in READY_VERDICTS


def _check_rows(kind: str, gid: int, tensors, want: List[RowSpec], lead: List[int]) -> str:
    if len(tensors) != len(want):
        return f"{kind}_arity:{gid}:{len(tensors)}!={len(want)}"
    for t, w, n in zip(tensors, want, lead):
        if list(t.shape) != [n] + list(w.shape):
            return f"{kind}_rows:{gid}:{list(t.shape)}!={[n] + list(w.shape)}"
        if str(t.dtype) != w.dtype:
            return f"{kind}_dtype:{gid}:{t.dtype}!={w.dtype}"
    return ""


def stage_parts(headers: List[th.TailHeader], held: HeldShapes, check_digest: bool) -> Staged:
    """Pure-ish (reads the part files): the local verdict and payload."""
    spec = headers[0].spec
    st = Staged(spec=spec, headers=list(headers), verdict="ready", qsa_ratio=held.qsa_ratio)
    for h in headers:
        if h.spec != spec:
            st.verdict = f"spec_differs:{h.part}"
            return st
    if held.holds_nothing:
        st.verdict = "not_mine"
        return st
    need_fa = {g: r[0].shape for g, r in held.fa.items()}
    need_gdn = {g: r[0].shape for g, r in held.gdn.items()}
    why = "dcp_active" if held.dcp else th.local_readiness(spec, headers, need_fa, need_gdn)
    if not why and held.fa and (not held.gdn or min(held.gdn) > min(held.fa)):
        # the install rides the GDN layer reads; an attention layer read
        # before the first of them would see unwritten rows
        why = "fa_before_gdn"
    if why:
        st.verdict = why
        return st
    fa: Dict[int, Tuple[torch.Tensor, ...]] = {}
    gdn: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for h in headers:
        bundle, why = th.read_part(h, check_digest=check_digest)
        if bundle is None:
            st.verdict = f"{why}:{h.part}"
            return st
        fa.update({int(g): tuple(t) for g, t in bundle["fa"].items() if int(g) in held.fa})
        gdn.update({int(g): tuple(t) for g, t in bundle["gdn"].items() if int(g) in held.gdn})
    rows = spec.rows
    for g, want in held.fa.items():
        lead = [rows, rows] + ([rows // held.qsa_ratio] if held.qsa_ratio else [])
        why = _check_rows("fa", g, fa[g], want, lead)
        if why:
            st.verdict = why
            return st
    for g, want in held.gdn.items():
        why = _check_rows("gdn", g, gdn[g], want, [1] * len(want))
        if why:
            st.verdict = why
            return st
    pin = torch.cuda.is_available()
    if pin:
        fa = {g: tuple(t.pin_memory() for t in ts) for g, ts in fa.items()}
        gdn = {g: tuple(t.pin_memory() for t in ts) for g, ts in gdn.items()}
    st.fa, st.gdn = fa, gdn
    if check_digest:
        rb = {f"fa{g}": ts for g, ts in fa.items()}
        rb.update({f"gdn{g}": ts for g, ts in gdn.items()})
        st.readback = {
            k: tuple(torch.empty(_as_bytes(t).shape, dtype=torch.uint8, pin_memory=pin) for t in ts)
            for k, ts in rb.items()
        }
    return st


class _Job(msgspec.Struct):
    thread: threading.Thread
    box: List[Staged]


class Agreed(msgspec.Struct):
    """The group's outcome for one rid, identical on every rank in ``agreed``;
    ``staged`` is this rank's own reading (payload only on a holding rank)."""

    staged: Staged
    agreed: bool


_JOBS: Dict[str, _Job] = {}
_AGREED: Dict[str, Agreed] = {}


def _candidate(rid: str) -> bool:
    from sglang.srt.managers.schedule_policy import _WEG2_END_ANCHOR

    # group P (END_ANCHOR) publishes; only D adopts
    return adopt_enabled() and rid.startswith("weg2-") and bool(th._dir()) and not _WEG2_END_ANCHOR


def _stage_into(box: List[Staged], headers, held: HeldShapes, check_digest: bool, device: Optional[int]) -> None:
    try:
        # pin on the rank's own card context, never a fresh one on device 0
        ctx = torch.cuda.device(device) if device is not None else th._null_ctx()
        with ctx:
            box.append(stage_parts(headers, held, check_digest))
    except Exception as exc:  # noqa: BLE001 -- a failed staging is a 0 vote, named
        logger.warning("WEG2-TAIL stage failed rid=%s (%s: %s)", headers[0].spec.rid, type(exc).__name__, exc)
        box.append(Staged(spec=headers[0].spec, headers=list(headers), verdict=f"stage_raised:{type(exc).__name__}"))


def stage(rid: str, tree_cache) -> None:
    """Prefetch-progress entry (every round): start reading the parts of a
    D request once, in the background. Never raises into the tree."""
    try:
        if rid in _JOBS or not _candidate(rid):
            return
        headers = th.headers_for(rid)
        if not headers:
            return
        held = held_shapes(tree_cache.token_to_kv_pool_allocator.get_kvcache(), tree_cache.req_to_token_pool)
        box: List[Staged] = []
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        t = threading.Thread(target=_stage_into, args=(box, headers, held, verify_enabled(), device),
                             daemon=True, name="weg2-tail-stage")
        _JOBS[rid] = _Job(thread=t, box=box)
        while len(_JOBS) > KEEP_AGREED:  # a prefetch that never completed
            _JOBS.pop(next(iter(_JOBS)))
        t.start()
    except Exception as exc:  # noqa: BLE001 -- no staging = a 0 vote
        logger.warning("WEG2-TAIL stage refused rid=%s (%s: %s)", rid, type(exc).__name__, exc)


def local_vote(rid: str) -> int:
    """This rank's slot in the group MIN: 1 = can serve (or holds nothing)."""
    job = _JOBS.get(rid)
    if job is None:
        return 0
    job.thread.join(STAGE_JOIN_S)
    return 1 if job.box and job.box[0].ok else 0


def agree(rid: str, group_vote: int) -> None:
    """After the MIN: remember the group's answer for the admission."""
    job = _JOBS.pop(rid, None)
    if job is None:
        return
    st = job.box[0] if job.box else None
    if st is None:
        return  # still reading after STAGE_JOIN_S: voted 0, nothing to remember
    agreed = bool(group_vote) and st.ok
    if not agreed:
        st.fa, st.gdn, st.readback = {}, {}, {}  # never applied: drop the payload now
    _AGREED[rid] = Agreed(staged=st, agreed=agreed)
    while len(_AGREED) > KEEP_AGREED:
        _AGREED.pop(next(iter(_AGREED)))


# -- 2. ADOPT --------------------------------------------------------------------------
def uniform_refusal(spec: th.TailSpec, ids, fill_len: int, extra_key, prefix_len: int) -> str:
    """'' when the admission may take the tail; otherwise why not. Every
    input is identical on every rank of the group (the extend length the
    group runs is decided HERE)."""
    if len(ids) != spec.n_tokens or fill_len != spec.n_tokens:
        return f"n_tokens:{fill_len}!={spec.n_tokens}"
    if int(prefix_len) != spec.page_prefix:
        return f"prefix:{int(prefix_len)}!={spec.page_prefix}"
    if not (0 < spec.rows and spec.extend >= 1):
        return "geometry"
    if th.tail_key(ids, spec.cut, extra_key) != spec.key:
        return "key_mismatch"
    return ""


def _log_ready(st: Staged, adopt: str) -> None:
    spec = st.spec
    logger.info(
        "WEG2-TAIL-READY rid=%s page_prefix=%d tail_rows=%d state_at=%d extend=%d parts=%d key=%s "
        "verdict=%s adopt=%s",
        spec.rid, spec.page_prefix, spec.rows, spec.cut, spec.extend, len(st.headers), spec.key,
        st.verdict, adopt,
    )


def plan_adopt(req, prefix_len: int) -> Optional[Agreed]:
    """Admission, before the commit: the agreed outcome for ``req`` if the
    group takes the tail now, else None (today's extend). One-shot per rid."""
    if not adopt_enabled():
        return None
    entry = _AGREED.pop(str(req.rid), None)
    if entry is None:
        return None
    if not entry.agreed:
        _log_ready(entry.staged, "skipped:group_vote")
        return None
    why = uniform_refusal(entry.staged.spec, req.origin_input_ids, len(req.full_untruncated_fill_ids),
                          req.extra_key, prefix_len)
    if why:
        _log_ready(entry.staged, f"skipped:{why}")
        return None
    return entry


def commit_adopt(req, entry: Agreed, tree_cache, page_size: int) -> int:
    """Admission commit (the request is going into this batch): one page,
    the prefix grows by its first ``rows`` slots, the holding rank queues the
    install. Returns the new prefix length c."""
    from sglang.srt.mem_cache.common import alloc_token_slots

    st = entry.staged
    spec = st.spec
    page = alloc_token_slots(tree_cache, int(page_size))
    rows = page[: spec.rows].to(dtype=req.prefix_indices.dtype, device=req.prefix_indices.device)
    req.prefix_indices = torch.cat([req.prefix_indices, rows])
    _log_ready(st, "done")
    if st.verdict == "not_mine":
        _log_adopt(spec, fa_rows=0, fa_layers=0, gdn_layers=0, digest="not_mine", ms=0.0, issue_ms=0.0)
        return spec.cut
    _queue_install(req, st, rows, tree_cache)
    return spec.cut


# -- 3. INSTALL ------------------------------------------------------------------------
class Install(msgspec.Struct):
    spec: th.TailSpec
    headers: List[th.TailHeader]
    fa: Dict[int, Tuple[torch.Tensor, ...]]   # gid -> host rows
    gdn: Dict[int, Tuple[torch.Tensor, ...]]  # gid -> host state
    fa_dst: Dict[int, Tuple[torch.Tensor, ...]]   # gid -> (k, v[, c]) device buffers
    gdn_dst: Dict[int, Tuple[torch.Tensor, ...]]  # gid -> (temporal, conv...) device buffers
    rows: torch.Tensor
    groups: Optional[torch.Tensor]
    slot: Optional[torch.Tensor]
    t0: float
    fa_done: bool = False
    last_gid: int = -1
    issue_ms: float = 0.0
    readback: Dict[str, Tuple[torch.Tensor, ...]] = {}


#: installs waiting for the extend forward's GDN layer reads (memory_pool
#: HybridReqToTokenPool.mamba2_layer_cache checks this list per call).
PENDING_INSTALLS: List[Install] = []


def _queue_install(req, st: Staged, rows: torch.Tensor, tree_cache) -> None:
    kvpool = tree_cache.token_to_kv_pool_allocator.get_kvcache()
    rtp = tree_cache.req_to_token_pool
    fa_dst: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for gid in st.fa:
        local = kvpool.full_attention_layer_id_mapping[gid]
        full = kvpool.full_kv_pool
        dst = (full.k_buffer[local], full.v_buffer[local])
        if len(st.fa[gid]) == 3:
            dst = dst + (kvpool.qsa_compressed_k_buffer_pool[local],)
        fa_dst[gid] = dst
    gdn_dst: Dict[int, Tuple[torch.Tensor, ...]] = {}
    slot = None
    if st.gdn:
        if req.mamba_pool_idx is None:
            # the group already runs [c, N); without a slot there is no state
            # to overwrite -- the same wrongness as any prefix without a slot
            logger.error("WEG2-TAIL-ADOPT rid=%s FAILED: no mamba slot on a GDN-holding rank", st.spec.rid)
            return
        slot = rtp.translate_mamba_indices(req.mamba_pool_idx.reshape(1).to(torch.int64))
        cache = rtp.mamba_pool.mamba_cache
        for gid in st.gdn:
            local = rtp.mamba_map[gid]
            gdn_dst[gid] = (cache.temporal[local],) + tuple(c[local] for c in cache.conv)
    ratio = st.qsa_ratio
    groups = (rows[::ratio] // ratio) if (ratio and st.fa) else None
    PENDING_INSTALLS.append(Install(
        spec=st.spec, headers=st.headers, fa=st.fa, gdn=st.gdn, fa_dst=fa_dst, gdn_dst=gdn_dst,
        rows=rows, groups=groups, slot=slot, t0=time.perf_counter(), readback=st.readback,
    ))


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _put(dst: torch.Tensor, idx: torch.Tensor, src: torch.Tensor, back: Optional[torch.Tensor]) -> None:
    """dst[idx] = src bytewise (fp8 has no index_copy_ on every build), the
    H2D on the current stream; optionally read the written rows back into
    the preallocated host buffer ``back``."""
    d = _as_bytes(dst)
    i = idx.to(device=d.device, dtype=torch.int64)
    d.index_copy_(0, i, _as_bytes(src).to(d.device, non_blocking=True))
    if back is not None:
        back.copy_(d.index_select(0, i), non_blocking=d.is_cuda)


def install_layer(layer_id: int) -> None:
    """GDN layer read of a forward (after its load-stream join): write what
    the pending installs owe up to this layer. Never raises into a forward
    except through a device error."""
    for inst in list(PENDING_INSTALLS):
        _install_step(inst, int(layer_id))


def _install_step(inst: Install, layer_id: int) -> None:
    if layer_id < inst.last_gid:
        # a second forward began before every planned GDN layer was read
        # (a repeated read of the same layer is not a new forward)
        PENDING_INSTALLS.remove(inst)
        logger.error(
            "WEG2-TAIL-ADOPT rid=%s INCOMPLETE: GDN layers %s were never read by the extend forward",
            inst.spec.rid, sorted(inst.gdn),
        )
        return
    t = time.perf_counter()
    inst.last_gid = layer_id
    verify = bool(inst.readback)
    if not inst.fa_done:
        for gid in sorted(inst.fa):
            back = inst.readback.get(f"fa{gid}")
            for j, (dst, src) in enumerate(zip(inst.fa_dst[gid], inst.fa[gid])):
                _put(dst, inst.groups if j == 2 else inst.rows, src, None if back is None else back[j])
        inst.fa_done = True
    if layer_id in inst.gdn:
        src = inst.gdn.pop(layer_id)
        back = inst.readback.get(f"gdn{layer_id}")
        for j, (dst, s) in enumerate(zip(inst.gdn_dst[layer_id], src)):
            _put(dst, inst.slot, s, None if back is None else back[j])
    inst.issue_ms += (time.perf_counter() - t) * 1000.0
    if not inst.gdn:
        PENDING_INSTALLS.remove(inst)
        _finish(inst, verify)


def _finish(inst: Install, verify: bool) -> None:
    event = None
    if verify and torch.cuda.is_available() and inst.rows.is_cuda:
        event = torch.cuda.Event()
        event.record()
    threading.Thread(target=_verify_and_log, args=(inst, verify, event), daemon=True,
                     name="weg2-tail-verify").start()


def readback_digest(inst: Install) -> str:
    """match | MISMATCH:<part> | partial: the written rows/state, read back
    from the device, against each part's publish digests (same byte order as
    tail_handoff.digest over the part's sorted layers)."""
    held_fa = {int(k[2:]) for k in inst.readback if k.startswith("fa")}
    held_gdn = {int(k[3:]) for k in inst.readback if k.startswith("gdn")}
    for h in inst.headers:
        if not set(h.fa_layers) <= held_fa or not set(h.gdn_layers) <= held_gdn:
            return "partial"
        fa = [t for g in sorted(h.fa_layers) for t in inst.readback[f"fa{g}"]]
        gdn = [t for g in sorted(h.gdn_layers) for t in inst.readback[f"gdn{g}"]]
        if th.digest(fa) != h.fa_digest or th.digest(gdn) != h.gdn_digest:
            return f"MISMATCH:{h.part}"
    return "match"


def _verify_and_log(inst: Install, verify: bool, event) -> None:
    try:
        if event is not None:
            event.synchronize()
        dig = readback_digest(inst) if verify else "off"
    except Exception as exc:  # noqa: BLE001 -- an instrument, named
        dig = f"verify_raised:{type(exc).__name__}"
    _log_adopt(inst.spec, fa_rows=inst.spec.rows, fa_layers=len(inst.fa_dst), gdn_layers=len(inst.gdn_dst),
               digest=dig, ms=(time.perf_counter() - inst.t0) * 1000.0, issue_ms=inst.issue_ms)


def _log_adopt(spec: th.TailSpec, fa_rows: int, fa_layers: int, gdn_layers: int, digest: str, ms: float,
               issue_ms: float) -> None:
    level = logging.ERROR if digest.startswith("MISMATCH") else logging.INFO
    logger.log(
        level,
        "WEG2-TAIL-ADOPT rid=%s page_prefix=%d tail_rows=%d state_at=%d extend=%d fa_rows_written=%d "
        "fa_layers=%d gdn_layers=%d digest=%s ms=%.1f issue_ms=%.1f",
        spec.rid, spec.page_prefix, spec.rows, spec.cut, spec.extend, fa_rows, fa_layers, gdn_layers,
        digest, ms, issue_ms,
    )
