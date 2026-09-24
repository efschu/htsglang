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

4. SKIP (E2, H24, ``SGLANG_WEG2_TAIL_SKIP_EXTEND``). When P's parts also
   carry the END section (rows [floor_page(c), N), the open QSA group's
   pending-ring rows + RoPE positions, the GDN state after N, P's sampled
   token), the vote slot carries a LEVEL: 2 = this rank can serve the END
   state too, 1 = only E1, 0 = neither; the group's MIN is the one answer.
   On 2 (and rank-uniform admission checks: the batch is still empty, no
   logprob/hidden/grammar/penalty request, key over ids[0:N]) the batch has
   EXACTLY E1's shape -- prefix c inside ONE page, extend [c, N) -- but
   EAGLEWorkerV2 runs NO target forward for it (``skip_tokens`` /
   ``run_skip``). On the forward stream it joins the batch's HiCache load
   (last layer event of its consumer), writes the END rows at the request's
   slots [floor_page(c), N) (read from req_to_token, so the extend's own
   slots [c, N) included), the complete groups' compressed rows, the ring
   rows at ``req_pool_idx * r + j``, and the GDN slot. The result is P's
   token as ``next_token_ids`` with ZERO target hidden states for [c, N),
   and the extend's DRAFT phase then runs exactly as after a real target
   forward (H24b): the solo host runs ``_draft_extend_for_prefill`` over
   [c, N) (draft KV + draft-pool QSA rows of those positions, a real
   draft seed), the shadows take their stub. The request then merges into
   the running batch like any finished prefill. A batch holding a skip
   request holds nothing else (PrefillAdder refuses the next request, like
   the born-spilled-deep batch).

5. WAIT (H45, ``SGLANG_WEG2_TAIL_WAIT_MS``). P's PP ranks write their parts
   from background publish threads, and D's first prefetch check can come
   before the last of them landed (metal fnFL2x150/x151: TP0 staged parts=1-2
   of 3, 'fa_layer_missing' -> vote 0 -> a real extend, flip 3.6-3.9 s; the
   one flip whose manifest was complete adopted in 1.95 s). The header names
   P's part count (``TailHeader.n_parts``), so ``stage`` starts reading only a
   COMPLETE manifest and re-reads the part list at every check until then;
   ``vote_hold`` is this rank's "not staged yet" (bounded by the env, 300 ms
   while no part exists at all), MAX-reduced with the prefetch termination
   verdict, so the group holds the termination -- and with it the vote --
   uniformly. The READY line carries ``waited_ms`` (how long the hold kept a
   finished read from terminating) and ``token_src`` (P's sampled token, read
   from the END headers of the parts -- the one source on every rank).

QSA at an OPEN group (N % r != 0): decode compresses the group its length
completes, from the per-request pending ring (qwen_sparse_attn_backend
``_qsa_build_write_plan``: members come from the ring). P's final chunk wrote
the members [floor_r(N), N) into ITS ring rows (P's req_pool_idx); they move
to D's ring rows, so D's decode completes the group bit-identically.

Any refusal, on any rank, before the commit: today's path (extend from
floor_page(c)), never an abort. A refused SKIP (level < 2 or an admission
refusal) is the E1 path, byte for byte. After the commit the extend length is fixed
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
#: H45: the hold while NO part of the rid exists (P may publish none: no
#: partial page below the cut, END refused before the write) -- short, the
#: partial-manifest bound is the env's.
NO_PARTS_WAIT_S = 0.3


def tail_wait_s() -> float:
    """SGLANG_WEG2_TAIL_WAIT_MS as seconds (H45); 0 = no hold."""
    return max(0.0, float(envs.SGLANG_WEG2_TAIL_WAIT_MS.get() or 0) / 1000.0)


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
    #: E2: per held attention layer the QSA pending-ring row (index-K state)
    ring: Dict[int, RowSpec] = {}
    #: E2: the layer-independent RoPE position row of the ring
    rope: Optional[RowSpec] = None

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
    ring: Dict[int, RowSpec] = {}
    rope: Optional[RowSpec] = None
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
                ring[int(gid)] = _row(kvpool.qsa_key_state_buffer_pool[local])
                rope = _row(kvpool.qsa_rope_position_buffer)
            fa[int(gid)] = rows
    gdn: Dict[int, List[RowSpec]] = {}
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        cache = req_to_token_pool.mamba_pool.mamba_cache
        for gid, local in sorted(req_to_token_pool.mamba_map.items()):
            t = cache.temporal[local]
            if math.prod(t.shape[1:]) == 0:
                continue
            gdn[int(gid)] = [_row(t)] + [_row(c[local]) for c in cache.conv]
    return HeldShapes(fa=fa, gdn=gdn, qsa_ratio=ratio, dcp=bool(fa) and bool(uneven_dcp_active()),
                      ring=ring, rope=rope)


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
    #: E2 (H24): "ready" | "not_mine" | "absent" | refusal reason, the END
    #: payload of the held layers and P's sampled token (every rank reads it)
    end_verdict: str = "absent"
    first_token: int = -1
    end_fa: Dict[int, Tuple[torch.Tensor, ...]] = {}
    end_gdn: Dict[int, Tuple[torch.Tensor, ...]] = {}
    end_ring: Dict[int, Tuple[torch.Tensor, ...]] = {}
    end_rope: Optional[torch.Tensor] = None
    #: keyed "fa<gid>" / "gdn<gid>" / "ring<gid>" / "rope"
    end_readback: Dict[str, Tuple[torch.Tensor, ...]] = {}
    #: H45: the part count P's manifest names (0 = unknown / pre-H45) and
    #: where ``first_token`` came from ("publish" = the parts' END headers,
    #: else "none:<why>")
    n_parts: int = 0
    token_src: str = "none"

    @property
    def ok(self) -> bool:
        return self.verdict in READY_VERDICTS

    @property
    def end_ok(self) -> bool:
        return self.ok and self.end_verdict in READY_VERDICTS

    def drop_end(self) -> None:
        self.end_fa, self.end_gdn, self.end_ring, self.end_rope, self.end_readback = {}, {}, {}, None, {}


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
    """Pure-ish (reads the part files): the local verdict and payload, E1
    and -- when the parts carry it and SKIP is on -- the END section (E2)."""
    bundles: List[dict] = []
    st = _stage_e1(headers, held, check_digest, bundles)
    st.n_parts = max((int(h.n_parts) for h in headers), default=0)
    st.first_token, st.token_src = end_token(headers)
    if st.ok and th.skip_extend_enabled():
        st.end_verdict = _stage_end(st, bundles, held, check_digest)
        if st.end_verdict not in READY_VERDICTS:
            st.drop_end()
    return st


def _stage_e1(headers: List[th.TailHeader], held: HeldShapes, check_digest: bool, bundles: List[dict]) -> Staged:
    """E1 (H21): rows [floor_page(c), c) + state at c; ``bundles`` receives
    the part payloads read (holding rank only)."""
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
        bundles.append(bundle)
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
        st.readback = _readback_buffers(fa, gdn, {}, None, pin)
    return st


def end_token(headers: List[th.TailHeader]) -> Tuple[int, str]:
    """(P's sampled token, source) from the parts' END headers -- the same
    source on every rank, whatever this rank's own E1 verdict (H45: TP0 used
    to print first_token=-1 whenever its E1 staging refused, because only
    ``_stage_end`` read it). (-1, 'none:<why>') when the headers do not name
    one token."""
    why = _end_header_refusal(headers) if headers else "no_parts"
    if why:
        return -1, f"none:{why}"
    return int(headers[0].end.first_token), "publish"


def _end_header_refusal(headers: List[th.TailHeader]) -> str:
    """The END sections of all parts name ONE state: same token, key, rows."""
    ends = [h.end for h in headers]
    if any(e is None for e in ends):
        return "end_missing"
    first = ends[0]
    shape = (first.first_token, first.key, first.rows, first.groups, first.ring_rows)
    for h, e in zip(headers, ends):
        if (e.first_token, e.key, e.rows, e.groups, e.ring_rows) != shape:
            return f"end_differs:{h.part}"
    if first.first_token < 0:
        return "no_first_token"
    return ""


def _stage_end(st: Staged, bundles: List[dict], held: HeldShapes, check_digest: bool) -> str:
    """E2 staging: the verdict; on a holding rank also the END payload of
    the layers it holds (pinned) and its readback buffers, set on ``st``."""
    headers = st.headers
    why = _end_header_refusal(headers)
    if why:
        return why
    end = headers[0].end
    st.first_token = int(end.first_token)
    if held.holds_nothing:
        return "not_mine"
    if (end.rows, end.groups, end.ring_rows) != th.end_geometry(st.spec, held.qsa_ratio):
        return f"end_geometry:{end.rows}/{end.groups}/{end.ring_rows}"
    fa: Dict[int, Tuple[torch.Tensor, ...]] = {}
    gdn: Dict[int, Tuple[torch.Tensor, ...]] = {}
    ring: Dict[int, Tuple[torch.Tensor, ...]] = {}
    ropes: List[torch.Tensor] = []
    for h, bundle in zip(headers, bundles):
        why = th.end_digest_refusal(h, bundle) if check_digest else ("" if "end" in bundle else "end_missing")
        if why:
            return f"{why}:{h.part}"
        sec = bundle["end"]
        fa.update({int(g): tuple(t) for g, t in sec["fa"].items() if int(g) in held.fa})
        gdn.update({int(g): tuple(t) for g, t in sec["gdn"].items() if int(g) in held.gdn})
        ring.update({int(g): tuple(t) for g, t in sec["ring"].items() if int(g) in held.fa})
        if sec["rope"] is not None:
            ropes.append(sec["rope"])
    why = _end_shape_refusal(end, held, fa, gdn, ring, ropes)
    if why:
        return why
    pin = torch.cuda.is_available()
    st.end_fa = {g: tuple(_pin(t, pin) for t in ts) for g, ts in fa.items()}
    st.end_gdn = {g: tuple(_pin(t, pin) for t in ts) for g, ts in gdn.items()}
    st.end_ring = {g: tuple(_pin(t, pin) for t in ts) for g, ts in ring.items()}
    st.end_rope = _pin(ropes[0], pin) if ropes and held.ring else None
    if check_digest:
        st.end_readback = _readback_buffers(st.end_fa, st.end_gdn, st.end_ring, st.end_rope, pin)
    return "ready"


def _end_shape_refusal(end: th.EndHeader, held: HeldShapes, fa, gdn, ring, ropes: List[torch.Tensor]) -> str:
    rows, groups, ring_rows = end.rows, end.groups, end.ring_rows
    for g, want in held.fa.items():
        if g not in fa:
            return f"end_fa_layer_missing:{g}"
        lead = [rows, rows] + ([groups] if held.qsa_ratio else [])
        why = _check_rows("end_fa", g, fa[g], want, lead)
        if why:
            return why
        if held.ring:
            if g not in ring:
                return f"end_ring_layer_missing:{g}"
            why = _check_rows("end_ring", g, ring[g], [held.ring[g]], [ring_rows])
            if why:
                return why
    for g, want in held.gdn.items():
        if g not in gdn:
            return f"end_gdn_layer_missing:{g}"
        why = _check_rows("end_gdn", g, gdn[g], want, [1] * len(want))
        if why:
            return why
    if held.ring and held.fa:
        if not ropes:
            return "end_rope_missing"
        why = _check_rows("end_rope", -1, (ropes[0],), [held.rope], [ring_rows])
        if why:
            return why
        if any(not torch.equal(r, ropes[0]) for r in ropes[1:]):
            return "end_rope_differs"
    return ""


def _pin(t: torch.Tensor, pin: bool) -> torch.Tensor:
    return t.pin_memory() if pin else t


def _readback_buffers(fa, gdn, ring, rope, pin: bool) -> Dict[str, Tuple[torch.Tensor, ...]]:
    src = {f"fa{g}": ts for g, ts in fa.items()}
    src.update({f"gdn{g}": ts for g, ts in gdn.items()})
    src.update({f"ring{g}": ts for g, ts in ring.items()})
    if rope is not None:
        src["rope"] = (rope,)
    return {
        k: tuple(torch.empty(_as_bytes(t).shape, dtype=torch.uint8, pin_memory=pin) for t in ts)
        for k, ts in src.items()
    }


class _Job(msgspec.Struct):
    """One rid's staging on this rank. H45: made at the FIRST progress check
    (``thread`` None while the manifest is incomplete; the part list is
    re-read at every check), started once the manifest is complete."""

    box: List[Staged]
    thread: Optional[threading.Thread] = None
    #: perf_counter of the first progress check of the rid
    t_first: float = 0.0
    #: the manifest as last read: state / parts seen / parts named
    state: str = "none"
    have: int = 0
    want: int = 0
    headers: List[th.TailHeader] = []
    #: perf_counter at which the hold first kept a finished read from
    #: terminating (-1 = never held)
    held_since: float = -1.0

    @property
    def staged(self) -> bool:
        return bool(self.box)


class Agreed(msgspec.Struct):
    """The group's outcome for one rid, identical on every rank in ``agreed``;
    ``staged`` is this rank's own reading (payload only on a holding rank)."""

    staged: Staged
    agreed: bool
    #: E2: the group voted 2 (every rank can serve the END state); cleared
    #: by a uniform admission refusal (then E1)
    skip: bool = False
    skip_note: str = ""
    #: H45: how long the hold kept this rid's finished read from terminating
    waited_ms: float = 0.0

    @property
    def resume_at(self) -> int:
        """Where the admission's extend starts: c, for E1 and SKIP alike.
        H24b: SKIP keeps E1's batch shape [c, N) -- the draft extend that
        follows needs a group-aligned prefix (QSA ``prefix_lens % r == 0``)
        and is the proven order before the first decode draft."""
        return self.staged.spec.cut


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


#: manifest states ``stage`` starts reading at (a 'legacy' manifest has no
#: count: taken as it is, the H21 form)
_STAGE_STATES = ("complete", "legacy")


def stage(rid: str, tree_cache) -> None:
    """Prefetch-progress entry (every round): start reading the parts of a
    D request once its manifest is complete (H45), in the background. Never
    raises into the tree."""
    try:
        job = _JOBS.get(rid)
        if job is not None and (job.thread is not None or job.staged):
            return
        if job is None:
            if not _candidate(rid):
                return
            job = _Job(box=[], t_first=time.perf_counter())
            _JOBS[rid] = job
            while len(_JOBS) > KEEP_AGREED:  # a prefetch that never completed
                _JOBS.pop(next(iter(_JOBS)))
        headers = th.headers_for(rid)
        job.state, job.have, job.want = th.manifest_state(headers)
        job.headers = list(headers)
        if job.state in ("none", "partial"):
            return  # P's publish threads are still writing: re-read next check
        if job.state not in _STAGE_STATES:
            # a count that can never complete (stale / disagreeing parts): a
            # named 0 vote now, no hold
            job.box.append(Staged(spec=headers[0].spec, headers=list(headers), n_parts=job.want,
                                  verdict=f"parts_{job.state}:{job.have}/{job.want}"))
            return
        held = held_shapes(tree_cache.token_to_kv_pool_allocator.get_kvcache(), tree_cache.req_to_token_pool)
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        job.thread = threading.Thread(target=_stage_into, args=(job.box, headers, held, verify_enabled(), device),
                                      daemon=True, name="weg2-tail-stage")
        job.thread.start()
    except Exception as exc:  # noqa: BLE001 -- no staging = a 0 vote
        logger.warning("WEG2-TAIL stage refused rid=%s (%s: %s)", rid, type(exc).__name__, exc)


def vote_hold(rid: str) -> int:
    """H45: this rank's slot in the prefetch termination MAX -- 1 while its
    staging of the rid is not finished (manifest incomplete or the read still
    running) and inside the bound (``SGLANG_WEG2_TAIL_WAIT_MS`` from the first
    check; ``NO_PARTS_WAIT_S`` while no part exists at all), else 0. The
    group's MAX makes the hold uniform; every input here is rank-local."""
    job = _JOBS.get(rid)
    if job is None or job.staged:
        return 0
    wait = tail_wait_s()
    if wait <= 0.0:
        return 0
    if job.thread is None and job.have == 0:
        wait = min(wait, NO_PARTS_WAIT_S)
    return 1 if time.perf_counter() - job.t_first < wait else 0


def note_held(rid: str) -> None:
    """H45: the group's read of ``rid`` could terminate, the tail hold kept
    it (``can_terminate_prefetch``) -- stamp the start of the wait."""
    job = _JOBS.get(rid)
    if job is not None and job.held_since < 0:
        job.held_since = time.perf_counter()


#: E2 needs a worker that honours a skip batch (EAGLEWorkerV2's extend
#: branch). A process whose model worker would run the extend forward anyway
#: -- over a GDN state already at N -- never votes 2 (it adopts E1 instead).
_SKIP_SERVER = [False]


def register_skip_server() -> None:
    """Called by a model worker that routes ``skip_tokens`` batches to
    ``run_skip`` instead of the target forward."""
    _SKIP_SERVER[0] = True


def local_vote(rid: str) -> int:
    """This rank's slot in the group MIN: 2 = can serve the END state (E2),
    1 = can serve E1 (or holds nothing), 0 = neither."""
    job = _JOBS.get(rid)
    if job is None:
        return 0
    if job.thread is not None and not job.box:
        job.thread.join(STAGE_JOIN_S)
    if not job.box or not job.box[0].ok:
        return 0
    return 2 if job.box[0].end_ok and _SKIP_SERVER[0] else 1


def agree(rid: str, group_vote: int) -> None:
    """After the MIN: remember the group's answer for the admission."""
    job = _JOBS.pop(rid, None)
    if job is None:
        return
    waited_ms = (time.perf_counter() - job.held_since) * 1000.0 if job.held_since >= 0 else 0.0
    st = job.box[0] if job.box else None
    if st is None:
        # voted 0: the manifest never completed inside the bound, or the read
        # was still running after STAGE_JOIN_S -- named at the admission
        why = "stage_unfinished" if job.thread is not None else f"parts_{job.state}:{job.have}/{job.want}"
        if not job.headers:
            logger.info(
                "WEG2-TAIL-READY rid=%s parts=0/? verdict=no_parts adopt=skipped:no_parts end=absent "
                "first_token=-1 token_src=none:no_parts waited_ms=%.0f",
                rid, waited_ms,
            )
            return
        first, src = end_token(job.headers)
        st = Staged(spec=job.headers[0].spec, headers=list(job.headers), verdict=why, n_parts=job.want,
                    first_token=first, token_src=src)
    agreed = group_vote >= 1 and st.ok
    skip = group_vote >= 2 and st.end_ok
    if not skip:
        st.drop_end()
    if not agreed:
        st.fa, st.gdn, st.readback = {}, {}, {}  # never applied: drop the payload now
    _AGREED[rid] = Agreed(staged=st, agreed=agreed, skip=skip,
                          skip_note="" if skip else f"level{int(group_vote)}:{st.end_verdict}",
                          waited_ms=waited_ms)
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


def _log_ready(st: Staged, adopt: str, skip: bool = False, skip_note: str = "", waited_ms: float = 0.0) -> None:
    """tail_rows/state_at/extend name the geometry the group RUNS: E1's
    [page_prefix, c) + extend [c, N) unless the END state is taken (skip).
    ``parts`` is seen/named (``?`` = a pre-H45 manifest without a count)."""
    spec = st.spec
    if skip:
        rows, state_at, extend = spec.n_tokens - spec.page_prefix, spec.n_tokens, 0
    else:
        rows, state_at, extend = spec.rows, spec.cut, spec.extend
    logger.info(
        "WEG2-TAIL-READY rid=%s page_prefix=%d tail_rows=%d state_at=%d extend=%d parts=%d/%s key=%s "
        "verdict=%s adopt=%s end=%s first_token=%d token_src=%s waited_ms=%.0f%s",
        spec.rid, spec.page_prefix, rows, state_at, extend, len(st.headers), st.n_parts or "?", spec.key,
        st.verdict, adopt, st.end_verdict, st.first_token, st.token_src, waited_ms,
        f" skip_refused={skip_note}" if skip_note else "",
    )


def skip_refusal(entry: Agreed, req, batch_empty: bool) -> str:
    """E2 admission: '' when the group may take the END state; every input
    is rank-uniform (the batch's contents, the request's own parameters,
    its token ids), so every rank decides the same."""
    spec = entry.staged.spec
    if not batch_empty:
        return "batch_not_empty"  # the skipped batch must hold nothing else
    if req.return_logprob or req.return_hidden_states:
        return "logprob_or_hidden"
    if req.grammar is not None:
        return "grammar"
    sp = req.sampling_params
    if sp.frequency_penalty or sp.presence_penalty or sp.repetition_penalty != 1.0 or sp.min_new_tokens:
        return "penalty_or_min_new_tokens"  # P's token never reached D's penalizer state
    end = entry.staged.headers[0].end
    if th.tail_key(req.origin_input_ids, spec.n_tokens, req.extra_key) != end.key:
        return "end_key_mismatch"
    return ""


def plan_adopt(req, prefix_len: int, batch_empty: bool = True) -> Optional[Agreed]:
    """Admission, before the commit: the agreed outcome for ``req`` if the
    group takes the tail now, else None (today's extend). One-shot per rid.
    ``entry.skip`` says whether it takes the END state (E2) or E1."""
    if not adopt_enabled():
        return None
    entry = _AGREED.pop(str(req.rid), None)
    if entry is None:
        return None
    if not entry.agreed:
        _log_ready(entry.staged, "skipped:group_vote", waited_ms=entry.waited_ms)
        return None
    why = uniform_refusal(entry.staged.spec, req.origin_input_ids, len(req.full_untruncated_fill_ids),
                          req.extra_key, prefix_len)
    if why:
        _log_ready(entry.staged, f"skipped:{why}", waited_ms=entry.waited_ms)
        return None
    if entry.skip:
        why = skip_refusal(entry, req, batch_empty)
        if why:
            entry.skip, entry.skip_note = False, f"admission:{why}"
            entry.staged.drop_end()
    return entry


def commit_adopt(req, entry: Agreed, tree_cache, page_size: int) -> int:
    """Admission commit (the request is going into this batch): one page,
    the prefix grows by its first ``rows`` slots, the holding rank queues the
    install. Returns the new prefix length c."""
    from sglang.srt.mem_cache.common import alloc_token_slots

    st = entry.staged
    spec = st.spec
    if entry.skip:
        return _commit_skip(req, entry, tree_cache, page_size)
    page = alloc_token_slots(tree_cache, int(page_size))
    rows = page[: spec.rows].to(dtype=req.prefix_indices.dtype, device=req.prefix_indices.device)
    req.prefix_indices = torch.cat([req.prefix_indices, rows])
    _log_ready(st, "done", skip_note=entry.skip_note, waited_ms=entry.waited_ms)
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
    #: E2 (SKIP): the END state -- written in one go by ``run_skip``; rows,
    #: groups, ring slots and the mamba slot are resolved there (the
    #: request's req_pool_idx and its extend slots [c, N) exist only after
    #: prepare_for_extend)
    end: bool = False
    ring: Dict[int, Tuple[torch.Tensor, ...]] = {}
    ring_dst: Dict[int, torch.Tensor] = {}
    rope: Optional[torch.Tensor] = None
    rope_dst: Optional[torch.Tensor] = None
    req_to_token: Optional[torch.Tensor] = None
    translate: Optional[object] = None
    ratio: int = 0


#: installs waiting for the extend forward's GDN layer reads (memory_pool
#: HybridReqToTokenPool.mamba2_layer_cache checks this list per call).
PENDING_INSTALLS: List[Install] = []


class SkipPlan(msgspec.Struct):
    """E2: a request admitted with the END state; its batch runs no forward."""

    spec: th.TailSpec
    first_token: int
    install: Optional[Install]  # None on a rank that holds no layer
    t0: float


#: rid -> plan, from the admission commit to the batch's (skipped) forward.
SKIP_PLANS: Dict[str, SkipPlan] = {}


def _fa_dst(kvpool, fa: Dict[int, Tuple[torch.Tensor, ...]]) -> Dict[int, Tuple[torch.Tensor, ...]]:
    out: Dict[int, Tuple[torch.Tensor, ...]] = {}
    for gid in fa:
        local = kvpool.full_attention_layer_id_mapping[gid]
        full = kvpool.full_kv_pool
        dst = (full.k_buffer[local], full.v_buffer[local])
        if len(fa[gid]) == 3:
            dst = dst + (kvpool.qsa_compressed_k_buffer_pool[local],)
        out[gid] = dst
    return out


def _gdn_dst(rtp, gdn: Dict[int, Tuple[torch.Tensor, ...]]) -> Dict[int, Tuple[torch.Tensor, ...]]:
    cache = rtp.mamba_pool.mamba_cache
    return {gid: (cache.temporal[rtp.mamba_map[gid]],) + tuple(c[rtp.mamba_map[gid]] for c in cache.conv)
            for gid in gdn}


def _queue_install(req, st: Staged, rows: torch.Tensor, tree_cache) -> None:
    kvpool = tree_cache.token_to_kv_pool_allocator.get_kvcache()
    rtp = tree_cache.req_to_token_pool
    fa_dst = _fa_dst(kvpool, st.fa)
    gdn_dst: Dict[int, Tuple[torch.Tensor, ...]] = {}
    slot = None
    if st.gdn:
        if req.mamba_pool_idx is None:
            # the group already runs [c, N); without a slot there is no state
            # to overwrite -- the same wrongness as any prefix without a slot
            logger.error("WEG2-TAIL-ADOPT rid=%s FAILED: no mamba slot on a GDN-holding rank", st.spec.rid)
            return
        slot = rtp.translate_mamba_indices(req.mamba_pool_idx.reshape(1).to(torch.int64))
        gdn_dst = _gdn_dst(rtp, st.gdn)
    ratio = st.qsa_ratio
    groups = (rows[::ratio] // ratio) if (ratio and st.fa) else None
    PENDING_INSTALLS.append(Install(
        spec=st.spec, headers=st.headers, fa=st.fa, gdn=st.gdn, fa_dst=fa_dst, gdn_dst=gdn_dst,
        rows=rows, groups=groups, slot=slot, t0=time.perf_counter(), readback=st.readback,
    ))


# -- 4. SKIP (E2) ----------------------------------------------------------------------
def _commit_skip(req, entry: Agreed, tree_cache, page_size: int) -> int:
    """E2 admission commit: ONE page, the prefix grows to c inside it (its
    first c-page_prefix slots, E1's shape), the extend [c, N) runs no target
    forward; the holding rank prepares the END install. Returns c."""
    from sglang.srt.mem_cache.common import alloc_token_slots

    st = entry.staged
    spec = st.spec
    page = alloc_token_slots(tree_cache, int(page_size))
    rows = page[: spec.rows].to(dtype=req.prefix_indices.dtype, device=req.prefix_indices.device)
    req.prefix_indices = torch.cat([req.prefix_indices, rows])
    _log_ready(st, "done", skip=True, waited_ms=entry.waited_ms)
    inst = None
    if st.end_verdict == "ready":
        inst = _skip_install(st, tree_cache)
    SKIP_PLANS[str(req.rid)] = SkipPlan(spec=spec, first_token=st.first_token, install=inst,
                                        t0=time.perf_counter())
    while len(SKIP_PLANS) > KEEP_AGREED:  # an admitted batch that never ran
        SKIP_PLANS.pop(next(iter(SKIP_PLANS)))
    return spec.cut


def _skip_install(st: Staged, tree_cache) -> Install:
    kvpool = tree_cache.token_to_kv_pool_allocator.get_kvcache()
    rtp = tree_cache.req_to_token_pool
    ring_dst = {g: kvpool.qsa_key_state_buffer_pool[kvpool.full_attention_layer_id_mapping[g]]
                for g in st.end_ring}
    return Install(
        spec=st.spec, headers=st.headers, fa=st.end_fa, gdn=dict(st.end_gdn), fa_dst=_fa_dst(kvpool, st.end_fa),
        gdn_dst=_gdn_dst(rtp, st.end_gdn), rows=torch.empty(0, dtype=torch.int64), groups=None, slot=None,
        t0=time.perf_counter(), readback=st.end_readback, end=True, ring=st.end_ring, ring_dst=ring_dst,
        rope=st.end_rope, rope_dst=kvpool.qsa_rope_position_buffer if st.end_rope is not None else None,
        req_to_token=rtp.req_to_token, translate=rtp.translate_mamba_indices, ratio=st.qsa_ratio,
    )


def skip_tokens(batch) -> Optional[List[int]]:
    """Worker entry, extend branch: P's tokens when EVERY request of the
    batch was admitted with the END state (then no forward runs), else None.
    A batch mixing the two cannot be formed (PrefillAdder) and is refused
    loudly: running the extend over a state already at N would advance it
    twice."""
    if not SKIP_PLANS:
        return None
    rids = [str(r.rid) for r in batch.reqs]
    hits = [rid in SKIP_PLANS for rid in rids]
    if not any(hits):
        return None
    if not all(hits):
        raise RuntimeError(
            f"WEG2-TAIL-SKIP-EXTEND mixed batch {rids}: a request already at its END state shares an "
            f"extend batch with one that needs a forward -- the adder's batch separation was bypassed"
        )
    return [SKIP_PLANS[rid].first_token for rid in rids]


def run_skip(batch, counter, draft: str = "extend") -> None:
    """Worker entry, on the forward stream, instead of the target forward:
    join the batch's HiCache load, write every END install, log. ``draft``
    names what seeds the chain on this rank (the host's draft extend, or a
    shadow's stub)."""
    t = time.perf_counter()
    if counter is not None and batch.hicache_consumer_index >= 0:
        # the load-back of [0, floor_page(c)) (+ its GDN anchor into the
        # same slot) rides the load stream; the next forward is a graph
        # replay that joins nothing, and our state must land after the anchor
        counter.set_consumer(batch.hicache_consumer_index)
        counter.wait_until(counter.num_layers - 1)
    for req in batch.reqs:
        plan = SKIP_PLANS.pop(str(req.rid))
        if plan.install is not None:
            _install_end(plan.install, req)
        logger.info(
            "WEG2-TAIL-SKIP-EXTEND rid=%s prefix=%d first_token=%d draft=%s draft_rows=%d held=%s ms=%.1f "
            "since_commit_ms=%.1f (no target forward: P's END state + token; the draft phase of [c, N) runs "
            "as after a real extend, off zero target hidden states)",
            plan.spec.rid, plan.spec.n_tokens, plan.first_token, draft, plan.spec.extend,
            "yes" if plan.install is not None else "not_mine",
            (time.perf_counter() - t) * 1000.0, (time.perf_counter() - plan.t0) * 1000.0,
        )


def _install_end(inst: Install, req) -> None:
    """Write the END state at this request's own slots (device indices only,
    no host sync): rows [page_prefix, N) from req_to_token, the complete
    groups at slot // r, the open group's ring rows, the GDN slot."""
    t = time.perf_counter()
    spec = inst.spec
    end = inst.headers[0].end
    rows = inst.req_to_token[int(req.req_pool_idx), spec.page_prefix:spec.n_tokens].to(torch.int64)
    inst.rows = rows
    if inst.ratio:
        inst.groups = th.group_slots(rows, inst.ratio, end.groups)
    for gid in sorted(inst.fa):
        back = inst.readback.get(f"fa{gid}")
        for j, (dst, src) in enumerate(zip(inst.fa_dst[gid], inst.fa[gid])):
            _put(dst, inst.groups if j == 2 else rows, src, None if back is None else back[j])
    if inst.ratio and (inst.ring or inst.rope is not None):
        ring_idx = th.ring_slots(req.req_pool_idx, inst.ratio, end.ring_rows)
        for gid in sorted(inst.ring):
            back = inst.readback.get(f"ring{gid}")
            _put(inst.ring_dst[gid], ring_idx, inst.ring[gid][0], None if back is None else back[0])
        if inst.rope is not None:
            back = inst.readback.get("rope")
            _put(inst.rope_dst, ring_idx, inst.rope, None if back is None else back[0])
    if inst.gdn:
        inst.slot = inst.translate(req.mamba_pool_idx.reshape(1).to(torch.int64))
        for gid in sorted(inst.gdn):
            back = inst.readback.get(f"gdn{gid}")
            for j, (dst, s) in enumerate(zip(inst.gdn_dst[gid], inst.gdn[gid])):
                _put(dst, inst.slot, s, None if back is None else back[j])
    inst.issue_ms = (time.perf_counter() - t) * 1000.0
    _finish(inst, bool(inst.readback))


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
        want = (h.end.fa_digest, h.end.gdn_digest) if inst.end else (h.fa_digest, h.gdn_digest)
        if (th.digest(fa), th.digest(gdn)) != want:
            return f"MISMATCH:{h.part}"
        if inst.end and h.end.ring_digest != _ring_readback_digest(inst, h):
            return f"MISMATCH:{h.part}:ring"
    return "match"


def _ring_readback_digest(inst: Install, h: th.TailHeader) -> str:
    ring = [inst.readback[f"ring{g}"][0] for g in sorted(h.fa_layers) if f"ring{g}" in inst.readback]
    rope = inst.readback.get("rope")
    return th.digest(ring + ([rope[0]] if rope is not None else []))


def _verify_and_log(inst: Install, verify: bool, event) -> None:
    try:
        if event is not None:
            event.synchronize()
        dig = readback_digest(inst) if verify else "off"
    except Exception as exc:  # noqa: BLE001 -- an instrument, named
        dig = f"verify_raised:{type(exc).__name__}"
    rows = inst.spec.n_tokens - inst.spec.page_prefix if inst.end else inst.spec.rows
    _log_adopt(inst.spec, fa_rows=rows, fa_layers=len(inst.fa_dst), gdn_layers=len(inst.gdn_dst),
               digest=dig, ms=(time.perf_counter() - inst.t0) * 1000.0, issue_ms=inst.issue_ms, end=inst.end)


def _log_adopt(spec: th.TailSpec, fa_rows: int, fa_layers: int, gdn_layers: int, digest: str, ms: float,
               issue_ms: float, end: bool = False) -> None:
    level = logging.ERROR if digest.startswith("MISMATCH") else logging.INFO
    rows, state_at, extend = (fa_rows, spec.n_tokens, 0) if end else (spec.rows, spec.cut, spec.extend)
    logger.log(
        level,
        "WEG2-TAIL-ADOPT rid=%s page_prefix=%d tail_rows=%d state_at=%d extend=%d fa_rows_written=%d "
        "fa_layers=%d gdn_layers=%d digest=%s ms=%.1f issue_ms=%.1f",
        spec.rid, spec.page_prefix, rows, state_at, extend, fa_rows, fa_layers, gdn_layers,
        digest, ms, issue_ms,
    )
