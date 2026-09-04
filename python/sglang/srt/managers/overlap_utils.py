from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.srt.debug_utils import index_race_guard
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    server_args: ServerArgs,
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO (it reads the
    CPU mirror outside the backend layer to split the batch) or ngram (its
    USE_FULL_MASK verify path reads the host mirror regardless of backend).
    """
    # Local import: keep overlap_utils' module-level deps leaf-only so it stays
    # importable everywhere; spec_info pulls in the spec/schedule_batch graph.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if server_args.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        return True
    algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if algo.is_ngram():
        # ngram's USE_FULL_MASK verify path reads seq_lens_cpu per req to size
        # the tree mask, regardless of the attn backend (e.g. Triton opts out).
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def decide_needs_confidence_relay(server_args: ServerArgs) -> bool:
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


class FutureMapPhaseMismatch(RuntimeError):
    """A FutureMap consulted by a phase that did not build it (#1201 B3)."""


def build_future_map(scheduler: Any) -> "FutureMap":
    """Build the cross-iter relay for the phase the scheduler is in RIGHT NOW.

    Extracted from ``Scheduler.init_overlap`` for one reason (#1201 B3): the map
    is stamped with two things the phase flip REPLACES, and until this was
    callable a second time the TP decode phase consulted the map the PP prefill
    phase had built.

      * ``spec_algo``.  A phase-flip instance boots with ``spec_algorithm``
        forced to NONE (scheduler.py:820-823, #631 -- the PP prefill phase has
        no draft worker), so the boot map's stamp is NONE for the life of the
        process.  ``resolve_forward_inputs`` branches on exactly that field, and
        the NON-overlap spec path reaches it -- the ``elif not
        batch.spec_algorithm.is_none():`` arm of ``run_batch``, a SIBLING of
        the ``if self.enable_overlap:`` arm and not nested inside it -- so the
        frozen stamp is read on every decode round of the TP phase.  (Named by
        its branch rather than by a line number on purpose: the number that
        stood here, scheduler.py:13174-13177, had already drifted onto the
        OVERLAP call INSIDE ``if self.enable_overlap:`` -- onto the line that
        refutes the sentence instead of the one that proves it.)  Nothing raises: the only
        assertion on that branch is gated on ``SGLANG_IS_IN_CI``.
      * the request pool.  ``req_pool_size``, ``max_context_len`` and
        ``ConfidenceRelay.pool`` all come from the pool that existed when this
        ran.  This is the fourth holder of the #1201 request-pool class; the
        other three are re-stamped by ``phase_req_pool_binding``.

    Duck-typed on the scheduler on purpose: it needs five attributes, and typing
    it to ``Scheduler`` would make the cutover import that module to call it.

    CALL ORDER AT THE SEAM.  Both stamps must already be settled, so this runs
    after the algorithm swap AND after the request-pool rebind.  Called between
    them it would carry the incoming algorithm and the OUTGOING pool.
    """
    # Workers without the spec_v2_attn_backends override fall back to
    # target-only so the helper still produces a safe decision (no accidental
    # opt-out for unaudited shapes).
    if scheduler.draft_worker is not None:
        attn_backends = getattr(
            scheduler.draft_worker,
            "spec_v2_attn_backends",
            (scheduler.tp_worker.model_runner.attn_backend,),
        )
    else:
        attn_backends = (scheduler.tp_worker.model_runner.attn_backend,)
    return scheduler.spec_algorithm.create_future_map(
        scheduler.device,
        scheduler.req_to_token_pool,
        needs_cpu_seq_lens=decide_needs_cpu_seq_lens(
            scheduler.server_args, attn_backends
        ),
        needs_confidence_relay=decide_needs_confidence_relay(scheduler.server_args),
    )


def assert_future_map_identity(scheduler: Any, previous: Any = None) -> None:
    """The map this phase consults was built BY this phase, or the seam stops.

    The registry's second obligation for ``future_map_req_pool_holder``: a probe
    proving the rebuild ran.  It is not a re-implementation of the rebuild -- it
    refuses when a stamp is still the outgoing phase's, whatever the reason (a
    rebuild placed too early, a second map cached somewhere else, a future field
    stamped at construction that nobody moved).

    Loud on purpose, because neither divergence fails on its own.  A stale
    ``spec_algo`` silently relays input ids the spec worker owns; a stale pool
    reads a tensor of the same shape belonging to the other phase.

    Duck-typed and absence-tolerant, like ``assert_req_pool_identity``: an
    instance that never built a map is not a divergence.

    WHAT THE TWO STAMP ARMS BELOW CAN AND CANNOT CATCH -- WRITTEN OUT BECAUSE
    THE FIRST CUT SHIPPED WITHOUT IT.  ``build_future_map`` calls
    ``scheduler.spec_algorithm.create_future_map(...)``, which passes ``self``
    as ``spec_algo`` (spec_info.py) and ``scheduler.req_to_token_pool`` straight
    into ``ConfidenceRelay(pool=...)``.  Both stamps are therefore identical to
    the scheduler's own fields BY CONSTRUCTION, and at the seam -- where this
    runs immediately after the rebuild -- neither arm can fire.  Measured: four
    combinations of (algo NONE|EAGLE) x (pool pp|tp), zero fires.  They are
    live only for a caller that runs them somewhere the map was NOT just
    rebuilt (a later decode round, the next arm), which is where a fifth holder
    or a second cached map would actually show.

    ``previous`` IS THE ARM THAT CAN FAIL AT THE SEAM, and it is the one the
    registry's obligation actually asks for: hand in the map the scheduler held
    BEFORE the rebuild and this refuses when the rebuild did not replace it.
    That is the whole content of "a probe proving the rebuild ran" -- a memoised
    ``build_future_map``, a deleted assignment, or a rebuild moved out of the
    cutover all land here, and none of them are visible to the stamp arms.
    """
    future_map = getattr(scheduler, "future_map", None)
    if future_map is None:
        return

    if previous is not None and future_map is previous:
        raise FutureMapPhaseMismatch(
            "#1201 the cutover did not replace the future map: the scheduler "
            "still holds the object the OUTGOING phase built. Its stamps "
            "cannot show this -- they are equal to the scheduler's fields by "
            "construction whenever the map was built from them -- so the only "
            "evidence that the rebuild ran is that the object changed."
        )

    algo = getattr(scheduler, "spec_algorithm", None)
    stamped = getattr(future_map, "spec_algo", None)
    if algo is not None and stamped is not algo:
        raise FutureMapPhaseMismatch(
            f"#1201 the future map is stamped spec_algo={stamped}, but this "
            f"phase runs spec_algorithm={algo}. resolve_forward_inputs branches "
            "on the map's stamp, so a NONE stamp under a speculative phase "
            "gathers last iteration's sampled token into batch.input_ids -- "
            "silently, since _assert_nonneg_and_invalidate is CI-gated."
        )

    pool = getattr(scheduler, "req_to_token_pool", None)
    relay = getattr(future_map, "confidence_relay", None)
    held = getattr(relay, "pool", None) if relay is not None else None
    if pool is not None and held is not None and held is not pool:
        raise FutureMapPhaseMismatch(
            "#1201 the future map still names the outgoing phase's request "
            f"pool (map binding={getattr(held, 'binding_tag', '<untagged>')}, "
            f"scheduler binding={getattr(pool, 'binding_tag', '<untagged>')}). "
            "Both pools hold the same row count, so this does not fail on its "
            "own: the relay would address another phase's rows in range."
        )


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).
    """
    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and (
        future_map.spec_algo.is_none()
        or getattr(batch, "kv_session_spill_tick", False)
    ):
        # kv-session-offload spill tick: a bs=1 host-resident decode carrying
        # spec_algorithm=NONE inside an otherwise-spec server. _build_spill_batch
        # stashed the session's last committed token into output_tokens_buf (via
        # future_map.stash) and left input_ids=None; the server-global spec_algo
        # gate would skip this resolution, so admit the spill-tick batch
        # explicitly. Gather the last token exactly as the non-spec decode path.
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # Only the overlap path relays spec extras through the future_map; the
    # synchronous (non-overlap) V2 path installs next_draft_input directly.
    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)


CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):

    confidence: torch.Tensor
    generation: torch.Tensor


def relay_field(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Normalize one relay extra to "present" (a tensor) or "absent" (None).

    A drafter that does not OWN an Eagle-shaped field carries a zero-width
    PLACEHOLDER for it rather than nothing: DFlashDraftInputV2's topk_p /
    topk_index / hidden_states are documented as "legacy Eagle-shaped fields;
    DFLASH relays via FutureMap so these are unused" and
    draft_worker_common.make_draft_input_v2 builds them as ``(bs, 0)``. That
    placeholder is semantically ABSENT, and the stash path already has a
    complete absent-field contract -- but it keys on ``is None``, so a
    zero-width tensor slipped past it and was written into a slot sized from
    a real payload.

    Under cross-algorithm switching that used to be invisible: the per-round
    warm-keep of the idle NEXTN rung overwrote all three fields with real
    seeds on EVERY DFLASH round, so a placeholder never reached the stash.
    Dropping that per-round warm-keep is exactly what lazy single capture
    does (#156-4), which is why the crash surfaced there first -- but the
    latent hole is in the relay, not in the lazy path, so it is fixed here.

    Zero-numel is a pure local SHAPE test, identical on every rank (batch
    size and draft width are rank-uniform), so the branch cannot desync.
    """
    if t is None or t.numel() == 0:
        return None
    return t


@dataclass
class RelayPayload:
    """Per-iteration stash payload for the FutureMap bufs. Non-spec fills only
    `bonus_tokens`; which spec extras get relayed is decided by
    `FutureMap.spec_algo`, not by this payload's shape.

    Spec extras are None when ABSENT -- see relay_field: a drafter that does
    not own a field must not present its zero-width placeholder as data."""

    bonus_tokens: torch.Tensor
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=relay_field(draft_input.topk_p),
            topk_index=relay_field(draft_input.topk_index),
            hidden_states=relay_field(draft_input.hidden_states),
            draft_probs=relay_field(getattr(draft_input, "draft_probs", None)),
            dsa_topk_indices=relay_field(
                getattr(draft_input, "dsa_topk_indices", None)
            ),
        )


class ConfidenceRelay(msgspec.Struct):

    device: torch.device
    req_pool_size: int
    pool: Any
    confidence_buf: Optional[torch.Tensor] = None
    conf_ring: Optional[torch.Tensor] = None
    gen_ring: Optional[torch.Tensor] = None
    copy_done: Optional[list] = None
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        if _is_cuda:
            depth = CONFIDENCE_RELAY_RING_DEPTH
            self.conf_ring = torch.empty(
                (depth, self.req_pool_size, gamma),
                dtype=torch.float32,
                pin_memory=True,
            )
            self.gen_ring = torch.zeros((depth, self.req_pool_size), dtype=torch.int64)
            self.copy_done = [
                torch.get_device_module(self.device).Event() for _ in range(depth)
            ]

    def scatter(self, indices: torch.Tensor, confidence: torch.Tensor) -> None:
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)

    def issue_ring_copy(self, *, stream, publish_ready) -> None:
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.gen_ring[slot].copy_(self.pool.req_generation)
        self.ring_pos += 1

    def resolve(
        self, batch: ScheduleBatch, *, stream, publish_ready
    ) -> Optional[ResolvedConfidence]:
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        if stream is None or publish_ready is None:
            idx = batch.req_pool_indices
            idx_cpu = batch.req_pool_indices_cpu
            return ResolvedConfidence(
                confidence=self.confidence_buf[idx].cpu(),
                generation=self.pool.req_generation[idx_cpu].clone(),
            )

        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            generation=self.gen_ring[slot][idx_cpu],
        )


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.
    """

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]
        # Row width of req_to_token: the bound every relayed seq_len must
        # respect, since seq_lens address that row. Kept for the #616 guard.
        self.max_context_len = req_to_token_pool.req_to_token.shape[1]

        if _DEBUG_ASSERT:
            # Poisoned init: every row must be written before its first gather.
            self.output_tokens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
        else:
            self.output_tokens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
        # Lazy-inited on the first non-empty stash (peeks tensor shapes); non-spec's is a no-op.
        self._forward_buf_initialized = False

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it
        # Debug consume-once state: armed by a recording publish, consumed by
        # resolve; arm/consume strictly alternate across all batch interleavings.
        self._publish_fresh = False

        self.confidence_relay = ConfidenceRelay(
            device=self.device,
            req_pool_size=self.req_pool_size,
            pool=req_to_token_pool,
        )

    def _lazy_init_forward_buf(self, payload: RelayPayload):
        # Local import (see decide_needs_cpu_seq_lens): keep module-level deps leaf.
        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        self._forward_buf_initialized = True

        # Spec extras are gated by spec_algo, not by the payload's shape, so a
        # non-spec stash allocates no extra bufs (only output_tokens_buf).
        self.need_topk = self.spec_algo.is_some() and self.spec_algo.need_topk()
        # Deliberately NOT gated on `payload.hidden_states is not None`: which
        # extras this server relays is a property of the CONFIGURATION, and
        # latching it from whichever payload happened to be stashed first
        # makes it depend on stash ORDER. That order is algorithm-dependent
        # under cross-algorithm switching -- a DFLASH round (or any bonus-only
        # stash) can legitimately come first, and it carries no hidden states,
        # which would pin need_hidden_states to False for the FutureMap's whole
        # lifetime and silently strip the NEXTN/MTP rung of its hidden-state
        # relay forever after. Not a crash: a silent accept-rate regression.
        # The per-field lazy allocation below is what tolerates an absent
        # field, and it already does so correctly (_ensure_hidden_buf returns
        # False and nothing is written), so no buffer is allocated until a
        # payload that HAS the field arrives.
        self.need_hidden_states = (
            self.spec_algo.is_some() and spec_need_hidden_states()
        )

        # Buf allocation is per-field lazy (see _ensure_topk_bufs /
        # _ensure_hidden_buf): under cross-algorithm switching (T156 stage 3)
        # a payload may legitimately carry only a subset of the spec extras,
        # so shapes are peeked from the first payload that HAS the field.
        self.topk_p_buf = None
        self.topk_index_buf = None
        self.hidden_states_buf = None
        if self.need_topk:
            self._ensure_topk_bufs(payload)
        if self.need_hidden_states:
            self._ensure_hidden_buf(payload)

        self.draft_probs_buf = None
        if payload.draft_probs is not None:
            draft_probs0 = payload.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

        self.dsa_topk_indices_buf = None
        if payload.dsa_topk_indices is not None:
            seed0 = payload.dsa_topk_indices[0]
            self.dsa_topk_indices_buf = torch.empty(
                (self.req_pool_size, *seed0.shape),
                dtype=payload.dsa_topk_indices.dtype,
                device=self.device,
            )

    def _ensure_topk_bufs(self, payload: RelayPayload) -> bool:
        if self.topk_p_buf is not None:
            return True
        # Shape PEEK: never size the pool buffers from a zero-width
        # placeholder (relay_field) -- a (req_pool_size, 0) buffer would
        # silently swallow every later write and then mismatch the first
        # real payload.
        topk_p = relay_field(payload.topk_p)
        topk_index = relay_field(payload.topk_index)
        if topk_p is None or topk_index is None:
            return False
        topk_p0 = topk_p[0]
        topk_index0 = topk_index[0]
        self.topk_p_buf = torch.empty(
            (self.req_pool_size, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.req_pool_size, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        return True

    def _ensure_hidden_buf(self, payload: RelayPayload) -> bool:
        if self.hidden_states_buf is not None:
            return True
        hidden_states = relay_field(payload.hidden_states)
        if hidden_states is None:
            return False
        hidden_states0 = hidden_states[0]
        self.hidden_states_buf = torch.empty(
            (self.req_pool_size, *hidden_states0.shape),
            dtype=hidden_states0.dtype,
            device=self.device,
        )
        return True

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(
            batch,
            stream=self.fwd_prepare_d2h_stream,
            publish_ready=self.publish_ready,
        )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            return
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        if self.need_topk and self.topk_p_buf is not None:
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            (
                draft_input.topk_p,
                draft_input.topk_index,
                bonus_tokens,
                hidden_states,
            ) = gather_spec_extras(
                indices,
                self.topk_p_buf,
                self.topk_index_buf,
                self.output_tokens_buf,
                hidden_states_buf,
            )
            draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        else:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if (
            self.need_hidden_states
            and not (self.need_topk and self.topk_p_buf is not None)
            and self.hidden_states_buf is not None
        ):
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if self.dsa_topk_indices_buf is not None:
            draft_input.dsa_topk_indices = self.dsa_topk_indices_buf[indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # schedule). The CPU mirror is gated by needs_cpu_seq_lens; backends that
        # opt out take the GPU-only path below. A private D2H stream overlaps the copy.
        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            if _DEBUG_ASSERT:
                # Consume-once: every event wait must be re-armed by a fresh
                # forward publish; a stale consume means a publish went missing.
                assert self._publish_fresh, "resolve without a fresh forward publish"
                self._publish_fresh = False
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

        # #616 instrument: the relay is the prime suspect for a cross-stream
        # read -- new_seq_lens_buf is written by the forward stream and gathered
        # here on the schedule stream. A hit means the gather observed a row
        # mid-update despite the publish_ready fence.
        if index_race_guard.is_enabled():
            index_race_guard.guard("future_indices", fi, 0, self.req_pool_size)
            index_race_guard.guard(
                "relayed_seq_lens", batch.seq_lens, 0, self.max_context_len
            )

        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            if _DEBUG_ASSERT:
                # Poison consumed rows: each row must be re-published/seeded
                # before the next resolve gathers it (safe here: the forward's
                # re-publish is fenced behind this stream via wait_stream).
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()

        # FIXME: fi == batch.req_pool_indices; unify future_indices and req_pool_indices.
        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        if _DEBUG_ASSERT:
            # After the D2H copy completed (synchronize above), so the pinned
            # mirror is not poisoned.
            _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        if self.spec_algo.is_some():
            device_module = torch.get_device_module(self.device)
            if self.publish_ready is None:
                self.publish_ready = device_module.Event()
            else:
                # Chain the records: event fire implies every prior publish is
                # visible, so an off-forward-stream publish (PD-decode prebuilt
                # seeding) cannot drop the in-flight forward's fence.
                device_module.current_stream().wait_event(self.publish_ready)
            self.publish_ready.record()
            self._publish_fresh = True
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                stream=self.fwd_prepare_d2h_stream,
                publish_ready=self.publish_ready,
            )

    def stash(self, future_indices: torch.Tensor, payload: RelayPayload) -> None:
        if self.spec_algo.is_ngram():
            # FIXME: remove once precomputed draft is supported.
            return
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            return
        if not self._forward_buf_initialized:
            self._lazy_init_forward_buf(payload)
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )

        # Absent-tolerant per field (cross-algorithm switching may stash
        # bonus-only payloads); an absent field simply keeps the previous
        # relayed value, and the consumer contract guarantees a round that
        # READS a field was preceded by a round that stashed it.
        # "Absent" means None OR a zero-width placeholder -- see relay_field;
        # re-normalized here so a directly built payload cannot bypass it.
        topk_p = relay_field(payload.topk_p)
        topk_index = relay_field(payload.topk_index)
        hidden_states = relay_field(payload.hidden_states)
        if self.need_topk and topk_p is not None and topk_index is not None:
            if self._ensure_topk_bufs(payload):
                self.topk_p_buf[indices] = topk_p.to(self.topk_p_buf.dtype)
                self.topk_index_buf[indices] = topk_index.to(
                    self.topk_index_buf.dtype
                )
        if self.need_hidden_states and hidden_states is not None:
            if self._ensure_hidden_buf(payload):
                self.hidden_states_buf[indices] = hidden_states.to(
                    self.hidden_states_buf.dtype
                )
        if self.draft_probs_buf is not None and payload.draft_probs is not None:
            self.draft_probs_buf[indices] = payload.draft_probs
        if (
            self.dsa_topk_indices_buf is not None
            and payload.dsa_topk_indices is not None
        ):
            self.dsa_topk_indices_buf[indices] = payload.dsa_topk_indices.to(
                self.dsa_topk_indices_buf.dtype
            )
