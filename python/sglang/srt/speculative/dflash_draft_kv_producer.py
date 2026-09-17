# DFlash draft-KV producer for Weg 2's prefill group (P).
#
# THE NEXTN PRODUCER (draft_kv_producer.DraftKvProducer) runs the MTP layer's
# draft-extend over each target chunk and writes the draft rows into a device
# draft pool that MIRRORS the target pool's slot space (4 KiB/token); the
# flip's write-back then copies those rows to the draft arena by target row.
#
# A DFlash2 draft row is 20 KiB/token (5 layers x 8 kv heads x 128 x K+V x
# bf16). A mirror of P's pool (~460k slots) would be 9.4 GB on the last
# stage's 3080 -- there is no such room. So this producer does NOT mirror:
#
#   * the draft KV of each chunk is materialised into a CHUNK RING (a small
#     MHATokenToKVPool the size of one prefill batch),
#   * and published in the same call, hash-keyed, straight into the draft
#     arena (ArenaMHAHostPool.publish_direct): one claim per page hash, one
#     all-layer device->arena copy, one completion. The ring is free again
#     when ``produce`` returns.
#
# The page hashes are the canonical position-aware chain the radix cache
# computes for the target's pages (mem_cache.utils.get_hash_str, page_size 1,
# chained from position 0), so a page D asks for at admission is the page P
# wrote -- same key function, same component suffix (drafter identity).
# Chunked prefills continue the chain from the previous chunk's last hash.
#
# What this rank captures for the draft is assembled across the PP stages by
# distributed/pp_aux_capture (part A); the LogitsProcessor hands it to the
# scheduler's ``_draft_kv_produce`` as ``logits_output.hidden_states`` in
# CaptureHiddenMode.FULL, exactly as it hands the MTP producer its hidden.
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

import torch

from sglang.srt.speculative.draft_kv_producer import (
    Weg2DraftRegistrationOffStage,
    _cuda_free_mib,
    _draft_server_args,
    _live_weight_mib,
)

logger = logging.getLogger(__name__)

_HASH_POS_ATTR = "_weg2_dflash_hash_pos"
_HASH_LAST_ATTR = "_weg2_dflash_hash_last"


class DFlashDraftKvProduceError(RuntimeError):
    """A chunk's draft rows could not be produced or published."""


def chunk_page_hashes(req, prefix_len: int, extend_len: int) -> List[str]:
    """The canonical page hashes of ``req``'s positions
    ``[prefix_len, prefix_len + extend_len)``, continuing the chain cached
    on the request when the previous chunk ended exactly at ``prefix_len``.

    ``req.fill_ids`` is the token stream the target consumed (origin prompt
    plus any output already filled), which is what the radix keys hash.
    """
    from sglang.srt.mem_cache.utils import get_hash_str

    if extend_len <= 0:
        return []
    tokens = list(req.fill_ids)
    end = prefix_len + extend_len
    if end > len(tokens):
        raise DFlashDraftKvProduceError(
            f"rid={getattr(req, 'rid', '?')}: chunk [{prefix_len}, {end}) "
            f"reaches past the {len(tokens)} filled token(s)"
        )
    cached_pos = getattr(req, _HASH_POS_ATTR, None)
    cached_last = getattr(req, _HASH_LAST_ATTR, None)
    if prefix_len == 0:
        hashes = get_hash_str(tokens[:end], None, page_size=1)
    elif cached_pos == prefix_len and cached_last:
        hashes = get_hash_str(tokens[prefix_len:end], cached_last, page_size=1)
    else:
        # A prefix hit or a chunk boundary we did not see: rebuild the chain
        # from position 0 (same function, same result, only slower).
        hashes = get_hash_str(tokens[:end], None, page_size=1)[prefix_len:]
    if not isinstance(hashes, list) or len(hashes) != extend_len:
        raise DFlashDraftKvProduceError(
            f"rid={getattr(req, 'rid', '?')}: expected {extend_len} page hash(es) "
            f"for the chunk, got {len(hashes) if isinstance(hashes, list) else type(hashes).__name__}"
        )
    setattr(req, _HASH_POS_ATTR, end)
    setattr(req, _HASH_LAST_ATTR, hashes[-1])
    return hashes


def batch_page_hashes(batch) -> List[str]:
    out: List[str] = []
    for req, pl, el in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
        out.extend(chunk_page_hashes(req, int(pl), int(el)))
    return out


def ring_capacity(server_args) -> int:
    """One prefill batch of draft rows fits the ring, by construction of the
    batch: the scheduler bounds a prefill batch by max_prefill_tokens and a
    single request's chunk by chunked_prefill_size."""
    cands = [
        int(getattr(server_args, "max_prefill_tokens", 0) or 0),
        int(getattr(server_args, "chunked_prefill_size", 0) or 0),
    ]
    cap = max(cands)
    if cap <= 0:
        raise ValueError(
            "DFlash draft-KV producer: neither --max-prefill-tokens nor "
            "--chunked-prefill-size bounds a prefill batch; the chunk ring "
            "cannot be sized"
        )
    return cap


class DFlashDraftKvProducer:
    """Group P's DFlash draft-KV producer (last pipeline stage)."""

    def __init__(self, scheduler, algorithm):
        from sglang.srt.distributed.parallel_state import draft_pp_scope, get_pp_group
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        server_args = scheduler.server_args
        if not get_pp_group().is_last_rank:
            raise Weg2DraftRegistrationOffStage(
                "the draft-KV producer belongs on the LAST pipeline stage only "
                f"(this is pp_rank {get_pp_group().rank_in_group} of "
                f"{get_pp_group().world_size})."
            )
        self.scheduler = scheduler
        self.algorithm = algorithm
        self.stage = int(get_pp_group().rank_in_group)
        self.stages = int(get_pp_group().world_size)
        self._chunks = 0
        self._rows = 0
        self._published = 0
        self._peak_mib = 0.0
        self._free_before_mib = _cuda_free_mib()
        # allocator view before the build: the NVML delta minus the allocator's
        # reserved delta is context growth (kernel modules, cuBLAS/JIT
        # workspaces) -- the fourth W11b instrument, measured not guessed.
        try:
            self._reserved_before_mib = torch.cuda.memory_reserved() / float(2**20)
        except Exception:  # noqa: BLE001 -- no CUDA context: not measured
            self._reserved_before_mib = -1.0
        self.context_growth_mib = 0.0
        # The fields Scheduler.maybe_init_draft_worker prints for every producer.
        self.resident_mib = -1.0
        self.nvml_delta_mib = -1.0
        self.head_released_mib = 0.0
        self.head_deferred = False
        self.embed_dtype = "n/a"
        self.ring_slots = ring_capacity(server_args)
        draft_args = _draft_server_args(server_args)
        t0 = time.monotonic()
        with draft_pp_scope():
            self.draft_worker = DFlashWorkerV2(
                server_args=draft_args,
                gpu_id=scheduler.ps.gpu_id,
                tp_rank=scheduler.ps.tp_rank,
                dp_rank=scheduler.ps.dp_rank,
                moe_ep_rank=scheduler.ps.moe_ep_rank,
                attn_cp_rank=scheduler.ps.attn_cp_rank,
                moe_dp_rank=scheduler.ps.moe_dp_rank,
                nccl_port=scheduler.nccl_port,
                target_worker=scheduler.tp_worker,
            )
        if not getattr(self.draft_worker, "draft_kv_only", False):
            raise DFlashDraftKvProduceError(
                "DFlashWorkerV2 was built without draft_kv_only although "
                "--speculative-draft-kv-only is set; the producer form did not "
                "reach the worker"
            )
        self.draft_worker.producer_ring_slots = self.ring_slots
        self.draft_runner = self.draft_worker.draft_model_runner
        self.build_s = time.monotonic() - t0

    # -- the producer contract the scheduler drives ---------------------------
    def load_resident_embedding(self, model_path: str) -> float:
        """A DFlash draft borrows the target's embedding only to embed the
        decode block; the producer embeds nothing. Nothing to load -- but the
        launcher's W11/W11b accounting reads three instruments off the armed
        line, the same three DraftKvProducer reports: the live bytes of the
        draft (resident_mib), the NVML free delta across the build
        (nvml_delta_mib) and the released head (0 here). A -1 delta is
        'not measured' and refuses the boot (xsn253, W11b)."""
        torch.cuda.empty_cache()
        after = _cuda_free_mib()
        if after >= 0 and self._free_before_mib >= 0:
            self.nvml_delta_mib = self._free_before_mib - after
        self.resident_mib = _live_weight_mib(self.draft_runner.model)
        try:
            reserved_after = torch.cuda.memory_reserved() / float(2**20)
        except Exception:  # noqa: BLE001
            reserved_after = -1.0
        if (
            self.nvml_delta_mib >= 0
            and reserved_after >= 0
            and self._reserved_before_mib >= 0
        ):
            allocator_delta = reserved_after - self._reserved_before_mib
            self.context_growth_mib = max(0.0, self.nvml_delta_mib - allocator_delta)
        return 0.0

    def alloc_memory_pool(self, **kw):
        from sglang.srt.distributed.parallel_state import draft_pp_scope

        with draft_pp_scope():
            self.draft_worker.alloc_memory_pool(**kw)
        pool = self.draft_runner.token_to_kv_pool
        size = int(getattr(pool, "size", -1))
        if size < self.ring_slots:
            raise DFlashDraftKvProduceError(
                f"the draft chunk ring holds {size} slot(s), fewer than the "
                f"{self.ring_slots} a prefill batch may carry"
            )
        # The cache controller must never treat the ring as a mirror of the
        # target pool: row-addressed backup/load are meaningless on it.
        pool.weg2_direct_publish = True

    def init_attention_backends(self):
        from sglang.srt.distributed.parallel_state import draft_pp_scope

        with draft_pp_scope():
            self.draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        from sglang.srt.distributed.parallel_state import draft_pp_scope

        with draft_pp_scope():
            self.draft_worker.init_cuda_graphs()

    # -- the per-chunk work --------------------------------------------------
    def _cache_controller(self):
        cc = getattr(getattr(self.scheduler, "tree_cache", None), "cache_controller", None)
        if cc is None or not getattr(cc, "has_draft", False):
            raise DFlashDraftKvProduceError(
                "no HiCache cache controller with a registered draft pool on "
                "this stage; the DFlash producer has nowhere to publish "
                "(never silently skipped)"
            )
        return cc

    def produce(self, batch, hidden_states, next_token_ids) -> dict:
        """Materialise this chunk's draft rows into the ring and publish them
        hash-keyed into the draft arena. Returns the L1 terms."""
        from sglang.srt.distributed.parallel_state import draft_pp_scope
        from sglang.srt.speculative.dflash_worker_v2 import compute_position

        rows = int(batch.extend_num_tokens or 0)
        if rows <= 0:
            return {"rows": 0, "ms": 0.0, "peak_mib": 0.0, "published": 0}
        if rows > self.ring_slots:
            raise DFlashDraftKvProduceError(
                f"prefill batch carries {rows} token(s), more than the "
                f"{self.ring_slots}-slot chunk ring"
            )
        if hidden_states is None or int(hidden_states.shape[0]) != rows:
            raise DFlashDraftKvProduceError(
                "the target chunk's aux hidden rows do not match the batch: "
                f"{None if hidden_states is None else tuple(hidden_states.shape)} "
                f"vs {rows} token(s)"
            )
        device = hidden_states.device
        dev = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(dev)
        before = torch.cuda.memory_allocated(dev)
        t0 = time.monotonic()
        ctx_lens = torch.tensor(batch.extend_lens, dtype=torch.int32, device=device)
        prefix_lens = torch.tensor(batch.prefix_lens, dtype=torch.int32, device=device)
        positions, _ = compute_position(
            self.scheduler.tp_worker.model_runner.server_args.attention_backend,
            prefix_lens,
            ctx_lens,
            rows,
        )
        ring_locs = torch.arange(rows, dtype=torch.int64, device=device)
        with draft_pp_scope(), torch.inference_mode():
            self.draft_worker._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=hidden_states,
                cache_loc=ring_locs,
                positions=positions,
            )
        hashes = batch_page_hashes(batch)
        if len(hashes) != rows:
            raise DFlashDraftKvProduceError(
                f"{len(hashes)} page hash(es) for {rows} draft row(s)"
            )
        published = int(
            self._cache_controller().publish_draft_rows_direct(
                hashes, self.draft_runner.token_to_kv_pool, ring_locs
            )
        )
        ms = (time.monotonic() - t0) * 1000.0
        peak = (torch.cuda.max_memory_allocated(dev) - before) / float(2**20)
        self._chunks += 1
        self._rows += rows
        self._published += published
        self._peak_mib = max(self._peak_mib, peak)
        if published != rows:
            logger.warning(
                "WEG2 DFLASH DRAFT-KV-PRODUCE: %d of %d page(s) refused by the "
                "draft arena (cumulative published=%d rows=%d)",
                rows - published,
                rows,
                self._published,
                self._rows,
            )
        return {"rows": rows, "ms": ms, "peak_mib": peak, "published": published}
