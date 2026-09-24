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
    DraftBuildMeter,
    Weg2DraftRegistrationOffStage,
    _draft_server_args,
    _live_weight_mib,
)

logger = logging.getLogger(__name__)

_HASH_POS_ATTR = "_weg2_dflash_hash_pos"
_HASH_LAST_ATTR = "_weg2_dflash_hash_last"


#: The rank-side switch of ``--dflash-produce-on-p`` (weg2 launcher, set on
#: group P through ``spec_form_env("P")``). See environ.py.
DFLASH_PRODUCE_ENV = "SGLANG_WEG2_DFLASH_PRODUCE"


def dflash_produce_on_p() -> bool:
    """Does group P's DFlash draft-KV producer COMPUTE (user decision
    2026-09-24: default off in the launcher)?

    False: the producer is still built -- its weights stay cold-resident on
    the last stage, so VRAM, the planner cut, the exchange census and the
    flip are unchanged -- but nothing runs on it: the scheduler's
    ``_draft_kv_producer_wants`` answers False (no FULL capture, no
    ``produce()``, no ``publish_draft_rows_direct``) and the target model
    runner arms no aux-hidden capture on any PP stage. Unset -> True, the
    producer form as before."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_DFLASH_PRODUCE.get())


def dflash_produce_off_line(*, where: str) -> str:
    """The one rank-side line naming the form (never silent)."""
    return (
        f"WEG2 DFLASH-PRODUCE-ON-P off ({DFLASH_PRODUCE_ENV}=0) at {where}: the "
        "DFlash draft stays cold-resident on group P's last stage (weights, "
        "chunk ring, graphs built as before) and computes nothing -- no aux "
        "capture, no produce(), no publish_draft_rows_direct; group D finds "
        "no draft pages and takes its cold path (user decision 2026-09-24)"
    )


class DFlashDraftKvProduceError(RuntimeError):
    """A chunk's draft rows could not be produced or published."""


def _token_stream(req) -> List[int]:
    """The token stream the target consumed for ``req``: origin prompt plus
    the output already filled. weg2xsn262 (17.09.): this fork's ``Req`` has
    no ``fill_ids`` attribute -- it keeps ``full_untruncated_fill_ids`` and
    answers ``get_fill_ids()`` (cut at ``extend_range.end``); PP2 died on the
    first P prefill with ``AttributeError: 'Req' object has no attribute
    'fill_ids'`` while the hermetic double carried exactly that attribute.
    Three sources, in order: ``get_fill_ids()``, a plain ``fill_ids``
    (upstream/doubles), ``origin_input_ids + output_ids``."""
    getter = getattr(req, "get_fill_ids", None)
    if callable(getter):
        try:
            return list(getter())
        except Exception:  # noqa: BLE001 -- extend_range unset: fall through
            pass
    direct = getattr(req, "fill_ids", None)
    if direct is not None:
        return list(direct)
    return list(getattr(req, "origin_input_ids", ()) or ()) + list(
        getattr(req, "output_ids", ()) or ())


def _key_units(tokens: List[int], bigram: bool):
    """The hash input for ``tokens`` in the radix cache's own key scheme.
    Unigram: the token list itself. Bigram (weg2xsn269, 18.09.): a
    ``RadixKey(is_bigram=True)`` over the raw tokens -- ``get_hash_str``
    reads its ``is_bigram`` flag and hashes the N-1 units (t_i, t_i+1)
    exactly as ``compute_node_hash_values`` does for a tree node."""
    if not bigram:
        return tokens
    from array import array

    from sglang.srt.mem_cache.radix_cache import RadixKey

    return RadixKey(array("q", tokens), None, is_bigram=True)


def chunk_page_hashes(req, prefix_len: int, extend_len: int, *,
                      bigram: bool = False) -> List[str]:
    """The canonical page hashes of ``req``'s positions
    ``[prefix_len, prefix_len + extend_len)``, continuing the chain cached
    on the request when the previous chunk ended exactly at ``prefix_len``.

    The token stream (``_token_stream``: ``get_fill_ids()`` on this fork's
    ``Req``) is what the radix keys hash.

    ``bigram`` (weg2xsn268/269, 18.09.): under SGLANG_HICACHE_BIGRAM_KEYS=1
    (every Weg 2 group, so that P's pages are readable by D) the tree keys
    page ``i`` by the bigram chain up to (t_i, t_i+1). The unigram chain this
    producer published under before was a DISJOINT key space: PP2 published
    4316 draft pages, D's draft L3 READ found 0 of them (xsn268, xsn269).
    In bigram form position ``i`` needs token ``i+1``; the chunk's last
    position has none when the chunk ends the filled stream, so that page is
    NOT produced (the tree cannot key it either) and the list is one short.
    """
    from sglang.srt.mem_cache.utils import get_hash_str

    if extend_len <= 0:
        return []
    tokens = _token_stream(req)
    end = prefix_len + extend_len
    if end > len(tokens):
        raise DFlashDraftKvProduceError(
            f"rid={getattr(req, 'rid', '?')}: chunk [{prefix_len}, {end}) "
            f"reaches past the {len(tokens)} filled token(s)"
        )
    # raw tokens the chain needs: one past the last position in bigram form
    raw_end = min(end + 1, len(tokens)) if bigram else end
    want = (raw_end - 1 - prefix_len) if bigram else extend_len
    if want <= 0:
        return []
    cached_pos = getattr(req, _HASH_POS_ATTR, None)
    cached_last = getattr(req, _HASH_LAST_ATTR, None)
    if prefix_len == 0:
        hashes = get_hash_str(_key_units(tokens[:raw_end], bigram), None, page_size=1)
    elif cached_pos == prefix_len and cached_last:
        hashes = get_hash_str(
            _key_units(tokens[prefix_len:raw_end], bigram), cached_last, page_size=1
        )
    else:
        # A prefix hit or a chunk boundary we did not see: rebuild the chain
        # from position 0 (same function, same result, only slower).
        hashes = get_hash_str(_key_units(tokens[:raw_end], bigram), None, page_size=1)[prefix_len:]
    if not isinstance(hashes, list) or len(hashes) != want:
        raise DFlashDraftKvProduceError(
            f"rid={getattr(req, 'rid', '?')}: expected {want} page hash(es) "
            f"for the chunk, got {len(hashes) if isinstance(hashes, list) else type(hashes).__name__}"
        )
    setattr(req, _HASH_POS_ATTR, prefix_len + want)
    setattr(req, _HASH_LAST_ATTR, hashes[-1])
    return hashes


def batch_page_hashes(batch, *, bigram: bool = False) -> List[str]:
    out: List[str] = []
    for req, pl, el in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
        out.extend(chunk_page_hashes(req, int(pl), int(el), bigram=bigram))
    return out


def _rows_to_publish(batch, ring_locs, *, bigram: bool):
    """Per request, the hashes of its chunk and the ring rows they key --
    in batch order, so that ``ring_locs`` (one row per extend token, in the
    batch's token order) is sliced per request. A request whose bigram
    chain is one short (see ``chunk_page_hashes``) contributes one row
    fewer; no row is ever published under the wrong request's key."""
    hashes: List[str] = []
    keep: List[int] = []
    base = 0
    for req, pl, el in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
        el = int(el)
        h = chunk_page_hashes(req, int(pl), el, bigram=bigram)
        hashes.extend(h)
        keep.extend(range(base, base + len(h)))
        base += el
    if len(keep) == int(ring_locs.numel()):
        return hashes, ring_locs
    import torch

    idx = torch.tensor(keep, dtype=torch.int64, device=ring_locs.device)
    return hashes, ring_locs[idx]


def tree_keys_are_bigram(scheduler) -> bool:
    """Whether this group's radix cache keys its pages by bigram (NEXTN/EAGLE
    form, or SGLANG_HICACHE_BIGRAM_KEYS=1 forcing it) -- read from the tree
    itself, the one carrier of that fact, never from the env."""
    tree = getattr(scheduler, "tree_cache", None)
    return bool(getattr(tree, "is_eagle", False))


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
        # #66 / weg2xsn417: the W11b build instruments, read by the SAME meter
        # the NEXTN producer uses (draft_kv_producer.DraftBuildMeter) -- the
        # scheduler's armed line prints them and the launcher's W11b adds them
        # up. This handle used to carry its own two-term copy (context growth
        # + allocator cache) of the pre-#66 formula; the armed line of the NF
        # line printed -1 for every #66 term, and W11b refused xsn417 with
        # 897.5 MiB 'unaccounted' -- the allocator cache the W8 draft load
        # leaves behind (xsn411, same build: allocator_cache_mib=891.8,
        # context growth 0.0, remainder 5.7 MiB live non-model bytes).
        self._build_meter = DraftBuildMeter()
        DraftBuildMeter.init_fields(self)
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
        launcher's W11/W11b accounting reads the build instruments off the
        armed line, measured by the one DraftBuildMeter both producers use:
        the live bytes of the draft (resident_mib, parameters + buffers --
        the draft's rotary cos/sin cache is a buffer, 128.2 MiB on
        Qwen3.8-27B-DFlash2), the NVML free delta across the build, the
        allocator cache it left (default_pool_inactive_mib, its tag-pool part
        tag_pool_inactive_mib), what grew outside torch, the live non-model
        bytes, and the card's free margin. The released head is 0 here. A -1
        term is 'not measured' and W11b refuses it (xsn253, xsn417).
        Measured BEFORE the chunk ring, the attention backend and the graphs
        of this handle exist (alloc_memory_pool / init_attention_backends /
        init_cuda_graphs run later), so none of those is in the build."""
        meter = getattr(self, "_build_meter", None) or DraftBuildMeter.unmeasured()
        meter.finish(self, resident_mib=lambda: _live_weight_mib(self.draft_runner.model))
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
        bigram = tree_keys_are_bigram(self.scheduler)
        hashes, locs = _rows_to_publish(batch, ring_locs, bigram=bigram)
        if len(hashes) != int(locs.numel()):
            raise DFlashDraftKvProduceError(
                f"{len(hashes)} page hash(es) for {int(locs.numel())} draft row(s)"
            )
        published = int(
            self._cache_controller().publish_draft_rows_direct(
                hashes, self.draft_runner.token_to_kv_pool, locs
            )
        ) if hashes else 0
        if len(hashes) != rows and self._chunks == 0:
            logger.info(
                "WEG2 DFLASH DRAFT-KV-PRODUCE keys=%s: %d of %d row(s) of this "
                "chunk carry a page key; a request's last filled position has "
                "no bigram partner yet, so its draft page is produced with the "
                "next chunk of that request (or never, at the prompt end -- the "
                "tree cannot key it either)",
                "bigram" if bigram else "unigram",
                len(hashes),
                rows,
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
