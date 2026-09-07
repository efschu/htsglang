# SPDX-License-Identifier: Apache-2.0
"""The draft-KV PRODUCER of Weg 2's prefill group (#1233, R1).

User order 2026-09-07: "der draft (mtp head aus dem originalmodell) muss
selbstverständlich auch im pp3 layout laufen". Group P (PP=3 prefill) runs
the checkpoint's own ``mtp.*`` head on its LAST attention stage after every
target chunk, so the draft KV of every prompt token exists and reaches the
HiCache store in the canonical draft page that group D (TP=3 + NEXTN) reads.

What this is NOT: a proposing drafter. There is no verify, no draft round,
no draft cuda graph, no proposal loop; ``EagleDraftWorker`` is used for the
ONE primitive that writes draft KV rows -- ``_draft_extend_for_prefill`` --
and returns right after the draft forward (``draft_kv_only``, C21). The
scheduler keeps ``spec_algorithm == NONE`` and ``draft_worker is None``, so
every spec-keyed branch of the PP loop takes the path it takes today; the
producer is a separate handle.

Placement A (Q4): the MTP head's ``embed_tokens`` is loaded RESIDENT on this
stage from the checkpoint (the target's own embedding on a non-first stage
is a ``PPMissingLayer``, which the head refuses by name, S3); ``lm_head`` is
shared from the target, which does exist on the last stage.
"""

from __future__ import annotations

import copy
import glob
import json
import logging
import os
import time


import torch

logger = logging.getLogger(__name__)


class Weg2DraftRegistrationOffStage(RuntimeError):
    """S4: the producer was asked for on a stage that is not the last one."""


class DraftKvProducer:
    """The producer handle: shaped like ``EAGLEWorkerV2`` where the scheduler
    and ``kv_cache_builder.get_draft_kv_pool`` reach into it
    (``.draft_worker.draft_runner.token_to_kv_pool``), and nothing more."""

    def __init__(self, scheduler, algorithm):
        from sglang.srt.distributed.parallel_state import draft_pp_scope, get_pp_group
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        server_args = scheduler.server_args
        if not get_pp_group().is_last_rank:
            raise Weg2DraftRegistrationOffStage(
                "the draft-KV producer belongs on the LAST pipeline stage only "
                f"(this is pp_rank {get_pp_group().rank_in_group} of "
                f"{get_pp_group().world_size})."
            )
        self.algorithm = algorithm
        self.stage = int(get_pp_group().rank_in_group)
        self.stages = int(get_pp_group().world_size)
        self._chunks = 0
        self._rows = 0
        self._peak_mib = 0.0

        # The draft build sees a pp_size-1 world: its one layer would fail
        # `make_layers`' `num_hidden_layers >= pp_size` assert under the
        # primary pp_size, and its embedding/norm/lm_head are built on
        # `is_first_rank`/`is_last_rank` of the group `get_pp_group()` returns.
        draft_args = copy.deepcopy(server_args)
        draft_args.override(
            "weg2.draft_kv_producer",
            pp_size=1,
            pp_stage_ratio=None,
            pp_attn_stage_ratio=None,
            pp_layer_ratio=None,
            skip_tokenizer_init=True,
        )
        t0 = time.monotonic()
        with draft_pp_scope():
            self.draft_worker = EagleDraftWorker(
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
        self.draft_worker.draft_kv_only = True
        self.draft_runner = self.draft_worker.draft_runner
        self.build_s = time.monotonic() - t0

    # -- the scheduler's three lifecycle hooks (mirror of the draft worker's) --

    def alloc_memory_pool(self, **kw):
        from sglang.srt.distributed.parallel_state import draft_pp_scope

        with draft_pp_scope():
            self.draft_worker.alloc_memory_pool(**kw)

    def init_attention_backends(self):
        from sglang.srt.distributed.parallel_state import draft_pp_scope

        with draft_pp_scope():
            self.draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        # Producer: no draft/verify graphs. The prefill draft-extend is eager.
        return None

    # -- placement A --------------------------------------------------------

    def load_resident_embedding(self, model_path: str) -> float:
        """Load the checkpoint's ``embed_tokens`` tensors into the draft's own
        embedding (INT8 via the same quant_config the target uses, #727) and
        share the target's ``lm_head``. Returns the embedding's MiB."""
        from sglang.srt.layers.utils import PPMissingLayer

        draft_model = self.draft_runner.model
        embed = getattr(getattr(draft_model, "model", None), "embed_tokens", None)
        if embed is None or isinstance(embed, PPMissingLayer):
            raise RuntimeError(
                "draft-KV producer: the draft model built no embed_tokens of its "
                "own (PPMissingLayer) -- the build did not run under draft_pp_scope."
            )
        params = dict(embed.named_parameters())
        wanted = {f"embed_tokens.{n}" for n in params}
        loaded = set()
        for name, tensor in _iter_checkpoint_tensors(model_path, "embed_tokens."):
            leaf = name[name.rindex("embed_tokens.") :]
            if leaf not in wanted:
                continue
            param = params[leaf[len("embed_tokens.") :]]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(tensor.to(param.dtype))
            else:
                loader(param, tensor)
            loaded.add(leaf)
        missing = wanted - loaded
        if missing:
            raise RuntimeError(
                f"draft-KV producer: embed_tokens parameter(s) {sorted(missing)} "
                f"not found in the checkpoint at {model_path} (placement A needs "
                "a resident embedding on the last stage)."
            )
        target_model = self.draft_worker.target_worker.model_runner.model
        head = getattr(target_model, "lm_head", None)
        if head is not None and hasattr(draft_model, "set_lm_head_from_target"):
            draft_model.set_lm_head_from_target(head)
        mib = sum(p.numel() * p.element_size() for p in params.values()) / float(2**20)
        return mib

    # -- the per-chunk primitive ---------------------------------------------

    def produce(self, batch, hidden_states, next_token_ids) -> dict:
        """Write the draft KV rows of this chunk (C7). Returns the L1 terms."""
        from sglang.srt.distributed.parallel_state import draft_pp_scope
        from sglang.srt.speculative.eagle_worker_v2 import (
            speculative_moe_a2a_backend_context,
            speculative_moe_backend_context,
        )

        rows = int(batch.extend_num_tokens or 0)
        dev = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(dev)
        before = torch.cuda.memory_allocated(dev)
        t0 = time.monotonic()
        with (
            draft_pp_scope(),
            self.draft_worker.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker._draft_extend_for_prefill(batch, hidden_states, next_token_ids)
        ms = (time.monotonic() - t0) * 1000.0
        peak = (torch.cuda.max_memory_allocated(dev) - before) / float(2**20)
        self._chunks += 1
        self._rows += rows
        self._peak_mib = max(self._peak_mib, peak)
        return {"rows": rows, "ms": ms, "peak_mib": peak}


def _iter_checkpoint_tensors(model_path: str, needle: str):
    """Yield ``(name, tensor)`` for every safetensors entry whose name carries
    ``needle`` -- read through the index when there is one, else every
    ``*.safetensors`` file. CPU tensors; the caller's loader moves them."""
    from safetensors import safe_open

    index = os.path.join(model_path, "model.safetensors.index.json")
    files = []
    if os.path.isfile(index):
        with open(index) as f:
            weight_map = json.load(f).get("weight_map", {})
        files = sorted({os.path.join(model_path, v) for k, v in weight_map.items() if needle in k})
    if not files:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if needle in name:
                    yield name, f.get_tensor(name)
