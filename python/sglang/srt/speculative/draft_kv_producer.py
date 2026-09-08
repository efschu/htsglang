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
        # Section 6 / W11: the resident cost of this producer on the stage,
        # MEASURED, not budgeted. -1 until load_resident_embedding ran.
        #
        # FIX 3, boot weg2dk3: the instrument used to be the NVML-free delta
        # across the build, and it cannot measure what it claims. Under
        # --enable-memory-saver the draft weights are loaded inside
        # `memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS)`
        # (model_runner.py:2329), and that region IS
        # `torch.cuda.use_mem_pool(self._primary_mem_pool)`
        # (torch_memory_saver/entrypoint.py:89-91). A block freed back into
        # that MemPool is NOT handed to the driver by
        # `torch.cuda.empty_cache()` -- the saver map records exactly this
        # (weg2/maps/memsaver.md N4: "allocator cache OUTSIDE the saver's own
        # mempool"). So the NVML delta reports the BUILD (3998.0 MiB on
        # weg2dk3) no matter how the head's own lm_head is released, and W11
        # refused a boot whose residue was fine. The block is not lost: the
        # draft/target KV pools allocate from the SAME primary pool and reuse
        # it. `resident_mib` is therefore the draft model's own live weight
        # bytes (parameters + buffers, the shared target lm_head excluded --
        # the target already paid for it); the NVML delta stays beside it on
        # the L2 line as its own named term, never as the residue.
        self._free_before_mib = _cuda_free_mib()
        self.resident_mib = -1.0
        self.nvml_delta_mib = -1.0
        self.head_released_mib = 0.0
        self.embed_dtype = "?"

        # The draft build sees a pp_size-1 world: its one layer would fail
        # `make_layers`' `num_hidden_layers >= pp_size` assert under the
        # primary pp_size, and its embedding/norm/lm_head are built on
        # `is_first_rank`/`is_last_rank` of the group `get_pp_group()` returns.
        draft_args = _draft_server_args(server_args)
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
        """The producer captures NO draft/verify graphs -- through the tree's
        own refusal, not by skipping the call (FIX 3, boot weg2dk3).

        EAGER-ONLY IS A TERMINAL STATE THAT HAS TO BE CONSTRUCTED. It is not
        the absence of the graph path: ``ModelRunner.init_cuda_graphs`` is
        what builds ``eager_runner`` (model_runner.py:1484-1499
        ``_install_eager_only_runners``), and the same mistake is already
        catalogued in that file at :1528-1540 (#656 spec item 8, "the first
        attempt skipped init_cuda_graphs wholesale from the draft worker and
        all three ranks died at the first draft extend"). This method used to
        ``return None``; on boot weg2dk3 the first prefill chunk killed all
        three P ranks with ``AttributeError: 'ModelRunner' object has no
        attribute 'eager_runner'`` at ``_draft_extend_for_prefill`` ->
        ``_forward_raw``, and DRAFT-KV-PRODUCE ran 0 times.

        ``draft_args`` carries ``disable_draft_cuda_graph=True``
        (``_draft_server_args``), so the call below enters exactly the
        upstream refusal -- ``EagleDraftWorker.init_cuda_graphs``
        (eagle_worker_v2.py:515-540) skips the two captures, and
        ``ModelRunner.init_cuda_graphs`` installs the eager-only terminal
        state. No fork twin of that state is built here.
        """
        from sglang.srt.distributed.parallel_state import draft_pp_scope

        with draft_pp_scope():
            self.draft_worker.init_cuda_graphs()

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
        # Placement A pays the embedding ONCE: the checkpoint rows go INTO
        # the tensors the MTP build materialised (`named_parameters` of the
        # built module; no second table). A checkpoint that does not match
        # the built parameters is refused by name -- int8 codes cast into a
        # bf16 table without their scale, or a scale with no parameter to
        # land in, would both load "successfully" as token soup.
        params = dict(embed.named_parameters())
        wanted = {f"embed_tokens.{n}" for n in params}
        loaded = set()
        for name, tensor in _iter_checkpoint_tensors(model_path, "embed_tokens."):
            leaf = name[name.rindex("embed_tokens.") :]
            if leaf not in wanted:
                raise RuntimeError(
                    f"draft-KV producer: checkpoint tensor {name} has no parameter "
                    f"on the built embedding (built: {sorted(wanted)}); a "
                    f"{leaf} with nowhere to go means the built table and the "
                    "checkpoint disagree on the vocab quantization (#727)."
                )
            param = params[leaf[len("embed_tokens.") :]]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                if tensor.dtype != param.dtype:
                    raise RuntimeError(
                        f"draft-KV producer: checkpoint tensor {name} is "
                        f"{tensor.dtype} but the built embedding parameter is "
                        f"{param.dtype} and has no weight_loader -- refusing the "
                        "silent dtype cast (int8 codes without their scale)."
                    )
                param.data.copy_(tensor)
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
        # The head's OWN lm_head: `Qwen3_5ForCausalLMMTP.__init__` builds it
        # on `is_last_rank` (true inside the scope) and never loads it; the
        # target's module is shared instead -- which the draft forward needs
        # anyway, because `_draft_extend_for_prefill` goes through the
        # logits processor before C21 returns. Boot weg2dk2 measured the
        # build at 3994 MiB (mtp + int8 embed + this bf16 [vocab x hidden]
        # table, 2426 MiB).
        #
        # FIX 3: the release DELETES THE PARAMETERS, it does not merely swap
        # the module. That is the upstream form -- `set_embed_and_head`
        # (qwen3_5_mtp.py:185-193 and qwen3_vl.py:1298-1310) frees by
        # `del self.lm_head.weight`, precisely because dropping a module only
        # frees when nobody else still holds it, and a module the loader,
        # a quant method or a backup path still references keeps its whole
        # table alive with no error anywhere. Boot weg2dk3 shipped the swap
        # alone and measured a residue equal to the build.
        target_model = self.draft_worker.target_worker.model_runner.model
        head = getattr(target_model, "lm_head", None)
        own_head = getattr(draft_model, "lm_head", None)
        if head is not None and hasattr(draft_model, "set_lm_head_from_target"):
            draft_model.set_lm_head_from_target(head)
        if (
            own_head is not None
            and own_head is not embed
            and own_head is not getattr(draft_model, "lm_head", None)
        ):
            # `own_head is not embed` covers tie_word_embeddings: there the
            # head IS the resident embedding this method just loaded, and
            # gutting it would hand the producer token soup.
            self.head_released_mib = _drop_parameters(own_head)
            del own_head
        torch.cuda.empty_cache()
        mib = sum(p.numel() * p.element_size() for p in params.values()) / float(2**20)
        self.embed_dtype = str(params["weight"].dtype) if "weight" in params else "?"
        after = _cuda_free_mib()
        if after >= 0 and self._free_before_mib >= 0:
            self.nvml_delta_mib = self._free_before_mib - after
        # `shared` is the TARGET's head and only ever that: under
        # tie_word_embeddings `lm_head` is this producer's own resident
        # embedding, and excluding it would under-report the residue W11
        # grades by the whole vocab table.
        now_head = getattr(draft_model, "lm_head", None)
        self.resident_mib = _live_weight_mib(
            draft_model, shared=(now_head if (head is not None and now_head is head) else None)
        )
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


def _draft_server_args(server_args):
    """The producer's own ServerArgs copy.

    The draft build sees a pp_size-1 world: its one layer would fail
    `make_layers`' `num_hidden_layers >= pp_size` assert under the primary
    pp_size, and its embedding/norm/lm_head are built on
    `is_first_rank`/`is_last_rank` of the group `get_pp_group()` returns.

    `disable_draft_cuda_graph` is part of that copy and not of the launcher's
    argv: it is a property of THIS handle (a producer runs one eager draft
    extend per target chunk and never a draft round), not of the boot, and
    the target's own graph settings must stay untouched. It is the flag
    `should_capture_draft_graphs` (base_spec_worker.py:79-122) reads, so the
    eager-only terminal state is installed by the upstream refusal inside
    `ModelRunner.init_cuda_graphs` -- see `DraftKvProducer.init_cuda_graphs`.
    """
    draft_args = copy.deepcopy(server_args)
    draft_args.override(
        "weg2.draft_kv_producer",
        pp_size=1,
        pp_stage_ratio=None,
        pp_attn_stage_ratio=None,
        pp_layer_ratio=None,
        skip_tokenizer_init=True,
        disable_draft_cuda_graph=True,
    )
    return draft_args


def _drop_parameters(module) -> float:
    """Delete every parameter of ``module`` (recursively) and return the MiB
    that were live in them. The upstream release form (`del module.weight`),
    generalised over quantized heads that carry a scale beside the codes."""
    freed = 0
    for sub in module.modules():
        for pname in list(sub._parameters.keys()):
            p = sub._parameters[pname]
            if p is not None:
                freed += p.numel() * p.element_size()
            delattr(sub, pname)
    return freed / float(2**20)


def _live_weight_mib(model, shared=None) -> float:
    """The model's own live weight bytes: parameters + buffers, each storage
    counted once, everything reachable from ``shared`` (a module owned by
    another model, e.g. the target's lm_head) excluded because that model
    already paid for it. This is the residue W11 grades -- see the note in
    `DraftKvProducer.__init__` for why the NVML free delta cannot be."""
    foreign = set()
    if shared is not None:
        for t in list(shared.parameters()) + list(shared.buffers()):
            foreign.add(t.data_ptr())
    seen = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        ptr = t.data_ptr()
        if ptr in foreign or ptr in seen:
            continue
        seen.add(ptr)
        total += t.numel() * t.element_size()
    return total / float(2**20)


def _cuda_free_mib() -> float:
    """NVML-visible free MiB of this process's device after the caching
    allocator handed its free blocks back; -1 without CUDA."""
    if not torch.cuda.is_available():
        return -1.0
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info(torch.cuda.current_device())
    return free / float(2**20)


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
