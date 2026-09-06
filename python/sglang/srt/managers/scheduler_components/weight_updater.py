from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.managers.weg2_memory_saver import (
    Weg2PcieLockTimeout,
    Weg2WakeRefused,
    assert_memory_saver_active,
    pcie_transfer_lock,
    sleep_acceptance_census,
)

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            success, message = self.tp_worker.update_weights_from_disk(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_disk(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            success, message = self.tp_worker.update_weights_from_distributed(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = worker.update_weights_from_tensor(recv_req)
            if success:
                self.flush_cache_after_weight_update(recv_req)
            else:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    # ------------------------------------------------------------------
    # Weg-2 slice S1 helpers.  See srt/managers/weg2_memory_saver.py for why
    # each exists and what upstream mechanism it reuses.
    # ------------------------------------------------------------------

    def _weg2_server_args(self):
        return getattr(self.scheduler, "server_args", None)

    @contextmanager
    def _weg2_pcie_lock(self, label: str) -> Iterator[None]:
        """Serialise this card's host<->device transfer against its sibling.

        Two failure modes, kept apart on purpose:

        * the card's NVML UUID cannot be resolved -- there is no key to
          serialise on.  Reported and the transfer proceeds unserialised: this
          lock is a throughput guard (an overlap halves both legs on the
          x4-linked 3080), and the correctness guards on this path are W12, W4
          and the barriers, not this lock.
        * the lock is still held at the deadline -- that is a bounded wait
          whose expiry is a named refusal (``Weg2PcieLockTimeout``) and it
          PROPAGATES.  Never a longer wait, never a silent overlap.
        """
        try:
            lock = pcie_transfer_lock(label=label)
            lock.__enter__()
        except Weg2PcieLockTimeout:
            # A bounded wait that expired is a refusal, not a reason to
            # overlap the link.  It propagates to the caller of the RPC.
            raise
        except Exception as exc:
            logger.warning(
                "[weg2 pcie] no PCIe serialisation for %s (card key "
                "unresolved): %s -- proceeding unserialised",
                label,
                exc,
            )
            yield
            return
        try:
            yield
        finally:
            lock.__exit__(None, None, None)

    def _weg2_log_sleep_acceptance(self) -> None:
        """Design (S) 2.4 step 11, EXECUTED on the flip path, not declared."""
        census = sleep_acceptance_census()
        if census.accepted:
            logger.info("%s", census.format_line())
        else:
            logger.warning("%s", census.format_line())

    def _weg2_wake_reload_weights(self) -> None:
        """The backup-OFF half of the wake, behind the per-group launcher flag.

        With ``--enable-weights-cpu-backup`` the TMS restore already carried
        the bytes (measured 2.08 s / 27 GiB, campaign (a)) and this is a no-op.
        Without it, ``resume(GPU_MEMORY_TYPE_WEIGHTS)`` recommitted VMM pages
        whose CONTENT IS UNDEFINED, so the weights are refilled through the
        upstream ``update_weights_from_disk`` endpoint from the page-cached
        checkpoint.  No fork loader: the upstream path is the path.
        """
        server_args = self._weg2_server_args()
        if server_args is None:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the weight wake path is undecidable -- no "
                "server_args reachable from the weight updater, so whether the "
                "cpu backup carried the bytes cannot be determined. VRAM has "
                "already been mutated; refusing rather than serving undefined "
                "weights."
            )
        if getattr(server_args, "enable_weights_cpu_backup", False):
            return

        # flush_cache MUST stay False here: at this point in the resume the KV
        # tag is still paused, and flush_cache() -> MambaPool.reset_state()
        # would write into unmapped pages -- the CAMPAIGN (a) fault, mirrored
        # onto the wake path.  torch_empty_cache likewise: the KV pool is about
        # to be recommitted.
        out = self.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(
                model_path=server_args.model_path,
                load_format=getattr(server_args, "load_format", None),
                flush_cache=False,
                torch_empty_cache=False,
            )
        )
        if not getattr(out, "success", False):
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the backup-OFF wake could not refill the "
                f"weights from {server_args.model_path!r}: "
                f"{getattr(out, 'message', '')!r}. The VMM pages are committed "
                "but their content is undefined; this group is fatal."
            )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        # W12 Weg2MemorySaverInactive, BEFORE anything is mutated: with the
        # no-op adapter every pause() below is `pass` and this RPC returns
        # success having released nothing.  A refusal that has already paused
        # a tag is not a refusal, so this is the first statement.
        assert_memory_saver_active(self.memory_saver_adapter, context="first sleep")

        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.release_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.release_memory_occupation()
            # CAMPAIGN (a) MUST_FIX, measured 2026-09-06 on Qwen3.8-27B-INT8
            # (hybrid mamba/GDN), boot a3: flush_cache() must run BEFORE the
            # pause, not after.  flush_cache() -> HybridReqToTokenPool.clear()
            # -> MambaPool.reset_state() ZEROES the mamba conv/temporal tensors
            # and then synchronizes; those tensors are allocated under
            # region(GPU_MEMORY_TYPE_KV_CACHE) (memory_pool.py:1017), so the
            # pause has already unmapped their pages and the zero_() is a write
            # to unmapped memory -> "CUDA error: an illegal memory access was
            # encountered" inside MambaPool._sync_device, killing the scheduler
            # on the FIRST sleep.  Flushing first is strictly more correct: the
            # reset runs while the pages are still mapped, and the pause then
            # releases an already-quiesced pool.  Nothing that touches the
            # device may be appended after the pause in this block.
            self.flush_cache()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            # #89 hibernate: destination="disk" parks the FINAL post-transform
            # weights to hibernate_dir before the normal release/pause, so a
            # later boot can restore them fast (LoadFormat.HIBERNATE). The
            # default path (destination None/"gpu") is unchanged.
            if getattr(recv_req, "destination", None) == "disk":
                self._hibernate_park_weights(recv_req)
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            # The PCIe serialisation lock is taken AFTER the barrier and around
            # the D2H leg only: with --enable-weights-cpu-backup this pause
            # copies the whole shard to host (~2.1 s / 27 GiB measured), and a
            # co-located rank's wake-H2D on the same card would halve both.
            # Never held across a collective.
            with self._weg2_pcie_lock("sleep-D2H weights"):
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()
        # Upstream's release does not empty the allocator cache, so the freed
        # pages sit in torch's reserve and NVML free does not move -- the
        # dormant group would look resident to every host-side instrument.
        torch.get_device_module().empty_cache()
        self._weg2_log_sleep_acceptance()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            # Wake-H2D: with the cpu backup this recommit refills from the host
            # buffer.  Same lock, same reason as the sleep leg above.
            with self._weg2_pcie_lock("wake-H2D weights"):
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            # Wake path (ii): without --enable-weights-cpu-backup the recommit
            # above restored PAGES, not CONTENT.  Refill from disk BEFORE the
            # static-state import, so the stash exported from the live model at
            # sleep stays the last writer for the buffers.
            self._weg2_wake_reload_weights()
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None:
                if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                    for queue_name in (
                        "disagg_decode_transfer_queue",
                        "disagg_decode_prealloc_queue",
                    ):
                        queue = getattr(scheduler, queue_name, None)
                        if queue is not None:
                            queue.resume_memory_occupation()
                elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
                    if queue is not None:
                        queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


    def _hibernate_park_weights(self, recv_req):
        """#89: park this rank's FINAL post-transform weights to disk."""
        from sglang.srt.model_loader.hibernate import park_weights_to_disk

        model_runner = self.tp_worker.model_runner
        server_args = model_runner.server_args
        model = model_runner.model
        # #510 (audit #506 A2-F1): the request may narrow the directory but
        # never leave the configured root. Enforced here as well as in the HTTP
        # handler because this is the sink -- os.makedirs()/os.path.join() are
        # two calls further down -- and it is reachable from the Engine API
        # without passing through the route.
        from sglang.srt.utils.path_confinement import confine_to_root

        hibernate_dir = confine_to_root(
            getattr(recv_req, "hibernate_dir", None),
            server_args.hibernate_dir,
        )
        tp_rank = model_runner.tp_rank
        park_weights_to_disk(
            model=model,
            server_args=server_args,
            hibernate_dir=hibernate_dir,
            tp_rank=tp_rank,
            tp_cpu_group=self.tp_cpu_group,
            export_static_state=_export_static_state,
            tie_word_embeddings=bool(
                getattr(model, "_gguf_tie_word_embeddings", False)
            ),
            unquantized_prefixes=list(
                getattr(model, "_gguf_unquantized_prefixes", [])
            ),
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
