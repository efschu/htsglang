from __future__ import annotations

import hashlib
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

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
    WEG2_SLEEP_MIN_RELEASED_FRACTION,
    WEG2_SLEEP_TAGS,
    Weg2WakeRefused,
    assert_backup_off_wake_refill_is_defined,
    assert_memory_saver_active,
    checkpoint_quantization,
    pcie_transfer_lock,
    resolve_pcie_lock_key,
    is_weights_family_tag,
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
    #: #1233: the pre-pause census of the sleep in progress (taken at the
    #: first release RPC of a sleep, graded at the RPC that completes the
    #: WEG2_SLEEP_TAGS population).  A slots dataclass: a field, not an
    #: ad-hoc attribute.
    weg2_sleep_before: Any = None

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

    @staticmethod
    def _weg2_rss_shmem_mib() -> float:
        """This process's RssShmem (/proc/self/status), the instrument that
        sees the torch_memory_saver cpu backup (cudaMallocHost pages are
        shared file-backed; RssAnon is blind to them -- boot weg2s1 N5)."""
        try:
            with open("/proc/self/status") as f:
                for ln in f:
                    if ln.startswith("RssShmem:"):
                        return int(ln.split()[1]) / 1024.0
        except OSError:
            pass
        return -1.0

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

        The two are separated STRUCTURALLY, not by clause order.  The earlier
        shape put `except Weg2PcieLockTimeout: raise` above a broad
        `except Exception -> log + yield`, so deleting those two lines silently
        downgraded every expiry to an unserialised overlap -- and the log line
        then blamed "card key unresolved" for a lock that was merely held (own
        mutant B-M2, which the whole suite survived green).  Here the only
        thing inside a broad handler is the key resolution; the lock itself is
        taken outside every `except`, so the refusal cannot be downgraded
        without deleting the `with` statement that the AST gates pin.
        """
        try:
            uuid_key = resolve_pcie_lock_key()
        except Exception as exc:
            logger.warning(
                "[weg2 pcie] no PCIe serialisation for %s (card key "
                "unresolved: %s) -- proceeding unserialised",
                label,
                exc,
            )
            yield
            return
        with pcie_transfer_lock(nvml_uuid=uuid_key, label=label):
            yield

    def _weg2_log_sleep_acceptance(
        self, before: Optional[Any] = None, tags: Optional[List[str]] = None
    ) -> None:
        """Design (S) 2.4 step 11, EXECUTED on the flip path, not declared.

        ``tags`` is the RESOLVED tag list of the request being graded -- the
        population the verdict is about.  It is printed on the line, so a boot
        postmortem can attribute a verdict to the request that produced it, and
        it decides whether the delta floor (a fraction of the WHOLE process
        residency) is the criterion in force at all: a #89 park's
        ``tags=["weights"]`` is a proper subset and is not gradeable by it.

        ``before`` is the PRE-PAUSE census this same RPC took, and it is what
        turns the reading into a verdict: without a criterion the census
        reported ``accepted=True`` for a rank still holding its entire 30 GiB
        shard, which is the silent-no-op condition the instrument exists to
        catch.  Graded on the SAME instrument at both ends, so the delta is a
        difference of two readings and not of two definitions.

        A ``before`` whose own NVML read failed carries ``proc_used_bytes``
        None; the census then finds no usable criterion and refuses, which is
        the honest outcome -- a blind pre-reading cannot grade an after-reading.
        """
        census = sleep_acceptance_census(
            tags=tags,
            before_bytes=None if before is None else before.proc_used_bytes,
            min_released_fraction=(
                None if before is None else WEG2_SLEEP_MIN_RELEASED_FRACTION
            ),
        )
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

        GATED on ``--enable-memory-saver``.  Without that flag every
        ``pause()`` was a no-op, so nothing was ever released and there is
        nothing to refill: a stock resume must be byte-for-byte the upstream
        path.  Ungated, the ORDINARY upstream RL configuration
        (``--enable-memory-saver`` alone, both backup flags default False --
        ``server_args.py:6636`` / ``:6640``) paid a full checkpoint reload on
        every wake (12.073/14.143/16.749 s measured on this rig, record
        (S) 2.6), and that reload is not side-effect-free: ``model_runner.py``
        ``:2866-2872`` rewrites ``self.load_config`` from a bare
        ``LoadConfig(load_format=...)`` built at ``:2825``, discarding the
        boot-time ``download_dir`` / ``model_loader_extra_config`` /
        ``ignore_patterns``, and records a ``model_runner.update_weights``
        override event for a weights update nobody requested.
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
        if not getattr(server_args, "enable_memory_saver", False):
            # Stock boot: pause() was `pass`, the pages were never released,
            # the content is whatever it always was.  Upstream path, untouched.
            return

        # The backup verdict is NOT `server_args.enable_weights_cpu_backup`
        # alone.  model_runner.py:2342-2344 builds the WEIGHTS region with
        #   enable_weights_cpu_backup or (is_draft_worker and
        #   enable_draft_weights_cpu_backup)
        # so the draft shard in this process can be cpu-backed while the main
        # shard is not.  The main shard is the binding term for "is a reload
        # needed at all" (the draft's expression contains the main's), but a
        # process whose two shards disagree cannot be served by ONE
        # `update_weights_from_disk`: that call refills the main runner and
        # then hands the SAME request -- and therefore the MAIN model_path --
        # to the draft runner (weight_updater.py:120-121,
        # eagle_worker_v2.py:3104-3107, multi_layer_eagle_worker_v2.py:1352-1362).
        # Shards never disagree: STOP, never compensate.
        main_carried = bool(getattr(server_args, "enable_weights_cpu_backup", False))
        draft_carried = main_carried or bool(
            getattr(server_args, "enable_draft_weights_cpu_backup", False)
        )
        if self.draft_worker is not None and draft_carried != main_carried:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: --enable-draft-weights-cpu-backup is set "
                "without --enable-weights-cpu-backup, so the draft shard's "
                "bytes were carried by the TMS restore and the main shard's "
                "were not (model_runner.py:2342-2344). One "
                "update_weights_from_disk serves both shards with one "
                "model_path, so there is no arrangement that refills the main "
                "shard without also pushing the main checkpoint through the "
                "draft runner. Launch this group with both flags or neither."
            )
        if self.draft_worker is not None:
            draft_path = getattr(server_args, "speculative_draft_model_path", None)
            if draft_path is not None and draft_path != server_args.model_path:
                raise Weg2WakeRefused(
                    "W4 Weg2WakeRefused: the draft worker loads from "
                    f"{draft_path!r}, not from {server_args.model_path!r}, and "
                    "the backup-OFF wake has exactly one model_path to give. "
                    "Refilling would push the main checkpoint through the "
                    "draft runner. V1 runs MTP out of the main checkpoint "
                    "(record 1b round-2 Q2, option (ii)); a separate draft "
                    "checkpoint needs its own wake leg, which S1 does not build."
                )
        if main_carried:
            return

        # W4, BEFORE anything is locked, entered or mutated: on a quantized
        # checkpoint the refill below is not a defined operation, because
        # load_weights_and_postprocess writes into parameters an earlier
        # process_weights_after_loading already replaced.  One definition,
        # shared with the launch arm (scheduler.py), so the boot refuses before
        # the first request and this call is the backstop for a process whose
        # launch check could not read the model config.
        assert_backup_off_wake_refill_is_defined(
            quantization=checkpoint_quantization(
                getattr(
                    getattr(self.tp_worker, "model_runner", None),
                    "model_config",
                    None,
                ),
                server_args,
            ),
            context="backup-OFF wake refill",
        )

        # flush_cache MUST stay False here: at this point in the resume the KV
        # tag is still paused, and flush_cache() -> MambaPool.reset_state()
        # would write into unmapped pages -- the CAMPAIGN (a) fault, mirrored
        # onto the wake path.  torch_empty_cache likewise: the KV pool is about
        # to be recommitted.
        #
        # The PCIe lock is taken HERE, a second time, and this is the take that
        # matters in the V1 arm: with the cpu backup OFF,
        # `resume(GPU_MEMORY_TYPE_WEIGHTS)` is a pure VMM recommit that moves
        # no bytes, and the entire 12-17 s host->device refill is this call.
        # Locking only the resume would leave the configured arm unserialised
        # against a co-located sibling's sleep-D2H (spec (S) 2.7).  Safe to take
        # after the caller's barrier(tp_cpu_group): update_weights_from_disk
        # holds no collective (weight_updater.py:115-128 -> tp_worker.py:103-109
        # -> model_runner.update_weights_from_disk, none of them collective).
        # THE REGION IS THE POINT OF THIS BLOCK, not decoration.  The refill
        # ALLOCATES: load_weights_and_postprocess ends in
        # `quant_method.process_weights_after_loading(module)`, which for every
        # repacking scheme replaces the parameter with a FRESH device
        # allocation (loader.py:921-931).  An allocation made with no region
        # active is not under the TMS weights tag, so after the first such wake
        # the live weights are no longer what `pause(GPU_MEMORY_TYPE_WEIGHTS)`
        # releases: the next sleep unmaps a region the model no longer points
        # at, the shard stays resident, and the RPC returns success -- the
        # silent-no-op class W12 and the sleep census exist to catch, re-entered
        # one level down and invisible until the SECOND sleep.
        #
        # Same region, same tag and the same `enable_cpu_backup` expression the
        # boot load uses (model_runner.py:2342-2348), so the repacked
        # parameters land exactly where the boot's did.  `enable_cpu_backup` is
        # False by the two guards above: `main_carried` is False (we returned
        # otherwise) and `draft_carried != main_carried` already raised, so
        # model_runner's `enable_weights_cpu_backup or (is_draft_worker and
        # enable_draft_weights_cpu_backup)` is False for BOTH shards here.
        #
        # Region outside, PCIe lock inside: the region must cover every
        # allocation the loader makes, the lock only the link.
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=False,
        ):
            with self._weg2_pcie_lock("wake-H2D weights reload"):
                try:
                    out = self.update_weights_from_disk(
                        UpdateWeightFromDiskReqInput(
                            model_path=server_args.model_path,
                            load_format=getattr(server_args, "load_format", None),
                            flush_cache=False,
                            torch_empty_cache=False,
                        )
                    )
                except Weg2WakeRefused:
                    raise
                except Exception as exc:
                    # model_runner.update_weights_from_disk catches the FIRST
                    # load failure and then re-runs `model_load_weights`
                    # OUTSIDE any try as its rollback (model_runner.py:2857-
                    # 2865), so a raw exception reaches here instead of the
                    # (False, message) tuple.  A wake leg that ends in an
                    # unnamed RuntimeError is the same undefined state as one
                    # that returns success=False; name it identically.
                    raise Weg2WakeRefused(
                        "W4 Weg2WakeRefused: the backup-OFF wake raised while "
                        f"refilling the weights from {server_args.model_path!r}: "
                        f"{type(exc).__name__}: {exc}. The VMM pages are "
                        "committed but their content is undefined; this group "
                        "is fatal."
                    ) from exc
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
        #
        # GATED ON THE REQUEST SHAPE, not on --enable-memory-saver.  This RPC
        # is SHARED with the fork's #89 hibernate: /hibernate sets
        # destination="disk", tags=["weights"] and posts it
        # (http_server.py:2034-2036), and hibernate is not gated on the memory
        # saver (server_args.py:18214-18222 requires only --hibernate-dir).
        # Refusing there deleted the feature on every stock engine stop: the
        # raise fired before _hibernate_park_weights ever ran, the dispatcher
        # does not catch it (scheduler.py:2963), so it reached
        # parent_process.send_signal(SIGQUIT), and the registry's Class-1
        # adapter swallows the resulting HTTP error
        # (registry/adapters/class1_srt.py:401-408) -- a silent death.
        #
        # Gating on the FLAG instead was the over-correction: it made the one
        # hazard (S) 2.1 names -- "a launcher edit that drops the flag makes
        # every sleep a no-op that returns success" -- unreachable, i.e. spec
        # (S) 10 S1 mutant M1 unsatisfiable, because the launch arm is gated on
        # the same flag and therefore also silent.  The request object already
        # carries the discriminator the two cases actually differ by
        # (io_struct.py:1964 `destination: Optional[str] = None`), and the
        # weights block below already branches on it.
        #
        # So: destination="disk" is the #89 park and is exempt; every OTHER
        # release is a genuine sleep, and a genuine sleep on a no-op adapter
        # frees nothing while returning success.  Upstream only warns about
        # that (check_validity, never called from this path); the fork's own
        # launcher comment states the same fact and calls the opt-out valid
        # "only for an engine that will never leave HOT"
        # (registry/adapters/class1_srt.py:318-324).  Refusing makes an
        # already-broken call loud instead of silent; it is a DELIBERATE
        # narrowing of stock behaviour on this endpoint, recorded as such.
        # Undecidable (no server_args reachable) -> refuse: a Weg-2 group
        # always has one, and a sleep whose gate cannot be read is not a sleep.
        server_args = self._weg2_server_args()
        weg2_memory_saver_on = server_args is None or bool(
            getattr(server_args, "enable_memory_saver", False)
        )
        if weg2_memory_saver_on or getattr(recv_req, "destination", None) != "disk":
            assert_memory_saver_active(self.memory_saver_adapter, context="first sleep")

        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        # #1233 one-backup flip: the weights are a FAMILY of tags (the base
        # GPU_MEMORY_TYPE_WEIGHTS plus weights_<k> per layer chunk, see
        # weg2_memory_saver.weights_family_tags) and one sleep may arrive as
        # several RPCs, one tag each, interleaved by the front with the other
        # group's wake.  Upstream's own offload_tags set is the ledger of what
        # is paused: the FIRST family tag of a sleep exports the static state
        # (buffers are read while every page is still mapped), and the sleep
        # is graded when the whole WEG2_SLEEP_TAGS population is paused.
        weights_tags = [t for t in tags if is_weights_family_tag(t)]
        family_paused_before = any(is_weights_family_tag(t) for t in self.offload_tags)
        sleep_begins = len(self.offload_tags) == 0

        for tag in tags:
            self.offload_tags.add(tag)

        # The PRE-PAUSE reading, on the same instrument the acceptance census
        # reads after.  Without it the census has no criterion and grades
        # nothing (see _weg2_log_sleep_acceptance).  Taken at the FIRST RPC of
        # a sleep (nothing paused yet), after the idle assert and before any
        # pause, so the difference is exactly what the whole sleep released.
        # Weg-2 path only: a stock release must stay upstream.
        if weg2_memory_saver_on and sleep_begins:
            self.weg2_sleep_before = sleep_acceptance_census(tags=tags)
        weg2_before_census = self.weg2_sleep_before
        t_rpc0 = time.perf_counter()

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
            # W25 Weg2DormantRefused (S1 boot killer K2): from this statement
            # on the req-index / KV / mamba pools are unmapped, so the
            # admission seams (Scheduler.handle_generate_request /
            # handle_embedding_request) must refuse by name instead of
            # walking into prepare_for_extend.  ONE flag on the object that
            # owns the pools; cleared after resume(KV_CACHE) below.
            if scheduler is not None:
                scheduler.weg2_dormant = True
                logger.info(
                    "WEG2-DORMANT set: kv_cache paused, admission seams refuse "
                    "with %s until resume_memory_occupation",
                    "W25 Weg2DormantRefused",
                )

        if weights_tags:
            # #89 hibernate: destination="disk" parks the FINAL post-transform
            # weights to hibernate_dir before the normal release/pause, so a
            # later boot can restore them fast (LoadFormat.HIBERNATE). The
            # default path (destination None/"gpu") is unchanged.
            if (
                GPU_MEMORY_TYPE_WEIGHTS in tags
                and getattr(recv_req, "destination", None) == "disk"
            ):
                self._hibernate_park_weights(recv_req)
            if not family_paused_before:
                self.stashed_model_static_state = _export_static_state(
                    self.tp_worker.model_runner.model
                )
            torch.distributed.barrier(self.tp_cpu_group)
            # The PCIe serialisation lock is taken AFTER the barrier and around
            # the D2H leg only: with --enable-weights-cpu-backup this pause
            # copies the whole shard to host (~2.1 s / 27 GiB measured), and a
            # co-located rank's wake-H2D on the same card would halve both.
            # Never held across a collective.
            #
            # SAME PROHIBITION AS THE kv_cache BLOCK ABOVE, and for the same
            # measured reason: nothing that touches the device may be appended
            # after this pause.  The model's parameters and buffers are
            # allocated inside region(GPU_MEMORY_TYPE_WEIGHTS)
            # (model_runner.py:2344-2348), so a later read of them -- a second
            # _export_static_state, a checksum, a .clone() -- is a read of
            # unmapped pages, which is the campaign (a) fault on the sibling
            # tag.  Pinned by test_weights_block_pause_is_the_last_statement.
            shm0 = self._weg2_rss_shmem_mib()
            with self._weg2_pcie_lock("sleep-D2H " + ",".join(weights_tags)):
                for tag in weights_tags:
                    self.memory_saver_adapter.pause(tag)
            # The MEASURED bytes of this tag set's host image (the ledger's
            # chunk term is image / N, derived; this is the instrument).
            logger.info(
                "WEG2-CHUNK-BYTES sleep tags=%s host_image_delta=%.0f MiB (RssShmem %.0f -> %.0f MiB, /proc/self/status)",
                weights_tags, self._weg2_rss_shmem_mib() - shm0, shm0, self._weg2_rss_shmem_mib(),
            )

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()
        sleep_complete = WEG2_SLEEP_TAGS.issubset(self.offload_tags)
        if weg2_memory_saver_on and not sleep_complete:
            # A chunk RPC of an interleaved sleep: the census (whole-process
            # residency) is graded once the population is complete, below.
            logger.info(
                "WEG2-SLEEP-CHUNK tags=%s paused in %.0f ms (offload_tags now %s)",
                tags,
                (time.perf_counter() - t_rpc0) * 1000,
                sorted(self.offload_tags),
            )
        if weg2_memory_saver_on and sleep_complete:
            # PROVENANCE, because the obvious justification for this call is
            # MEASURED FALSE on this rig: campaign (a) measured, WITHOUT it,
            # NVML free 1870.8 -> 30730.8 MiB and per-process 30,154 -> 1,294
            # MiB, identically in 9/9 steady cycles across two cold boots
            # (CAMPAIGN_a_0906.md §2 arm 1; carried into
            # WEG2_BUILD_DECISIONS_0906.md §1d).  TMS unmaps the TAGGED
            # segments' physical pages directly, so their release is already
            # visible to NVML and empty_cache() buys nothing there -- spec
            # (S) 2.4 step 10's premise is refuted for the tagged regions and
            # the record outranks the spec.  What it is kept for is the
            # UNTAGGED remainder torch still holds in its reserve, whose
            # benefit is unmeasured; the S1 slice boot re-checks it.  The S1
            # acceptance line "NVML free rises by weights+KV bytes" must NOT
            # be attributed to this call in the postmortem.
            #
            # Gated with the census below, for the same reason W12 is gated on
            # the request shape: a stock POST /hibernate must stay byte-for-
            # byte the upstream path, and an extra post-pause device call plus
            # two NVML reads on it is not that.
            torch.get_device_module().empty_cache()
            self._weg2_log_sleep_acceptance(
                weg2_before_census, sorted(self.offload_tags)
            )
            self.weg2_sleep_before = None

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        weights_tags = [t for t in tags if is_weights_family_tag(t)]
        if weights_tags:
            # Wake-H2D: with the cpu backup this recommit refills from the host
            # buffer (and, with the #1233 patched hook, frees that buffer).
            # Same lock, same reason as the sleep leg above.
            t_w0 = time.perf_counter()
            shm0 = self._weg2_rss_shmem_mib()
            with self._weg2_pcie_lock("wake-H2D " + ",".join(weights_tags)):
                for tag in weights_tags:
                    self.memory_saver_adapter.resume(tag)
            logger.info(
                "WEG2-CHUNK-BYTES wake tags=%s host_image_delta=%.0f MiB (RssShmem %.0f -> %.0f MiB, /proc/self/status; negative = the patched saver freed it)",
                weights_tags, self._weg2_rss_shmem_mib() - shm0, shm0, self._weg2_rss_shmem_mib(),
            )
            torch.distributed.barrier(self.tp_cpu_group)
            family_complete = not any(
                is_weights_family_tag(t) for t in self.offload_tags
            )
            logger.info(
                "WEG2-WAKE-CHUNK tags=%s resumed in %.0f ms (offload_tags now %s, weights family %s)",
                tags,
                (time.perf_counter() - t_w0) * 1000,
                sorted(self.offload_tags),
                "COMPLETE" if family_complete else "partial",
            )
            if family_complete:
                # Wake path (ii): without --enable-weights-cpu-backup the
                # recommit above restored PAGES, not CONTENT.  Refill from
                # disk BEFORE the static-state import, so the stash exported
                # from the live model at sleep stays the last writer for the
                # buffers.  Both only once the WHOLE family is mapped again.
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
                # W25: the pools are mapped again; the admission seams admit.
                scheduler.weg2_dormant = False
                logger.info(
                    "WEG2-DORMANT cleared: kv_cache resumed, admission seams admit"
                )
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
