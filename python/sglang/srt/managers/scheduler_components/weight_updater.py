from __future__ import annotations

import hashlib
import functools
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

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
    Weg2FlipRankDisagree,
    Weg2VramCreditRefused,
    Weg2WakeRefused,
    assert_backup_off_wake_refill_is_defined,
    assert_memory_saver_active,
    checkpoint_quantization,
    pcie_transfer_lock,
    resolve_pcie_lock_key,
    is_weights_family_tag,
    sleep_acceptance_census,
    vram_credit,
    weg2_graph_tag_armed,
)

#: C21 / spec R13.  The group fence's budget, named so the ONE place that owns
#: it can be re-justified and so C14's credit wait can be bounded by the SAME
#: number instead of inventing a second one (spec section 10.9 forbids a new
#: timeout constant for the credit).
#:
#: PROVENANCE, and it is the re-justification R13 demands.  The 120 s was chosen
#: against "the largest single pause measured is ~1.5 s (3336 MiB D2H)".  That
#: basis DIED with C9: the fence now closes once per LEG, not once per tag, and
#: a leg is the whole weights family -- measured 11.4 s of D2H on this rig
#: (WEG2_FLIPCOST_SPEC_0907 section 2, from the per-tag byte table), against a
#: flip whose measured sleep leg was 14 481 ms on boot weg2dk5.  120 s is
#: therefore ~8x the measured whole-leg payload rather than ~80x a single tag,
#: which is still a fence budget and no longer a number justified by an
#: obsolete unit.  It is NOT a performance bound: a leg that needs more than
#: this is a wedged leg, and the barrier names the rank that did not join.
WEG2_GROUP_FENCE_BUDGET_S = 120.0

MIB_ = 1024 * 1024


def _weg2_group_stop_on_leg_failure(fn):
    """C15: a rank that could not finish its half of a leg VOTES, then re-raises.

    A DECORATOR rather than a try inside each body, for two reasons.  The
    exception may come from ANY statement of the leg -- the pause itself, the
    device credit, the static-state export -- and a group STOP that only covers
    the statements someone remembered to wrap is exactly the rank-local silent
    failure spec section 10.5 forbids.  And ``functools.wraps`` keeps the
    method's name and its ``__wrapped__``, so the AST and ``inspect.getsource``
    gates that pin the ORDER of statements inside these handlers keep reading
    the real body: a wrapper that made those gates read an empty shim would
    disarm them while looking green, which is the same class of defect.
    """

    @functools.wraps(fn)
    def wrapper(self, recv_req):
        try:
            return fn(self, recv_req)
        except Exception as exc:
            self._weg2_leg_failed(f"{fn.__name__} FAILED on this rank", exc)
            raise

    return wrapper

#: The ring's granule, spec C1 ("2 MiB granules") -- a fixed GEOMETRY of the
#: region, not a measured size, so it is written here rather than solved.  L1's
#: ``granules=`` is derived from it; when a ring is actually published the
#: saver's own ``ring_stats()['granule_bytes']`` is the authority and this value
#: is only the pre-ring reading.
TMS_RING_GRANULE_BYTES = 2 * 1024 * 1024

#: The POPULATION token every ``WEG2-FLIP-TAG`` line carries, read back by
#: ``ring_table.parse_group_log``.  The names are imported from the reader so
#: the emitter and the parser cannot spell them differently -- a mismatch would
#: read as "weights only" and silently size the ring from a lower bound.
from sglang.srt.weg2.ring_table import (  # noqa: E402
    TAG_POPULATION_ALL as WEG2_TAG_POPULATION_ALL,
    TAG_POPULATION_WEIGHTS as WEG2_TAG_POPULATION_WEIGHTS,
)

logger = logging.getLogger(__name__)


def _weg2_flip_index_of(epoch) -> int:
    """The FLIP half of ``weg2_memory_saver.credit_epoch``'s ``<boot>.<flip>``.

    #1273 S5b.  The shadow needs a leg index for its rotation and for the
    region's per-flip stamp, and this class holds no flip counter of its own --
    the front owns it and publishes it exactly here, on the request.  Parsing
    it back out is a READ of the front's number, not a second counter beside
    it; inventing one would be the cross-boot regression ``credit_epoch``'s own
    docstring describes, one seam over.

    ``-1`` when there is no epoch or its tail is not an integer, and ``-1`` is
    load-bearing: ``XchgRegion.begin_flip`` refuses an index that does not
    advance past ``flip_index`` (initialised to ``-1``), so an undated leg
    cannot stamp a region and the shadow reports ``reason=no-region`` instead
    of adopting some other flip's rows.
    """
    token = str(epoch or "")
    if "." not in token:
        return -1
    try:
        return int(token.rsplit(".", 1)[1])
    except (TypeError, ValueError):
        return -1



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
    #: C15: set by :meth:`_weg2_group_fence` when the fence itself raised, read
    #: and cleared by :meth:`_weg2_leg_failed`.  A FIELD, not an ad-hoc
    #: attribute -- this is a ``slots=True`` dataclass, and the comment above
    #: already learnt that lesson once.
    weg2_fence_raised: bool = False
    #: C16: this rank's card, resolved once.  ``"unset"`` is distinct from
    #: ``None``, which is the resolved answer "no card key" -- so an
    #: unresolvable card is not re-resolved (and re-logged) on every tag.
    weg2_card_uuid_cache: Any = "unset"

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
    def _weg2_pcie_lock(self, label: str, direction: Optional[str] = None) -> Iterator[None]:
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
        # C13: the leg's DIRECTION goes to the key's owner.  On a card whose
        # measured duplex ratio reaches R17's gate the key becomes
        # ``<uuid>.<d2h|h2d>`` and a sleep no longer excludes a co-located
        # wake; below the gate the key is unchanged and they still serialise.
        # This module states the direction, weg2_memory_saver decides what the
        # key does with it -- one fact, one owner.
        with pcie_transfer_lock(nvml_uuid=uuid_key, label=label, direction=direction):
            yield

    def _weg2_card_uuid(self) -> Optional[str]:
        """This rank's physical card, or None with the reason logged once."""
        if self.weg2_card_uuid_cache != "unset":
            return self.weg2_card_uuid_cache
        try:
            uuid_key = resolve_pcie_lock_key()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[weg2] card key unresolved (%s) -- L1 lines carry card=unknown", exc)
            uuid_key = None
        self.weg2_card_uuid_cache = uuid_key
        return uuid_key

    def _weg2_tag_bytes(self, tag: str) -> int:
        """C7/C16: the saver's OWN byte sum for one tag, or 0 with a reason.

        THE INSTRUMENT, and the reason RssShmem is no longer it (spec R8): the
        shared host ring's granules are tmpfs pages mapped by BOTH co-located
        processes, so an RssShmem delta across a pause collapses to ~0 the
        moment the ring lands and the old ``WEG2-CHUNK-BYTES host_image_delta``
        would read a real 13 GiB image as nothing.  ``tms_tag_bytes`` reads the
        allocator's own metadata and is indifferent to where the backup lives.

        A 0 here means "the saver could not answer", printed as such by the
        caller -- never "this tag is empty".
        """
        adapter = getattr(self, "memory_saver_adapter", None)
        getter = getattr(adapter, "tag_bytes", None)
        if getter is None:
            return 0
        try:
            value = getter(tag)
        except Exception:  # noqa: BLE001
            return 0
        return int(value or 0)

    @staticmethod
    def _weg2_with_graph_tag(
        tags: Sequence[str], weg2_memory_saver_on: bool
    ) -> List[str]:
        """Add ``cuda_graph`` to a Weg-2 kv_cache RPC, or return ``tags`` as-is.

        Item `dormant` commit 2.  THE COUPLING IS RANK-LOCAL ON PURPOSE.  The
        front could have been taught to send the tag, but then the sleep and
        the wake would read two different processes' idea of whether it is
        armed, and the wake's ``offload_tags.remove`` raises ``KeyError`` on a
        tag the sleep never added -- a group-fatal fault from an env drift.
        Both legs call this, both read ``weg2_graph_tag_armed()``, whose env
        term is cached per process, so the pair cannot disagree.

        Three ways this returns the input untouched, each of them the correct
        answer rather than a fallback: the stock (non-Weg-2) path, a request
        that is not the kv_cache carrier (the weights family legs, the #89
        disk park), and a request that already names the tag (``tags=None`` ->
        ``GPU_MEMORY_ALL_TYPES``, which contains it).

        FIX 2, finding 1: "the stock path" is decided by
        ``weg2_memory_saver.weg2_group_name()``, INSIDE ``weg2_graph_tag_armed``
        -- not by ``weg2_memory_saver_on``, which is true on every upstream
        engine launched with ``--enable-memory-saver``.  Gating on that alone
        silently widened a stock ``POST /release_memory_occupation
        {"tags":["kv_cache"]}`` into ``kv_cache + cuda_graph`` (and added a
        ``zero_flashinfer_workspaces`` memset to the paired resume) on any such
        engine that also set ``SGLANG_MEMORY_SAVER_CUDA_GRAPH`` -- the
        documented configuration, and the one this file's own rule at the top
        of ``release_memory_occupation`` forbids touching: "a stock POST
        /hibernate must stay byte-for-byte the upstream path".
        """
        if not weg2_memory_saver_on or not weg2_graph_tag_armed(
            weg2_memory_saver_on
        ):
            return list(tags)
        if GPU_MEMORY_TYPE_KV_CACHE not in tags:
            return list(tags)
        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            return list(tags)
        return list(tags) + [GPU_MEMORY_TYPE_CUDA_GRAPH]

    @staticmethod
    def _weg2_zero_graph_scratch() -> Optional[int]:
        """Re-zero the registered flashinfer FLOAT workspaces; count, or None.

        None means the helper could not be reached -- printed as ``n/a`` by the
        caller, never as ``0``, because "no workspace was zeroed" and "the
        zeroing never ran" are the difference between a restored contract and
        a silent one.
        """
        try:
            from sglang.srt.layers.attention.flashinfer_backend import (
                zero_flashinfer_workspaces,
            )

            return int(zero_flashinfer_workspaces())
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _weg2_allocator_cache_bytes() -> Optional[Tuple[int, int]]:
        """``(reserved, allocated)`` of the torch caching allocator, or None.

        NOT an NVML reading and NOT the dormant image: ``memory_reserved`` counts
        only what THIS process's torch caching allocator holds in cudaMalloc'd
        segments, and the memory-saver's tagged regions are unmapped by
        ``tms_pause`` underneath torch, so their bytes are still inside
        ``reserved`` while being physically gone.  The only quantity this pair
        licenses is the DELTA of ``reserved`` across ``empty_cache()`` -- the
        untagged cache actually handed back -- which is what the caller prints.

        None (never a 0) when the counters cannot be read: a cache figure that
        was not taken must not read as a cache that was empty.
        """
        try:
            module = torch.get_device_module()
            return int(module.memory_reserved()), int(module.memory_allocated())
        except Exception:  # noqa: BLE001
            return None

    def _weg2_log_allocator_cache_released(
        self,
        before: Optional[Tuple[int, int]],
        after: Optional[Tuple[int, int]],
    ) -> None:
        """Print what ``empty_cache()`` on the sleep path actually gave back.

        Item `dormant`, record section [1y] row R6.  The line states its
        instrument in full because the number next to it on every other Weg-2
        line (``proc_used``) is a DIFFERENT instrument over a DIFFERENT
        population, and the two must never be added: NVML per-process bytes
        include the CUDA context, the barlink BAR1 windows and the comm buffers,
        none of which the torch allocator knows about.
        """
        if before is None or after is None:
            logger.info(
                "WEG2-SLEEP allocator_cache_released_mib=n/a (torch caching-allocator "
                "counters unreadable on this rank -- absence of a reading, NOT an "
                "empty cache)"
            )
            return
        reserved_before, allocated_before = before
        reserved_after, allocated_after = after
        logger.info(
            "WEG2-SLEEP allocator_cache_released_mib=%.1f "
            "(instrument: torch.cuda.memory_reserved delta across empty_cache() on "
            "THIS process's caching allocator -- reserved %.1f -> %.1f MiB, allocated "
            "%.1f -> %.1f MiB; population = the UNTAGGED reserve only, because "
            "tms_pause unmaps the tagged regions underneath torch and their bytes stay "
            "inside reserved; NOT NVML, NOT the dormant image, never add it to proc_used)",
            (reserved_before - reserved_after) / MIB_,
            reserved_before / MIB_,
            reserved_after / MIB_,
            allocated_before / MIB_,
            allocated_after / MIB_,
        )

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
        # #1273 S2 (refuter F6): the region tag is DERIVED, and the same
        # statement that opens the region publishes it, so this site and
        # model_runner's boot load cannot silently disagree about the tag the
        # post-load repack (`weight_chunk_scope`, model_loader/loader.py:941)
        # restores.  Under --weg2-weight-source exchange with a draft shard in
        # this process the derivation REFUSES: one region carries one tag and
        # this call refills two shards that now want two.
        from sglang.srt.managers.weg2_memory_saver import weights_region
        from sglang.srt.weg2.weight_exchange import (
            roll_forward_refusal_message,
            roll_forward_weights_tag,
        )

        weights_reload_tag = roll_forward_weights_tag(
            has_draft_shard=self.draft_worker is not None
        )
        if weights_reload_tag is None:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: " + roll_forward_refusal_message()
            )

        # Region outside, PCIe lock inside: the region must cover every
        # allocation the loader makes, the lock only the link.
        with weights_region(
            self.memory_saver_adapter,
            weights_reload_tag,
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

    def _weg2_group_fence(
        self,
        what: str,
        *,
        ok: bool = True,
        failure: str = "",
        per_tag: Optional[Dict[str, List[float]]] = None,
        leg_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """:meth:`_weg2_group_fence_impl`, plus the marker that stops re-entry.

        THE RE-ENTRY HAZARD is a HANG, not an error: the fence takes two
        collectives, so an exception raised inside it -- a W29 from a peer's
        vote, a ``monitored_barrier`` expiry -- has already consumed this rank's
        turn in them.  ``_weg2_leg_failed`` would then vote again and post a
        third collective the other ranks are not in, and gloo would sit there.
        The marker is set here and read (and cleared) there.
        """
        self.weg2_fence_raised = False
        try:
            return self._weg2_group_fence_impl(
                what, ok=ok, failure=failure, per_tag=per_tag, leg_ms=leg_ms
            )
        except BaseException:
            self.weg2_fence_raised = True
            raise

    def _weg2_group_fence_impl(
        self,
        what: str,
        *,
        ok: bool = True,
        failure: str = "",
        per_tag: Optional[Dict[str, List[float]]] = None,
        leg_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """Every rank of the group joins here before the owner rank answers.

        C15 -- THE OK-BIT, AND WHY IT IS SECOND.  After the barrier every rank
        also contributes ``ok`` (did MY half of this leg complete?) through one
        ``all_gather_object``, and any False makes EVERY rank raise
        :class:`Weg2FlipRankDisagree` (W29).  The order is ruled, not stylistic
        (spec R15): gloo's ``all_gather_object`` has NO timeout, so a rank that
        died mid-leg would hang the survivors in it forever; the existing
        ``monitored_barrier`` runs FIRST because it is bounded and NAMES the
        rank that did not join.  The gather is only ever reached by ranks that
        are all present.

        The same gather carries the leg's per-tag report (C16/C17) and each
        rank's own leg cost, so the answering rank can name the CRITICAL PATH
        (L5) without a second collective and without a second bookkeeping of
        the same numbers.  Returns that reduction; ``{}`` when there is no
        group to gather over (world <= 1, or no cpu group -- a single-rank
        engine cannot disagree with itself).

        The RPC answer is ONE rank's: scheduler.py process_input_requests
        sends the output from the rank that owns the tokenizer socket (PP0 /
        TP0), and a PP follower runs this handler only on ITS next pass,
        after the chain forward.  The Weg-2 front reads the answer as "the
        GROUP has released / mapped these pages" and moves the OTHER group's
        pages onto the same cards.  MEASURED 2026-09-07, boot weg2onebackup2:
        PP2's base-weights pause (3336 MiB, nvml2) completed 0.4 s after the
        RPC had returned; the front had already woken D's kv_cache on that
        card -> cu_mem_create out of memory, group D dead.  The front's dc
        line read 4714 MiB and was mis-read as residue growth; the rank's own
        census 0.4 s later read 1378, unchanged.

        Same mechanism as upstream's barrier(tp_cpu_group) in this handler,
        over the group's world cpu group instead (P: 3 PP stages x TP 1,
        D: 1 x TP 3).  On a PP stage the request-chain send to the next
        stage is COMMITTED first -- clause (ii) of
        _pp_forward_and_process_input_requests: the forward is posted
        async before the handler runs and is otherwise progressed only at the
        end of the pass, so blocking here without the commit is variant A
        (the owner waits for a peer that never received the request).  The
        last stage owes no forward; its list is empty and the commit is a
        no-op.  Weg-2 path only (the callers gate on the memory saver), so a
        stock hibernate stays byte-for-byte upstream.
        """
        scheduler = self.scheduler
        if scheduler is None:
            return {}
        world_group = getattr(scheduler, "world_group", None)
        cpu_group = getattr(world_group, "cpu_group", None)
        if cpu_group is None:
            return {}
        world = torch.distributed.get_world_size(group=cpu_group)
        if world <= 1:
            return {}
        t0 = time.perf_counter()
        ps = getattr(scheduler, "ps", None)
        joined = "n/a"
        if ps is not None and getattr(ps, "pp_size", 1) > 1:
            # The bounded join of the fork (#973 deadline, clears the list),
            # not a naked wait(): the peer is idle in its chain recv (the
            # group is drained before every sleep/wake), so this returns as
            # soon as the message is taken.
            pending = getattr(scheduler, "send_req_work", None)
            joined = str(len(pending)) if pending is not None else "none"
            if pending:
                scheduler._pp_join_comm_work(pending)
        # gloo-only monitored barrier: bounded, and on expiry it NAMES the
        # ranks that did not join (a plain barrier on this group would sit for
        # the group's two-hour timeout).
        #
        # C21 / R13 -- THE BUDGET, RE-JUSTIFIED AGAINST A WHOLE-LEG PAYLOAD.
        # Its old basis was "the largest single pause measured is ~1.5 s
        # (3336 MiB D2H)", which was true while the front sent one RPC PER TAG
        # and this fence closed once per tag.  C9 sends one RPC per FAMILY, so
        # what has to fit inside this budget is now an entire leg: 11.4 s of
        # D2H by the spec's per-tag byte table, and 14 481 ms measured for the
        # sleep leg of boot weg2dk5.  The number did not move; its
        # justification did, from ~80x a single tag to ~8x the measured leg.
        # See WEG2_GROUP_FENCE_BUDGET_S for the whole argument -- it is stated
        # once, at the constant, and this fence and C14's credit wait both read
        # it so there is no second timeout constant anywhere on this path.
        torch.distributed.monitored_barrier(
            group=cpu_group,
            timeout=timedelta(seconds=WEG2_GROUP_FENCE_BUDGET_S),
            wait_all_ranks=True,
        )
        # C15: the ok-bit, strictly AFTER the barrier that names a non-joiner.
        rank = torch.distributed.get_rank(group=cpu_group)
        mine = {
            "rank": rank,
            "ok": bool(ok),
            "failure": str(failure or ""),
            "card": self._weg2_card_uuid() or "unknown",
            "leg_ms": float(leg_ms),
            "per_tag": dict(per_tag or {}),
        }
        gathered: List[Optional[Dict[str, Any]]] = [None] * world
        torch.distributed.all_gather_object(gathered, mine, group=cpu_group)
        votes = [v for v in gathered if isinstance(v, dict)]
        bad = [v for v in votes if not v.get("ok", False)]
        logger.info(
            "WEG2-GROUP-FENCE %s joined in %.0f ms (world=%d ranks, chain sends joined=%s; "
            "the RPC answer now means every rank finished this WHOLE LEG) "
            "ok=%d/%d (denominator: the ranks that answered the gather)",
            what,
            (time.perf_counter() - t0) * 1000,
            world,
            joined,
            len(votes) - len(bad),
            len(votes),
        )
        if bad:
            named = ", ".join(
                f"rank {v.get('rank')} card={v.get('card')}: {v.get('failure') or 'no reason given'}"
                for v in bad
            )
            line = (
                f"W29 Weg2FlipRankDisagree rank={rank} tag={what} "
                f"exc={named} -- every rank stops in this fence"
            )
            logger.error("%s", line)
            raise Weg2FlipRankDisagree(line)
        # The reduction the answering rank turns into C17's per_tag / L5's
        # critical path.  Per tag: the MAXIMUM ms over ranks, because the leg
        # is finished when its slowest holder is -- the same rule the tokenizer
        # side applies over engines, stated once here and once there because
        # they reduce over two different populations and both must name theirs.
        merged: Dict[str, List[float]] = {}
        for v in votes:
            for tag_name, pair in (v.get("per_tag") or {}).items():
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                cand = [float(pair[0]), float(pair[1])]
                have = merged.get(str(tag_name))
                if have is None or cand[1] > have[1]:
                    merged[str(tag_name)] = cand
        slowest = max(votes, key=lambda v: float(v.get("leg_ms", 0.0)), default=None)
        critical = (
            ""
            if slowest is None
            else (
                f"rank={slowest.get('rank')} card={slowest.get('card')} "
                f"ms={float(slowest.get('leg_ms', 0.0)):.0f} "
                f"(slowest of {len(votes)} rank(s) in this leg)"
            )
        )
        return {"per_tag": merged, "critical_path": critical}

    def _weg2_drain_hicache_before_sleep(self, bound_s: float = 30.0) -> None:
        """Drain HiCache in-flight terms (write-through / storage backup /
        load-back / prefetch) before a sleep is judged -- see the caller."""
        sch = self.scheduler
        tc = getattr(sch, "tree_cache", None)
        if not getattr(sch, "enable_hierarchical_cache", False) or tc is None:
            return
        if not hasattr(tc, "check_hicache_events") or not hasattr(sch, "idle_blockers"):
            return
        t0 = time.time()
        polls = 0
        first = list(sch.idle_blockers())
        while time.time() - t0 < bound_s:
            blockers = list(sch.idle_blockers())
            if not blockers or any(not b.startswith("hicache") for b in blockers):
                break
            tc.check_hicache_events()
            polls += 1
            if sch.is_fully_idle():
                break
            time.sleep(0.01)
        logger.warning(
            "WEG2 SLEEP-DRAIN: waited %.2f s (%d polls) for HiCache in-flight terms "
            "before the sleep; blockers at entry %s, now %s",
            time.time() - t0, polls, first, list(sch.idle_blockers()),
        )

    # ------------------------------------------------------------------
    # C15 -- ranks never disagree: a rank that could not finish its half of a
    # leg joins the SAME fence the others are in and votes False, so the group
    # stops together (W29) instead of the front reading one rank's success as
    # the group's.  The two public handlers are these wrappers; the work is in
    # the ``_weg2_*`` bodies.  Written as a wrapper rather than as a try inside
    # the body because the exception may come from ANY statement of the leg
    # (the pause itself, the credit, the static-state export), and a group STOP
    # that only covers the statements someone remembered to wrap is the exact
    # rank-local-silent-failure shape spec section 10.5 forbids.
    # ------------------------------------------------------------------

    def _weg2_fence_is_armed(self) -> bool:
        server_args = self._weg2_server_args()
        return server_args is None or bool(
            getattr(server_args, "enable_memory_saver", False)
        )

    def _weg2_leg_failed(self, what: str, exc: BaseException) -> None:
        """Join the fence with a False vote -- unless the fence is where we died.

        THE RE-ENTRY HAZARD, and it is a hang rather than an error: the fence
        takes two collectives, so an exception raised INSIDE it (a W29 from a
        peer's vote, a monitored_barrier expiry) has already consumed this
        rank's turn in them.  Voting again would post a third collective the
        other ranks are not in, and gloo would sit there.  So a failure that
        came from the fence propagates untouched -- the group has already
        stopped, which is the outcome this method exists to produce.
        """
        if self.weg2_fence_raised:
            self.weg2_fence_raised = False
            return
        if not self._weg2_fence_is_armed():
            return
        self._weg2_group_fence(
            what, ok=False, failure=f"{type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # C14 -- the device-side credit, from this rank's side of the corridor
    # ------------------------------------------------------------------

    def _weg2_group_name(self) -> str:
        """``"P"``/``"D"``, or ``"?"`` when this rank is not a Weg-2 group.

        FIX 2 (finding 1, carried): this read WAS
        ``getattr(server_args, "weg2_group", "")`` -- an attribute that is
        assigned NOWHERE in either tree, so every ``WEG2-FLIP-TAG group=`` line
        of every boot printed ``?`` while claiming to name a group.  That is
        the Klasse-A instrument defect in its plainest form, and it is the same
        root as the finding: there was no in-process Weg-2 identity at all.
        There is one now, published by ``launcher.build_env(group=...)``.
        ``ring_table._TAG_RE`` reads this token as ``group=\\S+`` and does not
        capture it, so a real name is strictly more information, not a format
        change any parser depends on.
        """
        from sglang.srt.managers.weg2_memory_saver import weg2_group_name

        return weg2_group_name() or "?"

    def _weg2_rank(self) -> int:
        scheduler = self.scheduler
        for attr in ("tp_rank", "pp_rank"):
            value = getattr(scheduler, attr, None)
            if isinstance(value, int):
                return value
        return -1

    def _weg2_backup_census(self, weights_tags, tag_bytes):
        """``({tag: bytes}, population)`` -- A1-2's per-card dormant image.

        FIX 1 round 1, finding 2.  The population that matters is EVERY tag with
        a host backup, not the weights family: ``GPU_MEMORY_TYPE_CUDA_GRAPH`` is
        paused in this same RPC under ``SGLANG_ADAPTIVE_CAPTURE_CPU_BACKUP``, and
        the draft pools of the parallel slice are the next instance.  The saver's
        own ``enable_cpu_backup`` metadata is the only place that population
        exists, so it is asked; ``tag_bytes`` cannot stand in for it, because it
        counts device bytes whether or not they are ever copied to the host and
        would charge ``kv_cache`` (paused WITHOUT backup, R20) into the ring.

        When the running hook has no such entry the answer is the weights family
        and the SECOND element says so -- the caller prints it on every line and
        the planner reads it back, so a lower bound can never be quoted as a
        measurement.  That is the floor A1-2 sets on honesty.
        """
        weights_only = ({t: int(tag_bytes.get(t, 0)) for t in weights_tags},
                        WEG2_TAG_POPULATION_WEIGHTS)
        adapter = getattr(self, "memory_saver_adapter", None)
        getter = getattr(adapter, "backed_up_tag_bytes", None)
        if getter is None:
            return weights_only
        try:
            census = getter()
        except Exception:  # noqa: BLE001
            return weights_only
        if not census:
            return weights_only
        return {str(t): int(v) for t, v in census.items()}, WEG2_TAG_POPULATION_ALL

    def _weg2_open_credit_for_leg(self, epoch):
        """S's side: the counter for THIS card, opened for THIS FLIP.

        ``None`` (and no waiting anywhere) whenever the card key cannot be
        resolved, the counter cannot be created, or THE FLIP EPOCH IS NOT KNOWN
        -- the credit is a deadlock-avoidance instrument for a co-located pair,
        and a rank that cannot even name its card has no co-located pair to fund.
        The failure is logged, never swallowed into a success value.

        THE EPOCH IS THE FLIP'S, NOT A CLOCK (FIX 1 round 1, finding 4).  The
        predecessor stamped ``int(time.time())`` and nothing ever read it back,
        while C9 issues both legs in ONE ``asyncio.gather`` -- so there is no
        happens-before between this reset and the co-located waking rank's first
        ``wait_for``, the per-card file alternates writers across flips, and at
        flip N+1 W typically read flip N's TERMINAL state: ``leg_complete`` with
        a whole image of credit.  Both outcomes were silent and wrong -- a resume
        licensed by bytes nobody freed (the CUDA OOM C14 exists to prevent, spec
        10.5), or an instant W35 killing a flip nothing was wrong with.  The
        front owns ``self.epoch`` and sends both RPCs, so it carries it; a
        request without one arms nothing rather than reading a counter it cannot
        date.
        """
        if not self._weg2_fence_is_armed():
            return None
        if epoch is None:
            logger.warning(
                "[weg2 credit] the release request carries no flip epoch, so the "
                "device-side credit is NOT opened for this leg: a counter that "
                "cannot be dated cannot be told from the previous flip's terminal "
                "state, and reading one as funding is the silent failure C14 "
                "exists to prevent"
            )
            return None
        uuid_key = self._weg2_card_uuid()
        if uuid_key is None:
            return None
        try:
            credit = vram_credit(uuid_key)
            credit.begin_leg(str(epoch))
        except OSError as exc:
            logger.warning(
                "[weg2 credit] no VRAM credit published on %s (%s) -- a "
                "co-located waking rank will fall back to the card's own free "
                "bytes and refuse by name if they are short",
                uuid_key, exc,
            )
            return None
        return credit

    def _weg2_credit_reader(self, epoch):
        """W's side: ``(counter, epoch)``, read-only.  ``(None, None)`` disarms.

        Same epoch rule as :meth:`_weg2_open_credit_for_leg` and for the same
        reason: without the flip's own epoch this rank cannot tell this leg's
        counter from the last one's, so it consults none.
        """
        if not self._weg2_fence_is_armed() or epoch is None:
            return None, None
        uuid_key = self._weg2_card_uuid()
        if uuid_key is None:
            return None, None
        try:
            return vram_credit(uuid_key), str(epoch)
        except OSError:
            return None, None

    def _weg2_free_bytes(self) -> Optional[int]:
        """NVML free on THIS rank's card, or None when it cannot be read.

        None is load-bearing: :meth:`VramCredit.wait_for` then cannot take the
        "the card already holds the bytes" exit and waits on the peer, which is
        the conservative direction.  A blind reading must never license a
        resume that has no bytes.
        """
        uuid_key = self._weg2_card_uuid()
        if uuid_key is None:
            return None
        try:
            from sglang.srt.registry import nvml as nvml_registry

            return int(nvml_registry.memory_info_for_uuid(uuid_key).free_bytes)
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # #1273 S5b -- THE TWO SHADOW HOOKS.
    #
    # Two THIN adapters, and deliberately nothing else: they gather the
    # arguments this leg already holds and call
    # ``weight_exchange_shadow.run_leg_hook``, which owns every decision, every
    # refusal and the run's lifetime.  The bodies live in that module because
    # this class cannot be constructed without a model runner, a torch process
    # group and a device, so a hook written HERE is a hook no hermetic test can
    # drive -- and the S5 round has now paid twice for logic that no arm
    # reached (refuter finding 1's ``bind_scratch``, review finding 2's
    # unpinned guard).
    #
    # UPSTREAM-MINIMAL: this is two calls appended to the two existing weights
    # legs, not a second scheduler, not a thread, not an RPC.  The ring remains
    # the only authority for weight bytes; both adapters catch
    # ``BaseException`` and return, so no shadow failure can reach a flip.
    # ------------------------------------------------------------------

    def _weg2_device_index(self) -> int:
        """THIS RANK's CUDA ordinal, read from torch.  ``-1`` when unreadable.

        MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 3): this was the
        literal ``0``.  The launcher hands EVERY rank the same
        ``CUDA_VISIBLE_DEVICES`` -- all three card uuids, one string, both
        groups (``launcher.py`` builds ``cvd`` from every card) -- exactly so
        that ``weight_exchange_region``'s "rank *n* of either group runs on
        ``cards[n]``" holds.  So a rank's own device is its ordinal, not 0, and
        a hardcoded 0 makes the observer (i) ``cudaMalloc`` on ANOTHER rank's
        card, (ii) price that allocation against its own card's free column and
        uuid, and (iii) leave the SCHEDULER THREAD -- the flip leg's own thread
        -- on device 0, because ``CudartDeviceOps.set_device`` is a bare
        ``cudaSetDevice`` with no restore.  A zero-authority observer that moves
        the authoritative thread's current device is danger (a) in its plainest
        form.

        Read from torch and never assumed: torch is what the leg thread's
        device actually is, and ``_weg2_card_uuid`` already resolves this card
        through it, so the two cannot name different cards.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 -- an observer never raises
            pass
        return -1

    def _weg2_shadow_gate_rows(self, hook: str, group: str):
        """The rows this hook may expect to see voted AT THIS INSTANT.

        MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 1).  The two hooks sit
        at OPPOSITE ENDS OF THE FLIP: the source runs on the SLEEPING group
        before its ``pause``, the destination on the WAKING group after its
        ``resume`` and its disk reload -- and on a co-located card the waking
        rank is fenced (C14) on the credit the sleeping rank publishes INSIDE
        its pause loop, i.e. after its own hook.  A source rank that waits for
        all six rows therefore waits for three rows that cannot be written
        until it stops waiting: a circular wait, resolved only by the gate
        expiry, paid by every sleeping rank on every flip, on the critical path
        of that credit.

        So the source expects its OWN group's three rows -- the ranks that are
        at the same instant of the same flip -- and the destination expects all
        six, because the source rows were sealed earlier in this same flip with
        this region's ``epoch_hash`` and this ``leg`` and are therefore free.
        ``None`` means all six.
        """
        from sglang.srt.weg2 import weight_exchange_region as xr

        if hook != "source" or group not in ("P", "D"):
            return None
        return tuple(xr.rank_row(group, r) for r in range(xr.N_CARDS))

    def _weg2_shadow_hook(self, hook: str, *, recv_req, reserve_bytes: int = 0,
                          ring_ms=None) -> None:
        """Run one leg's shadow, or return having touched nothing.

        THE ARM IS CHECKED FIRST AND CHEAPLY.  ``shadow_armed()`` reads
        ``--weg2-weight-source``; on ``ring`` -- the default and every boot that
        has run to date -- it is False and this method returns after ONE module
        import, before a device call, an allocation, an env write or a single
        line of the exchange's machinery.  That is what keeps the default leg
        byte-identical, and it is what
        ``test_the_hooks_are_never_reached_on_the_ring_arm`` proves with a
        tripwire on every door out of the arm check.

        (The docstring used to claim it returned "before importing" anything;
        it did not, and does not -- ``shadow_armed`` lives in the shadow module
        and reading it is an import.  S5b refuter, non-blocking finding: a
        claim a reader can refute by looking two lines up is worse than no
        claim.  What is byte-identical is the DEVICE and the ALLOCATOR, not the
        import table.)
        """
        try:
            from sglang.srt.weg2 import weight_exchange_shadow as sh

            if not sh.shadow_armed():
                return
            from sglang.srt.weg2 import weight_exchange_region as xr

            group = self._weg2_group_name()
            rank = self._weg2_rank()
            if group not in ("P", "D") or rank < 0:
                return
            device = self._weg2_device_index()
            if device < 0:
                logger.info(sh.rank_local_skip_message(
                    reason="no-device", rank=rank,
                    leg=_weg2_flip_index_of(getattr(recv_req, "epoch", None)),
                    epoch=str(getattr(recv_req, "epoch", "") or ""),
                    detail="torch reports no CUDA device on this rank"))
                return
            peer = "D" if group == "P" else "P"
            free_bytes = self._weg2_free_bytes()
            inputs = sh.ShadowLegInputs(
                leg=_weg2_flip_index_of(getattr(recv_req, "epoch", None)),
                epoch=str(getattr(recv_req, "epoch", "") or ""),
                direction="d2h" if hook == sh.HOOK_SOURCE else "h2d",
                hook=str(hook),
                rank=int(rank),
                row=xr.rank_row(group, int(rank)),
                peer_row=xr.rank_row(peer, int(rank)),
                device=int(device),
                card_uuid=self._weg2_card_uuid() or "unknown",
                free_mib=0 if free_bytes is None else int(free_bytes // MIB_),
                resume_reserve_bytes=int(reserve_bytes),
                ring_ms=ring_ms,
                gate_rows=self._weg2_shadow_gate_rows(str(hook), group),
            )
            try:
                sh.run_leg_hook(inputs, log=logger.info)
            finally:
                # THE LEG THREAD'S DEVICE IS PUT BACK, ALWAYS.  The shadow's
                # raw ``cudaMalloc`` and its stream both go through
                # ``CudartDeviceOps.set_device``, which is a bare
                # ``cudaSetDevice`` with no save/restore, so without this the
                # observer decides what device the AUTHORITATIVE leg continues
                # on.  Restoring through torch (not through the ops layer) is
                # deliberate: torch's current device is the one the rest of
                # this leg reads.
                self._weg2_restore_device(device)
        except BaseException as exc:  # noqa: BLE001 -- an observer never raises
            logger.warning(
                "[weg2 shadow] the %s hook failed and the flip is unaffected "
                "(%s: %s) -- the ring is and stays the only authority for "
                "weight bytes",
                hook, type(exc).__name__, exc,
            )

    def _weg2_restore_device(self, device: int) -> None:
        """Put the calling thread's CUDA device back where the hook found it."""
        try:
            import torch

            if torch.cuda.is_available() and int(device) >= 0:
                torch.cuda.set_device(int(device))
        except Exception:  # noqa: BLE001 -- an observer's unwind
            pass

    def _weg2_shadow_source_leg(self, recv_req) -> None:
        """SOURCE hook, sleep leg.  Placed BEFORE the pause loop, not after it.

        **STATED DEVIATION from the S5b brief's parenthetical** ("after the ring
        save has the bytes"), with the reason: the exporter READS this rank's
        live weight tensors, and they are allocated inside
        ``region(GPU_MEMORY_TYPE_WEIGHTS)`` (``model_runner.py:2344-2348``), so
        after ``memory_saver_adapter.pause(tag)`` they are UNMAPPED PAGES.  A
        read there is the campaign (a) fault on the sibling tag, and the
        prohibition is pinned by
        ``test_weights_block_pause_is_the_last_statement``.  So the hook sits at
        the last instant the bytes exist on the device: after the census and the
        C14 credit -- i.e. after the ring has everything it needs from this
        rank -- and before the pause that takes the pages away.

        IT PASSES NO ``resume_reserve_bytes``, and 0 is the CORRECT value here
        rather than an unwired one: nothing on this leg is waiting to be
        mapped -- the leg's whole job is to give pages BACK -- so there is no
        future demand to hold free VRAM for.  ``hook=source`` on the line says
        which of the two a ``resume_reserve_mib=0`` came from.

        THE LEG'S OWN WALL INCLUDES IT.  ``weg2_leg_t0`` is taken BEFORE this
        call (S5b refuter, must_fix 4): with the clock started after the hook,
        ``weg2_leg_ms`` and every WEG2-FLIP-TAG wall excluded the entire
        shadow, so the sleep leg's own instrument reported a wall that was
        short by exactly the cost the observer added -- and that wall gates the
        C14 credit a co-located waking rank is fenced on.  ``shadow_ms`` on the
        shadow's own line is the subtrahend, so the two numbers on one log
        separate the ring's wall from the observer's.
        """
        self._weg2_shadow_hook("source", recv_req=recv_req)

    def _weg2_shadow_destination_leg(self, recv_req, *, reserve_bytes: int = 0,
                                     ring_ms=None) -> None:
        """DESTINATION hook, wake leg, after ``family_complete`` and the reload.

        This is the only place in the boot where the ground truth exists: the
        ring has restored every weight byte and the static state is imported, so
        the shadow's pulled stripes have something to be compared AGAINST.  It
        runs before the leg reports done, so a mismatch appears in the log
        beside the flip that produced it rather than one flip later.

        ``reserve_bytes`` IS THE STILL-UNMAPPED DEMAND AT *THIS* INSTANT, and
        the number it used to carry was wrong in the one way that matters.
        MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 2): it carried the
        WEIGHTS image (the sum of ``tag_bytes``, read before the resume loop) --
        but this hook runs AFTER that resume completed, and ``free_mib`` is read
        here too, so the image is already OUT of the free column.  Subtracting
        it a second time removed 9-13 GiB from a free column the VRAM corridor
        law holds at 819-1229 MiB: ``price_shadow.affordable`` was False by
        construction and every rank voted NO on every leg.  The demand that is
        genuinely still unmapped when this runs is the REST OF THIS RPC's tags
        -- kv_cache and anything else the resume has not reached -- read from
        the same saver instrument, at the instant the term is consumed.
        """
        self._weg2_shadow_hook("destination", recv_req=recv_req,
                               reserve_bytes=reserve_bytes, ring_ms=ring_ms)

    def _weg2_await_vram_credit(self, credit, tag: str, need_bytes: int,
                                epoch=None) -> None:
        if credit is None or need_bytes <= 0:
            return
        try:
            rec = credit.wait_for(
                need_bytes,
                budget_s=WEG2_GROUP_FENCE_BUDGET_S,
                tag=tag,
                free_bytes_now=self._weg2_free_bytes(),
                epoch=epoch,
            )
        except Weg2VramCreditRefused:
            raise
        except OSError as exc:
            logger.warning("[weg2 credit] unreadable on %s: %s -- not waiting", tag, exc)
            return
        if rec.get("waited_s", 0.0) > 0.0:
            logger.info(
                "WEG2-VRAM-CREDIT card=%s tag=%s waited=%.0f ms credit=%d MiB "
                "requested=%d MiB (%s)",
                self._weg2_card_uuid() or "unknown", tag,
                float(rec["waited_s"]) * 1000,
                int(rec.get("credit_bytes", 0)) // MIB_,
                int(need_bytes) // MIB_,
                rec.get("reason", ""),
            )

    @_weg2_group_stop_on_leg_failure
    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        # C16/C17: this rank's own per-tag report of THIS leg, filled by the
        # weights block below and reduced over the group at the fence.
        weg2_per_tag: Dict[str, List[float]] = {}
        weg2_leg_ms = 0.0
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

        # #1233 zero-remainder (boot weg2zr1 killer, W4): under PP the front's
        # quiesce witness is PP0's /flush_cache verdict, while this assert runs
        # on EVERY rank. A follower that finished the last request later than
        # PP0 can still hold its own write-through / storage backups of that
        # request in flight when the sleep RPC lands (PP2: 'not-idle because:
        # hicache_backup(2)' -> AssertionError -> group death, 15:34:06Z).
        # Those terms drain by themselves through check_hicache_events, which
        # only the scheduler loop drives; drive it here, bounded, before
        # judging. Any non-HiCache blocker still fails the assert below.
        if not self.is_fully_idle():
            self._weg2_drain_hicache_before_sleep()

        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        # ITEM `dormant` COMMIT 2, the SLEEP half of the coupling.  The graph
        # tag rides the kv_cache RPC, and it is added HERE, rank-locally, not
        # by the front: both legs then read the same process-local resolver, so
        # a tag can never be paused by one side and not removed by the other
        # (the wake does `offload_tags.remove`, which raises on a tag that was
        # never added).  kv_cache is the right carrier because it is the one
        # RPC with the same shape -- no cpu backup, content-free, and already
        # ordered first on the sleep and last on the wake, which is exactly
        # upstream's pause/resume order for cuda_graph.
        tags = self._weg2_with_graph_tag(tags, weg2_memory_saver_on)

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
            # C16: the per-tag bytes are read from the SAVER, before the pause
            # that moves them, and they are metadata reads -- no device call is
            # added after the pause (the prohibition above is intact).
            tag_bytes = {tag: self._weg2_tag_bytes(tag) for tag in weights_tags}
            # A1-2 / FIX 1 round 1: the CENSUS is a different population from the
            # timed loop.  The loop walks the weights family (that is the leg);
            # the census must name every tag that holds HOST bytes while this
            # group sleeps, or the ring is sized from a lower bound while the
            # provenance line says MEASURED.
            census, population = self._weg2_backup_census(weights_tags, tag_bytes)
            # C14: S opens the device-side credit for this leg BEFORE it frees
            # anything.  The epoch is THE FLIP'S, carried on the request by the
            # front that owns it -- see :meth:`_weg2_open_credit_for_leg`.
            credit = self._weg2_open_credit_for_leg(getattr(recv_req, "epoch", None))
            # #1273 S5b: the SOURCE half of the shadow, at the last instant the
            # weight pages are mapped.  See _weg2_shadow_source_leg for why it
            # is here and not after the pause.  Never raises; on every arm but
            # --weg2-weight-source shadow it returns having touched nothing.
            #
            # THE LEG'S CLOCK STARTS BEFORE IT (S5b refuter, must_fix 4).  With
            # t0 after the hook, weg2_leg_ms -- the number this leg publishes
            # as its own wall, and the one the sb5f flip band is read from --
            # excluded the whole shadow, so an observer could add seconds to
            # the leg that gates a co-located rank's C14 credit while the
            # leg's own instrument reported no change at all.  On the ring arm
            # the hook returns after one import, so the band is unaffected.
            weg2_leg_t0 = time.perf_counter()
            self._weg2_shadow_source_leg(recv_req)
            with self._weg2_pcie_lock("sleep-D2H " + ",".join(weights_tags), direction="d2h"):
                for tag in weights_tags:
                    t_tag = time.perf_counter()
                    self.memory_saver_adapter.pause(tag)
                    weg2_per_tag[tag] = [
                        float(tag_bytes.get(tag, 0)),
                        (time.perf_counter() - t_tag) * 1000,
                    ]
                    if credit is not None:
                        # The device bytes this tag's pause just gave back --
                        # the same number, from the same instrument, that the
                        # waking rank is waiting on.
                        credit.publish(tag, tag_bytes.get(tag, 0))
            weg2_leg_ms = (time.perf_counter() - weg2_leg_t0) * 1000
            if credit is not None:
                credit.leg_complete()
            card_uuid = self._weg2_card_uuid() or "unknown"
            # L1, one line per tag of the CENSUS -- not of the timed loop.  Every
            # line states the population it belongs to, because the ring planner
            # that reads them (ring_table.parse_group_log) must be able to tell
            # A1-2's measured dormant image from a weights-only lower bound, and
            # it cannot get that from the line's mere existence.  ``ms`` is the
            # timed pause where the loop above took one and -1 where the tag is
            # in the census but not in this leg (its bytes are still part of the
            # dormant image; its pause is not part of this RPC's cost).
            for tag, nbytes in sorted(census.items()):
                tms = weg2_per_tag.get(tag, [0.0, -1.0])[1]
                logger.info(
                    "WEG2-FLIP-TAG group=%s rank=%d card=%s dir=d2h tag=%s bytes=%d MiB "
                    "population=%s (source: tms_tag_bytes, NOT RssShmem) ms=%.0f "
                    "GB/s=%.2f granules=%d",
                    self._weg2_group_name(), self._weg2_rank(), card_uuid, tag,
                    int(nbytes) // MIB_, population, tms,
                    (nbytes / 1e9) / max(1e-6, tms / 1000.0) if tms > 0 else 0.0,
                    int(nbytes) // TMS_RING_GRANULE_BYTES,
                )
            # RssShmem stays on the line as the CROSS-CHECK it now is, and its
            # own death is stated: once the ring lands both co-located ranks map
            # the same tmpfs granules and this delta collapses to ~0 while the
            # bytes above are unchanged (spec R8).  A reader comparing the two
            # must know which one is the instrument.
            logger.info(
                "WEG2-CHUNK-BYTES sleep tags=%s host_image_delta=%.0f MiB (RssShmem %.0f -> %.0f MiB, "
                "/proc/self/status; CROSS-CHECK ONLY -- tms_tag_bytes above is the instrument, and this "
                "delta reads ~0 once the shared ring carries the backup)",
                weights_tags, self._weg2_rss_shmem_mib() - shm0, shm0, self._weg2_rss_shmem_mib(),
            )

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            # ITEM `dormant` COMMIT 2: the pause is upstream's, unchanged.  What
            # is added is the byte count, read from the saver's own metadata
            # BEFORE the pause that moves it -- the same instrument and the same
            # ordering rule as the weights family above (a metadata read, no
            # device call after a pause).
            graph_bytes = self._weg2_tag_bytes(GPU_MEMORY_TYPE_CUDA_GRAPH)
            t_graph = time.perf_counter()
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)
            logger.info(
                "WEG2-SLEEP released tags=['%s'] mib=%.1f ms=%.0f "
                "(instrument: tms_tag_bytes for this ONE tag, read before the pause; "
                "population = whatever was allocated inside region_config(cuda_graph) "
                "-- the capture pool plus the flashinfer FLOAT workspace; the int "
                "workspace, the CUDA context and the barlink BAR1 windows are NOT in "
                "this denominator and are not released by it. 0.0 = the saver could "
                "not answer, not an empty tag)",
                GPU_MEMORY_TYPE_CUDA_GRAPH,
                graph_bytes / MIB_,
                (time.perf_counter() - t_graph) * 1000,
            )

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
            #
            # ITEM `dormant` COMMIT 1.  The paragraph above says the benefit of
            # this call is UNMEASURED for the untagged remainder, and it has
            # stayed unmeasured for every Weg-2 boot: the sleep-acceptance
            # census reads NVML per-process bytes, which is the WHOLE dormant
            # image and cannot separate the allocator's reserve out of it.
            # Attribution section [1y] had to leave it inside a 306-358 MiB
            # UNATTRIBUTED remainder for exactly that reason.  Read the torch
            # allocator's own counters on both sides of the call, so the row
            # exists as a number instead of an argument.
            cache_before = self._weg2_allocator_cache_bytes()
            torch.get_device_module().empty_cache()
            cache_after = self._weg2_allocator_cache_bytes()
            self._weg2_log_allocator_cache_released(cache_before, cache_after)
            self._weg2_log_sleep_acceptance(
                weg2_before_census, sorted(self.offload_tags)
            )
            self.weg2_sleep_before = None

        report: Dict[str, Any] = {}
        if weg2_memory_saver_on:
            report = self._weg2_group_fence(
                "release tags=%s" % (list(tags),),
                per_tag=weg2_per_tag,
                leg_ms=weg2_leg_ms,
            )

        # C17: the group's answer carries what the group moved.  ``per_tag``
        # falls back to THIS rank's own numbers when there was no group to
        # gather over (a single-rank engine), and is None on the stock path so
        # that answer is byte-identical to what it always was.
        return ReleaseMemoryOccupationReqOutput(
            per_tag=(report.get("per_tag") or weg2_per_tag or None)
            if weg2_memory_saver_on
            else None,
            critical_path=(report.get("critical_path") or None)
            if weg2_memory_saver_on
            else None,
        )

    @_weg2_group_stop_on_leg_failure
    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        # C16/C17: this rank's own per-tag report of THIS leg, filled by the
        # weights block below and reduced over the group at the fence.
        weg2_per_tag: Dict[str, List[float]] = {}
        weg2_leg_ms = 0.0
        # Same gate as the release path: the Weg-2 fence below runs only on
        # a --enable-memory-saver engine (a stock resume stays upstream).
        server_args = self._weg2_server_args()
        weg2_memory_saver_on = server_args is None or bool(
            getattr(server_args, "enable_memory_saver", False)
        )
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        # ITEM `dormant` COMMIT 2, the WAKE half of the coupling.  Same resolver,
        # same carrier tag, so `offload_tags.remove` below sees exactly what the
        # sleep added.  It runs BEFORE the remove loop for that reason.
        tags = self._weg2_with_graph_tag(tags, weg2_memory_saver_on)

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            t_graph = time.perf_counter()
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)
            graph_ms = (time.perf_counter() - t_graph) * 1000
            graph_bytes = self._weg2_tag_bytes(GPU_MEMORY_TYPE_CUDA_GRAPH)
            # WAKE INVARIANT FOR THE GRAPH TAG, the exact mirror of the
            # kv_cache one further down and for the same reason: the resume maps
            # FRESH physical pages under this region -- on the two-group form
            # they are pages the other group just released -- and this tag has
            # no cpu backup.  The flashinfer FLOAT workspace lives in that
            # region (flashinfer_backend.py, item `dormant` R3a) and its
            # kernels' contract is that unwritten regions read ZERO (NOTE(#50):
            # "fresh cudaMalloc pages read as zeros ... the zeroed state is the
            # contract they were validated against").  Recycled pages are not
            # zero.  zero_flashinfer_workspaces() is the fork's own restore of
            # that state; here it runs after the resume instead of at a request
            # boundary, so the FIRST forward after a wake sees the boot
            # contract rather than the other group's residue.
            zeroed = self._weg2_zero_graph_scratch()
            logger.info(
                "WEG2-RESUME remapped mib=%.1f ms=%.0f workspaces_zeroed=%s "
                "(instrument: tms_tag_bytes for tag '%s' read AFTER the resume, so it "
                "is the bytes now mapped again, not a copy figure -- this tag has no "
                "cpu backup and no H2D leg; workspaces_zeroed counts the registered "
                "flashinfer FLOAT workspaces re-zeroed to the NOTE(#50) contract, "
                "n/a = the helper was unavailable, never 0)",
                graph_bytes / MIB_,
                graph_ms,
                "n/a" if zeroed is None else zeroed,
                GPU_MEMORY_TYPE_CUDA_GRAPH,
            )

        weights_tags = [t for t in tags if is_weights_family_tag(t)]
        if weights_tags:
            # Wake-H2D: with the cpu backup this recommit refills from the host
            # buffer (and, with the #1233 patched hook, frees that buffer).
            # Same lock, same reason as the sleep leg above.
            t_w0 = time.perf_counter()
            shm0 = self._weg2_rss_shmem_mib()
            tag_bytes = {tag: self._weg2_tag_bytes(tag) for tag in weights_tags}
            # S7 (#1273): tag -> the saver's own pass-1/pass-2 decomposition of
            # that tag's resume, or None where the instrument is absent.
            weg2_map_stats: Dict[str, Optional[Dict[str, float]]] = {}
            credit, credit_epoch = self._weg2_credit_reader(
                getattr(recv_req, "epoch", None)
            )
            with self._weg2_pcie_lock("wake-H2D " + ",".join(weights_tags), direction="h2d"):
                for tag in weights_tags:
                    # C14: the device bytes this tag needs may only exist once
                    # the co-located SLEEPING rank has released them, and with
                    # C9 both legs are in flight.  Waiting here turns a race
                    # into a bounded wait whose expiry is a NAMED refusal; when
                    # the card is not short the call returns without waiting at
                    # all, which is every non-co-located boot.
                    self._weg2_await_vram_credit(
                        credit, tag, tag_bytes.get(tag, 0), credit_epoch
                    )
                    t_tag = time.perf_counter()
                    self.memory_saver_adapter.resume(tag)
                    weg2_per_tag[tag] = [
                        float(tag_bytes.get(tag, 0)),
                        (time.perf_counter() - t_tag) * 1000,
                    ]
                    # S7 (#1273): READ THE MAP COST WHILE IT IS STILL THIS TAG'S.
                    # Resume pass 1 maps every allocation of the tag one at a
                    # time, so its wall is proportional to an allocation count
                    # that no log has ever carried -- and with the count absent,
                    # pass 1's time was charged to the copy rate, which is the
                    # unexplained remainder of ADDENDUM 3 section 4 and risk R2
                    # of #1273.  The read is a metadata call on the saver (no
                    # device call), and it is issued INSIDE the loop because the
                    # recorder holds ONE record: read after the next tag's resume
                    # and the number would belong to that tag.  The adapter
                    # returns None -- never a fabricated 0 -- when the running
                    # hook has no such symbol or the record names another tag.
                    weg2_map_stats[tag] = self.memory_saver_adapter.resume_stats(tag)
            weg2_leg_ms = (time.perf_counter() - t_w0) * 1000
            card_uuid = self._weg2_card_uuid() or "unknown"
            for tag, (nbytes, tms) in weg2_per_tag.items():
                # S7 (#1273): THREE FIELDS APPENDED, and the ring planner's
                # parser is unaffected because they are appended -- its regex
                # (ring_table._TAG_RE) anchors on the fields before them and
                # ends at the optional population token.
                #
                # ``allocations`` is the DENOMINATOR of ``map_ms``: pass 1 does
                # one cu_mem_create + cuMemMap + cu_mem_set_access per
                # allocation of the tag, and that count appeared in no log.
                # ``map_ms`` is pass 1 alone, ``copy_ms`` is pass 2's H2D issue
                # plus pass 3's single synchronise; pass 4's granule release is
                # in neither, so ``map_ms + copy_ms`` is a LOWER bound on ``ms``
                # and not a partition of it.  ``n/a`` means the instrument is
                # ABSENT -- hook without the symbol, a record naming another
                # tag, or a ROCm build, whose resume path never records at all.
                # The absence and a measured zero are different findings and
                # this is the whole denominator law: a 0 printed for an absent
                # instrument would read as "the remap was free", which is the
                # claim under test.  A tag whose resume matched no allocation
                # DOES print ``allocations=0 map_ms=0.0`` -- that is a real
                # measurement of "nothing was mapped", not an absence, and the
                # guard above deliberately does not hide it (round-2 refuter
                # F8: the earlier wording claimed a 0 was never printed).
                st = weg2_map_stats.get(tag)
                logger.info(
                    "WEG2-FLIP-TAG group=%s rank=%d card=%s dir=h2d tag=%s bytes=%d MiB "
                    "population=%s (source: tms_tag_bytes, NOT RssShmem) ms=%.0f "
                    "GB/s=%.2f granules=%d allocations=%s map_ms=%s copy_ms=%s",
                    self._weg2_group_name(), self._weg2_rank(), card_uuid, tag,
                    int(nbytes) // MIB_, WEG2_TAG_POPULATION_WEIGHTS, tms,
                    (nbytes / 1e9) / max(1e-6, tms / 1000.0),
                    int(nbytes) // TMS_RING_GRANULE_BYTES,
                    "n/a" if st is None else int(st["allocations"]),
                    "n/a" if st is None else "%.1f" % st["map_ms"],
                    "n/a" if st is None else "%.1f" % st["copy_ms"],
                )
            logger.info(
                "WEG2-CHUNK-BYTES wake tags=%s host_image_delta=%.0f MiB (RssShmem %.0f -> %.0f MiB, "
                "/proc/self/status; negative = the patched saver freed it; CROSS-CHECK ONLY -- "
                "tms_tag_bytes above is the instrument)",
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
                # #1273 S5b: the DESTINATION half.  The ring's bytes are final
                # here -- that is the whole reason this hook is after
                # family_complete and after the reload -- so the shadow's
                # stripes have a ground truth to be compared against.
                # ``ring_ms`` is the leg's OWN per-tag wall, the same instrument
                # the WEG2-FLIP-TAG lines above print, so the two numbers on one
                # log can be subtracted.
                # THE STILL-UNMAPPED DEMAND, READ HERE, WHERE IT IS
                # CONSUMED (S5b refuter, must_fix 2): every tag of THIS rpc
                # that the resume has not reached yet.  The graph tag was
                # resumed above this block and the weights family inside it,
                # so what is left is kv_cache and anything else in `tags` --
                # the bytes that will be mapped after this hook returns and
                # that the shadow's own buffers must not have taken.  Read
                # from the SAVER (`_weg2_tag_bytes`), the same instrument the
                # ring sizes itself from; a 0 there means "the saver could not
                # answer" and prints as resume_reserve_mib=0.
                pending_tags = [
                    t for t in tags
                    if t != GPU_MEMORY_TYPE_CUDA_GRAPH
                    and not is_weights_family_tag(t)
                ]
                self._weg2_shadow_destination_leg(
                    recv_req,
                    reserve_bytes=sum(self._weg2_tag_bytes(t)
                                      for t in pending_tags),
                    ring_ms=sum(float(v[1]) for v in weg2_per_tag.values()),
                )

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            scheduler = self.scheduler
            if scheduler is not None and weg2_memory_saver_on:
                # WAKE INVARIANT (boot weg2ls2b1 killer, 2026-09-07): the
                # resume maps FRESH physical pages under the kv_cache region
                # -- on the two-group form they are the pages the OTHER group
                # released one RPC earlier -- and this tag has no cpu backup,
                # so every table that was created WITH A VALUE inside the
                # region is garbage now: req_to_token (torch.zeros,
                # memory_pool.py ReqToTokenPool.__init__), the hybrid
                # req->mamba index maps, MambaPool's conv/temporal states and
                # cursors.  The fork states the invariant itself ("freshly
                # booted pools are torch.zeros", zero_kv_data_buffers) and
                # relies on it: a reader of an unwritten index position read a
                # benign 0 on every boot before this one and read a page
                # index from another group's KV after the first wake -> PP1
                # 'CUDA error: an illegal memory access' in the first GDN
                # extend after the wake (qwen3_5.py linear_attn), group P
                # dead.  flush_cache() is the fork's own restore of that
                # state (ReqToTokenPool.clear -> req_to_token.zero_(),
                # HybridReqToTokenPool.clear -> mamba maps + reset_state,
                # allocator + tree reset, KV bytes under SGLANG_FLUSH_ZERO_KV),
                # and upstream already runs it in this handler; here it runs
                # AFTER the resume, the mirror image of the MUST_FIX flush
                # BEFORE the pause.  The group is drained (idle assert on the
                # sleep) so the flush cannot refuse.
                t_f0 = time.perf_counter()
                flushed = self.flush_cache()
                logger.info(
                    "WEG2-WAKE-INVARIANT kv_cache pools re-zeroed after resume: flush_cache=%s in %.0f ms "
                    "(fresh-boot zero invariant restored on recycled pages)",
                    flushed,
                    (time.perf_counter() - t_f0) * 1000,
                )
                if not flushed:
                    raise RuntimeError(
                        "W26 Weg2WakeInvariantRefused: flush_cache() refused after resume(kv_cache) "
                        "(the group is not idle?) -- the pools hold recycled pages, serving on them is unsafe"
                    )
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

        report: Dict[str, Any] = {}
        if weg2_memory_saver_on:
            report = self._weg2_group_fence(
                "resume tags=%s" % (list(tags),),
                per_tag=weg2_per_tag,
                leg_ms=weg2_leg_ms,
            )

        return ResumeMemoryOccupationReqOutput(
            per_tag=(report.get("per_tag") or weg2_per_tag or None)
            if weg2_memory_saver_on
            else None,
            critical_path=(report.get("critical_path") or None)
            if weg2_memory_saver_on
            else None,
        )

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
