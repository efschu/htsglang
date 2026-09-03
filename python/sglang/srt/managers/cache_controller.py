from __future__ import annotations

"""
Copyright 2023-2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import sys
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING, List, NamedTuple, Optional

import torch

from sglang.srt.mem_cache.hicache_phase_guard import device_tier_disarmed
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    compute_model_identity_hash,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.pool_host import HostKVCache

from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)

device_module = get_device_module()


class LayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_events = [device_module.Event() for _ in range(num_layers)]
        self.start_event = device_module.Event()  # start event on controller stream

    def complete(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        self.load_events[layer_index].record()

    def wait(self, layer_index: int):
        device_module.current_stream().wait_event(self.load_events[layer_index])

    @property
    def finish_event(self):
        return self.load_events[-1]


class LayerDoneCounter:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # extra producer and consumer counters for overlap mode
        self.num_counters = 3
        self.events = [LayerLoadingEvent(num_layers) for _ in range(self.num_counters)]
        self.producer_index = -1
        self.consumer_index = -1

    def update_producer(self):
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ].finish_event.query(), (
            "Producer finish event should be ready before being reused."
        )
        return self.producer_index

    def set_consumer(self, index: int):
        self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        self.events[self.consumer_index].wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1


class CacheOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
    ):
        self.host_indices = host_indices
        self.device_indices = device_indices
        self.node_ids = [node_id]
        # #760: EVERY op carries the binding it was built against, stamped at
        # construction rather than at one enqueue site -- an op created by any
        # other path would otherwise be unstamped, and an unstamped op must be
        # refused, which would silently drop legitimate write-backs.
        from sglang.srt.mem_cache.hicache_phase_binding import current_generation

        self.binding_generation = current_generation()
        self.data = None

        self.id = CacheOperation.counter
        CacheOperation.counter += 1
        # default priority is the order of creation
        self.priority = priority if priority is not None else self.id

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        assert len(ops) > 0
        if len(ops) == 1:
            return ops[0]

        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged_op = CacheOperation(host_indices, device_indices, -1, priority)
        merged_op.node_ids = node_ids
        return merged_op

    def __lt__(self, other: CacheOperation):
        return self.priority < other.priority


class HiCacheAck(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    node_ids: List[int]


#: #943: sentinel for "this operation has never been stamped", distinct from a
#: stamp of None (which `StorageOperation.__init__` records when the generation
#: could not be read). The two must not compare equal: an unstamped operation may
#: still receive its first stamp, while a None-stamped one has already been
#: through the constructor's except path and is refused by
#: `write_back_stamp_is_current` anyway.
_UNSTAMPED = object()


class StaleStampRewrite(RuntimeError):
    """A completed operation's binding generation was rewritten (#943).

    Never downgraded to a warning: the rewrite's only effect is to make a span
    fetched from a replaced host pool look publishable to #937's check.
    """


class StorageOperation:
    counter = 0

    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.host_indices = host_indices
        self.token_ids = token_ids
        self.last_hash = last_hash
        self.completed_tokens = 0
        self.hash_value = hash_value if hash_value is not None else []
        self.prefix_keys = prefix_keys
        # W35: the binding this operation was OPENED under. Stamped here, at
        # construction, because that is when its host slots were allocated --
        # so every consumer downstream (the backup/prefetch threads, the
        # revoke drain, the release queue) can ask which pool the slots came
        # from instead of assuming the currently-bound one. One authority: the
        # value comes from #719's generation and nowhere else.
        try:
            from sglang.srt.mem_cache.hicache_phase_binding import current_generation

            self.binding_generation = current_generation()
        except Exception:  # noqa: BLE001 - a stamp may never break an op
            self.binding_generation = None

        self.id = StorageOperation.counter
        StorageOperation.counter += 1

    #: #943: THE STAMP IS WRITE-ONCE, AND THAT IS THE WHOLE SAFETY PROPERTY.
    #
    # #937 refuses to PUBLISH a completed prefetch whose binding generation is
    # no longer current, because the span was fetched into a host pool that a
    # cutover has since replaced. Publishing it anyway is what the 2j soak
    # measured as garbage at every prompt at or above the 256-token prefetch
    # threshold, non-deterministic at temperature 0.
    #
    # The bisection (#943, window-943-bisect-0827) put that refusal at the exact
    # commit where the anti-correlation flips: merges 2/9/13 publish stale spans
    # and return 1/7 coherent, merges 15/16 refuse them and return 7/7. So the
    # refusal is the garbage fix, and the "lost anchors" are its cost.
    #
    # THE CHEAP-LOOKING WAY TO WIN THAT COST BACK IS THE ONE THAT MUST BE
    # IMPOSSIBLE. `write_back_stamp_is_current(operation.binding_generation)` is
    # the only thing standing between a stale span and the model, so anyone
    # restoring cache hits after a refusal can make the symptom disappear by
    # re-stamping the operation instead of re-fetching under the new binding.
    # That reads as "the re-issue works" and reinstates the corruption exactly:
    # same bytes, same replaced pool, now with a stamp that passes the check.
    #
    # A re-issue must therefore mint a NEW operation -- new stamp, new host
    # slots, a fresh fetch from the content-keyed store -- and never revive this
    # one. Making the rewrite raise is what keeps that a property of the code
    # rather than a note in a commit message. Idempotent writes are allowed so
    # the constructor and any equal re-assignment stay legal; only a CHANGE to
    # an already-stamped operation is refused.
    #
    # A property rather than `__setattr__`: this class carries hot fields
    # (`completed_tokens` is updated per transfer), and a Python-level hook on
    # every attribute write would tax all of them to guard one.
    @property
    def binding_generation(self):
        return self._binding_generation

    @binding_generation.setter
    def binding_generation(self, value):
        prev = getattr(self, "_binding_generation", _UNSTAMPED)
        if prev is not _UNSTAMPED and prev is not None and value != prev:
            raise StaleStampRewrite(
                f"refusing to re-stamp operation {getattr(self, 'id', '?')} from "
                f"binding generation {prev} to {value}. The host slots this "
                f"operation holds were allocated under {prev}; a cutover has "
                f"since rebound the tier, and re-stamping would let #937's "
                f"publish check pass for a span fetched out of a pool that no "
                f"longer exists -- the 2j garbage, restored. A re-issue mints a "
                f"NEW operation and fetches again; it never revives this one."
            )
        self._binding_generation = value

    def __lt__(self, other: StorageOperation):
        return self.id < other.id


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ):
        self.request_id = request_id

        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()

        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)

    #: #1157: the store probe's answer in tokens (MIN-reduced across the
    #: prefetch group), stamped by `prefetch_thread_func`; None until the
    #: probe has run. Read by the tree's reap line and the seam witness.
    probed_hit_tokens: Optional[int] = None

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


#: #1068 (WEG 1 slice 2): the speculative-prefetch budget in the upstream
#: ``buffer_only`` form (upstream c66a285c94 cache_controller.py:253 and
#: :575-584). A STAGING host tier lends this fraction of its rows to
#: prefetch; the remainder is the write-through ring's capacity
#: (staging_write_ring.py).
HICACHE_LOAD_POOL_USAGE_FRACTION = 0.9
#: A RETENTION host tier (upstream "cache mode", the else branch of upstream
#: c66a285c94 :575-584) keeps half for prefetch. The upstream literal is
#: 0.5; the name is this fork's (A12.5: no rename, the SHA carries the
#: provenance). Neither number is a stored quantity: see
#: ``HiCacheController.prefetch_capacity_limit``, a property of the pool the
#: controller is bound to RIGHT NOW, so a phase rebind that swaps
#: ``mem_pool_host`` needs no re-derivation pass and no floor.
HICACHE_CACHE_MODE_LOAD_FRACTION = 0.5

#: #1068 (WEG 1 slice 2 fix 3 + fix 4, spec A12.4 and its amendment): the
#: bound, in seconds, that ``HiCacheController._stop_storage_threads`` gives
#: EACH storage thread to leave its loop AFTER every in-flight prefetch
#: operation has been terminated. The helper and both loops are inherited
#: unchanged by ``HybridCacheController``
#: (mem_cache/hybrid_cache/hybrid_cache_controller.py), the controller the
#: serving path actually runs, and the bound holds by construction on BOTH
#: classes only because both carry the guard: a terminated operation aborts
#: ``_page_transfer`` at the next 128-page batch boundary (~4 MiB,
#: milliseconds; the hybrid override calls the base one first and inherits
#: that guard) and ``_storage_hit_query`` before its next ``batch_exists``
#: (per batch in the base class; at the top of the hybrid override, which
#: asks the store ONCE for the whole span). Without the termination a
#: 39365-token span in flight at the cutover needed ~12.6 s at the spec's N5
#: rate (0.32 s/KiToken) and outlived any bound of this size (review finding
#: (d)); without the hybrid guard a terminated probe still walked the whole
#: span on the store (review round 4, B1b).
#: PRE-EXISTING PEER-SKEW TERM, named and not removed here: the prefetch
#: loop drains its queue with one rank-uniform gloo MIN all_reduce per
#: operation (``_all_reduce_prefetch_groups`` in ``prefetch_thread_func``),
#: so the bound holds only if the peer rank's prefetch thread reaches the
#: paired collective within it. A thread that is STILL alive after this
#: bound is a torn pipeline, which is the raenge-nie-uneins
#: (ranks-never-disagree) crash-stop form: the helper raises, it never
#: returns.
STORAGE_THREAD_JOIN_BOUND_S = 10.0


class StorageStopResult(NamedTuple):
    """What ``_stop_storage_threads`` did, for the ONE '#1068 RESET JOIN' line
    ``reset()`` prints: which thread roles were joined, how many distinct
    prefetch operations were terminated before the join, how many operations
    the two loops still consumed after the stop event (each one cheap,
    because terminated), and the seconds the joins took."""

    threads: List[str]
    terminated_ops: int
    drained_ops: int
    joined_s: float


def canonical_identity_hash_for(server_args, canonical_page: bool) -> str:
    """#706 remainder: the identity hash for a key that must be geometry-FREE.

    Under ``--phase-flip-canonical-kv-page`` the stored page holds EVERY
    attention layer at full width, so no parallel split can change its bytes.
    The key already drops the tp and pp suffixes for exactly that reason. The
    identity hash was not given the same treatment: its
    ``include_parallel_vectors`` defaults to True, so ``rank_tp_ratio`` and
    ``rank_kv_ratio`` -- which say how the KV is SPLIT across ranks, not what a
    canonical page contains -- were still moving the key.

    Live, not hypothetical: the harvest boot ran ``rank_tp_ratio=None``
    (falsy, skipped) but ``rank_kv_ratio='coupled'`` (truthy, appended), so a
    geometry term was in the key today. Two boots of the same model and
    kv-dtype with different kv-ratios write byte-identical canonical pages and
    miss each other -- which is precisely the cross-boot retention the whole
    format exists to provide.

    WHAT IS NOT DROPPED, and must never be: revision, dtype, quantization and
    kv_cache_dtype. Those describe the BYTES, and confusing two byte formats is
    the silent wrong hit the hash was introduced to turn into a clean miss.
    Only the parallel tail goes, and only when the canonical format is on --
    without it a stage's file really does hold just that stage's layers and
    geometry belongs in the key.
    """
    return compute_model_identity_hash(
        server_args, include_parallel_vectors=not canonical_page
    )


def consume_gate(controller, queue_attr: str, direction: str) -> bool:
    """May the batch queued in ``queue_attr`` be CONSUMED now? (#760/#719)

    ONE AUTHORITY, FOUR CALLERS. The write path had this logic inline and the
    load path had none; the hybrid subclass had neither. That is four sites
    that must agree about one question, which is exactly the shape that cost
    W32 (a second copy of a rule silently overriding the first) -- so the
    question is asked in one place and the four consume points call it.

    THE TWO CHECKS ARE NOT REDUNDANT, and #760 records why both are needed:

    * ``device_tier_disarmed(direction)`` -- the phase predicate. ``write()``
      and ``load()`` ask it at ENQUEUE and get the right answer there: the copy
      is queued while the model computes in the phase these pools are bound to.
      Nothing re-asked afterwards, and the cutover lands in between. Both #760
      crash specimens died three seconds AFTER a pp_to_tp cutover completed.
    * the binding generation stamp -- which cannot cover it alone: with
      ``--phase-flip-rebind-hicache`` off the binding never advances, so every
      stamp matches by construction and the check is dead code.

    WHY THE LOAD SIDE MATTERS AT LEAST AS MUCH AS THE WRITE SIDE, and why its
    absence was worse: a stale WRITE corrupts the host copy and is caught by
    the pool's own double-free/ownership assertions or simply persists wrong
    bytes. A stale LOAD fills device rows from host slots the incoming phase
    does not own and the tree then marks that prefix RESIDENT -- attention
    reads KV nobody wrote, with no assertion anywhere. A silent wrong answer,
    which this codebase ranks worse than a crash.

    ``check_shapes`` cannot substitute for either: under ``layer_first`` a
    stale binding is shape-IDENTICAL to the live one, which is why #760 records
    the shape guard armed on three ranks, refusing zero, and a SIGSEGV anyway.

    Refusing costs a cache MISS later -- the same cheap failure the #718 disarm
    already accepts. Proceeding costs the scheduler.

    Returns True to proceed. Returns False after CLEARING the queue: the batch
    is refused, loudly and counted by name.
    """
    from sglang.srt.mem_cache.hicache_phase_binding import (
        write_back_stamp_is_current,
    )

    queue = getattr(controller, queue_attr, None)
    if not queue:
        return False

    # W36 rung 3: COUNT EVERY CHECK, not only every refusal. W36 logged zero
    # refusals across eight flips and the rung was INCONCLUSIVE, because
    # "nothing crossed a cutover" and "this gate was never reached" produced
    # byte-identical logs. Counting the checks makes clean and blind
    # distinguishable; the seam prints the pair once per cutover.
    controller._gate_checked = getattr(controller, "_gate_checked", 0) + len(queue)
    label = direction.upper()
    if device_tier_disarmed(direction):
        attr = f"_{direction}_phase_refusals"
        setattr(controller, attr, getattr(controller, attr, 0) + len(queue))
        n = getattr(controller, attr)
        if n <= 3 or n % 200 == 0:
            logger.warning(
                "#760 %s REFUSED AT CONSUME: the phase moved after these %d "
                "operation(s) were queued, so their device indices name the "
                "pool of a phase that is no longer computing. Dropping them; "
                "those prefixes miss later. (%d so far.)",
                label,
                len(queue),
                n,
            )
        controller._gate_refused = getattr(controller, "_gate_refused", 0) + len(queue)
        queue.clear()
        return False

    fresh = [
        o
        for o in queue
        if write_back_stamp_is_current(getattr(o, "binding_generation", None))
    ]
    if len(fresh) != len(queue):
        dropped = len(queue) - len(fresh)
        attr = f"_{direction}_stamp_refusals"
        setattr(controller, attr, getattr(controller, attr, 0) + dropped)
        n = getattr(controller, attr)
        if n <= 3 or n % 200 == 0:
            logger.warning(
                "#760 %s REFUSED: %d queued operation(s) were stamped against "
                "an older binding generation and are dropped rather than "
                "consumed after the rebind. Those prefixes miss later. "
                "(%d so far.)",
                label,
                dropped,
                n,
            )
        controller._gate_refused = getattr(controller, "_gate_refused", 0) + dropped
        setattr(controller, queue_attr, fresh)
    return bool(getattr(controller, queue_attr))


def gate_heartbeat(controller) -> str:
    """ "checked N, refused M" for this flip epoch, and reset for the next.

    W36 RUNG 3 EXISTS BECAUSE THIS DID NOT. Every stale-generation gate logged
    only on REFUSAL, so a boot with zero refusals looked exactly like a boot
    whose gates were never reached -- and W36 produced precisely that log:
    eight cutovers, zero refusal lines, rung INCONCLUSIVE. The ambiguity was
    reintroduced by the very lines meant to remove it.

    Emitted from the SEAM, once per cutover, because the seam always runs. A
    gate that is never reached therefore still produces a line, reading
    ``checked=0`` -- which is the can-fail: unreachable is now visible instead
    of silent. Per epoch, not per operation, so it cannot become spam.
    """
    checked = int(getattr(controller, "_gate_checked", 0) or 0)
    refused = int(getattr(controller, "_gate_refused", 0) or 0)
    controller._gate_checked = 0
    controller._gate_refused = 0
    return f"checked={checked} refused={refused}"


def operation_is_stale(controller, operation, kind: str) -> bool:
    """Was ``operation`` opened under a binding that has since moved? (#719/W35)

    THE SIBLING OF ``consume_gate``, and deliberately in the same module: that
    one answers the question for a QUEUED BATCH at a consume point, this one
    for a SINGLE operation on a background thread. One authority, two shapes --
    the alternative is a third copy of the rule, which is what cost W32.

    REFUSAL IS THE ONLY SAFE VERB HERE, unlike the release path. A stale
    RELEASE can be routed to the pool its generation names, because that pool
    still owns those slots. A stale BACKUP cannot: it would read
    ``mem_pool_host.get_data_page(...)`` for host slots whose pool may since
    have been repurposed, and persist those bytes to a CONTENT-ADDRESSED store
    under a hash computed from the tokens it was opened with. The hash would
    not match the payload, every later reader would trust it, and the
    corruption OUTLIVES THE PROCESS. There is no version of routing that makes
    that safe, so the operation is declined and nothing is written.

    A DECLINED BACKUP IS A CORRECT NON-PERSIST, not a loss: the prefix simply
    misses later and is recomputed -- the same cheap failure the #718 disarm
    and the #760 write refusal already accept.

    THREAD BOUNDARY: both generations are read EXACTLY ONCE, here, at the
    decision point. The consumers are always-running background threads and
    the current generation is mutated by the cutover on another thread; a
    second read mid-persist could straddle a rebind and answer two different
    questions about one operation. One read, one answer, one operation.
    """
    controller._gate_checked = getattr(controller, "_gate_checked", 0) + 1
    stamped = getattr(operation, "binding_generation", None)
    if stamped is None:
        return False
    from sglang.srt.mem_cache.hicache_phase_binding import current_generation

    now = current_generation()
    if int(stamped) == int(now):
        return False
    controller._gate_refused = getattr(controller, "_gate_refused", 0) + 1
    attr = f"_{kind}_stale_refusals"
    setattr(controller, attr, getattr(controller, attr, 0) + 1)
    n = getattr(controller, attr)
    if n <= 5 or n % 100 == 0:
        logger.warning(
            "#719/W35 STALE %s REFUSED: request %s was opened under binding "
            "generation %s and the binding is now %s. Declining it -- its host "
            "slots belong to a pool that may have been repurposed, and "
            "persisting them would write bytes that do not match the "
            "content-addressed hash they would be stored under. The prefix "
            "misses later instead. (%d so far.)",
            kind.upper(),
            getattr(operation, "request_id", "?"),
            stamped,
            now,
            n,
        )
    return True


def _record_quiesce_poison(reason: str, name: str, exc: BaseException) -> None:
    """Register a poison-class drain failure with the process-wide record. #917.

    The #760 drain sits at a NAMED BOUNDARY of the cutover -- between the
    device-tier writeback that precedes it and the resident release that
    follows -- which makes it one of the few sites whose POSITION IN THE WALK
    is worth recording. `barlink_abort_gate.record_poison` is FIRST-WINS, so
    this competes with nothing: it either names the origin or defers to
    whoever observed the fault earlier.

    WHY THIS IS NOT AN ACCUSATION. A stream synchronize raising an
    illegal-access does not mean a copy on that stream made the bad access; a
    poisoned context raises at whatever call synchronizes next, and in BOTH
    0826 specimens the barlink watchdog had already reported the same fault on
    the same rank before this drain ran. Recording the site is how the
    attribution bracket gets an end, not how blame gets assigned.

    A MODULE FUNCTION, not a method, and that is load-bearing: the drain's own
    hermetic harness calls `HiCacheController.quiesce_device_io` unbound
    against a `types.SimpleNamespace` (`test_flip_seam_guard_760.py`), where a
    `self.` lookup for this helper would raise `AttributeError` from inside an
    exception handler on the no-return path. The shape of the existing test is
    the shape the drain must survive.

    Degrades, never raises.
    """
    try:
        from sglang.srt.distributed.device_communicators import barlink_abort_gate

        if barlink_abort_gate.is_poison_error(exc):
            barlink_abort_gate.record_poison(
                f"#760 device-tier quiesce ({reason}, {name})", exc
            )
    except Exception:  # noqa: BLE001 - pragma: no cover, the no-return path owns this
        pass


class HiCacheController:
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: HostKVCache,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        enable_storage_metrics: bool = False,
    ):
        self.tp_group = tp_group
        self.attn_cp_group = attn_cp_group
        self.attn_tp_group = attn_tp_group
        self.pp_group = pp_group
        self.prefetch_sync_groups: List[torch.distributed.ProcessGroup] = []
        self.mem_pool_device_allocator = token_to_kv_pool_allocator
        mem_pool_device = token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        # #706 keeps the WRAPPER too: only the hybrid pool knows its layers by
        # GLOBAL id (full_attention_layer_id_mapping / mamba_map). The unwrapped
        # full_kv_pool reports start_layer 0 and a bare layer count on every
        # stage, which cannot distinguish stage 1 from stage 0.
        self.mem_pool_device_hybrid = mem_pool_device
        if isinstance(mem_pool_device, HybridLinearKVPool):
            mem_pool_device = mem_pool_device.full_kv_pool
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host
        # #706 x #719 (0828): attach-time constants for the cutover window
        # rebuild (`rebind_canonical_windows`). Model constants, resolved once
        # at storage attach; the PER-PHASE quantities (layer ranges, head
        # shards, page widths) are re-read from the live pools at every
        # rebuild instead.
        self._canonical_server_args = None
        self._canonical_model_config = None
        self._canonical_attn_layer_ids = None
        self._canonical_mamba_spec = None
        self._canonical_mamba_layer_ids = None
        # #923: the owner rule this controller translates device slots with is
        # a CUTOVER-DEPENDENT quantity, not a boot constant. Join the #297
        # registry here so every installer of a token vector -- the reshard
        # cutover and the phase flip alike -- drops the memo through the same
        # call, instead of each site remembering to reach in and delete it.
        self._register_owner_bounds_refresh()
        self.write_policy = write_policy
        self.page_size = page_size
        self.io_backend = io_backend
        self.enable_storage = False
        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage_metrics = enable_storage_metrics
        # #1068: the host tier's role, copied from the storage config at
        # attach (``_generate_storage_config`` is the one reader on the
        # controller; build_staging_write_ring and phase_flip_boot read the
        # --hicache-host-role flag too). Retention until a backend is
        # attached, matching that reader's default.
        self.host_role: str = "retention"

        # Draft KV pool support (best-effort piggyback on target L2/L3 ops).
        self.has_draft = False
        self.mem_pool_device_draft = None
        self.mem_pool_host_draft = None
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # #861: the phase that owns the drafter (None on a single-phase
        # instance), the #719 binding generation the registration was minted
        # at, and the drafter-identity suffix its persisted pages carry. All
        # three are read by ``draft_tier_armed`` and nowhere else.
        self.draft_owner_phase = None
        self.draft_binding_generation = None
        self.draft_identity = None
        self._draft_disarm_warned = set()
        # #993 EXECUTION PROOF for the draft READ path. The write half was
        # observable from the store (pages on disk); the read half had no
        # instrument at all, so "the draft tier is armed" and "draft pages
        # actually come back" were indistinguishable from outside. Counted
        # here, reported by `_log_draft_l3_progress`.
        self._draft_l3_hits = 0
        self._draft_l3_misses = 0
        self._draft_l3_logged_at = -1

        # Default storage page IO functions (may be overridden by attach).
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set

        # Dedicated stop event for storage background threads (prefetch/backup).
        self.storage_stop_event = threading.Event()
        # #1068 (A12.4): the operation each storage loop is working on RIGHT
        # NOW, one pointer per loop, written only by that loop. Together with
        # the two queues this is the controller's complete view of in-flight
        # prefetch work, which is what `_stop_storage_threads` terminates
        # before it joins. Not a registry: an operation is here only while a
        # loop holds it, and the tree's `ongoing_prefetch` stays the record.
        self._prefetch_current: Optional[PrefetchOperation] = None
        self._prefetch_io_current: Optional[PrefetchOperation] = None
        # Operations a loop consumed AFTER the stop event was set (one counter
        # per loop, single writer each); summed into the RESET JOIN line.
        self._prefetch_drained_after_stop = 0
        self._prefetch_io_drained_after_stop = 0

        self.device = self.mem_pool_device.device
        self.layer_num = self.mem_pool_device.layer_num
        self.layer_done_counter = LayerDoneCounter(self.layer_num)
        self.mem_pool_device.register_layer_transfer_counter(self.layer_done_counter)

        if write_policy not in [
            "write_through",
            "write_through_selective",
            "write_back",
        ]:
            raise ValueError(f"Invalid write policy: {write_policy}")

        # self.write_queue = PriorityQueue[CacheOperation]()
        self.load_queue: List[CacheOperation] = []
        self.write_queue: List[CacheOperation] = []
        self.ack_load_queue: List[HiCacheAck] = []
        self.ack_write_queue: List[HiCacheAck] = []

        self.write_stream = device_module.Stream()
        self.load_stream = device_module.Stream()

        # If a storage backend is provided at startup, treat it as an implicit attach,
        # so init/runtime share the same lifecycle semantics and code paths.
        if storage_backend is not None:
            try:
                self.attach_storage_backend(
                    storage_backend=storage_backend,
                    prefetch_threshold=prefetch_threshold,
                    model_name=model_name,
                    storage_backend_extra_config=storage_backend_extra_config,
                )
            except ValueError as e:
                # Preserve the historical error shape on init for unknown backends.
                raise ValueError(f"Failed to create storage backend: {e}") from e

    def get_attn_cp_rank_and_size(self) -> tuple[int, int]:
        """Derive CP rank/size from the attn_cp process group."""
        if self.attn_cp_group is not None:
            return (
                torch.distributed.get_rank(group=self.attn_cp_group),
                torch.distributed.get_world_size(group=self.attn_cp_group),
            )
        return 0, 1

    def _create_prefetch_sync_groups(self) -> None:
        from sglang.srt.distributed.parallel_state import create_custom_parallel_group

        self.prefetch_sync_groups = []
        seen_rank_sets = set()

        if self.attn_cp_group is not None or self.attn_tp_group is not None:
            base_groups = [self.attn_cp_group, self.attn_tp_group]
        else:
            base_groups = [self.tp_group]

        for group in base_groups:
            if group is None or torch.distributed.get_world_size(group=group) == 1:
                continue
            group_ranks = tuple(torch.distributed.get_process_group_ranks(group))
            if group_ranks in seen_rank_sets:
                continue
            seen_rank_sets.add(group_ranks)
            self.prefetch_sync_groups.append(
                create_custom_parallel_group(
                    group_ranks=list(group_ranks), backend="gloo"
                )
            )

    def _destroy_prefetch_sync_groups(self) -> None:
        for group in self.prefetch_sync_groups:
            try:
                torch.distributed.destroy_process_group(group)
            except Exception:
                pass
        self.prefetch_sync_groups = []

    def _all_reduce_prefetch_groups(self, tensor: torch.Tensor, op) -> None:
        for group in self.prefetch_sync_groups:
            torch.distributed.all_reduce(tensor, op=op, group=group)

    def _start_storage_threads(self):
        """Start storage prefetch/backup threads and their queues.

        This is used by runtime attach, and also by reset when storage is enabled.
        """
        assert self.enable_storage
        assert not self.storage_stop_event.is_set()

        self.prefetch_thread = threading.Thread(
            target=self.prefetch_thread_func, daemon=True
        )
        self.backup_thread = threading.Thread(
            target=self.backup_thread_func, daemon=True
        )
        self.prefetch_queue = Queue()
        self.backup_queue = Queue()

        self.prefetch_revoke_queue: Queue[str] = Queue()
        self.ack_backup_queue: Queue[StorageOperation] = Queue()
        self.host_mem_release_queue: Queue[torch.Tensor] = Queue()
        # #989: first-slot -> "module:lineno" of whoever queued that page.
        # Read by `_drain_release` when it finds a duplicate or an
        # already-free slot, so the crash names PRODUCERS and not only slots.
        self.host_release_provenance: dict[int, str] = {}
        # #1068 (A12.4): a fresh pipeline holds nothing and has drained nothing.
        self._prefetch_current = None
        self._prefetch_io_current = None
        self._prefetch_drained_after_stop = 0
        self._prefetch_io_drained_after_stop = 0

        self.prefetch_thread.start()
        self.backup_thread.start()

    def _terminate_inflight_prefetch(self) -> int:
        """#1068 (A12.4): mark every prefetch operation the controller still
        holds as terminated, and return how many DISTINCT operations that was.

        The controller's complete view of in-flight prefetch work is four
        places: ``prefetch_queue`` (registered, not yet probed), the prefetch
        loop's current operation (inside ``_storage_hit_query``),
        ``prefetch_buffer`` (probed, not yet transferred) and the aux loop's
        current operation (inside ``_page_transfer``). An operation that is
        in none of them has already completed or been revoked, so a record
        the tree still keeps in ``ongoing_prefetch`` for it needs no
        termination -- which is why ``UnifiedRadixCache._reset_full`` may
        clear that dict and delegate to ``reset()`` instead of walking it.

        Must run AFTER ``storage_stop_event`` is set: both loops terminate on
        their own whatever they dequeue after the stop, so the only operations
        this pass can miss are ones dequeued before the stop, and those are
        either still current (visible through the pointer) or already in the
        next queue (visible through its deque). The queues are read under
        their own mutex without dequeuing, so the loops keep the drain (the
        prefetch loop carries a rank-uniform gloo all_reduce per operation;
        a drain from the caller's thread would run that collective on one
        rank and not on a peer whose loop already holds the operation, the
        #580/#645 rank-divergence class).

        CLASS-AGNOSTIC on the operation (slice 2 fix 4, A12.4 amendment (a)):
        ``HybridCacheController``, the controller the serving path runs,
        queues its own ``hybrid_cache_controller.PrefetchOperation`` -- an
        upstream twin hierarchy (0986bed8e2) that does NOT subclass this
        module's ``PrefetchOperation`` and is not reparented. The pass keys
        on the termination protocol (``mark_terminate``/``is_terminated``),
        never on isinstance against the base class, which saw none of them
        and terminated nothing on the serving path (review round 4, B1).
        """
        seen: dict[int, object] = {}

        def take(op) -> None:
            if op is None:
                return
            if callable(getattr(op, "mark_terminate", None)) and callable(
                getattr(op, "is_terminated", None)
            ):
                seen.setdefault(id(op), op)

        for name in ("prefetch_queue", "prefetch_buffer"):
            q = getattr(self, name, None)
            if q is None:
                continue
            with q.mutex:
                for op in list(q.queue):
                    take(op)
        # getattr defaults: a controller stopped between attach/detach and its
        # first _start_storage_threads has not published these pointers yet.
        take(getattr(self, "_prefetch_current", None))
        take(getattr(self, "_prefetch_io_current", None))
        for op in seen.values():
            op.mark_terminate()
        return len(seen)

    def _stop_storage_threads(self) -> StorageStopResult:
        """Stop the storage prefetch/backup/io-aux threads: terminate every
        in-flight prefetch operation, then join each thread within
        ``STORAGE_THREAD_JOIN_BOUND_S``.

        THE ONE JOIN AUTHORITY, and its callers (#1068 graft G6 / A12.4):
        ``reset()`` (the tree reset, which runs at EVERY phase-flip cutover
        through ``UnifiedRadixCache._reset_full`` -- the caller that turned
        the bound into a boot killer while nothing terminated the transfer
        in flight, review finding (d)), the #1025 cutover quiesce
        (``hicache_phase_binding._quiesce_storage_io``, same join without a
        preceding termination), ``attach_storage_backend`` (idempotency
        guard) and ``detach_storage_backend``. The old precondition "caller
        should ensure no in-flight requests" is therefore no longer a
        precondition: in-flight work is terminated HERE, before the join,
        so the bound holds by construction on BOTH controller classes: on
        ``HiCacheController`` a terminated operation leaves
        ``_page_transfer`` at the next batch boundary and
        ``_storage_hit_query`` at the next batch of ``batch_exists``; on
        ``HybridCacheController`` (the serving controller, which inherits
        this helper and both loops) ``_page_transfer`` calls the base one
        first and inherits that guard, and its full ``_storage_hit_query``
        override carries the same guard at its top (slice 2 fix 4). The
        terminate pass is class-agnostic on the operation (duck-typed on
        ``mark_terminate``/``is_terminated``) because the hybrid controller
        queues its own PrefetchOperation twin.

        PRE-EXISTING PEER-SKEW TERM, named and not removed here: the prefetch
        loop's drain carries one rank-uniform gloo MIN all_reduce per queued
        operation, so the bound holds only if the peer rank's prefetch thread
        reaches the paired collective inside it; a peer late by more than
        the bound trips the raise on this rank first.

        Returns what it did for the caller's log line. Raises RuntimeError
        ONLY when a thread is still alive after termination plus the bound:
        that is a torn pipeline, and starting a second set of threads on
        top of it would double the hazard -- raenge-nie-uneins
        (ranks-never-disagree): crash-stop, never compensation. A caller on
        the cutover path must let that RuntimeError propagate (see
        ``phase_flip_runtime.drop_prefix_tree_returning_rows``).
        """
        t0 = time.monotonic()
        # Always request stop. This is safe even when storage is already disabled,
        # and makes detach truly idempotent (previous partial detach may have left
        # threads alive).
        # NOTE: do NOT clear storage_stop_event unless threads have fully stopped; otherwise
        # a still-alive thread may resume and touch released state.
        self.storage_stop_event.set()

        # A12.4: terminate BEFORE the join, so the join is bounded by the
        # batch boundary and not by the length of the span in flight.
        terminated_ops = self._terminate_inflight_prefetch()

        # Best-effort wakeups so threads exit promptly even if blocked on queues.
        try:
            if hasattr(self, "prefetch_queue"):
                self.prefetch_queue.put_nowait(None)
            if hasattr(self, "backup_queue"):
                self.backup_queue.put_nowait(None)
            if hasattr(self, "prefetch_buffer"):
                self.prefetch_buffer.put_nowait(None)
        except Exception:
            pass

        # Bounded joins (threads are daemon, but join keeps state clean).
        threads = []
        if hasattr(self, "prefetch_thread"):
            threads.append(("prefetch", self.prefetch_thread))
        if hasattr(self, "backup_thread"):
            threads.append(("backup", self.backup_thread))
        if hasattr(self, "prefetch_io_aux_thread"):
            threads.append(("prefetch_io_aux", self.prefetch_io_aux_thread))

        for _, t in threads:
            try:
                t.join(timeout=STORAGE_THREAD_JOIN_BOUND_S)
            except Exception:
                pass

        alive = [
            role for role, t in threads if getattr(t, "is_alive", lambda: False)()
        ]
        if alive:
            logger.error(
                "Failed to stop HiCache storage threads cleanly: %s still alive "
                "%.1f s after %d in-flight prefetch operation(s) were terminated "
                "(bound_s=%d per thread)",
                alive,
                time.monotonic() - t0,
                terminated_ops,
                int(STORAGE_THREAD_JOIN_BOUND_S),
            )
            raise RuntimeError("Failed to stop HiCache storage threads cleanly.")

        return StorageStopResult(
            threads=[role for role, _ in threads],
            terminated_ops=terminated_ops,
            drained_ops=self._prefetch_drained_after_stop
            + self._prefetch_io_drained_after_stop,
            joined_s=time.monotonic() - t0,
        )

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        """Attach (enable) storage backend at runtime.

        Requirement: no in-flight requests. This call is expected to run on the scheduler
        thread (control path), not concurrently with prefetch/backup.
        """
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")

        # Defensive: a previous partial detach may have flipped `enable_storage` but
        # left background threads alive. Attaching on top of them is unsafe.
        try:
            self._stop_storage_threads()
        except Exception as e:
            raise RuntimeError(
                "Cannot attach storage backend: previous detach did not stop storage threads cleanly."
            ) from e

        # Rollback-safe init: if creation fails, keep controller state consistent
        # for future attach attempts.
        self.storage_backend_type = storage_backend
        from sglang.srt.mem_cache.utils import get_hash_str

        self.get_hash_str = get_hash_str
        self.storage_config = self._generate_storage_config(
            model_name, storage_backend_extra_config
        )
        # #1068: the role the budget property reads; same source as the
        # backend's own host_role field, never a second read of the flag.
        self.host_role = str(getattr(self.storage_config, "host_role", "retention"))
        # Weighted uneven-DCP owner mode: page files are owner-written and
        # rank-shared; only the file backend implements that key scheme, and
        # the per-page owner rule needs page_size == 1 (a multi-token page
        # would span owner ranks). Fail fast instead of silently writing an
        # allocation-dependent (corrupt) store (task #60).
        if self.storage_config.dcp_owner_mode:
            if storage_backend != "file":
                raise NotImplementedError(
                    "Weighted uneven-DCP HiCache storage currently supports "
                    f"only the 'file' backend, got '{storage_backend}'."
                )
            if self.page_size != 1:
                raise NotImplementedError(
                    "Weighted uneven-DCP HiCache storage requires page_size == 1, "
                    f"got {self.page_size}."
                )
        # #706 whole-page protocol: same two conditions, for the layer axis.
        # Only the file backend can assemble one page from several stages
        # (partial byte-range writes), and a canonical page is ONE token's
        # attention layers. The boot-time twin of this check lives in
        # ServerArgs._handle_phase_flip; both exist because this one also
        # covers a storage backend attached at RUNTIME, where no ServerArgs
        # validation runs.
        if self.storage_config.canonical_kv_page is not None:
            if storage_backend != "file":
                raise NotImplementedError(
                    "The #706 canonical KV page currently supports only the "
                    f"'file' backend, got '{storage_backend}'."
                )
            if self.page_size != 1:
                raise NotImplementedError(
                    "The #706 canonical KV page requires page_size == 1, got "
                    f"{self.page_size}."
                )
        # for MLA models, only one rank needs to backup the KV cache
        self.backup_skip = (
            self.storage_config.is_mla_model
            # todo: load balancing
            and self.storage_config.tp_rank != 0
        )

        # Use storage backend factory for dynamic backend creation
        from sglang.srt.mem_cache.storage import StorageBackendFactory

        try:
            self.storage_backend = StorageBackendFactory.create_backend(
                storage_backend, self.storage_config, self.mem_pool_host
            )
            self.storage_backend.register_mem_pool_host(self.mem_pool_host)

            self.enable_storage = True
            # todo: threshold policy for prefetching
            self.prefetch_threshold = max(prefetch_threshold, self.page_size)
            # #1068: the prefetch budget is the ``prefetch_capacity_limit``
            # PROPERTY of the bound host pool; nothing is stored here. This
            # counter is the number of tokens currently locked by registered
            # prefetches, updated by the main scheduler thread, and it is what
            # ``prefetch_rate_limited`` compares against the property (the
            # upstream cache-mode form; the method says why not the live one).
            self.prefetch_tokens_occupied = 0

            # Use dedicated gloo groups so storage prefetch sync is isolated
            # from other collectives and consistent across CPxTP participants.
            self._create_prefetch_sync_groups()

            # Select the get and set functions
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set

            if (
                self.storage_backend_type
                in ["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]
            ) or (
                self.storage_backend_type == "dynamic"
                and bool(self.storage_config.extra_config.get("interface_v1", 0))
            ):
                self.page_get_func = self._page_get_zero_copy
                self.page_set_func = self._page_set_zero_copy

            self._maybe_register_draft_with_storage()

            # Ensure stop_event is clear before starting threads.
            self.storage_stop_event.clear()
            self._start_storage_threads()
        except Exception:
            # Best-effort cleanup for partial init.
            try:
                self._stop_storage_threads()
            except Exception:
                pass
            self._destroy_prefetch_sync_groups()
            try:
                if (
                    hasattr(self, "storage_backend")
                    and self.storage_backend is not None
                ):
                    if hasattr(self.storage_backend, "close"):
                        self.storage_backend.close()
            except Exception:
                pass
            self.storage_backend = None
            self.storage_backend_type = None
            self.enable_storage = False
            self.page_get_func = self._generic_page_get
            self.page_set_func = self._generic_page_set
            self.draft_page_get_func = None
            self.draft_page_set_func = None
            raise

    def detach_storage_backend(self):
        """Detach (disable) storage backend at runtime.

        Requirement: no in-flight requests. This will stop storage threads and release
        the backend instance (best-effort close).
        """
        # Idempotent cleanup: even if `enable_storage` is already False,
        # we may still have leftover resources (threads/backend/process group) from a
        # previous partial detach. We attempt cleanup whenever possible.
        try:
            self._stop_storage_threads()
        except Exception as e:
            # Do not proceed tearing down backend/process group if threads are not
            # fully stopped; otherwise still-alive threads may touch released state.
            # Caller can retry detach.
            logger.exception("Stop storage threads failed: %s", e)
            # IMPORTANT: Do not silently succeed. Upper layers rely on exceptions here
            # to avoid flipping `enable_storage` flags while threads are still alive.
            raise RuntimeError("Stop storage threads failed; detach aborted.") from e

        # Best-effort destroy process groups created for storage ops.
        self._destroy_prefetch_sync_groups()

        # Best-effort close (some backends rely on GC/destructor).
        try:
            if (
                hasattr(self, "storage_backend")
                and self.storage_backend is not None
                and hasattr(self.storage_backend, "close")
            ):
                self.storage_backend.close()
        except Exception:
            logger.exception("Failed to close storage backend cleanly.")

        self.storage_backend = None
        self.storage_backend_type = None
        self.enable_storage = False
        self.page_get_func = self._generic_page_get
        self.page_set_func = self._generic_page_set
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        # Now it's safe to clear the stop event for future re-attach.
        self.storage_stop_event.clear()

    def _generate_storage_config(
        self,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
    ):
        if storage_backend_extra_config is None:
            storage_backend_extra_config = {}

        if is_dp_attention_enabled():
            self.tp_rank = get_parallel().attn_tp_rank
            self.tp_size = get_parallel().attn_tp_size
            self.dp_rank = get_attention_dp_rank()
        else:
            self.tp_rank = get_parallel().tp_rank
            self.tp_size = get_parallel().tp_size
            self.dp_rank = 0

        self.pp_rank = get_parallel().pp_rank
        self.pp_size = get_parallel().pp_size

        # Currently, NPUMLATokenToKVPool is the subclass of MLATokenToKVPool.
        # DeepSeekV4TokenToKVPool has compressed MLA-style rank-replicated cache
        # data. storage only needs rank 0 to write it back.
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        is_mla_model = isinstance(self.mem_pool_device, MLATokenToKVPool)
        is_compressed_mla_model = isinstance(
            self.mem_pool_device, DeepSeekV4TokenToKVPool
        )
        is_rank_replicated = is_mla_model or is_compressed_mla_model
        # Least Common Multiple among heterogeneous tp size
        tp_lcm_size = storage_backend_extra_config.pop("tp_lcm_size", None)
        should_split_heads = False

        if tp_lcm_size:
            assert (
                tp_lcm_size % self.tp_size == 0
            ), "tp_lcm_size must be divisible by tp_size."
            should_split_heads = (
                not is_rank_replicated
                and self.mem_pool_host.layout == "page_head"
                and tp_lcm_size > self.tp_size
            )

        attn_cp_rank, attn_cp_size = self.get_attn_cp_rank_and_size()

        # Page hashes cover token ids only and the backend suffix covers
        # served_model_name + parallel geometry; the KV byte format
        # (dtype/quantization/kv_cache_dtype) and the weights revision are in
        # neither. Persistent-tier entries outlive the process, so without
        # this hash a later run sharing served_model_name and storage location
        # could silently read pages written in another byte format. The
        # scheduler process publishes ServerArgs before any storage attach;
        # the fallback only covers bare-controller unit tests.
        try:
            server_args = get_server_args()
        except ValueError:
            server_args = None
        model_identity_hash = (
            canonical_identity_hash_for(
                server_args,
                bool(getattr(server_args, "phase_flip_canonical_kv_page", False)),
            )
            if server_args is not None
            else None
        )

        # #706: this rank's window in the geometry-neutral page. Built only
        # when the format is switched on, and built from the pools that KNOW
        # their global layer ids -- never inferred from a layer count, because
        # the host pool's own start_layer is 0 on every PP stage.
        canonical_kv_page = None
        canonical_mamba_blob = None
        if server_args is not None and getattr(
            server_args, "phase_flip_canonical_kv_page", False
        ):
            from sglang.srt.mem_cache.canonical_page_store import (
                build_page_window,
                resolve_attn_layer_ids,
            )

            model_config = server_args.get_model_config()
            # #706: the id list is RESOLVED, not guessed. The previous
            # `full_attention_layer_ids or range(n)` read an empty SWA-scoped
            # list as proof of a dense model and cut this GDN hybrid's page
            # against 64 slots instead of 16. See resolve_attn_layer_ids for
            # the ladder and why the fix is not in get_hybrid_layer_ids.
            attn_layer_ids = resolve_attn_layer_ids(model_config)
            # #706 x #719: recorded for the cutover window rebuild
            # (`rebind_canonical_windows`) -- resolved once, from the same
            # ladder the attach itself trusts.
            self._canonical_server_args = server_args
            self._canonical_model_config = model_config
            self._canonical_attn_layer_ids = [int(i) for i in attn_layer_ids]
            canonical_kv_page = build_page_window(
                attn_layer_ids, self.mem_pool_device_hybrid, self.mem_pool_host
            )
            logger.info(
                "#706 canonical KV page active: slots [%d, %d) of %d, %d B per "
                "slot; KV keys carry content only (no tp/pp suffix).",
                canonical_kv_page.first_slot,
                canonical_kv_page.first_slot + canonical_kv_page.num_slots,
                canonical_kv_page.spec.num_attn_layers,
                canonical_kv_page.cell_bytes,
            )
            canonical_mamba_blob = self._canonical_mamba_window(
                server_args, model_config
            )

        return HiCacheStorageConfig(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=attn_cp_size,
            # TODO(hzh): Rename is_mla_model to is_rank_replicated.
            is_mla_model=is_rank_replicated,
            enable_storage_metrics=self.enable_storage_metrics,
            is_page_first_layout=self.mem_pool_host.layout == "page_first",
            model_name=model_name,
            model_identity_hash=model_identity_hash,
            tp_lcm_size=tp_lcm_size,
            should_split_heads=should_split_heads,
            extra_config=storage_backend_extra_config,
            # #810: a staging host tier makes this backend the retention tier.
            host_role=getattr(server_args, "hicache_host_role", "retention"),
            # Weighted uneven-DCP: KV pages are token-sharded with FULL
            # replicated kv-heads -> owner-written, rank-shared page files.
            dcp_owner_mode=self._dcp_owner_ctx() is not None,
            # #706: full-width pages, stage-offset writes, suffix-free KV keys.
            canonical_kv_page=canonical_kv_page,
            canonical_mamba_blob=canonical_mamba_blob,
        )

    def _canonical_mamba_window(self, server_args, model_config, phase=None):
        """This rank's window in the canonical GDN blob (#706 slice 2).

        ``None`` only when the model has NO linear/GDN layers -- then there is
        no blob and the KV page alone is the whole prefix. For a hybrid model
        this must succeed or attach fails: a canonical KV page beside a
        phase-local GDN blob delivers ZERO usable prefix, because
        ``batch_exists_v2`` takes the minimum across pools and the mamba pool is
        registered TRAILING_PAGES. Silently running KV-only would look like the
        feature was on while every cross-phase lookup missed.
        """
        from sglang.srt.mem_cache.canonical_page_store import (
            CanonicalPageError,
            build_mamba_window,
            derive_mamba_blob_spec,
            local_mamba_layer_range,
            resolve_linear_layer_ids,
        )

        # #931: RESOLVED, not guessed -- the same ladder the KV half above got,
        # for the same reason. The line that stood here read
        # `getattr(model_config, "mamba2_cache_params", None)`, and that
        # property belongs to the CHECKPOINT config, never to sglang's
        # ModelConfig (`grep -c` in configs/model_config.py is 0). It therefore
        # missed on every model, the id list was always empty, and the
        # `return None` below fired on every boot since a38f39f1ee -- so this
        # blob has never once attached. Boot 2g proves it from the other end:
        # "#706 canonical KV page active" x3, "#706 canonical GDN blob active"
        # x0, refusals x0, with the format explicitly armed.
        #
        # An empty list is now a PROVEN "this model has no linear layers"
        # (ladder step (c)); every unresolvable hybrid raises there instead of
        # arriving here as a silent skip. That is what makes the `return None`
        # below sound rather than the hole it was.
        mamba_layer_ids = getattr(self, "_canonical_mamba_layer_ids", None)
        if mamba_layer_ids is None:
            mamba_layer_ids = resolve_linear_layer_ids(model_config)
        if not mamba_layer_ids:
            return None

        hybrid = self.mem_pool_device_hybrid
        mamba_pool = getattr(hybrid, "mamba_pool", None)
        if mamba_pool is None:
            raise NotImplementedError(
                "#706: this model has GDN/linear layers but the KV pool exposes "
                "no mamba pool, so the GDN blob cannot be made phase-uniform. "
                "Refusing rather than serving a store whose KV pages are "
                "geometry-free while its GDN blobs are not -- that combination "
                "hits nothing at all."
            )
        spec = getattr(self, "_canonical_mamba_spec", None)
        if spec is None:
            spec = derive_mamba_blob_spec(
                model_config, mamba_pool, num_linear_layers=len(mamba_layer_ids)
            )
        # #931 second hop: `mamba_pool`, NOT `hybrid`. The parameter is named
        # `mamba_pool` and the callee's docstring says `MambaPool.mamba_map`;
        # what was passed was the KV pool (HybridLinearKVPool), which holds a
        # `.mamba_pool` but no map of its own, so the read missed and the blob
        # refused to attach -- caught on metal by the refusal the first half of
        # #931 installed, which is what it was for. Both calls now take the
        # SAME object, which is what removes the asymmetry that allowed one of
        # them to be wrong while the other was right.
        layer_lo, layer_hi = local_mamba_layer_range(mamba_pool, mamba_layer_ids)
        # Head sharding: candidate (ratios, rank) resolutions, VERIFIED
        # against the pool the read path actually fills, in a fixed order.
        #
        # #706 x #719 (0828): in the flip's TP phase the mamba states are
        # head-sharded by the SAME vector that sizes the TP pools
        # (--phase-flip-tp-vector; the #871 staging pin logs 2:1:1 mamba
        # halves for 32,16,16), and this process's shard is its boot rank's
        # entry -- the flip reuses the same ranks (server_args.py). The
        # attach-time identity (tp_size=1 under the PP boot) cannot see that,
        # so the PHASE selects the primary candidate. The pre-0828 resolution
        # (rank_tp_ratio when set, equal shares otherwise) stays as the
        # attach-time and fallback rule -- under the PP phase it is [1] and
        # the window carries full heads, exactly the layer-only cut.
        candidates = []
        raw_ratio = getattr(server_args, "rank_tp_ratio", None)
        if raw_ratio:
            candidates.append(
                ([int(x) for x in str(raw_ratio).split(",")], int(self.tp_rank))
            )
        flip_vector = getattr(server_args, "phase_flip_tp_vector", None)
        if phase == "tp" and flip_vector:
            candidates.insert(
                0,
                (
                    [int(x) for x in str(flip_vector).split(",")],
                    int(getattr(self, "pp_rank", 0) or 0),
                ),
            )
        candidates.append(
            ([1] * max(1, int(self.tp_size)), int(self.tp_rank))
        )
        # The CURRENT mamba host pool's page is the authority a candidate has
        # to match: `_read_page` fills exactly that buffer, so a window of any
        # other size would refuse at every read. Verified here, once, instead
        # of once per refused page. A host tier that exposes no mamba entry
        # (bare-controller attach, non-grouped pools) keeps the first
        # candidate -- the attach-time behaviour, byte-identical.
        expected_bytes = None
        # getattr on self too: a bare controller probe (the #931 attach tests)
        # legitimately carries no host pool, and verification is best-effort
        # by design -- absence of the authority skips the check, it never
        # invents one.
        host_group = getattr(self, "mem_pool_host", None)
        entry_names = getattr(host_group, "entry_map", None) or {}
        get_pool = getattr(host_group, "get_pool", None)
        if get_pool is not None and PoolName.MAMBA in entry_names:
            try:
                probe_page = get_pool(PoolName.MAMBA).get_dummy_flat_data_page()
                expected_bytes = int(probe_page.numel()) * int(
                    probe_page.element_size()
                )
            except Exception:  # noqa: BLE001 - verification is best-effort
                expected_bytes = None
        window = None
        tried = []
        for ratios, rank in candidates:
            candidate = build_mamba_window(
                spec,
                ratios=ratios,
                rank=rank,
                layer_lo=layer_lo,
                layer_hi=layer_hi,
            )
            tried.append((list(ratios), int(rank), candidate.payload_bytes))
            if expected_bytes is None or candidate.payload_bytes == expected_bytes:
                window = candidate
                break
        if window is None:
            raise CanonicalPageError(
                f"no head-shard candidate cuts a mamba window matching the "
                f"bound mamba host pool's page of {expected_bytes} bytes "
                f"(tried (ratios, rank, payload): {tried}). Such a window "
                f"would refuse at every read; refusing to install it instead."
            )
        self._canonical_mamba_layer_ids = [int(i) for i in mamba_layer_ids]
        self._canonical_mamba_spec = spec
        logger.info(
            "#706 canonical GDN blob active: layers [%d, %d) of %d, %d of %d "
            "blob bytes on this rank, %d extent(s).",
            layer_lo,
            layer_hi,
            spec.num_layers,
            window.payload_bytes,
            window.total_bytes,
            len(window.extents),
        )
        return window

    def rebind_canonical_windows(self, incoming_phase: str) -> bool:
        """#706 x #719 (0828 specimen): re-derive the canonical windows from
        the pools the cutover has just bound.

        The windows are DERIVED QUANTITIES of the pool geometry: the KV
        window from the host pool's own flat page and the device pool's
        global layer ids, the mamba window from the current pool's layer map
        and this rank's head shard. `_generate_storage_config` builds them
        ONCE, at attach, from the pools bound at that moment -- and the #719
        rebind swaps every reader's pools without touching the storage
        backend. Measured on boot_943bx_0828: after the first pp_to_tp rebind
        (generation 1, 09:14:10) every canonical read refused
        deterministically ("read target holds 32768 bytes but this KV page
        window is 8192 bytes", 128x at 09:25:23), the prefetch completed 0 on
        all three ranks, and the #928 anchor re-prefilled everything -- with
        24k healthy pages sitting in the store the whole time (#872).

        Called from `rebind_for_cutover` AFTER the pool swap committed, so it
        reads exactly the pools the readers now name. Returns False when
        there is nothing to rebuild (no storage, canonical format not armed,
        a backend without the installer). Raises on a geometry that cannot be
        derived: the caller logs it loudly, the old windows stay installed,
        and `batch_exists_v2`'s compatibility probe keeps that state an
        honest cold miss instead of a broken promise.
        """
        if not getattr(self, "enable_storage", False) or self.storage_backend is None:
            return False
        backend = self.storage_backend
        if getattr(backend, "canonical_kv_page", None) is None:
            return False
        install = getattr(backend, "install_canonical_windows", None)
        if install is None:
            return False
        from sglang.srt.mem_cache.canonical_page_store import (
            CanonicalPageError,
            build_page_window,
        )

        attn_layer_ids = getattr(self, "_canonical_attn_layer_ids", None)
        if not attn_layer_ids:
            raise CanonicalPageError(
                "the attach never recorded the model's attention layer ids, "
                "so the canonical windows cannot be re-derived at the "
                "cutover."
            )
        kv_window = build_page_window(
            attn_layer_ids, self.mem_pool_device_hybrid, self.mem_pool_host
        )
        mamba_window = None
        if getattr(backend, "canonical_mamba_blob", None) is not None:
            mamba_window = self._canonical_mamba_window(
                self._canonical_server_args,
                self._canonical_model_config,
                phase=incoming_phase,
            )
        install(kv_window, mamba_window)
        # The v2 component reads/writes must land in the pools bound NOW, not
        # the pools bound at attach -- the same frozen-binding class one layer
        # down (`_batch_io_v2` resolves `registered_pools[name]`, registered
        # once at attach).
        backend.register_mem_pool_host(self.mem_pool_host)
        entries = getattr(self.mem_pool_host, "entries", None) or []
        for entry in entries:
            backend.register_mem_host_pool_v2(entry.host_pool, entry.name)
        import dataclasses

        self.storage_config = dataclasses.replace(
            self.storage_config,
            canonical_kv_page=kv_window,
            canonical_mamba_blob=mamba_window,
        )
        logger.info(
            "#706 canonical windows rebound for the '%s' phase: KV slots "
            "[%d, %d) of %d, mamba %s.",
            incoming_phase,
            kv_window.first_slot,
            kv_window.first_slot + kv_window.num_slots,
            kv_window.spec.num_attn_layers,
            (
                f"{mamba_window.payload_bytes} of {mamba_window.total_bytes} B"
                if mamba_window is not None
                else "none"
            ),
        )
        return True

    def reset(self):
        """Quiesce and restart the storage pipeline through the ONE join
        authority (#1068 graft G6).

        ``_stop_storage_threads`` joins the prefetch thread, the backup thread
        AND ``prefetch_io_aux_thread`` (the transfer thread that
        ``prefetch_thread_func`` starts); the hand-rolled joins this method
        carried before omitted the aux thread -- #1025's own docstring names
        it as the half ``reset()`` "hand-rolls and omits" -- so a tree reset
        at the cutover left it alive on the OUTGOING binding while the pools
        swapped (boot 22, #1052). ``_start_storage_threads`` rebuilds the
        queues fresh, which subsumes the per-queue clears that lived here.

        Slice 2 fix 3 (A12.4): the helper terminates every in-flight prefetch
        operation BEFORE it joins, so the join bound holds at the batch
        boundary of the span in flight and not at its length; this method
        prints the ONE line that names the bound, the joined thread roles,
        the terminated and drained operation counts and the seconds spent.
        """
        stop = self._stop_storage_threads()
        logger.info(
            "#1068 RESET JOIN bound_s=%d threads=%s terminated_ops=%d "
            "drained_ops=%d joined_s=%.2f",
            int(STORAGE_THREAD_JOIN_BOUND_S),
            stop.threads,
            stop.terminated_ops,
            stop.drained_ops,
            stop.joined_s,
        )

        self.write_queue.clear()
        self.load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()

        self.storage_stop_event.clear()

        if self.enable_storage:
            # N2 (spec section 12): the restart helper does not touch the
            # instrument counter, so it is zeroed here with the pipeline.
            self.prefetch_tokens_occupied = 0
            self._start_storage_threads()

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Back up KV caches from device memory to host memory.
        """
        # #718: this controller is bound to the pool it was BUILT with. While
        # the flip routes to its TP stack, that is not the pool the model
        # writes into, so this copy would persist another row's bytes under a
        # content-addressed key. Refuse: a prefix that is not staged is a miss
        # later, which is the cheap failure.
        if device_tier_disarmed("write"):
            return None
        # #923: and the row this copy would READ must be a row this rank's
        # device pool has. Asked before the host allocation, so a refusal
        # strands nothing.
        if self._refuse_unaddressable_kv_rows(device_indices, "write"):
            return None
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if len(self.write_queue) == 0:
            return

        # #760/W35: ONE AUTHORITY, FOUR CALLERS. This block used to live here
        # inline while the load path had no equivalent and the hybrid subclass
        # had neither -- four consume points that must agree about one
        # question. The question now lives in `consume_gate` and this is one of
        # its callers; see that function for the full #760 argument and for why
        # the phase predicate and the generation stamp are BOTH required.
        if not consume_gate(self, "write_queue", "write"):
            return

        op = CacheOperation.merge_ops(self.write_queue)
        # Kernel write-back keeps host indices on CPU only for page_first AND only
        # when the staged JIT write-back kernel is available (it stages through
        # device memory and accepts CPU destination indices). Otherwise we fall back
        # to the plain transfer kernel, whose CUDA/HIP implementation requires
        # device-resident destination indices -- so the indices must be moved to the
        # device first. Without the can_use_write_back_jit check this crashes on
        # backends where the JIT kernel is unavailable, with
        # "Destination indices must be a CUDA tensor".
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices, device_indices = op.host_indices, op.device_indices
        else:
            host_indices, device_indices = self.move_indices(
                op.host_indices, op.device_indices
            )
        self.write_queue.clear()

        start_event = device_module.Event()
        finish_event = device_module.Event()

        # Weighted uneven-DCP: back up only this rank's owned tokens, from
        # their COMPACT device slots (identity when the gate is off). The
        # draft pool below is NOT DCP-token-sharded (full token context,
        # global indices) and keeps the raw pair list.
        kv_host_indices, kv_device_indices = self._dcp_kv_transfer_pairs(
            host_indices, device_indices
        )

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                kv_host_indices,
                kv_device_indices,
                self.io_backend,
            )
            if self.draft_tier_armed("write"):
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            finish_event.record()
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the write stream is executing.
            for indices in (
                host_indices,
                device_indices,
                kv_host_indices,
                kv_device_indices,
            ):
                if indices.is_cuda:
                    indices.record_stream(self.write_stream)

        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Load KV caches from host memory to device memory.
        """
        # #718: the mirror hazard. This copy would fill rows in the bound (PP)
        # pool while the model reads the flip stack's, and the tree would
        # report the prefix resident -- so attention would read rows nobody
        # filled. Refuse: a prefetch that does not land is a miss now.
        if device_tier_disarmed("load"):
            return None
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        # #923: the write direction's sibling. The slots were just allocated,
        # so they are in the ALLOCATOR's id space by construction -- the
        # question this asks is whether the owner rule this controller holds
        # maps them into THIS pool's rows, which is exactly what a cutover
        # changes. Free the allocation on refusal: the caller only learns None.
        if self._refuse_unaddressable_kv_rows(device_indices, "load"):
            self.mem_pool_device_allocator.free(device_indices)
            return None
        self.load_queue.append(
            CacheOperation(host_indices, device_indices, node_id, priority)
        )
        return device_indices

    def _dcp_owner_ctx(self) -> Optional[tuple]:
        """(S, lo, hi) of this rank's weighted uneven-DCP owner range, or None.

        Memoized, and the memo is DROPPED at every token-vector cutover:
        ``refresh_dcp_owner_bounds`` is the #297 registry's hook and this
        controller registers for it in ``__init__``. When active, the radix
        tree hands the controller GLOBAL allocator indices while the device KV
        pool only holds this rank's COMPACT owned slots -- every device-side KV
        transfer must go through _dcp_kv_transfer_pairs (task #60).

        #923: the memo used to be taken once for process life, on the reasoning
        "the token vector is installed once at engine init". The phase flip
        falsifies that reasoning -- ``dcp_size`` goes 1 -> 3 at every pp_to_tp
        cutover and back, and ``phase_flip_runtime._cutover`` reinstalls the
        token vector on each leg. A memo taken in the PP phase says None, which
        makes _dcp_kv_transfer_pairs the IDENTITY, which hands the TP phase's
        compact device pool a GLOBAL slot id. Below the pool's row count that
        addresses a different token's row (silent wrong KV in the host tier);
        above it, ``at::Tensor::slice`` CLAMPS to an empty slice and the copy
        dies as "The size of tensor a (N) must match the size of tensor b (0)".
        """
        if not hasattr(self, "_dcp_owner_ctx_cache"):
            from sglang.srt.distributed.utils import uneven_dcp_owner_bounds

            self._dcp_owner_ctx_cache = uneven_dcp_owner_bounds()
        return self._dcp_owner_ctx_cache

    def refresh_dcp_owner_bounds(self) -> None:
        """#297 cutover hook (#923): forget the memoized owner ctx.

        The registry calls this on every registered consumer whenever a token
        vector is installed -- the #297 reshard cutover and the phase flip both
        go through ``refresh_all_owner_bounds()``. Dropping the attribute
        rather than re-deriving it here keeps the derivation in ONE place
        (:meth:`_dcp_owner_ctx`) and makes the next reader pay for it lazily,
        which is what a cutover wants: the groups are re-routed a few lines
        later in the same cutover.
        """
        self.__dict__.pop("_dcp_owner_ctx_cache", None)

    def _register_owner_bounds_refresh(self) -> None:
        """#923: join the ONE cutover registry instead of being invalidated by
        hand at each cutover site.

        There used to be exactly one hand-written invalidation, in
        ``kv_reshard._cutover_fn_for``, guarded by the comment "Stage A refuses
        to arm with hicache active, but a stale memo must still not survive".
        The phase flip is a second installer of the same payload and had no
        such line, so the memo survived every cutover on the live rig. A second
        bespoke ``delattr`` would have been a third mover of the same payload;
        registering is the reconciliation.
        """
        from sglang.srt.layers.dcp.owner import register_owner_bounds_consumer

        register_owner_bounds_consumer(self)

    def _dcp_owned_device_rows(self, device_indices: torch.Tensor):
        """``(owned_mask, compact_rows)`` for a vector of GLOBAL device slots.

        ``(None, device_indices)`` when the token-sharded gate is off, which is
        the identity the stock path relies on. THE single place the owner rule
        is applied to a HiCache transfer -- :meth:`_dcp_kv_transfer_pairs` and
        the #923 addressability refusal both read it, so the two can never
        disagree about which row a transfer would touch.
        """
        ctx = self._dcp_owner_ctx()
        if ctx is None:
            return None, device_indices
        S, lo, hi = ctx
        dev = device_indices.to(torch.int64)
        off = dev % S
        owned = (off >= lo) & (off < hi)
        compact = (dev // S) * (hi - lo) + (off - lo)
        return owned, compact[owned]

    def _kv_device_row_capacity(self) -> Optional[int]:
        """Rows of this rank's device KV pool, read off the buffer the transfer
        kernels actually slice. ``None`` when it cannot be read -- absence is
        not a mismatch, same contract as the #760 seam guard.
        """
        buffers = getattr(self.mem_pool_device, "k_buffer", None)
        try:
            return int(buffers[0].shape[0])
        except (TypeError, IndexError, AttributeError):
            return None

    def _refuse_unaddressable_kv_rows(self, device_indices, where: str) -> bool:
        """#923: True when this transfer would index a row the device KV pool
        does not have -- SAID, counted, and refused before anything commits.

        WHY REFUSE HERE AND NOT AT THE KERNEL. ``write()``/``load()`` may return
        None; that is the established contract ("a refused write is a prefix
        that misses later, which is the cheap failure") and, crucially, the
        caller has not yet marked the node backuped. A refusal further down --
        after ``commit_hicache_transfer`` -- would leave the tree believing the
        node is on the host tier while nothing was copied, which is the #767
        silent-wrongness direction and strictly worse than the crash it
        replaces. So: never a silent skip, and never a late one.
        """
        cap = self._kv_device_row_capacity()
        if cap is None or device_indices is None or device_indices.numel() == 0:
            return False
        _, rows = self._dcp_owned_device_rows(device_indices)
        if rows.numel() == 0:
            return False
        row_max = int(rows.max())
        row_min = int(rows.min())
        if row_min >= 0 and row_max < cap:
            return False
        n = getattr(self, "_unaddressable_kv_rows_refused", 0) + 1
        self._unaddressable_kv_rows_refused = n
        # Rate-limited on the same cadence as the other HiCache refusals in
        # this file: this can fire per operation, and an unbounded emitter is
        # its own outage. The COUNT is the finding, not the line.
        if n <= 3 or n % 200 == 0:
            logger.error(
                "#923 HICACHE %s REFUSED: the transfer would index device KV "
                "row(s) in [%d, %d] of a pool that has %d row(s). Owner ctx "
                "(S, lo, hi)=%s. A global allocator slot reaching a compact "
                "pool means the owner rule this controller holds is not the "
                "one the pools were built under -- the copy is refused rather "
                "than clamped to an empty slice (which dies in the kernel) or "
                "wrapped onto another token's row (which does not die at all). "
                "The affected prefix simply misses later. (%d so far.)",
                where,
                row_min,
                row_max,
                cap,
                self._dcp_owner_ctx(),
                n,
            )
        return True

    def _dcp_kv_transfer_pairs(
        self, host_indices: torch.Tensor, device_indices: torch.Tensor
    ):
        """Translate a (host page, GLOBAL device slot) pair list into the
        (host page, COMPACT device slot) pairs this rank actually owns.

        Weighted uneven-DCP stores token L's KV on the rank whose owner range
        [lo, hi) contains L % S, at compact physical slot
        (L // S) * (hi - lo) + (L % S - lo). Tokens outside [lo, hi) do not
        exist on this rank's device pool: their host pages are neither backed
        up nor loaded here (their L3 page file is written by the owner rank
        and holds the FULL replicated kv-heads). Gate off -> identity, keeping
        the stock path byte-identical."""
        owned, compact = self._dcp_owned_device_rows(device_indices)
        if owned is None:
            return host_indices, device_indices
        host_owned = host_indices[owned.to(host_indices.device)]
        return host_owned, compact

    def move_indices(self, host_indices: torch.Tensor, device_indices: torch.Tensor):
        # move indices to GPU if using kernels, to host if using direct indexing
        if self.io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(self.device, non_blocking=True)
            return host_indices, device_indices
        elif self.io_backend == "direct":
            if self.mem_pool_host.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                return host_indices, device_indices.index_select(0, idx)
            elif self.mem_pool_host.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            else:
                raise ValueError(
                    f"Unsupported layout {self.mem_pool_host.layout!r} for io backend 'direct'"
                )
        elif self.io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        else:
            raise ValueError(f"Unsupported io backend")

    def start_loading(self) -> int:
        if len(self.load_queue) == 0:
            return -1

        # #760/W35: THE CONSUME-TIME CHECKS THE LOAD PATH NEVER HAD.
        # `load()` asks the phase question at ENQUEUE and is right there; the
        # cutover lands between enqueue and here, and `load()` is a SEPARATE
        # call from this one, so the gap is real rather than theoretical. A
        # stale load fills device rows from host slots this phase does not own
        # and the tree marks the prefix RESIDENT -- attention then reads KV
        # nobody wrote, with no assertion anywhere. Checked before a producer
        # is allocated, so a refused batch costs nothing downstream.
        if not consume_gate(self, "load_queue", "load"):
            return -1

        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices = self.move_indices(
            op.host_indices, op.device_indices
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()

        # Weighted uneven-DCP: load only this rank's owned tokens into their
        # COMPACT device slots (identity when the gate is off). Draft pool is
        # not DCP-token-sharded and keeps the raw pair list.
        kv_host_indices, kv_device_indices = self._dcp_kv_transfer_pairs(
            host_indices, device_indices
        )

        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    kv_host_indices,
                    kv_device_indices,
                    i,
                    self.io_backend,
                )
                if (
                    self.draft_tier_armed("load")
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            # NOTE: We must save the host indices and device indices here,
            # this is because we need to guarantee that these tensors are
            # still alive when the load stream is executing.
            for indices in (
                host_indices,
                device_indices,
                kv_host_indices,
                kv_device_indices,
            ):
                if indices.is_cuda:
                    indices.record_stream(self.load_stream)

        self.ack_load_queue.append(
            HiCacheAck(
                start_event=producer_event.start_event,
                finish_event=producer_event.finish_event,
                node_ids=op.node_ids,
            )
        )
        return producer_id

    def evict_device(self, device_indices: torch.Tensor) -> int:
        self.mem_pool_device_allocator.free(device_indices)
        return len(device_indices)

    def quiesce_device_io(self, reason: str) -> float:
        """#760: finish every in-flight device-tier copy, bounded by the copies.

        Called by the flip runtime at the seam's no-return point, BEFORE any
        pool byte moves. The copies on ``write_stream`` / ``load_stream`` were
        enqueued while the pools they name were live, so FINISHING them is
        correct -- they become durable cache entries -- while letting them ride
        into the seam races the release of the outgoing phase's backing, which
        is the SIGSEGV both #760 crash specimens died of (3 s after a
        pp_to_tp cutover, inside backup_from_device_all_layer, below the
        Python seam). No Python-side predicate can close that window: the
        enqueue-time and consume-time checks both pass legitimately, the copy
        is torn by the STREAM's asynchrony.

        The wait is bounded by construction: a stream synchronize completes
        when the already-enqueued copies do (PCIe transfer time of the
        backlog), and this thread is the only producer of device-tier I/O, so
        nothing refills the streams while it waits. Not a collective -- each
        rank drains its own streams -- so it cannot wedge the group (#630).

        Returns the wait in seconds; the caller logs it into the seam record
        so a slow drain is attributable instead of vanishing into the flip's
        residual (#690).
        """
        # (#917) A DRAIN THAT DID NOT DRAIN MUST NOT REPORT THAT IT DID.
        #
        # The per-stream failure was already loud, but the summary below was
        # emitted unconditionally and said "streams drained" either way -- so
        # the one line every reader of this corpus greps was byte-identical
        # for a clean drain and for a blind one. Measured in the specimen
        # (`boot_rerun0826_0826_2149.log`): PP1 logged two synchronize failures
        # at L2189/L2195 and then "device-tier I/O quiesced ... in 0.2 ms" at
        # L2201, and PP2 the same at L2464/L2470/L2476. A grep for the success
        # line finds three clean drains in a boot that had one.
        #
        # This is the same class as #867 and as `SeamCensus.mark`: an
        # exception handler on the no-return path that cannot tell a survivable
        # failure from a context kill, feeding an instrument that then reports
        # the survivable reading. Fixed the same way -- classify, then say what
        # actually happened.
        t0 = time.perf_counter()
        failed: list = []
        for name in ("write_stream", "load_stream"):
            stream = getattr(self, name, None)
            if stream is None:
                continue
            try:
                stream.synchronize()
            except Exception as e:  # noqa: BLE001 - a dead stream must be loud
                failed.append(name)
                logger.error(
                    "#760 quiesce (%s): synchronizing %s failed (%s). "
                    "In-flight copies on it may still race the seam.",
                    reason,
                    name,
                    e,
                )
                _record_quiesce_poison(reason, name, e)
        elapsed = time.perf_counter() - t0
        # (#917) THE OUTCOME TRAVELS WITH THE CALL, because the runtime emits a
        # SECOND success claim off this one's return value
        # (`phase_flip_runtime._quiesce_hicache`, "[#760] SEAM DRAIN ...
        # device-tier streams quiesced"), and a float carries no outcome. Fixing
        # only the line in this module would have left the correlated,
        # direction-tagged line -- the one a three-rank log is actually read by
        # -- still claiming a drain that did not happen.
        #
        # An ATTRIBUTE rather than a widened return type: the signature is
        # pinned by `test_flip_seam_guard_760.py` and read by two callers, and
        # a drain is not the place to break a contract for a diagnostic.
        try:
            self.last_quiesce_failed = tuple(failed)
        except Exception:  # noqa: BLE001 - pragma: no cover
            pass
        if failed:
            logger.error(
                "#760 device-tier I/O NOT quiesced for %s after %.1f ms: "
                "%s did not drain. Their in-flight copies are unaccounted for "
                "and the seam proceeds without them -- this is NOT the clean "
                "drain the success line reports.",
                reason,
                elapsed * 1000.0,
                " and ".join(failed),
            )
            return elapsed
        logger.info(
            "#760 device-tier I/O quiesced for %s in %.1f ms (write and load "
            "streams drained while their pools are still live).",
            reason,
            elapsed * 1000.0,
        )
        return elapsed

    def evict_host(self, host_indices: torch.Tensor, backup_only: bool = True) -> int:
        if not backup_only:
            raise ValueError("Other eviction policies are not supported yet.")

        self.mem_pool_host.free(host_indices)
        return len(host_indices)

    def draft_tier_armed(self, direction: str) -> bool:
        """THE ONE GATE for draft-half I/O. Six consume points, one answer.

        #861, and the shape is 2bf0f53498's ("give all four consume points one
        gate") applied to the participant that never had one. Before this, six
        sites each read ``self.has_draft`` directly -- four here and two in
        ``HybridCacheController``'s overrides, which is the lane this rig
        actually runs (the W31/W32/W33 shape: a correct mechanism a second copy
        overrides is a mechanism that never runs).

        Three conditions, and each one is a corruption if it is skipped:

        1. REGISTERED AT ALL. The pre-#861 state on a flip boot, and the
           acceptance collapse this fixes.
        2. THE ACTIVE PHASE OWNS THE DRAFTER. There is one drafter in a
           phase-flip process and it lives on the TP stack; the PP prefill
           phase has none. A draft backup taken in PP would persist rows no
           drafter ever wrote under a content-addressed key, for the TP phase
           to load as valid -- strictly worse than the missing registration.
           ``None`` means "single-phase instance", which skips this term and
           keeps every non-flip deployment byte-identical.
        3. THE BINDING HAS NOT MOVED UNDER US. Draft host indices are 1-to-1
           with the TARGET host pool's, and ``hicache_phase_binding._stamp``
           re-points ``mem_pool_host`` at every rebind. A registration minted
           at generation g indexes generation g's slot space; consumed at g+1
           it addresses a different pool. Same authority as the releases and
           the write-backs (``write_back_stamp_is_current``), one more
           consumer -- not a second stamp scheme.

        The TARGET tier's own ``device_tier_disarmed`` is NOT re-asked here:
        every device-side consume point already sits behind it, and asking
        twice would let the two answers drift. The L3 points are storage-side
        and correctly outside it.
        """
        if not self.has_draft:
            return False
        if self.draft_owner_phase is not None:
            from sglang.srt.mem_cache.hicache_phase_guard import active_phase

            if active_phase() != self.draft_owner_phase:
                self._warn_draft_disarmed(
                    direction,
                    f"the active phase is not '{self.draft_owner_phase}', which "
                    f"is the only phase that owns a drafter",
                )
                return False
        if self.draft_binding_generation is not None:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                current_generation,
            )

            if int(self.draft_binding_generation) != int(current_generation()):
                self._warn_draft_disarmed(
                    direction,
                    f"registered at binding generation "
                    f"{self.draft_binding_generation}, current is "
                    f"{current_generation()}",
                )
                return False
        return True

    def _warn_draft_disarmed(self, direction: str, reason: str) -> None:
        """One line per (direction, reason) per process; the state is phase-long."""
        key = f"{direction}:{reason}"
        if key in self._draft_disarm_warned:
            return
        self._draft_disarm_warned.add(key)
        logger.warning(
            "#861 draft-half HiCache %s DISARMED: %s. Target-tier I/O is "
            "unaffected; a prefix restored while this holds carries NO draft "
            "rows, so requests admitted on it must be marked draft-cold "
            "(phase_flip_draft_bootstrap.mark_draft_cold) rather than allowed "
            "to speculate over rows nothing wrote.",
            direction,
            reason,
        )

    def set_draft_kv_pool(
        self,
        draft_device_pool,
        draft_host_pool,
        *,
        owner_phase=None,
        binding_generation=None,
        drafter_identity=None,
    ) -> None:
        """Register draft KV pools so L2/L3 ops piggyback draft transfers.

        Idempotent by design: ``rebind_hicache_draft_for_phase`` calls this on
        every pp->tp cutover with the same pools and a fresh generation, so
        re-registration is the normal case rather than an error.
        """
        self.has_draft = True
        self.mem_pool_device_draft = draft_device_pool
        self.mem_pool_host_draft = draft_host_pool
        self.draft_owner_phase = owner_phase
        self.draft_binding_generation = binding_generation
        self.draft_identity = drafter_identity
        logger.info(
            "HiCache draft KV registered: %s (host %d slots), owner_phase=%s, "
            "binding_generation=%s, drafter=%s",
            type(draft_device_pool).__name__,
            draft_host_pool.size,
            owner_phase,
            binding_generation,
            drafter_identity,
        )

        # If storage is already attached, wire up the draft I/O path now.
        # Otherwise this will be deferred until attach_storage_backend().
        self._maybe_register_draft_with_storage()

    def disarm_draft_kv_pool(self, reason: str) -> None:
        """#861: leave the draft half unarmed for the phase being entered.

        Not a teardown: the pools stay referenced so the next cutover into the
        drafter's own phase re-arms by re-stamping rather than re-allocating
        (a pinned host pool per flip would charge the host budget every time,
        and on this box that budget binds -- DESIGN_706 C1).

        Called unconditionally on the leg into a phase without a drafter, so
        "armed" is never a latch. The gate would refuse anyway on the phase
        term, and that redundancy is deliberate: a latched True that only the
        gate contradicts is one refactor away from being trusted.
        """
        if self.has_draft:
            logger.info(
                "#861 draft-half HiCache DISARMED: %s. The pools are kept for "
                "the next cutover into the drafter's phase.",
                reason,
            )
        self.has_draft = False
        self.draft_page_get_func = None
        self.draft_page_set_func = None

    def _maybe_register_draft_with_storage(self) -> None:
        """Pick the draft L3 IO implementation."""
        self.draft_page_get_func = None
        self.draft_page_set_func = None
        if not self.has_draft or not self.enable_storage:
            return

        backend = self.storage_backend_type

        # Multi-pool zero-copy backends.
        if backend == "mooncake":
            # #861 GUARD. The v2 route keys a page by the POOL NAME
            # (`register_mem_host_pool_v2(pool, PoolName.DRAFT)` ->
            # `_get_component_key`), so the drafter-identity suffix the generic
            # route carries cannot ride along without changing the registered
            # pool identity itself. Until task #861 item (a) puts the drafter
            # into `compute_model_identity_hash` -- where every backend and both
            # key routes pick it up -- a v2 draft page would be readable by a
            # different drafter as valid. Refused by name rather than left
            # write-only: a page nobody may read is still a page the NEXT boot
            # may read.
            logger.warning(
                "HiCache draft L3 disabled on the mooncake v2 route (#861): a "
                "v2 page is keyed by pool name, so it cannot carry the drafter "
                "identity the generic route puts in its component key, and a "
                "draft page readable across a drafter change is a silently "
                "wrong draft KV. The L2 host tier is unaffected. Lift this "
                "once #861 (a) folds the drafter into the model identity hash."
            )
            return

        # TODO: support "hf3fs", "eic", "nixl", "simm"
        if backend in {"hf3fs", "eic", "nixl", "simm"}:
            logger.warning(
                "HiCache draft L3 disabled: backend %s does not yet support "
                "draft pool registration.",
                backend,
            )
            return

        # Generic backends.
        self.draft_page_get_func = self._draft_page_get_generic
        self.draft_page_set_func = self._draft_page_set_generic

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
    ) -> PrefetchOperation:
        """
        Prefetch KV caches from storage backend to host memory.
        """
        operation = PrefetchOperation(
            request_id, host_indices, new_input_tokens, last_hash, prefix_keys
        )
        self.prefetch_queue.put(operation)
        return operation

    def terminate_prefetch(self, operation):
        operation.mark_terminate()
        return operation.completed_tokens, operation.hash_value

    def append_host_mem_release(self, host_indices: torch.Tensor, generation=None):
        """Queue host slots for release, ROUTED BY THE BINDING THEY CAME FROM.

        W35. `_drain_release` frees whatever is on this queue against
        `self.mem_pool_host` -- the pool bound NOW. Three producers can put
        entries here after a cutover that name slots from the pool bound
        BEFORE it: `_drain_revoke`, the prefetch transfer thread, and the
        direct path. Freeing those against the current pool is the measured
        W35 double-free ("slots not currently allocated"); dropping them leaks
        slots in a pool that returns on the very next flip, because the flip
        alternates and nothing tears the outgoing pool down.

        ROUTED AT PRODUCE TIME, which is the only point where the generation is
        still known without changing what this queue carries. A stale batch is
        freed immediately against the pool its generation names -- the queue is
        a batching convenience, not a correctness mechanism, so settling one
        batch early costs nothing. Only current-generation slots are queued, so
        by construction nothing on the queue can outlive its binding.

        A stale batch whose pool is UNKNOWN is refused loudly and NOT queued:
        freeing it here would corrupt and queueing it would corrupt later.

        #989 PROVENANCE AT THE ENQUEUE, BECAUSE THE FREE CANNOT RECONSTRUCT IT.
        `_drain_release` frees whatever arrives, so when two producers queue the
        same span the pool's assert names the SLOTS and nothing names the
        PRODUCERS -- which is why boot 18's double-free (span 60934-60971+,
        plain PP=3, flip provably off) could be localised to this queue and no
        further. The site is taken from the caller's frame rather than added as
        a parameter to all fifteen call sites, so a producer that is added
        LATER is stamped without anyone remembering to: eight sites in
        unified_radix_cache.py, plus swa_component, mamba_component,
        hiradix_cache (x3), hi_mamba_radix_cache (x2) and this file.

        THE W35 GUARD ABOVE DOES NOT COVER THIS AND THAT IS THE POINT. It routes
        by BINDING GENERATION -- it catches a producer naming slots from the
        pool bound BEFORE a cutover. Two producers queueing the same span
        WITHIN one generation are, to that guard, two ordinary current-
        generation batches: structurally invisible to it, not merely missed.
        The guard names the hazard, not the gap.
        """
        if host_indices.numel() == 0:
            return
        try:
            f = sys._getframe(1)
            site = f"{f.f_globals.get('__name__', '?').rsplit('.', 1)[-1]}:{f.f_lineno}"
        except Exception:  # noqa: BLE001 - provenance may never break a release
            site = "?"
        if generation is not None:
            from sglang.srt.mem_cache.hicache_phase_binding import (
                host_pool_for_generation,
                write_back_stamp_is_current,
            )

            if not write_back_stamp_is_current(generation):
                owner = host_pool_for_generation(generation)
                self._stale_release_routed = (
                    getattr(self, "_stale_release_routed", 0) + 1
                )
                if owner is None:
                    self._stale_release_orphaned = (
                        getattr(self, "_stale_release_orphaned", 0) + 1
                    )
                    logger.error(
                        "#719/W35 STALE RELEASE ORPHANED: %d host slot(s) were "
                        "opened under binding generation %s, which names no "
                        "known pool. Not queued and not freed -- freeing them "
                        "against the current pool is the W35 double-free and "
                        "queueing them defers the same crash. (%d so far.)",
                        int(host_indices.numel()),
                        generation,
                        self._stale_release_orphaned,
                    )
                    return
                n = self._stale_release_routed
                if n <= 3 or n % 200 == 0:
                    logger.warning(
                        "#719/W35 STALE RELEASE ROUTED: %d host slot(s) opened "
                        "under binding generation %s are freed against THAT "
                        "generation's pool rather than the one bound now. "
                        "(%d so far.)",
                        int(host_indices.numel()),
                        generation,
                        n,
                    )
                owner.free(host_indices)
                return
        pages = host_indices.split(self.mem_pool_host.page_size)
        for page in pages:
            # #989: record who queued this page, keyed by its first slot. The
            # map is authoritative only until the page is drained; the drain
            # pops what it reads, so it cannot grow without bound while the
            # drain keeps up, and `_drain_release`'s own clamp covers the case
            # where it does not.
            try:
                self.host_release_provenance[int(page[0])] = site
            except Exception:  # noqa: BLE001 - provenance may never break a release
                pass
            self.host_mem_release_queue.put(page)

    def host_release_site(self, first_slot: int) -> str:
        """#989: who queued the page starting at this slot, '?' if unrecorded."""
        return self.host_release_provenance.get(int(first_slot), "?")

    def _page_get_zero_copy(
        self, operation, hash_values, host_indices, extra_info=None
    ):
        results = self.storage_backend.batch_get_v1(
            hash_values, host_indices, extra_info
        )
        inc = 0
        for i in range(len(hash_values)):
            if not results[i]:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            inc += self.page_size
        operation.increment(inc)

    # todo: deprecate
    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if page_data is None:
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    f"Prefetch operation {operation.request_id} failed to retrieve page {hash_values[i]}."
                )
                break
            # Must set the data before increasing the completed tokens.
            # Otherwise this page may be read before being set.
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size],
                page_data[i],
            )
            if not operation.increment(self.page_size):
                break  # Operation terminated by controller

    def _page_transfer(self, operation):
        # Transfer batch by batch
        prefix_keys = operation.prefix_keys
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            if operation.is_terminated():
                # #1068 (A12.4): a stop or a revoke lands at the next batch
                # boundary (~4 MiB), never after one more storage read. The
                # per-page increment below would refuse anyway; checking
                # here saves the batch_get and makes the join bound of
                # `_stop_storage_threads` hold by construction.
                break
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]

            # Best-effort draft L3 read before publishing target completion.
            # Otherwise wait_complete can race and load back target KV before
            # draft KV reaches host memory.
            if self.draft_tier_armed("l3-load"):
                self._draft_page_get(batch_hashes, batch_host_indices)

            prev_completed_tokens = operation.completed_tokens
            # Get one batch token, and update the completed_tokens if succeed
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            self.page_get_func(operation, batch_hashes, batch_host_indices, extra_info)
            # Check termination
            if (
                operation.completed_tokens
                != prev_completed_tokens + len(batch_hashes) * self.page_size
            ):
                operation.mark_terminate()
                break  # Some operations fail or operation terminated by controller

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

    def prefetch_io_aux_func(self):
        """
        Auxiliary function conducting IO operations for prefetching.

        #1033d: ONE FAILED OPERATION MUST NOT END THE THREAD.
        Measured, boot 22 (`boot_855_1033b_0840f82601_0831_131955`, 13:23:57,
        three seconds before the cutover re-dispatch): a page whose geometry
        did not match the incoming binding raised out of `_page_transfer` --

            pool_host/mha.py:556  data_page.reshape(...)
            RuntimeError: shape '[2, 16, 1, 4, 256]' is invalid for input of
            size 16384          (2*16*1*4*256 = 32768 against a 16384 page)

        -- and because only `Empty` was caught, the exception left the `while`
        and the thread was GONE for the rest of the boot. Nothing noticed: a
        worker thread's death kills no process, sets no exit code and appears
        in no health probe. Storage prefetch was simply dead from then on.

        THE UNDERLYING DEFECT IS NOT FIXED HERE and this guard must not be
        mistaken for its fix. Two lines above that traceback the log of boot
        22 carried the fork's own diagnosis of the same mismatch (`source
        page 16384 elems vs destination page 32768 elems`, printed by the
        re-homing pass of that day) alongside `[#719 hicache-rebind]
        rebound 3 reader(s) to the 'tp' pools (generation 3)`. The host pool
        carries TWO page geometries across a flip. That is the
        #718/#719/#875 family and a violation of the standing
        phase-uniformity law (two geometries for one key is a bug). It is
        its own posten. HISTORY (2026-09-02, #1068 slice 4 af399f19c1 and
        its fix 285e3685b6): the re-homing pass and its page re-reader were
        deleted as second bookkeeping beside the slice-3 re-issue, so
        nothing in this file re-reads across geometries any more and that
        diagnosis line no longer exists to look for.

        WHY A BROAD `except` IS DEFENSIBLE HERE, given that warn-then-continue
        is a catalogued defect class in this tree: the comparison is not
        "swallow vs. propagate", it is "swallow vs. SILENT THREAD DEATH",
        because there is no supervisor above this loop to receive the raise.
        The three properties that keep it from becoming the anti-pattern:
        the operation is FAILED rather than silently skipped (its host memory
        is released, so the requester gets a refusal instead of waiting for a
        completion that can never come); the line is loud and names the
        consequence; and it CARRIES A COUNTER, so a path that fails constantly
        cannot read like a path that failed once.
        """
        n_failed = 0
        while not self.storage_stop_event.is_set():
            operation = None
            try:
                operation = self.prefetch_buffer.get(block=True, timeout=1)
                if operation is None:
                    continue
                # #1068 (A12.4): publish the current operation BEFORE reading
                # the stop event, so a stop that lands between the two lines
                # is seen by exactly one side: either this loop terminates
                # the operation itself (stop already set), or the pointer was
                # published before `_terminate_inflight_prefetch` read it.
                self._prefetch_io_current = operation
                if self.storage_stop_event.is_set():
                    operation.mark_terminate()
                    self._prefetch_io_drained_after_stop += 1
                self._page_transfer(operation)
                # operation terminated by controller, release pre-allocated memory
                # W35: this thread runs across cutovers, so the slots it
                # releases may have been opened under an older binding.
                self.append_host_mem_release(
                    operation.host_indices[operation.completed_tokens :],
                    generation=getattr(operation, "binding_generation", None),
                )
            except Empty:
                continue
            except Exception:  # noqa: BLE001 - see the docstring: the only
                # alternative is a dead thread nobody is told about.
                n_failed += 1
                logger.error(
                    "#1033d PREFETCH IO REFUSED: page transfer raised; this "
                    "operation is FAILED and its host slots are released, and "
                    "the aux thread STAYS ALIVE (before #1033d this raise ended "
                    "the thread and storage prefetch died silently for the rest "
                    "of the boot). Failures on this thread so far: %d. A "
                    "reshape/geometry error here is the two-geometry host pool "
                    "across a flip (#718/#719/#875 family: the page this "
                    "operation reads was minted under a host tier geometry "
                    "other than the one bound now).",
                    n_failed,
                    exc_info=True,
                )
                if operation is not None:
                    try:
                        self.append_host_mem_release(
                            operation.host_indices[operation.completed_tokens :],
                            generation=getattr(operation, "binding_generation", None),
                        )
                    except Exception:  # noqa: BLE001 - a release may not re-kill
                        logger.error(
                            "#1033d and the release of the failed operation's "
                            "host slots ALSO raised; those slots leak until the "
                            "binding is torn down. Named, not hidden.",
                            exc_info=True,
                        )
                continue
            finally:
                self._prefetch_io_current = None

    @property
    def prefetch_capacity_fraction(self) -> float:
        """The share of the bound host pool that speculative prefetch may
        hold: 0.9 for a staging tier, 0.5 for a retention tier (upstream
        c66a285c94 cache_controller.py:253 / :575-584)."""
        if self.host_role == "staging":
            return HICACHE_LOAD_POOL_USAGE_FRACTION
        return HICACHE_CACHE_MODE_LOAD_FRACTION

    @property
    def prefetch_capacity_limit(self) -> int:
        """The speculative-prefetch budget, in tokens, of the host pool the
        controller is bound to RIGHT NOW (#1068; upstream c66a285c94
        :575-584 sizes it the same way: the buffer_only fraction for a
        staging tier, the cache-mode half otherwise).

        A property and not a number on purpose: the phase flip rebinds
        ``mem_pool_host`` (hicache_phase_binding._stamp), and a number stored
        at attach kept describing the pool of the previous phase -- which is
        what the deleted floor and the two symmetrize passes compensated for.
        With ``--hicache-size`` both phase pools are MIN-synced across ranks
        (pool_host/base.py sync_fixed_hicache_size), so this value is
        rank-uniform by construction and no group reduce is needed.

        With NO host pool bound this RAISES (A12.4; #606 class): a silent 0
        would make ``prefetch_rate_limited`` read True forever and refuse
        every prefetch without a line that names why. ``__init__`` binds
        ``mem_pool_host`` unconditionally, so the raise is reachable only
        from a shell or a torn binding, and both must be loud.
        """
        pool = self.mem_pool_host
        if pool is None:
            raise RuntimeError(
                "#1068 prefetch_capacity_limit read with no host pool bound "
                "(mem_pool_host is None): the budget is a property of the bound "
                "pool and has no value without one; a 0 here would refuse every "
                "prefetch silently"
            )
        return int(self.prefetch_capacity_fraction * int(pool.size))

    def prefetch_rate_limited(self) -> bool:
        """Refuse a new prefetch registration once the registered prefetches
        hold the budget: ``prefetch_tokens_occupied >= prefetch_capacity_limit``
        (upstream c66a285c94 cache_controller.py:1164-1166, the cache-mode
        form), in BOTH host roles.

        WHAT THIS BOUNDS, honestly (slice 2 review round 3, finding (g), spec
        A12.1): ``prefetch_tokens_occupied`` is released at COMPLETION
        (unified_radix_cache.py, the decrement in the completion path), so
        the budget bounds CONCURRENT in-flight spans, not resident rows: at
        most floor(prefetch_capacity_limit / prompt_max) spans of the largest
        prompt class can be registered at once (Boot-2 sizing: 329589 /
        39365 = 8 spans; the 9th concurrent span reads True here). Today a
        True verdict in ``prefetch_from_storage`` is a DECLINE before its
        alloc -> evict_host -> retry path, and ``_prefetch_kvcache`` is
        issued once per request at intake, so the declined request recomputes
        its whole prefix. That is the double-prefill class this brake was
        never meant to cause; it is owned by slice 3 (A12.2: a rate_limited
        verdict becomes a DEFERRAL that retries on every scheduling pass and
        lands sequentially, never a decline). Until A12.2 lands, this method
        is exactly the concurrency bound above, no more.

        WHY THE COUNTER AND NOT LIVE OCCUPANCY (slice 2 review, finding B1).
        Upstream carries two forms and dispatches on ``host_memory_mode``: the
        live form ``max(0, size - available_size() - write_staged) >= limit``
        (c66a285c94 :1155-1163) is for ``buffer_only``, where the host pool is a
        TRANSIENT buffer and every row is freed once its storage write is
        acked. This fork's host tier RETAINS in both roles:
        ``UnifiedRadixCache._drain_backup`` answers a storage ack with
        ``ring.release`` + ``dec_host_lock_ref`` and never frees the rows,
        ``evict_host`` has no role branch, and ``available_size()`` is
        ``len(free_slots)`` (memory_pool_host.py). On a warm tier the live
        form therefore reads ``used == size``, trips permanently at 0.5
        (retention, the default) or 0.9 (staging) of the pool, and refuses
        every prefetch BEFORE ``prefetch_from_storage`` reaches its
        alloc -> evict_host -> retry path: store-read path B dead, every
        re-admission recomputing the whole prefix, the double-prefill class
        WEG 1 exists to remove. Slice 2 shipped that form unconditionally and
        its suite never pinned it (mutant retention -> counter form survived
        20/20).

        The live form becomes the right form for the staging role on the day
        the staging drain frees host rows after ``ack_backup`` (transient
        buffer = upstream buffer_only semantics). That day brings back the
        staged-tokens reader of upstream c66a285c94 :331 together with its
        consumer and its tests; this controller carries no such reader until then, because
        a wire that nobody reads is second bookkeeping (slice 2 fix 2). Until
        then ``prefetch_tokens_occupied`` is the gate: the tokens locked by
        registered prefetches (written by the three tree caches), zeroed at
        attach and in ``reset()``.
        """
        return self.prefetch_tokens_occupied >= self.prefetch_capacity_limit

    def _presence_pool_transfers(self) -> Optional[list[PoolTransfer]]:
        """The component transfers the REAL fetch carries, rebuilt for the probe.

        #869b, and it is the half `store_presence_pages` promised but did not
        deliver. Its own paragraph below says the probe "asks the identical
        question with the identical helper" as the real fetch. It did not: the
        fetch runs through `_storage_hit_query`, which calls `batch_exists_v2`
        WITH the tree's component transfers, and that call MIN-CLAMPS the
        usable KV prefix to the last page that also carries the component's
        blob (`hicache_storage.py` file backend, and the hf3fs/mooncake twins).
        The probe called plain `batch_exists`, which can only ever see KV pages.

        MEASURED, hermetically, on the real file backend: four KV pages
        present, the mamba anchor present only at page 1. `batch_exists`
        answers 4; `batch_exists_v2` answers 2. So the gate green-lit a fetch
        whose mamba half could not land at the depth it promised -- the KV
        bytes arrived and the match walk then refused the node for want of a
        recurrent state. That is the #873 census reading
        `refusers=MambaComponent` on 671 of 675 walks with the KV bytes
        demonstrably present: not a validator refusing wrongly, but a fetch
        nobody had asked the anchor about.

        REBUILT HERE rather than taken from the tree, because the probe runs
        on the admission path BEFORE any prefetch operation exists to carry
        transfers. Only a pool whose hit policy is a CONSTANT of the pool can
        be reproduced this way. Mamba's is: exactly one trailing page, fixed
        identically by `MambaComponent.build_hicache_transfers`
        (`mamba_component.py:1231-1232`) and by its HiMamba twin
        (`hi_mamba_radix_cache.py:2503-2504`).

        SWA IS DELIBERATELY LEFT OUT. Its trailing count is a function of the
        window, not of the pool, and is not derivable here. Under-including a
        pool keeps the probe optimistic in exactly the direction it was
        already optimistic in before this change -- no regression, no new
        wrong answer -- whereas guessing a window width would make the gate
        wrong in a NEW direction, and the direction it would be wrong in is
        "declare a prefix usable that is not".

        Returns None when there is nothing extra to ask about, which keeps
        every non-hybrid deployment on the exact pre-#869b call.
        """
        pools = getattr(self.storage_backend, "registered_pools", None)
        if not pools or PoolName.MAMBA not in pools:
            return None
        return [
            PoolTransfer(
                name=PoolName.MAMBA,
                keys=["__placeholder__"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]

    def store_presence_pages(self, token_ids, last_hash, prefix_keys=None) -> int:
        """#950: how many pages the STORE holds for this span, by CONTENT KEY.

        THE PRECONDITION REPLACEMENT. `Scheduler._prefetch_kvcache` gated the
        fetch on `last_host_node.backuped` -- "the full KV is ALREADY in this
        rank's host pool" -- as the entry price for an operation whose whole
        purpose is to obtain what is NOT resident. Window-946rf-0828 measured
        the consequence exactly: `reason=anchor_no_vote` 5 of 5, and
        `[#915 prefetch-gate] no observation` confirming the gate was never
        even reached. The criterion was anti-correlated with the situation the
        escape exists for.

        Presence by content key is the honest question, and under #706 it is a
        well-posed one: a page carries all attention layers for its tokens and
        the cut happens at READ time, so the key is a function of the CONTENT,
        not of any rank's layout or residency. That is also why a fetch answered
        here lands in the CURRENT layout and sidesteps the layer-sharded seam
        problem of #941.

        SAME KEY DERIVATION AS THE REAL FETCH, deliberately. `_storage_hit_query`
        below computes `get_hash_str(tokens, last_hash, page_size)`; this asks
        the identical question with the identical helper. A second spelling of a
        key chain would be a second installer of the same payload -- the exact
        rule the #949b tensor landmine was fixed under, one layer up.

        SAME COMPONENTS, TOO, SINCE #869b -- and until then this paragraph was
        half false in a way that cost the whole fetch. The key chain matched;
        the QUESTION did not. `_storage_hit_query` asks `batch_exists_v2` with
        the tree's component transfers, which min-clamps the answer to the last
        page that also carries a mamba anchor; this probe asked plain
        `batch_exists`, which sees KV pages alone. Measured on the file backend
        with the anchor only at page 1 of 4: the probe said 4, the fetch could
        use 2. The gate therefore admitted fetches whose recurrent half could
        not land, and the match walk refused the result -- #873's 671-of-675
        `refusers=MambaComponent` with the KV bytes present. `_presence_pool_
        transfers` above rebuilds the mamba transfer so both sides ask one
        question about BOTH components, and the number returned here is already
        the anchor-capped one: the caller inherits the honest depth instead of a
        KV-only promise. A backend without the v2 interface falls back to the
        KV-only question and SAYS SO, so "cannot be asked" never reads as
        "the anchor is there".

        ONE ROUND-TRIP: the whole key chain goes in a single `batch_exists`, and
        the caller caches the verdict per streak increment, so the escape costs
        at most one query per attempt and never one per pass.

        Returns 0 on any failure. A store that cannot be asked is treated as not
        holding the pages, which declines the fetch rather than issuing one that
        cannot land -- and the decline is NAMED, so "we could not ask" never
        reads as "it is not there".
        """
        if not token_ids:
            return 0
        try:
            page_hashes = self.get_hash_str(
                list(token_ids), last_hash, page_size=self.page_size
            )
            if not page_hashes:
                return 0
            extra_info = HiCacheStorageExtraInfo(
                prefix_keys=list(prefix_keys) if prefix_keys else None
            )
            batch = page_hashes[:STORAGE_BATCH_SIZE]
            pool_transfers = self._presence_pool_transfers()
            if pool_transfers:
                try:
                    return int(
                        self.storage_backend.batch_exists_v2(
                            batch, pool_transfers, extra_info
                        ).kv_hit_pages
                        or 0
                    )
                except NotImplementedError:
                    # A backend without the v2 interface cannot be asked about
                    # component blobs at all. Fall through to the KV-only
                    # question rather than reporting 0: this is the pre-#869b
                    # behaviour, and it is NAMED here so "this backend cannot
                    # be asked about the anchor" never reads as "the anchor is
                    # there".
                    logger.debug(
                        "#869b: %s has no batch_exists_v2; the presence probe "
                        "answers on KV pages alone and the mamba anchor is not "
                        "part of this issuance decision",
                        type(self.storage_backend).__name__,
                    )
            return int(
                self.storage_backend.batch_exists(batch, extra_info) or 0
            )
        except Exception as exc:  # noqa: BLE001 - a probe never breaks admission
            logger.warning(
                "#950 store presence probe failed (%s); treating the span as "
                "ABSENT, which declines the re-fetch rather than issuing one "
                "that cannot land",
                exc,
            )
            return 0

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        last_hash = operation.last_hash
        tokens_to_fetch = operation.token_ids
        prefix_keys = operation.prefix_keys.copy() if operation.prefix_keys else None

        storage_query_count = 0
        hash_value = []
        page_hashes = self.get_hash_str(
            tokens_to_fetch, last_hash, page_size=self.page_size
        )

        for start in range(0, len(page_hashes), STORAGE_BATCH_SIZE):
            if operation.is_terminated():
                # #1068 (A12.4): a terminated operation asks the store no
                # further batch; the partial count is what the rank-uniform
                # MIN all_reduce in `prefetch_thread_func` then agrees on.
                break
            batch_hashes = page_hashes[start : start + STORAGE_BATCH_SIZE]
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
            hash_value.extend(batch_hashes[:hit_page_num])
            storage_query_count += hit_page_num * self.page_size
            if hit_page_num < len(batch_hashes):
                break
            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes

        return hash_value, storage_query_count

    def prefetch_thread_func(self):
        """
        Manage prefetching operations from storage backend to host memory.
        """
        self.prefetch_buffer = Queue()
        self.prefetch_io_aux_thread = threading.Thread(
            target=self.prefetch_io_aux_func, daemon=True
        )
        self.prefetch_io_aux_thread.start()
        while (not self.storage_stop_event.is_set()) or not self.prefetch_queue.empty():
            try:
                operation = self.prefetch_queue.get(block=True, timeout=1)
            except Empty:
                continue
            if operation is None:
                continue
            # #1068 (A12.4): publish-then-check, same protocol as the aux
            # loop, so an operation dequeued around the stop is terminated
            # by exactly one side. The drain of this queue stays HERE (loop
            # condition above): it carries one rank-uniform MIN all_reduce
            # per operation, and after the stop each one is terminated, so
            # `_storage_hit_query` asks the store nothing and the drain is
            # one collective per operation, not one storage walk.
            self._prefetch_current = operation
            try:
                if self.storage_stop_event.is_set():
                    operation.mark_terminate()
                    self._prefetch_drained_after_stop += 1
                hash_value, storage_hit_count = self._storage_hit_query(operation)
                storage_hit_count_tensor = torch.tensor(
                    storage_hit_count, dtype=torch.int
                )
                self._all_reduce_prefetch_groups(
                    storage_hit_count_tensor, torch.distributed.ReduceOp.MIN
                )
                storage_hit_count = storage_hit_count_tensor.item()
                # #1157: the probe has ANSWERED (rank-uniform after the MIN).
                # Stamped before the threshold branch so a revoke and a
                # transfer both carry it; a reap that finds None was ahead of
                # the probe.
                operation.probed_hit_tokens = int(storage_hit_count)

                if storage_hit_count < self.prefetch_threshold:
                    # not to prefetch if not enough benefits
                    self.prefetch_revoke_queue.put(operation.request_id)
                    # #1068 (A12.5 addition, decided in the slice 4 fix): the
                    # LOST-REVOKE-AT-QUIESCE candidate is MOOT on every
                    # re-entry of `_start_storage_threads`. The candidate: a
                    # TERMINATED operation drained here after the stop lands
                    # on the revoke queue and on the release queue, then
                    # `_start_storage_threads` rebuilds both queues, the revoke
                    # is lost, and a tree record that survived would reach
                    # `check_prefetch_progress` and free the same slots a
                    # second time (#989 class). No record survives, by the seam
                    # order in `phase_flip_runtime._execute_body` (the whole
                    # seam runs on the scheduler thread, the only issuer):
                    #   1. `_release_residents_for_cutover` -> `_drop_tree` ->
                    #      `drop_prefix_tree_returning_rows` -> `tree.reset()`
                    #      -> `UnifiedRadixCache._reset_full`: ongoing_prefetch
                    #      = {} BEFORE `cache_controller.reset()` (stop, then
                    #      `_start_storage_threads`), then
                    #      `mem_pool_host.clear()`. Records gone, the pool
                    #      forgets every allocation: a revoke or a release lost
                    #      with the old queues names nothing.
                    #   2. `_cutover_fn` -> `rebind_for_cutover` ->
                    #      `_quiesce_storage_io` / `_resume_storage_io`
                    #      (hicache_phase_binding.py): the queues rebuilt in
                    #      step 1 are still EMPTY -- nothing between step 1 and
                    #      the rebind calls `prefetch_from_storage` (the
                    #      pre-cutover fns are the labelled movers; re-admission
                    #      is `_post_cutover_readmit`, AFTER the rebind) -- so
                    #      this branch drains no operation at that quiesce.
                    #   3. `attach_storage_backend` starts the first pipeline;
                    #      there is no predecessor to drain.
                    # The release is stamped with the operation's binding
                    # generation all the same, as the aux loop and
                    # `_drain_revoke` already do (W35 sibling): a producer that
                    # names slots without its generation is the shape W35
                    # measured, and the stamp is a no-op while the generation
                    # is current.
                    self.append_host_mem_release(
                        operation.host_indices,
                        generation=getattr(operation, "binding_generation", None),
                    )
                    logger.debug(
                        f"Revoking prefetch for request {operation.request_id} due to insufficient hits ({storage_hit_count})."
                    )
                else:
                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    # free the pre-allocated memory for pages that are not hit
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                    operation.host_indices = operation.host_indices[:storage_hit_count]
                    logger.debug(
                        f"Prefetching {len(operation.hash_value)} pages for request {operation.request_id}."
                    )
                    self.prefetch_buffer.put(operation)
            finally:
                self._prefetch_current = None

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        kv_page_owner_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Write KV caches from host memory to storage backend.
        """
        operation = StorageOperation(
            host_indices, token_ids, hash_value=hash_value, prefix_keys=prefix_keys
        )
        operation.kv_page_owner_mask = kv_page_owner_mask
        self.backup_queue.put(operation)
        return operation.id

    # todo: deprecate
    def _generic_page_set(self, hash_values, host_indices, extra_info=None) -> bool:
        data = [
            self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
            for i in range(len(hash_values))
        ]
        return self.storage_backend.batch_set(hash_values, data)

    def _page_set_zero_copy(self, hash_values, host_indices, extra_info=None) -> bool:
        return all(
            self.storage_backend.batch_set_v1(hash_values, host_indices, extra_info)
        )

    def _draft_page_set(self, hash_values, host_indices) -> None:
        """Best-effort write draft KV pages to L3 alongside the target backup."""
        if self.draft_page_set_func is None:
            return
        try:
            self.draft_page_set_func(hash_values, host_indices)
        except Exception:
            logger.debug(
                "Draft L3 write failed (best-effort), skipping.", exc_info=True
            )

    def _draft_page_get(self, hash_values, host_indices) -> None:
        """Best-effort read draft KV pages from L3 (mirrors `_draft_page_set`)."""
        if self.draft_page_get_func is None:
            return
        try:
            self.draft_page_get_func(hash_values, host_indices)
        except Exception:
            logger.debug("Draft L3 read failed (best-effort), skipping.", exc_info=True)
        self._log_draft_l3_progress()

    def _log_draft_l3_progress(self) -> None:
        """#993: one line per doubling of the pages this read path has seen.

        THE INSTRUMENT, and it is written so that it cannot read as success
        when nothing happened: it fires off the page count, so a process that
        never reads a draft page emits NOTHING and the absence of the line is
        itself the finding. `hit` and `miss` are counted at the one site that
        knows which is which (`_draft_page_get_generic`), not inferred.
        """
        total = self._draft_l3_hits + self._draft_l3_misses
        if total == 0:
            return
        if self._draft_l3_logged_at >= 0 and total < 2 * self._draft_l3_logged_at:
            # First batch always speaks (execution proof), then one line per
            # doubling so a long run does not drown in them.
            return
        self._draft_l3_logged_at = total
        logger.info(
            "#993 draft L3 READ: %d page(s) requested, %d hit / %d miss "
            "(%.1f %% hit). A miss writes ZEROS into its draft host row, so a "
            "restored prefix's draft half is real-or-zero and never the "
            "previous occupant's bytes; that is what lets admission keep the "
            "rows that did come back instead of scrubbing the whole prefix.",
            total,
            self._draft_l3_hits,
            self._draft_l3_misses,
            100.0 * self._draft_l3_hits / total,
        )

    # UNREACHABLE SINCE #861, and kept rather than deleted: these are the
    # landing site for task #861 item (a). The v2 route keys a page by POOL
    # NAME, so it cannot carry the drafter identity the generic route puts in
    # its component key; `_maybe_register_draft_with_storage` refuses to wire
    # them until (a) folds the drafter into `compute_model_identity_hash`,
    # where every backend picks it up. Deleting them would make (a) rebuild
    # what already works, for the sake of a dead-code count.
    def _draft_page_set_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_page_get_v2(self, hash_values, host_indices) -> None:
        self.storage_backend.batch_get_v2(
            [
                PoolTransfer(
                    name=PoolName.DRAFT,
                    host_indices=host_indices,
                    keys=list(hash_values),
                )
            ]
        )

    def _draft_component_name(self) -> str:
        """The component name a persisted draft page is keyed under.

        #861 GUARD, and it is the cheapest correct form rather than the full
        fix. Fix (0) newly makes these pages READABLE, and every HiCache key
        suffix is built from ``compute_model_identity_hash``, which covers the
        TARGET (model_path, revision, dtype, quantization, kv_cache_dtype) and
        carries nothing about the drafter. Two boots agreeing on the target and
        differing in drafter -- another NEXTN checkpoint, MTP<->EAGLE, the #156
        cross-algorithm switch -- would read each other's draft KV as valid,
        with blob length the only accidental guard and equal geometry the
        common case.

        Folding the drafter into the component name makes such a page simply
        NOT EXIST for the other drafter: a clean miss, which is the same
        argument ``HiCacheFile`` already makes for the identity hash it does
        carry. Task #861 item (a) -- drafter identity inside
        ``compute_model_identity_hash`` itself, so EVERY backend and both key
        routes carry it -- is filed and deliberately not built here.

        Falls back to the bare pool name when no identity was supplied, which
        keeps a caller that predates this parameter writing exactly the keys it
        wrote before.
        """
        if not self.draft_identity:
            return str(PoolName.DRAFT)
        return f"{PoolName.DRAFT}-{self.draft_identity}"

    def _draft_page_set_generic(self, hash_values, host_indices) -> None:
        # `{hash}.draft-{drafter}` mirrors HiCacheStorage._get_component_key's
        # `{key}.{pool_name}` convention so target/draft pages never collide,
        # and never collide across drafters either (#861).
        component = self._draft_component_name()
        draft_keys = [f"{h}.{component}" for h in hash_values]
        draft_data = [
            self.mem_pool_host_draft.get_data_page(host_indices[i * self.page_size])
            for i in range(len(draft_keys))
        ]
        self.storage_backend.batch_set(draft_keys, draft_data)

    def _draft_page_get_generic(self, hash_values, host_indices) -> int:
        """Fill the draft host rows for these pages. Returns the number HIT.

        #993: A MISS NOW WRITES ZEROS RATHER THAN LEAVING THE ROW ALONE.

        The row this page would occupy is a recycled host slot holding the
        PREVIOUS occupant's draft bytes. Leaving it untouched on a miss made
        "restored target prefix" and "draft rows of an unrelated request"
        indistinguishable downstream, which is why the admission path had to
        scrub the WHOLE prefix and could never keep the pages that did come
        back. Writing the zero page (``get_dummy_flat_data_page`` is
        ``torch.zeros``) makes a missing draft page DETERMINISTIC instead of
        arbitrary -- the same argument, and the same value, that
        ``phase_flip_draft_bootstrap.scrub_draft_kv`` already makes for the
        rows it clears, moved to the one place that knows WHICH pages missed.
        It also removes the fp8 NaN/Inf path into the draft softmax that
        arbitrary recycled bytes carry.

        Correctness is unchanged either way -- the target verifies every
        proposed token -- so this buys acceptance, not answers.
        """
        component = self._draft_component_name()
        draft_keys = [f"{h}.{component}" for h in hash_values]
        draft_dummy = [
            self.mem_pool_host_draft.get_dummy_flat_data_page() for _ in draft_keys
        ]
        draft_pages = self.storage_backend.batch_get(draft_keys, draft_dummy)
        if draft_pages is None:
            # The whole batch failed: no row is known-good, so every one of
            # them still holds its previous occupant. Zero them all rather
            # than return early -- an early return here is exactly the state
            # the admission scrub existed to clean up after.
            draft_pages = [None] * len(draft_keys)
        hits = 0
        for i, p in enumerate(draft_pages):
            if p is None:
                # `draft_dummy[i]` is untouched zeros for a miss.
                p = draft_dummy[i]
            else:
                hits += 1
            self.mem_pool_host_draft.set_from_flat_data_page(
                host_indices[i * self.page_size], p
            )
        self._draft_l3_hits += hits
        self._draft_l3_misses += len(draft_keys) - hits
        return hits

    # Backup batch by batch
    def _page_backup(self, operation):
        # Backup batch by batch
        prefix_keys = operation.prefix_keys
        # Weighted uneven-DCP owner mode: only the pages this rank owned at
        # backup time carry real data in the host pool -- write exactly those
        # (the rank-shared page file is complete: full replicated kv-heads).
        # Pages owned by other ranks are persisted by their owners; they still
        # count as completed here so ack/host-release semantics are unchanged.
        owner_mask = getattr(operation, "kv_page_owner_mask", None)
        for i in range(0, len(operation.hash_value), STORAGE_BATCH_SIZE):
            batch_hashes = operation.hash_value[i : i + STORAGE_BATCH_SIZE]
            batch_host_indices = operation.host_indices[
                i * self.page_size : (i + len(batch_hashes)) * self.page_size
            ]
            # Set one batch token, and record if success.
            # todo: allow partial success
            extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
            if owner_mask is not None:
                owned_pos = [
                    j for j in range(len(batch_hashes)) if bool(owner_mask[i + j])
                ]
                if owned_pos:
                    owned_hashes = [batch_hashes[j] for j in owned_pos]
                    owned_host_indices = torch.cat(
                        [
                            batch_host_indices[
                                j * self.page_size : (j + 1) * self.page_size
                            ]
                            for j in owned_pos
                        ]
                    )
                    success = self.page_set_func(
                        owned_hashes, owned_host_indices, extra_info
                    )
                else:
                    success = True
            else:
                success = self.page_set_func(
                    batch_hashes, batch_host_indices, extra_info
                )
            if not success:
                logger.warning(
                    f"Write page to storage: {len(batch_hashes)} pages failed."
                )
                break

            # Best-effort draft L3 write alongside target.
            if self.draft_tier_armed("l3-write"):
                self._draft_page_set(batch_hashes, batch_host_indices)

            if prefix_keys and len(prefix_keys) > 0:
                prefix_keys += batch_hashes
            operation.completed_tokens += self.page_size * len(batch_hashes)

    def backup_thread_func(self):
        """
        Manage backup operations from host memory to storage backend.
        """
        while not self.storage_stop_event.is_set():
            try:
                operation = self.backup_queue.get(block=True, timeout=1)
                if operation is None:
                    continue

                # W35 CLASS 4: the durable-corruption gate. Checked HERE, on
                # the consumer thread, immediately before the persist and
                # after the operation has been dequeued -- the one point where
                # this operation's fate is decided. See `operation_is_stale`
                # for why a stale backup is REFUSED rather than routed.
                # Acked either way: an unacked operation stalls the queue, and
                # a declined backup is a correct non-persist, exactly as
                # `backup_skip` already is.
                if operation_is_stale(self, operation, "backup"):
                    self.ack_backup_queue.put(operation)
                    continue

                if not self.backup_skip:
                    # Capacity watchdog: rate-limited inside the backend, so this
                    # is cheap per operation. A backend that has stopped writing
                    # (out of disk) still runs the backup: the individual page
                    # writes turn into refusals, which the ack path already
                    # treats as a partial backup, so the caller sees a cache
                    # miss rather than a stalled queue.
                    self.storage_backend.check_disk_space()
                    # #1061: stamp every key this operation hands to the store
                    # with the #719 binding generation it was OPENED under --
                    # the stamp `StorageOperation.__init__` already carries
                    # (the existing carrier; no second scheme). This is the
                    # producer-phase ledger's only production writer; disarmed
                    # it records nothing. Placed BEFORE `_page_backup` so a
                    # partial backup can only OVER-stamp: a stamped-but-
                    # unwritten key reads as a store miss later, never as a
                    # wrongly-attributed phase.
                    try:
                        from sglang.srt.mem_cache.producer_phase_census import (
                            note_backup_keys,
                        )

                        note_backup_keys(
                            getattr(operation, "hash_value", None),
                            getattr(operation, "binding_generation", None),
                        )
                    except Exception:  # noqa: BLE001 - never break the backup thread
                        pass
                    self._page_backup(operation)
                self.ack_backup_queue.put(operation)

            except Empty:
                # Idle: an idle backend is exactly when a filesystem filling up
                # from other writers would otherwise go unnoticed.
                if self.enable_storage and self.storage_backend is not None:
                    try:
                        self.storage_backend.check_disk_space()
                    except Exception as e:  # never kill the worker over a probe
                        logger.warning(f"HiCache capacity watchdog failed: {e}")
                continue
