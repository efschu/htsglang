from __future__ import annotations

import hashlib
import functools
import logging
import os
import time
import traceback
from collections import OrderedDict
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
from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind
from sglang.srt.weg2 import seam_digest

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

#: #1284: the NEED series and the W51 refusal that reads it.  Imported here
#: rather than inlined because the guard is pure arithmetic over a clock and a
#: stats callable, and that is the half that can be tested with no ring, no
#: CUDA and no boot.
#:
#: Only the guard is imported.  ``Weg2HostRingUnfunded`` is deliberately NOT
#: caught here: it must propagate out of :meth:`release_memory_occupation` so
#: the leg's RPC answers non-200 and the front issues its own named stop, the
#: same way any other leg failure is reported.  Catching it would turn a
#: refusal into a silent partial sleep.
from sglang.srt.weg2.ring_guard import RingNeedGuard  # noqa: E402

logger = logging.getLogger(__name__)


def _weg2_exc_note(exc: BaseException, *, limit: int = 120) -> str:
    """``Type: message @ file:line`` for a SWALLOWED exception (#1328).

    The observer arms of the shadow may never raise into a flip leg, so they
    catch ``BaseException`` and return a reason string. Until this existed the
    string was ``type(exc).__name__`` alone, and boot weg2xsn6 spent 24 of 24
    legs reporting ``manifest-failed:AttributeError`` -- a hint that named
    neither the attribute nor the site, and on which three separate
    hypotheses were built and then refuted (``region.boot_hash``, which
    ``__init__`` always sets; ``agreed.theirs``, which ``AgreedPieces``
    carries; ``derive_card_manifest``, whose failures are clean returns).

    The last frame of the exception's OWN traceback is the site that raised,
    which is the one fact a type cannot carry. Bounded and newline-free so it
    stays one grep-able field on an existing line, and defensive throughout:
    an instrument that raises while describing a failure replaces the finding
    with its own.
    """
    try:
        import traceback as _tb

        msg = " ".join(str(exc).split())
        site = ""
        frames = _tb.extract_tb(exc.__traceback__)
        if frames:
            last = frames[-1]
            site = f" @ {last.filename.rsplit(chr(47), 1)[-1]}:{last.lineno}"
        note = (f"{type(exc).__name__}: {msg}{site}" if msg
                else f"{type(exc).__name__}{site}")
        return note[:limit]
    except BaseException:  # noqa: BLE001 -- describing a failure may not fail
        return type(exc).__name__


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


# ---------------------------------------------------------------------------
# #1285: EPOCH-SCOPED LEG DEDUP -- what makes a retried flip leg safe
#
# NEITHER LEG IS IDEMPOTENT.  Measured on this tree, not assumed:
#   * `resume_memory_occupation` does `self.offload_tags.remove(tag)` for every
#     tag (below).  `set.remove` raises KeyError on a tag that is not present,
#     so a second wake for the same tags dies at its first statement -- and
#     `memory_saver_adapter.resume(tag)` on an already-mapped tag is a second
#     recommit of pages that are already there.
#   * `release_memory_occupation` calls `memory_saver_adapter.pause(tag)`
#     unconditionally, and `sleep_begins`/`family_paused_before` -- which gate
#     the census, the static-state export and the credit -- are derived from
#     `len(self.offload_tags)`, i.e. they read FALSE on a repeat and silently
#     turn the second sleep into a different operation.
# So the front may not simply re-send a leg whose fate it does not know, which
# is exactly the weg2sb5e situation: `ServerDisconnectedError` with no access
# line on the peer means applied / partly applied / not applied at all are all
# consistent with what the client saw.
#
# The handlers ALREADY receive the flip's epoch (io_struct.py, `epoch` on both
# ReqInputs; the front sends it at front.py's gathered legs).  They did NOT key
# on it: the only reader was the VRAM-credit counter
# (`_weg2_open_credit_for_leg` / `_weg2_credit_reader`).  This ledger makes the
# epoch mean what its presence implies -- a leg identified by (op, epoch, tag
# set) is applied AT MOST ONCE per rank, and a repeat returns the recorded
# answer without touching VRAM.
#
# WHY A COMPLETED-OUTCOME LEDGER IS ENOUGH (no in-flight state).  The handler
# runs inside the scheduler loop, which processes one control request at a time
# per rank.  A retry therefore cannot interleave with a first attempt still
# executing on the same rank: by the time the repeat is dispatched, the first
# has RETURNED on that rank, or the rank is already dead --
# `_weg2_group_stop_on_leg_failure` stops the group on a leg that raised.
#
# WHY IT CANNOT SPLIT THE RANKS (memory `raenge-nie-uneins`).  The decision is
# a pure function of the REQUEST (op, epoch, tags) and of a per-rank record
# every rank writes at the same leg, and the RPC fans out to every rank through
# the same communicator.  All ranks therefore hit or miss together, and the
# group fence further down is entered by all or by none.
#
# WHY IT CANNOT TOUCH THE STOCK PATH.  It engages ONLY when `epoch` is not
# None.  Upstream never sets it, and neither do the front's two un-epoched
# single-tag RPCs, so those requests are byte-identical to what they were.
#: How many completed legs a rank remembers.  A flip has two; the window only
#: has to outlive one RPC retry, and the record is three small tuples.
WEG2_LEG_LEDGER_MAX = 8


class Weg2LegLedger:
    """Per-rank record of COMPLETED epoch-scoped flip legs (#1285)."""

    __slots__ = ("_cap", "_done")

    def __init__(self, cap: int = WEG2_LEG_LEDGER_MAX):
        self._cap = cap
        self._done: OrderedDict[Tuple, Any] = OrderedDict()

    @staticmethod
    def key(op: str, epoch: Any, tags: Optional[Sequence[str]]) -> Tuple:
        # The TAG SET, sorted: the front sends a permutation-checked order and
        # the leg's effect does not depend on it, so two orders of the same
        # family are the same leg and must dedup against each other.
        return (op, str(epoch), tuple(sorted(tags or ())))

    def recorded(self, key: Tuple) -> Any:
        return self._done.get(key)

    def record(self, key: Tuple, out: Any) -> None:
        self._done[key] = out
        self._done.move_to_end(key)
        while len(self._done) > self._cap:
            self._done.popitem(last=False)


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
    #: #1295: this rank's W8b verdict on the L3 store index it rebuilt at the
    #: wake, empty when there is none.  Written by
    #: :meth:`_weg2_rescan_store_index`, read and cleared by the resume fence,
    #: which votes it through C15's ok-bit so ONE owner's finding stops EVERY
    #: rank instead of killing the owner alone.  A FIELD for the fourth time in
    #: this class, for the reason the three comments above give.
    weg2_store_rescan_failure: str = ""
    #: C16: this rank's card, resolved once.  ``"unset"`` is distinct from
    #: ``None``, which is the resolved answer "no card key" -- so an
    #: unresolvable card is not re-resolved (and re-logged) on every tag.
    weg2_card_uuid_cache: Any = "unset"
    #: #1285: this rank's record of COMPLETED epoch-scoped flip legs, so a
    #: front retry after a ServerDisconnectedError replays instead of
    #: re-applying.  A FIELD for the third time in this class: ``slots=True``
    #: turns a lazily-assigned ``self._weg2_leg_ledger`` into an
    #: ``AttributeError`` raised ONLY on the retry path, i.e. only after a
    #: failure has already happened -- the worst possible place to learn it.
    weg2_leg_ledger: Any = None

    #: #1329: FIELDS FOR THE FOURTH AND FIFTH TIME IN THIS CLASS, and the
    #: comment above called it three commits early. ``slots=True`` turns a
    #: lazily-assigned attribute into an ``AttributeError`` ON THE WRITE, and
    #: the two S6b shadow caches were assigned lazily
    #: (``self._weg2_shadow_region_cache = region``, weight_updater.py:1613 in
    #: :meth:`_weg2_shadow_region`; ``self._weg2_shadow_manifest_cache =
    #: entries``, weight_updater.py:1655 in :meth:`_weg2_shadow_manifest`.
    #: Both line numbers are THIS tree's; the boot log named 1590 because the
    #: fields above did not exist yet and shifted everything below them.)
    #:
    #: MEASURED, boot weg2xsn7 @ 376ae2a475 -- 24 of 24 legs, BOTH groups,
    #: both hooks: ``manifest=manifest-failed:AttributeError:
    #: 'SchedulerWeightUpdaterManager' object has no attribute
    #: '_weg2_shadow_region_cache' @ weight_updater.py:1590``. Every shadow leg
    #: of every boot on this arm died on the first write, which is why
    #: ``WEG2-XCHG-PLAN`` was 0 on P and D and the shadow has never once run.
    #:
    #: The READS were already safe (``getattr(self, ..., "unset")`` /
    #: ``..., None``), so the sentinel semantics are preserved exactly: the
    #: region cache defaults to the same ``"unset"`` the getattr default used,
    #: which is DISTINCT from a cached ``None`` (the region legitimately opens
    #: to None on a boot with no region, and that answer must be cached rather
    #: than retried on every leg).
    _weg2_shadow_region_cache: Any = "unset"
    _weg2_shadow_manifest_cache: Any = None

    #: #1350 SEAM GRADER: this rank's pre-pause reading of its own pieces, or
    #: ``None``.  A FIELD for the sixth time in this class, for the reason the
    #: five comments above give -- ``slots=True`` turns a lazily assigned
    #: attribute into an ``AttributeError`` ON THE WRITE, i.e. on the first
    #: armed flip of an instrument boot, which is the worst place to learn it.
    #:
    #: LIFECYCLE, per the standing rule that every new state field carries its
    #: own table BEFORE the boot:
    #:   WRITER  -- :meth:`_weg2_seam_digest_before`, at the FIRST weights RPC
    #:              of a sleep (``not family_paused_before``), while every page
    #:              is still mapped.
    #:   READER  -- :meth:`_weg2_seam_digest_after`, after the landing of the
    #:              next wake (inside ``family_complete``, after the reload).
    #:   DELETER -- :meth:`_weg2_seam_digest_after` itself, read-and-clear, and
    #:              the unarmed path, which CLEARS rather than inherits.
    #: SEPARATING EVENT -- the pause and the whole dormancy between the two.
    #: Nothing else touches it: no cutover, no fence, no replay.  A deleter
    #: between writer and reader is what the rule looks for, and there is none.
    weg2_seam_before: Any = None

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

    def _weg2_ring_stats(self) -> Optional[dict]:
        """#1284: the live host-ring counters, or ``None`` for ABSENCE.

        ``None`` covers every way there is no reading to be had -- no adapter,
        the no-op adapter (which RAISES ``NotImplementedError`` rather than
        returning anything), no ring published on this boot, or a saver too old
        to export ``tms_ring_stats``.  It is deliberately NOT a zero: a zero
        free would make :class:`RingNeedGuard` refuse every leg on a boot that
        simply has no ring, which is the ordinary non-Weg-2 path.
        """
        adapter = getattr(self, "memory_saver_adapter", None)
        getter = getattr(adapter, "ring_stats", None)
        if getter is None:
            return None
        try:
            return getter()
        except NotImplementedError:
            return None
        except Exception:  # noqa: BLE001
            # A broken stats path must not decide a flip.  The acquire keeps
            # whatever behaviour it had; only the guard stands down.
            return None

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

    #: The four possible carriers of the weight bytes on a wake.  Module-level
    #: strings on the class rather than literals at the branches: the seam's
    #: test substitutes them, and a typo in a literal would silently take the
    #: `disk` path -- which is the one that costs 12-17 s and puts the dormant
    #: image back on the disk route the exchange exists to remove.
    CARRIER_STOCK = "stock"
    CARRIER_TMS_BACKUP = "tms-backup"
    CARRIER_EXCHANGE = "exchange"
    CARRIER_DISK = "disk"

    def _weg2_wake_weight_carrier(self) -> str:
        """WHO carries the weight bytes on this wake.  One of four, or W4.

        #1273 S6 step 5.  This is the decision the refill used to make inline,
        lifted out for the reason #1329 cost three boots in this same file: a
        decision inside a long method has no executing test, and every test of
        that slice drove the module functions while the mixin's own methods --
        the only callers the product has -- had none.

        THE FOUR, and they are mutually exclusive:

        ``stock``       ``--enable-memory-saver`` absent: ``pause()`` was
                        ``pass``, nothing was released, there is nothing to
                        refill.  This wins over every other answer -- a stock
                        resume must be byte-for-byte the upstream path.
        ``exchange``    ``--weg2-weight-source exchange`` AND
                        ``--weg2-xchg-inject authoritative``: the peer group's
                        live VRAM is the source, through the bounded host
                        bounce.  The bytes MUST NOT come from disk here.
                        ASKED FIRST of the three below ``stock`` (#1342 S1):
                        it used to sit behind ``tms-backup``, and because the
                        weg2 launcher arms the backup UNCONDITIONALLY that
                        made this answer unreachable on every arm at every
                        argv -- see the comment at the branch itself.
        ``tms-backup``  ``--enable-weights-cpu-backup``: the TMS restore
                        already carried the bytes (2.08 s / 27 GiB, campaign
                        (a)).  Wins over ``disk`` and over a NON-authoritative
                        exchange, so a ``shadow`` wake with a backup armed is
                        this and not ``disk``.  It no longer wins over an
                        AUTHORITATIVE exchange: two writers for one payload is
                        the ein-job-ein-mover defect only while neither writer
                        is the declared authority, and under ``authoritative``
                        one of them is.
        ``disk``        none of the above: the upstream
                        ``update_weights_from_disk`` path, unchanged, which is
                        every ordinary RL boot.

        The three W4 refusals are UNCHANGED and stay AHEAD of every answer: an
        undecidable wake, shards that disagree about the backup, and a separate
        draft checkpoint.  VRAM has already been mutated by the time this runs,
        so a wake that cannot be decided refuses rather than guesses -- and a
        new branch that answered its own question before these checks would
        turn an undecidable wake into a silent injection.
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
            return self.CARRIER_STOCK

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
        # #1342 S1: THE EXCHANGE IS ASKED FIRST -- ahead of `main_carried`,
        # still BEHIND `stock` and behind all three W4 refusals above.
        #
        # WHY THE ORDER MOVED, measured on boot weg2xsn17 (571a3da963, record
        # BOOT_weg2xsn17_0911.md).  `main_carried` used to return here FIRST,
        # and it is `server_args.enable_weights_cpu_backup`, which the weg2
        # launcher passes UNCONDITIONALLY in `common_flags`.  So this method
        # could never reach the test below, `CARRIER_EXCHANGE` was unreachable
        # on EVERY arm at EVERY argv, and FOUR boots of S6I instruments graded
        # nothing.  The proof is positive rather than inferred: on that boot
        # `Weg2WakeRefused` was bare 0 / genuine 0, which is what a
        # never-entered raiser looks like and what an entered one could not be.
        #
        # THE ein-job-ein-mover OBJECTION IS NOT REFUTED, IT IS SUPERSEDED,
        # and it is written out because it is the argument this reorder
        # overturns.  The old precedence said: if the TMS backup carried the
        # bytes the exchange must not write them too, because two writers for
        # one payload is the defect.  True -- but only while neither writer is
        # the declared authority.  Under `--weg2-xchg-inject authoritative`
        # the exchange IS the authority by definition, so the situation is
        # "one authority plus a redundant restore of the same bytes":
        # wasteful, not incorrect.  B7 closes it properly by taking the host
        # image from 42.96 GiB to 0.00; until then the redundancy is the price
        # of the arm being REACHABLE, and an unreachable arm cannot be graded.
        #
        # `shadow` MUST NOT CHANGE ANSWER, and after this reorder it does not
        # -- but it now declines for the RIGHT REASON.  It fails the
        # `inject_authoritative()` conjunct below and falls through to
        # `main_carried`, so a shadow wake with a backup is `tms-backup` (not
        # `disk`: the backup really did carry the bytes) and without one is
        # `disk`.  Before, the first of those two was decided by a flag's
        # position rather than by the arm.
        try:
            from sglang.srt.weg2 import weight_exchange as wx

            # BOTH CONDITIONS, and the second is step 6c's whole point.
            # `--weg2-weight-source exchange` says the exchange is the weight
            # SOURCE; `--weg2-xchg-inject authoritative` says its injection has
            # REPLACED the refill.  Under the default `shadow` the refill stays
            # the authority and the injection grades itself beside it -- a boot
            # that armed the exchange and has not been graded must not become
            # the authority BY OMISSION, which is exactly the direction S6I
            # exists to close.
            if wx.exchange_armed() and wx.inject_authoritative():
                return self.CARRIER_EXCHANGE
        except Exception:  # noqa: BLE001 -- an unreadable arm is not an arm
            # An unreadable arm falls through to the carriers below, which is
            # the conservative direction: `tms-backup` if the backup is armed,
            # else `disk`.  Both are answers that serve correct bytes; only
            # `exchange` would depend on the arm this except clause could not
            # read.
            pass
        if main_carried:
            return self.CARRIER_TMS_BACKUP
        return self.CARRIER_DISK

    def _weg2_xchg_inject_weights(self, **kw) -> None:
        """Fill the remapped weight pages from the PEER GROUP, not from disk.

        #1273 S6 step 5, the authoritative half.  The bytes come from the peer
        group's live VRAM instead of from ``update_weights_from_disk``
        (12.073/14.143/16.749 s per wake on this rig -- the disk route
        #1317/#1323/#1325 exists to remove).

        CORRECTED #1342: this docstring used to say that under ``exchange``
        the weights region is opened ``enable_cpu_backup=False``, so the
        resume recommits pages whose content is undefined.  THAT IS STALE AND
        IT MISREADS THE ARM.  ``enable_cpu_backup`` is computed at
        ``model_runner.py:2440-2442`` from ``server_args`` with NO arm
        predicate, and the weg2 launcher passes ``--enable-weights-cpu-backup``
        unconditionally, so the region IS cpu-backed on this arm and the TMS
        restore does carry the bytes (boot weg2xsn17 served correctly through
        8 flips on exactly this configuration, which is the corroborating
        reading).  The stale sentence mattered: it makes ``tms-backup`` look
        like a wrong answer for this arm, and it is not -- it is a redundant
        one, which is why the carrier decision now asks the AUTHORITY question
        rather than the "who could have carried it" question.

        THE TERM IS THE LAUNCHER'S, READ AND NEVER DERIVED.  A rank may not
        re-run the checkpoint census: that would be a second sizing authority
        beside ``bounce_terms``, which is the defect class AMENDMENT 5 retired
        one instance of.  ``xchg_bounce.read_published_terms`` rebuilds the
        launcher's own priced inputs through the one sizing function, and an
        ABSENT publication is a REFUSAL rather than a locally chosen size --
        injecting against a buffer nobody priced is how the reap mark gets
        crossed by bytes no ledger carries, which
        ``host-schwelle-nie-uebertreten`` forbids.

        IT RAISES, unlike every shadow hook in this file.  An observer that
        took a flip down over its own bookkeeping would be wrong; an AUTHORITY
        that swallowed would serve UNDEFINED WEIGHTS, which is worse than a
        refusal by exactly the margin this campaign is about.
        """
        from sglang.srt.weg2 import xchg_bounce as xb

        terms = xb.read_published_terms()
        if terms is None:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: --weg2-weight-source exchange owns this "
                f"wake's weight bytes, but {xb.ENV_BOUNCE_TERMS} was not "
                "published, so the bounce geometry the injection needs was "
                "never priced by the launcher. Refusing rather than sizing a "
                "pinned host buffer locally: the resume has already remapped "
                "the weight pages and their content is undefined, so serving "
                "is not an option either. Launch through the weg2 launcher, "
                "which publishes the term it charged on the ARM line."
            )
        self._weg2_xchg_inject_from_peer(terms=terms, **kw)

    def _weg2_xchg_inject_from_peer(self, *, terms, **kw) -> None:
        """The transfer itself, once the term is known.

        SEPARATE FROM THE DECISION ABOVE so the seam's test can drive the
        decision without a device, and so the transport work has one entry
        point.  It delegates to :meth:`_weg2_xchg_bounce_leg`, which carries
        the whole plan.  (It used to name a second leg for the byte-identical
        pieces; that path was deleted in #1342 S3 -- see the note at the
        section header below for why, and the "ONLY PATH (b)" paragraph.)

        WIRED #1342 S2.  This docstring used to end "NOT YET REACHABLE ON THE
        METAL, and it says so rather than pretending", and the body was a bare
        ``raise`` -- so the sentence was true and the delegation it described
        was fiction.  Boot weg2xsn17 is what that cost: zero
        ``WEG2-XCHG-INJECT`` lines on a boot whose flip path demonstrably ran.

        BOTH OF THE OLD REFUSAL'S PREMISES WERE STALE, which is why it could be
        removed rather than merely relaxed:

        * *"the plan provider has no registrant (TODO(S6))"* -- it does.
          ``arm_coverage_at_load`` is called from ``model_runner.py:2564`` at
          load time and registers it via ``install_default_plan_provider``
          (``weight_exchange.py:3109``).  Corroborated on metal: weg2xsn17
          emitted 18 (P) / 21 (D) real ``WEG2-XCHG-PLAN card=`` lines.
        * *"the on-card diagonal lane is being fixed ... after XSN9 died on
          ``pair_id(c, c)``"* -- fixed (#1334), and weg2xsn17 realised
          ``oncard-*.bin`` at 3 x 33,554,432 B.

        THE REFUSALS THAT REMAIN ARE THE IDENTITY SWEEP, and they keep the
        property the old body had by accident: an input this method cannot
        resolve is a NAMED refusal, never a silent return.  The resume has
        already remapped the weight pages, so returning without filling them
        serves whatever the remap left behind -- that direction is pinned by
        ``test_a_missing_plan_still_refuses_by_name``.

        ONLY PATH (b) IS DRIVEN, and the reason is recorded because the
        docstring used to promise two.  Path (a) -- the byte-identical
        card-to-card set -- was DELETED in this slice, not left unwired: its
        input is the per-flip-leg agreement verdict from
        ``reconcile_card_manifest``, which does not exist at a wake refill
        (this method has no leg identity: ``_weg2_wake_reload_weights`` calls
        it with no request), and path (b) carries those bytes anyway.  Measured
        on boot weg2xsn8 the agreed set was 4.90 MiB of a 27.52 GiB image --
        0.018 %, a second mover for the same payload rather than a throughput
        argument, which is the UPSTREAM-MINIMAL delete shape.
        """
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import weight_exchange_region as xr

        priced = f"({terms.total_bytes} B, {terms.expression()})"

        # -- the identity sweep, in the order the legs consume it ------------
        boot_nonce = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
        if not boot_nonce:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the exchange owns this wake's weight "
                f"bytes and the launcher priced its bounce {priced}, but "
                f"{xr.ENV_REGION_BOOT} is empty, so no region epoch names the "
                "shared buffer the legs key every offset by. Refusing rather "
                "than guessing a nonce: the resume has already remapped the "
                "weight pages and their content is undefined."
            )
        group = self._weg2_group_name()
        rank = self._weg2_rank()
        if group not in ("P", "D") or rank < 0:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the exchange owns this wake's weight "
                f"bytes, but this rank has no Weg-2 identity (group={group!r} "
                f"rank={rank}). Every row the plan addresses is keyed by the "
                "rank's index inside its group; a sentinel may not travel into "
                "a leg (the #1273 S6 fix-D class)."
            )
        device = self._weg2_device_index()
        if device < 0:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the exchange owns this wake's weight "
                f"bytes, but torch reports no CUDA device on rank {rank} "
                f"(group {group}), so there is no device for the bounce leg to "
                "slice into. Refusing rather than injecting onto device -1."
            )

        # -- the plan, through the ONE product call site of build_plan -------
        # #1345: NOT NARROWED, and ``agreed=None`` is now a STATED position
        # rather than an input this path failed to gather.  There is no peer
        # question here: path (b) carries the bytes, the compare's reference is
        # the LOCAL landed weights (``ptr_of``), and the agreed set was 0.018 %
        # of the image.  The reconcile is deliberately NOT called from this
        # path -- it would be a second writer of a row whose writer is declared
        # unique, and this method has no leg identity to key one by anyway
        # (``_weg2_shadow_manifest`` requires ``leg`` and ``epoch``; both
        # callers of this one pass neither).  Pinned by
        # ``test_no_manifest_writer_is_reachable_from_the_injection_path``.
        plan, plan_reason = self._weg2_shadow_plan(
            "authoritative", group, int(rank), agreed=None,
            require_agreement=False)
        if plan is None:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the exchange owns this wake's weight "
                f"bytes and the launcher priced its bounce {priced}, but no "
                f"weight-exchange PLAN reached rank {rank} (group {group}): "
                f"{plan_reason}. Refusing: returning would serve whatever the "
                "remap left behind, and falling back to the disk refill would "
                "silently restore the very route this arm exists to remove."
            )

        ops = self._weg2_xchg_device_ops()
        if ops is None:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: the exchange owns this wake's weight "
                f"bytes, but the CUDA device-ops layer could not be opened on "
                f"rank {rank}, so no copy can be issued at all."
            )

        # THE MODE IS THE CALLER'S IF IT GAVE ONE, else the flag's, and it is
        # resolved ONCE here and passed down -- a leg that re-read it further
        # in could act on a different answer than the one it was entered with,
        # and the two differ by "does this write into the live weights".
        #
        # THE CALLER'S `mode` MUST WIN, and getting this wrong was a real bug
        # in the first version of this wiring: it computed `wx.inject_mode()`
        # and IGNORED `kw` entirely. There are two callers, and one of them
        # passes the mode explicitly --
        # `_weg2_xchg_shadow_compare` calls `_weg2_xchg_inject_weights(
        # mode=wx.INJECT_SHADOW)` (this file, the step-6c grader). On the S6I
        # order's own argv the two agree, so the bug would have been INVISIBLE
        # and still wrong: an argument published by the caller, read into
        # `**kw`, and never acted upon -- the exact class #1256 names and this
        # slice exists to remove. Silently dropping it would also break the
        # grader the day the flag and the call site disagree, which is
        # precisely when a grade matters.
        mode = str(kw.get("mode") or "") or wx.inject_mode()
        # `descs`, NOT `raw_descs` (#1342 B4, the FIFTH defect of this slice's
        # own form, found by the chain smoke at the desk).
        #
        # `_weg2_shadow_plan` returns `derive_leg_plan`'s **LegPlan**, and a
        # LegPlan has `descs` and no `raw_descs` at all -- `raw_descs` is an
        # XchgPlan field.  The chain died here with
        # `AttributeError: 'LegPlan' object has no attribute 'raw_descs'`,
        # swallowed by the observer into a single `verdict=NO-COMPARE` line,
        # which the grading plan scores as a FAIL and not a neutral.
        #
        # `descs` IS THE RIGHT POPULATION, not merely the attribute that exists:
        # `derive_leg_plan` sets `descs=tuple(plan.descs)` from the XchgPlan, so
        # this is the COALESCED set -- the pieces the leg should actually move.
        # `raw_descs` would have been the uncoalesced population even where it
        # existed.
        #
        # Why the unit test missed it, recorded because it is the same miss
        # twice: the double was shaped after what THIS line reads, so it grew a
        # `raw_descs` the producer never had. The chain smoke builds a real
        # LegPlan from the producer's field set instead.
        self._weg2_xchg_bounce_leg(
            descs=list(plan.descs), ops=ops, boot_nonce=boot_nonce,
            terms=terms, mode=mode, device=int(device),
        )

        # THE INSTRUMENTS ARE EMITTED BY THE LEG, NOT HERE, and the first
        # version of this method got that wrong in a way worth recording: it
        # called `inject_summary_line([result])` with a `BounceResult` where a
        # `Sequence[InjectVerdict]` belongs.  It raised on the first real
        # object the test handed it -- which is exactly the value of driving
        # the product's own types instead of a hand-written double.
        #
        # THE DEEPER REASON IT WAS WRONG: this path runs only under
        # `authoritative`, and `run_bounce_leg` sets `comparing = mode ==
        # INJECT_SHADOW`, so an authoritative leg produces NO `InjectVerdict`
        # at all.  A summary emitted here would have counted a leg that never
        # compared, printing `NOT-CLEAN` for a correctly-running authoritative
        # wake -- an instrument lie in the same class as #1336.  The per-leg
        # `WEG2-XCHG-INJECT` line and the running summary are now emitted
        # inside `run_bounce_leg`, at the one site that holds the verdict.

    def _weg2_wake_reload_weights(self) -> None:
        """Fill the weight pages the resume recommitted, by whatever carries them.

        A ROUTER SINCE #1273 S6 step 5, not a decider: it asks
        :meth:`_weg2_wake_weight_carrier` and obeys the answer.  It used to
        read the flags itself, and the whole reason it no longer does is that
        the exchange arm needs a THIRD answer -- inject from the peer group --
        and a second reading of the flags at a second call site is how two
        answers to one question start to diverge (`ein-job-ein-mover`).
        `test_the_refill_asks_the_decision_and_does_not_re_read_the_flags`
        pins that by substitution: it forces the decision to say `tms-backup`
        on a configuration whose raw flags say `disk`, and a refill that
        re-read the flags would reload.

        With ``--enable-weights-cpu-backup`` the TMS restore already carried
        the bytes (measured 2.08 s / 27 GiB, campaign (a)) and this is a no-op.
        Without it, ``resume(GPU_MEMORY_TYPE_WEIGHTS)`` recommitted VMM pages
        whose CONTENT IS UNDEFINED, so the weights are refilled through the
        upstream ``update_weights_from_disk`` endpoint from the page-cached
        checkpoint.  No fork loader: the upstream path is the path.  Under
        ``--weg2-weight-source exchange`` neither applies and the peer group's
        live VRAM is the source (:meth:`_weg2_xchg_inject_weights`).

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
        carrier = self._weg2_wake_weight_carrier()
        if carrier in (self.CARRIER_STOCK, self.CARRIER_TMS_BACKUP):
            # Nothing to refill: either pause() never released, or the TMS
            # restore already wrote every byte.  A second writer here would be
            # the ein-job-ein-mover defect.
            return
        if carrier == self.CARRIER_EXCHANGE:
            self._weg2_xchg_inject_weights()
            return
        server_args = self._weg2_server_args()

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
        # THE STEP-6c COMPARE WAS HERE AND IS GONE (#1342 S1b).  GRADING IS NOT
        # REFILLING, and this method is a refill.
        #
        # It sat at the end of this branch with a comment that was half right:
        # "the refill has just written every weight byte" IS the correct
        # precondition, but it is only reached when the refill actually runs.
        # This method has FOUR carriers and two of them return at :1137-1141 --
        # and `tms-backup` is the carrier EVERY weg2 boot on this rig selects,
        # because the launcher passes `--enable-weights-cpu-backup`
        # unconditionally.  So the grader was unreachable on the only
        # configuration that grades, which boot weg2xsn18 measured as 0
        # `WEG2-XCHG-INJECT` lines with `Weg2WakeRefused`/`W4`/`NO-COMPARE` all
        # 0 -- a never-entered method, not a failing one.
        #
        # The early return above is NOT the defect and is untouched: its own
        # comment is right that a second writer here would be the
        # ein-job-ein-mover defect.  The grader moved to the WAKE path, after
        # the weights-family resume and after this method returns, which is
        # where the bytes have landed whichever carrier wrote them -- the same
        # placement rule the shadow's destination hook states and follows.
        # See `resume_memory_occupation`.

    def _weg2_xchg_shadow_compare(self) -> None:
        """Grade the exchange's assembled bytes against the weights as landed.

        STEP 6c.  Called from the WAKE path (`resume_memory_occupation`), after
        `family_complete` and after `_weg2_wake_reload_weights` -- i.e. once the
        bytes are final no matter WHICH carrier wrote them.  #1342 S1b moved it
        there out of the refill branch, where it was only reachable when the
        disk refill actually ran; see the note left at the old site.

        THREE ARMS DECLINE, and each for its own reason rather than by
        placement luck:

        ``ring``          nothing was exchanged, so there is nothing to grade.
        ``authoritative`` the injection IS the authority and grades itself, and
                          its disagreement is a GATE (the refusal lives in
                          :meth:`_weg2_xchg_inject_weights`).  Two graders for
                          one leg would be the ein-job-ein-mover shape.
        ``stock``         DANGER DIRECTION 1, and it is now a NAMED gate rather
                          than a consequence of where this call sits.  Without
                          ``--enable-memory-saver`` every ``pause()`` was a
                          no-op: the weights were never released, never
                          recommitted and never rewritten, so a compare would
                          grade the same bytes against themselves and any
                          MISMATCH could only be an instrument fault.  The
                          carrier is asked through the ONE authority,
                          :meth:`_weg2_wake_weight_carrier` -- a READ of a pure
                          decision, not a second decider.

        AN OBSERVER, so it never raises.  Here the refill (or the TMS restore)
        is the authority and the model is already correct; a compare that took
        the flip down over its own bookkeeping would be the thing S6I exists to
        avoid -- the grade is evidence, not a gate.
        """
        try:
            from sglang.srt.weg2 import weight_exchange as wx

            if not (wx.exchange_armed()
                    and wx.inject_mode() == wx.INJECT_SHADOW):
                return
            # DANGER DIRECTION 1, gated here so the call site stays a plain
            # call and the whole question lives in ONE place.
            if self._weg2_wake_weight_carrier() == self.CARRIER_STOCK:
                return
            self._weg2_xchg_inject_weights(mode=wx.INJECT_SHADOW)
        except BaseException as exc:  # noqa: BLE001 -- an observer never raises
            logger.info(
                "WEG2-XCHG-INJECT mode=shadow verdict=NO-COMPARE pieces=0 "
                "bytes=0 rows=0 mismatches=0 mismatch_first=- -- the grade "
                "could not be taken (%s: %s); the refill remains the "
                "authority and the weights are unaffected",
                type(exc).__name__, exc,
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
        # B4g wall 1: THE COVERAGE VERDICT, consumed at last.  `arm_coverage`
        # records it at load and cannot raise there (no fence in scope, refuter
        # F5); this IS the fence its own docstring names, so the verdict becomes
        # an action here and nowhere else.  Under `shadow` the leg is disarmed
        # and the flip proceeds on the ring; under `authoritative` it votes
        # not-ok and every rank raises W29 carrying W84's own text.  Boot
        # weg2xsn14 printed uncovered=12 on 39 of 40 COVER lines with W84
        # genuine 0 because this consumer did not exist.
        from sglang.srt.weg2 import weight_exchange as wx

        coverage_armed, coverage_reason, coverage_stop = wx.coverage_leg_decision()
        if coverage_reason:
            logger.error("%s", coverage_reason)
        mine = {
            "rank": rank,
            "ok": bool(ok) and not coverage_stop,
            "failure": (str(failure or "")
                        or (str(coverage_reason) if coverage_stop else "")),
            "coverage_armed": bool(coverage_armed),
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

    def _weg2_rescan_store_index(self) -> None:
        """Re-read the L3 store directory into the LRU index at this wake.

        #1295: ``HiCacheStorage.rescan_eviction_index`` was written as THE
        mitigation for two eviction owners over one directory -- its own
        docstring says "an index built once at boot is wrong after hours of the
        sibling's writes ... Each owner therefore re-scans when it wakes" -- and
        at 57fef0ce6e it had ZERO callers anywhere in ``python/sglang``.
        Measured on boot weg2sb5h: ``re-scanned at wake`` = 0 in both the P and
        the D log, while the store reached 1.34x its cap. The index the cap is
        enforced against, and the measurement of the bytes a sibling owner holds
        in the same directory, both go stale exactly across a sleep; this is the
        moment they are corrected, which is why it runs here rather than on the
        write path (re-walking 655k files per page would buy nothing -- a
        sleeping group performs no writes).

        ``Weg2StoreIndexBlind`` IS ESCALATED, AND NOT BY RAISING HERE.
        ``rescan_eviction_index`` records the obligation in its own docstring
        -- "a rank that dies HERE alone while its siblings wake on is still the
        disagreement §0 forbids" -- and a bare ``raise`` on this line does
        exactly that: ``LRUFileEvictor.rescan`` returns early for a non-owner
        (``_eviction_enabled`` = configured AND elected owner), so only PP0 / TP0
        can ever reach the raise, and PP1-2 / TP1-2 would clear dormancy and
        wake on. Fix 2 shipped that raise with a docstring CLAIMING a group-
        fatal escalation nothing in the code performed.

        The escalation is the OK-BIT of the fence that already closes this leg
        (``_weg2_group_fence`` at the end of ``resume_memory_occupation``, C15):
        every rank of the group joins it, the verdict is all-gathered, and ANY
        False makes EVERY rank raise ``Weg2FlipRankDisagree``. So this method
        RECORDS the refusal in ``weg2_store_rescan_failure`` and the fence
        turns one owner's finding into one group-wide STOP, with no new
        collective, no cross-rank protocol, and no rank left running.
        Everything else -- no store, no evictor, an OSError from the walk --
        leaves the wake alone; a stale index degrades the hit rate, and
        refusing the wake over it would be worse than the staleness.
        """
        sch = self.scheduler
        tc = getattr(sch, "tree_cache", None)
        if tc is None or not getattr(sch, "enable_hierarchical_cache", False):
            return
        controller = getattr(tc, "cache_controller", None)
        backend = getattr(controller, "storage_backend", None)
        if backend is None or not hasattr(backend, "rescan_eviction_index"):
            return
        t0 = time.perf_counter()
        try:
            census = backend.rescan_eviction_index()
        except Weg2StoreIndexBlind as e:
            self.weg2_store_rescan_failure = (
                f"W4 Weg2WakeRefused (#1295, via the C15 ok-bit): the L3 store "
                f"index rebuilt at this wake is blind -- {e}. A cap enforced "
                f"over a fraction of the sole handback carrier is not a cap, "
                f"and this group's whole awake phase would evict against "
                f"numbers that do not describe the disk. Recorded here and "
                f"voted at the resume fence so every rank of the group stops "
                f"together; a raise on this line would kill the eviction owner "
                f"alone while its siblings woke on."
            )
            logger.error("%s", self.weg2_store_rescan_failure)
            return
        except Exception as e:
            logger.warning(
                "WEG2-STORE-RESCAN skipped at wake: %s (%s) -- the eviction "
                "index keeps the numbers it had at the last census, so the cap "
                "is enforced against a stale reading of the directory",
                e,
                type(e).__name__,
            )
            return
        logger.info(
            "WEG2-STORE-RESCAN at wake: %d of %d files, %d of %d B (%.1f%%) "
            "indexed here; of the indexed bytes %d B are the SIBLING GROUP's "
            "pages under the shared #706 canonical suffix -- counted against "
            "this cap, never unlinked by this owner, because this store is the "
            "handback carrier -- and a further %d B carry a suffix this group "
            "does not scan; %d B are staging/partial files "
            "(instrument: os.scandir + os.stat over the store, charged at "
            "max(st_blocks*512, st_size) -- the #410 unit, not apparent size) "
            "in %.0f ms",
            census.get("indexed_entries", 0),
            census.get("seen_entries", 0),
            census.get("indexed_bytes", 0),
            census.get("seen_bytes", 0),
            100.0 * float(census.get("fraction", 1.0)),
            census.get("foreign_indexed_bytes", 0),
            census.get("foreign_bytes", 0),
            census.get("staging_bytes", 0),
            (time.perf_counter() - t0) * 1000,
        )

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
        """THIS RANK's 0..2 index inside its Weg-2 group, or ``-1``.

        MEASURED-BY-BOOT DEFECT (weg2shadowC, #1273 S6 fix C).  This read
        ``getattr(scheduler, "tp_rank")`` then ``"pp_rank"`` -- and **the
        Scheduler has neither**.  It keeps its parallel identity on the
        ``ParallelState`` wrapper, which the tree already states in three
        places, each written after the same read raised somewhere else
        (``scheduler.py:1766``, ``:8157-8161``, ``:14502-14504``).  Here it
        could not raise: the read is ``getattr(..., None)`` behind an
        ``isinstance(..., int)`` test, so it degraded SILENTLY to -1 on every
        rank of every Weg-2 boot.  Two consequences, one root:

        * every ``WEG2-FLIP-TAG`` line ever emitted printed ``rank=-1`` --
          210 of 210 in boot weg2shadowC's four rank logs -- and instead of
          the emitter being fixed, ``ring_table`` WIDENED its parser to
          ``rank=(-?\\d+)`` and grew a synthetic per-card index for it
          (``ring_table.py:159-166``, W37's docstring at ``:866``);
        * ``_weg2_shadow_hook``'s ``rank < 0`` gate returned before
          ``run_leg_hook`` on all four flips of both shadow arms, so the whole
          observer -- plan, sems, on-card refusals, the byte compare -- was a
          single silent ``return``.

        THE IDENTITY IS THE WORLD GROUP'S, NOT ``ps``.  ``scheduler.ps`` is
        PHASE state: the cutover REPLACES it with ``pp_rank=0`` on every rank
        (``phase_flip_runtime.py:3366-3374``, stated verbatim at ``:12235``),
        so it cannot tell ranks apart on a flip boot.  ``world_group`` is bound
        once (``scheduler.py:1867``) and the cutover rebinds the tp/attn/pp
        handles beside it but never this one; it is also the identity the
        cutover itself reads (``:3323``).

        AND ``ps.tp_rank`` ALONE WOULD BE WORSE THAN -1.  Group P runs
        ``pp_size=3, tp_size=1`` and group D ``pp_size=1, tp_size=3``
        (``launcher.py:6525``), so on P ``ps.tp_rank`` is 0 on all three ranks:
        three publishers on row 0 of the six-row gate matrix -- silently WRONG
        where -1 was merely silently absent.  The ``ps`` fallback below is
        therefore the FLAT world rank, the same arithmetic
        ``Scheduler._admin_world_rank`` uses, which reduces correctly on both
        group shapes and in both ``ps`` states.

        ``-1`` survives as the answer where there is genuinely no identity to
        read.  It may not become 0: rank 0 is a real row another rank owns.
        """
        scheduler = self.scheduler
        world_rank = getattr(
            getattr(scheduler, "world_group", None), "rank_in_group", None)
        if isinstance(world_rank, int):
            return world_rank
        ps = getattr(scheduler, "ps", None)
        try:
            return int(ps.pp_rank) * int(ps.tp_size) + int(ps.tp_rank)
        except Exception:  # noqa: BLE001 -- an unreadable identity is -1
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

    def _weg2_shadow_gate_rows(self, hook: str, group: str, leg: int):
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

        **#1337: THAT LAST SENTENCE IS FALSE ON LEG 1, and it cost 6/24 legs
        per group on XSN12.**  Rooted by boot seat 3: on the FIRST flip there is
        no earlier instant in which source rows could have been sealed, so the
        destination waited the whole ``SHADOW_GATE_BUDGET_S`` = 5.0 s for three
        rows nobody can write (measured 4.953/4.956/4.960 s) and then read
        ``ran=no``.  Under ``--weg2-xchg-inject authoritative`` that would be
        weight bytes nobody injected.

        The EXPECTATION is what was wrong, not the gate -- so on a leg whose
        source rows cannot yet exist the destination expects its OWN rows, for
        the mirror image of the reason the source already does.  From leg 2 the
        justification holds again and the cross-group expectation is restored,
        because narrowing it everywhere would silently stop checking the
        agreement this gate exists for.

        ``leg`` IS REQUIRED, deliberately with no default: the call site had
        ``leg`` in scope the whole time (``:2078``) and simply did not pass it,
        so a default is exactly the shape that let this survive.
        """
        from sglang.srt.weg2 import weight_exchange_region as xr

        if group not in ("P", "D"):
            return None
        if hook != "source" and int(leg) > 1:
            return None
        return tuple(xr.rank_row(group, r) for r in range(xr.N_CARDS))

    def _weg2_shadow_param_census(self, group: str, rank: int) -> None:
        """ONE LINE PER RANK, ONCE: the pp/tp asymmetry, read off the LOG.

        #1273 S6 fix F2, and it is the "cheap evidence" SECTION 1ai-F-root's
        UNPROVEN 2 named: the claim that group P holds PIPELINE STAGES and
        group D TENSOR SHARDS of the same card was read out of ``launcher.py``
        and corroborated only indirectly, by six disagreeing storage digests.
        A count plus the first and last parameter name per rank settles it from
        the boot's own log instead, and costs one sorted walk of
        ``named_parameters`` at the first hook of the boot.

        Never raises and never repeats: an observer that cost a leg its wall on
        every flip would be paying for evidence with the thing it observes.
        """
        if getattr(type(self), "_weg2_param_census_done", False):
            return
        try:
            type(self)._weg2_param_census_done = True
            runner = getattr(self.tp_worker, "model_runner", None)
            model = getattr(runner, "model", None)
            if model is None:
                return
            names = sorted(n for n, _ in model.named_parameters())
            if not names:
                return
            logger.info(
                "WEG2-XCHG-SHADOW PARAM-CENSUS group=%s rank=%s params=%d "
                "first=%s last=%s -- the pp/tp asymmetry, from this rank's own "
                "named_parameters(): a PP stage carries whole layers over a "
                "SUBSET of layer indices, a TP shard carries slices of EVERY "
                "layer, and the two are why the co-located pair's whole-storage "
                "digests differ while their on-card PIECE sets agree (#1273 S6 "
                "fix F2, SECTION 1ai-F-fix)",
                group, rank, len(names), names[0], names[-1],
            )
        except BaseException:  # noqa: BLE001 -- an observer never raises
            pass

    def _weg2_shadow_region(self):
        """This process's shadow region, opened once and kept.  ``None`` if absent.

        #1311 S6b.  The card manifest has to be published and read BEFORE the
        plan is derived (the plan is narrowed by the agreement), and the plan is
        derived before ``run_leg_hook`` opens its own region for the transport.
        So this adapter opens one itself, from the SAME two env vars
        ``ShadowLeg.attach`` reads -- no second channel is invented.

        CACHED PER PROCESS, and that is correct rather than convenient: the
        region file is per BOOT (its ``boot_hash`` names the boot nonce) and is
        merely re-STAMPED per flip, while what this adapter reads and writes --
        the manifest rows -- are keyed on ``boot_hash`` and are boot constants.
        The transport's own region handle stays :class:`ShadowLeg`'s; this one
        is never handed to a leg and never begins a flip.

        Never raises: an observer that took a flip down over its own bookkeeping
        would be the thing this whole slice exists not to be.
        """
        cached = getattr(self, "_weg2_shadow_region_cache", "unset")
        if cached != "unset":
            return cached
        region = None
        try:
            from sglang.srt.weg2 import weight_exchange_region as xr

            path = (os.environ.get(xr.ENV_REGION_PATH, "") or "").strip()
            boot = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
            if path and boot:
                region = xr.XchgRegion.open(path, expect_boot=boot)
        except BaseException:  # noqa: BLE001 -- an observer never raises
            region = None
        self._weg2_shadow_region_cache = region
        return region

    def _weg2_shadow_manifest(self, group: str, peer: str, rank: int, *,
                              leg: int, epoch: str):
        """Publish this rank's card manifest and agree with the co-located peer.

        ``(AgreedPieces | None, state)``.  THE FIX FOR BOOT weg2xsn5's W80.

        The manifest itself is a BOOT constant and is derived once per process;
        the reconciliation runs per leg because the PEER's row appears at the
        peer's own first hook, which is a different instant of a different flip
        half.  Reconciling is one shared-memory read plus a set intersection --
        microseconds against a derivation that walks every parameter.

        Never raises.  A ``None`` agreement is a NAMED state on the log, never a
        quiet fall-back to this rank's own view: the fall-back IS the defect
        this closes.
        """
        try:
            from sglang.srt.weg2 import weight_exchange as wx
            from sglang.srt.weg2 import weight_exchange_region as xr
            from sglang.srt.weg2 import weight_exchange_shadow as sh

            region = self._weg2_shadow_region()
            if region is None:
                return None, "no-region"
            entries = getattr(self, "_weg2_shadow_manifest_cache", None)
            if entries is None:
                runner = getattr(self.tp_worker, "model_runner", None)
                model = getattr(runner, "model", None)
                region_tag = ""
                if runner is not None:
                    try:
                        region_tag = wx.weights_region_tag_for(
                            wx.RunnerShape.of(runner))
                    except BaseException:  # noqa: BLE001
                        region_tag = ""
                entries, reason = sh.derive_card_manifest(
                    rank=int(rank), model=model, region_tag=region_tag)
                if entries is None:
                    return None, f"no-manifest:{reason}"
                self._weg2_shadow_manifest_cache = entries
            row = xr.rank_row(group, int(rank))
            peer_row = xr.rank_row(peer, int(rank))
            agreed, state = sh.reconcile_card_manifest(
                region, row=row, peer_row=peer_row, entries=entries)
            # ONE LINE PER LEG, AND IT NAMES BOTH CARDINALITIES.  The states
            # that are not ``agreed`` are the ones a boot record has to be able
            # to count, and two of them are ordinary startup rather than a
            # fault -- which is why only the overflow carries a W-code.
            logger.info(sh.manifest_state_message(
                state=state, rank=int(rank), row=row, peer_row=peer_row,
                leg=int(leg), epoch=str(epoch), mine=len(entries),
                theirs=(agreed.theirs if agreed is not None else -1)))
            return agreed, state
        except BaseException as exc:  # noqa: BLE001 -- an observer never raises
            # #1328: THE TYPE ALONE IS A HINT, NOT A FINDING. Boot weg2xsn6
            # printed `manifest=manifest-failed:AttributeError` on 24 of 24
            # legs, in both groups, and that string is all the evidence the
            # boot left: no message, no attribute name, no site. The whole
            # campaign then had to guess which attribute -- and three
            # plausible candidates (`region.boot_hash`, `agreed.theirs`,
            # `derive_card_manifest`'s clean returns) were each refuted by
            # reading or by local reproduction, costing a desk pass.
            #
            # An observer still never raises. It now REPORTS: the message and
            # the last frame of its own traceback, so the next boot names the
            # attribute in one line instead of licensing another guess.
            return None, f"manifest-failed:{_weg2_exc_note(exc)}"

    def _weg2_shadow_plan(self, hook: str, group: str, rank: int, *,
                          agreed=None, require_agreement: bool):
        """THE PRODUCT CALL SITE OF ``weight_exchange.build_plan``.

        ``require_agreement`` IS THE CALLER'S DECISION AND HAS NO DEFAULT
        (#1345).  It was hard-set ``True`` here for every caller, and boot
        weg2xsn19 measured what that cost: the two callers of this method have
        OPPOSITE and both-documented needs, so the injection caller could only
        ever read ``manifest-unagreed`` -- 12 of 12 legs on both groups, on
        every argv, with the refusal reached, named and static.

        * the SHADOW hook compares stripes ACROSS the co-located pair, so it
          MUST narrow to the agreed set (#1311 S6b; boot weg2xsn5's W80 refused
          7 of 8 legs because each end hashed its own inventory);
        * the INJECTION lane has no peer question at all.  Its bytes come from
          path (b), its acceptance is ``xchg_bounce == expression`` rather than
          ``== peer``, and its reference is the LOCAL landed weights
          (``ptr_of`` resolves pointers only for this rank's own group and rank
          and returns ``None`` on the peer side by ``XchgDesc``'s contract).
          Narrowing it also made the grader vacuous: the agreed set measured
          4.90 MiB of a 27.52 GiB image, 0.018 %.

        NO DEFAULT is deliberate: a default is what let one caller inherit the
        other's requirement silently.  ``derive_leg_plan`` itself defaults to
        ``require_agreement=False``, so forwarding ``False`` here is the
        library's sanctioned path and not a relaxed gate.

        SECTION 1ai-S5b-fix's UNPROVEN 2 in its own words -- *"``build_plan``
        has no product caller ... unchanged, and still the largest gap"* -- is
        closed here.  The derivation itself lives in
        ``weight_exchange_shadow.derive_leg_plan``, deliberately, and this
        method is four lines of argument gathering: the mixin cannot be
        constructed without a model runner, a process group and a device, so
        anything written INTO it is code no hermetic test can drive (the same
        reason the two hooks are free functions over plain arguments).

        THE MODEL IS THIS RANK'S OWN LIVE ONE and the region tag is the one its
        weights were actually opened with (``weights_region_tag_for``), so a
        runner whose weights are OUT of the exchanged family (the drafter under
        an armed exchange) derives no family tags and refuses by name rather
        than planning bytes nobody exchanges.

        ``(None, reason)`` on every failure, including an exception: a
        derivation that raised into a flip leg would be the observer taking the
        authority this slice exists not to have.
        """
        try:
            from sglang.srt.weg2 import weight_exchange as wx
            from sglang.srt.weg2 import weight_exchange_shadow as sh

            runner = getattr(self.tp_worker, "model_runner", None)
            model = getattr(runner, "model", None)
            region_tag = ""
            if runner is not None:
                try:
                    region_tag = wx.weights_region_tag_for(
                        wx.RunnerShape.of(runner))
                except BaseException:  # noqa: BLE001 -- an unclassified shape
                    region_tag = ""
            plan, reason = sh.derive_leg_plan(
                hook=str(hook), group=str(group),
                peer=("D" if group == "P" else "P"), rank=int(rank),
                model=model, region_tag=region_tag,
                # #1311 S6b.  ``require_agreement`` is the half that must not be
                # forgotten: without it a leg whose peer has not published would
                # silently fall back to this rank's own inventory, which is
                # exactly the rank-local derivation boot weg2xsn5 refused 7 of 8
                # legs on.  The product refuses by name instead.
                agreed=agreed, require_agreement=bool(require_agreement))
            # THE ACCEPTANCE LINE IS NOT EMITTED HERE, and the reason is
            # recorded because #1342 S3 wired it here and boot weg2xsn18 paid
            # for it: this method only ever holds `derive_leg_plan`'s
            # **LegPlan**, while `WEG2-XCHG-PLAN dir=.../plan_id=...` is an
            # **XchgPlan** line.  The emit raised
            # `AttributeError: 'LegPlan' object has no attribute 'log_line'`
            # 21x on D and 18x on P, and census (d) read `plan_id` 0.
            #
            # It now lives one frame earlier, at the plan-construction call
            # inside `weight_exchange_shadow.derive_leg_plan`, which is the only
            # production frame that holds an XchgPlan.  (The constructor is
            # deliberately NOT named literally here: a single test asserts there
            # is exactly ONE call site for it across the package, and a mention
            # in prose counts -- which this comment discovered by tripping it.)
            # `test_the_wrong_site_no_longer_emits` keeps this site quiet.
            return plan, reason
        except BaseException as exc:  # noqa: BLE001 -- an observer never raises
            # #1328, same reason as the manifest arm one method up: a swallowed
            # exception that reports only its type cannot be acted on.
            return None, f"derivation-failed:{_weg2_exc_note(exc)}"

    def _weg2_shadow_hook(self, hook: str, *, recv_req, reserve_bytes: int = 0,
                          ring_ms=None) -> None:
        """Run one leg's shadow, or return having touched nothing.

        THE ARM IS CHECKED FIRST AND CHEAPLY.  ``bounce_lane_armed()`` reads
        ``--weg2-weight-source``; on ``ring`` -- the default and every boot that
        has run to date -- it is False and this method returns after ONE module
        import, before a device call, an allocation, an env write or a single
        line of the exchange's machinery.  (It read ``shadow_armed()`` until
        #1273 B4q; that is one READING of a three-valued flag and it was not
        the one the S6I order arms, so the hook was dead on the `exchange`
        arm by construction -- boot weg2xsn16.)  That is what keeps the default leg
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

            # #1273 B4q: THE AXIS, not the S5 arm.  `shadow_armed()` reads
            # `weight_source() == "shadow"`, and the S6I order launches
            # `--weg2-weight-source exchange` with
            # `SGLANG_WEG2_XCHG_INJECT=shadow` -- mutually exclusive readings
            # of a three-valued flag, so this `return` fired on every leg of
            # boot weg2xsn16 while the flip path itself ran (48
            # `WEG2-GROUP-FENCE` on D, 0 `WEG2-XCHG-INJECT`).  `ring` still
            # returns here, which is what keeps the default leg byte-identical.
            if not sh.bounce_lane_armed():
                return
            from sglang.srt.weg2 import weight_exchange_region as xr

            group = self._weg2_group_name()
            rank = self._weg2_rank()
            if group not in ("P", "D") or rank < 0:
                # THE DISCRIMINATOR BOOT weg2shadowC DID NOT HAVE.  Both
                # pre-flight gates used to return without a word, so "the arm
                # never reached this rank" and "this rank has no identity"
                # produced byte-identical evidence -- nothing -- and the boot
                # could not separate the two readings that mattered.  The ARM
                # gate above stays silent on purpose: it fires on every ring
                # boot, i.e. every boot that has ever run.  THIS one fires only
                # under an armed shadow, where a silent return is the defect;
                # it is the same W79 line every other rank-local refusal on
                # this path already uses, so no reader learns a new shape.
                logger.info(sh.rank_local_skip_message(
                    reason="no-identity", rank=rank,
                    leg=_weg2_flip_index_of(getattr(recv_req, "epoch", None)),
                    epoch=str(getattr(recv_req, "epoch", "") or ""),
                    detail=f"hook={hook} group={group} rank={rank} -- the "
                           f"rank's own index inside its Weg-2 group is what "
                           f"xr.rank_row keys the six gate rows by"))
                return
            device = self._weg2_device_index()
            if device < 0:
                logger.info(sh.rank_local_skip_message(
                    reason="no-device", rank=rank,
                    leg=_weg2_flip_index_of(getattr(recv_req, "epoch", None)),
                    epoch=str(getattr(recv_req, "epoch", "") or ""),
                    detail="torch reports no CUDA device on this rank"))
                return
            # ------------------------------------------------------------
            # #1273 S6 fix D -- THE IDENTITY SWEEP, not one more instance.
            #
            # Boot weg2shadowD found the SAME SHAPE one level up: the flip
            # index degraded to -1, was COMPOSED INTO A STAMP
            # (``weight_exchange_shadow.py:2753``,
            # ``f"{boot_nonce}.{int(i.leg)}"``) and handed to
            # ``XchgRegion.begin_flip``, which refused it as **W68
            # Weg2XchgPlanDisagree** -- a name that says "the plan disagrees"
            # about a leg whose actual condition is "this is not a flip".
            #
            # The three legs that produced it were group P's BOOT-TIME initial
            # sleep at its own READY (14:33:55Z), 73 s BEFORE the first flip
            # began (14:35:08Z).  That leg carries no epoch because no flip
            # exists yet -- the W79 printed ``epoch=`` EMPTY, not malformed --
            # so the -1 was honest and only its downstream use was not.
            #
            # UPSTREAM-MINIMAL, and it is why no counter is added here: the
            # FRONT owns the flip counter and publishes it on the request
            # (``front.py:2667`` composes ``credit_epoch(boot, self.epoch)``
            # and sends it on BOTH gathered legs, ``:2676-2680``); every real
            # flip leg of shadowD carried ``epoch=1788964408.0`` / ``.1``
            # correctly.  ``credit_epoch``'s own docstring records what a
            # rank-local counter costs -- a cross-boot collision that fed C14
            # a previous boot's credit.  So the authority is the request, and
            # the right behaviour when it is absent is to REFUSE BY NAME, never
            # to invent an index and never to let the sentinel travel.
            #
            # THE CLASS, swept here in one place: every identity this hook
            # reads is resolved BEFORE the inputs are built, and any one that
            # is missing becomes a NAMED W79 refusal instead of a sentinel that
            # downstream code has to recognise.  The sentinels that used to
            # travel were: leg=-1 / epoch="" (W68, above), card_uuid="unknown"
            # (an unnamed card priced and charged as if it were a real one),
            # and free_mib=0 (an UNREADABLE NVML free column priced as a FULL
            # card -- ``price_shadow`` would then refuse UNAFFORDABLE giving
            # the wrong reason, which is worse than not pricing).
            # ------------------------------------------------------------
            leg = _weg2_flip_index_of(getattr(recv_req, "epoch", None))
            epoch_token = str(getattr(recv_req, "epoch", "") or "")
            if leg < 0:
                logger.info(sh.rank_local_skip_message(
                    reason="no-flip-epoch", rank=rank, leg=leg,
                    epoch=epoch_token,
                    detail=f"hook={hook} -- this leg carries no flip epoch, so "
                           f"it is not a flip: the front publishes the index on "
                           f"the request (front.py:2667) and the boot-time "
                           f"initial sleep runs before any flip exists.  A leg "
                           f"with no flip identity may not stamp a region"))
                return
            card_uuid = self._weg2_card_uuid()
            if not card_uuid:
                logger.info(sh.rank_local_skip_message(
                    reason="no-card", rank=rank, leg=leg, epoch=epoch_token,
                    detail=f"hook={hook} -- NVML could not name this rank's "
                           f"card, and every shadow term (the price, the "
                           f"deposit charge, the W77 line) is keyed by it"))
                return
            peer = "D" if group == "P" else "P"
            free_bytes = self._weg2_free_bytes()
            if free_bytes is None:
                logger.info(sh.rank_local_skip_message(
                    reason="no-free-column", rank=rank, leg=leg,
                    epoch=epoch_token,
                    detail=f"hook={hook} card={card_uuid} -- the LIVE NVML free "
                           f"column is what price_shadow grades against and it "
                           f"could not be read; pricing against 0 would refuse "
                           f"UNAFFORDABLE naming a full card that is not full"))
                return
            self._weg2_shadow_param_census(group, rank)
            # #1311 S6b -- THE CARD MANIFEST, BEFORE THE PLAN AND NOT AFTER.
            # The plan is narrowed to the pair's agreed piece set, and
            # ``coalesce`` merges descriptors across parameter names, so the
            # narrowing has to happen on the INVENTORY inside the derivation.
            # That is why the agreement is reconciled here, one call earlier,
            # rather than inside ``run_leg_hook`` where the region is opened
            # for the transport.
            agreed, manifest_state = self._weg2_shadow_manifest(
                group, peer, int(rank), leg=leg, epoch=epoch_token)
            # #1345: THE SHADOW LANE NARROWS, and says so at the call rather
            # than relying on the adapter.  This is the danger direction of
            # that slice: a shadow leg planned over a set the peer never agreed
            # to is boot weg2xsn5's W80 exactly.
            plan, plan_reason = self._weg2_shadow_plan(str(hook), group,
                                                       int(rank),
                                                       agreed=agreed,
                                                       require_agreement=True)
            # THE MANIFEST STATE RIDES THE PLAN REASON, so the one line a
            # refused leg prints (W79 ``no-plan``) says WHY the pair could not
            # agree and not merely that it did not.
            if plan_reason:
                plan_reason = f"{plan_reason} manifest={manifest_state}"
            inputs = sh.ShadowLegInputs(
                leg=leg,
                epoch=epoch_token,
                direction="d2h" if hook == sh.HOOK_SOURCE else "h2d",
                hook=str(hook),
                rank=int(rank),
                row=xr.rank_row(group, int(rank)),
                peer_row=xr.rank_row(peer, int(rank)),
                device=int(device),
                card_uuid=card_uuid,
                free_mib=int(free_bytes // MIB_),
                resume_reserve_bytes=int(reserve_bytes),
                ring_ms=ring_ms,
                gate_rows=self._weg2_shadow_gate_rows(str(hook), group,
                                                      leg=int(leg)),
                plan_reason=str(plan_reason),
                # THE ON-CARD LANE HAS NO CONCURRENT PEER ON THIS PLACEMENT.
                # Same structural fact as ``_weg2_shadow_gate_rows`` above, one
                # consequence further on (S5c refuter, must_fix 3): the source
                # hook is UPSTREAM of the pause loop whose ``credit.publish``
                # the co-located waking rank's ``resume`` is fenced on (C14),
                # and the destination hook runs after that resume -- so while
                # either hook holds this scheduler thread, the other end of the
                # bounce cannot be running.  It is FALSE HERE and nowhere else
                # -- S6's RPC handler drives both ends inside one call and
                # passes the default.
                #
                # S6 CHANGED WHAT THIS FLAG BUYS, not where it is set.  It used
                # to mean "refuse the lane"; it now means "this lane must be
                # STORE-AND-FORWARD", which is the shape that needs no
                # concurrent peer: the source fills one slot per batch, seals
                # the rows and returns inside its own leg, and the destination
                # hook -- after the C14-fenced resume, in its own later leg --
                # opens the same ``/dev/shm`` file and reads them.  The drain
                # that could not be drained is gone by arithmetic (every
                # ``seq - slots`` is negative) rather than by a branch, so
                # ``blocked_ms`` stays 0.000 and the destination now has bytes
                # to COMPARE.  A shape that does not fit is still refused by
                # name, and the refusal is now W81 (the deposit) instead of the
                # blameless placement line.
                oncard_drainable=False,
                # #1311 S6b -- WHY THE ON-CARD LANE STILL DOES NOT RUN, and
                # it is NOT this flag.  ``run_leg_hook``'s ShadowLegInputs path
                # already derives ``store_forward = not oncard_drainable``
                # (``weight_exchange_shadow.py:3855``), so setting
                # ``oncard_drainable=False`` here IS asking for the deposit.
                # Boot weg2xsn5's 24 ``WEG2-XCHG-SHADOW-ONCARD-REFUSED`` lines
                # came through the OTHER half of that refusal's condition:
                # ``deposit_refusal_reason`` returns ``DEPOSIT_REASON_IPC`` for
                # every mode that is not ``host``
                # (``weight_exchange_transport.py:1607``) and the boot ran
                # ``--weg2-xchg-oncard ipc``.  An exported VRAM bounce is freed
                # with the leg that exported it, so store-and-forward is a
                # ``host``-arm shape by construction and no code change here can
                # give the ipc arm one.  What #1311 changed is that the refusal
                # LINE now names the arm and the action instead of blaming the
                # placement the deposit already defeats -- and the ledger term
                # below is only read on the host arm
                # (``_weg2_shadow_host_budget``), which is the other half a
                # host-arm boot needs.
                # WHAT THE #1269 LEDGER CHARGED FOR ONE CARD'S DEPOSIT.  The
                # adapter is the producer because the budget is a property of
                # the BOOT's arm and a rank hook cannot read the launcher's
                # ladder; the value comes from the ledger's own function, so
                # the charge and the bound are one number and cannot drift.
                # An unreadable ledger yields 0, which REFUSES -- pinned host
                # bytes nothing charged for are exactly what
                # host-schwelle-nie-uebertreten forbids.
                host_bounce_budget_bytes=self._weg2_shadow_host_budget(),
            )
            try:
                sh.run_leg_hook(inputs, log=logger.info, plan=plan)
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

    def _weg2_shadow_host_budget(self) -> int:
        """One card's charged deposit budget, in bytes, or 0 (#1273 S6).

        ONE READER OF ONE NUMBER.  ``xchg_bounce.staging_bytes_per_card``
        is the same function the launcher's ARM line charges with, so the
        budget a rank enforces and the term the ledger carries are the same
        arithmetic rather than two copies of it.  Zero on any failure, and zero
        REFUSES the deposit downstream: an observer that could not read its own
        budget must not pin host memory on a guess.

        IT ASKS THE ARM FIRST, AND THAT IS THE HALF THAT WAS MISSING (S6
        refuter, finding 6 + must_fix 4).  The launcher charges the deposit
        ONLY on the ``host`` on-card arm; a rank that returned the full budget
        on every arm would authorise a deposit against a term the ledger did
        not carry -- host bytes above the reap mark by exactly the amount
        nobody paid for, which ``host-schwelle-nie-uebertreten`` forbids.  The
        arm is read from the value the launcher PUBLISHED
        (``resolve_shadow_oncard_mode``), i.e. the same string that decided the
        charge, so the two cannot disagree.  With that, the ``except`` arm
        below is no longer the only path to 0: ``ipc`` reaches it by design.
        """
        try:
            from sglang.srt.weg2 import weight_exchange_shadow as wxs
            from sglang.srt.weg2 import weight_exchange_transport as tp
            from sglang.srt.weg2 import xchg_bounce as xb

            if wxs.resolve_shadow_oncard_mode() != tp.ONCARD_MODE_HOST:
                return 0
            # AMENDMENT 5: the PUBLISHED SLOT, not the retired ceiling.  This
            # read `host_ledger.xchg_bounce_bytes_per_card()` -- the S6b
            # deposit's `ONCARD_SLOTS_MAX 8 x ONCARD_SLOT_BYTES_MAX` -- while
            # the ledger charged `SLOTS_PER_PAIR 2 x slot_bytes`, so the budget
            # a rank enforced and the term that was paid for disagreed by 4x on
            # ONE payload.  Both functions are now deleted and the arithmetic
            # has a single owner (`xchg_bounce.staging_bytes_per_card`).
            #
            # `tp.ONCARD_SLOT_BYTES_MAX` is the LAUNCHER'S PUBLISHED VALUE
            # despite its name: it resolves `ENV_ONCARD_SLOT_MIB` at import
            # (`_resolve_oncard_slot_bytes_max`), which is exactly the
            # `--weg2-xchg-oncard-slot-mib` the launcher parsed and published,
            # and it is the same number the ARM line printed as `slot_mib=`.
            # Under S6-BOUNCE it is not a ceiling but THE slot, and the deposit
            # is never sized up: a plan needing more batches than this term
            # funds is refused by name as `DEPOSIT_REASON_UNFUNDED` -- graded
            # against THIS number, which is the third and last copy of the
            # sentence #1333 corrected.  `DEPOSIT_REASON_BATCHES` is a
            # different lever (`ONCARD_SLOTS_MAX`, the transport's row area)
            # and is not what binds a 3-batch plan here.
            return int(xb.staging_bytes_per_card(tp.ONCARD_SLOT_BYTES_MAX))
        except Exception:  # noqa: BLE001 -- an observer never raises
            return 0

    def _weg2_restore_device(self, device: int) -> None:
        """Put the calling thread's CUDA device back where the hook found it."""
        try:
            import torch

            if torch.cuda.is_available() and int(device) >= 0:
                torch.cuda.set_device(int(device))
        except Exception:  # noqa: BLE001 -- an observer's unwind
            pass

    # =======================================================================
    # #1350 SEAM GRADER -- "did the exchange bring MY bytes back?"
    #
    # Rank-local, awake-only, default OFF.  The two hooks below are the ONLY
    # call sites in the tree, and that is asserted rather than intended
    # (`test_the_only_call_sites_are_the_two_awake_seams`).  There is no third
    # one at the cutover, and there cannot be: at flip time the sleeping group
    # holds NO weight bytes in VRAM (`understand_tensor-map.md` 5.4 with
    # WEG2_SLEEP_TAGS), so a reading there would hash unmapped pages.
    # =======================================================================

    def _weg2_seam_inventory(self):
        """This card's pieces as ``([(ParamGeom, tensor), ...], reason)``.

        Delegated to ``weight_exchange_shadow.card_inventory``, which is the
        tree's ONE producer of a card's placement.  The grader deliberately
        does not walk ``named_parameters()`` itself: placement is decided by
        the loader at boot and already published, and a second inventory here
        would be second bookkeeping beside it (operator direction 2026-09-12).
        """
        try:
            from sglang.srt.weg2 import weight_exchange_shadow as shadow

            runner = getattr(self.tp_worker, "model_runner", None)
            model = getattr(runner, "model", None)
            return shadow.card_inventory(rank=self._weg2_rank(), model=model)
        except BaseException as exc:  # noqa: BLE001 -- an observer's unwind
            return None, f"inventory-failed:{type(exc).__name__}:{exc}"

    def _weg2_seam_digest_before(self, recv_req, weights_tags) -> None:
        """THE PRE-PAUSE READING.  At the last instant the pages are mapped.

        Called only at the FIRST weights RPC of a sleep, for the same reason
        ``_export_static_state`` is: once any family tag is paused, reading a
        parameter is a read of unmapped pages -- the campaign (a) fault on the
        weights tag.

        UNARMED CLEARS RATHER THAN INHERITS.  A boot that turns the grader off
        between two flips must not leave a stale reading behind for the next
        wake to grade against; that would be a verdict about two different
        boots wearing this one's name.
        """
        if not seam_digest.seam_digest_armed():
            self.weg2_seam_before = None
            return
        # The rank's OWN reading of the knob, once per process.  What the
        # launcher intended and what this process resolved are two facts, and
        # only the second one decides what the flip does.
        seam_digest.announce_once(logger.info)
        inventory, reason = self._weg2_seam_inventory()
        if inventory is None:
            self.weg2_seam_before = None
            logger.info(
                "%s",
                seam_digest.unarmed_verdict(
                    f"no-inventory:{reason}",
                    group=self._weg2_group_name(),
                    rank=self._weg2_rank(),
                    card=self._weg2_device_index(),
                    tags=weights_tags,
                ).line(),
            )
            return
        reading = seam_digest.take_reading(
            "before",
            inventory,
            group=self._weg2_group_name(),
            rank=self._weg2_rank(),
            card=self._weg2_device_index(),
            tags=weights_tags,
            epoch=getattr(recv_req, "epoch", None),
        )
        self.weg2_seam_before = reading
        logger.info("%s", reading.line())

    def _weg2_seam_digest_after(self, recv_req, weights_tags) -> None:
        """THE POST-LANDING READING AND THE VERDICT.

        Placement rule, shared with the shadow's destination hook and with the
        step-6c compare: after ``family_complete``, after the reload, after the
        shadow compare.  A reading before the bytes settle grades undefined
        content and its MISMATCH would mean nothing.

        READ-AND-CLEAR on every path, including the raising one: a refusal that
        left the pre-pause reading standing would grade the NEXT wake against a
        landing that already failed.

        THE MISMATCH RAISES.  A grader whose verdict nobody consumes is the
        class this campaign has paid for repeatedly (counter-vs-actuator), and
        the bytes this rank is about to serve on are not the bytes it had.  The
        line reaches the log FIRST, so the evidence survives the raise.
        """
        if not seam_digest.seam_digest_armed():
            self.weg2_seam_before = None
            return
        before = self.weg2_seam_before
        self.weg2_seam_before = None
        group = self._weg2_group_name()
        rank = self._weg2_rank()
        card = self._weg2_device_index()
        inventory, reason = self._weg2_seam_inventory()
        if inventory is None:
            logger.info(
                "%s",
                seam_digest.unarmed_verdict(
                    f"no-inventory:{reason}", group=group, rank=rank, card=card,
                    tags=weights_tags,
                ).line(),
            )
            return
        after = seam_digest.take_reading(
            "after",
            inventory,
            group=group,
            rank=rank,
            card=card,
            tags=weights_tags,
            epoch=getattr(recv_req, "epoch", None),
        )
        logger.info("%s", after.line())
        verdict = seam_digest.compare(before, after)
        logger.info("%s", verdict.line())
        refusal = verdict.refusal()
        if refusal is not None:
            raise refusal

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

    # -- #1273 S6-BOUNCE: THE AUTHORITATIVE PATHS ------------------------
    #
    # These two are the product call sites of the exchange, and they differ
    # from the shadow's two hooks above in the one way that matters: the
    # shadow OBSERVES beside a ring that owns the bytes, and these OWN them.
    # Under ``--weg2-weight-source exchange`` the bytes come from the peer
    # group's live VRAM, through a bounded host buffer, instead of from
    # ``update_weights_from_disk`` (:meth:`_weg2_wake_reload_weights`, measured
    # 12.073/14.143/16.749 s per wake on this rig).
    #
    # CORRECTED #1342, same stale claim as the one removed from
    # :meth:`_weg2_xchg_inject_weights`: this comment used to assert that the
    # arm opens the weights region ``enable_cpu_backup=False``.  It does not --
    # ``model_runner.py:2440-2442`` computes that from ``server_args`` with no
    # arm predicate.  The exchange is the authority here because
    # ``--weg2-xchg-inject authoritative`` SAYS SO, not because nothing else
    # could have carried the bytes.
    #
    # THEY RAISE.  An observer that took a flip down over its own bookkeeping
    # would be wrong, and that is why every shadow method above swallows; an
    # AUTHORITY that swallowed would serve undefined weights, which is worse
    # than a refusal by exactly the margin this whole campaign is about.  The
    # refusals are W68 (a run no slot can hold), W71 (a unit the buffer cannot
    # assemble whole) and W74 (a slice no source covers) -- all three from
    # modules that already own them, and no new W-code.

    def _weg2_xchg_device_ops(self):
        """This rank's CUDA device-ops layer, or ``None`` if it cannot open.

        A SEAM, and that is its whole purpose: the transport's ops object is
        the one input of the authoritative inject that cannot be constructed
        without a GPU, so it lives behind a one-line method the hermetic tests
        substitute.  Everything else the inject needs (the nonce, the identity,
        the plan) is plain data and is driven for real in the tests.

        ``None`` rather than an exception, because the caller turns it into the
        same NAMED W4 refusal as every other missing input -- the shadow's
        ``attach`` reports ``"no-ops"`` for the identical condition and this
        mirrors it instead of inventing a second shape.
        """
        try:
            from sglang.srt.weg2 import weight_exchange_transport as tp

            return tp.CudartDeviceOps()
        except BaseException:  # noqa: BLE001 -- no ops is not an error to raise
            return None

    def _weg2_xchg_bounce_leg(self, *, descs, ops, boot_nonce,
                              slot_bytes=None, depth=None, terms=None,
                              mode=None, shm_root=None, device: int = 0):
        """PATH (b): assemble each unit in the host bounce, every card slices.

        The whole of the user law's fallback sentence, at the one call site
        that has both the plan and the device: *"notfalls wird das layer auf
        einem (vertretbar kleinen) hostpuffer vollstaendig zusammengesetzt und
        jede karte nimmt sich von dem was er braucht (oder ihn komplett)"*.

        A THIN METHOD ON PURPOSE, and the thinness is the point rather than an
        omission: #1329 was three boots spent on a shadow whose first write
        raised on a slots dataclass, and every test of that slice drove the
        module functions while the mixin's own methods -- the only callers the
        product has -- had no executing test at all.  So this method exists to
        BE the call site the smoke drives, and it holds no state (this is a
        ``slots=True`` dataclass, for the fifth time in this file).

        ``terms`` is the ARM's priced decision (``xchg_bounce.BounceTerms``)
        and is the normal way in; the explicit ``slot_bytes``/``depth`` pair is
        for tests and tools. The geometry is never derived here -- section
        10.8's "one reader, or they drift".
        """
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import weight_exchange_bounce as bx
        from sglang.srt.weg2 import weight_exchange_region as xr

        # THE MODE IS READ ONCE, from the one reader, and passed down.  A leg
        # that re-read it further in could act on a different answer than the
        # one it was entered with, and the two differ by "does this write into
        # the live weights".
        return bx.run_bounce_leg(
            descs, ops, boot_nonce,
            slot_bytes=slot_bytes, depth=depth, terms=terms,
            mode=wx.inject_mode() if mode is None else mode,
            shm_root=xr.SHM_ROOT if shm_root is None else shm_root,
            device=device, log=logger.info,
        )

    # PATH (a) WAS DELETED HERE (#1342 S3), and the deletion is recorded
    # rather than silent so a re-introduction has to argue with it.
    #
    # `_weg2_xchg_agreed_leg` moved the byte-identical card-to-card set to live
    # storage.  It had ZERO production callers -- only the two docstring
    # mentions in the section header above -- and it was a SECOND MOVER for a
    # payload path (b) already carries: measured on boot weg2xsn8 the agreed
    # set is 4.90 MiB of a 27.52 GiB image, i.e. 0.018 %, so keeping it is not
    # a throughput argument.  Its input was unavailable at the one call site
    # that exists, too: the agreement verdict comes from
    # `reconcile_card_manifest`, a per-flip-leg artifact, and the wake refill
    # has no leg identity.  UPSTREAM-MINIMAL: a second accounting of one
    # payload is a delete candidate, and the repair carries the burden of
    # proof.  `weight_exchange_bounce.run_agreed_leg`, `agreed_descs` and
    # `AgreedResult` went with it; `test_the_deleted_second_mover_is_really_gone`
    # pins the absence by name.

    def _weg2_corridor_floor_bytes(self) -> Optional[int]:
        """This card's corridor lower bound in bytes, or None (#1331).

        THE CORRIDOR AUTHORITY IS THE CORRIDOR AUTHORITY. The floor is read
        by calling `managers.corridor_guard.corridor_floor_mib` -- the same
        function `weg2/corridor_budget` calls to build its per-card map -- never
        re-derived here and never replaced by a constant: the VRAM-korridor law
        says the per-card lower bound is the MEASURED transient peak plus the
        user reserve knob, and a second reading of that would be exactly the
        1024-MiB constant the law deleted.

        `None` when the map is not available, which the caller turns into a
        floor of 0 -- i.e. the refusal then grades free against the request
        alone. That is the conservative direction for THIS gate: a missing
        floor may not manufacture a refusal, only fail to tighten one.
        """
        uuid_key = self._weg2_card_uuid()
        if uuid_key is None:
            return None
        try:
            from sglang.srt.managers.corridor_guard import corridor_floor_mib

            mib = corridor_floor_mib(uuid_key, group=self._weg2_group_name())
            return None if mib is None else int(mib) * 1024 * 1024
        except Exception:  # noqa: BLE001 -- an absent authority is an absence
            return None

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
                # #1331: the SAME reader, handed in so the grant can re-read
                # the card at the moment it grants rather than trusting the
                # snapshot taken before the wait. `free_bytes_now` above is
                # still the early-exit term; this is the grant-time one, and
                # they are one reader (`registry.nvml`, the #1250 v2 free
                # column) so a second NVML loop cannot drift from the first.
                free_reader=self._weg2_free_bytes,
                floor_bytes=self._weg2_corridor_floor_bytes(),
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
                "requested=%d MiB free_mib=%s allocatable_est=%s "
                "corridor_floor_mib=%s (%s)",
                self._weg2_card_uuid() or "unknown", tag,
                float(rec["waited_s"]) * 1000,
                int(rec.get("credit_bytes", 0)) // MIB_,
                int(need_bytes) // MIB_,
                # #1331: PRINTED, so the next boot can ATTRIBUTE instead of
                # infer. weg2xsn7's line carried neither number, which is why
                # "allocatable < credited" had to be reconstructed from a
                # `cu_mem_create` line one row below it. `n/a` and never 0 --
                # an unreadable free column is an absence.
                "n/a" if rec.get("free_bytes") is None
                else int(rec["free_bytes"]) // MIB_,
                "n/a" if rec.get("allocatable_est_bytes") is None
                else int(rec["allocatable_est_bytes"]) // MIB_,
                int(rec.get("corridor_floor_bytes", 0) or 0) // MIB_,
                rec.get("reason", ""),
            )

    # ---- #1285 epoch-scoped leg dedup (module docstring above) -------------
    def _weg2_leg_key(self, op: str, recv_req) -> Optional[Tuple]:
        """The dedup key of THIS leg, or None when the request carries no epoch.

        None is the stock path: no epoch, no dedup, byte-identical behaviour.
        """
        epoch = getattr(recv_req, "epoch", None)
        if epoch is None:
            return None
        return Weg2LegLedger.key(op, epoch, getattr(recv_req, "tags", None))

    def _weg2_leg_ledger_obj(self) -> Weg2LegLedger:
        led = self.weg2_leg_ledger
        if led is None:
            led = Weg2LegLedger()
            self.weg2_leg_ledger = led
        return led

    def _weg2_leg_replay(self, op: str, recv_req):
        """The recorded answer of an already-applied leg, or None."""
        key = self._weg2_leg_key(op, recv_req)
        if key is None:
            return None
        out = self._weg2_leg_ledger_obj().recorded(key)
        if out is None:
            return None
        logger.info(
            "WEG2-LEG REPLAY op=%s epoch=%s tags=%s -- this leg is already "
            "applied on this rank; returning the recorded answer and touching "
            "no VRAM (#1285: neither leg is idempotent, so a front retry after "
            "a ServerDisconnectedError must not re-apply it)",
            op, getattr(recv_req, "epoch", None), sorted(getattr(recv_req, "tags", None) or []),
        )
        return out

    def _weg2_leg_commit(self, op: str, recv_req, out):
        key = self._weg2_leg_key(op, recv_req)
        if key is not None:
            self._weg2_leg_ledger_obj().record(key, out)
        return out

    @_weg2_group_stop_on_leg_failure
    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        # #1285 FIRST STATEMENT, before the idle assert and before any mutation:
        # a leg that is already applied returns its recorded answer.  It has to
        # precede the assert too -- a repeat arrives with the group in whatever
        # state the first attempt left it, and refusing there would turn a safe
        # no-op into a group death.
        replay = self._weg2_leg_replay("release", recv_req)
        if replay is not None:
            return replay
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
            # #1350 SEAM GRADER, source side.  HERE for two reasons, both of
            # which are the same ones the shadow hook above is placed by:
            #
            # BEFORE THE PAUSE, because the reading walks this rank's live
            # parameters and they live inside region(GPU_MEMORY_TYPE_WEIGHTS)
            # -- after `pause(tag)` they are unmapped pages.
            #
            # INSIDE THE TIMED BAND (`weg2_leg_t0` is above it), because the
            # hash is not free and a leg whose instrument hid its own observer
            # is exactly the defect S5b refuter must_fix 4 recorded.  On an
            # unarmed boot -- every serving boot -- the call returns after one
            # predicate and the band is unchanged.
            #
            # Only at the FIRST weights RPC of a sleep: a later chunk RPC
            # arrives with part of the family already paused, and reading then
            # would be the campaign (a) fault.
            if not family_paused_before:
                self._weg2_seam_digest_before(recv_req, weights_tags)
            # #1284: the NEED series, and the refusal that reads it.  The pause
            # below is what enters ``host_ring.cpp``'s blocking acquire, and on
            # weg2sb5e that acquire sat out its whole 110 s budget in silence
            # and then died blaming L6.  The guard runs IMMEDIATELY BEFORE the
            # pause so that (a) every tag's need/free/delta reaches the log
            # whether or not anything goes wrong -- that series is what answers
            # "creep or launch arithmetic" without an archaeologist -- and (b) a
            # leg whose peer is not releasing is refused in ~2 s with the series
            # attached, before any waiter is parked and while this tag's device
            # bytes are still mapped.  See ring_guard.RingNeedGuard.
            weg2_ring_guard = RingNeedGuard(
                self._weg2_card_uuid() or "unknown",
                group=self._weg2_group_name(),
                rank=self._weg2_rank(),
            )
            with self._weg2_pcie_lock("sleep-D2H " + ",".join(weights_tags), direction="d2h"):
                for tag in weights_tags:
                    weg2_ring_guard.guard_tag(
                        tag,
                        int(tag_bytes.get(tag, 0)),
                        self._weg2_ring_stats,
                        peer_hint="the group waking on this card",
                    )
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
        return self._weg2_leg_commit("release", recv_req, ReleaseMemoryOccupationReqOutput(
            per_tag=(report.get("per_tag") or weg2_per_tag or None)
            if weg2_memory_saver_on
            else None,
            critical_path=(report.get("critical_path") or None)
            if weg2_memory_saver_on
            else None,
        ))

    @_weg2_group_stop_on_leg_failure
    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        # #1285: see the release leg.  This one is the sharper case -- the wake's
        # very first mutation below drops each tag from the offload set, which
        # raises KeyError on a repeat, so without this the retry kills the group.
        replay = self._weg2_leg_replay("resume", recv_req)
        if replay is not None:
            return replay
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
                # STEP 6c: THE SHADOW COMPARE (#1342 S1b -- moved here out of
                # `_weg2_wake_reload_weights`, where it was parked at the end of
                # the DISK-REFILL branch and therefore unreachable on the
                # `tms-backup` carrier every boot on this rig selects).
                #
                # THIS IS WHERE THE BYTES HAVE LANDED, for every carrier:
                # `tms-backup` restored them in the resume above, `disk` and
                # `exchange` wrote them inside the reload call, and `stock`
                # never released them at all -- which is why the grader's own
                # gate declines that one BY NAME rather than relying on this
                # placement.  Same rule the destination leg above follows:
                # after `family_complete`, after the reload.
                #
                # AFTER the reload and AFTER the destination leg, deliberately:
                # a compare that ran earlier would grade content the reload has
                # not settled, and a MISMATCH from that is an instrument fault
                # dressed as a finding.
                # `test_the_compare_comes_after_the_reload_in_the_wake_path`
                # pins the order; mutant M11 pulls it above the reload.
                self._weg2_xchg_shadow_compare()
                # #1350 SEAM GRADER, destination side.  THE SAME PLACEMENT RULE
                # the two hooks above follow, and for the same reason: this is
                # where the bytes have landed for every carrier, so a reading
                # here grades settled content.  Last of the three deliberately
                # -- it is the only one that can RAISE, and the other two must
                # have reached the log before it does.
                self._weg2_seam_digest_after(recv_req, weights_tags)

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
                self._weg2_rescan_store_index()
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
        # #1295 MUST_FIX 2: THE STORE VERDICT RIDES THE FENCE THAT IS ALREADY
        # HERE. ``_weg2_rescan_store_index`` above can find this group's L3
        # index blind (W8b), and only ONE rank of the group holds an index to
        # rebuild -- so raising there kills the eviction owner while its
        # siblings clear dormancy and walk into the next collective, which is
        # the disagreement §0 forbids. C15's ok-bit is all-gathered over the
        # group's world cpu group and ANY False makes EVERY rank raise, so the
        # verdict is voted, not raised. Read-and-clear, so a later leg cannot
        # inherit it.
        store_failure = self.weg2_store_rescan_failure
        self.weg2_store_rescan_failure = ""
        if weg2_memory_saver_on:
            report = self._weg2_group_fence(
                "resume tags=%s" % (list(tags),),
                ok=not store_failure,
                failure=store_failure,
                per_tag=weg2_per_tag,
                leg_ms=weg2_leg_ms,
            )
        if store_failure and not report:
            # The fence did not gather: no memory saver, no cpu group, or
            # world <= 1. A single-rank engine cannot disagree with itself, so
            # here -- and only here -- the local raise IS the group-wide stop.
            raise Weg2WakeRefused(store_failure)

        return self._weg2_leg_commit("resume", recv_req, ResumeMemoryOccupationReqOutput(
            per_tag=(report.get("per_tag") or weg2_per_tag or None)
            if weg2_memory_saver_on
            else None,
            critical_path=(report.get("critical_path") or None)
            if weg2_memory_saver_on
            else None,
        ))

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
