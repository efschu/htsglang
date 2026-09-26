from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace as _dataclasses_replace
from datetime import timedelta
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
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
    weg2_tms_resume,
    WEG2_SLEEP_MIN_RELEASED_FRACTION,
    WEG2_SLEEP_TAGS,
    Weg2FlipRankDisagree,
    Weg2VramCreditRefused,
    Weg2WakeRefused,
    Weg2XchgLaneNeverDrainedRefused,
    Weg2XchgWakeSourceGapRefused,
    assert_backup_off_wake_refill_is_defined,
    assert_memory_saver_active,
    checkpoint_quantization,
    is_weights_family_tag,
    pcie_transfer_lock,
    resolve_pcie_lock_key,
    sleep_acceptance_census,
    vram_credit,
    weg2_graph_tag_armed,
)
from sglang.srt.managers.weg2_sleep_drain import (
    WEG2_SLEEP_DRAIN_BOUND_S,
    Weg2SleepDrainRefused,
    drain_until_group_verdict,
    refusal_message,
)
from sglang.srt.mem_cache.hicache_collective import collective_rank_desc
from sglang.srt.mem_cache.weg2_store_gates import Weg2StoreIndexBlind
from sglang.srt.model_executor.vram_peak_window import flip_leg as _vram_peak_leg  # H55
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

#: H11: ONE derivation per plan key at a time. x105 (D TP0, first wake): the
#: two collect workers of the first two tags both missed the per-tag key and
#: derived the same 20-GiB plan side by side (WEG2-XCHG-PLAN h2d twice,
#: first collect 0,2-0,33 s after the first resume). Re-entrant: the
#: derivation may ask for a plan itself.
_WEG2_PLAN_LOCK = threading.RLock()

#: H11: every lane key a wake collect can ride (``c<card>`` diagonal,
#: ``p<k>`` directed cross pair) -- the turn registry's universe.
_WEG2_TURN_LANES = ("c0", "c1", "c2", "p0", "p1", "p2", "p3", "p4", "p5")


def _weg2_plan_key(hook, group, rank, require_agreement, agreed_key) -> tuple:
    """The leg-cache key of a shadow plan -- the ONE producer, read by the
    cached lookup and by the boot warm-up alike (H11)."""
    return ("plan", str(hook), str(group), int(rank), bool(require_agreement),
            agreed_key)


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

# #1358 [W102-wired]: the lanes-specific refusal lives in the ledger, which is
# also where the count it disagrees with was priced.
# #1348: the unexecuted-line instrument. Imported at module scope because the
# module itself is inert and dependency-free until armed -- it imports
# `coverage` lazily inside `arm()` and only when the launcher published its
# directory, so an unarmed boot pays this import and nothing else.
# #1348: the unexecuted-line instrument. Imported at module scope because the
# module itself is inert and dependency-free until armed -- it imports
# `coverage` lazily inside `arm()` and only when the launcher published its
# directory, so an unarmed boot pays this import and nothing else.
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
from sglang.srt.weg2 import host_ledger as hl  # noqa: E402
from sglang.srt.weg2 import lane_coverage as wlc  # noqa: E402
from sglang.srt.weg2 import ring_guard  # noqa: E402
from sglang.srt.weg2.ring_guard import RingNeedGuard  # noqa: E402

#: The POPULATION token every ``WEG2-FLIP-TAG`` line carries, read back by
#: ``ring_table.parse_group_log``.  The names are imported from the reader so
#: the emitter and the parser cannot spell them differently -- a mismatch would
#: read as "weights only" and silently size the ring from a lower bound.
from sglang.srt.weg2.ring_table import (  # noqa: E402
    TAG_POPULATION_ALL as WEG2_TAG_POPULATION_ALL,
)
from sglang.srt.weg2.ring_table import (  # noqa: E402
    TAG_POPULATION_WEIGHTS as WEG2_TAG_POPULATION_WEIGHTS,
)

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



def _weg2_identity(owner, method: str, default):
    """#1358. This rank's group or index, or a default -- never an exception.

    An INSTRUMENT'S identity may not be able to take a leg down. The scheduler
    mixin has both methods; the smoke harnesses that drive the same product
    path with stubs do not, and an unguarded call there killed every leg
    (train [17b], caught by execution smoke, not by unit tests).
    """
    fn = getattr(owner, method, None)
    if fn is None:
        return default
    try:
        got = fn()
    except BaseException:  # noqa: BLE001 -- an instrument never raises
        return default
    return default if got is None else got


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


def _weg2_wake_collect_workers() -> int:
    """Collects in flight on the wake side (SGLANG_WEG2_WAKE_COLLECT_WORKERS,
    default 2, 1 = the 2026-09-15 form)."""
    import os as _os
    try:
        return max(1, min(4, int(_os.environ.get("SGLANG_WEG2_WAKE_COLLECT_WORKERS", "2") or "2")))
    except ValueError:
        return 2


def _weg2_wake_overlap_armed() -> bool:
    """SGLANG_WEG2_WAKE_OVERLAP (default 1): the wake side collects tag t on
    a single worker thread while resuming tag t+1."""
    import os as _os
    return str(_os.environ.get("SGLANG_WEG2_WAKE_OVERLAP", "1") or "1"
               ).strip().lower() in ("1", "true", "yes", "on")


def _weg2_seam_per_tag_armed() -> bool:
    """SGLANG_WEG2_SEAM_PER_TAG (default 1): the after reading is folded per
    tag on the wake worker and assembled (Punkt 3)."""
    import os as _os
    return str(_os.environ.get("SGLANG_WEG2_SEAM_PER_TAG", "1") or "1"
               ).strip().lower() in ("1", "true", "yes", "on")


def _weg2_seam_reuse_armed() -> bool:
    """SGLANG_WEG2_SEAM_REUSE (default 1): a leg's ``before`` reading is the
    rank's last graded reading instead of a fresh device walk."""
    import os as _os
    return str(_os.environ.get("SGLANG_WEG2_SEAM_REUSE", "1") or "1"
               ).strip().lower() in ("1", "true", "yes", "on")


def bx_mod_shadow_hook_armed() -> bool:
    """The region-slot shadow grader (``run_leg_hook``): off by default since
    2026-09-15, on with ``SGLANG_WEG2_XCHG_SHADOW_HOOK=1``."""
    import os as _os
    return str(_os.environ.get("SGLANG_WEG2_XCHG_SHADOW_HOOK", "0") or "0"
               ).strip().lower() in ("1", "true", "yes", "on")


def _weg2_drafter_of(scheduler):
    """The draft ModelRunner THIS PROCESS hosts, or None.

    #1378 xsn78: two producers, one accessor. Group D holds the drafter in
    the speculative worker (``scheduler.draft_worker``); group P's LAST stage
    holds it in the draft-KV producer (``scheduler.draft_kv_producer
    .draft_runner``, scheduler.py:1619) and has NO ``draft_worker`` at all --
    every site that asked ``draft_worker`` alone planned no draft leg on P,
    refused nothing, and let D's 25 deposited units rot on the lane. The
    manifest writer (model_runner.load_model -> arm_coverage_at_load) runs
    inside the draft runner itself and never needed the accessor, which is
    how P's draft manifest existed while P's plan did not.

    A MODULE FUNCTION, not a method: the smoke harnesses drive these paths
    with stubs that borrow the methods and carry none of their own (#1358).
    """
    draft_worker = getattr(scheduler, "draft_worker", None)
    if draft_worker is not None:
        try:
            drafter = _get_draft_model_runner(draft_worker)
        except BaseException:  # noqa: BLE001 -- an observer never raises
            drafter = None
        if drafter is not None:
            return drafter
    producer = getattr(scheduler, "draft_kv_producer", None)
    if producer is None:
        # xsn82 ("no-drafter" on P rank 2): the caller is the
        # SchedulerWeightUpdaterManager, a dataclass that COPIES tp_worker
        # and draft_worker off the scheduler and keeps the scheduler itself
        # under `.scheduler`; the producer was never copied.
        host = getattr(scheduler, "scheduler", None)
        producer = getattr(host, "draft_kv_producer", None)
    return getattr(producer, "draft_runner", None)


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



def _diagonal_card_of(group, device: int, phase: str) -> int:
    """Die Karte der DIAGONAL-Lane -- aus der Halterschaft, nicht aus mir.

    #82 (fnFL2w8, 21.09.): hier stand
    ``int(getattr(group[0], "dst_rank", device))``, und der Default war
    nicht der Ausnahmefall, sondern der stille Normalfall. Ein Deskriptor
    ohne ``dst_rank`` landet schon eine Stufe vorher in der Diagonal-Gruppe
    (``group_descs_by_pair`` liest ihn mit demselben Default,
    weight_exchange_bounce.py:2064 -> ``pair_of(-1, -1)``), und DANN nimmt
    diese Zeile MEINE Karte. Zwei Defaults, dieselbe Luecke, und zusammen
    ergeben sie eine Lane, auf der niemand deponiert.

    GEMESSEN an fnFL2w8: P sammelte auf 9 Lanes, D bediente 6. Die drei
    ueberzaehligen waren c1, c2 und p3 -- genau die Diagonal-Lanen der
    Karten, auf denen die sammelnden PP-Stufen SELBST sitzen. Der Tensor
    ``layers.29.attn_hyper_connection.block_inject_weight.weight`` liegt auf
    BEIDEN Seiten auf rank 0 (der 5090), P wartete trotzdem auf c1. Drei
    Boots (w3/w7/w8) sind daran gestorben.

    Unter symmetrischen Layouts faellt der Fehler nicht auf: da IST die
    eigene Karte die Zielkarte. Unter Form A haelt der Attention-Host alles,
    und die Worker-Karten haben auf ihrer Diagonalen nichts abzulegen.

    Deshalb VERWEIGERT diese Funktion, statt zu raten. Ein Flip, der auf der
    falschen Lane lauscht, stirbt ohnehin -- nur 120 s spaeter und ohne zu
    sagen, woran. Die Verweigerung nennt Phase, Tensor und die Karte, die
    ich genommen haette.
    """
    first = group[0] if group else None
    card = getattr(first, "dst_rank", None)
    if card is None:
        name = getattr(first, "name", None) or getattr(first, "tag", "?")
        raise RuntimeError(
            f"W82 Weg2DiagonalCardUnknown: phase={phase} the diagonal lane "
            f"needs the card that HOLDS the tensor, and this descriptor "
            f"carries no dst_rank (first of {len(group)}: {name!r}). Falling "
            f"back to my own card ({device}) is what made P wait on c1/c2/p3 "
            f"in fnFL2w8 while D deposited on c0 -- a lane nobody serves. "
            f"Set dst_rank on the descriptor, or route this tensor as a "
            f"cross pair."
        )
    return int(card)


def _any_scheduler_process_alive() -> bool:
    """Lebt ueberhaupt noch ein Scheduler ausser mir? (#79)

    FAIL-OPEN wie der Aufrufer: kann diese Funktion nichts lesen, antwortet
    sie True -- weiterwarten im Budget. Sie liest /proc direkt statt pgrep zu
    starten, weil sie im Collect-Pfad je Einheit laufen kann.
    """
    import os as _os

    me = _os.getpid()
    try:
        for entry in _os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == me:
                continue
            try:
                with open(f"/proc/{pid}/comm") as fh:
                    if "sglang" in fh.read():
                        return True
            except OSError:
                continue
    except OSError:
        return True
    return False

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
    #: #108/2: (collected descs, expected descs) of the last inject, written
    #: by :meth:`_weg2_xchg_inject_from_peer`, read by the cover check after
    #: it.  The third time for the lesson above: 7a3d8f5ceb assigned it as an
    #: ad-hoc attribute, and fnFL2x5/x7 -- the first boots whose wake got
    #: this far -- died on ``AttributeError`` in the assignment itself on all
    #: three P ranks.
    _weg2_last_inject_cover: Any = None
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
    #: #1330 B4n SLICE 3.  A FIELD for the same reason as the two above, and
    #: the reason is measured rather than remembered: boot weg2xsn7 lost 24 of
    #: 24 legs on BOTH groups to `AttributeError: ... has no attribute
    #: '_weg2_shadow_region_cache'` because `slots=True` turns a lazily
    #: assigned attribute into an error ON THE WRITE.  This cache is written
    #: lazily in :meth:`_weg2_xchg_sems`, so it is declared here.
    _weg2_xchg_sems_cache: Any = "unset"

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
    #: 2026-09-15 (Nutzer-Order "digest nur einmal je leg"): the rank's last
    #: GRADED reading of its own bytes -- the wake leg's ``after`` (verdict
    #: MATCH) or a sleep leg's real ``before``. Serving never writes weights,
    #: so the next leg's ``before`` IS this reading: one device walk per leg
    #: (the waking side's ``after``) instead of two on each side.
    weg2_seam_ref: Any = None
    #: #1376 W10: THE PER-TAG LOCKSTEP'S OWN STATE, DECLARED. This class is
    #: `@dataclass(kw_only=True, slots=True)`, so an attribute that is not a
    #: field cannot be assigned at all -- and #1374's F1b assigned two of them
    #: lazily. Boot weg2xsn31/2 died three seconds after
    #: `WEG2-DORMANT set: kv_cache paused` on all three D ranks:
    #:   weight_updater.py:4089 release_memory_occupation
    #:     -> :1172 _weg2_xchg_deposit_before_sleep
    #:       -> :3528 _weg2_xchg_bounce_leg  self._weg2_xchg_tag_seen = seen
    #:   AttributeError: 'SchedulerWeightUpdaterManager' object has no
    #:   attribute '_weg2_xchg_tag_seen'
    #: then W29 on every rank, the NCCL heartbeat break and W17 Weg2GroupDead.
    #: The wall sat BEHIND the deadlock F1 removed, which is why no earlier
    #: boot reached it.
    #:
    #: `_weg2_xchg_collected_per_tag` is the SAME defect one call site over and
    #: had not been reached yet: it would have been the next AttributeError, on
    #: the wake side. Declared here rather than written lazily, and read as a
    #: field rather than through a `getattr` default -- a default would hide
    #: exactly this and is what the operator's order rules out.
    #: Which tags this leg has already deposited, so the per-tag drain knows
    #: whether it is at the first tag (prime the counter) or a later one (wait
    #: for the collector's `drained`). `None` until the first leg sets it.
    _weg2_xchg_tag_seen: Optional[set] = None
    #: weg2xsn86 (#1378): ``{(tag, param_name)}`` this rank consumes but
    #: never writes -- MEASURED target shares of the draft (embed_tokens on
    #: D IS the target's tensor); set by ``_weg2_shadow_plan``'s draft branch,
    #: read by ``_weg2_xchg_bounce_leg`` for ``run_sequential_units``.
    _weg2_xchg_no_write: Optional[frozenset] = None
    #: 2026-09-15 (Beschleunigung): per-lane count of tags this rank has run
    #: through ``run_sequential_units`` -- the buffer slot (seq % depth) and
    #: the depth-2 drain rule read it; BOTH sides count the same tags per
    #: lane (the join is symmetric), so the slots agree without a message.
    _weg2_xchg_lane_seq: Optional[dict] = None
    #: 2026-09-15 (weg2xsn94): per-LEG cache of the join, the lane plan and
    #: the address books -- derived ONCE per leg instead of per lane and tag
    #: (~1 s per tag on every rank: join_manifests + plan_from_join + books,
    #: and the shadow plan with its 1.2M-piece pointer profile).
    _weg2_xchg_leg_cache: Optional[dict] = None
    #: Punkt 2 (weg2xsn107): True while the wake worker collects a tag --
    #: the credit wait's stuck-lane reader must not read those in-flight
    #: bands as 'never drained' (PP0 W108 at the credit wait on xsn107).
    _weg2_wake_inflight: bool = False
    #: Punkt 3 (2026-09-15): the wake worker folds each tag's pieces right
    #: after that tag's collect (per-tag `after` parts, keyed by piece key)
    #: and the leg's `after` reading is ASSEMBLED from them in inventory
    #: order -- the reading digest is an ordered fold over the pieces, so the
    #: assembled reading equals a whole walk. The leg's inventory is taken
    #: once (cache) so the worker does not re-walk the model per tag.
    _weg2_seam_after_parts: Optional[dict] = None
    _weg2_seam_after_threads: Optional[list] = None  # #1437: slots=True dataclass, the field must be declared
    #: #1450: a seam-digest refusal graded BEHIND the wake -- raised at this
    #: rank's next leg or idle tick, never lost.  slots=True: declared here.
    weg2_seam_pending_refusal: Optional[BaseException] = None
    #: #1450b: the finisher thread of the deferred grade -- JOINED at the head
    #: of the next leg before any page is paused (boot weg2xsn208: the fold
    #: read weight pages the next release had just unmapped -> illegal memory
    #: access on D, P's leg W68).  slots=True: declared here.
    _weg2_seam_finisher: Any = None
    #: xsn261: the boot-time lane-buffer registration thread (or None).
    _weg2_prewarm_thread: Any = None
    #: fnFL2x40: the boot-time manifest-join warm-up thread (or None).
    _weg2_join_prewarm_thread: Any = None
    #: H46: the boot-time ring-file registration thread (or None).
    _weg2_ring_prereg_thread: Any = None
    _weg2_bar1: Any = None            # BAR1 lanes registry (weg2/bar1_lanes.py), built at boot
    _weg2_bar1_thread: Any = None     # its setup thread (windows served, peers mapped)
    _weg2_flip_index_now: object = None  # the flip index of the leg in progress (BAR1 flag seq = '<flip>-<tag>')
    _weg2_leg_tag_order: Any = None   # the wake leg's tag list (index = tag order for the tag-order gate)
    _weg2_tag_done: Any = None        # {tag index: threading.Event}, set when that tag's collect is through
    #: H11: the wake leg's per-lane turns for host/IPC lanes (weg2/lane_turns.py),
    #: None = the every-earlier-tag gate (SGLANG_WEG2_WAKE_LANE_TURNS=0).
    _weg2_lane_turns: Any = None
    #: weg2xsn269/270: the VramCredit of the leg this rank is SLEEPING
    #: through (set in release_memory_occupation, read by
    #: _weg2_stage_charge). A slots dataclass: an undeclared attribute
    #: killed group P 60 s after launch (xsn270, AttributeError at the
    #: first sleep leg) -- declare, never just assign.
    _weg2_leg_credit: Any = None
    #: weg2xsn271: WEG2-CREDIT-FLOOR logged once per rank.
    _weg2_floor_noted: bool = False
    _weg2_sleep_count: int = 0
    #: H15: the local-memory park of the last complete sleep (weg2/sleep_lmem.py
    #: LmemPark), None once the wake restored it. slots=True: declared here.
    _weg2_lmem_park: Any = None
    #: H47: the boot's base stack (bytes) from the first park; 0 = no park yet.
    _weg2_lmem_base_stack: int = 0
    #: H25: the draft's pinned host image (weg2/draft_park.DraftHostPark),
    #: created at the first park and reused for the life of the process.
    _weg2_draft_park: Any = None
    #: H25e: the park's own verdict "this rank's draft tag holds nothing"
    #: ('storages=0 tms_bytes=0', a Form-A worker's shadow draft), recorded
    #: at the sleep; None = no such verdict (parked, or never asked).
    _weg2_draft_not_carried: Any = None
    #: H25: (phase list, index) of the wake RPC where the draft H2D started.
    _weg2_draft_unpark_ph0: Any = None
    _weg2_kv_deferred: bool = False       # Wake-Parallel: kv resume deferred to the weights call
    _weg2_kv_epoch_done: object = None    # Wake-Parallel: flip epoch whose kv resume is done
    _weg2_kv_resumed_epoch: object = None  # Wake-Parallel: flip epoch whose kv_cache tms resume (the RESUME half) already ran
    _weg2_leg_min_free_mib: object = None  # xsn323: the LOWEST card-free (MiB) seen at a tag claim of this rank's last wake legs
    _weg2_leg_min_free_epoch: object = None  # the epoch that minimum belongs to (reset at the first claim of a new epoch)
    _weg2_graph_deferred: bool = False    # Wake-Parallel: cuda_graph resume deferred to the weights call
    _weg2_weights_epoch_done: object = None  # Wake-Parallel: flip epoch whose weight legs are collected
    #: #1452b: snapshot counter -- slots=True, so it is a FIELD (boot weg2xsn208
    #: printed 'n/a (AttributeError ... _1452_snapshots)' on every rank).
    _1452_snapshots: int = 0
    _weg2_seam_leg_inventory: Any = None
    #: True once the resume loop has collected tag by tag, so the once-per-wake
    #: entry stands down instead of injecting a second time over bytes already
    #: written.
    _weg2_xchg_collected_per_tag: bool = False
    #: #1391 (DESK10) ROUND 4: the SAME field, for the SHADOW carrier.
    #: Declared for the SAME reason `_weg2_xchg_collected_per_tag` is: a
    #: lazily-assigned attribute is the exact AttributeError class boot
    #: weg2xsn31/2 died of (see the comment above). Set True the moment the
    #: per-tag loop runs a SHADOW-mode collect for a tag (mirroring the
    #: EXCHANGE branch beside it), read by `_weg2_xchg_shadow_compare` to
    #: stand down its own once-per-wake, whole-plan compare instead of
    #: grading the same bytes twice.
    _weg2_xchg_shadow_compared_per_tag: bool = False
    #: #1378 xsn55: the lane audit's per-boot cache of THIS rank's own
    #: (region, name) keys. Declared for the SAME reason the three fields
    #: above are -- the class is ``slots=True``, so the lazy write the first
    #: version used raised ``AttributeError`` on its first call (measured,
    #: not inferred: executing ``_weg2_owned_name_keys`` on a constructed
    #: manager raised exactly that) and the audit's ``owned=`` would have
    #: died on the first lane of the next boot instead of counting.
    _weg2_owned_name_keys_cache: Optional[set] = None

    # ---- xsn261: lane buffers registered at boot, not in the first flip -----

    def _weg2_prewarm_lanes_start(self) -> None:
        """weg2xsn261 (17.09.): the first flip of every boot paid ~10 s before
        its second tag -- cudaHostRegister of the lane buffers at first use
        (P side 5.3-6.4 s per lane, register rate ~0.4 GB/s over freshly
        truncated tmpfs pages) and, on the shared card, the co-located
        sleeper's pause() blocked behind it (D-TP0 pause_ms=6174 while TP1/TP2
        took 20 ms). The same buffers are persistent for the boot anyway
        (SGLANG_WEG2_SEQ_PERSIST_BUFFERS), so they are created and registered
        HERE, on a daemon thread that waits for both groups' manifests, sizes
        every lane from the same join the legs use, and touches nothing the
        flip would not have touched. SGLANG_WEG2_LANE_PREWARM=0 keeps the old
        first-use form."""
        try:
            from sglang.srt.weg2 import weight_exchange as wx

            if not wx.exchange_armed():
                return
            # xsn263 (17.09.): DEFAULT OFF. The boot-time form pinned every
            # lane at its max for BOTH slots on all six ranks at once
            # (10-12 GiB per rank), shmem rose to 64 GiB during the launch
            # and the host ledger latched W98 (cushion 1.42 < 1.50 GiB) --
            # the flip's lazy growth reaches a smaller steady state (slot 1
            # only where a lane carries a second tag) and never inside the
            # launch transient. What made the first flip cheap was the tmpfs
            # populate before cudaHostRegister (register_ms 22-92 instead of
            # 5000-21000), and that stays on the lazy path. An exact
            # slot-parity sizing can re-enable this later (=1).
            # 18.09. BAR1 lanes (user order): the registry object exists on
            # EVERY rank before its thread runs, so a depositor without a
            # mapped peer writes mode=host at once and no collector waits
            # for a mode file that never comes.
            self._weg2_bar1_start()
            self._weg2_join_prewarm_start()
            # H46: the H44 ring files of this rank's diagonal lane, created
            # and registered now instead of at the first host-path tag.
            try:
                self._weg2_ring_preregister_start()
            except Exception as _rp_exc:  # noqa: BLE001 -- never costs the other warm-ups
                logger.info("WEG2-SEQ preregister not started: %r", _rp_exc)
            # H46b: D's draft host image, allocated now (main thread, the
            # rank's device current) instead of in the first sleep's park.
            try:
                self._weg2_draft_prealloc_at_boot()
            except Exception as _dp_exc:  # noqa: BLE001 -- the first park allocates as before
                logger.info("WEG2-DRAFT-PARK host image not preallocated: %r", _dp_exc)
            if (os.environ.get("SGLANG_WEG2_LANE_PREWARM", "0") or "0") != "1":
                return
            import threading

            t = threading.Thread(target=self._weg2_prewarm_lanes,
                                 name="weg2-lane-prewarm", daemon=True)
            self._weg2_prewarm_thread = t
            t.start()
        except Exception as exc:  # noqa: BLE001 -- a warm-up never breaks a boot
            logger.info("WEG2-LANE-PREWARM not started: %r", exc)

    def _weg2_ring_preregister_start(self) -> None:
        """H46: create, populate and cudaHostRegister this rank's lane ring
        files (``c<rank>[_s<k>]_ring.bin``, H44) on a daemon thread at boot.

        x148 paid every lane-buffer registration of the boot inside the first
        D->P flip (register_ms 5918-7867 for sizes that x147/x150 registered in
        293-342 ms); D deposits then waited 6-8.4 s for their collectors. The
        ring's size is pure geometry (``seq_ring_bytes`` of the ring slots and
        the sync batch), so nothing waits for manifests: the files exist
        before either group can flip, and the flip's ``_persistent_host_buffer``
        finds them (``persist ... reuse``). The co-card peer process maps and
        registers the same files (the tmpfs pages exist once).
        SGLANG_WEG2_SEQ_LANE_RING_PREREGISTER=0 keeps the first-use form."""
        from sglang.srt.weg2 import weight_exchange_bounce as bx
        from sglang.srt.weg2 import weight_exchange_region as xr

        why = bx.ring_preregister_skip_reason()
        if why:
            logger.info("WEG2-SEQ preregister skipped: %s", why)
            return
        boot_nonce = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
        rank = self._weg2_rank()
        if not boot_nonce or rank is None or int(rank) < 0:
            logger.info("WEG2-SEQ preregister skipped: boot=%r rank=%r",
                        boot_nonce, rank)
            return
        device = -1
        try:
            import torch

            if torch.cuda.is_available():
                device = int(torch.cuda.current_device())
        except Exception:  # noqa: BLE001 -- the thread then names the device it lacks
            device = -1
        import threading

        t = threading.Thread(
            target=self._weg2_ring_preregister,
            kwargs={"boot_nonce": boot_nonce, "rank": int(rank), "device": device},
            name="weg2-ring-prereg", daemon=True)
        self._weg2_ring_prereg_thread = t
        t.start()

    def _weg2_ring_preregister(self, *, boot_nonce: str, rank: int, device: int,
                               ops=None, set_device=None) -> dict:
        """The thread body: bind THIS rank's device first (a fresh thread's
        runtime device is 0 -- a register there would open a context on a
        foreign card), then register. The hooks exist for the hermetic test."""
        from sglang.srt.weg2 import weight_exchange_bounce as bx

        try:
            if ops is None:
                ops = self._weg2_xchg_device_ops()
            if ops is None:
                logger.info("WEG2-SEQ preregister skipped: no device ops on rank %d",
                            int(rank))
                return {}
            if int(device) < 0:
                logger.info("WEG2-SEQ preregister skipped: rank %d has no CUDA "
                            "device to bind the registering thread to", int(rank))
                return {}
            if set_device is None:
                def set_device(dev):
                    import torch

                    torch.cuda.set_device(int(dev))
                    ops.set_device(int(dev))
            set_device(int(device))
            return bx.preregister_ring_lanes(boot_nonce, [f"c{int(rank)}"], ops,
                                             log=logger.info)
        except Exception as exc:  # noqa: BLE001 -- a warm-up never breaks a boot
            logger.info("WEG2-SEQ preregister stopped: %r -- the first host-path "
                        "tag maps its ring file as before", exc)
            return {}

    def _weg2_join_prewarm_start(self) -> None:
        """fnFL2x40: join this boot's manifests on a daemon thread, before the
        first flip needs them.

        The first leg of the boot joined them on its critical path, once per
        region in the hook and again in every lane thread of the first tag
        (0.55-0.57 s per join, three concurrent 2.09 s wall, measured on x40's
        manifests): 6.3-7.7 s in the first deposit of every rank against
        0.1-0.4 s for every later tag. ``xchg_manifest.join_manifests`` keeps
        each distinct input once per process, so a join done here is the one
        the flip finds. SGLANG_WEG2_JOIN_PREWARM=0 keeps the first-use form.
        """
        if (os.environ.get("SGLANG_WEG2_JOIN_PREWARM", "1") or "1") == "0":
            return
        import threading

        t = threading.Thread(target=self._weg2_prewarm_joins,
                             name="weg2-join-prewarm", daemon=True)
        self._weg2_join_prewarm_thread = t
        t.start()

    def _weg2_prewarm_joins(self, *, poll_s: float = 1.0, stable_s: float = 5.0,
                            budget_s: float = 900.0, rounds: int = 3) -> int:
        """Wait until the manifest files stop changing, join them, repeat when
        they change again (the drafter rewrites its manifest after the load).
        Returns the number of warm-ups done; a warm-up never breaks a boot."""
        from sglang.srt.weg2 import xchg_manifest as xm

        done = 0
        last_warm = None
        deadline = time.monotonic() + float(budget_s)
        try:
            while done < int(rounds) and time.monotonic() < deadline:
                if done:
                    # after the first warm-up only a REWRITE is left to catch
                    deadline = min(deadline, time.monotonic() + 120.0)
                sig = xm.manifest_files_signature()
                t_sig = time.monotonic()
                while time.monotonic() < deadline:
                    time.sleep(float(poll_s))
                    now = xm.manifest_files_signature()
                    if now != sig:
                        sig, t_sig = now, time.monotonic()
                    elif time.monotonic() - t_sig >= float(stable_s):
                        break
                if sig == last_warm:
                    continue
                ms = xm.prewarm_joins()
                if not ms:
                    continue  # not every rank's manifest is there yet
                last_warm = sig
                done += 1
                ms.update(self._weg2_warm_leg_cache())
                logger.info(
                    "WEG2-JOIN-PREWARM group=%s rank=%s round=%d files=%d %s "
                    "memo=%s -- the first flip finds these joins done "
                    "(fnFL2x40: they were its first deposit's critical path)",
                    self._weg2_group_name(), self._weg2_rank(), done, len(sig),
                    " ".join(f"{k}_ms={v:.0f}" for k, v in ms.items()),
                    xm.join_memo_stats())
        except Exception as exc:  # noqa: BLE001 -- a warm-up never breaks a boot
            from sglang.srt.weg2 import weight_exchange as wx

            if isinstance(exc, (wx.Weg2XchgSourceMissing,
                                wx.Weg2XchgPlanDisagree)):
                # fnFL2x100: the SAME join the first flip will ask, refused
                # at boot -- said loudly here, where it is minutes before the
                # flip, instead of as "stopped" among the info lines.
                logger.error("WEG2-JOIN-PREWARM REFUSED group=%s rank=%s: %s "
                             "-- the first flip leg will refuse on every rank",
                             self._weg2_group_name(), self._weg2_rank(), exc)
            else:
                logger.info("WEG2-JOIN-PREWARM stopped: %r", exc)
        return done

    def _weg2_warm_leg_cache(self, *, agree_budget_s: float = 60.0,
                             agree_poll_s: float = 1.0) -> Dict[str, float]:
        """fnFL2x82: derive this rank's lane PLAN for both hooks at boot, into
        the same cache ``_weg2_seq_lane_descs`` reads (``("join", hook, group,
        rank)``), so the first flip's lane threads find it.

        x80 (D TP0, first wake): the join was warm (memo hit 11) but the plan
        from it was not -- three lane threads derived it at once (400 ms each,
        WEG2-LANE-DERIVE), the first collect started 0,33 s after the first
        resume and PP0's first tag waited 632 ms for it. Returns ``{what: ms}``;
        an empty dict when the manifests are not all there or the rank has no
        group name. Never breaks a boot."""
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import weight_exchange_shadow as sh
        from sglang.srt.weg2 import xchg_manifest as xm

        group = self._weg2_group_name()
        rank = self._weg2_rank()
        if not group or rank is None or int(rank) < 0:
            return {}
        mans, _why = xm.manifests_for_boot(pp_group="P", tp_group="D")
        if mans is None:
            return {}
        _lc = getattr(self, "_weg2_xchg_leg_cache", None)
        if _lc is None:
            _lc = {}
            try:
                self._weg2_xchg_leg_cache = _lc
            except AttributeError:
                return {}
        out: Dict[str, float] = {}
        join = xm.join_manifests(mans, pp_group="P", tp_group="D")
        for hook in (sh.HOOK_SOURCE, "authoritative"):
            t0 = time.perf_counter()
            plan = xm.plan_from_join(join, direction=wx.leg_direction(hook, group))
            _lc[("join", str(hook), str(group), int(rank))] = (join, plan)
            out[f"leg:{hook}"] = (time.perf_counter() - t0) * 1000.0
        # fnFL2x84: THE HOOK'S OWN PLAN TOO. x83: the source hook re-derived
        # its region plans before P's first deposit (HOOK-TIME plan=125 ms)
        # and the destination hook after D's last collect (plan=168 ms,
        # regions weights + weights_draft), both on the critical path, both
        # keyed by the pair's manifest agreement -- a boot constant. The same
        # agreement is made here, so the flip's key hits.
        peer = "D" if str(group) == "P" else "P"
        # x84: the agreement is a two-sided handshake ("peer-unready: the
        # peer's manifest is published but has not yet seen ours") -- the
        # first call publishes our side, the peer's own warm-up answers it,
        # so the agreement is polled for a bounded while before giving up.
        agreed = None
        deadline = time.monotonic() + float(agree_budget_s)
        while True:
            agreed, state = self._weg2_shadow_manifest(
                str(group), peer, int(rank), leg=0, epoch="prewarm")
            if agreed is not None or time.monotonic() >= deadline:
                break
            time.sleep(float(agree_poll_s))
        for hook in (sh.HOOK_SOURCE, sh.HOOK_DESTINATION):
            if agreed is None:
                out[f"hook:{hook}"] = -1.0
                continue
            t0 = time.perf_counter()
            self._weg2_shadow_plan(str(hook), str(group), int(rank),
                                   agreed=agreed, require_agreement=True)
            out[f"hook:{hook}"] = (time.perf_counter() - t0) * 1000.0
        # H11: THE PER-TAG KEY TOO. x83-x105: the deposit of every tag
        # (hook=source) and the collect (hook=authoritative) read the plan
        # WITHOUT an agreement -- a key neither warm-up above fills. The first
        # tag of the first P->D flip derived it: PP0's no-op weights_13 took
        # 106/121/517/134 ms (x83/x87/x104/x105) before its first byte, D's
        # first collect began 0,2-0,33 s after its first resume.
        from sglang.srt.environ import envs

        if envs.SGLANG_WEG2_TAG_PLAN_PREWARM.get():
            for hook in (sh.HOOK_SOURCE, "authoritative"):
                out[f"tag:{hook}"] = self._weg2_warm_tag_plan(
                    _lc, str(hook), str(group), int(rank))
        return out

    def _weg2_warm_tag_plan(self, cache: dict, hook: str, group: str,
                            rank: int) -> float:
        """H11: derive the per-tag plan (``agreed=None``,
        ``require_agreement=False``) into ``cache`` and return its ms, -1.0
        when no plan came out. OVERWRITES the key: a later warm-up round runs
        because a manifest changed (the drafter rewrites its own after the
        load), and the plan of the newer manifests is the one the flip must
        find."""
        t0 = time.perf_counter()
        with _WEG2_PLAN_LOCK:
            out = SchedulerWeightUpdaterManager._weg2_shadow_plan_uncached(
                self, hook, group, rank, agreed=None, require_agreement=False)
            if out is None or out[0] is None:
                return -1.0
            cache[_weg2_plan_key(hook, group, rank, False, None)] = out
        return (time.perf_counter() - t0) * 1000.0

    def _weg2_bar1_start(self) -> None:
        """Build the BAR1 lane registry (weg2/bar1_lanes.py) and run its
        setup on a helper thread: this rank serves a window for every cross
        lane it RECEIVES on and maps the peer window of every lane it SENDS
        on. Refusals are named per lane; the leg falls back to the host
        buffer for that lane only."""
        try:
            from sglang.srt.weg2 import bar1_lanes as b1
            from sglang.srt.weg2 import weight_exchange_region as xr
            import torch

            if not b1.lanes_on():
                logger.info("WEG2-BAR1 off (%s=0): host lanes", b1.ENV_ON)
                return
            boot_nonce = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
            group = self._weg2_group_name()
            rank = self._weg2_rank()
            if not boot_nonce or not group or rank is None or int(rank) < 0:
                logger.info("WEG2-BAR1 skipped: boot=%r group=%r rank=%r",
                            boot_nonce, group, rank)
                return
            rank = int(rank)
            device = int(torch.cuda.current_device())
            lanes = [f"p{i}" for i, (s, d) in enumerate(xr.CROSS_PAIRS)
                     if rank in (int(s), int(d))]
            reg = b1.Bar1Lanes(boot_nonce, group, rank, device, xr.CROSS_PAIRS,
                               log=logger.info)
            self._weg2_bar1 = reg
            import threading

            t = threading.Thread(target=reg.setup, args=(lanes,),
                                 name="weg2-bar1-setup", daemon=True)
            self._weg2_bar1_thread = t
            t.start()
        except Exception as exc:  # noqa: BLE001 -- a lane setup never breaks a boot
            logger.info("WEG2-BAR1 not started: %r", exc)

    def _weg2_prewarm_lanes(self, *, manifests_ready=None, lane_bytes_of=None,
                            persist=None, family=None, poll_s: float = 2.0,
                            budget_s: float = 900.0) -> Dict[str, int]:
        """Size and register every lane buffer this rank will map (both
        roles: depositor and collector, both buffer slots) from the join.
        Returns ``{lane_key: bytes}``; ``{}`` and a line when nothing could be
        derived. The hooks exist for the hermetic test."""
        from sglang.srt.weg2 import weight_exchange_bounce as bx
        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2 import weight_exchange_shadow as sh
        from sglang.srt.weg2 import xchg_manifest as xm
        from sglang.srt.managers import weg2_memory_saver as ms

        t0 = time.perf_counter()
        boot_nonce = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
        group = self._weg2_group_name()
        rank = self._weg2_rank()
        if not boot_nonce or not group or rank is None or int(rank) < 0:
            logger.info("WEG2-LANE-PREWARM skipped: boot=%r group=%r rank=%r",
                        boot_nonce, group, rank)
            return {}
        rank = int(rank)
        if manifests_ready is None:
            def manifests_ready():
                mans, _why = xm.manifests_for_boot(pp_group="P", tp_group="D")
                return mans is not None
        deadline = time.monotonic() + float(budget_s)
        while not manifests_ready():
            if time.monotonic() >= deadline:
                logger.info("WEG2-LANE-PREWARM NOT-READY: the manifests of both "
                            "groups did not appear within %.0f s; the first flip "
                            "registers at first use as before", float(budget_s))
                return {}
            time.sleep(float(poll_s))
        if family is None:
            _chunk_layers, chunk_count = ms.weight_chunk_geometry()
            family = list(ms.weights_family_tags(int(chunk_count))) if int(chunk_count) > 0 else []
        if lane_bytes_of is None:
            def lane_bytes_of(hook, lane_key, tag):
                pair = int(lane_key[1:]) if lane_key.startswith("p") else None
                card = None if pair is not None else int(lane_key[1:])
                descs = self._weg2_seq_lane_descs(
                    hook=hook, group=group, rank=rank, pair=pair, card=card,
                    tag=tag, log=None)
                return sum(int(getattr(d, "nbytes", 0) or 0) for d in (descs or ()))
        lanes = [f"c{rank}"] + [f"p{i}" for i, (s, d) in enumerate(xr.CROSS_PAIRS)
                                 if rank in (int(s), int(d))]
        biggest: Dict[str, int] = {}
        failures = 0
        for hook in (sh.HOOK_SOURCE, "authoritative"):
            for lk in lanes:
                for tag in family:
                    try:
                        b = int(lane_bytes_of(hook, lk, tag) or 0)
                    except Exception as exc:  # noqa: BLE001 -- sized lanes still register
                        failures += 1
                        if failures <= 3:
                            logger.info("WEG2-LANE-PREWARM lane=%s hook=%s tag=%s "
                                        "not sized: %r", lk, hook, tag, exc)
                        b = 0
                    if b > biggest.get(lk, 0):
                        biggest[lk] = b
        if persist is None:
            ops = self._weg2_xchg_device_ops()
            root = xr.SHM_ROOT

            def persist(path, nbytes, lk):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                return bx._persistent_host_buffer(path, int(nbytes), ops, lk, logger.info)
        depth = max(1, int(bx.seq_buffer_depth()))
        n = 0
        total = 0
        for lk in sorted(biggest):
            b = int(biggest[lk])
            if b <= 0:
                continue
            for slot in range(depth):
                path = bx.sequential_buffer_path(
                    boot_nonce, xr.SHM_ROOT, lane=bx.seq_lane_file_name(lk, slot))
                try:
                    persist(path, b, lk)
                    n += 1
                    total += b
                except Exception as exc:  # noqa: BLE001 -- the leg registers at first use
                    logger.info("WEG2-LANE-PREWARM lane=%s slot=%d failed: %r", lk, slot, exc)
        logger.info("WEG2-LANE-PREWARM group=%s rank=%d lanes=%s buffers=%d "
                    "pinned=%.2f GiB sizing_failures=%d ms=%.0f -- the first flip "
                    "finds every lane buffer registered (reuse), nothing to "
                    "register on its critical path",
                    group, rank,
                    ",".join(f"{k}:{v >> 20}MiB" for k, v in sorted(biggest.items()) if v > 0),
                    n, total / (1 << 30), failures, (time.perf_counter() - t0) * 1000)
        return biggest

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

    def _weg2_pcie_lock_retired(self, label: str, direction: Optional[str] = None):
        """#1378 xsn36: the WHOLE-LEG card lock is retired for the flip legs.

        It deadlocked the co-located pair by construction: the holder waits
        INSIDE it for the sibling's semaphore posts, which the sibling cannot
        produce without the same lock (measured twice -- weg2xsn35's
        PcieLockTimeout held=124.45s vs budget 120s, weg2xsn36's W68 after a
        FULL 600 s wait). The serialisation moved per COPY into
        ``run_bounce_leg`` (``pcie_uuid``); the rendezvous waits sit outside
        every lock. Named no-op rather than deleted so the call sites keep
        their shape auditable; the OTHER pcie-lock users (the disk reloads)
        keep the real lock -- they are single copies with no rendezvous wait
        inside.
        """
        import contextlib
        return contextlib.nullcontext()

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

    def _weg2_tag_resident_bytes(self, tag: str) -> Optional[int]:
        """The saver's OWN byte sum for ``tag`` as a THREE-VALUED reading:
        ``None`` when the saver cannot answer (no adapter, no ``tms_tag_bytes``
        symbol, or the call raised), else the integer it answered -- a real
        ``0`` INCLUDED.

        :meth:`_weg2_tag_bytes` folds "could not answer" and "answered 0"
        into one 0 on purpose (its callers print a byte count and must never
        print a manufactured zero).  The wake-source gap check needs the two
        apart: on a PIPELINE stage a family tag that lives on ANOTHER stage is
        a genuine 0 by the allocator's own metadata (weg2xsn83: PP0 owns
        weights_0..4, asked about weights_6 -> tms_tag_bytes=0, plan
        descs=0), and that 0 is the fact that makes an empty deposit for the
        tag a no-op instead of a gap -- while an UNMEASURABLE absence keeps
        the refusal, because nothing then vouches that the bytes are elsewhere.
        """
        adapter = getattr(self, "memory_saver_adapter", None)
        getter = getattr(adapter, "tag_bytes", None)
        if getter is None:
            return None
        try:
            value = getter(tag)
        except Exception:  # noqa: BLE001
            return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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

    def _weg2_nvml_self_bytes(self) -> Optional[int]:
        """NVML per-process bytes for THIS pid on THIS card, or None (never 0)."""
        try:
            import pynvml  # noqa: PLC0415
            uuid = self._weg2_card_uuid()
            if not uuid:
                return None
            handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            me = os.getpid()
            for pr in pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle):
                if int(pr.pid) == me and pr.usedGpuMemory is not None:
                    return int(pr.usedGpuMemory)
            return None
        except BaseException:  # noqa: BLE001 -- an instrument never breaks a leg
            return None

    def _weg2_dump_dc_snapshot(self, stage: str) -> Optional[str]:
        """#1452: torch.cuda.memory._dump_snapshot of THIS rank into the
        evidence dir (SGLANG_WEG2_RANKDUMP_DIR), once per sleep.  Read with
        the debugtools memsnapshot_analyze tool.  Fail-soft, never raises."""
        try:
            n = getattr(self, "_1452_snapshots", 0) + 1
            self._1452_snapshots = n
            if n > 4:
                return None
            root = os.environ.get("SGLANG_WEG2_RANKDUMP_DIR") or "/tmp"
            path = os.path.join(
                root, "memsnap_%s_rank%s_n%d.pickle"
                % (self._weg2_group_name() or "g", self._weg2_rank(), n))
            torch.cuda.memory._dump_snapshot(path)
            logger.info("WEG2-DC-SNAPSHOT stage=%s wrote %s (untagged census; analyse with memsnapshot_analyze)",
                        stage, path)
            return path
        except BaseException as exc:  # noqa: BLE001
            logger.info("WEG2-DC-SNAPSHOT stage=%s n/a (%s: %s)", stage, type(exc).__name__, exc)
            return None

    def _weg2_log_dc_breakdown(self, stage: str) -> Optional[Dict[str, Any]]:
        """#1446: print this rank's dormant-residue attribution (see
        weg2_memory_saver.dc_breakdown).  Fail-soft: never raises."""
        try:
            from sglang.srt.managers.weg2_memory_saver import (  # noqa: PLC0415
                dc_breakdown, format_dc_breakdown, weights_family_tags,
            )
            tags = set(str(t) for t in (getattr(self, "offload_tags", None) or ()))
            tags |= {GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH,
                     GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_WEIGHTS_DRAFT}
            try:
                tags |= set(weights_family_tags())
            except Exception:  # noqa: BLE001
                pass
            tag_bytes = {t: self._weg2_tag_bytes(t) for t in sorted(tags)}
            try:
                module = torch.get_device_module()
                reserved, allocated = int(module.memory_reserved()), int(module.memory_allocated())
            except Exception:  # noqa: BLE001
                reserved, allocated = None, None
            # #1491: the CARD terms, so the line can say whether the bytes
            # that went missing since the last wake are this process's at all.
            _card_total = None
            _card_free = None
            try:
                uuid_key = self._weg2_card_uuid()
                if uuid_key is not None:
                    from sglang.srt.registry import nvml as _nvml_registry

                    _info = _nvml_registry.memory_info_for_uuid(uuid_key)
                    _card_total = int(_info.total_bytes)
                    _card_free = int(_info.free_bytes)
            except Exception:  # noqa: BLE001 -- an absent reading is n/a, not 0
                _card_total = _card_free = None
            rec = dc_breakdown(
                nvml_proc_bytes=self._weg2_nvml_self_bytes(),
                torch_reserved=reserved, torch_allocated=allocated,
                tag_bytes=tag_bytes, offload_tags=getattr(self, "offload_tags", None),
                card_total_bytes=_card_total, card_free_bytes=_card_free,
            )
            logger.info("%s", format_dc_breakdown(rec, stage=stage))
            # #1491: and the WAKE-TO-WAKE delta, which is the question boots
            # weg2xsn406/408 could not answer -- 8387 MiB free at D TP1's
            # first wake, 5974 at its second, and no line anywhere saying
            # which post took the difference.
            _creep_key = stage.split()[0]
            if _creep_key.startswith("wake"):
                from sglang.srt.managers.weg2_memory_saver import (  # noqa: PLC0415
                    dc_creep, format_dc_creep, remember_dc,
                )

                _prev = remember_dc(_creep_key, rec)
                logger.info("%s", format_dc_creep(dc_creep(_prev, rec), stage=stage))
            # #1452: with SGLANG_WEG2_DC_SNAPSHOT=1 the allocator's snapshot of
            # the UNTAGGED remainder (794 / 772 MiB per card at weg2xsn207)
            # goes to the evidence dir once all tags are paused -- the census
            # of what torch still holds, by allocation stack (history is
            # recorded from scheduler start under the same env).
            if (os.environ.get("SGLANG_WEG2_DC_SNAPSHOT", "0") == "1"
                    and int(rec.get("tms_resident_mib") or 0) == 0):
                self._weg2_dump_dc_snapshot(stage)
            return rec
        except BaseException:  # noqa: BLE001
            logger.info("WEG2-DC-BREAKDOWN stage=%s n/a (instrument failed)", stage)
            return None

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
        self._weg2_log_sleep_residue(census, tags)

    def _weg2_log_sleep_residue(self, census: Any, tags: Optional[List[str]]) -> None:
        """weg2xsn296: name what is STILL on the card after the sleep and dump
        the allocator snapshot for it (Nutzer-Order: everything that can go
        down without killing the process goes to host RAM). Never raises."""
        try:
            import os

            import torch

            from sglang.srt.managers.weg2_memory_saver import (
                _MEMHIST_ARMED,
                sleep_residue_terms,
            )

            stats = torch.cuda.memory_stats()
            active = int(stats.get("active_bytes.all.current", 0))
            reserved = int(stats.get("reserved_bytes.all.current", 0))
            family = set(tags or ()) | {"weights", "kv_cache", "cuda_graph", "weights_draft"}
            family |= {f"weights_{i}" for i in range(16)}
            tagged = 0
            for t in sorted(family):
                a = int(self._weg2_tag_bytes(t) or 0)
                b = int(self._weg2_tag_resident_bytes(t) or 0)
                tagged += max(a, b)
            terms = sleep_residue_terms(
                active_bytes=active, reserved_bytes=reserved, tagged_bytes=tagged,
                nvml_used_bytes=getattr(census, "proc_used_bytes", None),
            )
            n = int(getattr(self, "_weg2_sleep_count", 0) or 0) + 1
            try:
                self._weg2_sleep_count = n
            except Exception:  # noqa: BLE001 -- slots dataclass without the field
                pass
            snap = "off"
            out_dir = os.environ.get("SGLANG_WEG2_RANKDUMP_DIR", "")
            if _MEMHIST_ARMED and out_dir:
                path = os.path.join(
                    out_dir,
                    f"memsnap_{os.environ.get('SGLANG_WEG2_GROUP', 'X')}_pid{os.getpid()}_sleep{n}.pickle",
                )
                try:
                    torch.cuda.memory._dump_snapshot(path)
                    snap = path
                except Exception as exc:  # noqa: BLE001
                    snap = f"failed:{type(exc).__name__}"
            logger.info(
                "WEG2-SLEEP-RESIDUE sleep=%d untagged_live=%d MiB tagged=%d MiB torch_active=%d MiB "
                "torch_reserved=%d MiB nvml_proc_used=%s MiB outside_torch=%s MiB snapshot=%s "
                "(untagged_live = torch active minus every tag's bytes: what no pause covers; "
                "outside_torch = NVML per-process minus untagged_live: context + comm windows) "
                + self._weg2_residue_posts().replace("%", "%%"),
                n, terms["untagged_live"] >> 20, terms["tagged_bytes"] >> 20, active >> 20,
                reserved >> 20,
                "n/a" if getattr(census, "proc_used_bytes", None) is None
                else int(census.proc_used_bytes) >> 20,
                "n/a" if terms["outside_torch"] < 0 else terms["outside_torch"] >> 20,
                snap,
            )
        except Exception as exc:  # noqa: BLE001 -- an instrument never kills the sleep
            logger.info("WEG2-SLEEP-RESIDUE instrument raised (%s: %s)", type(exc).__name__, str(exc)[:160])
        self._weg2_trim_host_heap_at_sleep()

    @staticmethod
    def _weg2_sm_threads() -> int:
        """SMs x max resident threads per SM: the driver's local-memory multiplier."""
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(props.multi_processor_count) * int(props.max_threads_per_multi_processor)

    def _weg2_park_lmem_at_sleep(self) -> None:
        """H15: lower the stack limit of this sleeping context (weg2/sleep_lmem.py).
        An unrestored park is kept as is: parking again would save the PARKED
        limit as the one to restore. Never raises."""
        from sglang.srt.environ import envs  # noqa: PLC0415

        if not envs.SGLANG_WEG2_SLEEP_RELEASE_LMEM.get() or self._weg2_lmem_park is not None:
            return
        try:
            from sglang.srt.weg2.sleep_lmem import CudaDriverStackLimit, park_lmem  # noqa: PLC0415

            park = park_lmem(
                driver=CudaDriverStackLimit(),
                threads=self._weg2_sm_threads(),
                nvml_bytes=self._weg2_nvml_self_bytes,
                base_stack_bytes=self._weg2_lmem_base_stack or None,
            )
        except Exception as exc:  # noqa: BLE001 -- an unparked context is the old state
            logger.info("WEG2-SLEEP-LMEM n/a (%s: %s)", type(exc).__name__, str(exc)[:160])
            return
        self._weg2_lmem_park = park
        if not park.refused:
            self._weg2_lmem_base_stack = park.base_stack_bytes
        # H101: the per-rank stack high-water of the loaded-kernel census rides
        # on the H15 line (the planner's LMEM census reads this line).
        try:
            from sglang.srt.utils.lmem_census import census_max  # noqa: PLC0415

            c_bytes, c_kernel = census_max()
            census = f" census={c_bytes}({c_kernel or '-'})"
        except Exception:  # noqa: BLE001 -- the H15 line stays
            census = ""
        logger.info("WEG2-SLEEP-LMEM %s%s", park.format_post(), census)

    def _weg2_draft_park_armed(self) -> bool:
        """H25: does THIS rank park its draft at the sleep?  Exactly when a
        draft runner lives here, the exchange arm is on and ``weights_draft``
        is NOT a family member (group P carries no draft, so nobody moves
        these bytes and the region has no cpu backup on this arm)."""
        from sglang.srt.managers.weg2_memory_saver import draft_tag_in_family
        from sglang.srt.weg2.weight_exchange import exchange_armed

        return (_weg2_drafter_of(self) is not None and bool(exchange_armed())
                and not draft_tag_in_family())

    def _weg2_draft_carried_by_park(self) -> bool:
        """H25d: does this rank's draft wake from its own pinned host image?
        True once a park wrote the image (``DraftHostPark.holds_image``); the
        image is kept for the process's life and every later wake of the
        draft reads it (``_weg2_unpark_draft_start``), never the disk."""
        park = self._weg2_draft_park
        return park is not None and park.holds_image

    def _weg2_draft_prealloc_at_boot(self) -> None:
        """H46b: allocate the draft's pinned host image at scheduler init
        (after the draft runner and its graphs exist), sized from the SAME
        population the first park lays out, so that park finds ``host image
        reused``. Only where a park would run (``_weg2_draft_park_armed`` and
        bytes to park -- TP0 on the Next-Flash form); the ledger books
        ``d_draft_host`` for the whole boot already, so the image exists
        earlier, not bigger. SGLANG_WEG2_DRAFT_PARK_PREALLOC=0 keeps the
        first-park allocation."""
        from sglang.srt.environ import envs

        if not envs.SGLANG_WEG2_DRAFT_PARK_PREALLOC.get():
            logger.info("WEG2-DRAFT-PARK host image not preallocated: "
                        "SGLANG_WEG2_DRAFT_PARK_PREALLOC=0")
            return
        if not self._weg2_draft_park_armed():
            return
        from sglang.srt.managers.weg2_memory_saver import GPU_MEMORY_TYPE_WEIGHTS_DRAFT
        from sglang.srt.weg2.draft_park import DraftHostPark, park_population

        drafter = _weg2_drafter_of(self)
        target = self.tp_worker.model_runner.model
        population = park_population(drafter.model, target)
        tag_bytes = int(self._weg2_tag_bytes(GPU_MEMORY_TYPE_WEIGHTS_DRAFT) or 0)
        if not population or tag_bytes <= 0:
            logger.info("WEG2-DRAFT-PARK host image not preallocated: nothing to "
                        "park on this rank (storages=%d tms_bytes=%d)",
                        len(population), tag_bytes)
            return
        if self._weg2_draft_park is None:
            self._weg2_draft_park = DraftHostPark(new_stream=torch.cuda.Stream)
        nbytes, ms = self._weg2_draft_park.preallocate(population)
        logger.info("WEG2-DRAFT-PARK host image preallocated bytes=%d mib=%.1f "
                    "storages=%d ms=%.0f tag=%s tms_bytes=%d -- the first sleep's "
                    "park finds it (host image reused), no cudaHostAlloc in the flip",
                    int(nbytes), int(nbytes) / float(1 << 20), len(population), ms,
                    GPU_MEMORY_TYPE_WEIGHTS_DRAFT, tag_bytes)

    def _weg2_park_draft_at_sleep(self, credit) -> None:
        """H25 (C): D2H the draft into the pinned host image, pause its tag,
        and credit the released VRAM to the waking group on this card. Runs
        at the FIRST weights RPC of a sleep, before any family tag pauses."""
        if not self._weg2_draft_park_armed():
            return
        # an H2D of the previous wake that no admission joined: join it now,
        # before the D2H reads the same storages
        self._weg2_unpark_draft_join([], where="sleep")
        from sglang.srt.managers.weg2_memory_saver import GPU_MEMORY_TYPE_WEIGHTS_DRAFT
        from sglang.srt.weg2.draft_park import DraftHostPark, park_population

        drafter = _weg2_drafter_of(self)
        target = self.tp_worker.model_runner.model
        population = park_population(drafter.model, target)
        tag_bytes = int(self._weg2_tag_bytes(GPU_MEMORY_TYPE_WEIGHTS_DRAFT) or 0)
        if not population or tag_bytes <= 0:
            # H25e: the ONE verdict the wake's W4 exemption reads.
            self._weg2_draft_not_carried = (
                f"storages={len(population)} tms_bytes={tag_bytes}")
            logger.info("WEG2-DRAFT-PARK tag=%s nothing to park on this rank "
                        "(%s) -- the wake needs no draft source here (H25e)",
                        GPU_MEMORY_TYPE_WEIGHTS_DRAFT, self._weg2_draft_not_carried)
            return
        self._weg2_draft_not_carried = None
        if self._weg2_draft_park is None:
            self._weg2_draft_park = DraftHostPark(new_stream=torch.cuda.Stream)
        rec = self._weg2_draft_park.park(
            population, tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            pause=self.memory_saver_adapter.pause,
            sync=torch.cuda.synchronize)
        if credit is not None:
            credit.publish(GPU_MEMORY_TYPE_WEIGHTS_DRAFT, tag_bytes)
        logger.info("%s tms_bytes=%d credited=%s", rec.line(), tag_bytes,
                    "yes" if credit is not None else "no-credit")

    def _weg2_unpark_draft_start(self, credit, epoch, submitted, phases) -> None:
        """H25 (C): after the family legs -- the peer's release on this card
        is complete, so the draft's bytes are covered -- claim them, resume
        the tag (same VA: the verifier's CUDA graphs stay valid) and issue the
        H2D on a side stream. :meth:`_weg2_unpark_draft_join` closes it at
        the ADMISSION (DORMANT clear), the last instant before a forward can
        read the draft -- so the copy overlaps the legs' tail, the fence, the
        front's round trip and the KV wake."""
        park = self._weg2_draft_park
        if park is None or not park.parked:
            return
        from sglang.srt.managers.weg2_memory_saver import (
            GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            weg2_tms_resume,
        )

        need = int(self._weg2_tag_bytes(GPU_MEMORY_TYPE_WEIGHTS_DRAFT) or park.nbytes)
        self._weg2_await_vram_credit(credit, GPU_MEMORY_TYPE_WEIGHTS_DRAFT, need,
                                     epoch, submitted=list(submitted))
        resume_ms = park.unpark_start(
            tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            resume=lambda t: weg2_tms_resume(self.memory_saver_adapter, t))
        self._weg2_draft_unpark_ph0 = (phases, len(phases))
        logger.info("WEG2-DRAFT-UNPARK start tag=%s resume_ms=%.0f issue_ms=%.0f bytes=%d",
                    GPU_MEMORY_TYPE_WEIGHTS_DRAFT, resume_ms, park.unpark_issue_ms,
                    park.nbytes)

    def _weg2_unpark_draft_join(self, phases, *, where: str) -> None:
        """H25 (C): block until the draft's H2D is done and name what it
        overlapped: the phases of the starting RPC after the start, and --
        when the join runs in a LATER RPC -- that RPC's phases so far."""
        park = self._weg2_draft_park
        if park is None or self._weg2_draft_unpark_ph0 is None:
            return
        started_in, idx = self._weg2_draft_unpark_ph0
        self._weg2_draft_unpark_ph0 = None
        names = [str(n) for n, _ms in list(started_in)[int(idx):]]
        if phases is not started_in:
            names.append("rpc")
            names.extend(str(n) for n, _ms in list(phases))
        names.append(where)
        line = park.join(overlap="+".join(names))
        if line:
            logger.info("%s", line)

    def _weg2_rearm_defer_armed(self) -> bool:
        """H31b: defer the extra rows past the first token? Only on the
        DECODE group (its MoE layers run the device pool; P's prefill plans
        on the host with full residency and would land every layer at once)."""
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_REARM_DEFER.get()) and self._weg2_group_name() == "D"

    def _weg2_rearm_defer_settle(self) -> None:
        """H31b: before the first pause of a sleep -- wait for a running
        deferred fill and forget what is pending (the next wake rewrites the
        tables). Only a module lookup when nothing is pending."""
        from sglang.srt.layers.moe.expert_offload import deferred_rows_fill

        deferred_rows_fill().settle()

    def _weg2_zero_local_scratch(self, models) -> list:
        """fnFL2 v43: zero the runtime-built parameters (Marlin workspaces)
        of these models; the names zeroed. A failure is NAMED, never raised
        and never swallowed silently."""
        out: list = []
        if not models:
            return out
        try:
            from sglang.srt.weg2.weight_exchange import zero_local_scratch

            for _m in models:
                out.extend(zero_local_scratch(_m))
        except Exception as _sexc:  # noqa: BLE001
            # a failure here means the first forward after this wake runs
            # on the peer's semaphores
            logger.error(
                "WEG2-RESUME local-scratch zeroing FAILED (%s) -- the "
                "first forward after this wake reads the peer's residue "
                "in every Marlin workspace", _sexc,
            )
        return out

    def _weg2_rearm_prefetch_begin(self, phases):
        """H31: plan the Platztausch pad+extra rows of every wake model by the
        tag their buffer lives under, so each tag's rows can be issued right
        behind its resume. None = the serial rearm (switch off, no offload
        layer with a map, or the plan failed -- named, never raised)."""
        from sglang.srt.environ import envs

        if not envs.SGLANG_WEG2_REARM_PREFETCH.get():
            return None
        try:
            from sglang.srt.layers.moe.expert_offload import ExpertRearmPrefetch
            from sglang.srt.managers.weg2_memory_saver import (
                GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )

            target = getattr(getattr(getattr(self, "tp_worker", None), "model_runner", None),
                             "model", None)
            regions = [(m, GPU_MEMORY_TYPE_WEIGHTS if m is target else GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
                       for m in self._weg2_wake_models()]
            pf = ExpertRearmPrefetch(regions, phases=phases)
        except Exception as exc:  # noqa: BLE001 -- the serial rearm stays
            logger.info("WEG2-REARM-PREFETCH off (%s: %s) -- the rearm loads serially",
                        type(exc).__name__, str(exc)[:160])
            return None
        if not pf.planned_layers:
            return None
        logger.info("WEG2-REARM-PREFETCH plan layers=%d rows=%d tags=%d plan_ms=%.1f",
                    pf.planned_layers, pf.planned_rows, len(pf.planned_tags), pf.plan_ms)
        return pf

    def _weg2_rearm_prefetch_issue(self, pf, tag):
        """H31: right behind ``resume(tag)``. A failed issue drops the prefetch
        for the rest of this wake (its layers load serially at the rearm);
        what it already queued is still joined there."""
        if pf is None:
            return None
        try:
            pf.issue(tag)
        except Exception as exc:  # noqa: BLE001 -- the rearm's own path raises it again
            logger.warning("WEG2-REARM-PREFETCH issue tag=%s FAILED (%s: %s) -- the "
                           "remaining layers load serially at the rearm",
                           tag, type(exc).__name__, str(exc)[:160])
        return pf

    def _weg2_restore_lmem_at_wake(self) -> None:
        """H15: put the limit saved at the park back. Never raises."""
        park = self._weg2_lmem_park
        if park is None:
            return
        self._weg2_lmem_park = None
        if park.refused:
            return
        try:
            from sglang.srt.utils.lmem_census import census_max  # noqa: PLC0415
            from sglang.srt.weg2.sleep_lmem import CudaDriverStackLimit, restore_lmem  # noqa: PLC0415

            # H101: the largest LOCAL_SIZE of a kernel this process loaded is a
            # booked need -- restored here, never regrown inside a launch.
            booked, booked_kernel = census_max()
            rec = restore_lmem(
                driver=CudaDriverStackLimit(), park=park, nvml_bytes=self._weg2_nvml_self_bytes,
                booked_stack_bytes=booked or None, booked_kernel=booked_kernel,
            )
        except Exception as exc:  # noqa: BLE001 -- the driver grows it on demand
            logger.warning("WEG2-WAKE-LMEM n/a (%s: %s)", type(exc).__name__, str(exc)[:160])
            return
        (logger.warning if rec.refused else logger.info)("%s", rec.format_line(park=park))
        if rec.skip_line():
            logger.warning("%s", rec.skip_line())

    def _weg2_residue_posts(self) -> str:
        """H15: the per-post split of the sleep residue that is readable in-process.
        Never raises: a failed read prints n/a, the residue line stays."""
        try:
            from sglang.srt.distributed.device_communicators.barlink_matrix_transport import (  # noqa: PLC0415
                ledger_balance,
            )
            from sglang.srt.weg2.bar1_lanes import Bar1Lanes  # noqa: PLC0415
            from sglang.srt.weg2.sleep_lmem import format_residue_posts  # noqa: PLC0415

            lanes = self._weg2_bar1
            lane_windows = []
            if isinstance(lanes, Bar1Lanes):
                lane_windows = [
                    (k, int(w.size)) for k, w in sorted(lanes.recv.items()) if not w.borrowed
                ]
            return format_residue_posts(
                park=self._weg2_lmem_park,
                group_windows=ledger_balance(torch.cuda.current_device()),
                lane_windows=lane_windows,
            )
        except Exception as exc:  # noqa: BLE001 -- an instrument never kills the sleep
            return f"posts=n/a({type(exc).__name__}: {str(exc)[:80]})"

    def _weg2_trim_host_heap_at_sleep(self) -> None:
        """weg2xsn297 (Nutzer-Order 18.09.: Scheduler-Heap -- bauen, verdrahten,
        Standard). Each rank held ~2.5 GiB of glibc heap while serving
        (/proc/<pid>/smaps [heap], xsn296 desk reading); a sleeping rank is the
        moment to hand freed arenas back to the kernel (malloc_trim(0)) and
        to say what the heap is made of (a gc census by type on sleeps
        1, 2, 4, 8, ..., top 8). H16: the census runs AFTER the sleep answered
        unless SGLANG_WEG2_SLEEP_HEAP_CENSUS=2 (weg2/heap_census.py has the
        fnFL2x127 tails it took off the flip's critical path). Never raises."""
        try:
            from sglang.srt.environ import envs  # noqa: PLC0415
            from sglang.srt.weg2 import heap_census  # noqa: PLC0415

            before = self._weg2_rss_anon()
            trimmed = self._weg2_malloc_trim()
            after = self._weg2_rss_anon()
            n = int(self._weg2_sleep_count or 0)
            action = heap_census.census_action(
                mode=envs.SGLANG_WEG2_SLEEP_HEAP_CENSUS.get(), sleep_count=n
            )
            census = self._weg2_census_now(heap_census=heap_census, action=action, n=n)
            logger.info(
                "WEG2-SLEEP-HOST-HEAP rss_anon=%d MiB -> %d MiB after malloc_trim(0) (returned=%d MiB, "
                "trim_rc=%d) census_mode=%s%s (instrument: /proc/self/status RssAnon; census = "
                "sys.getsizeof over gc-tracked objects by type, shallow sizes, top 8 -- containers' "
                "payloads not included)",
                before >> 20, after >> 20, max(0, before - after) >> 20, trimmed, action, census,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("WEG2-SLEEP-HOST-HEAP instrument raised (%s: %s)", type(exc).__name__, str(exc)[:160])

    @staticmethod
    def _weg2_rss_anon() -> int:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        return -1

    @staticmethod
    def _weg2_malloc_trim() -> int:
        try:
            import ctypes  # noqa: PLC0415

            return int(ctypes.CDLL("libc.so.6").malloc_trim(0))
        except Exception:  # noqa: BLE001
            return -1

    @staticmethod
    def _weg2_census_now(*, heap_census: Any, action: str, n: int) -> str:
        """The census suffix of the trim line: inline text, or the deferral
        (the timer logs WEG2-SLEEP-HOST-HEAP-CENSUS itself)."""
        if action == "inline":
            try:
                text, ms = heap_census.timed_census()
                return f" census_ms={ms:.0f} census={text}"
            except Exception as exc:  # noqa: BLE001
                return f" census=n/a({type(exc).__name__})"
        if action == "defer":
            heap_census.defer_census(sleep_count=n, log=logger.info)
            return f" census=deferred({heap_census.DEFER_S:.0f}s)"
        return ""

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
        #
        # #1369 CORRECTION (DESK12's find, this line, 2026-09-14): this used
        # to read the RAW `server_args.enable_weights_cpu_backup` -- the
        # argv bit the launcher sets UNCONDITIONALLY (launcher.py:2702) and
        # therefore always True, regardless of arm.  Once #1369's own knob
        # (`weight_exchange.weights_cpu_backup_armed()`) gates the PHYSICAL
        # backup at `model_runner.py`'s `pause()` (Paket C, not this file),
        # the raw bit and the true backup state diverge in exactly the two
        # cases the completeness refusal (W106, weg2_memory_saver.py) cannot
        # reach on its own: (a) `--weg2-weights-cpu-backup off` -- W106 never
        # runs for a tag whose exchange is not armed at all, because
        # `_weg2_xchg_deposit_before_sleep` returns immediately for an
        # unarmed exchange (:1080-1081) -- ring mode was never its half to
        # guard; (b) `exchange`+`shadow`+`off`, which W106 DOES refuse at the
        # sleep leg, so this second case never reaches a wake here at all,
        # but the read must still be correct for defense in depth. Reading
        # the RAW bit here would choose `CARRIER_TMS_BACKUP` -- "the TMS
        # restore already carried the bytes" -- for a tag nothing backed up,
        # and `_weg2_xchg_inject_from_peer`'s TMS branch does nothing: the
        # remapped pages stay on their post-`resume()` undefined content,
        # silently, with no exception and no distinguishing log line except
        # weights that read wrong. Reading the PREDICATE instead makes the
        # fallback fall all the way through to `CARRIER_DISK` (:993) for
        # exactly the tags case (a) covers -- the checkpoint on disk, the
        # user's own "das ist das Netz" (2026-09-14), not an undefined page.
        # An unreadable predicate (the function raising, e.g. an env this
        # process never saw published) keeps the PRE-#1369 raw-flag answer:
        # the conservative direction here is the one every boot before
        # tonight already ran, not a guess about a contract that failed to
        # answer.
        try:
            from sglang.srt.weg2.weight_exchange import weights_cpu_backup_armed

            main_carried = bool(weights_cpu_backup_armed())
        except Exception:  # noqa: BLE001 -- see comment above: fall back, don't guess
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
            # weg2xsn264 (17.09., DFLASH form): a SEPARATE draft checkpoint is
            # legitimate when the EXCHANGE carries the draft -- the draft
            # runner's region `weights_draft` is a member of the weights
            # family (`draft_tag_in_family`), its bytes come from the peer
            # group's live VRAM through the same legs as every other tag, and
            # `_weg2_xchg_draft_reload_from_disk` reloads from the DRAFT's own
            # path if the legs carried nothing. The "one model_path" objection
            # below is about the disk refill and does not apply. Measured:
            # all three D ranks refused W4 at the first P->D wake with the
            # DFlash2 draft (`Qwen3.8-27B-DFlash2-W8-lued`) beside the target.
            _draft_via_exchange = False
            try:
                from sglang.srt.managers.weg2_memory_saver import draft_tag_in_family as _dtf
                from sglang.srt.weg2 import weight_exchange as _wx

                _draft_via_exchange = (bool(_wx.exchange_armed())
                                       and bool(_wx.inject_authoritative())
                                       and bool(_dtf()))
            except Exception:  # noqa: BLE001 -- unreadable arm: keep the refusal
                _draft_via_exchange = False
            # H25d (fnFL2x142, 24.09.): the draft's bytes may also come from
            # its OWN pinned host image. Under SGLANG_WEG2_DRAFT_ON_P=0 the
            # draft tag is out of the family (no partner on P), so the clause
            # above is False, and x142 died here on all three D ranks at the
            # first P->D wake -- AFTER `WEG2-DRAFT-PARK ... credited=yes` on
            # TP0 and BEFORE `_weg2_unpark_draft_start` could copy the image
            # back. Nothing is refilled from disk on that path; the objection
            # below does not apply. Keyed on the image EXISTING on this rank
            # (a park happened), never on the arm alone: a rank that never
            # parked has no host source and keeps the refusal.
            _draft_via_park = self._weg2_draft_carried_by_park()
            # H25e (fnFL2x143, 24.09.): THE THIRD SOURCE -- none needed. A
            # Form-A worker's draft is a shadow whose tag holds no byte on the
            # card; its sleep logged 'WEG2-DRAFT-PARK ... nothing to park on
            # this rank (storages=0 tms_bytes=0)' and x143's TP1/TP2 then died
            # here (W4) at the first P->D wake, while TP0 (parked) passed. A
            # rank whose draft tag carries nothing has nothing to refill; W4
            # guards nothing there. The verdict is the park's own, recorded at
            # the sleep (`_weg2_draft_not_carried`), never re-guessed here.
            _draft_not_carried = self._weg2_draft_not_carried is not None
            if (draft_path is not None and draft_path != server_args.model_path
                    and not _draft_via_exchange and not _draft_via_park
                    and not _draft_not_carried):
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

    def _weg2_xchg_inject_weights(self, *, tag=None, **kw) -> bool:
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
        # #73 (fnFL2v97): ZWEI PRAEDIKATE FUER EINE FRAGE -- und nur eines
        # davon stimmte.
        #
        # Der Launcher publiziert die Bounce-Terme unter
        # `xchg_bounce_arm_pins_host(weight_source, oncard_mode)`: BEIDE Arme
        # entscheiden, denn `--weg2-xchg-oncard ipc` exportiert eine
        # VRAM-Bounce, die mit ihrem eigenen Leg stirbt -- "no bounce FILE
        # exists and no host byte is pinned" (launcher.py, Docstring). Er
        # publiziert dort also ZU RECHT nichts.
        #
        # Diese Pruefung fragte nur nach der QUELLE und verweigerte jeden
        # exchange-Wake ohne Terme -- auch den, bei dem es per Konstruktion
        # keine Host-Bounce-Geometrie zu publizieren GIBT. fnFL2v97 erreichte
        # beide Gruppen READY, den ersten Flip und starb hier: W4 auf allen
        # drei P-Raengen, danach W29 (rank disagree) und ein
        # `cudaErrorInvalidValue` aus `MemPool::~MemPool()` unter
        # `_PyModule_Clear` -- der Shutdown-Folgeschaden, den man leicht fuer
        # die Wurzel haelt.
        #
        # Die Leg ist fuer `terms=None` GEBAUT, nicht bloss tolerant dagegen:
        # `_weg2_xchg_bounce_leg` fuehrt `terms=None` in der Signatur und
        # prueft `if terms is not None` an jeder Stelle, die davon liest.
        #
        # EXPLIZIT 'ipc', nie "fehlt": eine fehlende Variable heisst
        # "unbekannter Arm", und der bleibt eine Verweigerung. Nur der Arm,
        # der sich als ipc AUSWEIST, hat nachweislich keinen Host-Bounce.
        if terms is None:
            from sglang.srt.weg2 import (
                weight_exchange_transport as _wt,
            )

            _oncard = (os.environ.get(_wt.ENV_ONCARD_MODE, "") or "").strip()
            if _oncard == _wt.ONCARD_MODE_IPC:
                logger.info(
                    "WEG2-XCHG-BOUNCE TERMS ABSENT BY ARM oncard=%s -- this arm "
                    "pins no host byte, so the launcher priced no bounce "
                    "geometry and there is none to read. The leg assembles "
                    "from the exported VRAM bounce; `terms` stays None, which "
                    "it is built for (#73).", _oncard)
            else:
                raise Weg2WakeRefused(
                    "W4 Weg2WakeRefused: --weg2-weight-source exchange owns "
                    f"this wake's weight bytes, but {xb.ENV_BOUNCE_TERMS} was "
                    "not published, so the bounce geometry the injection "
                    "needs was never priced by the launcher. Refusing rather "
                    "than sizing a pinned host buffer locally: the resume has "
                    "already remapped the weight pages and their content is "
                    "undefined, so serving is not an option either. Launch "
                    "through the weg2 launcher, which publishes the term it "
                    f"charged on the ARM line. (Arm here: oncard="
                    f"{_oncard!r} -- only an explicit "
                    f"{_wt.ONCARD_MODE_IPC!r} pins no host byte and may go "
                    "without terms, #73.)"
                )
        return self._weg2_xchg_inject_from_peer(terms=terms, tag=tag, **kw)

    def _weg2_xchg_deposit_before_sleep(self, *, flip_index: int = -1,
                                        tag: Optional[str] = None) -> None:
        """THE DEPOSIT HALF -- the group going dormant stages its card bytes.

        WEG2XSN25 MEASURED THE HOLE: `SEAM-DIGEST MATCH 0/6`, and the cause was
        not a phase bug but a MISSING CALLER. There was exactly one production
        call site (`_weg2_xchg_inject_from_peer`), it runs on the WAKING group,
        and it hard-coded `hook="authoritative"` -- which `leg_direction`
        derives as `collect`. So every rank that ran a leg collected, nobody
        deposited, and the collect legs waited on a band no one would post:
        `carries 0 bytes` against a derivation of 756323776. D emitted zero
        `WEG2-XCHG-INJECT` lines all boot; it never ran a leg at all.

        THE ROLE IS NOT A NEW DERIVATION. `weight_exchange.leg_direction`
        (:2223) and `weight_exchange_shadow.py:3345` already answer it from the
        hook, and this method's only contribution is to BE the source hook:
        the sleeping rank holds the bytes, so it exports. The phase falls out
        (`_weg2_xchg_bounce_leg` -> `PHASE_DEPOSIT`) and the INJECT line prints
        it.

        WHERE IT IS CALLED MATTERS MORE THAN WHAT IT DOES: before
        `memory_saver_adapter.pause(tag)` releases the weight pages. A deposit
        after the pause would read pages this rank no longer owns.

        An unarmed boot returns immediately -- this is the exchange's half, not
        the ring's.
        """
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2 import weight_exchange_shadow as sh
        from sglang.srt.weg2 import xchg_bounce as xb

        if not wx.exchange_armed():
            return
        boot_nonce = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
        group = self._weg2_group_name()
        rank = self._weg2_rank()
        device = self._weg2_device_index()
        if not boot_nonce or not group or rank is None or int(device) < 0:
            # NAMED, NOT SILENT: this is the class weg2xsn25 paid a boot for.
            logger.info(
                "WEG2-XCHG-DEPOSIT-SKIPPED boot=%r group=%r rank=%r device=%s "
                "-- the dormant group could not identify itself, so no bytes "
                "were staged; the waking group's collect will have nothing to "
                "read and will say so", boot_nonce, group, rank, device)
            return
        # #1368: ONLY THE SOURCE GROUP OF THIS FLIP'S DIRECTION DEPOSITS.
        # This method fired on EVERY sleeping rank of BOTH groups, while the
        # collect half runs only on the group being woken -- boot weg2xsn27
        # counted 22 deposit-hook legs against 6 collect legs, so most bands
        # had no counterpart: D's next deposit found `slot=0 seq=0 was still
        # full ... the collecting rank has not drained` (x9) while P's collect
        # found the same slot `was not posted full by any depositing rank`
        # (x2). The rendezvous classes are not at fault -- a hermetic probe
        # with both sides on one pair drains cleanly.
        #
        # NO NEW STATE: `leg_enabled` is the arm's own predicate over
        # `leg_direction`, the same derivation the transport's `is_source`
        # uses. The caller had keyed on "I am going to sleep", which is a
        # property of this rank and not of the flip.
        # #1368 FIX 2 -- THE CONDITION IS "IS THERE A FLIP", and the boot said
        # so in its own words. My first gate asked `leg_enabled(HOOK_SOURCE,
        # group)`, which is the LEGS CONFIGURATION (may this group ever be a
        # source) and not the direction of THIS flip: under `legs=both` -- the
        # default, which xsn27 ran and xsn28 runs -- it is true for both
        # groups and separates nothing.
        #
        # BUT THE REPLACEMENT IS NOT "source_of(direction)" EITHER, and the
        # evidence is weg2xsn27's own W79 on P:
        #   `reason=no-flip-epoch detail=hook=source -- this leg carries no
        #    flip epoch, so it is not a flip: the front publishes the index on
        #    the request (front.py:2667) and the boot-time initial sleep runs
        #    before any flip exists.`
        # Within a real flip exactly ONE group sleeps, and the sleeping group
        # IS the source -- this method runs on the sleeper by construction, so
        # a direction predicate derived from the group would be circular. What
        # was missing is that the BOOT-TIME INITIAL SLEEP is not a flip at all
        # and must not deposit: it claims a slot before any collect exists.
        #
        # `-1` is the front's own "no flip" (`_weg2_flip_index_of`), the same
        # number `XchgRegion.begin_flip` refuses to stamp with.
        if int(flip_index) < 0:
            logger.info(
                "WEG2-XCHG DEPOSIT skipped role=no-flip direction=%s "
                "group=%s rank=%s flip_index=%s -- this sleep carries no flip "
                "epoch (the boot-time initial sleep runs before any flip "
                "exists), so a deposit here would claim a slot no collect "
                "will ever drain",
                wx.xchg_legs(), group, rank, flip_index)
            return
        # THE CONFIGURATION GATE STAYS, and only as that: whether this
        # direction is exchanged at all on this boot.
        if not wx.leg_enabled(sh.HOOK_SOURCE, group):
            logger.info(
                "WEG2-XCHG DEPOSIT skipped role=direction-not-armed "
                "direction=%s group=%s rank=%s -- this boot does not exchange "
                "in the direction this group would source",
                wx.xchg_legs(), group, rank)
            return
        plan, reason = self._weg2_shadow_plan(
            sh.HOOK_SOURCE, group, int(rank), agreed=None,
            require_agreement=False)
        if plan is None:
            # fnFL2x100: NO PLAN IS NOT "NOTHING TO DEPOSIT". A refused join
            # (W74: a tensor one group publishes and the other does not) ends
            # here on EVERY rank of the sleeping group, and the pause right
            # after this return releases the tag's bytes with nobody holding
            # them. With the ring off and the exchange authoritative that is
            # the W106 gap exactly -- the same predicate, asked with the plan
            # refusal as its reason, before VRAM is mutated for this tag.
            if tag is not None:
                _gap = self._weg2_xchg_wake_source_gap(
                    tag, cdescs_present=False,
                    resident_bytes=self._weg2_tag_resident_bytes(tag))
                if _gap is not None:
                    _expected_bytes = self._weg2_tag_bytes(tag)
                    raise Weg2XchgWakeSourceGapRefused(
                        f"W106 Weg2XchgWakeSourceGapRefused: group={group} "
                        f"rank={rank} tag={tag} expected_bytes="
                        f"{_expected_bytes if _expected_bytes > 0 else 'unmeasurable'}"
                        f": this rank has NO exchange plan at all ({reason}) "
                        f"-- {_gap}")
            logger.info(
                "WEG2-XCHG-DEPOSIT-SKIPPED group=%s rank=%s no plan: %s",
                group, rank, reason)
            return
        # THE ARM'S OWN DECISION, THE SAME ONE P GETS. This passed `terms=None`
        # and killed boot weg2xsn26: `run_bounce_leg` derives no size of its
        # own (the sizing expression has ONE owner), so D's sleep leg raised
        # `ValueError: ... needs either terms ... or an explicit
        # slot_bytes/depth pair` nine times, took the group fence down through
        # `_weg2_leg_failed` and ended the boot at W17 Weg2GroupDead with
        # 0 legs and 0 SEAM-DIGEST lines.
        #
        # I WROTE `terms=None` BELIEVING THE LEG WOULD DERIVE THEM. The
        # docstring's "the normal way in" describes the terms/slot_bytes PAIR,
        # not a fallback -- and the call site that proves it is the one I did
        # not exercise. P reaches `read_published_terms` at :997 and passes the
        # result; D now reads the SAME publication, so both halves of one flip
        # are sized by one decision.
        terms = xb.read_published_terms()
        if terms is None:
            logger.info(
                "WEG2-XCHG-DEPOSIT-SKIPPED group=%s rank=%s: the arm published "
                "no bounce terms (%s), so this rank cannot size a deposit; the "
                "waking group's collect will find nothing and say so",
                group, rank, xb.ENV_BOUNCE_TERMS)
            return
        # #1374 F1 PER-TAG LOCKSTEP. `tag` restricts this deposit to ONE
        # weight tag, because the caller now runs it INSIDE the pause loop:
        # deposit(t) -> pause(t) -> credit(t) -> deposit(t+1). Before #1374 the
        # whole plan was deposited before the first pause, and on the
        # co-located card that cannot complete -- the collector needs the
        # credit, the credit needs the pause, the pause needs this deposit
        # (weg2xsn30: D0 W68 and PP0 W35, both after a full 120 s).
        #
        # `None` keeps the whole-plan behaviour for any caller that is not the
        # pause loop; the descs carry their own tag (`XchgDesc.tag`), so the
        # filter is a property of the plan and not a second bookkeeping.
        _descs = list(plan.descs)
        if tag is not None:
            _descs = [d for d in _descs if str(getattr(d, "tag", "")) == str(tag)]
            # #1369/#1394 (DESK10 Paket B): THE VOLLSTAENDIGKEITS-REFUSAL.
            # Checked HERE, on the SOURCE/sleep side, before the pause that
            # would make a gap real: once this tag's ring backup is off,
            # this deposit (or its structural absence) is the LAST chance to
            # say "nobody will restore this tag's bytes at wake" while VRAM
            # has not yet been mutated for it.
            # weg2xsn83 (#1378, leg 1 = P->D, the first time a PIPELINE
            # group ever reached this check as the SOURCE): the pause loop
            # walks the whole family (weights_0..7, weights_draft, weights),
            # but a PP stage holds only ITS layers' tags -- PP0 met
            # weights_6 first, had no descriptor for it (correct: PP2 owns
            # it), and W106 killed the leg before any deposit. The tag's
            # residency on THIS rank, by the saver's own census, is what
            # tells that no-op apart from the real gap the check exists for.
            _resident = self._weg2_tag_resident_bytes(tag)
            _gap = self._weg2_xchg_wake_source_gap(
                tag, cdescs_present=bool(_descs), resident_bytes=_resident)
            if _gap is not None:
                # NUTZER-ORDER 2026-09-14 (PRONTO, den Ring abschalten):
                # "jeder pausierte Tag, der beim resume KEINE Quelle hat,
                # muss den Boot mit NAMEN UND BYTE-ZAHL toeten" -- `tag` was
                # already the name; this adds the count. `_weg2_tag_bytes`
                # is the file's own established instrument for it (C7/C16,
                # `tms_tag_bytes`, already read at this exact call site's
                # caller for the credit publish two lines below the pause
                # this deposit precedes) -- not a new derivation, the SAME
                # number the sleep leg is about to release. A `0` here is
                # NAMED as unmeasurable rather than printed as a real zero
                # (NULL-NUR-BEI-ERREICHTEM-EMITTER): a refusal that could
                # itself lie about a byte count would be exactly the "still
                # wrong text" class this whole ticket exists to end.
                _expected_bytes = self._weg2_tag_bytes(tag)
                _bytes_text = (f"{_expected_bytes}" if _expected_bytes > 0
                              else "unmeasurable (tms_tag_bytes answered 0 "
                                   "or nothing -- see _weg2_tag_bytes)")
                raise Weg2XchgWakeSourceGapRefused(
                    f"W106 Weg2XchgWakeSourceGapRefused: group={group} "
                    f"rank={rank} tag={tag} expected_bytes={_bytes_text}: "
                    f"{_gap}"
                )
            if not _descs:
                # Not an error BEYOND the check above: a tag this rank does
                # not carry has nothing to deposit, and saying so keeps the
                # per-tag census honest -- reachable only because the gap
                # check just cleared it (ring still on, or the exchange is
                # authoritative and genuinely has nothing FOR THIS RANK,
                # which is fine as long as SOME rank does).
                logger.info(
                    "WEG2-XCHG DEPOSIT tag=%s group=%s rank=%s pieces=0 "
                    "resident_bytes=%s -- this rank's plan carries no desc "
                    "for this tag, so the lockstep step is a no-op rather "
                    "than a missing band (resident_bytes=0 by the saver's own "
                    "census: the tag lives on another stage of this group)",
                    tag, group, rank,
                    "unmeasurable" if _resident is None else int(_resident))
                return
        self._weg2_xchg_bounce_leg(
            descs=_descs, ops=self._weg2_xchg_device_ops(),
            boot_nonce=boot_nonce, terms=terms, mode=wx.inject_mode(),
            device=int(device), hook=sh.HOOK_SOURCE,
            region=self._weg2_shadow_region(),
            sems=self._weg2_xchg_sems(), tag=tag, rank=int(rank),
        )

    def _weg2_xchg_wake_source_gap(self, tag, *, cdescs_present: bool,
                                   resident_bytes: Optional[int] = None,
                                   ) -> Optional[str]:
        """``None`` if ``tag``'s wake, WITHOUT its ring backup, has something
        that will actually restore its bytes -- a short GAP REASON string
        otherwise.

        ``resident_bytes`` is the saver's three-valued census of ``tag`` on
        THIS rank (:meth:`_weg2_tag_resident_bytes`): a real ``0`` with an
        empty plan is NOT a gap -- the tag lives on another stage of this
        group, which deposits it from its own pause loop (weg2xsn83, PP0
        asked about PP2's ``weights_6``).  ``None`` (unmeasurable) keeps
        case 2's refusal: an absence nobody measured vouches for nothing.
        A POSITIVE count with an empty plan is case 2 exactly -- bytes here,
        nobody deposits them.

        #1369/#1394 (DESK10 Paket B), the completeness half of turning the
        host ring off: with the ring gone, the exchange is the ONLY source
        of a tag's wake bytes, so a gap here is not a missing optimisation,
        it is undefined weights served with every counter green. Two
        independent ways a gap can exist, checked in order (either one is
        enough to refuse):

        1. THE TAG'S RING BACKUP IS OFF, BUT THE EXCHANGE IS NOT
           AUTHORITATIVE. ``--weg2-xchg-inject shadow`` compares, it never
           writes (#1391's own lesson: ``_weg2_xchg_shadow_compare`` and the
           per-tag shadow branch beside it are graders, not movers -- see
           ``_weg2_xchg_inject_from_peer``'s ``mode`` argument, which the
           shadow path sets to ``INJECT_SHADOW`` explicitly). A tag with the
           ring off and nothing authoritative writing it has NO source at
           all, independent of how complete its plan is: this check fires
           even when ``cdescs_present`` is True, because a perfect plan
           under ``shadow`` still writes nothing.
        2. THE EXCHANGE IS AUTHORITATIVE, BUT THIS RANK'S OWN PLAN IS EMPTY
           FOR THIS TAG. ``weights_draft`` is EXEMPTED from this one case
           only: #1394's fix gives it a dedicated disk-reload wake path
           (:meth:`_weg2_xchg_draft_reload_from_disk`) instead of the
           exchange, because ``_weg2_shadow_plan`` structurally never
           selects the draft runner's region (it reads
           ``self.tp_worker.model_runner`` alone, on every call, regardless
           of tag) -- an empty plan for ``weights_draft`` is therefore the
           EXPECTED shape, not a gap, as long as the reload path is reachable
           (``self.draft_worker is not None``). It is NOT exempted from
           needing case 1's authoritative writer.

        ``weights_cpu_backup_armed`` unreadable (the contract not yet on
        this tree, or an env this process cannot parse) answers ``None``
        rather than manufacturing a refusal from an absence -- the same
        rule :meth:`_weg2_xchg_undrained_lanes` follows for an unopened
        semaphore set.
        """
        from sglang.srt.managers.weg2_memory_saver import (
            GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
        )
        from sglang.srt.weg2 import weight_exchange as wx

        try:
            from sglang.srt.weg2.weight_exchange import weights_cpu_backup_armed

            if weights_cpu_backup_armed():
                return None
        except Exception:  # noqa: BLE001 -- an unreadable contract is not a gap
            return None
        # The ring is off for this tag. The exchange has to be the writer.
        if not (wx.exchange_armed() and wx.inject_authoritative()):
            return (
                f"weights_cpu_backup_armed()=False for this tag and the "
                f"exchange is not authoritative (exchange_armed="
                f"{wx.exchange_armed()} inject_authoritative="
                f"{wx.inject_authoritative()}) -- shadow mode only COMPARES "
                f"the exchange's assembled bytes, it never writes them, so "
                f"nothing would restore this tag at wake"
            )
        if str(tag) == GPU_MEMORY_TYPE_WEIGHTS_DRAFT:
            if self.draft_worker is None:
                return None
            _drafter = _weg2_drafter_of(self)
            if getattr(_drafter, "is_draft_solo_shadow", False):
                # 19.09. (xsn390): a solo-draft SHADOW holds no draft bytes
                # by construction (a meta draft, the saver has no allocation
                # under this tag), so an empty plan for it is the whole
                # truth, not a gap -- the host restores its own draft
                # through the exchange's draft leg.
                return None
            if cdescs_present:
                return None  # the real join covers this tag on this leg
            if resident_bytes is not None and int(resident_bytes) == 0:
                # fnFL2x10: a META SHADOW (Form A's expert workers under
                # --speculative-draft-placement solo) -- the allocator holds
                # no byte of this tag here, so nothing is released and
                # nothing needs a source at wake. x10's D TP1/TP2 refused
                # W106 on exactly this; the real holder (TP0) carries the
                # descriptors and runs this check with them.
                return None
            # NUTZER-ORDER 2026-09-14 (A2, xsn32, W4 Weg2WakeRefused at
            # weg2_memory_saver.py:368): the exemption below used to answer
            # "covered by _weg2_xchg_draft_reload_from_disk" UNCONDITIONALLY
            # -- an ASSUMPTION that disk is always a safe net, which #1394's
            # own doctrine ("der Checkpoint ist das Netz") never actually
            # verified against THIS process's checkpoint. It is not: on a
            # quantized checkpoint, `update_weights_from_disk` writes into
            # parameters `process_weights_after_loading` already REPLACED
            # (`assert_backup_off_wake_refill_is_defined`'s own W4 reason,
            # weg2_memory_saver.py:328-382) -- the SAME undefined operation
            # the launch-time guard exists to name, reached here via a
            # DIFFERENT caller it does not know about. "Vollstaendigkeit als
            # BEDINGUNG, nicht als Annahme": the exemption now checks
            # whether the fallback it names is ACTUALLY safe before
            # granting it, and refuses BY NAME (still W106, the tag's own
            # completeness refusal -- not a new code) when it is not.
            _draft_quant = self._weg2_draft_checkpoint_quantization()
            if _draft_quant:
                return (
                    f"weights_cpu_backup_armed()=False for this tag, the "
                    f"exchange's own join carries zero descriptors for it "
                    f"on this leg, AND the disk-reload fallback (#1394, "
                    f"_weg2_xchg_draft_reload_from_disk) is undefined on "
                    f"this {_draft_quant!r} checkpoint -- "
                    f"update_weights_from_disk would write into parameters "
                    f"process_weights_after_loading already replaced (W4 "
                    f"Weg2WakeRefused's own reason, "
                    f"weg2_memory_saver.py:328-382). No third net exists "
                    f"for this tag on this checkpoint"
                )
            return None  # unquantized checkpoint: the disk reload is a genuine net
        if not cdescs_present and resident_bytes is not None and int(resident_bytes) == 0:
            # A PIPELINE stage's view of a family tag it does not hold: the
            # allocator has no bytes of it here, so there is nothing a peer
            # could miss FROM THIS RANK. The stage that holds the tag runs
            # this same check with its own descriptors (weg2xsn83 PP2 for
            # weights_6/weights_7) and is the one that would refuse.
            return None
        if not cdescs_present and self._weg2_tag_live_tensor_count(tag) == 0:
            # fnFL2x11: Form A's expert workers hold 2 MiB (one allocator
            # granule) under the BASE tag and not one tensor of it -- the
            # boot coverage printed no row for it. Segment slack is carried
            # by the exchange on no rank; the check exists for TENSORS whose
            # bytes nobody deposits, and the walk that fed the coverage
            # vote finds none here.
            return None
        if not cdescs_present:
            return (
                "weights_cpu_backup_armed()=False for this tag and the "
                "exchange is authoritative, but this rank's own plan "
                "carries zero descriptors for it on the source side -- "
                "nobody will deposit these bytes for the peer to collect "
                "at wake"
            )
        return None

    def _weg2_tag_live_tensor_count(self, tag) -> Optional[int]:
        """How many live tensors of ``tag`` the MAIN model holds on this rank,
        by the boot coverage's own walk (parameters, buffers, attribute
        tensors); ``None`` when unreadable -- the caller then keeps its
        refusal."""
        try:
            from sglang.srt.weg2 import weight_exchange as wx

            model = self.tp_worker.model_runner.model
            if model is None:
                return None
            return sum(1 for t in wx.walk_live_tensors(model)
                       if str(t.tag) == str(tag))
        except BaseException:  # noqa: BLE001 -- unreadable is not "none"
            return None

    def _weg2_draft_checkpoint_quantization(self) -> Optional[str]:
        """The quantization ACTUALLY IN FORCE for the draft checkpoint
        ``_weg2_xchg_draft_reload_from_disk`` would reload -- the same
        question ``checkpoint_quantization`` (weg2_memory_saver.py) already
        answers for the main shard at weight_updater.py:1753, asked here
        for the runner ``update_weights_from_disk`` actually targets in
        THIS method rather than the main one.

        The draft runner's OWN ``model_config`` is read first (a draft
        checkpoint may be a genuinely different file with its own
        quantization scheme, per ``_weg2_xchg_draft_reload_from_disk``'s
        own ``speculative_draft_model_path`` fallback), ``server_args``
        second -- the identical two-holder order
        ``checkpoint_quantization`` itself already defines, so this is not
        a second reading of that decision, only a different runner's view
        of it.

        ``None`` (never a manufactured answer) when neither the draft
        runner nor server_args is reachable: an unreadable checkpoint
        identity is not evidence of an UNQUANTIZED one, and the disk-reload
        exemption above already treats ``None`` as "no gap" -- the same
        conservative direction :meth:`_weg2_xchg_wake_source_gap` takes for
        every other unreadable contract in this method.
        """
        server_args = self._weg2_server_args()
        drafter = _weg2_drafter_of(self)
        draft_model_config = getattr(drafter, "model_config", None)
        return checkpoint_quantization(draft_model_config, server_args)

    def _weg2_xchg_draft_reload_from_disk(self) -> bool:
        """``weights_draft``'s FALLBACK wake source, tried only when the
        real exchange has nothing.

        SUPERSEDED 2026-09-14, PARTIALLY (user order, "draft ist auch nur
        ein layer... warum muss er ueber den ring gehen?"): #1394's own
        first cut treated the exchange as STRUCTURALLY unable to cover this
        tag at all, because ``_weg2_shadow_plan`` resolved ``region_tag``
        from ``self.tp_worker.model_runner`` alone -- the MAIN runner --
        on EVERY call (verified #1391 round 2, executed). That is fixed
        now: ``_weg2_shadow_plan`` additionally asks
        :meth:`_weg2_xchg_draft_plan_or_none` for the draft runner's OWN
        region, through the SAME join every other tag already uses, and
        unions the descriptors -- see that method's docstring for the
        (a)/(b) population finding (re-materialised embed/head shards vs.
        the real MTP layer) that makes a genuine, all-or-nothing join
        failure a REAL possibility on some boots, not a defensive default.

        THIS METHOD IS THEREFORE NO LONGER THE FIRST THING TRIED for
        ``weights_draft`` -- the caller (``resume_memory_occupation``'s
        per-tag loop) asks the exchange FIRST, exactly like any other tag,
        and calls this ONLY when that leg's own collect found ZERO
        descriptors for the tag. With the host ring OFF
        (``weights_cpu_backup_armed()=False``) and the exchange genuinely
        empty for this tag, ``resume`` has only remapped pages -- their
        content is undefined until something writes them -- so this reload
        straight from the checkpoint on disk, through the SAME per-shard
        call ``update_weights_from_disk`` already uses for the draft
        worker independently of the main one, is what makes the ring
        removal safe for a boot where the draft join genuinely could not
        be built -- ON AN UNQUANTIZED CHECKPOINT ONLY (never the ring
        again, never a mini-ring).

        CORRECTED 2026-09-14 (A2, xsn32, coordinator relay of a real boot
        death, W4 Weg2WakeRefused at weg2_memory_saver.py:368): the
        sentence above used to end at "be built", stated as an
        unconditional safety net. It is not one on a quantized checkpoint
        -- ``update_weights_from_disk`` is the SAME operation
        :func:`weg2_memory_saver.assert_backup_off_wake_refill_is_defined`
        already names undefined there (``process_weights_after_loading``
        replaced the parameters this reload would write into), reached
        here through a caller that guard did not know about. This method
        now asks that SAME guard, with the DRAFT checkpoint's own
        quantization (:meth:`_weg2_draft_checkpoint_quantization`), before
        touching anything -- so a quantized checkpoint refuses BY NAME
        here (W4) instead of committing undefined VRAM, exactly the
        completeness gate :meth:`_weg2_xchg_wake_source_gap` already
        applies one level higher, at the sleep leg, for every flip except
        the boot-time initial one.

        Returns ``True`` when it did the reload, ``False`` when there is
        nothing to reload (no draft shard in this process, or the ring
        still covers it) -- never silently both-or-neither.
        """
        if self.draft_worker is None:
            return False
        from sglang.srt.managers.weg2_memory_saver import (
            GPU_MEMORY_TYPE_WEIGHTS_DRAFT as _DRAFT_TAG,
        )

        if getattr(_weg2_drafter_of(self), "is_draft_solo_shadow", False):
            # 19.09. (xsn391): a solo-draft SHADOW holds no draft bytes at
            # all (meta draft) -- nothing to refill, and the refill's own
            # quantization gate (W4 on compressed-tensors) must not fire for
            # bytes that never existed; the host restores its draft through
            # the exchange's draft leg.
            return False
        if self._weg2_tag_resident_bytes(_DRAFT_TAG) == 0:
            # fnFL2x10: the meta shadow holds no byte of the draft tag -- its
            # empty collect is complete, and a disk reload into a meta
            # drafter would be W4 on a quantized checkpoint for nothing.
            logger.info("WEG2-XCHG-DRAFT-RELOAD-SKIPPED tag=%s: tms_tag_bytes=0 "
                        "on this rank (meta shadow) -- nothing to reload",
                        _DRAFT_TAG)
            return False
        try:
            from sglang.srt.weg2.weight_exchange import weights_cpu_backup_armed

            if weights_cpu_backup_armed():
                return False  # the ring already covers this tag
        except Exception:  # noqa: BLE001 -- an unreadable contract changes nothing
            return False
        server_args = self._weg2_server_args()
        if server_args is None:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: weights_draft's disk-reload wake path "
                "needs server_args to name the checkpoint and none is "
                "reachable from the weight updater. VRAM has already been "
                "remapped; refusing rather than serving undefined weights."
            )
        from sglang.srt.managers.weg2_memory_saver import (
            GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            weights_region,
        )

        # NUTZER-ORDER 2026-09-14 (A2, xsn32, W4 Weg2WakeRefused at
        # weg2_memory_saver.py:368; second half of the same finding):
        # BEFORE ANYTHING IS LOCKED, ENTERED OR MUTATED, the SAME question
        # the main shard's own disk-carrier branch already asks
        # (weight_updater.py's `_weg2_wake_reload_weights`, "one definition,
        # shared with the launch arm" -- reused here rather than
        # re-derived, exactly that doctrine one caller wider): is
        # `update_weights_from_disk` a DEFINED operation for the checkpoint
        # THIS call is about to reload? On a quantized checkpoint it is
        # not -- `process_weights_after_loading` has already replaced the
        # parameters this reload would write into with transposed,
        # weight_loader-less ones, and the raise happens inside the loader
        # with the VMM pages already committed. This is defense in depth
        # for the case the sleep-leg's own completeness check
        # (`_weg2_xchg_wake_source_gap`) never ran for this tag -- the
        # boot-time INITIAL sleep is explicitly not a flip and skips that
        # check entirely (`_weg2_xchg_deposit_before_sleep`'s own
        # `flip_index < 0` return) -- so this is the one check that reaches
        # every call to this method regardless of how it got here.
        assert_backup_off_wake_refill_is_defined(
            quantization=self._weg2_draft_checkpoint_quantization(),
            context="weights_draft disk-reload wake path",
        )

        draft_path = (
            getattr(server_args, "speculative_draft_model_path", None)
            or server_args.model_path
        )
        with weights_region(
            self.memory_saver_adapter, GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            enable_cpu_backup=False,
        ):
            with self._weg2_pcie_lock("wake-H2D weights_draft reload"):
                try:
                    success, message = self.draft_worker.update_weights_from_disk(
                        UpdateWeightFromDiskReqInput(
                            model_path=draft_path,
                            load_format=getattr(server_args, "load_format", None),
                            flush_cache=False,
                            torch_empty_cache=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- name it, do not swallow
                    raise Weg2WakeRefused(
                        "W4 Weg2WakeRefused: weights_draft's disk-reload wake "
                        f"path raised while refilling from {draft_path!r}: "
                        f"{type(exc).__name__}: {exc}. The VMM pages are "
                        "committed but their content is undefined; this group "
                        "is fatal."
                    ) from exc
        if not success:
            raise Weg2WakeRefused(
                "W4 Weg2WakeRefused: weights_draft's disk-reload wake path "
                f"could not refill from {draft_path!r}: {message!r}. The VMM "
                "pages are committed but their content is undefined; this "
                "group is fatal."
            )
        return True

    def _weg2_xchg_inject_from_peer(self, *, terms, tag=None, **kw) -> bool:
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

        # #73: `terms` ist auf dem ipc-Arm None (kein Host-Bounce, also nichts
        # zu bepreisen) -- die Zeile sagt das, statt an einem Attribut zu
        # sterben, das es dort per Konstruktion nicht gibt.
        priced = ("(no host bounce priced: oncard=ipc)" if terms is None
                  else f"({terms.total_bytes} B, {terms.expression()})")

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
        # #1330 B4n SLICE 3: THE PHASE AND ITS HANDSHAKE, from the hook.
        #
        # `authoritative` is an IMPORTING hook -- this rank owns the pages the
        # bytes must land in -- so its cross pairs run `collect` and wait on
        # the depositing rank's `full`. The region and the semaphore set are
        # the ones this process already opens for the shadow; opening a second
        # pair would be two handles on one handshake.
        #
        # A CROSS LEG WITHOUT A HANDSHAKE IS REFUSED, NOT DOWNGRADED: if either
        # is unavailable the adapter takes the unsplit form, and
        # `run_bounce_leg` then refuses every cross pair by name rather than
        # running it as `both` -- which would ask this rank for the peer's
        # device address again, i.e. weg2xsn20's wall.
        # #1374 F1 PER-TAG LOCKSTEP, the COLLECT half. `tag` restricts this
        # collect to the tag the resume loop has just mapped, in the SAME order
        # the depositing rank walks -- both sides read `weights_family_tags`,
        # so the order has one producer and neither side invents it. The
        # collector posts `drained(tag)` at the end of the lane loop, which is
        # what releases the peer's deposit of the NEXT tag.
        # #1391 (DESK10) ROUND 3, CORRECTED: the guard used to live HERE,
        # checking `_cdescs` for emptiness before calling the leg. WRONG
        # AUFRUFER (coordinator finding, boot weg2xsn31 instrument run): this
        # method's own "pieces=0" log line appears ZERO times in either
        # rank's log on the metal, because the shadow-mode grader
        # (`_weg2_xchg_shadow_compare`) calls `_weg2_xchg_inject_weights`
        # with NO tag at all (mode=shadow, whole-plan-at-once -- see that
        # method, weight_updater.py:1693) -- so `tag` here is `None` on the
        # boot's ACTUAL collect call, this whole `if tag is not None:` branch
        # never executes, and a guard placed inside it is dead on the metal
        # even though every desk test that called this method WITH an
        # explicit tag (hand-built, not the real caller) passed. The guard
        # now lives in :meth:`_weg2_xchg_bounce_leg` itself, the one frame
        # BOTH this method and the deposit side actually reach regardless of
        # whether `tag` is a string or `None`.
        _sems = self._weg2_xchg_sems()
        _cdescs = list(plan.descs)
        if tag is not None:
            _cdescs = [d for d in _cdescs
                       if str(getattr(d, "tag", "")) == str(tag)]
            if not _cdescs:
                logger.info(
                    "WEG2-XCHG COLLECT tag=%s pieces=0 plan_tags=%s -- this "
                    "rank's plan carries no desc for this tag; the lockstep "
                    "step is a no-op and the peer's drain still has to be "
                    "posted",
                    tag, sorted({str(getattr(d, "tag", "")) for d in plan.descs}))
        # weg2xsn85 (#1378): the W108 check inside the leg needs the lanes
        # the WHOLE plan covers, not this tag's slice -- see the leg's
        # `covered_lanes` for the lane-sparse-tag shape a pipeline source
        # produces.
        from sglang.srt.weg2 import weight_exchange_bounce as _bx
        try:
            _covered_lanes = set(_bx.group_descs_by_pair(list(plan.descs)).keys())
        except Exception:  # noqa: BLE001 -- an unbuildable coverage keeps the per-tag set
            _covered_lanes = None
        self._weg2_xchg_bounce_leg(
            descs=_cdescs, ops=ops, boot_nonce=boot_nonce,
            terms=terms, mode=mode, device=int(device),
            hook="authoritative",
            region=self._weg2_shadow_region(),
            sems=_sems, tag=tag, rank=int(rank),
            covered_lanes=_covered_lanes,
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
        #
        # NUTZER-ORDER 2026-09-14: THE WAKE SIDE NEEDS TO KNOW whether this
        # leg actually carried bytes for `tag`, so it can try the real
        # exchange FIRST for `weights_draft` too (draft is "just another
        # layer") and fall back to disk (#1394) only when the exchange
        # genuinely found nothing -- never the other way around, and never
        # unconditionally. `bool(_cdescs)` -- computed above, not
        # re-derived -- is exactly that answer for a caller that named a
        # `tag`; a caller with `tag=None` (the whole-plan shadow grader)
        # gets `bool(plan.descs)` instead, which is the same question one
        # level up.
        #
        # #108: DIE DECKUNG FESTHALTEN, an der Stelle die sie KENNT.
        # `_weg2_adopt_cover` liest sie spaeter zurueck, statt sie ein
        # zweites Mal abzuleiten -- dieselbe Regel, an der heute #106 und
        # #107/2 gescheitert sind: wer eine Zahl zweimal herleitet, bekommt
        # irgendwann zwei verschiedene. `plan.descs` ist, was die Karte
        # diesem Rang zuschreibt; `_cdescs` ist, was der Leg davon
        # tatsaechlich getragen hat.
        try:
            _erwartet = len(getattr(plan, "descs", ()) or ())
            self._weg2_last_inject_cover = (len(_cdescs or ()), _erwartet)
        except (AttributeError, TypeError):
            self._weg2_last_inject_cover = (0, 0)
        return bool(_cdescs)

    def _weg2_adopt_cover(self) -> tuple:
        """(gefuellt, erwartet) fuer #108 -- aus der KARTE, nicht geschaetzt.

        Erwartet ist, was die Halter-Karte diesem Rang zuschreibt: jeder
        Parameter des lebenden Modells, der in der letzten Join-Runde
        Descriptors mit diesem Rang als Ziel hatte. Gefuellt ist, was der
        Inject davon tatsaechlich beschrieben hat.

        WARUM NICHT "alle Parameter des Modells": ein Rang haelt unter
        Form A nur SEINEN Ausschnitt, und der meta-Draft-Schatten haelt
        bewusst NICHTS (#103-#105). Wuerde hier die Modellgroesse als
        Erwartung stehen, koennte kein Rang je vollstaendig decken und der
        Riegel bliebe immer stehen -- ein Guard, der nie oeffnet, ist so
        unbrauchbar wie einer, der nie schliesst.

        KONSERVATIV: laesst sich die Zahl nicht ermitteln, gibt diese
        Methode ``(0, 0)`` zurueck. ``adopt.mark_adopted`` wertet das NICHT
        als Erfolg, der Riegel bleibt also stehen. Ein Boot, der mit Grund
        nicht antwortet, ist besser als einer, der auf Zufallszahlen
        rechnet.
        """
        try:
            letzter = getattr(self, "_weg2_last_inject_cover", None)
            if isinstance(letzter, tuple) and len(letzter) == 2:
                return int(letzter[0]), int(letzter[1])
        except (TypeError, ValueError):
            pass
        return 0, 0

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
            # #1374 F1: the per-tag collect runs INSIDE the resume loop now, so
            # by the time this is reached every tag has been collected beside
            # its own resume. Collecting the whole plan again here would be a
            # second writer over bytes already injected -- the ein-job-ein-mover
            # defect the branch above names for the other carriers.
            if self._weg2_xchg_collected_per_tag:
                self._weg2_xchg_collected_per_tag = False
                logger.info(
                    "WEG2-XCHG INJECT per-tag=done -- the resume loop "
                    "collected each tag beside its own resume (#1374); this "
                    "once-per-wake entry stands down rather than re-injecting")
                return
            self._weg2_xchg_inject_weights()
            # #108 ERSTBOOT-ADOPTION: hier faellt der Riegel -- oder er bleibt.
            #
            # Haelt dieser Rang Platzhalter (D unter `--weg2-d-adopt on`), war
            # GENAU DIESER Inject der Grund seines Bootens. Die Deckung kommt
            # aus der Halter-Karte, nicht aus dem Gefuehl: `_weg2_adopt_cover`
            # zaehlt, wieviele der Tensoren, die dieser Rang halten MUSS,
            # tatsaechlich Bytes bekommen haben. Deckt der Inject sie nicht
            # alle, bleibt der Riegel stehen und der Rang verweigert weiter --
            # ein halb gefuelltes Modell rechnet, und das ist schlimmer als
            # eines, das nicht antwortet.
            try:
                from sglang.srt.weg2 import adopt as _adopt

                if _adopt.weights_are_placeholder():
                    _filled, _expected = self._weg2_adopt_cover()
                    _adopt.mark_adopted(_filled, _expected)
                    logger.info(
                        "#108 ADOPT-COVER filled=%d expected=%d -> %s",
                        _filled, _expected,
                        "PLATZHALTER GELOEST, dieser Rang rechnet"
                        if not _adopt.weights_are_placeholder()
                        else f"RIEGEL BLEIBT ({_adopt.placeholder_reason()})")
            except ImportError:
                pass
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

            carrier = self._weg2_wake_weight_carrier()
            if not self._weg2_xchg_shadow_armed_for(carrier):
                return
            # #1391 (DESK10) ROUND 4: STAND DOWN, THE SAME PATTERN
            # `_weg2_wake_reload_weights`'s CARRIER_EXCHANGE branch already
            # uses for `_weg2_xchg_collected_per_tag`. The resume loop above
            # (`resume_memory_occupation`) now runs this exact compare PER
            # TAG, from inside the per-tag loop, precisely so the per-tag
            # `post_drained` gate fires during the resume instead of never
            # (#1391's wedge: shadow mode's only collect used to be THIS
            # call, once, with `tag=None`, after the whole family had
            # already resumed -- too late for D's lockstep to ever see a
            # drain). Running it again here over the WHOLE plan would grade
            # every tag's bytes a second time for no reason and, worse, look
            # like independent corroboration of a single measurement.
            if self._weg2_xchg_shadow_compared_per_tag:
                self._weg2_xchg_shadow_compared_per_tag = False
                logger.info(
                    "WEG2-XCHG-INJECT mode=shadow per-tag=done -- the resume "
                    "loop compared each tag beside its own resume (#1391); "
                    "this once-per-wake entry stands down rather than "
                    "grading the same bytes twice")
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

    def _weg2_xchg_shadow_armed_for(self, carrier) -> bool:
        """Is a SHADOW-mode collect due for this wake, on this carrier?

        #1391 (DESK10) ROUND 4: ONE PREDICATE, read from BOTH the per-tag
        loop (:meth:`resume_memory_occupation`) and the once-per-wake
        stand-down check (:meth:`_weg2_xchg_shadow_compare`) above, so the
        two can never disagree about whether a given wake is a shadow wake
        -- the exact class of defect a second reading of one decision
        always risks (`ein-job-ein-mover`). Mirrors what
        `_weg2_xchg_shadow_compare` checked inline before this round:
        the exchange is armed, the inject mode is shadow, and the carrier
        is not `stock` (an unreleased-pages wake has nothing to compare --
        DANGER DIRECTION 1, unchanged from before this round).
        """
        from sglang.srt.weg2 import weight_exchange as wx

        if not (wx.exchange_armed() and wx.inject_mode() == wx.INJECT_SHADOW):
            return False
        return carrier != self.CARRIER_STOCK

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
        #
        # B4d: THE WAVE PARTITION IS A GROUP FACT, so it is decided HERE and
        # nowhere else.  `waves_for_plan` RECORDS a published-vs-derived
        # mismatch instead of raising, because a raise inside the plan builder
        # is rank-local and the other five ranks would carry on and meet a peer
        # that is gone.  This fence is the wired group-uniform mechanism of the
        # wake path, so the reason rides its per-rank dict and any one rank's
        # mismatch makes EVERY rank raise W29 with the same peer list.  No new
        # bus, and no second group-uniform error: W29's contract already says
        # "every rank stops in this fence".
        #
        # THE DIGESTS TRAVEL EVEN WHEN THEY AGREE, which is the half a reader
        # needs at 3am: the fence line then states what each rank actually
        # planned, so "all six agreed on the priced partition" is a MEASUREMENT
        # rather than an absence of complaint.
        from sglang.srt.weg2 import weight_exchange as wx

        coverage_armed, coverage_reason, coverage_stop = wx.coverage_leg_decision()
        if coverage_reason:
            logger.error("%s", coverage_reason)
        wave_reason = wx.wave_disagreement()
        mine = {
            "rank": rank,
            "ok": bool(ok) and not coverage_stop and not wave_reason,
            "failure": (str(failure or "")
                        or (str(coverage_reason) if coverage_stop else "")
                        or str(wave_reason or "")),
            "coverage_armed": bool(coverage_armed),
            "card": self._weg2_card_uuid() or "unknown",
            "leg_ms": float(leg_ms),
            "per_tag": dict(per_tag or {}),
            "waves_published": wx.waves_digest(wx.published_waves() or ()),
            "waves_planned": wx.waves_digest(wx.planned_waves() or ()),
        }
        gathered: List[Optional[Dict[str, Any]]] = [None] * world
        torch.distributed.all_gather_object(gathered, mine, group=cpu_group)
        votes = [v for v in gathered if isinstance(v, dict)]
        bad = [v for v in votes if not v.get("ok", False)]
        logger.info(
            "WEG2-GROUP-FENCE %s joined in %.0f ms t=" + f"{time.time():.3f}" + " (world=%d ranks, chain sends joined=%s; "
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

    def _weg2_drain_hicache_before_sleep(
        self, bound_s: float = WEG2_SLEEP_DRAIN_BOUND_S
    ) -> None:
        """Drain HiCache in-flight terms (write-through / storage backup /
        load-back / prefetch) before a sleep is judged -- see the caller.

        fnFL2x105: a GROUP loop (``weg2_sleep_drain``). Called on EVERY rank,
        idle or not: the first pass is a reduction over the attention group
        that ``check_hicache_events`` posts its collectives on, and every rank
        polls exactly as often as the others. A group still blocked when the
        reduced verdict stops the loop raises W120 on every rank at once.
        """
        sch = self.scheduler
        if sch is None or not sch.enable_hierarchical_cache:
            return
        tc = sch.tree_cache
        t0 = time.monotonic()
        first = list(sch.idle_blockers())
        verdict, polls = drain_until_group_verdict(
            idle_blockers=sch.idle_blockers,
            check_hicache_events=tc.check_hicache_events,
            group_max=functools.partial(
                tc.hicache_group_max, label="weg2_sleep_drain"
            ),
            bound_s=bound_s,
        )
        waited_s = time.monotonic() - t0
        now = list(sch.idle_blockers())
        if first or polls or not verdict.idle:
            logger.warning(
                "WEG2 SLEEP-DRAIN: waited %.2f s (%d group polls) for HiCache "
                "in-flight terms before the sleep; this rank's blockers at entry "
                "%s, now %s; group verdict %s",
                waited_s, polls, first, now, verdict,
            )
        if not verdict.idle:
            raise Weg2SleepDrainRefused(
                refusal_message(
                    verdict=verdict,
                    own_blockers=now,
                    waited_s=waited_s,
                    polls=polls,
                    bound_s=bound_s,
                    rank_desc=collective_rank_desc(tc),
                )
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
        # fnFL2x100: NAME THE DEAD LEG TO THE WAITERS FIRST. The fence below
        # reaches only THIS group, and only once every member arrives -- x100's
        # D TP1 sat in it 121 s while P PP1 (credit), D TP0 (BAR1 'free') and
        # P PP0 (lane mode) waited out their own 120 s budgets for a leg that
        # had died at W106. Posted before the fence, read by every liveness
        # probe and credit wait of this flip (weg2/leg_abort.py).
        self._weg2_post_leg_abort(f"{what}: {type(exc).__name__}: {exc}")
        if self.weg2_fence_raised:
            self.weg2_fence_raised = False
            return
        if not self._weg2_fence_is_armed():
            return
        self._weg2_group_fence(
            what, ok=False, failure=f"{type(exc).__name__}: {exc}"
        )

    def _weg2_leg_abort_key(self) -> Optional[Dict[str, Any]]:
        """``{boot_nonce, flip, group, rank}`` of the leg in progress, or
        ``None`` when this rank is in no flip (no boot nonce, no flip index,
        no group identity) -- then there is nothing to post or to read."""
        from sglang.srt.weg2 import weight_exchange_region as xr

        boot_nonce = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
        flip = self._weg2_flip_index_now
        group = self._weg2_group_name()
        rank = self._weg2_rank()
        if (not boot_nonce or not isinstance(flip, int) or flip < 0
                or group == "?" or rank is None or int(rank) < 0):
            return None
        return {"boot_nonce": boot_nonce, "flip": int(flip),
                "group": str(group), "rank": int(rank)}

    def _weg2_post_leg_abort(self, reason: str) -> None:
        """Publish this rank's failed leg (weg2/leg_abort.py). Never raises:
        the failure being reported is the one that must propagate."""
        from sglang.srt.weg2 import leg_abort as la

        key = self._weg2_leg_abort_key()
        if key is None:
            return
        try:
            path = la.post(reason=reason, **key)
        except OSError as exc:
            logger.error("WEG2-LEG-ABORT post failed: %r", exc)
            return
        logger.error("WEG2-LEG-ABORT posted group=%s rank=%s flip=%s path=%s "
                     "reason=%s", key["group"], key["rank"], key["flip"], path,
                     reason[: la.REASON_MAX_CHARS])

    def _weg2_foreign_leg_aborts(self) -> str:
        """The OTHER ranks' aborts of this flip as one clause, ``""`` if none.
        Read by every wait of the leg (liveness, credit, lane refusal)."""
        from sglang.srt.weg2 import leg_abort as la

        key = self._weg2_leg_abort_key()
        if key is None:
            return ""
        try:
            return la.describe(la.foreign(**key))
        except OSError:
            return ""

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

    def _weg2_wake_kv_first_ok(self, tags) -> bool:
        """Wake-Parallel (user 18.09.): may the kv_cache pool be resumed BEFORE
        the weight legs? Only when the card can fund it right now: free -
        floor >= kv bytes (+256 MiB margin). On the 3080s this holds (the
        sleeper's KV pool went at sleep-kv); on the 5090 it holds only once
        the sleeper's weights are far enough paused -- else the old order.
        SGLANG_WEG2_WAKE_KV_FIRST=0 disables it."""
        if str(os.environ.get("SGLANG_WEG2_WAKE_KV_FIRST", "1")).strip().lower() in ("0", "false", "no", "off"):
            return False
        try:
            need = int(self._weg2_tag_bytes(GPU_MEMORY_TYPE_KV_CACHE) or 0)
            free = self._weg2_free_bytes()
            floor = int(self._weg2_corridor_floor_bytes() or 0)
        except Exception as exc:  # noqa: BLE001 -- no probe, old order
            logger.info("WEG2-WAKE-KV-FIRST skipped (%s: %s)", type(exc).__name__, exc)
            return False
        if free is None or need <= 0:
            logger.info("WEG2-WAKE-KV-FIRST skipped free=%s need=%d", free, need)
            return False
        margin = 256 << 20
        ok = int(free) - floor - margin >= need
        # xsn323: the pool up BEFORE the legs takes `need` off every tag claim
        # of the legs. The legs' tightest point is known from this rank's last
        # wake (`_weg2_leg_min_free_mib`); with the pool up it must still hold
        # the legs' reserve (the largest tag + the peer's on-card staging,
        # SGLANG_WEG2_WAKE_KV_LEG_RESERVE_MIB, default 4352 = 2918 + 1309 + margin
        # measured on the 5090 in xsn322/323). No record yet: the old order.
        ref = getattr(self, "_weg2_leg_min_free_mib", None)
        try:
            reserve_mib = int(os.environ.get("SGLANG_WEG2_WAKE_KV_LEG_RESERVE_MIB", "4352"))
        except ValueError:
            reserve_mib = 4352
        if ok:
            if ref is None:
                ok = False
                why = "no leg record yet (first wake of this rank): old order"
            else:
                legs_ok = int(ref) - (need >> 20) >= (floor >> 20) + (margin >> 20) + reserve_mib
                ok = bool(legs_ok)
                why = (f"legs' tightest free last wake={int(ref)} MiB - kv={need >> 20} MiB "
                       f"{'>=' if legs_ok else '<'} floor+margin+reserve={(floor >> 20) + (margin >> 20) + reserve_mib} MiB")
        else:
            why = "free - floor - margin < kv"
        logger.info("WEG2-WAKE-KV-FIRST %s free=%d MiB floor=%d MiB need=%d MiB (kv_cache resumed %s the weight legs; %s)",
                    "EARLY" if ok else "LATE", int(free) >> 20, floor >> 20, need >> 20,
                    "before" if ok else "after", why)
        return ok

    def _weg2_stage_charge(self):
        """weg2xsn269 (18.09.): the deposit lanes' on-card IPC staging is a
        cudaMalloc on the card the WAKING rank resumes into. Booked here
        against this leg's VRAM credit (``debit`` under the same lock the
        waker claims under) so that a staging can only take bytes the waker
        was never promised; freed stagings ``refund``. Returns the
        ``(nbytes) -> refund | None`` the bounce expects, or None when this
        leg has no credit (then the staging is unbooked, as before)."""
        credit = getattr(self, "_weg2_leg_credit", None)
        if credit is None or not hasattr(credit, "debit"):
            return None

        def _charge(nbytes: int):
            n = int(nbytes)
            try:  # Task #20: free, unpromised VRAM may be staged (overdraw)
                _free = self._weg2_free_bytes()
                _floor = self._weg2_corridor_floor_bytes()
            except Exception:  # noqa: BLE001 -- no probe, no overdraw
                _free, _floor = None, None
            if not credit.debit("ipc-stage", n, free_bytes=_free, floor_bytes=_floor):
                logger.info(
                    "WEG2-SEQ stage-charge REFUSED %d MiB: the card's credit balance "
                    "is what the waking rank is promised; host path for this tag",
                    n >> 20,
                )
                return None
            # H11: a booking, not a bare refund -- _stage_alloc marks it live
            # once the cudaMalloc returned (the waker stops counting it twice)
            from sglang.srt.managers.weg2_memory_saver import StageBooking

            return StageBooking(credit, "ipc-stage", n)

        return _charge

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
        """Per-leg cache in front of :meth:`_weg2_shadow_plan_uncached`
        (2026-09-15, weg2xsn94): the plan is the same for every tag of a leg
        and cost ~1 s per derivation; the cache lives for the BOOT.

        ORDER POINT 2 (xsn125, WEG2-SLEEP-PRELOOP / WEG2-WAKE-TAIL): the
        hook ALWAYS passes the manifest agreement, and the first form of this
        cache bypassed itself for ``agreed is not None`` -- so both hooks
        re-derived the plan on every leg: source_hook=580-640 ms before the
        first deposit, dest_hook_compare=600-740 ms after the last collect,
        1.2-1.3 s of every flip on the critical path. ``AgreedPieces`` is a
        frozen dataclass (a frozenset of piece keys plus two counts) and the
        manifest is a boot constant, so the agreement IS a valid cache key:
        the same agreed set yields the same plan. An unhashable agreement
        (a desk double) falls back to the uncached call as before."""
        _lc = getattr(self, "_weg2_xchg_leg_cache", None)
        if _lc is None:
            _lc = {}
            try:
                self._weg2_xchg_leg_cache = _lc
            except AttributeError:
                _lc = None
        # class-level call: the execution smokes' stubs copy this method
        # alone (the #1358 lesson) and carry no `_uncached` attribute.
        _impl = SchedulerWeightUpdaterManager._weg2_shadow_plan_uncached
        _agreed_key = None
        if agreed is not None:
            try:
                _agreed_key = hash(agreed)
            except TypeError:
                _lc = None
        if _lc is None:
            return _impl(self, hook, group, rank, agreed=agreed,
                         require_agreement=require_agreement)
        key = _weg2_plan_key(hook, group, rank, require_agreement, _agreed_key)
        if key in _lc:
            return _lc[key]
        with _WEG2_PLAN_LOCK:
            # H11: single flight -- a thread that waited here finds the key
            if key in _lc:
                return _lc[key]
            out = _impl(self, hook, group, rank, agreed=agreed,
                        require_agreement=require_agreement)
            if out is not None and out[0] is not None:
                _lc[key] = out
        return out

    def _weg2_shadow_plan_uncached(self, hook: str, group: str, rank: int, *,
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
            # #1330 B4n: THE JOIN IS THE PRODUCER WHENEVER THE EXCHANGE IS
            # ARMED, and there is NO fallback from it to the derivation.
            #
            # The derivation builds the on-card DIAGONAL -- inventory
            # `shard_axis=REPLICATED` (weight_exchange_shadow.py:3237), BOTH
            # GroupLayouts `tp_size=1` (:3332-3334) -- and asks a rank for the
            # PEER's pointer, which no process can answer. That is the measured
            # source of `src_resolved=0/N` on all 24 legs of weg2xsn20. It may
            # never step in silently again, so a missing peer manifest is a
            # NAMED refusal carrying the expected FILE PATH, not a quiet
            # re-derivation. `test_no_join_path_can_reach_derive_leg_plan`
            # pins that there is no route from the armed arm to the diagonal.
            #
            # THE GATE IS THE ARM, not the presence of files: on `ring` and on
            # the pure `shadow` arm `arm_coverage_at_load` writes no manifests
            # at all (it returns early on `not exchange_armed()`), so those arms
            # keep the derivation and the plan line SAYS which producer
            # answered (`facts.source`). Choosing the producer by whether files
            # happen to exist would be exactly the silent fallback this
            # refusal exists to remove.
            if wx.exchange_armed():
                from sglang.srt.weg2 import xchg_manifest as xm

                mans, why = xm.manifests_for_boot(
                    pp_group="P", tp_group="D")
                if mans is None:
                    return None, why
                plan, reason = xm.leg_plan_from_join(
                    hook=str(hook), group=str(group), rank=int(rank),
                    manifests=mans,
                    src_addr=self._weg2_join_src_addr(
                        str(hook), str(group), int(rank), model,
                        region=region_tag),
                    dst_addr=self._weg2_join_dst_addr(
                        str(hook), str(group), int(rank), model,
                        region=region_tag),
                    # The materialisation check runs against THIS rank's own
                    # live tensors, at the flip -- where the manifest could
                    # have drifted since it was written at the end of loading.
                    model=model,
                    # THE REGION THIS RUNNER OWNS. A leg addresses only its own
                    # region's tensors: weg2xsn25 measured what mixing costs --
                    # P rank 0's leg carried the draft head's eleven pieces,
                    # which live in ANOTHER runner, and read dst_resolved=893/
                    # 904 (893 being exactly the main runner's own count).
                    region_tag=region_tag,
                    log=logger.info)
                # NUTZER-ORDER 2026-09-14: "draft ist auch nur ein layer...
                # warum muss er ueber den ring gehen?" -- ONE REGION PER
                # RUNNER (the shape `roll_forward_weights_tag`'s own
                # docstring already named as the correct fix and deferred to
                # "S6 with W73" -- we are in S6). If this rank ALSO carries
                # a draft runner whose tag is in the exchange family, ask
                # the SAME join for THAT runner's OWN region too, and union
                # the descriptors: the per-tag filter both callers already
                # apply downstream (`str(getattr(d, "tag", "")) ==
                # str(tag)`) then scopes correctly to `weights_draft`
                # without any caller-side special case, the same "one
                # authority, no special case anywhere" shape
                # `is_weights_family_tag` already established.
                #
                # FAILS SOFT, ON PURPOSE, and ONLY FOR THE DRAFT HALF: a
                # draft join failure (no counterpart on the peer -- see
                # `_weg2_xchg_draft_plan_or_none`'s own docstring for the
                # (a)/(b) population finding that makes this a REAL
                # possibility, not a defensive default) must not fail the
                # MAIN region's leg, which is unrelated. The draft tag then
                # keeps zero descriptors here, exactly the pre-existing
                # shape `_weg2_xchg_wake_source_gap`'s exemption and
                # `_weg2_xchg_draft_reload_from_disk` (#1394) already
                # handle -- disk, never the ring, never a mini-ring.
                if plan is not None:
                    draft_plan, draft_reason = self._weg2_xchg_draft_plan_or_none(
                        hook=str(hook), group=str(group), rank=int(rank),
                        mans=mans)
                    if draft_plan is not None and draft_plan.descs:
                        plan = _dataclasses_replace(
                            plan, descs=tuple(plan.descs) + tuple(draft_plan.descs),
                            tags=tuple(sorted(set(plan.tags) | set(draft_plan.tags))))
                    elif draft_reason:
                        logger.info(
                            "WEG2-XCHG-DRAFT-PLAN-SKIPPED hook=%s group=%s "
                            "rank=%s: %s -- weights_draft keeps zero "
                            "exchange descriptors this leg; the wake side "
                            "falls back to its own disk-reload path (#1394) "
                            "rather than the ring",
                            hook, group, rank, draft_reason)
                return plan, reason
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

    def _weg2_xchg_draft_plan_or_none(self, *, hook: str, group: str,
                                      rank: int, mans):
        """THE DRAFT RUNNER'S OWN LEG, joined against the SAME manifests.

        NUTZER-ORDER 2026-09-14 ("draft ist auch nur ein layer... warum muss
        er ueber den ring gehen?"): ONE REGION PER RUNNER, the shape
        ``weight_exchange.roll_forward_weights_tag``'s own docstring already
        names as the correct fix, deferred there to "S6 with W73" -- this is
        that fix's exchange-plan half. Returns ``(None, "")`` whenever the
        draft runner is not applicable at all (no draft worker in this
        process, exchange not armed, or the tag is not in the family) --
        that is not a failure, it is "nothing to add", and callers must not
        log it as one. Returns ``(None, reason)`` when a draft runner IS
        applicable but its OWN join could not be built, and ``(plan, "")``
        on success.

        THE (a)/(b) POPULATION FINDING, verified by reading (no boot this
        round): ``weight_exchange.weights_region_tag_for``'s own docstring
        measures group D's ``weights_draft`` tag at 1382/1311/1311 MiB, of
        which the checkpoint's ``mtp.*`` term is only 0.396 GiB -- the
        REMAINDER is embed/head shards the draft runner RE-MATERIALISES
        rather than loads a peer-exchanged copy of. ``launcher.py``'s
        ``DRAFT_KV_ON_P_DEFAULT = "on"`` (read, not edited -- DESK9's file)
        means group P's shipped default DOES carry the four speculative
        flags (``P_DRAFT_KV_FLAGS``), so the STALE half of that docstring's
        own premise ("Group P carries no --speculative-* in this form") is
        wrong on the shipped default -- P's own draft/producer runner is a
        real candidate SOURCE for the ``mtp.*`` layer's bytes.

        WHETHER THE RE-MATERIALISED SHARDS (population a) ALSO HAPPEN TO
        MATCH ACROSS GROUPS -- because both sides load them deterministically
        from the identical checkpoint bytes -- is NOT something this file can
        determine without a real boot's manifests: ``xchg_manifest.
        join_manifests`` (xchg_manifest.py:800-934, DESK9's file, read not
        edited) is ALL-OR-NOTHING per region -- ANY name present on one side
        and absent on the other raises ``Weg2XchgSourceMissing`` (W74) for
        the WHOLE region's join, discarding tensors that DID match along with
        the ones that did not (:930-934, the `unsourced` list is checked only
        AFTER the matching loop finishes). There is therefore no
        `file:line`-clean way to move ONLY population (b) through this join
        while leaving population (a) to disk within ONE region/tag -- that
        would need per-PARAMETER tagging inside the draft runner's own memory
        region (model_runner.py / the model class, outside this file's
        boundary), not a change to how this method calls the join.

        SO THIS METHOD MAKES NO ASSUMPTION EITHER WAY: it tries the real
        join for the WHOLE ``weights_draft`` region, exactly like the main
        region above. If group P's producer construction happens to hold a
        matching name for every one of group D's draft-region tensors (which
        this file cannot verify without a boot), the WHOLE tag -- MTP layer
        included -- moves through the real exchange, byte-identical
        population (a) bytes and all, which is correct (not merely
        harmless): both sides already compute them identically from the same
        checkpoint, so exchanging them is not a new source of truth, just a
        redundant one. If even ONE name is genuinely one-sided (population
        (a) really is asymmetric on this boot's actual argv), the WHOLE
        region's join refuses (W74) and this method reports that reason
        upward rather than inventing a partial success -- the caller then
        keeps the pre-existing disk-reload fallback (#1394,
        :meth:`_weg2_xchg_draft_reload_from_disk`) for the tag, never the
        ring, never a mini-ring.
        """
        try:
            from sglang.srt.managers import weg2_memory_saver as ms
            from sglang.srt.weg2 import weight_exchange as wx
            from sglang.srt.weg2 import xchg_manifest as xm

            if not wx.exchange_armed():
                return None, ""
            # #1378 xsn81: EVERY early exit is NAMED. P rank 2 logged
            # "COLLECT tag=weights_draft pieces=0" and nothing else while D
            # deposited 25+4 units towards its card and refused W108 -- an
            # empty reason here is what kept the merge site's SKIPPED line
            # silent for six boots.
            drafter = _weg2_drafter_of(self)
            if drafter is None:
                return None, ("no-drafter: neither draft_worker nor "
                              "draft_kv_producer.draft_runner on this rank")
            draft_model = getattr(drafter, "model", None)
            if draft_model is None:
                return None, f"no-draft-model: {type(drafter).__name__}"
            # #1378 xsn77 -- THE SAME RULE THE MANIFEST WRITER USES. On group
            # P the drafter is resident for the flip but P runs no speculative
            # decoding, so `classify_runner` answers None and
            # `weights_region_tag_for` RAISES under an armed exchange; this
            # site swallowed that as "draft-shape-unclassified" and planned no
            # draft leg on P, while `_weg2_seam_inventory` had written P's
            # draft manifest under weights_draft (19 pieces) and D deposited
            # 25 units on lane c0 that P never collected. The handshake tokens
            # stayed posted, the NEXT tag's collect on that lane (weights,
            # embed_tokens) read the stale unit and refused -- four boots of
            # "embed_tokens content-changed". A drafter reached through
            # `_get_draft_model_runner` under an armed exchange IS the draft
            # region, on both groups, exactly as the manifest says.
            if ms.draft_tag_in_family():
                draft_region_tag = wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT
            else:
                try:
                    draft_region_tag = wx.weights_region_tag_for(
                        wx.RunnerShape.of(drafter))
                except BaseException:  # noqa: BLE001 -- an unclassified shape
                    return None, "draft-shape-unclassified"
            # #1378 xsn34 root: the DRAFT's own ``lm_head.weight`` is the
            # shared TARGET module on this arm (eagle_worker_v2's
            # set_embed_and_head_modules / set_lm_head_from_target hand the
            # co-located target's table in by reference, qwen3_5_mtp.py:351
            # ``self.lm_head = target_lm_head``; the producer side is
            # #1259(b)'s lm_head_from_target, head_released_mib=2425
            # measured on weg2tr1). The counterpart therefore CANNOT exist
            # in the peer's draft manifest -- P's drafter has no such
            # parameter -- and the bytes the exchange would move are the
            # TARGET's lm_head, which the MAIN region's leg already moves
            # (tag=weights, both sides hold it). Excluding it here is not a
            # hole: the proof is MEASURED at this rank (data_ptr identity of
            # draft head vs target head), fail-closed -- no proof, no
            # exclusion, and any OTHER missing counterpart still refuses
            # W74 (test_weg2_draft_lmhead_share_1378.py's mutant).
            _lm_head_excluded: frozenset = frozenset()
            if draft_region_tag == wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT:
                # fnFL2x10: PER PARAMETER, like the embed below. A quantized
                # head carries weight_packed/weight_scale/... and no `weight`,
                # so the single-tensor proof answered "no-weight" and the
                # shared head stayed in the join -- once the manifests name
                # the live drafter, both sides would move the TARGET's head
                # through the draft lane into a still-paused region.
                _proof = self._weg2_draft_lm_head_is_target_share()
                if _proof.startswith("MEASURED-SHARED"):
                    _lm_head_excluded = frozenset({"lm_head.weight"})
                else:
                    _lm_head_excluded, _proof = (
                        self._weg2_draft_lm_head_target_shares())
                if _lm_head_excluded:
                    logger.info(
                        "WEG2-XCHG-DRAFT-LMHEAD-TARGET-SHARED %s -> skip=%s",
                        _proof, sorted(_lm_head_excluded))
                # weg2xsn86 (#1378): THE EMBEDDING IS THE OTHER SHARE.
                # frozen_kv_mtp_worker_v2 hands the draft the target's
                # embed_tokens AND lm_head (set_embed_and_head); on D the
                # draft's `model.embed_tokens.weight` IS the target's tensor
                # (region `weights`, still PAUSED while `weights_draft` is
                # collected -> SIGSEGV in cuMemcpyAsync at draft unit 2 on
                # all three TP ranks). lm_head is excluded from the JOIN on
                # both sides because both measure the share; the embed is
                # shared on D only (PP2 holds its own copy and deposits it),
                # so it stays IN the lane -- consumed, never written -- to
                # keep the two sides' unit indices aligned.
                _shared, _eproof = self._weg2_draft_embed_target_shares()
                # fnFL2x33/x34 (23.09.): THE HEAD IS THE OTHER SHARE'S TWIN.
                # `skip_names` narrows the DRAFT PLAN (leg_plan_from_join,
                # 35 descs), but the LANE derives its unit list from the raw
                # join (`_weg2_seq_lane_descs` -> plan_from_join, 38 units)
                # so both sides agree without metadata -- and unit 2,
                # lm_head.weight_packed (636 MB), was written into the
                # TARGET's head at 0x7f0000000, region `weights`, still
                # PAUSED: SIGSEGV in cudaMemcpyAsync on D TP0, over BAR1
                # (x33) and over the SEQ host lane (x34) alike. The measured
                # head share takes the embed's form: consumed, never written.
                self._weg2_xchg_no_write = frozenset(
                    (str(draft_region_tag), str(n))
                    for n in set(_shared) | set(_lm_head_excluded))
                logger.info(
                    "WEG2-XCHG-DRAFT-EMBED-TARGET-SHARED %s -> no_write=%s "
                    "(+ the measured head share %s, x34)",
                    _eproof, sorted(_shared), sorted(_lm_head_excluded))
            if draft_region_tag != wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT:
                # A drafter this arm classifies as something other than the
                # draft region (e.g. #631/#274's shapes, which stay in the
                # BASE tag by `weights_region_tag_for`'s own design) has
                # nothing separate to add here -- its bytes are already part
                # of the main region's plan above.
                return None, ""
            if not ms.is_weights_family_tag(draft_region_tag):
                # The membership predicate itself says no (e.g. the ring arm,
                # where `draft_tag_in_family()` is False by construction) --
                # not this method's call to make differently.
                return None, ""
            draft_plan, draft_reason = xm.leg_plan_from_join(
                hook=str(hook), group=str(group), rank=int(rank),
                manifests=mans,
                src_addr=self._weg2_join_src_addr(
                    str(hook), str(group), int(rank), draft_model,
                    region=draft_region_tag),
                dst_addr=self._weg2_join_dst_addr(
                    str(hook), str(group), int(rank), draft_model,
                    region=draft_region_tag),
                model=draft_model,
                # #1378 xsn34: the one-sided shared head drops out of the
                # DRAFT side's own tensor list BEFORE the join, so the W74
                # "no counterpart" refusal keeps guarding every OTHER name.
                skip_names=(_lm_head_excluded or None),
                # THE DRAFT RUNNER'S OWN REGION, never the main one -- MUTANT
                # 1's own danger direction (a draft leg reading the main
                # region's bands would write foreign bytes into the draft
                # head, silently). `test_the_draft_leg_never_reads_the_main_
                # regions_bands` pins this by construction, not by hope.
                region_tag=draft_region_tag,
                log=logger.info)
            return draft_plan, ("" if draft_plan is not None
                               else (draft_reason or "draft-join-refused"))
        except BaseException as exc:  # noqa: BLE001 -- an observer never
            # raises: a draft-region failure must never take the MAIN
            # region's leg down with it.
            return None, f"draft-derivation-failed:{_weg2_exc_note(exc)}"

    def _weg2_draft_lm_head_is_target_share(self) -> str:
        """MEASURED, not assumed: is this rank's draft head the TARGET's
        lm_head module (data_ptr identity of the weight tensors)?  Returns
        "MEASURED-SHARED draft_ptr=0x.. target_ptr=0x.." or the reason the
        share is NOT proven -- the caller excludes the draft's lm_head from
        the join ONLY on the MEASURED-SHARED prefix (fail-closed)."""
        try:
            drafter = _weg2_drafter_of(self)
            draft_model = getattr(drafter, "model", None)
            target_runner = getattr(getattr(self, "tp_worker", None),
                                    "model_runner", None)
            target_model = getattr(target_runner, "model", None)
            d_head = getattr(draft_model, "lm_head", None)
            t_head = getattr(target_model, "lm_head", None)
            d_w = getattr(d_head, "weight", None)
            t_w = getattr(t_head, "weight", None)
            if d_w is None or t_w is None:
                return "no-weight (draft head deferred or target head absent)"
            d_ptr, t_ptr = d_w.data_ptr(), t_w.data_ptr()
            if d_ptr == t_ptr:
                return (f"MEASURED-SHARED draft_ptr={d_ptr:#x} "
                        f"target_ptr={t_ptr:#x}")
            return (f"NOT-SHARED draft_ptr={d_ptr:#x} target_ptr={t_ptr:#x} "
                    f"(own table, {d_w.numel()} elems -- needs a source, "
                    f"never silently excluded)")
        except BaseException as exc:  # noqa: BLE001 -- fail-closed observer
            return f"proof-failed:{_weg2_exc_note(exc)}"

    def _weg2_draft_lm_head_target_shares(self):
        """The draft's ``lm_head`` parameter names to skip in the draft join,
        plus a proof string -- MEASURED per parameter (data_ptr identity with
        the target head's parameter of the same name), fnFL2x10.

        ALL OR NOTHING: the set is non-empty only when EVERY parameter of the
        draft head is the target's. A head that is partly its own needs a
        source, and dropping any of it would leave its bytes undefined -- the
        W74 refusal keeps guarding that shape (fail-closed, like
        :meth:`_weg2_draft_lm_head_is_target_share` for the BF16 head).
        """
        try:
            drafter = _weg2_drafter_of(self)
            draft_model = getattr(drafter, "model", None)
            target_runner = getattr(getattr(self, "tp_worker", None),
                                    "model_runner", None)
            target_model = getattr(target_runner, "model", None)
            d_head = getattr(draft_model, "lm_head", None)
            t_head = getattr(target_model, "lm_head", None)
            if d_head is None or t_head is None:
                return frozenset(), ("no-head (draft head=%s target head=%s)"
                                     % (d_head is not None, t_head is not None))
            t_ptrs = {str(n): int(p.data_ptr())
                      for n, p in t_head.named_parameters(recurse=False)}
            d_params = [(str(n), int(p.data_ptr()))
                        for n, p in d_head.named_parameters(recurse=False)]
            if not d_params:
                return frozenset(), "no-params (draft head deferred)"
            own = [n for n, ptr in d_params if t_ptrs.get(n) != ptr]
            if own:
                return frozenset(), (f"NOT-SHARED own={sorted(own)} of "
                                     f"{len(d_params)} -- needs a source")
            return (frozenset(f"lm_head.{n}" for n, _ in d_params),
                    f"MEASURED-SHARED {len(d_params)} params "
                    f"first={d_params[0][0]}:{d_params[0][1]:#x}")
        except BaseException as exc:  # noqa: BLE001 -- fail-closed observer
            return frozenset(), f"proof-failed:{_weg2_exc_note(exc)}"

    def _weg2_draft_embed_target_shares(self):
        """MEASURED, not assumed (weg2xsn86): the draft's ``embed_tokens``
        parameters that ARE the target's (data_ptr identity), as a set of
        the draft-side parameter names (``model.embed_tokens.weight``,
        ``model.embed_tokens.weight_scale`` ...) plus a proof string.
        Fail-closed: anything unmeasurable answers an EMPTY set -- an
        unshared embed needs its bytes written, never silently skipped.
        """
        try:
            drafter = _weg2_drafter_of(self)
            draft_model = getattr(drafter, "model", None)
            target_runner = getattr(getattr(self, "tp_worker", None),
                                    "model_runner", None)
            target_model = getattr(target_runner, "model", None)
            d_inner = getattr(draft_model, "model", None)
            t_inner = getattr(target_model, "model", None)
            d_emb = getattr(d_inner, "embed_tokens", None)
            t_emb = getattr(t_inner, "embed_tokens", None)
            if d_emb is None or t_emb is None:
                return frozenset(), ("no-embed (draft embed=%s target embed=%s)"
                                     % (d_emb is not None, t_emb is not None))
            t_ptrs = {}
            for pname, p in t_emb.named_parameters(recurse=False):
                try:
                    t_ptrs[str(pname)] = int(p.data_ptr())
                except BaseException:  # noqa: BLE001
                    pass
            shared, notes = set(), []
            for pname, p in d_emb.named_parameters(recurse=False):
                try:
                    d_ptr = int(p.data_ptr())
                except BaseException:  # noqa: BLE001
                    continue
                t_ptr = t_ptrs.get(str(pname))
                if t_ptr is not None and t_ptr == d_ptr:
                    shared.add(f"model.embed_tokens.{pname}")
                    notes.append(f"{pname}:MEASURED-SHARED {d_ptr:#x}")
                else:
                    notes.append(f"{pname}:NOT-SHARED draft={d_ptr:#x} "
                                 f"target={'-' if t_ptr is None else hex(t_ptr)}")
            return frozenset(shared), " ".join(notes) or "no-params"
        except BaseException as exc:  # noqa: BLE001 -- fail-closed observer
            return frozenset(), f"proof-failed:{_weg2_exc_note(exc)}"

    def _weg2_xchg_drain_outstanding(self) -> None:
        """The depositor's leg end (2026-09-15): wait for the collector's
        drain of the last ``depth`` tags on every lane this rank deposited
        into, then free the on-card IPC staging. Leaves ZERO leftover
        credits, so the next leg's per-leg counters start clean."""
        from sglang.srt.weg2 import weight_exchange_bounce as bx
        seq = dict(getattr(self, "_weg2_xchg_lane_seq", None) or {})
        try:
            sems = self._weg2_xchg_sems()
        except Exception:  # noqa: BLE001
            sems = None
        depth = int(bx.seq_buffer_depth())
        if sems is not None:
            for lane_key, n in seq.items():
                outstanding = min(int(n), depth)
                if outstanding <= 0:
                    continue
                try:
                    if str(lane_key).startswith("p"):
                        rv = bx.CrossSlotRendezvous(
                            sems, None, pair=int(str(lane_key)[1:]),
                            budget_s=WEG2_GROUP_FENCE_BUDGET_S)
                    else:
                        rv = bx.CrossSlotRendezvous(
                            sems, None, card=int(str(lane_key)[1:]),
                            budget_s=WEG2_GROUP_FENCE_BUDGET_S)
                    got = 0
                    for _ in range(outstanding):
                        if not rv.wait_drained(tag="leg-end"):
                            break
                        got += 1
                    logger.info("WEG2-SEQ leg-end drain lane=%s outstanding=%d "
                                "drained=%d", lane_key, outstanding, got)
                except Exception as exc:  # noqa: BLE001
                    logger.info("WEG2-SEQ leg-end drain lane=%s failed: %s",
                                lane_key, exc)
        bx.release_stage_buffers(None, log=logger.info)
        # xsn265: the depositor's host lane buffers go at its leg end.
        # xsn266: UNMAP ONLY, NEVER TRUNCATE HERE -- D-TP0 died of SIGSEGV
        # mid-collect (lane c0, piece 3/224) while PP0, the depositor, had
        # already passed this leg end and cut the files to 0: the leg-end
        # drain wait above consumes the lane's PRIMED credits as readily as
        # real drains, so it is not a proof that the collector is done. The
        # COLLECTOR is the last reader and truncates after its own leg end
        # (resume_memory_occupation, after the wake worker joined).
        if bx.seq_release_lanes():
            bx.release_host_lane_buffers(truncate=False, log=logger.info)

    def _weg2_bar1_order_key(self, tag):
        """(flip index, index of the tag in the wake order) or None."""
        try:
            fi = self._weg2_flip_index_now
            order = self._weg2_leg_tag_order or []
            if fi is None or int(fi) < 0 or str(tag) not in order:
                return None
            return (int(fi), order.index(str(tag)))
        except Exception:  # noqa: BLE001 -- stubs without the fields
            return None

    def _weg2_collect_lane_is_bar1(self, lane_key: str, pair) -> bool:
        """H89: the collector's tag-order gate reads the depositor's mapping,
        not only this side's window (bar1_lanes.collect_gate_is_bar1)."""
        from sglang.srt.weg2 import bar1_lanes as b1
        return b1.collect_gate_is_bar1(getattr(self, "_weg2_bar1", None), lane_key, pair)

    def _weg2_bar1_register(self, tag) -> None:
        """Main thread, in tag order: the tag is pending on every BAR1 lane
        this rank receives on (bar1_lanes.register_turns)."""
        try:
            b1 = self._weg2_bar1
            if b1 is not None:
                b1.register_turns(self._weg2_bar1_order_key(tag))
        except Exception:  # noqa: BLE001
            pass

    def _weg2_bar1_release(self, tag, used=None) -> None:
        try:
            b1 = self._weg2_bar1
            if b1 is not None:
                b1.release_turns(self._weg2_bar1_order_key(tag), used)
        except Exception:  # noqa: BLE001
            pass

    def _weg2_turn_index(self, tag) -> int:
        """The tag's index in this wake leg's order, -1 when it has none."""
        order = self._weg2_leg_tag_order or []
        return order.index(str(tag)) if str(tag) in order else -1

    def _weg2_turns_register(self, tag) -> None:
        """H11, main thread, in tag order: the tag is pending on every
        host/IPC lane until its collect says which lanes it uses."""
        turns = self._weg2_lane_turns
        idx = self._weg2_turn_index(tag)
        if turns is not None and idx >= 0:
            turns.register(idx)

    def _weg2_turns_leave(self, tag, lane: str) -> None:
        """H11: the tag's run on ``lane`` is over."""
        turns = self._weg2_lane_turns
        idx = self._weg2_turn_index(tag)
        if turns is not None and idx >= 0:
            turns.leave(lane, idx)

    def _weg2_turns_release(self, tag, used=None) -> None:
        """H11: drop the tag from the lanes it does not use (``used``) or
        from every lane (its collect is over)."""
        turns = self._weg2_lane_turns
        idx = self._weg2_turn_index(tag)
        if turns is not None and idx >= 0:
            turns.release(idx, used)

    def _weg2_preload_hold(self) -> int:
        """18.09. (Flip-Schwanz): the held requests' host pages start loading
        into the (just resumed) kv pool NOW -- the same match + init_load_back
        the adder runs at init_new, on the same thread (the wake handler IS
        the scheduler thread), so the first extend pass finds them on the
        device. Returns the number of requests with a load issued."""
        sched = self.scheduler
        hold = list(getattr(sched, "weg2_dormant_hold", None) or []) if sched is not None else []
        if not hold:
            return 0
        from sglang.srt.managers import schedule_policy as sp
        tree = sched.tree_cache
        n = 0
        t0 = time.perf_counter()
        for req in hold:
            try:
                sp.match_prefix_for_req(tree, req, include_req=True)
                ext = sp._pp_load_back_extent(req)
                if not ext:
                    logger.info("WEG2-PRELOAD rid=%s no host extent (device hit %d)",
                                str(getattr(req, "rid", "?"))[:12], len(getattr(req, "prefix_indices", []) or []))
                    continue
                res = tree.inc_lock_ref(req.last_node)
                dec = res.to_dec_params() if tree.is_tree_cache() else None
                old_last = req.last_node
                try:
                    req.mamba_loadback_anchor_adopted = False
                    new_indices, req.last_node = tree.init_load_back(
                        sp.InitLoadBackParams(best_match_node=req.best_match_node,
                                              host_hit_length=ext, req=req))
                finally:
                    if dec is not None:
                        tree.dec_lock_ref(old_last, dec)
                n += 1
                logger.info("WEG2-PRELOAD rid=%s extent=%d issued=%d tokens",
                            str(getattr(req, "rid", "?"))[:12], int(ext), int(new_indices.numel()))
            except Exception as exc:  # noqa: BLE001 -- init_new loads it later as before
                logger.info("WEG2-PRELOAD rid=%s skipped: %r", str(getattr(req, "rid", "?"))[:12], exc)
        logger.info("WEG2-PRELOAD held=%d issued=%d ms=%.0f", len(hold), n, (time.perf_counter() - t0) * 1000)
        return n

    def _weg2_tag_done_set(self, tag) -> None:
        """Mark this tag's collect as through for the tag-order gate."""
        try:
            order = self._weg2_leg_tag_order or []
            events = self._weg2_tag_done or {}
            if str(tag) in order:
                ev = events.get(order.index(str(tag)))
                if ev is not None:
                    ev.set()
        except Exception:  # noqa: BLE001 -- stubs without the fields
            pass

    def _weg2_wake_collect_one(self, tag) -> None:
        """One tag's collect on the wake side (the exchange carrier), run
        inline or on the wake worker (2026-09-15, Punkt 2)."""
        from sglang.srt.managers.weg2_memory_saver import (
            GPU_MEMORY_TYPE_WEIGHTS_DRAFT as _DRAFT_TAG,
        )
        try:
            self._weg2_wake_inflight = True
        except AttributeError:
            pass
        try:
            _collected = self._weg2_xchg_inject_weights(tag=tag)
        finally:
            try:
                self._weg2_wake_inflight = False
            except AttributeError:
                pass
            self._weg2_bar1_release(tag)       # every lane: this tag's collect is over
            self._weg2_turns_release(tag)
            self._weg2_tag_done_set(tag)
        self._weg2_xchg_collected_per_tag = True
        self._weg2_seam_after_part(tag)
        if (str(tag) == str(_DRAFT_TAG)
                and not _collected
                and self._weg2_xchg_draft_reload_from_disk()):
            pass  # the exchange carried nothing for this tag

    @staticmethod
    def _weg2_seam_after_part_items(tag, inventory, nw_keys, w_names):
        """The pieces the after-part digest of `tag` may read NOW, on the
        collect thread, while the main thread resumes the next tag.

        #1405 excluded a foreign tag's NO-WRITE pieces (aliases of target
        tensors in the `weights` region). #1418 (boots xsn165/xsn170, D TP0,
        first wake, 2 of 12 boots): the draft pieces that SHARE their name
        with a `weights` piece (lm_head.weight, model.norm.weight -- measured
        shares that land with the target's own leg, xsn115) are the same
        alias class and were still read under `weights_draft`; the `weights`
        remap on the main thread made them an illegal address. They are
        graded under `weights`, where the refold clause already includes
        them, so the draft part leaves them out too.
        """
        tag = str(tag)
        out = []
        for idn, t in inventory:
            itag, name = str(idn.tag), str(idn.name)
            if itag == tag:
                if tag == "weights":
                    out.append((idn, t))
                elif (itag, name) in nw_keys:
                    continue
                elif itag == "weights_draft" and name in w_names:
                    continue
                else:
                    out.append((idn, t))
            elif tag == "weights" and (
                (itag, name) in nw_keys or (itag == "weights_draft" and name in w_names)
            ):
                out.append((idn, t))
        return out

    def _weg2_seam_after_part(self, tag) -> None:
        """Punkt 3: fold THIS tag's pieces now (on the wake worker, while
        the main thread resumes the next tag); the leg's `after` reading is
        assembled from the parts. Any failure leaves the part out -- the
        assembly then falls back to the whole walk, never to a gap."""
        from sglang.srt.weg2 import seam_digest
        try:
            if not (_weg2_seam_per_tag_armed() and seam_digest.seam_digest_armed()):
                return
            parts = getattr(self, "_weg2_seam_after_parts", None)
            if parts is None:
                return
            inv = getattr(self, "_weg2_seam_leg_inventory", None)
            if inv is None:
                inv = self._weg2_seam_inventory()
                self._weg2_seam_leg_inventory = inv
            inventory = inv[0]
            if inventory is None:
                return
            # xsn114: the draft's NO-WRITE pieces (the target's embed, see
            # _weg2_xchg_no_write) get their bytes with the target's own
            # `weights` tag, so they are folded again with that tag and the
            # later part overrides the draft part's stale entry.
            # xsn115: the same holds for every draft piece that SHARES its
            # name with a weights piece on this rank (lm_head.weight is
            # excluded from the draft join as a measured share and lands
            # with the target's weights tag) -- refold those with `weights`.
            _nw = getattr(self, "_weg2_xchg_no_write", None) or frozenset()
            _nw_keys = {(str(t), str(n)) for (t, n) in _nw}
            _w_names = {str(idn.name) for idn, _t in inventory if str(idn.tag) == "weights"}
            # #1405 (boots xsn136/xsn141-class without the coverage tracer): a
            # NO-WRITE piece of another tag is an ALIAS of a target tensor
            # whose memory lives in the `weights` region -- resumed by the
            # main thread while this digest runs on the collect thread. Read
            # before that resume it is an unmapped address (xsn136: D TP0
            # "after-part tag=weights_draft FAILED ... illegal memory
            # access", then the BAR1 poll died). The tracer only hid the race
            # by slowing this thread down. Alias pieces are graded under
            # `weights`, where the clause below already includes them.
            items = self._weg2_seam_after_part_items(tag, inventory, _nw_keys, _w_names)
            if not items:
                return
            _grp, _rk, _card = self._weg2_group_name(), self._weg2_rank(), self._weg2_device_index()

            def _run(_items=items, _tag=str(tag), _parts=parts):
                try:
                    part = seam_digest.take_reading(
                        "after-part", _items, group=_grp, rank=_rk, card=_card,
                        tags=[_tag], epoch=None, budget_s=0.0)
                    _parts[_tag] = (part, {p.identity.key: p for p in part.pieces})
                    logger.info("WEG2-SEAM-DIGEST stage=after-part tag=%s pieces=%d ms=%.0f",
                                _tag, len(part.pieces), part.ms)
                except Exception as exc:  # noqa: BLE001 -- an observer never takes the leg down
                    logger.info("WEG2-SEAM-DIGEST stage=after-part tag=%s FAILED %s: %s",
                                _tag, type(exc).__name__, exc)

            # #1437 (xsn196, D wake leg 1800 ms of which ~100 ms per tag was
            # THIS reading, on the critical path of every resume): the
            # after-part digest is an observer, the pieces it reads are
            # resident and static until the next sleep -- take it on a side
            # thread and join before the verdict. SGLANG_WEG2_SEAM_DIGEST_ASYNC=0
            # keeps it inline.
            if os.environ.get("SGLANG_WEG2_SEAM_DIGEST_ASYNC", "1") == "1":
                import threading as _threading
                th = _threading.Thread(target=_run, name=f"seam-after-{tag}", daemon=True)
                lst = getattr(self, "_weg2_seam_after_threads", None)
                if lst is None:
                    lst = self._weg2_seam_after_threads = []
                lst.append(th)
                th.start()
            else:
                _run()
        except Exception as exc:  # noqa: BLE001 -- an observer never takes the leg down
            logger.info("WEG2-SEAM-DIGEST stage=after-part tag=%s FAILED %s: %s",
                        tag, type(exc).__name__, exc)

    def _weg2_cocard_peer_alive(self) -> bool:
        """Is the co-located rank on THIS card still alive?  NVML per-process
        pids on the rank's own device, minus self -- during a flip the OTHER
        compute process on this card is the pair's sibling rank.

        #1378 xsn36/37 (the coordinator's requirement (a)): the collect's
        wait must distinguish "the deposit is slow" (keep waiting, within
        the 120 s budget) from "the deposit rank is DEAD" (die NOW, named).
        FAIL-OPEN: any instrument error answers True -- keep waiting within
        the budget; the 120 s rendezvous budget remains the hard bound and
        the detector (the coordinator's requirement (b)).

        fnFL2x100: A PEER WHOSE LEG ALREADY DIED IS NOT COMING either. Its
        process may well be alive (x100's D TP1 sat in its group fence for
        121 s), so NVML says nothing; its posted abort does, and ends this
        wait at the next poll instead of at the budget.
        """
        if self._weg2_foreign_leg_aborts():
            return False
        try:
            uuid_key = resolve_pcie_lock_key()
            import pynvml  # noqa: PLC0415
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid_key)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            me = os.getpid()
            others = {int(pr.pid) for pr in procs} - {me}
            if others:
                return True
            # #79 (fnFL2w5, 21.09.): EIN SCHLAFENDER RANG IST KEIN TOTER RANG.
            # NVML listet nur Prozesse, die auf der Karte ALLOKIERT haben. Die
            # schlafende Gruppe gibt beim Sleep genau das frei -- sie faellt
            # aus dieser Liste, ohne zu sterben. Der Collect las das als Tod:
            #
            #   W68: 3 lane(s) refused: c0/weights_1: PeerGone at unit 0
            #   'model.layers.3.attn_hyper_connection.block_inject_weight...'
            #
            # und riss ueber W29 (resume_memory_occupation) alle sechs Raenge
            # mit, obwohl bis zur selben Sekunde 6/6 Scheduler liefen. Und der
            # Flip legt die Quelle IMMER schlafen (WEG2-FLIP begin sleep=D
            # wake=P), also trifft es jeden Flip, nicht einen Sonderfall.
            #
            # Eine leere NVML-Liste ist damit KEIN Todesbeweis mehr. Sie wird
            # erst einer, wenn auch kein Scheduler-Prozess mehr lebt -- dann
            # ist wirklich niemand mehr da, der deponieren koennte. Bis dahin
            # gilt weiter das Budget als harte Schranke (die urspruengliche
            # Anforderung (b) aus #1378 xsn36/37), und der Irrtum faellt auf
            # die sichere Seite: warten statt einen lebenden Peer erschiessen.
            return _any_scheduler_process_alive()
        except BaseException:  # noqa: BLE001 -- fail-open, budget is the bound
            return True

    def _weg2_leg_pcie_uuid(self):
        """THIS rank's NVML uuid for the leg's per-copy locks, fail-soft.

        A stubbed caller (the smoke/replay harnesses borrow this path) has no
        card cache -- it gets ``None`` and the leg runs unserialised, exactly
        like the pre-lock boots. Never raises: the leg must not die on an
        instrument."""
        try:
            return self._weg2_card_uuid()
        except BaseException:  # noqa: BLE001
            return None

    def _weg2_rank_param_table(self):
        """Every live parameter THIS PROCESS holds, across BOTH its runners.

        BOOT weg2xsn24's FIRST ROOT: the collect hook refused with
        ``W74 bounce: fc.weight has no destination pointer ...
        dst_resolved=894/904``. ``fc.weight`` is the DRAFT head's, and the
        address book was built from ``self.tp_worker.model_runner`` alone --
        one runner. But the manifest join unions a rank's runner files
        (`merge_region_tags`, added for weg2xsn20's two-runner write), so the
        PLAN covers both runners while the ADDRESSES covered one. Ten
        descriptors of 904 had no home, and one hole refuses the whole leg.

        The two populations must therefore be the same two: this walks the main
        runner AND the draft runner, exactly as `arm_coverage_at_load` is
        reached once per runner. `_get_draft_model_runner` is the tree's own
        accessor (DFlash / FrozenKVMTP shapes); absent, the table is the main
        runner's and the join simply has nothing draft-shaped to place.
        """
        # KEYED BY (REGION, NAME) -- the fourth of the four sites that share
        # one key. `setdefault` on the NAME alone made the winner depend on
        # walk ORDER once two runners of a rank share a parameter name, which
        # weg2xsn25 measured eight times (the drafter is a one-layer block, so
        # its parameters are `model.layers.0.*` too).
        table = {}
        runners = []
        main = getattr(self.tp_worker, "model_runner", None)
        if main is not None:
            runners.append(main)
        drafter = _weg2_drafter_of(self)
        if drafter is not None:
            runners.append(drafter)
        from sglang.srt.weg2 import weight_exchange as _wx
        from sglang.srt.weg2 import xchg_manifest as _xm

        for runner in runners:
            model = getattr(runner, "model", None)
            if model is None:
                continue
            if runner is not main:
                # #1378 xsn77: the drafter is filed under the draft region by
                # the same rule the manifest writer and the draft plan use
                # (see _weg2_xchg_draft_plan_or_none). On P the classifier
                # raises (no speculative config) and the old fallback filed
                # the draft's tensors under `weights`, where the target's
                # names shadow them and the draft-only ones are unreachable.
                from sglang.srt.managers.weg2_memory_saver import (
                    draft_tag_in_family as _dtif,
                )
                if _dtif():
                    region = _xm.region_of_tag(_wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
                    try:
                        for name, param in model.named_parameters():
                            if getattr(getattr(param, "device", None), "type", "") == "meta":
                                continue  # xsn389: a solo-shadow meta draft holds no bytes
                            table.setdefault((region, str(name)), param)
                    except BaseException:  # noqa: BLE001
                        pass
                    continue
            try:
                region = _xm.region_of_tag(
                    _wx.weights_region_tag_for(_wx.RunnerShape.of(runner)))
            except BaseException:  # noqa: BLE001 -- an unclassified shape
                region = _wx.GPU_MEMORY_TYPE_WEIGHTS
            try:
                for name, param in model.named_parameters():
                    if getattr(getattr(param, "device", None), "type", "") == "meta":
                        continue  # xsn389: a meta parameter has no device address
                    table.setdefault((region, str(name)), param)
            except BaseException:  # noqa: BLE001
                continue
            # PLATZTAUSCH: die Experten-Puffer (#135) stehen seit #135-#139 in
            # Manifest, Plan und Coverage -- aber nie in DIESEM Adressbuch, das
            # nur `named_parameters()` las. Ihre Deskriptoren hatten damit auf
            # keiner Seite einen Zeiger (gebaut, nie erreicht). Dieselbe
            # Quelle wie das Manifest (`expert_buffer_tensors`), damit Name und
            # Zeiger aus EINEM Walk kommen.
            try:
                from sglang.srt.weg2.weight_exchange_shadow import (
                    expert_buffer_tensors as _ebt,
                )

                for name, tensor in _ebt(model):
                    table.setdefault((region, str(name)), tensor)
            except BaseException:  # noqa: BLE001
                pass
        return table

    def _weg2_join_src_addr(self, hook: str, group: str, rank: int, model,
                            region: str = ""):
        """#1330 B4n. The SOURCE address book for a join-backed leg.

        OWN DEVICE ADDRESS ONLY, and never the ring. On the SOURCE hook this
        rank holds the bytes it must supply, so ``data_ptr()`` of its own live
        tensor is the honest answer and the deposit reads from there. On every
        other hook the source is the bounce slot, which the collect addresses
        itself; this returns ``None`` there.

        Reading the source out of the ring would be the ring restore through
        another door and would defeat the goal the exchange exists for (zero
        layer bytes resident in host RAM), so the ring appears nowhere here.
        """
        if str(hook) != "source":
            return None
        table = self._weg2_rank_param_table()
        if not table:
            return None

        from sglang.srt.weg2 import weight_exchange as _wx
        from sglang.srt.weg2 import xchg_manifest as _xm

        my_region = (_xm.region_of_tag(region) if region
                     else _wx.GPU_MEMORY_TYPE_WEIGHTS)

        def src_addr(name: str, r: int):
            if int(r) != int(rank):
                return None
            tensor = table.get((my_region, str(name)))
            return None if tensor is None else int(tensor.data_ptr())

        return src_addr

    def _weg2_join_dst_addr(self, hook: str, group: str, rank: int, model,
                            region: str = ""):
        """#1330 B4n. The DESTINATION address book for a join-backed leg.

        The mirror of the source book, over the SAME two-runner population --
        see :meth:`_weg2_rank_param_table` for why one runner was not enough.

        THE COLLECT'S TARGET IS A PLAN PARAMETER, NOT A REFILL DERIVATION, and
        that is deliberate for the remap slice (AMENDMENT 8): when the remap
        replaces WHERE the destination pages come from, only this book changes
        -- the deposit and the collect stay the same two primitives.
        """
        if str(hook) == "source":
            return None
        table = self._weg2_rank_param_table()
        if not table:
            return None

        from sglang.srt.weg2 import weight_exchange as _wx
        from sglang.srt.weg2 import xchg_manifest as _xm

        my_region = (_xm.region_of_tag(region) if region
                     else _wx.GPU_MEMORY_TYPE_WEIGHTS)

        def dst_addr(name: str, r: int):
            if int(r) != int(rank):
                return None
            tensor = table.get((my_region, str(name)))
            return None if tensor is None else int(tensor.data_ptr())

        return dst_addr

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
            # #1348 (review MF-1): THE ARM GOES BEFORE THE LANE GATE.
            #
            # It used to sit ~55 lines further down, after this gate, after the
            # identity gate and after the device gate -- so a rank whose hook
            # returned early wrote NO dump, and an ingest that iterated the
            # dumps it FOUND said nothing whatsoever about that rank. That is
            # the #1329 shape itself (the shadow returned early for three
            # boots): group P reports, group D is absent from the report, and
            # the reader takes P's reading list for the boot's.
            #
            # Arming here leaves a `legs=0` dump behind for every rank that
            # reaches this hook at all, which is what turns "this rank ran no
            # leg" into a printable fact instead of a silence. The tracer is
            # NOT started by the arm (it is bracketed to the leg below), so
            # this costs a rank on the ring arm nothing but one JSON write.
            #
            # Guarded on `armed_by_launcher()` -- a single env read -- so an
            # unarmed boot does not even resolve group and rank here.
            # The phase stamps (WEG2-XCHG-HOOK-TIME) exist for EVERY boot:
            # boot xsn135 (2026-09-15, the first without --xchg-coverage-diff)
            # had them defined only under the coverage arm below, so the
            # first `_hk_ph(...)` raised UnboundLocalError, the hook's
            # observer-except swallowed it as "the flip is unaffected", P
            # never deposited, and D's wake waited until the front's W4.
            _hk_t = [time.perf_counter()]
            _hk_l = []

            def _hk_ph(name):
                _n = time.perf_counter()
                _hk_l.append((name, (_n - _hk_t[0]) * 1000))
                _hk_t[0] = _n

            if wlc.armed_by_launcher():
                wlc.arm(group=self._weg2_group_name(), rank=self._weg2_rank())
                _hk_ph("arm")
            if not sh.bounce_lane_armed():
                return
            from sglang.srt.weg2 import weight_exchange as wx_legs
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
            # #1330 B4n: THE DIRECTION KNOB, and it is checked HERE -- after
            # the identity is known (the direction is a function of hook AND
            # group) and BEFORE any device, region or semaphore is opened, so
            # a skipped direction costs the leg nothing at all.
            #
            # THE SKIP IS NAMED.  A direction that is off prints its count, its
            # reason and BOTH directions; silence is what made four boots of
            # this campaign unreadable, and a skipped leg reads in a log
            # exactly like a leg that ran clean.  Under the default `both` this
            # branch is never taken and the leg is byte-identical.
            #
            # THE COUNTER IS THE MODULE'S, NOT THIS OBJECT'S.  The mixin is a
            # `slots=True` dataclass: `self._weg2_legs_skipped = ...` would
            # raise AttributeError inside the flip leg -- which is #1329's own
            # shape, a first write that raised on a slots dataclass and cost
            # three boots.  `weight_exchange` keeps the cumulative count beside
            # the pointer-profile legs, for the same reason it keeps those.
            if not wx_legs.leg_enabled(str(hook), str(group)):
                logger.info("%s", wx_legs.legs_skipped_line(
                    wx_legs.record_leg_skipped(),
                    hook=str(hook), group=str(group)))
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
            _hk_ph("free_bytes")
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
            _hk_ph("census")
            # #1311 S6b -- THE CARD MANIFEST, BEFORE THE PLAN AND NOT AFTER.
            # The plan is narrowed to the pair's agreed piece set, and
            # ``coalesce`` merges descriptors across parameter names, so the
            # narrowing has to happen on the INVENTORY inside the derivation.
            # That is why the agreement is reconciled here, one call earlier,
            # rather than inside ``run_leg_hook`` where the region is opened
            # for the transport.
            agreed, manifest_state = self._weg2_shadow_manifest(
                group, peer, int(rank), leg=leg, epoch=epoch_token)
            _hk_ph("manifest")
            # #1345: THE SHADOW LANE NARROWS, and says so at the call rather
            # than relying on the adapter.  This is the danger direction of
            # that slice: a shadow leg planned over a set the peer never agreed
            # to is boot weg2xsn5's W80 exactly.
            plan, plan_reason = self._weg2_shadow_plan(str(hook), group,
                                                       int(rank),
                                                       agreed=agreed,
                                                       require_agreement=True)
            _hk_ph("plan")
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
            _hk_ph("inputs_gate_budget")
            # #1348 (review MF-4): THE TRACER IS ON ONLY FOR THE LEG.
            # sys.settrace is paid by every Python call in the process, not
            # just the allowlisted files (measured: 4.04x on NON-allowlisted
            # code), so leaving it on after the leg confounded every other
            # figure of the boot and ate into the 2.0 s / 20.0 s wall-clock
            # ring deadlines. In, run, out.
            wlc.begin_leg(hook)
            try:
                # NUTZER-ORDER 2026-09-15 (Beschleunigung): the region-slot
                # shadow grader ran into a stale slot on PP0/PP2 every leg
                # (W69 after its 5 s budget, weg2xsn84-87) -- 5 s of every
                # flip for a grader the sequential transport's own witnesses
                # (identity record, SEAM-DIGEST) have replaced. Off unless
                # SGLANG_WEG2_XCHG_SHADOW_HOOK=1.
                if bx_mod_shadow_hook_armed():
                    sh.run_leg_hook(inputs, log=logger.info, plan=plan)
                else:
                    logger.info(
                        "WEG2-XCHG-SHADOW-HOOK off hook=%s leg=%s -- "
                        "SGLANG_WEG2_XCHG_SHADOW_HOOK is not 1; the "
                        "sequential transport's identity record and the "
                        "SEAM-DIGEST are the witnesses of this leg",
                        hook, getattr(inputs, "leg", "?"))
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
                # #1348: THIS LEG'S READING IS ON DISK BEFORE THE NEXT ONE
                # STARTS. In the `finally` on purpose -- the legs whose
                # coverage matters most are the ones that RAISED, and a dump
                # written only on the success path is absent on exactly the
                # boots this instrument exists for. A rank that dies between
                # two legs still leaves every leg it completed behind.
                wlc.note_leg_end(hook)
                _hk_ph("leg_hook")
                logger.info("WEG2-XCHG-HOOK-TIME hook=%s " + " ".join(f"{_n}={_ms:.0f}" for _n, _ms in _hk_l) + f" t={time.time():.3f}", hook)
        except BaseException as exc:  # noqa: BLE001 -- an observer never raises
            # #1403 (xsn135): this line used to say "the flip is unaffected".
            # It was wrong the one time it mattered -- on the exchange arm the
            # sequential transport's deposit runs INSIDE this hook, so a raise
            # here is a leg that never deposited and a peer that waits for it
            # until the front's W4. Say so, at ERROR, with the traceback.
            logger.error(
                "[weg2 shadow] the %s hook RAISED (%s: %s) -- on the exchange "
                "arm the transport's deposit/collect runs in this hook, so "
                "this leg may not have moved its bytes and the peer's wake "
                "will wait for them (xsn135: W4 after 200 s). The ring, where "
                "armed, stays the authority for weight bytes.",
                hook, type(exc).__name__, exc, exc_info=True,
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
        """This card's pieces: ``(inventory, skipped, walked, reason)``.

        Delegated to ``weight_exchange_shadow.card_inventory``, which is the
        tree's ONE producer of a card's placement.  The grader deliberately
        does not walk ``named_parameters()`` itself: placement is decided by
        the loader at boot and already published, and a second inventory here
        would be second bookkeeping beside it (operator direction 2026-09-12).

        BOTH RUNNERS, not just the target model (#1350 F2 / review R4).  The
        MTP/NEXTN draft shard is flipped like every other layer since B4k
        (``weg2_memory_saver.weights_family_tags`` puts ``weights_draft``
        between the chunks and the base tag), so its bytes are bytes the
        EXCHANGE MOVES -- 1440/1280/1280 MiB on D's three ranks.  Grading the
        target model alone meant those bytes carried no verdict while the line
        said MATCH, and the only thing naming the gap was a word in the
        population prose.  The draft runner is walked through the SAME producer
        under its own region tag, so it is one walk, not a second inventory.
        """
        try:
            from sglang.srt.managers.weg2_memory_saver import (
                GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
            )
            from sglang.srt.weg2 import weight_exchange_shadow as shadow

            rank = self._weg2_rank()
            runner = getattr(self.tp_worker, "model_runner", None)
            model = getattr(runner, "model", None)
            inv, skipped, walked, reason = shadow.card_inventory(
                rank=rank, model=model
            )
            if inv is None:
                return None, skipped, walked, reason
            draft_runner = _weg2_drafter_of(self)
            draft_model = getattr(draft_runner, "model", None)
            # ONLY WHERE THE DRAFT IS ACTUALLY A FAMILY MEMBER.  Under the
            # `exchange` arm `weights_draft` joins the family and is flipped as
            # its own member (B4k), which is what makes it gradable here.  Under
            # `ring` the drafter never receives its own tag -- it is paused as
            # part of the base tag and travels the host ring -- and forcing a
            # region tag on it would be the region-blind reading
            # `tag_of_parameter_name` warns about, i.e. a census over a family
            # tag that does not exist on that arm.  So on `ring` it is a NAMED
            # skip with its reason, never a silent omission and never a
            # mis-tagged grade.
            from sglang.srt.managers.weg2_memory_saver import draft_tag_in_family

            if draft_model is not None and not draft_tag_in_family():
                skipped = list(skipped) + [
                    ("<draft-runner>", "not-in-family-on-this-arm")
                ]
            elif draft_model is not None:
                d_inv, d_skipped, d_walked, d_reason = shadow.card_inventory(
                    rank=rank, model=draft_model,
                    region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                )
                walked += d_walked
                skipped = list(skipped) + list(d_skipped)
                if d_inv:
                    inv = list(inv) + list(d_inv)
                else:
                    # A draft runner the producer cannot walk is NAMED, never
                    # dropped: its bytes move either way.
                    skipped.append(("<draft-runner>", d_reason or "no-inventory"))
            return inv, skipped, walked, ""
        except Exception as exc:  # noqa: BLE001 -- an observer never raises
            return None, [], 0, f"inventory-failed:{type(exc).__name__}:{exc}"

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
        inventory, skipped, walked, reason = self._weg2_seam_inventory()
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
        _ref = getattr(self, "weg2_seam_ref", None)
        # weg2xsn97: the before reading is taken under the leg's FIRST
        # release request, which on the waking side carries only kv_cache/
        # cuda_graph (no weights tags) -- so the tag field is NOT part of
        # the reading's identity; the placement key (checked by compare)
        # and the tensor count are.
        try:
            _n_inv = len(inventory)
        except TypeError:
            _n_inv = -1
        if _ref is not None and not (_weg2_seam_reuse_armed()
                                     and int(getattr(_ref, "n_tensors", -1)) == _n_inv):
            logger.info("WEG2-SEAM-DIGEST stage=before NOT-REUSED reuse=%s ref_n=%s "
                        "inventory_n=%s", _weg2_seam_reuse_armed(),
                        getattr(_ref, "n_tensors", None), _n_inv)
        if (_ref is not None and _weg2_seam_reuse_armed()
                and int(getattr(_ref, "n_tensors", -1)) == _n_inv):
            self.weg2_seam_before = _ref
            logger.info(
                "WEG2-SEAM-DIGEST stage=before REUSED group=%s rank=%s "
                "digest=%s epoch=%s n_tensors=%d -- this rank's bytes were "
                "graded at that reading and serving does not write weights; "
                "the walk (~0.8 s per 18 GB) is spent once per leg, on the "
                "waking side's `after` (SGLANG_WEG2_SEAM_REUSE=0 walks here too)",
                self._weg2_group_name(), self._weg2_rank(),
                getattr(_ref, "digest", "?"), getattr(_ref, "epoch", "?"),
                len(inventory))
            return
        reading = seam_digest.take_reading(
            "before",
            inventory,
            group=self._weg2_group_name(),
            rank=self._weg2_rank(),
            card=self._weg2_device_index(),
            tags=weights_tags,
            epoch=getattr(recv_req, "epoch", None),
            skipped=skipped,
            walked=walked,
        )
        self.weg2_seam_before = reading
        self.weg2_seam_ref = reading
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
        inventory, skipped, walked, reason = self._weg2_seam_inventory()
        if inventory is None:
            logger.info(
                "%s",
                seam_digest.unarmed_verdict(
                    f"no-inventory:{reason}", group=group, rank=rank, card=card,
                    tags=weights_tags,
                ).line(),
            )
            return
        _epoch = getattr(recv_req, "epoch", None)
        _kw = dict(before=before, inventory=inventory, skipped=skipped, walked=walked,
                   group=group, rank=rank, card=card, weights_tags=weights_tags, epoch=_epoch)
        _threads = list(getattr(self, "_weg2_seam_after_threads", None) or [])
        if os.environ.get("SGLANG_WEG2_SEAM_DIGEST_DEFER", "1") == "1" and _threads:
            # #1450: the whole grade -- join, assemble, compare, verdict --
            # leaves the wake RPC.  py-spy on TP0 (boot weg2xsn206, flip P->D):
            # 47 % of the samples sat in the fold while the wake was on the
            # clock.  A refusal is parked on the object and raised at this
            # rank's next leg or idle tick (RAENGE-NIE-UNEINS still crashes,
            # later and named); the wake answers now.
            self._weg2_seam_after_threads = []
            import threading as _threading

            def _finish():
                for _th in _threads:
                    try:
                        _th.join(timeout=60.0)
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    _refusal = self._weg2_seam_digest_finish(**_kw)
                except BaseException as exc:  # noqa: BLE001 -- the grade itself failed: that IS a refusal
                    _refusal = exc
                if _refusal is not None:
                    self.weg2_seam_pending_refusal = _refusal
                    logger.error("WEG2-SEAM-DIGEST DEFERRED REFUSAL epoch=%s: %s -- raised at this rank's "
                                 "next leg or idle tick (#1450)", _epoch, _refusal)

            _fin = _threading.Thread(target=_finish, name=f"seam-finish-{_epoch}", daemon=True)
            self._weg2_seam_finisher = _fin
            _fin.start()
            logger.info("WEG2-SEAM-DIGEST stage=after DEFERRED behind the wake (#1450): %d part "
                        "thread(s) are joined, assembled and compared on a worker; the RPC returns now",
                        len(_threads))
            return
        refusal = self._weg2_seam_digest_finish(**_kw)
        if refusal is not None:
            raise refusal

    def _weg2_seam_digest_finish(self, *, before, inventory, skipped, walked,
                                 group, rank, card, weights_tags, epoch):
        """#1450: join the per-tag part threads, assemble the AFTER reading,
        compare, log -- and RETURN the refusal instead of raising it, so the
        caller decides whether it lands on the RPC (sync form) or on this
        rank's next leg / idle tick (deferred form).  Body unchanged from the
        pre-#1450 tail of _weg2_seam_digest_after."""
        after = None
        # #1437: the after-part readings may still be running on side threads
        for _th in list(getattr(self, "_weg2_seam_after_threads", None) or []):
            try:
                _th.join(timeout=60.0)
            except Exception:  # noqa: BLE001
                pass
        self._weg2_seam_after_threads = []
        _parts = getattr(self, "_weg2_seam_after_parts", None) or {}
        if _parts and _weg2_seam_per_tag_armed():
            _by_key = {}
            _ms = 0.0
            _syncs = 0
            for _p, _m in _parts.values():
                _by_key.update(_m)
                _ms += float(_p.ms)
                _syncs += int(getattr(_p, "syncs", 0))
            # xsn114: the whole walk orders its pieces by read chunking, not
            # by inventory -- the fold is ordered, so the assembly follows
            # the BEFORE reading's piece order (same identities), and only
            # without one the inventory order.
            if before is not None and getattr(before, "pieces", None):
                _keys = [p.identity.key for p in before.pieces]
            else:
                _keys = [seam_digest.identity_of(idn, card=int(card)).key
                         for idn, _t in inventory]
            if all(k in _by_key for k in _keys):
                after = seam_digest.SeamReading(
                    stage="after", group=str(group), rank=int(rank), card=int(card),
                    tags=tuple(str(t) for t in weights_tags),
                    epoch=epoch,
                    pieces=tuple(_by_key[k] for k in _keys), ms=_ms,
                    walked=len(inventory) + len(skipped),
                    skipped=tuple((str(n), str(r)) for n, r in skipped),
                    skipped_bytes=0, syncs=_syncs)
                logger.info("WEG2-SEAM-DIGEST stage=after ASSEMBLED parts=%d "
                            "pieces=%d fold_ms=%.0f -- folded per tag on the wake "
                            "worker; no whole walk on the leg's critical path",
                            len(_parts), len(_keys), _ms)
            else:
                logger.info("WEG2-SEAM-DIGEST stage=after parts=%d cover=%d/%d -- "
                            "whole walk", len(_parts),
                            sum(1 for k in _keys if k in _by_key), len(_keys))
        if after is None:
            after = seam_digest.take_reading(
            "after",
            inventory,
            group=group,
            rank=rank,
            card=card,
            tags=weights_tags,
            epoch=epoch,
            skipped=skipped,
            walked=walked,
        )
        logger.info("%s", after.line())
        verdict = seam_digest.compare(before, after)
        logger.info("%s", verdict.line())
        refusal = verdict.refusal()
        if refusal is not None:
            self.weg2_seam_ref = None
            return refusal
        self.weg2_seam_ref = after
        return None

    def _weg2_release_carrier_hold_at_wake(self) -> int:
        """H81: the wake of a group gives back the hand-over anchors its
        last sleep reset held (``UnifiedRadixCache.weg2_release_carrier_hold``;
        only group P holds any). Host bookkeeping only; never fails a wake.
        Returns the references released."""
        sched = self.scheduler
        tc = getattr(sched, "tree_cache", None) if sched is not None else None
        release = getattr(tc, "weg2_release_carrier_hold", None)
        if not callable(release):
            return 0
        try:
            return int(release("wake") or 0)
        except Exception as exc:  # noqa: BLE001 -- a release never takes the wake down
            logger.warning("WEG2 CARRIER-HOLD wake release raised %s: %s", type(exc).__name__, exc)
            return 0

    def _weg2_wake_restore_pools(self) -> bool:
        """#1455: Scheduler.flush_cache minus tree_cache.reset(): the pool
        state the remap left undefined is restored, the radix tree with the
        hold's prefetched host nodes stays.  Returns True when it ran."""
        sched = self.scheduler
        if sched is None:
            return False
        try:
            sched.req_to_token_pool.clear()
            sched.token_to_kv_pool_allocator.clear()
            try:
                from sglang.srt.environ import envs as _envs  # noqa: PLC0415
                if _envs.SGLANG_FLUSH_ZERO_KV.get():
                    sched._flush_zero_kv_buffers()
            except Exception:  # noqa: BLE001
                pass
            if getattr(sched, "draft_worker", None):
                sched.draft_worker.clear_cache_pool()
            logger.info("WEG2-WAKE-RESTORE pools cleared, radix tree KEPT (#1455: the hold's prefetch survives the wake)")
            return True
        except Exception as exc:  # noqa: BLE001 -- fall back to the full flush, never leave pools undefined
            logger.warning("WEG2-WAKE-RESTORE failed (%s: %s) -> full flush_cache", type(exc).__name__, exc)
            return bool(self.flush_cache())

    def _weg2_raise_pending_seam_refusal(self, *, join: bool = True) -> None:
        """#1450: a refusal graded behind the wake is raised here -- called at
        the head of every Weg-2 leg (join=True: the finisher is WAITED FOR
        first, so no pause overtakes a grade still reading the pages --
        #1450b, boot weg2xsn208) and from the scheduler's idle tick
        (join=False: never blocks the loop; a grade still running is simply
        not graded yet)."""
        fin = getattr(self, "_weg2_seam_finisher", None)
        if fin is not None and fin.is_alive():
            if not join:
                return
            t0 = time.perf_counter()
            fin.join(timeout=180.0)
            logger.info("WEG2-SEAM-DIGEST finisher joined at the next leg in %.0f ms (#1450b)%s",
                        (time.perf_counter() - t0) * 1000,
                        "" if not fin.is_alive() else " -- STILL RUNNING after 180 s")
        if fin is not None and not fin.is_alive():
            self._weg2_seam_finisher = None
        pending = getattr(self, "weg2_seam_pending_refusal", None)
        if pending is not None:
            self.weg2_seam_pending_refusal = None
            raise pending

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

        This is the place in the boot where the ground truth for this wake
        already exists: whatever carrier ``_weg2_wake_reload_weights``
        selected (ring, TMS backup, disk, or -- under
        ``--weg2-weight-source exchange`` + ``--weg2-xchg-inject
        authoritative`` -- the peer group's own live VRAM via the bounce
        lane) has already written the bytes, and the static state is
        imported, so the shadow's pulled stripes have something to be
        compared AGAINST regardless of which carrier that was (CORRECTED
        2026-09-14, #1334: this used to name the ring specifically as the
        only possible ground truth, which is stale once
        ``CARRIER_EXCHANGE`` exists). It runs before the leg reports done,
        so a mismatch appears in the log beside the flip that produced it
        rather than one flip later.

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

    def _weg2_xchg_sems(self):
        """This process's semaphore set for the boot's CROSS pairs, or None.

        ONE HANDLE PER PROCESS, cached: `SemSet` opens each name lazily and
        `sem_open` without O_CREAT is the only legal form in a rank (the class
        says why -- a creating open would silently adopt a name the launcher
        did not make). Two sets would be two handles on one handshake.

        `None` rather than an exception: the caller turns it into the unsplit
        form, where every cross pair is then REFUSED by name. That is the
        honest degradation -- a missing handshake must stop a cross leg, never
        turn it into a `both` leg that asks for the peer's address.
        """
        from sglang.srt.weg2 import weight_exchange as wx

        cached = getattr(self, "_weg2_xchg_sems_cache", "unset")
        if cached != "unset":
            return cached
        sems = None
        try:
            from sglang.srt.weg2 import weight_exchange_transport as tp

            region = self._weg2_shadow_region()
            if region is not None:
                sems = tp.SemSet(region.boot_nonce)
        except BaseException as exc:  # noqa: BLE001
            # #505 SILENT SWALLOW, CLOSED. This was a bare
            # `except BaseException: sems = None`, and it is the reason a
            # cutover leg could lose its handshake without one line saying so:
            # the boot then read `W74 ... src_resolved=0/N` (an ADDRESS
            # complaint) for a cause that was a missing semaphore set. The
            # exception is now named -- type AND text -- at the moment it
            # happens.
            logger.info(
                "WEG2-XCHG-SEMS-UNAVAILABLE type=%s text=%s armed=%s -- the "
                "semaphore set for this boot's CROSS pairs could not be "
                "opened; a leg without it cannot run the deposit/collect "
                "handshake",
                type(exc).__name__, exc, bool(wx.exchange_armed()))
            # AND NOT SWALLOWED WHERE IT DECIDES A FLIP. With the exchange
            # armed, this process IS one of the two weg2 groups and a leg
            # without semaphores is not a degraded observer -- it is a flip
            # that would move weights with no ordering at all. Unarmed boots
            # (`ring`, the pure `shadow` arm) keep the observer behaviour,
            # which is what the `None` return was written for.
            if wx.exchange_armed():
                raise wx.Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: the exchange is armed and "
                    f"this rank's semaphore set could not be opened: "
                    f"{type(exc).__name__}: {exc}. Refusing by name beats "
                    f"returning None, which sends the leg to the unsplit "
                    f"form and reports the missing PEER address instead of "
                    f"the missing handshake."
                ) from exc
            sems = None
        self._weg2_xchg_sems_cache = sems
        return sems

    def _weg2_xchg_undrained_lanes(self, rank: int, sems, *,
                                   covered: Optional[set] = None) -> list:
        """``[(lane, full, empty)]`` for every lane whose DESTINATION is
        ``rank``'s own card and whose ``full`` count is already nonzero.

        #1391 (DESK10): THE CHECK boot weg2xsn31 needed and did not have. The
        #1374 contract keeps a lane's counting ``full`` semaphore at exactly 0
        in the gap between tags -- the collector drains every band of tag t
        before the depositor's ``wait_drained`` for tag t+1 may pass -- so a
        nonzero ``full`` here is never a race and never a matter of timing: it
        is bands a peer already posted that this rank's OWN plan (whatever
        called this) says nothing about. Checked over BOTH families this rank
        could be a destination for: the on-card diagonal (``card=rank``) and
        every directed cross pair whose ``dst == rank``.

        ``covered``, when given, is the SET of ``group_descs_by_pair`` keys
        (``None`` for the diagonal, a ``CROSS_PAIRS`` index for a cross pair)
        this call's OWN descriptors already account for -- those are skipped:
        a lane this leg is about to actually walk is not a silent miss, no
        matter its current count. ``None`` (the default) checks every lane
        unconditionally, which is what a caller with no descriptor set of its
        own (the credit-wait, #1391 round 3) needs.

        ``sems=None`` (the unsplit/hermetic form, or an unopened set) answers
        an empty list rather than raising: this check exists to CATCH a
        collect-side no-op earlier, not to demand a handshake exists where the
        caller's own contract already tolerates its absence.
        """
        if sems is None:
            return []
        from sglang.srt.weg2 import weight_exchange_bounce as bx
        from sglang.srt.weg2 import weight_exchange_region as xr

        slot = int(bx.CrossSlotRendezvous._COUNT_SLOT)
        out = []
        if covered is None or None not in covered:
            try:
                full = sems.diagonal_getvalue(int(rank), slot, "full")
                empty = sems.diagonal_getvalue(int(rank), slot, "empty")
            except OSError:
                full = None
            if full:
                out.append((f"card{int(rank)}-{slot}", int(full),
                            -1 if empty is None else int(empty)))
        for pair, (src, dst) in enumerate(xr.CROSS_PAIRS):
            if int(dst) != int(rank):
                continue
            if covered is not None and pair in covered:
                continue
            try:
                full = sems.getvalue(pair, slot, "full")
                empty = sems.getvalue(pair, slot, "empty")
            except OSError:
                continue
            if full:
                out.append((f"{src}-{dst}-{slot}", int(full), int(empty)))
        return out

    def _weg2_seq_lane_descs(self, *, hook: str, group: str, rank: int,
                             pair: Optional[int], card: Optional[int],
                             tag, log=None) -> list:
        """THE LANE'S OWN DESC LIST, derived from the JOIN -- one producer.

        The xsn52 commit derived the units from the join's raw TENSORS, which
        carries neither the lane (every card would move every tensor) nor the
        shard cut (the buffer would hold the unsharded extents of the whole
        image).  What the two ends of a lane must share is the DESC LIST OF
        THAT LANE, and the join already has a deterministic producer for it:
        ``plan_from_join`` builds the cross-group plan from the manifests
        alone, with no pointer input, so both ranks derive the same list
        without exchanging metadata.  The role-narrowed ``descs`` this method's
        caller received are NOT used for the layout -- they are the two halves
        that diverged (measured on weg2xsn43: P rank0 6 tags/1937 descs, D
        rank0 9 tags/1116 descs) and produced the W90.

        THE LANE KEY is the same predicate ``pair_of`` answers: ``src ==
        dst`` is the on-card diagonal (rank n of either group runs on
        cards[n]), everything else is one of the directed cross pairs.

        THE POINTERS are resolved per side, by the side that owns them: the
        deposit answers ``src_ptr`` for the ranks it holds, the collect
        ``dst_ptr``.  A side that resolves nothing has nothing to move on
        this lane, and the transport refuses that shape rather than silently
        moving zero bytes.
        """
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import xchg_manifest as xm
        from sglang.srt.weg2 import weight_exchange_region as xr

        if (pair is None) == (card is None):
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: the sequential lane needs exactly "
                f"one of pair= (cross) or card= (diagonal), got pair={pair!r} "
                f"card={card!r} -- a lane that cannot say which family it "
                f"belongs to is the xsn52 shape")
        # #1378 xsn55 (THE DIRECTION KEY).  `group` is this leg's GROUP NAME
        # ('P'/'D') -- the same identity `leg_plan_from_join` documents as the
        # direction's second input ("THE DIRECTION IS DERIVED, NOT PASSED",
        # xchg_manifest.py) and the same one this method's own callers refuse
        # on when they cannot answer it.  A caller that hands down anything
        # else -- a desc list, a lane key, a plan -- makes
        # ``leg_direction(hook, group)`` answer from the HOOK ALONE, because
        # ``str(<non-name>)`` is neither "P" nor "D".  That is correct for
        # group D by coincidence (source->tp_to_pp, importing->pp_to_tp are
        # exactly D's two directions) and MIRRORED for group P, which is the
        # measured weg2xsn53 wall: P rank 1's wake (hook=authoritative)
        # planned pp_to_tp, its lane (0,1) then carried P rank 0's 102
        # layer-32..38 descs -- the manifest-true count, on the wrong side --
        # and P rank 1's own address book correctly answered None for every
        # one of them ("102 of 102 descs have no address").  The same swap
        # printed the second wall ("NO desc for lane src=0 dst=2
        # tag='weights_6' (direction=pp_to_tp)") on P rank 2.  Refusing here
        # closes the CLASS: a wrong-typed group can no longer plan a mirrored
        # direction, it names what it was handed instead.
        if str(group) not in ("P", "D"):
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: the sequential lane derivation "
                f"needs the GROUP NAME of the rank running this leg ('P' or "
                f"'D'), got {type(group).__name__}: {str(group)[:96]!r}.  "
                f"leg_direction() answers from (hook, group); a non-name keys "
                f"it off the hook alone, which plans group P's legs in the "
                f"mirrored direction -- the weg2xsn53 102-of-102 'no address' "
                f"wall.  The caller must hand down the identity "
                f"_weg2_group_name() resolves, not a lane's desc list")
        if pair is not None:
            src_card, dst_card = xr.CROSS_PAIRS[int(pair)]
        else:
            src_card = dst_card = int(card)

        _lc = getattr(self, "_weg2_xchg_leg_cache", None)
        if _lc is None:
            _lc = {}
            try:
                self._weg2_xchg_leg_cache = _lc
            except AttributeError:
                pass
        _jk = ("join", str(hook), str(group), int(rank))
        # fnFL2x82: SINGLE-FLIGHT. x80 (D TP0, first wake): the three lane
        # threads of the first tag all missed this key at once and each derived
        # the 20-GiB plan (400/402/420 ms, thread=weg2-lane_0/1/1) -- the
        # first collect started 0,33 s after the first resume. One derives,
        # the others wait for its entry; and `_weg2_warm_leg_cache` fills the
        # key at boot, so the first flip normally finds it.
        _dl = getattr(self, "_weg2_xchg_leg_lock", None)
        if _dl is None:
            import threading as _th
            _dl = _th.Lock()
            try:
                self._weg2_xchg_leg_lock = _dl
            except AttributeError:
                pass
        with _dl:
            if _lc is not None and _jk in _lc:
                join, plan = _lc[_jk]
            else:
                _t_derive = time.perf_counter()
                mans, why = xm.manifests_for_boot(pp_group="P", tp_group="D")
                if mans is None:
                    raise wx.Weg2XchgPlanDisagree(
                        f"W68 Weg2XchgPlanDisagree: the sequential transport could "
                        f"not read the manifests it must derive its lane from: {why}")
                join = xm.join_manifests(mans, pp_group="P", tp_group="D")
                plan = xm.plan_from_join(join, direction=wx.leg_direction(hook, group))
                if _lc is not None:
                    _lc[_jk] = (join, plan)
                # fnFL2x40: this derivation sat on the first deposit's critical
                # path in every lane thread; the line says what it costs now.
                import threading

                logger.info(
                    "WEG2-LANE-DERIVE hook=%s group=%s rank=%s ms=%.0f memo=%s "
                    "thread=%s", hook, group, rank,
                    (time.perf_counter() - _t_derive) * 1000,
                    xm.join_memo_stats(), threading.current_thread().name)

        model = self._weg2_model_for_group(group)
        # THE SAME REGION KEY THE PLAN WAS BUILT WITH: `_weg2_shadow_plan`
        # hands the books `region=region_tag` (the runner's own region tag), so
        # a book built without it would key the table by the default region
        # and answer None for every tensor of a runner whose tag resolves
        # elsewhere -- a refusal that names the wrong mechanism.
        region_tag = ""
        try:
            runner = getattr(self.tp_worker, "model_runner", None)
            if runner is not None:
                region_tag = wx.weights_region_tag_for(wx.RunnerShape.of(runner))
        except BaseException:  # noqa: BLE001 -- an unclassified shape
            region_tag = ""
        if rank is None or int(rank) < 0:
            # The shadow grader and the leg replay call the leg with no rank
            # (whole-plan-at-once); the address books need THIS rank's identity
            # to answer, so take the same resolved identity the instruments
            # use rather than handing the book a None it cannot compare.
            rank = self._weg2_rank()
        if rank is None or int(rank) < 0:
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: the sequential lane derivation "
                f"needs this rank's identity and neither the caller nor "
                f"_weg2_rank() could answer (group={group!r}) -- an "
                f"unaddressable rank cannot say which bytes it holds")
        _bk = ("books", str(hook), str(group), int(rank), str(region_tag))
        if _lc is not None and _bk in _lc:
            src_book, dst_book = _lc[_bk]
        else:
            src_book = self._weg2_join_src_addr("source", group, rank, model,
                                                region=region_tag)
            dst_book = self._weg2_join_dst_addr("destination", group, rank, model,
                                                region=region_tag)
            if _lc is not None:
                _lc[_bk] = (src_book, dst_book)
        # #1378 xsn73 -- ONE BOOK PER REGION OF THE DESC'S OWN TAG. The pair
        # above is keyed by the MAIN runner's region (`weights`), and this
        # method fills every desc from it -- so for the draft tag the book
        # answered the 9 names the draft shares with the target
        # (embed_tokens, input_layernorm, mlp.*, model.norm) and None for the
        # 10 it does not (fc.*, qkv_proj, o_proj, q_norm, k_norm,
        # pre_fc_norm_*: the target's layer 0 is a GDN layer). MEASURED on
        # weg2xsn73 with the W68 census: `book[weights_draft]=20 names`
        # on the very rank that refused "10 of 19 descs have no address".
        # The plan built at :4274 carries no addresses at all, so THIS is the
        # only resolution the lane ever gets; it has to ask the region the
        # desc's tag lives in. Built lazily per region, the default pair
        # stays for a desc without a tag.
        _books = {}

        def _books_for(desc_tag):
            reg = xm.region_of_tag(str(desc_tag)) if desc_tag else ""
            if not reg or reg == xm.region_of_tag(str(region_tag or "")):
                return src_book, dst_book
            if reg not in _books:
                _books[reg] = (
                    self._weg2_join_src_addr("source", group, rank, model,
                                             region=reg),
                    self._weg2_join_dst_addr("destination", group, rank, model,
                                             region=reg),
                )
            return _books[reg]

        # DOES THIS RANK OWN AN END OF THE LANE?  The lane key is a CARD pair
        # and rank n of either group runs on cards[n], so under the leg's
        # direction this rank is the lane's source (source hook, src_card ==
        # its rank) or its destination (every other hook, dst_card == its
        # rank).  A lane whose BOTH ends are other ranks is not this rank's
        # business -- the plan legitimately carries every pair's descs, and a
        # rank that refused those would refuse lanes it never staged bytes
        # for.  Skipping is the honest answer; refusing names a defect that
        # is not there.
        is_source_hook = str(hook) == "source"
        my_rank = int(rank)
        owns_lane = (src_card == my_rank) if is_source_hook else (dst_card == my_rank)

        out = []
        #: #1378 xsn58: how many destinations the plan carried STALE (see the
        #: re-resolve below).  A field-free local on purpose -- it lives for one
        #: lane and is reported on this lane's own line, so a second lane cannot
        #: inherit a count that is not its own.
        _stale_dst = 0
        for d in plan.descs:
            if int(d.src_rank) != int(src_card) or int(d.dst_rank) != int(dst_card):
                continue
            if tag is not None and str(getattr(d, "tag", "")) != str(tag):
                continue
            src_book, dst_book = _books_for(getattr(d, "tag", ""))
            if src_book is not None and d.src_ptr is None:
                d = d.replace(src_ptr=src_book(str(d.param_name),
                                               int(d.src_rank)))
            # #1378 xsn58 -- THE DESTINATION IS RE-RESOLVED, NOT INHERITED,
            # AND A DIVERGENCE IS NAMED.
            #
            # The old condition was `d.dst_ptr is None`, i.e. a pointer the
            # PLAN already carries was kept. But the plan resolves every
            # destination ONCE, before the resume loop -- weg2xsn57 logged
            # `POINTER-PROFILE ... dst_resolved=1937/1937` for all tags at
            # 05:28:37 -- while `memory_saver_adapter.resume(tag)` maps that
            # tag's pages AFTERWARDS, per tag (weight_updater:5671, collect at
            # :5707). So between the plan's reading and this copy-out the
            # pages were re-committed, and `dst_resolved` counts RESOLVED
            # ADDRESSES, never MAPPED PAGES -- the same distinction that has
            # already cost this ticket two walls.
            #
            # xsn57 died exactly here: `collect-first ... dst_ptr=
            # 134250049263104 window_off=0 nbytes=10240` for
            # `layers.0.input_layernorm.weight` (5120 x 2 B, the correct
            # geometry, 10 KB into a 1.97 GB buffer -- source, offset and
            # length all provably right), then SIGSEGV two seconds later.
            #
            # THIS IS A FIX AND A MEASUREMENT AT ONCE, on purpose: if the
            # fresh address equals the planned one, the staleness theory is
            # REFUTED and the line never appears -- a no-op, and the next
            # round does not have to guess again. If it differs, the first
            # divergence is named with both numbers and the theory is proven
            # on the metal. Not a silent repair either way.
            if dst_book is not None:
                _fresh = dst_book(str(d.param_name), int(d.dst_rank))
                if _fresh is not None:
                    if (d.dst_ptr is not None
                            and int(_fresh) != int(d.dst_ptr)):
                        _stale_dst += 1
                        if _stale_dst == 1:
                            log(f"WEG2-SEQ-STALE-DST hook={hook} "
                                f"tag={tag!r} first={d.param_name!r} "
                                f"planned={int(d.dst_ptr)} "
                                f"fresh={int(_fresh)} "
                                f"delta={int(_fresh) - int(d.dst_ptr)} -- the "
                                f"plan's destination was resolved before this "
                                f"tag's resume re-mapped it")
                    d = d.replace(dst_ptr=_fresh)
            out.append(d)
        if not owns_lane:
            return []
        # #1378 xsn54 (DIE ZWEI ZAHLEN NEBENEINANDER): `owned` zaehlt die
        # Descs, deren Name im EIGENEN Manifest dieses Rangs steht; `planned`
        # ist die Lane-Groesse. Lesevorschrift (praezisiert 02:3xZ, der
        # fruehere Verdacht "12 Stuecke Layer 39 auf P rank 0's Lane" ist
        # ZURUECKGEZOGEN -- gemessen: bei pp_to_tp sammelt D legitim von
        # BEIDEN Stages, also ist planned=114 gegen owned=102 auf einem
        # COLLECT-Leg die ERWARTETE Form, weil die Quelle mehrere Stages
        # stellt). Ein Verdacht ist diese Zahl nur auf dem DEPOSIT-Leg:
        # dort nennt ein owned < planned Namen, die dieser Rang
        # hinlegt, ohne sie zu halten.
        
        _own = self._weg2_owned_name_keys()
        owned_n = sum(1 for d in out
                      if (xm.region_of_tag(d.tag), str(d.param_name)) in _own)
        if log is not None:
            log(f"WEG2-SEQ-LANE "
                f"lane={'p%d' % pair if pair is not None else 'c%d' % card} "
                f"owned={owned_n} planned={len(out)} tag={tag!r} "
                # #1378 xsn58: ALWAYS printed, including the 0.  A staleness
                # count that only appears when it is non-zero cannot tell
                # "measured, none found" from "never measured" -- the
                # absence-without-an-emitter trap this campaign keeps paying.
                f"stale_dst={_stale_dst}")
        if not out:
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: the join's plan carries NO desc "
                f"for lane src={src_card} dst={dst_card} tag={tag!r} "
                f"(direction={wx.leg_direction(hook, group)}), which THIS "
                f"rank owns an end of.  An empty owned lane is a plan defect "
                f"here, not an empty one: the caller grouped this lane from "
                f"descs it did hold, so a join that answers nothing for it is "
                f"exactly the two-sides-built-from-different-lists shape that "
                f"the seam digest had to catch on the metal")
        unresolved = [str(d.param_name) for d in out
                      if (d.src_ptr is None if is_source_hook
                          else d.dst_ptr is None)]
        if unresolved:
            # #1378 xsn71: ALL the names, and what the book HOLDS for this
            # region on this rank -- "first: fc.weight" could not say whether
            # the draft runner was missing from the table, filed under another
            # region, or named differently (D rank 0: 10 of 19 unresolved
            # while its own weights_draft manifest listed every one of them).
            try:
                _region = xm.region_of_tag(tag)
                _have = sorted(n for (r, n) in self._weg2_rank_param_table()
                               if r == _region)
                _book = (f" book[{_region}]={len(_have)} names, first: "
                         f"{_have[:6]}; regions in book: "
                         f"{sorted({r for (r, _n) in self._weg2_rank_param_table()})}")
            except BaseException as _bexc:  # noqa: BLE001
                _book = f" book=NOT-READABLE({type(_bexc).__name__})"
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: lane src={src_card} dst={dst_card} "
                f"tag={tag!r}: {len(unresolved)} of {len(out)} descs have no "
                f"address on the side this rank owns (first: "
                f"{unresolved[0]}; all: {unresolved}) -- the address book "
                f"answered None, which means this rank does not hold the bytes "
                f"the lane says it moves.  Copying them would move garbage; "
                f"refusing names the tensors instead.{_book}")
        return out

    def _weg2_owned_name_keys(self) -> set:
        """The (region, name) keys of THIS rank's own manifest, cached per boot.

        The lane audit's `owned=` side: a desc whose (region, name) is not in
        here names bytes this rank does not hold, however many of them the
        plan assigns to it (xsn54: 12 layer-39 pieces on P rank 0's lane).
        """
        cached = self._weg2_owned_name_keys_cache
        if cached is not None:
            return cached
        # #1378 xsn55: `xm` and `region_of_tag` were BARE NAMES here and at
        # the audit's `owned_n` line -- neither is imported at module level
        # (this file's weg2 imports are function-local by design), so the
        # first lane died with `NameError: region_of_tag` inside the audit's
        # comprehension and this method's own read was swallowed by the
        # `except BaseException` below into an EMPTY key set, i.e. an
        # instrument that answered `owned=0` for every lane. Measured by
        # executing both, not by reading them.
        from sglang.srt.weg2 import xchg_manifest as xm

        # THE RANK IS COMPARED, NEVER `or`-DEFAULTED: `_weg2_rank()` answers
        # -1 when it has no identity, and an unknown identity must match
        # nothing -- but ``self._weg2_rank() or -1`` turns the REAL rank 0
        # into -1 too, so rank 0 (the holder of the widest share on every
        # PP boot) would have collected no keys of its own and its audit
        # would have read `owned=0` on every lane. Measured: the audit on a
        # rank-0 manager returned set() for a manifest that carries the name.
        _rank = self._weg2_rank()
        my_rank = int(_rank) if isinstance(_rank, int) and _rank >= 0 else -1

        keys = set()
        try:
            for man in xm.manifests_for_boot(pp_group="P", tp_group="D")[0] or []:
                if my_rank < 0 or int(man.rank) != my_rank:
                    continue
                if str(man.group) != str(self._weg2_group_name() or ""):
                    continue
                for pc in man.pieces:
                    keys.add((xm.region_of_tag(pc.tag), str(pc.param_name)))
        except BaseException:  # noqa: BLE001 -- an audit never takes the leg
            keys = set()
        self._weg2_owned_name_keys_cache = keys
        return keys

    def _weg2_wake_models(self) -> list:
        """fnFL2x38: every model this rank computes with after a wake -- the
        TARGET (tp_worker's runner: MoE layers, Marlin workspaces, expert
        pool) and, when present and distinct, the DRAFT.  The exchange asks
        :meth:`_weg2_model_for_group` for the REGION's runner (D: the draft);
        the rearm and the scratch zeroing must reach the layers that run."""
        out: list = []
        for m in (
            getattr(getattr(getattr(self, "tp_worker", None), "model_runner", None), "model", None),
            self._weg2_model_for_group("D"),
        ):
            if m is not None and all(m is not o for o in out):
                out.append(m)
        return out

    def _weg2_model_for_group(self, group: str):
        """The model runner for the given group."""
        if group == "D":
            worker = getattr(self, "draft_worker", None)
            runner = _get_draft_model_runner(worker) if worker else None
            return getattr(runner, "model", None) if runner else None
        worker = getattr(self, "tp_worker", None)
        runner = getattr(worker, "model_runner", None) if worker else None
        return getattr(runner, "model", None) if runner else None

    def _weg2_xchg_bounce_leg(self, *, descs, ops, boot_nonce,
                              slot_bytes=None, depth=None, terms=None,
                              mode=None, shm_root=None, device: int = 0,
                              #: #1330 B4n SLICE 3.  Which half this rank runs.
                              #: ``None`` derives it from the hook, which is the
                              #: only honest default: a rank on the SOURCE hook
                              #: holds the bytes and deposits them, a rank on
                              #: any importing hook collects them.
                              hook: str = "",
                              region=None, sems=None,
                              #: #1374 F1: WHICH TAG this leg is the step for,
                              #: or None for the whole plan. It names the leg in
                              #: the log and is the key of the per-tag `drained`
                              #: handshake -- the one wait the contract keeps,
                              #: taken between tags and never within one.
                              tag=None,
                              #: #1391 (DESK10) ROUND 3: this rank's own
                              #: resolved identity (`self._weg2_rank()`,
                              #: group-local pp_rank/tp_rank), for the
                              #: undrained-lane check below. ``None`` skips
                              #: the check rather than guessing -- a caller
                              #: with no identity to give is exactly the
                              #: shape ``_weg2_rank`` itself answers ``-1``
                              #: for, and ``-1`` is not a card.
                              rank=None,
                              #: weg2xsn85 (#1378): the lanes this rank's
                              #: WHOLE-LEG plan covers (every tag), for the
                              #: W108 check below. ``descs`` is one tag's
                              #: slice of the lockstep; under a PIPELINE
                              #: source a tag is lane-SPARSE (weights_7 rides
                              #: only the lanes from PP2's card) and the
                              #: source runs ahead by its no-op tags, so a
                              #: lane the plan covers for a LATER tag holds
                              #: that tag's bands while this tag is walked.
                              #: ``None`` keeps the per-tag set (a caller
                              #: without a whole plan).
                              covered_lanes=None):
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
        from sglang.srt.weg2 import weight_exchange_transport as tp

        # THE MODE IS READ ONCE, from the one reader, and passed down.  A leg
        # that re-read it further in could act on a different answer than the
        # one it was entered with, and the two differ by "does this write into
        # the live weights".
        resolved_mode = wx.inject_mode() if mode is None else mode
        root = xr.SHM_ROOT if shm_root is None else shm_root

        # #1330 B4n SLICE 3: SPLIT BY DIRECTED CARD PAIR, because that is how
        # the semaphores are named.  A band that mixed two pairs would post one
        # pair's `full` for another pair's bytes.
        #
        # The DIAGONAL (`src_rank == dst_rank`) keeps `phase=both` and its own
        # on-card carrier: it is one card, both pointers live in reach, and
        # `sem_name` refuses a diagonal id by name (W15) because it has no
        # cross pair at all.  That is not a fallback -- it is the lane the
        # diagonal has always had.
        # `region` IS NO LONGER AN INPUT OF THE PHASED PATH and is not
        # required here: the byte count moved to the bounce's OWN record
        # (`BounceSlots`) precisely because sharing the region's slot records
        # with `run_producer_pair` is what made weg2xsn24 read `carries 1080
        # bytes`. The parameter stays for the existing caller and is unused;
        # requiring it would make a leg fall back to the unsplit form for a
        # dependency it does not have.
        if not hook or sems is None:
            # THE REFUSAL IS HERE, AND IT WAS NOT.  This comment used to say
            # that a CROSS leg reaching the unsplit form "is refused inside
            # `run_bounce_leg` rather than run as `both`".  That was FALSE and
            # is the #1358 class stated at the desk: `_require_rendezvous`
            # returns immediately for `PHASE_BOTH`
            # (weight_exchange_bounce.py:937-938) and nothing below it
            # distinguishes a cross descriptor from a diagonal one, so a cross
            # leg here ran as `both`, asked THIS rank for the PEER's device
            # address, and died as `W74 ... src_resolved=0/N` -- weg2xsn24's
            # own wall, reachable on the cutover path whenever
            # `_weg2_xchg_sems` returns None (its region is None, or its
            # `except BaseException` swallowed anything at all).
            #
            # A ratchet that lives in a comment is not a ratchet.
            # THE CONDITION IS `hook AND NOT sems`, not "cross": a hermetic
            # caller passes NO hook and holds both pointers in one process, so
            # `both` is honest for it and 20 existing tests drive exactly that.
            # Only a leg that NAMES a hook can be a cutover leg, and a cutover
            # leg without its handshake is the one case that cannot work.
            cross = [d for d in descs
                     if int(getattr(d, "src_rank", -1))
                     != int(getattr(d, "dst_rank", -2))] if hook else []
            if cross:
                raise wx.Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {len(cross)} of {len(descs)} "
                    f"descriptors cross cards (first "
                    f"{getattr(cross[0], 'param_name', '?')}: "
                    f"{getattr(cross[0], 'src_rank', '?')} -> "
                    f"{getattr(cross[0], 'dst_rank', '?')}) and this leg "
                    f"names hook={hook!r} but has no semaphore set. The "
                    f"unsplit form runs phase=both, which needs BOTH pointers "
                    f"in one process -- and the source of a cross pair is the "
                    f"peer, whose address this rank cannot resolve by "
                    f"construction. Refusing the leg beats reporting "
                    f"src_resolved=0/N after the plan was built."
                )
            # The unsplit form, for the hermetic callers and the diagonal-only
            # case -- both of which hold both pointers in one process.
            return bx.run_bounce_leg(
                descs, ops, boot_nonce,
                slot_bytes=slot_bytes, depth=depth, terms=terms,
                mode=resolved_mode, shm_root=root, device=device,
                log=logger.info,
            )

        phase = (bx.PHASE_DEPOSIT if str(hook) == "source"
                 else bx.PHASE_COLLECT)
        # #1374 F1: THE PER-TAG DRAIN, and where it sits is the whole point.
        # The source takes it BEFORE tag t's bands and AFTER tag t-1's credit
        # was published (the pause loop publishes it immediately after the
        # previous deposit), so the collector it waits for is already able to
        # run -- no cycle. The first tag primes the counter to 0 instead,
        # because `create_semaphores` arms every `empty` at 1 for the OLD
        # meaning and at tag 0 there is no previous tag.
        # #1376 W10: FIELDS, NOT `getattr` DEFAULTS. The default is what let a
        # lazy assignment onto a `slots=True` dataclass look survivable at the
        # desk and die on the metal -- the read never raised, so nothing said
        # the write could not land.
        _first_tag = False
        if tag is not None:
            if self._weg2_xchg_tag_seen is None:
                self._weg2_xchg_tag_seen = set()
            _first_tag = not self._weg2_xchg_tag_seen
            self._weg2_xchg_tag_seen.add(str(tag))
        # THE BOUNCE'S OWN SLOT RECORD, created by whoever gets there first and
        # mapped by both ends. NOT the region's: weg2xsn24 read `carries 1080
        # bytes` because `run_producer_pair` publishes the RING's counts into
        # the same (pair, slot) records (weight_exchange_transport.py:1786).
        # #1377 (c) W11: THE RECORD'S ROW COUNT COMES FROM THE SAME TERMS AS
        # THE LEG'S BAND COUNT. It did not, and boot weg2xsn31/3 died of it:
        #   W68 Weg2XchgPlanDisagree: band 2 has no row -- this lane's record
        #   was built for 2 band(s) per pair
        # raised by our own `_row` guard (weight_exchange_bounce.py:1646) from
        # release_memory_occupation at 17:01:47Z, four seconds before the W98.
        #
        # THE COUPLING, verified and not assumed: `run_bounce_leg` takes its
        # slot count from `terms.lane_slots` -- 24 under #1374's Option 1
        # sizing (2907 MiB largest tag / 128 MiB slots, +1 compare slot) --
        # while this line built the record with the DEFAULT `rows_per_pair`,
        # i.e. `xr.SLOTS_PER_PAIR` = 2. The depositor then indexed band 2 into
        # a two-row table. Two numbers for one geometry: #1374 F1a made `_row`
        # REFUSE instead of folding (which is why this surfaced by name rather
        # than as silent aliasing), and #1374 F1b raised the leg's slots
        # without ever wiring the record to the same source.
        from sglang.srt.weg2 import weight_exchange_region as xr_mod

        _rows_per_pair = (int(terms.lane_slots) if terms is not None
                          else int(xr_mod.SLOTS_PER_PAIR))
        slots = bx.BounceSlots(str(boot_nonce), shm_root=root, create=True,
                               rows_per_pair=_rows_per_pair)
        last = None
        try:
            # #1358 THE LEDGER'S LANE COUNT IS CHECKED AGAINST THE LANE'S OWN
            # ENUMERATION, here, where both exist for the first time. The term
            # charges `buffer_bytes x n_lanes`; this dict IS the lane set, so a
            # ledger that counted fewer lanes than the leg creates is caught
            # before the first buffer is allocated instead of showing up as a
            # cushion that vanishes in service.
            #
            # That is not hypothetical: on boot weg2xsn28 the term charged ONE
            # buffer (2.160 GiB) while the boot created FIVE (7.044 GiB of
            # bounce.bin files, 0 holders at teardown), and W98 latched nine
            # seconds after `serving` needing 0.51 GiB it did not have.
            #
            # A NECESSARY CONDITION, NOT A COMPLETE ONE, and the difference is
            # a population and must be stated: `n_lanes` is BOOT-WIDE (the host
            # holds every rank's file at once -- that is what the ledger
            # charges), while `_lanes` here is THIS RANK's share. The five
            # xsn28 files came from several ranks: `p1,p2,p4` are pair lanes
            # and `c0,c1` are two ranks' diagonal groups, and no single rank
            # ever sees all five. So this catches a price BELOW what one rank
            # alone needs and CANNOT catch a boot-wide undercount on its own.
            # Comparing a per-rank count against a boot-wide charge as though
            # they were the same number is exactly the error this whole ticket
            # is about; the weaker check is kept because it is sound, and the
            # complete one belongs where the boot-wide set exists.
            # fnFL2x18: THE DECLARED PAD NEVER BECOMES A LANE. A ZEROFILL desc
            # (the zero pad expert row every Form-A D rank carries per expert
            # tensor, #74; the vocab pad) has no source, and
            # `group_descs_by_pair` files it under its destination's DIAGONAL.
            # On a tag whose layers the co-located P stage does not hold, that
            # diagonal carried ONLY pad, and `_weg2_seq_lane_descs` -- which
            # asks the join for (card, card) pieces, and a pad row has
            # src_rank=-1 -- refused W68 "NO desc for lane src=2 dst=2
            # tag='weights_9'" on D TP2 at the first P->D wake. Nor was the
            # pad ever zeroed on this path: `apply_zerofill` ran only in the
            # ring transport. It is an initialisation case, done below after
            # the tag's lanes, on the collect side.
            _zerofill = [d for d in descs if getattr(d, "kind", None) == wx.ZEROFILL]
            _lanes = bx.group_descs_by_pair(
                [d for d in descs if getattr(d, "kind", None) != wx.ZEROFILL])
            _priced = int(getattr(terms, "n_lanes", 1) or 1)
            if terms is not None and _priced < len(_lanes):
                # #1358 [W102-wired] THE CODE MATCHES THE DEFECT. This raised
                # W68 Weg2XchgPlanDisagree -- the code for a leg whose PHASE or
                # lane KEY is wrong -- while the condition here is a lane COUNT
                # the ledger priced too low. `Weg2XchgLanesUnmeasured` (W102)
                # is the lanes-specific code and, until this commit, had NO
                # raise site anywhere in the tree while a comment in
                # host_ledger.py claimed the leg driver used it. An intention
                # recorded as a state: the exact thing a W-code census reads as
                # covered.
                raise hl.Weg2XchgLanesUnmeasured(
                    f"W102 Weg2XchgLanesUnmeasured: the ledger priced "
                    f"n_lanes={_priced} BOOT-WIDE and THIS RANK alone needs "
                    f"{len(_lanes)} "
                    f"({sorted(str(k) for k in _lanes)}). Each lane is its own "
                    f"assemble buffer of {int(getattr(terms, 'buffer_bytes', 0))} "
                    f"bytes on tmpfs, so the host would hold "
                    f"{(len(_lanes) - _priced) * int(getattr(terms, 'buffer_bytes', 0))} "
                    f"bytes the arm never charged. Refusing BEFORE the first "
                    f"allocation: an under-priced bounce does not fail here, it "
                    f"fails later as a cushion nobody can account for."
                )
            # #1391 (DESK10) ROUND 3: THE GUARD, MOVED HERE FROM THE WRAPPER.
            # It used to live in `_weg2_xchg_inject_from_peer`, gated on
            # `tag is not None` -- and boot weg2xsn31's own instrument run
            # showed that gate is DEAD on the metal: the shadow-mode grader
            # (`_weg2_xchg_shadow_compare`) calls this leg with `tag=None`
            # (grading the WHOLE plan at once, correctly, for its own
            # purpose -- weight_updater.py:1693), so a check inside
            # `if tag is not None:` never runs on THIS boot's actual collect
            # call, only in a desk test that hand-supplied a tag. THIS frame
            # is the one both real callers (deposit AND collect, whichever
            # `tag` they pass) reach every time, which is why it belongs
            # here instead.
            #
            # THE CHECK ITSELF is unchanged in substance: on the COLLECT
            # side (`phase == PHASE_COLLECT`), a lane targeting THIS rank's
            # own card that `_lanes` does NOT cover (no descriptor for it in
            # `descs`, for ANY tag -- not just the one this call names) is
            # only ever an honest silence when nobody deposited into it
            # either (#1233's "this stage owns no layers of that chunk").
            # A nonzero `full` there is real, undrained work this call is
            # about to walk past without a word.
            if phase == bx.PHASE_COLLECT and rank is not None and int(rank) >= 0 and sems is not None:
                # weg2xsn85 (#1378): "for ANY tag" was the docstring, the
                # per-tag `descs` was the code. With P as the source D
                # refused at weights_7 (lanes from card 2 only) because
                # lane 1-0-0 held PP1's weights_4 bands and card0-0 PP0's
                # weights_4 deposit in flight -- the source had moved on
                # past its no-op tags (#1374 lockstep is per LANE, and the
                # per-lane order still matches). The whole-leg coverage is
                # what the check was written to mean.
                _covered = set(_lanes.keys()) | set(covered_lanes or ())
                _stuck = self._weg2_xchg_undrained_lanes(
                    int(rank), sems, covered=_covered)
                if _stuck:
                    raise Weg2XchgLaneNeverDrainedRefused(
                        f"W108 Weg2XchgLaneNeverDrainedRefused: hook={hook} "
                        f"rank={rank} tag={tag} boot_nonce={boot_nonce}: "
                        f"this leg's own descriptor set (this tag plus the "
                        f"whole-leg coverage {sorted(str(k) for k in _covered)}) "
                        f"carries no entry for "
                        f"{len(_stuck)} lane(s) targeting this rank's card, "
                        f"and they already hold undrained bands: "
                        + ", ".join(
                            f"lane={lane} full={full} empty={empty}"
                            for lane, full, empty in _stuck)
                        + ". Measured shape (boot weg2xsn31): a COMPLETE, "
                        f"correctly-sized deposit from the peer's own "
                        f"TP-shard split that this leg's plan does not "
                        f"cover for the tag(s) at hand -- not a double "
                        f"post, not a counting-semaphore invariant break. "
                        f"Refusing now beats the peer's own wait_drained "
                        f"and this rank's next credit wait each burning "
                        f"their full {WEG2_GROUP_FENCE_BUDGET_S:.0f}s "
                        f"budget and naming the wrong mechanism (W68 / "
                        f"W35)."
                    )
            # #1385 (Wand 11b step 2): THE RUNTIME HALF OF THE CAP. Step 1
            # (this branch's earlier commit) wired `lanes_concurrent` through
            # PRICING only -- the ledger charged `lanes_priced` while every
            # rank still pinned every lane it owned unconditionally, which is
            # #1358's under-charge reproduced in the other direction (the
            # WEG2-XCHG-LANES-CONCURRENT-CAVEAT line named exactly this gap).
            # This closes it: a rank opens the boot-wide permit ONLY when a
            # real cap is active, so the unset/default path performs the
            # zero extra syscalls the flag promises (`terms.lanes_concurrent`
            # is read from the SAME `terms` the ledger priced with -- one
            # cap value, never a second one computed here).
            # DIE DOKTRIN (zweite Sperre, zweite Auspraegung -- gemessen
            # dreimal: weg2xsn35 PcieLockTimeout held=124,45s vs Budget 120s;
            # weg2xsn36 W68 nach vollen 600s; weg2xsn37 derselbe Deadlock auf
            # dem falschen SHA):
            #   EINE SPERRE SERIALISIERT KOPIEN, NIEMALS WAITS.
            # Das LanePermit wird nur unter dem Cap-Flag armiert
            # (--xchg-lanes-concurrent; ohne Flag ist lanes_concurrent 0 und
            # das Permit existiert nicht), und sein Hold-Bereich enthaelt die
            # Deposit-Waits (wait_drained) -- genau die Form, die das
            # ko-lokalierte Paar blockiert. Der Ring-off-Boot faehrt OHNE
            # Cap (arm_xsn39: das Flag ist aus dem Argv genommen); die
            # 15,06-GiB-Lane-Pinning-Groesse ist vom Nutzer ausdruecklich
            # erlaubt ("bis dahin darf es auch 15GB 'ringpuffer' geben") und
            # steht in der ARM-Zeile. Ein Cap auf dieser Form waere der
            # falsche Trade: ein Deadlock, um Bytes zu sparen, die wir
            # ausgeben duerfen.
            _lane_permit_active = (
                terms is not None
                and int(getattr(terms, "lanes_concurrent", 0) or 0) > 0
            )
            _lane_permit = tp.LanePermit(str(boot_nonce)) if _lane_permit_active else None
            try:
                _lane_failures = []  # #1378 xsn77: every lane's refusal, not the last lane's
                _depth = bx.seq_buffer_depth()
                # getattr: the execution smokes drive this method with stubs
                # that carry no such field (the #1358 lesson, same frame).
                _lane_seq = getattr(self, "_weg2_xchg_lane_seq", None)
                if _lane_seq is None:
                    _lane_seq = {}
                    try:
                        self._weg2_xchg_lane_seq = _lane_seq
                    except AttributeError:
                        pass

                def _run_lane(pair, group):
                    # #1378 xsn34 root fix: the lanes' wait budget is the
                    # boot's own leg bound (600 s, = the deadman's GRACE_S),
                    # NOT the 120 s tool default -- measured on weg2xsn34:
                    # the first flip's collect expired 3 s before the sleep
                    # side's first full post landed (the pause chain IS the
                    # staging time). See LANE_RENDEZVOUS_BUDGET_S.
                    rv = (bx.CrossSlotRendezvous(sems, slots, pair=pair,
                                                 budget_s=bx.LANE_RENDEZVOUS_BUDGET_S)
                          if pair is not None else
                          bx.CrossSlotRendezvous(
                              sems, slots,
                              card=int(getattr(group[0], "dst_rank", device)),
                              budget_s=bx.LANE_RENDEZVOUS_BUDGET_S))
                    # #1374 F1: THE PER-TAG DRAIN, wired where the lane's own
                    # rendezvous exists. Ordering, on the source:
                    #   tag 0: prime the counter to 0 (create_semaphores arms
                    #          every `empty` at 1 for the OLD per-slot
                    #          meaning, and at tag 0 there is no previous tag
                    #          to have been drained)
                    #   tag t: wait for the collector's `drained(t-1)` BEFORE
                    #          touching the buffer, which is safe because the
                    #          pause loop published credit(t-1) before
                    #          calling us
                    # On the collector: post `drained(t)` after the tag's
                    # last band. Without this, tag t+1 overwrites bands tag t's
                    # collector may still be reading -- silently, which is
                    # worse than the deadlock the per-band claim caused.
                    #
                    # DELIBERATELY OUTSIDE THE PERMIT HOLD (below): this wait
                    # is for a PEER'S PROGRESS, not for this lane's own
                    # buffer, and no buffer exists yet at this point in the
                    # loop. Holding a permit across it would tie up a
                    # boot-wide slot while idle, making the cap less
                    # effective than the number it charges for and risking a
                    # spurious W69 on a healthy but merely slow peer.
                    _lane_key = (f"p{pair}" if pair is not None
                                 else f"c{int(getattr(group[0], 'dst_rank', device))}")
                    # 18.09. (xsn369/370): the tag-order gate for HOST/IPC
                    # lanes, BEFORE the slot counter below is read -- with two
                    # collects in flight, tag t+1 read the same `_seq` as tag
                    # t and their unit records collided on the lane (xsn370:
                    # c0/weights_2 identity mismatch). H89: a lane is BAR1 for
                    # this gate only when the DEPOSITOR mapped the window, not
                    # when this side merely serves one (cu130 image: P's boot
                    # connect ran out, P deposited p0/p2/p4 on the host ring,
                    # D skipped the gate, two collects shared slot 0 -> W68
                    # p0/weights_0 identity mismatch).
                    if phase == bx.PHASE_COLLECT:
                        _is_bar1_lane = self._weg2_collect_lane_is_bar1(_lane_key, pair)
                        _turns = self._weg2_lane_turns
                        if not _is_bar1_lane and _turns is not None:
                            # H11: THE LANE'S TURN, not every earlier tag.
                            # Only the earlier tags that ride THIS lane can
                            # collide on its slot counter; the others (other
                            # source cards under the round-robin order) left
                            # it when their collect started. x83-x105: D TP0
                            # waited 790-889 ms per P->D flip in this gate.
                            _ord = self._weg2_leg_tag_order or []
                            _ti = _ord.index(str(tag)) if str(tag) in _ord else -1
                            _tg0 = time.perf_counter()
                            _ok, _waited = _turns.take(_lane_key, _ti, 600.0)
                            if not _ok:
                                _lane_failures.append(
                                    f"{_lane_key}/{tag}: lane turn: the earlier tag(s) "
                                    f"{','.join(_ord[j] for j in _waited)} did not leave "
                                    f"the lane within 600 s")
                                return
                            if _waited:
                                logger.info("WEG2-TAG-GATE lane=%s tag=%s waited_ms=%.0f for=%s mode=lane",
                                            _lane_key, tag, (time.perf_counter() - _tg0) * 1000,
                                            ",".join(_ord[j] for j in _waited))
                        elif not _is_bar1_lane:
                            _ord = getattr(self, "_weg2_leg_tag_order", None) or []
                            _evs = getattr(self, "_weg2_tag_done", None) or {}
                            _ti = _ord.index(str(tag)) if str(tag) in _ord else -1
                            # EVERY earlier tag, not only the previous one: with
                            # two collects in flight a small tag t-1 finishes
                            # while t-2 still holds the lane (xsn371).
                            _pending = [j for j in range(_ti) if j in _evs and not _evs[j].is_set()]
                            if _pending:
                                _tg0 = time.perf_counter()
                                for j in _pending:
                                    if not _evs[j].wait(600.0):
                                        _lane_failures.append(
                                            f"{_lane_key}/{tag}: tag-order gate: the earlier tag "
                                            f"{_ord[j]} was not collected within 600 s")
                                        return
                                logger.info("WEG2-TAG-GATE lane=%s tag=%s waited_ms=%.0f for=%s",
                                            _lane_key, tag, (time.perf_counter() - _tg0) * 1000,
                                            ",".join(_ord[j] for j in _pending))
                    _seq = int(_lane_seq.get(_lane_key, 0))
                    _slot = _seq % max(1, int(_depth))
                    # 2026-09-15 (Beschleunigung, depth 2): the drain wait
                    # reaches back `_depth` tags -- seq 0 primes, seq 1 uses
                    # the second buffer without waiting, seq >= depth waits
                    # for the collector's drain of seq-depth (the counting
                    # `empty` hands posts out in order). Depth 1 is the
                    # xsn87 form: every tag after the first waits.
                    if tag is not None and phase == bx.PHASE_DEPOSIT:
                        if _seq == 0:
                            rv.prime_drain()
                            logger.info(
                                "WEG2-SEQ prime lane=%s drained=%d depth=%d",
                                _lane_key, int(getattr(rv, "last_primed", 0)),
                                int(_depth))
                        elif _seq >= int(_depth) and not rv.wait_drained(tag=str(tag)):
                            raise bx.Weg2XchgBouncePhaseUnordered(
                                f"W68 Weg2XchgPlanDisagree: tag={tag} lane="
                                f"{'p%s' % pair if pair is not None else 'diag'} "
                                f"waited for the collector to drain the "
                                f"previous tag and it did not. The credit for "
                                f"that tag was published before this wait. "
                                f"Either the peer is stalled or dead, or it "
                                f"sits in its own credit wait with a balance "
                                f"the earlier resumes overdrew (fnFL2x26: the "
                                f"pause order let the destination run ahead "
                                f"of this card's supply -- read its "
                                f"WEG2-VRAM-CREDIT waiting line and "
                                f"front.interleave_chain_card). Refusing "
                                f"rather than overwriting bands a consumer "
                                f"may still read.")
                    # #1385: ACQUIRE RIGHT BEFORE THE BUFFER, RELEASE RIGHT
                    # AFTER IT CLOSES -- never earlier, never later. Acquiring
                    # here (not before the drain wait above) keeps the permit
                    # held for exactly the window `run_bounce_leg` actually
                    # pins the buffer, which is `lanes_serialised`'s own
                    # promise: the boot pays flip time only while a lane
                    # really is waiting for tmpfs room, not for a peer's
                    # unrelated progress. `run_bounce_leg`'s OWN `finally`
                    # closes its `LayerBounce` on every exit path (success or
                    # raise) BEFORE returning or propagating, so by the time
                    # this `finally` releases the permit the memory is
                    # already unpinned -- the buffer is freed BEFORE the next
                    # acquire can succeed, never after.
                    if _lane_permit_active:
                        _lane_permit.acquire(
                            budget_s=tp.LANE_PERMIT_TIMEOUT_S, lane=_lane_key,
                            boot=str(boot_nonce))
                    try:
                        # #1378 xsn45 (THE WIRING): the sequential unit
                        # transport replaces the lane machinery on this leg.
                        # #1378 xsn46 (NUTZER-FRAGE beantwortet): die
                        # nbytes kommen von der EMPFANGENDEN Seite (d.nbytes
                        # = die Desc's Empfangsgroesse, NICHT die Quell-
                        # groesse). Die Empfangsseite bestimmt, wie gross
                        # die Einheit im Puffer sein muss.
                        # #1378 xsn52 (DER MANIFEST-LAYOUT-FIX), CORRECTED IN
                        # xsn53: the units come from the JOIN, but through the
                        # join's OWN plan builder and filtered to THIS LANE --
                        # not the join's raw tensors.  The raw tensor list has
                        # no lane in it (every card would move every tensor)
                        # and no shard cut (the buffer would hold the
                        # unsharded extents, the whole image per rank).  What
                        # both sides must share is the DESC LIST OF THE LANE,
                        # and ``plan_from_join`` is the one producer of that:
                        # deterministic from the manifests alone, so the
                        # deposit rank and the collect rank derive the same
                        # list without exchanging a byte of metadata.  That is
                        # also what fixes the NameError this line died of
                        # (`join` was read here and defined nowhere --
                        # provider smoke 15/19, four red, measured).
                        # #1378 xsn55 (THE DIRECTION KEY): `group` in THIS
                        # frame is the LANE's desc list (`for pair, group in
                        # _lanes.items()` above), not a group name.  Handing
                        # it down made `leg_direction(hook, group)` answer
                        # from the hook alone -- right for group D by
                        # coincidence, MIRRORED for group P, which is the
                        # measured weg2xsn53 wall (P rank 1's wake lane (0,1)
                        # carried P rank 0's 102 layer-32..38 descs and its
                        # own address book answered None for all of them).
                        # The group NAME is resolved from the identity this
                        # process already carries; a caller without one is
                        # refused by name inside the derivation rather than
                        # planned in a mirrored direction.  The getattr guard
                        # is the same shape the leg already uses one frame
                        # down (an execution smoke drives this method with a
                        # stub that carries no identity methods, and that
                        # stub overrides the derivation, so the read is dead
                        # for it by construction -- not skipped silently for
                        # a caller that reaches it).
                        _group_reader = getattr(self, "_weg2_group_name", None)
                        _group_name = (str(_group_reader() or "")
                                       if _group_reader is not None else "")
                        _lane_descs = self._weg2_seq_lane_descs(
                            hook=hook, group=_group_name, rank=rank,
                            pair=pair, card=None if pair is not None
                            else int(getattr(group[0], "dst_rank", device)),
                            tag=tag, log=logger.info)
                        if not _lane_descs:
                            # A lane whose both ends are other ranks: this
                            # rank holds neither the source bytes nor the
                            # destination window, so it has nothing to stage
                            # and no handshake to meet.  Continuing would
                            # post a token nobody's peer waits for.
                            return
                        # #1378 xsn56 -- THE LANE'S BUFFER IS THE LANE'S OWN
                        # SUM, not the priced single slot.  Operator order
                        # 2026-09-15: "er limitiert die groesse des
                        # ringpuffers wohl immernoch."
                        #
                        # `terms.buffer_bytes` prices the WIDEST SINGLE LAYER
                        # (756323776 B = 721 MiB at depth=1).  A PP-form lane
                        # carries a whole BAND -- 114 descs, ~2.16 GiB on
                        # weg2xsn55 -- so `batch_descs` cut it into 3 batches
                        # and this form refuses that by construction: one
                        # handshake, one buffer.  `slot_bytes` means the
                        # LANE'S BUFFER SIZE here and the PRICING slot size in
                        # `bounce_terms`; two meanings, one name, and the
                        # caller was handing over the wrong one.
                        #
                        # THE SUM IS EXACT, not an estimate: `batch_descs`
                        # packs byte-exactly (FLAT `take = min(room,
                        # remaining)`, STRIDED2D `cur_bytes += take * run`)
                        # with no alignment padding, so a slot equal to the
                        # sum yields exactly one batch and the buffer is then
                        # sized from `_batch.total_bytes`, not from this
                        # number.  Funding, measured 2026-09-15: /dev/shm 63
                        # GiB with 62 free, and the host ring this replaces
                        # was 43.83 GiB -- the transient buffer is the
                        # mechanism that keeps the dormant image OFF the host,
                        # so it is the cheap half of the trade.
                        _lane_bytes = sum(int(getattr(d, "nbytes", 0) or 0)
                                          for d in _lane_descs)
                        # 18.09. BAR1 lanes: the depositor decides per tag
                        # (peer window mapped or not) and writes mode.<seq>;
                        # the collector waits for that file first. Diagonal
                        # lanes (pair None) stay on the on-card IPC form.
                        from sglang.srt.weg2 import bar1_lanes as b1
                        _b1 = getattr(self, "_weg2_bar1", None)
                        _b1_role = (_b1.role(_lane_key)
                                    if (_b1 is not None and pair is not None) else None)
                        _b1_mode = None
                        if (_b1_role is not None
                                and (_b1_role == "src") == (phase == bx.PHASE_DEPOSIT)):
                            _fi = getattr(self, "_weg2_flip_index_now", None)
                            _b1_seq = (f"{int(_fi)}-{tag}" if _fi is not None and int(_fi) >= 0
                                       else int(_seq))
                            _b1_mode = _b1.lane_mode(
                                _lane_key, _b1_role, seq=_b1_seq,
                                liveness=self._weg2_cocard_peer_alive)
                            if _b1_mode != b1.MODE_BAR1:
                                logger.info("WEG2-BAR1 lane=%s phase=%s seq=%d via=host "
                                            "reason=%s", _lane_key, phase, int(_seq),
                                            (_b1.refusals.get(_lane_key, "peer decided host")
                                             if _b1_role == "src" else "depositor decided host"))
                        if _b1_mode == b1.MODE_BAR1:
                            # the lane's turn (flip, index in the wake order): with two
                            # collects in flight one lane's tags stay in order on its
                            # credit channel
                            _ord_k = getattr(self, "_weg2_leg_tag_order", None) or []
                            _b1_order = ((int(_fi), _ord_k.index(str(tag)))
                                         if (phase == bx.PHASE_COLLECT and _fi is not None
                                             and int(_fi) >= 0 and str(tag) in _ord_k) else None)
                            last = b1.run_bar1_units(
                                _lane_descs, ops, lanes=_b1, lane_key=_lane_key,
                                role=_b1_role, seq=_b1_seq, phase=phase, order_key=_b1_order,
                                no_write=getattr(self, "_weg2_xchg_no_write", None),
                                liveness=self._weg2_cocard_peer_alive,
                                device=device, log=logger.info)
                        else:
                          last = bx.run_sequential_units(
                            _lane_descs, ops, boot_nonce,
                            shm_root=root, device=device, phase=phase,
                            slot_bytes=_lane_bytes or xr.SLOT_BYTES,
                            pair=None if pair is None else int(pair),
                            card=(None if pair is not None else
                                  _diagonal_card_of(group, device, phase)),
                            liveness=self._weg2_cocard_peer_alive,
                            # weg2xsn86: (tag, name) pairs consumed but not
                            # written -- MEASURED target shares of the draft.
                            no_write=getattr(self, "_weg2_xchg_no_write", None),
                            buffer_slot=int(_slot),
                            stage_charge=self._weg2_stage_charge(),
                            # #1358: the identity the host-slot lines carry.
                            # This is the only frame where the group and the
                            # rank both exist.
                            #
                            # GUARDED, AND THE GUARD IS THE FIX FOR A DEFECT I
                            # SHIPPED. The first version called
                            # `self._weg2_group_name()` unprotected and
                            # claimed "every existing caller unchanged". The
                            # train seat's execution smoke refuted it by
                            # bisection ([15]/[16]/[17a] all 19/19, [17b]
                            # 15/19): the two smoke harnesses drive this exact
                            # product path with STUBS (`_LegStub` in
                            # xchg_provider_smoke.py, `_Stub` in
                            # xchg_leg_replay.py) that carry no such methods,
                            # so every leg died with AttributeError before it
                            # ran.
                            #
                            # A getattr default rather than two more stub
                            # methods: this closes the CLASS (any future
                            # caller without the methods) instead of the two
                            # instances that happened to exist, and it
                            # matches what the emitter already does one frame
                            # down, where an absent group prints as `group=?`.
                            log=logger.info)
                        _lane_seq[_lane_key] = _seq + 1
                        if last:
                            _lane_failures.append(f"{_lane_key}/{tag}: {last}")
                    finally:
                        if _lane_permit_active:
                            _lane_permit.release()
                    # #1374 F1: THIS TAG IS OUT OF THE BUFFER. Posted once per
                    # tag by the collector, after its last band; it is what
                    # the source's `wait_drained` for the NEXT tag releases.
                    # The per-band `empty` post is gone (that row IS this
                    # handshake), so N posts per tag cannot hand the source N
                    # tokens.
                    if tag is not None and phase == bx.PHASE_COLLECT:
                        rv.post_drained(tag=str(tag))
                        # #1385 STEP 3 (boot weg2xsn31/6): FREE THE FILE HERE,
                        # not only the permit. `LayerBounce.close()` never
                        # unlinks (by design, for store-and-forward), so
                        # without this a lane's tmpfs pages stay committed
                        # until full boot teardown regardless of the
                        # concurrency cap -- measured on xsn31/6: cap=1
                        # correctly serialised PINNING (never >1 buffer being
                        # actively registered) while FOUR lanes' files
                        # (c0+p1+p2+p4, 12.00 GiB) coexisted on tmpfs because
                        # none of the earlier ones were ever removed. Safe
                        # exactly here: this is the SAME point that already
                        # tells the depositor (via `wait_drained`) it may
                        # reuse the buffer for the next tag, and both sides'
                        # own file descriptors are already closed by this
                        # point (inside their respective `run_bounce_leg`
                        # calls' own `finally`). Gated on `_lane_permit_active`
                        # so the unset/default path -- whose own pricing
                        # already charges for every lane accumulating, by
                        # design -- is untouched.
                        if _lane_permit_active:
                            bx.unlink_lane_buffer(boot_nonce, root, _lane_key)

                # 2026-09-15 (Beschleunigung): a rank's lanes in threads.
                # Each lane owns its buffer, record file, handshake and
                # rendezvous; the ctypes copies release the GIL. Off with
                # SGLANG_WEG2_SEQ_LANES_PARALLEL=0 (the xsn87 serial form).
                def _run_lane_turned(pair, group):
                    # H11: the lane's turn ends with its run, whichever way
                    # _run_lane returns -- the next tag on it may start.
                    try:
                        _run_lane(pair, group)
                    finally:
                        if phase == bx.PHASE_COLLECT:
                            self._weg2_turns_leave(
                                tag, f"p{pair}" if pair is not None
                                else f"c{int(getattr(group[0], 'dst_rank', device))}")

                if phase == bx.PHASE_COLLECT:
                    # the BAR1 lanes this tag does NOT use give up their turn now
                    _used_keys = [(f"p{p}" if p is not None
                                   else f"c{int(getattr(g[0], 'dst_rank', device))}")
                                  for p, g in _lanes.items()]
                    self._weg2_bar1_release(tag, used=_used_keys)
                    self._weg2_turns_release(tag, used=_used_keys)
                if bx.seq_lanes_parallel() and len(_lanes) > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    _t0 = time.perf_counter()
                    with ThreadPoolExecutor(
                            max_workers=len(_lanes),
                            thread_name_prefix="weg2-lane") as _ex:
                        _futs = [(p, _ex.submit(_run_lane_turned, p, g))
                                 for p, g in _lanes.items()]
                        _errs = [(p, f.exception()) for p, f in _futs
                                 if f.exception() is not None]
                    # #101 (fnFL2w36): `errors=%d` NENNT DEN FEHLER NICHT.
                    # Gemessen: TP1 loggte `phase=deposit tag=weights_9 ms=5
                    # errors=1` und danach stand im ganzen D-Log kein
                    # Traceback -- die Lane c1 deponierte nie, P wartete 90 s
                    # auf genau sie ("c1/weights_9: budget expired at unit 0")
                    # und riss beim Fence alle drei P-Raenge mit. Die Zahl
                    # allein kostete einen Boot: sie sagt DASS, nie WAS.
                    # `raise _errs[0][1]` wirft zwar die erste Ausnahme, aber
                    # der Aufrufer oben faengt sie in den Gruppen-Fence, und
                    # die Ausnahmen der UEBRIGEN Lanes fallen still weg.
                    _err_txt = "; ".join(
                        f"{p}: {type(e).__name__}: {e}" for p, e in _errs)
                    logger.info(
                        "WEG2-SEQ-LANES parallel=%d phase=%s tag=%s ms=%.0f "
                        "errors=%d%s t0=%.3f t=%.3f", len(_lanes), phase, tag,
                        (time.perf_counter() - _t0) * 1000, len(_errs),
                        (f" | {_err_txt}" if _errs else ""),
                        time.time() - (time.perf_counter() - _t0), time.time())
                    if _errs:
                        # Mit Stack, damit die Wurzel im Log steht und nicht
                        # nur die Zaehlung -- der Fence weiter oben macht aus
                        # jeder Lane-Ausnahme dieselbe Gruppen-Meldung.
                        logger.error(
                            "#101 WEG2-SEQ-LANES phase=%s tag=%s: %d von %d "
                            "Lane(s) gescheitert: %s -- die erste wird "
                            "geworfen, die uebrigen stehen nur hier.",
                            phase, tag, len(_errs), len(_lanes), _err_txt,
                            exc_info=_errs[0][1])
                        raise _errs[0][1]
                else:
                    for pair, group in _lanes.items():
                        _run_lane_turned(pair, group)
            finally:
                if _lane_permit is not None:
                    _lane_permit.close()
        finally:
            slots.close()
        if _lane_failures:
            # #1378 xsn77: `last` was overwritten by every following lane, so
            # a lane that refused (P rank 0, lane c0, tag weights: "digest
            # mismatch at unit 0 model.embed_tokens.weight") vanished behind
            # two lanes that returned "" -- the embed rows 0..82815 were never
            # written and four boots read the SEAM-DIGEST mismatch as a
            # content question. A lane's refusal refuses the leg, by name.
            # fnFL2x100: a lane stopped by ANOTHER rank's dead leg says only
            # "gone or stuck"; the posted abort names who died and why.
            _aborted = self._weg2_foreign_leg_aborts()
            raise bx.Weg2XchgBouncePhaseUnordered(
                f"W68 Weg2XchgPlanDisagree: {len(_lane_failures)} lane(s) of "
                f"this leg refused: " + " | ".join(_lane_failures[:6])
                + (f" -- W121 PEER LEG ABORTED this flip: {_aborted}"
                   if _aborted else ""))
        if _zerofill and phase == bx.PHASE_COLLECT:
            self._weg2_xchg_apply_zerofill(
                ops=ops, descs=_zerofill, rank=rank, device=device, tag=tag)
        return last

    def _weg2_xchg_apply_zerofill(self, *, ops, descs, rank, device, tag) -> int:
        """Zero this rank's declared pad of one tag, after the tag's lanes.

        The pad rows exist on no card and in no checkpoint, so nothing ever
        delivers them; after a backup-off resume their pages hold whatever
        the remap left. A pad row without an address is refused by name --
        skipping it would leave exactly those bytes in place.
        """
        from sglang.srt.weg2 import weight_exchange as wx
        from sglang.srt.weg2 import weight_exchange_transport as tp

        my_rank = int(rank) if rank is not None and int(rank) >= 0 else int(self._weg2_rank())
        mine = [d for d in descs if int(d.dst_rank) == my_rank]
        unresolved = [str(d.param_name) for d in mine if d.dst_ptr is None]
        if unresolved:
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {len(unresolved)} of {len(mine)} "
                f"ZEROFILL descs of tag={tag!r} have no destination address on "
                f"rank {my_rank} (first {unresolved[0]!r}) -- the pad rows would "
                f"keep whatever the resumed pages held")
        stream = ops.create_stream(int(device))
        try:
            nbytes = tp.apply_zerofill(ops, stream, mine, my_rank)
        finally:
            ops.destroy_stream(stream)
        logger.info(
            "WEG2-XCHG ZEROFILL tag=%s rank=%d descs=%d bytes=%d -- the declared "
            "pad (no source anywhere) zeroed after this tag's lanes",
            tag, my_rank, len(mine), nbytes)
        return nbytes

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
            from sglang.srt.managers.corridor_guard import (
                corridor_floor_mib,
                user_reserve_by_card,
            )

            # weg2xsn271 (18.09.): `corridor_floor_mib` returns a CorridorFloor
            # OBJECT (transient + reserve, with provenance), never a number.
            # `int(floor)` raised TypeError, the except below turned it into
            # None, and every credit wait on every card ran with floor 0 --
            # the D-side reader logged corridor_floor_mib=0 while the boot
            # had MEASURED-D 3567 MiB (767 transient + 2800 reserve) for the
            # 5090. TP0 then entered resume(weights_4) with 8 MiB to spare
            # and the memory saver's cu_mem_create exit(1)'d the rank.
            reserve = user_reserve_by_card().get(uuid_key)
            floor = corridor_floor_mib(
                uuid_key,
                group=self._weg2_group_name(),
                user_reserve_mib=0 if reserve is None else int(reserve),
            )
            if floor is None:
                return None
            mib = int(getattr(floor, "mib"))
            if not getattr(self, "_weg2_floor_noted", False):
                self._weg2_floor_noted = True
                logger.info(
                    "WEG2-CREDIT-FLOOR card=%s group=%s floor_mib=%d "
                    "(transient=%s reserve=%s source=%s) -- the credit wait "
                    "grades free VRAM against need + this floor",
                    uuid_key, self._weg2_group_name(), mib,
                    getattr(floor, "transient_mib", "?"),
                    getattr(floor, "reserve_mib", "?"),
                    getattr(floor, "source", "?"),
                )
            return mib * 1024 * 1024
        except Exception as exc:  # noqa: BLE001 -- an absent authority is an absence
            if not getattr(self, "_weg2_floor_noted", False):
                self._weg2_floor_noted = True
                logger.warning(
                    "WEG2-CREDIT-FLOOR card=%s group=%s UNREADABLE (%s: %s) -- "
                    "the credit wait grades free VRAM against the request "
                    "alone (floor 0)", uuid_key, self._weg2_group_name(),
                    type(exc).__name__, exc)
            return None

    def _weg2_xchg_whole_leg_lanes(self) -> Optional[set]:
        """The lanes this rank's WHOLE-LEG collect plan covers (every tag),
        as ``group_descs_by_pair`` keys -- ``None`` for the on-card diagonal,
        a ``CROSS_PAIRS`` index for a directed cross pair -- or ``None`` when
        no plan can be derived here (a desk double, an unarmed boot).

        weg2xsn257 (17.09.): the credit wait's W108 reader checked EVERY lane
        with a nonzero ``full`` and refused ``0.0s into the wait`` on PP0 at
        tag weights_1 -- lanes 1-0-0 and 2-0-0 held 106 bands each. Those
        were weights_1's OWN bands: all six ranks walk one tag order
        (0,4,7,1,5,2,6,3,draft,weights on both groups, read from the
        SLEEP-TAG-TIME / RESUME-begin stamps), the depositors had finished
        weights_1 while PP0 still waited for its co-located sleeper's credit,
        and PP0's very next step was to collect exactly those bands. Under
        #1397's band credit a lane legally holds the NEXT tag's bands while
        this rank waits, so "full > 0 between tags" no longer proves a tag
        nobody will drain. The collect path already knew this (weg2xsn85:
        ``covered_lanes`` = the whole-leg set); the credit wait did not. This
        is the one producer of that set for both call sites: the same
        boot-cached plan ``_weg2_xchg_inject_from_peer`` collects with, so a
        lane the plan covers for ANY tag is never a "never drained" lane.
        """
        try:
            from sglang.srt.weg2 import weight_exchange_bounce as _bx
            group = self._weg2_group_name()
            rank = self._weg2_rank()
            if not group or rank is None or int(rank) < 0:
                return None
            plan, _reason = self._weg2_shadow_plan(
                "authoritative", group, int(rank), agreed=None,
                require_agreement=False)
            if plan is None:
                return None
            return set(_bx.group_descs_by_pair(list(plan.descs)).keys())
        except Exception:  # noqa: BLE001 -- no plan is "not measurable", never a refusal
            return None

    def _weg2_await_vram_credit(self, credit, tag: str, need_bytes: int,
                                epoch=None, submitted=None) -> None:
        if credit is None or need_bytes <= 0:
            return
        # #1391 (DESK10) ROUND 3: A CALLBACK, NOT A ONE-SHOT CHECK. The first
        # version checked ONCE at entry and lost a real race the metal
        # measured directly (boot weg2xsn31 instrument run): this rank
        # entered clean, the peer's saturating post landed four seconds
        # later, and nothing looked again while the 120s poll ran. A
        # predicate taken once at the door cannot see a condition that
        # develops WHILE waiting; it has to live inside the poll.
        # `VramCredit.wait_for` (weg2_memory_saver.py) owns that loop, so
        # the check moves there via this callback -- `_stuck_lane_reader`
        # -- called on its own cadence and raising `Weg2XchgLaneNeverDrainedRefused`
        # (W108, renumbered 2026-09-14 from W100 -- TRAIN2's census found it
        # colliding with host_ledger.py's pre-existing Weg2SleepLegCushionDeficit)
        # itself the moment it finds something, well before this call's own
        # `budget_s` would otherwise expire into a W35.
        _rank = self._weg2_rank()
        _reader_state = {"noted": False}

        def _stuck_lane_reader():
            if _rank < 0:
                return []
            if getattr(self, "_weg2_wake_inflight", False):
                return []   # a collect is in flight on the wake worker
            # weg2xsn257: the SAME whole-leg coverage the collect path applies
            # (`_weg2_xchg_bounce_leg`'s `covered_lanes`). A lane this rank's
            # plan walks for ANY tag holds the next tag's bands legally; only a
            # lane the plan never walks is bands nobody will drain. No plan ->
            # no proof -> no refusal: the credit budget (W35) stays the
            # detector, exactly as for every caller that passes no reader.
            covered = self._weg2_xchg_whole_leg_lanes()
            if covered is None:
                if not _reader_state["noted"]:
                    _reader_state["noted"] = True
                    logger.info(
                        "WEG2-CREDIT-WAIT tag=%s lane check NOT-MEASURED: no "
                        "whole-leg plan on this rank, so a full lane cannot be "
                        "told from the next tag's bands; W108 is not raised "
                        "from this wait and the credit budget remains the "
                        "detector", tag)
                return []
            sems = self._weg2_xchg_sems()
            stuck = self._weg2_xchg_undrained_lanes(_rank, sems, covered=covered)
            if not stuck and not _reader_state["noted"]:
                held = self._weg2_xchg_undrained_lanes(_rank, sems)
                if held:
                    _reader_state["noted"] = True
                    logger.info(
                        "WEG2-CREDIT-WAIT tag=%s %d full lane(s) targeting this "
                        "card are COVERED by this rank's whole-leg plan (%s) -- "
                        "the next tag's bands, not a wedge; waiting on",
                        tag, len(held),
                        ", ".join(f"{lane}:full={full}" for lane, full, _e in held))
            return stuck

        # #22: name this wait for the other wakers' cycle readers (tag + the
        # tags whose collect is already submitted), and read theirs.
        _b1 = getattr(self, "_weg2_bar1", None)
        _cycle_reader = None
        if _b1 is not None:
            _b1.post_credit_wait(tag, submitted)
            _cycle_reader = _b1.credit_cycle
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
                stuck_lane_reader=_stuck_lane_reader,
                cycle_reader=_cycle_reader,
                abort_reader=self._weg2_foreign_leg_aborts,
            )
        except Weg2VramCreditRefused:
            raise
        except OSError as exc:
            logger.warning("[weg2 credit] unreadable on %s: %s -- not waiting", tag, exc)
            return
        finally:
            if _b1 is not None:
                _b1.clear_credit_wait()
        # xsn323: remember the tightest point of these legs. The kv-first gate
        # of the NEXT wake reads it: kv_cache resumed before the legs must not
        # eat the free space the legs' tags need (5090: free 9028 MiB at
        # weights_3 in the old order, 2360 with the pool up -> credit wait ->
        # the sleeper's tail never paused -> W35 after 120 s).
        try:
            _fb = rec.get("free_bytes")
            if _fb is not None:
                _fm = int(_fb) >> 20
                if self._weg2_leg_min_free_epoch != epoch or self._weg2_leg_min_free_mib is None:
                    self._weg2_leg_min_free_epoch = epoch
                    self._weg2_leg_min_free_mib = _fm
                else:
                    self._weg2_leg_min_free_mib = min(int(self._weg2_leg_min_free_mib), _fm)
        except Exception:  # noqa: BLE001 -- an instrument for the next gate, never a gate itself
            pass
        # #1349: a tag that TOOK credit is logged even when it waited 0 ms. The
        # debit is the half that was missing, so it has to be readable per tag,
        # and the quiet case stays quiet by construction: with no co-located peer
        # the counter is empty, nothing is ever claimed, and this stays at the
        # pre-#1349 "only a real wait prints" volume.
        if rec.get("waited_s", 0.0) > 0.0 or int(rec.get("claimed_bytes", 0)) > 0:
            logger.info(
                "WEG2-VRAM-CREDIT card=%s tag=%s waited=%.0f ms credit=%d MiB "
                "published=%d MiB consumed=%d MiB available_after=%d MiB "
                "requested=%d MiB free_mib=%s allocatable_est=%s "
                "corridor_floor_mib=%s (%s)",
                self._weg2_card_uuid() or "unknown", tag,
                float(rec["waited_s"]) * 1000,
                int(rec.get("credit_bytes", 0)) // MIB_,
                # #1349: BOTH HALVES OF THE BOOK ON EVERY LINE. `credit` is the
                # UNSPENT balance the grant was decided on; `published` is the
                # peer's gross release total and `consumed` what this leg's tags
                # have already claimed off it. On weg2xsn21b the two consecutive
                # grants read `credit=2560` BOTH times -- with these three fields
                # that double-spend is visible in the log instead of having to be
                # reconstructed from a `free_mib` that fell by the request size.
                int(rec.get("credit_published_bytes", rec.get("credit_bytes", 0)))
                // MIB_,
                int(rec.get("consumed_bytes", 0)) // MIB_,
                int(rec.get("available_bytes", 0)) // MIB_,
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
    @_vram_peak_leg("release")
    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        # #1285 FIRST STATEMENT, before the idle assert and before any mutation:
        # a leg that is already applied returns its recorded answer.  It has to
        # precede the assert too -- a repeat arrives with the group in whatever
        # state the first attempt left it, and refusing there would turn a safe
        # no-op into a group death.
        self._weg2_raise_pending_seam_refusal()  # #1450
        replay = self._weg2_leg_replay("release", recv_req)
        _weg2_ph_t = [time.perf_counter()]
        _weg2_ph_l = []
        
        def _weg2_ph(name):
            _n = time.perf_counter()
            _weg2_ph_l.append((name, (_n - _weg2_ph_t[0]) * 1000))
            _weg2_ph_t[0] = _n
        
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
        # fnFL2x105: UNCONDITIONAL. The drain's polls are collectives over the
        # attention group, so the question "drain or not" is a group question
        # -- boot fnFL2x104: TP0 (idle) skipped this, slept and waited in the
        # fence while TP1/TP2 (hicache_backup(1)) waited for it in the drain's
        # all_reduce, 120 s until the fence expired. The drain now reduces its
        # verdict over that group first and raises W120 on every rank alike.
        self._weg2_drain_hicache_before_sleep()
        _weg2_ph("drain_hicache")

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
        # Task #47 Scheibe 6a: under --flip-weights resident the weights family
        # is never paused. A release naming one is a wrong front, not a sleep --
        # refused by name before any tag is touched.
        from sglang.srt.managers.weg2_memory_saver import weights_resident_armed as _wra

        if _wra():
            _foreign = [t for t in tags if is_weights_family_tag(t)]
            if _foreign:
                raise Weg2WakeRefused(
                    "W4 Weg2WakeRefused: this boot keeps the weights RESIDENT "
                    f"(SGLANG_WEG2_WEIGHTS_RESIDENT=1) but the release named {_foreign}; "
                    "only kv_cache/cuda_graph may sleep here -- nothing paused"
                )

        # #1233 one-backup flip: the weights are a FAMILY of tags (the base
        # GPU_MEMORY_TYPE_WEIGHTS plus weights_<k> per layer chunk, see
        # weg2_memory_saver.weights_family_tags) and one sleep may arrive as
        # several RPCs, one tag each, interleaved by the front with the other
        # group's wake.  Upstream's own offload_tags set is the ledger of what
        # is paused: the FIRST family tag of a sleep exports the static state
        # (buffers are read while every page is still mapped), and the sleep
        # is graded when the whole WEG2_SLEEP_TAGS population is paused.
        weights_tags = [t for t in tags if is_weights_family_tag(t)]
        # 2026-09-15: per-LEG lane counters (buffer slot, drain rule) -- both
        # sides walk the same tags per leg, so both start at 0 here.
        try:
            self._weg2_xchg_lane_seq = {}
            self._weg2_seam_after_parts = {}
            self._weg2_seam_leg_inventory = None
            # the derivation cache is PER BOOT (weg2xsn98): the manifests
            # are written at load and the placement key is identical on
            # every leg, so the join/plan/books/shadow plan derived on the
            # first leg serve every later one -- ~1 s off each leg start,
            # which is what the waking side's first collect waited for.
            if getattr(self, "_weg2_xchg_leg_cache", None) is None:
                self._weg2_xchg_leg_cache = {}
        except AttributeError:
            pass
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
            # #1457: no KV zeroing before a pause that discards the pages (the
            # mamba/req_to_token resets in the flush still run on mapped pages).
            self.flush_cache(zero_kv=False)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            _weg2_ph("kv_pause")
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
            # weg2xsn269: the deposit lanes of this leg book their on-card
            # IPC stagings against this credit (see _weg2_stage_charge).
            self._weg2_leg_credit = credit
            _weg2_ph("census_credit")
            # H25 (C): the draft goes to host RAM FIRST, while every page is
            # still mapped, and its VRAM is credited before the family's.
            # H31b: a deferred extra-row fill still running writes into pages
            # this sleep is about to pause (and the park reads); idempotent
            self._weg2_rearm_defer_settle()
            if not family_paused_before:
                self._weg2_park_draft_at_sleep(credit)
                _weg2_ph("draft_park")
            # #1273 S5b: the SOURCE half of the shadow, at the last instant the
            # weight pages are mapped.  See _weg2_shadow_source_leg for why it
            # is here and not after the pause.  Never raises; on the `ring`
            # arm it returns having touched nothing (`bounce_lane_armed()`
            # is False there and only there, #1273 B4q -- CORRECTED 2026-09-14,
            # #1334: this comment used to say "every arm but
            # --weg2-weight-source shadow", which stopped being true the
            # moment B4q widened the gate from `shadow_armed()` to
            # `bounce_lane_armed()` and was never reconciled here -- under
            # `--weg2-weight-source exchange` (BOOT7's A1) this hook ALSO
            # runs, proven by execution in
            # test_weg2_1334_bounce_lane_axis.py).
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
            _weg2_ph("source_hook")
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
                _weg2_ph("seam_before")
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
            # #1350g: NAME THE LEG, or the whole NEED series is unusable.
            # Measured on boot weg2xsn25: 90 of 90 `WEG2-RING NEED` lines read
            # `leg=unknown`, because `set_leg` existed and nothing called it.
            # Without a leg identity the per-card sums cannot be split "per
            # sleep leg" and the ring-anchor measuring boot produces nothing --
            # the emitter's three fields (#1350d) are inert until this one line.
            # The epoch is the SAVER's own flip epoch where it is published and
            # the leg is a d2h sleep by construction here; an absent epoch
            # prints as `?` rather than as a number this file invents.
            weg2_ring_guard.set_leg(
                os.environ.get("TMS_HOST_RING_EPOCH", "") or "?",
                self._weg2_group_name(), "peer",
            )
            # THE DEPOSIT, BEFORE THE PAGES GO. The pause below releases the
            # weight tags; a deposit after it would read pages this rank has
            # already given back. weg2xsn25 ran with no depositor at all.
            # #1378 xsn36: RETIRED as a whole-leg lock -- it deadlocked the
            # co-located pair (weg2xsn35/36). The serialisation now happens
            # per COPY inside run_bounce_leg (pcie_uuid); the waits between
            # the copies stay outside every lock.
            try:
                self._weg2_flip_index_now = _weg2_flip_index_of(getattr(recv_req, "epoch", None))
            except Exception:  # noqa: BLE001 -- the stubs carry no epoch: the counter seq stays
                pass
            with self._weg2_pcie_lock_retired("sleep-D2H " + ",".join(weights_tags)):
                _t_prev_end = None
                logger.info("WEG2-SLEEP-PRELOOP ms " + " ".join(f"{_n}={_ms:.0f}" for _n, _ms in _weg2_ph_l) + f" t={time.time():.3f}")
                for tag in weights_tags:
                    weg2_ring_guard.guard_tag(
                        tag,
                        int(tag_bytes.get(tag, 0)),
                        self._weg2_ring_stats,
                        peer_hint="the group waking on this card",
                    )
                    # #1374 F1: THE DEPOSIT OF THIS TAG, IMMEDIATELY BEFORE
                    # ITS PAUSE. The bytes are still mapped here and the
                    # credit below follows, so the collector this deposit
                    # needs can run -- which is the ordering boot weg2xsn30
                    # did not have. The buffer holds a whole tag (Option 1),
                    # so this completes without its collector.
                    _t_dep0 = time.perf_counter()
                    _gap_ms = ((_t_dep0 - _t_prev_end) * 1000
                               if _t_prev_end is not None else 0.0)
                    self._weg2_xchg_deposit_before_sleep(
                        flip_index=_weg2_flip_index_of(
                            getattr(recv_req, "epoch", None)),
                        tag=tag)
                    # S1 (PLAN_FLIP_LANES_0917): the native pause() ends in
                    # cuMemUnmap, which synchronises the WHOLE device -- every
                    # copy this process still has in flight (the HiCache KV
                    # write-back among them) is paid inside 'pause_ms'. Take
                    # that wait out on its own clock so the line says which
                    # of the two it was: the device's backlog (sync_ms) or the
                    # unmap itself (pause_ms).
                    _t_sync0 = time.perf_counter()
                    try:
                        torch.cuda.synchronize()
                    except Exception:  # noqa: BLE001 -- no device: nothing to wait for
                        pass
                    _sync_ms = (time.perf_counter() - _t_sync0) * 1000
                    t_tag = time.perf_counter()
                    self.memory_saver_adapter.pause(tag)
                    weg2_per_tag[tag] = [
                        float(tag_bytes.get(tag, 0)),
                        (time.perf_counter() - t_tag) * 1000,
                    ]
                    _t_cr0 = time.perf_counter()
                    if credit is not None:
                        # The device bytes this tag's pause just gave back --
                        # the same number, from the same instrument, that the
                        # waking rank is waiting on.
                        credit.publish(tag, tag_bytes.get(tag, 0))
                    _t_prev_end = time.perf_counter()
                    # 2026-09-15 (Nutzer-Order: die Schlaefer-Schleife je Tag
                    # messen): deposit = plan filter + gap check + lanes,
                    # pause = tms unmap, credit = publish, gap = the loop's own
                    # work between the previous tag's credit and this deposit.
                    logger.info(
                        "WEG2-SLEEP-TAG-TIME tag=%s deposit_ms=%.0f sync_ms=%.0f pause_ms=%.0f "
                        "credit_ms=%.0f gap_ms=%.0f total_ms=%.0f t0=%.3f t=%.3f",
                        tag, (_t_sync0 - _t_dep0) * 1000, _sync_ms,
                        weg2_per_tag[tag][1], (_t_prev_end - _t_cr0) * 1000,
                        _gap_ms, (_t_prev_end - _t_dep0) * 1000,
                        # wall-clock stamps (order point 2 timeline): the
                        # front log is wall-clock ms, so the legs align on it
                        time.time() - (time.perf_counter() - _t_dep0),
                        time.time())
            # #1360b: ONE `WEG2-RING NEED` LINE PER SAVED TAG, not per
            # ring-carried tag.  The loop above guards the `weights_*` family
            # because those are the tags whose bytes the peer has to release --
            # but the RING-ANCHOR question is a different one ("does
            # weights + weights_draft cover everything `enable_cpu_backup`
            # saves?"), and it cannot be answered from a population that only
            # contains the weights family.  Measured on weg2xsn24 and weg2xsn25:
            # the whole NEED tag census reads `weights_0..7`, `weights_draft`,
            # `weights` and NOTHING else, so the "extra" set the anchor needs is
            # empty BY CONSTRUCTION and the manifest coverage is unmeasurable.
            #
            # OBSERVATION ONLY, and that is the load-bearing restriction: these
            # tags are emitted through `record()`, never `guard_tag()`, so
            # nothing waits, nothing refuses and no flip timing moves.  A tag
            # outside the weights family is not ring-carried, so a `free`
            # comparison would be meaningless for it -- the line exists to state
            # its BYTES under its own name, which is exactly what the anchor
            # sums.  `free` is carried from the guard's own last reading so the
            # five fixed fields stay on every line and the series still parses.
            _free_now = weg2_ring_guard.free_mib_or_none(self._weg2_ring_stats)
            for _tag, _nb in sorted(census.items()):
                if _tag in weights_tags:
                    continue
                weg2_ring_guard.record(
                    _tag,
                    ring_guard.need_mib(int(_nb), weg2_ring_guard.granule_bytes),
                    int(_free_now if _free_now is not None else 0),
                )
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
            # 2026-09-15: the depositor confirms every outstanding drain
            # (the last `depth` tags per lane) and frees its on-card staging.
            self._weg2_xchg_drain_outstanding()
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
                "WEG2-SLEEP-CHUNK tags=%s paused in %.0f ms (offload_tags now %s) t=" + f"{time.time():.3f}",
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
            # H15: the context's local-memory reservation, which no tag and no
            # empty_cache() reaches -- parked BEFORE the census below, so the
            # WEG2-SLEEP-RESIDUE line reads the card after it.
            self._weg2_park_lmem_at_sleep()
            self._weg2_log_sleep_acceptance(
                weg2_before_census, sorted(self.offload_tags)
            )
            self.weg2_sleep_before = None

        if weg2_memory_saver_on:
            self._weg2_log_dc_breakdown("release tags=%s" % (list(tags),))  # #1446
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
    @_vram_peak_leg("resume")
    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        # #1285: see the release leg.  This one is the sharper case -- the wake's
        # very first mutation below drops each tag from the offload set, which
        # raises KeyError on a repeat, so without this the retry kills the group.
        self._weg2_raise_pending_seam_refusal()  # #1450
        replay = self._weg2_leg_replay("resume", recv_req)
        _weg2_ph_t = [time.perf_counter()]
        _weg2_ph_l = []
        
        def _weg2_ph(name):
            _n = time.perf_counter()
            _weg2_ph_l.append((name, (_n - _weg2_ph_t[0]) * 1000))
            _weg2_ph_t[0] = _n
        
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

        def _weg2_kv_resume_part():
            # Wake-Parallel: kv_cache resume + pool restore ONLY. Safe before
            # the weight legs (xsn315: clearing DORMANT here admitted requests
            # onto paused weights -- two P ranks died, the flip hung).
            # Wake-Parallel (user 18.09.): the kv_cache resume, pool restore, DORMANT
            # clear and hold release -- one block, called EARLY (before the weight
            # legs, so the held requests' arena loads overlap the legs) when the
            # card can fund the pool now, else LATE (the old order).
            # xsn317/318/321: the early kv-only call resumed kv_cache and the
            # weights call's late site resumed it AGAIN -- torch_memory_saver
            # answers a second resume of an unpaused tag with exit(1) ("Cannot
            # resume allocation that is not paused"), the rank dies without a
            # Python traceback. The RESUME half runs once per epoch.
            if _kv_epoch is not None and self._weg2_kv_resumed_epoch == _kv_epoch:
                logger.info("WEG2-WAKE-KV-RESUME already ran in epoch=%s: resume half skipped", _kv_epoch)
                return True
            # #1490: REFUSE A RESUME THE CARD CANNOT FUND, AND REFUSE TO
            # BELIEVE ONE THAT DID NOT LAND. `_weg2_wake_kv_first_ok` above
            # already computed this rank's fit and printed it --
            #   WEG2-WAKE-KV-FIRST LATE free=5974 MiB floor=700 MiB need=6904 MiB
            # -- and then used the answer ONLY to pick EARLY vs LATE. Boots
            # weg2xsn406 and weg2xsn408 both resumed LATE into a card that
            # arithmetic had just called short; the hook rolled the tag back,
            # its void ABI reported nothing, and the pool work below zeroed
            # unmapped memory. TP0 and TP1 of xsn408 died there with no Python
            # traceback. The fit figure is re-read HERE because the legs run
            # between the plan and this site and change it.
            from sglang.srt.utils.torch_memory_saver_adapter import (
                Weg2TmsResumeRefused,
            )
            from sglang.srt.weg2.wake_kv import kv_resume_fit_refusal

            # #1491: the census BEFORE the resume, on every wake. This is the
            # reading that did not exist: `stage=release` runs at the SLEEP,
            # so a creep that happens between a sleep and the next wake -- the
            # co-resident group's prefill, on the same card -- was never in
            # any log. Emitted before the fit check so that a refused wake is
            # explained by the line above it, not only named by the line below.
            # H15: the local-memory reservation goes back FIRST, so the census
            # and the fit check below read the card with it in place.
            self._weg2_restore_lmem_at_wake()
            self._weg2_log_dc_breakdown("wake-pre-kv epoch=%s" % (_kv_epoch,))

            _kv_need = None
            _kv_free = None
            try:
                _kv_need = int(self._weg2_tag_bytes(GPU_MEMORY_TYPE_KV_CACHE) or 0)
                _kv_free = self._weg2_free_bytes()
                _kv_floor = int(self._weg2_corridor_floor_bytes() or 0)
            except Exception as _exc:  # noqa: BLE001 -- no probe, no refusal
                logger.info("WEG2-WAKE-KV-FIT skipped (%s: %s)", type(_exc).__name__, _exc)
                _kv_floor = 0
            _unfit = kv_resume_fit_refusal(_kv_free, _kv_need, _kv_floor)
            if _unfit is not None:
                logger.error(
                    "W114 Weg2KvResumeRefused epoch=%s: %s. The kv_cache tag "
                    "stays PAUSED and this rank stays DORMANT -- the pool is "
                    "NOT cleared, NOT zeroed and NOT marked resumed, because "
                    "every one of those touches unmapped memory. That touch is "
                    "what killed TP0 and TP1 of boot weg2xsn408 with no Python "
                    "traceback at all. This is a NAMED refusal of the wake, "
                    "not a silent skip: the group stays alive and asleep, and "
                    "the front's W4 Weg2WakeRefused now states a fact instead "
                    "of marking a grave.",
                    _kv_epoch, _unfit,
                )
                return False
            try:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
            except Weg2TmsResumeRefused as _exc:
                logger.error(
                    "W114 Weg2KvResumeRefused epoch=%s: %s The pool is NOT "
                    "cleared and NOT marked resumed; this rank stays DORMANT.",
                    _kv_epoch, _exc,
                )
                return False
            self._weg2_kv_resumed_epoch = _kv_epoch
            _weg2_ph("kv_resume")
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
                # #1455: the tree keeps what the dormant hold prefetched during
                # the flip; only the POOL state that the remap left undefined is
                # restored (req_to_token, mamba maps, allocator, KV zero).
                # SGLANG_WEG2_WAKE_FLUSH=1 restores the full flush (tree reset).
                if os.environ.get("SGLANG_WEG2_WAKE_FLUSH", "0") == "1":
                    flushed = self.flush_cache()
                else:
                    flushed = self._weg2_wake_restore_pools()
                # H81: D's phase is over (the front drained it, #1011) -- the
                # END anchors P's sleep reset held for D go back to the arena
                self._weg2_release_carrier_hold_at_wake()
                _weg2_ph("flush")
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
        def _weg2_kv_clear_part():
            # Wake-Parallel: DORMANT cleared, hold release, store rescan, disagg
            # queues -- only once the weights are resumed (the late site).
            scheduler = self.scheduler
            # H25 (C): the parked draft's H2D is joined HERE, the last instant
            # before a request can reach the verifier.
            self._weg2_unpark_draft_join(_weg2_ph_l, where="admit")
            if scheduler is not None:
                # W25: the pools are mapped again; the admission seams admit.
                scheduler.weg2_dormant = False
                # fnFL2x36: the standstill pass bound gets a grace of one
                # stall window from here (scheduler._weg2_note_prefetch_progress)
                scheduler._weg2_last_wake_t = time.perf_counter()
                logger.info(
                    "WEG2-DORMANT cleared: kv_cache resumed, admission seams admit"
                )
                # 19.09. (xsn380): the admission-wedge clocks are ABSOLUTE per
                # rank and kept running through this rank's sleep -- the last
                # first token was 26 s before the wake, a 100k arrival needed
                # 3 s of store prefetch, and the 20-s alarm fired 3 s after the
                # wake ("28 s since first-token progress"), the recovery turned
                # the arrival into an intake stall and the front flipped away
                # with the batch still queued. A request queued during the
                # flip is not older than the wake: restart both clocks here.
                try:
                    scheduler.note_first_token_progress()
                    scheduler.note_prefill_progress()
                    logger.info("WEG2-WAKE wedge clocks restarted (first-token, prefill): "
                                "queue age counts from this wake")
                except Exception as _clk_exc:  # noqa: BLE001 -- stubs without the clocks
                    logger.info("WEG2-WAKE wedge clocks not restarted: %r", _clk_exc)
                scheduler._weg2_post_wake_pass_n = 0  # arm the post-wake pass timer
                scheduler._weg2_post_wake_t = None
                # weg2xsn288: a new phase -- no intake-stall hold and no
                # reported rid from the previous one survives the wake.
                _iw = getattr(scheduler, "_weg2_intake_watch", None)
                if _iw is not None:
                    _iw.reset()
                self._weg2_rescan_store_index()
                _rel = getattr(scheduler, "_weg2_release_dormant_hold", None)  # #1443
                if callable(_rel):
                    _rel()
                _weg2_ph("store_rescan")
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
        def _weg2_kv_block():
            # #1490: the CLEAR half zeroes req_to_token, the mamba maps and the
            # KV buffers. Every one of those writes into the kv_cache region,
            # so it may run ONLY behind a resume that actually mapped it.
            if _weg2_kv_resume_part() is False:
                logger.error(
                    "W114 Weg2KvResumeRefused: the clear half is SKIPPED "
                    "because the resume did not land -- zeroing a paused pool "
                    "is the fault, not the report of it."
                )
                return False
            _weg2_kv_clear_part()
            return True

        _weg2_kv_done = False
        from sglang.srt.weg2.wake_kv import wake_kv_plan as _wk_plan
        _kv_epoch = getattr(recv_req, "epoch", None)
        _kv_in = GPU_MEMORY_TYPE_KV_CACHE in tags
        _plan = _wk_plan(
            kv_in_tags=_kv_in,
            weights_in_tags=any(is_weights_family_tag(t) for t in tags),
            fundable=(self._weg2_wake_kv_first_ok(tags) if _kv_in else False),
            deferred=bool(self._weg2_kv_deferred),
            epoch=_kv_epoch, epoch_done=self._weg2_kv_epoch_done,
            weights_done=(self._weg2_weights_epoch_done is not None
                          and self._weg2_weights_epoch_done == _kv_epoch),
        )
        logger.info("WEG2-WAKE-KV-PLAN %s epoch=%s kv=%s weights=%s deferred=%s", _plan, _kv_epoch,
                    _kv_in, any(is_weights_family_tag(t) for t in tags), bool(self._weg2_kv_deferred))
        _weg2_kv_resumed_early = False
        if _plan == "early":
            # #1490: a refused resume must not be recorded as an early one --
            # the late site below reads this flag to decide whether the CLEAR
            # half may run, and a False here is what sends it back through
            # `_weg2_kv_block` (which retries, the legs having freed memory in
            # between) instead of straight into the pool.
            _weg2_kv_resumed_early = _weg2_kv_resume_part() is not False
            self._weg2_kv_deferred = False
            if not any(is_weights_family_tag(t) for t in tags):
                # a kv-only call: the CLEAR half and the cuda_graph resume belong
                # to the weights call (its late site), after the legs
                self._weg2_kv_deferred = True
                self._weg2_graph_deferred = True
                _weg2_kv_done = True
        elif _plan == "defer":
            self._weg2_kv_deferred = True
            _weg2_kv_done = True
            logger.info("WEG2-WAKE-KV-FIRST DEFERRED epoch=%s: kv_cache resumes inside the weights call after its legs", _kv_epoch)
        elif _plan == "done":
            _weg2_kv_done = True
            logger.info("WEG2-WAKE-KV-FIRST already resumed in epoch=%s: nothing to do", _kv_epoch)

        for tag in tags:
            self.offload_tags.remove(tag)

        def _weg2_graph_block():
            # Wake-Parallel (xsn317): the cuda_graph resume references weight VA;
            # it must never run before the weight legs. An early kv-only call
            # defers it to the weights call's late site.
            t_graph = time.perf_counter()
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)
            _weg2_ph("cg_resume")
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

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags and not self._weg2_graph_deferred:
            _weg2_graph_block()
        elif GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            logger.info("WEG2-WAKE-KV-FIRST cuda_graph deferred to the weights call (legs first)")
            try:
                if GPU_MEMORY_TYPE_CUDA_GRAPH not in self.offload_tags:
                    self.offload_tags.add(GPU_MEMORY_TYPE_CUDA_GRAPH)
            except Exception:  # noqa: BLE001 -- list-typed offload_tags
                try:
                    self.offload_tags.append(GPU_MEMORY_TYPE_CUDA_GRAPH)
                except Exception:  # noqa: BLE001
                    pass
        weights_tags = [t for t in tags if is_weights_family_tag(t)]
        # 2026-09-15: per-LEG lane counters (buffer slot, drain rule) -- both
        # sides walk the same tags per leg, so both start at 0 here.
        try:
            self._weg2_xchg_lane_seq = {}
            self._weg2_seam_after_parts = {}
            self._weg2_seam_leg_inventory = None
            # the derivation cache is PER BOOT (weg2xsn98): the manifests
            # are written at load and the placement key is identical on
            # every leg, so the join/plan/books/shadow plan derived on the
            # first leg serve every later one -- ~1 s off each leg start,
            # which is what the waking side's first collect waited for.
            if getattr(self, "_weg2_xchg_leg_cache", None) is None:
                self._weg2_xchg_leg_cache = {}
        except AttributeError:
            pass
        if weights_tags:
            # Wake-H2D: with the cpu backup this recommit refills from the host
            # buffer (and, with the #1233 patched hook, frees that buffer).
            # Same lock, same reason as the sleep leg above.
            t_w0 = time.perf_counter()
            _weg2_ph("pre_leg")
            shm0 = self._weg2_rss_shmem_mib()
            tag_bytes = {tag: self._weg2_tag_bytes(tag) for tag in weights_tags}
            # H31: the Platztausch pad+extra rows, issued per tag behind its
            # resume below and joined at the expert-rearm after the legs.
            _rearm_pf = self._weg2_rearm_prefetch_begin(_weg2_ph_l)
            # S7 (#1273): tag -> the saver's own pass-1/pass-2 decomposition of
            # that tag's resume, or None where the instrument is absent.
            weg2_map_stats: Dict[str, Optional[Dict[str, float]]] = {}
            credit, credit_epoch = self._weg2_credit_reader(
                getattr(recv_req, "epoch", None)
            )
            # #1378 xsn36: RETIRED, same shape as the sleep leg above.
            try:
                self._weg2_flip_index_now = _weg2_flip_index_of(getattr(recv_req, "epoch", None))
            except Exception:  # noqa: BLE001 -- the stubs carry no epoch: the counter seq stays
                pass
            # 18.09. (xsn369): with two collects in flight, a HOST/IPC lane
            # (unit semaphores, slot counter) must still see its tags in
            # order -- tag t's non-BAR1 lanes wait for tag t-1's collect
            # (an Event per tag); BAR1 lanes run free (seq per tag, done flag).
            import threading as _thr
            self._weg2_leg_tag_order = [str(t) for t in weights_tags]
            self._weg2_tag_done = {i: _thr.Event() for i in range(len(weights_tags))}
            from sglang.srt.environ import envs as _envs_h11
            from sglang.srt.weg2.lane_turns import LaneTurns

            self._weg2_lane_turns = (LaneTurns(_WEG2_TURN_LANES)
                                     if _envs_h11.SGLANG_WEG2_WAKE_LANE_TURNS.get() else None)
            try:
                if self._weg2_wake_weight_carrier() != self.CARRIER_EXCHANGE:
                    for _ev in self._weg2_tag_done.values():
                        _ev.set()   # no exchange collects at all: the gate never waits
            except Exception:  # noqa: BLE001 -- stubs without a carrier
                for _ev in self._weg2_tag_done.values():
                    _ev.set()
            with self._weg2_pcie_lock_retired("wake-H2D " + ",".join(weights_tags)):
                # 2026-09-15 (Punkt 2, SGLANG_WEG2_WAKE_OVERLAP default 1)
                _wake_worker = None
                _wake_futs = []
                if _weg2_wake_overlap_armed():
                    from concurrent.futures import ThreadPoolExecutor as _TPE
                    # 18.09. (xsn367): TWO collects in flight -- the flip's critical chain
                    # is PP0's tags in series, the other ranks' tags overlap it
                    # (they arrive over other links); with one worker PP0 idled
                    # 0.8 s behind weights_6/7 at every flip
                    _n_wake_workers = _weg2_wake_collect_workers()
                    # H11: the run-ahead bound below keeps up to
                    # _n_wake_workers + 1 collects submitted (t-2 finishing,
                    # t-1, t); a pool of exactly _n_wake_workers queued the
                    # just-resumed tag t behind them -- x105 PP0 weights_1/2
                    # waited 115/139 ms on its lane p0 while D TP1's two
                    # workers held PP2's and PP1's tags. The spare worker(s)
                    # change no resume and no VRAM, only who may collect.
                    _wake_worker = _TPE(
                        max_workers=_n_wake_workers + max(0, _envs_h11.SGLANG_WEG2_WAKE_COLLECT_SPARE.get()),
                        thread_name_prefix="weg2-wake-collect")
                for _ti, tag in enumerate(weights_tags):
                    # C14: the device bytes this tag needs may only exist once
                    # the co-located SLEEPING rank has released them, and with
                    # C9 both legs are in flight.  Waiting here turns a race
                    # into a bounded wait whose expiry is a NAMED refusal; when
                    # the card is not short the call returns without waiting at
                    # all, which is every non-co-located boot.
                    # #1378 xsn67: PP0 went silent for >180 s after the resume
                    # request, printed no W35 (credit budget 120 s) and no
                    # PTRATTR -- the log could not say WHICH step it was in.
                    # One line per tag BEFORE the credit wait and the resume,
                    # so a stuck rank names its phase; nothing to derive.
                    logger.info(
                        "WEG2-RESUME begin tag=%s need_mib=%d credit=%s epoch=%s "
                        "free_mib=%s -- next: credit wait (budget %.0fs), then "
                        "memory_saver_adapter.resume(tag)",
                        tag, int(tag_bytes.get(tag, 0)) // MIB_,
                        "armed" if credit is not None else "none", credit_epoch,
                        (self._weg2_free_bytes() or 0) // MIB_,
                        WEG2_GROUP_FENCE_BUDGET_S,
                    )
                    self._weg2_await_vram_credit(
                        credit, tag, tag_bytes.get(tag, 0), credit_epoch,
                        submitted=list(weights_tags[:_ti]),
                    )
                    t_tag = time.perf_counter()
                    logger.info("WEG2-RESUME credit-ok tag=%s -- entering resume(tag)", tag)
                    # weg2xsn269: a refused remap raises here (rc from the
                    # hook's tms_resume_rc) instead of exit(1) inside cuMemCreate.
                    weg2_tms_resume(self.memory_saver_adapter, tag)
                    # H31: this tag's buffers exist now -- their pad+extra rows
                    # go on the side stream while the legs run on.
                    _rearm_pf = self._weg2_rearm_prefetch_issue(_rearm_pf, tag)
                    # #1378 xsn62: DID THE RESUME ACTUALLY MAP ANYTHING?
                    #
                    # weg2xsn61 read the destination of the first copy-out and
                    # got `dst_rc=0 dst_type=0 dst_device=-1` -- the call
                    # SUCCEEDED and the driver does not know the address, which
                    # is what a reserved-but-not-committed VMM range looks like.
                    # Two seconds later: SIGSEGV. So the bytes the collect
                    # writes into are not mapped.
                    #
                    # I had already checked that this `resume(tag)` stands
                    # BEFORE the collect (:5757) and concluded the pages were
                    # there. That check only proves the CALL precedes it, never
                    # that it TOOK EFFECT -- "aufgerufen != gewirkt", one level
                    # past the "resolved != mapped" trap that cost two earlier
                    # walls. This closes that gap with a reading instead of an
                    # inference: same probe, same producer (`bx.ptr_attrs`), one
                    # row per tag, right after the resume.
                    #
                    # `type=2` here and `0` at the copy-out means the mapping is
                    # lost BETWEEN the two -- a second actor. `type=0` already
                    # here means THIS resume does not map this tag's tensors,
                    # and the root is in the memory saver or the tag-to-tensor
                    # attribution, not in the transport at all.
                    try:
                        from sglang.srt.weg2 import (
                            weight_exchange_bounce as _bx_probe,
                        )
                        # #1378 xsn63 -- THE PROBE'S OWN FIRST DEFECT, fixed by
                        # its first execution. xsn62 printed `no-model` on every
                        # rank for every tag: `self.model_runner` does not exist
                        # here. The class has its own accessor and it is the one
                        # every other site uses -- `_weg2_model_for_group`
                        # (:4468), which reaches the model via `tp_worker` for P
                        # and `draft_worker` for D. Guessing an attribute name
                        # instead of using the accessor the file already has is
                        # the prior-art failure this ticket keeps repeating.
                        _pm = self._weg2_model_for_group(
                            self._weg2_group_name())
                        _probe_row = None
                        if _pm is not None:
                            # xsn63 -- THE PROBE'S THIRD DEFECT: "the first
                            # parameter" was `visual.patch_embed.proj.weight`,
                            # which the manifest files under tag `weights`,
                            # not under the tag just resumed, so its type=0
                            # said nothing. Two readings now, labelled: the
                            # first parameter (whatever it is) AND the tensor
                            # the collect died on, `layers.0.input_layernorm`
                            # (tag weights_0), after EVERY tag's resume -- a
                            # time series over the resume sequence. ABSENT
                            # is a true statement on a PP rank that does not
                            # hold layer 0, not a reading.
                            _first = None
                            _layer0 = None
                            for _pn, _pt in _pm.named_parameters():
                                if _first is None:
                                    _first = (_pn, _pt)
                                if _pn.endswith(
                                        "layers.0.input_layernorm.weight"):
                                    _layer0 = (_pn, _pt)
                                if _first is not None and _layer0 is not None:
                                    break
                            _rows = []
                            for _lbl, _pair in (("first", _first),
                                                ("layer0", _layer0)):
                                if _pair is None:
                                    _rows.append(f"{_lbl}=ABSENT")
                                    continue
                                _prc, _pty, _pdev = _bx_probe.ptr_attrs(
                                    int(_pair[1].data_ptr()))
                                _rows.append(
                                    f"{_lbl} name={_pair[0]!r} rc={_prc} "
                                    f"type={_pty} device={_pdev}")
                            # xsn66 -- THE PURITY CENSUS: of the parameters
                            # whose NAME puts them under this tag (the same
                            # attribution the manifest uses), how many does
                            # the driver see mapped after this tag's resume?
                            # `unmapped>0` here is the weg2xsn66 defect
                            # (a small tensor in a segment another tag owns);
                            # 0 is the acceptance of the per-tag pool.
                            try:
                                from sglang.srt.layers.utils.common import (
                                    get_layer_id as _gli,
                                )
                                from sglang.srt.managers.weg2_memory_saver import (
                                    weight_chunk_tag as _wct,
                                )
                                _n_map = _n_unmap = 0
                                _first_unmapped = None
                                for _pn, _pt in _pm.named_parameters():
                                    _lid = _gli(_pn)
                                    _own = (_wct(_lid) if _lid is not None
                                            else None) or "weights"
                                    if _own != str(tag):
                                        continue
                                    _ty = _bx_probe.ptr_attrs(
                                        int(_pt.data_ptr()))[1]
                                    if _ty == 2:
                                        _n_map += 1
                                    else:
                                        _n_unmap += 1
                                        if _first_unmapped is None:
                                            _first_unmapped = (
                                                f"{_pn}({int(_pt.numel()) * int(_pt.element_size())}B,type={_ty})")
                                _rows.append(
                                    f"own_tag={tag} mapped={_n_map} "
                                    f"unmapped={_n_unmap} "
                                    f"first_unmapped={_first_unmapped}")
                            except Exception as _cx:  # noqa: BLE001
                                _rows.append(
                                    f"census=NOT-MEASURED({type(_cx).__name__})")
                            _probe_row = " | ".join(_rows)
                        if _probe_row is None:
                            # NO explanatory text on a row that measured
                            # NOTHING. xsn62's version appended "type 0 means
                            # the driver does not know this tensor's pages" to a
                            # `no-model` row, which reads as a FINDING where
                            # there was no reading at all -- the instrument-text
                            # trap, in my own instrument.
                            logger.info(
                                "WEG2-RESUME-PTRATTR tag=%s NOT-MEASURED "
                                "reason=no-model-for-group -- this row carries "
                                "no reading and says nothing about the pages",
                                tag)
                        else:
                            logger.info(
                                "WEG2-RESUME-PTRATTR tag=%s %s -- type 0 means "
                                "the driver does not know this tensor's pages "
                                "AFTER the resume (0=unregistered 1=host "
                                "2=device)", tag, _probe_row)
                    except Exception as _probe_exc:  # noqa: BLE001
                        logger.info("WEG2-RESUME-PTRATTR tag=%s unavailable=%s",
                                    tag, type(_probe_exc).__name__)
                    weg2_per_tag[tag] = [
                        float(tag_bytes.get(tag, 0)),
                        (time.perf_counter() - t_tag) * 1000,
                    ]
                    # order point 2 timeline: when THIS tag's resume ended
                    # (wall clock, aligns with the front's ms log and the
                    # depositor's SLEEP-TAG-TIME t0/t stamps)
                    logger.info("WEG2-WAKE-TAG-TIME tag=%s resume_ms=%.0f t0=%.3f t=%.3f",
                                tag, weg2_per_tag[tag][1],
                                time.time() - (time.perf_counter() - t_tag),
                                time.time())
                    # 18.09. (Flip-Schwanz): the kv pool comes back MID-LEGS as soon
                    # as the card funds it plus the next tag, and the held
                    # requests' pages start loading while the remaining legs run
                    # (xsn368: 1.1 s of init_new loads AFTER the wake).
                    # (the front wakes with TWO RPCs: the weights family first,
                    # kv_cache + cuda_graph after the legs -- so the pool is
                    # still PAUSED here whatever this call's tag list says; the
                    # kv RPC then finds it resumed and runs only the clear half)
                    if (GPU_MEMORY_TYPE_KV_CACHE in self.offload_tags
                            and not _weg2_kv_resumed_early
                            and not (_kv_epoch is not None and self._weg2_kv_resumed_epoch == _kv_epoch)
                            # DEFAULT OFF (xsn377): the decision is per rank, and it
                            # differed (TP1 6.5 GB funded, TP0 12.3 GB never) -- the
                            # preload then moved one rank's prefixes and the extend
                            # died on PrefixLensRankDivergence. A group-uniform verdict
                            # would be the tightest rank's (TP0: never before the last
                            # tag), so this stays a lever for a STAGED pool, not for
                            # the whole one.
                            and str(os.environ.get("SGLANG_WEG2_WAKE_KV_MID", "0")).strip().lower()
                            not in ("0", "false", "no", "off")):
                        try:
                            from sglang.srt.weg2.wake_kv import kv_mid_ok as _kv_mid_ok
                            _ti_mid = list(weights_tags).index(tag)
                            _rest = list(weights_tags)[_ti_mid + 1:]
                            _rest_need = sum(int(tag_bytes.get(t, 0) or 0) for t in _rest)
                            _kv_need = int(self._weg2_tag_bytes(GPU_MEMORY_TYPE_KV_CACHE) or 0)
                            _free_mid = self._weg2_free_bytes()
                            _floor_mid = int(self._weg2_corridor_floor_bytes() or 0)
                            _mid_ok = _kv_mid_ok(_free_mid, _floor_mid, _kv_need, _rest_need)
                            logger.info("WEG2-WAKE-KV-MID %s after tag=%s free=%s MiB floor=%d MiB kv=%d MiB "
                                        "remaining=%d tags(%d MiB)", "RESUME" if _mid_ok else "wait", tag,
                                        (int(_free_mid) >> 20) if _free_mid is not None else None,
                                        _floor_mid >> 20, _kv_need >> 20, len(_rest), _rest_need >> 20)
                            if _mid_ok:
                                _t_mid = time.perf_counter()
                                # #1490: same rule mid-legs. A refusal here is
                                # not an error -- the remaining legs still free
                                # memory, so the late site gets the next try --
                                # but it may never be recorded as a resume.
                                if _weg2_kv_resume_part() is not False:
                                    _weg2_kv_resumed_early = True
                                    _n_pre = self._weg2_preload_hold()
                                    logger.info("WEG2-WAKE-KV-MID resumed after tag=%s preload=%d ms=%.0f",
                                                tag, _n_pre, (time.perf_counter() - _t_mid) * 1000)
                                else:
                                    logger.info("WEG2-WAKE-KV-MID refused after tag=%s: the late "
                                                "site retries once the remaining legs have run", tag)
                        except Exception as _mid_exc:  # noqa: BLE001 -- the late site stays
                            logger.info("WEG2-WAKE-KV-MID skipped after tag=%s: %r", tag, _mid_exc)
                    from sglang.srt.managers.weg2_memory_saver import (
                        GPU_MEMORY_TYPE_WEIGHTS_DRAFT as _WEIGHTS_DRAFT_TAG,
                    )

                    # NUTZER-ORDER 2026-09-14, REVERSING #1394'S OWN ORIGINAL
                    # PRIORITY: "draft ist auch nur ein layer... warum muss er
                    # ueber den ring gehen?" -- he is right, and the join now
                    # gives him the source (`_weg2_shadow_plan`'s per-runner
                    # merge, above). So the disk-reload short-circuit that
                    # used to run BEFORE even asking the carrier -- correct
                    # ONLY while the exchange structurally never covered this
                    # tag at all -- would now silently PREVENT the real
                    # exchange from ever being tried, on every boot, forever.
                    # The exchange is asked FIRST, exactly like any other
                    # tag; disk is the FALLBACK, taken only when this leg's
                    # own collect found zero descriptors for `weights_draft`
                    # (`_weg2_xchg_inject_weights`'s return value, computed at
                    # the one frame that holds the filtered `_cdescs` --
                    # `_weg2_xchg_inject_from_peer`, not re-derived here).
                    # `_weg2_xchg_draft_reload_from_disk` stays itself a
                    # no-op (returns False) whenever the ring still covers
                    # this tag or there is no draft shard here, so this whole
                    # branch costs nothing on every OTHER tag and every
                    # pre-#1369 boot.
                    # #1374 F1: COLLECT THIS TAG NOW, while it is the tag the
                    # resume has just mapped -- resume(t) -> collect(t) ->
                    # post_drained(t), the mirror of the source's deposit(t) ->
                    # pause(t) -> credit(t). Before #1374 the whole plan was
                    # collected after the loop, which is why the peer's
                    # per-tag deposit had nobody to drain it.
                    if (_weg2_carrier_this_tag := self._weg2_wake_weight_carrier()) == self.CARRIER_EXCHANGE:
                        if _wake_worker is not None:
                            # 2026-09-15 (Punkt 2): the collect of tag t runs
                            # on ONE worker (sequential, so every lane's unit
                            # semaphores keep their order) while this thread
                            # waits for tag t+1's credit and resumes it.
                            self._weg2_bar1_register(tag)
                            self._weg2_turns_register(tag)
                            _wake_futs.append((str(tag), _wake_worker.submit(
                                self._weg2_wake_collect_one, tag)))
                            # weg2xsn110: BOUNDED run-ahead. Unbounded, the
                            # main thread resumed all ten tags in one second
                            # (card 0 free 13442 -> 12 MiB) before the worker
                            # collected the first; the sleeper's staging then
                            # found no VRAM and the chain wedged (W68 at the
                            # sleeper's drain wait). Resume t+1 may overlap
                            # collect t, nothing further: wait for t-1 here.
                            if len(_wake_futs) > _n_wake_workers:
                                _wake_futs[-(_n_wake_workers + 1)][1].result()
                        else:
                            self._weg2_bar1_register(tag)
                            self._weg2_turns_register(tag)
                            self._weg2_wake_collect_one(tag)
                    elif (str(tag) == _WEIGHTS_DRAFT_TAG
                            and self._weg2_xchg_draft_reload_from_disk()):
                        self._weg2_tag_done_set(tag)  # no collect: the gate must not wait for it
                        pass  # not CARRIER_EXCHANGE at all (e.g. ring, or a
                        # non-authoritative arm) -- the disk path is the only
                        # candidate; itself a no-op when the ring covers it
                    # #1391 (DESK10) ROUND 4: THE SAME PER-TAG STEP, UNDER
                    # SHADOW. Coordinator's hypothesis, verified at this exact
                    # site before building the fix: under CARRIER_EXCHANGE the
                    # branch above already collects per tag, with a real
                    # `tag`, from INSIDE this loop -- so
                    # `_weg2_xchg_bounce_leg`'s `post_drained(tag=...)` gate
                    # fires and D's per-tag `wait_drained` is satisfied. Under
                    # `--weg2-xchg-inject shadow` the ONLY collect used to be
                    # `_weg2_xchg_shadow_compare`, called once AFTER this
                    # whole loop with `tag=None` -- so the SAME gate never
                    # fired and D's lockstep wait never resolved (#1391's
                    # wedge). The comparison itself was never the problem;
                    # WHEN it ran was. Running it HERE, per tag, gives the
                    # per-tag drain its signal without touching the drain
                    # count's own contract (`post_drained` is still called
                    # exactly once per tag, from exactly one branch below --
                    # see the danger-direction mutant in
                    # test_weg2_shadow_per_tag_collect_1391.py for what a
                    # wrong pairing between "which tag was graded" and "which
                    # tag was drained" would cost).
                    elif self._weg2_xchg_shadow_armed_for(_weg2_carrier_this_tag):
                        from sglang.srt.weg2 import weight_exchange as _wx_shadow

                        self._weg2_xchg_inject_weights(
                            tag=tag, mode=_wx_shadow.INJECT_SHADOW)
                        self._weg2_xchg_shadow_compared_per_tag = True
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
            if _wake_worker is not None:
                _t_join = time.perf_counter()
                _errs = []
                for _ftag, _fut in _wake_futs:
                    try:
                        _fut.result()
                    except BaseException as _fexc:  # noqa: BLE001
                        _errs.append((_ftag, _fexc))
                _wake_worker.shutdown(wait=True)
                logger.info("WEG2-WAKE-OVERLAP collects=%d joined_ms=%.0f errors=%d",
                            len(_wake_futs), (time.perf_counter() - _t_join) * 1000,
                            len(_errs))
                if _errs:
                    raise _errs[0][1]
            # xsn265/266: the collector is the LAST reader of every lane it
            # mapped -- the depositor finished writing before it posted full
            # -- so THIS is where the files are truncated (tmpfs bytes
            # return); the depositor only unmaps its side.
            try:
                from sglang.srt.weg2 import weight_exchange_bounce as _bx_rel

                if _bx_rel.seq_release_lanes():
                    _bx_rel.release_host_lane_buffers(truncate=True, log=logger.info)
            except Exception as _rel_exc:  # noqa: BLE001 -- a release never fails a wake
                logger.info("WEG2-SEQ lane-release skipped: %r", _rel_exc)
            weg2_leg_ms = (time.perf_counter() - t_w0) * 1000
            _weg2_ph("leg_collects")
            # fnFL2 v43: THE WEIGHTS-SIDE MIRROR of the graph tag's
            # `_weg2_zero_graph_scratch` above, and for the identical reason.
            # `marlin_make_workspace` registers a semaphore array as a
            # PARAMETER on every Marlin layer (272 int32, created with
            # `torch.zeros`, in no checkpoint and therefore in no exchange
            # plan).  The resume maps FRESH physical pages under the weights
            # region -- on the two-group form, pages the other group just
            # released -- and nothing writes this parameter, so it holds the
            # peer's residue.  Marlin's kernels require it to start at zero:
            # a non-zero semaphore makes them spin or read a partial tile.
            # Runs on the waking side, after the tags are mapped and before
            # any forward.
            # fnFL2x38 (23.09.): BOTH RUNNERS OF THIS RANK, never the group's
            # exchange runner alone. `_weg2_model_for_group("D")` answers the
            # DRAFT (the runner whose region the draft leg addresses); the
            # MoE layers, the Marlin workspaces and the expert pool live in
            # the TARGET. D woke with 0 rearmed layers (P: 3), ran its first
            # forward on P's expert rows and stale pool tables, TP2 died of an
            # illegal memory access two seconds after layer 47.
            _wake_models = self._weg2_wake_models()
            # H31b: the TARGET is zeroed and rearmed BEFORE the draft unpark,
            # the draft behind it. The unpark makes the current stream wait for
            # its 1.5 GB H2D (5090); rearmed behind it, the target's closing
            # sync waited for that copy too (x146 TP0 rearm 139 ms for 232
            # rows, the longest pre-fence tail once the extras are deferred).
            _draft_m = self._weg2_model_for_group("D")
            _early = [_m for _m in _wake_models if _m is not _draft_m]
            _late = [_m for _m in _wake_models if _m is _draft_m]
            # PLATZTAUSCH (Nutzer-Entscheid 22.09.): der Austausch hat nur den
            # Experten-PRAEFIX gefuellt. Pad nullen, Extra-Zeilen aus ihren
            # festen Store-Plaetzen laden, LRU und Pool-Tabellen verwerfen --
            # auf der aufwachenden Seite, vor dem ersten Forward. Ein Fehler
            # hier ist KEIN Warnfall: ein Layer mit Resten der anderen Gruppe
            # rechnet falsch und sagt es nicht.
            from sglang.srt.layers.moe.expert_offload import (
                REARM_PREFETCH_OFF_FIELDS,
                deferred_rows_fill,
                rearm_expert_offload_after_wake,
            )

            _defer = self._weg2_rearm_defer_armed()
            _t_rearm = time.perf_counter()
            # H31: join the side stream first -- only the rest is paid here.
            _pf_join = _rearm_pf.join(_weg2_ph_l) if _rearm_pf is not None else None
            _scratch = self._weg2_zero_local_scratch(_early)
            _rl = _rz = 0
            for _m in _early:
                _l, _z = rearm_expert_offload_after_wake(
                    _m, prefetch=_rearm_pf, defer=_defer, sync=True)
                _rl += int(_l)
                _rz += int(_z)
            _early_ms = (time.perf_counter() - _t_rearm) * 1000
            # H25 (C): the parked draft comes back BEHIND the legs, on a side
            # stream, and overlaps everything below up to the fence.
            self._weg2_unpark_draft_start(credit, credit_epoch, weights_tags, _weg2_ph_l)
            # the draft's own tensors: stream-ordered behind the unpark, no
            # host wait (the admission joins the unpark; forward_stream waits
            # this stream before the first forward)
            _scratch += self._weg2_zero_local_scratch(_late)
            for _m in _late:
                _l, _z = rearm_expert_offload_after_wake(
                    _m, prefetch=_rearm_pf, defer=_defer, sync=False)
                _rl += int(_l)
                _rz += int(_z)
            if _scratch:
                logger.info(
                    "WEG2-RESUME local-scratch zeroed=%d first=%s "
                    "(runtime-built parameters the exchange has no source "
                    "for; see weight_exchange.LOCAL_SCRATCH_REASON)",
                    len(_scratch), _scratch[0],
                )
            if _rl:
                _pf_rows = _pf_join.rows if _pf_join is not None else 0
                _deferred = deferred_rows_fill().rows_pending()
                logger.info(
                    "WEG2-RESUME expert-rearm layers=%d rows_from_store=%d "
                    "serial=%d deferred=%d %s ms=%.0f target_ms=%.0f models=%d "
                    "(Platztausch: Praefix kam ueber den Austausch, Pad+Extra aus dem "
                    "Store, LRU verworfen; H31: prefetched = waehrend der Legs auf dem "
                    "Seitenstrom; H31b: deferred = erst nach dem ersten Decode-Forward, "
                    "bis dahin kalt in den Tabellen)",
                    _rl, _rz + _pf_rows + _deferred, _rz, _deferred,
                    _pf_join.fields() if _pf_join is not None else REARM_PREFETCH_OFF_FIELDS,
                    (time.perf_counter() - _t_rearm) * 1000, _early_ms, len(_wake_models),
                )
            else:
                logger.warning(
                    "WEG2-RESUME expert-rearm NONE: no offload layer on %d wake model(s) "
                    "of group %s -- a MoE group waking without a rearm computes on the "
                    "other group's expert rows (x38)", len(_wake_models), self._weg2_group_name())
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
                _weg2_ph("reload")
                _import_static_state(
                    self.tp_worker.model_runner.model,
                    self.stashed_model_static_state,
                )
                del self.stashed_model_static_state
                # #1273 S5b: the DESTINATION half.  This hook runs after
                # family_complete and after the reload, so whatever DID
                # carry the wake's bytes has already written them -- under
                # `ring`/`tms-backup` that is the ring, and the shadow's
                # stripes have that ground truth to be compared against
                # (CORRECTED 2026-09-14, #1334: this comment used to say
                # "the ring's bytes are final" unconditionally, which is
                # stale under `--weg2-weight-source exchange` +
                # `--weg2-xchg-inject authoritative`, where
                # `_weg2_wake_reload_weights` routes to `CARRIER_EXCHANGE`
                # and neither the ring nor disk ever wrote these bytes --
                # the peer group's own live VRAM did, via the bounce lane.
                # This hook still runs there too (`bounce_lane_armed()` is
                # True for `exchange`, #1273 B4q), it is simply comparing
                # against a different, still-correct ground truth).
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
                _weg2_ph("dest_hook_compare")
                # #1350 SEAM GRADER, destination side.  THE SAME PLACEMENT RULE
                # the two hooks above follow, and for the same reason: this is
                # where the bytes have landed for every carrier, so a reading
                # here grades settled content.  Last of the three deliberately
                # -- it is the only one that can RAISE, and the other two must
                # have reached the log before it does.
                self._weg2_seam_digest_after(recv_req, weights_tags)
                _weg2_ph("seam_after")

        if any(is_weights_family_tag(t) for t in tags):
            self._weg2_weights_epoch_done = _kv_epoch  # the legs of this epoch are collected
        if (GPU_MEMORY_TYPE_KV_CACHE in tags or self._weg2_kv_deferred) and not _weg2_kv_done:
            # the late site: legs first, then kv (old order), or the CLEAR half of
            # a kv resume that already happened early (this call or a deferred one)
            if self._weg2_graph_deferred:
                _weg2_graph_block()  # the weights are resident now
                self._weg2_graph_deferred = False
            if _weg2_kv_resumed_early or (_kv_epoch is not None and self._weg2_kv_resumed_epoch == _kv_epoch):
                _weg2_kv_clear_part()
                _weg2_kv_ok = True
            else:
                _weg2_kv_ok = _weg2_kv_block() is not False
            # #1490: the epoch is DONE only when the pool is actually back. A
            # refused resume that marked the epoch done would make the next
            # call answer "done: nothing to do" and leave the group dormant
            # forever with no second chance and no further line in the log.
            if _weg2_kv_ok:
                self._weg2_kv_epoch_done = _kv_epoch
                self._weg2_kv_deferred = False

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
            _weg2_ph("fence")
            logger.info("WEG2-WAKE-TAIL ms " + " ".join(f"{_n}={_ms:.0f}" for _n, _ms in _weg2_ph_l) + f" t={time.time():.3f}")
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
