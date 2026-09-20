# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 9 -- the GROUP side: who actually runs the transient stage.

Slices 1-6 built the term, the loader, the hand-off and the ordering with its
teardown invariant.  All of it was callable and none of it was CALLED.  This
module is the caller, and the only real decision it makes is WHICH PROCESS.

WHICH PROCESS, and why it is not a rank
---------------------------------------
The obvious answer -- "a P rank, it has the card" -- is the wrong one, and the
reason is delivery.  Under PP3 only stage 0 turns tokens into embeddings; a
tower run on rank 2's card would have to ship its rows to rank 0 before the
prefill, which is a new cross-rank transport for 10 MiB, in the one window
where the layout is supposed to be quiet.

The right answer is the process that already holds the pixels: **the group's
tokenizer / multimodal-processor process.**  Four facts, each checked:

1. It already has the pixel values -- it is what built the ``mm_items``.
2. It is where upstream ITSELF fills this field: ``moss_vl.py:587`` sets
   ``item.precomputed_embeddings`` in the processor, and
   ``base_processor.py:1556-1563`` wraps it for transport afterwards.  So the
   seam is upstream's, not ours.
3. It can bind ANY card (``cuda:<n>``) -- it is not pinned to a rank's device.
   Its CUDA context is a real cost and it is exactly the post
   ``TowerSpec.ctx_bytes`` already carries.
4. Delivery is free.  ``attach_precomputed_embeddings`` REFUSES a tensor still
   on the stage's card, so the rows are on the host by construction, and
   ``_get_precomputed_embedding`` moves them to whatever device asks
   (``mm_utils.py:459-467``).  A host tensor travels with the request through
   the normal path; no transport is added.

THE LIMIT THIS FORM HAS, named instead of hidden
------------------------------------------------
Band displacement (``pause_tag`` / ``resume_tag``,
``offload_movement.py:338-341``) happens inside a RANK's process, on that
rank's own weights.  The tokenizer process cannot reach it.  So:

* **no-eviction placements work end to end here, today.**
* **a placement that would need a band refuses by name**
  (``VisionStageNoRoom``), and the refusal says that eviction needs the
  rank-side path.  It is not silently attempted and it is not silently
  skipped.

That is a real restriction on WHICH REQUESTS SUCCEED, not on whether the
stage works.  It is priced in the refusal, where an operator reading the log
sees the numbers and can decide.

THE TIMEOUT IS A DEADLINE BETWEEN LEGS, NOT AN INTERRUPT
---------------------------------------------------------
A stage that is killed mid-flight is the one outcome worse than a slow one: a
band paused and never resumed is a HOLE in the weights, and a tower left on
the card is memory the prefill will not find.  ``run_vision_stage``'s whole
value is that it unwinds itself; an interrupt takes that away.  So the
deadline is CHECKED AT LEG BOUNDARIES and never interrupts a leg in progress,
and :class:`VisionStageTimeout` names the leg that overran plus the legs that
already completed.  A stage that is slow finishes and is reported slow; only
the NEXT one is refused early.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from sglang.srt.planner.vision_stage import (
    GIB,
    MIB,
    CardAir,
    TowerSpec,
    VisionEncoderConfig,
    VisionStageFlipInFlight,
    VisionStageNoRoom,
    VisionStageRefused,
    VisionStageTowerUnreadable,
)
from sglang.srt.planner.vision_stage_load import (
    VisionEmbeddingRefused,
    VisionStageLoadRefused,
)
from sglang.srt.weg2.vision_stage_runtime import (
    StageHooks,
    VisionStageDisplacementFailed,
    VisionStageEncodeFailed,
    VisionStageResult,
    VisionStageTeardownIncomplete,
    run_vision_stage,
)

logger = logging.getLogger(__name__)

#: W-codes, one per seam.  The front maps a group refusal back onto these, so
#: the code a caller sees and the code the group logged are ONE string.
W_STAGE_OK = "W102 Weg2VisionStage"
W_NO_ROOM = "W105 Weg2VisionNoRoom"
W_LOAD = "W106 Weg2VisionLoadFailed"
W_ENCODE = "W107 Weg2VisionEncodeFailed"
W_FLIP = "W108 Weg2VisionFlipInFlight"
W_TIMEOUT = "W109 Weg2VisionTimeout"
#: NOT a refusal.  The boot is compromised; the front takes the group out.
W_TEARDOWN = "W110 Weg2VisionTeardownIncomplete"
#: BOOT TIME.  ``--weg2-vision transient`` was asked for and the service could
#: not be built -- the named precondition is in the message.  See
#: :mod:`sglang.srt.weg2.vision_stage_boot`.
W_ARM_REFUSED = "W111 Weg2VisionArmRefused"
#: REQUEST TIME.  An image arrived in a group that is running ``transient`` and
#: has NO service armed.  This is the one shape the whole arming path exists to
#: prevent: without it the seam is a silent no-op, the items leave with
#: ``feature`` and no rows, and the failure surfaces three hops later as
#: ``_require_visual`` -- or, on a path that does not check, as wrong text.
W_NOT_ARMED = "W112 Weg2VisionNotArmed"

#: The launcher's arming variable, and the group that runs the stage.  They
#: live in THIS module rather than in ``vision_stage_boot`` because this is the
#: light module every multimodal boot already imports: a caller that has to
#: decide whether a failed import of the heavy one matters can read the key
#: without importing it, and one key cannot be spelled two ways.
VISION_ENV = "SGLANG_WEG2_VISION"
VISION_GROUP_ENV = "SGLANG_WEG2_GROUP"
VISION_GROUP = "P"
VISION_TRANSIENT = "transient"

#: Default deadline for one stage, seconds.  Derived, not chosen: the modelled
#: worst leg on this box is the BUFFERED read of the tower (0.858 GiB at
#: 1.08 GB/s = 0.85 s) plus a 1536x1536 encode at a pessimistic 10 TFLOP/s
#: (1.83 s) plus load and teardown -- call it 3 s of work, doubled for a box
#: that is also prefilling.  A caller with a measured encoder rate should pass
#: its own.
DEFAULT_STAGE_DEADLINE_S = 6.0

#: The legs, in the order ``run_vision_stage`` runs them.  The deadline is
#: checked BEFORE each one.
LEGS = ("plan", "displace", "load", "encode", "attach", "teardown")


#: The request id of the request currently being tokenized, for the log lines.
#: A ``ContextVar`` rather than a parameter because the seam that runs the
#: stage (``base_processor.process_and_combine_mm_data``) is four upstream
#: frames below the only place that knows the rid, and widening four upstream
#: signatures to carry a log field is a worse trade than one task-local.
#: Measured cost of NOT having it: ``W105 Weg2VisionNoRoom rid= --`` on
#: xsn405, a refusal that could not be tied to the request that caused it.
_REQUEST_RID: ContextVar[str] = ContextVar("weg2_vision_rid", default="")


def set_request_rid(rid: str) -> None:
    """Record the rid for this request's task.  Never raises.

    Per-task by construction: each HTTP request is its own asyncio task with
    its own context, so a value set here cannot leak into another request.
    """
    try:
        _REQUEST_RID.set(str(rid or ""))
    except Exception:  # noqa: BLE001 -- a log field never breaks a request
        pass


def current_rid() -> str:
    try:
        return _REQUEST_RID.get()
    except Exception:  # noqa: BLE001
        return ""


class VisionStageRequestRefused(ValueError):
    """A refused stage ENDS the request, here, in the processor process.

    THE DEFECT THIS EXISTS TO END (metal boot xsn405, 20.09. 16:18Z): the seam
    logged the refusal and then *let the request continue*.  The comment that
    justified it said the items "still carry ``feature``, and the normal
    refusal downstream is what the caller sees -- one failure, named once, not
    two."  That premise is FALSE, and the boot proved it: the downstream
    refusal is ``_require_visual`` (``qwen3_vl.py:1421``), which raises inside
    the SCHEDULER THREAD during prefill.  A RuntimeError there is not a request
    refusal, it is a dead rank -- all of PP0/PP1/PP2 went down and the front
    logged ``W17 Weg2GroupDead``.  One refused image killed the group.

    So a refusal is terminal AT THE SEAM.  Raising out of the processor ends
    the request before the tokenizer builds a ``TokenizedGenerateReqInput``,
    which means nothing carrying mm_items is ever sent to a scheduler.

    It derives from :class:`ValueError` deliberately: that is the one exception
    class every entrypoint route already catches
    (``http_server.py:1235/1248/1260``) and turns into a clean error envelope
    instead of an unhandled 500.  ``weg2_http_status`` lifts the status to 501
    -- the same code the front already answers an image with under
    ``--weg2-vision off`` (``W101``), so a client sees ONE status for "this
    server will not encode that image", however it was decided.
    """

    #: Read by ``http_server._create_error_response``; the precedent is
    #: ``ServerShuttingDown`` -> 503 (#840).
    weg2_http_status = 501

    def __init__(self, outcome: "VisionStageOutcome"):
        self.outcome = outcome
        self.code = outcome.code
        self.fatal = bool(outcome.fatal)
        super().__init__(
            f"{outcome.code}: the transient vision stage did not produce "
            f"embeddings for this request (rid={outcome.rid or '<unset>'}), so "
            "it is refused here, in the tokenizer process. Nothing carrying "
            "image items is handed to a scheduler: a rank built without a "
            "vision tower would raise _require_visual inside its scheduler "
            "thread, which kills the group rather than the request. "
            f"{outcome.detail}"
        )


class VisionStageUnstaged(VisionStageRequestRefused):
    """Image items reached the transport with no embeddings and no verdict.

    The second line of defence, and it exists because the first one can only
    catch failures it was TOLD about.  ``maybe_run`` returns ``None`` for
    "nothing to do", and one of the ways to reach that is an installed service
    that this boot expected and does not have.  This check asks the only
    question that actually matters at this seam -- *are any image items about
    to leave for a scheduler without rows?* -- and refuses by name if so.
    """

    def __init__(self, outcome: "VisionStageOutcome"):
        super().__init__(outcome)


class VisionStageNotArmed(RuntimeError):
    """An image reached a ``transient`` group with no service installed.

    NOT a :class:`VisionStageRefused`: a refusal is a verdict the stage
    reached, and no stage ran here at all.  It is raised out of
    :func:`maybe_run` -- the ONE thing in this module that raises -- because
    the alternative is the silent shape: ``maybe_run`` returning ``None``
    looks exactly like "text request, nothing to do", and a boot that asked
    for ``transient`` and armed nothing would then serve image requests by
    handing the prefill items with no rows.

    The message carries the BOOT-TIME reason (W111) when there was one, so
    the operator reading the first failed request sees the precondition that
    was missing at arming and does not have to go back through the log.
    """

    def __init__(self, reason: str, items: int):
        self.reason = str(reason)
        self.items = int(items)
        super().__init__(
            f"{W_NOT_ARMED}: {self.items} image item(s) reached the group's "
            "multimodal processor, this boot runs --weg2-vision transient, and "
            "NO vision stage service is installed in this process, so the tower "
            "would never run. Reason recorded at arming: "
            f"{self.reason or '<none recorded: install() was never called>'}. "
            "Refusing this request by name rather than passing items with no "
            "precomputed embeddings downstream, where a rank without a tower "
            "either raises _require_visual or returns plausible wrong text."
        )


class VisionStageTimeout(VisionStageRefused):
    """The stage ran past its deadline.  Names the leg and what was done.

    A refusal and not an escalation: the stage that overran still unwound
    itself, so the card is intact.  What is refused is this REQUEST.
    """

    def __init__(self, leg: str, elapsed_s: float, deadline_s: float, done: Sequence[str]):
        self.leg = str(leg)
        self.elapsed_s = float(elapsed_s)
        self.deadline_s = float(deadline_s)
        self.done = tuple(str(d) for d in done)
        super().__init__(
            f"the transient vision stage passed its {self.deadline_s:.1f} s "
            f"deadline after {self.elapsed_s:.1f} s, at leg {self.leg!r} "
            f"(completed: {list(self.done)}). The legs that ran were NOT "
            "interrupted -- an interrupted stage can strand a weights band, "
            "which is a hole in the weights and worse than a slow request. "
            "This request is refused; the card is intact."
        )


@dataclass(frozen=True)
class VisionStageOutcome:
    """The wire form: what the group tells the front about one stage."""

    ok: bool
    code: str
    detail: str
    rid: str = ""
    card: int = -1
    rows: int = 0
    seconds: float = 0.0
    #: True only for the teardown class: the GROUP is compromised, not just
    #: this request.  The front takes the group out of rotation on this.
    fatal: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code,
            "detail": self.detail,
            "rid": self.rid,
            "card": self.card,
            "rows": self.rows,
            "seconds": round(self.seconds, 4),
            "fatal": self.fatal,
        }


def code_for(exc: BaseException) -> Tuple[str, bool]:
    """(W-code, fatal) for a stage exception.  PURE, and total over the seams.

    Total on purpose: an exception class with no entry here would otherwise
    reach the caller as a bare 500, which is the silent shape the whole path
    exists to avoid.  The fallback is the LOAD code with ``fatal=False``, and
    the detail carries the class name, so an unmapped class is visible as
    itself rather than as nothing.
    """
    if isinstance(exc, VisionStageTeardownIncomplete):
        return W_TEARDOWN, True
    if isinstance(exc, VisionStageTimeout):
        return W_TIMEOUT, False
    if isinstance(exc, VisionStageFlipInFlight):
        return W_FLIP, False
    if isinstance(exc, (VisionStageNoRoom, VisionStageDisplacementFailed)):
        return W_NO_ROOM, False
    if isinstance(exc, (VisionStageLoadRefused, VisionStageTowerUnreadable)):
        return W_LOAD, False
    if isinstance(exc, (VisionStageEncodeFailed, VisionEmbeddingRefused)):
        return W_ENCODE, False
    return W_LOAD, False


def _device_total_mib(dev: Any, mem: Any) -> float:
    """Total MiB of a card, from the object that actually carries it.

    THE DEFECT THIS FUNCTION EXISTS TO END (metal boot xsn405, 20.09. 16:18Z).
    The predecessor read ``getattr(mem, "total_mib", 0.0)`` -- and
    ``registry.nvml.MemoryInfo`` has ``free_mib``, ``reserved_mib``,
    ``allocatable_mib`` and ``tenant_used_mib``, but NO ``total_mib``: that
    property lives on ``DeviceInfo`` (``registry/nvml.py:88``).  The
    ``getattr`` default turned a wrong-object read into ``0.0``, the
    ``total_mib <= 0`` guard then dropped EVERY card silently, and the
    placement was handed an empty list.  What the operator saw was
    ``W105 Weg2VisionNoRoom ... (evictions FORBIDDEN by the caller): . Best
    card is short by nan GiB`` -- a capacity verdict over zero cards.

    So: the number is read from ``dev`` first (where it is defined), then from
    ``mem`` only as an explicit second source, and a card whose total cannot be
    read at ALL is reported to the caller instead of vanishing.  No default:
    ``getattr`` with a numeric default on a budget path is the #606 class.
    """
    for src in (dev, mem):
        total = getattr(src, "total_mib", None)
        if total is not None:
            return float(total)
        total_bytes = getattr(src, "total_bytes", None)
        if total_bytes is not None:
            return float(total_bytes) / MIB
    return -1.0


def census_from_nvml(
    snapshot: Sequence[Tuple[Any, Any]],
    *,
    h2d_gbps: Dict[int, float],
    ranks_of_card: Optional[Dict[int, Tuple[int, ...]]] = None,
    on_drop: Optional[Callable[[int, str], None]] = None,
) -> Tuple[CardAir, ...]:
    """Turn ``registry.nvml.memory_snapshot()`` into the planner's input.

    ``memory_snapshot`` is THE reader for "how much can still be allocated on
    this card right now" (``registry/nvml.py:533``), and it is usable from any
    process -- including this one, which is not a rank.  Its ``free_mib`` is
    the driver's own allocatable free, NOT ``total - used``: the defect that
    reading cost boot weg2rg6 is recorded in that docstring, 424-518 MiB of
    driver carve-out reported as free.

    This is the IDLE reading by construction -- it is taken before the stage
    starts, while the P layout is quiet.  The ranks' own ``[vram-idle]`` line
    (``vram_family_census.log_vram_idle``) is the CROSS-CHECK on it, sampled
    inside the rank's own process; two instruments on one number, which is why
    both exist.

    A card with no measured H2D rate is DROPPED, not defaulted: placing
    against an assumed link rate is how a modelled time becomes fiction.

    EVERY DROP IS REPORTED.  ``on_drop(card, reason)`` is called for each one
    and the reason is logged at WARNING.  A census that silently returns fewer
    cards than the rig has is indistinguishable from a full rig, and that
    confusion is exactly what cost boot xsn405: see :func:`_device_total_mib`.
    """
    out = []
    for dev, mem in snapshot:
        idx_raw = getattr(dev, "index", None)
        if idx_raw is None:
            idx_raw = getattr(dev, "nvml_index", -1)
        idx = int(idx_raw)

        def _drop(reason: str, _idx: int = idx) -> None:
            logger.warning("vision stage census: card%d DROPPED -- %s", _idx, reason)
            if on_drop is not None:
                on_drop(_idx, reason)

        if idx not in h2d_gbps:
            _drop(
                "no MEASURED h2d rate for this card (have rates for "
                f"{sorted(h2d_gbps)}); placing against an assumed link rate "
                "would make the modelled time fiction"
            )
            continue
        free_mib = getattr(mem, "free_mib", None)
        if free_mib is None:
            _drop(
                f"the memory record {type(mem).__name__} carries no 'free_mib'; "
                "the placement input cannot be read from it"
            )
            continue
        total_mib = _device_total_mib(dev, mem)
        if total_mib <= 0:
            _drop(
                f"total capacity unreadable from {type(dev).__name__}/"
                f"{type(mem).__name__} (got {total_mib}); 'total_mib' lives on "
                "DeviceInfo, not on MemoryInfo"
            )
            continue
        try:
            out.append(
                CardAir(
                    card=idx,
                    ranks=tuple((ranks_of_card or {}).get(idx, ())),
                    total_bytes=int(total_mib * MIB),
                    free_bytes=int(float(free_mib) * MIB),
                    h2d_gbps=float(h2d_gbps[idx]),
                    evictable=(),  # see the module docstring: not from this process
                    provenance="nvml memory_snapshot (idle)",
                )
            )
        except ValueError as exc:  # CardAir's own self-consistency guards
            _drop(f"census is self-inconsistent: {exc}")
    return tuple(out)


@dataclass
class VisionStageService:
    """The group's transient-vision entry point.

    Everything it touches is injected, for the same reason
    :class:`StageHooks` is: the whole path -- front verdict, group service,
    placement, stage, teardown -- has to be drivable at a desk, and a service
    that reaches for NVML and CUDA in its constructor is not.

    ``pause_tag``/``resume_tag`` are ``None`` by default and that is the
    honest default for this process: see the module docstring.  With both
    ``None``, eviction is FORBIDDEN and a placement that would need a band
    refuses with numbers instead of being attempted.
    """

    nvml_snapshot: Callable[[], Sequence[Tuple[Any, Any]]]
    h2d_gbps: Dict[int, float]
    tower: TowerSpec
    load_tower: Callable[[int], Any]
    encode: Callable[[Any, Sequence[Any]], Sequence[Any]]
    release_tower: Callable[[Any], None]
    read_gbps: float
    flip_armed: Callable[[], bool] = lambda: False
    pause_tag: Optional[Callable[[str], None]] = None
    resume_tag: Optional[Callable[[str], None]] = None
    encoder_config: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    prefer_cards: Tuple[int, ...] = ()
    ranks_of_card: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    achieved_tflops: Optional[float] = None
    deadline_s: float = DEFAULT_STAGE_DEADLINE_S
    clock: Callable[[], float] = time.perf_counter

    @property
    def eviction_available(self) -> bool:
        return self.pause_tag is not None and self.resume_tag is not None

    def _census(self) -> Tuple[CardAir, ...]:
        """The placement input, and a RECORD of what it contained.

        ``_last_census`` is kept so a refusal can print the candidates it was
        actually offered instead of leaving the reader to guess.  A capacity
        refusal over an empty candidate list is not a capacity finding, and
        the only way to tell the two apart afterwards is for the census to say
        what it saw.
        """
        drops: list = []
        cards = census_from_nvml(
            self.nvml_snapshot(),
            h2d_gbps=self.h2d_gbps,
            ranks_of_card=self.ranks_of_card,
            on_drop=lambda card, why: drops.append((card, why)),
        )
        self._last_census = cards
        self._last_census_drops = tuple(drops)
        return cards

    def census_note(self, need_bytes: int) -> str:
        """One sentence naming every candidate with numbers, and every drop.

        This is what makes the ``W105`` line actionable: card, free_idle,
        need and slack for each candidate the placement was offered -- or, if
        it was offered none, why.
        """
        cards = getattr(self, "_last_census", ())
        drops = getattr(self, "_last_census_drops", ())
        if cards:
            candidates = "; ".join(
                f"card{c.card}: free_idle {c.free_bytes / GIB:.3f} GiB, need "
                f"{need_bytes / GIB:.3f} GiB, slack "
                f"{(c.free_bytes - need_bytes) / GIB:.3f} GiB"
                for c in cards
            )
            note = f"CENSUS offered {len(cards)} card(s): {candidates}."
        else:
            note = (
                "CENSUS offered ZERO cards, so no card was measured against the "
                "need -- this refusal is an INPUT failure, not a capacity "
                "finding."
            )
        if drops:
            note += " DROPPED: " + "; ".join(
                f"card{c} ({why})" for c, why in drops
            )
        elif not cards:
            note += (
                " No card was dropped either, so nvml_snapshot() itself "
                "returned an empty sequence in this process."
            )
        return note

    def _deadline(self, started: float):
        """A ``check_deadline`` for :func:`run_vision_stage`.

        Called by the runtime BEFORE each leg and never before the teardown,
        so refusing here aborts at a boundary and the teardown still unwinds
        whatever is standing.  Records the legs that completed, so the refusal
        says what was done and not only what was not.
        """
        done: list = []

        def _check(leg: str) -> None:
            elapsed = self.clock() - started
            if elapsed > self.deadline_s:
                raise VisionStageTimeout(leg, elapsed, self.deadline_s, done)
            done.append(leg)

        return _check

    def encode_items(
        self, items: Sequence[Any], *, rid: str = "", expected_width: Optional[int] = None
    ) -> VisionStageOutcome:
        """Run the stage for one request's items.  Never raises.

        Every seam becomes a :class:`VisionStageOutcome` with a W-code, so the
        caller has ONE shape to handle and no exception can escape as a bare
        500.  ``fatal=True`` is the teardown class and only that: it means the
        GROUP is compromised, not this request.
        """
        started = self.clock()
        rid = rid or current_rid()
        self._last_census = ()
        self._last_census_drops = ()
        hooks = StageHooks(
            census=self._census,
            load_tower=self.load_tower,
            encode=self.encode,
            release_tower=self.release_tower,
            pause_tag=self.pause_tag or _no_eviction_pause,
            resume_tag=self.resume_tag or _no_eviction_resume,
            flip_armed=self.flip_armed,
        )
        rows_hint = 0
        try:
            rows_hint = int(
                getattr(items[0], "_vision_patch_rows", 0)
            ) if items else 0
        except Exception:  # noqa: BLE001
            rows_hint = 0
        flops = (
            self.encoder_config.encoder_flops(rows_hint) if rows_hint > 0 else 0
        )
        try:
            result: VisionStageResult = run_vision_stage(
                items,
                tower=self.tower,
                hooks=hooks,
                read_gbps=self.read_gbps,
                encoder_config=self.encoder_config,
                expected_width=expected_width,
                prefer_cards=self.prefer_cards,
                allow_eviction=self.eviction_available,
                encoder_flops=flops,
                achieved_tflops=self.achieved_tflops,
                clock=self.clock,
                check_deadline=self._deadline(started),
            )
        except BaseException as exc:  # noqa: BLE001 -- every seam becomes an outcome
            code, fatal = code_for(exc)
            detail = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, VisionStageNoRoom):
                # The candidates WITH NUMBERS, always -- a placement refusal
                # that does not say what it was offered cannot be acted on,
                # and on xsn405 it was the missing half of the line.
                detail += " " + self.census_note(int(exc.need_bytes))
                if not self.eviction_available:
                    detail += (
                        " NOTE: band displacement was not attempted -- "
                        "pause_tag/resume_tag live in the RANK processes and this "
                        "stage runs in the group's processor process, which cannot "
                        "reach them. A placement needing a band requires the "
                        "rank-side path."
                    )
            (logger.error if fatal else logger.warning)(
                "%s rid=%s -- %s", code, rid or "<unset>", detail
            )
            return VisionStageOutcome(
                ok=False,
                code=code,
                detail=detail,
                rid=rid,
                seconds=self.clock() - started,
                fatal=fatal,
            )
        # THE ACCEPTANCE LINE (design §6 row e).  It was BUILT and never
        # emitted: `log_line()` was only stuffed into the outcome's `detail`,
        # and `base_processor` discarded the outcome -- so the one line the
        # metal test greps for could not appear in any log, success or not.
        # The stage's first measured encoder time lives here.
        logger.info("%s rid=%s", result.log_line(), rid or "<unset>")
        return VisionStageOutcome(
            ok=True,
            code=W_STAGE_OK,
            detail=result.log_line(),
            rid=rid,
            card=result.plan.card,
            rows=result.rows,
            seconds=self.clock() - started,
        )


def _no_eviction_pause(name: str) -> None:
    raise RuntimeError(
        f"band {name!r} cannot be displaced from this process: pause_tag lives "
        "in the rank processes. This should be unreachable -- the service "
        "forbids eviction when it has no hooks."
    )


def _no_eviction_resume(name: str) -> None:  # pragma: no cover - same reason
    raise RuntimeError(f"band {name!r}: resume_tag not available in this process")


# ---------------------------------------------------------------------------
# The installed-service seam.  The processor calls `maybe_run` unconditionally;
# with no service installed it is a no-op and the boot is byte-for-byte what it
# was.  One global because there is one processor process per group.
# ---------------------------------------------------------------------------

_SERVICE: Optional[VisionStageService] = None
#: Set INSTEAD of a service when this boot asked for ``transient`` and the
#: arming refused.  Two states are NOT one: ``None`` here means "no transient
#: boot, nothing expected" (``off``/``resident``, or a plain upstream engine),
#: and a non-empty string means "transient was asked for and is NOT armed".
#: Collapsing them is exactly how a broken arming becomes a silent no-op.
_ARM_REFUSAL: str = ""


def install(service: Optional[VisionStageService]) -> None:
    """Arm (or disarm, with ``None``) the transient stage for this process.

    Installing a service CLEARS any recorded arming refusal: the two are
    mutually exclusive states of one process, and a stale reason beside a
    live service would print a refusal for a stage that ran.
    """
    global _SERVICE, _ARM_REFUSAL
    _SERVICE = service
    if service is not None:
        _ARM_REFUSAL = ""
    logger.info(
        "vision stage service %s", "ARMED" if service is not None else "disarmed"
    )


def install_refusal(reason: str) -> None:
    """Record that ``transient`` was asked for and could NOT be armed.

    The counterpart to :func:`install`, and the reason :func:`maybe_run` can
    tell "nothing to do" from "this boot is broken for images".  Logged at
    ERROR with the W-code at the moment it happens, so the boot log says it
    once at arming -- and said again, with this reason quoted, on the first
    image request.
    """
    global _SERVICE, _ARM_REFUSAL
    _SERVICE = None
    _ARM_REFUSAL = str(reason)
    logger.error(
        "%s -- the transient vision stage is NOT armed in this process: %s. "
        "Image requests to this group will be refused by name (%s); text "
        "requests are unaffected.",
        W_ARM_REFUSED,
        _ARM_REFUSAL,
        W_NOT_ARMED,
    )


def arm_refusal() -> str:
    """The recorded arming refusal, or ``""`` when none was recorded."""
    return _ARM_REFUSAL


def reset_for_test() -> None:
    """Drop both states.  Only tests call this; a boot arms once."""
    global _SERVICE, _ARM_REFUSAL
    _SERVICE = None
    _ARM_REFUSAL = ""


def installed() -> Optional[VisionStageService]:
    return _SERVICE


def maybe_run(items: Sequence[Any], *, rid: str = "") -> Optional[VisionStageOutcome]:
    """Run the stage for these items if a service is armed and they need it.

    Returns ``None`` when there was nothing to do -- no service and no
    transient boot, no items, or items that already carry embeddings (a
    re-entry, or an encoder-disagg boot that filled them upstream).

    Raises exactly one thing, and only one: :class:`VisionStageNotArmed`, when
    there ARE items to stage and this boot asked for ``transient`` without
    arming.  Every other failure is a :class:`VisionStageOutcome` with a
    W-code, because a stage that reached a verdict has a verdict to report.

    THE ORDER HERE IS THE TEXT-PATH GUARANTEE: the "nothing to stage" exits
    come FIRST, so a text-only request takes exactly the same two ``getattr``
    scans it took before this path existed and can never reach the refusal.
    """
    if not items:
        return None
    pending = _unstaged(items)
    if not pending:
        return None
    service = _SERVICE
    if service is None:
        if _ARM_REFUSAL:
            raise VisionStageNotArmed(_ARM_REFUSAL, len(pending))
        return None
    return service.encode_items(pending, rid=rid or current_rid())


def _unstaged(items: Sequence[Any]) -> list:
    """Items that carry pixels and no rows -- the ones a tower has to run for."""
    return [
        it
        for it in items
        if getattr(it, "precomputed_embeddings", None) is None
        and getattr(it, "feature", None) is not None
    ]


def this_boot_is_transient() -> bool:
    """Does the launcher say this process serves ``--weg2-vision transient``?

    The launcher publishes the key and pops it otherwise (design R19), so the
    variable is an OUTPUT of the boot, never an operator's input.
    """
    import os

    return os.environ.get(VISION_ENV, "").strip() == VISION_TRANSIENT


def assert_nothing_unstaged(items: Sequence[Any], *, rid: str = "") -> None:
    """Refuse if any image item would reach a scheduler with no embeddings.

    Called by the processor AFTER :func:`maybe_run`, and only in a transient
    boot.  It is the check that does not depend on a verdict having been
    reached: ``maybe_run`` returns ``None`` both for "text request, nothing to
    do" and for "no service installed and no refusal recorded", and the second
    of those, in a boot whose ranks have no tower, is the shape that killed
    group P on xsn405.

    The text path cannot reach it: with no image item, :func:`_unstaged` is
    empty and this returns before looking at anything else.
    """
    if not this_boot_is_transient():
        return
    pending = _unstaged(items)
    if not pending:
        return
    rid = rid or current_rid()
    outcome = VisionStageOutcome(
        ok=False,
        code=W_NOT_ARMED,
        detail=(
            f"{len(pending)} image item(s) still carry pixels and NO "
            "precomputed embeddings after the transient vision stage seam. "
            "This boot runs --weg2-vision transient, so every rank was built "
            "with language_model_only and has no vision tower: handing these "
            "items on would reach _require_visual inside a scheduler thread "
            "and kill the group. Recorded arming refusal: "
            f"{arm_refusal() or '<none>'}."
        ),
        rid=rid,
    )
    logger.error("%s rid=%s -- %s", outcome.code, rid or "<unset>", outcome.detail)
    raise VisionStageUnstaged(outcome)
