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


def census_from_nvml(
    snapshot: Sequence[Tuple[Any, Any]],
    *,
    h2d_gbps: Dict[int, float],
    ranks_of_card: Optional[Dict[int, Tuple[int, ...]]] = None,
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
    """
    out = []
    for dev, mem in snapshot:
        idx = int(getattr(dev, "index", getattr(dev, "nvml_index", -1)))
        if idx not in h2d_gbps:
            logger.warning(
                "vision stage census: card%d has no MEASURED h2d rate; dropping "
                "it from the placement rather than assuming one",
                idx,
            )
            continue
        free_mib = float(getattr(mem, "free_mib", 0.0))
        total_mib = float(getattr(mem, "total_mib", 0.0))
        if total_mib <= 0:
            continue
        out.append(
            CardAir(
                card=idx,
                ranks=tuple((ranks_of_card or {}).get(idx, ())),
                total_bytes=int(total_mib * MIB),
                free_bytes=int(free_mib * MIB),
                h2d_gbps=float(h2d_gbps[idx]),
                evictable=(),  # see the module docstring: not from this process
                provenance="nvml memory_snapshot (idle)",
            )
        )
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
        return census_from_nvml(
            self.nvml_snapshot(),
            h2d_gbps=self.h2d_gbps,
            ranks_of_card=self.ranks_of_card,
        )

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
            if isinstance(exc, VisionStageNoRoom) and not self.eviction_available:
                detail += (
                    " NOTE: band displacement was not attempted -- "
                    "pause_tag/resume_tag live in the RANK processes and this "
                    "stage runs in the group's processor process, which cannot "
                    "reach them. A placement needing a band requires the "
                    "rank-side path."
                )
            (logger.error if fatal else logger.warning)(
                "%s rid=%s -- %s", code, rid, detail
            )
            return VisionStageOutcome(
                ok=False,
                code=code,
                detail=detail,
                rid=rid,
                seconds=self.clock() - started,
                fatal=fatal,
            )
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


def install(service: Optional[VisionStageService]) -> None:
    """Arm (or disarm, with ``None``) the transient stage for this process."""
    global _SERVICE
    _SERVICE = service
    logger.info(
        "vision stage service %s", "ARMED" if service is not None else "disarmed"
    )


def installed() -> Optional[VisionStageService]:
    return _SERVICE


def maybe_run(items: Sequence[Any], *, rid: str = "") -> Optional[VisionStageOutcome]:
    """Run the stage for these items if a service is armed and they need it.

    Returns ``None`` when there was nothing to do -- no service, no items, or
    items that already carry embeddings (a re-entry, or an encoder-disagg
    boot that filled them upstream).  Never raises: a stage failure is an
    outcome, and the caller decides.
    """
    service = _SERVICE
    if service is None or not items:
        return None
    pending = [
        it
        for it in items
        if getattr(it, "precomputed_embeddings", None) is None
        and getattr(it, "feature", None) is not None
    ]
    if not pending:
        return None
    return service.encode_items(pending, rid=rid)
