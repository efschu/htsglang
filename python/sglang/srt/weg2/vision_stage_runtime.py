# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 6 -- the transient vision stage, end to end.

Slices 1-3 built the pieces: ``planner/vision_stage.py`` picks the card,
``planner/vision_stage_load.py`` reads the tower and hands the rows on.  This
module is the ORDER those pieces run in, and -- the part that actually earns a
module -- the guarantee that the card is left exactly as it was found, on every
path out, including the ones that raise.

    flip gate -> plan -> displace band(s) -> load tower -> encode
              -> attach rows -> release tower -> restore band(s)

WHY THE TEARDOWN IS THE HARD PART
---------------------------------
Every step in the middle can fail, and each failure leaves a different amount
of the stage standing.  A tower left on the card is 0.858 GiB the prefill will
not find; a band left in host RAM is a hole in the weights that the next
forward reads as garbage -- and unlike the tower, THAT one does not OOM, it
returns plausible wrong text.  So the invariant is not "clean up on success",
it is:

    whatever was displaced is restored, and whatever was loaded is released,
    before this function returns OR raises -- and if a restore itself fails,
    that is escalated by name and never swallowed.

:class:`VisionStageTeardownIncomplete` is that escalation.  It is deliberately
NOT a subclass of :class:`~sglang.srt.planner.vision_stage.VisionStageRefused`:
a refusal means "this request cannot be served and the rig is fine"; a failed
teardown means the RIG is not fine and the next request must not be served at
all.

EVERY SIDE EFFECT IS AN INJECTED CALLABLE
-----------------------------------------
:class:`StageHooks` carries the six things this module cannot do itself --
read the census, pause and resume a weights band, load and release the tower,
run the encoder -- plus the flip predicate.  That is not test decoration: it is
what lets the WHOLE stage, including every failure path and the teardown
invariant, run at a desk with no GPU, no NVML and no checkpoint.  The metal
wiring passes the real six; the probe passes fakes that count their calls.

THE SEAMS, and the named refusal at each
----------------------------------------
============================ ==========================================
seam                         refusal
============================ ==========================================
a flip is running            ``VisionStageFlipInFlight``   (a WAIT)
no card, no displaceable band ``VisionStageNoRoom``
tower bytes unknown/scattered ``VisionStageTowerUnreadable``
a band will not pause         ``VisionStageDisplacementFailed``
the tower will not load       ``VisionStageLoadRefused``
the encoder raises            ``VisionStageEncodeFailed``
rows do not fit the request   ``VisionEmbeddingRefused``
a band will not come back     ``VisionStageTeardownIncomplete``  (NOT a refusal)
============================ ==========================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from sglang.srt.planner.vision_stage import (
    CardAir,
    EvictableBlock,
    TowerSpec,
    VisionEncoderConfig,
    VisionStageFlipInFlight,
    VisionStageNoRoom,
    VisionStagePlan,
    VisionStageRefused,
    VisionStageTowerUnreadable,
    plan_vision_stage,
)
from sglang.srt.planner.vision_stage_load import (
    VisionEmbeddingRefused,
    VisionStageLoadRefused,
    attach_precomputed_embeddings,
)

logger = logging.getLogger(__name__)


class VisionStageDisplacementFailed(VisionStageRefused):
    """A ``weights_<k>`` band would not move out of the way.

    A refusal and not an escalation: nothing has been displaced that is not
    put back, so the rig is intact and only this request is refused.
    """


class VisionStageEncodeFailed(VisionStageRefused):
    """The encoder forward raised.  The tower and the bands are torn down
    before this leaves; the rig is intact."""


class VisionStageTeardownIncomplete(RuntimeError):
    """The card was NOT left as it was found.

    NOT a :class:`VisionStageRefused`, and the distinction is the whole
    point: a refusal is "this request cannot be served"; this is "a weights
    band is still in host RAM and the next forward will read a hole".  A
    caller that catches ``VisionStageRefused`` to answer 501 must NOT catch
    this -- it has to reach the scheduler.
    """

    def __init__(self, card: int, stranded: Sequence[str], cause: str):
        self.card = int(card)
        self.stranded = tuple(str(s) for s in stranded)
        self.cause = str(cause)
        super().__init__(
            f"vision stage on card{self.card} did not restore "
            f"{list(self.stranded)}: {self.cause}. These weights are in host "
            "RAM and the card has a HOLE where they belong -- the next forward "
            "reads garbage and returns plausible wrong text rather than "
            "failing. This is not a refusal; the boot is compromised until "
            "the band is back."
        )


@dataclass(frozen=True)
class StageHooks:
    """Every side effect the stage has, injectable one by one.

    ``census()`` must return the cards' **IDLE** free air (see
    :class:`~sglang.srt.planner.vision_stage.CardAir`) -- the stage runs
    before the prefill and is gone before it, so the load reading is the wrong
    instrument here.

    ``flip_armed()`` is read FAIL-CLOSED: an exception out of it counts as
    "armed", the same way ``kv_backing_relief.py:4559-4567`` reads it, because
    an unreadable flip is not an idle one.
    """

    census: Callable[[], Sequence[CardAir]]
    load_tower: Callable[[int], Any]
    encode: Callable[[Any, Sequence[Any]], Sequence[Any]]
    release_tower: Callable[[Any], None]
    pause_tag: Callable[[str], None]
    resume_tag: Callable[[str], None]
    flip_armed: Callable[[], bool] = lambda: False


@dataclass
class VisionStageResult:
    """What the stage did, in numbers a log line can carry."""

    plan: VisionStagePlan
    rows: int
    displaced: Tuple[str, ...]
    #: measured wall time of each leg, seconds.  These are MEASURED, unlike
    #: the plan's modelled ones -- both are kept so the first metal boot can
    #: compare the model against the clock.
    measured: dict = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return float(sum(self.measured.values()))

    def log_line(self) -> str:
        legs = ", ".join(
            f"{k} {v * 1e3:.0f}" for k, v in sorted(self.measured.items())
        )
        return (
            f"W102 Weg2VisionStage card={self.plan.card} rows={self.rows} "
            f"displaced={list(self.displaced)} "
            f"free_idle_before={self.plan.free_before_bytes / (1 << 30):.3f}GiB "
            f"need={self.plan.need_bytes / (1 << 30):.3f}GiB "
            f"slack={self.plan.slack_bytes / (1 << 30):.3f}GiB "
            f"legs_ms=({legs}) total_ms={self.total_seconds * 1e3:.0f}"
        )


def _flip_is_armed(hook: Callable[[], bool]) -> bool:
    """Fail-closed, and the precedent is named in the module docstring."""
    try:
        return bool(hook())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "vision stage: the flip predicate raised (%s); reading it as ARMED "
            "-- an unreadable flip is not an idle one",
            exc,
        )
        return True


def run_vision_stage(
    items: Sequence[Any],
    *,
    tower: TowerSpec,
    hooks: StageHooks,
    read_gbps: float,
    encoder_config: Optional[VisionEncoderConfig] = None,
    expected_width: Optional[int] = None,
    prefer_cards: Sequence[int] = (),
    allow_eviction: bool = True,
    encoder_flops: int = 0,
    achieved_tflops: Optional[float] = None,
    clock: Callable[[], float] = time.perf_counter,
    check_deadline: Optional[Callable[[str], None]] = None,
) -> VisionStageResult:
    """Run the tower once, on one card, and leave the card as it was found.

    Returns a :class:`VisionStageResult` on success.  Raises exactly one of
    the seam refusals in the module docstring, or
    :class:`VisionStageTeardownIncomplete` when the card could NOT be restored.

    The ordering is not incidental.  The flip gate is first because a census
    taken mid-flip is fiction.  The plan is second because nothing should move
    before it is known that everything will fit.  The displacement is third
    and the load fourth, because a load that fails with nothing displaced is a
    cheaper failure than one with a band already in host RAM.

    ``check_deadline(leg)`` is called BEFORE each leg and NEVER before the
    teardown.  Raising from it aborts the stage at a leg BOUNDARY, which is
    the only place an abort is safe: the teardown still runs and unwinds
    whatever is standing.  There is deliberately no way to interrupt a leg in
    progress -- a band paused and never resumed is a hole in the weights, and
    that is worse than any slow request.
    """
    def _gate(leg: str) -> None:
        if check_deadline is not None:
            check_deadline(leg)
    cfg = encoder_config or VisionEncoderConfig()
    width = int(expected_width if expected_width is not None else cfg.embed_width)

    if _flip_is_armed(hooks.flip_armed):
        raise VisionStageFlipInFlight("armed")

    # --- plan -------------------------------------------------------------
    _gate("plan")
    t0 = clock()
    cards = list(hooks.census())
    plan = plan_vision_stage(
        cards,
        tower,
        read_gbps=read_gbps,
        prefer_cards=prefer_cards,
        allow_eviction=allow_eviction,
        encoder_flops=encoder_flops,
        achieved_tflops=achieved_tflops,
    )
    measured = {"plan": clock() - t0}

    displaced: List[EvictableBlock] = []
    handle = None
    try:
        # --- displace ------------------------------------------------------
        _gate("displace")
        t0 = clock()
        for blk in plan.evicted:
            try:
                hooks.pause_tag(blk.name)
            except Exception as exc:  # noqa: BLE001
                # Nothing is left half-moved: the finally below puts back
                # whatever DID move, then this refusal leaves.
                raise VisionStageDisplacementFailed(
                    f"card{plan.card}: weights band {blk.name!r} "
                    f"({blk.bytes / (1 << 20):.0f} MiB) would not pause: {exc}. "
                    "The bands that did move are restored; this request is "
                    "refused and the rig is intact."
                ) from exc
            displaced.append(blk)
        measured["displace"] = clock() - t0

        # --- load ----------------------------------------------------------
        _gate("load")
        t0 = clock()
        try:
            handle = hooks.load_tower(plan.card)
        except VisionStageLoadRefused:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VisionStageLoadRefused(
                f"card{plan.card}: the tower ({tower.pieces} pieces, "
                f"{tower.weight_bytes / (1 << 30):.3f} GiB) would not load: {exc}"
            ) from exc
        if handle is None:
            raise VisionStageLoadRefused(
                f"card{plan.card}: the loader returned None. A stage that "
                "proceeds on a null tower encodes nothing and hands the "
                "prefill zeros, which is wrong text and not an error."
            )
        measured["load"] = clock() - t0

        # --- encode --------------------------------------------------------
        _gate("encode")
        t0 = clock()
        try:
            embeddings = hooks.encode(handle, items)
        except Exception as exc:  # noqa: BLE001
            raise VisionStageEncodeFailed(
                f"card{plan.card}: the encoder forward over {len(items)} item(s) "
                f"raised: {exc}. The tower is released and every displaced band "
                "restored before this leaves."
            ) from exc
        measured["encode"] = clock() - t0

        # --- hand on -------------------------------------------------------
        _gate("attach")
        t0 = clock()
        rows = attach_precomputed_embeddings(
            items, embeddings, expected_width=width
        )
        measured["attach"] = clock() - t0
    finally:
        # --- teardown, on EVERY path -----------------------------------------
        t0 = clock()
        release_error = None
        if handle is not None:
            try:
                hooks.release_tower(handle)
            except Exception as exc:  # noqa: BLE001
                # Recorded, not raised here: a stranded band is worse than a
                # stranded tower (wrong text against an OOM), so the restores
                # below run first and this is escalated after them.
                release_error = exc
                logger.error(
                    "vision stage: the tower would not release on card%d: %s",
                    plan.card,
                    exc,
                )
        stranded: List[str] = []
        causes: List[str] = []
        for blk in displaced:
            try:
                hooks.resume_tag(blk.name)
            except Exception as exc:  # noqa: BLE001
                stranded.append(blk.name)
                causes.append(f"{blk.name}: {exc}")
        measured["teardown"] = clock() - t0
        if stranded:
            raise VisionStageTeardownIncomplete(
                plan.card, stranded, "; ".join(causes)
            )
        if release_error is not None:
            raise VisionStageTeardownIncomplete(
                plan.card,
                [f"tower ({tower.weight_bytes / (1 << 30):.3f} GiB)"],
                str(release_error),
            )

    result = VisionStageResult(
        plan=plan,
        rows=rows,
        displaced=tuple(b.name for b in displaced),
        measured=measured,
    )
    logger.info("%s", result.log_line())
    return result
