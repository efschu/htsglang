# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The dashboard's self-benchmark, as idle work (DESIGN #347 W7).

ANALYSE #347 item 1, last bullet: "the dashboard's 'absent' factor tiles
re-measure themselves when the rig is idle". The planner's factor board
(``POST /api/bench_factors``) reports each limiting factor as *measured*,
*estimated* or *absent*, and an absent tile deliberately carries no value at
all -- a placeholder is indistinguishable from a measurement once it has been
rendered. Absent tiles therefore stay absent until somebody runs the study.

This tenant runs one of those studies when nobody is using the rig. It wraps
the cheapest factor that already has a job endpoint: the short card probe
behind ``POST /api/card_probe``, which fills the ``card_rates`` and
``pair_link`` tiles in one run and costs about 30 s of GPU time. It owns no
measurement code -- it launches ``python -m sglang.srt.rigmon.card_probe
--run``, the same subprocess entry point ``ProbeJobStore`` uses, for the same
reason ``ProbeJobStore`` uses it: a probe allocates a CUDA context on every
card and the serving process must not acquire one.

Work exists when the cached profile is absent or older than the configured
maximum age. That is the whole queue: this tenant's work is derived from the
state of the rig, so it has no enqueue surface.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Callable, Optional

from sglang.srt.workbench.tenant import (
    DEFAULT_CUDA_CONTEXT_BYTES,
    MIB,
    EventSink,
    IdleWorkTenant,
    SubprocessSegment,
    WorkEstimate,
    WorkEvent,
    WorkGrant,
    WorkSegment,
)

logger = logging.getLogger(__name__)

#: Seven days. A card's rates change with a driver, a power-limit change or a
#: reseated riser, none of which happens often; refreshing more eagerly would
#: spend GPU time re-measuring a number that did not move.
DEFAULT_MAX_AGE_S = 7 * 24 * 3600.0

#: What the probe allocates per card, named. From
#: ``sglang/srt/rigmon/card_probe.py``: an fp8 GEMM at 2048x5120x17408 with a
#: bf16 draft of each operand, plus two 64 MiB transfer buffers per direction.
_FP8_M, _FP8_K, _FP8_N = 2048, 5120, 17408
_XFER_BYTES = 64 * MIB


def probe_posts() -> dict[str, int]:
    """The named posts of one probe run on one card. No safety factor."""
    return {
        "cuda_context": DEFAULT_CUDA_CONTEXT_BYTES,
        "gemm_operand_drafts": (_FP8_M * _FP8_K + _FP8_N * _FP8_K) * 2,
        "gemm_operands": (_FP8_M * _FP8_K + _FP8_N * _FP8_K) * 1,
        "gemm_output": _FP8_M * _FP8_N * 2,
        "transfer_buffers": 2 * _XFER_BYTES,
    }


class CardProbeTenant(IdleWorkTenant):
    """Refreshes the card-rate and pair-link factors when they go stale."""

    name = "card_probe"
    priority = 70

    def __init__(
        self,
        *,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        node_id: str = "local",
        python_executable: str = "",
        profile_loader: Optional[Callable[[], Any]] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self.max_age_s = float(max_age_s)
        self.node_id = node_id
        self.python_executable = python_executable or sys.executable
        self._profile_loader = profile_loader
        self._clock = clock
        self._last_error = ""

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        try:
            self._load()
        except ImportError as exc:
            return False, f"the card probe module cannot be imported: {exc}"
        return True, ""

    def describe(self) -> str:
        return (
            "re-measures the planner's card-rate and pair-link factors when "
            f"the cached probe is missing or older than {self.max_age_s / 3600:.0f}h"
        )

    # -- the queue ----------------------------------------------------------

    def _load(self) -> Any:
        if self._profile_loader is not None:
            return self._profile_loader()
        from sglang.srt.rigmon.card_probe import load_card_probe

        return load_card_probe()

    def age_s(self) -> Optional[float]:
        """Age of the cached profile, or ``None`` when there is none."""
        try:
            profile = self._load()
        except Exception as exc:  # noqa: BLE001 - an unreadable cache is an absent one
            logger.debug("workbench card_probe: cache unreadable: %s", exc)
            return None
        if profile is None:
            return None
        age = profile.age_s(self._clock())
        return None if age is None else float(age)

    def pending(self) -> int:
        age = self.age_s()
        if age is None:
            return 1
        return 1 if age > self.max_age_s else 0

    # -- pricing ------------------------------------------------------------

    def estimate(self) -> WorkEstimate:
        posts = probe_posts()
        return WorkEstimate(
            per_card_bytes=sum(posts.values()),
            posts=posts,
            # Every visible card: the probe measures each card and the pair
            # matrix between them, and a partial run would produce a profile
            # whose missing pairs look like unmeasurable links.
            cards_wanted=0,
            expected_seconds=30.0,
        )

    # -- running ------------------------------------------------------------

    async def start_segment(self, grant: WorkGrant, sink: EventSink) -> WorkSegment:
        argv = [
            self.python_executable,
            "-m",
            "sglang.srt.rigmon.card_probe",
            "--run",
            "--node-id",
            self.node_id,
        ]
        sink(
            WorkEvent(
                "info",
                "refreshing the card-rate and pair-link factors over "
                f"{len(grant.card_indices)} card(s)",
                data={"age_s": self.age_s(), "max_age_s": self.max_age_s},
            )
        )
        segment = SubprocessSegment(
            argv=argv,
            cwd=None,
            env={"CUDA_VISIBLE_DEVICES": grant.visible_devices},
            sink=sink,
            label="card_probe",
            artifact_path=self._cache_path(),
            line_filter=_probe_line,
        )
        return await segment.start()

    def _cache_path(self) -> Optional[str]:
        try:
            from sglang.srt.rigmon.card_probe import default_cache_path

            return default_cache_path()
        except Exception:  # noqa: BLE001 - a path we cannot name is not a failure
            return None

    def snapshot(self) -> dict[str, Any]:
        body = super().snapshot()
        age = self.age_s()
        body.update(
            {
                "profile_age_s": None if age is None else round(age, 1),
                "max_age_s": self.max_age_s,
                "cache_path": self._cache_path(),
                "factors": ["card_rates", "pair_link"],
            }
        )
        return body


def _probe_line(line: str) -> Optional[WorkEvent]:
    lowered = line.lower()
    if "traceback" in lowered or "error" in lowered:
        return WorkEvent("error", line)
    if "wrote" in lowered or "probe" in lowered:
        return WorkEvent("info", line, type="metrics")
    return None


__all__ = ["CardProbeTenant", "DEFAULT_MAX_AGE_S", "probe_posts"]
