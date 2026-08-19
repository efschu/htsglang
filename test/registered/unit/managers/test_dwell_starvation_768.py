# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#768: the min-dwell bound must not hold a STARVED scheduler.

The dwell is a thrash bound. Thrashing needs work in flight, so with nothing
running and prefill pending the hold protects no throughput -- and nothing can
end it either, because the request that would restart the clock is the one the
hold refuses to admit.

Observed on the full-feature boot: a tp_to_pp cutover committed, the very next
decision logged

    PHASE-POLICY holding in pp: min dwell: 0.0s since last flip < 3s
    (pending prefill 5813 tok, running bs 0)

and serving then sat idle for 464s across 41 ADMISSION-WEDGE reports -- with
health answering 200 the whole time -- until a watchdog killed the tree.
"""

from __future__ import annotations

import unittest

from sglang.srt.managers.phase_policy import (
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    decide,
)

PHASE_PP = "pp"


def _cfg(**kw):
    base = dict(enabled=True, flip_tokens=1, min_dwell_s=3.0)
    base.update(kw)
    return PhasePolicyConfig(**base)


def _inp(pending: int, running: int, now: float = 100.0, phase: str = PHASE_PP):
    return PhasePolicyInputs(
        phase=phase, pending_prefill_tokens=pending, running_bs=running, now=now
    )


class TestTheDwellStillBoundsThrash(unittest.TestCase):
    """The bound must survive: work in flight means the hold is legitimate."""

    def test_it_still_holds_while_requests_are_running(self):
        # running_bs > 0 -> there IS throughput to protect, dwell applies.
        state = PhasePolicyState()
        state.last_flip_at = 100.0
        d = decide(_cfg(), state, _inp(pending=5813, running=2))
        self.assertIn("min dwell", d.reason)
        self.assertFalse(d.wants_flip)


class TestAStarvedSchedulerIsNotHeld(unittest.TestCase):
    """The #768 specimen, pinned: 5813 pending, 0 running, 0.0s since flip."""

    def test_the_specimen_no_longer_reports_a_dwell_hold(self):
        state = PhasePolicyState()
        state.last_flip_at = 100.0  # 0.0s since the last flip
        d = decide(_cfg(), state, _inp(pending=5813, running=0))
        self.assertNotIn(
            "min dwell",
            d.reason,
            "the dwell must not be what blocks a scheduler with nothing "
            f"running and work pending; got: {d.reason}",
        )

    def test_no_pending_work_means_the_dwell_may_still_hold(self):
        # Nothing queued: this is not starvation, so the bound is free to bind.
        state = PhasePolicyState()
        state.last_flip_at = 100.0
        d = decide(_cfg(), state, _inp(pending=0, running=0))
        self.assertFalse(d.wants_flip)


if __name__ == "__main__":
    unittest.main()
