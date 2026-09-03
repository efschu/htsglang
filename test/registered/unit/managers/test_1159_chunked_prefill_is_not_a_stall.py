"""#1159 -- an ADVANCING chunked prefill is not a stalled decode bundle.

THE MEASUREMENT, boot weg1b3
(/spinning/evidence-665-f1/boot_855_weg1b3_6980c75eac_0902_234752.log:100502)::

    [2026-09-02 23:56:35 PP0] PHASE-POLICY arming tp_to_pp: decode bundle
    STALLED, not draining: 1 of 1 req still decoding and the set has not
    shrunk for 11.4s (deadline 10.0s), while 33504 tok of prefill waits.

The one request in that bundle was rid 679e4568, and it was not stalled: its
PP0 ADMIT prefix ladder is strictly monotone, 4096, 6281, 8192, ... 20480 in
that window -- one 4096-token chunk every ~3 s. A chunked prefill making
exactly the progress the runtime asks of it. The arm cost a 70 s cutover
(RECONCILED at 23:57:47, total 70.091 s) and the cutover retracted the request
whose context it was building -- the #939 breach measured two lines later
(worst=79931 tok recomputed, 19.5 chunks against the one-chunk law).

WHY THE CLOCK WAS WRONG. ``last_bundle_progress_at`` was stamped on exactly
two axes: ``running_bs`` shrinking (#833) and ``seam_cohort_pending_tokens``
sinking (#1069). A request that is being CHUNK-PREFILLED holds ``running_bs``
flat by construction -- it occupies one slot for its whole prefill -- and it is
not a seam cohort member, so neither axis moved and the set read as frozen
while it was advancing 4096 tokens every 3 s.

THE AXIS THIS ADDS, and it is the same shape as its two neighbours: the
in-flight chunked request's OWN computed prefix. Progress = the SAME request's
prefix strictly growing. A different request appearing with a large prefix is
admission REFILLING the bundle, which is precisely what #833 must keep reading
as non-progress -- so the rid is compared, never just the number, and a set
that neither shrinks nor advances still arms exactly as it does today.
"""

import unittest

from sglang.srt.managers.phase_policy import (
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    TP_TO_PP,
    observe_idle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

#: The boot's own numbers, verbatim from log line 100502.
B3_PENDING = 33504
B3_BUNDLE = 1
B3_RID = "679e45685bf74fc9a7c564be9d972089"
#: Two consecutive rungs of that request's measured prefix ladder.
B3_PREFIX_BEFORE = 16384
B3_PREFIX_AFTER = 20480


def _cfg(**kw):
    fields = dict(
        enabled=True,
        drain_mode=True,
        flip_tokens=1,
        pp_exit_tokens=1,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        flip_cost_s=3.2,
    )
    fields.update(kw)
    return PhasePolicyConfig(**fields)


def _inp(now: float, **kw):
    fields = dict(
        phase="tp",
        running_bs=B3_BUNDLE,
        pending_prefill_tokens=B3_PENDING,
        now=now,
    )
    fields.update(kw)
    return PhasePolicyInputs(**fields)


def _state(now: float):
    """A TP phase already observed once, with the bundle flat since t=0."""
    return PhasePolicyState(
        last_phase="tp",
        phase_since=0.0,
        bundle_at_phase_entry=B3_BUNDLE,
        last_bundle_progress_at=0.0,
        last_running_bs=B3_BUNDLE,
    )


def _decide(cfg, state, inp):
    from sglang.srt.managers.phase_policy import _decide_from_load

    return _decide_from_load(cfg, state, inp)


class AnAdvancingChunkedPrefillIsNotAStall(unittest.TestCase):
    """The weg1b3 arm, replayed. It must not fire."""

    def test_the_weg1b3_arm_does_not_fire(self):
        cfg = _cfg()
        state = _state(0.0)
        # Round 1, t=8.4 s into the flat bundle: the request has 16384
        # computed tokens.
        first = _inp(
            8.4,
            chunked_prefill_rid=B3_RID,
            chunked_prefill_computed_tokens=B3_PREFIX_BEFORE,
        )
        observe_idle(state, first)
        # Round 2, t=11.4 s (the boot's own stall figure), one chunk later.
        second = _inp(
            11.4,
            chunked_prefill_rid=B3_RID,
            chunked_prefill_computed_tokens=B3_PREFIX_AFTER,
        )
        observe_idle(state, second)
        self.assertEqual(
            state.last_bundle_progress_at,
            11.4,
            "a chunk landing on the SAME request is bundle progress",
        )
        decision = _decide(cfg, state, second)
        self.assertFalse(
            decision.direction is not None and "STALLED" in (decision.reason or ""),
            f"armed against an advancing chunked prefill: {decision.reason!r}",
        )

    def test_a_genuinely_stalled_bundle_still_arms(self):
        """#833's own case, unchanged: flat bs AND a flat prefix."""
        cfg = _cfg()
        state = _state(0.0)
        for t in (5.0, 11.4):
            observe_idle(
                state,
                _inp(
                    t,
                    chunked_prefill_rid=B3_RID,
                    chunked_prefill_computed_tokens=B3_PREFIX_BEFORE,
                ),
            )
        self.assertEqual(state.last_bundle_progress_at, 0.0)
        decision = _decide(
            cfg,
            state,
            _inp(
                11.4,
                chunked_prefill_rid=B3_RID,
                chunked_prefill_computed_tokens=B3_PREFIX_BEFORE,
            ),
        )
        self.assertIsNotNone(decision.direction)
        self.assertEqual(decision.direction, TP_TO_PP)
        self.assertIn("STALLED", decision.reason or "")

    def test_a_different_request_is_refill_not_progress(self):
        """Admission refilling the bundle must still read as non-progress."""
        state = _state(0.0)
        observe_idle(
            state,
            _inp(
                5.0,
                chunked_prefill_rid=B3_RID,
                chunked_prefill_computed_tokens=B3_PREFIX_AFTER,
            ),
        )
        observe_idle(
            state,
            _inp(
                11.4,
                chunked_prefill_rid="c4e85437098642179eb28151bc132b8d",
                # A LARGER number, but on another request: refill, not drain.
                chunked_prefill_computed_tokens=B3_PREFIX_AFTER + 4096,
            ),
        )
        self.assertEqual(state.last_bundle_progress_at, 0.0)

    def test_an_unsupplied_field_reproduces_the_old_behaviour(self):
        """No stand-in and no non-flip deployment changes behaviour."""
        state = _state(0.0)
        observe_idle(state, _inp(5.0))
        observe_idle(state, _inp(11.4))
        self.assertEqual(state.last_bundle_progress_at, 0.0)


if __name__ == "__main__":
    unittest.main()
