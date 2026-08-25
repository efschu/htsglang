"""#858: quiescence re-derived against the no-carry seam.

TWO SITES IN ONE PREDICATE, PULLING OPPOSITE WAYS.
`build_flip_quiescence_fn` carried two rules whose justifications both died
with #856 (2026-08-24, "the flip carries no KV"):

  * the BETWEEN-CHUNKS allowance (#631 defect O, 2026-08-09) LET a flip
    through mid-chunked-prefill, justified verbatim as "exactly the state the
    carry moves". Post-#856 that state is freed, not moved -- so the flip
    discards the prefill. W37-H arm A: 51 flips, 0 decode rounds, 0
    completions; W38-B logged the mechanism directly, "prefill still chunked
    (allocated=4096, needs=6045)".
  * the ORPHAN gate (#631 defect L) BLOCKED a flip on requests "not yet merged
    into the resident set the carry harvests". There is no harvest, and
    de4f541b41 made `_live_reqs` enumerate the identical population.

One let through what must now be refused; the other refused for a reason that
no longer exists. Fixing only the first would leave the predicate half
governed by a deleted mechanism -- and would plausibly convert the livelock
into a stall rather than into progress.

WHY `strict` AND NOT A REVERSAL. Blocking every incomplete chunk re-creates
defect O, where a flip armed FOR a prefill could not land until that prefill
had finished. Under strict batching the flip is armed for the DECODE after the
drain, so waiting IS drain-and-flip. The two greens below pin that distinction;
without them a predicate that blocks everything would read as a pass.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest

from sglang.srt.managers.phase_flip_runtime import chunk_blocks_quiescence
from sglang.test.test_utils import CustomTestCase


def _chunked(*, allocated, total, has_row=True):
    """A chunked prefill: `extend_range.end` rows written of `total` needed."""
    return types.SimpleNamespace(
        rid="r0",
        req_pool_idx=0 if has_row else None,
        extend_range=types.SimpleNamespace(start=0, end=allocated),
        full_untruncated_fill_ids=list(range(total)),
    )


class TestStrictRefusesAnIncompleteChunkedPrefill(CustomTestCase):
    def test_strict_blocks_an_incomplete_chunked_prefill(self):
        """RED before the fix. The W37-H shape: 4096 of 6045 rows written."""
        req = _chunked(allocated=4096, total=6045)
        self.assertTrue(
            chunk_blocks_quiescence(req, strict=True),
            "post-#856 a mid-chunk flip DISCARDS this prefill; strict must wait",
        )

    def test_strict_permits_a_completed_prefill(self):
        """GREEN BEFORE AND AFTER -- the control that makes the red mean
        something. A predicate that simply blocked every chunked_req would
        pass the test above and fail this one."""
        req = _chunked(allocated=6045, total=6045)
        self.assertFalse(
            chunk_blocks_quiescence(req, strict=True),
            "a finished prefill is a settled boundary and must not block",
        )


class TestNonStrictIsUnchanged(CustomTestCase):
    """GREEN BEFORE AND AFTER. Non-strict keeps #631 defect O's fix exactly.

    RESIDUAL, deliberate and filed: post-#856 a mid-chunk flip discards the
    prefill in non-strict too. We decline to block there because an
    unconditional block re-creates defect O. The correct non-strict answer is
    UNKNOWN and is filed, not solved -- this test pins today's behaviour, it
    does not certify it as sound.
    """

    def test_non_strict_still_allows_between_chunks(self):
        req = _chunked(allocated=4096, total=6045)
        self.assertFalse(
            chunk_blocks_quiescence(req, strict=False),
            "blocking here unconditionally re-creates #631 defect O",
        )

    def test_mid_admission_still_blocks_in_both_modes(self):
        """The original rule survives untouched: no pool row yet means the KV
        has no home at all."""
        req = _chunked(allocated=0, total=6045, has_row=False)
        self.assertTrue(chunk_blocks_quiescence(req, strict=False))
        self.assertTrue(chunk_blocks_quiescence(req, strict=True))

    def test_no_chunked_request_never_blocks(self):
        self.assertFalse(chunk_blocks_quiescence(None, strict=True))
        self.assertFalse(chunk_blocks_quiescence(None, strict=False))


class TestTheOrphanGateIsGone(CustomTestCase):
    """RED before the fix: the gate blocked on a population `_live_reqs`
    already enumerates."""

    def test_quiescence_does_not_consult_the_carry_orphan_query(self):
        import inspect

        from sglang.srt.managers.phase_flip_runtime import (
            build_flip_quiescence_fn,
        )

        src = inspect.getsource(build_flip_quiescence_fn)
        self.assertNotIn(
            "orphan_resident_reqs(",
            src,
            "the orphan gate blocked for a harvest that #856 deleted; "
            "de4f541b41 gave _live_reqs the identical population",
        )

    def test_live_reqs_still_covers_that_population(self):
        """The removal is only safe because the enumeration moved, not
        vanished. If `_live_reqs` ever stops reading last_mbs/last_batch, the
        gate's removal becomes a real hole and this reddens."""
        import inspect

        from sglang.srt.managers.phase_flip_runtime import _live_reqs

        src = inspect.getsource(_live_reqs)
        self.assertIn("last_mbs", src)
        self.assertIn("last_batch", src)


class TestBothCallersAskTheSameQuestion(CustomTestCase):
    """The helper's own docstring: "ONE definition with TWO callers, and they
    must never disagree." They drifted once already (2026-08-09 20:31:38Z), so
    the strict term must reach BOTH."""

    def test_the_park_site_passes_strict_too(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.get_next_batch_to_run)
        self.assertIn("chunk_blocks_quiescence(", src)
        self.assertIn("strict=", src)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# #858b: DIRECTION. The block above was direction-BLIND, and every test in this
# file passed anyway -- which is exactly why it shipped. A direction-blind
# predicate satisfies every single-direction test, so the tests are
# parametrised over BOTH directions from here on.
# ---------------------------------------------------------------------------

from sglang.srt.managers.phase_flip_runtime import (  # noqa: E402
    prefill_runnable_in_current_layout,
)
from sglang.srt.managers.phase_policy import PP_TO_TP, TP_TO_PP  # noqa: E402


class _Purity:
    def __init__(self, prefill_in_tp: bool):
        self._p = prefill_in_tp
        self.strict = not prefill_in_tp

    def prefill_allowed_in_tp(self) -> bool:
        return self._p


class TestTheDirectionTerm(CustomTestCase):
    """`pp_to_tp` is armed for the DECODE after the drain -- waiting is
    drain-and-flip. `tp_to_pp` is armed FOR the prefill, and under strict that
    prefill may not run in the TP layout that holds while we wait."""

    def test_pp_to_tp_prefill_can_progress(self):
        self.assertTrue(
            prefill_runnable_in_current_layout(PP_TO_TP, _Purity(prefill_in_tp=False))
        )

    def test_tp_to_pp_under_strict_cannot(self):
        self.assertFalse(
            prefill_runnable_in_current_layout(TP_TO_PP, _Purity(prefill_in_tp=False))
        )

    def test_tp_to_pp_when_tp_may_prefill_can(self):
        self.assertTrue(
            prefill_runnable_in_current_layout(TP_TO_PP, _Purity(prefill_in_tp=True))
        )

    def test_unknown_direction_does_not_invent_a_hold(self):
        """No armed direction -> do not manufacture a reason to wait."""
        self.assertTrue(
            prefill_runnable_in_current_layout(None, _Purity(prefill_in_tp=False))
        )


class TestBothDirectionsAgainstTheSamePrefill(CustomTestCase):
    """THE TEST THAT WOULD HAVE CAUGHT #858. One incomplete chunked prefill,
    asked in both directions: the answers must DIFFER under strict."""

    def _req(self):
        return _chunked(allocated=4096, total=6045)

    def test_strict_blocks_pp_to_tp(self):
        runnable = prefill_runnable_in_current_layout(
            PP_TO_TP, _Purity(prefill_in_tp=False)
        )
        self.assertTrue(
            chunk_blocks_quiescence(
                self._req(), strict=True, prefill_runnable_here=runnable
            ),
            "pp_to_tp must still wait: the prefill CAN finish in PP",
        )

    def test_strict_does_not_block_tp_to_pp(self):
        """RED before #858b. This is the 225-of-228 hold from
        boot_w40_857strict_0825_1931, and it has no exit: TP has only a
        MINIMUM dwell (`tp_decode_floor_s`), never a bound."""
        runnable = prefill_runnable_in_current_layout(
            TP_TO_PP, _Purity(prefill_in_tp=False)
        )
        self.assertFalse(
            chunk_blocks_quiescence(
                self._req(), strict=True, prefill_runnable_here=runnable
            ),
            "tp_to_pp is armed FOR this prefill, which strict forbids in TP: "
            "waiting deadlocks",
        )


class TestTheTpExitIsValidatedAtBoot(CustomTestCase):
    """#858b: the configuration that deadlocks is refused at parse time, in
    the shape `validate_purity_policy_pair` already uses for PP."""

    def _cfg(self, **kw):
        base = dict(drain_mode_strict=True, decode_stall_slo_s=0.0, tp_window_s=0.0)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_the_deadlocking_triple_is_refused(self):
        from sglang.srt.managers.phase_purity import (
            PhasePurityError,
            parse_purity,
            validate_tp_exit_pair,
        )

        with self.assertRaises(PhasePurityError):
            validate_tp_exit_pair(parse_purity("strict"), self._cfg())

    def test_a_declared_slo_is_an_exit(self):
        from sglang.srt.managers.phase_purity import parse_purity, validate_tp_exit_pair

        validate_tp_exit_pair(parse_purity("strict"), self._cfg(decode_stall_slo_s=180))

    def test_non_strict_is_not_refused(self):
        """GREEN BEFORE AND AFTER -- the guard must not fire on the mode that
        never had this problem."""
        from sglang.srt.managers.phase_purity import parse_purity, validate_tp_exit_pair

        validate_tp_exit_pair(parse_purity("off"), self._cfg())
