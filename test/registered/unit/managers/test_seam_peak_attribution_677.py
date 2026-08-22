"""#677: the seam draw must be attributable per component at its peak.

WHY THIS EXISTS. The arming floor is one scalar per rank -- 909 / 1006 /
1648 MiB on this rig -- with no recorded composition. Nothing in the tree
says what those bytes are made of, so no part of them can be traded: a
component that could live in host RAM between flips cannot be identified,
and one that genuinely must stay resident cannot be defended. Every
per-component decision #702 and #677 want to take is, until this exists,
arithmetic on an undivided number.

WHERE THE MARK SITS. ``_staging_affordable`` is the instant the flip's
demand is weighed against free VRAM -- the peak the floor is sized to
survive. Earlier the buffers do not exist; later the decision is already
taken.

WHAT THESE PINS PROTECT, in order of how badly getting them wrong would
mislead a reader:

  1. Unmeasured components are NULL, never 0. A zero reads as "costs
     nothing" and would retire a question that was never asked (#606).
  2. The residual is SIGNED. A negative residual means the named terms
     over-count, which is a different defect from an unattributed
     remainder, and max(0, ...) would hide it.
  3. The instrument cannot break a flip. It sits on the seam path, which is
     the no-return region, so it may cost a missing line and never a
     cutover (the #631 census contract, re-pinned at a second site).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers import phase_flip_runtime as pfr


class _Runtime:
    """A shell carrying only what ``_record_seam_peak`` reads."""

    def __init__(self, arena_tail=4096, rank=2):
        self._staging_reserve_bytes = 1024
        self._epoch = 7
        self._world_rank = rank
        self._arena_tail = arena_tail
        self._record_seam_peak = pfr.PhaseFlipRuntime._record_seam_peak.__get__(
            self, _Runtime
        )

    def _arena_tail_bytes(self, direction):
        if self._arena_tail is None:
            raise RuntimeError("no arena on this rank")
        return self._arena_tail


class _Capture:
    """Stands in for the #605 recorder and keeps every mark."""

    def __init__(self):
        self.marks = []

    def mark(self, phase, *, rank=0, extra=None, **kw):
        self.marks.append((phase, rank, dict(extra or {})))
        return {}


def _run(runtime, staging=8192, driver_free=1 << 30, cached=1 << 20):
    cap = _Capture()
    import sys

    saved = sys.modules.get("sglang.srt.mem_ledger.flight_recorder")
    sys.modules["sglang.srt.mem_ledger.flight_recorder"] = cap
    fake_pkg = SimpleNamespace(flight_recorder=cap)
    saved_pkg = sys.modules.get("sglang.srt.mem_ledger")
    sys.modules["sglang.srt.mem_ledger"] = fake_pkg
    try:
        runtime._record_seam_peak("pp_to_tp", staging, driver_free, cached)
    finally:
        if saved is not None:
            sys.modules["sglang.srt.mem_ledger.flight_recorder"] = saved
        if saved_pkg is not None:
            sys.modules["sglang.srt.mem_ledger"] = saved_pkg
    return cap


class TestItEmitsOnTheRecorderChannel(unittest.TestCase):
    def test_a_seam_peak_mark_is_emitted_with_the_rank(self):
        cap = _run(_Runtime())

        self.assertEqual(len(cap.marks), 1)
        phase, rank, extra = cap.marks[0]
        self.assertEqual(phase, "seam_peak")
        self.assertEqual(rank, 2, "the mark must carry the rank it describes")
        self.assertEqual(extra["direction"], "pp_to_tp")
        self.assertEqual(extra["epoch"], 7)

    def test_the_named_components_are_present(self):
        _phase, _rank, extra = _run(_Runtime(arena_tail=4096)).marks[0]

        self.assertEqual(extra["staging_bytes"], 8192)
        self.assertEqual(extra["refill_destination_bytes"], 4096)
        self.assertEqual(extra["staging_reserve_bytes"], 1024)
        self.assertEqual(extra["driver_free_bytes"], 1 << 30)
        self.assertEqual(extra["allocator_cached_free_bytes"], 1 << 20)


class TestUnmeasuredIsNullNotZero(unittest.TestCase):
    """#606: a defaulted measurement is worse than an absent one."""

    def test_graph_workspace_is_null_because_it_is_not_visible_here(self):
        _phase, _rank, extra = _run(_Runtime()).marks[0]

        self.assertIsNone(
            extra["graph_workspace_bytes"],
            "0 would read as 'graph workspace costs nothing' and retire a "
            "question nobody asked",
        )

    def test_an_unreadable_arena_tail_is_null_not_zero(self):
        _phase, _rank, extra = _run(_Runtime(arena_tail=None)).marks[0]

        self.assertIsNone(extra["refill_destination_bytes"])


class TestTheResidualIsSigned(unittest.TestCase):
    def test_the_residual_is_what_the_named_terms_do_not_explain(self):
        _phase, _rank, extra = _run(_Runtime(arena_tail=2048), staging=8192).marks[0]

        self.assertEqual(extra["named_bytes"], 8192 + 2048)
        self.assertEqual(extra["unattributed_bytes"], 8192 - (8192 + 2048))

    def test_a_negative_residual_is_reported_not_floored(self):
        """Over-counting is a real defect and must stay visible. Flooring it
        at 0 would make an over-counted floor look perfectly explained."""
        _phase, _rank, extra = _run(_Runtime(arena_tail=1 << 20), staging=4096).marks[0]

        self.assertLess(
            extra["unattributed_bytes"],
            0,
            "a named total above the peak must surface as a negative residual",
        )


class TestItCannotBreakAFlip(unittest.TestCase):
    """The no-return-path contract, re-pinned at this second site."""

    def test_a_recorder_that_raises_does_not_escape(self):
        class _Angry:
            def mark(self, *a, **kw):
                raise RuntimeError("recorder is having a day")

        import sys

        saved = sys.modules.get("sglang.srt.mem_ledger")
        sys.modules["sglang.srt.mem_ledger"] = SimpleNamespace(flight_recorder=_Angry())
        try:
            _Runtime()._record_seam_peak("pp_to_tp", 1, 2, 3)
        finally:
            if saved is not None:
                sys.modules["sglang.srt.mem_ledger"] = saved

    def test_a_runtime_missing_its_fields_does_not_escape(self):
        bare = SimpleNamespace()
        bound = pfr.PhaseFlipRuntime._record_seam_peak.__get__(bare, SimpleNamespace)
        bound("pp_to_tp", 1, 2, 3)


if __name__ == "__main__":
    unittest.main()
