"""#714: a KV rung whose floor sits ABOVE the pool cap is dead, and says so.

Specimen, 3x FLIP ABANDONED on 0b61699cc3:

    staging 1748 MiB needed but only 1693 MiB is spendable
    KV rung: current=137216 rows, floor=398471, slack=0

The 55 MiB is the symptom. The rung line is the cause, and it is arithmetically
impossible rather than merely tight.

THE FLOOR FORMULA IS CORRECT. ``_floor_rows`` is
``max_live + 1 + margin_rows + admission_reserve_rows``; ``margin_rows``
defaults to 0 and is never passed at the construction site, and
``admission_reserve_rows`` is ``chunked_prefill_size`` = 512. So
``floor = max_live + 513``, and floor=398471 means **max_live = 397,958** --
against a current cap of 137,216. A live row id 2.9x above the cap.

Compare the healthy shape the module itself documents (line 875):
``max_live=644 + admission reserve 512, slack=405894``. There max_live is tiny
and slack is enormous. Here the high-water id outlived the pool it was measured
in -- ids from a larger id space surviving a reshard/shrink.

WHY IT MATTERS BEYOND ONE LINE. ``slack = max(0, current - floor_rows)`` pins to
0 whenever floor exceeds current, so the rung can never propose a shrink. The
evict-rung funding path (#688) is therefore PERMANENTLY unavailable at this
operating point, and every flip falls back on the raw seam fund alone. The 55
MiB shortfall is simply what the raw fund misses once its backstop is gone --
which is why the same instance abandons three times instead of funding once.

A rung that is structurally dead must not report ``slack=0`` and nothing else:
that is indistinguishable from a rung that merely had no room this round. These
tests pin the distinction.

Hermetic: pure arithmetic on the summary, no CUDA.
"""

import unittest


class _Rung:
    """Minimal stand-in exposing only what last_proposal_summary reads."""

    def __init__(self, current, floor_rows, deficit=0, skipped=None, desire=None):
        self._last_proposal_terms = {
            "current": current,
            "floor_rows": floor_rows,
            "deficit": deficit,
            "skipped": skipped,
            "desire": current if desire is None else desire,
        }

    summary = None


def _summary(rung):
    from sglang.srt.managers.kv_backing_relief import KvBackingRelief

    return KvBackingRelief.last_proposal_summary(rung)


class AnUnreachableFloorIsNamed(unittest.TestCase):
    def test_the_specimen_reports_the_floor_as_unreachable(self):
        """THE FALSIFIER: slack=0 alone hides an impossible configuration."""
        s = _summary(_Rung(current=137216, floor_rows=398471))
        self.assertIn("slack=0", s)
        self.assertIn("UNREACHABLE", s.upper())

    def test_it_names_the_deficit_in_rows(self):
        """An operator needs the size of the impossibility, not just its fact."""
        s = _summary(_Rung(current=137216, floor_rows=398471))
        self.assertIn("261255", s.replace(",", ""))

    def test_a_healthy_rung_says_nothing_extra(self):
        """The module's own documented shape must be untouched."""
        s = _summary(_Rung(current=407051, floor_rows=1157))
        self.assertIn("slack=405894", s)
        self.assertNotIn("UNREACHABLE", s.upper())

    def test_floor_exactly_at_the_cap_is_not_unreachable(self):
        """Boundary: slack 0 with floor == current is tight, not impossible."""
        s = _summary(_Rung(current=1000, floor_rows=1000))
        self.assertIn("slack=0", s)
        self.assertNotIn("UNREACHABLE", s.upper())

    def test_a_rung_that_never_ran_still_reports_that_instead(self):
        class _NoTerms:
            _last_proposal_terms = None

        s = _summary(_NoTerms())
        self.assertIn("NO proposal", s)


if __name__ == "__main__":
    unittest.main()
