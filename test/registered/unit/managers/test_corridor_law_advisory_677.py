"""USER DECISION 2026-08-16: the corridor law is a SOFT target at seam entry.

THE DECISION, in the user's terms. The ~1024 MiB corridor line was introduced
only because the planner was not filling VRAM well enough; it stays as the
FILL-QUALITY target, and it stays the planner's job. It is not a safety
device. The single hard constraint is OOM avoidance. So no mechanism may
block, delay, or refuse solely on the 1024-line: dipping below it is
acceptable as long as no OOM results, and it is logged as a WARNING.

WHAT THIS COST BEFORE IT WAS DECIDED, on this box, in one morning:

* 06:47:48 -- ``ensure_headroom`` refused a seam whose staging FIT in free
  (PP1 want 2163 MiB against 2456 free) purely because the residual 293 MiB
  sat under the law. 76 refusals in a row, 727004 tok waiting, GPU idle.
* 07:02:15 -- a rank WITHHELD its entry-margin yield on a PREDICTED sub-law
  trough (864 MiB against the 1024 law) and emitted a delay that is exempt
  from the stand-down cap, so the delay streak climbed with no exit. bs 0,
  GPU 0%, 794179 tok waiting.

Both wedges protected a fill-quality target by stopping the machine. The
trade the law was defending -- a few hundred MiB of headroom -- was never
worth an idle instance, and the user has now said so directly.

WHAT REMAINS HARD, and this file pins the boundary: an allocation LARGER THAN
FREE cannot proceed, because that is not a corridor dip, it is an OOM. The
worst-measured-draw prediction stays as INPUT -- to the warning text and to
how much the spill provider tries to free -- never as a gate.
"""

import unittest

from sglang.srt.managers import corridor_guard as cg

MIB = 1024 * 1024


class _Card:
    def __init__(self, free_mib: int):
        self.free = free_mib * MIB

    def probe(self) -> int:
        return self.free


def _guard(card, floor=1536, delta=256, law=1024):
    return cg.CorridorGuard(
        0, floor_mib=floor, delta_mib=delta, law_floor_mib=law, probe=card.probe
    )


class TheLawNoLongerRefuses(unittest.TestCase):
    def test_the_0647_wedge_now_clears(self):
        """PP1's REAL numbers. want 2163 fits in 2456 free; the residual 293
        is under the law. That is a dip to warn about, not a reason to idle
        the instance with 727004 tokens waiting."""
        g = _guard(_Card(2456))
        r = g.ensure_headroom(2163 * MIB)
        self.assertTrue(r.ok, "a want that FITS must never be refused for the law")

    def test_the_pp2_numbers_clear_too(self):
        g = _guard(_Card(3560))
        self.assertTrue(g.ensure_headroom(2858 * MIB).ok)

    def test_a_dip_under_the_law_is_reported_as_a_breach_for_the_warning(self):
        """The caller has to be able to SAY it dipped -- point (a) requires a
        WARNING, which needs the fact to survive the verdict."""
        g = _guard(_Card(2456))
        r = g.ensure_headroom(2163 * MIB)
        self.assertTrue(getattr(r, "law_breached", False))
        self.assertIn("law", r.detail.lower())

    def test_a_clearance_that_holds_the_law_is_not_flagged(self):
        g = _guard(_Card(4096))
        r = g.ensure_headroom(1000 * MIB)
        self.assertTrue(r.ok)
        self.assertFalse(getattr(r, "law_breached", False))


class OomIsStillHard(unittest.TestCase):
    def test_an_allocation_larger_than_free_is_still_refused(self):
        """THE BOUNDARY. Not a corridor dip -- there is no memory. Softening
        this would turn a warning into a dead worker."""
        g = _guard(_Card(1000))
        r = g.ensure_headroom(2000 * MIB)
        self.assertFalse(r.ok)

    def test_the_refusal_counter_now_counts_only_real_impossibility(self):
        """A refuse_count inflated by law dips would make a healthy rig look
        like a failing one, and this counter feeds pool sizing."""
        g = _guard(_Card(2456))
        g.ensure_headroom(2163 * MIB)
        self.assertEqual(0, g.refuse_count)
        g2 = _guard(_Card(1000))
        g2.ensure_headroom(2000 * MIB)
        self.assertEqual(1, g2.refuse_count)


if __name__ == "__main__":
    unittest.main()


class TheSeamCommitChunkDefaultIsArmed(unittest.TestCase):
    """#688: the row-blocking machinery must be reachable in a default boot.

    IT WAS NOT. `SGLANG_FLIP_SEAM_CHUNK_MIB` defaulted to 0, and
    `_effective_row_blocks` returns 1 when the arena cannot do span ops -- so
    the shipped 16-block default was unreachable in every fresh deployment.
    This rig only ever exercised row-blocking because its operator env carried
    `SGLANG_FLIP_SEAM_CHUNK_MIB=8` by hand, which is also what made the first
    attempt to price it come out as "no effect": that A/B was 8 against 16,
    both above the arena chunk floor, and it measured nothing because there
    was nothing between them to measure.

    Priced properly at 0 against 16 on 7b706e8b89, same argv, same load, at
    matched live-slot counts: 216-377 MiB per rank saved, as a CONSTANT offset
    rather than a slope, with corridor law warnings falling 6 -> 3 over the
    same work. That is the same order as the margin the 06:47:48 wedge was
    short by.
    """

    def test_the_default_is_no_longer_zero(self):
        import inspect as _inspect

        from sglang.srt.mem_cache import memory_pool

        src = _inspect.getsource(
            memory_pool.MHATokenToKVPool._alloc_post_capture_buffers
        )
        self.assertIn('"SGLANG_FLIP_SEAM_CHUNK_MIB", "8"', src)
        self.assertNotIn('"SGLANG_FLIP_SEAM_CHUNK_MIB", "0"', src)

    def test_the_measurement_travels_with_the_default(self):
        """A default whose provenance is not written down gets reverted by the
        next person who reads the old 'defaults to 0' comment."""
        import inspect as _inspect

        from sglang.srt.mem_cache import memory_pool

        src = _inspect.getsource(
            memory_pool.MHATokenToKVPool._alloc_post_capture_buffers
        )
        self.assertIn("MEASURED UNDER LOAD", src)
        self.assertIn("1438.25", src)
