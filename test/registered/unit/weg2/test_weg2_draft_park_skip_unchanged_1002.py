"""LAYER-REST-1002 (NF y7t D->P): the draft park skips its D2H when the
device bytes are the image's.

y7t ep3/5/7/9/11: D TP0's pre-loop carried ``draft_park`` 139-162 ms (D2H
113 ms of 1523 MiB + pause 9); P PP0's first claim (weights_0) was granted
exactly when that pre-loop ended (collect t0 within 2-13 ms of TP0's loop
start), and D TP1/TP2's first deposits (weights_0 -> PP0, BAR1) waited
~150 ms on that collector in every flip. The draft is static: every wake
H2D's the image back, so the next park's D2H rewrites the same bytes.

Pinned, CPU-only:

1. second park after a round trip with unchanged bytes: no D2H, the line says
   so, the pause still follows, the host image is intact;
2. any write into the draft between parks (even inside one MiB granule):
   the D2H runs and the new bytes reach the image;
3. a moved layout (other storages): the D2H runs;
4. the switch off: the D2H on every park, as before;
5. the digest is position-sensitive at its granule.
"""

import unittest

import pytest
import torch

try:
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import draft_park as dpk
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # > 1 MiB so the per-MiB granule path and the tail path both run
        self.w = torch.nn.Parameter(
            torch.arange((1 << 18) + 37, dtype=torch.float32), requires_grad=False)
        self.register_buffer("odd", torch.arange(7, dtype=torch.uint8))  # n % 4 != 0


def _round_trip(park, draft, pop, events):
    """park -> (unmap) -> resume -> H2D -> join, the saver faked."""
    saved = [v.clone() for _n, v in pop]

    def pause(tag):
        events.append("pause")
        for _n, v in pop:
            v.zero_()      # the unmap: device bytes gone

    rec = park.park(pop, tag="weights_draft", pause=pause, sync=lambda: events.append("sync"))
    park.unpark_start(tag="weights_draft", resume=lambda t: events.append("resume"))
    park.join(overlap="")
    for (_n, v), s in zip(pop, saved):
        assert torch.equal(v, s), "the wake must restore the bytes from the image"
    return rec


class TestSkipUnchanged(unittest.TestCase):
    def test_unchanged_bytes_skip_the_d2h_and_keep_the_order(self):
        draft = _Draft()
        park = dpk.DraftHostPark(pin=False)
        pop = dpk.park_population(draft, None)
        with envs.SGLANG_OPT_WEG2_DRAFT_PARK_SKIP_UNCHANGED.override(True):
            ev = []
            r1 = _round_trip(park, draft, pop, ev)
            self.assertFalse(r1.d2h_skipped)
            self.assertEqual(r1.digest, "first")
            image = park.host.clone()
            ev2 = []
            r2 = _round_trip(park, draft, dpk.park_population(draft, None), ev2)
        self.assertTrue(r2.d2h_skipped)
        self.assertEqual(r2.digest, "match")
        self.assertIn("d2h skipped, image current: digest match", r2.line())
        self.assertIn("reused", r2.line())
        # the pause still runs after the sync, and the image is untouched
        self.assertEqual(ev2[:2], ["sync", "pause"])
        self.assertTrue(torch.equal(park.host, image))

    def test_a_write_between_parks_forces_the_d2h(self):
        draft = _Draft()
        park = dpk.DraftHostPark(pin=False)
        with envs.SGLANG_OPT_WEG2_DRAFT_PARK_SKIP_UNCHANGED.override(True):
            _round_trip(park, draft, dpk.park_population(draft, None), [])
            # one word inside the first MiB granule changes
            draft.w.data[5] = -1.0
            pop = dpk.park_population(draft, None)
            r = park.park(pop, tag="weights_draft", pause=lambda t: None, sync=lambda: None)
        self.assertFalse(r.d2h_skipped)
        self.assertEqual(r.digest, "differs")
        self.assertEqual(float(park.host[:park.entries[0].nbytes].view(torch.float32)[5]), -1.0)

    def test_a_moved_layout_forces_the_d2h(self):
        draft = _Draft()
        park = dpk.DraftHostPark(pin=False)
        with envs.SGLANG_OPT_WEG2_DRAFT_PARK_SKIP_UNCHANGED.override(True):
            _round_trip(park, draft, dpk.park_population(draft, None), [])
            draft.w = torch.nn.Parameter(draft.w.data.clone(), requires_grad=False)  # new storage
            r = park.park(dpk.park_population(draft, None), tag="weights_draft",
                          pause=lambda t: None, sync=lambda: None)
        self.assertFalse(r.d2h_skipped)
        self.assertEqual(r.digest, "layout-moved")

    def test_switch_off_copies_every_park(self):
        draft = _Draft()
        park = dpk.DraftHostPark(pin=False)
        with envs.SGLANG_OPT_WEG2_DRAFT_PARK_SKIP_UNCHANGED.override(False):
            _round_trip(park, draft, dpk.park_population(draft, None), [])
            r = _round_trip(park, draft, dpk.park_population(draft, None), [])
        self.assertFalse(r.d2h_skipped)
        self.assertEqual(r.digest, "off")
        self.assertIn("d2h ", r.line())


class TestDigest(unittest.TestCase):
    def test_granules_are_position_sensitive(self):
        a = torch.zeros(3 * dpk.DIGEST_CHUNK_BYTES + 5, dtype=torch.uint8)
        b = a.clone()
        a[10] = 1                                   # granule 0
        b[dpk.DIGEST_CHUNK_BYTES + 10] = 1          # granule 1, same word sum overall
        self.assertFalse(torch.equal(dpk.content_digest([a]), dpk.content_digest([b])))
        self.assertTrue(torch.equal(dpk.content_digest([a]), dpk.content_digest([a.clone()])))

    def test_tail_bytes_count(self):
        a = torch.zeros(9, dtype=torch.uint8)
        b = a.clone()
        b[8] = 3
        self.assertFalse(torch.equal(dpk.content_digest([a]), dpk.content_digest([b])))


if __name__ == "__main__":
    unittest.main()
