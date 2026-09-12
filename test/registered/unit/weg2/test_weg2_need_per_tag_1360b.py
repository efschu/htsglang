# SPDX-License-Identifier: Apache-2.0
"""#1360b -- one WEG2-RING NEED line per SAVED tag, so the anchor is measurable.

THE GAP, measured on weg2xsn24 and weg2xsn25: the entire NEED tag census of both
boots reads `weights_0..weights_7`, `weights_draft`, `weights` and NOTHING else.
The leg only ever guarded the weights family -- correctly, because those are the
tags whose bytes the waking peer has to release -- so the "extra" population the
ring anchor needs (capture pool, workspace, static clones, everything else
`enable_cpu_backup` saves) was empty BY CONSTRUCTION, and the question "does
weights + weights_draft cover what the saver puts away?" could not be answered
from a boot at all. #1350c left it open with exactly that name.

OBSERVATION ONLY, and that restriction is the whole safety of this change: the
extra tags go through `record()`, never `guard_tag()`, so nothing waits, nothing
refuses, and no flip timing moves. They are not ring-carried, so a `free`
comparison is meaningless for them -- the line states their BYTES under their
own name, which is what the anchor sums.
"""

import logging
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_guard as rg
from sglang.test.test_utils import CustomTestCase


class _Log(logging.Logger):
    def __init__(self):
        super().__init__("t")
        self.seen = []

    def info(self, msg, *a):
        self.seen.append(msg % a if a else msg)


class NeedLinePerSavedTag1360b(CustomTestCase):
    def test_the_leg_emits_for_every_census_tag_not_only_the_weights_family(self):
        """The call site itself -- desk-written-never-executed."""
        import inspect
        from sglang.srt.managers.scheduler_components import weight_updater as wu
        src = inspect.getsource(wu)
        self.assertIn("#1360b", src)
        self.assertIn("for _tag, _nb in sorted(census.items()):", src)
        self.assertIn("if _tag in weights_tags:", src)
        # OBSERVATION ONLY: the extra pass must use record(), never guard_tag()
        block = src[src.index("#1360b"):src.index("weg2_leg_ms = (time.perf")]
        # CODE lines only: the block's own prose explains why it does NOT
        # guard, so a naive substring check would read its own comment.
        code = "\n".join(
            ln for ln in block.splitlines() if not ln.strip().startswith("#"))
        self.assertIn("weg2_ring_guard.record(", code)
        self.assertNotIn("guard_tag", code)

    def test_an_extra_tag_gets_its_own_named_line_with_its_bytes(self):
        log = _Log()
        g = rg.RingNeedGuard("card-x", group="P", rank=0, log=log)
        g.set_leg(3, "P", "D")
        g.record("capture_pool", rg.need_mib(512 * 1024 * 1024), 0)
        line = log.seen[-1]
        self.assertIn("tag=capture_pool", line)
        self.assertIn("leg=3/P->D", line)
        self.assertIn("need_mib=512", line)

    def test_the_five_fixed_fields_stay_on_every_line(self):
        log = _Log()
        g = rg.RingNeedGuard("card-x", group="D", rank=2, log=log)
        g.record("weights_0", 100, 50)
        g.record("workspace", 8, 50)
        for line in log.seen:
            for field in ("tag=", "card=", "leg=", "need_mib=", "free_mib=",
                          "delta_mib="):
                self.assertIn(field, line)

    def test_free_mib_or_none_is_an_absence_never_a_zero(self):
        g = rg.RingNeedGuard("card-x")
        self.assertIsNone(g.free_mib_or_none(lambda: None))
        self.assertIsNone(g.free_mib_or_none(lambda: {}))
        self.assertEqual(
            g.free_mib_or_none(
                lambda: {"granules_free": 4, "granule_bytes": rg.RING_GRANULE_BYTES}),
            4 * rg.RING_GRANULE_BYTES // (1024 * 1024),
        )

    def test_the_extra_population_is_what_the_anchor_was_missing(self):
        """The coverage sum is only computable once both families are on the line."""
        log = _Log()
        g = rg.RingNeedGuard("card-x", group="P", rank=0, log=log)
        g.set_leg(0, "P", "D")
        for tag, mib in (("weights_0", 5976), ("weights_draft", 3144),
                         ("capture_pool", 512), ("workspace", 96)):
            g.record(tag, mib, 0)
        weights = sum(s.need_mib for s in g.samples if s.tag.startswith("weights"))
        extra = sum(s.need_mib for s in g.samples if not s.tag.startswith("weights"))
        self.assertEqual(weights, 9120)
        self.assertEqual(extra, 608)      # the set that was empty before #1360b
        self.assertGreater(extra, 0)


if __name__ == "__main__":
    unittest.main()
