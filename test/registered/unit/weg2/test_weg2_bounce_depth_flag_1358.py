# SPDX-License-Identifier: Apache-2.0
"""#1358 -- the bounce depth becomes a flag, and ONE source prices it.

weg2xsn27 died in `shm,pinned`, and the #1358 reader attributed the WHOLE
10.565 GiB of that region to five slot holders x 2.113 GiB from one emit site.
The buffer is `widest x assemble_slots(depth)` and `slots = depth + 1` under
shadow, so one step of depth is one widest layer per holder:

    depth=2   slots=3   2.113 GiB/holder   10.566 GiB over 5
    depth=1   slots=2   1.409 GiB/holder    7.044 GiB over 5
    saving                                  3.52 GiB

at the sleep leg, in the region the boot died in.

DEFAULT 2 IS BYTE-IDENTICAL. The flag exposes the constant; it does not move
it. And the value is published with the rest of the bounce terms, so P, D and
the host ledger read ONE source -- a second spelling of the depth is exactly
how `2.16 GiB` came to have three causes (#1361 [23a]).

THE COST IS NOT ASSERTED HERE. depth=1 leaves one pipeline stage fewer and
nothing on this rig has measured what that does to flip duration; operator
order 2026-09-13 is that it gets measured at the metal, never estimated. This
file pins the MEMORY arithmetic and the single source, and claims nothing about
throughput.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import xchg_bounce as xb
from sglang.test.test_utils import CustomTestCase

WIDEST = 756323776  # gdncov INT8's widest layer, 721.3 MiB
HOLDERS = 5         # measured on weg2xsn27 by the #1358 reader


class TheDepthFlagChangesTheSlotsAndTheTerm(CustomTestCase):
    def test_depth_1_gives_two_slots_under_shadow(self):
        self.assertEqual(xb.assemble_slots(1, comparing=True), 2)
        self.assertEqual(xb.assemble_slots(2, comparing=True), 3)

    def test_the_ledger_term_falls_from_10566_to_7044(self):
        """The ordered numbers, asserted as numbers."""
        g = 1024 ** 3
        at2 = xb.assemble_buffer_bytes(WIDEST, 2, comparing=True) * HOLDERS / g
        at1 = xb.assemble_buffer_bytes(WIDEST, 1, comparing=True) * HOLDERS / g
        self.assertAlmostEqual(at2, 10.566, places=2)
        self.assertAlmostEqual(at1, 7.044, places=2)
        self.assertAlmostEqual(at2 - at1, 3.52, places=2)

    def test_the_default_is_byte_identical(self):
        self.assertEqual(xb.ASSEMBLE_DEPTH_DEFAULT, 2)

    def test_the_flag_exists_and_defaults_to_the_constant(self):
        from sglang.srt.weg2 import launcher as lc

        ns = lc.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.xchg_bounce_depth, xb.ASSEMBLE_DEPTH_DEFAULT)
        ns1 = lc.build_parser().parse_args(
            ["--tree", "/t", "--tag", "t", "--xchg-bounce-depth", "1"])
        self.assertEqual(ns1.xchg_bounce_depth, 1)


class OneSourcePricesIt(CustomTestCase):
    """No second bookkeeping: the term, the slots and the publication agree."""

    def test_the_arm_helper_takes_the_depth_rather_than_reading_the_constant(self):
        import inspect

        from sglang.srt.weg2 import launcher as lc

        sig = inspect.signature(lc.xchg_bounce_terms_for_arm)
        self.assertIn("bounce_depth", sig.parameters)
        body = inspect.getsource(lc.xchg_bounce_terms_for_arm)
        self.assertIn("depth=int(bounce_depth)", body)
        self.assertNotIn("depth=xchg_bounce.ASSEMBLE_DEPTH_DEFAULT", body)

    def test_the_published_terms_carry_the_depth_and_round_trip(self):
        """EXECUTION SMOKE of the publication path: publish -> read back.

        P and D both go through `read_published_terms`, so if the depth does
        not survive this round trip the ranks price a different buffer than the
        ledger did -- the two-sources defect this flag exists to avoid.
        """
        for d in (1, 2):
            t = xb.bounce_terms(bytes_per_direction=29119878266, n_layers=64,
                                widest_layer_bytes=WIDEST, pairs=3, depth=d,
                                slot_bytes=134217728)
            back = xb.read_published_terms(xb.publish_terms(t))
            self.assertIsNotNone(back, f"depth={d} did not survive publication")
            self.assertEqual(int(back.depth), d)
            self.assertEqual(int(back.buffer_bytes), int(t.buffer_bytes))

    def test_the_arm_line_prints_the_FLAG_depth_not_the_constant(self):
        """THE RATCHET the train seat asked for, now assertable.

        On the earlier base this file could only assert an absence, because
        [23a]'s `bounce_depth=` field did not exist there. It does now, and the
        danger it guards is concrete: if `_bounce_prov` kept reading the
        CONSTANT while the arm is priced at depth=1, the ARM line would print
        `bounce_depth=2` for a depth=1 arm -- the field introduced to stop a
        number having three readings would itself be the lie. The merge that
        would have produced exactly that was refused rather than resolved
        mechanically.
        """


        from sglang.srt.weg2 import host_ledger as hl

        mt, ma = 126 * 1024 ** 3, 110 * 1024 ** 3
        for d in (1, 2):
            with self.subTest(depth=d):
                prov = {"depth": d,
                        "slots": xb.assemble_slots(d, comparing=True),
                        "inject_mode": "shadow"}
                _, _, lines = hl.choose(mt, ma, ring_bytes=1,
                                        ring_span1_bytes=1, ring_provenance="t",
                                        xchg_bounce_prov=prov)
                arm = [x for x in lines if "WEG2-HOST-LEDGER ARM " in x]
                self.assertTrue(arm)
                for ln in arm:
                    self.assertIn(f"bounce_depth={d} ", ln)
                    self.assertIn(f"bounce_slots={d + 1} ", ln)
                    self.assertNotIn(f"bounce_depth={3 - d} ", ln)

    def test_the_prov_block_reads_the_flag_variable(self):
        """Source pin: the printed field must come from `bounce_depth`."""
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = "\n".join(
            ln for ln in inspect.getsource(lc).splitlines()
            if not ln.strip().startswith("#"))
        i = src.index('"depth": int(bounce_depth)')
        self.assertIn("int(bounce_depth)", src[i:i + 400],
                      "assemble_slots must be given the flag, not the constant")

    def test_no_call_site_still_reads_the_bare_constant_for_its_depth(self):
        """The single-source property, asserted as an ABSENCE.

        [23a]'s `bounce_depth=` ARM-line field is NOT on this base -- the train
        has not picked 20ddcde21d yet -- so this file does not assert it. What
        it CAN assert here is the half that must hold either way: no launcher
        site derives a depth from the constant behind the flag's back. When the
        five-chain lands, `_bounce_prov["depth"]` reads `bounce_depth` and the
        printed field follows the flag for free.
        """
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc)
        self.assertNotIn("depth=xchg_bounce.ASSEMBLE_DEPTH_DEFAULT", src,
                         "a call site still prices a depth the flag cannot move")


if __name__ == "__main__":
    unittest.main()
