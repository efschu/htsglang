# SPDX-License-Identifier: Apache-2.0
"""#1374 F1b: the tag bound, validated against boot weg2xsn30's own manifests.

The operator's wiring rule was: derive it, validate per rank against the
manifests, and if any rank deviates by more than 5 % report instead of wiring.
This file IS that validation, kept as a test so the derivation cannot drift
away from the boot that grounds it.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import checkpoint_census as cc
from sglang.srt.weg2 import launcher as lc
from sglang.test.test_utils import CustomTestCase

MODEL = ("/spinning/llm_stuff/club-3090/models-cache/"
         "Qwen3.8-27B-INT8-gdncov-vocabembed")
MIB = 1 << 20
#: weg2xsn30, P group, per rank: the tag manifest's own MiB figures.
#:   PP0 weights_0 2988 / _1 2916 / _2 2916 / _3 2916 / _4 2560
#:   PP1 weights_4 486 / _5 2856 / _6 1458
#:   PP2 weights_6 1588 / _7 2856
#: The PARAM SUM for PP0's weights_0 is 2908.1 MiB over 114 tensors; the 2988
#: on the pause line is the saver ARENA footprint. The buffer holds what the
#: deposit moves, so the param sum is what this is graded against.
XSN30_PARAM_SUM_MIB = 2908.1
XSN30_TAG_LINE_MIB = 2988
CHUNK_LAYERS = 8          # 64 layers / 8 chunks, measured: weights_0 = layers 0..7


@unittest.skipUnless(os.path.isdir(MODEL), "the shipped checkpoint is rig-only")
class TheTagBoundMatchesTheBootThatGroundsIt(CustomTestCase):
    def test_the_derivation_is_within_five_percent_of_the_manifest(self):
        got = cc.max_tag_bytes_from_census(MODEL, CHUNK_LAYERS) / MIB
        dev = 100.0 * (got - XSN30_PARAM_SUM_MIB) / XSN30_PARAM_SUM_MIB
        self.assertLess(abs(dev), 5.0,
                        f"derived {got:.1f} MiB against the manifest's "
                        f"{XSN30_PARAM_SUM_MIB} MiB is {dev:+.1f}% -- the "
                        f"operator rule is: report, do not wire")

    def test_the_mtp_tree_is_what_closed_the_first_overshoot(self):
        """+9.2 % on stage 0 until `mtp.layers.<k>` was excluded BY NAME."""
        with_mtp = cc.layer_census_from_headers(MODEL)
        without = cc.layer_census_from_headers(
            MODEL, exclude_prefixes=cc.MTP_TREE_PREFIXES)
        l0_with = dict(with_mtp.layer_bytes)[0] / MIB
        l0_without = dict(without.layer_bytes)[0] / MIB
        self.assertAlmostEqual(l0_with - l0_without, 355.1, delta=1.0)
        self.assertAlmostEqual(l0_without, 366.2, delta=1.0)

    def test_the_unlayered_total_is_group_wide_and_must_not_set_the_bound(self):
        """4516 MiB group-wide against a 1213 MiB per-rank share (xsn30's own
        plan lines: embed 404.375 + lm_head 808.750), which is BELOW the
        window. `max(window, unlayered)` overshot by +51 %."""
        c = cc.layer_census_from_headers(
            MODEL, exclude_prefixes=cc.MTP_TREE_PREFIXES)
        self.assertGreater(c.unlayered_bytes / MIB, XSN30_PARAM_SUM_MIB)
        self.assertEqual(cc.max_tag_bytes_from_census(MODEL, CHUNK_LAYERS),
                         max(sum(b for i, b in c.layer_bytes
                                 if k * CHUNK_LAYERS <= i < (k + 1) * CHUNK_LAYERS)
                             for k in range(8)),
                         "the bound is the widest window and nothing else")


class TheProducerRefusesRatherThanDefaulting(CustomTestCase):
    def test_an_unpublished_chunking_refuses_by_name(self):
        saved = lc._WEIGHT_CHUNK_LAYERS
        try:
            lc._WEIGHT_CHUNK_LAYERS = None
            with self.assertRaises(
                    cc.Weg2XchgWidestLayerUnreadable) as caught:
                lc.xchg_max_tag_bytes(MODEL)
            self.assertIn("weg2xsn30", str(caught.exception))
        finally:
            lc._WEIGHT_CHUNK_LAYERS = saved

    def test_publishing_it_makes_the_producer_answer(self):
        saved = lc._WEIGHT_CHUNK_LAYERS
        try:
            lc.publish_weight_chunk_layers(CHUNK_LAYERS)
            self.assertEqual(lc._WEIGHT_CHUNK_LAYERS, CHUNK_LAYERS)
        finally:
            lc._WEIGHT_CHUNK_LAYERS = saved


if __name__ == "__main__":
    unittest.main()
