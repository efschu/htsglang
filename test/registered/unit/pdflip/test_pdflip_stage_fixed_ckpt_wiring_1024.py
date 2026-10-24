# SPDX-License-Identifier: Apache-2.0
"""PPCUT-INT4AR 24.10.: the launcher wiring of the stage-fixed checkpoint move.

The arithmetic is tested hermetically in
``test/registered/unit/planner/test_pp_cut_stage_fixed_ckpt_1024.py``.  This
file pins the three launcher facts it rests on:

* the qwen27b record names its reference checkpoint's non-layer bytes
  (Qwen3.8-27B-INT8-gdncov-vocabembed: I8 embedding 1212.97, BF16 lm_head
  2425.0 MiB) -- a profile without an entry keeps its record unchanged;
* an operator-given ``--pp-cut-stage-fixed-mib`` is recorded as given and is
  therefore NOT moved (it is the operator's measurement of this checkpoint);
* ``solve_p_cut`` moves the vector BEFORE the pool model is built from it, so
  every candidate is priced on the moved post, not only a log line.

Hermetic: no GPU, no NVML, no checkpoint, no launcher run.
"""

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import launcher as L
from flliper.test.test_utils import CustomTestCase


class TheReferenceTable(CustomTestCase):
    def test_qwen27b_reference_is_the_record_checkpoint(self):
        label, emb, head = L.P_PP_STAGE_FIXED_REF_NONLAYER_MIB["qwen27b"]
        self.assertIn("vocabembed", label)
        self.assertAlmostEqual(emb, 1212.97, places=2)
        self.assertAlmostEqual(head, 2425.0, places=2)

    def test_nextflash_has_no_entry(self):
        self.assertNotIn(L.PROFILE_NEXTFLASH, L.P_PP_STAGE_FIXED_REF_NONLAYER_MIB)


class TheGivenFlag(CustomTestCase):
    def _ns(self, words):
        ns = L.build_parser().parse_args(["--tree", "/", "--tag", "x"] + list(words))
        ns.profile = L.PROFILE_QWEN27B
        L.apply_profile_arg_defaults(ns, ["--tree", "/", "--tag", "x"] + list(words))
        return ns

    def test_unset_flag_is_not_given(self):
        self.assertFalse(self._ns([]).pp_cut_stage_fixed_given)

    def test_stated_flag_is_given(self):
        ns = self._ns(["--pp-cut-stage-fixed-mib", "2700,5000"])
        self.assertTrue(ns.pp_cut_stage_fixed_given)
        self.assertEqual(ns.pp_cut_stage_fixed_mib, "2700,5000")


class TheOrderInSolvePCut(CustomTestCase):
    def test_vector_is_moved_before_the_pool_model_is_built(self):
        src = inspect.getsource(L.solve_p_cut)
        move = src.index("stage_fixed_checkpoint_delta(")
        build = src.index("stage_fixed_mib=_stage_fixed")
        self.assertLess(move, build)
        self.assertIn("pp_cut_stage_fixed_given", src[:move])


if __name__ == "__main__":
    unittest.main()
