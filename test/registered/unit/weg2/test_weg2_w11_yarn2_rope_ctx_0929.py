# SPDX-License-Identifier: Apache-2.0
"""W11/W11b for YaRN x2 27B: the draft's RoPE context is budgeted and accounted (heu5g3, 29.09.).

METAL (boot dkr27browauthorityyarn2nopinbar1fs09291701, z30y2, P last stage):
  x2 (524288): 'resident_mib=2302.5 ... nvml_delta_mib=2578.0' -> W11 resident 2302.5 vs budget 2072.1 (+230,
      under tol 256) but W11b unaccounted 275.5 > 256 -> REFUSED; draft 'RotaryEmbedding 524672x128 float32 256.2 MiB'.
  x1 (262144, dkr27browauthorityz30ybar1fs09291331): resident 2174.5, nvml_delta 2324.0, unaccounted 149.5.
The x2 build is the x1 build + the draft RoPE context delta TWICE (resident +128, NVML +254): the cache once
resident, once through the allocator while it is written. 48505d750a priced that delta in the P pool floor
(p_rope_context_delta_mib) but not in W11/W11b; the fix feeds the SAME function's draft share into both.
"""
import os
import tempfile
import types
import unittest

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BUDGET = 2072.1  # dflash_draft_family_bytes of Qwen3.8-27B-DFlash2-W8-lued (W11 line of both boots)
X2 = ("[2026-09-29 17:02:17 PP2] WEG2 DRAFT-KV-PRODUCER armed stage=2/3 drafter=f0c73316257fec1d layout=v1 heads=8 "
      "head_dim=128 page_bytes=2048 embed=resident mtp_mib=2046.3 embed_mib=0.0 resident_mib=2302.5 "
      "head_released_mib=0.0 head_deferred=False nvml_delta_mib=2578.0 tag_pool_inactive_mib=-1.0 "
      "outside_torch_mib=-1.0 default_pool_inactive_mib=-1.0 card_free_mib=-1.0 other_live_mib=-1.0 "
      "embed_dtype=n/a build_s=1.4\n")
X1 = X2.replace("resident_mib=2302.5", "resident_mib=2174.5").replace("nvml_delta_mib=2578.0", "nvml_delta_mib=2324.0")
TARGET = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
DRAFT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-DFlash2-W8-lued"


def _log(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".P.log", delete=False)
    f.write(text)
    f.close()
    return f.name


def _ns(ctx):
    return types.SimpleNamespace(
        extra_p=f"--context-length {ctx}", model=TARGET, spec_form="DFLASH", dflash_draft_path=DRAFT)


class W11Yarn2RopeCtxTest(CustomTestCase):
    def test_metal_x2_refused_without_the_rope_term_accepted_with_it(self):
        p = _log(X2)
        try:
            before = L.check_draft_resident(p, budget_mib=BUDGET)
            self.assertFalse(before["accounted"])            # the heu5g3 death, reproduced
            self.assertAlmostEqual(before["unaccounted_mib"], 275.5, places=1)
            fixed = L.check_draft_resident(p, budget_mib=BUDGET, rope_ctx_mib=128.0)
            self.assertTrue(fixed["resident_ok"])
            self.assertTrue(fixed["accounted"], fixed)
            self.assertAlmostEqual(fixed["unaccounted_mib"], 147.5, places=1)  # == the x1 build's 149.5 +- 2
            self.assertAlmostEqual(fixed["over_mib"], 102.4, places=1)          # == x1's over 102.4
        finally:
            os.unlink(p)

    @unittest.skipUnless(os.path.isdir(TARGET) and os.path.isdir(DRAFT), "rig checkpoints not mounted")
    def test_draft_share_is_the_pool_floor_function(self):
        self.assertAlmostEqual(L.p_rope_draft_delta_mib(_ns(524288)), 128.0, places=3)
        delta, _ = L.p_rope_context_delta_mib(_ns(524288), 3)
        self.assertAlmostEqual(delta[-1] - delta[0], L.p_rope_draft_delta_mib(_ns(524288)), places=6)
        self.assertEqual(L.p_rope_draft_delta_mib(_ns(262144)), 0.0)

    def test_x1_is_byte_identical(self):
        """Byte-equality guard: at x1 (rope 0) the result dict and the gate's log line are what they were."""
        p = _log(X1)
        try:
            a = L.check_draft_resident(p, budget_mib=BUDGET)
            b = L.check_draft_resident(p, budget_mib=BUDGET, rope_ctx_mib=0.0)
            self.assertEqual(a, b)
            self.assertNotIn("rope_ctx_mib", b)
            self.assertAlmostEqual(b["unaccounted_mib"], 149.5, places=1)
            lines = []
            with unittest.mock.patch.object(L, "spec_form_is_dflash", lambda: False):
                try:  # the non-DFLASH budget refuses this build; the line is logged before the verdict
                    L.gate_w11(p, lines.append, rope_ctx_mib=0.0)
                except L.Weg2LaunchRefused:
                    pass
            self.assertTrue(lines and lines[0].startswith("W11 DRAFT-RESIDENT"))
            self.assertNotIn("rope_ctx_mib", lines[0])
        finally:
            os.unlink(p)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
