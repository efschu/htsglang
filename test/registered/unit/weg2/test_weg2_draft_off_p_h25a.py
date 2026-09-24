"""H25a (Nutzer-Order 24.09. 08:25Z): group P without the MTP draft head.

``SGLANG_WEG2_DRAFT_ON_P`` defaults to False. Pinned here:

1. THE ARGV (bug regression, H1 23.09.): the Next-Flash arm passes
   ``--speculative-draft-model-path`` through ``--extra-p`` for both forms;
   ``ring_table.p_carries_drafter`` reads ANY ``--speculative-*`` token as
   "P carries a drafter", so without the strip P still loaded the head.
2. THE FAMILY: no partner on P -> ``weights_draft`` leaves the family, on the
   one published value.
3. ONE RESOLUTION: flag and env never disagree silently (W127); the unset env
   overrules the arms' explicit ``--draft-kv-on-p on`` (the producer has no
   reader since fnFL2x63) and says so.
"""

import json
import os
import unittest
from unittest import mock

import pytest

try:
    from sglang.srt.environ import envs
    from sglang.srt.managers import weg2_memory_saver as ms
    from sglang.srt.weg2 import launcher as L
    from sglang.srt.weg2 import ring_table, weight_exchange
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: conftest: this module pins its own draft form per case.
H25_OWN_DRAFT_FORM = True

DRAFT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-MTP-INT4-g32-albucino"
#: arm_fnFL2.sh EXTRA_P, shape-exact: the draft path in the middle.
NF_EXTRA_P = [
    "--max-total-tokens", "262144", "--kv-cache-dtype", "fp8_e4m3", "--page-size", "64",
    "--hicache-size", "2", "--speculative-draft-model-path", DRAFT,
    "--rank-moe-resident-fraction", "0.26,0.45,0.39", "--disable-cuda-graph",
]


def _p(draft_kv_on_p):
    return L.argv_p(py="/nonexistent/python", model="/nonexistent/model",
                    budgets=[28208, 17840, 17168], s_gb=1, m_mib=600,
                    store_cfg=json.dumps({"max_size": "1"}), extra=list(NF_EXTRA_P),
                    p_max_total_tokens=1277631, draft_kv_on_p=draft_kv_on_p)


class TestOffArgvCarriesNoDrafter(unittest.TestCase):
    def test_off_with_the_nf_extra_carries_zero_speculative_tokens(self):
        argv = _p(False)
        self.assertEqual([t for t in argv if t.startswith("--speculative-")], [])
        self.assertNotIn(DRAFT, argv)
        self.assertFalse(ring_table.p_carries_drafter(argv))
        i = argv.index("--rank-moe-resident-fraction")
        self.assertEqual(argv[i + 1], "0.26,0.45,0.39")

    def test_on_keeps_the_extra_draft_path(self):
        argv = _p(True)
        self.assertEqual(argv[argv.index("--speculative-draft-model-path") + 1], DRAFT)
        self.assertTrue(ring_table.p_carries_drafter(argv))

    def test_strip_value_shapes(self):
        kept, stripped = L.strip_speculative_flags(
            ["--speculative-draft-kv-only", "--page-size", "64",
             "--speculative-num-steps=3", "--speculative-draft-model-path", "/d", "--x"])
        self.assertEqual(kept, ["--page-size", "64", "--x"])
        self.assertEqual(stripped, ["--speculative-draft-kv-only", "--speculative-num-steps=3",
                                    "--speculative-draft-model-path", "/d"])


class TestOneResolution(unittest.TestCase):
    def test_default_env_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(L.DRAFT_ON_P_ENV, None)
            self.assertFalse(envs.SGLANG_WEG2_DRAFT_ON_P.get())

    def test_explicit_contradiction_is_refused_by_name(self):
        with self.assertRaisesRegex(L.Weg2LaunchRefused, "W127"):
            L.resolve_draft_on_p("on", True, True, False)
        with self.assertRaisesRegex(L.Weg2LaunchRefused, "W127"):
            L.resolve_draft_on_p("off", True, True, True)

    def test_unset_env_overrules_the_arms_explicit_on_and_says_so(self):
        on, why = L.resolve_draft_on_p("on", True, False, False)
        self.assertFalse(on)
        self.assertIn("OVERRULES", why)

    def test_explicit_env_decides(self):
        self.assertTrue(L.resolve_draft_on_p("on", False, True, True)[0])
        self.assertTrue(L.resolve_draft_on_p("on", True, True, True)[0])
        self.assertFalse(L.resolve_draft_on_p("off", True, True, False)[0])

    def test_main_publishes_before_the_cut_and_any_family_reader(self):
        import inspect

        src = inspect.getsource(L.main)
        pub = src.index("envs.SGLANG_WEG2_DRAFT_ON_P.set(")
        self.assertLess(pub, src.index("solve_p_cut("))
        self.assertLess(pub, src.index("weights_family_tags(chunk_count)"))


class TestDraftLeavesTheFamily(unittest.TestCase):
    def _family(self, draft_on_p):
        with mock.patch.dict(os.environ, {weight_exchange.WEIGHT_SOURCE_ENV: "exchange"}):
            with envs.SGLANG_WEG2_DRAFT_ON_P.override(draft_on_p):
                return (ms.draft_tag_in_family(), ms.weights_family_tags(2),
                        ms.is_weights_family_tag("weights_draft"))

    def test_off_removes_weights_draft(self):
        member, family, pred = self._family(False)
        self.assertFalse(member)
        self.assertFalse(pred)
        self.assertNotIn("weights_draft", family)
        self.assertEqual(family[-1], "weights")

    def test_on_keeps_weights_draft_before_the_base_tag(self):
        member, family, pred = self._family(True)
        self.assertTrue(member and pred)
        self.assertEqual(family[-2:], ["weights_draft", "weights"])

    def test_the_draft_region_keeps_its_own_tag_off_the_family(self):
        # the park pauses THIS tag alone: D's draft region must stay
        # 'weights_draft' (never the base tag) when it leaves the family.
        with mock.patch.dict(os.environ, {weight_exchange.WEIGHT_SOURCE_ENV: "exchange"}):
            with envs.SGLANG_WEG2_DRAFT_ON_P.override(False):
                shape = weight_exchange.RunnerShape(is_draft_worker=True,
                                                    speculative_configured=True)
                self.assertEqual(weight_exchange.weights_region_tag_for(shape), "weights_draft")


if __name__ == "__main__":
    unittest.main()
