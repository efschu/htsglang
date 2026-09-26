"""UNIFY S7/S8: the 27B mamba-anchor grid (34965fc3fa) follows the profile's
``mamba_anchor`` field: grid4096 (qwen27b) = group P anchors every 4096 tokens,
at most 4 per path -- the 27B arm's values; deepest (nextflash) = off, the NF form.
An explicit env value wins; a desk caller's explicit mapping keeps the plain read.
Hermetic: no GPU.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import mamba_ckpt_utils as M
from sglang.srt.weg2 import form as F
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KEYS = (F.FORM_ENV, "SGLANG_WEG2_GROUP", M.ANCHOR_INTERVAL_ENV, M.MAX_STATES_PER_PATH_ENV)


def _form_env(profile):
    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile != "nextflash"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return F.Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                      flip="family", vision="off", profile=profile, model="m").env_value()


class TestAnchorGridProfile(CustomTestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in _KEYS}

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _set(self, profile, group="P", **explicit):
        for k in _KEYS:
            os.environ.pop(k, None)
        if profile:
            os.environ[F.FORM_ENV] = _form_env(profile)
        os.environ["SGLANG_WEG2_GROUP"] = group
        os.environ.update({k: str(v) for k, v in explicit.items()})

    def test_rows(self):
        q = F.PROFILE_SWITCH_DEFAULTS["qwen27b"]
        n = F.PROFILE_SWITCH_DEFAULTS["nextflash"]
        self.assertEqual((q["SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL"], q["SGLANG_WEG2_MAMBA_MAX_STATES_PER_PATH"]), (4096, 4))
        self.assertEqual((n["SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL"], n["SGLANG_WEG2_MAMBA_MAX_STATES_PER_PATH"]), (0, 0))

    def test_qwen27b_group_p_takes_the_grid(self):
        self._set("qwen27b")
        self.assertEqual(M.weg2_anchor_interval(), 4096)
        self.assertEqual(M.weg2_max_states_per_path(), 4)

    def test_nextflash_and_no_form_stay_off(self):
        self._set("nextflash")
        self.assertEqual(M.weg2_anchor_interval(), 0)
        self.assertEqual(M.weg2_max_states_per_path(), -1)
        self._set(None)
        self.assertEqual(M.weg2_anchor_interval(), 0)
        self.assertEqual(M.weg2_max_states_per_path(), -1)

    def test_group_d_never(self):
        self._set("qwen27b", group="D")
        self.assertEqual(M.weg2_anchor_interval(), 0)
        self.assertEqual(M.weg2_max_states_per_path(), -1)

    def test_explicit_wins_and_mapping_reads_plain(self):
        self._set("qwen27b", **{M.ANCHOR_INTERVAL_ENV: 8192, M.MAX_STATES_PER_PATH_ENV: 2})
        self.assertEqual(M.weg2_anchor_interval(), 8192)
        self.assertEqual(M.weg2_max_states_per_path(), 2)
        mapping = {F.FORM_ENV: _form_env("qwen27b"), "SGLANG_WEG2_GROUP": "P"}
        self.assertEqual(M.weg2_anchor_interval(mapping), 0)
        self.assertEqual(M.weg2_max_states_per_path(mapping), -1)


if __name__ == "__main__":
    unittest.main()
