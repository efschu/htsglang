"""29.09.: #251d (KV stage by demand) is the default after its metal proof.

z30x2 flip boot kvdemand 09291210 (D.log _121057): '#251 WAKE-RESHARD n=2
stage=S1 tokens=393216 demand=262276 over=no' at bs2 x 128k, every other wake
S0 (demand 17k-131k), needle MATCH, no death. The rank and the launcher read the
same default, so the launcher prices the form the rank runs; an explicit 0 in
--env-d still turns it off; without a stage form (the 27B) nothing is read.
"""

import os
import unittest

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as SV  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402


class TheDemandSwitchIsOnByDefault(CustomTestCase):
    def test_env_default_is_on(self):
        self.assertIs(envs.SGLANG_WEG2_D_KV_STAGE_BY_DEMAND.default, True)

    def test_launcher_follows_the_rank_default_when_not_named(self):
        self.assertTrue(L.d_kv_stage_by_demand({}))
        self.assertTrue(L.d_kv_stage_by_demand({"SGLANG_WEG2_D_KV_STAGE_TOKENS": "262144,393216"}))

    def test_an_explicit_off_stays_off(self):
        for off in ("0", "false", "off", "no"):
            self.assertFalse(L.d_kv_stage_by_demand({L.D_KV_STAGE_BY_DEMAND_KEY: off}), off)
        self.assertTrue(L.d_kv_stage_by_demand({L.D_KV_STAGE_BY_DEMAND_KEY: "1"}))

    def test_without_a_stage_form_nothing_is_read(self):
        # the 27B (and any D without >= 2 stage tokens): no StageForm, the
        # by-demand default has no reader -- byte-identical
        with envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.override(""):
            self.assertIsNone(SV.stage_form({"SGLANG_WEG2_PHASE_GROUP": "D"}))
        with envs.SGLANG_WEG2_D_KV_STAGE_TOKENS.override("262144"):
            self.assertIsNone(SV.stage_form({"SGLANG_WEG2_PHASE_GROUP": "D"}))


if __name__ == "__main__":
    unittest.main()
