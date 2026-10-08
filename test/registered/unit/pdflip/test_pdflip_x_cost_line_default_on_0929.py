"""29.09.: X-COST-LINE is the default after its metal proof (Leistungsschalter rule).

27B z30y 09291331 (895559fed2) front: "PDFLIP X COST-LINE RE-SOLVE X=5120 <-
X_prev=4096 X*=5878 (ok)", RECORD side=D a=190 b=0.575 n=9, side=P b=0.138
n=62 -- the re-solve from D's measured cost line ran under load (before:
"X NO-SOLVE: no r_d" in NF 09290827 and 27B 09290020). Model-neutral, so the
default goes on for both; an explicit 0 turns it off (the solo-r_D re-solve).
RED on 895559fed2: the default there is off.
"""

import os
import unittest

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.environ import envs  # noqa: E402

KEY = "FLLIPER_PDFLIP_ENABLE_X_COST_LINE"


class TheCostLineIsOnByDefault(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(KEY, None)

    def tearDown(self):
        os.environ.pop(KEY, None)
        if self._saved is not None:
            os.environ[KEY] = self._saved

    def test_unset_is_on(self):
        self.assertIs(envs.FLLIPER_PDFLIP_ENABLE_X_COST_LINE.default, True)
        self.assertTrue(envs.FLLIPER_PDFLIP_ENABLE_X_COST_LINE.get())

    def test_an_explicit_zero_turns_it_off(self):
        for off in ("0", "false"):
            os.environ[KEY] = off
            self.assertFalse(envs.FLLIPER_PDFLIP_ENABLE_X_COST_LINE.get(), off)
        os.environ[KEY] = "1"
        self.assertTrue(envs.FLLIPER_PDFLIP_ENABLE_X_COST_LINE.get())


if __name__ == "__main__":
    unittest.main()
