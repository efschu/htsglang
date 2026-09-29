# SPDX-License-Identifier: Apache-2.0
"""D-EXPECT (29.09.): group D's EXPECTATION budget (map pass, dry pass) charges
group P's MEASURED dormant residue as ``dormant_other`` on Next Flash, not D's
own reserve moved by the window difference.

Why (rc12z30r3, token cut, Form A): the map pass booked dormant_other
2054/1188/1586 MiB (nvml1/0/2) = ``dc_expect_d`` 2006/1140/1538 + 120 - 72,
while P's front-stamped record measured 1288/682/666 MiB and the launcher's own
reading after P's sleep 1252/646/630. The D form (S3f ownership, FR_D, scratch)
is solved in that pass and pinned into the Platztausch map; the real pass
could not raise it later, so 540-960 MiB per card stayed free all run.

DANGER DIRECTIONS this file guards:
* the 27B line is byte-identical (its dormant_other is its resident draft);
* no record, a record of another weight form or a record missing a card:
  legacy term, reason named -- never a partial dictionary;
* the measurement is priced BARE (as the real pass charges dc_p);
* the kill switch prices the legacy term;
* both expectation passes go through the one selector (AST pin).

Hermetic: no NVML, no boot, no GPU.
"""
import ast
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
S0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
S2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
CARDS = [
    SimpleNamespace(uuid=S0, name="NVIDIA GeForce RTX 3080", nvml_index=0),
    SimpleNamespace(uuid=BIG, name="NVIDIA GeForce RTX 5090", nvml_index=1),
    SimpleNamespace(uuid=S2, name="NVIDIA GeForce RTX 3080", nvml_index=2),
]
# rc12z30r3 (boot ...bz3bar1dauer09290548): D's census-corrected reserve and
# P's record stamped at its first sleep (05:53:05Z).
DC_EXPECT_D = {BIG: 2006, S0: 1140, S2: 1538}
P_MEASURED = {BIG: 1288, S0: 682, S2: 666}
LEGACY = {BIG: 2054, S0: 1188, S2: 1586}
XCHG = launcher.WEIGHT_SOURCE_EXCHANGE
NF = "nextflash"


def _prec(form=XCHG, vals=None):
    return {"group": "P", "boot_tag": "dkrnfh91dprsavisadoptstcutvsyncodxbz3bar1dauer09290548",
            "at": "2026-09-29T05:53:05Z", "vram_residue_form": form,
            "vram_residue_mib": dict(P_MEASURED if vals is None else vals)}


class DExpectDormantOther(CustomTestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(launcher.P_DORMANT_EXPECT_ENV, None)

    def tearDown(self):
        self._env.stop()

    def test_nf_charges_p_measured_residue_bare(self):
        got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, _prec(), XCHG, NF)
        self.assertEqual(got, P_MEASURED)
        self.assertIn("group-P record", why)
        # the z30r3 gap this frees, per card
        self.assertEqual({u: LEGACY[u] - got[u] for u in got}, {BIG: 766, S0: 506, S2: 920})

    def test_27b_is_byte_identical(self):
        for prof in (None, "qwen27b"):
            got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, _prec(), XCHG, prof)
            self.assertEqual(got, LEGACY)
            self.assertIn("legacy", why)

    def test_no_record_is_named_legacy(self):
        got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, None, XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn("UNMEASURED", why)

    def test_foreign_form_is_legacy(self):
        got, why = launcher.d_expect_dormant_other(
            CARDS, DC_EXPECT_D, _prec(form="serving"), XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn("UNMEASURED", why)

    def test_missing_card_is_legacy_not_partial(self):
        vals = dict(P_MEASURED)
        vals.pop(S2)
        got, why = launcher.d_expect_dormant_other(
            CARDS, DC_EXPECT_D, _prec(vals=vals), XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn(S2, why)

    def test_kill_switch(self):
        os.environ[launcher.P_DORMANT_EXPECT_ENV] = "0"
        got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, _prec(), XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn("=0", why)

    def test_p_dormant_from_record_unmeasured(self):
        self.assertEqual(launcher.p_dormant_from_record(None, CARDS, XCHG)[0], None)
        self.assertEqual(launcher.p_dormant_from_record(_prec(), CARDS, XCHG)[0], P_MEASURED)


class ExpectationPassesUseTheSelector(CustomTestCase):
    """Both expectation passes of ``main`` price dormant_other through the one
    selector; the legacy expression is gone from them."""

    def test_wiring(self):
        src = inspect.getsource(launcher.main)
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "d_expect_dormant_other"]
        # map pass, dry pass, and the early D start (BOOTZEIT 3) -- one selector
        self.assertEqual(len(calls), 3)
        for label in ('"D(Karte, Erwartung)"', '"D(dry, expectation)"'):
            line = next(l for l in src.splitlines() if "budgets_from_dc(" in l and label in l)
            self.assertNotIn("P_WINDOWS_MIB - D_WINDOWS_MIB", line)


if __name__ == "__main__":
    unittest.main()
