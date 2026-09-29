# SPDX-License-Identifier: Apache-2.0
"""D-RESERVE (29.09.): group D's dormant-residue reserve on Next Flash is the
maximum of D's own recent records, not the old xchg census.

rc12z30r3: the census (fnFL2x86 form) priced 1140/1538 MiB (nvml0/2) + slack
while every D record of this form since 28.09. measured 612-680 MiB on the
3080s. The reserve is P's budget's dormant_other, the front's --dc-reserve and
the W19 bolt -- ~460/860 MiB per 3080 sat idle that P's resident experts could
hold. The fnFL2x111 lesson stays: a single low sample never prices it (maximum
over the newest boots), and without records the old max(record, census) rule
is byte-identical.
"""
import ast
import inspect
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
S0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
S2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
CARDS = [
    launcher.Card(0, S0, "NVIDIA GeForce RTX 3080", 20480),
    launcher.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    launcher.Card(2, S2, "NVIDIA GeForce RTX 3080", 20480),
]
XCHG = launcher.WEIGHT_SOURCE_EXCHANGE
SLACK = 64
M = launcher.DC_RECORD_MARGIN_MIB
# z30r3: newest D record 1686/676/678 (+256 margin +64 slack), census 1942/1076/1474 (+64)
RECORD_PRICED = {BIG: 1686 + M + SLACK, S0: 676 + M + SLACK, S2: 678 + M + SLACK}
CENSUS = {BIG: 1942 + SLACK, S0: 1076 + SLACK, S2: 1474 + SLACK}
D_ROWS = [("09290450", "2026-09-29T04:55:26Z", 1690, 676, 678),
          ("09290500", "2026-09-29T05:05:22Z", 1692, 676, 678),
          ("09290349", "2026-09-29T03:53:53Z", 1662, 654, 656),
          ("09290654", "2026-09-29T06:59:51Z", 1692, 678, 680)]


def _drecs(rows=D_ROWS):
    return [{"group": "D", "boot_tag": "dkrnf" + t, "at": at, "vram_residue_form": XCHG,
             "vram_residue_mib": {BIG: b, S0: s0, S2: s2}} for t, at, b, s0, s2 in rows]


class DReserve(CustomTestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(launcher.D_RESERVE_RECORD_ENV, None)

    def tearDown(self):
        self._env.stop()

    def test_record_max_replaces_the_old_census(self):
        d_max, why = launcher.dormant_max_from_records(_drecs(), CARDS, XCHG, group="D")
        self.assertEqual(d_max, {BIG: 1692, S0: 678, S2: 680})
        self.assertIn("group-D", why)
        got, won = launcher.d_reserve_from_records_or_census(
            CARDS, RECORD_PRICED, CENSUS, d_max, SLACK, True)
        self.assertEqual(got, {BIG: 1692 + M + SLACK, S0: 678 + M + SLACK, S2: 680 + M + SLACK})
        # the 3080s give back what the census over-reserved
        self.assertEqual({u: CENSUS[u] - got[u] for u in (S0, S2)}, {S0: 142, S2: 538})
        self.assertTrue(all("record-max" in w for w in won))

    def test_without_records_the_old_rule_is_byte_identical(self):
        got, won = launcher.d_reserve_from_records_or_census(
            CARDS, RECORD_PRICED, CENSUS, None, SLACK, True)
        self.assertEqual(got, {BIG: CENSUS[BIG], S0: CENSUS[S0], S2: CENSUS[S2]})
        got, _ = launcher.d_reserve_from_records_or_census(
            CARDS, {BIG: 2400, S0: 900, S2: 900}, CENSUS, None, SLACK, True)
        self.assertEqual(got[BIG], 2400)  # fnFL2x111: a newer record above the census wins

    def test_kill_switch(self):
        os.environ[launcher.D_RESERVE_RECORD_ENV] = "0"
        d_max, _ = launcher.dormant_max_from_records(_drecs(), CARDS, XCHG, group="D")
        got, _ = launcher.d_reserve_from_records_or_census(
            CARDS, RECORD_PRICED, CENSUS, d_max, SLACK, True)
        self.assertEqual(got[S2], CENSUS[S2])

    def test_one_low_sample_does_not_price_it(self):
        rows = D_ROWS + [("09290700", "2026-09-29T07:30:00Z", 1500, 500, 500)]
        d_max, _ = launcher.dormant_max_from_records(_drecs(rows), CARDS, XCHG, group="D")
        self.assertEqual(d_max, {BIG: 1692, S0: 678, S2: 680})

    def test_p_wrapper_unchanged(self):
        recs = [dict(r, group="P") for r in _drecs()]
        a = launcher.p_dormant_from_records(recs, CARDS, XCHG)
        b = launcher.dormant_max_from_records(recs, CARDS, XCHG, group="P")
        self.assertEqual(a, b)


class Wiring(CustomTestCase):
    def test_census_loop_uses_the_selector(self):
        src = inspect.getsource(launcher.main)
        calls = [getattr(n.func, "id", None) or getattr(n.func, "attr", None)
                 for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("d_reserve_from_records_or_census"), 1)
        self.assertNotIn("_census_reserve = int(_xr[c.uuid]) + slack_mib", src)


if __name__ == "__main__":
    unittest.main()
