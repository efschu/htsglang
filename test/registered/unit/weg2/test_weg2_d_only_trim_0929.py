"""29.09.: the --d-only pass writes EXTEND-TRIM and EXTEND-STUECKELUNG like the
flip boot's D pass, and D_EXTEND_GROWTH_MIB carries the 3080 ranks.

kvstage-donly (29.09. 11:02Z, 424346f693): the map pass wrote
SGLANG_WEG2_EXTEND_TRIM_MIB=3071,1724,1725, restored env_d, and the d-only pass
priced its budgets without card terms -- no ledger, no trim: 0x
WEG2-EXTEND-CACHE-TRIM in the D.log, allocator cache 3.4-3.8 GiB per rank. The
z30x2 flip boot of the same image (…_113056) trimmed 125x, but the 3080 ranks
still dropped to card_free 16/78 MiB with 4 alloc retries: their growth record
was null, the threshold floor + 1024 sat under the growth 2176/1728.
"""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L  # noqa: E402


def _rc12c_test_module():
    p = Path(__file__).with_name("test_weg2_d_awake_rest_rc12c.py")
    spec = importlib.util.spec_from_file_location("_rc12c_terms_donly", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_d_only(profile):
    T = _rc12c_test_module()
    seen = {}

    def _solve(ns, cards, budgets, log, label, **kw):
        seen["label"] = label
        seen["card_terms"] = kw.get("card_terms")
        seen["budgets"] = list(budgets)

    ns = SimpleNamespace(profile=profile, corridor_budget_sample=None)
    with mock.patch.object(L.corridor_budget, "floors_for_cards", T._floors), \
            mock.patch.object(L, "log_d_rank_vram_solve", _solve):
        budgets = L.d_only_solve(ns, T._cards(), dict(T.DORMANT), [].append,
                                 user_reserve_by_card={}, p_split=None, chunk_layers=None)
    return T, seen, budgets


class TheDOnlyPassHandsTheLedgerItsTerms(CustomTestCase):
    def test_nf_d_only_pass_gets_card_terms(self):
        _T, seen, budgets = _run_d_only("nextflash")
        self.assertEqual(seen["label"], "D(d-only, expectation)")
        self.assertIsNotNone(seen["card_terms"])
        self.assertEqual(len(seen["card_terms"]), 3)
        self.assertEqual(seen["budgets"], list(budgets))

    def test_nf_d_only_pass_prices_like_the_map_pass(self):
        # same terms as the map pass -> same budgets as the rc12c D pass
        T, _seen, budgets = _run_d_only("nextflash")
        _c, ref = T._d_pass("nextflash", [], terms=[])
        self.assertEqual(list(budgets), list(ref))

    def test_nf_d_only_ledger_writes_the_trim(self):
        T, seen, budgets = _run_d_only("nextflash")
        led = L.d_card_ledger(seen["card_terms"], budgets, "D(d-only, expectation)")
        fits = T._fits(budgets, (91, 48, 48))
        growth, _src = L.d_extend_growth_record("nextflash")
        self.assertEqual(L.d_extend_trim_env(led, fits, growth), "3201,2876,2429")

    def test_27b_d_only_stays_without_terms(self):
        # qwen27b has no D_AWAKE_REST record: no terms, no ledger, byte-identical
        _T, seen, _b = _run_d_only("qwen27b")
        self.assertIsNone(seen["card_terms"])


class TheGrowthRecordCarriesTheWorkers(CustomTestCase):
    def test_nf_growth_every_rank(self):
        growth, boots = L.d_extend_growth_record("nextflash")
        self.assertEqual(growth, [2434.0, 2176.0, 1728.0])
        self.assertIn("dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutz30x2bar1dauer09291130", boots)

    def test_27b_growth_untouched(self):
        self.assertEqual(L.d_extend_growth_record("qwen27b"), (None, ""))


if __name__ == "__main__":
    unittest.main()
