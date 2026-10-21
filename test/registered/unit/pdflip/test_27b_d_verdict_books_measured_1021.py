"""27B port 21.10.: the #145 D verdict books what is measured beside torch (rc29b INT8 boot dkr27bbar1drei10210113).

Metal: the verdict printed ``rang1 nvml0 RTX 3080: total 20480 - fremd 0 - nichttorch 0 - reserve 0 = 20480 verfuegbar,
gefragt 17384 -> REST 3096 MiB [NICHT GEBUCHT: --d-foreign-context-mib, --d-nontorch-mib]``, while the budget of the same
pass had charged ``driver_carve 425 (NVML reserved) - dormant_other 588`` and the card held 1332 MiB free at idle
(PDFLIP-VRAM-PEAK rank=1 phase=idle allocated 16690 reserved 17480 card_free 1332, card_total 20055). The verdict now books
fremd = the budget's dormant term and nichttorch = the budget's carve term + the record D_TORCH_CAP_OTHER_MIB.
"""
import json
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

from flliper.srt.pdflip import launcher as L  # noqa: E402
from flliper.srt.planner import pp_cut  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_RECORDS = os.path.join(_HERE, "..", "..", "..", "..", "python", "flliper", "srt", "pdflip", "profile_records_data",
                        "qwen27b.json")

# the budget terms of the rc29b boot (front log 01:14:52Z 'budget D group=D ...') and its asked budgets
TERMS = [dict(total=32607, carve=518, dormant=1104, awake=3191),
         dict(total=20480, carve=425, dormant=588, awake=2079),
         dict(total=20480, carve=425, dormant=814, awake=2067)]
BUDGETS = [27792, 17384, 17168]
TOTALS = [32607.0, 20480.0, 20480.0]
#: measured idle card free of the same boot (PDFLIP-VRAM-PEAK phase=idle card_free_mib, TP0 / TP1 / TP2)
IDLE_FREE_TP1 = 1332


def _other_record():
    with open(_RECORDS, encoding="utf-8") as fh:
        d = json.load(fh)
    recs = d if isinstance(d, list) else d.get("records", d)
    for r in recs:
        if r.get("name") == "D_TORCH_CAP_OTHER_MIB":
            return list(r["value"])
    raise AssertionError("record D_TORCH_CAP_OTHER_MIB missing")


class DVerdictBooksMeasured(unittest.TestCase):
    def test_tp1_rest_matches_the_measured_idle_free(self):
        other = _other_record()
        self.assertEqual(other, [1537, 853, 793])
        fremd, nt, booked = L.d_verdict_measured_terms(None, None, TERMS, other, 3, "D_TORCH_CAP_OTHER_MIB")
        self.assertEqual(fremd, [1104.0, 588.0, 814.0])
        self.assertEqual(nt, [518.0 + 1537, 425.0 + 853, 425.0 + 793])
        self.assertEqual(len(booked), 2)
        v = pp_cut.d_rank_budget_verdict(budgets_mib=BUDGETS, card_total_mib=TOTALS, foreign_context_mib=fremd,
                                         nontorch_mib=nt, reserve_mib_by_rank=None, corridor_floor_mib=819.0)
        rest_tp1 = -v[1].over_mib if v[1].fits else v[1].over_mib
        rest_tp1 = abs(v[1].over_mib)
        self.assertTrue(v[1].fits)
        # booked 20480 - 588 - 425 - 853 - 17384 = 1230 MiB against 1332 measured: within the record's max-over-8-boots
        # margin (the old verdict said 3096 -- 1764 MiB more than the card held)
        self.assertEqual(round(rest_tp1), 1230)
        self.assertLessEqual(rest_tp1, IDLE_FREE_TP1)
        self.assertLess(IDLE_FREE_TP1 - rest_tp1, 200)

    def test_a_given_flag_wins_and_missing_inputs_stay_unbooked(self):
        fremd, nt, booked = L.d_verdict_measured_terms([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], TERMS, [1, 1, 1], 3, "R")
        self.assertEqual((fremd, nt, booked), ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], []))
        fremd, nt, booked = L.d_verdict_measured_terms(None, None, None, [1, 1, 1], 3, "R")
        self.assertEqual((fremd, nt, booked), (None, None, []))
        fremd, nt, booked = L.d_verdict_measured_terms(None, None, TERMS, None, 3, "R")
        self.assertEqual(fremd, [1104.0, 588.0, 814.0])
        self.assertIsNone(nt)                              # no record: nichttorch stays NICHT GEBUCHT

    def test_the_verdict_uses_the_helper(self):
        import inspect

        src = inspect.getsource(L.log_d_rank_vram_solve)
        self.assertIn("d_verdict_measured_terms(fremd, nicht_torch, card_terms, _other, n,", src)
        self.assertIn("GEBUCHT AUS MESSUNG", src)


if __name__ == "__main__":
    unittest.main()
