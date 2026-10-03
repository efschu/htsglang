"""NF PORTS 1003 (order 980 item 2): the NF extend chunk-cap rate for the ranks
its record leaves null.

Report 780 (A.4): Next Flash carries ``D_EXTEND_GROWTH_PER_ROW_MIB =
[0.5871, null, null]`` -- only TP0 (5090) votes in the rc12g width vote; TP1/TP2
(3080) read "0" = no rate = no vote, although they have LESS room (z30x2: card
free 16 / 78 MiB). Q-694b (27B line, 1de15e82ca) measured the rate per rank at
run time; here the same measurement is ported and armed ONLY on the null ranks,
with a start rate DERIVED (never copied from another rank, no rig constant):

* ``ratio``        = recorded rank's rate x D_EXTEND_GROWTH_MIB[r] / D_EXTEND_GROWTH_MIB[recorded rank]
* ``geometry``     = ``extend_trim.derived_rate_mib(config.json)`` (0.4622 on NF: UNDER the measured 0.5871)
* ``record-floor`` = the profile's highest recorded (measured) rate: no rank starts under it
                     (coordinator, report 960)
* start            = the largest of the three; the rank then votes with ``max(start, measured x 1.15)``.

The recorded rank keeps its recorded rate byte for byte (TP0: 0.5871, no
measurement). With the record complete, or without a null rank, nothing changes.
"""

import inspect
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs
from sglang.srt.weg2 import extend_trim as ET
from sglang.srt.weg2 import launcher as L

RECORD = [0.5871, None, None]
GROWTH = [2434.0, 2176.0, 1728.0]  # D_EXTEND_GROWTH_MIB of nextflash.json


def _nf_config(**over):
    t = {"hidden_size": 2560, "intermediate_size": 0, "moe_intermediate_size": 512,
         "num_experts_per_tok": 10, "num_experts": 512, "shared_expert_intermediate_size": 512,
         "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "attn_output_gate": True,
         "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 32,
         "linear_value_head_dim": 128, "torch_dtype": "bfloat16"}
    t.update(over)
    return {"text_config": t}


def _model_dir(cfg):
    d = tempfile.mkdtemp(prefix="nf_ports_model_")
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return d


FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "nf_ports_1003",
                       "nextflash_int4mixed_config.json")


def _real_nf_model_dir():
    """A model dir carrying the REAL NF INT4-mixed config.json (fixture of the profile S3 report 960)."""
    d = tempfile.mkdtemp(prefix="nf_ports_real_")
    with open(FIXTURE) as src, open(os.path.join(d, "config.json"), "w") as dst:
        dst.write(src.read())
    return d


class TheStartRateNeverFallsBelowTheMeasuredNfValue(unittest.TestCase):
    """Coordinator, report 960 (extend rate): ``derived_rate_mib`` on the NF geometry is 0.4622 MiB/row,
    21 % UNDER the measured NF record 0.5871 (reserved growth); on 27B it is conservative (0.4422 vs 0.3091).
    So no NF rank may start under the measured value: the start of a null rank is
    max(ratio, geometry, the recorded rank's own rate)."""

    MEASURED = 0.5871

    def test_the_formula_alone_is_below_the_measured_value_on_the_nf_geometry(self):
        with open(FIXTURE) as fh:
            geo = ET.derived_rate_mib(json.load(fh))
        self.assertEqual(geo, 0.4622)
        self.assertLess(geo, self.MEASURED)  # why the floor exists; the formula itself is not changed

    def test_every_nf_rank_starts_at_or_above_the_measured_record(self):
        ns = SimpleNamespace(profile="nextflash", model=_real_nf_model_dir(), env_d="")
        rates, fill = L.d_extend_rate_fill_null(ns, list(RECORD))
        self.assertEqual(fill["geometry"], 0.4622)
        for r, v in enumerate(rates):
            self.assertGreaterEqual(v, self.MEASURED, f"rank {r}")

    def test_the_emitted_env_text_never_carries_a_rate_under_it(self):
        ns = SimpleNamespace(profile="nextflash", model=_real_nf_model_dir(), env_d="")
        rates, _ = L.d_extend_rate_fill_null(ns, list(RECORD))
        for v in ET.launcher_rates(rates).split(","):
            self.assertGreaterEqual(float(v), self.MEASURED)

    def test_an_unreadable_geometry_does_not_lower_it_either(self):
        ns = SimpleNamespace(profile="nextflash", model="/nonexistent/model", env_d="")
        rates, _ = L.d_extend_rate_fill_null(ns, list(RECORD))
        for v in rates:
            self.assertGreaterEqual(v, self.MEASURED)

    def test_a_higher_derived_candidate_still_wins(self):
        rates, src = ET.fill_null_rates(RECORD, GROWTH, 0.9)
        self.assertEqual(rates[1:], [0.9, 0.9])
        self.assertEqual(src[1:], ["geometry", "geometry"])


class FillNullRates(unittest.TestCase):

    def test_ratio_of_the_recorded_rank_without_the_floor(self):
        rates, src = ET.fill_null_rates(RECORD, GROWTH, None, floor_to_record=False)
        # 0.5871 x 2176 / 2434 = 0.5249 ; 0.5871 x 1728 / 2434 = 0.4169 (the ratio alone, for the record)
        self.assertEqual(rates, [0.5871, 0.5249, 0.4169])
        self.assertEqual(src, ["record", "ratio", "ratio"])

    def test_the_record_floor_lifts_every_null_rank(self):
        rates, src = ET.fill_null_rates(RECORD, GROWTH, 0.4622)
        self.assertEqual(rates, [0.5871, 0.5871, 0.5871])
        self.assertEqual(src, ["record", "record-floor", "record-floor"])

    def test_the_larger_candidate_wins(self):
        rates, src = ET.fill_null_rates(RECORD, GROWTH, 0.9)
        self.assertEqual(rates[1:], [0.9, 0.9])
        self.assertEqual(src[1:], ["geometry", "geometry"])
        # a rank that grows MORE than the recorded one wins by its ratio
        rates, src = ET.fill_null_rates(RECORD, [2000.0, 2500.0, 1000.0], None)
        self.assertEqual(rates, [0.5871, 0.7339, 0.5871])
        self.assertEqual(src, ["record", "ratio", "record-floor"])

    def test_geometry_alone_without_a_record_floor(self):
        rates, src = ET.fill_null_rates([None, None, None], None, 0.4422)
        self.assertEqual(rates, [0.4422] * 3)
        self.assertEqual(src, ["geometry"] * 3)

    def test_nothing_derivable_stays_unarmed_and_named(self):
        rates, src = ET.fill_null_rates([None, None, None], GROWTH, None)
        self.assertEqual(rates, [None, None, None])
        self.assertEqual(src, ["none"] * 3)
        self.assertEqual(ET.launcher_rates(rates), "")

    def test_a_complete_record_is_untouched(self):
        rates, src = ET.fill_null_rates([0.5, 0.6, 0.7], GROWTH, 9.0)
        self.assertEqual(rates, [0.5, 0.6, 0.7])
        self.assertEqual(src, ["record"] * 3)

    def test_no_recorded_rank_means_no_anchor_and_no_floor(self):
        rates, src = ET.fill_null_rates([None, None, None], GROWTH, None)
        self.assertEqual(src, ["none"] * 3)


class LauncherFillAndArm(unittest.TestCase):

    def _ns(self, model="", env_d=""):
        return SimpleNamespace(profile="nextflash", model=model, env_d=env_d)

    def test_the_real_nf_record_is_null_on_the_3080_ranks(self):
        rate, _ = L.d_extend_growth_per_row_record("nextflash")
        growth, _ = L.d_extend_growth_record("nextflash")
        self.assertEqual(rate, RECORD)
        self.assertEqual(growth, GROWTH)

    def test_fill_with_the_geometry_of_the_model(self):
        d = _model_dir(_nf_config())
        rates, fill = L.d_extend_rate_fill_null(self._ns(d), list(RECORD))
        geo = ET.derived_rate_mib(_nf_config())
        self.assertIsNotNone(geo)
        self.assertEqual(rates, [0.5871, 0.5871, 0.5871])  # the synthetic geometry is under the record too
        self.assertLess(geo, 0.5871)
        self.assertEqual(fill["ranks"], [1, 2])
        self.assertEqual(fill["geometry"], geo)
        self.assertEqual(fill["sources"], ["record", "record-floor", "record-floor"])

    def test_an_unreadable_model_is_named_not_fatal(self):
        rates, fill = L.d_extend_rate_fill_null(self._ns("/nonexistent/model"), list(RECORD))
        self.assertEqual(rates, [0.5871, 0.5871, 0.5871])  # the record floor alone
        self.assertIn("FileNotFoundError", fill["geo_err"])
        self.assertEqual(fill["ranks"], [1, 2])

    def test_a_complete_record_returns_no_fill(self):
        rates, fill = L.d_extend_rate_fill_null(self._ns(), [0.5, 0.6, 0.7])
        self.assertEqual(rates, [0.5, 0.6, 0.7])
        self.assertIsNone(fill)

    def test_arm_writes_the_measure_env_on_the_filled_ranks_only(self):
        ns, lines = self._ns("/nonexistent/model"), []
        rates, fill = L.d_extend_rate_fill_null(ns, list(RECORD))
        L.d_extend_rate_measure_ranks_env(ns, lines.append, "D(test)", fill, ET.launcher_rates(rates))
        env = L.parse_group_env(ns.env_d)
        self.assertEqual(env["SGLANG_WEG2_EXTEND_RATE_MEASURE"], "1")
        self.assertEqual(env["SGLANG_WEG2_EXTEND_RATE_MEASURE_RANKS"], "1,2")
        (line,) = lines
        self.assertIn("EXTEND-RATE source=record+derived start=0.5871,0.5871,0.5871 measure=on ranks=1,2", line)
        self.assertIn("rank0=record", line)
        self.assertIn("rank1=record-floor", line)
        self.assertIn("kein Rang startet unter dem gemessenen Wert", line)

    def test_a_second_solve_pass_sees_its_own_write_not_a_user_value(self):
        ns, lines = self._ns("/nonexistent/model"), []
        rates, fill = L.d_extend_rate_fill_null(ns, list(RECORD))
        text = ET.launcher_rates(rates)
        L.d_extend_rate_measure_ranks_env(ns, lines.append, "D(1)", fill, text)
        first = ns.env_d
        L.d_extend_rate_measure_ranks_env(ns, lines.append, "D(2)", fill, text)
        self.assertEqual(ns.env_d, first)
        self.assertNotIn("Vorrang", lines[1])

    def test_a_user_value_in_env_d_wins(self):
        ns, lines = self._ns("/nonexistent/model", "SGLANG_WEG2_EXTEND_RATE_MEASURE=0"), []
        rates, fill = L.d_extend_rate_fill_null(ns, list(RECORD))
        L.d_extend_rate_measure_ranks_env(ns, lines.append, "D(test)", fill, ET.launcher_rates(rates))
        env = L.parse_group_env(ns.env_d)
        self.assertEqual(env["SGLANG_WEG2_EXTEND_RATE_MEASURE"], "0")
        self.assertNotIn("SGLANG_WEG2_EXTEND_RATE_MEASURE_RANKS", env)
        self.assertIn("measure=off", lines[0])
        self.assertIn("Vorrang", lines[0])

    def test_an_unarmed_rank_is_named(self):
        # nothing derivable: no recorded rank (no floor, no ratio) and no readable geometry
        ns, lines = self._ns("/nonexistent/model"), []
        rates, fill = L.d_extend_rate_fill_null(ns, [None, None, None])
        self.assertEqual(rates, [None, None, None])
        self.assertEqual(fill["ranks"], [])
        L.d_extend_rate_measure_ranks_env(ns, lines.append, "D(test)", fill, ET.launcher_rates(rates))
        self.assertIn("measure=off ranks=-", lines[0])
        self.assertIn("Rang 0,1,2: weder Wachstums-Record noch Geometrie", lines[0])
        self.assertNotIn("SGLANG_WEG2_EXTEND_RATE_MEASURE", ns.env_d)

    def test_the_d_rank_solve_calls_the_fill_inside_the_rate_block(self):
        src = inspect.getsource(L.log_d_rank_vram_solve)
        i = src.index("d_extend_growth_per_row_record(ns.profile)")
        j = src.index("d_extend_rate_fill_null(ns, _rate)")
        k = src.index("d_extend_rate_measure_ranks_env(ns, log, label, _fill, _rtext)")
        self.assertLess(i, j)
        self.assertLess(j, k)


class RankSideMeasurementOnlyOnTheNamedRanks(unittest.TestCase):

    def setUp(self):
        ET.reset_for_tests()
        self.addCleanup(ET.reset_for_tests)

    def _arm(self, measure, ranks):
        ET.reset_for_tests()
        ET._CACHE["measure"] = measure
        ET._CACHE["measure_ranks"] = ranks

    def test_off_is_off_for_every_rank(self):
        self._arm(False, [1, 2])
        for r in (None, 0, 1, 2):
            self.assertFalse(ET.measure_armed(r))
        self.assertEqual(ET.effective_rate(0.5871, 1), 0.5871)

    def test_ranks_narrow_the_switch(self):
        self._arm(True, [1, 2])
        self.assertFalse(ET.measure_armed(0))
        self.assertTrue(ET.measure_armed(1))
        self.assertTrue(ET.measure_armed(2))
        self.assertTrue(ET.measure_armed(None))  # unknown rank: the caller filters later
        self._arm(True, None)
        self.assertTrue(ET.measure_armed(0))

    def test_effective_rate_ratchets_only_on_a_measuring_rank(self):
        self._arm(True, [1, 2])
        ET._CACHE["measured"] = 0.60  # x 1.15 = 0.69
        self.assertEqual(ET.effective_rate(0.5249, 1), 0.69)
        self.assertEqual(ET.effective_rate(0.5871, 0), 0.5871)  # TP0 never measures
        ET._CACHE["measured"] = 0.30
        self.assertEqual(ET.effective_rate(0.5249, 1), 0.5249)  # never below the start

    def test_the_environ_names_exist(self):
        self.assertFalse(envs.SGLANG_WEG2_EXTEND_RATE_MEASURE.get())
        self.assertIsNone(envs.SGLANG_WEG2_EXTEND_RATE_MEASURE_RANKS.get())

    def test_measure_ranks_read_the_env(self):
        with envs.SGLANG_WEG2_EXTEND_RATE_MEASURE_RANKS.override("1, 2"), \
                envs.SGLANG_WEG2_EXTEND_RATE_MEASURE.override(True):
            ET.reset_for_tests()
            self.assertEqual(ET.measure_ranks(), [1, 2])
            self.assertFalse(ET.measure_armed(0))
            self.assertTrue(ET.measure_armed(2))

    def test_a_rank_0_close_is_dropped_when_only_1_2_measure(self):
        self._arm(True, [1, 2])
        ET._CACHE["pending"] = (None, 0, 0, 0)
        self.assertIsNone(ET.measure_close(object(), object(), object(), rank=0))
        self.assertIsNone(ET._CACHE["pending"])
        self.assertIsNone(ET.measured_rate())

    def test_a_target_extend_prices_the_rank_per_row(self):
        MIB = 1 << 20

        class Cuda:
            def __init__(self, stats):
                self.stats = stats

            def memory_stats(self):
                return self.stats

        class Mode:
            def is_extend(self):
                return True

            def is_target_verify(self):
                return False

        class FB:
            forward_mode = Mode()
            input_ids = SimpleNamespace(shape=(4096,))

        self._arm(True, [1, 2])
        ET._CACHE["measure"] = True
        ET._CACHE["pending"] = (None, 1000 * MIB, 900 * MIB, 5000 * MIB)
        cuda = Cuda({"allocated_bytes.all.peak": 2200 * MIB, "allocated_bytes.all.current": 900 * MIB,
                     "reserved_bytes.all.current": 7200 * MIB})
        line = ET.measure_close(SimpleNamespace(), FB(), cuda, rank=1)
        # cost = max(2200 - 900, 7200 - 5000) = 2200 MiB / 4096 rows = 0.5372 (ceil 4)
        self.assertIsNotNone(line)
        self.assertIn("rate=0.5372", line)
        self.assertEqual(ET.measured_rate(), 0.5372)


class NfVoteWouldBindFarAboveTheCardFloor(unittest.TestCase):
    """Item 3 evidence, as a test: the cap=1 class (27B y8vb, STUECKELUNG cap=1 ->
    4096 to 1) needs post <= 300 + rate x page. NF pages are 64 rows; the real NF abl
    boots (5444cf8cde 13:52, 25 boots scanned) show post >= 1738 MiB on every rank, so
    even with TP1/TP2 voting the cap stays in the thousands."""

    def test_cap_at_the_lowest_post_of_the_real_boots(self):
        # lowest card_free_after in the 13:52 abl boot: TP0 1915, TP1 1876, TP2 1738; every rank now
        # starts at 0.5871 (the record floor)
        for post, lo in ((1915, 2600), (1876, 2600), (1738, 2400)):
            self.assertGreaterEqual(ET.rows_cap(post, 0.5871, 64), lo)

    def test_the_page_is_the_floor_not_one_row(self):
        self.assertEqual(ET.rows_cap(100.0, 0.5871, 64), 64)  # starved card: one NF page, never 1 row
        self.assertEqual(ET.rows_cap(100.0, 0.3091, 1), 1)  # the 27B INT8 page-1 case of y8vb


if __name__ == "__main__":
    unittest.main()
