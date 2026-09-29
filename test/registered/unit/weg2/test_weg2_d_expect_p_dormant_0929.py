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

DANGER DIRECTIONS this file guards (27B review 29.09.):
* the 27B line is byte-identical AND never reads the sidecar -- a broken or
  foreign sidecar cannot touch a 27B boot (the reader sits behind the gate);
* no record, a record of another weight form or a record missing a card:
  skipped by name, never a partial dictionary; none left: legacy, UNMEASURED;
* the maximum over the newest N boots, not the newest one -- a lucky low
  sample must not size the D form too large;
* P holding MORE after P's sleep than the records say: the real pass prices
  the lower budget, the pinned form is checked against it and refused by name
  (W122), and the check line names the card and the difference;
* the measurement is priced BARE (as the real pass charges dc_p);
* the kill switch prices the legacy term;
* both expectation passes go through the one selector (AST pin).

Hermetic: no NVML, no boot, no GPU.
"""
import ast
import inspect
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
S0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
S2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
CARDS = [
    launcher.Card(0, S0, "NVIDIA GeForce RTX 3080", 20480),
    launcher.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    launcher.Card(2, S2, "NVIDIA GeForce RTX 3080", 20480),
]
# rc12z30r3 (boot ...bz3bar1dauer09290548): D's census-corrected reserve and
# P's record stamped at its first sleep (05:53:05Z).
DC_EXPECT_D = {BIG: 2006, S0: 1140, S2: 1538}
P_MEASURED = {BIG: 1288, S0: 682, S2: 666}
LEGACY = {BIG: 2054, S0: 1188, S2: 1586}
XCHG = launcher.WEIGHT_SOURCE_EXCHANGE
NF = "nextflash"
# the sidecar's newest group-P records of this checkpoint and form, oldest
# first (29.09.): the 0450/0500 boots carry the nvml2 outlier 924.
SERIES = [
    ("09290109", "2026-09-29T01:14:57Z", 1286, 680, 666),
    ("09290122", "2026-09-29T01:29:01Z", 1308, 682, 668),
    ("09290232", "2026-09-29T02:39:21Z", 1342, 736, 774),
    ("09290349", "2026-09-29T03:54:22Z", 1288, 684, 668),
    ("09290450", "2026-09-29T04:55:51Z", 1304, 684, 924),
    ("09290500", "2026-09-29T05:05:45Z", 1304, 684, 924),
    ("09290528", "2026-09-29T05:32:57Z", 1286, 680, 666),
    ("09290548", "2026-09-29T05:53:05Z", 1288, 682, 666),
    ("09290654", "2026-09-29T07:00:12Z", 1330, 702, 690),
]


def _prec(tag="09290548", at="2026-09-29T05:53:05Z", form=XCHG, vals=None):
    return {"group": "P", "boot_tag": "dkrnf" + tag, "at": at, "vram_residue_form": form,
            "vram_residue_mib": dict(P_MEASURED if vals is None else vals)}


def _series(rows=SERIES):
    return [_prec(t, at, vals={BIG: b, S0: s0, S2: s2}) for t, at, b, s0, s2 in rows]


class DExpectDormantOther(CustomTestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(launcher.P_DORMANT_EXPECT_ENV, None)

    def tearDown(self):
        self._env.stop()

    def test_nf_charges_p_measured_residue_bare(self):
        got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, [_prec()], XCHG, NF)
        self.assertEqual(got, P_MEASURED)
        self.assertIn("MAX over the newest 1", why)
        # the z30r3 gap this frees, per card
        self.assertEqual({u: LEGACY[u] - got[u] for u in got}, {BIG: 766, S0: 506, S2: 920})

    def test_27b_is_byte_identical(self):
        for prof in (None, "qwen27b"):
            got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, [_prec()], XCHG, prof)
            self.assertEqual(got, LEGACY)
            self.assertIn("legacy", why)

    def test_no_record_is_named_legacy(self):
        for recs in (None, []):
            got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, recs, XCHG, NF)
            self.assertEqual(got, LEGACY)
            self.assertIn("UNMEASURED", why)

    def test_foreign_form_is_legacy(self):
        got, why = launcher.d_expect_dormant_other(
            CARDS, DC_EXPECT_D, [_prec(form="serving")], XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn("UNMEASURED", why)

    def test_missing_card_is_skipped_not_partial(self):
        vals = dict(P_MEASURED)
        vals.pop(S2)
        got, why = launcher.d_expect_dormant_other(
            CARDS, DC_EXPECT_D, [_prec(vals=vals)], XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn("nvml2", why)
        # a complete older record behind it is priced alone, the partial one named
        got, why = launcher.d_expect_dormant_other(
            CARDS, DC_EXPECT_D,
            [_prec(tag="old", at="2026-09-29T01:00:00Z"),
             _prec(tag="new", at="2026-09-29T06:00:00Z", vals=vals)], XCHG, NF)
        self.assertEqual(got, P_MEASURED)
        self.assertIn("dkrnfold", why)

    def test_kill_switch(self):
        os.environ[launcher.P_DORMANT_EXPECT_ENV] = "0"
        got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, [_prec()], XCHG, NF)
        self.assertEqual(got, LEGACY)
        self.assertIn("=0", why)


class MaximumOverTheNewestBoots(CustomTestCase):
    """27B review (3): a lucky low newest sample must not size the D form."""

    def test_the_outlier_inside_the_window_is_priced(self):
        got, why = launcher.p_dormant_from_records(_series(), CARDS, XCHG)
        # newest 8 boots = 09290122 .. 09290654; nvml2 924 from 09290450/0500
        self.assertEqual(got, {BIG: 1342, S0: 736, S2: 924})
        self.assertIn(f"newest 8 of N={launcher.P_DORMANT_EXPECT_N}", why)
        self.assertIn("nvml2 924 (min 666)", why)
        self.assertIn("dkrnf09290654", why)
        self.assertNotIn("dkrnf09290109", why)
        # the newest record alone would have priced 690 on nvml2: 234 MiB short
        self.assertEqual(launcher.p_dormant_from_records(_series()[-1:], CARDS, XCHG)[0][S2], 690)

    def test_the_window_ends_at_n(self):
        old_high = [("09280000", "2026-09-28T00:00:00Z", 1850, 1130, 1120)]
        got, _ = launcher.p_dormant_from_records(_series(old_high + SERIES), CARDS, XCHG)
        self.assertEqual(got, {BIG: 1342, S0: 736, S2: 924})

    def test_one_boot_counts_once(self):
        recs = _series(SERIES[-1:]) * 9 + _series(SERIES[:1])
        got, why = launcher.p_dormant_from_records(recs, CARDS, XCHG)
        self.assertIn("newest 2 of", why)
        self.assertEqual(got, {BIG: 1330, S0: 702, S2: 690})


class TheReaderSitsBehindTheGate(CustomTestCase):
    """27B review (1): the 27B path never reads the sidecar."""

    def test_27b_does_not_read(self):
        with mock.patch.object(host_ledger, "read_measured_records",
                               side_effect=AssertionError("27B read the sidecar")):
            for prof in (None, "qwen27b"):
                self.assertIsNone(launcher.p_dormant_records(prof, lambda e: True))
            # no calibration identity: no read either
            self.assertIsNone(launcher.p_dormant_records(NF, None))

    def test_broken_sidecar_on_27b_is_byte_identical_and_on_nf_unmeasured(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "weg2_measured_record.json")
            with open(bad, "w") as f:
                f.write("{not json")
            with mock.patch.object(launcher, "measured_record_path", return_value=bad):
                recs_27b = launcher.p_dormant_records("qwen27b", lambda e: True)
                got, _ = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, recs_27b, XCHG,
                                                         "qwen27b")
                self.assertEqual(got, LEGACY)
                recs_nf = launcher.p_dormant_records(NF, lambda e: True)
                self.assertEqual(recs_nf, [])
                got, why = launcher.d_expect_dormant_other(CARDS, DC_EXPECT_D, recs_nf, XCHG, NF)
                self.assertEqual(got, LEGACY)
                self.assertIn("UNMEASURED", why)

    def test_nf_reads_every_p_record_of_its_identity(self):
        import tempfile

        rows = _series() + [dict(_prec(tag="d"), group="D")]
        for r in rows:
            r["rss_shmem_gib"] = 1.0
        rows.append(dict(_prec(tag="foreign"), rss_shmem_gib=1.0, model_digest="other"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weg2_measured_record.json")
            with open(path, "w") as f:
                json.dump({"samples": rows}, f)
            with mock.patch.object(launcher, "measured_record_path", return_value=path):
                recs = launcher.p_dormant_records(
                    NF, lambda e: e.get("model_digest", "") != "other")
        self.assertEqual([r["boot_tag"] for r in recs], [r["boot_tag"] for r in _series()])


class PHoldsMoreThanTheRecords(CustomTestCase):
    """27B review (2): the record UNDER P's real residue. The real pass prices
    the launcher's own reading after P's sleep BARE; the pinned form is checked
    against the lower budget and refused by name -- never started into an OOM,
    never a silent overdraw."""

    REAL = {BIG: 1288, S0: 682, S2: 1012}  # 09281447: nvml2 1012 after serving

    def test_the_real_budget_is_lower_by_exactly_the_difference(self):
        lines = []
        expect = launcher.budgets_from_dc(CARDS, P_MEASURED, lines.append, "D")
        real = launcher.budgets_from_dc(CARDS, self.REAL, lines.append, "D")
        gap = [e - r for e, r in zip(expect, real)]
        # bare: the whole difference, up to the budget's own 16 MiB rounding
        self.assertEqual(gap[:2], [0, 0])
        self.assertTrue(346 - 16 <= gap[2] <= 346, gap)
        check = launcher.d_expect_check_lines(CARDS, P_MEASURED, self.REAL)
        self.assertEqual(len(check), 1)
        self.assertIn("nvml2 P real 1012 vs expectation 666 MiB (UNDER by 346)", check[0])
        self.assertIn("nvml1 P real 1288 vs expectation 1288 MiB (ok, 0 spare)", check[0])
        self.assertEqual(launcher.d_expect_check_lines(CARDS, None, self.REAL), [])

    def test_the_pinned_form_is_refused_by_name(self):
        from sglang.srt.planner import expert_residency as er
        from sglang.srt.weg2 import draft_post as dp

        lines = []
        expect = launcher.budgets_from_dc(CARDS, P_MEASURED, lines.append, "D")
        real = launcher.budgets_from_dc(CARDS, self.REAL, lines.append, "D")
        ns = SimpleNamespace(
            extra_d="--rank-moe-ratio 183,137,168 --rank-moe-resident-fraction 0.06,0.44,0.365 "
                    "--rank-tp-ratio 1,0,0",
            extra_p="--rank-moe-resident-fraction 0.26,0.45,0.39",
            env_d="SGLANG_MOE_SCRATCH_SLOTS=91,48,48;SGLANG_MOE_RESIDENT_EXPERT_FRACTION="
                  "0.06,0.44,0.365",
            pp_cut_expert_device_fraction="0.26,0.45,0.39", tag="t", model="m",
            d_kv_token_cut="owned")
        launcher.publish_d_owner_ratio(
            ns, SimpleNamespace(solved_owner_ratio=(215, 113, 160), owner_record={}),
            "D(Karte, Erwartung)", lines.append)
        ns.extra_d = dp.replace_vector_flag(ns.extra_d, "--rank-moe-resident-fraction",
                                            (0.096, 0.504, 0.431))
        ns._d_solved_cut = (0.0, 48.0, 16.0)
        self.assertIsNotNone(launcher.pin_d_form_for_map(ns, lines.append))
        for attr in ("d_foreign_context_mib", "d_nontorch_mib", "d_reserve_mib",
                     "d_residency_reference_logs", "d_card_reference_logs", "profile"):
            setattr(ns, attr, "")
        ns.weg2_boot_form = None

        def _plan(**kw):
            # the planner's own verdict: the pinned form was solved to the edge of
            # the EXPECTATION budget, so any rank below it does not fit (W122)
            short = [(r, e - b) for r, (e, b) in enumerate(zip(expect, kw["budgets_mib"]))
                     if b < e]
            refusal = None if not short else (
                f"{er.REFUSAL_CODE} ({kw['label']}): die gefahrene Experten-Fraction passt auf "
                + str(["rang%d" % r for r, _ in short]) + " nicht ins Budget -- "
                + "; ".join(f"rang{r} {m:.0f} MiB zuviel" for r, m in short))
            return SimpleNamespace(lines=(), refusal=refusal, fits=(), solved_owner_ratio=(),
                                   solved_fractions=(), kv_token_cut=(0.0, 48.0, 16.0),
                                   overflow_waves=None)

        with mock.patch.object(er, "plan_d_residency", _plan), \
                mock.patch.object(launcher, "d_seat_table_lines", lambda *a, **k: []):
            launcher.log_d_rank_vram_solve(ns, CARDS, expect, lines.append, "D(Karte, Erwartung)")
            with self.assertRaises(launcher.Weg2LaunchRefused) as cm:
                launcher.log_d_rank_vram_solve(ns, CARDS, real, lines.append, "D")
        self.assertIn(er.REFUSAL_CODE, str(cm.exception))
        self.assertIn("['rang2']", str(cm.exception))
        self.assertIn(f"rang2 {expect[2] - real[2]} MiB zuviel", str(cm.exception))
        self.assertTrue(any("KARTE-FORM" in ln and "nicht neu" in ln for ln in lines))


class ExpectationPassesUseTheSelector(CustomTestCase):
    """Both expectation passes of ``main`` price dormant_other through the one
    selector; the legacy expression is gone from them; the records are read
    through the gated reader only; the real pass prints the check."""

    def test_wiring(self):
        src = inspect.getsource(launcher.main)
        tree = ast.parse(src)
        calls = [getattr(n.func, "id", None) or getattr(n.func, "attr", None)
                 for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("d_expect_dormant_other"), 2)
        self.assertEqual(calls.count("p_dormant_records"), 1)
        # the only direct read in main is group D's (D-RESERVE, inside the
        # NF census branch); group P's goes through the gated reader
        direct = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "attr", None) == "read_measured_records"]
        self.assertEqual([ast.literal_eval(n.args[1]) for n in direct], ["D"])
        self.assertEqual(calls.count("d_expect_check_lines"), 1)
        for label in ('"D(Karte, Erwartung)"', '"D(dry, expectation)"'):
            line = next(l for l in src.splitlines() if "budgets_from_dc(" in l and label in l)
            self.assertNotIn("P_WINDOWS_MIB - D_WINDOWS_MIB", line)


if __name__ == "__main__":
    unittest.main()
