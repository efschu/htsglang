# SPDX-License-Identifier: Apache-2.0
"""VRAM-GRUNDGESETZ 29.09. (desk/27b-no-reserve-0929): D books its MEASURED awake rest, no reserve.

The 27B D budget booked, per card, three constants above the corridor floor's
transient: ``--user-reserve-mib`` 1800/1400/1400, ``awake_overshoot`` 404 and
``D_OVERSHOOT_MIB`` 489/0/0 (07.09.). The same boots MEASURED what an awake D
holds beyond its budget line (``WEG2-VRAM-PEAK card_free_mib``): the D ranks
left 273 / 4 / 14 MiB free at their tightest instant -- the "reserve" was the
awake D's own rest, never free VRAM. The fix books that rest from a record
(weg2/budget_rest.py, ``D_AWAKE_REST_BOOKED_MIB``: max over the newest boots)
instead of floor + reserve + 404 + 489; a card the record does not price keeps
those terms, NAMED UNMEASURED; the NF budgets stay byte-identical.

Fixture lines are verbatim from the 27B evidence
(/spinning/docker-acceptance/27b/evidence, boots w109290020 bb82fbcb68,
fs09291152 4e15b21564, w109281851).
"""

import os
import unittest.mock as mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import budget_rest as BR  # noqa: E402
from sglang.srt.weg2 import form as F  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
U0 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
U2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

BUDGET_D = (
    "[{ts}] WEG2-LAUNCH budget D group=D ordinal=0 nvml_idx=1 NVIDIA GeForce RTX 5090: 27520 MiB = "
    "total 32607 - corridor 2971 (floor 2567 source=MEASURED-D reserve=1800 + awake_overshoot 404) - "
    "dormant_other 1104 - driver_carve 518 (NVML reserved) - measured_awake_overshoot 489 (boot "
    "weg2ls4b1) MiB\n"
    "[{ts}] WEG2-LAUNCH budget D group=D ordinal=1 nvml_idx=0 NVIDIA GeForce RTX 3080: 17384 MiB = "
    "total 20480 - corridor 2504 (floor 2100 source=MEASURED-D reserve=1400 + awake_overshoot 404) - "
    "dormant_other 588 MiB\n"
    "[{ts}] WEG2-LAUNCH budget D group=D ordinal=2 nvml_idx=2 NVIDIA GeForce RTX 3080: 17160 MiB = "
    "total 20480 - corridor 2505 (floor 2101 source=MEASURED-D reserve=1400 + awake_overshoot 404) - "
    "dormant_other 814 MiB\n"
)
#: the dry pass prints the same shape under another label -- never read as the real pass
DRY_D = ("[2026-09-29T00:21:07Z] WEG2-LAUNCH budget D(dry, expectation) group=D ordinal=0 nvml_idx=1 "
         "NVIDIA GeForce RTX 5090: 1 MiB = total 32607 - corridor 2971 (floor 2567) - dormant_other 1 MiB\n")


def _peak(ts, tp, phase, free, total):
    return (f"[{ts} TP{tp}] WEG2-VRAM-PEAK rank={tp} phase={phase} n=1 t0_unix_ms=1 t_unix_ms=2 "
            f"window_ms=1 peak_allocated_mib=1 peak_reserved_mib=1 start_allocated_mib=1 "
            f"transient_mib=1 allocated_mib=1 reserved_mib=1 card_free_start_mib=1 "
            f"card_free_mib={free} card_total_mib={total} alloc_retries=0 ooms=0 "
            f"alloc_retries_total=0")


#: (tag, front text, D text): the tightest line per rank of each boot, verbatim values
BOOTS = (
    ("dkr27browauthoritybar1fs09291152", DRY_D + BUDGET_D.format(ts="2026-09-29T11:53:30Z"), "\n".join([
        _peak("2026-09-29 11:55:02", 0, "round", 389, 32088),
        _peak("2026-09-29 11:55:36", 1, "round", 4, 20055),
        _peak("2026-09-29 11:55:36", 2, "round", 176, 20055),
        "[2026-09-29 11:55:37 TP2] WEG2-VRAM-PEAK rank=2 phase=idle n=0 card_free_mib=na card_total_mib=20055",
    ])),
    ("dkr27browauthoritybar1w109290020", BUDGET_D.format(ts="2026-09-29T00:22:14Z"), "\n".join([
        _peak("2026-09-29 00:26:56", 0, "chunk", 273, 32088),
        _peak("2026-09-29 00:47:46", 1, "round", 118, 20055),
        _peak("2026-09-29 00:37:10", 2, "round", 116, 20055),
        _peak("2026-09-29 00:40:43", 0, "idle", 1405, 32088),
    ])),
    ("dkr27browauthoritybar1w109281851", BUDGET_D.format(ts="2026-09-28T18:52:33Z"), "\n".join([
        _peak("2026-09-28 18:56:36", 0, "round", 303, 32088),
        _peak("2026-09-28 19:00:22", 1, "round", 22, 20055),
        _peak("2026-09-28 19:00:29", 2, "round", 14, 20055),
    ])),
)

#: the w109290020 real D pass (its launcher.log / front.log, verbatim): budgets and terms
LEGACY_D = [27520, 17384, 17160]
DORMANT_P = {U5090: 1104, U0: 588, U2: 814}
RESERVE = {U5090: 1800, U0: 1400, U2: 1400}


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U0, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid=U5090, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid=U2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
    ])


def _d_pass(profile, lines, **extra):
    cards = _cards()
    over, over_prov = L.d_overshoot_record(profile)
    kw = dict(L.booked_rest_kwargs(cards, profile, "D", lines.append))
    kw.update(extra)
    return cards, L.budgets_from_dc(
        cards, dict(DORMANT_P), lines.append, "D", overshoot_mib=over, overshoot_provenance=over_prov,
        user_reserve_by_card=dict(RESERVE), charge_driver_carve=L.budget_charges_driver_carve(profile),
        driver_carve_min_total_mib=L.driver_carve_min_total_mib(profile), **kw)


class TheRestComesFromTheBootsOwnLines(CustomTestCase):
    def test_the_record_is_the_max_over_the_boots(self):
        vals, lines = BR.record_from_boots(BOOTS, "D", 3)
        # 5090: 32088 - 273 - 27520 - 1104 (w109290020 chunk); nvml0: 20055 - 4 - 17384 - 588
        # (fs09291152); nvml2: 20055 - 14 - 17160 - 814 (w109281851)
        self.assertEqual(vals, [3191, 2079, 2067])
        self.assertTrue(any("rest 3191 = card_total 32088 - min card_free 273 (chunk)" in ln for ln in lines))

    def test_the_registry_record_is_that_measurement(self):
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.record_name("D")]
        self.assertEqual(list(rec.value), BR.record_from_boots(BOOTS, "D", 3)[0])
        for tag, _, _ in BOOTS:
            self.assertIn(tag, rec.boots)

    def test_the_dry_pass_and_na_samples_are_not_read(self):
        lines = BR.budget_lines(BOOTS[0][1], "D")
        self.assertEqual(lines[0], (1, 27520, 1104))  # the real pass, not the dry "1 MiB"
        self.assertEqual(BR.tightest(BOOTS[0][2])[2], (176, 20055, "round"))

    def test_a_card_whose_budget_does_not_bind_is_unmeasured(self):
        # 27B group P w109290020: PP1 idle free 12787 against budget 15896 -> rest < 0: the
        # P-CUT cap holds the pool below the budget, the rest cannot be priced there
        front = ("[2026-09-29T00:21:07Z] WEG2-LAUNCH budget P group=P ordinal=1 nvml_idx=0 NVIDIA GeForce "
                 "RTX 3080: 15896 MiB = total 20480 - corridor 2899 (floor 2495 source=MEASURED-P "
                 "reserve=1400 + awake_overshoot 404) - dormant_other 1678 MiB\n")
        grp = _peak("2026-09-29 00:22:12", 1, "idle", 12787, 20055).replace("TP1]", "PP1]")
        vals, lines = BR.record_from_boots([("w109290020", front, grp)], "P", 3)
        self.assertEqual(vals, [None, None, None])
        self.assertTrue(any("does not bind" in ln and "UNMEASURED" in ln for ln in lines))


class TheDBudgetBooksTheRestInsteadOfTheReserve(CustomTestCase):
    def test_the_27b_line_names_the_record_and_no_reserve(self):
        lines = []
        cards, budgets = _d_pass(F.PROFILE_QWEN27B, lines)
        blines = [ln for ln in lines if ln.startswith("budget D")]
        self.assertEqual(len(blines), 3)
        for ln in blines:
            self.assertIn("awake_rest_booked", ln)
            self.assertIn("RECORD D_AWAKE_REST_BOOKED_MIB", ln)
            self.assertNotIn("reserve=", ln)
            self.assertNotIn(" - corridor ", ln)
            self.assertNotIn("- measured_awake_overshoot", ln)
        c = cards[0]
        self.assertEqual(budgets[0], ((c.total_mib - c.reserved_mib - 1104 - 3191) // 8) * 8)

    def test_budget_total_against_the_w109290020_boot(self):
        """THE BUDGET-GESAMTTEST: the real D pass of w109290020 (budgets 27520/17384/17160 under
        reserve 1800/1400/1400 + 404 + 489) rises by what the awake D really left free -- 272 /
        0 / 8 MiB after the 8-MiB grain -- NOT by the 2693 / 1804 / 1804 MiB of constants: the
        measured rest (3191 / 2079 / 2067) consumed the rest of them."""
        lines = []
        _, budgets = _d_pass(F.PROFILE_QWEN27B, lines)
        gain = [b - a for a, b in zip(LEGACY_D, budgets)]
        self.assertEqual(budgets, [27792, 17384, 17168])
        self.assertEqual(gain, [272, 0, 8])
        # the constants that fall away, per card: reserve + 404 + D_OVERSHOOT_MIB
        dropped = [1800 + 404 + 489, 1400 + 404, 1400 + 404]
        self.assertTrue(all(g < d for g, d in zip(gain, dropped)))

    def test_switch_off_is_byte_identical(self):
        with mock.patch.dict(os.environ, {BR.ENV: "0"}):
            self.assertEqual(L.booked_rest_kwargs(_cards(), F.PROFILE_QWEN27B, "D"), {})
            a, b = [], []
            _, off = _d_pass(F.PROFILE_QWEN27B, a)
            _, ref = _d_pass(F.PROFILE_QWEN27B, b, booked_rest_mib=None)
        self.assertEqual(off, ref)
        self.assertEqual(a, b)
        self.assertTrue(any("reserve=1800" in ln for ln in a))


class AMissingRecordIsNamedNeverSilent(CustomTestCase):
    def test_a_profile_without_the_record_logs_unmeasured(self):
        lines = []
        with mock.patch.dict(os.environ, {BR.ENV: "1"}):
            kw = L.booked_rest_kwargs(_cards(), F.PROFILE_NEXTFLASH, "D", lines.append)
        self.assertEqual(kw, {})
        self.assertTrue(any(BR.SOURCE_UNMEASURED in ln and "D_AWAKE_REST_BOOKED_MIB" in ln for ln in lines))

    def test_an_unpriced_card_keeps_its_terms_by_name(self):
        lines = []
        cards, budgets = _d_pass(F.PROFILE_QWEN27B, lines, booked_rest_mib=[3191, None, None],
                                 booked_rest_provenance="D_AWAKE_REST_BOOKED_MIB test")
        _, legacy = _d_pass(F.PROFILE_QWEN27B, [], booked_rest_mib=None)
        self.assertEqual(budgets[1:], legacy[1:])  # the unpriced cards: the legacy pass, unchanged
        self.assertNotEqual(budgets[0], legacy[0])
        un = [ln for ln in lines if BR.MARKER in ln and BR.SOURCE_UNMEASURED in ln]
        self.assertEqual(len(un), 2)
        self.assertTrue(all("reserve 1400" in ln and "awake_overshoot 404" in ln for ln in un))
        b1 = [ln for ln in lines if ln.startswith("budget D") and "ordinal=1 " in ln][0]
        self.assertIn("reserve=1400", b1)  # the legacy term, charged and printed
        self.assertIn("awake_overshoot 404", b1)

    def test_a_record_of_another_card_count_is_refused(self):
        with self.assertRaises(L.Weg2LaunchRefused):
            L.budgets_from_dc(_cards(), dict(DORMANT_P), [].append, "D", booked_rest_mib=[1, 2])


class NextFlashStaysByteIdentical(CustomTestCase):
    def test_the_nf_row_is_off_and_carries_no_record(self):
        self.assertFalse(F.PROFILES[F.PROFILE_NEXTFLASH].budget_rest_from_records)
        self.assertTrue(F.PROFILES[F.PROFILE_QWEN27B].budget_rest_from_records)
        self.assertNotIn(BR.record_name("D"), F.PROFILES[F.PROFILE_NEXTFLASH].constants)
        self.assertEqual(L.booked_rest_kwargs(_cards(), F.PROFILE_NEXTFLASH, "D"), {})

    def test_the_nf_d_pass_equals_the_pass_without_the_kwargs(self):
        a, b = [], []
        _, with_kw = _d_pass(F.PROFILE_NEXTFLASH, a)
        _, plain = _d_pass(F.PROFILE_NEXTFLASH, b, booked_rest_mib=None)
        self.assertEqual(with_kw, plain)
        self.assertEqual(a, b)


if __name__ == "__main__":
    import unittest

    unittest.main()
