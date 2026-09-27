# SPDX-License-Identifier: Apache-2.0
"""rc12b OOM (D TP0, 27.09. 00:26:34Z): the D budget never took the driver carve.

Numbers from boot dkrnfh91bar1dauer09270007 (c6868a1e13):
* front.log:178  budget D ordinal=0 5090: 29144 = 32607 - corridor 1171
  - dormant_other 1320 - served_dormant_growth 482 - awake_overshoot 489
* front.log:184  corridor pass terms[total=32607 ... carve=518 ...]:
  the carve is known there, but only to predict free, never to budget
* D.log:83732    OOM: 68.75 MiB free, this process 29.19 GiB (29890 MiB),
                 Process 440 (P PP0 asleep) 1.76 GiB (1802 MiB)
nvidia-smi -q on the 5090: Reserved 519 MiB (torch capacity 32092 of 32607).

The corridor law (floor 767, measured on D) is what must stay free for D's
transients. With the carve booked, the same D (29890 = budget + 746) leaves it.
"""

import os

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L  # noqa: E402

FRESH = {"GPU-31d7ef41": 1320, "GPU-5c648f96": 712, "GPU-62dbbae1": 696}
SERVED_P_5090 = 1802
D_ABOVE_BUDGET = 29890 - 29144  # measured at the OOM, rc12b


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid="GPU-5c648f96", name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid="GPU-31d7ef41", name="NVIDIA GeForce RTX 5090",
               total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid="GPU-62dbbae1", name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
    ])


def _d_budgets(profile, lines):
    """The launcher's real D pass (the terms it passes there)."""
    cards = _cards()
    g, prov = L.served_dormant_growth(cards, profile)
    kw = {}
    charge = getattr(L, "budget_charges_driver_carve", None)
    if charge is not None:
        kw["charge_driver_carve"] = charge(profile)
    return cards, L.budgets_from_dc(cards, dict(FRESH), lines.append, "D",
                                     overshoot_mib=[489, 0, 0], overshoot_provenance="boot weg2ls4b1",
                                     dormant_growth_mib=g, dormant_growth_provenance=prov, **kw)


class TheNextFlashBudgetBooksTheDriverCarve(CustomTestCase):
    def test_the_rc12b_d_leaves_the_corridor_free(self):
        lines = []
        cards, budgets = _d_budgets("nextflash", lines)
        c = cards[0]
        b0 = [l for l in lines if l.startswith("budget D") and "ordinal=0 " in l][0]
        floor = int(b0.split("(floor ")[1].split(" ")[0])  # 767 MEASURED-D on the rig
        free = c.total_mib - c.reserved_mib - SERVED_P_5090 - (budgets[0] + D_ABOVE_BUDGET)
        self.assertGreaterEqual(
            free, floor,
            f"5090: D budget {budgets[0]} + the {D_ABOVE_BUDGET} MiB D ran above it, P asleep "
            f"{SERVED_P_5090}, driver carve {c.reserved_mib}: {free} MiB free < corridor law "
            f"{floor} -- rc12b D TP0 OOM 00:26:34Z")

    def test_the_budget_line_names_the_carve(self):
        lines = []
        _d_budgets("nextflash", lines)
        b0 = [l for l in lines if l.startswith("budget D") and "ordinal=0 " in l][0]
        self.assertIn("driver_carve 518", b0)

    def test_every_card_is_charged_its_own_carve(self):
        lines = []
        cards, with_carve = _d_budgets("nextflash", lines)
        g, prov = L.served_dormant_growth(cards, "nextflash")
        without = L.budgets_from_dc(cards, dict(FRESH), [].append, "D", overshoot_mib=[489, 0, 0],
                                    dormant_growth_mib=g, dormant_growth_provenance=prov)
        for i, c in enumerate(cards):
            self.assertLessEqual(with_carve[i], without[i] - c.reserved_mib + 8, c.name)
            self.assertGreaterEqual(with_carve[i], without[i] - c.reserved_mib - 8, c.name)


#: the D TP0 peak demand at the card edge, per served NF Dauerlauf boot (torch
#: OOM line: per-process NVML + the refused request) minus that boot's budget
NF_D_ABOVE_BUDGET = {"09270007": 29890 + 80 - 29144, "09262249": 29972 + 80 - 29624,
                     "09262302": 29819 + 158 - 29624}
EXTEND_TRANSIENT = 270  # WEG2-VRAM-PEAK phase=chunk, max transient_mib over rc12b


class TheNextFlashBooksItsOwnAwakeOvershoot(CustomTestCase):
    """H94: records only from the same checkpoint. The 489 the NF D budget
    charged was the 27B's (boot weg2ls4b1); NF measured up to 826."""

    def _nf_d(self, lines):
        cards = _cards()
        g, prov = L.served_dormant_growth(cards, "nextflash")
        over = list(L._pconst("D_OVERSHOOT_MIB", "nextflash"))
        boots = getattr(L, "_pconst_boots", lambda n, p: "boot weg2ls4b1")("D_OVERSHOOT_MIB", "nextflash")
        kw = {}
        if getattr(L, "budget_charges_driver_carve", None) is not None:
            kw["charge_driver_carve"] = L.budget_charges_driver_carve("nextflash")
        return cards, L.budgets_from_dc(cards, dict(FRESH), lines.append, "D", overshoot_mib=over,
                                        overshoot_provenance=boots, dormant_growth_mib=g,
                                        dormant_growth_provenance=prov, **kw)

    def test_the_record_is_the_nf_maximum(self):
        self.assertEqual(list(L._pconst("D_OVERSHOOT_MIB", "nextflash"))[0], max(NF_D_ABOVE_BUDGET.values()))

    def test_the_budget_line_names_the_nf_boot(self):
        lines = []
        self._nf_d(lines)
        b0 = [l for l in lines if l.startswith("budget D") and "ordinal=0 " in l][0]
        self.assertIn("measured_awake_overshoot 826 (boot dkrnfh91bar1dauer09270007", b0)

    def test_the_measured_worst_case_leaves_the_corridor_free(self):
        lines = []
        cards, budgets = self._nf_d(lines)
        c = cards[0]
        b0 = [l for l in lines if l.startswith("budget D") and "ordinal=0 " in l][0]
        floor = int(b0.split("(floor ")[1].split(" ")[0])
        d_max = budgets[0] + max(NF_D_ABOVE_BUDGET.values())
        free = c.total_mib - c.reserved_mib - SERVED_P_5090 - d_max
        self.assertGreaterEqual(free, floor, f"{b0}: worst case D {d_max} leaves {free} < law {floor}")

    def test_27b_keeps_its_own_overshoot_and_provenance(self):
        self.assertEqual(list(L._pconst("D_OVERSHOOT_MIB", "qwen27b")), [489, 0, 0])
        boots = getattr(L, "_pconst_boots", None)
        if boots is not None:
            self.assertEqual(boots("D_OVERSHOOT_MIB", "qwen27b"), "boot weg2ls4b1")


class TheQwen27bBudgetIsByteIdentical(CustomTestCase):
    def test_27b_does_not_charge_the_carve(self):
        charge = getattr(L, "budget_charges_driver_carve", None)
        if charge is not None:
            self.assertFalse(charge("qwen27b"))
        lines = []
        cards, budgets = _d_budgets("qwen27b", lines)
        plain = []
        ref = L.budgets_from_dc(cards, dict(FRESH), plain.append, "D", overshoot_mib=[489, 0, 0],
                                overshoot_provenance="boot weg2ls4b1")
        self.assertEqual(budgets, ref)
        self.assertEqual(lines, plain)


if __name__ == "__main__":
    import unittest

    unittest.main()
