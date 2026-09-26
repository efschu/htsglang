# SPDX-License-Identifier: Apache-2.0
"""rc12 OOM (D TP0, 22:59:07Z): the D budget must leave room for P as it sleeps
AFTER serving, not as it slept once at boot.

Numbers from boot dkrnfh91bar1dauer09262249 (530fb713ce):
* front.log:178  budget D ordinal=0 5090: 29624 = 32607 - corridor 1171
  - dormant_other 1320 - measured_awake_overshoot 489
* P.log:3476     PP0 first sleep nvml_proc 1320 MiB (never served)
* P.log:15412    PP0 after serving nvml_proc 1802 MiB (1.76 GiB in the OOM
                 precursor line D.log:16619 "Process 477 has 1.76 GiB")
* D.log:46733    torch.OutOfMemoryError ... 50.75 MiB free, Process 477 1.69 GiB

A test that is red on 530fb713ce names the overbooking with those numbers.
"""

import os

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L  # noqa: E402

try:  # absent on 530fb713ce: the budget tests must still run there and go red
    from sglang.srt.weg2 import dormant_residue as dr
except ImportError:  # pragma: no cover
    dr = None

FRESH = {"GPU-31d7ef41": 1320, "GPU-5c648f96": 712, "GPU-62dbbae1": 696}
SERVED = {"GPU-31d7ef41": 1802, "GPU-5c648f96": 1284, "GPU-62dbbae1": 1080}
OVERSHOOT = [489, 0, 0]


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid="GPU-5c648f96", name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=1, uuid="GPU-31d7ef41", name="NVIDIA GeForce RTX 5090",
               total_mib=32607, reserved_mib=518),
        L.Card(nvml_index=2, uuid="GPU-62dbbae1", name="NVIDIA GeForce RTX 3080",
               total_mib=20480, reserved_mib=425),
    ])


def _d_budgets(profile, log=None):
    """The real D pass as the launcher runs it after P's first sleep."""
    cards = _cards()
    grow_fn = getattr(L, "served_dormant_growth", None)
    kw = {}
    if grow_fn is not None:
        g, prov = grow_fn(cards, profile)
        kw = {"dormant_growth_mib": g, "dormant_growth_provenance": prov}
    lines = [] if log is None else log
    return cards, L.budgets_from_dc(cards, dict(FRESH), lines.append, "D",
                                     overshoot_mib=OVERSHOOT, overshoot_provenance="boot weg2ls4b1",
                                     **kw)


class TheNextFlashDBudgetLeavesTheServedPResidue(CustomTestCase):
    def test_no_card_is_booked_past_what_the_sleeping_p_leaves(self):
        lines = []
        cards, budgets = _d_budgets("nextflash", lines)
        for i, c in enumerate(cards):
            corridor = int([l for l in lines if f"ordinal={i} " in l and l.startswith("budget D")][0]
                           .split("corridor ")[1].split(" ")[0])
            room = c.total_mib - corridor - SERVED[c.uuid] - OVERSHOOT[i]
            self.assertLessEqual(
                budgets[i], room,
                f"{c.name} ordinal {i}: D budget {budgets[i]} MiB, but with P asleep AFTER "
                f"serving ({SERVED[c.uuid]} MiB, booked {FRESH[c.uuid]}) only {room} MiB exist "
                f"-- rc12 D TP0 OOM 22:59:07Z")

    def test_the_budget_line_names_the_growth_as_its_own_term(self):
        lines = []
        _d_budgets("nextflash", lines)
        b0 = [l for l in lines if l.startswith("budget D") and "ordinal=0 " in l][0]
        self.assertIn("dormant_other 1320", b0)
        self.assertIn("served_dormant_growth 482", b0)


class TheQwen27bBudgetIsByteIdentical(CustomTestCase):
    def test_27b_has_no_record_and_keeps_its_budgets(self):
        grow_fn = getattr(L, "served_dormant_growth", None)
        if grow_fn is not None:
            self.assertEqual(grow_fn(_cards(), "qwen27b"), (None, ""))
        lines = []
        cards, budgets = _d_budgets("qwen27b", lines)
        plain = []
        _, ref = cards, L.budgets_from_dc(cards, dict(FRESH), plain.append, "D",
                                          overshoot_mib=OVERSHOOT, overshoot_provenance="boot weg2ls4b1")
        self.assertEqual(budgets, ref)
        self.assertEqual(lines, plain)


def _release(ts, rank, tags, mib):
    return (f"[2026-09-26 {ts} {rank}] WEG2-DC-BREAKDOWN stage=release tags={tags} "
            f"nvml_proc={mib} MiB = tms_resident 0 {{}} + torch_untagged 1\n")


class TheMeasurementReadsTheReleaseLines(CustomTestCase):
    LOG = [
        "[2026-09-26 22:50:57 PP0] WEG2-XCHG-MANIFEST-WRITE group=P rank=0 card=0 region_tag=weights\n",
        "[2026-09-26 22:50:57 PP1] WEG2-XCHG-MANIFEST-WRITE group=P rank=1 card=1 region_tag=weights\n",
        _release("22:51:54", "PP0", "['kv_cache', 'weights_0', 'weights', 'cuda_graph']", 1320),
        _release("22:51:54", "PP1", "['kv_cache', 'weights_0', 'weights', 'cuda_graph']", 712),
        _release("22:56:07", "PP0", "['kv_cache', 'cuda_graph']", 19076),
        _release("22:56:08", "PP0", "['weights_9', 'weights']", 1802),
        _release("22:56:08", "PP1", "['weights_9', 'weights']", 1094),
        _release("22:57:48", "PP0", "['weights_9', 'weights']", 1730),
        _release("22:57:48", "PP1", "['weights_9', 'weights']", 1276),
    ]

    def setUp(self):
        super().setUp()
        self.assertIsNotNone(dr, "weg2/dormant_residue.py (WEG2-DORMANT-SERVED) missing")

    def test_fresh_is_the_first_sleep_and_served_the_max_after_it(self):
        rr ={r.rank: r for r in dr.rank_residues(self.LOG)}
        self.assertEqual((rr["PP0"].card, rr["PP0"].fresh_mib, rr["PP0"].served_max_mib), (0, 1320, 1802))
        self.assertEqual(rr["PP0"].growth_mib, 482)
        self.assertEqual(rr["PP1"].growth_mib, 564)
        self.assertEqual(rr["PP0"].sleeps, 3, "the kv_cache/cuda_graph stage line does not close a sleep")

    def test_a_rank_without_a_card_is_refused(self):
        with self.assertRaises(dr.DormantResidueError):
            dr.rank_residues(self.LOG[2:])

    def test_the_nextflash_record_is_what_the_measurement_prints(self):
        from sglang.srt.weg2 import form

        self.assertEqual(list(form.profile_constant("P_DORMANT_SERVED_GROWTH_MIB", "nextflash")),
                         [482, 572, 384])


if __name__ == "__main__":
    import unittest

    unittest.main()
