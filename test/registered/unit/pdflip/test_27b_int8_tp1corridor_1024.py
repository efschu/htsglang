# SPDX-License-Identifier: Apache-2.0
"""27B INT8 drei-Weg 24.10.: D's budget books the extend vote's moved line beside the measured rest.

Metal (boot_weg2_dkr27bbar1drei10240803_b9cddabda1_1024_080355, rc39a, D.log 08:16:36):
TP1 (3080, GPU-62dbbae1) budget 17384 = 20480 - carve 425 - dormant_other 592 -
awake_rest_booked 2079 (D_AWAKE_REST_BOOKED_MIB, eight boots of 28./29.09.). The width
vote kept its line 701 (EXTEND-CORRIDOR-LINE source=MEASURED-D:701) and funded
``floor((790 - 701) / 0.3091) = 287`` / ``floor((734 - 701) / 0.3091) = 105`` rows
(#794 GROUP-NARROWED 4096 -> 287 / 256). The record was measured while the vote's line
was rc12g's 300 MiB halt line (786fe038dc moved it 21.10., the NACHT2 port to the card's
measured floor 22.10.): the room the vote now keeps (701 - 300 = 401 MiB on TP1) is not
in the rest, the budget handed it to the KV pool, and the vote took it out of the chunk.

Fix (pdflip/budget_rest.py PDFLIP-BUDGET-REST-VOTE-LINE, launcher.booked_rest_vote_line_deltas):
the delta ``max(0, max(300, floor) - 300)`` is booked beside the rest where the floor
actuates. Danger direction named: a budget that does NOT book it is the pre-fix budget
(higher, the KV pool takes the vote's room); the tests below pin the booked budget and
the identity that the vote's end state is unchanged (no OOM direction).
"""

import os
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import corridor_guard as CG  # noqa: E402
from flliper.srt.pdflip import budget_rest as BR  # noqa: E402
from flliper.srt.pdflip import corridor_budget as CB  # noqa: E402
from flliper.srt.pdflip import extend_trim as ET  # noqa: E402
from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

# this boot's cards (front.log 08:05:00 PDFLIP-DC / budget D lines)
U5090 = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"  # nvml0, D TP0
U_TP1 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"  # nvml1, D TP1
U_TP2 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"  # nvml2, D TP2
DORMANT = {U5090: 1108, U_TP1: 592, U_TP2: 818}
#: PDFLIP-LAUNCH CORRIDOR-FLOOR group=D transient=767/701/700 source=MEASURED-D reserve=0
MEASURED = {U5090: 767, U_TP1: 701, U_TP2: 700}
#: the live boot's budgets (HEAD): 27808 / 17384 / 17168
HEAD_BUDGETS = [27808, 17384, 17168]
RATE = 0.3091  # D_EXTEND_CAP_PER_ROW_MIB, rate_src=start


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U5090, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=499),
        L.Card(nvml_index=1, uuid=U_TP1, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=2, uuid=U_TP2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
    ])


def _floor(uuid, source, transient):
    return CG.CorridorFloor(card_uuid=uuid, group="D", transient_mib=int(transient), reserve_mib=0,
                            source=source, provenance="test", basis=CG.BASIS_ALLOC_DELTA)


def _floors(source="MEASURED-D", values=None):
    vals = MEASURED if values is None else values
    return lambda cards, group, user_reserve_mib=None: {c.uuid: _floor(c.uuid, source, vals[c.uuid])
                                                       for c in cards}


def _d_pass(lines, floors):
    cards = _cards()
    with mock.patch.object(CB, "floors_for_cards", floors):
        kw = dict(L.booked_rest_kwargs(cards, F.PROFILE_QWEN27B, "D", lines.append))
        budgets = L.budgets_from_dc(
            cards, dict(DORMANT), lines.append, "D", user_reserve_by_card={U5090: 1800, U_TP1: 1400, U_TP2: 1400},
            charge_driver_carve=L.budget_charges_driver_carve(F.PROFILE_QWEN27B),
            driver_carve_min_total_mib=L.driver_carve_min_total_mib(F.PROFILE_QWEN27B), **kw)
    return cards, budgets, kw


class TheLiveBootsBudgetIsReproduced(CustomTestCase):
    def test_head_budgets_are_the_live_boots_without_a_measured_floor(self):
        # the record path alone (no actuating floor): the live boot's 27808 / 17384 / 17168
        lines = []
        _, budgets, _ = _d_pass(lines, _floors(source=CG.FLOOR_SOURCE_FALLBACK,
                                               values={U5090: 1024, U_TP1: 1024, U_TP2: 1024}))
        self.assertEqual(budgets, HEAD_BUDGETS)


class TheMovedVoteLineIsBookedBesideTheRest(CustomTestCase):
    def test_measured_floors_book_the_delta(self):
        """RED on HEAD (budgets 27808 / 17384 / 17168): the vote's line 767 / 701 / 700 minus
        the record's 300 is booked beside the rest -> 27336 / 16976 / 16768."""
        lines = []
        cards, budgets, kw = _d_pass(lines, _floors())
        self.assertEqual(budgets, [27336, 16976, 16768])
        # per card exactly: total - carve - dormant - rest - (line - 300), down to 8 MiB
        rest = [3191, 2079, 2067]
        for i, c in enumerate(cards):
            want = ((c.total_mib - c.reserved_mib - DORMANT[c.uuid] - rest[i] - (MEASURED[c.uuid] - 300)) // 8) * 8
            self.assertEqual(budgets[i], want, c.uuid)
        self.assertIn(BR.VOTE_LINE_MARKER, kw["booked_rest_provenance"])
        self.assertTrue(any(BR.VOTE_LINE_MARKER in ln and "ordinal=1 nvml1: delta 401 MiB" in ln for ln in lines))

    def test_the_tp1_corridor_gains_the_delta_and_the_vote_ends_where_it_did(self):
        """The vote's END state is unchanged (no OOM direction): with the delta in the post the
        vote at line 701 funds exactly what the record's regime funded at line 300; the width
        at the live tight instants (post 790 / 733.5 MiB, logged 790 / 734) goes 287 -> 1585 and
        105 -> 1402 rows."""
        _, budgets, _ = _d_pass([], _floors())
        delta = HEAD_BUDGETS[1] - budgets[1]  # 408 = 401 + the 8-MiB grain
        self.assertGreaterEqual(delta, MEASURED[U_TP1] - 300)
        for post, before in ((790.0, 287), (733.5, 105)):
            self.assertEqual(ET.rows_cap(post, RATE, 1, 701), before)
            after = ET.rows_cap(post + (MEASURED[U_TP1] - 300), RATE, 1, 701)
            self.assertEqual(after, ET.rows_cap(post, RATE, 1, 300))
            # the chunk's priced end free stays >= the vote's line: the same end as before
            self.assertGreaterEqual(post + (MEASURED[U_TP1] - 300) - RATE * after, 701)
            self.assertGreaterEqual(after, 1400)


class NoDeltaWhereTheFloorDoesNotActuateOrTheRegimeIsUnknown(CustomTestCase):
    def test_an_unmeasured_floor_never_cuts_kv(self):
        lines = []
        _, budgets, kw = _d_pass(lines, _floors(source=CG.FLOOR_SOURCE_FALLBACK,
                                               values={U5090: 1024, U_TP1: 1024, U_TP2: 1024}))
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertNotIn(BR.VOTE_LINE_MARKER, kw["booked_rest_provenance"])
        self.assertTrue(any("does not actuate" in ln for ln in lines))

    def test_a_record_of_unknown_boots_books_nothing_and_says_so(self):
        self.assertIsNone(BR.record_vote_line("D_AWAKE_REST_BOOKED_MIB", ("someotherboot",)))
        self.assertEqual(BR.record_vote_line("D_AWAKE_REST_BOOKED_MIB", BR.BOOTS_0929), 300)
        lines = []
        with mock.patch.object(BR, "RECORD_VOTE_LINE_MIB", {}):
            _, budgets, _ = _d_pass(lines, _floors())
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertTrue(any(BR.VOTE_LINE_MARKER in ln and "unknown" in ln for ln in lines))

    def test_the_registry_record_carries_the_named_boots(self):
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.record_name("D")]
        self.assertEqual(tuple(rec.boots), BR.BOOTS_0929)

    def test_pure_deltas(self):
        self.assertEqual(BR.vote_line_deltas([3191, None, 2067], [767, 701, 700.2], [True, True, True], 300),
                         [467, 0, 401])
        self.assertEqual(BR.vote_line_deltas([1, 1], [200, 701], [True, False], 300), [0, 0])
        self.assertEqual(BR.vote_line_deltas([1], [701], [True], None), [0])
        with self.assertRaises(ValueError):
            BR.vote_line_deltas([1, 2], [701], [True], 300)


class TheTorchCapStaysWhereItWas(CustomTestCase):
    def test_cap_budget_plus_rest_is_unchanged(self):
        # PDFLIP-ALLOC-OVERHANG: cap = budget + rest - OTHER; the delta leaves the budget and
        # enters the rest, so torch may still grow into the chunk's room
        lines = []
        _, budgets, kw = _d_pass(lines, _floors())
        _, head, head_kw = _d_pass([], _floors(source=CG.FLOOR_SOURCE_FALLBACK,
                                               values={U5090: 1024, U_TP1: 1024, U_TP2: 1024}))
        for b, r, hb, hr in zip(budgets, kw["booked_rest_mib"], head, head_kw["booked_rest_mib"]):
            self.assertLessEqual(abs((b + r) - (hb + hr)), 8)


if __name__ == "__main__":
    import unittest

    unittest.main()
