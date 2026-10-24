# SPDX-License-Identifier: Apache-2.0
"""27B INT8 drei-Weg 24.10.: D's budget books the extend vote's moved line beside the measured rest.

Metal (boot_weg2_dkr27bbar1drei10240803_b9cddabda1_1024_080355, rc39a, D.log 08:16:36):
TP1 (3080, GPU-62dbbae1) budget 17384 = 20480 - carve 425 - dormant_other 592 -
awake_rest_booked 2079 (D_AWAKE_REST_BOOKED_MIB, eight boots of 28./29.09.). The width
vote kept its line 701 (EXTEND-CORRIDOR-LINE source=MEASURED-D:701) and funded
``floor((790 - 701) / 0.3091) = 287`` / ``floor((733.5 - 701) / 0.3091) = 105`` rows
(#794 GROUP-NARROWED 4096 -> 287 / 256). The record was measured while the vote's line
was rc12g's 300 MiB halt line (786fe038dc moved it 21.10., the NACHT2 port to the card's
measured floor 22.10.): the booked rest UNDER-books the current line by 701 - 300 = 401
MiB on TP1, the budget handed that room to the KV pool, and the vote took it out of the chunk.

Fix (pdflip/budget_rest.py PDFLIP-VOTE-LINE-BOOK, launcher.booked_rest_vote_line_deltas):
the DERIVED delta ``ceil8(max(0, max(300, floor) - 300))`` is booked beside the rest where
the floor is measured and of this form.

Danger directions pinned here (review 24.10.):
- OOM: the torch cap ``budget + rest - OTHER`` must never RISE (A2: the delta is rounded UP
  to the budget's 8-MiB grain; an unrounded delta raises it by up to 7 MiB).
- reserve: a user reserve must never enter the delta (S1: 1800/1400/1400 would be booked as
  "line"); an unmeasured, hand-set or foreign-form floor books nothing (S3).
- The pre-fix budget (delta not booked) is the narrow-chunk state, NOT an OOM direction.
"""

import json
import os
import tempfile
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
FALLBACK = {U5090: 1024, U_TP1: 1024, U_TP2: 1024}
#: the arm's --user-reserve-mib (front:37) -- must NEVER reach the delta
RESERVE = {U5090: 1800, U_TP1: 1400, U_TP2: 1400}
#: the live boot's budgets (HEAD): 27808 / 17384 / 17168
HEAD_BUDGETS = [27808, 17384, 17168]
#: the booked deltas: ceil8(767/701/700 - 300)
DELTAS = [472, 408, 400]
RATE = 0.3091  # D_EXTEND_CAP_PER_ROW_MIB, rate_src=start

_TMP = tempfile.mkdtemp(prefix="tp1corridor-")


def _store(cards, tp_pp=(3, 1), name="drei"):
    """A phase-footprint store shaped like /root/.cache/flliper/phase_footprint-a191a0712717-cc0ac5caf91e.json."""
    path = os.path.join(_TMP, f"phase_footprint-{name}.json")
    with open(path, "w") as fh:
        json.dump({"version": 1, "hw_fingerprint": "a191a0712717", "profile_digest": "cc0ac5caf91e",
                   "profile": [["Qwen3_5ForConditionalGeneration"], 4096, tp_pp[0], tp_pp[1], "fp8_e4m3", 3, 24],
                   "cards": {u: {"activation_mib": MEASURED.get(u, 700), "capture_mib": 198} for u in cards}}, fh)
    return path


STORE_DREI = _store([U5090, U_TP1, U_TP2])


def _cards():
    return L.order_cards([
        L.Card(nvml_index=0, uuid=U5090, name="NVIDIA GeForce RTX 5090", total_mib=32607, reserved_mib=499),
        L.Card(nvml_index=1, uuid=U_TP1, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
        L.Card(nvml_index=2, uuid=U_TP2, name="NVIDIA GeForce RTX 3080", total_mib=20480, reserved_mib=425),
    ])


def _floor(uuid, source, transient, reserve, store):
    prov = (f"{store}#cards.{uuid}.activation_mib={transient}MiB (measured_peak)"
            if source.startswith("MEASURED-") else "test")
    return CG.CorridorFloor(card_uuid=uuid, group="D", transient_mib=int(transient), reserve_mib=int(reserve),
                            source=source, provenance=prov, basis=CG.BASIS_ALLOC_DELTA)


def _floors(source="MEASURED-D", values=None, store=None):
    """floors_for_cards with the REAL reserve semantics (mib = transient + reserve)."""
    vals = MEASURED if values is None else values
    st = STORE_DREI if store is None else store

    def fn(cards, group, user_reserve_mib=None):
        out = {}
        for c in cards:
            r = (user_reserve_mib or {}).get(c.uuid, 0) if isinstance(user_reserve_mib, dict) \
                else int(user_reserve_mib or 0)
            out[c.uuid] = _floor(c.uuid, source, vals[c.uuid], r, st)
        return out
    return fn


def _d_pass(lines, floors, dormant=None, single="0"):
    cards = _cards()
    dc = dict(DORMANT if dormant is None else dormant)
    env = {CG.USER_RESERVE_ENV: json.dumps(RESERVE)}  # a reserve published for the front is in the env
    # VOTE-LINE-SINGLE (desk/27b-vote-line-single-1024): this file pins the FLOOR-line book (the switch-off path,
    # FLLIPER_PDFLIP_VOTE_LINE_SINGLE=0, = b130e9dab4); the single line is pinned in test_27b_vote_line_single_1024
    env[BR.VOTE_LINE_SINGLE_ENV] = single
    with mock.patch.object(CB, "floors_for_cards", floors), mock.patch.dict(os.environ, env):
        kw = dict(L.booked_rest_kwargs(cards, F.PROFILE_QWEN27B, "D", lines.append))
        budgets = L.budgets_from_dc(
            cards, dc, lines.append, "D", user_reserve_by_card=dict(RESERVE),
            charge_driver_carve=L.budget_charges_driver_carve(F.PROFILE_QWEN27B),
            driver_carve_min_total_mib=L.driver_carve_min_total_mib(F.PROFILE_QWEN27B), **kw)
    return cards, budgets, kw


class TheLiveBootsBudgetIsReproduced(CustomTestCase):
    def test_head_budgets_are_the_live_boots_without_a_measured_floor(self):
        _, budgets, _ = _d_pass([], _floors(source=CG.FLOOR_SOURCE_FALLBACK, values=FALLBACK))
        self.assertEqual(budgets, HEAD_BUDGETS)


class TheMovedVoteLineIsBookedBesideTheRest(CustomTestCase):
    def test_measured_floors_book_the_delta(self):
        """RED on 5255bf2d5d (budgets 27808 / 17384 / 17168): the vote's line 767 / 701 / 700
        minus the record's 300, up to the 8-MiB grain (472 / 408 / 400), is booked beside the
        rest -> 27336 / 16976 / 16768."""
        lines = []
        cards, budgets, kw = _d_pass(lines, _floors())
        self.assertEqual(budgets, [27336, 16976, 16768])
        self.assertEqual([h - b for h, b in zip(HEAD_BUDGETS, budgets)], DELTAS)
        self.assertEqual(kw["booked_rest_mib"], [3191 + 472, 2079 + 408, 2067 + 400])
        # A3: the delta is named DERIVED, in the provenance and on every card line
        self.assertIn(BR.VOTE_LINE_MARKER, kw["booked_rest_provenance"])
        self.assertIn("DERIVED", kw["booked_rest_provenance"])
        self.assertIn("not a measurement", kw["booked_rest_provenance"])
        tp1 = [ln for ln in lines if BR.VOTE_LINE_MARKER in ln and "ordinal=1 nvml1: delta 408 MiB" in ln]
        self.assertEqual(len(tp1), 1)
        self.assertIn("DERIVED", tp1[0])

    def test_the_line_is_the_ranks_line(self):
        # S3: one formula for the rank's vote line and the budget's delta
        self.assertEqual(ET.line_for_floor_mib(701), 701.0)
        self.assertEqual(ET.line_for_floor_mib(120), float(ET.CHUNKING_FLOOR_MIB))

    def test_the_tp1_chunk_widens_and_the_vote_still_ends_on_its_line(self):
        """Hochgerechnet: at the live tight instants (post 790 / 733.5 MiB, logged 790 / 734)
        the TP1 vote goes 287 -> 1607 and 105 -> 1425 rows; the priced end free stays >= 701."""
        _, budgets, _ = _d_pass([], _floors())
        delta = HEAD_BUDGETS[1] - budgets[1]
        self.assertEqual(delta, 408)
        for post, before, after_want in ((790.0, 287, 1607), (733.5, 105, 1425)):
            self.assertEqual(ET.rows_cap(post, RATE, 1, 701), before)
            after = ET.rows_cap(post + delta, RATE, 1, 701)
            self.assertEqual(after, after_want)
            self.assertGreaterEqual(after, ET.rows_cap(post, RATE, 1, 300))
            self.assertGreaterEqual(post + delta - RATE * after, 701)


class TheTorchCapNeverRises(CustomTestCase):
    """A2 -- the OOM direction: cap = budget + rest - OTHER must not rise by the delta's rounding."""

    def _sums(self, floors_fix, floors_head, dormant=None):
        _, b, kw = _d_pass([], floors_fix, dormant)
        _, hb, hkw = _d_pass([], floors_head, dormant)
        return ([x + r for x, r in zip(b, kw["booked_rest_mib"])],
                [x + r for x, r in zip(hb, hkw["booked_rest_mib"])])

    def test_live_boot_cap_is_byte_identical(self):
        new, old = self._sums(_floors(), _floors(source=CG.FLOOR_SOURCE_FALLBACK, values=FALLBACK))
        self.assertEqual(new, old)

    def test_a_grain_unaligned_delta_does_not_raise_the_cap(self):
        """Budget grain worst case: TP1 total-carve-dormant-rest = 17391 (== 7 mod 8), floor 707
        (raw delta 407 == 7 mod 8). Unrounded: floor8(17391 - 407) + 407 = 17391 > 17384 = +7 MiB
        on the cap. Rounded up: 17384 - 408 + 408 = 17384."""
        dormant = dict(DORMANT, **{U_TP1: 585})
        vals = dict(MEASURED, **{U_TP1: 707})
        new, old = self._sums(_floors(values=vals), _floors(source=CG.FLOOR_SOURCE_FALLBACK, values=FALLBACK),
                              dormant)
        for n, o in zip(new, old):
            self.assertLessEqual(n, o)
        self.assertEqual(new, old)


class NoReserveEntersTheDelta(CustomTestCase):
    def test_a_user_reserve_is_never_booked_as_line(self):
        """S1: the arm passes --user-reserve-mib 1800/1400/1400 and the launcher publishes it for the
        front (USER_RESERVE_ENV, set in _d_pass); with the real reserve semantics in the floor
        (mib = transient + reserve) the delta stays 472 / 408 / 400."""
        _, budgets, kw = _d_pass([], _floors())
        self.assertEqual([h - b for h, b in zip(HEAD_BUDGETS, budgets)], DELTAS)
        self.assertEqual(BR.vote_line_deltas([1, 1, 1], [767 + 1800, 701 + 1400, 700 + 1400],
                                             [True, True, True], 300), [2272, 1808, 1800])


class NoDeltaWhereTheFloorIsNotMeasuredOfThisFormOrTheRegimeIsUnknown(CustomTestCase):
    def test_an_unmeasured_floor_never_cuts_kv(self):
        lines = []
        _, budgets, kw = _d_pass(lines, _floors(source=CG.FLOOR_SOURCE_FALLBACK, values=FALLBACK))
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertNotIn(BR.VOTE_LINE_MARKER, kw["booked_rest_provenance"])
        self.assertTrue(any("is not measured" in ln for ln in lines))

    def test_a_hand_set_law_books_nothing(self):
        _, budgets, _ = _d_pass([], _floors(source=CG.FLOOR_SOURCE_ENV, values=FALLBACK))
        self.assertEqual(budgets, HEAD_BUDGETS)

    def test_a_footprint_of_another_form_books_nothing(self):
        """S3: the launcher reads the LAST published D digest; a duo store (2 cards, tp 2) after
        a drei boot is not the line this boot's ranks keep."""
        duo = _store([U5090, U_TP1], tp_pp=(2, 1), name="duo")
        lines = []
        _, budgets, _ = _d_pass(lines, _floors(store=duo))
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertTrue(any("not of this form" in ln for ln in lines))
        self.assertIsNone(BR.footprint_form_mismatch({"cards": {U5090: 1, U_TP1: 1, U_TP2: 1},
                                                      "profile": [["x"], 4096, 3, 1]}, [U5090, U_TP1, U_TP2]))
        self.assertIn("tp x pp", BR.footprint_form_mismatch({"cards": {U5090: 1, U_TP1: 1, U_TP2: 1},
                                                             "profile": [["x"], 4096, 2, 1]},
                                                            [U5090, U_TP1, U_TP2]))

    def test_an_unreadable_footprint_books_nothing(self):
        lines = []
        _, budgets, _ = _d_pass(lines, _floors(store=os.path.join(_TMP, "missing.json")))
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertTrue(any("unreadable" in ln for ln in lines))

    def test_a_record_of_unknown_boots_books_nothing_and_says_so(self):
        self.assertIsNone(BR.record_vote_line("D_AWAKE_REST_BOOKED_MIB", ("someotherboot",)))
        self.assertEqual(BR.record_vote_line("D_AWAKE_REST_BOOKED_MIB", BR.BOOTS_0929), 300)
        lines = []
        with mock.patch.object(BR, "RECORD_VOTE_LINE_MIB", {}):
            _, budgets, _ = _d_pass(lines, _floors())
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertTrue(any(BR.VOTE_LINE_MARKER in ln and "unknown" in ln for ln in lines))

    def test_the_switch_off_restores_the_pre_fix_budgets(self):
        lines = []
        with mock.patch.dict(os.environ, {BR.VOTE_LINE_ENV: "0"}):
            _, budgets, _ = _d_pass(lines, _floors())
        self.assertEqual(budgets, HEAD_BUDGETS)
        self.assertTrue(any(BR.VOTE_LINE_ENV + "=0" in ln for ln in lines))
        self.assertTrue(BR.vote_line_switch_on({}))
        self.assertFalse(BR.vote_line_switch_on({BR.VOTE_LINE_ENV: "off"}))

    def test_the_registry_record_carries_the_named_boots(self):
        rec = F.PROFILES[F.PROFILE_QWEN27B].constants[BR.record_name("D")]
        self.assertEqual(tuple(rec.boots), BR.BOOTS_0929)

    def test_pure_deltas(self):
        # rounded UP to the 8-MiB grain
        self.assertEqual(BR.vote_line_deltas([3191, None, 2067], [767, 701, 700.2], [True, True, True], 300),
                         [472, 0, 408])
        self.assertEqual(BR.vote_line_deltas([1, 1], [200, 701], [True, False], 300), [0, 0])
        self.assertEqual(BR.vote_line_deltas([1], [701], [True], None), [0])
        with self.assertRaises(ValueError):
            BR.vote_line_deltas([1, 2], [701], [True], 300)


if __name__ == "__main__":
    import unittest

    unittest.main()
