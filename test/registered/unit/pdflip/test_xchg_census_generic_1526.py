"""AP4 1526: the W71 exchange census is joined to the LIVE inventory, not to this rig's three card UUIDs.

User order 05.10. 20:4xZ: "das muss generisch sein bzw. per flags oder env gesetzt werden"; addendum 20:5xZ: the
genericity holds across MODEL and FORMAT too (NF, INT8, FP8, NVFP4, GGUF; flip mode without Dual).

Before: ``xchg_residency.solve`` and ``launcher.xchg_form_dormant_reserve`` looked every live card up by UUID in the census
file and raised ``card ... is not in the census`` for any card of another rig (nf.env:63,130 point at a census keyed by
GPU-31d7ef41 / GPU-5c648f96 / GPU-62dbbae1) -- not forcebar, not a register code.

Now (``xchg_residency.resolve_census`` + ``launcher.load_xchg_census_for_cards``):
  (a) every live card is a census key (the reference rig, or any inventory the census was measured on): the SAME census object
      comes back, nothing is logged, no refusal is remembered, ``solve`` is value-identical -- also with ``--force`` armed;
  (b) two identical foreign cards: a named W71 ``HW-BORROWED`` without ``--force``; with ``--force`` the boot goes on,
      one ``FORCED-PAST HW-BORROWED`` line per card, and the peak is graded against the card's LIVE NVML total;
  (c) four mixed cards (two known, one same-class unknown, one foreign class): no IndexError, per-card donor kind
      (class / conservative) named, armed under ``--force``;
  (d) an unknown UUID is a NAMED refusal without ``--force``, runs with ``--force`` and the HW-BORROWED line; a card too
      small for the borrowed row still refuses by W71 (force does not lift the peak check);
  (e) the SAME code with a second model profile -- the 27B census fixture (INT8/FP8 tags, N = 2 on census keys) -- gives
      today's result, and the 27B reserve (``xchg_census_is_reserve`` false) is the weg2xsn14 constant, untouched.

GPU-free, NVML-free.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import launcher as L
from flliper.srt.pdflip import refusals as R
from flliper.srt.pdflip import xchg_residency as X
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-pdflip-unit")

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_SRC = os.path.join(HERE, "..", "..", "..", "..", "python", "flliper", "srt", "pdflip", "launcher.py")
FIXTURE_27B = os.path.join(HERE, "fixtures", "xchg_launch_replay_0911", "census_weg2sn5b_48f55fb393.json")

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"

CARD_5090 = "NVIDIA GeForce RTX 5090"
CARD_3080 = "NVIDIA GeForce RTX 3080"


def ref_cards():
    return [L.Card(1, BIG, CARD_5090, 32607, cc=(12, 0)),
            L.Card(0, SM1, CARD_3080, 20480, cc=(8, 6)),
            L.Card(2, SM2, CARD_3080, 20480, cc=(8, 6))]


def nf_like_census(path):
    """16 equal weight bands per group, like xchg_census_fnFL2_graph.json (NF graph form); the numbers are that file's."""
    per = {BIG: (910, 983, 1896), SM1: (564, 970, 1076), SM2: (494, 836, 1474)}
    cards = {u: {"tags": {"P": {f"weights_{k}": p for k in range(16)}, "D": {f"weights_{k}": d for k in range(16)}},
                 "dormant_proc_used_mib": dm, "dormant_source": f"test row {u[:12]}"}
             for u, (p, d, dm) in per.items()}
    blob = {"cards": cards, "waves": [[f"weights_{k}"] for k in range(16)], "provenance": "test NF-like census"}
    with open(path, "w") as fh:
        json.dump(blob, fh)
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(R.ENV_FORCED_BOOT, None)
        R.arm(False)
        L.set_xchg_census_map("")
        self.tmp = tempfile.mkdtemp()
        self.census_path = nf_like_census(os.path.join(self.tmp, "census.json"))
        self.lines = []

    def tearDown(self):
        R.arm(False)
        L.set_xchg_census_map("")
        os.environ.pop(R.ENV_FORCED_BOOT, None)
        if self._env is not None:
            os.environ[R.ENV_FORCED_BOOT] = self._env

    def log(self, line):
        self.lines.append(str(line))

    def forced_codes(self):
        return [d["code"] for d in R.forced_list()]


def foreign(i, uuid, name, mib, cc):
    return L.Card(i, uuid, name, mib, cc=cc)


# ---------------------------------------------------------------------------
class ReferenceUnchanged(Base):
    """(a) the old path is the default whenever the inventory is the census' own."""

    def test_same_object_no_borrows_no_notes(self):
        census = X.load_census(self.census_path)
        out, borrows, notes = X.resolve_census(census, ref_cards())
        self.assertIs(out, census)
        self.assertEqual((borrows, notes), ([], []))

    def test_subset_of_the_census_keys_is_unchanged_too(self):
        census = X.load_census(self.census_path)
        out, borrows, notes = X.resolve_census(census, ref_cards()[:2])
        self.assertIs(out, census)
        self.assertEqual((borrows, notes), ([], []))

    def test_solve_is_value_identical_and_nothing_is_logged_or_remembered(self):
        for forced in (False, True):
            R.arm(forced)
            before = X.solve(ref_cards(), X.load_census(self.census_path), L.ARMING_FLOOR_MIB)
            census = L.load_xchg_census_for_cards(self.census_path, ref_cards(), self.log)
            after = X.solve(ref_cards(), census, L.ARMING_FLOOR_MIB)
            self.assertEqual(before.rows, after.rows)
            self.assertEqual(before.lines, after.lines)
            self.assertEqual(before.refusals, after.refusals)
            self.assertEqual(self.lines, [])
            self.assertEqual(R.forced_list(), [])
            self.assertEqual(R.flush(self.log), 0)

    def test_prepare_weight_exchange_and_reserve_on_the_reference(self):
        res = L.prepare_weight_exchange(ref_cards(), self.log, "exchange", self.census_path, 7, 0)
        self.assertTrue(res.armed)
        self.assertFalse([ln for ln in self.lines if "BORROWED" in ln or "CENSUS-MAP" in ln])
        out, _ = L.xchg_form_dormant_reserve(ref_cards(), self.census_path, profile="nextflash")
        self.assertEqual(out, {BIG: 1896, SM1: 1076, SM2: 1474})
        self.assertEqual(R.forced_list(), [])


# ---------------------------------------------------------------------------
class TwoIdenticalForeignCards(Base):
    """(b)"""

    def cards(self):
        return [foreign(0, "GPU-aaaa-3090-0", "NVIDIA GeForce RTX 3090", 24576, (8, 6)),
                foreign(1, "GPU-aaaa-3090-1", "NVIDIA GeForce RTX 3090", 24576, (8, 6))]

    def test_without_force_a_named_refusal_for_every_card(self):
        with self.assertRaises(X.PdFlipXchgResidencyUnarmable) as cm:
            L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log)
        msg = str(cm.exception)
        self.assertIn("HW-BORROWED", msg)
        self.assertIn("GPU-aaaa-3090-0", msg)
        self.assertIn("--force", msg)
        self.assertIn("W71 PdFlipXchgResidencyUnarmable", msg)
        self.assertIsInstance(cm.exception, L.REFUSALS)        # cli() exits 2 with the one named line
        self.assertEqual(R.forced_list(), [])

    def test_with_force_it_runs_and_prints_one_forced_past_line_per_card(self):
        R.arm(True)
        res = L.prepare_weight_exchange(self.cards(), self.log, "exchange", self.census_path, 7, 0)
        self.assertTrue(res.armed)
        past = [ln for ln in self.lines if ln.startswith("FORCED-PAST HW-BORROWED")]
        self.assertEqual(len(past), 2, self.lines)
        self.assertEqual(self.forced_codes(), ["HW-BORROWED", "HW-BORROWED"])
        self.assertEqual(res.order, ["GPU-aaaa-3090-0", "GPU-aaaa-3090-1"])
        # the peak is graded against the LIVE total of the foreign card, not the donor's
        self.assertEqual({r.total_mib for r in res.rows}, {24576})
        # AP2b 1006: the census rows are labelled from the reference rig's registry (BIG = RTX5090, SM1/SM2 = RTX3080),
        # so a 3090 (sm_86) borrows the heaviest row of its ARCH TWIN class RTX3080 (SM1), no longer the 5090 row
        # (before: "conservative", the heaviest row of the census -- 24330 MiB of a 5090 image on 4x3090 was W71)
        self.assertIn("ARCH TWIN", past[0])
        self.assertIn(SM1[:12], past[0])
        self.assertNotIn("from census row " + BIG, past[0])

    def test_with_force_the_second_reader_adds_no_second_line(self):
        # known class (the W19 class selector, AP1, is a separate stop for classes without a record), unknown UUIDs
        R.arm(True)
        cards = [foreign(0, "GPU-aaaa-5090-0", CARD_5090, 32607, (12, 0)), foreign(1, "GPU-aaaa-5090-1", CARD_5090, 32607, (12, 0))]
        L.prepare_weight_exchange(cards, self.log, "exchange", self.census_path, 7, 0)
        L.xchg_form_dormant_reserve(cards, self.census_path, log=self.log, profile="nextflash")
        self.assertEqual(self.forced_codes(), ["HW-BORROWED", "HW-BORROWED"])


# ---------------------------------------------------------------------------
class FourMixedCards(Base):
    """(c)"""

    def cards(self):
        return [L.Card(1, BIG, CARD_5090, 32607, cc=(12, 0)),
                L.Card(0, SM1, CARD_3080, 20480, cc=(8, 6)),
                foreign(2, "GPU-bbbb-3080-new", CARD_3080, 20480, (8, 6)),            # same class, unknown uuid
                foreign(3, "GPU-bbbb-4090", "NVIDIA GeForce RTX 4090", 24564, (8, 9))]  # foreign class

    def test_donor_kind_per_card(self):
        census = X.load_census(self.census_path)
        out, borrows, notes = X.resolve_census(census, self.cards())
        self.assertEqual(notes, [])
        self.assertEqual({b.uuid: b.kind for b in borrows}, {"GPU-bbbb-3080-new": "class", "GPU-bbbb-4090": "conservative"})
        by = {b.uuid: b for b in borrows}
        self.assertEqual(by["GPU-bbbb-3080-new"].donor_uuid, SM1)           # the 3080 row (labelled by the live SM1)
        self.assertEqual(by["GPU-bbbb-4090"].donor_uuid, BIG)               # heaviest row overall
        self.assertEqual(set(out.cards), set(census.cards) | {"GPU-bbbb-3080-new", "GPU-bbbb-4090"})
        self.assertEqual(out.cards[SM1], census.cards[SM1])                 # own rows untouched

    def test_no_index_error_and_armed_under_force(self):
        with self.assertRaises(X.PdFlipXchgResidencyUnarmable):
            L.prepare_weight_exchange(self.cards(), self.log, "exchange", self.census_path, 7, 0)
        R.arm(True)
        res = L.prepare_weight_exchange(self.cards(), self.log, "exchange", self.census_path, 7, 0)
        self.assertTrue(res.armed)
        self.assertEqual(len(res.order), 4)
        self.assertEqual(self.forced_codes(), ["HW-BORROWED", "HW-BORROWED"])


# ---------------------------------------------------------------------------
class UnknownUuid(Base):
    """(d)"""

    def cards(self):
        return ref_cards()[:2] + [foreign(2, "GPU-cccc-unknown", CARD_3080, 20480, (8, 6))]

    def test_refusal_is_named_and_forcebar_in_the_register(self):
        r = R.by_code("HW-BORROWED")
        self.assertIsNotNone(r)
        self.assertTrue(r.forcebar)
        self.assertIn("HW-BORROWED", R.wired_codes(open(LAUNCHER_SRC, encoding="utf-8").read()))
        with self.assertRaises(X.PdFlipXchgResidencyUnarmable) as cm:
            L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log)
        self.assertIn("GPU-cccc-unknown", str(cm.exception))
        self.assertIn("HW-BORROWED", str(cm.exception))

    def test_forced_runs_and_logs_the_warning_line(self):
        R.arm(True)
        census = L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log)
        self.assertIn("GPU-cccc-unknown", census.cards)
        self.assertTrue(any(ln.startswith("FORCED-PAST HW-BORROWED") and "GPU-cccc-unknown" in ln for ln in self.lines))
        self.assertFalse(R.records_allowed())            # a forced boot writes no records

    def test_force_does_not_lift_the_peak_check(self):
        R.arm(True)
        tiny = [foreign(0, "GPU-dddd-tiny-0", "NVIDIA GeForce RTX 3070", 8192, (8, 6)),
                foreign(1, "GPU-dddd-tiny-1", "NVIDIA GeForce RTX 3070", 8192, (8, 6))]
        with self.assertRaises(X.PdFlipXchgResidencyUnarmable) as cm:
            L.prepare_weight_exchange(tiny, self.log, "exchange", self.census_path, 7, 0)
        self.assertIn("predicted VRAM residency", str(cm.exception))

    def test_operator_map_is_an_assertion_not_a_borrow(self):
        L.set_xchg_census_map("nvml2=%s" % SM2)
        census = L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log)
        self.assertEqual(census.cards["GPU-cccc-unknown"].tags, X.load_census(self.census_path).cards[SM2].tags)
        self.assertTrue(any(ln.startswith("PDFLIP-XCHG-CENSUS-MAP") for ln in self.lines))
        self.assertEqual(R.forced_list(), [])
        self.assertFalse(R.forced_boot())

    def test_class_label_and_uuid_keys_of_the_map(self):
        for key in ("RTX3080", "GPU-cccc-unknown"):
            self.lines.clear()
            census = L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log,
                                                  census_map="%s=%s" % (key, BIG))
            self.assertEqual(census.cards["GPU-cccc-unknown"].dormant_proc_used_mib, 1896)

    def test_bad_map_is_hard(self):
        for bad in ("nvml2", "nvml2=GPU-not-in-census"):
            with self.assertRaises(X.PdFlipXchgResidencyUnarmable):
                L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log, census_map=bad)
        R.arm(True)
        with self.assertRaises(X.PdFlipXchgResidencyUnarmable):
            L.load_xchg_census_for_cards(self.census_path, self.cards(), self.log, census_map="nvml2=GPU-not-in-census")

    def test_card_class_field_of_the_census_file_selects_the_donor(self):
        with open(self.census_path) as fh:
            blob = json.load(fh)
        blob["cards"][SM2]["card_class"] = "RTX3090/24576MiB/sm86"
        blob["cards"][BIG]["card_class"] = "RTX5090"
        path = os.path.join(self.tmp, "classed.json")
        with open(path, "w") as fh:
            json.dump(blob, fh)
        cards = [foreign(0, "GPU-eeee-3090", "NVIDIA GeForce RTX 3090", 24576, (8, 6))]
        out, borrows, _ = X.resolve_census(X.load_census(path), cards)
        self.assertEqual((borrows[0].kind, borrows[0].donor_uuid), ("class", SM2))
        self.assertEqual(out.cards["GPU-eeee-3090"].dormant_proc_used_mib, 1474)


# ---------------------------------------------------------------------------
class SecondModelProfile27B(Base):
    """(e) the same code, the 27B census (INT8/FP8 tags, ``qwen27b`` profile), N = 2."""

    def cards(self):
        return [L.Card(1, BIG, CARD_5090, 32607, cc=(12, 0)), L.Card(0, SM1, CARD_3080, 20480, cc=(8, 6))]

    def test_n2_on_census_keys_is_todays_result(self):
        census = X.load_census(FIXTURE_27B)
        out, borrows, notes = X.resolve_census(census, self.cards())
        self.assertIs(out, census)
        self.assertEqual((borrows, notes), ([], []))
        for forced in (False, True):
            R.arm(forced)
            direct = X.solve(self.cards(), census, L.ARMING_FLOOR_MIB)
            via = X.solve(self.cards(), L.load_xchg_census_for_cards(FIXTURE_27B, self.cards(), self.log), L.ARMING_FLOOR_MIB)
            self.assertEqual((direct.rows, direct.lines, direct.refusals), (via.rows, via.lines, via.refusals))
        self.assertEqual(self.lines, [])

    def test_27b_reserve_is_the_constant_and_untouched(self):
        for profile in ("qwen27b", None):
            out, _ = L.xchg_form_dormant_reserve(self.cards(), FIXTURE_27B, profile=profile)
            for c in self.cards():
                self.assertEqual(out[c.uuid], L.dc_measured_d_mib(c, L.WEIGHT_SOURCE_EXCHANGE))
        self.assertEqual(R.forced_list(), [])

    def test_27b_census_on_a_foreign_n2_pair(self):
        foreign_pair = [foreign(0, "GPU-ffff-0", "NVIDIA GeForce RTX 3090", 24576, (8, 6)),
                        foreign(1, "GPU-ffff-1", "NVIDIA GeForce RTX 3090", 24576, (8, 6))]
        with self.assertRaises(X.PdFlipXchgResidencyUnarmable):
            L.load_xchg_census_for_cards(FIXTURE_27B, foreign_pair, self.log)
        R.arm(True)
        census = L.load_xchg_census_for_cards(FIXTURE_27B, foreign_pair, self.log)
        res = X.solve(foreign_pair, census, L.ARMING_FLOOR_MIB)
        self.assertEqual(res.order, ["GPU-ffff-0", "GPU-ffff-1"])
        # the 27B tags (not the NF ones) priced the borrowed rows
        self.assertEqual(set(census.cards["GPU-ffff-0"].tags["P"]), set(X.load_census(FIXTURE_27B).cards[BIG].tags["P"]))


class FlagAndRegister(Base):
    def test_flag_exists_with_empty_default(self):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.pdflip_xchg_census_map, "")
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--pdflip-xchg-census-map", "nvml2=%s" % SM2])
        self.assertEqual(ns.pdflip_xchg_census_map, "nvml2=%s" % SM2)

    def test_main_arms_the_map_next_to_force(self):
        src = open(LAUNCHER_SRC, encoding="utf-8").read()
        self.assertIn('set_xchg_census_map(getattr(ns, "pdflip_xchg_census_map", ""))', src)
        # all three census readers go through the one resolver
        self.assertEqual(src.count("xchg_residency.load_census("), 1)   # only inside load_xchg_census_for_cards
        self.assertIn("load_xchg_census_for_cards(census_path, cards, log)", src)


if __name__ == "__main__":
    unittest.main()
