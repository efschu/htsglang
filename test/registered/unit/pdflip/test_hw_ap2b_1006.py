"""AP2b 1006 (nf-hwgen-ap2b): the two walls that still stopped a --force dry run of NF at N = 4 after AP2.

User order 06.10.: "4 Karten muss natuerlich moeglich sein".  AP2 report (deskq/done/nf-hwgen-ap2-1006-bericht.md):

  (1) W71 residency fit.  The shipped census files (/spinning/gpu-arb/weg2/census/) carry no ``card_class``; on a
      foreign inventory no live card carries their UUIDs, every row was "unlabelled", and every live card borrowed the
      census's HEAVIEST row -- the 5090 image (24330 MiB) on a 3080 (20480 MiB) and on a 3090 (24576 MiB, 246 MiB left,
      floor 1229).  The only way out was the hand flag ``--pdflip-xchg-census-map RTX5090=<uuid>,RTX3080=<uuid>``.
      NOW (``xchg_residency.reference_census_classes`` + ``resolve_census(known_classes=, twin_of=)``, passed by
      ``launcher.load_xchg_census_for_cards``): the launcher labels the census rows from the reference rig's card
      registry (UUID -> NVML board name, ``planner/power_limit.py``) and a live card borrows the row of its OWN class,
      else of its ARCH TWIN (AP1 W19 rule: sm_86 -> RTX3080).  The census files are not touched.  The borrow is still the
      named, forcebar HW-BORROWED refusal: without --force it refuses as before, with --force it is FORCED-PAST.
  (2) W120.  The planner (AP-C ``propose_rules.form_a_d``) proposed FR_D 1.000 for four D ranks (every card holds its
      whole share); a rank without two scratch rows builds no Platztausch buffer, and the launcher refuses that rightly.
      NOW the ceiling belongs to the proposal: ``planner.expert_residency.d_rank_fraction_caps`` -- the largest fraction
      that still builds the buffer, per rank, from owned + pad (the SAME count ``expert_map.unbuilt_platztausch_buffers``
      grades).  The AP-C side (``propose_rules.fr_d_with_buffer``) lives on the planner line; this file pins the rule
      and that the refusal stays as the guard.

Honesty: every borrowed census row is UNMEASURED on the card it prices; green here is the desk gate, not a boot.
GPU-free, NVML-free.
"""

import hashlib
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.layers.moe import expert_map as EM
from flliper.srt.planner import expert_residency as ER
from flliper.srt.pdflip import launcher as L
from flliper.srt.pdflip import refusals as R
from flliper.srt.pdflip import xchg_residency as X
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-pdflip-unit")

#: the reference rig's UUIDs, the keys of every shipped census (and of planner/power_limit.py RIG_*)
BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
SHIPPED = "/spinning/gpu-arb/weg2/census/xchg_census_fnFL2_graph.json"

N5090, N3080, N3090, N4090 = ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3090",
                              "NVIDIA GeForce RTX 4090")


def card(i, uuid, name, mib, cc):
    return L.Card(i, uuid, name, mib, cc=cc)


def ref_cards():
    return [card(1, BIG, N5090, 32607, (12, 0)), card(0, SM1, N3080, 20480, (8, 6)), card(2, SM2, N3080, 20480, (8, 6))]


def mixed4():
    """2 x 5090 + 2 x 3080, none of them a census key (the AP2 oracle inventory n4_mixed)."""
    return [card(0, "GPU-b2-5090-0", N5090, 32607, (12, 0)), card(1, "GPU-b2-5090-1", N5090, 32607, (12, 0)),
            card(2, "GPU-b2-3080-2", N3080, 20480, (8, 6)), card(3, "GPU-b2-3080-3", N3080, 20480, (8, 6))]


def n4_3090():
    return [card(i, f"GPU-b2-3090-{i}", N3090, 24576, (8, 6)) for i in range(4)]


def nf_like_census(path):
    """The NF graph census (xchg_census_fnFL2_graph.json) in 16 equal bands, WITHOUT card_class (like the shipped files):
    5090 row P 14560 / D 15728 / dormant 1896 MiB, 3080 rows 9024/15520/1076 and 7904/13376/1474."""
    per = {BIG: (910, 983, 1896), SM1: (564, 970, 1076), SM2: (494, 836, 1474)}
    cards = {u: {"tags": {"P": {f"weights_{k}": p for k in range(16)}, "D": {f"weights_{k}": d for k in range(16)}},
                 "dormant_proc_used_mib": dm, "dormant_source": f"test row {u[:12]}"}
             for u, (p, d, dm) in per.items()}
    with open(path, "w") as fh:
        json.dump({"cards": cards, "waves": [[f"weights_{k}"] for k in range(16)], "provenance": "test NF census"}, fh)
    return path


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class _Census(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(R.ENV_FORCED_BOOT, None)
        R.arm(False)
        L.set_xchg_census_map("")
        self.tmp = tempfile.mkdtemp()
        self.path = nf_like_census(os.path.join(self.tmp, "census.json"))
        self.lines = []

    def tearDown(self):
        R.arm(False)
        L.set_xchg_census_map("")
        os.environ.pop(R.ENV_FORCED_BOOT, None)
        if self._env is not None:
            os.environ[R.ENV_FORCED_BOOT] = self._env

    def log(self, line):
        self.lines.append(str(line))

    def past(self):
        return [ln for ln in self.lines if ln.startswith("FORCED-PAST HW-BORROWED")]


# ---------------------------------------------------------------------------
# (1) W71: the donor of a missing census row is derived from class / UUID, no hand flag
# ---------------------------------------------------------------------------
class TestReferenceRegistry(unittest.TestCase):
    def test_the_registry_names_the_reference_cards(self):
        self.assertEqual(X.reference_census_classes(), {BIG: "RTX5090", SM1: "RTX3080", SM2: "RTX3080"})

    def test_the_shipped_census_is_keyed_by_exactly_those_uuids(self):
        if not os.path.exists(SHIPPED):
            self.skipTest("shipped census not on this box")
        census = X.load_census(SHIPPED)
        self.assertEqual(set(census.cards), set(X.reference_census_classes()))
        self.assertTrue(all(not c.card_class for c in census.cards.values()))     # the files carry no class (unchanged)


class TestDerivedDonor(_Census):
    def test_mixed_four_borrow_their_own_class_without_a_map(self):
        R.arm(True)
        census = L.load_xchg_census_for_cards(self.path, mixed4(), self.log)
        src = X.load_census(self.path).cards
        for u in ("GPU-b2-5090-0", "GPU-b2-5090-1"):
            self.assertEqual(census.cards[u].tags, src[BIG].tags)
        for u in ("GPU-b2-3080-2", "GPU-b2-3080-3"):
            self.assertEqual(census.cards[u].tags, src[SM1].tags)               # the heaviest 3080 row, not the 5090
            self.assertEqual(census.cards[u].dormant_proc_used_mib, 1076)
        past = self.past()
        self.assertEqual(len(past), 4, self.lines)
        self.assertTrue(all("reference rig's card registry" in ln for ln in past))
        self.assertFalse([ln for ln in self.lines if ln.startswith("PDFLIP-XCHG-CENSUS-MAP")])   # no hand flag involved

    def test_mixed_four_arms_the_exchange_under_force(self):
        R.arm(True)
        res = L.prepare_weight_exchange(mixed4(), self.log, "exchange", self.path, 7, 0)
        self.assertTrue(res.armed, res.refusals)
        self.assertEqual([d["code"] for d in R.forced_list()], ["HW-BORROWED"] * 4)

    def test_four_3090_borrow_the_arch_twin_row_and_arm(self):
        R.arm(True)
        census = L.load_xchg_census_for_cards(self.path, n4_3090(), self.log)
        src = X.load_census(self.path).cards
        self.assertTrue(all(census.cards[c.uuid].tags == src[SM1].tags for c in n4_3090()))
        self.assertTrue(all("ARCH TWIN class RTX3080" in ln for ln in self.past()))
        self.lines.clear()
        res = L.prepare_weight_exchange(n4_3090(), self.log, "exchange", self.path, 7, 0)
        self.assertTrue(res.armed, res.refusals)
        self.assertEqual({r.total_mib for r in res.rows}, {24576})              # graded against the LIVE board

    def test_without_force_the_derived_borrow_is_still_refused(self):
        for cards in (mixed4(), n4_3090()):
            with self.assertRaises(X.PdFlipXchgResidencyUnarmable) as cm:
                L.load_xchg_census_for_cards(self.path, cards, self.log)
            self.assertIn("HW-BORROWED", str(cm.exception))
            self.assertIn("--force", str(cm.exception))
            self.assertEqual(R.forced_list(), [])

    def test_an_arch_without_a_twin_stays_conservative(self):
        census = X.load_census(self.path)
        _out, borrows, _n = X.resolve_census(census, [card(0, "GPU-b2-4090", N4090, 24564, (8, 9))],
                                             known_classes=X.reference_census_classes())
        self.assertEqual([(b.kind, b.donor_uuid) for b in borrows], [("conservative", BIG)])

    def test_the_operator_map_still_wins_and_is_not_refused(self):
        L.set_xchg_census_map("RTX3080=%s" % SM2)
        census = L.load_xchg_census_for_cards(self.path, mixed4()[2:], self.log)
        self.assertEqual(census.cards["GPU-b2-3080-2"].dormant_proc_used_mib, 1474)
        self.assertEqual(R.forced_list(), [])

    def test_the_census_file_is_not_touched(self):
        before = sha(self.path)
        R.arm(True)
        L.prepare_weight_exchange(mixed4(), self.log, "exchange", self.path, 7, 0)
        L.prepare_weight_exchange(n4_3090(), self.log, "exchange", self.path, 7, 0)
        self.assertEqual(sha(self.path), before)
        if os.path.exists(SHIPPED):
            s0 = sha(SHIPPED)
            L.prepare_weight_exchange(mixed4(), self.log, "exchange", SHIPPED, 7, 0)
            self.assertEqual(sha(SHIPPED), s0)

    def test_the_shipped_census_arms_n4_under_force(self):
        if not os.path.exists(SHIPPED):
            self.skipTest("shipped census not on this box")
        R.arm(True)
        for cards in (mixed4(), n4_3090()):
            res = L.prepare_weight_exchange(cards, self.log, "exchange", SHIPPED, 7, 0)
            self.assertTrue(res.armed, res.refusals)


class TestReferenceRigUnchanged(_Census):
    def test_same_object_nothing_logged_nothing_forced(self):
        census = X.load_census(self.path)
        out, borrows, notes = X.resolve_census(census, ref_cards(), known_classes=X.reference_census_classes())
        self.assertIs(out, census)
        self.assertEqual((borrows, notes), ([], []))
        for forced in (False, True):
            R.arm(forced)
            res = L.prepare_weight_exchange(ref_cards(), self.log, "exchange", self.path, 7, 0)
            self.assertTrue(res.armed)
            self.assertFalse([ln for ln in self.lines if "BORROWED" in ln or "FORCED-PAST" in ln])
            self.assertEqual(R.forced_list(), [])


# ---------------------------------------------------------------------------
# (2) W120: FR_D of the planner proposal is capped at the largest fraction WITH a Platztausch buffer
# ---------------------------------------------------------------------------
#: Form A ownership of the AP2 oracle runs (W120 text: "D-Rang 0: 95 von 95 Experten", ... -> owned = local - pad)
OWNED_MIXED = [94, 190, 114, 114]
OWNED_3090 = [53, 153, 153, 153]


class TestFrDCap(unittest.TestCase):
    def test_caps_are_the_numbers_the_w120_refusal_names(self):
        self.assertEqual(ER.d_rank_fraction_caps(OWNED_MIXED, 1), [0.978, 0.989, 0.982, 0.982])
        self.assertEqual(ER.d_rank_fraction_caps(OWNED_3090, 1), [0.962, 0.987, 0.987, 0.987])

    def test_the_cap_builds_the_buffer_and_is_the_largest_such_fraction(self):
        for e in range(3, 600):
            (cap,) = ER.d_rank_fraction_caps([e - 1], 1)
            self.assertLessEqual(ER.resident_rows(e, cap), e - 2, e)              # two scratch rows remain
            nxt = round(cap + 0.001, 3)
            if nxt < 1.0:
                self.assertGreater(ER.resident_rows(e, nxt), e - 2, e)            # one step more loses them
        self.assertEqual(ER.d_rank_fraction_caps([1], 1), [None])                 # 2 local rows: no fraction builds one

    def _karte(self, owned, fr_tp):
        stages = [s for s in range(4) for _ in range(12)]
        return EM.build_nested(sum(owned), owned, [0.5] * 4, fr_tp, stages, pad_tp=1)

    def test_the_map_builds_every_d_buffer_at_the_cap(self):
        for owned in (OWNED_MIXED, OWNED_3090):
            caps = ER.d_rank_fraction_caps(owned, 1)
            self.assertEqual([u for u in EM.unbuilt_platztausch_buffers(self._karte(owned, caps))
                              if u.phase == EM.PHASE_TP], [])
            unbuilt = [u for u in EM.unbuilt_platztausch_buffers(self._karte(owned, [1.0] * 4))
                       if u.phase == EM.PHASE_TP]
            self.assertEqual([u.max_fraction for u in unbuilt], caps)          # ONE rule on both sides of the seam

    def test_the_refusal_stays_the_guard(self):
        with self.assertRaises(L.PdFlipLaunchRefused) as cm:
            L._refuse_unbuilt_platztausch_buffers(self._karte(OWNED_MIXED, [1.0] * 4), chunk_layers=3)
        self.assertIn("W120 PdFlipPlatztauschBufferUnbuilt", str(cm.exception))
        self.assertIn("groesste Fraction mit Puffer 0.978", str(cm.exception))
        L._refuse_unbuilt_platztausch_buffers(self._karte(OWNED_MIXED, ER.d_rank_fraction_caps(OWNED_MIXED, 1)),
                                              chunk_layers=3)


if __name__ == "__main__":
    unittest.main()
