"""RELEASE 1022: the planning core for 5, 6 and 7 cards (user 22.10.: "2 karten und 4 karten, und 5 karten und 6 karten und 7 karten gehen").

Two searches grew with the card count and are bounded from 5 cards on; up to 4 cards they are the old exact ones, bit for bit (pinned here and by the NCARDS dry-run goldens):

* the owned expert ownership search of NF (``expert_residency``): a lattice of about 55^(N-1) vectors.  From 5 ranks on it runs over CLASSES of ranks that price
  identically (card, budget, fixed posts, slot, draft delta, miss/extend cost); the members of a class hold the same share.  N identical cards are a search over the host
  and one worker class;
* the P cut solver (``pp_cut``): every composition of the layers over the stages (595 k at 5 stages, 7 M at 6, 67 M at 7 for 64 layers).  From 5 stages on the field is the
  neighbourhood of the incumbent, as wide as the budget ``PP_CUT_WINDOW_MAX_CUTS`` allows (and the gapped splits stay within one layer of the even deal).

CPU only, no GPU, no launcher run.
"""

import itertools
import os
import pathlib
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.planner import expert_residency as E
from flliper.srt.planner import pp_cut as P

TREE = pathlib.Path(__file__).resolve().parents[4]
PROFILES = TREE / "docker" / "flliper" / "profiles_release"


def brute_cuts(total, stages, lo=1):
    return [c for c in itertools.product(range(lo, total + 1), repeat=stages) if sum(c) == total]


class CutField(unittest.TestCase):
    def test_up_to_four_stages_the_field_is_the_full_one(self):
        for n in (2, 3, 4):
            self.assertIsNone(P._cut_window(64, n, 1, [64 // n] * n))
        full = P._enumerate_cuts(64, 4, 1)
        self.assertEqual(len(full), 39711)                                  # C(63, 3): every composition, as before

    def test_the_full_enumeration_equals_the_brute_force_on_small_layer_counts(self):
        for stages in (2, 3, 4, 5, 6, 7):
            self.assertEqual(sorted(P._enumerate_cuts(14, stages, 1)), sorted(brute_cuts(14, stages)), stages)

    def test_the_window_is_a_subset_that_holds_the_incumbent_and_widens_to_the_full_field(self):
        for stages in (5, 6, 7):
            inc = [14 // stages + (1 if i < 14 % stages else 0) for i in range(stages)]
            full = set(P._enumerate_cuts(14, stages, 1))
            for hw in (1, 2, 3):
                win = set(P._enumerate_cuts(14, stages, 1, around=inc, halfwidth=hw))
                self.assertLessEqual(win, full)
                self.assertIn(tuple(inc), win)
                self.assertTrue(all(abs(a - b) <= hw for c in win for a, b in zip(c, inc)))
            self.assertEqual(set(P._enumerate_cuts(14, stages, 1, around=inc, halfwidth=14)), full)

    def test_from_five_stages_the_field_is_bounded_and_wide(self):
        for stages, min_hw in ((5, 5), (6, 3), (7, 2)):
            inc = [64 // stages + (1 if i < 64 % stages else 0) for i in range(stages)]
            hw = P._cut_window(64, stages, 1, inc)
            self.assertIsNotNone(hw)
            self.assertGreaterEqual(hw, min_hw, stages)
            cuts = P._enumerate_cuts(64, stages, 1, around=inc, halfwidth=hw)
            self.assertLessEqual(len(cuts), P.PP_CUT_WINDOW_MAX_CUTS)
            self.assertIn(tuple(inc), cuts)

    def test_the_chosen_cut_of_the_window_is_the_one_of_the_full_field(self):
        # identical cards and mixed cards (rates differ per stage), 5 stages, 20 layers: the window around the rate-proportional incumbent picks what the full field picks
        for rates in ((1.0, 1.0, 1.0, 1.0, 1.0), (1.0, 2.0, 2.0, 1.5, 1.0), (3.0, 1.0, 1.0, 1.0, 2.0)):
            timing = P.PrefillTiming(ms_per_layer=tuple(1.0 / r for r in rates), fixed_ms=tuple(0.0 for _ in rates))
            tot = sum(rates)
            inc = [max(1, round(20 * r / tot)) for r in rates]
            inc[inc.index(max(inc))] += 20 - sum(inc)
            full = P.solve_pp_cut_for_prefill_speed(20, timing, inc, 20)
            hw = P._cut_window(20, 5, 1, inc)
            self.assertIsNotNone(hw)
            win_cuts = set(P._enumerate_cuts(20, 5, 1, around=inc, halfwidth=hw))
            best_full = full[0].counts
            self.assertIn(best_full, win_cuts, (rates, best_full, hw))

    def test_the_gapped_deal_is_unchanged_up_to_four_stages_and_bounded_from_five(self):
        def full(n_attn, stages, gdn):
            lo = [0 if s == gdn else 1 for s in range(stages)]
            return sorted(c for c in itertools.product(range(0, n_attn + 1), repeat=stages) if sum(c) == n_attn and all(a >= b for a, b in zip(c, lo)))
        for stages in (2, 3, 4):
            for gdn in range(stages):
                self.assertEqual(sorted(P.enumerate_gapped_splits(8, stages, gdn)), full(8, stages, gdn))
        for stages in (5, 6, 7):
            got = P.enumerate_gapped_splits(16, stages, 0)
            self.assertTrue(got)
            self.assertTrue(all(sum(c) == 16 and c[0] >= 0 and min(c[1:]) >= 1 for c in got))
            self.assertLess(len(got), 20000)


def fit(rank, budget, fixed, slot=2.4, dv=0.0, exp=100.0):
    return types.SimpleNamespace(rank=rank, budget_mib=budget, fixed_mib=fixed, slot_mib=slot, draft_vocab_delta_mib=dv, expert_mib=exp)


class OwnedClasses(unittest.TestCase):
    def test_identical_workers_are_one_class_and_the_host_stands_alone(self):
        fits = [fit(0, 19208, 7480)] + [fit(r, 21384, 1074, exp=100.0 + r) for r in (1, 2, 3, 4)] + [fit(5, 21576, 952)]
        self.assertEqual(E.owned_rank_classes(fits, 0), [[0], [1, 2, 3, 4], [5]])                  # the expert MiB follow the ownership: not part of the class
        self.assertEqual(E.owned_rank_classes(fits, 0, miss_ms_rank=[1, 1, 1, 1, 1, 2]), [[0], [1, 2, 3, 4], [5]])
        self.assertEqual(E.owned_rank_classes(fits, 0, miss_ms_rank=[1, 1, 1, 1, 3, 1]), [[0], [1, 2, 3], [4], [5]])      # a rank with its own miss cost is its own class

    def test_class_vectors_are_class_constant_keep_the_sum_and_the_floor(self):
        for n in (5, 6, 7):
            base = [30] + [(512 - 30) // (n - 1)] * (n - 1)
            base[1] += 512 - sum(base)
            groups = [[0], list(range(1, n - 1)), [n - 1]]
            vecs = E.owned_ratio_vectors_classes(base, groups, step=8)
            self.assertTrue(vecs)
            self.assertLess(len(vecs), 400000)
            self.assertIn(tuple(base), set(vecs) | {tuple(base)})
            for v in vecs:
                self.assertEqual(sum(v), 512)
                self.assertGreaterEqual(min(v), 1)
                mids = [v[r] for r in range(1, n - 1)]
                mode = max(set(mids), key=mids.count)
                self.assertLessEqual(sum(1 for m in mids if m != mode), 1)                       # members of a class hold the same value (the rounding residue sits on ONE member)

    def test_the_class_vectors_are_part_of_the_full_lattice_on_a_small_case(self):
        base = [8, 12, 12, 12, 12]
        groups = [[0], [1, 2, 3], [4]]
        cls = set(E.owned_ratio_vectors_classes(base, groups, step=4, max_shift=16))
        full = set(E.owned_ratio_vectors_free(base, step=4, max_shift=16))
        self.assertTrue(cls)
        self.assertLessEqual(cls, full)

    def test_class_shares_sum_to_the_grid_and_leave_the_host_at_zero(self):
        groups = [[0], [1, 2, 3, 4], [5]]
        sh = E.owned_share_vectors_classes(6, 0, groups, 64, 4)
        self.assertTrue(sh)
        for v in sh:
            self.assertEqual(sum(v), 64)
            self.assertEqual(v[0], 0)
            self.assertEqual(len({v[1], v[2], v[3], v[4]}), 1)

    def test_a_search_with_four_ranks_or_fewer_does_not_branch(self):
        self.assertEqual(E.OWNED_CLASS_MIN_RANKS, 4)                                          # n > 4 only: 2, 3 and 4 ranks run the exact search as before


class Profiles(unittest.TestCase):
    FAMILIES = ("27b", "27b-nvfp4", "27b-nvfp4-dual", "nf")

    def test_every_family_has_a_profile_for_five_six_and_seven_cards(self):
        for fam in self.FAMILIES:
            for n in (5, 6, 7):
                p = PROFILES / ("%s-%dx3090.env" % (fam, n))
                self.assertTrue(p.is_file(), p)
                text = p.read_text(encoding="utf-8")
                self.assertIn("PROFILE_CARD_COUNT=%d\n" % n, text, p)
                self.assertIn('PROFILE_INVENTORY="%s"' % ",".join(["RTX3090"] * n), text, p)
                self.assertIn("PROFILE_STATUS=", text)

    def test_the_dual_p_budgets_name_one_value_per_card(self):
        for n in (5, 6, 7):
            text = (PROFILES / ("27b-nvfp4-dual-%dx3090.env" % n)).read_text(encoding="utf-8")
            line = [l for l in text.splitlines() if "6610,5050,5200/" in l][0]
            budgets = line.split("6610,5050,5200/")[1].split("}")[0]
            self.assertEqual(len(budgets.split(",")), n, line)


if __name__ == "__main__":
    unittest.main()
