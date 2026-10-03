"""#1471b: every rank names its own usable-match vote and why.

z30m 03:19:50 / 03:21:40 / 03:22:09 (D, rid weg2-116-141): the X gate priced
uncached = the WHOLE prompt (38687, 39693, 39693) right after every rank had
read 28096, 4352 and 36800 of it. The group's usable floor was 0 -- and the
lines that name which rank voted 0 and on which branch (#1424d PROOF-CUT,
RU FLOOR ZERO, X-PRICE-FLOOR) were all past their throttles, so the log could
not say. ``tp_match_floor.vote_reason`` keeps this rank's last vote per rid
with its branch, and the X gate's W31 line prints it.

Driven through the REAL ``admission_probe`` / ``local_usable_matches`` on the
H98 tree model. RED on 09993f1153 (no ``vote_reason``), GREEN with the change."""
from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import tp_match_floor as m  # noqa: E402

_S4B = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_nf_s4b_worker_proof_cut_239.py"
)
_spec = importlib.util.spec_from_file_location("_s4b_harness", _S4B)
s4b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s4b)
h98 = s4b.h98

REACH = 18112
CUT = 15552


class TestVoteReason(unittest.TestCase):
    def setUp(self):
        m._VOTE_WHY.clear()

    def test_host_vote_names_its_admission(self):
        tree = s4b._ProvenTree(REACH, [CUT, REACH])
        with h98._switch(True), h98._as_rank(0):
            got = m.admission_probe(tree, h98._req("h"), follow=False)
        self.assertEqual(got, REACH)
        self.assertEqual(m.vote_reason("h"), (REACH, REACH, "host_admission"))

    def test_host_proof_cut_below_every_anchor_votes_zero_by_name(self):
        # The shape that zeroes the group floor: the host's chain is proven
        # only to CUT and no anchor sits at or below it -- its vote is 0 and
        # the reason says it was the proof cut, not a missing read.
        tree = s4b._ProvenTree(REACH, [REACH], proven=CUT)
        with h98._switch(True), h98._as_rank(0):
            got = m.admission_probe(tree, h98._req("z"), follow=False)
        self.assertEqual(got, 0)
        raw, vote, why = m.vote_reason("z")
        self.assertEqual((raw, vote), (REACH, 0))
        self.assertTrue(why.startswith(f"proof_cut@{CUT}"), why)

    def test_worker_proof_cut_is_named(self):
        tree = s4b._ProvenTree(REACH, [], proven=CUT)
        with h98._switch(True), h98._as_rank(1), s4b._HoldsKv(True):
            got = m.admission_probe(tree, h98._req("w"), follow=True)
        self.assertEqual(got, CUT)
        self.assertEqual(m.vote_reason("w"), (REACH, CUT, f"worker_proof_cut@{CUT}"))

    def test_worker_reach_is_named(self):
        tree = s4b._ProvenTree(REACH, [])
        with h98._switch(True), h98._as_rank(2), s4b._HoldsKv(False):
            m.admission_probe(tree, h98._req("r"), follow=True)
        self.assertEqual(m.vote_reason("r"), (REACH, REACH, "worker_reach"))

    def test_classic_usable_arm_names_the_zeroing_branch(self):
        tree = s4b._ProvenTree(REACH, [REACH], proven=CUT)
        req = h98._req("c")
        req.best_match_node = tree._node(REACH)
        with h98._switch(False):
            out = m.local_usable_matches(tree, {"c": req}, {"c": REACH})
        self.assertEqual(out, {"c": 0})
        self.assertEqual(m.vote_reason("c"), (REACH, 0, "proof_cut"))

    def test_unknown_rid_has_no_reason(self):
        self.assertIsNone(m.vote_reason("never-voted"))

    def test_table_is_bounded(self):
        for i in range(m._VOTE_WHY_CAP + 10):
            m._note_vote(h98._req(f"r{i}"), 1, 1, "x")
        self.assertEqual(len(m._VOTE_WHY), m._VOTE_WHY_CAP)
        self.assertIsNone(m.vote_reason("r0"))
        self.assertIsNotNone(m.vote_reason(f"r{m._VOTE_WHY_CAP + 9}"))


if __name__ == "__main__":
    unittest.main()
