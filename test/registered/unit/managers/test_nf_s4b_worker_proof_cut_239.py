"""#239 S4b (F14) part 6: #1424d on the worker trees.

Under the token cut (``--d-kv-token-cut 0,48,16``) a Form A worker OWNS token
rows: since S4b part 3 it carries a real arena (``get_mha_host_pool_cls`` gives
a KV-holding worker ``ArenaMHAHostPool``), and its load-back proves its own
chain page by page (``UnifiedRadixCache._1424_verify_load_chain`` ->
``arena_pool.verify_load_chain``). Its usable-match vote, though, was still
H98's bare KV reach (``admission_probe(follow=True)``): the #1424d proof cut
ran only on the host's side. A page the WORKER cannot prove therefore entered
the group MIN at full depth -- every rank admitted it, and the worker alone
stopped at its load (``#1424 CHAIN MISMATCH``), the one-rank death #1424d was
built to prevent on TP0.

The worker now cuts its reach at its last proven page, so the group MIN (and
H97's realize round, which reads the same vote) takes every rank there. A
byteless worker (every form without the cut) votes its reach unchanged.

Driven through the REAL pure helpers (``local_usable_matches``,
``admission_probe``, the MIN/MAX/realize payloads) with each rank's REAL Form A
role plan, on the H98 tree model plus a proof depth. RED on ebeccfd190 (the
worker votes its reach), GREEN with the fix."""
from __future__ import annotations

import importlib.util
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt import rank_role  # noqa: E402
from sglang.srt.managers import tp_match_floor as m  # noqa: E402

_H98 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_nf_form_a_follow_h98.py")
_spec = importlib.util.spec_from_file_location("_h98_harness", _H98)
h98 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h98)


class _ProvenTree(h98._Tree):
    """An H98 tree whose host chain is proven only up to ``proven`` tokens
    (``weg2_chain_proof_depth``: None = everything proven)."""

    def __init__(self, kv, anchors, proven=None):
        super().__init__(kv, anchors)
        self.proven = proven
        self.proof_asks = 0

    def weg2_chain_proof_depth(self, node, req=None):
        self.proof_asks += 1
        if self.proven is None:
            return None
        return self.proven


class _HoldsKv:
    """Pin ``form_a_worker_holds_kv`` (the owner bounds come from the parallel
    runtime, absent on the desk)."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.prev = rank_role.form_a_worker_holds_kv
        rank_role.form_a_worker_holds_kv = lambda: self.value
        return self

    def __exit__(self, *exc):
        rank_role.form_a_worker_holds_kv = self.prev
        return False


REACH = 18112
CUT = 15552  # the last proven page of the worker's chain (a page multiple)


class TestWorkerVote(unittest.TestCase):
    def test_kv_holding_worker_votes_its_last_proven_page(self):
        tree = _ProvenTree(REACH, [], proven=CUT)
        req = h98._req("a")
        with h98._switch(True), h98._as_rank(1), _HoldsKv(True):
            got = m.admission_probe(tree, req, follow=True)
        self.assertEqual(
            got, CUT,
            "a worker that owns token rows must vote its last PROVEN page -- on "
            "ebeccfd190 it votes its KV reach and stops ALONE at the load")
        self.assertFalse(getattr(tree, h98.FOLLOW_ATTR, False), "the follow walk must not leak")

    def test_proven_chain_votes_the_reach(self):
        tree = _ProvenTree(REACH, [], proven=None)
        with h98._switch(True), h98._as_rank(1), _HoldsKv(True):
            self.assertEqual(m.admission_probe(tree, h98._req("a"), follow=True), REACH)

    def test_byteless_worker_never_asks_the_proof(self):
        tree = _ProvenTree(REACH, [], proven=CUT)
        with h98._switch(True), h98._as_rank(1), _HoldsKv(False):
            self.assertEqual(m.admission_probe(tree, h98._req("a"), follow=True), REACH)
        self.assertEqual(tree.proof_asks, 0, "a byteless worker has nothing to prove")

    def test_usable_arm_carries_the_cut(self):
        tree = _ProvenTree(REACH, [], proven=CUT)
        req = h98._req("a")
        with h98._switch(True), h98._as_rank(2), _HoldsKv(True):
            self.assertEqual(m.local_usable_matches(tree, {"a": req}, {"a": REACH}), {"a": CUT})


class TestGroup(unittest.TestCase):
    def _plant_and_admit(self, trees, holds):
        """H98's reduce (usable MIN, MAX, H97 realize) and each rank's admission,
        with ``holds`` pinning form_a_worker_holds_kv for the worker ranks."""
        with h98._switch(True), _HoldsKv(holds):
            return h98._run(trees, "weg2-7-2")

    def test_group_takes_every_rank_to_the_workers_proven_page(self):
        # TP0 (share 0 under the cut: its KV rows are byteless, its mamba
        # anchors real) reaches 18112 with anchors at 15552 and 18112; TP1
        # proves its chain only to 15552.
        trees = {
            0: _ProvenTree(REACH, [CUT, REACH]),
            1: _ProvenTree(REACH, [], proven=CUT),
            2: _ProvenTree(REACH, []),
        }
        planted, geometry = self._plant_and_admit(trees, True)
        self.assertEqual(planted, {"weg2-7-2": CUT}, geometry)
        self.assertEqual(set(geometry.values()), {CUT},
                         f"every rank must admit the proven depth: {geometry}")

    def test_no_host_anchor_there_reprefills_uniformly(self):
        # TP0 holds no anchor at the worker's proven page: H97's realize round
        # plants 0 -- every rank re-prefills (slower, never a split).
        trees = {
            0: _ProvenTree(REACH, [REACH]),
            1: _ProvenTree(REACH, [], proven=CUT),
            2: _ProvenTree(REACH, []),
        }
        planted, geometry = self._plant_and_admit(trees, True)
        self.assertEqual(set(geometry.values()), {0}, f"{planted} {geometry}")

    def test_without_the_cut_nothing_changes(self):
        trees = {
            0: _ProvenTree(REACH, [CUT, REACH]),
            1: _ProvenTree(REACH, [], proven=CUT),
            2: _ProvenTree(REACH, []),
        }
        planted, geometry = self._plant_and_admit(trees, False)
        self.assertEqual(planted, {"weg2-7-2": REACH})
        self.assertEqual(set(geometry.values()), {REACH})


if __name__ == "__main__":
    unittest.main()
