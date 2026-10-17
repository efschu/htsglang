"""K23 (NF h16u2 17.10. 20:01:10Z, D TP0/TP1/TP2, boot 25812b36f9).

THE MEASUREMENT. D (a Form A TP group, NOT the local-PP floor) died in
``alloc_req_slots`` at ``mamba_available=2/38``: 19 mamba states counted as
evictable, every one an unlocked device leaf whose eviction needs a host backup,
and the KV arena (6485 slots, 6077 pinned, tree=6186 refs) refused every claim
(``#1421 BACKUP-REFUSED why=arena_claim``, ``ARENA-DROP freed=0``). TP0's W3
spill answered ``W3-ARENA FORMA-REQUEST released_pages=0`` for ten minutes:
the arena was full of HOST-ONLY nodes and the host-only leaf spill
(``weg2/host_only_spill``) was gated to the local-PP floor, so ``W3-HOST-LEAF
SPILL`` appeared 0 times on D.

THE FIX. On TP0 of a Form A group (R12 role ``host``) the host-only leaves
leave too (L3 copy first), and each drop is sent to the workers as a node STATE
(``record_state``): TP0 decides, the group follows -- rank-uniform by
construction. A worker never spills on its own.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest
from unittest import mock

from sglang.srt.mem_cache import form_a_host_shadow as R12
from sglang.srt.weg2 import pp_slot_fidelity as SF
from sglang.test.test_utils import CustomTestCase

from test_w3_spill_host_leaves_int22_1008 import SPAN, _AnchorPool, _death_shape

URC = "sglang.srt.mem_cache.unified_radix_cache"




def _spill(cache, pool, claimer, role):
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
    with mock.patch.object(R12, "role", return_value=role):
        return cache._w3_arena_spill(pool, SPAN, claimer=claimer)


class TestK23FormAHostLeafSpill(CustomTestCase):
    def setUp(self):
        R12._reset_state_for_tests()

    def tearDown(self):
        R12._reset_state_for_tests()

    def test_tp0_of_a_tp_group_spills_host_only_leaves_and_sends_their_state(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        keys = {R12.node_key(mid), R12.node_key(leaf)}
        got = _spill(cache, pool, top, "host")
        self.assertEqual(got, 2 * SPAN, "TP0 must free the host-only leaves (was 0: W3-HOST-LEAF 0x on D)")
        self.assertEqual(pool.secured, [SPAN, SPAN], "L3 copy first, for every page")
        self.assertNotIn(leaf, cache._collect_all_nodes())
        sent = {(e[1], int(e[2])) for e in R12._S.ledger if e[0] == R12.STATE and e[3] == 0}
        self.assertEqual(sent, keys, "every drop goes to the workers as a STATE with exists=0")

    def test_a_worker_never_spills_on_its_own(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        with mock.patch.object(R12, "request_spill") as req:
            got = _spill(cache, pool, top, "worker")
        self.assertEqual(got, 0)
        self.assertIn(leaf, cache._collect_all_nodes())
        self.assertEqual(pool.secured, [])
        req.assert_called_once()

    def test_no_role_and_no_floor_edits_nothing(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool()
        self.assertEqual(_spill(cache, pool, top, None), 0)
        self.assertIn(leaf, cache._collect_all_nodes())

    def test_a_page_without_l3_copy_stays_on_tp0(self):
        cache, top, mid, leaf = _death_shape()
        pool = _AnchorPool(lose={100})
        self.assertEqual(_spill(cache, pool, top, "host"), 0)
        self.assertIn(leaf, cache._collect_all_nodes())
        self.assertFalse([e for e in R12._S.ledger if e[0] == R12.STATE])


if __name__ == "__main__":
    unittest.main()
