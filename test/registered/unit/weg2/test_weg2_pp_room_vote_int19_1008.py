# SPDX-License-Identifier: Apache-2.0
"""PR: every PP stage applies the same agreed room R_m in pass m (NF, 08.10.).

Two deaths of one class -- on the carrierless PP form every stage decides the
admission and the load-back room from ITS OWN pool, and the pools diverge:

* int19 13:37:49Z PP2 (P log ..._1008_133232.P.log): ``#969N ADMIT ... extend=16384``
  with ``full_available_size=15232``; the peel paid 0 of ``reported_evictable=
  132224`` -> ``Prefill out of memory`` -> RANK-DEATH.
* int20 14:11:26-33Z (P log ..._1008_135412.P.log 52524-53124): weg2-8-75, load-back
  49792; all three stages ``SF LOADBACK-ROOM PP-RESIDUAL ... avail=20928``; PP1
  loaded it a pass later on its own room (14:11:30, slot 2, fwd_ct=78), PP0 only at
  14:11:32 (slot 0) -> ``#1004 SLOT DISAGREEMENT`` -> RANK-DEATH.

RED on bff5913a62 (driven by behaviour through the readers that exist there; the
agreed value is set by its attribute name): the readers ignore it -- PP0's budget
stays its own, PP1 loads alone, the residual refuses rank-locally, the empty P
calls the head FITS. GREEN: one R_m on every stage -> one verdict; the residual
stops by name; no fact -> no R_m -> byte-identical.
"""

import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache import common  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

ATTR = "weg2_pp_room_cap"  # mem_cache.common.PP_ROOM_CAP_ATTR
AVAIL_PP2 = 15232  # 13:37 'full_available_size=15232'
REPORTED = 132224  # 13:37 'full_evictable_size_=132224'
CHUNK = 16384  # 13:37 'Try to allocate 16384 tokens'
KV_8_75 = 49792  # 14:11 'kv_tokens=49792'
AVAIL_1411 = 20928  # 14:11 'avail=20928'
EVICTABLE_1411 = 73600  # 14:11 'evictable=73600 evicted=0'


class _PP0Tree:
    """PP0's own pool says plenty (its tree diverged from PP2's)."""

    uniform_avail_floor = None

    def __init__(self):
        self.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 60000)

    def evictable_size(self):
        return 100000


class _StageTree:
    """One stage on the local-PP floor; its peel pays ``payable`` of what it reports."""

    def __init__(self, avail, evictable, payable=0):
        self.alloc = types.SimpleNamespace(free=avail)
        self.alloc.available_size = lambda: self.alloc.free
        self.token_to_kv_pool_allocator = self.alloc
        self._ev, self._pay = evictable, payable
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def evictable_size(self):
        return self._ev

    def evict(self, params):
        got = min(int(params.num_tokens), self._pay)
        self._pay -= got
        self._ev -= got
        self.alloc.free += got
        return types.SimpleNamespace(num_tokens_evicted=got)


def _sf_on():
    os.environ[SF.ENV] = "1"


class LoadBackAgreementTest(unittest.TestCase):
    """14:11 form: three stages, different local room at PP1's pass, one R_m."""

    def setUp(self):
        _sf_on()

    def test_14_11_no_stage_loads_alone(self):
        # PP1 had room in its pass (something freed there); PP0/PP2 did not
        stages = {0: _StageTree(AVAIL_1411, EVICTABLE_1411),
                  1: _StageTree(60000, EVICTABLE_1411),
                  2: _StageTree(AVAIL_1411, EVICTABLE_1411)}
        r_m = AVAIL_1411  # MIN over the stages' payable room for this pass
        verdicts = {}
        for r, t in stages.items():
            setattr(t, ATTR, r_m)
            verdicts[r] = bool(SF.local_pp_room(t, KV_8_75, AVAIL_1411, "weg2-8-75"))
        self.assertEqual(verdicts, {0: False, 1: False, 2: False}, "one verdict, no #1004")

    def test_14_11_all_load_in_the_same_pass_once_all_have_room(self):
        stages = [_StageTree(30000, 30000, payable=30000), _StageTree(60000, 0),
                  _StageTree(25000, 40000, payable=40000)]
        for t in stages:
            setattr(t, ATTR, 55000)
        self.assertEqual([bool(SF.local_pp_room(t, KV_8_75, 20000, "weg2-8-75")) for t in stages],
                         [True, True, True])

    def test_the_residual_stops_by_name_instead_of_refusing_alone(self):
        """A stage whose own pool cannot pay what the stages agreed (a fact older
        than its pool's last change) must not refuse alone -- that is the
        14:11 split. It stops by name and is counted."""
        t = _StageTree(AVAIL_1411, EVICTABLE_1411, payable=0)
        setattr(t, ATTR, 60000)
        with self.assertRaises(RuntimeError) as cm:
            SF.local_pp_room(t, KV_8_75, AVAIL_1411, "weg2-8-75")
        self.assertIn("PR AGREED-ROOM SHORT", str(cm.exception))

    def test_without_an_agreement_the_load_back_is_unchanged(self):
        t = _StageTree(60000, 0)
        self.assertTrue(SF.local_pp_room(t, KV_8_75, AVAIL_1411, "weg2-8-75"))
        t2 = _StageTree(AVAIL_1411, EVICTABLE_1411)
        self.assertFalse(SF.local_pp_room(t2, KV_8_75, AVAIL_1411, "weg2-8-75"))


class AdmissionAgreementTest(unittest.TestCase):
    """13:37 form: the chunk budget is the agreed R_m on every stage."""

    def test_13_37_pp0_no_longer_admits_16384(self):
        tree = _PP0Tree()
        self.assertGreaterEqual(common.fundable_extend_tokens(tree), CHUNK, "PP0's own pool says yes")
        setattr(tree, ATTR, AVAIL_PP2)
        self.assertEqual(common.fundable_extend_tokens(tree), AVAIL_PP2)
        self.assertEqual(common.published_fundable_floor(tree), AVAIL_PP2)
        self.assertLess(common.chunk_tokens_the_pool_can_fund(
            common.fundable_extend_tokens(tree), 64, CHUNK), CHUNK)

    def test_13_37_empty_p_is_the_named_stall_not_fits(self):
        """Deadlock case: nothing runs on P, no stage can hold the head. Without
        R_m the intake verdict says FITS (PP0's 160000) and P waits for room no
        pass frees; with it the stall is named (503, re-route, flip)."""
        from sglang.srt.weg2 import p_intake
        from sglang.srt.weg2.intake_stall import INTAKE_IMPOSSIBLE

        tree = _PP0Tree()
        setattr(tree, ATTR, AVAIL_PP2)
        sched = types.SimpleNamespace(tree_cache=tree, token_to_kv_pool_allocator=tree.token_to_kv_pool_allocator)
        self.assertEqual(p_intake.intake_phase_verdict(sched, "weg2-0-9", CHUNK, 500000, "gate=adder_no_token"),
                         INTAKE_IMPOSSIBLE)

    def test_no_agreement_is_byte_identical(self):
        tree = _PP0Tree()
        self.assertEqual(common.fundable_extend_tokens(tree), 160000)
        self.assertIsNone(common.published_fundable_floor(tree))
        setattr(tree, ATTR, None)
        self.assertEqual(common.fundable_extend_tokens(tree), 160000)
        self.assertIsNone(common.published_fundable_floor(tree))


# --------------------------------------------------------------------------
# the vote itself (new module)
# --------------------------------------------------------------------------


class _CD:
    def __init__(self, n=0, lock=0, host_lock=0):
        self.value = torch.arange(n) if n else None
        self.lock_ref = lock
        self.host_lock_ref = host_lock


class _Node:
    def __init__(self, parent, *, n=0, backuped=False, evicted=False, lock=0, host_lock=0):
        self.id = id(self)
        self.parent = parent
        self.children = {}
        self.backuped = backuped
        self.evicted = evicted
        self.component_data = [_CD(n, lock, host_lock), _CD(), _CD()]
        if parent is not None:
            parent.children[self.id] = self


def _wb(root):
    return types.SimpleNamespace(root_node=root, ongoing_write_through={},
                                 cache_controller=types.SimpleNamespace(write_policy="write_back"))


def _fact(PR, *, room=AVAIL_PP2, executed=10, rank=2):
    return PR.Weg2PpRoomFact(rank=rank, executed=executed, room=room, available=AVAIL_PP2,
                             payable=room - AVAIL_PP2, reported=REPORTED)


class PayableEstimateTest(unittest.TestCase):
    def test_13_37_one_peel_pays_nothing(self):
        """128768 tokens behind two frontier leaves (3456) no peel pays: un-backed,
        arena full, a host-LOCKED host child each (UD/UD-H may not drop them)."""
        from sglang.srt.weg2 import pp_room_vote as PR

        root = _Node(None)
        chain = _Node(root, n=128768)
        for n in (2304, 1152):
            leaf = _Node(chain, n=n)
            _Node(leaf, backuped=True, evicted=True, host_lock=1)
        self.assertEqual(PR.estimate_payable(_wb(root), local_floor=True, host_free=0), 0)

    def test_payable_where_a_peel_can_pay(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        root = _Node(None)
        a = _Node(root, n=4096)                # un-backed: dropped on the local-PP floor
        _Node(a, n=1024, backuped=True)        # backed: demoted
        _Node(root, n=777, lock=1)             # a running request: never
        t = _wb(root)
        self.assertEqual(PR.estimate_payable(t, local_floor=True, host_free=0), 5120)
        self.assertEqual(PR.estimate_payable(t, local_floor=False, host_free=0), 1024)
        self.assertEqual(PR.estimate_payable(t, local_floor=False, host_free=4096), 5120)


class RoomBookTest(unittest.TestCase):
    def test_r_m_is_the_min_of_pp0_and_the_followers(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        book = PR.RoomBook()
        book.begin_pass()
        book.absorb([_fact(PR, room=90000, rank=1), _fact(PR, room=AVAIL_PP2, rank=2)])
        self.assertEqual((book.cap(200000).cap, book.cap(200000).rank), (AVAIL_PP2, 2))
        self.assertEqual((book.cap(1000).cap, book.cap(1000).rank), (1000, 0))

    def test_no_fact_no_r_m(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        book = PR.RoomBook()
        book.begin_pass()
        self.assertIsNone(book.cap(1000), "PP0's own room alone agrees nothing new")

    def test_inflight_rows_count_against_a_fact(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        book = PR.RoomBook()
        book.begin_pass()
        book.absorb([_fact(PR, room=50000, executed=10)])
        ext = types.SimpleNamespace(is_extend=lambda: True)
        book.note_batches([types.SimpleNamespace(forward_iter=c, forward_mode=ext, extend_num_tokens=r, reqs=[1])
                           for c, r in ((10, 16384), (11, 16384), (12, 8000))])
        v = book.cap(None)
        self.assertEqual((v.inflight, v.cap, v.rank), (24384, 25616, 2))

    def test_an_old_fact_is_not_read(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        book = PR.RoomBook()
        book.begin_pass()
        book.absorb([_fact(PR)])
        for _ in range(PR.FACT_MAX_AGE_PASSES):
            book.begin_pass()
        self.assertIsNotNone(book.cap(None))
        book.begin_pass()
        self.assertIsNone(book.cap(None), "a silent follower costs the agreement, never a wedge")


class WireTest(unittest.TestCase):
    def test_r_m_rides_list_m_and_every_stage_applies_the_same(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        reqs = ["r1", "r2"]
        wire = PR.stamp_cap(reqs, PR.CapVerdict(cap=AVAIL_1411, rank=2, room=AVAIL_1411, inflight=0))
        self.assertEqual(reqs, ["r1", "r2"], "the dispatched list stays PP0's own")
        rest, cap = PR.absorb_cap(wire)
        self.assertEqual((rest, cap), (["r1", "r2"], AVAIL_1411))
        trees = [_PP0Tree(), _PP0Tree()]
        PR.set_cap(trees[0], AVAIL_1411)          # PP0, at its pass top
        PR.apply_cap(trees[1], cap)               # a follower, after relaying list m
        self.assertEqual([common.fundable_extend_tokens(t) for t in trees], [AVAIL_1411] * 2)

    def test_no_r_m_on_the_list_is_byte_identical(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        reqs = ["r1"]
        self.assertIs(PR.stamp_cap(reqs, None)[0], "r1")
        rest, cap = PR.absorb_cap(reqs)
        self.assertIs(rest, reqs)
        self.assertIsNone(cap)
        t = _PP0Tree()
        PR.apply_cap(t, None)
        self.assertFalse(hasattr(t, ATTR), "nothing agreed, nothing written")

    def test_the_sender_never_blocks(self):
        from sglang.srt.weg2 import pp_room_vote as PR

        sent, busy = [], [False]
        ch = types.SimpleNamespace(send_nowait=lambda f: (not busy[0]) and (sent.append(f) or True))
        s = PR.FollowerSender()
        self.assertTrue(s.offer(_fact(PR), 0.0))
        self.assertTrue(s.flush(ch, 0.0))
        self.assertFalse(s.offer(_fact(PR), 0.1), "unchanged within RESEND_S")
        busy[0] = True
        self.assertTrue(s.offer(_fact(PR, room=1), 0.2))
        self.assertFalse(s.flush(ch, 0.2), "previous send in flight: kept pending, no block")
        busy[0] = False
        self.assertTrue(s.flush(ch, 0.3))
        self.assertEqual([x.room for x in sent], [AVAIL_PP2, 1])


if __name__ == "__main__":
    unittest.main()
