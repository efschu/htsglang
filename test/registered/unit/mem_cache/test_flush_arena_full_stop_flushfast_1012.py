"""FLUSHFAST (NF int24g, boot ..._09235ed28a_1012_185900): D>P flips of 28-30 s.

THE MEASUREMENT (D.log, TP0, epoch 105, 21:36:29-21:36:58). The quiesce
``/flush_cache`` and the sleep leg's flush each ran the ``#1470 FLUSH-PUBLISH``
loop of ``Scheduler.flush_cache`` over 66 un-backed nodes (epoch 115: 36)
against a full KV arena: one ``arena_claim`` refusal and one fruitless
``#1427 ARENA-DROP need=.. freed=0 stages=i:0,ii:0,iii:0`` (~45 ms) PER NODE,
~3 s per round. The loop ended only at round 4 (``issued==0 and _round>=3``)
because 48 store writes stood in flight (``in_flight_after=48`` in every
round), so 4 x ~3 s = 12.5 s per leg (``FLUSH-PUBLISH issued=22 unbacked_left=66
in_flight_after=48 waited_ms=12474``). The SWEEP-FULL-ARENA skip that spares
those claims never engaged on D: its gate excluded the Form A shadow (R12),
which D always runs ("R12 HOST-SHADOW ACTIVE role=host"; 0 lines of
``arena_full_skipped`` in any D log, 52 in one P log).

THE FIX, two parts, both only where the wait is futile:
 B. the skip engages under R12 too; the refusals R12 mirrors to its workers
    are recorded by the skip itself, per skipped node, as the claim would have.
 A. a flush round that issued nothing, left every un-backed node refused by
    the arena AND saw no write land since the round before (in-flight count
    not lower) ends the loop (marker ``#1470 FLUSH-PUBLISH ARENA-FULL-STOP``).
    Round 0 never stops (nothing to compare with): the metal's sleep leg ran
    0/2/3/0/0 issued with in_flight 16 -> 3, an all-refused round 0 followed by
    rounds that published 5 nodes (review flushfast-1012 finding 1). The writes
    are not dropped.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import logging
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.mem_cache import form_a_host_shadow
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

from test_sweep_full_arena_skip_1007 import (
    CHAIN,
    _chains,
    _DirectPool,
    _tree,
)

N = 2 * CHAIN          # un-backed nodes of the fixture tree
IN_FLIGHT = 48         # the metal's stuck store writes (in_flight_after=48)


class _Sched:
    """The scheduler attributes ``flush_cache`` reads on its way to the verdict;
    the real ``Scheduler.flush_cache`` runs on it (it is borrowed, as the other
    weg2 flush tests do)."""

    def __init__(self, tree, *, writes_in_flight=0):
        tree.ongoing_backup = {i: object() for i in range(writes_in_flight)}
        self.tree_cache = tree
        self.enable_hierarchical_cache = True
        self.running_batch = types.SimpleNamespace(is_empty=lambda: True, reqs=[])
        self.chunked_req = None
        self.anchor_tails = None
        self.waiting_queue = []
        # not idle -> the flush answers False after the loop and never resets
        self.group_idle_verdict = lambda tp_group_verdict=False: (False, "model")
        self.idle_blockers = lambda: ["model"]
        self.flush_cache = sched_mod.Scheduler.flush_cache.__get__(self)


def _ack_all(self, write_back=False):
    """The write-through join of the model: every device->host copy acked."""
    self.ongoing_write_through.clear()


def _flush(sched, pool, *, role=None, skip=True, on_wait=None):
    """``on_wait(tree)``: what the wait of one loop round lets happen (a write
    lands, room frees); called after the write-through ack, once per wait."""
    from sglang.srt.environ import envs

    def _writing_check(self, write_back=False):
        _ack_all(self, write_back)
        if on_wait is not None:
            on_wait(self)

    recorded = []
    with envs.SGLANG_WEG2_ENABLE_SWEEP_FULL_ARENA_SKIP.override(skip), \
            mock.patch.object(UnifiedRadixCache, "_weg2_direct_pool", lambda self: pool), \
            mock.patch.object(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                              lambda self, node, write_back=False: True), \
            mock.patch.object(form_a_host_shadow, "role", lambda: role), \
            mock.patch.object(UnifiedRadixCache, "writing_check", _writing_check), \
            mock.patch.object(form_a_host_shadow, "record_state",
                              lambda tree, node, why="": recorded.append((node.id, why))):
        sched.flush_cache(empty_cache=False, zero_kv=False)
    return recorded


class TestTheFullArenaFlushLoop(CustomTestCase):
    """The metal's shape: N un-backed nodes, a full arena, writes in flight, R12 host."""

    def test_the_repro_one_futile_claim_per_round_not_one_per_node(self):
        """RED on c59d8bda8d: 4 rounds x N claims (= 24), the break only at
        round 4. GREEN: two rounds (round 0, then round 1 shows no write landed)
        x ONE claim each."""
        sched = _Sched(_tree(), writes_in_flight=IN_FLIGHT)
        pool = _DirectPool(room=())
        _flush(sched, pool, role="host")
        self.assertEqual(len(pool.claims), 2, "claims = the ARENA-DROPs of the metal")

    def test_the_loop_ends_after_the_first_stuck_round_with_the_named_marker(self):
        sched = _Sched(_tree(), writes_in_flight=IN_FLIGHT)
        sweeps = []
        real = UnifiedRadixCache.publish_unbacked_sweep

        def counting(self, *a, **kw):
            out = real(self, *a, **kw)
            sweeps.append(dict(out))
            return out

        with mock.patch.object(UnifiedRadixCache, "publish_unbacked_sweep", counting), \
                self.assertLogs(sched_mod.logger.name, level=logging.INFO) as cm:
            _flush(sched, _DirectPool(room=()), role="host")
        self.assertEqual(len(sweeps), 2, "round 0 + the round that proves it stuck, not four")
        for sw in sweeps:
            self.assertEqual(sw["arena_refused"], sw["unbacked"])
            self.assertEqual(sw["pending"], IN_FLIGHT)
        marker = [m for m in cm.output if "FLUSH-PUBLISH ARENA-FULL-STOP" in m]
        self.assertEqual(len(marker), 1, cm.output)
        self.assertIn("round=1", marker[0])
        self.assertTrue(any("#1470 FLUSH-PUBLISH issued=0" in m for m in cm.output))

    def test_the_writes_in_flight_are_not_dropped_or_cancelled(self):
        """The loop stops WAITING; it does not touch what is in flight."""
        tree = _tree()
        sched = _Sched(tree, writes_in_flight=IN_FLIGHT)
        _flush(sched, _DirectPool(room=()), role="host")
        self.assertEqual(len(tree.ongoing_backup), IN_FLIGHT)
        self.assertEqual(tree.cache_controller.copied, [], "nothing was issued, nothing aborted")

    def test_the_same_shape_without_the_form_a_shadow_ends_after_two_rounds_too(self):
        sched = _Sched(_tree(), writes_in_flight=IN_FLIGHT)
        pool = _DirectPool(room=())
        _flush(sched, pool, role=None)
        self.assertEqual(len(pool.claims), 2)

    def test_the_r12_records_are_the_ones_the_claims_made(self):
        """Exactness for the shadow: one sweep with the skip, one without --
        the same (node, why) records, in the same order. (Round count aside:
        compared at ONE sweep each, the loop's own stop removed from the test.)"""
        out = {}
        for skip in (False, True):
            tree = _tree()
            pool = _DirectPool(room=())
            from sglang.srt.environ import envs

            recorded = []
            with envs.SGLANG_WEG2_ENABLE_SWEEP_FULL_ARENA_SKIP.override(skip), \
                    mock.patch.object(UnifiedRadixCache, "_weg2_direct_pool", lambda self: pool), \
                    mock.patch.object(UnifiedRadixCache, "_mamba_write_through_pin_admissible",
                                      lambda self, node, write_back=False: True), \
                    mock.patch.object(form_a_host_shadow, "role", lambda: "host"), \
                    mock.patch.object(form_a_host_shadow, "record_state",
                                      lambda t, node, why="": recorded.append((node.id, why))):
                stats = tree.publish_unbacked_sweep(max_issue=256)
            nodes = sorted(n.id for n in tree._collect_all_nodes() if n is not tree.root_node)
            base = nodes[0]
            out[skip] = ([(i - base, w) for i, w in recorded], stats["refused"], stats["unbacked"],
                         len(pool.claims))
        self.assertEqual(out[True][0], out[False][0])
        self.assertEqual(out[True][1:3], out[False][1:3])
        self.assertEqual(out[False][3], N, "off: one claim per un-backed head chain node")
        self.assertEqual(out[True][3], 1)
        # the parent rule's shape: every node's recursion re-asks its chain's top
        # (arena_claim on the top), then parent_unbacked on each node below it:
        # per chain of CHAIN nodes 1 + 2 + .. + CHAIN refusals, CHAIN - 1 .. 0 of them below the top
        whys = [w for _, w in out[True][0]]
        self.assertEqual(whys.count("arena_claim"), N)
        self.assertEqual(whys.count("parent_unbacked"), 2 * sum(range(CHAIN)))


class TestTheLoopStaysWhereWaitingCanHelp(CustomTestCase):
    @staticmethod
    def _counting():
        sweeps = []
        real = UnifiedRadixCache.publish_unbacked_sweep

        def counting(self, *a, **kw):
            out = real(self, *a, **kw)
            sweeps.append(dict(out))
            return out

        return sweeps, counting

    def test_arena_full_in_round_0_writes_land_room_frees_the_loop_issues(self):
        """The metal's sleep leg (D.log 610486 -> 610738 -> 610804): round 0 is
        ALL arena-refused (issued=0, in_flight 16), the wait lets writes land
        (16 -> 3) and frees room, round 1 issues. Review finding 1: the first
        version of the stop ended the loop at round 0 and those nodes were lost
        at the reset. The loop must go on and publish them."""
        tree = _tree()
        (_, second) = _chains(tree)
        pool = _DirectPool(room=())
        sched = _Sched(tree, writes_in_flight=16)

        def on_wait(t):
            if not getattr(t, "_landed", False):
                t._landed = True
                for k in list(t.ongoing_backup)[:13]:     # 16 -> 3
                    del t.ongoing_backup[k]
                pool.room |= {n.hash_value[0] for n in second}

        sweeps, counting = self._counting()
        with mock.patch.object(UnifiedRadixCache, "publish_unbacked_sweep", counting), \
                self.assertLogs(sched_mod.logger.name, level=logging.INFO) as cm:
            _flush(sched, pool, role="host", on_wait=on_wait)
        self.assertEqual(sweeps[0]["issued"], 0)
        self.assertEqual(sweeps[0]["arena_refused"], sweeps[0]["unbacked"], "round 0 all arena-refused")
        self.assertEqual(sweeps[1]["issued"], CHAIN, "round 1 published what the landed writes made room for")
        self.assertLess(sweeps[1]["pending"], 16, "the landed writes show in the count")
        stops = [m for m in cm.output if "ARENA-FULL-STOP" in m]
        self.assertTrue(all("round=0" not in m and "round=1" not in m for m in stops), stops)

    def test_writes_landing_keep_the_loop_going_even_before_room_frees(self):
        """Progress is the in-flight count falling: 48 -> 28 in the first wait
        keeps round 1 alive, round 2 (28 -> 28) is the stuck one."""
        tree = _tree()
        sched = _Sched(tree, writes_in_flight=IN_FLIGHT)
        waits = []

        def on_wait(t):
            waits.append(len(t.ongoing_backup))
            if len(waits) == 1:
                for k in list(t.ongoing_backup)[:20]:
                    del t.ongoing_backup[k]

        sweeps, counting = self._counting()
        with mock.patch.object(UnifiedRadixCache, "publish_unbacked_sweep", counting), \
                self.assertLogs(sched_mod.logger.name, level=logging.INFO) as cm:
            _flush(sched, _DirectPool(room=()), role="host", on_wait=on_wait)
        self.assertEqual([s["pending"] for s in sweeps], [IN_FLIGHT, IN_FLIGHT - 20, IN_FLIGHT - 20])
        marker = [m for m in cm.output if "ARENA-FULL-STOP" in m]
        self.assertEqual(len(marker), 1, cm.output)
        self.assertIn("round=2", marker[0])

    def test_an_arena_with_room_is_unchanged(self):
        """'Arena has room': every node is backed, no skip, no marker, one sweep."""
        tree = _tree()
        sched = _Sched(tree, writes_in_flight=0)
        pool = _DirectPool(room={n.hash_value[0] for c in _chains(tree) for n in c})
        sweeps = []
        real = UnifiedRadixCache.publish_unbacked_sweep

        def counting(self, *a, **kw):
            out = real(self, *a, **kw)
            sweeps.append(dict(out))
            return out

        with mock.patch.object(UnifiedRadixCache, "publish_unbacked_sweep", counting), \
                self.assertLogs(sched_mod.logger.name, level=logging.INFO) as cm:
            _flush(sched, pool, role="host")
        self.assertEqual(sweeps[0]["issued"], N)
        self.assertEqual(pool.asked, [])
        self.assertFalse(any("ARENA-FULL-STOP" in m for m in cm.output))

    def test_a_refusal_that_is_not_the_arena_keeps_the_patience_loop(self):
        """Pin-budget refusals free as write-throughs ack: not arena_refused, so
        the loop keeps its four rounds exactly as before."""
        tree = _tree()
        sched = _Sched(tree, writes_in_flight=IN_FLIGHT)
        sweeps = []

        def refusing(self, node, write_back=False, kv_only_if_mamba_refused=False):
            return 0     # no arena_claim reason is ever set

        real = UnifiedRadixCache.publish_unbacked_sweep

        def counting(self, *a, **kw):
            out = real(self, *a, **kw)
            sweeps.append(dict(out))
            return out

        with mock.patch.object(UnifiedRadixCache, "publish_unbacked_sweep", counting), \
                mock.patch.object(UnifiedRadixCache, "write_backup", refusing):
            _flush(sched, _DirectPool(room=()), role="host")
        self.assertEqual(len(sweeps), 4)
        self.assertNotIn("arena_refused", sweeps[0])

    def test_one_issued_node_is_not_a_stuck_round(self):
        """A round that issued something goes on (room for the second chain only)."""
        tree = _tree()
        (_, second) = _chains(tree)
        pool = _DirectPool(room={n.hash_value[0] for n in second})
        sched = _Sched(tree, writes_in_flight=IN_FLIGHT)
        sweeps = []
        real = UnifiedRadixCache.publish_unbacked_sweep

        def counting(self, *a, **kw):
            out = real(self, *a, **kw)
            sweeps.append(dict(out))
            return out

        with mock.patch.object(UnifiedRadixCache, "publish_unbacked_sweep", counting):
            _flush(sched, pool, role="host")
        self.assertEqual(sweeps[0]["issued"], CHAIN)
        self.assertGreaterEqual(len(sweeps), 2, "the first round issued: the loop continues")


if __name__ == "__main__":
    unittest.main()
