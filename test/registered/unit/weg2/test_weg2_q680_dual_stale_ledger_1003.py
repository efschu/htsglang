# SPDX-License-Identifier: Apache-2.0
"""Q-680 DUAL STALE LEDGER (27B dual y8w, boot dkr27bnvfp4dual1mpsleepbar1fs10031623, 006cd2955e).

METAL (P + front log, 03.10.), in this order:

  16:45:41  front P-PAUSE weg2-0-219 (+ abort of 218/219/220 on P)
  16:45:42  PP1/PP2 FOLLOWER-ABORT-APPLIED weg2-0-218 -> RELEASE 114688 tokens -> 0
  16:45:42  PP0 regrants the requeued 218 (8192) and 220 (49152); PP1/PP2
            MAPPED-BY-GRANT 49152 committed=201326592 / 301989888 B
  16:45:43  PP1/PP2 START-LOADING weg2-0-220 (39099 prefix rows), the request
            finishes in that pass (END-ANCHOR, finished=1)
  16:45:44  PP0 RELEASE 49152 -> 0; PP1/PP2 never release: lock census
            ongoing_lb=1 tracked_protected=39168 until 17:02 -- the load-back
            lock is dropped only by loading_check, which an idle stage never ran
  16:45:42.. front RESUME-WAIT rid=weg2-0-219 waited 905 s, per card
            [(0,0,0),(0,201326592,0),(0,301989888,0)] -- 0 requests for 15 min

RED on 006cd2955e:
  * an idle dual P follower never drains its load-back acks -> its release is
    held for ever, its card keeps "P committed";
  * the front's RESUME-WAIT is unbounded on a ledger whose only non-zero term is
    "P committed" while P has no leg in flight.
"""
from __future__ import annotations

import asyncio
import collections
import os
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import card_kv_ledger as K  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as S  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
ENV = ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


class _Tree:
    """The follower's tree as far as the idle release reads it: a load-back
    holds its rows protected until loading_check pops its ack."""

    def __init__(self):
        self.ongoing_load_back = {}
        self.ongoing_write_through = {}
        self.acks = []
        self.locked = 0

    def start_loading(self, node_id, rows):
        self.ongoing_load_back[node_id] = rows
        self.locked += rows
        self.acks.append(node_id)                 # the copy landed; the ack waits for a poll

    def loading_check(self):
        while self.acks:
            self.locked -= self.ongoing_load_back.pop(self.acks.pop(0))

    def flush_write_through_acks(self):
        pass

    def evictable_size(self):
        return 0

    def protected_size(self):
        return self.locked

    def evict(self, params):
        pass


class _Actor:
    def __init__(self):
        self.mapped_tokens = 0
        self._committed = 0
        self._phys_next = float("inf")            # no NVML read in a unit test
        self.ledger = None
        self.released = []

    def map_granted(self, tokens, charged=None):
        self.mapped_tokens = max(self.mapped_tokens, int(tokens))
        self._committed = self.mapped_tokens * 4096

    def release_all(self):
        n = self._committed
        self.released.append(n)
        self.mapped_tokens = 0
        self._committed = 0
        return n


class _EnvCase(CustomTestCase):
    DUAL = True

    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ENV}
        if self.DUAL:
            os.environ["SGLANG_WEG2_DUAL_LAYOUT"] = "1"
            os.environ["SGLANG_WEG2_GROUP"] = "P"
        else:
            for k in ENV:
                os.environ.pop(k, None)
        self._orig_actor = S._actor
        self.actor = _Actor()
        S._actor = lambda sched: self.actor

    def tearDown(self):
        S._actor = self._orig_actor
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _follower(self):
        tree = _Tree()
        sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1, pp_size=3),
                                      enable_hierarchical_cache=True, tree_cache=tree, forward_ct=1264)
        return sched, tree

    def _idle(self, sched):
        """The scheduler's on_idle order for a dual P follower: acks, then the release."""
        S.flush_acks_when_idle(sched)
        return S.on_idle(sched)


class FollowerIdleLoadAck(_EnvCase):
    """(a) the metal order: release, late grant, load-back, finish, idle -> the follower releases."""

    def test_release_then_late_grant_then_loadback_the_idle_follower_releases(self):
        sched, tree = self._follower()
        self.actor.map_granted(114688)
        self.assertEqual(self._idle(sched), 114688 * 4096, "precondition: the first release (16:45:42)")
        item = types.SimpleNamespace(rid="weg2-0-220")
        setattr(item, S.WIRE_DUAL_KV, 49152)
        S.on_told(sched, item)                     # the grant arrives after the release
        self.assertEqual(self.actor._committed, 49152 * 4096)
        tree.start_loading(540, 39099)             # 16:45:43 START-LOADING, the request finishes
        with self.assertLogs(S.logger.name, level="INFO") as cm:
            freed = self._idle(sched)
        self.assertEqual(freed, 49152 * 4096,
                         "the idle follower kept its grant (metal: 201/302 MB committed for 905 s)")
        self.assertEqual(self.actor._committed, 0)
        self.assertEqual(tree.ongoing_load_back, {})
        self.assertTrue(any("P-KV IDLE-LOAD-ACK pp_rank=1 drained=1 left=0" in m for m in cm.output), cm.output)

    def test_a_held_release_names_why(self):
        sched, tree = self._follower()
        self.actor.map_granted(49152)
        tree.locked = 4096                        # held by something that is not a load-back
        with self.assertLogs(S.logger.name, level="WARNING") as cm:
            self.assertEqual(self._idle(sched), 0)
        self.assertTrue(any("P-KV IDLE-HELD pp_rank=1 mapped=49152" in m and "protected=4096" in m
                            for m in cm.output), cm.output)


class FlipIdleUnchanged(_EnvCase):
    """Flip form (no dual env): the idle ack flush is not taken at all -- as on 006cd2955e."""

    DUAL = False

    def test_flush_acks_when_idle_is_inert_without_the_dual_layout(self):
        sched, tree = self._follower()
        tree.start_loading(540, 39099)
        self.assertFalse(S.flush_acks_when_idle(sched))
        self.assertEqual(list(tree.ongoing_load_back), [540], "the flip form drained a load-back ack")


def _front(dual=True):
    f = F.Front(prefill="http://p", decode="http://d", awake="D" if dual else "P", tag="dual",
                store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=dual)

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.rpc = rpc
    return f


def _pending(rid, uncached, paused=0):
    fut = asyncio.get_event_loop().create_future()
    p = F.Pending(rid, "/generate", {}, "x", time.time(), fut, est_prompt=uncached, est_uncached=uncached)
    p.dual_paused_n = paused
    return p


def _metal_ledgers(d_demand=False):
    """Card 0 clean, cards 1/2 with P committed 201326592 / 301989888 B (the metal rows)."""
    root = tempfile.mkdtemp(prefix="wkv680")
    paths = [os.path.join(root, "card%d" % i) for i in range(3)]
    for i, pth in enumerate(paths):
        K.CardKvLedger(pth, "D").contribute(2000 * MIB, committed=0)
        K.CardKvLedger(pth, "P").contribute(0)
    K.CardKvLedger(paths[1], "P").request(201326592)
    K.CardKvLedger(paths[2], "P").request(301989888)
    if d_demand:
        K.CardKvLedger(paths[1], "D").request(4000 * MIB)
    return paths


class FrontResumeStale(CustomTestCase):
    """(b) RESUME-WAIT is never unbounded on a stale 'P committed'."""

    def _held(self, f):
        return f._dual_resume_held()

    def test_metal_rows_with_p_idle_resume_after_the_stale_bound(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = _metal_ledgers()
            f.queue = collections.deque([_pending("weg2-0-219", 12288, paused=1), _pending("weg2-0-221", 90000)])
            first = self._held(f)
            f._dual_resume_stale_since = time.time() - 11.0       # > SGLANG_WEG2_DUAL_RESUME_STALE_S (10)
            with self.assertLogs(F.logger.name, level="WARNING") as cm:
                second = self._held(f)
            return f, first, second, cm.output

        f, first, second, out = _run(run())
        self.assertTrue(first, "a fresh 'P committed' is still waited for")
        self.assertFalse(second, "RESUME-WAIT stayed unbounded (metal: 905 s, 0 requests in 15 min)")
        self.assertTrue(any("DUAL RESUME-STALE-LEDGER rid=weg2-0-219" in m for m in out), out)
        self.assertEqual(f.counters["dual_resume_stale"], 1)

    def test_a_leg_in_flight_on_p_is_not_stale(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = _metal_ledgers()
            f.queue = collections.deque([_pending("weg2-0-219", 12288, paused=1)])
            f._dual_inflight["weg2-0-222"] = _pending("weg2-0-222", 4000)
            self._held(f)
            f._dual_resume_stale_since = time.time() - 60.0
            return self._held(f)

        self.assertTrue(_run(run()), "P is working: its committed bytes are real")

    def test_d_demand_is_not_stale(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = _metal_ledgers(d_demand=True)
            f.queue = collections.deque([_pending("weg2-0-219", 12288, paused=1)])
            self._held(f)
            f._dual_resume_stale_since = time.time() - 60.0
            return self._held(f)

        self.assertTrue(_run(run()), "D still asks for KV: the head waits")


if __name__ == "__main__":
    import unittest

    unittest.main()
