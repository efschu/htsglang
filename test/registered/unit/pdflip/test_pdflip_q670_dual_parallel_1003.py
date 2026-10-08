# SPDX-License-Identifier: Apache-2.0
"""Q-670 DUAL-PARALLEL (27B dual y8w, boot dkr27bnvfp4dual1mpsleepbar1fs10031623, 006cd2955e).

METAL (front + P log, 03.10.):

  16:37:20,515  front LEG1-INPUT-IDS pdflip-0-144 (91477 tokens) -- the only leg in flight;
                the queue was empty for 20 ms, then 145..152 arrived behind it
  16:37:20      P PP0 ``P-KV PP0 WAIT rid=pdflip-0-144 tokens=91477: a card is short``
  16:39:43      P PP0 ``GRANT rid=pdflip-0-144 ... after 90254 waits over 142.9 s``
  16:39:48      front LEG1 pdflip-0-145 (25 tokens) -- 148 s after its arrival
  16:40:22      the OpenWebUI request pdflip-0-151 (6366 tokens): PDFLIP-CLIENT-GONE wait_s=142.9 -> 503
  16:39:49..    DUAL RESUME-WAIT up to 26.8 s with a paused 91k head: every new pass held

RED on 006cd2955e:
  * the drain pool sleeps on its one leg while a free slot and queued work wait;
  * a paused head in RESUME-WAIT holds every short request behind it;
  * PP0 has no age bound: a waiting head is overtaken for ever (and nothing names it).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import tempfile
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_kv_ledger as K  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as S  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MIB = 1 << 20


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _front():
    f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="dual", store_dir="/tmp",
                prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                flip_min_work_tokens=1, dual_layout=True)

    async def rpc(g, path, body, timeout):
        return 200, "{}"

    f.rpc = rpc
    return f


def _pending(rid, uncached, paused=0, age=0.0):
    fut = asyncio.get_event_loop().create_future()
    p = F.Pending(rid, "/generate", {}, "x", time.time() - age, fut, est_prompt=uncached,
                  est_uncached=uncached)
    p.dual_paused_n = paused
    return p


class PoolWakesForArrivals(CustomTestCase):
    """(1) front: a leg that waits on P (its card grant) must not hide the queue."""

    def test_an_arrival_behind_a_held_leg_is_dispatched_before_that_leg_ends(self):
        async def run():
            q = collections.deque(["long"])
            started = {}
            t0 = time.monotonic()

            async def one(item):
                started[item] = time.monotonic() - t0
                await asyncio.sleep(1.0 if item == "long" else 0.0)
                return item

            async def arrive():
                await asyncio.sleep(0.05)
                q.append("short")                       # 20 ms after the long one went out

            asyncio.ensure_future(arrive())
            await F._p_drain_pool(q, 2, one, lambda p: None, lambda: True, poll_s=0.05,
                                  dual_wake=True)
            return started

        started = _run(run())
        self.assertIn("short", started)
        self.assertLess(started["short"], 0.5, "the short arrival waited for the long leg (metal: 148 s)")


class FrontShortBypass(CustomTestCase):
    """(2)+(3) front: a paused head blocks only itself; a short request overtakes a young long head."""

    def _ledgers_with_p_committed(self):
        root = tempfile.mkdtemp(prefix="wkv670")
        paths = [os.path.join(root, "card%d" % i) for i in range(3)]
        for pth in paths:
            d, p = K.CardKvLedger(pth, "D"), K.CardKvLedger(pth, "P")
            d.contribute(2000 * MIB, committed=2000 * MIB)
            p.contribute(1000 * MIB)
            d.release(1000 * MIB)
        K.CardKvLedger(paths[0], "P").request(700 * MIB)      # a P stage still commits: RESUME-WAIT
        return paths

    def test_a_short_request_passes_a_paused_head_in_resume_wait(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = self._ledgers_with_p_committed()
            head = _pending("pdflip-0-148", 91638, paused=1)
            short = _pending("pdflip-0-151", 2270)
            f.queue = collections.deque([head, short])
            return f, f._dual_dispatch_held(), list(f.queue)

        f, held, order = _run(run())
        self.assertFalse(held, "the paused head held the short request (metal: RESUME-WAIT 26.8 s)")
        self.assertEqual([p.rid for p in order], ["pdflip-0-151", "pdflip-0-148"])
        self.assertEqual(f.counters["dual_short_bypass"], 1)

    def test_a_paused_head_without_a_short_behind_still_waits(self):
        async def run():
            f = _front()
            f.dual_kv_ledgers = self._ledgers_with_p_committed()
            f.queue = collections.deque([_pending("pdflip-0-148", 91638, paused=1),
                                         _pending("pdflip-0-149", 90000)])
            return f._dual_dispatch_held()

        self.assertTrue(_run(run()), "a long request must not jump a paused head (D priority)")

    def test_short_first_past_a_young_long_head_but_not_past_an_aged_one(self):
        async def run():
            f = _front()
            f.queue = collections.deque([_pending("pdflip-0-144", 91477, age=1.0), _pending("pdflip-0-145", 25)])
            moved = f._dual_short_reorder(head_blocked=False)
            first = [p.rid for p in f.queue]
            g = _front()
            g.queue = collections.deque([_pending("pdflip-0-144", 91477, age=600.0), _pending("pdflip-0-145", 25)])
            aged = g._dual_short_reorder(head_blocked=False)
            return moved, first, aged

        moved, first, aged = _run(run())
        self.assertTrue(moved)
        self.assertEqual(first, ["pdflip-0-145", "pdflip-0-144"])
        self.assertFalse(aged, "an aged long head must not be overtaken any more (no starvation)")

    def test_flip_form_never_reorders(self):
        async def run():
            f = F.Front(prefill="http://p", decode="http://d", awake="P", tag="flip", store_dir="/tmp",
                        prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                        flip_min_work_tokens=1, dual_layout=False)
            f.queue = collections.deque([_pending("a", 91477), _pending("b", 25)])
            return f._dual_short_reorder(head_blocked=False), [p.rid for p in f.queue]

        moved, order = _run(run())
        self.assertFalse(moved)
        self.assertEqual(order, ["a", "b"])


class _Pp0Fixture(CustomTestCase):
    """Temp card ledgers: D holds all but 96 MiB per card; P PP0 decides the group grant."""

    DUAL_ENV = "1"

    def setUp(self):
        self._env = os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT")
        if self.DUAL_ENV is None:
            os.environ.pop("FLLIPER_PDFLIP_DUAL_LAYOUT", None)
        else:
            os.environ["FLLIPER_PDFLIP_DUAL_LAYOUT"] = self.DUAL_ENV
        S._reset_wait_log()
        self.root = tempfile.mkdtemp(prefix="wkv670p")
        self.paths = []
        for r, per in enumerate((2048, 4096, 6144)):
            pth = os.path.join(self.root, "card%d" % r)
            K.CardKvLedger(pth, "D").contribute(4096 * MIB, committed=0)
            K.CardKvLedger(pth, "P").contribute(0)
            # D holds all but 96 MiB of each card: a 4096-token grant fits (<= 24 MiB),
            # the 91k head (94208 tokens: 184 / 368 / 552 MiB) never does
            K.CardKvLedger(pth, "D").request(4000 * MIB)
            with open(os.path.join(self.root, "stage%d" % r), "w") as f:
                json.dump({"ledger": pth, "step": 4096, "top": 196608,
                           "bytes": [k * 4096 * per for k in range(196608 // 4096 + 1)]}, f)
            self.paths.append(pth)
        self._orig = (S.stage_file, S._actor, S._now)
        S.stage_file = lambda tag, r, root="/dev/shm": os.path.join(self.root, "stage%d" % r)
        actor = types.SimpleNamespace(page=64, _committed=0, map_granted=lambda lvl, charged=None: None)
        S._actor = lambda sched: actor
        self.t = [1000.0]
        S._now = lambda: self.t[0]

    def tearDown(self):
        S.stage_file, S._actor, S._now = self._orig
        S._reset_wait_log()
        if self._env is None:
            os.environ.pop("FLLIPER_PDFLIP_DUAL_LAYOUT", None)
        else:
            os.environ["FLLIPER_PDFLIP_DUAL_LAYOUT"] = self._env

    def _sched(self):
        return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), waiting_queue=[],
                                     _pdflip_store_held={})

    def _req(self, rid, tokens):
        return types.SimpleNamespace(rid=rid, origin_input_ids=list(range(tokens)), _dual_grant_untold=None)


class Pp0GrantBypass(_Pp0Fixture):
    """(1) P, PP0: a grant that fits now is taken past a waiting head; past the age the head is the head."""

    def test_a_small_grant_passes_the_waiting_head_and_is_named(self):
        sched = self._sched()
        head = self._req("pdflip-0-144", 91477)
        self.assertEqual(S.pp0_grant(sched, head), 0, "precondition: the 91k head does not fit")
        head._dual_kv_wait = True
        sched._pdflip_store_held[head.rid] = head
        self.t[0] += 5.0
        small = self._req("pdflip-0-145", 25)
        with self.assertLogs(S.logger.name, level="INFO") as cm:
            lvl = S.pp0_grant(sched, small)
        self.assertGreater(lvl, 0, "the 25-token request waited behind the head (metal: 148 s)")
        self.assertTrue(any("P-KV GRANT-BYPASS rid=pdflip-0-145" in m and "past=pdflip-0-144" in m
                            for m in cm.output), cm.output)

    def test_past_the_age_newcomers_wait_for_the_head(self):
        sched = self._sched()
        head = self._req("pdflip-0-144", 91477)
        S.pp0_grant(sched, head)
        head._dual_kv_wait = True
        sched._pdflip_store_held[head.rid] = head
        self.t[0] += 61.0                                   # > FLLIPER_PDFLIP_DUAL_BYPASS_HEAD_AGE_S (60)
        self.assertEqual(S.pp0_grant(sched, self._req("pdflip-0-160", 25)), 0,
                         "an aged head was overtaken -- it could starve for ever")

    def test_a_younger_waiter_never_blocks_the_older_head(self):
        sched = self._sched()
        head = self._req("pdflip-0-144", 91477)
        S.pp0_grant(sched, head)
        head._dual_kv_wait = True
        sched._pdflip_store_held[head.rid] = head
        self.t[0] += 61.0
        young = self._req("pdflip-0-160", 25)
        S.pp0_grant(sched, young)                            # held by the age rule
        young._dual_kv_wait = True
        sched._pdflip_store_held[young.rid] = young
        self.t[0] += 61.0                                   # both waited past the age now
        older = S._older_waits(sched, "pdflip-0-144")
        self.assertEqual(older, [], "the younger waiter counted against the head: two aged waiters deadlock")


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class FlipLayoutUnchanged(CustomTestCase):
    """User order 03.10.: loops happened ONLY on dual NVFP4 -- the flip form (27B INT8
    row authority, NF) must work byte for byte as on 006cd2955e. Green on base AND fix."""

    def test_flip_drain_pool_does_not_poll_without_a_ro_slot(self):
        async def run():
            q = collections.deque(["long"])
            started = {}
            t0 = time.monotonic()

            async def one(item):
                started[item] = time.monotonic() - t0
                await asyncio.sleep(0.6 if item == "long" else 0.0)
                return item

            async def arrive():
                await asyncio.sleep(0.05)
                q.append("short")

            asyncio.ensure_future(arrive())
            await F._p_drain_pool(q, 2, one, lambda p: None, lambda: True, poll_s=0.05)
            return started

        started = _run(run())
        self.assertIn("short", started)
        self.assertGreaterEqual(started["short"], 0.5,
                                "flip form: the pool woke for an arrival (base sleeps on the leg)")

    def test_flip_front_queue_order_is_untouched(self):
        async def run():
            f = F.Front(prefill="http://p", decode="http://d", awake="P", tag="flip", store_dir="/tmp",
                        prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                        flip_min_work_tokens=1, dual_layout=False)
            f.queue = collections.deque([_pending("a", 91477), _pending("b", 25, paused=1)])
            reorder = getattr(f, "_dual_short_reorder", None)
            moved = [reorder(head_blocked=hb) for hb in (True, False)] if reorder else []
            return moved, [p.rid for p in f.queue]

        moved, order = _run(run())
        self.assertFalse(any(moved))
        self.assertEqual(order, ["a", "b"])


class FlipPp0GrantUnchanged(_Pp0Fixture):
    """pp0_grant without FLLIPER_PDFLIP_DUAL_LAYOUT=1: no older-wait refusal, no GRANT-BYPASS line."""

    DUAL_ENV = None

    def test_newcomer_is_tried_as_on_base_and_nothing_is_named(self):
        sched = self._sched()
        head = self._req("pdflip-0-144", 91477)
        self.assertEqual(S.pp0_grant(sched, head), 0)
        head._dual_kv_wait = True
        sched._pdflip_store_held[head.rid] = head
        self.t[0] += 61.0                                   # past the dual age bound
        h = _ListHandler()
        S.logger.addHandler(h)
        old = S.logger.level
        S.logger.setLevel(logging.DEBUG)
        try:
            lvl = S.pp0_grant(sched, self._req("pdflip-0-160", 25))
        finally:
            S.logger.removeHandler(h)
            S.logger.setLevel(old)
        self.assertGreater(lvl, 0, "flip form: a newcomer was held behind an aged head (base grants it)")
        self.assertFalse(any("GRANT-BYPASS" in m for m in h.lines), h.lines)
