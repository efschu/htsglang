"""P-POOL-SHARED (rc29 pphase 21.10.; SOLL R1a, R1u, R11): the P overlap plan books a leg that continues an in-flight
leg's LOADED prefix only beyond that prefix.

Metal rc27 boot ..._893d09fa3f_1021_023225 (front.log / P.log):
* 02:37:02.5 the pool dispatched pdflip-0-17 (149584), -0-20 (63484), -0-18 (111411), -0-19 (132448) = 456927 booked;
  pdflip-0-21 (prompt 149741, u 2029) did not fit 524288 and was held (``L2-READ-AHEAD HINT rid=pdflip-0-21`` 02:37:04.9,
  ``pool_holds=98``) until pdflip-0-17 was SERVED at 02:37:25.7 -- 23.2 s.
* pdflip-0-21 continues pdflip-0-17: P matched it at ``P-FORK-CUT TOLD rid=pdflip-0-21 fork=149568 src=tree``; the
  in-flight -0-17 had loaded its store prefix 147712 (``#TK ABS-TOLD rid=pdflip-0-17 head=18432 span=129280
  told=147712``). Its real addition to P's pool was 2029 tokens; P's ``full token usage`` read 0.17-0.43 meanwhile.
* the head stayed the head, so pdflip-1-23 (needs P, 02:37:19.8) waited behind it too, and P's commit at 02:37:20-25
  ran with queue 0 (COMMIT-WARTEN-P-OHNE-QUEUE 02:37:25 pass=5817 commit=5447).
"""

import asyncio
import collections
import types
import unittest

from flliper.srt.pdflip import front, phase_policy


class ContinuationCredit(unittest.TestCase):
    def test_the_metal_pair(self):
        # 0-21 shares >= 0-17's whole loaded prefix (147712) -> credited 147712
        self.assertEqual(phase_policy.p_continuation_credit([(149584, 147712)]), 147712)

    def test_a_shorter_common_prefix_credits_nothing(self):
        # session siblings: common 18486 < the leg's loaded 147712 -> the front knows no anchor there
        self.assertEqual(phase_policy.p_continuation_credit([(18486, 147712)]), 0)

    def test_largest_wins_and_junk_is_ignored(self):
        pairs = [(18904, 132448), (200000, 184576), (149584, 147712), (5, 0), (0, 64), (None, None)]
        self.assertEqual(phase_policy.p_continuation_credit(pairs), 184576)

    def test_empty(self):
        self.assertEqual(phase_policy.p_continuation_credit([]), 0)


def _item(rid, cost, giver=None):
    return types.SimpleNamespace(rid=rid, cost=cost, giver=giver)


def _run_pool(items, budget, shared, finish_order):
    """Every leg waits on its own event; ``finish_order`` releases them one per loop turn. Returns (dispatch order,
    stats, inflight-token readings at each dispatch)."""

    async def main():
        evs = {it.rid: asyncio.Event() for it in items}
        q = collections.deque(items)
        started = []
        stats = {}

        async def one(it):
            started.append(it.rid)
            await evs[it.rid].wait()
            return it

        async def releaser():
            for rid in finish_order:
                for _ in range(5):
                    await asyncio.sleep(0)
                evs[rid].set()

        rel = asyncio.ensure_future(releaser())
        await front._p_drain_pool(q, 16, one, lambda _p: None, lambda: True,
                                  cost=lambda it: it.cost, budget=budget, stats=stats, shared=shared)
        await rel
        return started, stats

    return asyncio.run(main())


def _shared_by_giver(it, others):
    for o in others:
        if it.giver is not None and o.rid == it.giver[0]:
            return it.giver[1], o
    return 0, None


class PoolBooksTheContinuationBeyondTheGiver(unittest.TestCase):
    METAL = [_item("0-17", 149584), _item("0-20", 63484), _item("0-18", 111411), _item("0-19", 132448),
             _item("0-21", 149741, giver=("0-17", 147712))]

    def test_metal_held_without_the_credit(self):
        # the old booking: 456927 + 149741 > 524288 -> 0-21 is held until a leg finishes
        started, stats = _run_pool(self.METAL, 524288, None, ["0-20", "0-17", "0-18", "0-19", "0-21"])
        self.assertEqual(started[:4], ["0-17", "0-20", "0-18", "0-19"])
        self.assertGreater(stats["pool_holds"], 0)
        self.assertEqual(stats["shared_dispatched"], 0)

    def test_metal_dispatched_at_once_with_the_credit(self):
        # 456927 + (149741 - 147712) = 458956 <= 524288 -> all five go in the first round, no hold
        started, stats = _run_pool(self.METAL, 524288, _shared_by_giver, ["0-20", "0-17", "0-18", "0-19", "0-21"])
        self.assertEqual(started, ["0-17", "0-20", "0-18", "0-19", "0-21"])
        self.assertEqual(stats["pool_holds"], 0)
        self.assertEqual(stats["shared_dispatched"], 1)
        self.assertEqual(stats["shared_credit"], 147712)
        self.assertEqual(stats["peak_tokens"], 456927 + 2029)

    def test_the_credit_ends_when_the_giver_leaves_first(self):
        # giver g finishes first: the taker's booking returns to its full cost, so a third leg that fits only
        # against the credited booking is held until the taker is gone (danger direction: never under-count P)
        items = [_item("g", 100), _item("t", 100, giver=("g", 90)), _item("x", 95)]
        # budget 200: g(100) + t(10) = 110; x(95) fits (205? no: 110 + 95 = 205 > 200) -> held at once anyway;
        # use budget 210: x dispatched beside g and t only while the credit stands
        started, stats = _run_pool(items, 210, _shared_by_giver, ["g", "t", "x"])
        self.assertEqual(started, ["g", "t", "x"])
        self.assertEqual(stats["peak_tokens"], 205)

    def test_revert_holds_the_next_leg(self):
        # g and t in flight (100 + 10); g leaves -> t books 100; y (150) must now wait for t: 100 + 150 > 210
        items = [_item("g", 100), _item("t", 100, giver=("g", 90)), _item("y", 150)]
        async def main():
            evs = {it.rid: asyncio.Event() for it in items}
            q = collections.deque(items)
            started, stats = [], {}

            async def one(it):
                started.append((it.rid, tuple(sorted(r for r, e in evs.items() if not e.is_set()))))
                await evs[it.rid].wait()
                return it

            async def releaser():
                for _ in range(5):
                    await asyncio.sleep(0)
                evs["g"].set()          # the giver leaves first
                for _ in range(5):
                    await asyncio.sleep(0)
                evs["t"].set()
                for _ in range(5):
                    await asyncio.sleep(0)
                evs["y"].set()

            rel = asyncio.ensure_future(releaser())
            await front._p_drain_pool(q, 16, one, lambda _p: None, lambda: True, cost=lambda it: it.cost,
                                      budget=210, stats=stats, shared=_shared_by_giver)
            await rel
            return started, stats

        started, stats = asyncio.run(main())
        order = [r for r, _ in started]
        self.assertEqual(order, ["g", "t", "y"])
        # y started only after t (and g) had finished: with the reverted booking 100 + 150 > 210
        y_unfinished_at_start = dict(started)["y"]
        self.assertNotIn("t", y_unfinished_at_start)
        self.assertGreater(stats["pool_holds"], 0)

    def test_a_credit_error_keeps_the_full_cost(self):
        def boom(_it, _others):
            raise RuntimeError("x")
        started, stats = _run_pool(self.METAL, 524288, boom, ["0-20", "0-17", "0-18", "0-19", "0-21"])
        self.assertGreater(stats["pool_holds"], 0)
        self.assertEqual(stats["shared_dispatched"], 0)

    def test_a_giver_not_in_flight_credits_nothing(self):
        items = [_item("a", 400000), _item("b", 200000, giver=("zz", 190000))]
        started, stats = _run_pool(items, 524288, _shared_by_giver, ["a", "b"])
        self.assertEqual(stats["shared_dispatched"], 0)
        self.assertGreater(stats["pool_holds"], 0)


class FrontReadsIdsAndDepth(unittest.TestCase):
    """The front's reader: ids from ``_set_ids`` (front tokenizer), the giver's loaded prefix from its priced depth."""

    def _front(self):
        f = front.Front.__new__(front.Front)
        f.counters = collections.Counter()
        f.epoch = 1
        base = list(range(147712))
        f._set_ids = collections.OrderedDict()
        f._set_ids["pdflip-0-17"] = base + [7] * 1872            # prompt 149584
        f._set_ids["pdflip-0-21"] = base + [7] * 1856 + [9] * 173  # prompt 149741, continues 0-17 past 147712
        f._set_ids["pdflip-0-19"] = base[:18880] + [5] * (132448 - 18880)   # session sibling only
        f._set_depth = collections.OrderedDict([("pdflip-0-17", 147712), ("pdflip-0-19", 131328)])
        return f

    @staticmethod
    def _p(rid, est_prompt, est_uncached):
        return types.SimpleNamespace(rid=rid, est_prompt=est_prompt, est_uncached=est_uncached,
                                     leg1_prompt_tokens=0, skip_leg1=False)

    def test_metal_credit_and_giver(self):
        f = self._front()
        p = self._p("pdflip-0-21", 149741, 2029)
        g17, g19 = self._p("pdflip-0-17", 149584, 1872), self._p("pdflip-0-19", 132448, 1120)
        cr, gv = front.Front._p_pool_shared(f, p, [g19, g17])
        self.assertEqual((cr, gv.rid), (147712, "pdflip-0-17"))
        self.assertEqual(f.counters["p_pool_shared_credit"], 1)

    def test_sibling_only_credits_nothing(self):
        f = self._front()
        p = self._p("pdflip-0-21", 149741, 2029)
        cr, gv = front.Front._p_pool_shared(f, p, [self._p("pdflip-0-19", 132448, 1120)])
        self.assertEqual((cr, gv), (0, None))

    def test_ids_not_kept_credit_nothing(self):
        f = self._front()
        p = self._p("pdflip-9-99", 149741, 2029)
        cr, gv = front.Front._p_pool_shared(f, p, [self._p("pdflip-0-17", 149584, 1872)])
        self.assertEqual((cr, gv), (0, None))

    def test_depth_falls_back_to_the_entry_estimate(self):
        f = self._front()
        f._set_depth = collections.OrderedDict()   # no book entry: est_prompt - est_uncached = 147712
        p = self._p("pdflip-0-21", 149741, 2029)
        cr, _gv = front.Front._p_pool_shared(f, p, [self._p("pdflip-0-17", 149584, 1872)])
        self.assertEqual(cr, 147712)


class Wired(unittest.TestCase):
    def test_the_p_drain_passes_the_reader(self):
        import inspect
        src = inspect.getsource(front.Front)
        self.assertIn("shared=(lambda p, others: Front._p_pool_shared(self, p, others))", src)


if __name__ == "__main__":
    unittest.main()
