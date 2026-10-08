"""X-SUM-PRICE (ed42d79b5f) -- "Flip unveraendert" for the NF tree (Auftrag 1503).

test_dual_fixes_flip_unchanged_1003.py lives on the 27B line only and drives dual modules
(dual_anchor_release, dual_d_kv_stage, ...) this tree does not carry, and X-SUM-PRICE is not
a dual fix. This is the NF counterpart for that one fix. What it proves:

  1. Below X the take is the identity: ``_sum_priced_take`` returns every entry, oldest first,
     the same objects, and keeps nothing back -- and ``Front._asr_queued_short_to_d`` then
     moves exactly what the pre-fix body moved (differential against that body's filter).
  2. Above X the ONLY change is that the youngest overflow stays queued: what is moved is a
     subset of what the pre-fix body moved; moved + kept is a partition (nothing lost/doubled).
  3. The gate is unchanged: not serving / not awake D / not admitting -> returns [] and touches
     nothing, ``_sum_priced_take`` is not even called; ineligible entries never reach it.
  4. The flip itself is untouched: the hand-over writes only queue, _ready_for_d, the two
     counters and ``d_direct``; every other attribute (state, awake, admit_d, t_awake, ...)
     is the same afterwards.
  5. Reach (source, ast): ``_sum_priced_take`` has ONE call site, in ``_asr_queued_short_to_d``;
     that one has ONE call site, in ``_arrival_seat_step``; that one has ONE call site, the
     D-awake ARRIVAL-SEAT branch of the serving loop, under ``_asr.enabled()``.

One difference is real and documented (test_unsorted_queue_is_taken_oldest_first): the take
sorts by ``t_arrive``; the pre-fix body kept the queue order. With a queue in arrival order
(the normal case) both are the same.
"""

import ast
import collections
import inspect
import random
import types
import unittest
from unittest import mock

from flliper.srt.pdflip import front

X = 12288


def _p(rid, uncached, t, **kw):
    d = dict(rid=rid, est_uncached=uncached, t_arrive=t, d_eligible=True, intake_stalled=False,
             leg1_done=False, reroutes=0, x_requeues=0, p_only=False, x_deferred=False, d_direct=False)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _pre_fix_moved(live_q, x):
    """The body ed42d79b5f replaced: every eligible entry, queue order, no sum."""
    return [p for p in live_q
            if p.d_eligible and not p.intake_stalled and not p.leg1_done and not p.reroutes
            and not p.x_requeues and not p.p_only and not p.x_deferred and 0 <= int(p.est_uncached or 0) <= x]


class _Fake:
    def __init__(self, queue, ready=(), *, admit_d=True, state="serving", awake="D"):
        self.admit_d, self.state, self.awake = admit_d, state, awake
        self.x_split = False
        self.tp_prefill_max_tokens = X
        self.t_awake = 123.0
        self.queue = collections.deque(queue)
        self._ready_for_d = list(ready)
        self.counters = collections.Counter()
        self.synced = 0

    def _sync_batch_gate(self):
        self.synced += 1

    def _x_band_floor(self):
        return 4096

    def _d_holds_work(self):
        return False


def _hand(fake, live=None):
    return front.Front._asr_queued_short_to_d(fake, list(fake.queue) if live is None else live, 10.0)


class TheTakeIsTheIdentityBelowX(unittest.TestCase):
    def test_no_overflow_returns_every_entry_in_order_and_keeps_none(self):
        es = [_p("a", 25, 1.0), _p("b", 400, 2.0), _p("c", 3000, 3.0)]
        taken, kept = front._sum_priced_take(es, carried=0, limit=X)
        self.assertEqual(len(taken), 3)
        for got, want in zip(taken, es):
            self.assertIs(got, want)
        self.assertEqual(kept, [])

    def test_a_sum_of_exactly_x_is_no_overflow(self):
        es = [_p("a", X - 100, 1.0), _p("b", 100, 2.0)]
        taken, kept = front._sum_priced_take(es, carried=0, limit=X)
        self.assertEqual([p.rid for p in taken], ["a", "b"])
        self.assertEqual(kept, [])

    def test_one_token_over_x_keeps_only_the_youngest(self):
        es = [_p("a", X - 100, 1.0), _p("b", 101, 2.0)]
        taken, kept = front._sum_priced_take(es, carried=0, limit=X)
        self.assertEqual([p.rid for p in taken], ["a"])
        self.assertEqual([p.rid for p in kept], ["b"])

    def test_the_input_list_is_not_mutated(self):
        es = [_p("b", 5, 2.0), _p("a", 5, 1.0)]
        snap = list(es)
        front._sum_priced_take(es, carried=0, limit=X)
        self.assertEqual(len(es), len(snap))
        for got, want in zip(es, snap):
            self.assertIs(got, want)

    def test_empty_input(self):
        self.assertEqual(front._sum_priced_take([], carried=5, limit=X), ([], []))

    def test_random_queues_partition_and_match_pre_fix_below_x(self):
        rng = random.Random(1503)
        for _ in range(300):
            n = rng.randint(0, 8)
            es = [_p("r%d" % i, rng.randint(0, 3000), float(i)) for i in range(n)]
            carried = rng.choice([0, 0, 500, 5000])
            taken, kept = front._sum_priced_take(es, carried=carried, limit=X)
            self.assertEqual(sorted(id(p) for p in taken + kept), sorted(id(p) for p in es))  # partition
            self.assertEqual([p.t_arrive for p in taken], sorted(p.t_arrive for p in taken))   # oldest first
            if carried + sum(p.est_uncached for p in es) <= X:                                 # below X: identity
                self.assertEqual([id(p) for p in taken], [id(p) for p in es])
                self.assertEqual(kept, [])
            else:                                                                               # above X: kept is non-empty
                self.assertTrue(kept)


class TheHandOverMatchesThePreFixBody(unittest.TestCase):
    def test_below_x_moves_exactly_what_the_pre_fix_body_moved(self):
        rng = random.Random(15031)
        for _ in range(300):
            n = rng.randint(0, 9)
            qs = []
            for i in range(n):
                qs.append(_p("r%d" % i, rng.randint(0, 1400), float(i),
                             d_eligible=rng.random() < 0.8, p_only=rng.random() < 0.1,
                             reroutes=int(rng.random() < 0.1), x_deferred=rng.random() < 0.1))
            want = _pre_fix_moved(qs, X)
            fake = _Fake(qs)
            moved = _hand(fake)
            self.assertEqual([id(p) for p in moved], [id(p) for p in want])
            self.assertEqual([p.rid for p in fake.queue], [p.rid for p in qs if p not in want])
            self.assertEqual([id(p) for p in fake._ready_for_d], [id(p) for p in want])
            self.assertEqual(fake.counters["arrival_seat_queue_sum_kept"], 0)
            self.assertEqual(fake.counters["arrival_seat_queue_to_d"], len(want))

    def test_above_x_moved_is_a_subset_and_the_rest_stays_queued_in_order(self):
        rng = random.Random(15032)
        saw_overflow = False
        for _ in range(300):
            qs = [_p("r%d" % i, rng.randint(2000, X), float(i)) for i in range(rng.randint(2, 7))]
            want = _pre_fix_moved(qs, X)
            fake = _Fake(qs)
            moved = _hand(fake)
            self.assertTrue({id(p) for p in moved} <= {id(p) for p in want})
            left = [p.rid for p in fake.queue]
            self.assertEqual(left, [p.rid for p in qs if p not in moved])  # untouched order of the rest
            self.assertEqual(len(moved) + len(left), len(qs))
            self.assertEqual(fake.counters["arrival_seat_queue_sum_kept"], len(want) - len(moved))
            saw_overflow = saw_overflow or len(moved) < len(want)
        self.assertTrue(saw_overflow)

    def test_unsorted_queue_is_taken_oldest_first(self):
        """The one real difference to the pre-fix body (it kept queue order): documented, not hidden."""
        qs = [_p("young", 10, 5.0), _p("old", 10, 1.0)]
        fake = _Fake(qs)
        moved = _hand(fake)
        self.assertEqual([p.rid for p in moved], ["old", "young"])
        self.assertEqual([p.rid for p in _pre_fix_moved(qs, X)], ["young", "old"])


class TheGateIsUnchanged(unittest.TestCase):
    def _assert_untouched(self, fake, qs):
        with mock.patch.object(front, "_sum_priced_take", side_effect=AssertionError("called behind the gate")):
            self.assertEqual(_hand(fake), [])
        self.assertEqual([p.rid for p in fake.queue], [p.rid for p in qs])
        self.assertEqual(fake._ready_for_d, [])
        self.assertEqual(+fake.counters, collections.Counter())
        self.assertEqual(fake.synced, 0)
        self.assertTrue(all(p.d_direct is False for p in qs))

    def test_not_admitting_not_serving_not_awake_d_touches_nothing(self):
        for kw in ({"admit_d": False}, {"state": "sleeping"}, {"state": "flipping"}, {"awake": "P"}):
            qs = [_p("a", 10, 1.0), _p("b", 20, 2.0)]
            self._assert_untouched(_Fake(qs, **kw), qs)

    def test_ineligible_entries_never_reach_the_take_and_stay_queued(self):
        bad = [_p("not-elig", 10, 1.0, d_eligible=False), _p("stalled", 10, 2.0, intake_stalled=True),
               _p("leg1", 10, 3.0, leg1_done=True), _p("rer", 10, 4.0, reroutes=1),
               _p("xrq", 10, 5.0, x_requeues=1), _p("ponly", 10, 6.0, p_only=True),
               _p("xdef", 10, 7.0, x_deferred=True), _p("big", X + 1, 8.0)]
        good = _p("good", 10, 9.0)
        fake = _Fake(bad + [good])
        seen = []
        real = front._sum_priced_take

        def spy(entries, **kw):
            seen.append([p.rid for p in entries])
            return real(entries, **kw)

        with mock.patch.object(front, "_sum_priced_take", side_effect=spy):
            moved = _hand(fake)
        self.assertEqual(seen, [["good"]])
        self.assertEqual([p.rid for p in moved], ["good"])
        self.assertEqual([p.rid for p in fake.queue], [p.rid for p in bad])
        self.assertTrue(all(p.d_direct is False for p in bad))


class TheFlipStateIsUntouched(unittest.TestCase):
    def test_only_queue_ready_counters_and_d_direct_are_written(self):
        qs = [_p("8-32", 12177, 1.0), _p("8-34", 12281, 2.0), _p("8-37", 12213, 3.0)]
        fake = _Fake(qs)
        before = {k: v for k, v in vars(fake).items() if k not in ("queue", "_ready_for_d", "counters", "synced")}
        moved = _hand(fake)
        after = {k: v for k, v in vars(fake).items() if k not in ("queue", "_ready_for_d", "counters", "synced")}
        self.assertEqual(before, after)  # state, awake, admit_d, t_awake, x_split, tp_prefill_max_tokens
        self.assertEqual(set(fake.counters), {"arrival_seat_queue_sum_kept", "arrival_seat_d_prefill",
                                              "arrival_seat_queue_to_d"})
        self.assertEqual(fake.synced, 1 if moved else 0)
        self.assertEqual([p.rid for p in moved], ["8-32"])
        self.assertEqual([p.d_direct for p in qs], [True, False, False])

    def test_an_empty_hand_over_does_not_sync_the_batch_gate(self):
        waiting = _p("w", 9000, 0.5, d_direct=True)
        nxt = _p("n", 5000, 1.0)
        fake = _Fake([nxt], ready=[waiting])
        self.assertEqual(_hand(fake), [])
        self.assertEqual(fake.synced, 0)
        self.assertEqual(fake.counters["arrival_seat_queue_sum_kept"], 1)


class TheReachIsOneCallChain(unittest.TestCase):
    """Source-level: where the fix can run at all (front.py of THIS tree)."""

    @classmethod
    def setUpClass(cls):
        cls.src = inspect.getsource(front)
        cls.tree = ast.parse(cls.src)

    def _sites(self, name):
        """(enclosing function, line) of every call of ``name`` (plain or attribute)."""
        out = []

        def visit(node, fn):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = node.name
            if isinstance(node, ast.Call):
                f = node.func
                called = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                if called == name:
                    out.append((fn, node.lineno))
            for ch in ast.iter_child_nodes(node):
                visit(ch, fn)

        visit(self.tree, None)
        return out

    def test_sum_priced_take_is_called_only_from_the_queued_short_hand_over(self):
        self.assertEqual([f for f, _ in self._sites("_sum_priced_take")], ["_asr_queued_short_to_d"])

    def test_the_hand_over_is_called_only_from_the_arrival_seat_step(self):
        self.assertEqual([f for f, _ in self._sites("_asr_queued_short_to_d")], ["_arrival_seat_step"])

    def test_the_arrival_seat_step_is_called_once_in_the_arrival_seat_branch(self):
        sites = self._sites("_arrival_seat_step")
        self.assertEqual(len(sites), 1)
        lines = self.src.splitlines()
        line = sites[0][1]
        window = "\n".join(lines[max(0, line - 6):line])
        self.assertIn("if _asr.enabled():", window)  # the ARRIVAL-SEAT decision site while D is awake

    def test_the_counter_is_written_only_by_the_hand_over(self):
        writers = [i + 1 for i, ln in enumerate(self.src.splitlines()) if "arrival_seat_queue_sum_kept" in ln]
        self.assertEqual(len(writers), 1)


if __name__ == "__main__":
    unittest.main()
