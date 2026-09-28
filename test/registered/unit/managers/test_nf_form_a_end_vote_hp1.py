"""HP1 (rc12z20, boot dkrnfh91dprsavisnoadoptstbar1dauer09281220 on
3a86888ba5, D 12:30:32-12:31:36, rid weg2-8-2): on a Form A D group the #580
prefetch vote compares span ENDS, not lengths from different starts.

MEASURED: TP0 matched 0 (``[#904 match-census] ... refusers=MambaComponent:
32704``) and asked 33600 tokens; TP1/TP2 matched 32704 on their byteless
shadow tree and asked 896. H99's workers abstained in the SPAN pair only, so
the LENGTH MIN was 896: ``#915 PREFETCH TRUNCATED need=33600 got=896
lost=32704 cut_rank=1`` on TP0 while TP1 had ``available=319552`` -- deferred
as host_pool_shortfall, re-voted on every pass for 64 s (638 group
truncations), D decoding at bs1 with the request queued.

Driven through the REAL ``UnifiedRadixCache.prefetch_from_storage`` on three
simulated ranks (the H99 harness). RED on ebeccfd190: no ``span_base``, no
END pack. GREEN with HP1: the host registers its whole span, each worker its
own share of it, and the completion MIN, the stall witness and the retry
back-off agree.
"""

from __future__ import annotations

import types
import unittest

import test_nf_form_a_prefetch_span_h99 as h99

from sglang.srt.managers import tp_match_floor as m
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.match_refusal_census import PREFETCH_GATE_COUNTS
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

END = 33600
PROMPT = list(range(END))


def _intake(bases, rid, *, span_base=True, end=END):
    """One intake pass: every rank enters the vote with its own start."""
    group = h99.MockGlooGroup()
    caches = {}
    prompt = list(range(end))

    def _rank(r):
        caches[r] = c = h99._carrier(r, group)
        c._hp1_note_end_vote = types.MethodType(UnifiedRadixCache._hp1_note_end_vote, c)
        kw = {"span_base": bases[r]} if span_base else {}
        c.prefetch_from_storage(
            rid, h99._host_node(), prompt[bases[r]:], last_hash=None, prefix_keys=None, **kw
        )
        return h99._registered_len(c, rid)

    with h99._Env():
        results, errors = h99.run_ranks(_rank)
    return caches, results, errors, group


RC12Z20 = {0: 0, 1: 32704, 2: 32704}


class TestRc12z20HostReadsItsWholeSpan(unittest.TestCase):
    def test_no_group_cut_host_33600_workers_896(self):
        before = PREFETCH_GATE_COUNTS.get("host_pool_truncated_group", 0)
        _c, results, errors, group = _intake(RC12Z20, "weg2-8-2")
        self.assertEqual(errors, {}, errors)
        self.assertEqual(group.errors, [])
        self.assertEqual(
            results, {0: (33600, 33600), 1: (896, 896), 2: (896, 896)},
            "rc12z20: the workers' 896 capped TP0's 33600 (lost=32704)",
        )
        self.assertEqual(
            PREFETCH_GATE_COUNTS.get("host_pool_truncated_group", 0), before,
            "a Form A read whose spans END together is no group truncation",
        )

    def test_the_completion_min_keeps_the_hosts_read(self):
        caches, _r, errors, _g = _intake(RC12Z20, "weg2-8-2")
        self.assertEqual(errors, {})
        bases = {r: caches[r]._hp1_end_base_by_rid["weg2-8-2"] for r in caches}
        self.assertEqual(bases, RC12Z20)
        completed = {0: 33600, 1: 896, 2: 896}
        packs = {r: urc._hp1_end_pack(bases[r], completed[r], completed[r], urc._ANCHOR_ABSTAIN)
                 for r in caches}
        group = [min(p[i] for p in packs.values()) for i in range(3)]
        got = {r: urc._hp1_end_unpack(bases[r], group[0]) for r in caches}
        self.assertEqual(got, completed, "each rank keeps its own whole share")
        self.assertEqual(urc._hp1_end_unpack(bases[0], group[2]), urc._ANCHOR_ABSTAIN)

    def test_a_short_host_read_leaves_the_workers_nothing_past_it(self):
        bases = RC12Z20
        completed = {0: 20000, 1: 896, 2: 896}
        packs = [urc._hp1_end_pack(bases[r], completed[r], completed[r], 16384 if r == 0 else urc._ANCHOR_ABSTAIN)
                 for r in range(3)]
        g = [min(p[i] for p in packs) for i in range(3)]
        self.assertEqual([urc._hp1_end_unpack(bases[r], g[0]) for r in range(3)], [20000, 0, 0])
        self.assertEqual(urc._hp1_end_unpack(bases[0], g[2]), 16384)

    def test_worker_short_tail_no_longer_votes_the_host_down(self):
        # A new turn of 128 tokens: the worker's own span is below the
        # prefetch threshold, which was a group decline (vote_negative) --
        # the host then prefilled 33k from scratch.
        bases = {0: 0, 1: END - 128, 2: END - 128}
        _c, results, errors, _g = _intake(bases, "r-tail")
        self.assertEqual(errors, {}, errors)
        self.assertEqual(results, {0: (END, END), 1: (128, 128), 2: (128, 128)}, results)

    def test_rc9o_shape_workers_register_to_the_hosts_end(self):
        bases = {0: h99.TP0_PREFIX, 1: h99.WORKER_PREFIX, 2: h99.WORKER_PREFIX}
        _c, results, errors, _g = _intake(bases, "weg2-33-33", end=h99.MATCH_END)
        self.assertEqual(errors, {}, errors)
        self.assertEqual(results[0], (320, 320))
        self.assertEqual(results[1], (6720, 6720), "worker ends where the host ends")

    def test_without_span_base_the_vote_is_the_old_one(self):
        _c, results, errors, _g = _intake(RC12Z20, "r-old", span_base=False)
        self.assertEqual(errors, {})
        self.assertEqual(set(results.values()), {(896, 896)}, results)


class TestPieces(unittest.TestCase):
    def test_end_base_only_on_form_a(self):
        with h99._Env():
            self.assertEqual(m.form_a_end_base(32704), 32704)
            self.assertIsNone(m.form_a_end_base(None))
        with h99._Env(switch=False):
            self.assertIsNone(m.form_a_end_base(32704))

    def test_host_base_vote_worker_abstains(self):
        with h99._Env():
            h99._ROLE.worker = True
            try:
                self.assertEqual(m.form_a_host_base_vote(32704), m.PREFETCH_SPAN_ABSTAIN)
                self.assertTrue(m.form_a_null_tier_span(128, 32704))
                self.assertFalse(m.form_a_null_tier_span(0, 32704))
            finally:
                h99._ROLE.worker = False
            self.assertEqual(m.form_a_host_base_vote(0), 0)
            self.assertFalse(m.form_a_null_tier_span(128, 0))

    def test_scheduler_passes_the_span_start(self):
        import inspect

        from sglang.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler._prefetch_kvcache)
        self.assertIn("span_base=int(_matched_len)", src)


class _Sched:
    def __init__(self):
        self.observed = 0

    def _weg2_note_prefetch_progress(self, req):
        self.observed += 1
        return "stalled"


def _due(sched, req):
    from sglang.srt.managers.scheduler import _hp1_group_retry_due

    return _hp1_group_retry_due(sched, req)


class TestRetryBackoff(unittest.TestCase):
    def test_group_mark_retries_on_a_pass_backoff(self):
        s = _Sched()
        req = types.SimpleNamespace(rid="weg2-8-2", prefetch_deferred="host_pool_shortfall",
                                    prefetch_defer_since=1.0)
        due = [n for n in range(1, 65) if _due(s, req)]
        self.assertEqual(due, [1, 2, 4, 8, 16, 32, 48, 64])
        self.assertEqual(s.observed, 64 - len(due), "a skipped pass still observes")

    def test_fresh_mark_restarts_and_rate_arm_is_untouched(self):
        s = _Sched()
        req = types.SimpleNamespace(rid="r", prefetch_deferred="store_prefix_short",
                                    prefetch_defer_since=1.0)
        [_due(s, req) for _ in range(5)]
        req.prefetch_defer_since = 2.0
        self.assertTrue(_due(s, req), "a fresh mark re-issues at once")
        req.prefetch_deferred = "rate_limited"
        self.assertTrue(all(_due(s, req) for _ in range(10)))

    def test_backoff_is_a_pure_function_of_the_pass_count(self):
        # Rank-uniformity: three ranks with the same marks skip the same passes.
        seqs = []
        for _rank in range(3):
            s = _Sched()
            req = types.SimpleNamespace(rid="r", prefetch_deferred="host_pool_shortfall",
                                        prefetch_defer_since=float(_rank))  # rank-local clock
            seqs.append([_due(s, req) for _ in range(100)])
        self.assertEqual(seqs[0], seqs[1])
        self.assertEqual(seqs[0], seqs[2])


if __name__ == "__main__":
    unittest.main()
