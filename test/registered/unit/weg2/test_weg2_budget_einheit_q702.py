"""Q-702 BUDGET-EINHEIT: the SEAT-AGE KV verdict asks the ADDER's question.

NF y9nf, boot ...10032328, D 23:33:28Z (Auftrag 1030, Fall A). weg2-4-14 (raw extend 189248,
uncached 384 behind a host hit) got ``add_result_NO_TOKEN`` from the adder, while the verdict
(``kv_displace_would_fit``) logged ``need=189248 free=197376 ... -> fits free, nobody leaves``
(basis: raw extend against available + evictable). The adder charges the LIFETIME price
(extend + max_new + page + mamba gap) against ``rem_total_tokens`` (pool + evictable - the running
batch's decode reserves - group floor - commitment ledger): two budgets for one fit question, so the
verdict displaced for nothing in one pass and refused to displace in the next (782 displacements).

Pinned: the adder keeps the numbers of its first lifetime refusal (``lifetime_refusal``), the
scheduler hands them to the verdict (``note_adder_refusal``), and the verdict decides on THEM --
price + 1 against the budget plus what the victims give back (KV rows AND decode reserve), carried
by the pool drift -- with ``basis=adder`` in the log. Without a view the legacy reading stands.
"""

import os
import types
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.schedule_policy import (  # noqa: E402
    CLIP_MAX_NEW_TOKENS,
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as DPR  # noqa: E402
from sglang.srt.weg2 import d_seats as DS  # noqa: E402

VIEW_ATTR = "_weg2_sa_no_token_view"   # literal: the base has no constant, the red must be behavioural
OLDER = "weg2-4-14"
VICTIM = "weg2-4-16"
EXTEND = 189248        # the specimen's need=
FREE = 197376          # the specimen's free=
MAX_NEW = 2048
PAGE = 64
RESERVE = 2048         # one running request's decode reserve (ratio 1.0)


def _req(rid, span, prefix=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * span, output_ids=[],
                                 prefix_indices=[0] * prefix)


def _sched(running, waiting, pool=FREE, view=None, no_token=OLDER, group_min=None):
    batch = types.SimpleNamespace(reqs=list(running), released=[], spec_algorithm=None)
    batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))
    batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [batch.reqs[i] for i in keep_indices])
    sched = types.SimpleNamespace(
        waiting_queue=list(waiting), server_args=types.SimpleNamespace(max_running_requests=6),
        _weg2_sa_no_token=no_token, calls=[],
        tree_cache=types.SimpleNamespace(page_size=64, evictable_size=lambda: 0),
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: pool))
    setattr(sched, VIEW_ATTR, view)
    sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)

    def gm(flags):
        sched.calls.append(list(flags))
        return group_min(flags) if group_min else flags

    sched._weg2_group_min_flags = gm
    return sched, batch


def _view(price, budget, pool=FREE, reserve=None):
    return {"rid": OLDER, "price": price, "budget": budget, "pool": pool,
            "reserve": {VICTIM: RESERVE, "weg2-4-15": RESERVE} if reserve is None else reserve}


def _run(fn):
    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
         mock.patch.object(DPR, "seat_cap", lambda s: None):
        return fn()


class VerdictAsksTheAddersQuestion(unittest.TestCase):
    def test_raw_extend_fits_free_but_the_adder_refused_so_the_victim_leaves(self):
        """The specimen's 'fits free, nobody leaves': raw 189248 <= 197376, yet the adder's
        price (+ max_new + page) is above its budget (pool - the running reserves). The
        victim's KV rows and its reserve close the gap: the older one is let in."""
        price = EXTEND + MAX_NEW + PAGE                       # 191360
        budget = FREE - 4 * RESERVE                           # 189184 < price
        victim = _req(VICTIM, 14781)
        sched, batch = _sched([_req("weg2-4-15", 1000), victim], [_req(OLDER, EXTEND)],
                              view=_view(price, budget))
        with self.assertLogs(DPR.logger, level="INFO"):
            got = _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(got, VICTIM)
        self.assertEqual(sched.calls, [[True]])               # one group MIN, as before

    def test_raw_extend_fits_the_victim_but_the_adder_price_does_not_so_nobody_leaves(self):
        """Victim for nothing: raw 189248 fits free 100000 + 100000 (victim), the adder's price
        does not fit budget + victim rows + reserve."""
        price = EXTEND + MAX_NEW + PAGE
        victim = _req(VICTIM, 100000)
        sched, batch = _sched([_req("weg2-4-15", 1000), victim], [_req(OLDER, EXTEND)], pool=100000,
                              view=_view(price, budget=80000, pool=100000))
        with self.assertLogs(DPR.logger, level="INFO") as cap:
            got = _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertIsNone(got)
        self.assertTrue(any("basis=adder" in r for r in cap.output))
        self.assertEqual(batch.released, [])

    def test_the_adder_refuses_at_price_ge_budget_so_the_fit_needs_price_plus_one(self):
        old = _req(OLDER, EXTEND)
        victim = _req(VICTIM, 1000)
        sched, _ = _sched([victim], [old])
        free_after = 1000 + RESERVE                           # what the victim gives back
        v = _view(price=5000, budget=5000 - free_after, reserve={VICTIM: RESERVE})
        self.assertFalse(DPR.kv_displace_would_fit(sched, OLDER, [victim], view=v))   # price == budget': refused
        v = _view(price=5000, budget=5000 - free_after + 1, reserve={VICTIM: RESERVE})
        self.assertTrue(DPR.kv_displace_would_fit(sched, OLDER, [victim], view=v))

    def test_the_pool_drift_since_the_refusal_is_carried(self):
        """20000 rows freed since the refusal: the view's budget alone says 'a victim must leave',
        the carried budget says it fits already -> nobody leaves."""
        old = _req(OLDER, EXTEND)
        victim = _req(VICTIM, 5000)
        v = _view(price=EXTEND + MAX_NEW + PAGE, budget=EXTEND, pool=FREE)
        sched, _ = _sched([victim], [old], pool=FREE + 20000)
        self.assertFalse(DPR.kv_displace_would_fit(sched, OLDER, [victim], view=v))   # k == 0
        sched, _ = _sched([victim], [old], pool=FREE)
        self.assertTrue(DPR.kv_displace_would_fit(sched, OLDER, [victim], view=v))    # k == 1

    def test_a_foreign_or_missing_view_keeps_the_legacy_reading(self):
        old = _req(OLDER, EXTEND)
        victim = _req(VICTIM, 14781)
        sched, _ = _sched([victim], [old])
        foreign = dict(_view(10**9, 0), rid="weg2-9-9")
        with self.assertLogs(DPR.logger, level="INFO") as cap:
            self.assertFalse(DPR.kv_displace_would_fit(sched, OLDER, [victim], view=foreign))
        self.assertIn("basis=legacy", cap.output[0])
        self.assertIn("fits free, nobody leaves", cap.output[0])

    def test_the_view_is_consumed_with_the_refusal(self):
        sched, batch = _sched([_req(VICTIM, 1000)], [_req(OLDER, 50)], view=_view(10, 0))
        _run(lambda: DPR.displace_for_age(sched, batch))
        self.assertIsNone(getattr(sched, VIEW_ATTR))
        self.assertIsNone(sched._weg2_sa_no_token)


class NoteAdderRefusal(unittest.TestCase):
    def test_it_keeps_the_adders_numbers_and_the_reserves_of_the_running(self):
        adder = SimpleNamespace(lifetime_refusal=(OLDER, 191360, 189184),
                                released_by_leaving=lambda r: {"a": 7, "b": 9}[r.rid])
        sched, _ = _sched([], [])
        run = SimpleNamespace(reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")])
        _run(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), run))
        v = getattr(sched, VIEW_ATTR)
        self.assertEqual((v["rid"], v["price"], v["budget"]), (OLDER, 191360, 189184))
        self.assertEqual(v["reserve"], {"a": 7, "b": 9})
        self.assertEqual(v["pool"], FREE)

    def test_another_rids_refusal_or_none_sets_no_view(self):
        sched, _ = _sched([], [])
        run = SimpleNamespace(reqs=[])
        _run(lambda: DPR.note_adder_refusal(
            sched, SimpleNamespace(lifetime_refusal=("other", 1, 1)), SimpleNamespace(rid=OLDER), run))
        self.assertIsNone(getattr(sched, VIEW_ATTR))
        _run(lambda: DPR.note_adder_refusal(sched, SimpleNamespace(), SimpleNamespace(rid=OLDER), run))
        self.assertIsNone(getattr(sched, VIEW_ATTR))

    def test_off_a_d_park_group_nothing_is_kept(self):
        sched, _ = _sched([], [])
        adder = SimpleNamespace(lifetime_refusal=(OLDER, 5, 4), released_by_leaving=lambda r: 1)
        with mock.patch.object(DS, "d_flip_park_active", lambda: False):
            DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), SimpleNamespace(reqs=[]))
        self.assertIsNone(getattr(sched, VIEW_ATTR))

    def test_the_scheduler_hands_the_adders_numbers_over_where_it_keeps_the_rid(self):
        from sglang.srt.managers import scheduler as S

        src = open(S.__file__).read()
        i = src.index("self._weg2_sa_no_token = str(req.rid)")
        self.assertIn("note_adder_refusal(self, adder, req, running_batch)", src[i:i + 700])


def _tree_cache():
    tc = MagicMock()
    tc.supports_mamba.return_value = False
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.swa_evictable_size.return_value = 0
    tc.disable = False
    tc.uniform_avail_floor = None
    tc.inc_lock_ref.return_value = IncLockRefResult()
    tc.dec_lock_ref.return_value = DecLockRefResult()
    return tc


def _allocator(available):
    a = MagicMock()
    a.available_size.return_value = available
    a.full_available_size.return_value = available
    a.swa_available_size.return_value = 0
    return a


def _running(n):
    b = MagicMock()
    b.reqs = [SimpleNamespace(rid=f"run-{i}", output_ids=[], sampling_params=SimpleNamespace(max_new_tokens=RESERVE))
              for i in range(n)]
    return b


def _adder(available, running):
    return PrefillAdder(
        page_size=PAGE, tree_cache=_tree_cache(), token_to_kv_pool_allocator=_allocator(available),
        running_batch=running, new_token_ratio=1.0, rem_input_tokens=10**9, rem_chunk_tokens=None,
        num_mixed_decode_tokens=0, priority_scheduling_preemption_threshold=0,
    )


def _adder_req(fill):
    req = MagicMock(spec=Req)
    req.rid = OLDER
    req.priority = 0
    req.prefix_indices = []
    req.full_untruncated_fill_ids = list(range(fill))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=MAX_NEW, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


class TheAdderKeepsItsOwnNumbers(unittest.TestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_the_specimen_the_raw_extend_fits_the_pool_but_the_adder_refuses_and_says_why(self):
        adder = _adder(FREE, _running(4))
        self.assertIsNone(adder.lifetime_refusal)
        req = _adder_req(EXTEND)
        self.assertLess(EXTEND, FREE)                          # the verdict's 'fits free'
        res = adder.add_one_req(req, truncation_align_size=None)
        self.assertEqual(res, AddReqResult.NO_TOKEN)
        rid, price, budget = adder.lifetime_refusal
        self.assertEqual(rid, OLDER)
        self.assertEqual(price, EXTEND + MAX_NEW + PAGE)       # extend + max_new + page
        self.assertEqual(budget, FREE - 4 * min(RESERVE, CLIP_MAX_NEW_TOKENS))
        self.assertGreaterEqual(price, budget)

    def test_a_request_that_fits_leaves_no_refusal(self):
        adder = _adder(FREE, _running(0))
        res = adder.add_one_req(_adder_req(1000), truncation_align_size=None)
        self.assertNotEqual(res, AddReqResult.NO_TOKEN)
        self.assertIsNone(adder.lifetime_refusal)

    def test_what_a_leaving_request_gives_back_is_its_decode_reserve(self):
        adder = _adder(FREE, _running(2))
        r = SimpleNamespace(rid="x", output_ids=[1, 2, 3], sampling_params=SimpleNamespace(max_new_tokens=RESERVE))
        self.assertEqual(adder.released_by_leaving(r), RESERVE - 3)


if __name__ == "__main__":
    unittest.main()
