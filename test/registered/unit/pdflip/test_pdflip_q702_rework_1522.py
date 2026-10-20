"""Q-702 BUDGET-EINHEIT, Nacharbeit (Auftrag 1522) on top of 44d92050bf / review 1140.

Review 1140 found two things the first Q-702 commit does not cover:

1. FORM A (NF D; TP0 = attention host, TP1/TP2 = expert workers): only the host has a
   lifetime refusal of its own. The workers' gates say ADMIT (D.log 10032328 23:33:28Z,
   ``local_price=4544 local_budget=176587`` against ``host_price=193408 host_budget=176587``,
   64 of 64 WAIT lines ``local=ADMIT``), they take NO_TOKEN from the host via ``_fa_follow``
   and never reach ``_note_lifetime_refusal`` -> no view -> ``basis=legacy`` -> their
   ``fits free, nobody leaves`` vetoes the host's ``basis=adder`` in the group MIN: the fix
   does nothing on the form the case happened on. Fix: on a Form A group every rank builds
   the view from the GROUP's gate numbers (``form_a_admission_verdict(sink=...)`` ->
   ``_follow.last_verdict`` -> ``d_park_runtime._group_refusal``).
2. AGE: the view is read one pass after it was written; when ``admission()`` is not reached
   in between (the pass declines above it) the numbers are several passes old. Fix: the view
   carries the scheduler's pass count and is dropped (legacy basis) when older than one pass.

Each test names the mutation that turns it red (see the 1522 report).
"""

import ast
import inspect
import os
import re
import textwrap
import types
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers import tp_match_floor as TMF  # noqa: E402
from flliper.srt.managers.schedule_batch import Req  # noqa: E402
from flliper.srt.managers.schedule_policy import AddReqResult, PrefillAdder  # noqa: E402
from flliper.srt.managers.scheduler import Scheduler  # noqa: E402
from flliper.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    DecLockRefResult,
    IncLockRefResult,
)
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as DPR  # noqa: E402
from flliper.srt.pdflip import d_seats as DS  # noqa: E402

VIEW_ATTR = "_pdflip_sa_no_token_view"
PASS_ATTR = "_pdflip_sa_pass"
OLDER = "pdflip-4-14"
V_OLD = "pdflip-4-15"        # an older running request (not a victim of OLDER)
VICTIM = "pdflip-4-16"       # the youngest running request


# ---------------------------------------------------------------- scheduler / adder doubles
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


def _run_req(rid, kv, max_new):
    return SimpleNamespace(rid=rid, origin_input_ids=[0] * kv, output_ids=[],
                           sampling_params=SimpleNamespace(max_new_tokens=max_new))


def _batch(reqs):
    b = MagicMock()
    b.reqs = list(reqs)
    return b


def _adder(available, running, page=1, chunk=None):
    return PrefillAdder(
        page_size=page, tree_cache=_tree_cache(), token_to_kv_pool_allocator=_allocator(available),
        running_batch=running, new_token_ratio=1.0, rem_input_tokens=10**9, rem_chunk_tokens=chunk,
        num_mixed_decode_tokens=0, priority_scheduling_preemption_threshold=0,
    )


def _mock_req(fill, max_new, rid=OLDER, device_prefix=0):
    req = MagicMock(spec=Req)
    req.rid = rid
    req.priority = 0
    req.prefix_indices = list(range(device_prefix))
    req.full_untruncated_fill_ids = list(range(fill))
    req.output_ids = []
    req.sampling_params = SimpleNamespace(max_new_tokens=max_new, ignore_eos=False)
    req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
    req.retracted_stain = False
    req.finished.return_value = False
    req.needs_host_load_back.return_value = False
    req.host_hit_length = 0
    req.last_node = MagicMock()
    req.born_spilled = False
    req.born_spilled_deep = False
    return req


def _queued(rid, kv, prefix=0):
    return SimpleNamespace(rid=rid, origin_input_ids=[0] * kv, output_ids=[],
                           prefix_indices=[0] * prefix)


def _sched(running, waiting, pool, evictable=0, pass_n=0, group_min=None):
    batch = SimpleNamespace(reqs=list(running), released=[], spec_algorithm=None)
    batch.release_req = lambda idx, rem, sa, retain=False: batch.released.append((batch.reqs[idx].rid, retain))
    batch.filter_batch = lambda keep_indices: setattr(batch, "reqs", [batch.reqs[i] for i in keep_indices])
    sched = SimpleNamespace(
        waiting_queue=list(waiting), server_args=SimpleNamespace(max_running_requests=6),
        _pdflip_sa_no_token=None, calls=[],
        tree_cache=SimpleNamespace(page_size=64, evictable_size=lambda: evictable),
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: pool))
    setattr(sched, VIEW_ATTR, None)
    setattr(sched, PASS_ATTR, pass_n)
    sched._add_request_to_queue = lambda req, is_retracted=False: sched.waiting_queue.append(req)

    def gm(flags):
        sched.calls.append(list(flags))
        return group_min(flags) if group_min else flags

    sched._pdflip_group_min_flags = gm
    return sched, batch


def _in_d(fn):
    with mock.patch.object(DS, "d_flip_park_active", lambda: True), \
         mock.patch.object(DPR, "seat_cap", lambda s: None):
        return fn()


def _basis(logs):
    return [m.group(1) for line in logs.output for m in [re.search(r"basis=(\w+)", line)] if m]


# ------------------------------------------------------------------ Form A group (3 ranks)
class _Group:
    """Three ranks of one Form A group; the TP broadcast is a mailbox the host fills first
    (the same shape as test_nf_form_a_admission_follow_h105)."""

    def __init__(self):
        self.mailbox = {}

    def scheduler(self, tp_rank):
        s = SimpleNamespace(ps=SimpleNamespace(tp_size=3, pp_size=1),
                            tp_group=SimpleNamespace(rank=tp_rank, ranks=[0, 1, 2]), tp_cpu_group=None)

        def _exchange(site, payload):
            if payload is not None:
                self.mailbox[site] = payload
            return self.mailbox.get(site)

        s._form_a_tp_exchange = _exchange
        for name in ("_form_a_is_host", "_form_a_admission_follow_fn"):
            setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
        return s


def _install(adder, sched, tp_rank):
    with mock.patch.object(TMF, "form_a_follow_active", return_value=True), \
         mock.patch.object(TMF, "this_rank_follows", return_value=tp_rank != 0):
        adder.form_a_admission_follow = sched._form_a_admission_follow_fn()


FILL = 82081          # the older request's extend, as the host prices it (no device prefix)
HOST_HIT = 75264      # what a worker holds as device rows -> it prices 6817
MAX_NEW = 64
HOST_POOL = 82100     # raw extend 82081 <= pool: the legacy 'fits free'
WORKER_POOL = 90000
V_OLD_KV, VICTIM_KV = 1000, 6000


def _form_a_pass(*, pass_n=7, host_pool=HOST_POOL, host_prefix=0):
    """One admission pass of the older request on a 3-rank Form A group through the REAL
    PrefillAdder.add_one_req and the REAL Scheduler._form_a_admission_follow_fn, then the
    scheduler's NO_TOKEN-branch call ``note_adder_refusal`` per rank. Returns per-rank
    (result, adder, sched, batch)."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    group = _Group()
    out = {}
    for tp_rank, pool, prefix in ((0, host_pool, host_prefix), (1, WORKER_POOL, HOST_HIT),
                                  (2, WORKER_POOL, HOST_HIT)):
        running = [_run_req(V_OLD, V_OLD_KV, MAX_NEW), _run_req(VICTIM, VICTIM_KV, MAX_NEW)]
        adder = _adder(pool, _batch(running))
        _install(adder, group.scheduler(tp_rank), tp_rank)
        res = adder.add_one_req(_mock_req(FILL, MAX_NEW, device_prefix=prefix), truncation_align_size=None)
        sched, batch = _sched(running, [_queued(OLDER, FILL, prefix)], pool, pass_n=pass_n)
        if res == AddReqResult.NO_TOKEN:
            sched._pdflip_sa_no_token = OLDER
            _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), _batch(running)))
        out[tp_rank] = (res, adder, sched, batch)
    return out


class FormAEveryRankRunsTheAddersBasis(unittest.TestCase):
    """Review 1140 finding 1 (BLOCKING): the specimen's shape through the real adder and the
    real follow callable."""

    def test_the_gates_are_the_specimens_host_refuses_workers_admit_locally(self):
        got = _form_a_pass()
        host = got[0][1]
        self.assertEqual([got[r][0] for r in (0, 1, 2)], [AddReqResult.NO_TOKEN] * 3)
        # the host's own lifetime refusal exists; the workers' adders never wrote one
        self.assertIsNotNone(host.lifetime_refusal)
        self.assertIsNone(got[1][1].lifetime_refusal)
        self.assertIsNone(got[2][1].lifetime_refusal)
        price, budget = host.lifetime_refusal[1:]
        self.assertGreaterEqual(price, budget)

    def test_every_rank_holds_the_hosts_numbers_as_its_view(self):
        """RED without the group numbers (mutation M1: note_adder_refusal reads only the
        rank-local lifetime_refusal): TP1/TP2 have no view."""
        got = _form_a_pass()
        host_price, host_budget = got[0][1].lifetime_refusal[1:]
        views = {r: getattr(got[r][2], VIEW_ATTR) for r in (0, 1, 2)}
        for r in (0, 1, 2):
            self.assertIsNotNone(views[r], f"TP{r} has no view: it would read the legacy basis")
            self.assertEqual((views[r]["rid"], views[r]["price"], views[r]["budget"]),
                             (OLDER, host_price, host_budget), f"TP{r}")
        self.assertEqual(host_price, FILL + MAX_NEW + 1)
        # the budget is the host's (82100 - 2 x 64), not a worker's own pool reading
        self.assertEqual(host_budget, HOST_POOL - 2 * MAX_NEW)
        # the reserves are the adder's per running request, on every rank
        for r in (0, 1, 2):
            self.assertEqual(views[r]["reserve"], {V_OLD: MAX_NEW, VICTIM: MAX_NEW})
        # the pool reading is each rank's OWN (it carries the rank's own drift later)
        self.assertEqual([views[r]["pool"] for r in (0, 1, 2)], [HOST_POOL, WORKER_POOL, WORKER_POOL])

    def test_the_group_min_is_adder_on_all_three_ranks_and_the_victim_leaves(self):
        """RED without the fix (M1): TP0 'adder' says yes, TP1/TP2 'legacy' say 'fits free,
        nobody leaves' (6817 <= 90000) and veto -> MIN False -> nobody leaves."""
        got = _form_a_pass()
        flags, bases = [], []
        for r in (0, 1, 2):
            sched, batch = got[r][2], got[r][3]
            with self.assertLogs(DPR.logger, level="INFO") as lg:
                flags.append(DPR.kv_displace_would_fit(sched, OLDER, batch.reqs, view=getattr(sched, VIEW_ATTR)))
            bases += _basis(lg)
        self.assertEqual(bases, ["adder"] * 3)
        self.assertEqual(flags, [True, True, True])

    def test_displace_for_age_on_the_host_displaces_the_youngest_when_the_group_agrees(self):
        got = _form_a_pass(pass_n=7)
        sched, batch = got[0][2], got[0][3]
        setattr(sched, PASS_ATTR, 8)                      # the next pass
        sched._pdflip_group_min_flags = lambda flags: [True]   # TP1/TP2 said True as well (test above)
        with self.assertLogs(DPR.logger, level="INFO"):
            displaced = _in_d(lambda: DPR.displace_for_age(sched, batch))
        self.assertEqual(displaced, VICTIM)

    def test_a_worker_local_refusal_of_another_rid_does_not_leak_into_the_view(self):
        """On a Form A group the rank-local lifetime_refusal is not read: a stale note of
        another rid on a worker's adder neither blocks nor replaces the group's numbers."""
        got = _form_a_pass()
        adder = got[1][1]
        adder.lifetime_refusal = ("pdflip-9-9", 1, 1)
        sched = got[1][2]
        setattr(sched, VIEW_ATTR, None)
        _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), _batch([])))
        v = getattr(sched, VIEW_ATTR)
        self.assertEqual((v["rid"], v["price"]), (OLDER, FILL + MAX_NEW + 1))


class GroupRefusalUnit(unittest.TestCase):
    def _follow(self, last):
        return SimpleNamespace(last_verdict=last)

    def test_only_this_rids_lifetime_shaped_no_token_counts(self):
        f = DPR._group_refusal
        self.assertEqual(f(self._follow((OLDER, "NO_TOKEN", 100, 90)), OLDER), (OLDER, 100, 90))
        self.assertIsNone(f(self._follow((OLDER, "NO_TOKEN", 100, 90)), "pdflip-9-9"))    # other rid
        self.assertIsNone(f(self._follow((OLDER, "ADMIT", 100, 90)), OLDER))            # admitted
        self.assertIsNone(f(self._follow((OLDER, "OTHER", 100, 90)), OLDER))
        self.assertIsNone(f(self._follow(None), OLDER))                                 # no call yet
        self.assertIsNone(f(self._follow(("x",)), OLDER))                               # malformed

    def test_a_host_refusal_on_another_gate_carries_no_budget_question(self):
        """price < budget: the host refused on SWA / load-back room / cut room. Mutation M4
        (guard removed): every rank would build a view out of numbers that did not refuse."""
        self.assertIsNone(DPR._group_refusal(self._follow((OLDER, "NO_TOKEN", 80, 90)), OLDER))
        self.assertEqual(DPR._group_refusal(self._follow((OLDER, "NO_TOKEN", 90, 90)), OLDER), (OLDER, 90, 90))

    def test_unparseable_numbers_set_no_view_and_never_raise(self):
        sched, _ = _sched([], [], 1000)
        adder = SimpleNamespace(form_a_admission_follow=self._follow((OLDER, "NO_TOKEN", None, None)),
                                released_by_leaving=lambda r: 1)
        _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), _batch([])))
        self.assertIsNone(getattr(sched, VIEW_ATTR))


class GroupNumbersTravelWithTheVerdict(unittest.TestCase):
    """tp_match_floor: the sink carries what the group verdict decided on (broadcast / gather),
    and nothing else about the verdict changes."""

    def test_broadcast_sink_is_the_hosts_tuple_on_every_rank(self):
        box = {}

        def ex(p):
            if p is not None:
                box["v"] = p
            return box["v"]

        host, worker = {}, {}
        self.assertEqual(TMF.form_a_admission_verdict("r", "NO_TOKEN", is_host=True, exchange=ex,
                                                      price=193408, budget=176587, sink=host), "NO_TOKEN")
        self.assertEqual(TMF.form_a_admission_verdict("r", "ADMIT", is_host=False, exchange=ex,
                                                      price=4544, budget=176587, sink=worker), "NO_TOKEN")
        self.assertEqual(host, {"price": 193408, "budget": 176587})
        self.assertEqual(worker, host)       # the worker's own 4544 is not what it keeps

    def test_cut_gather_sink_is_the_deciders_tuple(self):
        mine = ("r", "ADMIT", 5, 100)
        got = [("r", "ADMIT", 10, 100), ("r", "NO_TOKEN", 300, 200), ("r", "ADMIT", 7, 100)]
        sink = {}
        code = TMF.form_a_admission_verdict("r", "ADMIT", is_host=False, exchange=None, price=5, budget=100,
                                            gather=lambda p: list(got), sink=sink)
        self.assertEqual(code, "NO_TOKEN")
        self.assertEqual(sink, {"price": 300, "budget": 200})
        del mine

    def test_without_a_sink_the_call_is_the_old_call(self):
        """Reference configuration: same code, same wire payload, with and without a sink."""
        posts = []

        def ex(p):
            posts.append(p)
            return p

        a = TMF.form_a_admission_verdict("r", "NO_TOKEN", is_host=True, exchange=ex, price=3, budget=2)
        b = TMF.form_a_admission_verdict("r", "NO_TOKEN", is_host=True, exchange=ex, price=3, budget=2, sink={})
        self.assertEqual((a, b), ("NO_TOKEN", "NO_TOKEN"))
        self.assertEqual(posts[0], posts[1])
        self.assertEqual(posts[0], ("r", "NO_TOKEN", 3, 2, 0, ""))


# ------------------------------------------------------------------------------- age lock
def _view(price, budget, pool, pass_n, reserve=None):
    return {"rid": OLDER, "price": price, "budget": budget, "pool": pool, "pass": pass_n,
            "reserve": {VICTIM: 64} if reserve is None else reserve}


class TheViewIsDroppedWhenItIsOlderThanOnePass(unittest.TestCase):
    """Review 1140 finding 2. The view below would displace (adder basis: price 5001 > budget
    4000 + victim 1000 + reserve 64? no: 5065 >= 5002 -> yes) while the legacy reading says
    'fits free' (raw 50 against pool 4000)."""

    def _setup(self, written, now):
        victim = _run_req(VICTIM, 1000, 64)
        sched, batch = _sched([victim], [_queued(OLDER, 50)], pool=4000, pass_n=now)
        sched._pdflip_sa_no_token = OLDER
        setattr(sched, VIEW_ATTR, _view(5001, 4000, 4000, written))
        return sched, batch

    def test_a_view_of_the_previous_pass_is_read(self):
        sched, batch = self._setup(written=10, now=11)
        with self.assertLogs(DPR.logger, level="INFO"):
            self.assertEqual(_in_d(lambda: DPR.displace_for_age(sched, batch)), VICTIM)

    def test_a_view_two_passes_old_is_dropped_and_the_legacy_basis_decides(self):
        """RED without the age lock (mutation M2: _view_fresh always True): the old numbers
        displace the victim."""
        sched, batch = self._setup(written=10, now=12)
        with self.assertLogs(DPR.logger, level="INFO") as lg:
            self.assertIsNone(_in_d(lambda: DPR.displace_for_age(sched, batch)))
        text = "\n".join(lg.output)
        self.assertIn("Q-702 SEAT-AGE VIEW-STALE older=pdflip-4-14 age=2 passes max=1", text)
        self.assertEqual(batch.released, [])
        self.assertIsNone(getattr(sched, VIEW_ATTR))               # consumed either way
        self.assertEqual(sched.calls, [[False]])                   # one group MIN, legacy 'fits free'

    def test_the_bound_is_exactly_one_pass(self):
        """Mutation M2b (bound 1 -> 2 or off by one) is red here."""
        for age, expect in ((0, True), (1, True), (2, False), (3, False), (50, False)):
            sched, _ = _sched([], [], 1000, pass_n=100)
            self.assertEqual(DPR._view_fresh(sched, _view(1, 1, 1, 100 - age)), expect, age)
        sched, _ = _sched([], [], 1000, pass_n=100)
        self.assertFalse(DPR._view_fresh(sched, _view(1, 1, 1, 101)), "a view from the future is no view")
        self.assertFalse(DPR._view_fresh(sched, None))

    def test_a_view_without_a_stamp_is_a_desk_double_and_counts_as_fresh(self):
        sched, _ = _sched([], [], 1000, pass_n=100)
        v = _view(1, 1, 1, 0)
        del v["pass"]
        self.assertTrue(DPR._view_fresh(sched, v))

    def test_declined_passes_age_the_view_the_way_the_scheduler_counts_them(self):
        """A view written in pass 3; passes 4 and 5 decline above admission() (batch_is_full)
        but still count; admission() runs in pass 6 -> stale. Only the counter moves in
        the declined passes, exactly as ``_get_new_batch_prefill_raw`` does it."""
        victim = _run_req(VICTIM, 1000, 64)
        sched, batch = _sched([victim], [_queued(OLDER, 50)], pool=4000, pass_n=3)
        adder = SimpleNamespace(lifetime_refusal=(OLDER, 5001, 4000), released_by_leaving=lambda r: 64)
        _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), batch))
        self.assertEqual(getattr(sched, VIEW_ATTR)["pass"], 3)
        sched._pdflip_sa_no_token = OLDER
        for _ in (4, 5, 6):
            setattr(sched, PASS_ATTR, getattr(sched, PASS_ATTR) + 1)
        with self.assertLogs(DPR.logger, level="INFO") as lg:
            self.assertIsNone(_in_d(lambda: DPR.displace_for_age(sched, batch)))
        self.assertIn("VIEW-STALE", "\n".join(lg.output))


class TheSchedulerCountsEveryPass(unittest.TestCase):
    def test_the_pass_counter_is_bumped_before_any_return_of_the_raw_pass(self):
        """Mutation M5 (the bump removed or moved below the first early exit) is red."""
        src = textwrap.dedent(inspect.getsource(Scheduler._get_new_batch_prefill_raw))
        fn = ast.parse(src).body[0]
        bump = [n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and t.attr == PASS_ATTR for t in n.targets)]
        rets = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Return)]
        self.assertEqual(len(bump), 1, bump)
        self.assertLess(bump[0], min(rets))

    def test_the_view_is_stamped_with_the_counter_it_is_aged_by(self):
        sched, batch = _sched([], [], 1000, pass_n=41)
        adder = SimpleNamespace(lifetime_refusal=(OLDER, 5, 4), released_by_leaving=lambda r: 1)
        _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), batch))
        self.assertEqual(getattr(sched, VIEW_ATTR)["pass"], 41)
        self.assertEqual(DPR.SA_PASS_ATTR, PASS_ATTR)


# ----------------------------------------------------------- Fall A (03.10.): not 782 victims
class FallAEndToEnd(unittest.TestCase):
    """Fall A with the REAL PrefillAdder: the older request is refused by the adder although
    its raw extend fits the pool; the verdict asks the adder's question and displaces exactly
    the victims that make the adder admit it -- then stops."""

    EXTEND, MAXNEW, PAGE, POOL, RES = 189248, 2048, 64, 197376, 2048

    def _drive(self, victims_kv, passes=20, pool=None):
        """Per pass, in the scheduler's order: ``admission()`` -> ``displace_for_age`` (reads the
        previous pass's refusal), then the adder's turn for the older request; a refusal is
        noted by ``note_adder_refusal``. A displaced victim's KV rows go back to the pool and its
        decode reserve leaves the running batch."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        state = {"pool": self.POOL if pool is None else pool}
        running = [_run_req(f"pdflip-4-{20 + i}", kv, self.RES) for i, kv in enumerate(victims_kv)]
        sched, batch = _sched(running, [_queued(OLDER, self.EXTEND)], pool=state["pool"], pass_n=0)
        sched.token_to_kv_pool_allocator.available_size = lambda: state["pool"]

        def release(idx, rem, sa, retain=False):
            victim = batch.reqs[idx]
            state["pool"] += len(victim.origin_input_ids) + len(victim.output_ids)
            batch.released.append((victim.rid, retain))

        batch.release_req = release
        admitted_at = None
        for n in range(1, passes + 1):
            setattr(sched, PASS_ATTR, n)
            _in_d(lambda: DPR.displace_for_age(sched, batch))
            adder = _adder(state["pool"], _batch(batch.reqs), page=self.PAGE)
            res = adder.add_one_req(_mock_req(self.EXTEND, self.MAXNEW), truncation_align_size=None)
            if res != AddReqResult.NO_TOKEN:
                admitted_at = n
                break
            sched._pdflip_sa_no_token = OLDER
            _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), _batch(batch.reqs)))
        return [rid for rid, _ in batch.released], admitted_at

    def test_the_raw_extend_fits_but_the_adder_refuses_one_victim_makes_room_and_the_older_is_admitted(self):
        """Specimen numbers: raw 189248 <= pool 197376 ('fits free'), adder price 191360 against
        budget 197376 - 4 x 2048 = 189184. Pass 1 refused; pass 2 the youngest (14781 rows +
        2048 reserve) leaves and the same pass's adder admits: ONE displacement, not 782."""
        released, admitted_at = self._drive([1000, 2000, 3000, 14781])
        self.assertEqual(released, ["pdflip-4-23"])
        self.assertEqual(admitted_at, 2)

    def test_victims_that_cannot_make_room_are_never_displaced(self):
        """Not even all younger seats make price + 1 fit: nobody leaves in any of 20 passes
        (a victim for nothing, the 782, cannot start)."""
        released, admitted_at = self._drive([100, 100, 100, 100], pool=150000)
        self.assertEqual(released, [])
        self.assertIsNone(admitted_at)

    def test_two_victims_are_needed_two_leave_one_per_pass_and_then_it_stops(self):
        released, admitted_at = self._drive([9000, 9000, 5000, 5000], pool=190000)
        self.assertEqual(released, ["pdflip-4-23", "pdflip-4-22"])
        self.assertEqual(admitted_at, 3)


# --------------------------------------------------------- reference configuration unchanged
def _base_kv_displace_would_fit(sched, older_rid, running, seat=False):
    """kv_displace_would_fit exactly as on 3bfee09511 (the base the rework is cut from)."""
    from flliper.srt.pdflip import seat_age as _sa
    n_ = DPR._n
    older = next((q for q in getattr(sched, "waiting_queue", ()) or () if str(q.rid) == str(older_rid)), None)
    if older is None:
        return False
    need = max(0, DPR._req_kv_tokens(older) - n_(getattr(older, "prefix_indices", None)))
    try:
        avail = int(sched.token_to_kv_pool_allocator.available_size())
    except Exception:  # noqa: BLE001
        return False
    try:
        avail += int(sched.tree_cache.evictable_size() or 0)
    except Exception:  # noqa: BLE001
        pass
    young = sorted((r for r in running if _sa.rid_age(str(r.rid)) > _sa.rid_age(str(older_rid))),
                   key=lambda r: _sa.rid_age(str(r.rid)), reverse=True)
    k = DPR.victims_needed(need, avail, [DPR._req_kv_tokens(r) for r in young])
    if seat and k is not None:
        k = None if not young else max(1, k)
    n = getattr(sched, "_sa_kv_fit_n", 0) + 1
    sched._sa_kv_fit_n = n
    if k != 1 and (n <= 8 or (n & (n - 1)) == 0):
        DPR.logger.info("SEAT-AGE %s-DISPLACE-VERDICT older=%s need=%d free=%d younger_running=%d -> %s (n=%d)",
                        "SEAT" if seat else "KV", str(older_rid), need, avail, len(young),
                        "fits free, nobody leaves" if k == 0 else
                        "not even with all younger seats: nobody leaves, backfill stays" if k is None else
                        "%d youngest must leave (one per pass)" % k, n)
    return bool(k)


class WithoutAFaultTheVerdictIsTheBases(unittest.TestCase):
    """Reference configuration: no adder refusal, so no view; and a view whose numbers ARE the
    legacy reading (adder and verdict agree) -- same decisions, same log text (the only
    difference is the added ``basis=`` token)."""

    GRID = [(need_kv, prefix, pool, ev, young_kvs, seat)
            for need_kv in (50, 5000, 189248)
            for prefix in (0, 384)
            for pool in (0, 4000, 197376)
            for ev in (0, 700)
            for young_kvs in ((), (1000,), (1000, 14781), (9000, 9000, 9000))
            for seat in (False, True)]

    def _mk(self, need_kv, prefix, pool, ev, young_kvs):
        running = [_run_req(f"pdflip-4-{20 + i}", kv, 64) for i, kv in enumerate(young_kvs)]
        sched, _ = _sched(running, [_queued(OLDER, need_kv, prefix)], pool=pool, evictable=ev)
        return sched, running

    @staticmethod
    def _norm(lines):
        return [re.sub(r" basis=\w+", "", ln) for ln in lines]

    def test_no_view_decides_and_logs_like_the_base(self):
        n = 0
        for need_kv, prefix, pool, ev, young_kvs, seat in self.GRID:
            s_new, run_new = self._mk(need_kv, prefix, pool, ev, young_kvs)
            s_old, run_old = self._mk(need_kv, prefix, pool, ev, young_kvs)
            with self.assertLogs(DPR.logger, level="INFO") as a:
                DPR.logger.info("tick")
                new = DPR.kv_displace_would_fit(s_new, OLDER, run_new, seat=seat)
            with self.assertLogs(DPR.logger, level="INFO") as b:
                DPR.logger.info("tick")
                old = _base_kv_displace_would_fit(s_old, OLDER, run_old, seat=seat)
            self.assertEqual(new, old, (need_kv, prefix, pool, ev, young_kvs, seat))
            self.assertEqual(self._norm(a.output), b.output)
            self.assertTrue(all("basis=legacy" in ln for ln in a.output if "DISPLACE-VERDICT" in ln))
            n += 1
        self.assertEqual(n, len(self.GRID))

    def test_a_view_that_states_the_legacy_numbers_decides_like_the_base(self):
        """Adder and verdict agree: price + 1 = legacy need, budget = legacy available, no
        reserves, the pool unchanged. The adder basis then gives the base's answer."""
        for need_kv, prefix, pool, ev, young_kvs, seat in self.GRID:
            s_new, run_new = self._mk(need_kv, prefix, pool, ev, young_kvs)
            s_old, run_old = self._mk(need_kv, prefix, pool, ev, young_kvs)
            need = max(0, need_kv - prefix)
            if need == 0:
                continue                       # price + 1 cannot state need 0
            view = {"rid": OLDER, "price": need - 1, "budget": pool + ev, "reserve": {}, "pool": pool + ev,
                    "pass": 5}
            setattr(s_new, PASS_ATTR, 6)
            with self.assertLogs(DPR.logger, level="INFO"):
                DPR.logger.info("tick")
                new = DPR.kv_displace_would_fit(s_new, OLDER, run_new, seat=seat, view=view)
            with self.assertLogs(DPR.logger, level="INFO"):
                DPR.logger.info("tick")
                old = _base_kv_displace_would_fit(s_old, OLDER, run_old, seat=seat)
            self.assertEqual(new, old, (need_kv, prefix, pool, ev, young_kvs, seat))

    def test_no_refusal_no_view_and_the_pass_runs_as_before(self):
        """displace_for_age with nothing refused (no rid, no view): the base's path -- one group
        MIN of ``[False]`` as on the base (the KV trigger enters it on every rank), no stale line."""
        victim = _run_req(VICTIM, 1000, 64)
        sched, batch = _sched([victim], [_queued(OLDER, 50)], pool=4000, pass_n=9)
        with self.assertNoLogs(DPR.logger, level="INFO"):
            self.assertIsNone(_in_d(lambda: DPR.displace_for_age(sched, batch)))
        self.assertEqual(sched.calls, [[False]])
        self.assertIsNone(getattr(sched, VIEW_ATTR))

    def test_off_form_a_the_adders_own_refusal_is_still_the_source(self):
        """27B D / single TP: no follow callable, the rank-local lifetime_refusal builds the view."""
        sched, batch = _sched([], [], 1000, pass_n=2)
        adder = SimpleNamespace(lifetime_refusal=(OLDER, 11, 10), released_by_leaving=lambda r: 3)
        self.assertFalse(hasattr(adder, "form_a_admission_follow"))
        _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), batch))
        v = getattr(sched, VIEW_ATTR)
        self.assertEqual((v["price"], v["budget"], v["pass"]), (11, 10, 2))
        adder.form_a_admission_follow = None
        _in_d(lambda: DPR.note_adder_refusal(sched, adder, SimpleNamespace(rid=OLDER), batch))
        self.assertEqual(getattr(sched, VIEW_ATTR)["price"], 11)


class TheAdderBookkeepingCannotRaise(unittest.TestCase):
    def test_non_finite_numbers_are_swallowed_review_1140_3c(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        adder = _adder(1000, _batch([]))
        adder._note_lifetime_refusal(_mock_req(10, 1), float("inf"), 5)
        adder._note_lifetime_refusal(_mock_req(10, 1), 5, float("nan"))
        self.assertIsNone(adder.lifetime_refusal)
        adder._note_lifetime_refusal(_mock_req(10, 1), 7, 5)
        self.assertEqual(adder.lifetime_refusal, (OLDER, 7, 5))


if __name__ == "__main__":
    unittest.main()
