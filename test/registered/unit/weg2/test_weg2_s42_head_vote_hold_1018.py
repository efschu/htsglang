# SPDX-License-Identifier: Apache-2.0
"""S42 HEAD-VOTE HOLD: a waiting D request is not re-measured in every decode round.

MEASURED (S40, boot ..._49f9c2ea4d_1018_133242.D.log, TP0, 14:13:00-14:13:59, ``#queue-req: 6``, bs 1): 902 decode
rounds, 66 ms each, ``DECODE-HOST-SPLIT sched_ms=56.9`` (2.6-3.2 ms without a waiting request). Per round 4x
``#1042 EXTENT LIFECYCLE set`` (rids 36-117/119/124, 38-126, ~155x each in the minute) and ~14
``MAMBA-HOST-RESUME`` acceptances, ``#1317 X-STORE-VOTE head=6 voted=6``, zero ``X-GATE`` lines: DQH held the
admission pass, the #823 head vote at the budget site (``_update_uniform_pool_budget`` ->
``_local_head_prefix_matches`` + the usable arm) still measured all six rids every round.

Driven through the REAL ``Scheduler._update_uniform_pool_budget`` (3 rank doubles on one rendezvous, the harness of
test_weg2_rank_gleichheit_1537b), the REAL ``_local_head_prefix_matches`` / ``_weg2_d_head_fingerprint`` /
``_weg2_dqh_holds`` and a REAL ``DHeadHold``; the admission pass is reduced to what it does with a refused head
(consume the vote, ``note_pass(built_nothing=True)``). The microbenchmark at the end walks a REAL
``UnifiedRadixCache`` (page 64, six 114k-token prompts) with the REAL ``match_prefix_for_req`` and the REAL Form A
``admission_probe``.

RED on b9592b6f12 (every round measures every waiting rid: 6 x M). GREEN: measured on the passes DQH evaluates
(the first, every change, every 17th), abstained on the held ones; never an evaluated pass on an unmeasured vote.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

N = 3
HANG_S = 2.0
_TLS = threading.local()
RIDS = ["r36-117", "r36-119", "r36-124", "r38-126", "r38-127", "r38-130"]
HEAD = RIDS[0]
RUNNING = ["r34-107"]


class Hang(Exception):
    pass


class Mismatch(Exception):
    pass


class Rendezvous:
    """The group collective: the k-th call of a rank meets the k-th call of every other; elementwise MIN; other
    vector length = Mismatch; a partner missing within the deadline = Hang."""

    def __init__(self, n=N, hang_s=HANG_S):
        self.n, self.hang_s = n, hang_s
        self.calls = [[] for _ in range(n)]
        self._slots, self._lock = {}, threading.Lock()

    def count(self):
        return [len(c) for c in self.calls]

    def reduce(self, rank, vals):
        vals = [int(v) for v in vals]
        seq = len(self.calls[rank])
        self.calls[rank].append(list(vals))
        with self._lock:
            slot = self._slots.setdefault(seq, {"v": {}, "cv": threading.Condition(self._lock)})
        cv = slot["cv"]
        with cv:
            slot["v"][rank] = vals
            cv.notify_all()
            if not cv.wait_for(lambda: len(slot["v"]) == self.n, timeout=self.hang_s):
                raise Hang("rank %d: collective #%d never met its partners" % (rank, seq))
            lens = {len(v) for v in slot["v"].values()}
            if len(lens) != 1:
                raise Mismatch("collective #%d: lengths %s" % (seq, sorted(lens)))
            return [min(slot["v"][r][i] for r in range(self.n)) for i in range(lens.pop())]


def lockstep(items, fn, *, join_s=20.0):
    out = [(None, None)] * len(items)

    def run(i):
        try:
            out[i] = (fn(items[i]), None)
        except BaseException as exc:  # noqa: BLE001 -- the test reads it
            out[i] = (None, exc)

    ts = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(items))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(join_s)
    return out


def _rank_class():
    from sglang.srt.managers.scheduler import Scheduler as Sch

    class _R:
        # the real methods under test
        _update_uniform_pool_budget = Sch._update_uniform_pool_budget
        _local_head_prefix_matches = Sch._local_head_prefix_matches
        _weg2_d_head_fingerprint = Sch._weg2_d_head_fingerprint
        _weg2_local_store_read_pending_ages = Sch._weg2_local_store_read_pending_ages
        _HOST_AVAIL_ABSENT = Sch._HOST_AVAIL_ABSENT
        _MAMBA_AVAIL_ABSENT = Sch._MAMBA_AVAIL_ABSENT

        # rank-uniform filler (not the subject)
        def _local_host_avail(self):
            return self._HOST_AVAIL_ABSENT

        def _local_mamba_avail(self):
            return self._MAMBA_AVAIL_ABSENT

        def _local_corridor_width_ceiling(self):
            return 4096

        def _local_admit_limit(self, running_batch):
            return 8

        def _local_seam_premise_vote(self):
            return 1

        def _drain_prefetch_progress(self):
            return dict(self.verdicts)

        def _weg2_store_read_is_pending(self, req):
            return req.rid in self.pending

        def _weg2_local_store_matches(self, canonical):
            self.store_votes += 1
            return {}

        def _publish_uniform_evict_floor(self, *a, **k):
            pass

        def _publish_uniform_host_floor(self, *a, **k):
            pass

        def _publish_uniform_mamba_floor(self, *a, **k):
            pass

    if hasattr(Sch, "_weg2_dqh_holds"):
        _R._weg2_dqh_holds = Sch._weg2_dqh_holds
    return _R


def _sched(rank, rids=RIDS):
    from sglang.srt.weg2 import d_head_hold as H

    s = _rank_class()()
    s.kv_session_offload = None
    s.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 1000)
    s.tp_cpu_group = "grp"
    s.tree_cache = types.SimpleNamespace(match_prefix=lambda params: None)
    s.server_args = types.SimpleNamespace(pp_size=1, tp_size=N, dcp_size=1)
    s.ps = types.SimpleNamespace(tp_rank=rank, tp_size=N, pp_size=1)
    s.waiting_queue = [types.SimpleNamespace(rid=r, best_match_node=None) for r in rids]
    s.running_batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid=r) for r in RUNNING])
    s.chunked_req = None
    s.verdicts, s.pending, s.store_votes = {}, set(), 0
    s.measured = []          # (iteration, rid) of every head match this rank took
    s.it = 0
    s.weg2_d_head_hold = H.DHeadHold(group_d=True, token_cut=lambda: True)
    s._weg2_hol = {"rid": HEAD, "passes": 1, "logged": 0}
    return s


class _Group:
    """3 ranks of the real budget site on one rendezvous; the elif chain's DQH decision; a pass that consumes the
    vote and refuses the head."""

    def __init__(self, monkeypatch, gen_fn=None):
        import sglang.srt.managers.schedule_policy as SP
        import sglang.srt.managers.scheduler as S
        from sglang.srt.managers import tp_match_floor as TMF

        self.rv = Rendezvous()
        self.gen_fn = gen_fn or (lambda it: 7)

        def reduce(t, op=None, group=None):
            out = self.rv.reduce(_TLS.rank, t.tolist())
            t.copy_(torch.tensor(out, dtype=t.dtype))

        monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda g=None: N)

        def match(tree, req, include_req=False, **k):
            s = _TLS.sched
            s.measured.append((s.it, req.rid))
            req._vote = 100000 + RIDS.index(req.rid) if req.rid in RIDS else 50000

        monkeypatch.setattr(SP, "match_prefix_for_req", match)
        monkeypatch.setattr(S, "_head_vote_len", lambda req: int(req._vote))
        monkeypatch.setattr(TMF, "local_usable_matches",
                            lambda tree, by_rid, matches: {k: int(v) for k, v in matches.items()})
        import sglang.srt.mem_cache.hicache_phase_binding as HB

        monkeypatch.setattr(HB, "current_generation", lambda: self.gen_fn(_TLS.sched.it))

    def round(self, s, *, mutate_between=None):
        """One get_next_batch_to_run of rank ``s`` reduced to the budget site + the DQH elif branch + the pass.
        Returns 'held' or 'evaluated:<consumed lens>'."""
        try:
            from sglang.srt.weg2 import head_vote_hold as HV
        except ImportError:  # b9592b6f12
            HV = None

        _TLS.rank, _TLS.sched = s.ps.tp_rank, s
        rb = s.running_batch
        if HV is not None:
            HV.arm(s)
        s._update_uniform_pool_budget(rb)
        if HV is not None:
            HV.disarm(s)
        if mutate_between is not None:
            mutate_between(s)
        if hasattr(s, "_weg2_dqh_holds"):
            held = s._weg2_dqh_holds(rb)
        else:  # b9592b6f12: the chain asks DQH
            held = s.weg2_d_head_hold.should_hold(
                s._weg2_d_head_fingerprint(rb, s.__dict__.get("_pass_prefetch_verdicts")))
        if held:
            out = "held"
        else:
            verdicts = s.__dict__.pop("_pass_prefetch_verdicts", None)
            inputs = s.__dict__.pop("_uniform_head_inputs", None)
            s.__dict__.pop("_uniform_usable_floor", None)
            lens = dict(zip(inputs.canonical, inputs.group_match_lens)) if inputs else {}
            out = ("evaluated", lens, {r for it, r in s.measured if it == s.it})
            s.weg2_d_head_hold.note_pass(fp=s._weg2_d_head_fingerprint(rb, verdicts), built_nothing=True)
        s.it += 1
        return out

    def run(self, scheds, rounds, *, per_round=None, mutate_between=None):
        def one(s):
            res = []
            for i in range(rounds):
                if per_round is not None:
                    per_round(i, s)
                res.append(self.round(s, mutate_between=(mutate_between(i) if mutate_between else None)))
            return res

        res = lockstep(scheds, one)
        errs = [e for _v, e in res]
        assert errs == [None] * len(scheds), [repr(e) for e in errs]
        return [v for v, _e in res]


def _evaluated(results):
    return [i for i, r in enumerate(results) if r != "held"]


# --------------------------------------------------------------------------------------------------------------
# Repro: N=6 waiting, M=34 decode rounds with an unchanged refused head
# --------------------------------------------------------------------------------------------------------------
def test_rc11_1413_six_waiting_34_rounds_measured_only_on_evaluated_passes(monkeypatch):
    from sglang.srt.weg2 import d_head_hold as H

    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    M = 2 * (H.REEVAL_EVERY_PASSES + 1)
    results = g.run(scheds, M)
    for s, res in zip(scheds, results):
        ev = _evaluated(res)
        # rc11 14:13: 6 rids x every round. Now: the first pass and the 16-pass bound only.
        assert ev == [0, H.REEVAL_EVERY_PASSES + 1], ev
        assert len(s.measured) == len(RIDS) * len(ev), (len(s.measured), "6 x %d" % M)
        assert {it for it, _r in s.measured} == set(ev), "a held round measured the vote"
        assert s.store_votes == len(ev), "the store arm ran on a held round"
    assert g.rv.count() == [M] * N, "one packed MIN per round on every rank, held or not"


def test_rc11_1413_no_refused_head_the_queue_is_held_all_the_same(monkeypatch):
    """rc11 14:12:43-14:17:46: the passes declined all six before add_one_req (no NO_TOKEN, no X-GATE) and the
    HOL state still named the head of 14:09 (r34-107), admitted and RUNNING since. RED with the head required
    (DQH never armed: 6 x M head matches)."""
    from sglang.srt.weg2 import d_head_hold as H

    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    for s in scheds:
        s._weg2_hol = {"rid": RUNNING[0], "passes": 40, "logged": 3}
    M = 2 * (H.REEVAL_EVERY_PASSES + 1)
    results = g.run(scheds, M)
    for s, res in zip(scheds, results):
        assert _evaluated(res) == [0, H.REEVAL_EVERY_PASSES + 1], _evaluated(res)
        assert len(s.measured) == len(RIDS) * 2, len(s.measured)


def test_every_evaluated_pass_reads_a_vote_measured_in_its_own_iteration(monkeypatch):
    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    results = g.run(scheds, 40)
    for res in results:
        for r in res:
            if r == "held":
                continue
            _tag, lens, measured_now = r
            assert set(RIDS) <= measured_now, "an evaluated pass read a vote not measured in its iteration"
            assert all(lens[rid] == 100000 + RIDS.index(rid) for rid in RIDS), lens


# --------------------------------------------------------------------------------------------------------------
# Re-evaluation on every relevant change (danger direction: a missed change -> the request starves)
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hol", ["refused_head", "no_head"])
@pytest.mark.parametrize("change", ["arrival", "left_queue", "decode_finished", "prefetch_landed", "flip", "bound"])
def test_a_relevant_change_measures_the_vote_at_once(monkeypatch, change, hol):
    from sglang.srt.weg2 import d_head_hold as H

    AT = 5
    g = _Group(monkeypatch, gen_fn=(lambda it: 8 if (change == "flip" and it >= AT) else 7))
    scheds = [_sched(r) for r in range(N)]
    if hol == "no_head":
        for s in scheds:
            s._weg2_hol = {"rid": RUNNING[0], "passes": 40, "logged": 3}

    def per_round(i, s):
        if i != AT:
            return
        if change == "arrival":
            s.waiting_queue.append(types.SimpleNamespace(rid="r40-200", best_match_node=None))
        elif change == "left_queue":
            s.waiting_queue = [q for q in s.waiting_queue if q.rid != RIDS[3]]
        elif change == "decode_finished":
            s.running_batch = types.SimpleNamespace(reqs=[])
        elif change == "prefetch_landed":
            s.verdicts = {RIDS[2]: True}

    rounds = AT + 2 if change != "bound" else H.REEVAL_EVERY_PASSES + 3
    results = g.run(scheds, rounds, per_round=per_round)
    for res in results:
        ev = _evaluated(res)
        want = [0, AT] if change != "bound" else [0, H.REEVAL_EVERY_PASSES + 1]
        assert ev == want, (change, ev)


def test_a_change_between_budget_site_and_chain_holds_this_round_and_measures_the_next(monkeypatch):
    """The vote of this iteration was not measured, so the pass must not run on it -- the next budget site sees
    the change, measures, and evaluates (one round)."""
    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]

    def mutate_between(i):
        if i != 4:
            return None

        def m(s):
            s.waiting_queue.append(types.SimpleNamespace(rid="r40-201", best_match_node=None))
        return m

    results = g.run(scheds, 8, mutate_between=mutate_between)
    for res in results:
        assert _evaluated(res) == [0, 5], _evaluated(res)
        assert res[4] == "held", "round 4 ran on a vote it never measured"
        _t, lens, measured_now = res[5]
        assert "r40-201" in measured_now and "r40-201" in lens


def test_a_held_pass_keeps_the_store_read_wait_running(monkeypatch):
    """The completion arm's stamp measures THE WAIT; a held pass must neither drop nor restart it."""
    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    for s in scheds:
        s.pending = {RIDS[1]}
    g.run(scheds, 1)
    t0 = {id(s): dict(s._weg2_x_defer_since) for s in scheds}
    time.sleep(0.05)
    g.run(scheds, 10)  # all held
    for s in scheds:
        assert s._weg2_x_defer_since == t0[id(s)], "a held pass touched the pending stamps"
        assert RIDS[1] in s._weg2_x_defer_since


# --------------------------------------------------------------------------------------------------------------
# Rank agreement (R11, raenge-nie-uneins)
# --------------------------------------------------------------------------------------------------------------
def test_every_rank_holds_and_measures_on_the_same_rounds(monkeypatch):
    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    results = g.run(scheds, 40)
    pattern = [[r == "held" for r in res] for res in results]
    assert pattern[0] == pattern[1] == pattern[2]
    assert g.rv.count()[0] == g.rv.count()[1] == g.rv.count()[2]


def test_a_rank_that_holds_alone_delays_the_head_and_splits_no_collective(monkeypatch):
    """Should DQH ever hold on one rank only, its abstention MIN-reduces every head arm to ABSENT on every rank:
    no usable skew, no realize round anywhere, the evaluating ranks see no group match (delay, never force)."""
    from sglang.srt.managers import tp_match_floor as TMF

    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    g.run(scheds, 1)                      # every rank armed on the refused head
    scheds[1].weg2_d_head_hold.state.fingerprint = None  # ranks 0/2 hold, rank 1 measures
    before = g.rv.count()
    seen = {}

    def one(s):
        from sglang.srt.weg2 import head_vote_hold as HV

        _TLS.rank, _TLS.sched = s.ps.tp_rank, s
        HV.arm(s)
        s._update_uniform_pool_budget(s.running_batch)
        HV.disarm(s)
        seen[s.ps.tp_rank] = (dict(s._uniform_usable_floor), s._uniform_head_inputs.group_match_lens[:len(RIDS)])
        return s._weg2_dqh_holds(s.running_batch)

    res = lockstep(scheds, one)
    assert [e for _v, e in res] == [None] * N, [repr(e) for _v, e in res]
    assert [v for v, _e in res] == [True, False, True]
    assert [c - b for c, b in zip(g.rv.count(), before)] == [1, 1, 1], "a realize round on some rank only"
    for rank in range(N):
        usable, lens = seen[rank]
        assert usable == {}, (rank, usable)
        assert set(lens) <= {TMF.ABSENT}, (rank, lens)


def test_an_unarmed_caller_of_the_budget_site_measures_and_leaves_no_decision(monkeypatch):
    """disaggregation/decode.py calls the budget site without the elif chain: no hold may come from there."""
    g = _Group(monkeypatch)
    scheds = [_sched(r) for r in range(N)]
    g.run(scheds, 1)

    def one(s):
        _TLS.rank, _TLS.sched = s.ps.tp_rank, s
        s._update_uniform_pool_budget()
        return s.__dict__.get("_s42_dqh_decision")

    n0 = [len(s.measured) for s in scheds]
    res = lockstep(scheds, one)
    assert [v for v, _e in res] == [None] * N
    assert [len(s.measured) - a for s, a in zip(scheds, n0)] == [len(RIDS)] * N


def test_an_early_return_of_the_budget_site_leaves_no_arm_for_another_caller(monkeypatch):
    """A single-rank (or offload) budget site returns before the decision; the arm of that iteration must not
    reach the next caller of the budget site (which has no elif chain behind it)."""
    from unittest import mock

    from sglang.srt.weg2 import head_vote_hold as HV

    g = _Group(monkeypatch)
    s = _sched(0)
    s.weg2_d_head_hold.note_pass(fp=s._weg2_d_head_fingerprint(s.running_batch, {}), built_nothing=True)
    _TLS.rank, _TLS.sched = 0, s
    HV.arm(s)
    with mock.patch.object(torch.distributed, "get_world_size", lambda g=None: 1):
        s._update_uniform_pool_budget(s.running_batch)
    HV.disarm(s)
    one = _OneRank()
    with mock.patch.object(torch, "distributed", one):
        s._update_uniform_pool_budget()
    assert s.__dict__.get(HV.DECISION_ATTR) is None
    assert len(s.measured) == len(RIDS), "the unarmed caller abstained on a stale arm"


def test_the_chain_consumes_the_budget_site_decision():
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.get_next_batch_to_run)
    i_arm = src.index("_weg2_head_vote_hold.arm(self)")
    i_budget = src.index("self._update_uniform_pool_budget(running_batch)")
    i_disarm = src.index("_weg2_head_vote_hold.disarm(self)")
    i_gate = src.index("elif self._weg2_dqh_holds(running_batch):")
    i_pass = src.index("prefill_plan = self.get_new_batch_prefill(running_batch)")
    assert i_arm < i_budget < i_disarm < i_gate < i_pass
    budget = inspect.getsource(Scheduler._update_uniform_pool_budget)
    assert budget.index("_weg2_head_vote_hold.decide(") < budget.index("self._local_head_prefix_matches()")
    assert budget.index("self._drain_prefetch_progress()") < budget.index("_weg2_head_vote_hold.decide(")


# --------------------------------------------------------------------------------------------------------------
# Microbenchmark: a REAL tree, the REAL head-vote walks, Form A host vote (as on rc11 D TP0)
# --------------------------------------------------------------------------------------------------------------
def _real_tree():
    import sys
    from array import array

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "mem_cache"))
    import sglang.srt.utils as _u
    import sglang.srt.utils.common as _c

    _c.get_device = lambda *a, **k: "cpu"
    _u.get_device = _c.get_device
    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.sampling.sampling_params import SamplingParams

    cache, alloc, _pool = build_fixture(CacheConfig(page_size=64, kv_size=300000, head_num=1, head_dim=8,
                                                    max_context_len=200000, max_num_reqs=16))
    base = list(range(1, 100001))     # the shared agent prefix (rc11: extents 113-120k)
    reqs = []
    for i, rid in enumerate(RIDS):
        toks = base + list(range(10_000_000 * (i + 1), 10_000_000 * (i + 1) + 13600 + 64 * i))
        n = len(toks) // 64 * 64
        cache.insert(InsertParams(key=RadixKey(array("q", toks[:n]), None), value=alloc.alloc(n).to(torch.int64)))
        prompt = array("q", toks + [7] * 200)
        r = Req(rid=rid, origin_input_text="", origin_input_ids=prompt,
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1))
        r.output_ids, r.extra_key, r.full_untruncated_fill_ids = array("q"), None, prompt
        reqs.append(r)
    return cache, reqs


class _OneRank:
    """Identity MIN over a group of 3 equal ranks (one process)."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self):
        self.calls = 0

    def get_world_size(self, group=None):
        return N

    def all_reduce(self, t, op=None, group=None):
        self.calls += 1


def test_microbenchmark_real_tree_six_waiting_sched_ms_per_round(monkeypatch, capsys):
    from unittest import mock

    import sglang.srt.managers.schedule_policy as SP
    from sglang.srt.managers import tp_match_floor as TMF
    from sglang.srt.weg2 import d_head_hold as H

    cache, reqs = _real_tree()
    s = _sched(0)
    s.tree_cache = cache
    s.waiting_queue = list(reqs)
    real_match, real_probe = SP.match_prefix_for_req, TMF.admission_probe
    counts = {"match": 0, "probe": 0}

    def match(*a, **k):
        counts["match"] += 1
        return real_match(*a, **k)

    def probe(*a, **k):
        counts["probe"] += 1
        return real_probe(*a, **k)

    monkeypatch.setattr(SP, "match_prefix_for_req", match)
    monkeypatch.setattr(TMF, "admission_probe", probe)
    monkeypatch.setattr(TMF, "form_a_follow_active", lambda: True)   # rc11 D: Form A host vote
    monkeypatch.setattr(TMF, "this_rank_follows", lambda: False)
    import sglang.srt.mem_cache.hicache_phase_binding as HB

    monkeypatch.setattr(HB, "current_generation", lambda: 7)
    g = _Group.__new__(_Group)
    g.gen_fn = lambda it: 7
    M = 4 * (H.REEVAL_EVERY_PASSES + 1)
    per = []
    with mock.patch.object(torch, "distributed", _OneRank()):
        _TLS.rank, _TLS.sched = 0, s
        for _ in range(M):
            t = time.perf_counter()
            r = g.round(s)
            per.append(((time.perf_counter() - t) * 1e3, r == "held"))
    held = [ms for ms, h in per if h]
    ev = [ms for ms, h in per if not h]
    rounds = len(per)
    n_eval = len(ev)
    with capsys.disabled():
        print("\nS42 MICROBENCH N=%d waiting x %d tok, M=%d rounds: evaluated=%d held=%d "
              "matches=%d probes=%d (%.2f re-evaluations/round) budget-site ms/round: mean=%.2f "
              "evaluated=%.2f held=%.2f (evaluated rounds: %s)" % (
                  len(reqs), len(reqs[0].origin_input_ids), rounds, n_eval, len(held), counts["match"],
                  counts["probe"], counts["match"] / rounds, sum(ms for ms, _h in per) / rounds,
                  (sum(ev) / n_eval) if ev else 0.0, (sum(held) / len(held)) if held else 0.0,
                  " ".join("%.1f" % ms for ms in ev[:6])))
    assert n_eval == 4, n_eval
    assert counts["match"] == len(reqs) * n_eval and counts["probe"] == len(reqs) * n_eval
