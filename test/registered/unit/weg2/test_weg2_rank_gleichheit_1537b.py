"""1537b -- Rang-Gleichheit der Kollektiv-Eintritte (Teile a, b, c).

Frage je Teil: tritt jeder Rang einer Gruppe in DASSELBE Kollektiv ein, wenn rang-lokale
Eingaenge abweichen? Ein MIN-Reduce gleicht Werte aus, nie die ANZAHL oder Form der Kollektive
(ein Rang wartet, der andere ging vorbei = Hang). Harness: ein Thread je Rang, ein Rendezvous
(Nachbau von ``Rendezvous`` / ``lockstep`` aus test_weg2_cand3_rank_stepping_1536.py, Zweig
desk/nf-rankstep-1536 @daa11377e3; hier selbstaendig, damit der Test ohne jenen Zweig laeuft).

(a) ``_local_head_prefix_matches``: Ausnahme -> leere Stimme -> Realize-Runde (scheduler.py:10792)?
(b) ``_weg2_post_wake_settle_tick``: ``_cap_wait`` (park_l3.capacity_waiting) vor dem Kollektiv :6906.
(c) ``capacity_park_precondition`` liest ``_weg2_store_short_fallback``; Form-A-Vorfall HFB rc12z21
    (``synced_end``, scheduler.py:15174-15183).

Rot/Gruen: ``xfail(strict=True)`` = der Fehlerfall ist ROT (belegt), der Test faellt auf, sobald er
gruen wird. Nichts davon aendert Produktcode.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

N = 3
HANG_S = 1.0
_TLS = threading.local()


class Hang(Exception):
    pass


class Mismatch(Exception):
    pass


class Rendezvous:
    """Das Gruppen-Kollektiv der Raenge: der k-te Aufruf eines Rangs trifft den k-ten Aufruf jedes
    anderen (so ordnet auch all_reduce auf einer Gruppe zu); MIN elementweise; andere Vektorlaenge =
    Mismatch; fehlender Partner innerhalb der Frist = Hang."""

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
                raise Hang("rank %d: collective #%d never met its %d partners" % (rank, seq, self.n - 1))
            lens = {len(v) for v in slot["v"].values()}
            if len(lens) != 1:
                raise Mismatch("collective #%d: vector lengths %s" % (
                    seq, sorted(len(v) for v in slot["v"].values())))
            return [min(slot["v"][r][i] for r in range(self.n)) for i in range(lens.pop())]

    def flags(self, rank):
        return lambda vals: [bool(x) for x in self.reduce(rank, [1 if v else 0 for v in vals])]

    def ints(self, rank):
        return lambda vals: self.reduce(rank, vals)


def lockstep(items, fn, *, join_s=10.0):
    """One call of ``fn(item)`` per rank, in parallel. Returns [(value | None, exc | None)]."""
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


# ---------------------------------------------------------------------------
# (a) _local_head_prefix_matches -> _update_uniform_pool_budget (real code, 3 rank doubles)
# ---------------------------------------------------------------------------
def _make_rank_class():
    from sglang.srt.managers.scheduler import Scheduler as Sch

    class _R:
        # the real methods under test
        _update_uniform_pool_budget = Sch._update_uniform_pool_budget
        _local_head_prefix_matches = Sch._local_head_prefix_matches
        _HOST_AVAIL_ABSENT = Sch._HOST_AVAIL_ABSENT
        _MAMBA_AVAIL_ABSENT = Sch._MAMBA_AVAIL_ABSENT

        # everything below is rank-uniform filler (not the subject)
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
            return {}

        def _weg2_local_store_read_pending_ages(self, canonical):
            return {}

        def _weg2_local_store_matches(self, canonical):
            return {}

        def _publish_uniform_evict_floor(self, *a, **k):
            pass

        def _publish_uniform_host_floor(self, *a, **k):
            pass

        def _publish_uniform_mamba_floor(self, *a, **k):
            pass

    return _R


def _rank_sched(rank, rids):
    s = _make_rank_class()()
    s.kv_session_offload = None
    s.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 1000)
    s.tp_cpu_group = "grp"
    s.tree_cache = types.SimpleNamespace(match_prefix=lambda params: None)   # only its presence is read
    s.server_args = types.SimpleNamespace(pp_size=1, tp_size=N, dcp_size=1)
    s.ps = types.SimpleNamespace(tp_rank=rank, tp_size=N, pp_size=1)
    s.waiting_queue = [types.SimpleNamespace(rid=r, best_match_node=None) for r in rids]
    return s


class _Budget:
    """Wires the 3 ranks of ``_update_uniform_pool_budget`` to one Rendezvous.

    ``votes[rank][rid]`` = the rank's head-match vote for the rid (the usable arm votes the same
    number: no anchor trouble). ``boom`` = ranks whose ``canonical_head_rids`` raises (= the except
    path of ``_local_head_prefix_matches``, scheduler.py:10974)."""

    def __init__(self, monkeypatch, votes, boom=(), realize_ok=True):
        import sglang.srt.managers.schedule_policy as SP
        import sglang.srt.managers.scheduler as S
        from sglang.srt.managers import tp_head_congruence as THC
        from sglang.srt.managers import tp_match_floor as TMF

        self.rv = Rendezvous()
        self.votes, self.boom = votes, set(boom)

        def reduce(t, op=None, group=None):
            out = self.rv.reduce(_TLS.rank, t.tolist())
            t.copy_(torch.tensor(out, dtype=t.dtype))

        monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda g=None: N)
        real_canon = THC.canonical_head_rids

        def canon(rids, *a, **k):
            if _TLS.rank in self.boom:
                raise RuntimeError("1537b: this rank's head walk failed")
            return real_canon(rids, *a, **k)

        monkeypatch.setattr(THC, "canonical_head_rids", canon)

        def match(tree, req, include_req=False, **k):
            req._vote = self.votes[_TLS.rank].get(req.rid)
            if req._vote is None:
                raise RuntimeError("unpriced")

        monkeypatch.setattr(SP, "match_prefix_for_req", match)
        monkeypatch.setattr(S, "_head_vote_len", lambda req: int(req._vote))
        monkeypatch.setattr(TMF, "local_usable_matches",
                            lambda tree, by_rid, matches: {k: int(v) for k, v in matches.items()})
        monkeypatch.setattr(TMF, "can_realize", lambda tree, req, depth: bool(realize_ok))

    def run(self, rids_by_rank):
        scheds = [_rank_sched(r, rids_by_rank[r]) for r in range(N)]

        def one(s):
            _TLS.rank = s.ps.tp_rank
            return s._update_uniform_pool_budget()

        res = lockstep(scheds, one)
        return scheds, [e for _v, e in res], self.rv.count()


SKEW = {0: {"weg2-1-1": 100}, 1: {"weg2-1-1": 50}, 2: {"weg2-1-1": 50}}   # MIN 50 < MAX 100 = skew


def _same_rids():
    return [["weg2-1-1"] for _ in range(N)]


def test_a0_equal_ranks_skew_takes_the_realize_round_on_every_rank(monkeypatch):
    b = _Budget(monkeypatch, SKEW)
    scheds, errs, count = b.run(_same_rids())
    assert errs == [None] * N, [repr(e) for e in errs]
    assert count == [2, 2, 2], count                  # packed MIN + realize round
    assert {s._uniform_usable_floor.get("weg2-1-1") for s in scheds} == {50}


def test_a0b_equal_ranks_without_skew_one_collective(monkeypatch):
    b = _Budget(monkeypatch, {r: {"weg2-1-1": 50} for r in range(N)})
    _s, errs, count = b.run(_same_rids())
    assert errs == [None] * N and count == [1, 1, 1], (errs, count)


def test_a1_one_rank_head_walk_raises_every_rank_still_takes_the_same_collectives(monkeypatch):
    """THE DOOR OF 1537 2.1: ``_local_head_prefix_matches`` swallows an exception and returns
    ``canonical=[]`` (scheduler.py:10974-10980). 1537 inferred: that rank has no ``_usable_skew``
    and skips the realize round (:10792) while the others enter it -> hang. NOT SO: an empty vote
    abstains with ABSENT (-1) in EVERY slot, MIN absorbs it, ``decode_group_usable`` then drops
    every slot (value > ABSENT) on ALL ranks, so ``skewed_rids`` is empty everywhere. One collective
    on each rank, no hang. (Belegt-gewollt: tp_match_floor.py:271-290, ABSENT = -1.)"""
    b = _Budget(monkeypatch, SKEW, boom={1})
    _s, errs, count = b.run(_same_rids())
    assert errs == [None] * N, [repr(e) for e in errs]
    assert count == [1, 1, 1], count


def test_a1b_the_group_usable_is_empty_on_every_rank_after_one_rank_abstains(monkeypatch):
    b = _Budget(monkeypatch, SKEW, boom={1})
    scheds, errs, _c = b.run(_same_rids())
    assert errs == [None] * N
    assert [s._uniform_usable_floor for s in scheds] == [{}, {}, {}]


def test_a2_the_head_rank_raises_instead_of_a_worker_still_uniform(monkeypatch):
    b = _Budget(monkeypatch, SKEW, boom={0})
    _s, errs, count = b.run(_same_rids())
    assert errs == [None] * N and count == [1, 1, 1], (errs, count)


def test_a3_two_ranks_raise_still_uniform(monkeypatch):
    b = _Budget(monkeypatch, SKEW, boom={0, 2})
    _s, errs, count = b.run(_same_rids())
    assert errs == [None] * N and count == [1, 1, 1], (errs, count)


def test_a4_a_rid_one_rank_cannot_price_is_absent_group_wide_uniform(monkeypatch):
    votes = {0: {"weg2-1-1": 100, "weg2-1-2": 80}, 1: {"weg2-1-1": 50}, 2: {"weg2-1-1": 50, "weg2-1-2": 80}}
    b = _Budget(monkeypatch, votes)
    rids = [["weg2-1-1", "weg2-1-2"] for _ in range(N)]
    _s, errs, count = b.run(rids)
    assert errs == [None] * N and count == [2, 2, 2], (errs, count)    # rid 1 skew -> realize on all


def test_a5_queue_sets_differ_by_rank_the_collective_count_stays_uniform(monkeypatch):
    """A rid one rank does not hold shifts the canonical slots (sorted set), so the same slot
    carries different rids on different ranks. The skew decision is still taken on the reduced
    NUMBERS of the common prefix (a shorter canonical votes ABSENT in the tail), so the realize
    round is entered by all or none; the digest stop (prefetch_ballot) comes after, on every rank."""
    votes = {0: {"weg2-1-1": 100, "weg2-1-3": 70}, 1: {"weg2-1-1": 50, "weg2-1-2": 90},
             2: {"weg2-1-1": 50, "weg2-1-3": 70}}
    b = _Budget(monkeypatch, votes)
    rids = [["weg2-1-1", "weg2-1-3"], ["weg2-1-1", "weg2-1-2"], ["weg2-1-1", "weg2-1-3"]]
    _s, _errs, count = b.run(rids)
    assert len(set(count)) == 1, count


def test_a6_control_a_neutral_abstention_WOULD_split_the_group(monkeypatch):
    """Sensitivity control (the mutant lives in the test): if the empty vote abstained with a
    MIN-NEUTRAL value instead of ABSENT, the raising rank would skip the realize round while the
    others enter it -- the hang 1537 feared. Proves the harness sees the split; production does not
    have it because ABSENT is MIN-absorbing (tp_match_floor.py:271-290)."""
    from sglang.srt.managers import tp_match_floor as TMF

    real = TMF.build_usable_match_payload

    def neutral(canonical, local_usable, slots):
        return [10 ** 9 if v == TMF.ABSENT else v for v in real(canonical, local_usable, slots)]

    b = _Budget(monkeypatch, SKEW, boom={1})
    monkeypatch.setattr(TMF, "build_usable_match_payload", neutral)
    _s, errs, count = b.run(_same_rids())
    assert any(isinstance(e, Hang) for e in errs) or len(set(count)) > 1, (errs, count)


# ---------------------------------------------------------------------------
# (b) _weg2_post_wake_settle_tick: _cap_wait is taken BEFORE the group MIN (scheduler.py:6855-6866)
# ---------------------------------------------------------------------------
PAGE = 64


def _hold_req(rid, tokens):
    import time

    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * tokens, output_ids=[], stream=True,
                                 _weg2_248_read_at_wake=True, _1471_since=time.monotonic() - 1.0,
                                 prefetch_deferred=None)


def _settle_rank(rank, rv, *, arena_slots, reqs):
    """One rank of the post-wake settle tick; the REAL ``Scheduler._weg2_post_wake_settle_tick`` and
    the REAL park_l3 issue/wait logic. ``arena_slots=None`` = this rank has no arena bound
    (``_arena_for`` -> None, hicache_storage.py:3048-3051 'disk only')."""
    from sglang.srt.managers.scheduler import Scheduler as Sch

    arena = None if arena_slots is None else types.SimpleNamespace(slots=arena_slots)
    cc = types.SimpleNamespace(page_size=PAGE, mem_pool_host=types.SimpleNamespace(arena=arena),
                               weg2_hold_rids=set())
    s = types.SimpleNamespace(
        weg2_post_wake_settle=list(reqs), weg2_dormant=False, waiting_queue=[], _weg2_wake_seq=1,
        tree_cache=types.SimpleNamespace(check_hicache_events=lambda: None, cache_controller=cc,
                                         prefetch_loaded_tokens_by_reqid={}),
        _weg2_group_min_flags=rv.flags(rank),
        _weg2_refetch_one=lambda req, now, allow_reissue=False: "complete",
        _prefetch_kvcache=lambda req: "issued",
        WEG2_POST_WAKE_SETTLE_S=Sch.WEG2_POST_WAKE_SETTLE_S)
    s.tick = lambda: Sch._weg2_post_wake_settle_tick(s)
    s.all_reqs = list(reqs)
    return s


def _issue_at_wake(s):
    from sglang.srt.weg2 import park_l3

    return park_l3.issue_deferred_reads(s, list(s.weg2_post_wake_settle))


def _settle_group(arena_slots_by_rank, *, tokens=(60 * PAGE, 60 * PAGE), pre_tick=None):
    """3 ranks, the same two held requests (60 pages each). Arena 100 slots: the 2nd read waits."""
    rv = Rendezvous()
    ranks = []
    for rk in range(N):
        reqs = [_hold_req("weg2-1-%d" % i, n) for i, n in enumerate(tokens)]
        ranks.append(_settle_rank(rk, rv, arena_slots=arena_slots_by_rank[rk], reqs=reqs))
    for s in ranks:
        _issue_at_wake(s)
    if pre_tick:
        pre_tick(ranks)
    res = lockstep(ranks, lambda s: s.tick())
    return ranks, [e for _v, e in res], [v for v, _e in res], rv.count()


def test_b0_equal_ranks_one_read_waits_for_arena_room_every_rank_takes_the_same_collectives():
    from sglang.srt.weg2 import park_l3

    ranks, errs, _vals, count = _settle_group([100, 100, 100])
    assert errs == [None] * N, [repr(e) for e in errs]
    assert [park_l3.capacity_waiting(s.all_reqs[1]) for s in ranks] == [True] * N
    assert len(set(count)) == 1 and count[0] >= 1, count


def test_b1_equal_ranks_every_member_cap_waiting_every_rank_takes_the_same_collectives():
    """1537b (b) patch: the cap-waiters are NOT stripped before the vote. With every member cap-waiting
    on every rank the tick still enters the same two collectives (due|decide, release flags) on each
    rank, votes 0 for each waiter, releases nothing and keeps both parked (the base returned 0 before
    any collective: [0,0,0])."""
    from sglang.srt.weg2 import park_l3

    def hold_all(ranks):
        for s in ranks:
            # an older read of this wake, released to the queue and not yet admitted, holds the arena
            s.waiting_queue.append(types.SimpleNamespace(
                rid="weg2-0-0", **{park_l3.PAGES_ATTR: 1000, park_l3.WAKE_ATTR: 1}))
            for r in s.weg2_post_wake_settle:
                setattr(r, park_l3.CAPWAIT_ATTR, True)
                setattr(r, park_l3.DEFER_ATTR, True)

    ranks, errs, vals, count = _settle_group([100, 100, 100], pre_tick=hold_all)
    assert errs == [None] * N and vals == [0] * N and count == [2, 2, 2], (errs, vals, count)
    assert [[str(r.rid) for r in s.weg2_post_wake_settle] for s in ranks] == [["weg2-1-0", "weg2-1-1"]] * N


def test_b2_one_rank_has_no_arena_bound_the_settle_collective_stays_uniform():
    """1537b (b) patch: a rank without an arena (cap None) issues EVERY hold read, the others keep the
    2nd one cap-waiting. The vote vector is now built from the replicated settle, so the three ranks
    meet in the same collectives with the same length, the 2nd member is not released (a waiter votes
    0, MIN) and every rank keeps the same settle list. (Red on the base: Mismatch [2,2,4].)"""
    ranks, errs, vals, count = _settle_group([100, None, 100])
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(count)) == 1, count
    assert len({tuple(str(r.rid) for r in s.weg2_post_wake_settle) for s in ranks}) == 1
    assert ["weg2-1-1" in {str(r.rid) for r in s.weg2_post_wake_settle} for s in ranks] == [True] * N


def test_b3_issue_capacity_waiters_raises_on_one_rank_the_settle_collective_stays_uniform(monkeypatch):
    """1537b (b) patch: park_l3.issue_capacity_waiters raising on ONE rank leaves its ``_cap_wait`` empty
    (scheduler.py `except`), the peers' waiters stay waiting. No strip any more -> same vector on all
    ranks; the waiter is not released (peers vote 0) and stays in every settle. (Red on the base.)"""
    from sglang.srt.weg2 import park_l3

    real = park_l3.issue_capacity_waiters

    def flaky(sched, settle):
        if getattr(sched, "_boom", False):
            raise RuntimeError("1537b: this rank's capacity issue failed")
        return real(sched, settle)

    monkeypatch.setattr(park_l3, "issue_capacity_waiters", flaky)

    def arm(ranks):
        ranks[1]._boom = True

    ranks, errs, _vals, count = _settle_group([100, 100, 100], pre_tick=arm)
    assert errs == [None] * N, [repr(e) for e in errs]
    assert len(set(count)) == 1, count
    assert len({tuple(str(r.rid) for r in s.weg2_post_wake_settle) for s in ranks}) == 1


def test_b4_the_capacity_decision_reads_no_rank_identity_and_no_pool_state():
    """What makes b0/b1 uniform: ``read_pages`` is request fields / page size, ``_in_flight_pages``
    the settle/queue lists and per-request stamps, ``arena_capacity`` the shared arena's slot count.
    A NEW rank-local term in the capacity decision (a rank id, a pool occupancy) turns this red
    (park_l3.py:144-186, :211-228, :238-268). It does NOT cover arena presence (b2) -- that is the
    open door."""
    import ast
    import inspect
    import textwrap

    from sglang.srt.weg2 import park_l3

    seen = set()
    for fn in (park_l3.read_pages, park_l3.capacity_waiting, park_l3._in_flight_pages,
               park_l3.issue_deferred_reads, park_l3.issue_capacity_waiters, park_l3.arena_capacity):
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
            if isinstance(node, ast.Attribute):
                seen.add(node.attr)
    banned = {"tp_rank", "rank", "attn_tp_rank", "dp_rank", "local_rank", "free_pages", "available_size",
              "evictable_size"}
    assert not (seen & banned), sorted(seen & banned)


# ---------------------------------------------------------------------------
# (c) capacity_park_precondition reads req._weg2_store_short_fallback BEFORE the vote
#     (scheduler.py:14177 / :14180; resume_via_p.py:270-286). HFB rc12z21 (Form A) = the split.
# ---------------------------------------------------------------------------
RID_C = "weg2-8-34"
SYNCED_END, DELIVERABLE_END = 25920, 28480        # HFB rc12z21 D 13:35:14: the group's one answer


def _outcome(matched, loaded, *, synced_end=None):
    from sglang.srt.mem_cache.hicache_storage import PrefetchOutcome

    o = PrefetchOutcome(loaded, matched=matched, deliverable=DELIVERABLE_END, synced=SYNCED_END)
    if synced_end is not None:
        o.synced_end, o.deliverable_end = synced_end, DELIVERABLE_END
    return o


def _short_rank():
    from sglang.srt.managers import scheduler as S

    s = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=1, tp_size=N), weg2_dormant=False,
        tree_cache=types.SimpleNamespace(prefetch_loaded_tokens_by_reqid={}),
        server_args=types.SimpleNamespace(tp_prefill_max_tokens=0),
        _clear_prefetch_deferral_fields=lambda req: None)
    # the drain hands the verdict to the deferral exactly as scheduler.py:9447 does for a fresh mark
    s._apply_prefetch_deferral = lambda req, verdict, site: S._weg2_store_short_fallback(
        s, req, S._DEFER_REASON_STORE_SHORT, 1, site)
    s.note = lambda req: S.Scheduler._weg2_note_store_shortfall(s, req)
    req = types.SimpleNamespace(rid=RID_C, stream=True, multimodal_inputs=None,
                                full_untruncated_fill_ids=[0] * 30000)
    return s, req


def _passes(monkeypatch, *, form_a_ends, passes=8):
    """3 ranks read the SAME store read for 8 passes. TP0: materialized 13376 (12544 + 832) every
    pass, no growth; the workers' span starts differ: 9536 + 64 per pass. ``form_a_ends`` = the
    outcome carries the group's absolute ends (the HFB fix reads them)."""
    from sglang.srt.weg2 import resume_via_p as rvp

    monkeypatch.setattr(rvp, "capacity_park_on", lambda: True)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    ranks = [_short_rank() for _ in range(N)]
    trace = []
    for p in range(passes):
        row = []
        for rk, (s, req) in enumerate(ranks):
            m = (12544, 832) if rk == 0 else (0, 9536 + 64 * p)
            s.tree_cache.prefetch_loaded_tokens_by_reqid[RID_C] = _outcome(
                *m, synced_end=SYNCED_END if form_a_ends else None)
            s.note(req)
            row.append((int(getattr(req, "_weg2_store_delivered", -1)),
                        bool(getattr(req, "_weg2_store_short_fallback", False)),
                        rvp.capacity_park_precondition(req)))
        trace.append(row)
    return trace


def test_c1_form_a_end_vote_the_precondition_is_the_same_on_every_rank_on_every_pass(monkeypatch):
    """HFB rc12z21 (scheduler.py:15174-15183): with the group's absolute ends in the record the
    witness term ``_weg2_store_delivered`` is the group's number, so the store-short cycle bound
    trips on the SAME pass on every rank and ``capacity_park_precondition`` -- which decides whether
    the group takes the MIN vote at :14180 -- never differs by rank."""
    trace = _passes(monkeypatch, form_a_ends=True)
    for p, row in enumerate(trace):
        assert len({r for r in row}) == 1, (p, row)
    assert {r[0] for r in trace[0]} == {SYNCED_END}
    assert [row[0][2] for row in trace].count(True) >= 1, "the bound never tripped: the test would pass vacuously"


def test_c2_control_without_the_group_ends_the_ranks_split_on_the_precondition(monkeypatch):
    """The incident itself (the mutant is the input): span-relative ``materialized`` -- TP0 sees no
    growth and trips the bound ('cycles=5 > bound=4'), the workers see growth and never do -> TP0
    takes the vote at :14180, the workers do not = Hang. This is why the ``synced_end`` read in
    ``_weg2_note_store_shortfall`` is load-bearing (and why it must stay ABOVE the witness write)."""
    trace = _passes(monkeypatch, form_a_ends=False)
    pre = [[r[2] for r in row] for row in trace]
    assert any(len(set(row)) > 1 for row in pre), pre
    assert pre[-1] == [True, False, False], pre[-1]


def test_c3_the_replicated_half_reads_exactly_these_request_terms():
    """``capacity_park_precondition`` is the entry condition of the group vote: it must read only
    replicated terms. Its read set is pinned (switch, env group, stream, multimodal_inputs,
    ``_weg2_store_short_fallback``); a new rank-local read here turns this red."""
    import ast
    import inspect
    import textwrap

    from sglang.srt.weg2 import resume_via_p as rvp

    tree = ast.parse(textwrap.dedent(inspect.getsource(rvp.capacity_park_precondition)))
    reads = sorted(n.args[1].value for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr"
                   and len(n.args) > 1 and isinstance(n.args[1], ast.Constant))
    assert reads == ["_weg2_store_short_fallback", "multimodal_inputs", "stream"], reads
    calls = sorted(n.func.id for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id != "getattr")
    assert calls == ["bool", "bool", "capacity_park_on"], calls


def test_c4_the_group_end_is_read_before_the_witness_term_is_written():
    """Order inside ``_weg2_note_store_shortfall``: ``synced_end`` is read (and ``delivered`` replaced by
    it) BEFORE ``req._weg2_store_delivered`` is written (scheduler.py:15183-15200). Moving the read
    below the write reproduces the HFB split for every Form A read."""
    import ast
    import inspect
    import textwrap

    from sglang.srt.managers.scheduler import Scheduler

    tree = ast.parse(textwrap.dedent(inspect.getsource(Scheduler._weg2_note_store_shortfall)))
    pos = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and n.value == "synced_end":
            pos.setdefault("read", n.lineno)
        if isinstance(n, ast.Attribute) and n.attr == "_weg2_store_delivered" and isinstance(n.ctx, ast.Store):
            pos.setdefault("write", n.lineno)
    assert set(pos) == {"read", "write"} and pos["read"] < pos["write"], pos
