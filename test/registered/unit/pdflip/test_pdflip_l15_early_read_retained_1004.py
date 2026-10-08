# SPDX-License-Identifier: Apache-2.0
"""L15-EARLY-READ (04.10., desk/27b-l15-fix-a-1004): under a RETAINED L1.5 hold
the #248 hold reads run beside the weight legs again -- for the rids the group
does NOT keep on the card, only when every rank predicts the hold, and a tree
reset after the issue re-arms them.

THE METAL (INT8 boot 4cf740ad50, deskq/done/1240-int8-flip-nachlauf.out): under
the retained hold early_enabled() was False -> no WAKE-READ-EARLY-SPREAD in any
hold wake, #1471 held_after_wake_s 1.3-2.4 s, a second EXTEND pass: P>D nachlauf
2.69 s with the hold against 1.81 s without (layer equal).

THE DANGER (N4f 1002_110321, park_l3.early_enabled docstring): the wake's L15
path RESET the tree after the early read; the settle reads an absent record as
"complete" (``_pdflip_refetch_one``), the request met the X gate with nothing on
the host: W31, W50-REROUTE, ping-pong. Closed at the cause: the fallback act
and the no-hold drop re-arm the early reads (``park_l3.rearm_early_reads``).

Hermetic, CPU, no arena: stand-in scheduler, the real park_l3 issue / filter /
re-arm, the real ``Scheduler._pdflip_release_dormant_hold`` and
``_pdflip_refetch_one`` bound to it.
"""
from __future__ import annotations

import functools
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import cache_controller as _cc  # noqa: E402
from flliper.srt.managers.scheduler import Scheduler  # noqa: E402
from flliper.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from flliper.srt.pdflip import park_l3  # noqa: E402

HELD_RID = "pdflip-0-7"      # kept on the card by the L1.5 hold (a parked decode)
NEW_A = "pdflip-1-14"        # a P hand-off: NOT held, must be read from the store
NEW_B = "pdflip-1-15"


class _Req(types.SimpleNamespace):
    pass


def _req(rid, n=12):
    return _Req(rid=rid, origin_input_ids=list(range(n)), output_ids=[], kv_arrival_seq=None,
                time_stats=types.SimpleNamespace(set_wait_queue_entry_time=lambda: None),
                extra_key=None)


class _Tree:
    """check_prefetch_progress: a registered read is 'reading' until done;
    reset() = the radix tree reset of the wake (records and pins wiped)."""

    def __init__(self):
        self.ongoing_prefetch = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.cache_controller = types.SimpleNamespace(pdflip_hold_rids=set())
        self.resets = 0

    def check_prefetch_progress(self, rid):
        return rid not in self.ongoing_prefetch

    def reset(self):
        self.resets += 1
        self.ongoing_prefetch.clear()
        self.prefetch_loaded_tokens_by_reqid.clear()


def _sched(tree, reads, retained):
    s = types.SimpleNamespace(tree_cache=tree, waiting_queue=[], pdflip_post_wake_settle=[],
                              ps=types.SimpleNamespace(tp_size=1), server_args=types.SimpleNamespace(),
                              _l15_tree_retained=retained)

    def _read(req, **kw):
        reads.append(req.rid)
        tree.ongoing_prefetch[req.rid] = object()
        tree.prefetch_loaded_tokens_by_reqid[str(req.rid)] = 12   # a complete read's record once done
        return "issued"

    s._prefetch_kvcache = _read
    s._apply_prefetch_deferral = lambda req, verdict, site: None
    s._pdflip_refetch_one = functools.partial(Scheduler._pdflip_refetch_one, s)
    s._pdflip_group_min_flags = functools.partial(Scheduler._pdflip_group_min_flags, s)
    s._pdflip_note_store_shortfall = lambda req: None
    return s


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_HANDOFF", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_L15", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_L15_SLEEP_AGREE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_L15_EARLY_READ", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_WAKE_READ_EARLY", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_WAKE_READ_EARLY_SPREAD", raising=False)
    yield monkeypatch
    _cc.PDFLIP_HANDOFF_PAGE_KEYS.clear()


def _hold(s, rids):
    reqs = [_req(r) for r in rids]
    for r in reqs:
        park_l3.defer_hold_read(s, r)
    s.pdflip_dormant_hold = list(reqs)
    return reqs


def _agree(monkeypatch, held):
    monkeypatch.setattr(park_l3, "l15_agreed_held_rids", lambda s, gather=None: set(held))


# (b) + (c) -----------------------------------------------------------------------
def test_retained_hold_reads_the_non_held_rids_early_and_not_the_held_one(env):
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    assert park_l3.early_enabled(s) is True
    got = park_l3.issue_reads_at_wake_begin(s)
    assert got == [a, b], "(c) the non-held rids are read early, in hold order"
    assert reads == [NEW_A, NEW_B], "(b) the rid the hold keeps on the card is NOT read early"
    assert park_l3.deferred(held) and not park_l3.issued(held)


# (a) -----------------------------------------------------------------------------
def test_verdict_hold_settle_complete_no_second_read_no_w31_path(env):
    """The hold verdict resets nothing: the early reads that completed beside the legs
    release 'complete' at the wake -- no settle, no re-read of them."""
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    park_l3.issue_reads_at_wake_begin(s)
    tree.ongoing_prefetch.clear()                       # the legs outlast the host reads
    n = Scheduler._pdflip_release_dormant_hold(s)
    assert reads.count(NEW_A) == 1 and reads.count(NEW_B) == 1, "no re-read of an early read"
    # the held rid's own read is issued at the release (still in flight in the stand-in): it alone waits
    assert s.pdflip_post_wake_settle == [held] and n == 2, "the early reads: state=complete, nothing parked"
    assert a in s.waiting_queue and b in s.waiting_queue


# DANGER DIRECTION (N4f) --------------------------------------------------------------
def test_n4f_a_tree_reset_after_the_early_read_leaves_a_blind_settle_without_the_rearm(env):
    """Documents the loss: reset wipes the records, the settle answers 'complete'."""
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    park_l3.issue_reads_at_wake_begin(s)
    tree.ongoing_prefetch.clear()
    tree.reset()                                        # the fallback drop, no re-arm
    Scheduler._pdflip_release_dormant_hold(s)
    assert a in s.waiting_queue and b in s.waiting_queue, \
        "blind: a/b released as complete with nothing on the host"
    assert reads.count(NEW_A) == 1, "and never re-read -- the N4f W31 shape"


def test_fallback_act_rearms_the_early_reads_so_the_release_reads_after_the_reset(env):
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    park_l3.issue_reads_at_wake_begin(s)
    tree.ongoing_prefetch.clear()
    mgr = types.SimpleNamespace(
        _l15_fallback_drop=lambda sched: tree.reset() or 3,
        _l15_wake_refill=False, _l15_release_host_hold_refs=None)
    dropped = wu.SchedulerWeightUpdaterManager._l15_wake_act(
        mgr, s, "fallback", group_ok=True, master_on=True)
    assert dropped == 3 and tree.resets == 1
    assert park_l3.deferred(a) and park_l3.deferred(b) and not park_l3.issued(a)
    Scheduler._pdflip_release_dormant_hold(s)
    assert reads.count(NEW_A) == 2 and reads.count(NEW_B) == 2, "re-read AFTER the reset, at the release"
    assert reads.count(HELD_RID) == 1


def test_no_hold_drop_rearms_too(env):
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    s.tree_cache.reset = tree.reset
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    park_l3.issue_reads_at_wake_begin(s)
    assert wu._l15_drop_retained_tree(s, 0) is True and tree.resets == 1
    assert park_l3.deferred(a) and park_l3.deferred(b)


def test_hold_and_none_verdicts_rearm_nothing(env):
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    park_l3.issue_reads_at_wake_begin(s)
    mgr = types.SimpleNamespace(_l15_fallback_drop=lambda sched: pytest.fail("no drop on hold"),
                                _l15_wake_refill=False, _l15_release_host_hold_refs=None)
    for verdict in ("hold", "none"):
        wu.SchedulerWeightUpdaterManager._l15_wake_act(mgr, s, verdict, group_ok=True, master_on=True)
    assert park_l3.issued(a) and park_l3.issued(b) and not park_l3.deferred(a)


# retained but the group does not predict the hold -> no early read ---------------------------------
def test_retained_without_an_agreed_hold_reads_at_the_release_only(env):
    _agree(env, set())                                  # a rank without its manifest / mixed fps
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    assert park_l3.issue_reads_at_wake_begin(s) == [] and reads == []
    Scheduler._pdflip_release_dormant_hold(s)
    assert sorted(reads) == sorted([HELD_RID, NEW_A, NEW_B])


# (d) rank agreement ---------------------------------------------------------------------------
def test_every_rank_decides_alike_from_the_gathered_votes(env, monkeypatch):
    """The decision depends only on group-uniform data: the retained flag (SLEEP-AGREE: every
    rank or none), the gathered vote list (identical on every rank), the hold list."""
    votes_ok = [(7, ["a"]), (7, ["a"]), (7, ["a"])]
    votes_split = [(7, ["a"]), None, (7, ["a"])]
    votes_fp = [(7, ["a"]), (8, ["a"]), (7, ["a"])]

    def run(votes, rank):
        monkeypatch.setattr(park_l3, "l15_agreed_held_rids",
                            lambda sched, gather=None: (
                                set(votes[0][1]) if votes and votes[0] is not None
                                and all(v == votes[0] for v in votes) else set()))
        reads, tree = [], _Tree()
        s = _sched(tree, reads, retained=True)
        _hold(s, ["a", NEW_A, NEW_B])
        park_l3.issue_reads_at_wake_begin(s)
        return reads

    for votes, want in ((votes_ok, [NEW_A, NEW_B]), (votes_split, []), (votes_fp, [])):
        assert [run(votes, r) for r in range(3)] == [want] * 3


def test_the_real_vote_reduction_is_identical_on_every_rank():
    vote = (5, ["a", "b"])
    gathered = [vote, vote, vote]
    outs = [park_l3.l15_agreed_held_rids(types.SimpleNamespace(), gather=lambda v: list(gathered))
            for _ in range(3)]
    # master off here (env not set): no collective, empty set on every rank alike
    assert outs == [set(), set(), set()]


def test_no_per_rank_term_before_a_collective(env):
    """early_enabled reads the retained flag and two env switches only -- no rank-local
    measurement; the gather inside issue_reads_at_wake_begin runs at the same position on
    every rank (hold list nonempty is the group's)."""
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    calls = []
    env.setattr(park_l3, "l15_agreed_held_rids", lambda sched, gather=None: calls.append(1) or {HELD_RID})
    _hold(s, [HELD_RID, NEW_A])
    s.pdflip_dormant_hold = []                             # empty hold: no gather, no read
    assert park_l3.issue_reads_at_wake_begin(s) == [] and calls == []


# (e) the switch -----------------------------------------------------------------------------------
def test_switch_off_is_the_4cf740ad50_shape(env):
    env.setenv("FLLIPER_PDFLIP_L15_EARLY_READ", "0")
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    assert park_l3.early_enabled(s) is False
    assert park_l3.issue_reads_at_wake_begin(s) == [] and reads == []
    Scheduler._pdflip_release_dormant_hold(s)
    assert sorted(reads) == sorted([HELD_RID, NEW_A, NEW_B]), "everything read at the release"
    assert park_l3.rearm_early_reads(s, "x") == 0


# mandatory: L15 off / no hold -> unchanged ------------------------------------------------------------
def test_l15_off_early_read_unchanged_and_no_rearm(env):
    env.setenv("FLLIPER_PDFLIP_L15", "0")
    called = []
    env.setattr(park_l3, "l15_agreed_held_rids", lambda s, gather=None: called.append(1) or set())
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=False)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    assert park_l3.early_enabled() is True and park_l3.early_enabled(s) is True
    got = park_l3.issue_reads_at_wake_begin(s)
    assert got == [held, a, b] and reads == [HELD_RID, NEW_A, NEW_B]
    assert park_l3.rearm_early_reads(s, "x") == 3 and park_l3.deferred(a)  # only a reset calls it


def test_l15_on_no_retained_hold_unchanged(env):
    """Plain sleep under L1.5 (L15-ON-READ-EARLY): reads everything early, no gating."""
    _agree(env, set())
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=False)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    assert park_l3.early_enabled(s) is True
    assert park_l3.issue_reads_at_wake_begin(s) == [held, a, b]


def test_the_wake_act_without_early_reads_rearms_nothing(env):
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    s.pdflip_dormant_hold = []
    mgr = types.SimpleNamespace(_l15_fallback_drop=lambda sched: 2, _l15_wake_refill=False,
                                _l15_release_host_hold_refs=None)
    assert wu.SchedulerWeightUpdaterManager._l15_wake_act(
        mgr, s, "fallback", group_ok=True, master_on=True) == 2
    assert wu._l15_rearm_early_reads(types.SimpleNamespace(), "x") == 0


def test_the_spread_asks_the_group_once_and_stays_rank_uniform(env):
    """PDFLIP-S under the retained hold: one read per tag after the first gather, the
    gather cached on the wake key; non-held only."""
    asks = []
    env.setattr(park_l3, "l15_agreed_held_rids", lambda s, gather=None: asks.append(1) or {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    s._pdflip_wake_seq, s.pdflip_dormant = 4, True
    _hold(s, [HELD_RID, NEW_A, NEW_B])
    out = []
    for _ in range(3):
        out += park_l3.issue_reads_at_wake_begin(s, max_n=1)
    assert [r.rid for r in out] == [NEW_A, NEW_B] and len(asks) == 1


# review 1260 (5): the full-flush paths reset the tree too ---------------------------------------
def test_flush_fallback_in_restore_pools_rearms_the_early_reads(env):
    """_pdflip_wake_restore_pools falls back to flush_cache() on an exception: the tree is reset after
    the early read -- re-armed (without it the settle is blind, see the N4f test above)."""
    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    held, a, b = _hold(s, [HELD_RID, NEW_A, NEW_B])
    park_l3.issue_reads_at_wake_begin(s)
    tree.ongoing_prefetch.clear()
    mgr = types.SimpleNamespace(
        scheduler=s, _l15_wake_hold_signal=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        flush_cache=lambda: tree.reset() or True)
    assert wu.SchedulerWeightUpdaterManager._pdflip_wake_restore_pools(mgr) is True
    assert tree.resets == 1 and park_l3.deferred(a) and park_l3.deferred(b)
    Scheduler._pdflip_release_dormant_hold(s)
    assert reads.count(NEW_A) == 2, "re-read after the flush's reset, not a blind 'complete'"


def test_the_wake_flush_branch_calls_the_rearm_after_the_flush():
    import inspect

    src = inspect.getsource(wu.SchedulerWeightUpdaterManager)
    i = src.index('flushed = self.flush_cache()')
    assert '_l15_rearm_early_reads(scheduler, "flush")' in src[i:i + 400]


# review 1260 (6): the prediction's milliseconds are logged, behaviour unchanged ------------------
def test_the_prediction_logs_its_milliseconds_once_per_wake(env, caplog):
    import logging

    _agree(env, {HELD_RID})
    reads, tree = [], _Tree()
    s = _sched(tree, reads, retained=True)
    s._pdflip_wake_seq, s.pdflip_dormant = 4, True
    _hold(s, [HELD_RID, NEW_A, NEW_B])
    with caplog.at_level(logging.INFO, logger=park_l3.logger.name):
        out = []
        for _ in range(3):
            out += park_l3.issue_reads_at_wake_begin(s, max_n=1)
    lines = [r.getMessage() for r in caplog.records if "L15-EARLY-READ predict_ms=" in r.getMessage()]
    assert len(lines) == 1 and "retained=True agreed=1" in lines[0]
    assert [r.rid for r in out] == [NEW_A, NEW_B]
