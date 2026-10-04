# SPDX-License-Identifier: Apache-2.0
"""Q-1500 UD-V2: the dual P group trims the shared KV arena BEFORE the wall (desk analysis 1290, y9d3).

THE METAL (1290 Gl. 7-10): the shared arena filled to 98-99 % pinned, the holder was P itself (host-only
leaves of every finished leg; P is never idle under load), D gave back ~91k pages (Q-1190) and arena_pinned
rose anyway, every backup was refused (#1421 arena_claim) and the P pool died on its last device leaf.

WHAT IS PINNED (the REAL ``UnifiedRadixCache`` on CPU, host-only leaves, a faked arena header whose
``pinned`` is the tree's own reference count -- plus one test on the real C arena):
* fill 0.97 with host-only leaves: the base gives nothing (RED: no seam), V2 gives back down to <= 0.80 with
  every released leaf L3-secured first (GREEN);
* fill below HI: nothing, and the strided header census is never read (``stats()`` is the pre-filter);
* hysteresis (an episode ends at LO, and a fill between LO and HI starts none), minimum distance between two
  decisions, backoff after an order PP0 itself could not serve;
* gates: flip / NF / INT8 (no dual layout), dual D, uncapped P, the switch at 0, a follower stage -- no header
  read, no order, the list is the very same object;
* RANK CONGRUENCE: three stages through the REAL ``_pp_forward_and_process_input_requests``: PP0's order rides
  the wire, the followers relay it and dispatch without it, all three stages execute the SAME order and end
  with the SAME tree; no follower reads the header;
* host-locked / pending-write / aux-state / child-carrying / L3-refused nodes stay; ``sanity_check`` holds.
The test file imports the V2 names through ``getattr`` so that on the base it fails by BEHAVIOUR (nothing is
given back), not by an import error.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import os
import pickle
import shutil
import sys
import types
from array import array

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(__file__))

from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.weg2 import dual_arena_spill as D  # noqa: E402
from sglang.srt.weg2 import dual_p_kv_stage as PK  # noqa: E402

from test_unified_radix_cache_unittest import CacheConfig, build_fixture  # noqa: E402

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
SLOTS = 100
# the V2 names through getattr: on the base this file loads and fails by BEHAVIOUR (nothing is given back)
TRIM_ENV = getattr(D, "TRIM_ENV", "SGLANG_WEG2_DUAL_ARENA_TRIM")
TRIM_HI_ENV = getattr(D, "TRIM_HI_ENV", TRIM_ENV + "_HI")
TRIM_LO_ENV = getattr(D, "TRIM_LO_ENV", TRIM_ENV + "_LO")
TRIM_MIN_S_ENV = getattr(D, "TRIM_MIN_S_ENV", TRIM_ENV + "_MIN_S")
TRIM_MAX_PAGES_ENV = getattr(D, "TRIM_MAX_PAGES_ENV", TRIM_ENV + "_MAX_PAGES")
TRIM_BUDGET_S_ENV = getattr(D, "TRIM_BUDGET_S_ENV", TRIM_ENV + "_BUDGET_S")
TRIM_EMPTY_BACKOFF_S_ENV = getattr(D, "TRIM_EMPTY_BACKOFF_S_ENV", TRIM_ENV + "_EMPTY_BACKOFF_S")
TRIM_MARK = getattr(D, "TRIM_MARK", "Q-1500 UD-V2 DUAL ARENA-TRIM")
DUAL_P = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", PK.MAX_TOKENS_ENV: "131072"}
NOT_DUAL_P = [
    {},                                                                    # flip / NF / INT8
    {"SGLANG_WEG2_GROUP": "P"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", PK.MAX_TOKENS_ENV: "131072"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"},           # no P KV cap: not armed
    dict(DUAL_P, **{TRIM_ENV: "0"}),                                     # the switch
]
_KEYS = ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", PK.MAX_TOKENS_ENV, TRIM_ENV, TRIM_HI_ENV,
         TRIM_LO_ENV, TRIM_MIN_S_ENV, TRIM_MAX_PAGES_ENV, TRIM_BUDGET_S_ENV,
         TRIM_EMPTY_BACKOFF_S_ENV)


def _env(monkeypatch, env):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(autouse=True)
def _fresh():
    # the module counters gate the first-8 log lines of Q-697c / Q-1190: a test of this file must not
    # use them up for the tests that read those lines out of caplog in the same process
    saved = (dict(D._N), dict(D._Y))
    _reset()
    yield
    _reset()
    D._N.clear(); D._N.update(saved[0])
    D._Y.clear(); D._Y.update(saved[1])


def _reset():
    getattr(D, "_reset_trim_for_tests", lambda: None)()


class _WriteBackController:
    write_policy = "write_back"

    def append_host_mem_release(self, *args, **kwargs):
        return None


class _FakeArena:
    """The shared header as the tree sees it: ``pinned`` = the pages this tree references (one stage = one
    holder here), ``complete`` adds slots nobody references. Counts what is read."""

    def __init__(self, cache, slots=SLOTS, unreferenced=0):
        self.cache, self.slots, self.unreferenced = cache, slots, unreferenced
        self.stats_calls = 0
        self.census_calls = 0

    def _pinned(self):
        return sum(int(n.component_data[FULL].host_value.numel()) for n in self.cache._collect_all_nodes()
                   if n is not self.cache.root_node and n.component_data[FULL].host_value is not None)

    def stats(self):
        self.stats_calls += 1
        return {"slots": self.slots, "complete": self._pinned() + self.unreferenced, "claimed": 0,
                "slot_bytes": 64, "file_bytes": 0}

    def ref_census(self):
        self.census_calls += 1
        p = self._pinned()
        return p, p, p + self.unreferenced


class _FakePool:
    """Carries what ``spill_host_only`` / ``_spill_pool`` ask for. ``refuse``: host-row bases whose L3 copy fails."""

    staging_rows = 0

    def __init__(self, cache, refuse=(), slots=SLOTS):
        self.arena = _FakeArena(cache, slots)
        self.refuse = set(refuse)
        self.secured = []

    def secure_rows_to_l3(self, rows):
        base = int(rows.min())
        if base in self.refuse:
            return {"lost": int(rows.numel()), "pages": 0, "on_disk": 0, "written": 0}
        self.secured.append(base)
        return {"lost": 0, "pages": int(rows.numel()), "on_disk": 0, "written": int(rows.numel())}


def _host_leaf(cache, token, base):
    key = RadixKey(array("q", [token]))
    host = torch.tensor([base], dtype=torch.int64)
    res = cache._insert_helper_host(cache.root_node, key, host, [f"h{token}"])
    assert res.inserted_host_node is not None
    return res.inserted_host_node


def _host_child(cache, parent_token, token, base):
    """A host-only node below the single-page leaf ``parent_token`` (the tree's own writer)."""
    key = RadixKey(array("q", [parent_token, token]))
    host = torch.tensor([base - 1, base], dtype=torch.int64)
    res = cache._insert_helper_host(cache.root_node, key, host, [f"h{parent_token}", f"h{token}"])
    assert res.inserted_host_node is not None
    return res.inserted_host_node


def _stage(n_leaves, *, pp_rank=0, refuse=()):
    """One P stage: a real tree of ``n_leaves`` host-only single-page leaves (node-id order = token order)."""
    # the dual gate armed while the fixture is built makes the pool constructor demand a saver allocation
    # (V1's note): build with the gate keys out, restore them for the test body
    saved = {k: os.environ.pop(k) for k in _KEYS if k in os.environ}
    try:
        cache, _alloc, _r2t = build_fixture(CacheConfig(page_size=1, components=(FULL, MAMBA)))
    finally:
        os.environ.update(saved)
    cache.cache_controller = _WriteBackController()
    nodes = [_host_leaf(cache, 10 + i, 7000 + i) for i in range(n_leaves)]
    pool = _FakePool(cache, refuse=[7000 + i for i in refuse])
    cache._weg2_direct_pool = lambda: pool
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=3), tree_cache=cache)
    return sched, cache, pool, nodes


def _live(cache):
    return {n.id for n in cache._collect_all_nodes() if n is not cache.root_node}


def _live_tokens(cache):
    """The tokens of the live nodes: node ids come from a process-wide counter, so the replica trees of
    one process differ in id but not in key."""
    return {tuple(int(t) for t in n.key.token_ids) for n in cache._collect_all_nodes() if n is not cache.root_node}


def _fill(pool):
    return pool.arena._pinned() / float(pool.arena.slots)


def _tick(sched, now, cfg=None):
    """The pass seam of PP0 without the wire: stamp, then serve the own order. Absent on the base: a no-op
    (the BEHAVIOUR of the base -- nothing is ever given back)."""
    stamp = getattr(D, "pp0_stamp", None)
    if stamp is None:
        return None
    wire, cmd = stamp(sched, [], now=now)
    if cmd is not None:
        D.execute(sched, cmd)
    return cmd


# -------------------------------------------------------------------------------- the defect and the fix
def test_fill_097_with_host_only_leaves_the_base_gives_nothing_v2_trims_to_lo(monkeypatch, caplog):
    """RED on the base (no seam: the arena stays at 0.97, y9d3). GREEN: <= 0.80, L3 copy first, oldest first."""
    _env(monkeypatch, DUAL_P)
    import logging

    caplog.set_level(logging.INFO)
    sched, cache, pool, nodes = _stage(97)
    assert abs(_fill(pool) - 0.97) < 1e-9
    t = 100.0
    for _ in range(6):                                  # a few passes, 2 s apart (> MIN_S)
        _tick(sched, t)
        t += 2.0
    assert _fill(pool) <= 0.80 + 1e-9, "the arena stayed at %.2f (the y9d3 wall)" % _fill(pool)
    live = _live(cache)
    gone = [n for n in nodes if n.id not in live]
    assert len(gone) == 17                              # exactly pinned - LO*slots = 97 - 80
    assert [n.id for n in gone] == sorted(n.id for n in nodes)[:17]        # node-id order: oldest first
    assert sorted(pool.secured) == [7000 + i for i in range(17)]            # every leaf had its L3 copy first
    assert any(TRIM_MARK in r.getMessage() and "START" in r.getMessage() for r in caplog.records)
    assert any(TRIM_MARK in r.getMessage() and "DONE" in r.getMessage() for r in caplog.records)
    cache.sanity_check()


def test_the_order_is_capped_and_an_episode_runs_several_orders(monkeypatch):
    _env(monkeypatch, dict(DUAL_P, **{TRIM_MAX_PAGES_ENV: "5"}))
    sched, cache, pool, nodes = _stage(97)
    cmds = []
    t = 10.0
    for _ in range(8):
        c = _tick(sched, t)
        if c is not None:
            cmds.append(c)
        t += 2.0
    assert [c.want for c in cmds[:3]] == [5, 5, 5]      # 97 -> 92 -> 87 -> 82 -> next asks 2 (82-80)
    assert [c.seq for c in cmds] == list(range(1, len(cmds) + 1))
    assert _fill(pool) <= 0.80 + 1e-9
    assert len(cmds) == 4 and cmds[3].want == 2


def test_below_hi_nothing_and_the_census_is_never_read(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(85)
    t = 0.0
    for _ in range(5):
        assert _tick(sched, t) is None
        t += 2.0
    assert len(_live(cache)) == 85
    assert pool.arena.census_calls == 0                 # stats() bounds the pinned count: no strided read
    assert pool.arena.stats_calls >= 1
    assert not pool.secured


def test_at_hi_exactly_nothing_above_hi_a_trim(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(90)              # 0.90 == HI: not above
    assert _tick(sched, 0.0) is None
    sched, cache, pool, nodes = _stage(91)
    _reset()
    c = _tick(sched, 0.0)
    assert c is not None and c.want == 11


def test_complete_but_unreferenced_slots_are_no_wall(monkeypatch):
    """``complete`` above HI but ``pinned`` (reader-referenced) below it: the census is read, no order --
    an unreferenced COMPLETE slot is evictable by the claim itself (ARENA-DROP), not the y9d3 wall."""
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(85)
    pool.arena.unreferenced = 10                       # complete = 95 of 100, pinned = 85
    assert _tick(sched, 0.0) is None
    assert pool.arena.census_calls == 1 and len(_live(cache)) == 85


def test_hysteresis_an_episode_ends_at_lo_and_a_fill_between_lo_and_hi_starts_none(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97)
    t = 0.0
    for _ in range(4):
        _tick(sched, t)
        t += 2.0
    assert _fill(pool) <= 0.80 + 1e-9 and not D._T["active"]
    for i in range(6):                                  # the arena grows back to 0.86: between LO and HI
        _host_leaf(cache, 5000 + i, 9000 + i)
    assert 0.80 < _fill(pool) <= 0.90
    assert _tick(sched, t) is None and not D._T["active"]
    for i in range(10):                                 # and above HI again: a new episode
        _host_leaf(cache, 6000 + i, 9100 + i)
    assert _fill(pool) > 0.90
    assert _tick(sched, t + 2.0) is not None and D._T["active"]


def test_minimum_distance_between_two_decisions(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97)
    assert _tick(sched, 100.0) is not None
    reads = pool.arena.stats_calls
    assert _tick(sched, 100.2) is None                  # < MIN_S: no decision, no header read at all
    assert pool.arena.stats_calls == reads
    # after MIN_S a decision again, while the first order did not bring the fill to LO
    _reset()
    sched, cache, pool, nodes = _stage(97)
    monkeypatch.setenv(TRIM_MAX_PAGES_ENV, "3")
    assert _tick(sched, 0.0) is not None
    assert _tick(sched, 0.5) is None
    assert _tick(sched, 1.01) is not None


def test_backoff_after_an_order_pp0_could_not_serve(monkeypatch):
    """Every leaf's L3 copy is refused: released 0 -> the next decision waits EMPTY_BACKOFF_S (2 s)."""
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97, refuse=range(97))
    c = _tick(sched, 100.0)
    assert c is not None and len(_live(cache)) == 97    # nothing left L2: no L3 copy, no release
    assert _tick(sched, 101.2) is None                  # > MIN_S but inside the backoff
    assert _tick(sched, 102.5) is not None              # backoff over


def test_mutants_the_tests_can_fail(monkeypatch):
    """Gate out / threshold out / backoff out each turn at least one pinned assertion red (shown in the
    desk log with the mutated source; here the observable of each mutation)."""
    _env(monkeypatch, DUAL_P)
    # threshold out (HI 1.0: 'never above'): the 0.97 fill stays -- the base behaviour
    monkeypatch.setenv(TRIM_HI_ENV, "1.0")
    sched, cache, pool, nodes = _stage(97)
    assert _tick(sched, 0.0) is None and len(_live(cache)) == 97
    # backoff out (0 s): a refused order is retried at the next MIN_S -- the instrument sees the retry
    monkeypatch.setenv(TRIM_HI_ENV, "0.9")
    monkeypatch.setenv(TRIM_EMPTY_BACKOFF_S_ENV, "0")
    _reset()
    sched, cache, pool, nodes = _stage(97, refuse=range(97))
    assert _tick(sched, 100.0) is not None
    assert _tick(sched, 101.2) is not None


# -------------------------------------------------------------------------------- gates
def test_gates_flip_nf_int8_dual_d_uncapped_and_the_switch_do_nothing(monkeypatch):
    """Flip / NF / INT8 (no dual layout), dual D, an uncapped P, the switch: no header read, no order, the
    very same list object back, no node moves."""
    for env in NOT_DUAL_P:
        _env(monkeypatch, env)
        _reset()
        sched, cache, pool, nodes = _stage(97)
        wire = ["req"]
        out, cmd = D.pp0_stamp(sched, wire, now=1.0)
        assert out is wire and cmd is None, env
        assert pool.arena.stats_calls == 0 and pool.arena.census_calls == 0, env
        assert D.follower_absorb(sched, wire) is wire, env
        assert len(_live(cache)) == 97 and not pool.secured, env
        forced = D.execute(sched, D.Weg2DualArenaTrim(1, 50, 970000), env=None)
        assert forced["released"] == 0 and len(_live(cache)) == 97, env


def test_a_follower_stage_never_decides_and_never_reads_the_header(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97, pp_rank=1)
    out, cmd = D.pp0_stamp(sched, [], now=1.0)
    assert cmd is None
    assert pool.arena.stats_calls == 0 and pool.arena.census_calls == 0


def test_pp_group_of_one_stage_is_not_armed_for_the_wire(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97)
    sched.ps.pp_size = 1
    assert D.pp0_stamp(sched, [], now=1.0)[1] is None


# -------------------------------------------------------------------------------- guards
def test_locked_pending_aux_child_and_l3_refused_nodes_stay_and_the_tree_is_sane(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97, refuse=[3])
    nodes[0].component_data[FULL].host_lock_ref = 1             # a load-back / prefetch pin reads it
    cache.ongoing_write_through[nodes[1].id] = object()          # slots not COMPLETE yet
    nodes[2].hash_value = None                                   # page count cannot be checked
    # nodes[3]: its L3 copy is refused (the pool's refuse list)
    child = _host_child(cache, 14, 60, 7500)                     # nodes[4] (token 14) carries a child: not an H-leaf
    assert child.parent is nodes[4]
    t = 0.0
    for _ in range(12):
        _tick(sched, t)
        t += 2.0
    live = _live(cache)
    for k in (0, 1, 2, 3, 4):
        assert nodes[k].id in live, k
    assert 3 not in pool.secured and 7003 not in pool.secured
    # the rest went in id order, never an orphan
    assert [n.id for n in nodes[5:] if n.id not in live] == sorted(n.id for n in nodes[5:] if n.id not in live)
    nodes[0].component_data[FULL].host_lock_ref = 0
    cache.ongoing_write_through.pop(nodes[1].id)
    cache.sanity_check()


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_post_d_switch_zero_posts_nothing(tmp_path, monkeypatch):
    import test_w3_dual_host_only_spill_q697c_1004 as Q

    _env(monkeypatch, dict(DUAL_P, **{"SGLANG_WEG2_DUAL_ARENA_TRIM_POST_D": "0"}))
    p, arena, root, t, nodes = Q._host_only_tree(tmp_path, slots=40, chain=False)
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), tree_cache=t)
    assert _tick(sched, 0.0) is not None
    assert not os.path.exists(str(arena.path) + D.NEED_SUFFIX)


def test_a_device_resident_node_is_not_a_host_only_leaf_and_stays(monkeypatch):
    """Not backed-and-evicted (it still has a device value): outside the V2 spill, as in Q-697c."""
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97)
    nodes[1].component_data[FULL].value = torch.arange(1, dtype=torch.int64)
    got = D.spill_host_only(cache, pool, 100, 1, set())
    assert nodes[1].id in _live(cache) and 7001 not in pool.secured
    assert got["leaves"] == 96


def test_a_stage_without_a_spill_pool_logs_a_rate_limited_stop(monkeypatch, caplog):
    import logging

    _env(monkeypatch, DUAL_P)
    caplog.set_level(logging.WARNING)
    sched, cache, pool, nodes = _stage(97, pp_rank=1)
    cache._weg2_direct_pool = lambda: None
    for i in range(20):
        got = D.execute(sched, D.Weg2DualArenaTrim(i + 1, 5, 970000))
        assert got["released"] == 0
    stops = [r for r in caplog.records if "STOP rank=1" in r.getMessage()]
    assert 1 <= len(stops) <= 8 and len(_live(cache)) == 97


def test_the_brake_stops_the_loop_before_the_next_leaf_and_the_next_order_goes_on(monkeypatch):
    _env(monkeypatch, DUAL_P)
    sched, cache, pool, nodes = _stage(97)
    clock = [0.0]

    def slow_secure(rows, orig=pool.secure_rows_to_l3):
        clock[0] += 0.3                                          # each L3 copy 'takes' 0.3 s of wall time
        return orig(rows)

    pool.secure_rows_to_l3 = slow_secure
    monkeypatch.setattr(D.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv(TRIM_BUDGET_S_ENV, "0.5")
    got = D.spill_host_only(cache, pool, 17, 1, set(), budget_s=0.5)
    assert got["braked"] == 1 and got["leaves"] == 2             # 0.3 -> 0.6 >= 0.5: stops before the third
    got = D.spill_host_only(cache, pool, 1, 1, set())            # the claim-driven callers: no brake, as before
    assert got["braked"] == 0 and got["leaves"] == 1
    cache.sanity_check()


# -------------------------------------------------------------------------------- the wire
def _pp_stage(rank, n_leaves=97):
    sched, cache, pool, nodes = _stage(n_leaves, pp_rank=rank)
    h = sched
    h.pp_group = types.SimpleNamespace(is_first_rank=rank == 0, is_last_rank=rank == 2)
    h.send_req_work = None
    h.flush_wrapper = types.SimpleNamespace(apply_pp0_verdict=lambda *a, **k: None)
    h._weg2_store_told_armed = False
    h._weg2_vote_pass_hook = lambda reqs: None
    h._weg2_vote_after_forward = lambda reqs: reqs
    h._pp_commit_comm_work = lambda work: None
    h.sent, h.dispatched = [], []
    h._pp_send_pyobj_to_next_stage = lambda reqs, async_send=True: h.sent.append(
        pickle.loads(pickle.dumps(list(reqs))))
    h.process_input_requests = lambda reqs: h.dispatched.append(list(reqs))
    h.waiting_queue = []
    return h, cache, pool, nodes


def _nc(lst):
    """The list without PP0's burst clock (another test of the process may leave that window armed)."""
    return [r for r in lst if type(r).__name__ != "Weg2BurstClock"]


def _intake(h, reqs):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    SchedulerPPMixin._pp_forward_and_process_input_requests(h, reqs)


def test_rank_congruence_one_order_on_the_wire_every_stage_executes_the_same(monkeypatch):
    _env(monkeypatch, DUAL_P)
    stages = [_pp_stage(r) for r in range(3)]
    h0, h1, h2 = (s[0] for s in stages)
    _intake(h0, ["req"])                                         # PP0 decides and serves its own order
    assert [type(r).__name__ for r in _nc(h0.sent[-1])] == ["str", "Weg2DualArenaTrim"]
    assert h0.dispatched[-1] == ["req"]                          # the dispatched list stays clean
    _intake(h1, h0.sent[-1])                                     # PP1: relay verbatim, dispatch without it
    assert h1.sent[-1] == h0.sent[-1] and _nc(h1.dispatched[-1]) == ["req"]
    _intake(h2, h1.sent[-1])                                     # PP2 (last: no send)
    assert h2.sent == [] and h2.dispatched[-1] == ["req"]
    cmd = [r for r in h0.sent[-1] if type(r).__name__ == "Weg2DualArenaTrim"][0]
    assert cmd.want == 17 and cmd.seq == 1
    lives = [_live_tokens(s[1]) for s in stages]
    assert lives[0] == lives[1] == lives[2]                      # the SAME tree on every stage
    assert len(lives[0]) == 80
    for s in stages:
        assert s[2].secured == [7000 + i for i in range(17)]     # same leaves, same order, L3 copy first
        s[1].sanity_check()
    # no follower read the header; PP0 read it once
    assert stages[1][2].arena.stats_calls == stages[1][2].arena.census_calls == 0
    assert stages[2][2].arena.stats_calls == stages[2][2].arena.census_calls == 0
    assert stages[0][2].arena.census_calls == 1


def test_off_the_gate_the_wire_is_untouched(monkeypatch):
    """Flip / NF / INT8: PP0 sends the dispatched list itself, followers find nothing to take."""
    for env in NOT_DUAL_P:
        _env(monkeypatch, env)
        _reset()
        stages = [_pp_stage(r) for r in range(3)]
        _intake(stages[0][0], ["req"])
        _intake(stages[1][0], stages[0][0].sent[-1])
        _intake(stages[2][0], stages[1][0].sent[-1])
        assert _nc(stages[0][0].sent[-1]) == ["req"] and _nc(stages[1][0].sent[-1]) == ["req"], env
        assert all(len(_live(s[1])) == 97 and s[2].arena.stats_calls == 0 for s in stages), env


def test_a_lagging_follower_with_a_node_less_executes_the_same_order_and_keeps_what_it_cannot_give(monkeypatch):
    """Replica trees may differ in what a rank CAN give. PP1 has one old leaf host-locked: it keeps that
    leaf (the slot stays pinned by that rank), everything else is the same set -- no inconsistent tree."""
    _env(monkeypatch, DUAL_P)
    stages = [_pp_stage(r) for r in range(2)]
    stages[1][3][0].component_data[FULL].host_lock_ref = 1
    _intake(stages[0][0], [])
    _intake(stages[1][0], stages[0][0].sent[-1])
    l0, l1 = _live_tokens(stages[0][1]), _live_tokens(stages[1][1])
    kept = (10,)                                                 # the host-locked oldest leaf of PP1
    assert kept in l1 and kept not in l0
    assert len(l0) == 80 and len(l1) == 80                       # PP1 gave 17 OTHER leaves: same count, one other set
    assert l1 - l0 == {kept}
    stages[1][3][0].component_data[FULL].host_lock_ref = 0
    for s in stages:
        s[1].sanity_check()


def test_wiring_order_in_the_pass(monkeypatch):
    import inspect

    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    src = inspect.getsource(SchedulerPPMixin._pp_forward_and_process_input_requests)
    stamp = src.index("_das_trim.pp0_stamp(")
    send = src.index("self._pp_send_pyobj_to_next_stage(")
    serve = src.index("_das_trim.execute(self, _arena_trim_cmd)")
    absorb = src.index("_das_trim.follower_absorb(")
    dispatch = src.index("self.process_input_requests(recv_reqs)")
    assert stamp < send < serve < absorb < dispatch


# -------------------------------------------------------------------------------- the real C arena
@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_real_arena_header_097_to_080_with_l3_files(tmp_path, monkeypatch):
    """The real ShmArena header (``stats`` + ``ref_census``), the real pool, the real ``secure_rows_to_l3``:
    fill 39/40 = 0.975 -> <= 0.80, the released pages are in the L3 store, the arena really frees them."""
    import test_w3_dual_host_only_spill_q697c_1004 as Q

    _env(monkeypatch, DUAL_P)
    p, arena, root, t, nodes = Q._host_only_tree(tmp_path, slots=40, chain=False)
    # drop the last node from tree + arena: 39 of 40 slots stay referenced
    last = nodes[-1]
    t.cache_controller.mem_pool_host.anchor_entry.host_pool.free(last.component_data[Q.FULL].host_value)
    last.parent.children.pop(next(k for k, c in last.parent.children.items() if c is last))
    pinned0 = arena.ref_census()[0]
    assert pinned0 == 39 and abs(pinned0 / 40.0 - 0.975) < 1e-9
    sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, pp_size=3), tree_cache=t)
    tt = 0.0
    for _ in range(4):
        _tick(sched, tt)
        tt += 2.0
    pinned = arena.ref_census()[0]
    assert pinned / 40.0 <= 0.80 + 1e-9, "pinned=%d of 40" % pinned
    gone = [n for n in nodes[:-1] if n.parent is None or n.component_data[Q.FULL].host_value is None]
    assert len(gone) == 39 - 32
    for n in gone:
        assert (root / (n.hash_value[0] + "_sfx.bin")).exists()   # L3 copy first, then the leaf went
    # the same beat for group D: the first order posted its need next to the shared arena (Q-1190 file)
    import struct

    need_file = str(arena.path) + D.NEED_SUFFIX
    assert os.path.exists(need_file)
    with open(need_file, "rb") as fh:
        assert struct.unpack("<q", fh.read(8))[0] == 7             # 39 pinned - int(0.8*40) = 7 pages
