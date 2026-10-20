"""NF-FOLLOWER-ROUND 1014 (inventory ``deskq/done/nf-row-inventory-1014.md``, C1 + C2).

C1. Under the row authority (31aa655fff) a follower of group P skips the plan on a
frameless cycle (PLAN BYPASS) -- and with the plan the whole HiCache housekeeping of
``_get_new_batch_prefill_raw``: write-through / load-back acks, prefetch REVOKES,
storage BACKUP acks, host RELEASES, extra-pool releases, deferred prefetch retries,
prefetch progress. e7cfda7fbe drained three of them (WT/LB/backup) plus the progress
in the #1268 vote only, i.e. only inside a quiesce. Revokes (A4) and releases (A5)
stayed queued on a follower until its next frame -- in the quiesce for good.

The ring below runs three stages over the REAL ``UnifiedRadixCache.check_hicache_events``
(``writing_check``, ``loading_check``, ``drain_storage_control_queues``,
``_drain_storage_control_queues_impl``, ``_count_ready_acks``, ``_attn_reduce_world``,
``_all_reduce_attn_groups``, ``_drain_async_work``): PP0 runs the planned round every
cycle, the followers run what a frameless cycle runs (``pdflip_follower_round`` where the
tree has it -- nothing on e7cfda7fbe) and a #1268 vote lap every few cycles. The census
guard: after N cycles every ack, revoke and release queue of every stage is empty.
Red on e7cfda7fbe for revokes and releases, green with the round.

C2. The #1400 told gate on a follower held a rid PP0's executed row names (B4): the
follower's batch came out one rid short of PP0's -> ``PPScheduleRefused`` / #1004. The
gate now yields to the row (as the LB-VERDICT gate does since 31aa, LBV_ROW_YIELD).
"""
from __future__ import annotations

import inspect
import itertools
import re
import types
from queue import Queue

import pytest
import torch

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

from flliper.srt.managers import scheduler_pp_mixin as mix  # noqa: E402
from flliper.srt.managers import pdflip_store_told as told_mod  # noqa: E402
from flliper.srt.managers.pdflip_idle_vote import PdFlipIdleVoteReq  # noqa: E402
from flliper.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import p_intake  # noqa: E402
from flliper.srt.pdflip import p_row_authority  # noqa: E402

ROUND_ENV = getattr(mix, "FOLLOWER_ROUND_ENV", "FLLIPER_PDFLIP_P_FOLLOWER_ROUND")
YIELD_ENV = getattr(p_intake, "TOLD_ROW_YIELD_ENV", "FLLIPER_PDFLIP_TOLD_ROW_YIELD")
REFUSED = getattr(mix, "FollowerRoundRefused", RuntimeError)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (ROUND_ENV, YIELD_ENV, "FLLIPER_PDFLIP_RELEASE_DRAIN_CAP", "FLLIPER_HICACHE_ROUND_TIMING",
              "FLLIPER_HICACHE_DRAIN_AGREE_EVERY", "FLLIPER_HICACHE_DRAIN_BUDGET_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)


# ---------------------------------------------------------------- the tree ----


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        assert self.done

    def elapsed_time(self, other):
        return 1.0


class _Node:
    def __init__(self, nid):
        self.id = nid
        self.l3_present = False


class _Op:
    def __init__(self, oid):
        self.id = oid
        self.completed_tokens = 64


class _Tree:
    """A #737 tree of one stage: the REAL per-round poll of ``UnifiedRadixCache``.
    Stubbed are only the leaves the queues end in (lock refs, the write-through ack's
    sidecar issue, host pool frees), recorded so a test can see what was retired."""

    check_hicache_events = UnifiedRadixCache.check_hicache_events
    writing_check = UnifiedRadixCache.writing_check
    loading_check = UnifiedRadixCache.loading_check
    _count_ready_acks = UnifiedRadixCache._count_ready_acks
    _drain_async_work = UnifiedRadixCache._drain_async_work
    _attn_reduce_world = UnifiedRadixCache._attn_reduce_world
    _all_reduce_attn_groups = UnifiedRadixCache._all_reduce_attn_groups
    _drain_agreement_is_collective = UnifiedRadixCache._drain_agreement_is_collective
    drain_storage_control_queues = UnifiedRadixCache.drain_storage_control_queues
    _drain_storage_control_queues_impl = UnifiedRadixCache._drain_storage_control_queues_impl
    drain_backup_acks_uncoordinated = getattr(
        UnifiedRadixCache, "drain_backup_acks_uncoordinated", None)

    def __init__(self, pp_rank):
        self.pp_rank = pp_rank
        self.pp_size = 3
        self.attn_cp_group = None
        self.attn_tp_group = None
        self.tp_world_size = 1
        self.work_list = []
        self._drain_depth_every = 0
        self._pin_trace_every = 0
        self.enable_storage = True
        self.enable_storage_metrics = False
        self.storage_metrics_collector = None
        self.staging_write_ring = None
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.freed = []
        self.extra_freed = []
        self.unlocked = []
        self.host_unlocked = []
        self._ops = itertools.count(1)
        self.collectives = []
        self.cache_controller = types.SimpleNamespace(
            ack_write_queue=[],
            ack_load_queue=[],
            ack_backup_queue=Queue(),
            prefetch_revoke_queue=Queue(),
            host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={PoolName.QSA_INDEXER: Queue()},
            mem_pool_host=types.SimpleNamespace(free=lambda t: self.freed.extend(t.tolist()),
                                                page_size=1),
            entry_for_extra_release=lambda pool: types.SimpleNamespace(
                host_pool=types.SimpleNamespace(free=lambda t: self.extra_freed.extend(t.tolist()))),
            append_host_mem_release=lambda **kw: None,
            prefetch_tokens_occupied=0,
        )

    # -- the leaves ---------------------------------------------------------
    def _finish_write_through_ack(self, ack_id):
        """``_pdflip_direct_complete`` + ``_pdflip_write_plain_sidecars``: one #106S sidecar
        store write per ack, acked by the store at once (20:48:52)."""
        self.ongoing_write_through.pop(ack_id)
        op = _Op(next(self._ops))
        self.ongoing_backup[op.id] = (_Node(ack_id), ("lock", ack_id))
        self.cache_controller.ack_backup_queue.put(op)

    def dec_lock_ref(self, node, params=None):
        self.unlocked.append(node.id)

    def dec_host_lock_ref(self, node, params=None):
        self.host_unlocked.append(getattr(node, "id", node))

    def _pdflip_rebind_host_to_arena(self, node):
        return False

    def _pdflip_host_is_transit(self):
        return False

    def _wait_bounded(self, work, label):
        raise AssertionError(f"a frameless round waited on a peer: {label}")

    # -- what a pass leaves behind -------------------------------------------
    def leave_pass_residue(self, base):
        """The state a forward pass leaves on this stage (h16l/h16j shapes): 3 direct
        write-throughs, 1 load-back, 1 revoked store read (A4), host rows given back
        (A5) and a sidecar pool release."""
        cc = self.cache_controller
        for i in range(3):
            wid = base + i
            self.ongoing_write_through[wid] = object()
            cc.ack_write_queue.append((_Event(), _Event(), [wid]))
        lid = base + 10
        self.ongoing_load_back[lid] = (_Node(lid), None, None)
        cc.ack_load_queue.append((_Event(), _Event(), [lid]))
        rid = f"pdflip-0-{base}"
        self.ongoing_prefetch[rid] = (_Node(f"pf{base}"), list(range(64)), None,
                                      types.SimpleNamespace(binding_generation=None,
                                                            probed_hit_tokens=0),
                                      None, {})
        cc.prefetch_revoke_queue.put(rid)
        cc.host_mem_release_queue.put(torch.arange(base * 100, base * 100 + 8))
        cc.extra_host_mem_release_queues[PoolName.QSA_INDEXER].put(
            torch.arange(base * 100 + 50, base * 100 + 52))
        return rid


def census(tc) -> dict:
    cc = tc.cache_controller
    return {
        "write_through": len(tc.ongoing_write_through),
        "load_back": len(tc.ongoing_load_back),
        "revoke": cc.prefetch_revoke_queue.qsize(),
        "backup": cc.ack_backup_queue.qsize() + len(tc.ongoing_backup),
        "release": cc.host_mem_release_queue.qsize(),
        "extra": sum(q.qsize() for q in cc.extra_host_mem_release_queues.values()),
        "prefetch": len(tc.ongoing_prefetch),
    }


def _blockers(tc):
    c = census(tc)
    return [f"{k}({v})" for k, v in c.items() if v]


class _Forbidden:
    """A scheduler decision the housekeeping round may never take."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        raise AssertionError(f"the follower round took a decision: {self.name}")


def _stage(pp_rank, storage=False):
    s = types.SimpleNamespace()
    s.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3, attn_tp_size=1, attn_cp_size=1)
    s.tree_cache = _Tree(pp_rank)
    s.enable_hierarchical_cache = True
    s.enable_hicache_storage = storage
    s.running_batch = None
    s.waiting_queue = []
    s.mbs = [None, None, None]
    s.chunked_req = None
    s.calls = []
    s._retry_deferred_prefetches = lambda: s.calls.append("retry")
    s._drain_prefetch_progress = lambda: s.calls.append("progress") or {}
    s._pp_microbatches_drained = lambda: True
    tc = s.tree_cache
    s.is_fully_idle = lambda: not _blockers(tc)
    s.idle_blockers = lambda: _blockers(tc)
    for name in ("process_pending_chunked_abort", "process_pending_pdflip_park",
                 "get_next_batch_to_run", "_get_new_batch_prefill_raw", "_pp_void_own_batch",
                 "_add_request_to_queue", "abort_request"):
        setattr(s, name, _Forbidden(name))
    return s


def _frameless_cycle(s):
    """What a follower's frameless cycle runs after the PLAN BYPASS: the round where it
    exists (getattr: on e7cfda7fbe nothing runs -- red by behaviour, not by import)."""
    rnd = getattr(mix, "pdflip_follower_round", None)
    if rnd is not None:
        rnd(s, site="frameless")


def _planned_round(s):
    """PP0 plans every cycle: ``_get_new_batch_prefill_raw`` runs the tree round."""
    s.tree_cache.check_hicache_events()


def _vote_lap(s, epoch):
    vote = PdFlipIdleVoteReq(epoch=epoch, origin=0, world=3, slots=[(0, 1, "none")])
    mix.SchedulerPPMixin._pdflip_vote_attach_own_slot(s, vote)
    return [sl for sl in vote.slots if sl[0] == s.ps.pp_rank][0]


def _ring(cycles=6, vote_every=2):
    stages = [_stage(r) for r in range(3)]
    for st in stages:
        st.tree_cache.leave_pass_residue(base=10 + st.ps.pp_rank)
    for k in range(cycles):
        _planned_round(stages[0])
        for st in stages[1:]:
            _frameless_cycle(st)
            if vote_every and k % vote_every == vote_every - 1:
                _vote_lap(st, k + 2)
    return stages


# -------------------------------------------------- C1: the census guard ----


def test_census_guard_every_queue_of_every_stage_is_empty_after_frameless_cycles():
    """THE GUARD (inventory D 4c): after N frameless cycles every ack, revoke and
    release queue of every stage is empty. e7cfda7fbe: PP1/PP2 keep the revoke (A4) and
    the host + sidecar releases (A5) -- the vote's single drains do not reach them."""
    stages = _ring()
    for st in stages:
        c = census(st.tree_cache)
        assert c == {k: 0 for k in c}, (st.ps.pp_rank, c)


def test_guard_a4_a5_named_separately():
    stages = _ring()
    for st in stages[1:]:
        c = census(st.tree_cache)
        assert c["revoke"] == 0 and c["prefetch"] == 0, ("A4 revoke left", st.ps.pp_rank, c)
        assert c["release"] == 0 and c["extra"] == 0, ("A5 release left", st.ps.pp_rank, c)


def test_without_any_vote_the_frameless_cycles_alone_drain_everything():
    """Outside the quiesce (no #1268 lap at all: idle between requests, a PF wait
    window) -- the case the vote drains never covered (A1, A3)."""
    stages = _ring(cycles=1, vote_every=0)
    for st in stages[1:]:
        c = census(st.tree_cache)
        assert c == {k: 0 for k in c}, (st.ps.pp_rank, c)


def test_the_revoke_is_recorded_as_the_probe_answer_and_its_locks_go():
    """A4: the drained revoke leaves the PrefetchOutcome(0) record (the PF ack then
    answers) and releases its host lock -- what PP0's round does with its own."""
    st = _ring(cycles=1, vote_every=0)[1]
    tc = st.tree_cache
    rid = "pdflip-0-11"
    assert rid not in tc.ongoing_prefetch
    assert int(tc.prefetch_loaded_tokens_by_reqid[rid]) == 0
    assert "pf11" in tc.host_unlocked
    assert sorted(tc.freed) == list(range(1100, 1108))
    assert sorted(tc.extra_freed) == [1150, 1151]


def test_follower_drains_equal_pp0s_planned_round():
    """Rank neutrality, measured: the follower's round leaves its tree in exactly the
    state PP0's planned round leaves PP0's -- same queues, same retirements."""
    stages = _ring(cycles=1, vote_every=0)
    ref = stages[0].tree_cache
    for st in stages[1:]:
        tc = st.tree_cache
        assert census(tc) == census(ref)
        assert len(tc.unlocked) == len(ref.unlocked) == 1
        assert len(tc.freed) == len(ref.freed) == 8
        assert len(tc.extra_freed) == len(ref.extra_freed) == 2


def test_round_decides_nothing_and_touches_no_membership():
    """raenge-nie-uneins: queue, slots, chunked continuation untouched; no admission,
    abort, park or void callable is reached (``_Forbidden`` raises)."""
    st = _stage(1, storage=True)
    sentinel_q = [types.SimpleNamespace(rid="pdflip-9-9")]
    st.waiting_queue = list(sentinel_q)
    st.mbs = ["batch-0", None, None]
    st.chunked_req = "chunk"
    st.tree_cache.leave_pass_residue(base=20)
    assert mix.pdflip_follower_round(st, site="frameless") is True
    assert st.waiting_queue == sentinel_q and st.mbs == ["batch-0", None, None]
    assert st.chunked_req == "chunk"
    assert st.calls == ["retry", "progress"]


def test_round_runs_retry_and_progress_only_with_storage():
    st = _stage(1, storage=False)
    assert mix.pdflip_follower_round(st, site="frameless") is True
    assert st.calls == []


def test_pp0_and_switch_off_run_no_round(monkeypatch):
    pp0 = _stage(0)
    pp0.tree_cache.leave_pass_residue(base=30)
    assert mix.pdflip_follower_round(pp0, site="frameless") is False
    assert census(pp0.tree_cache)["revoke"] == 1
    monkeypatch.setenv(ROUND_ENV, "0")
    st = _stage(1)
    st.tree_cache.leave_pass_residue(base=31)
    assert mix.pdflip_follower_round(st, site="frameless") is False
    assert census(st.tree_cache)["revoke"] == 1


def test_vote_site_needs_the_row_authority_frameless_site_does_not(monkeypatch):
    """The bypass itself proves a frameless cycle; the vote keeps the single drains'
    precondition (row authority)."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: False)
    st = _stage(1)
    st.tree_cache.leave_pass_residue(base=32)
    assert mix.pdflip_follower_round(st, site="vote") is False
    assert mix.pdflip_follower_round(st, site="frameless") is True
    assert census(st.tree_cache)["revoke"] == 0


def test_vote_runs_the_one_round_and_no_single_drain(monkeypatch):
    """A18: under the round the vote calls no single drain (no double run)."""
    called = []
    for name in ("pdflip_loadback_drain_on_idle_vote", "pdflip_writethrough_drain_on_idle_vote",
                 "pdflip_backup_ack_drain_on_idle_vote"):
        monkeypatch.setattr(mix, name, lambda s, _n=name: called.append(_n) or 0)
    st = _stage(1, storage=True)
    st.tree_cache.leave_pass_residue(base=33)
    slot = _vote_lap(st, 2)
    assert slot == (1, 1, "none"), slot
    assert called == []
    assert st.calls == ["retry", "progress"]


def test_vote_with_the_switch_off_is_e7cfda7fbe(monkeypatch):
    monkeypatch.setenv(ROUND_ENV, "0")
    st = _stage(1, storage=True)
    st.tree_cache.leave_pass_residue(base=34)
    slot = _vote_lap(st, 2)
    # the single drains: WT, LB, backup gone -- revoke and releases stay (A4/A5 open)
    c = census(st.tree_cache)
    assert c["write_through"] == c["load_back"] == c["backup"] == 0, c
    assert c["revoke"] == 1 and c["release"] == 1 and c["extra"] == 1, c
    assert slot[1] == 0
    assert st.calls == ["progress"]


# ------------------------------------------------ C1: the collective guard ----


class _GroupTree(_Tree):
    """A tree on a form with an attention group of 3: the drain agreement is a
    collective; every reduce is recorded instead of issued."""

    def _drain_agreement_is_collective(self):
        return True

    def _attn_reduce_world(self):
        return 3

    def _all_reduce_attn_groups(self, tensor, op, label="hicache"):
        self.collectives.append(label)


def _group_stage():
    st = _stage(1)
    st.tree_cache = _GroupTree(1)
    st.tree_cache.leave_pass_residue(base=40)
    return st


def test_attn_group_above_one_is_a_named_stop_and_issues_no_collective():
    st = _group_stage()
    with pytest.raises(REFUSED, match="FOLLOWER-ROUND REFUSED"):
        mix.pdflip_follower_round(st, site="frameless")
    assert st.tree_cache.collectives == []
    st2 = _stage(1)
    st2.ps.attn_tp_size = 3
    with pytest.raises(REFUSED, match="attn world 3"):
        mix.pdflip_follower_round(st2, site="frameless")


def test_pending_pp_sync_send_is_a_named_stop():
    st = _stage(1)
    st.tree_cache.work_list = [object()]
    with pytest.raises(REFUSED, match="PP-sync"):
        mix.pdflip_follower_round(st, site="frameless")


def _mutant(src_fn, old: str, new: str):
    """Compile a mutant of a module function in the module's own namespace."""
    src = inspect.getsource(src_fn)
    assert old in src, old
    ns = dict(vars(mix))
    exec(compile(src.replace(old, new, 1), "<mutant>", "exec"), ns)  # noqa: S102
    return ns[src_fn.__name__]


def test_mutant_round_without_the_guard_issues_a_rank_local_collective():
    """Mutant 'round without guard at attn group > 1': the round walks into the storage
    drain's MIN all_reduce alone -- the #580 class the guard names."""
    mutant = _mutant(mix.pdflip_follower_round, "if attn_world > 1 or collective or reduce_world > 1:",
                     "if False:")
    st = _group_stage()
    mutant(st, site="frameless")
    assert st.tree_cache.collectives, "the mutant must reach a collective"


def test_mutant_round_on_pp0_would_run_twice_per_pass():
    mutant = _mutant(mix.pdflip_follower_round, "    if int(getattr(ps, \"pp_rank\", 0) or 0) == 0:\n        return False\n", "")
    pp0 = _stage(0)
    pp0.tree_cache.leave_pass_residue(base=41)
    assert mutant(pp0, site="frameless") is True  # the guard caught by the real one above


def test_mutant_round_without_storage_drain_leaves_a4_a5():
    """Mutant 'the tree round skipped' (e.g. only on frames): revoke and releases stay."""
    mutant = _mutant(mix.pdflip_follower_round, "            tree_round()\n", "            pass\n")
    st = _stage(1)
    st.tree_cache.leave_pass_residue(base=42)
    mutant(st, site="frameless")
    c = census(st.tree_cache)
    assert c["revoke"] == 1 and c["release"] == 1


# ------------------------------------------------ C1: the call sites ----


def _frameless_block(src: str) -> str:
    i = src.index("if _row_skip_plan:")
    j = src.index("elif _pre_proxy is not None:", i)
    return src[i:j]


def test_the_frameless_branch_runs_the_round():
    """Mutant 'round only on a frame': the call must sit in the ``_row_skip_plan``
    branch of the loop body, not in a planned branch."""
    src = inspect.getsource(mix.SchedulerPPMixin._event_loop_pp_body)
    assert 'pdflip_follower_round(self, site="frameless")' in _frameless_block(src)
    mutated = src.replace('pdflip_follower_round(self, site="frameless")', "pass", 1)
    assert 'pdflip_follower_round(self, site="frameless")' not in _frameless_block(mutated)


def test_vote_order_791c_then_round_then_slot():
    src = inspect.getsource(mix.SchedulerPPMixin._pdflip_vote_attach_own_slot)
    i791 = src.index("pdflip_791c_release_on_idle_vote(self, vote)")
    iround = src.index('pdflip_follower_round(self, site="vote")')
    islot = src.index("attach_slot(vote, rank, self.is_fully_idle()")
    assert i791 < iround < islot
    # the single drains survive only in the not-ran branch
    k = src.index("if not _round_ran:")
    for name in ("pdflip_loadback_drain_on_idle_vote(self)", "pdflip_writethrough_drain_on_idle_vote(self)",
                 "pdflip_backup_ack_drain_on_idle_vote(self)", "self._drain_prefetch_progress()"):
        assert k < src.index(name) < islot, name


def test_round_does_not_plant_the_prefetch_memo():
    src = inspect.getsource(mix.pdflip_follower_round)
    assert "_pass_prefetch_verdicts" not in src.replace("`_pass_prefetch_verdicts` stays", "")


# ---------------------------------------------------------------- C2 ----


ROW_RID = "pdflip-0-10"


def _c2_stage(pp_rank=1, row=("pdflip-0-9", "pdflip-1-11", ROW_RID)):
    s = types.SimpleNamespace()
    s.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3)
    s._pdflip_store_told = {}
    s._pp_admission_incoming_effective = (
        None if row is None else {r: 4096 for r in row})
    return s


def _req(rid):
    return types.SimpleNamespace(rid=rid)


def _held(stage, rid):
    """W27-UNIFORM: the follower's tree is short of PP0's told for ``rid`` (item 180)."""
    told_mod._mark_unreached(stage, rid, 20096, 3008, None)


def _gate(stage, req, notes):
    """The scheduler's told gate (SCH, the #1400 branch) over the real pieces: the
    told verdict of the rid outstanding, the real W27 barrier."""
    note = lambda k, r: notes.append((k, r))  # noqa: E731
    pending = lambda s, r, n: (n(p_intake.SKIP_TOLD_PENDING, r.rid), None)[1]  # noqa: E731
    loaded = p_intake.told_admission(stage, req, note, pending)
    if loaded is not None:
        return "admit", loaded
    pg = getattr(p_intake, "told_pending_gate", None)
    if pg is not None:
        return pg(stage, req, note, told_mod.held_barrier), None
    # e7cfda7fbe's scheduler seam, spelled out (red by behaviour on the base)
    if told_mod.held_barrier(stage, req):
        note("w27_held_barrier", req.rid)
        return "break", None
    return "skip", None


def _follower_batch(stage, queue):
    """The follower's admission loop: told gate, then the row arm (``pp_not_named``)."""
    out, notes = [], []
    for req in queue:
        act, _credit = _gate(stage, req, notes)
        if act == "break":
            break
        if act == "skip":
            continue
        if req.rid not in stage._pp_admission_incoming_effective:
            continue
        out.append(req.rid)
    return out


def test_c2_a_held_row_named_rid_is_seated_like_pp0():
    """B4: PP0's row names [pdflip-0-9, pdflip-1-11, pdflip-0-10]; PP1 holds pdflip-0-10 (tree
    short of told) and has no told yet for the others (they come from the row). Without
    the yield the follower seats nothing PP0 seated -> PPScheduleRefused; with it the
    membership is PP0's."""
    stage = _c2_stage()
    _held(stage, ROW_RID)
    queue = [_req(ROW_RID), _req("pdflip-0-9"), _req("pdflip-1-11")]
    batch = _follower_batch(stage, queue)
    assert batch == [ROW_RID, "pdflip-0-9", "pdflip-1-11"], batch


def test_c2_switch_off_is_the_base_gate(monkeypatch):
    monkeypatch.setenv(YIELD_ENV, "0")
    stage = _c2_stage()
    _held(stage, ROW_RID)
    queue = [_req(ROW_RID), _req("pdflip-0-9"), _req("pdflip-1-11")]
    assert _follower_batch(stage, queue) == []  # the barrier: the base split


def test_c2_a_held_rid_the_row_does_not_name_is_no_barrier():
    stage = _c2_stage(row=("pdflip-0-9",))
    _held(stage, ROW_RID)
    queue = [_req(ROW_RID), _req("pdflip-0-9")]
    assert _follower_batch(stage, queue) == ["pdflip-0-9"]


def test_c2_without_a_row_the_barrier_stands():
    """The carrierless form (row authority off, no incoming row): W27-HELD-STOP as is."""
    stage = _c2_stage(row=None)
    stage._pp_admission_incoming_effective = None
    _held(stage, ROW_RID)
    notes = []
    assert _gate(stage, _req(ROW_RID), notes)[0] == "break"
    assert ("w27_held_barrier", ROW_RID) in notes


def test_c2_pp0_never_yields():
    stage = _c2_stage(pp_rank=0)
    assert p_intake.told_row_verdict(stage, _req(ROW_RID)) is None


def test_c2_gate_answers_are_the_scheduler_spellings():
    assert (p_intake.GATE_ADMIT, p_intake.GATE_SKIP, p_intake.GATE_BREAK) == ("admit", "skip", "break")
    stage = _c2_stage()
    assert p_intake.told_pending_gate(stage, _req(ROW_RID), lambda *a: None,
                                      told_mod.held_barrier) == p_intake.GATE_ADMIT


def test_c2_scheduler_seam_uses_the_gate():
    from flliper.srt.managers import scheduler as S

    src = inspect.getsource(S)
    i = src.index("_told_loaded = _p_intake.told_admission(")
    window = src[i:i + 900]
    assert "_p_intake.told_pending_gate(" in window
    assert "pdflip_store_told.held_barrier" in window
    assert re.search(r"if _told_gate == _p_intake\.GATE_BREAK:\s+break", window), window
    assert re.search(r"if _told_gate == _p_intake\.GATE_SKIP:\s+continue", window), window
    gsrc = inspect.getsource(p_intake.told_pending_gate)
    assert gsrc.index("told_row_verdict") < gsrc.index("held_barrier(scheduler, req)")


def test_c2_mutant_gate_ignoring_the_row_splits_membership(monkeypatch):
    """Mutant 'the gate consults the barrier before the row': the B4 split returns."""
    monkeypatch.setattr(p_intake, "told_row_verdict", lambda s, r: None)
    stage = _c2_stage()
    _held(stage, ROW_RID)
    queue = [_req(ROW_RID), _req("pdflip-0-9"), _req("pdflip-1-11")]
    assert _follower_batch(stage, queue) != list(stage._pp_admission_incoming_effective)
