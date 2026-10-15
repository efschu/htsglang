# SPDX-License-Identifier: Apache-2.0
"""NF-COLDSTART-RIG 1015: the first minutes of an NF P group, on CPU, against ANY tree.

WHY. h16j died twice at #968, h16l twice at W3, h16h after 4 h at ArenaChainMismatch -- classes no single unit
test caught, each in the first minutes of a live image. From now on an image goes live only if its NF revision
passes this rig, and the rig is only worth anything if it reproduces the known deaths RED on the trees that died.

WHAT. One P group of three stages (PP0 with the row authority, two followers) sharing one HiCache housekeeping
surface each, one shared C arena with a file store behind it, and a D side as the second arena user (evictor,
parked host writer). After every cycle the LAW GUARDS run (``Guards``):

  G1 every stage admits the same rids at the same depth in every pass (RAENGE-NIE-UNEINS);
  G2 every ack, revoke and release queue of every stage is empty after N frameless cycles;
  G3 the idle vote turns true once the work is done (no W3);
  G4 no referenced arena page loses its key; a read lands what it was told to land.

A guard that fires is a group stop by name -- the rig never repairs, never retries, never compensates (hard user
law raenge-nie-uneins-crash-stop): a detected disagreement is RED, full stop.

SCENARIOS (one test each, ``test_S<n>_...``; the runner ``tools/weg2_coldstart_rig.sh`` prints one line each):

  S1  first P->D flip with flush after prefills with direct write-through (sidecar backups)   red on 75e3ee8abd (W3)
  S2  two same-session requests, L3 load-back under the row authority, matched 0 vs 3008      red on 73f6a4e110 (#968)
  S3  revokes / host releases on frameless followers                                           red on e7cfda7fbe (A4/A5)
  S4a arena reap race: a live evictor's candidates reaped, PP0 fills + references them         red without F1 (01:09 death)
  S4b a D clock evicts PP0's landed pages before PP0 references them                           red without REF-AT-LAND (case 8)
  S4c D parks with forced host write-through, P reads the same stems from L3 seconds later     h54 (16/16 JOIN-WAIT tears)
  S5  a vision request over P (row form): every stage admits it in the same pass               self-check by mutant
  S6  LB-VERDICT burst with a late-starting twin on the carrierless form (19:33 hang)          red on 853534a9ff

WHAT IS REAL. The tree's own code wherever the death lived: the frameless branch of ``_event_loop_pp_body`` is cut
out of THIS tree's source and executed (so whatever the tree runs there runs here); every ``_weg2_vote_*`` hook, the
real ``Scheduler.flush_cache`` / ``group_idle_verdict`` / ``SchedulerFlushWrapper`` /flush_cache path; the real
``UnifiedRadixCache`` housekeeping (``check_hicache_events``, ``writing_check``, ``loading_check``,
``_drain_storage_control_queues_impl``); the real ``load_back`` / ``init_load_back`` / ``execute_scheduled_prefix`` /
``pp_room_vote``; the real C arena (gcc) with the real ``HiCacheFile._arena_evict_to_disk`` and
``arena_fill_from_disk`` (and a second PROCESS for D's host-write claims); the real ``vision_verdict`` /
``vision_rank_runner``; the real ``pp_loadback_verdict``. Every symbol a newer tree added is looked up by name, so on
an older tree the scenario runs the older behaviour and goes red by BEHAVIOUR, never by an import error. A rig
defect (a source anchor that is gone) is a ``RIG-FEHLER``, never green and never red.

WHAT IS MODELLED (the limits, see the report): no forward, no NCCL/gloo timing (the chain is one list per pass,
the guaranteed order chain-before-frame), the radix trees of S2 are stand-ins carrying the #1423 insert shape,
arena pressure is synthetic (8..64 slots), disk IO of the evictor is a stand-in (the arena state machine is real),
the scheduler admission loop of S5/S6 is reduced to its decisions.
"""
from __future__ import annotations

import inspect
import itertools
import multiprocessing as mp
import os
import shutil
import textwrap
import threading
import time
import types
from queue import Queue
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

RIG = "RIG-FEHLER"


class RigError(AssertionError):
    """The rig cannot drive this tree (a source anchor is gone) -- neither red nor green."""

    def __init__(self, msg):
        super().__init__(f"{RIG}: {msg}")


class LawViolation(AssertionError):
    """A law guard fired: the group stops by name (RED)."""

    def __init__(self, law, msg):
        super().__init__(f"GESETZ {law}: {msg}")


def _group_stop(exc: BaseException):
    """A production stop the tree raised itself (#968, ROW-ROOM SHORT, PPScheduleRefused, ArenaChainMismatch ...):
    the group died by name in the rig -- RED, reported with the tree's own text."""
    first = str(exc).splitlines()[0] if str(exc) else ""
    return LawViolation("GRUPPENSTOPP", f"{type(exc).__name__}: {first[:600]}")


#: switches a scenario or a self-check sets; cleared before every test so the tree's defaults decide
RIG_ENVS = ("SGLANG_WEG2_P_FOLLOWER_ROUND", "SGLANG_WEG2_TOLD_ROW_YIELD", "SGLANG_WEG2_P_ROW_AUTHORITY",
            "SGLANG_WEG2_LBV_ROW_ROOM", "SGLANG_WEG2_LBV_ROW_YIELD", "SGLANG_WEG2_LBV_ADMIT",
            "SGLANG_WEG2_PP_LOADBACK_VERDICT", "SGLANG_WEG2_L3FILL_REF_AT_LAND", "SGLANG_WEG2_L3FILL_JOIN_LIVENESS",
            "SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in RIG_ENVS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def nf_form(monkeypatch):
    """The launcher's contract to the ranks: the published NF form (profile nextflash) in ``SGLANG_WEG2_FORM`` --
    every profile default of the tree resolves from it exactly as on the NF image."""
    from sglang.srt.weg2 import form as FM

    try:
        value = FM.Weg2Form(arch="moe", experts="offload", draft="mtp", p_draft="none", kv="qsa_forma",
                            flip="family", vision="off", profile="nextflash", model="nf").env_value()
    except Exception as exc:  # noqa: BLE001
        raise RigError(f"cannot publish the NF form on this tree: {exc!r}")
    monkeypatch.setenv(FM.FORM_ENV, value)


# =====================================================================================================================
# the law guards
# =====================================================================================================================


class Guards:
    @staticmethod
    def same_admission(pass_no, batches, what="admission"):
        """G1: every stage admits the same rids at the same depth in this pass."""
        ref = batches[0]
        for rank, b in enumerate(batches[1:], start=1):
            if list(b) != list(ref):
                raise LawViolation(
                    "G1 RAENGE-NIE-UNEINS",
                    f"pass {pass_no}: {what} PP0={list(ref)} PP{rank}={list(b)} (all stages: {batches})")

    @staticmethod
    def queues_empty(stages, census, after_cycles):
        """G2: every ack, revoke and release queue of every stage is empty after N frameless cycles."""
        left = {}
        for s in stages:
            c = census(s.tree_cache)
            nz = {k: v for k, v in c.items() if v}
            if nz:
                left[f"PP{s.ps.pp_rank}"] = nz
        if left:
            raise LawViolation("G2 QUEUE-CENSUS", f"after {after_cycles} frameless cycles still queued: {left}")

    @staticmethod
    def idle_within(ok, polls, blockers):
        """G3: the idle vote turns true once the work is done -- else the front's W3."""
        if not ok:
            raise LawViolation(
                "G3 IDLE-VOTE (W3 Weg2DrainWitnessDisagreement)",
                f"/flush_cache never 200 in {polls} polls after the work ended; blockers {blockers}")

    @staticmethod
    def keys_kept(arena, refs):
        """G4a: a page a stage references keeps its key (find_slots(stem) == (slot, COMPLETE))."""
        lost = []
        for stem, slot in refs:
            got = arena.find_slots([stem])[0]
            if tuple(got) != (int(slot), 2):
                lost.append((stem, int(slot), tuple(got)))
        if lost:
            raise LawViolation("G4 REFERENCED-PAGE-KEY",
                               f"{len(lost)} referenced page(s) lost their key (stem, slot, find=(slot,state)): {lost[:6]}"
                               " -- the load-back's chain proof fails: ArenaChainMismatch #1424 'key slot -1 (state 0)'")

    @staticmethod
    def landed_full(landed, want, what):
        """G4b: a read lands the full store depth it was asked for."""
        if landed != want:
            raise LawViolation("G4 LANDED-DEPTH", f"{what}: landed {landed} of {want} pages")


# =====================================================================================================================
# housekeeping surface of one stage (S1, S3): the REAL UnifiedRadixCache per-round poll on a #737 tree
# =====================================================================================================================


def _urc():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        assert self.done

    def elapsed_time(self, other):
        return 1.0


class _HkNode:
    def __init__(self, nid):
        self.id = nid
        self.l3_present = False


class _Op:
    def __init__(self, oid):
        self.id = oid
        self.completed_tokens = 64


def _hk_tree_cls():
    U = _urc()
    names = ("check_hicache_events", "writing_check", "loading_check", "_count_ready_acks", "_drain_async_work",
             "_attn_reduce_world", "_all_reduce_attn_groups", "_drain_agreement_is_collective",
             "drain_storage_control_queues", "_drain_storage_control_queues_impl",
             "drain_backup_acks_uncoordinated", "drain_prefetch_revokes_uncoordinated")
    attrs = {n: getattr(U, n) for n in names if getattr(U, n, None) is not None}
    for must in ("check_hicache_events", "writing_check", "_drain_storage_control_queues_impl"):
        if must not in attrs:
            raise RigError(f"UnifiedRadixCache.{must} is gone -- the housekeeping surface moved")
    return type("_HkTree", (_HkTreeBase,), attrs)


class _HkTreeBase:
    """Stubbed are only the leaves the queues end in (lock refs, the write-through ack's #106S sidecar issue, host
    pool frees) -- recorded, so a guard can see what was retired."""

    def __init__(self, pp_rank):
        from sglang.srt.mem_cache.hicache_storage import PoolName

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
        self.freed, self.extra_freed, self.unlocked, self.host_unlocked = [], [], [], []
        self.resets = 0
        self._ops = itertools.count(1)
        self.cache_controller = SimpleNamespace(
            ack_write_queue=[], ack_load_queue=[], ack_backup_queue=Queue(), prefetch_revoke_queue=Queue(),
            host_mem_release_queue=Queue(), extra_host_mem_release_queues={PoolName.QSA_INDEXER: Queue()},
            mem_pool_host=SimpleNamespace(free=lambda t: self.freed.extend(t.tolist()), page_size=1),
            entry_for_extra_release=lambda pool: SimpleNamespace(
                host_pool=SimpleNamespace(free=lambda t: self.extra_freed.extend(t.tolist()))),
            append_host_mem_release=lambda **kw: None, prefetch_tokens_occupied=0,
        )

    def _finish_write_through_ack(self, ack_id):
        """``_weg2_direct_complete`` + ``_weg2_write_plain_sidecars``: one #106S sidecar store write per ack, acked by
        the store at once (h16l 20:48:52)."""
        self.ongoing_write_through.pop(ack_id)
        op = _Op(next(self._ops))
        self.ongoing_backup[op.id] = (_HkNode(ack_id), ("lock", ack_id))
        self.cache_controller.ack_backup_queue.put(op)

    def dec_lock_ref(self, node, params=None):
        self.unlocked.append(node.id)

    def dec_host_lock_ref(self, node, params=None):
        self.host_unlocked.append(getattr(node, "id", node))

    def _weg2_rebind_host_to_arena(self, node):
        return False

    def _weg2_host_is_transit(self):
        return False

    def _wait_bounded(self, work, label):
        raise LawViolation("G2 QUEUE-CENSUS", f"a housekeeping round waited on a peer: {label}")

    def reset(self):
        self.resets += 1

    # -- what a pass leaves behind -------------------------------------------------------------------------------
    def leave_write_through(self, base, n=3):
        """h16l: the last pass of weg2-0-1 leaves n direct write-throughs open on every rank."""
        cc = self.cache_controller
        for i in range(n):
            wid = base + i
            self.ongoing_write_through[wid] = object()
            cc.ack_write_queue.append((_Event(), _Event(), [wid]))

    def leave_load_back(self, base):
        lid = base + 10
        self.ongoing_load_back[lid] = (_HkNode(lid), None, None)
        self.cache_controller.ack_load_queue.append((_Event(), _Event(), [lid]))

    def leave_revoke_and_releases(self, base):
        """A4: a store read revoked under the prefetch threshold; A5: host rows and a QSA sidecar row given back."""
        from sglang.srt.mem_cache.hicache_storage import PoolName

        cc = self.cache_controller
        rid = f"weg2-0-{base}"
        self.ongoing_prefetch[rid] = (_HkNode(f"pf{base}"), list(range(64)), None,
                                      SimpleNamespace(binding_generation=None, probed_hit_tokens=0), None, {})
        cc.prefetch_revoke_queue.put(rid)
        cc.host_mem_release_queue.put(torch.arange(base * 100, base * 100 + 8))
        cc.extra_host_mem_release_queues[PoolName.QSA_INDEXER].put(torch.arange(base * 100 + 50, base * 100 + 52))
        return rid


def hk_census(tc) -> dict:
    cc = tc.cache_controller
    return {
        "write_through": len(tc.ongoing_write_through) + len(cc.ack_write_queue),
        "load_back": len(tc.ongoing_load_back),
        "revoke": cc.prefetch_revoke_queue.qsize(),
        "backup": cc.ack_backup_queue.qsize() + len(tc.ongoing_backup),
        "release": cc.host_mem_release_queue.qsize(),
        "extra": sum(q.qsize() for q in cc.extra_host_mem_release_queues.values()),
        "prefetch": len(tc.ongoing_prefetch),
    }


def hk_blockers(tc):
    """The idle witness as ``is_fully_idle``/``idle_blockers`` name it (``hicache_<queue>(n)``)."""
    out = []
    if tc.ongoing_write_through:
        out.append(f"hicache_write_through({len(tc.ongoing_write_through)})")
    if tc.ongoing_load_back:
        out.append(f"hicache_load_back({len(tc.ongoing_load_back)})")
    if tc.ongoing_backup:
        out.append(f"hicache_backup({len(tc.ongoing_backup)})")
    if tc.ongoing_prefetch:
        out.append(f"hicache_prefetch({len(tc.ongoing_prefetch)})")
    return out


class _Forbidden:
    """A scheduler decision a frameless cycle may never take."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        raise LawViolation("G1 RAENGE-NIE-UNEINS", f"a frameless follower cycle took a decision: {self.name}")


# ---------------------------------------------------------------------------------------------------- the loop body --

_FRAMELESS_CODE = {}


def frameless_code():
    """The ``if _row_skip_plan:`` branch of THIS tree's ``_event_loop_pp_body``, compiled: what a frameless follower
    cycle runs, whatever that is on the tree under test."""
    if "code" in _FRAMELESS_CODE:
        return _FRAMELESS_CODE["code"]
    from sglang.srt.managers import scheduler_pp_mixin as mix

    src = inspect.getsource(mix.SchedulerPPMixin._event_loop_pp_body)
    lines = src.splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if ln.strip() == "if _row_skip_plan:")
    except StopIteration:
        raise RigError("_event_loop_pp_body has no 'if _row_skip_plan:' branch")
    ind = len(lines[i]) - len(lines[i].lstrip())
    body = []
    for ln in lines[i + 1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= ind:
            if not ln.strip().startswith("elif _pre_proxy is not None"):
                raise RigError(f"the frameless branch ends at an unexpected line: {ln.strip()[:80]}")
            break
        body.append(ln)
    else:
        raise RigError("the frameless branch has no end")
    code = compile(textwrap.dedent("\n".join(body)), "<frameless-branch>", "exec")
    _FRAMELESS_CODE["code"] = code
    _FRAMELESS_CODE["text"] = textwrap.dedent("\n".join(l for l in body if l.strip() and not l.strip().startswith("#")))
    return code


def run_frameless_cycle(stage, mb_id):
    from sglang.srt.managers import scheduler_pp_mixin as mix

    exec(frameless_code(), dict(vars(mix)), {"self": stage, "mb_id": mb_id})  # noqa: S102


def run_planned_round(stage):
    """PP0 plans every cycle: ``_get_new_batch_prefill_raw`` runs ``check_hicache_events``."""
    stage.tree_cache.check_hicache_events()


# ------------------------------------------------------------------------------------------------- the P group -----

WORLD = 3


class FakeClock:
    def __init__(self, t=5000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class _HomeFrame:
    def __init__(self, wire):
        self.wire = wire

    def advance(self, _budget_s):
        return bool(self.wire.home)

    def take(self):
        return self.wire.home.pop(0)


def build_p_group(storage=False):
    """Group P, pp_size=3, tp_size=1 (the NF form), every vote hook bound REAL, PP0's /flush_cache path REAL."""
    from sglang.srt.managers import scheduler_pp_mixin as mix
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
    from sglang.srt.managers.weg2_idle_vote import WEG2_VOTE_TAG, home_ranks

    Tree = _hk_tree_cls()
    wire = SimpleNamespace(inbox={r: [] for r in range(WORLD)}, home=[], pp0_incoming=[])
    ranks = []
    for r in range(WORLD):
        tc = Tree(r)
        s = SimpleNamespace(
            ps=SimpleNamespace(pp_rank=r, pp_size=WORLD, tp_size=1, attn_tp_rank=0, attn_cp_rank=0,
                               attn_tp_size=1, attn_dp_rank=0, attn_cp_size=1),
            pp_group=SimpleNamespace(is_first_rank=(r == 0), is_last_rank=(r == WORLD - 1)),
            world_group=SimpleNamespace(cpu_group=None),
            tree_cache=tc, enable_hierarchical_cache=True, enable_hicache_storage=storage, weg2_dormant=False,
            mbs=[None] * WORLD, running_batch=None, last_batch=None, waiting_queue=[], chunked_req=None,
            calls=[],
        )
        s.is_fully_idle = lambda tc=tc: not hk_blockers(tc)
        s.idle_blockers = lambda tc=tc: hk_blockers(tc)
        s._pp_microbatches_drained = lambda: True
        s._pp_commit_comm_work = lambda _w: None
        s._drain_prefetch_progress = lambda s=s: s.calls.append("progress") or {}
        s._retry_deferred_prefetches = lambda s=s: s.calls.append("retry")
        s.process_pending_chunked_abort = lambda s=s: s.calls.append("chunked_abort")
        s._pp_slot_holds_unconsumed_launch = lambda mb_id, where: False
        for name in ("get_next_batch_to_run", "_get_new_batch_prefill_raw", "_pp_void_own_batch",
                     "_add_request_to_queue", "abort_request", "process_pending_weg2_park"):
            setattr(s, name, _Forbidden(name))
        for name in dir(Scheduler):
            if name.startswith("_weg2_vote_") and callable(getattr(Scheduler, name)):
                setattr(s, name, getattr(Scheduler, name).__get__(s))
        s._pp_null_frameless_slot = mix.SchedulerPPMixin._pp_null_frameless_slot.__get__(s)
        ranks.append(s)
    pp0 = ranks[0]
    src_global, _dst = home_ranks(0, WORLD, 1, 0)
    pp0._pp_object_recv_frames = {(src_global, WEG2_VOTE_TAG): _HomeFrame(wire)}
    pp0.running_batch = SimpleNamespace(is_empty=lambda: True, reqs=[])
    pp0.cur_batch_for_debug = None
    pp0.draft_worker = None
    pp0.req_to_token_pool = SimpleNamespace(clear=lambda: None)
    pp0.token_to_kv_pool_allocator = SimpleNamespace(clear=lambda: None)
    pp0.grammar_manager = SimpleNamespace(clear=lambda: None)
    pp0.metrics_reporter = SimpleNamespace(reset_metrics=lambda: None, is_stats_logging_rank=True)
    pp0._flush_zero_kv_wanted = lambda _z: False
    for name in ("group_idle_verdict", "_weg2_join_store_writes_before_reset", "_weg2_note_lost_anchors",
                 "flush_cache"):
        fn = getattr(Scheduler, name, None)
        if fn is not None:
            setattr(pp0, name, fn.__get__(pp0))
    if not hasattr(pp0, "flush_cache"):
        raise RigError("Scheduler.flush_cache is gone")
    wrapper = SchedulerFlushWrapper(flush_cache=lambda **kw: pp0.flush_cache(empty_cache=False, **kw),
                                    is_fully_idle=pp0.is_fully_idle, ipc_channels=None)
    return SimpleNamespace(ranks=ranks, wire=wire, wrapper=wrapper)


def _send_home(wire):
    def send(vote, _group, _src, _dst):
        wire.home.append(vote)
        return []

    return send


def group_cycle(g, mb_id):
    """One loop iteration of every rank in ``_event_loop_pp_body`` order: the chain hop with the vote hooks
    (``_pp_forward_and_process_input_requests``), then the plan -- PP0 plans (the round runs in
    ``_get_new_batch_prefill_raw``), a follower without a frame takes the tree's frameless branch."""
    from sglang.srt.managers import scheduler_pp_mixin as mix

    for r, s in enumerate(g.ranks):
        if r == 0:
            recv = list(g.wire.pp0_incoming)
            g.wire.pp0_incoming = []
        else:
            recv = g.wire.inbox[r].pop(0) if g.wire.inbox[r] else []
        s._weg2_vote_pass_hook(recv)
        if r < WORLD - 1:
            g.wire.inbox[r + 1].append(list(recv))
        with mock.patch.object(mix, "send_home", _send_home(g.wire)):
            s._weg2_vote_after_forward(recv)
        if r == 0:
            run_planned_round(s)
        else:
            run_frameless_cycle(s, mb_id)


def flush_poll(g):
    from sglang.srt.managers.io_struct import FlushCacheReqInput

    return g.wrapper.handle(FlushCacheReqInput()).success


# =====================================================================================================================
# S1 -- first P->D flip with flush after prefills with write-through (h16l, W3)
# =====================================================================================================================


@pytest.fixture
def row_authority(monkeypatch):
    from sglang.srt.weg2 import p_row_authority

    for k in ("SGLANG_WEG2_P_FOLLOWER_ROUND", "SGLANG_WEG2_TOLD_ROW_YIELD", "SGLANG_WEG2_P_ROW_AUTHORITY"):
        monkeypatch.delenv(k, raising=False)
    if not p_row_authority.applies(SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=1))):
        pytest.skip("n/a: the NF form of this tree runs P without the row authority (no frameless follower cycle)")
    clock = FakeClock()
    with mock.patch("time.monotonic", new=clock):
        yield clock


def s1_flip_with_flush(clock, polls=30, wt=3):
    """The h16l boot: weg2-0-1's last pass leaves ``wt`` direct write-throughs open on EVERY rank (their acks issue
    #106S sidecar backups), then the P->D quiesce: the front polls /flush_cache, the #1268 laps run between polls,
    followers have no frame. Returns (ok, polls_used, last_blockers, group)."""
    g = build_p_group()
    for s in g.ranks:
        s.tree_cache.leave_write_through(base=10 * (s.ps.pp_rank + 1), n=wt)
    mb = 0
    blockers = {}
    for k in range(polls):
        if flush_poll(g):
            return True, k + 1, {}, g
        for _ in range(3):
            group_cycle(g, mb)
            mb = (mb + 1) % WORLD
            clock.advance(0.002)
        clock.advance(0.05)
        blockers = {f"PP{s.ps.pp_rank}": hk_blockers(s.tree_cache) for s in g.ranks}
    return False, polls, blockers, g


def test_S1_first_flip_with_flush_after_write_through(row_authority):
    """W3 (h16l 75e3ee8abd): PP1/PP2 vote ``hicache_backup(3)`` forever, /flush_cache 400, the front stops W3."""
    ok, n, blockers, g = s1_flip_with_flush(row_authority)
    Guards.idle_within(ok, n, blockers)
    Guards.queues_empty(g.ranks, hk_census, n * 3)
    assert g.ranks[0].tree_cache.resets == 1, "PP0's flush did not reset its tree exactly once"


# =====================================================================================================================
# S3 -- revokes and releases on frameless followers (A4/A5)
# =====================================================================================================================


def test_S3_revokes_releases_on_frameless_followers(row_authority):
    """After a pass every stage holds write-throughs, a load-back, a revoked store read (A4) and host + sidecar
    releases (A5); PP0 plans, the followers have no frame and no vote (idle between requests). G2 after N cycles."""
    g = build_p_group()
    for s in g.ranks:
        b = 10 * (s.ps.pp_rank + 1)
        s.tree_cache.leave_write_through(base=b)
        s.tree_cache.leave_load_back(base=b)
        s.tree_cache.leave_revoke_and_releases(base=b)
    n = 6
    for k in range(n):
        run_planned_round(g.ranks[0])
        for s in g.ranks[1:]:
            run_frameless_cycle(s, k % WORLD)
    Guards.queues_empty(g.ranks, hk_census, n)


# =====================================================================================================================
# S2 -- two same-session requests, L3 load-back, row authority, matched 0 vs 3008 (h16j, #968)
# =====================================================================================================================

COMMON = 3008
R1, R2 = "weg2-0-1", "weg2-0-2"
LEN968 = {R1: 121600, R2: 108800}
POOL = 262144
PP0_FREE_BEFORE_R1 = 153152
FOLLOWER_FREE = 153216
FOLLOWER_EVICTABLE = 105920


def _urc_mod():
    from sglang.srt.mem_cache import unified_radix_cache as URC

    return URC


class _CD:
    def __init__(self):
        self.value = None
        self.host_value = None


class _Node968:
    _ids = itertools.count(1)

    def __init__(self, parent, label, n):
        URC = _urc_mod()
        self.id = next(_Node968._ids)
        self.parent = parent
        self.label = label
        self.n = int(n)
        self.children = {}
        self.component_data = {URC.BASE_COMPONENT_TYPE: _CD(), URC.ComponentType.MAMBA: _CD()}
        if parent is not None:  # host-only after a store read (L3 -> host)
            self.component_data[URC.BASE_COMPONENT_TYPE].host_value = torch.arange(self.n, dtype=torch.int64)
            self.component_data[URC.ComponentType.MAMBA].host_value = torch.zeros(1, dtype=torch.int64)

    @property
    def evicted(self):
        return self.component_data[_urc_mod().BASE_COMPONENT_TYPE].value is None


class _Alloc:
    def __init__(self, avail):
        self.avail = int(avail)

    def available_size(self):
        return self.avail


class _Lock:
    delta = 0

    def to_dec_params(self):
        return None


def _evicted_chain(node):
    chain = []
    while node is not None and node.parent is not None and node.evicted:
        chain.append(node)
        node = node.parent
    return list(reversed(chain))


class _KvXfer:
    def __init__(self, nodes):
        B = _urc_mod().BASE_COMPONENT_TYPE
        self.nodes_to_load = list(nodes)
        self.host_indices = torch.cat([n.component_data[B].host_value for n in nodes]) if nodes else torch.empty(0)
        self.device_indices = None


class _BaseComp:
    @property
    def component_type(self):
        return _urc_mod().BASE_COMPONENT_TYPE

    def build_hicache_transfers(self, node, phase, req=None):
        return [_KvXfer(_evicted_chain(node))]

    def commit_hicache_transfer(self, node, phase, xfers):
        B = _urc_mod().BASE_COMPONENT_TYPE
        for x in xfers:
            off = 0
            for n in x.nodes_to_load:
                n.component_data[B].value = torch.as_tensor(x.device_indices[off:off + n.n])
                off += n.n


class _MambaComp:
    @property
    def component_type(self):
        return _urc_mod().ComponentType.MAMBA

    def build_hicache_transfers(self, node, phase, req=None):
        if req is not None:
            req.mamba_loadback_anchor_adopted = True
        return [SimpleNamespace(name="mamba")]

    def commit_hicache_transfer(self, node, phase, xfers):
        pass


class _Controller968:
    def __init__(self, tree):
        self.tree = tree
        self.loads = []

    def load(self, host_indices, node_id, extra_pools=None):
        n = len(host_indices)
        alloc = self.tree.token_to_kv_pool_allocator
        if alloc.avail < n:
            return None
        alloc.avail -= n
        self.loads.append(n)
        return torch.arange(n, dtype=torch.int64)


def _stage968_cls():
    URC = _urc_mod()
    return type("_Stage968", (_Stage968Base,), {"load_back": URC.UnifiedRadixCache.load_back,
                                                "init_load_back": URC.UnifiedRadixCache.init_load_back})


class _Stage968Base:
    enable_storage = False
    prefetch_progress_is_collective_free = None

    def __init__(self, rank, avail, evictable=0):
        from sglang.srt.weg2 import pp_loadback_verdict as LBV
        from sglang.srt.weg2 import pp_slot_fidelity as SF

        URC = _urc_mod()
        B, M = URC.BASE_COMPONENT_TYPE, URC.ComponentType.MAMBA
        self.rank = rank
        self.root_node = _Node968(None, "root", 0)
        self.root_node.component_data[B].value = torch.empty(0, dtype=torch.int64)
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self._evictable = int(evictable)
        self.evicted_log = []
        self.uniform_avail_floor = int(avail)
        self.cache_controller = _Controller968(self)
        self.components = {B: _BaseComp(), M: _MambaComp()}
        self._components_tuple = (self.components[B], self.components[M])
        self.load_back_threshold = 10
        self.ongoing_load_back = {}
        self.metrics_collector = None
        self._empty_match_result = SimpleNamespace(device_indices=torch.empty(0, dtype=torch.int64))
        self.locks = 0
        # the P form: tp group of one under pp 3 -> the local-PP floor, the LB-VERDICT gate armed
        SF.mark_floor_scope(self, True)
        LBV.arm(SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=rank), tree_cache=self))

    def insert_read(self, rid):
        """#1423: a completed store read lands in the host tier -- (matched, inserted, deepest node)."""
        matched, node = 0, self.root_node
        for label, n in (("common", COMMON), (rid, LEN968[rid] - COMMON)):
            if label in node.children:
                node = node.children[label]
                matched += n
            else:
                child = _Node968(node, label, n)
                node.children[label] = child
                node = child
        return matched, LEN968[rid] - matched, node

    def inc_host_lock_ref(self, node):
        return _Lock()

    def dec_host_lock_ref(self, node, params=None):
        pass

    def inc_lock_ref(self, node):
        self.locks += 1
        return _Lock()

    def dec_lock_ref(self, node, params=None):
        self.locks -= 1

    def _1424_verify_load_chain(self, kv_xfer, req=None):
        return None

    def _build_sidecar_transfers(self, phase, kv, comp):
        return []

    def evictable_size(self):
        return self._evictable

    def evict(self, params):
        n = min(int(params.num_tokens), self._evictable)
        self._evictable -= n
        self.token_to_kv_pool_allocator.avail += n
        self.evicted_log.append(n)
        return SimpleNamespace(num_tokens_evicted=n)

    def _record_store_event(self, node, medium=None):
        pass

    def _update_evictable_leaf_sets(self, node):
        pass

    def check_prefetch_progress(self, rid):
        pass


class _Req968:
    def __init__(self, rid, best, root):
        self.rid = rid
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.best_match_node = best
        self.last_node = root
        self.host_hit_length = LEN968[rid]
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        self.full_untruncated_fill_ids = list(range(LEN968[rid] + 38))
        self.cache_protected_len = 0
        self.mamba_loadback_anchor_adopted = False

    def init_next_round_input(self, tree=None):
        pass


@pytest.fixture
def s2_env(monkeypatch):
    from sglang.srt.managers import pp_admission_congruence as PAC
    from sglang.srt.weg2 import pp_loadback_verdict as LBV
    from sglang.srt.weg2 import pp_slot_fidelity as SF

    for env in (LBV.ENV, getattr(LBV, "ROW_YIELD_ENV", "SGLANG_WEG2_LBV_ROW_YIELD"),
                getattr(LBV, "ROW_ROOM_ENV", "SGLANG_WEG2_LBV_ROW_ROOM"), getattr(SF, "ENV", "SGLANG_WEG2_SF"),
                "SGLANG_WEG2_P_ROW_AUTHORITY"):
        monkeypatch.delenv(env, raising=False)
    from sglang.srt.weg2 import p_row_authority

    if not p_row_authority.applies(SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=1))):
        pytest.skip("n/a: the NF form of this tree runs P without the row authority (no row to execute)")
    if not hasattr(PAC, "execute_scheduled_prefix"):
        raise RigError("pp_admission_congruence.execute_scheduled_prefix (the #968 station) is gone")
    if hasattr(PAC, "store_read_bound_s"):
        monkeypatch.setattr(PAC, "store_read_bound_s", lambda deficit: 0.05)
    if hasattr(PAC, "MATERIALISE_POLL_S"):
        monkeypatch.setattr(PAC, "MATERIALISE_POLL_S", 0.005)


def _pp0_loads(pp0, rid, best):
    """PP0's own plan: the LB-VERDICT gate holds, PP0 decides, the verdict is absorbed, PP0 loads."""
    from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
    from sglang.srt.weg2 import pp_loadback_verdict as LBV

    req = _Req968(rid, best, pp0.root_node)
    pp0.load_back(best, None, req=req)
    for v in LBV.take(pp0):
        LBV.absorb(pp0, v)
    new, _last = pp0.init_load_back(InitLoadBackParams(best_match_node=best, host_hit_length=LEN968[rid], req=req))
    req.prefix_indices = torch.cat([req.prefix_indices, new])
    return req


def _follower_executes_row(stage, rid, best, scheduled):
    from sglang.srt.managers import pp_admission_congruence as PAC

    req = _Req968(rid, best, stage.root_node)
    got = PAC.execute_scheduled_prefix(req, stage, scheduled)
    return req, got


def _cap_for_followers(pp0, followers):
    from sglang.srt.weg2 import pp_room_vote as PR

    book = PR.RoomBook()
    book.begin_pass()
    for s in followers:
        room = s.token_to_kv_pool_allocator.avail + s._evictable
        book.absorb([PR.Weg2PpRoomFact(rank=s.rank, executed=0, room=room, available=room - s._evictable,
                                       payable=s._evictable, reported=s._evictable)])
    verdict = book.cap(own_room=pp0.token_to_kv_pool_allocator.avail)
    wire = PR.stamp_cap(["req"], verdict)
    for s in followers:
        _rest, cap = PR.absorb_cap(list(wire))
        PR.set_cap(s, cap)


def test_S2_same_session_l3_loadback_under_row_authority(s2_env):
    """Boot, two same-session requests read from L3 on every stage; the reads land in a different order on PP0 than
    on the followers (#1423 matched 0 vs 3008). weg2-0-2 loads first everywhere, then PP0 loads weg2-0-1 and the
    followers execute that row under the R of PP0's NEXT pass. G1: every stage holds the row's prefix."""
    from sglang.srt.weg2 import pp_room_vote as PR

    Stage = _stage968_cls()
    pp0 = Stage(0, PP0_FREE_BEFORE_R1 + LEN968[R2])
    fol = [Stage(1, FOLLOWER_FREE + LEN968[R2]), Stage(2, FOLLOWER_FREE + LEN968[R2])]
    placed = {0: {R1: pp0.insert_read(R1), R2: pp0.insert_read(R2)}}
    for s in fol:
        placed[s.rank] = {R2: s.insert_read(R2), R1: s.insert_read(R1)}
    assert placed[0][R1][:2] == (0, 121600) and placed[1][R1][:2] == (COMMON, 118592), "constellation"
    try:
        # pass m-1: weg2-0-2 by PP0's plan, by the row on the followers
        r2 = _pp0_loads(pp0, R2, placed[0][R2][2])
        depths = [len(r2.prefix_indices)]
        for s in fol:
            PR.set_cap(s, POOL)
            req, _got = _follower_executes_row(s, R2, placed[s.rank][R2][2], LEN968[R2])
            depths.append(len(req.prefix_indices))
            s._evictable = FOLLOWER_EVICTABLE
            s.token_to_kv_pool_allocator.avail = FOLLOWER_FREE
            s.uniform_avail_floor = FOLLOWER_FREE
        Guards.same_admission(0, [[(R2, d)] for d in depths], "row depth")
        pp0.uniform_avail_floor = pp0.token_to_kv_pool_allocator.avail
        # pass m: PP0 loads weg2-0-1; pass m+1 stamps PP0's post-load R; the followers execute the row of pass m
        r1 = _pp0_loads(pp0, R1, placed[0][R1][2])
        _cap_for_followers(pp0, fol)
        depths = [len(r1.prefix_indices)]
        for s in fol:
            req, _got = _follower_executes_row(s, R1, placed[s.rank][R1][2], LEN968[R1])
            depths.append(len(req.prefix_indices))
    except LawViolation:
        raise
    except Exception as exc:  # noqa: BLE001 - the tree stopped the group by name
        raise _group_stop(exc) from exc
    Guards.same_admission(1, [[(R1, d)] for d in depths], "row depth")


# =====================================================================================================================
# S4 -- the shared arena: PP0, the followers and D on ONE C arena with a file store behind it
# =====================================================================================================================

TOTAL = 4096


def _need_gcc():
    if shutil.which("gcc") is None:
        pytest.skip("needs gcc for the C arena")


def _arena(path, slots):
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    return ShmArena(str(path), TOTAL, slots)


def _fill_complete(arena, stems):
    claims = arena.claim_slots(stems, [TOTAL] * len(stems))
    assert [c[1] for c in claims] == [0] * len(stems), claims
    assert arena.complete_slots([c[0] for c in claims], [c[2] for c in claims], [(0, TOTAL)]) == [1] * len(stems)
    return claims


def _store(root, stems, salt=0x40):
    root.mkdir(exist_ok=True)
    for i, s in enumerate(stems):
        (root / (s + ".bin")).write_bytes(bytes([(salt + i) % 256]) * TOTAL)


def _backend(store_root, secure_hook=None):
    """A ``HiCacheFile`` of one stage: the REAL ``_arena_evict_to_disk`` / ``arena_fill_from_disk``; the disk IO of
    the evictor (``arena_secure_to_disk``) is a stand-in that runs ``secure_hook`` while the copy is in flight."""
    from sglang.srt.mem_cache.hicache_storage import HiCacheFile

    be = object.__new__(HiCacheFile)
    be.pins = None

    def _path(stem):
        return os.path.join(str(store_root), stem + ".bin")

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}

    def _secure(arena, cands, writer="claim_room"):
        if secure_hook is not None:
            secure_hook(cands)
        return {"written": len(cands)}

    be.arena_secure_to_disk = _secure
    return be


def _ref_at_land_wired():
    """REF-AT-LAND: does THIS tree's ``_arena_page_get`` reference every page as it lands (``take_ref=True``)?"""
    try:
        from sglang.srt.managers.cache_controller import HiCacheController

        src = inspect.getsource(HiCacheController._arena_page_get)
    except Exception:  # noqa: BLE001
        return False
    return "take_ref=True" in src


def _fill(be, arena, stems, take_ref):
    from sglang.srt.mem_cache.hicache_storage import HiCacheFile

    kw = {"prefix": True}
    if take_ref and "take_ref" in inspect.signature(HiCacheFile.arena_fill_from_disk).parameters:
        kw["take_ref"] = True
    out = HiCacheFile.arena_fill_from_disk(be, arena, stems, TOTAL, **kw)
    reffed = bool(kw.get("take_ref")) and getattr(out, "_arena", None) is not None
    return list(out), reffed


def _walk_ref(arena, stems, out, reffed):
    """The walk of ``_arena_page_get``: reference each landed page (unless the fill did) -- a page that is no
    longer there ends the prefix. Returns the (stem, slot) pairs this stage now references."""
    refs = []
    for stem, slot in zip(stems, out):
        if slot is None:
            break
        if not reffed:
            if arena.find_slots([stem])[0] != (int(slot), 2) or arena.ref_slots([int(slot)], +1) != 1:
                break
        refs.append((stem, int(slot)))
    return refs


def test_S4a_arena_reap_race_referenced_page_keeps_key(tmp_path, monkeypatch):
    """The h16h death 01:09:10 (ArenaChainMismatch #1424, bad=47): D's clock (live) holds 4 candidates mid copy;
    PP2's eviction tail reaps; PP0's fill claims, lands and references; D's clock finishes and frees its
    candidates. G4: PP0's referenced pages keep their keys."""
    _need_gcc()
    arena = _arena(tmp_path / "kv.bin", 8)
    _fill_complete(arena, [f"old{i}" for i in range(8)])
    store = tmp_path / "store"
    stems = [f"page{i}" for i in range(5)]
    _store(store, stems)
    took = {}
    pp2 = _backend(store)
    pp0 = _backend(store)
    pp0._arena_evict_to_disk = lambda a, want, need=None: 0   # PP0 waits for room elsewhere; no own clock here
    refs_box = {}

    def while_d_copies(cands):
        took["d"] = [c[0] for c in cands]
        took["pp2"] = type(pp2)._arena_evict_to_disk(pp2, arena, 1)    # PP2's eviction tail: '... reap_stale()'
        out, reffed = _fill(pp0, arena, stems, _ref_at_land_wired())   # PP0's fill (prefetch io)
        refs_box["refs"] = _walk_ref(arena, stems, out, reffed)

    d = _backend(store, secure_hook=while_d_copies)
    type(d)._arena_evict_to_disk(d, arena, 4)                          # D's clock: candidates -> copy -> free
    assert "refs" in refs_box, "the interleave did not run"
    Guards.keys_kept(arena, refs_box["refs"])


def test_S4b_d_clock_takes_landed_unreferenced_pages(tmp_path, monkeypatch):
    """Case 8 (00:59:48, W3 over the group boundary): PP0's fill lands its pages; before PP0's walk references
    them, D's arena clock evicts COMPLETE unreferenced slots of the shared arena. G4: what landed is referenced."""
    _need_gcc()
    arena = _arena(tmp_path / "kv.bin", 16)
    _fill_complete(arena, [f"old{i}" for i in range(8)])
    store = tmp_path / "store"
    stems = [f"page{i}" for i in range(8)]
    _store(store, stems)
    pp0 = _backend(store)
    out, reffed = _fill(pp0, arena, stems, _ref_at_land_wired())
    landed = sum(1 for s in out if s is not None)
    Guards.landed_full(landed, len(stems), "PP0 fill (arena has room)")
    d = _backend(store)
    type(d)._arena_evict_to_disk(d, arena, 16, need=16)                # D's clock, between landing and the walk
    refs = _walk_ref(arena, stems, out, reffed)
    Guards.landed_full(len(refs), landed, "PP0 walk references what its fill landed")
    Guards.keys_kept(arena, refs)


# ---------------------------------------------------------------------------- S4c: D parks, P reads (h54) --------

S4C_PAGES = 24
#: the metal bounds scaled 1/10 (join wait 5 s, stale claim 5 s, join idle 1 s, cap 60 s); 3-45 s park-to-read
S4C_ENV = {"SGLANG_WEG2_L3FILL_JOIN_WAIT_MS": "500", "SGLANG_WEG2_L3FILL_JOIN_WAIT_MAX_MS": "500",
           "SGLANG_WEG2_L3FILL_JOIN_WAIT_PER_PAGE_MS": "0.2", "SGLANG_WEG2_L3FILL_STALE_CLAIM_S": "0.5",
           "SGLANG_WEG2_L3FILL_JOIN_IDLE_MS": "100", "SGLANG_WEG2_L3FILL_JOIN_CAP_S": "6",
           "SGLANG_WEG2_L3FILL_ROOM_WAIT_MS": "500"}


def _d_parks(path, slots, stems, ready, release, mode):
    """D (another process, its own PID): the park's forced host write-through claims the stems (role HOST_WRITE,
    ``arena_pool._host_write_claim``) and the device->host copy lands the bytes; the COMPLETION of those claims is
    D's own housekeeping (``writing_check`` -> ``complete_slots``) on each of D's TP ranks, each for its own extent,
    and D's group goes dormant after the flip D->P -- the claims stay CLAIMED, the writer stays alive.

    ``mode``: ``park_refs`` -- the parked requests hold their host pages (a reader reference per slot);
    ``tp_merges`` -- TP0 merged its extent at once, TP1's merges trickle in during P's read (each one refreshes the
    claim's progress, metal: ``ARENA-REF-CENSUS-CLAIM claimed=200 oldest=3603/10583ms/pid857/host-write/g76/open0``
    13 s after the park), TP2 is dormant before its own -- the slot never turns COMPLETE;
    ``none`` -- open claims, no reference, no merge (the control)."""
    try:
        from sglang.srt.mem_cache.storage.file import hicache_arena as ha

        arena = ha.ShmArena(path, TOTAL, slots)
        role = int(getattr(ha, "ROLE_HOST_WRITE", 0))
        claims = arena.claim_slots(stems, [TOTAL] * len(stems), role=role) if role else \
            arena.claim_slots(stems, [TOTAL] * len(stems))
        fresh = [(slot, g) for slot, st, g in claims if st == 0]
        for slot, _g in fresh:
            arena.slot_view(slot, TOTAL)[:TOTAL] = b"\x11" * TOTAL
        third = TOTAL // 3
        if mode == "park_refs" and fresh:
            arena.ref_slots([s for s, _ in fresh], +1)
        if mode == "tp_merges" and fresh:
            arena.complete_slots([s for s, _ in fresh], [g for _, g in fresh], [(0, third)])      # TP0
        ready.send(len(fresh))
        if mode == "tp_merges":
            k = 0
            while not release.poll(0.04):                                                    # TP1, one slot a tick
                slot, g = fresh[k % len(fresh)]
                if k < len(fresh):
                    arena.complete_slots([slot], [g], [(third, third)])
                k += 1
        release.recv()
    finally:
        os._exit(0)


def _s4c(tmp_path, monkeypatch, mode):
    _need_gcc()
    for k, v in S4C_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("SGLANG_WEG2_L3FILL_JOIN_LIVENESS", raising=False)
    slots = 64
    path = str(tmp_path / "kv.bin")
    arena = _arena(path, slots)
    _fill_complete(arena, [f"old{i}" for i in range(16)])
    store = tmp_path / "store"
    stems = [f"sess{i}" for i in range(S4C_PAGES)]
    _store(store, stems)
    ctx = mp.get_context("fork")
    r_ready, w_ready = ctx.Pipe(duplex=False)
    r_rel, w_rel = ctx.Pipe(duplex=False)
    overlap = stems[: S4C_PAGES * 2 // 3]                               # D's decode tail overlaps P's prefix
    d = ctx.Process(target=_d_parks, args=(path, slots, overlap, w_ready, r_rel, mode), daemon=True)
    d.start()
    try:
        if not r_ready.poll(30):
            raise RigError("D's park process did not claim in 30 s")
        claimed = r_ready.recv()
        assert claimed == len(overlap), f"D claimed {claimed} of {len(overlap)}"
        time.sleep(0.3)                                                 # 3 s after the park, scaled
        from sglang.srt.mem_cache.hicache_storage import PREFETCH_IO_THREAD_PREFIX

        box = {}
        p = _backend(store)
        p._arena_evict_to_disk = lambda a, want, need=None: 0

        def prefetch_io():
            try:
                box["out"] = _fill(p, arena, stems, _ref_at_land_wired())
            except Exception as exc:  # noqa: BLE001
                box["exc"] = exc

        t = threading.Thread(target=prefetch_io, name=f"{PREFETCH_IO_THREAD_PREFIX}-0")
        t0 = time.perf_counter()
        t.start()
        t.join(60)
        waited = time.perf_counter() - t0
        assert d.is_alive(), "D's writer died -- the scenario is a LIVE parked holder"
        if "exc" in box:
            raise _group_stop(box["exc"])
        if "out" not in box:
            raise LawViolation("G4 LANDED-DEPTH", "P's store read did not return in 60 s")
        out, reffed = box["out"]
        refs = _walk_ref(arena, stems, out, reffed)
        Guards.landed_full(len(refs), len(stems), f"P's L3 read after D's park (waited {waited:.2f} s, "
                                                  f"{len(overlap)} stems under D's host-write claims)")
        Guards.keys_kept(arena, refs)
    finally:
        try:
            w_rel.send(1)
        except Exception:  # noqa: BLE001
            pass
        d.join(5)
        if d.is_alive():
            d.kill()


def test_S4c_d_park_host_write_then_p_l3_read(tmp_path, monkeypatch):
    """h54 (16 of 16 JOIN-WAIT tears 3-40 s after a PARK-IMMEDIATE/PARK-RUNNING, 67-100k tokens each, once
    ArenaChainMismatch): D's park claims the stems with a forced host write-through, holds them for the resume and
    goes dormant before its housekeeping completes the claims; P reads the overlapping prefix from L3 seconds later.
    G4: the full store depth lands, every referenced page keeps its key."""
    _s4c(tmp_path, monkeypatch, mode="park_refs")


def test_S4c3_d_park_tp_merges_trickle_then_p_l3_read(tmp_path, monkeypatch):
    """S4c, second reading of the metal: no reference, but D's TP ranks merge their extents of the host-write claims
    at different times (TP2 dormant before its own): the claims stay CLAIMED and their progress keeps moving."""
    _s4c(tmp_path, monkeypatch, mode="tp_merges")


def test_S4c2_control_d_claims_without_park_reference(tmp_path, monkeypatch):
    """Control of S4c: the same dormant host-write claims, but no reader reference on them -- the stale-claim reap
    may take them (generation-safe) and P reads the pages itself."""
    _s4c(tmp_path, monkeypatch, mode="none")


# =====================================================================================================================
# S5 -- a vision request over P (row form)
# =====================================================================================================================


class _Item:
    def __init__(self, n=4):
        self.feature = torch.randn(n, 8)
        self.precomputed_embeddings = None
        self.modality = "image"


def _vreq(rid, items=()):
    return SimpleNamespace(rid=rid, multimodal_inputs=SimpleNamespace(mm_items=list(items)))


def _vision_stages(queue):
    from sglang.srt.weg2 import vision_verdict as vv

    pp0 = SimpleNamespace(
        waiting_queue=list(queue), weg2_dormant=False, _weg2_vision_refused=set(), _weg2_vision_arm_refusal="",
        _weg2_vision_origin_aborts=[], _weg2_vision_runs=0,
        running_batch=SimpleNamespace(is_empty=lambda: True), chunked_req=None,
        server_args=SimpleNamespace(model_path="/m"), model_config=SimpleNamespace(hf_config=None),
        ps=SimpleNamespace(pp_size=3, pp_rank=0), pp_flip_counters=None)
    vv.arm(pp0)
    fol = []
    for r in (1, 2):
        f = SimpleNamespace(waiting_queue=[_vreq(q.rid, [_Item()] if q.multimodal_inputs.mm_items else [])
                                           for q in queue],
                            weg2_dormant=False, ps=SimpleNamespace(pp_size=3, pp_rank=r))
        f._weg2_vision_follower_gate = vv.arm(f)
        fol.append(f)
    return pp0, fol


@pytest.fixture
def vision_stage(monkeypatch):
    from sglang.srt.weg2 import vision_rank_runner as vrr

    def fake(scheduler, reqs, **kw):
        for r in reqs:
            for it in vrr.unstaged_items(r):
                it.precomputed_embeddings, it.feature = torch.zeros(1, 6), None
        return vrr.StageOutcome()

    monkeypatch.setattr(vrr, "run_rank_stage", fake)
    monkeypatch.setattr(vrr, "_rank_device", lambda: torch.device("cpu"))
    if hasattr(vrr, "VISION_ASYNC_ENV"):
        monkeypatch.setenv(vrr.VISION_ASYNC_ENV, "0")


def s5_vision_over_p(passes=4, chain_lag=0):
    """Pass n: every stage dispatches the chain list PP0 put out at its intake (the request chain precedes the
    frame), PP0 plans (``vision_rank_pass``: stage, hold, verdict queued), the followers take PP0's row and admit
    what it names -- held by their own gate or not. ``chain_lag`` > 0 is the self-check mutant (the verdict reaches
    the followers passes late)."""
    from sglang.srt.weg2 import vision_rank_runner as vrr
    from sglang.srt.weg2 import vision_verdict as vv

    queue = [_vreq("t1"), _vreq("img", [_Item()]), _vreq("t2")]
    pp0, fol = _vision_stages(queue)
    in_flight = []
    for n in range(passes):
        wire = vrr.take_origin_aborts(pp0)
        in_flight.append(wire)
        for obj in wire:
            if isinstance(obj, vv.Weg2VisionVerdict):
                vv.absorb(pp0, obj)
        late = in_flight[n - chain_lag] if n - chain_lag >= 0 else []
        for f in fol:
            for obj in late:
                if isinstance(obj, vv.Weg2VisionVerdict):
                    vv.absorb(f, obj)
        parked = vrr.vision_rank_pass(pp0)
        row = [r.rid for r in pp0.waiting_queue]
        vrr.vision_unpark(pp0, parked)
        batches = [row]
        for f in fol:
            fp = vv.follower_pass(f) if f._weg2_vision_follower_gate else []
            have = {r.rid for r in f.waiting_queue}
            vrr.vision_unpark(f, fp)
            batches.append([rid for rid in row if rid in have])      # the row arm: admit what the row names
        Guards.same_admission(n, batches, "vision admission")
        for s in [pp0] + fol:
            s.waiting_queue = [r for r in s.waiting_queue if r.rid not in row]
    left = [[r.rid for r in s.waiting_queue] for s in [pp0] + fol]
    if any(left):
        raise LawViolation("G1 RAENGE-NIE-UNEINS", f"vision request never admitted in {passes} passes: {left}")


def test_S5_vision_request_over_p(vision_stage):
    s5_vision_over_p()


def test_selfcheck_S5_late_verdict_is_caught(vision_stage):
    """SELF-CHECK: a verdict that reaches the followers one pass after the row names the image must fire G1."""
    with pytest.raises(LawViolation, match="G1"):
        s5_vision_over_p(chain_lag=1)


# =====================================================================================================================
# S6 -- LB-VERDICT burst with a late-starting twin on the carrierless form (h16h 19:33:48)
# =====================================================================================================================

R659, R658, R657, R656 = "weg2-70-659", "weg2-70-658", "weg2-70-657", "weg2-70-656"
BURST = [R659, R658, R657, R656]
HOST_PP0 = {R659: 121600, R658: 90816, R657: 90240, R656: 0}
HOST_FOLLOWER = {R659: 121600, R658: 90816, R657: 90240, R656: 98112 - 94528}


class _LbvStage:
    """One carrierless P stage: its queue, its own host hits, PP0-only deferrals (the twin wait), the scheduler's
    LBV-ADMIT gate where the tree has one, and the ``load_back`` seam -- reduced to their decisions."""

    def __init__(self, rank, host, defer=None):
        from sglang.srt.weg2 import pp_loadback_verdict as LBV
        from sglang.srt.weg2.pp_slot_fidelity import FLOOR_LOCAL_PP_ATTR

        self.LBV = LBV
        self.rank = rank
        self.tree = SimpleNamespace()
        setattr(self.tree, FLOOR_LOCAL_PP_ATTR, True)
        self.sched = SimpleNamespace(ps=SimpleNamespace(pp_size=3, pp_rank=rank), tree_cache=self.tree)
        LBV.arm(self.sched)
        self.host = dict(host)
        self.defer = defer or (lambda stage, rid: False)
        self.queue = []

    def plan(self):
        LBV = self.LBV
        admit_holds = getattr(LBV, "admit_holds", None)
        admit_done = getattr(LBV, "admit_done", lambda req: None)
        live = getattr(LBV, "live", None) or (lambda tree, req: LBV.released(tree, req.rid))  # 853534a9ff seam
        gated = getattr(LBV, "gated", None) or (lambda tree, req: LBV.armed(tree))
        admitted = []
        for rid in list(self.queue):
            if self.defer(self, rid):
                continue
            kv = int(self.host.get(rid, 0))
            req = SimpleNamespace(rid=rid, pp_load_back_extent=kv, host_hit_length=kv)
            if admit_holds is not None and admit_holds(self.sched, req, pp0_room=lambda kv_, rid_: True):
                continue
            try:
                if kv:
                    if gated(self.tree, req) and not live(self.tree, req):
                        LBV.hold(self.tree, rid, kv, lambda: True)
                        break  # applied=0 under an adopted anchor -> NO_TOKEN -> the loop breaks
                    LBV.consume(self.tree, rid)
                admitted.append(rid)
            finally:
                admit_done(req)
        for rid in admitted:
            self.queue.remove(rid)
        return admitted


@pytest.fixture
def carrierless(monkeypatch):
    from sglang.srt.weg2 import pp_loadback_verdict as LBV

    for k in (LBV.ENV, getattr(LBV, "ADMIT_ENV", "SGLANG_WEG2_LBV_ADMIT"),
              getattr(LBV, "ROW_YIELD_ENV", "SGLANG_WEG2_LBV_ROW_YIELD"), "SGLANG_PP_ROW_AUTHORITY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SGLANG_WEG2_P_ROW_AUTHORITY", "0")


def test_S6_lbv_burst_late_twin_carrierless(carrierless):
    """Four load-back requests out of one collect window; PP0 verdicts 659/658/657, -656 is -657's twin whose
    prefetch started late (PP0-only twin wait), device-resident on PP0 and host-only on PP1/PP2. G1 per pass."""

    def twin_wait(stage, rid):
        return stage.rank == 0 and rid == R656 and R657 in stage.queue

    stages = [_LbvStage(0, HOST_PP0, defer=twin_wait), _LbvStage(1, HOST_FOLLOWER), _LbvStage(2, HOST_FOLLOWER)]
    for s in stages:
        s.queue = list(BURST)
    LBV = stages[0].LBV
    intake = []
    for n in range(12):
        for s in stages:
            for v in intake:
                LBV.absorb(s.tree, v)
        batches = [s.plan() for s in stages]
        try:
            Guards.same_admission(n, batches, "batch (a split = PP0 in recv_object[src=2], PP1/PP2 in the chain "
                                              "receive: RING-WAIT)")
        except LawViolation:
            raise
        intake = LBV.take(stages[0].tree)
    left = [s.queue for s in stages]
    if any(left):
        raise LawViolation("G1 RAENGE-NIE-UNEINS", f"burst not served in 12 passes: {left}")


# =====================================================================================================================
# SELF-CHECKS: every guard fires on the mutant of the decision that closed its death (a rig that cannot go red
# measures nothing). On a tree where the scenario is red anyway, the self-check passes trivially.
# =====================================================================================================================


def test_selfcheck_S1_without_any_follower_backup_drain_is_w3(row_authority, monkeypatch):
    from sglang.srt.managers import scheduler_pp_mixin as mix

    monkeypatch.setenv("SGLANG_WEG2_P_FOLLOWER_ROUND", "0")
    if hasattr(mix, "weg2_backup_ack_drain_on_idle_vote"):
        monkeypatch.setattr(mix, "weg2_backup_ack_drain_on_idle_vote", lambda s: 0)
    with pytest.raises(LawViolation, match="G3"):
        test_S1_first_flip_with_flush_after_write_through(row_authority)


def test_selfcheck_S2_row_room_off_is_968(s2_env, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_LBV_ROW_ROOM", "0")
    with pytest.raises(LawViolation, match="#968"):
        test_S2_same_session_l3_loadback_under_row_authority(s2_env)


def test_selfcheck_S3_round_off_leaves_a4_a5(row_authority, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_P_FOLLOWER_ROUND", "0")
    with pytest.raises(LawViolation, match="G2"):
        test_S3_revokes_releases_on_frameless_followers(row_authority)


def test_selfcheck_S4a_instant_reap_loses_the_key(tmp_path, monkeypatch):
    _need_gcc()
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    def instant_reap(self, *a, **k):
        lib = self._lib
        if hasattr(lib, "arena_reap_stale_aged"):
            return int(lib.arena_reap_stale_aged(self._base, 0))
        return int(lib.arena_reap_stale(self._base))

    monkeypatch.setattr(ShmArena, "reap_stale", instant_reap)
    if hasattr(ShmArena, "free_evicted"):   # ARENA-FREE-GEN: the evictor's free bound to its own candidates
        monkeypatch.setattr(ShmArena, "free_evicted", lambda self, slots, reason="unnamed": self.free_slots(slots, reason=reason))
    with pytest.raises(LawViolation, match="G4"):
        test_S4a_arena_reap_race_referenced_page_keeps_key(tmp_path, monkeypatch)


def test_selfcheck_S4b_ref_at_land_off_loses_the_landing(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L3FILL_REF_AT_LAND", "0")
    with pytest.raises(LawViolation, match="G4"):
        test_S4b_d_clock_takes_landed_unreferenced_pages(tmp_path, monkeypatch)


def test_selfcheck_S6_lbv_admit_off_is_the_ring_wait(carrierless, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_LBV_ADMIT", "0")
    with pytest.raises(LawViolation, match="G1"):
        test_S6_lbv_burst_late_twin_carrierless(carrierless)
