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
  G4 no referenced arena page loses its key; a read lands what it was told to land; a live writer's reference on
     its claim survives a take-over (G4c, the slot keeps its generation).
  G10 (S10) an abort PP0 decided reaches every stage on the same pass PP0 ran: no stage pops a rid a frame
     still names (#1180), none computes on a rid PP0 aborted (leak), none releases it twice.

A guard that fires is a group stop by name -- the rig never repairs, never retries, never compensates (hard user
law raenge-nie-uneins-crash-stop): a detected disagreement is RED, full stop.

SCENARIOS (one test each, ``test_S<n>_...``; the runner ``tools/weg2_coldstart_rig.sh`` prints one line each):

  S1  first P->D flip with flush after prefills with direct write-through (sidecar backups)   red on 75e3ee8abd (W3)
  S2  two same-session requests, L3 load-back under the row authority, matched 0 vs 3008      red on 73f6a4e110 (#968)
  S3  revokes / host releases on frameless followers                                           red on e7cfda7fbe (A4/A5)
  S4a arena reap race: a live evictor's candidates reaped, PP0 fills + references them         red without F1 (01:09 death)
  S4b a D clock evicts PP0's landed pages before PP0 references them                           red without REF-AT-LAND (case 8)
  S4c D parks with forced host write-through, P reads the same stems from L3 seconds later     h54 (16/16 JOIN-WAIT tears)
      + G4c: the D writer node's reference on its old slot survives P's take-over                 red under M1 (A2 1015)
  S5  a vision request over P (row form): every stage admits it in the same pass               self-check by mutant
  S6  LB-VERDICT burst with a late-starting twin on the carrierless form (19:33 hang)          red on 853534a9ff
  S7  PF Frist against the pass cadence (h64 "PF TOLD-FALLBACK reason=frist" on held reads)    red on 17c2563647
      (+ S7b two-hop ack in long passes) -- G7 HELD-PREFIX
  S8  slow D park on a TP3 D group (h16n D epoch 90, the park undid itself, W3)               red on 17c2563647
      (+ S8b clean park walks no L3, S8c one refusing rank -> every rank walks and votes) -- G8 PARK-SELF-UNDO,
      G9 PARK-WALK
  S9  (test_S9_arena_full_pinned_row_room; numbered 9 so the S7/S8 of desk/nf-coldstart-rig-s7s8-1015 merge
      without a name clash -- united on desk/nf-coldstart-rig-h16q-1015)
      arena full and pinned by the peers, a 210k load-back executed on a follower (h16n 07:30:35) red on 4f58ca0256,
      17c2563647 and 853534a9ff (the defect is latent since 8f2fe84949; h16h never refused a P claim at the metal)
      + G5 ROW-ROOM: the follower pays the room PP0's row needs (no ROW-ROOM SHORT, no PrefillOutOfPages)
  S10 K14: a client abort (H102) reaches the followers before PP0's row frame that names the rid (h16p
      21:52:38Z weg2-164-2265, #1180 RANK_EXCEPTION)                                          red on 1cfbf32584 (h16q2)
      (+ S10b PP0 had not admitted it, S10c chunked first chunk) -- G1 + G10 ABORT-UNIFORM: every stage that ran
      the frame's pass carries PP0's abort, no stage keeps the rid queued or held afterwards

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
            "SGLANG_WEG2_L3FILL_JOIN_IDLE_MS", "SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER", "SGLANG_WEG2_L3_DROP",
            "SGLANG_WEG2_L3_DROP_ACK_WAIT_MS", "SGLANG_WEG2_W3_SPILL_KEEP_PAYABLE", "SGLANG_WEG2_SF")


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
    def writer_ref_kept(arena, held, what):
        """G4c (nf-h54-review-1015 Auflage 3): a writer node's reference on its claim survives a take-over -- the
        old slot keeps its generation (never freed, never recycled) and its reference while the writer lives.
        ``held`` = [(slot, generation)] the writer process referenced."""
        if not held:
            return
        slots = [int(s) for s, _ in held]
        census = arena.claim_census(slots)
        refs = arena.slot_refs(slots) if hasattr(arena, "slot_refs") else [None] * len(slots)
        lost = [(s, g, c["gen"], c["state"], r) for (s, g), c, r in zip(held, census, refs)
                if c["gen"] != int(g) or (r is not None and r < 1)]
        if lost:
            raise LawViolation("G4 WRITER-REF",
                               f"{what}: {len(lost)} of {len(held)} slot(s) the live writer references lost it "
                               f"(slot, gen held, gen now, state now, refs now): {lost[:6]} -- the take-over freed a "
                               "referenced slot; the writer's node now addresses whatever the slot holds next")

    @staticmethod
    def row_room(paid, need, what):
        """G5 (ARENAFULL-1015): a follower executing PP0's row pays the room the row needs -- else the metal stop
        'LB-VERDICT ROW-ROOM SHORT' (07:30:35) / PrefillOutOfPages (07:11:54) ends the group."""
        if paid < need:
            raise LawViolation("G5 ROW-ROOM",
                               f"{what}: the peel paid {paid} of the {need} rows PP0's row needs -- "
                               "LB-VERDICT ROW-ROOM SHORT, the group stops by name")

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
    ``tp_merges`` -- TP0 merged its extent at once and its node references the partial slot (``complete_write``
    references status <= 2), TP1's merges trickle in during P's read (each one refreshes the
    claim's progress, metal: ``ARENA-REF-CENSUS-CLAIM claimed=200 oldest=3603/10583ms/pid857/host-write/g76/open0``
    13 s after the park), TP2 is dormant before its own -- the slot never turns COMPLETE;
    ``none`` -- open claims, no reference, no merge (the control).
    ``ready`` receives ``(fresh claims, [(slot, generation) the writer references])`` (G4c WRITER-REF)."""
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
        held = []
        if mode == "park_refs" and fresh:
            arena.ref_slots([s for s, _ in fresh], +1)
            held = list(fresh)
        if mode == "tp_merges" and fresh:
            st = arena.complete_slots([s for s, _ in fresh], [g for _, g in fresh], [(0, third)])      # TP0
            took = [f for f, x in zip(fresh, st) if int(x) <= 2]       # complete_write: its node's reference
            if took:
                arena.ref_slots([s for s, _ in took], +1)
            held = took
        ready.send((len(fresh), held))
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
        claimed, d_held = r_ready.recv()
        assert claimed == len(overlap), f"D claimed {claimed} of {len(overlap)}"
        if mode in ("park_refs", "tp_merges") and len(d_held) != claimed:
            raise RigError(f"D's writer referenced {len(d_held)} of its {claimed} claims")
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
        Guards.writer_ref_kept(arena, d_held, f"D's writer node after P's read (mode {mode})")
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
    """S4c, second reading of the metal: D's TP ranks merge their extents of the host-write claims at different times
    (TP2 dormant before its own): the claims stay CLAIMED, their progress keeps moving, and TP0's node references the
    partial slot it merged into (``complete_write``, status <= 2). G4c: that reference survives P's take-over."""
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
# S7 -- PF Frist against the pass cadence (h16n 17c2563647, h64: "PF TOLD-FALLBACK ... reason=frist" on held reads)
# =====================================================================================================================
# The PF group fallback (told := 0 when a follower's read is late) counted WALL time while the acks it waits for travel
# in PASSES: a follower acks in its pass, one chain hop per stage, and the gloo ack takes two PP0 harvests (size frame,
# then the payload receive PP0 posts only on seeing the size). With 3-7 s passes (a 16k-chunk prefill on PP0) the
# Frist ran out with every stage holding the SAME depth -- 130432 tokens recomputed although the store held them.
#   case A (weg2-6-44) a follower's early read said 93440 (read at intake, before the twin sibling was published), the
#          told-limited settle found the span LOCALLY (96640) and issued no read, but answered "short": its ack waited
#          for a read that never came, i.e. for the next follower pass -- one chain hop behind PP0's Frist.
#   case B (weg2-8-47) all three stages read 33792 alike; the 9.9 s Frist expired after three PP0 passes with
#          acks={1: 33792}, PP2's ack still on the wire; PP0 had no paced rid left and stopped harvesting.
# Law G7 HELD-PREFIX: every stage proved the same depth (read = told) -> the admission depth of EVERY rank is told
# (never the group fallback 0, never a depth the stages do not share: G1). PP0 decides, the followers follow the wire.
# The Ring (a PP3 ring double: pass k plans PP0's pass k - pp_rank, PP0 publishes at the top of its pass, the
# followers absorb one pass per hop later) and the two-step gloo stream are the ones of the h64 tests, carried here so
# the copy runs in trees that do not have them. Every H64 switch is looked up by NAME and never set: the tree's own
# defaults decide (on 17c2563647 there are none -> the h16n behaviour -> red by behaviour).

_PF_PAGE = 64


class _PfClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _PfLedgerTree:
    """Store reads that terminate at a wall-clock time, with the row ledger of the in-flight reads."""

    def __init__(self, clock):
        self.clock = clock
        self.ongoing = {}
        self.completed = self._prefetch_completed_tokens = {}
        self.loaded = {}
        self.op_refs = 0
        self.released = []
        self.is_eagle = False

    def start_read(self, rid, tokens, duration):
        rows = -(-int(tokens) // _PF_PAGE)
        self.ongoing[rid] = (self.clock() + duration, int(tokens), rows)
        self.op_refs += rows

    def check_prefetch_progress(self, rid):
        op = self.ongoing.get(rid)
        if op is None:
            return True
        done_at, tokens, rows = op
        if self.clock() < done_at:
            return False
        del self.ongoing[rid]
        self.op_refs -= rows
        self.completed[rid] = tokens
        self.loaded[rid] = tokens
        return True

    def completed_prefetch_tokens(self, rid):
        return self.completed.get(rid)

    @property
    def prefetch_loaded_tokens_by_reqid(self):
        return self.loaded

    def pop_prefetch_loaded_tokens(self, rid):
        return self.loaded.pop(rid, 0)

    def release_aborted_request(self, rid):
        self.released.append(rid)
        self.loaded.pop(rid, None)
        self.completed.pop(rid, None)
        op = self.ongoing.pop(rid, None)
        if op is not None:
            self.op_refs -= op[2]

    def prefetch_progress_is_collective_free(self):
        return True


class _PfStage:
    def __init__(self, pp_rank, clock, prompts, read_s):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.pp_flip_counters = None
        self.tree_cache = _PfLedgerTree(clock)
        self.waiting_queue = []
        self.prompts = prompts
        self.read_s = read_s
        self.admitted = []      # (planned PP0 pass, rid, told cap, credit)

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        hit = self.prompts.get(req.rid, 0)
        if limit_tokens is not None:
            hit = min(hit, int(limit_tokens))
        dur = self.read_s.get(req.rid, 0.0)
        if hit <= 0 or dur is None:
            return "declined:store_absent"
        self.tree_cache.start_read(req.rid, hit, dur)
        return "issued"


class _GlooLike:
    """The gloo ack stream, one step per PP0 harvest per follower: an ack is posted (size on the wire), its size is
    taken on PP0's next harvest (the payload receive is posted then), its payload on the harvest after; the follower's
    send stays in flight until PP0 posted that payload receive."""

    def __init__(self):
        self.q = {1: [], 2: []}
        self.harvests = 0


class _FollowerEnd:
    def __init__(self, g, rank):
        self.g, self.rank = g, rank

    def _inflight(self):
        q = self.g.q[self.rank]
        return bool(q) and q[-1][1] == 0

    def send_nowait(self, ack):
        if self._inflight():
            return False
        self.g.q[self.rank].append([ack, 0])
        return True

    def pump(self):
        return not self._inflight()


class _Pp0End:
    def __init__(self, g):
        self.g = g

    def harvest(self):
        self.g.harvests += 1
        out = []
        for _r, q in self.g.q.items():
            if not q:
                continue
            if q[0][1] == 0:
                q[0][1] = 1
            else:
                out.append(q.pop(0)[0])
        return out


class _PfRing:
    def __init__(self, m, monkeypatch, prompts, read_s_by_rank, dt):
        self.m, self.dt = m, dt
        self.clock = _PfClock()
        monkeypatch.setattr(m, "_clock", self.clock)
        self.stages = [_PfStage(r, self.clock, prompts, read_s_by_rank[r]) for r in range(3)]
        self.gloo = _GlooLike()
        self.stages[0]._weg2_fb_channel = _Pp0End(self.gloo)
        for s in self.stages[1:]:
            s._weg2_fb_channel = _FollowerEnd(self.gloo, s.ps.pp_rank)
        for s in self.stages:
            if not m.armed(s):
                pytest.skip("n/a: the NF form of this tree does not arm the PF told fallback")
        self.wire, self.pass_n = {}, 0
        self.sleeps = {0: 0.0, 1: 0.0, 2: 0.0}
        self._in = None

        def _sleep(sec):
            self.sleeps[self._in] += sec
            self.clock.t += sec

        monkeypatch.setattr(m.time, "sleep", _sleep)

    def arrive(self, rid):
        for s in self.stages:
            r = SimpleNamespace(rid=rid, prefetch_deferred=None)
            s.waiting_queue.append(r)
            self.m.intake(s, r, lambda gate: None)

    def step(self):
        m, k = self.m, self.pass_n
        self.wire[k] = m.pp0_publish(self.stages[0], [])
        for s in self.stages[1:]:
            src = k - s.ps.pp_rank
            if src >= 0:
                m.follower_absorb(s, list(self.wire.get(src, [])))
        for s in self.stages:
            self._in = s.ps.pp_rank
            for r in list(s.waiting_queue):
                credit = m.admission(s, r, lambda kind, rid: None)
                if credit is None:
                    continue
                s.waiting_queue.remove(r)
                s.admitted.append((k - s.ps.pp_rank, r.rid, getattr(r, "_weg2_prefix_cap", None), credit))
        self.pass_n += 1
        self.clock.t += self.dt

    def run(self, passes):
        for _ in range(passes):
            self.step()

    def plans(self, rid):
        return [[a[:3] for a in s.admitted if a[1] == rid] for s in self.stages]

    def told(self, rid):
        """The depth PP0 told the group for ``rid`` (the absolute twin told on the first PF wire object)."""
        for k in sorted(self.wire):
            for o in self.wire[k]:
                if getattr(o, "rid", None) == rid and hasattr(o, "told"):
                    return int(o.told)
        return None


class Guards7:
    @staticmethod
    def held_prefix(ring, rid, held, what):
        """G7: every stage proved ``held`` tokens of ``rid`` (its read ended at ``held``) -> every rank admits at
        ``held`` (one plan, all stages -- G1 -- and never the group fallback 0 over a depth they all hold)."""
        plans = ring.plans(rid)
        Guards.same_admission(0, [p for p in plans], f"PF told admission of {rid}")
        if any(len(p) != 1 for p in plans):
            raise LawViolation("G1 RAENGE-NIE-UNEINS", f"{what}: {rid} not admitted exactly once on every stage in "
                                                       f"the passes run: {plans}")
        cap = plans[0][0][2]
        if cap != held:
            fb = getattr(ring.stages[0], "_pf_fallback_n", 0)
            raise LawViolation(
                "G7 HELD-PREFIX",
                f"{what}: every stage held {held} tokens of {rid}, the group admitted {cap} on all ranks "
                f"(PF TOLD-FALLBACK fired {fb}x on PP0, acks it saw: {getattr(ring.stages[0], '_pf_admit_acks_n', 0)}) "
                f"-- {held - (cap or 0)} cached tokens ignored (CACHE-HIT IGNORIERT), recomputed")


def _pf_modules():
    try:
        from sglang.srt.managers import weg2_store_told as m
        from sglang.srt.managers import weg2_told_fallback as fb
    except ImportError:
        pytest.skip("n/a: this tree has no PF told fallback")
    return m, fb


def _pf_ring(monkeypatch, prompts, reads, dt):
    m, fb = _pf_modules()
    # the form decides what is armed (nf_form fixture); the twin told is absolute on the NF row (TREE_KEY)
    if not m._absolute_armed():
        pytest.skip("n/a: the NF form of this tree does not run the absolute twin told")
    return m, fb, _PfRing(m, monkeypatch, prompts, reads, dt)


RID_A, TOLD_A, STALE_A = "weg2-6-44", 96640, 93440
RID_B, TOLD_B = "weg2-8-47", 33792


def s7_case_a(monkeypatch, dt=2.2, passes=12):
    """weg2-6-44: PP0's absolute twin told 96640 (head == told, window 0 -> the Frist is the 2 s grace); every
    follower's early read ended at the STALE 93440 (read at intake before the sibling published), its told-limited
    settle finds the span locally (MATCH-CENSUS-DEEP reached=96640) and issues no read. Passes are DT s long."""
    reads = {r: {RID_A: 0.1} for r in range(3)}
    m, fb, ring = _pf_ring(monkeypatch, {RID_A: 0}, reads, dt)
    pp0 = ring.stages[0]

    def pp0_read(req, rematch=True, limit_tokens=None):
        req._prefetch_registered_prefix_len = TOLD_A      # TWIN-REGISTER head=96640 verdict=declined:too_short
        return "declined:too_short"

    pp0._prefetch_kvcache = pp0_read
    for s in ring.stages[1:]:
        def fol_read(req, rematch=True, limit_tokens=None, tree=s.tree_cache):
            if limit_tokens is None:                        # the early read at intake: stale 93440, done at once
                tree.start_read(req.rid, STALE_A, 0.0)
                return "issued"
            return "declined:too_short"                     # #915 PREFETCH REFUSED reason=too_short need=0

        s._prefetch_kvcache = fol_read
    ring.arrive(RID_A)
    for s in ring.stages[1:]:
        for r in s.waiting_queue:
            r.host_hit_length = TOLD_A
    ring.run(passes)
    return ring


def s7_case_b(monkeypatch, dt=3.3, passes=10, grace="5.5"):
    """weg2-8-47: all three stages read 33792 alike (0.1 s), passes are DT s (a 16k chunk of someone else's
    recompute on PP0), the grace is sized so that the wall Frist lapses at PP0's fourth pass after the read-ahead
    -- when PP2's ack has only had its size frame taken."""
    reads = {r: {RID_B: 0.1} for r in range(3)}
    monkeypatch.setenv("SGLANG_WEG2_TOLD_FALLBACK_GRACE_S", grace)
    m, fb, ring = _pf_ring(monkeypatch, {RID_B: TOLD_B}, reads, dt)
    ring.arrive(RID_B)
    ring.run(passes)
    return ring


def test_S7_pf_frist_against_pass_cadence(monkeypatch):
    """h64 case A and B, each its own ring: G1 + G7 on the admission of the rid."""
    Guards7.held_prefix(s7_case_a(monkeypatch), RID_A, TOLD_A, "case A (stale early record, satisfied locally, 2.2 s passes)")


def test_S7b_pf_frist_two_hop_ack_in_long_passes(monkeypatch):
    Guards7.held_prefix(s7_case_b(monkeypatch), RID_B, TOLD_B, "case B (equal reads, 3.3 s passes, PP2 two hops down)")


# =====================================================================================================================
# S8 -- slow D park (h16n 17c2563647, D epoch 90, 06:56-06:57Z host swap storm: the park undid itself, W3)
# =====================================================================================================================
# Metal: the front sent /weg2/park_running to D, D's park took 31.3 s on the scheduler thread (HY settle's L3 stat walk
# 21.7 s although nothing was refused, #248 mark 9.3 s). ``park_running`` had stamped the awake re-queue clock
# (``d_seats.SINCE_ATTR``, bound 30 s, read by ``park_tick``) right after the retraction, AHEAD of those phases: when the
# answer left, the stamp was 31 s old and the very next pass (same second) re-queued the whole park. D decoded the parked
# requests again (a 205760-token load-back held the scheduler thread 72 s) while the front, told they were parked, began
# the D->P flip: its drain witness got no /get_server_info answer -> W3 Weg2DrainWitnessUnreachable.
#   S8  (A) a 31 s park on a TP3 D group (real ``park_running`` / ``park_tick`` on three rank threads, rank-local
#           monotonic clocks, a group-MIN collective between them): no parked request leaves the park before the
#           awake bound has run since the ANSWER; every rank agrees (G1); the net still works afterwards.
#   S8b (B) a park that retracted and refused nothing walks no L3 on any rank (real ``park_hold_yield.settle``: the
#           stat walk is the leaf that costs the 21.7 s); a park where ONE rank refused makes EVERY rank walk and vote
#           the same collectives (a rank that skips a vote its peers enter hangs the group).
# Law G8 PARK-SELF-UNDO: a park does not end before its answer has been out for the awake bound -- the bound measures
# the front's time, never the park's own duration. G9 PARK-WALK: a clean park walks nothing. NOT modelled: the front's
# own lapse clock (``_d_parked`` stamped at t_park before a 54 s RPC), the PLE-warm GIL hold (C), the load-back.

PARK_S_RANKS = (31.3, 30.9, 31.1)       # epoch 90: stage=park ms=31311 on TP0, TP1/TP2 within a second of it
HY_SETTLE_S = 21.7                      # ... of which yield:21732 (hy.settle), l3mark:9334 (park_l3.mark_parked) the rest


class _RankClock:
    def __init__(self, t=1000.0):
        self.t = float(t)

    def monotonic(self):
        return self.t

    def time(self):
        return self.t

    def perf_counter(self):
        return self.t


class _RankTime:
    """Stand-in for the ``time`` module of ``d_park_runtime``: every rank thread reads its own clock."""

    def __init__(self):
        self.tl = threading.local()

    def _c(self):
        return self.tl.clock

    def monotonic(self):
        return self._c().monotonic()

    def time(self):
        return self._c().time()

    def perf_counter(self):
        return self._c().perf_counter()


class _GroupCollective:
    """A MIN all-reduce over the D ranks (barrier based). A rank that enters one collective more or fewer than its
    peers breaks the barrier: the real group would hang -- RED (G1), never repaired."""

    def __init__(self, n, timeout=4.0):
        self.n, self.timeout = n, timeout
        self.barrier = threading.Barrier(n, timeout=timeout)
        self.slots = [None] * n

    def reduce(self, rank, values, label):
        try:
            self.slots[rank] = list(values)
            self.barrier.wait()
            out = [min(col) for col in zip(*self.slots)]
            self.barrier.wait()
            return out
        except threading.BrokenBarrierError:
            raise LawViolation(
                "G1 RAENGE-NIE-UNEINS",
                f"collective {label!r}: the ranks did not enter the same collectives (rank {rank} waited "
                f"{self.timeout:.0f} s for its peers) -- on the metal a hung group")


class _DBatch:
    def __init__(self, reqs):
        self.reqs = list(reqs)
        self.batch_is_full = True

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, **_kw):
        pass

    def retract_all(self, server_args, offload_kv=True, retain=False):
        out, self.reqs = self.reqs, []
        return out


def _dreq(rid, seq):
    return types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 10, output_ids=[0] * 3,
                                 is_fast_lane=False, spill_class=None)


class _DSched:
    """The scheduler state ``d_park_runtime`` moves on one D rank, nothing more (the h91c2 stand-in)."""

    def __init__(self, rank, group, running=(), waiting=(), settle=()):
        from collections import deque

        self.rank = rank
        self.running_batch = _DBatch(running)
        self.waiting_queue = list(waiting)
        self.weg2_post_wake_settle = list(settle)
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = deque()
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = types.SimpleNamespace(max_running_requests=6)
        self.weg2_dormant = False
        self.enable_hicache_storage = False
        self.sent = []
        self.group = group
        self.ipc_channels = types.SimpleNamespace(
            send_to_tokenizer=types.SimpleNamespace(send_output=lambda o, r: self.sent.append(o)))

    def _969ad_note_retract(self, req, site):
        pass

    def _add_request_to_queue(self, req, is_retracted=False):
        self.waiting_queue.append(req)

    def _weg2_group_min_flags(self, flags):
        return [1 if v else 0 for v in self.group.reduce(self.rank, [1 if f else 0 for f in flags], "group_min_flags")]

    def uniform_min_avail(self):
        return 0


def _run_ranks(n, fn):
    """Run ``fn(rank)`` on n threads; the first exception of any rank is re-raised, the results returned in rank order."""
    out, errs = [None] * n, []

    def body(r):
        try:
            out[r] = fn(r)
        except BaseException as exc:  # noqa: BLE001
            errs.append((r, exc))

    ts = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    if errs:
        errs.sort(key=lambda e: (not isinstance(e[1], LawViolation), e[0]))
        raise errs[0][1]
    return out


def _d_modules():
    try:
        from sglang.srt.weg2 import d_park_runtime as rt
        from sglang.srt.weg2 import d_seats as ds
        from sglang.srt.weg2 import park_hold_yield as hy
    except ImportError:
        pytest.skip("n/a: this tree has no D park runtime")
    return rt, ds, hy


@pytest.fixture
def d_group(monkeypatch):
    """Group D (the launcher's SGLANG_WEG2_GROUP), the rank-local clocks in place of ``d_park_runtime.time``."""
    rt, ds, hy = _d_modules()
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    rtime = _RankTime()
    monkeypatch.setattr(rt, "time", rtime)
    return SimpleNamespace(rt=rt, ds=ds, hy=hy, rtime=rtime, clocks=[_RankClock() for _ in range(3)])


def _enter(g, rank):
    g.rtime.tl.clock = g.clocks[rank]


def s8_slow_park(g, monkeypatch, durations=PARK_S_RANKS):
    """epoch 90 on a TP3 D group: nothing running, one settle request (weg2-88-1053, its read short), two queued behind
    (weg2-88-1054, weg2-89-1056). The park's phases after the retraction take ``durations[rank]`` s on that rank's own
    monotonic clock, split like the metal's ``sub=...yield:21732,l3mark:9334``: ``HY_SETTLE_S`` in ``hy.settle`` (the
    yield phase) and the rest in ``park_l3.mark_parked`` (#248 PARK-MARK), each a wrapper around the REAL function.
    A stamp taken anywhere before the answer is therefore stale by the time ``park_running`` returns -- not only one
    taken at the retraction. Returns (scheds, answered[rank]); then the passes after the answer."""
    from sglang.srt.managers.io_struct import Weg2ParkRunningReqInput

    rt, ds, hy = g.rt, g.ds, g.hy
    group = _GroupCollective(3)
    real_settle = hy.settle
    scheds = [_DSched(r, group, waiting=[_dreq("weg2-88-1054", 2), _dreq("weg2-89-1056", 3)],
                      settle=[_dreq("weg2-88-1053", 1)]) for r in range(3)]
    try:
        from sglang.srt.weg2 import park_l3 as l3
        real_mark = l3.mark_parked
    except (ImportError, AttributeError):
        l3 = real_mark = None
    ran = {"settle": [0] * 3, "mark": [0] * 3}

    def slow(sched, *, retracted, parked):
        share = min(HY_SETTLE_S, durations[sched.rank]) if l3 is not None else durations[sched.rank]
        g.clocks[sched.rank].t += share                       # the yield phase: HY settle's stat walk under the storm
        ran["settle"][sched.rank] += 1
        return real_settle(sched, retracted=retracted, parked=parked)

    monkeypatch.setattr(hy, "settle", slow)
    if l3 is not None:
        def slow_mark(sched, parked, *a, **k):
            g.clocks[sched.rank].t += max(0.0, durations[sched.rank] - HY_SETTLE_S)   # #248 PARK-MARK (l3mark)
            ran["mark"][sched.rank] += 1
            return real_mark(sched, parked, *a, **k)

        monkeypatch.setattr(l3, "mark_parked", slow_mark)
    answered = [None] * 3

    def park(r):
        _enter(g, r)
        out = rt.park_running(scheds[r], Weg2ParkRunningReqInput(epoch=90, reason="immediate-over-x"),
                              late_hold_armed=True)
        if not out.success:
            raise RigError(f"park_running refused on this tree: {out.message}")
        answered[r] = g.clocks[r].t
        return out

    _run_ranks(3, park)
    if ran["settle"] != [1, 1, 1] or (l3 is not None and ran["mark"] != [1, 1, 1]):
        raise RigError(f"RIG-FEHLER: the park's phases did not run once on every rank (settle {ran['settle']}, "
                       f"mark {ran['mark']}) -- the rig would measure nothing")
    return scheds, answered


def _tick_all(g, scheds, at):
    """One pass at the common instant ``at`` (a pass starts when the slowest rank is through; ranks stand in the
    group-MIN collective of ``park_tick``): returns the number of requests each rank re-queued."""
    for c in g.clocks:
        c.t = at

    def tick(r):
        _enter(g, r)
        return g.rt.park_tick(scheds[r])

    return _run_ranks(3, tick)


class Guards8:
    @staticmethod
    def park_not_undone(g, scheds, answered, bound, what):
        """G8 + G1 over the passes after the answer: every pass is a common instant; a parked request leaves the park
        only once ``bound`` s of the FRONT's time have run since the last rank answered; all ranks agree on it; and the
        safety net (the sleep never came) still re-queues it right after the bound."""
        t_ans = max(answered)
        parked0 = sorted(str(r.rid) for r in scheds[0].weg2_d_parked)
        want = ["weg2-88-1053", "weg2-88-1054", "weg2-89-1056"]
        if parked0 != want:
            raise RigError(f"{what}: the park holds {parked0}, expected {want}")
        # the rank-local stamps differ by up to ``spread`` s (each rank's own clock stamped at its own answer): inside
        # the window [t_ans + bound - spread, t_ans + bound) SOME ranks' local verdict is "due" and some are not. Only
        # the group MIN (``_weg2_group_min_flags``) makes every rank act alike -- a rank-local verdict re-queues on
        # some ranks alone (G1 RAENGE-NIE-UNEINS, R2). The tick inside the window is where that is measured.
        spread = max(answered) - min(answered)
        tick_in_window = t_ans + bound - min(0.2, spread / 2.0)
        for label, at, may_requeue in (
                ("the pass right after the answer", t_ans, False),
                (f"{bound - 1.0:.0f} s after the answer", t_ans + bound - 1.0, False),
                (f"{bound - (t_ans + bound - tick_in_window):.1f} s after the answer (inside the ranks' disagreement "
                 f"window: some rank-local stamps are already {bound:.0f} s old, others not)", tick_in_window, False),
                (f"{bound + 1.0:.0f} s after the answer (the safety net)", t_ans + bound + 1.0, True)):
            moved = _tick_all(g, scheds, at)
            if len(set(moved)) != 1:
                raise LawViolation("G1 RAENGE-NIE-UNEINS",
                                   f"{what}: {label}: the D ranks re-queued {moved} requests (the group MIN verdict "
                                   "did not decide for every rank alike)")
            if may_requeue:
                if moved[0] == 0:
                    raise LawViolation("G8 PARK-SELF-UNDO",
                                       f"{what}: the park was never re-queued {label} -- the safety net is gone")
            elif moved[0]:
                raise LawViolation(
                    "G8 PARK-SELF-UNDO",
                    f"{what}: {label}: every D rank re-queued the parked requests ({moved[0]} each) although the "
                    f"answer had been out for {at - t_ans:.1f} s of a {bound:.0f} s bound (the park itself took "
                    f"{', '.join(f'{d:.1f}' for d in PARK_S_RANKS)} s): D decodes them again while the front, told "
                    "they are parked, flips -- W3 Weg2DrainWitnessUnreachable (epoch 90)")


def test_S8_slow_d_park_does_not_undo_itself(d_group, monkeypatch):
    """A 31 s park on a TP3 D group: neither the very next pass nor the pass 29 s later may re-queue it."""
    scheds, answered = s8_slow_park(d_group, monkeypatch)
    bound = d_group.ds.awake_requeue_s()
    Guards8.park_not_undone(d_group, scheds, answered, bound, "epoch 90 (31 s park under the swap storm)")


class _CountingTree:
    """The UnifiedRadixCache surface ``park_hold_yield.settle`` touches on one rank; the collectives go through the
    group, every one is counted by label."""

    def __init__(self, rank, group, refused=None):
        self.rank, self.group = rank, group
        self._weg2_park_track = None
        self.refused = refused or {}
        self.votes = []

    def _weg2_arena_pools(self):
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        return {ComponentType.FULL: types.SimpleNamespace(arena=None)}

    def _all_reduce_attn_groups(self, t, op, label=""):
        self.votes.append(label)
        t.copy_(torch.tensor(self.group.reduce(self.rank, t.tolist(), label), dtype=t.dtype))

    def _barrier_attn_groups(self, label=""):
        self.votes.append("barrier:" + label)


def s8b_park_walks(g, monkeypatch, refusing_rank=None):
    """Per rank: ``hy.settle`` over the park of epoch 90 (three held requests, nothing retracted). The L3 stat walk
    (``held_facts``, the leaf that cost 21.7 s) is counted and advances the rank clock by 7.2 s per held request.
    ``refusing_rank``: that rank's backups were refused by the arena (need 13 > the 12 on-disk pages (3 spans x 4): the refusal
    stands, but every collective of the verdict is entered). Returns (walks per rank, votes per rank)."""
    hy = g.hy
    group = _GroupCollective(3)
    walks = [[] for _ in range(3)]
    trees = [_CountingTree(r, group) for r in range(3)]

    def facts(tree, rid):
        walks[tree.rank].append(rid)
        g.clocks[tree.rank].t += 7.2
        return 4, 4

    monkeypatch.setattr(hy, "held_facts", facts)
    monkeypatch.setattr(hy, "need_pages", lambda tree, refused: 13 if refused else 0)

    def settle(r):
        _enter(g, r)
        tree = trees[r]
        hy.begin(tree)
        tree._weg2_park_track = {1: object()} if r == refusing_rank else {}
        parked = [types.SimpleNamespace(rid=x) for x in ("weg2-88-1053", "weg2-88-1054", "weg2-89-1056")]
        return hy.settle(types.SimpleNamespace(tree_cache=tree), retracted=[], parked=parked)

    verdicts = _run_ranks(3, settle)
    return walks, [t.votes for t in trees], verdicts


def test_S8b_clean_park_walks_no_l3(d_group, monkeypatch):
    """Nothing refused anywhere: no rank stats a single held page (G9), and all ranks vote alike (G1)."""
    walks, votes, verdicts = s8b_park_walks(d_group, monkeypatch)
    if any(walks):
        raise LawViolation(
            "G9 PARK-WALK",
            f"a park that retracted and refused nothing walked L3 for {[len(w) for w in walks]} held span(s) per rank "
            f"({7.2 * max(len(w) for w in walks):.1f} s on the metal's epoch 90: yield:21732 of a 31.3 s park); "
            f"verdicts {verdicts}")
    if any(v != votes[0] for v in votes):
        raise LawViolation("G1 RAENGE-NIE-UNEINS", f"the ranks voted different collectives: {votes}")


def test_S8c_one_refusing_rank_makes_every_rank_walk_and_vote(d_group, monkeypatch):
    """One rank's backups refused: the first vote's MIN is the same everywhere, so EVERY rank walks its held spans
    and enters the same further collectives (a rank skipping a vote its peers enter hangs the group)."""
    walks, votes, verdicts = s8b_park_walks(d_group, monkeypatch, refusing_rank=1)
    if any(len(w) != 3 for w in walks):
        raise LawViolation("G1 RAENGE-NIE-UNEINS",
                           f"rank 1 refused, the walks per rank were {[len(w) for w in walks]} (all must walk 3)")
    if any(v != votes[0] for v in votes):
        raise LawViolation("G1 RAENGE-NIE-UNEINS", f"the ranks voted different collectives: {votes}")
    if len(set(verdicts)) != 1:
        raise LawViolation("G1 RAENGE-NIE-UNEINS", f"the ranks reached different verdicts: {verdicts}")

#: switches the S7/S8 scenarios or their self-checks set (a separate fixture, so the tree's defaults decide in every test)
RIG_ENVS_S7S8 = (
    # S7 (h64): the PF Frist counted in passes, the satisfied settle's ack, the ack drain, the group floor
    "SGLANG_WEG2_TOLD_FALLBACK_MIN_PASSES", "SGLANG_WEG2_PF_SETTLE_ACK_NOW", "SGLANG_WEG2_TOLD_ACK_DRAIN",
    "SGLANG_WEG2_TOLD_GROUP_FLOOR", "SGLANG_WEG2_TOLD_FALLBACK_GRACE_S", "SGLANG_WEG2_FOLLOWER_EARLY_READ",
    # S8 (flipstall): the awake re-queue bound of a flip park and the park hold yield switch
    "SGLANG_WEG2_D_PARK_AWAKE_REQUEUE_S", "SGLANG_WEG2_ENABLE_PARK_HOLD_YIELD")


@pytest.fixture(autouse=True)
def _clean_env_s7s8(monkeypatch):
    for k in RIG_ENVS_S7S8:
        monkeypatch.delenv(k, raising=False)


def test_selfcheck_S7_wall_frist_and_late_settle_are_the_h64_fallback_a(monkeypatch):
    """SELF-CHECK case A: the pass floor of the Frist and the settle's immediate ack both off = the h16n code."""
    monkeypatch.setenv("SGLANG_WEG2_TOLD_FALLBACK_MIN_PASSES", "0")
    monkeypatch.setenv("SGLANG_WEG2_PF_SETTLE_ACK_NOW", "0")
    with pytest.raises(LawViolation, match="G7"):
        test_S7_pf_frist_against_pass_cadence(monkeypatch)


def test_selfcheck_S7b_wall_frist_without_the_pass_floor_is_the_h64_fallback_b(monkeypatch):
    """SELF-CHECK case B: the Frist counts wall time alone (pass floor off). ONLY the pass floor is turned back here:
    the ACK-DRAIN switch is NOT part of this self-check on purpose -- in S7b the pass floor keeps the rid open and PP0
    keeps harvesting, so ``TOLD_ACK_DRAIN=0`` alone changes nothing (mutant R4 survives; ACK-DRAIN is an uncovered
    fix, see the report 'Grenzen')."""
    monkeypatch.setenv("SGLANG_WEG2_TOLD_FALLBACK_MIN_PASSES", "0")
    with pytest.raises(LawViolation, match="G7"):
        test_S7b_pf_frist_two_hop_ack_in_long_passes(monkeypatch)


def test_selfcheck_S8_stamp_before_the_phases_is_the_epoch90_death(d_group, monkeypatch):
    """SELF-CHECK: the awake re-queue clock stamped at the START of the park (before its phases) instead of at its
    answer -- the h16n code. Where the tree has ``park_answer_stamp`` it is replaced by that; on a tree without it
    the scenario is red anyway and the self-check passes trivially."""
    rt, ds = d_group.rt, d_group.ds
    t_start = d_group.clocks[0].t
    if hasattr(rt, "park_answer_stamp"):
        def early(sched, flip_list, *, late_hold):
            for req in flip_list:
                setattr(req, ds.SINCE_ATTR, t_start)
            setattr(sched, rt.LATE_HOLD_ATTR, t_start if late_hold else None)
            return float(d_group.clocks[sched.rank].t)

        monkeypatch.setattr(rt, "park_answer_stamp", early)
    with pytest.raises(LawViolation, match="G8"):
        test_S8_slow_d_park_does_not_undo_itself(d_group, monkeypatch)


def test_selfcheck_S8_rank_local_verdict_is_the_g1_divergence(d_group, monkeypatch):
    """SELF-CHECK (the runner's mutant R2 as a monkeypatch): ``park_tick`` decides from the RANK-LOCAL clock verdict, no
    group MIN (``_weg2_group_min_flags`` returns what it was given). The rank-local stamps differ by 0.4 s, so the tick
    inside that window re-queues on some ranks alone: G1. On a tree without ``park_answer_stamp`` the scenario is
    G8-red already and the self-check accepts either law."""
    monkeypatch.setattr(_DSched, "_weg2_group_min_flags", lambda self, flags: [1 if f else 0 for f in flags])
    law = "G1" if hasattr(d_group.rt, "park_answer_stamp") else "G1|G8"
    with pytest.raises(LawViolation, match=law):
        test_S8_slow_d_park_does_not_undo_itself(d_group, monkeypatch)


def test_selfcheck_S8_stamp_after_the_yield_phase_is_still_the_epoch90_death(d_group, monkeypatch):
    """SELF-CHECK (the runner's mutant R3): the awake clock stamped right after the HY settle (``_ph("yield")``) instead of
    at the answer -- before #248 PARK-MARK (9.3 s on the metal), #59b and the clamp. The partial reversal of the
    flipstall fix A: a rig that stalls only the HY phase would stay green on it."""
    rt, ds, hy = d_group.rt, d_group.ds, d_group.hy
    if not hasattr(rt, "park_answer_stamp"):
        pytest.skip("n/a: this tree has no park_answer_stamp (the scenario is red on it already)")
    real_settle = hy.settle

    def stamp_after_yield(sched, *, retracted, parked):
        out = real_settle(sched, retracted=retracted, parked=parked)
        now = d_group.clocks[sched.rank].t
        for req in sched.weg2_d_parked:
            setattr(req, ds.SINCE_ATTR, now)
        setattr(sched, rt.LATE_HOLD_ATTR, now)
        return out

    monkeypatch.setattr(hy, "settle", stamp_after_yield)
    monkeypatch.setattr(rt, "park_answer_stamp", lambda sched, flip_list, *, late_hold: float(d_group.clocks[sched.rank].t))
    with pytest.raises(LawViolation, match="G8"):
        test_S8_slow_d_park_does_not_undo_itself(d_group, monkeypatch)


def test_selfcheck_S8b_walk_before_the_vote_is_the_epoch90_yield(d_group, monkeypatch):
    """SELF-CHECK: the L3 stat walk gathered BEFORE the first vote on every park (the h16n ``settle``)."""
    hy = d_group.hy
    real = hy.settle

    def walk_first(sched, *, retracted, parked):
        for r in parked:
            hy.held_facts(sched.tree_cache, str(r.rid))
        return real(sched, retracted=retracted, parked=parked)

    monkeypatch.setattr(hy, "settle", walk_first)
    with pytest.raises(LawViolation, match="G9"):
        test_S8b_clean_park_walks_no_l3(d_group, monkeypatch)


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


# =====================================================================================================================
# S9 -- the shared arena full and pinned by the peers, a 210k load-back executed on a follower (NF h16n 07:30:35)
# =====================================================================================================================
#
# Metal (boot ..._17c2563647_1015_071520.P.log, PP1): the arena (6485 slots) complete and referenced by every stage
# (P census 07:29:57 complete=6294 pinned=6292 refs=18876); a 3-page backup refused -> W3-ARENA SPILL released node
# 208's host copy (3232 pages, ref_free=0/3232: PP0/PP2 still held them), node 208 stays on the device l3_present;
# the peel demotes its tail, whose store write host-locks it; '#1421 BACKUP-REFUSED why=arena_claim node=208',
# 'UD-REFUSED why=host_children:child_host_locked node=208 holder=211' -> 'LB-VERDICT ROW-ROOM SHORT kv=209472
# avail=4096 evictable=255040 evicted=48192' -> group stop. REAL here: the C arena (gcc, the claim of the refused
# backup, the peers' references), UnifiedRadixCache._w3_arena_spill / _evict_device_leaf / _ud_clear_host_children /
# _is_device_leaf / _is_host_leaf (and whatever this tree's eviction calls by name, e.g. L3-DROP). MODELLED: the
# stage tree's nodes and I/O (store write ack queue, the demotion), the peel's leaf order (smallest first).

S9_NODE208, S9_TAIL, S9_OTHER, S9_FREE, S9_NEED = 206848, 231, 48192, 4096, 209472
S9_PAGE = 64
S9_SLOTS = 96


class _CD9:
    def __init__(self, value=None, host_value=None):
        self.value = value
        self.host_value = host_value
        self.lock_ref = 0
        self.host_lock_ref = 0


class _Node9:
    _ids = itertools.count(200)

    def __init__(self, parent, tokens, *, host=False, label=""):
        self.id = next(_Node9._ids)
        self.label = label
        self.parent = parent
        self.children = {}
        self.l3_present = False
        self.key = list(range(tokens))
        self.slots = []
        self.component_data = [_CD9(torch.arange(tokens, dtype=torch.int64) if tokens else None,
                                    torch.arange(tokens, dtype=torch.int64) if (tokens and host) else None),
                               _CD9(), _CD9()]
        if parent is not None:
            parent.children[self.id] = self

    @property
    def evicted(self):
        return self.parent is not None and self.component_data[0].value is None

    @property
    def backuped(self):
        return self.component_data[0].host_value is not None


class _SpillPool9:
    """The anchor pool the spill talks to: every page it is asked about is on disk already (the write-behind
    secured it -- 'l3=on_disk:3232,written:0'); no header read (node-id order, the tree's fallback)."""

    staging_rows = 0

    def __init__(self):
        self.arena = object()

    def secure_rows_to_l3(self, host_indices):
        n = int(host_indices.numel())
        return {"pages": n, "on_disk": n, "written": 0, "lost": 0}


def _s9_stage_cls():
    URC = _urc_mod()
    C = URC.UnifiedRadixCache
    names = ("_evict_device_leaf", "_is_device_leaf", "_is_host_leaf", "_ud_clear_host_children", "_w3_arena_spill")
    missing = [n for n in names if not hasattr(C, n)]
    if missing:
        raise RigError(f"UnifiedRadixCache lacks {missing}")
    return type("_Stage9", (_Stage9Base,), {n: getattr(C, n) for n in names})


class _Stage9Base:
    """One follower P stage: its own tree (tp group of one under pp 3 -> the local-PP floor) on the shared arena."""

    W3_SPILL_MIN_PAGES = 64
    W3_SPILL_EMPTY_BACKOFF_S = 0.5

    def __init__(self, arena):
        from sglang.srt.weg2 import pp_slot_fidelity as SF

        self.arena = arena
        self.root_node = _Node9(None, 0, label="root")
        self.root_node.component_data[0].value = torch.empty(0, dtype=torch.int64)
        self.cache_controller = SimpleNamespace(write_policy="write_back")
        self.ongoing_write_through = {}
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self.ack_queue = []
        self.tree_components = (_urc_mod().BASE_COMPONENT_TYPE,)
        base = SimpleNamespace(component_type=_urc_mod().BASE_COMPONENT_TYPE)
        self.components = {_urc_mod().BASE_COMPONENT_TYPE: base}
        self._components_tuple = (base,)
        self.evictable_host_leaves = set()
        self.evictable_device_leaves = set()
        self.page_size = 1
        self.backup_claims = []
        SF.mark_floor_scope(self, True)

    # ---- the tree surface the real methods touch ------------------------------------------------------------
    def _collect_all_nodes(self):
        out, stack = [], [self.root_node]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def _w3_spill_keys(self, pool, nodes):
        keys = {id(n): (1, 0, 0, n.id) for n in nodes}

        class _K(dict):
            def slots_of(self, node):
                return list(node.slots)

            def incomplete(self, node):
                # L3-COMPLETE (1995104e53): the rig stage's arena pages are all written (state 2);
                # an unfinished node would be no spill candidate, the rig has none.
                return False

        return _K(keys)

    def _w3_spill_ref_free(self, pool, slots):
        refs = self.arena.slot_refs(sorted(set(slots))) if slots else []
        return f"{sum(1 for r in refs if r == 0)}/{len(refs)}"

    def _evict_component_and_detach_lru(self, node, comp, target=None, tracker=None):
        """The spill's host release (#1407 transit, KV layer): this stage's references go back to the arena."""
        hv = node.component_data[0].host_value
        if node.slots:
            self.arena.ref_slots(node.slots, -1)
        node.component_data[0].host_value = None
        return 0, 0 if hv is None else int(hv.numel())

    def write_backup(self, node, write_back=False, kv_only_if_mamba_refused=False):
        """The backup IS an arena claim (_weg2_direct_claim -> alloc_write): claimed for real on the shared arena."""
        pages = max(1, len(node.key) // S9_PAGE)
        stems = [f"s9-backup-{node.id}-{i}" for i in range(pages)]
        got = self.arena.claim_slots(stems, [1024] * pages)
        self.backup_claims.append([st for _s, st, _g in got])
        fresh = [(s, g) for s, st, g in got if st == 0]
        if fresh:
            self.arena.release_claims([s for s, _ in fresh], [g for _, g in fresh])
        return 0 if any(st == 4 for _s, st, _g in got) else len(node.key)

    def writing_check(self, write_back=False):
        return None

    def _evict_to_host(self, node, tracker):
        """GPU -> CPU demotion of a backed leaf (the node stays, host-only)."""
        tracker[_urc_mod().BASE_COMPONENT_TYPE] += len(node.component_data[0].value)
        node.component_data[0].value = None

    def _evict_host_leaf(self, node, tracker):
        assert self._is_host_leaf(node)
        if node.slots:
            self.arena.ref_slots(node.slots, -1)
        node.parent.children.pop(node.id)

    def _ud_drop_unbacked_leaf(self, node, tracker):
        assert not node.children, "#841: an un-backed node with children is never dropped"
        tracker[_urc_mod().BASE_COMPONENT_TYPE] += len(node.component_data[0].value)
        node.parent.children.pop(node.id)

    def _update_evictable_leaf_sets(self, node):
        pass

    def drain_backup_acks_uncoordinated(self):
        n = 0
        while self.ack_queue:
            op = self.ack_queue.pop(0)
            node, _p = self.ongoing_backup.pop(op)
            node.l3_present = True
            node.component_data[0].host_lock_ref = max(0, node.component_data[0].host_lock_ref - 1)
            n += 1
        return n

    # ---- the scenario's own steps -----------------------------------------------------------------------------
    def store_write(self, node, op):
        """write_backup_storage: the node's host lock until its ack is drained; the backup thread acked it."""
        node.component_data[0].host_lock_ref += 1
        self.ongoing_backup[op] = (node, None)
        self.ack_queue.append(op)

    def peel(self, want):
        """The eviction: device leaves smallest first, until ``want`` rows are paid or a round pays nothing."""
        B = _urc_mod().BASE_COMPONENT_TYPE
        tracker = {B: 0}
        while tracker[B] < want:
            leaves = sorted((n for n in self._collect_all_nodes() if self._is_device_leaf(n)),
                            key=lambda n: (len(n.key), n.id))
            before = tracker[B]
            for n in leaves:
                self._evict_device_leaf(n, tracker)
                if tracker[B] >= want:
                    break
            if tracker[B] == before:
                break
        return tracker[B]


def _s9_pinned_arena(tmp_path):
    """The shared arena: every slot COMPLETE and referenced by the peers (PP0/PP2/D) -- 'complete=6294 pinned=6292'."""
    _need_gcc()
    from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

    path = str(tmp_path / "arena-s9.bin")
    arena = ShmArena(path, 1024, S9_SLOTS)
    peers = ShmArena(path, 1024, S9_SLOTS)
    stems = [f"s9-page-{i}" for i in range(S9_SLOTS)]
    got = arena.claim_slots(stems, [1024] * S9_SLOTS)
    slots = [s for s, _st, _g in got]
    assert arena.complete_slots(slots, [g for _s, _st, g in got], [(0, 1024)] * S9_SLOTS) == [1] * S9_SLOTS
    assert peers.ref_slots(slots, +1) == S9_SLOTS
    return arena, peers, slots


def test_S9_arena_full_pinned_row_room(tmp_path, monkeypatch):
    """Node 208 (206848 tokens) and its tail on the follower's device, both backed into the pinned arena; a small
    backup is refused and the spill takes 208's host copy; the tail's store write is in flight when the peel
    demotes it; PP0's row needs 209472 rows on this stage (4096 free). G5: the peel pays them."""
    monkeypatch.setenv("SGLANG_WEG2_SF", "1")
    arena, peers, slots = _s9_pinned_arena(tmp_path)
    try:
        Stage = _s9_stage_cls()
        st = Stage(arena)
        n208 = _Node9(st.root_node, S9_NODE208, host=True, label="208")
        tail = _Node9(n208, S9_TAIL, host=True, label="211")
        other = _Node9(st.root_node, S9_OTHER, host=True, label="other")
        n208.slots, tail.slots, other.slots = slots[:48], slots[48:50], slots[50:80]
        assert arena.ref_slots(n208.slots + tail.slots + other.slots, +1) == 80   # this stage's tree references
        # 07:30:30: a 192-token backup (weg2-17-57) is refused -- the arena has no free slot
        small = _Node9(st.root_node, 192, label="215")
        if st.write_backup(small) != 0:
            raise RigError("the pinned arena granted a claim -- the scenario's premise is gone")
        st.root_node.children.pop(small.id)
        try:
            st._w3_arena_spill(_SpillPool9(), 3, claimer=None)
        except Exception as exc:  # noqa: BLE001
            raise RigError(f"_w3_arena_spill cannot run on the rig stage: {exc!r}")
        if not n208.backuped and arena.slot_refs(n208.slots)[0] != 1:
            raise RigError("the spill freed 208's slots although the peers reference them")
        # the tail's store write (written back after the spill) is in flight when the peel comes
        st.store_write(tail, op=77)
        # PP0's row: load-back of 209472 rows on this stage, 4096 free -> the peel must pay the rest
        need = S9_NEED - S9_FREE
        try:
            paid = st.peel(need)
        except LawViolation:
            raise
        except Exception as exc:  # noqa: BLE001 - the tree stopped by name
            raise _group_stop(exc) from exc
        Guards.row_room(paid, need, "follower PP1 executing weg2-16-53's row (kv=209472, avail=4096)")
    finally:
        arena.close()
        peers.close()


def test_selfcheck_S9_l3_drop_off_is_row_room_short(tmp_path, monkeypatch):
    """With the L3-DROP switch off the store-backed node takes the claim-first path again -> G5 red."""
    from sglang.srt.mem_cache import unified_radix_cache as URC

    if not hasattr(URC.UnifiedRadixCache, "_weg2_l3_drop"):
        pytest.skip("n/a: this tree has no L3-DROP to switch off")
    monkeypatch.setenv("SGLANG_WEG2_L3_DROP", "0")
    with pytest.raises(LawViolation, match="G5"):
        test_S9_arena_full_pinned_row_room(tmp_path, monkeypatch)


# =====================================================================================================================
# S10 -- K14: a client abort while PP0's row frame names the rid (NF h16p death 15.10. 21:52:38Z, weg2-164-2265)
# =====================================================================================================================
# PP0 admitted weg2-164-2265 into slot 2 (#969N ADMIT fwd_ct=346) and launched it; the front's H102 client-gone abort
# reached PP0 one pass later, PP0 forwarded it (#1460 CTRL-FWD) and set to_finish on its in-flight request. The chain
# delivered the AbortReq to PP1/PP2 BEFORE their slot-2 frame (two wires, no cross-wire order), they still queued the
# rid and popped it at receipt; the frame then named a rid they no longer had: #631 ROW-PROBE DEFER x3 ->
# PpRowDeferCapExceeded (#1180) -> RANK_EXCEPTION. Real code: ``Scheduler.abort_request`` / ``_abort_request_now`` /
# ``process_pending_chunked_abort``, the probe ``SchedulerPPMixin._pp_proxy_frame_pending``, the PP0 stamp the tree's
# ``_pp_forward_and_process_input_requests`` calls before its send (looked up by name in that source), and the
# after-plan settle the tree's ``_event_loop_pp_body`` runs after ``_pp_void_pass.after_plan`` (looked up the same way).
# Modelled: the plan's admission (the rid leaves the waiting queue into the slot's microbatch), no forward.
#   G1  the slot-2 pass executes the same rids on every stage;
#   G10 ABORT-UNIFORM: every stage that ran the pass carries PP0's abort on the rid (to_finish / chunked abort), and
#       after the settle no stage keeps the rid queued or held (no live leftover = no KV/slot leak, no second release).

S10_RID, S10_OTHER, S10_SLOT, S10_EPOCH = "weg2-164-2265", "weg2-164-2266", 2, 165


class _Req10(SimpleNamespace):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _req10(rid=S10_RID):
    return _Req10(rid=rid, to_finish=None, finished_reason=None, mamba_pool_idx=None, req_pool_idx=0,
                  kv_committed_freed=True, finished=lambda: False,
                  time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=lambda **k: None)))


def _s10_modules():
    from sglang.srt.managers import scheduler as SM
    from sglang.srt.managers import scheduler_pp_mixin as PPM
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.weg2 import pp_abort as PA

    for name in ("_pp_proxy_frame_pending", "_pp_forward_and_process_input_requests", "_event_loop_pp_body"):
        if not hasattr(PPM.SchedulerPPMixin, name):
            raise RigError(f"SchedulerPPMixin.{name} is gone -- S10 cannot drive this tree")
    return SM, PPM, AbortReq, PA


def _s10_wired():
    """(stamp_fn or None, settle_after_plan) as THIS tree wires them."""
    SM, PPM, _, PA = _s10_modules()
    fwd = inspect.getsource(PPM.SchedulerPPMixin._pp_forward_and_process_input_requests)
    send = "self.send_req_work = self._pp_send_pyobj_to_next_stage("
    if send not in fwd:
        raise RigError("the request-chain send is gone from _pp_forward_and_process_input_requests")
    stamp = None
    for name in [n for n in dir(PA) if "stamp" in n and "abort" in n and callable(getattr(PA, n))]:
        if name in fwd and fwd.index(name) < fwd.index(send):
            stamp = getattr(PA, name)
    body = inspect.getsource(PPM.SchedulerPPMixin._event_loop_pp_body)
    anchor = "_pp_void_pass.after_plan(self, mb_id)"
    settle = False
    if anchor in body:
        tail = body[body.index(anchor):body.index(anchor) + 1500]
        settle = "_weg2_resolve_waiting_aborts" in tail and hasattr(SM.Scheduler, "_weg2_resolve_waiting_aborts")
    return stamp, settle


def _s10_stage(monkeypatch, pp_rank):
    SM, PPM, _, _ = _s10_modules()
    monkeypatch.setattr(SM, "prepare_abort", lambda req, why: setattr(req, "aborted_why", why))
    monkeypatch.setattr(SM, "release_kv_cache", lambda *a, **k: None)
    S = SM.Scheduler
    sent = []
    st = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=pp_rank, pp_size=WORLD, tp_size=1),
        chunked_req=None, _pending_chunked_abort_req=None, _pending_chunked_abort_delay=0,
        _pp_scheduled_extents=lambda: None, waiting_queue=[], running_mbs=[None] * WORLD, mbs=[None] * WORLD,
        running_batch=None, last_batch=None, kv_session_offload=None, disaggregation_mode=None,
        enable_hicache_storage=False, grammar_manager=SimpleNamespace(abort_requests=lambda r: None),
        _weg2_abort_dormant_hold=lambda r: 0, _weg2_d_park_abort=lambda r: 0,
        tree_cache=SimpleNamespace(supports_mamba=lambda: False),
        ipc_channels=SimpleNamespace(send_to_tokenizer=SimpleNamespace(send_output=lambda o, r: sent.append(o.rid))),
        pp_group=object(), _pp_row_chain_owed=False, _pp_flip_epoch=lambda: S10_EPOCH,
        _weg2_store_told_armed=False, _weg2_store_told={}, _weg2_store_held={}, sent=sent,
    )
    st._abort_request_now = lambda r: S._abort_request_now(st, r)
    if hasattr(S, "_weg2_resolve_waiting_aborts"):
        st._weg2_resolve_waiting_aborts = lambda pp0_drained=False: S._weg2_resolve_waiting_aborts(
            st, pp0_drained=pp0_drained)
    st.process_pending_chunked_abort = lambda: S.process_pending_chunked_abort(st)
    return st


def _s10_frame(rid):
    _, PPM, _, _ = _s10_modules()
    row = (rid, 0, 1, True, False, None, None, False, None, (), None)
    return {"__stamp__": (S10_SLOT, 347, 265, S10_EPOCH, 347, (rid, 60736, 61001)),
            PPM._ADMISSION_DECISION_PAYLOAD_KEY: (S10_SLOT, (row,))}


def _s10_probe(monkeypatch, st, inbox):
    _, PPM, _, _ = _s10_modules()
    monkeypatch.setattr(PPM, "resolve_src", lambda group, x: 0)
    monkeypatch.setattr(PPM, "typed_inbox", lambda group: {(0, "proxy"): inbox})
    return PPM.SchedulerPPMixin._pp_proxy_frame_pending(st, S10_SLOT)


def s10_client_abort_during_row_frame(monkeypatch, *, pp0_admitted=True, chunked=False):
    """One P group, the 21:52:38 order. Raises LawViolation (red) or returns the stages (green)."""
    SM, PPM, AbortReq, PA = _s10_modules()
    from sglang.srt.weg2 import p_row_authority

    if not p_row_authority.applies(SimpleNamespace(ps=SimpleNamespace(pp_size=WORLD, pp_rank=1))):
        pytest.skip("n/a: the NF form of this tree runs P without the row authority (no row frame)")
    S = SM.Scheduler
    stamp, settle = _s10_wired()
    pp0 = _s10_stage(monkeypatch, 0)
    r0 = _req10()
    if pp0_admitted:
        pp0.mbs[S10_SLOT] = SimpleNamespace(reqs=[r0])           # #969N ADMIT slot=2, launched
        pp0._pp_launched_batches = {S10_SLOT: pp0.mbs[S10_SLOT]}   # #1225: launched, not delivered
        if chunked:
            pp0.chunked_req = r0
    else:
        # still queued on PP0: no frame names it; slot 2 holds a COMPLETED batch of it (delivered, requeued
        # everywhere -- #968b / void park / retreat): mbs and last_batch keep it, _pp_launched_batches does not
        done = SimpleNamespace(reqs=[_req10()])
        pp0.mbs[S10_SLOT], pp0.last_batch, pp0._pp_launched_batches = done, done, {}
        pp0.waiting_queue = [r0]
    fol = [_s10_stage(monkeypatch, r) for r in range(1, WORLD)]
    reqs = [_req10() for _ in fol]
    for st, r in zip(fol, reqs):
        st.waiting_queue = [r]                                   # chain hop landed, frame not yet consumed
    wire = [AbortReq(rid=S10_RID)]                               # H102 client-gone
    if stamp is not None:
        wire = stamp(pp0, wire)                                  # #1460 CTRL-FWD (the tree's stamp, if wired)
    try:
        S.abort_request(pp0, wire[0])
        for st in fol:
            S.abort_request(st, wire[0])                         # chain before the slot-2 frame
    except Exception as exc:  # noqa: BLE001
        raise _group_stop(exc) from exc
    frame_rid = S10_RID if pp0_admitted else S10_OTHER
    executed = [[S10_RID] if pp0_admitted else []]
    for st, r in zip(fol, reqs):
        if not pp0_admitted:
            st.waiting_queue.append(_req10(S10_OTHER))
        inbox = [_s10_frame(frame_rid)]
        try:
            ok = False
            for _ in range(8):                                   # past the #1180 lap cap
                if _s10_probe(monkeypatch, st, inbox):
                    ok = True
                    break
        except Exception as exc:  # noqa: BLE001 -- PpRowDeferCapExceeded: the 21:52:38 group stop
            raise _group_stop(exc) from exc
        if not ok:
            raise LawViolation("G1 RAENGE-NIE-UNEINS", f"PP{st.ps.pp_rank} never executes the slot-2 frame")
        # the plan executes PP0's row: the named rid leaves the waiting queue into the slot's microbatch
        got = [q for q in st.waiting_queue if q.rid == frame_rid]
        st.waiting_queue = [q for q in st.waiting_queue if q.rid != frame_rid]
        st.mbs[S10_SLOT] = SimpleNamespace(reqs=got)
        if chunked and got:
            st.chunked_req = got[0]
        if settle:
            st._weg2_resolve_waiting_aborts()                    # the tree's after-plan settle
        executed.append([q.rid for q in got] if pp0_admitted else [])
    Guards.same_admission(f"slot {S10_SLOT} (rid {S10_RID})", [list(e) for e in executed],
                          what="aborted rid in the frame's pass")
    # G10: every stage carries PP0's abort, nothing of the rid is left queued or held
    pp0_abort = r0.to_finish is not None or pp0._pending_chunked_abort_req is r0 or r0 not in pp0.waiting_queue
    bad = []
    for st, r in zip(fol, reqs):
        carries = r.to_finish is not None or st._pending_chunked_abort_req is r
        left = [q.rid for q in st.waiting_queue if q.rid == S10_RID]
        held = dict(getattr(st, "_weg2_pp_held_waiting_aborts", None) or {})
        if pp0_admitted and (not carries or left or held):
            bad.append((f"PP{st.ps.pp_rank}", carries, left, sorted(held)))
        if not pp0_admitted and (left or held):
            bad.append((f"PP{st.ps.pp_rank}", "queued-after-abort", left, sorted(held)))
    if not pp0_abort or bad:
        raise LawViolation("G10 ABORT-UNIFORM",
                           f"PP0 aborted={pp0_abort}; stages without PP0's abort on {S10_RID} "
                           f"(stage, carries, still queued, still held): {bad} -- the follower would compute the "
                           "rid PP0 finished (KV/slot leak) or release it a second time")
    return pp0, fol


def test_S10_client_abort_during_row_frame(monkeypatch):
    """K14 (h16p 21:52:38Z): the abort reaches the followers before the frame that names the rid."""
    s10_client_abort_during_row_frame(monkeypatch)


def test_S10b_client_abort_pp0_had_not_admitted(monkeypatch):
    """Gegenprobe: PP0 still queued the rid -- every stage drops it at receipt, the frame names another rid."""
    s10_client_abort_during_row_frame(monkeypatch, pp0_admitted=False)


def test_S10c_client_abort_during_chunked_row_frame(monkeypatch):
    """The first chunk's frame names the rid: the followers' chunked abort follows PP0's schedule (#791C)."""
    s10_client_abort_during_row_frame(monkeypatch, chunked=True)


def test_selfcheck_S10_rank_local_pop_is_the_1180_death(monkeypatch):
    """The abort applied rank-locally at receipt again -> the 21:52:38 group stop (#1180)."""
    from sglang.srt.weg2 import pp_abort as PA

    if hasattr(PA, "hold_waiting_abort"):
        monkeypatch.setattr(PA, "hold_waiting_abort", lambda s, r: False)
    with pytest.raises(LawViolation, match="GRUPPENSTOPP|G1"):
        test_S10_client_abort_during_row_frame(monkeypatch)


def test_selfcheck_S10_held_abort_never_applied_is_g10(monkeypatch):
    """The held abort never applied after the admitting plan -> the follower computes the rid PP0 aborted (G10)."""
    from sglang.srt.weg2 import pp_abort as PA

    if hasattr(PA, "waiting_abort_verdict"):
        monkeypatch.setattr(PA, "waiting_abort_verdict", lambda still, drained: "keep")
    with pytest.raises(LawViolation, match="GRUPPENSTOPP|G1|G10"):
        test_S10_client_abort_during_row_frame(monkeypatch)


# =====================================================================================================================
# S11 (h16u 1017): the SET scheduler of SCHEDULER-SOLL v2 -- the Set verdict and its log line, cases SOLL §5 1-8 + H1-H9
# =====================================================================================================================
#
# WHAT. ``phase_policy.classify_arrival`` / ``phase_policy.set_verdict`` (SOLL §4: pure, no clock, no I/O) driven by a
# small model of the front's glue (arrival -> class -> set -> tick -> verdict -> one ``WEG2 SET-VERDICT`` line each),
# every line checked against the §4 format by a strict parser, S / X / halfX / the verdict / the wait against the
# SOLL numbers (X = 6000, halfX = 3000, T_halb = 4.5 s).  The danger directions are asserted both ways (R7: no
# FLIP_P under X; R3: no D_AFTER_TIMER at S >= X) over a grid, not at one point.
# WHAT IS REAL: the tree's own ``phase_policy`` functions.  WHAT IS MODELLED: the front glue (a model, not
# front.py); two source anchors bind the REAL front: the format string of the log line and the call edge
# (``classify_arrival`` / ``set_verdict`` called with every keyword of §4) -- so a front that stops passing
# ``d_decoding`` / ``seats_free`` / ``d_prefill_running_uncached`` goes red here, not at the metal.
# A symbol the tree does not have yet is RED (behaviour), never a RIG-FEHLER: the cases are the acceptance of S1.

import re as _re11  # noqa: E402

_SX = 6000
_SHALF = 0.5
_ST = 4.5
SET_LINE_RE = _re11.compile(
    r"^WEG2 SET-VERDICT epoch=(?P<epoch>\d+) set=(?P<set>\S+) n=(?P<n>\d+) members=(?P<members>\S+) "
    r"S=(?P<S>\d+) X=(?P<X>\d+) halfX=(?P<halfX>\d+) classes=(?P<classes>(?:kurz|halb|gross|p_only)(?:,(?:kurz|halb|gross|p_only))*) "
    r"d_decoding=(?P<dd>[01]) seats_free=(?P<seats>-?\d+) t_half_s=(?P<th>\d+(?:\.\d+)?) waited_s=(?P<w>\d+(?:\.\d+)?) "
    r"verdict=(?P<verdict>COLLECT|FLIP_P|D_AFTER_TIMER|PARK_NO_SEAT|REROUTE_P|D_NOW) why=(?P<why>\S.*)$")
SET_KEYS = ("epoch=", "set=", "n=", "members=", "S=", "X=", "halfX=", "classes=", "d_decoding=", "seats_free=",
            "t_half_s=", "waited_s=", "verdict=", "why=")


def _pp11():
    """The tree's phase_policy with the §4 interface -- a missing name is RED by behaviour (S1's job)."""
    from sglang.srt.weg2 import phase_policy as PP

    missing = [n for n in ("SetMember", "SetVerdict", "classify_arrival", "set_verdict") if not hasattr(PP, n)]
    assert not missing, f"phase_policy has no {missing} -- SOLL §4 interface (S1) not in this tree"
    return PP


class _SetFront:
    """The front's glue around the pure policy, as a model: one open set per epoch, a tick re-asks, a terminal
    verdict closes the set; every decision writes exactly one SET-VERDICT line (kurz: D_NOW why=R2)."""

    def __init__(self, *, d_decoding=True, seats_free=4, x=_SX, half_frac=_SHALF, t_half=_ST, epoch=7):
        self.PP = _pp11()
        self.x, self.frac, self.t_half, self.epoch = x, half_frac, t_half, epoch
        self.d_decoding, self.seats_free = d_decoding, seats_free
        self.d_running = 0
        self.members, self.t_open, self.set_no = [], None, 1
        self.lines, self.decisions = [], []
        self.policy_calls = []

    def _emit(self, members, classes, S, half_x, now, verdict, why, waited):
        line = ("WEG2 SET-VERDICT epoch=%d set=%d n=%d members=%s S=%d X=%d halfX=%d classes=%s d_decoding=%d seats_free=%d "
                "t_half_s=%.1f waited_s=%.1f verdict=%s why=%s") % (
            self.epoch, self.set_no, len(members), ",".join("%s:%d" % (m.rid, m.uncached) for m in members), S, self.x,
            half_x, ",".join(classes), int(self.d_decoding), self.seats_free, self.t_half, waited, verdict, why)
        assert SET_LINE_RE.match(line), "rig line is not SOLL §4: " + line
        self.lines.append(line)
        return line

    def arrive(self, t, rid, u, p_only=False):
        cls = self.PP.classify_arrival(u, self.x, self.frac, p_only)
        assert cls in ("kurz", "halb", "gross", "p_only"), cls
        m = self.PP.SetMember(rid=rid, uncached=u, t_arrive=t, p_only=p_only)
        if cls == "kurz":   # R2: never into the set, never waits
            self._emit([m], ["kurz"], u, int(self.frac * self.x), t, "D_NOW", "R2", 0.0)
            self.decisions.append(("D_NOW", u, 0.0))
            return "D_NOW"
        self.members.append(m)
        if self.t_open is None:
            self.t_open = t
        return self._decide(t, 1)

    def tick(self, t):
        return self._decide(t, 0) if self.members else None

    def _decide(self, t, new):
        members = tuple(self.members)
        assert all(self.PP.classify_arrival(m.uncached, self.x, self.frac, m.p_only) != "kurz" for m in members), \
            "a kurz request reached set_verdict (R2 comes before the set)"
        v = self.PP.set_verdict(members, x_tokens=self.x, half_frac=self.frac, t_half_s=self.t_half, now=t,
                                t_open=self.t_open, d_decoding=self.d_decoding, seats_free=self.seats_free,
                                d_prefill_running_uncached=self.d_running, new_since_last=new)
        assert isinstance(v, self.PP.SetVerdict), type(v)
        classes = [self.PP.classify_arrival(m.uncached, self.x, self.frac, m.p_only) for m in members]
        waited = t - self.t_open
        self._emit(list(members), classes, v.s_tokens, v.half_x, t, v.verdict, v.why, waited)
        self.decisions.append((v.verdict, v.s_tokens, round(waited, 3)))
        if v.verdict != "COLLECT":
            self.members, self.t_open, self.set_no = [], None, self.set_no + 1
        return v.verdict

    def parsed(self, i=-1):
        return SET_LINE_RE.match(self.lines[i]).groupdict()


def _check_line(f, i, *, verdict, S, n=None, classes=None, dd=None, waited=None):
    d = f.parsed(i)
    assert d["verdict"] == verdict, (verdict, f.lines[i])
    assert int(d["S"]) == S, (S, f.lines[i])
    assert int(d["X"]) == _SX and int(d["halfX"]) == int(_SHALF * _SX), f.lines[i]
    assert float(d["th"]) == _ST, f.lines[i]
    if n is not None:
        assert int(d["n"]) == n, f.lines[i]
    if classes is not None:
        assert d["classes"] == classes, (classes, f.lines[i])
    if dd is not None:
        assert int(d["dd"]) == dd, f.lines[i]
    if waited is not None:
        assert abs(float(d["w"]) - waited) <= 0.05, (waited, f.lines[i])


# --- SOLL §5 examples 1-8 ------------------------------------------------------------------------------------------


def test_S11_ex1_short_goes_to_d_at_once():
    f = _SetFront()
    assert f.arrive(100.0, "a", 300) == "D_NOW"
    _check_line(f, -1, verdict="D_NOW", S=300, n=1, classes="kurz", waited=0.0)
    assert f.parsed()["why"] == "R2" and f.members == [] and f.t_open is None   # no timer, no set


def test_S11_ex2_half_alone_waits_then_d():
    f = _SetFront()
    assert f.arrive(100.0, "a", 3500) == "COLLECT"
    _check_line(f, -1, verdict="COLLECT", S=3500, n=1, classes="halb", waited=0.0)
    assert f.tick(102.0) == "COLLECT" and f.tick(104.4) == "COLLECT"
    assert f.tick(104.5) == "D_AFTER_TIMER"
    _check_line(f, -1, verdict="D_AFTER_TIMER", S=3500, n=1, classes="halb", waited=4.5)


def test_S11_ex3_two_halves_flip_at_once():
    f = _SetFront()
    f.arrive(100.0, "a", 3500)
    assert f.arrive(102.0, "b", 3200) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=6700, n=2, classes="halb,halb", dd=1, waited=2.0)


def test_S11_ex4_short_between_is_d_the_set_flips():
    """SOLL §5 ex.4 names the third request 2800, but 2800 < halfX (3000) is kurz by §2/R2 -- the example is
    inconsistent with the classification (reported in the S2 report); the rig takes 3000 (= halfX, halb): S = 6500."""
    f = _SetFront()
    assert f.arrive(100.0, "a", 3500) == "COLLECT"
    assert f.arrive(100.5, "k", 300) == "D_NOW"
    _check_line(f, -1, verdict="D_NOW", S=300, classes="kurz", waited=0.0)
    assert f.arrive(101.0, "b", 3000) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=6500, n=2, classes="halb,halb")
    assert "k:300" not in f.lines[-1]           # the short one is not a member of the flipping set


def test_S11_ex4_literal_2800_is_kurz_by_classification():
    """The literal SOLL §5 ex.4 request: 2800 under halfX goes to D at once, the 3500 set does NOT flip (S = 3500)."""
    f = _SetFront()
    f.arrive(100.0, "a", 3500)
    f.arrive(100.5, "k", 300)
    assert f.arrive(101.0, "c", 2800) == "D_NOW"
    assert f.members and f.members[0].rid == "a" and sum(m.uncached for m in f.members) == 3500


def test_S11_ex5_big_waits_for_a_rider():
    f = _SetFront()
    assert f.arrive(100.0, "a", 9000) == "COLLECT"          # D decodes: big waits T_halb
    _check_line(f, -1, verdict="COLLECT", S=9000, n=1, classes="gross", waited=0.0)
    assert f.arrive(101.0, "b", 3100) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=12100, n=2, classes="gross,halb", waited=1.0)


def test_S11_ex6_no_decode_decides_at_once():
    f = _SetFront(d_decoding=False)
    assert f.arrive(100.0, "a", 9000) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=9000, n=1, classes="gross", dd=0, waited=0.0)
    assert f.arrive(100.5, "b", 400) == "D_NOW"
    _check_line(f, -1, verdict="D_NOW", S=400, classes="kurz", dd=0, waited=0.0)


def test_S11_ex7_running_d_prefill_is_rerouted():
    f = _SetFront()
    f.d_running = 4000                                          # D prefills 4000 right now
    assert f.arrive(100.0, "n", 5000) == "REROUTE_P"
    _check_line(f, -1, verdict="REROUTE_P", S=9000, n=1, classes="halb")


def test_S11_ex8_set_due_without_seats_parks():
    f = _SetFront(seats_free=0)
    f.arrive(100.0, "a", 3500)
    assert f.arrive(102.0, "b", 3200) == "PARK_NO_SEAT"
    _check_line(f, -1, verdict="PARK_NO_SEAT", S=6700, n=2, classes="halb,halb")


# --- H1-H9 (tagesplan §4, S2) ---------------------------------------------------------------------------------------


def test_S11_H1_half_plus_half_flips():
    f = _SetFront()
    f.arrive(10.0, "a", 4000)
    assert f.arrive(14.0, "b", 4000) == "FLIP_P"            # inside T_halb (4.0 s < 4.5 s)
    _check_line(f, -1, verdict="FLIP_P", S=8000, n=2, waited=4.0)


def test_S11_H2_half_alone_after_timer_on_d():
    f = _SetFront()
    f.arrive(10.0, "a", 5999)                               # the largest half
    assert f.tick(14.0) == "COLLECT"
    assert f.tick(14.6) == "D_AFTER_TIMER"
    _check_line(f, -1, verdict="D_AFTER_TIMER", S=5999, n=1)


def test_S11_H3_half_plus_short_short_now_half_after_timer():
    f = _SetFront()
    assert f.arrive(10.0, "h", 3000) == "COLLECT"           # exactly halfX is halb
    assert f.arrive(10.2, "k", 2999) == "D_NOW"             # one under halfX is kurz: at once, waited 0
    _check_line(f, -1, verdict="D_NOW", S=2999, classes="kurz", waited=0.0)
    assert f.tick(14.4) == "COLLECT"
    assert f.tick(14.5) == "D_AFTER_TIMER"
    _check_line(f, -1, verdict="D_AFTER_TIMER", S=3000, n=1, classes="halb")


def test_S11_H4_big_during_decode_collects_then_flips_after_timer():
    f = _SetFront()
    assert f.arrive(10.0, "g", 6000) == "COLLECT"           # exactly X is gross
    _check_line(f, -1, verdict="COLLECT", S=6000, classes="gross")
    assert f.tick(14.4) == "COLLECT"
    assert f.tick(14.5) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=6000, n=1, classes="gross", waited=4.5)


def test_S11_H5_big_plus_half_flips_at_once():
    f = _SetFront()
    f.arrive(10.0, "g", 7000)
    assert f.arrive(10.1, "h", 3000) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=10000, n=2, classes="gross,halb")


def test_S11_H6_without_decode_everything_at_once():
    f = _SetFront(d_decoding=False)
    assert f.arrive(10.0, "h", 3500) == "D_AFTER_TIMER"     # R8: S < X -> D, no waiting for the timer
    _check_line(f, -1, verdict="D_AFTER_TIMER", S=3500, dd=0, waited=0.0)
    assert f.arrive(11.0, "g", 8000) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=8000, dd=0, waited=0.0)


def test_S11_H7_no_seat_parks():
    f = _SetFront(seats_free=0)
    assert f.arrive(10.0, "g", 9000) == "COLLECT"
    assert f.tick(14.5) == "PARK_NO_SEAT"                   # the timer ran out, flip due, no seat (R5)
    _check_line(f, -1, verdict="PARK_NO_SEAT", S=9000, n=1)
    f2 = _SetFront(seats_free=0, d_decoding=False)
    assert f2.arrive(10.0, "g", 9000) == "PARK_NO_SEAT"     # R8 + R5
    assert f2.parsed()["verdict"] == "PARK_NO_SEAT"


def test_S11_H8_r6_reroute_p():
    f = _SetFront()
    f.d_running = 4000
    assert f.arrive(10.0, "n1", 3500) == "REROUTE_P"        # S = 7500 > X while D prefills
    _check_line(f, -1, verdict="REROUTE_P", S=7500, n=1)
    f2 = _SetFront()
    f2.d_running = 2000                                     # 2000 + 3500 = 5500 <= X: no reroute, collect
    assert f2.arrive(10.0, "n1", 3500) == "COLLECT"


def test_S11_H9_p_only_counts_as_big():
    f = _SetFront()
    assert f.arrive(10.0, "v", 500, p_only=True) == "COLLECT"      # vision, tiny uncached: still a P candidate
    _check_line(f, -1, verdict="COLLECT", S=500, n=1, classes="p_only")
    assert f.tick(14.5) == "FLIP_P"
    _check_line(f, -1, verdict="FLIP_P", S=500, classes="p_only")
    f2 = _SetFront(d_decoding=False)
    assert f2.arrive(10.0, "v", 500, p_only=True) == "FLIP_P"      # without decode: at once
    f3 = _SetFront()
    f3.arrive(10.0, "v", 500, p_only=True)
    assert f3.arrive(11.0, "h", 3000) == "FLIP_P"                  # a rider joins the p_only candidate (R4)


# --- classification boundaries and the danger directions over a grid ---------------------------------------------------


@pytest.mark.parametrize("u,want", [(0, "kurz"), (2999, "kurz"), (3000, "halb"), (5999, "halb"), (6000, "gross"),
                                    (90000, "gross")])
def test_S11_classify_arrival_boundaries(u, want):
    assert _pp11().classify_arrival(u, _SX, _SHALF, False) == want


def test_S11_classify_arrival_p_only_and_frac():
    PP = _pp11()
    assert PP.classify_arrival(10, _SX, _SHALF, True) == "p_only"
    assert PP.classify_arrival(2000, _SX, 0.25, False) == "halb"      # F_halb is the flag, halfX = 1500


@pytest.mark.parametrize("u1", [3000, 3500, 4999, 5999])
@pytest.mark.parametrize("u2", [3000, 3200, 5999])
@pytest.mark.parametrize("dt", [0.0, 1.0, 4.4])
def test_S11_grid_two_halves_flip_iff_sum_reaches_x(u1, u2, dt):
    """R7 (no FLIP_P under X) AND R3 (S >= X inside T_halb flips) -- both directions, the whole grid."""
    f = _SetFront()
    f.arrive(100.0, "a", u1)
    got = f.arrive(100.0 + dt, "b", u2)
    if u1 + u2 >= _SX:
        assert got == "FLIP_P", (u1, u2, dt, f.lines[-1])
    else:
        assert got == "COLLECT", (u1, u2, dt, f.lines[-1])          # never FLIP_P under X
        assert f.tick(100.0 + _ST + 0.01) == "D_AFTER_TIMER", f.lines[-1]
        assert f.parsed()["S"] == str(u1 + u2)


def test_S11_timer_starts_once_per_set():
    """A later member does not restart the timer: the set expires T_halb after the FIRST member."""
    f = _SetFront()
    f.arrive(100.0, "a", 3100)
    f.arrive(100.0 + 4.0, "b", 100 + 2000)     # kurz -> D_NOW, must not touch the set's timer
    assert f.tick(104.5) == "D_AFTER_TIMER", f.lines[-1]
    f2 = _SetFront()
    f2.arrive(100.0, "a", 3100)
    f2.d_decoding = True
    assert f2.tick(104.5) == "D_AFTER_TIMER"
    assert f2.members == [] and f2.t_open is None


def test_S11_timer_starts_once_two_members_below_x():
    """F_halb is a flag: at 0.25 two halb members stay under X (1600 + 1700 < 6000). The second one must NOT restart
    the timer -- the set expires T_halb after the FIRST member (R3 'Timer startet beim ersten Mitglied')."""
    f = _SetFront(half_frac=0.25)
    assert f.arrive(100.0, "a", 1600) == "COLLECT"
    assert f.arrive(103.0, "b", 1700) == "COLLECT"
    assert f.tick(104.4) == "COLLECT"
    assert f.tick(104.5) == "D_AFTER_TIMER", f.lines[-1]
    assert f.parsed()["S"] == "3300" and f.parsed()["n"] == "2" and abs(float(f.parsed()["w"]) - 4.5) < 0.05


def test_S11_decode_ends_during_timer_decides_at_once():
    f = _SetFront()
    f.arrive(100.0, "a", 3500)
    f.d_decoding = False                       # D's last decode finished mid-timer (R8)
    assert f.tick(101.0) == "D_AFTER_TIMER"
    f2 = _SetFront()
    f2.arrive(100.0, "g", 9000)
    f2.d_decoding = False
    assert f2.tick(101.0) == "FLIP_P"


def test_S11_seat_frees_during_timer_flips_instead_of_parking():
    f = _SetFront(seats_free=0)
    f.arrive(100.0, "a", 3500)
    f.members.append(f.PP.SetMember(rid="b", uncached=3200, t_arrive=101.0))
    f.seats_free = 2                           # a seat freed while the set waited
    assert f.tick(101.0) == "FLIP_P"


# --- the REAL front: format string and call edge (source anchors) -----------------------------------------------------


def _front_source():
    from sglang.srt.weg2 import front as FR

    return open(FR.__file__).read()


def test_S11_front_logs_the_set_verdict_line_in_the_soll_format():
    """The front's format string carries every §4 key, in order, after 'WEG2 SET-VERDICT'."""
    src = _front_source()
    at = src.find("WEG2 SET-VERDICT")
    assert at >= 0, "front.py never logs 'WEG2 SET-VERDICT' (S1 not in this tree)"
    window = src[at:at + 3000]
    pos = 0
    for key in SET_KEYS:
        nxt = window.find(key, pos)
        assert nxt >= 0, f"front's SET-VERDICT format lacks '{key}' in §4 order: {window[:400]!r}"
        pos = nxt + len(key)


def test_S11_front_calls_the_policy_with_every_keyword():
    """Call edge (not the helper): front.py calls set_verdict with each §4 keyword and classify_arrival at all."""
    import ast

    from sglang.srt.weg2 import front as FR

    tree = ast.parse(open(FR.__file__).read())
    calls = {"set_verdict": [], "classify_arrival": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in calls:
                calls[name].append({k.arg for k in node.keywords if k.arg})
    assert calls["classify_arrival"], "front.py never calls classify_arrival (R2: the class comes at arrival)"
    assert calls["set_verdict"], "front.py never calls set_verdict (R1/R11: the front decides over the set)"
    need = {"x_tokens", "half_frac", "t_half_s", "now", "t_open", "d_decoding", "seats_free",
            "d_prefill_running_uncached", "new_since_last"}
    assert any(need <= kws for kws in calls["set_verdict"]), \
        f"no set_verdict call passes every §4 keyword {sorted(need)}; calls pass {calls['set_verdict']}"
