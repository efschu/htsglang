"""L15-END-ANCHOR (weg2/l15_end_anchor.py): a finished DFLASH request on D leaves
a Mamba anchor at its EXACT committed end, under the overlap schedule.

Hermetic, CPU. A small state model stands in for the GPU: every Mamba slot
carries the POSITION of the state it holds (None = never written / garbage).
The REAL code under test drives it:

* ``DFlashWorkerV2._update_target_mamba_state_after_verify`` (grid + armed
  steps), whose backend call is the fake GPU: the live slot advances to
  ``pre + last_step + 1``, a track slot with step s >= 0 receives
  ``pre + s + 1``;
* ``SchedulerBatchResultProcessor._mamba_prefix_cache_update`` (result side);
* ``HybridReqToTokenPool.get_mamba_ping_pong_other_idx`` / ``_keep_idx``;
* ``l15_end_anchor.plan_verify`` (plan side; absent on the base commit).

The overlap order is the event loop's: plan+launch round k+1 BEFORE the result
of round k is processed; a request that finishes in round k has already been
launched in round k+1 (the discarded extra forward), whose GPU effect is
applied before the finish insert reads the slot (adversarial: stream order
puts it first). The insert check is POSITION, not existence: the donated slot
must hold the state at exactly ``mamba_last_track_seqlen``.
"""

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import sglang.srt.managers.scheduler_components.batch_result_processor as brp
import sglang.srt.runtime_context as rc
from sglang.srt.mem_cache import memory_pool as mp
from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.speculative import dflash_worker_v2 as dfw

try:  # absent on the base commit -> the armed tests are red there
    from sglang.srt.weg2 import l15_end_anchor as ea
except ImportError:  # pragma: no cover - base commit
    ea = None

INTERVAL = 256


# --------------------------------------------------------------------------- fakes
class FakePool:
    get_mamba_ping_pong_other_idx = mp.HybridReqToTokenPool.get_mamba_ping_pong_other_idx
    get_mamba_ping_pong_keep_idx = mp.HybridReqToTokenPool.get_mamba_ping_pong_keep_idx

    def __init__(self, size=2):
        self.mamba_ping_pong_track_buffer_size = size
        self.enable_mamba_extra_buffer_lazy = False


class Req:
    def __init__(self, rid, prompt, target_out, live, pp, first_out=1):
        self.rid = rid
        self.origin_input_ids = list(range(prompt))
        self.output_ids = [7] * first_out  # prefill produced the first (unfed) token
        self.kv_committed_len = prompt  # fed tokens after the prefill/extend
        self.target_out = target_out
        self.mamba_pool_idx = torch.tensor(live)
        self.mamba_ping_pong_track_buffer = torch.tensor(pp)
        self.mamba_next_track_idx = 0
        self.mamba_last_track_seqlen = None
        self.grammar = None
        self.session = None
        self.is_retracted = False

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def finished(self):
        return len(self.output_ids) >= self.target_out


@dataclasses.dataclass
class FakeSB:
    """Dataclass stand-in for ScheduleBatch: ``Scheduler._forward_isolation``
    snapshots/restores ``dataclasses.fields`` of the batch."""

    reqs: Any = None
    req_to_token_pool: Any = None
    tree_cache: Any = None
    spec_algorithm: Any = None
    mamba_track_indices: Any = None
    mamba_track_mask: Any = None
    mamba_track_seqlens: Any = None
    weg2_end_anchor: Any = None
    weg2_end_anchor_mask: Any = None
    sampling_info: Any = None
    device: Any = "cpu"


def _isolated(batch):
    """The REAL spec-v2 forward isolation (overlap=False: no batch_record_buf)."""
    return sched_mod.Scheduler._forward_isolation(SimpleNamespace(), batch, overlap=False)


class GPU:
    """slot -> position of the state it holds."""

    def __init__(self):
        self.pos = {}
        self.pre = None

    def update_mamba_state_after_mtp_verify(
        self, *, last_correct_step_indices, mamba_track_indices, mamba_steps_to_track, model
    ):
        pre = self.pre
        for r in range(len(last_correct_step_indices)):
            self.pos[self._live[r]] = int(pre[r]) + int(last_correct_step_indices[r]) + 1
        if mamba_track_indices is not None:
            for r in range(len(last_correct_step_indices)):
                s = int(mamba_steps_to_track[r])
                if s >= 0:
                    self.pos[int(mamba_track_indices[r])] = int(pre[r]) + s + 1


def _sargs():
    return SimpleNamespace(
        mamba_cache_chunk_size=64,
        mamba_track_interval=INTERVAL,
        enable_mamba_extra_buffer=lambda: True,
        enable_mamba_extra_buffer_lazy=lambda: False,
        mamba_checkpoint_interval=None,
        page_size=1,
        pp_size=1,
        speculative_algorithm="DFLASH",
    )


@pytest.fixture
def env(monkeypatch):
    sa = _sargs()
    monkeypatch.setattr(rc, "get_server_args", lambda: sa)
    monkeypatch.setattr(brp, "get_server_args", lambda: sa)
    monkeypatch.delenv("SGLANG_WEG2_L15_TREE_CAND_MIN_TOKENS", raising=False)
    if ea is not None:
        ea.reset_for_tests()
    yield sa
    if ea is not None:
        ea.reset_for_tests()


def _arm(monkeypatch, on=True):
    if on:
        monkeypatch.setenv("SGLANG_WEG2_L15", "1")
        monkeypatch.setenv("SGLANG_WEG2_L15_END_ANCHOR", "1")
    else:
        monkeypatch.delenv("SGLANG_WEG2_L15_END_ANCHOR", raising=False)
    if ea is not None:
        ea.reset_for_tests()


def _stock_rebuild(batch):
    # mirrors set_mamba_track_indices_from_reqs (pinned/device-free)
    batch.mamba_track_indices = torch.tensor(
        [int(r.mamba_ping_pong_track_buffer[r.mamba_next_track_idx]) for r in batch.reqs]
    )
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None


class Sim:
    """One running batch of DFLASH decode rounds under the overlap loop."""

    def __init__(self, reqs, *, overlap=True, track_indices="none", mutate=None):
        self.reqs = reqs
        self.pool = FakePool(2 if overlap else 1)
        self.gpu = GPU()
        self.gpu._live = [int(r.mamba_pool_idx) for r in reqs]
        self.overlap = overlap
        # base-commit DFLASH: after a filter/merge the indices are None; an
        # unfiltered batch keeps the extend's tensor ("stale")
        self.track_indices = track_indices
        self.mutate = mutate or {}
        self.batch = FakeSB(
            reqs=list(reqs),
            req_to_token_pool=self.pool,
            tree_cache=SimpleNamespace(page_size=1),
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            mamba_track_indices=(
                torch.tensor([int(r.mamba_ping_pong_track_buffer[0]) for r in reqs])
                if track_indices == "stale"
                else None
            ),
        )
        self.inserts = []  # (rid, cache_len, donated_slot, state_pos)
        self.extra_forwards = 0

    # -- plan + launch ------------------------------------------------------------
    def launch(self, commits):
        b = self.batch
        live = [r for r in b.reqs]
        with _isolated(b):
            done = self._launch_inner(b, live, commits)
        # event loop: batch.copy() AFTER run_batch (i.e. after the isolation)
        snap = SimpleNamespace(
            reqs=list(live),
            req_to_token_pool=self.pool,
            spec_algorithm=b.spec_algorithm,
            weg2_end_anchor=b.weg2_end_anchor,
            weg2_end_anchor_mask=b.weg2_end_anchor_mask,
        )
        return snap, done

    def _launch_inner(self, b, live, commits):
        # forward_batch_generation's top: a plan belongs to one verify
        b.weg2_end_anchor = None
        b.weg2_end_anchor_mask = None
        if ea is not None:
            other = self.mutate.get("other_idx")
            ea.plan_verify(b, rebuild=_stock_rebuild, other_idx=other)
        pre = torch.tensor([self._device_fed(r) for r in live], dtype=torch.int64)
        commit = torch.tensor([commits[r.rid] for r in live], dtype=torch.int32)
        self.gpu.pre = pre
        self.gpu._live = [int(r.mamba_pool_idx) for r in live]
        fake_self = SimpleNamespace(
            _need_mamba_verify_commit=True,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(attn_backend=self.gpu, model=object())
            ),
        )
        dfw.DFlashWorkerV2._update_target_mamba_state_after_verify(
            fake_self,
            batch=b,
            seq_lens_pre_verify=pre,
            seq_lens_post_verify=pre + commit.to(pre.dtype),
            commit_lens=commit,
        )
        for r, c in zip(live, commit.tolist()):
            r._device_fed = self._device_fed(r) + c  # the GPU's own length
        return {r.rid: c for r, c in zip(live, commit.tolist())}

    @staticmethod
    def _device_fed(r):
        return getattr(r, "_device_fed", r.kv_committed_len)

    # -- result -------------------------------------------------------------------
    def process(self, snap, commits):
        proc = SimpleNamespace(
            _mamba_check_track_boundary=lambda *a: brp.SchedulerBatchResultProcessor._mamba_check_track_boundary(
                None, *a
            ),
            mamba_lazy_post_decode_at_boundary=lambda *a: None,
        )
        finished = []
        for i, r in enumerate(snap.reqs):
            if r.finished() or r.is_retracted:
                self.extra_forwards += 1
                continue
            c = commits[r.rid]
            r.kv_committed_len += c  # _resolve_spec_v2_tokens: drafts + bonus
            r.output_ids.extend([7] * c)
            result = SimpleNamespace(num_correct_drafts_per_req_cpu=[0] * len(snap.reqs))
            result.num_correct_drafts_per_req_cpu[i] = c - 1
            brp.SchedulerBatchResultProcessor._mamba_prefix_cache_update(proc, r, snap, result, i)
            if r.finished():
                finished.append(r)
        return finished

    def insert(self, r):
        cache_len = r.mamba_last_track_seqlen
        if cache_len is None:
            self.inserts.append((r.rid, None, None, None))
            return
        if "donate" in self.mutate:
            slot, cache_len = self.mutate["donate"](self, r)
        else:
            keep = self.pool.get_mamba_ping_pong_keep_idx(r)
            slot = int(r.mamba_ping_pong_track_buffer[keep])
        self.inserts.append((r.rid, cache_len, slot, self.gpu.pos.get(slot)))

    def filter_finished(self, finished):
        if finished:
            self.batch.reqs = [r for r in self.batch.reqs if r not in finished]
            # filter_batch: the stock path nulls the track destinations
            self.batch.mamba_track_indices = None

    def run(self, schedule):
        """schedule: list of {rid: commit_len} per round (overlap: result k after
        launch k+1)."""
        pending = None
        for commits in schedule + [None]:
            if commits is not None and self.batch.reqs:
                c = {r.rid: commits.get(r.rid, 1) for r in self.batch.reqs}
                launched = self.launch(c)
            else:
                launched = None
            if not self.overlap and launched is not None:
                fin = self.process(*launched)
                for r in fin:
                    self.insert(r)
                self.filter_finished(fin)
                continue
            if pending is not None:
                fin = self.process(*pending)
                for r in fin:
                    self.insert(r)
                self.filter_finished(fin)
            pending = launched
            if launched is None:
                break
        return self.inserts


def _assert_exact(inserts, reqs):
    by = {r.rid: r for r in reqs}
    for rid, cache_len, slot, state_pos in inserts:
        assert cache_len is not None, f"{rid}: tombstone, no end anchor"
        assert state_pos == cache_len, (
            f"{rid}: anchor key length {cache_len} but the donated slot {slot} holds "
            f"the state at {state_pos}"
        )
        assert cache_len == by[rid].kv_committed_len, (
            f"{rid}: anchor {cache_len} is not the committed end {by[rid].kv_committed_len}"
        )


# ------------------------------------------------------------------ the leg-2 shape
def _leg2(prompt=60054, commits=(3, 1, 8, 2, 4, 5, 2, 3, 1, 6, 3, 2, 4, 3, 2), target=48):
    r = Req("weg2-8-7", prompt, target, live=5, pp=[10, 11])
    return r, [{"weg2-8-7": c} for c in commits] + [{"weg2-8-7": 2}] * 4


def test_off_is_byte_for_byte(env, monkeypatch):
    _arm(monkeypatch, on=False)
    if ea is None:
        pytest.skip("base commit: nothing to switch")
    b = SimpleNamespace(reqs=[], req_to_token_pool=FakePool())
    assert ea.plan_verify(b, rebuild=lambda _b: pytest.fail("touched")) is None
    assert not hasattr(b, "weg2_end_anchor")


def test_gate_refusals_by_name(env, monkeypatch):
    if ea is None:
        pytest.fail("l15_end_anchor missing (base commit)")
    _arm(monkeypatch)
    for field, val, code in [
        ("page_size", 64, "W-L15-EA-PAGE"),
        ("pp_size", 2, "W-L15-EA-PP"),
        ("speculative_algorithm", "EAGLE", "W-L15-EA-NOT-DFLASH"),
        ("mamba_checkpoint_interval", 4096, "W-L15-EA-CKPT-INTERVAL"),
    ]:
        sa = _sargs()
        setattr(sa, field, val)
        assert ea._evaluate_gate(sa) == (False, code)
    sa = _sargs()
    sa.enable_mamba_extra_buffer_lazy = lambda: True
    assert ea._evaluate_gate(sa) == (False, "W-L15-EA-LAZY")
    monkeypatch.setenv("SGLANG_WEG2_L15", "0")
    assert ea._evaluate_gate(_sargs()) == (False, "off")


def test_leg2_without_grid_crossing_leaves_exact_end_anchor(env, monkeypatch):
    """The measured class: 60054 + ~48 tokens crosses no 256 point (60160).
    Base: tombstone. Fix: anchor at kv_committed_len, slot holds exactly it,
    although the extra forward already ran."""
    _arm(monkeypatch)
    r, sched = _leg2()
    sim = Sim([r], track_indices="none")
    inserts = sim.run(sched)
    assert sim.extra_forwards == 1  # the discarded overlap forward happened
    assert len(inserts) == 1
    _assert_exact(inserts, [r])


def test_accept_lengths_above_one_and_full_block(env, monkeypatch):
    # every round commits > 1 (drafts + bonus), one full 8-block, and the
    # request finishes inside a long accept run (output overshoots target)
    _arm(monkeypatch)
    r = Req("r", 101106, 40, live=3, pp=[20, 21])
    sched = [{"r": c} for c in (8, 7, 2, 8, 5, 8, 6, 8)]
    sim = Sim([r])
    _assert_exact(sim.run(sched), [r])
    assert len(r.output_ids) > r.target_out  # stop inside the committed run


def test_grid_crossing_request_stays_exact(env, monkeypatch):
    # weg2-38-67's shape: anchor 101101, end 101154 crosses 101120; armed, the
    # end (not the grid point) is the anchor, and its state matches.
    _arm(monkeypatch)
    r = Req("weg2-38-67", 101106, 49, live=4, pp=[30, 31])
    sim = Sim([r], track_indices="stale")
    _assert_exact(sim.run([{"weg2-38-67": c} for c in (3, 2, 4, 3, 8, 2, 5, 3, 4, 6, 2, 3, 2, 2)]), [r])


def test_mixed_batch_unarmed_grid_request_is_exact_after_filter(env, monkeypatch):
    """Rebuild part (upstream 44fd17b696): a short background request (below
    MIN_TOKENS, NOT armed) crosses 256 after the batch was filtered. Base
    DFLASH writes no track then (indices None) while the scheduler records 256
    -> garbage anchor. Armed switch: rebuilt indices, the grid anchor holds."""
    _arm(monkeypatch)
    monkeypatch.setenv("SGLANG_WEG2_L15_TREE_CAND_MIN_TOKENS", "4096")
    long_r = Req("long", 60054, 10, live=1, pp=[10, 11])
    bg = Req("bg", 240, 60, live=2, pp=[12, 13])
    sched = [{"long": 3, "bg": 3}] * 8 + [{"bg": 4}] * 12
    sim = Sim([long_r, bg], track_indices="none")
    inserts = sim.run(sched)
    assert {i[0] for i in inserts} == {"long", "bg"}
    by = {i[0]: i for i in inserts}
    _assert_exact([by["long"]], [long_r])
    rid, cache_len, slot, state_pos = by["bg"]
    assert cache_len == 256 and state_pos == 256, by["bg"]
    assert getattr(bg, ea.ARMED_ATTR) is False


@pytest.mark.xfail(strict=True, reason="OPEN DEFAULT-PATH DEFECT: fork DFLASH verify never rebuilds "
                   "mamba_track_indices (upstream 44fd17b696 unported); after a filter the scheduler "
                   "records a grid anchor whose slot was never written")
def test_default_path_grid_anchor_after_filter_is_exact(env, monkeypatch):
    _arm(monkeypatch, on=False)
    a = Req("a", 100, 5, live=1, pp=[10, 11])  # finishes in round 2 -> filter
    bg = Req("bg", 240, 60, live=2, pp=[12, 13])
    sim = Sim([a, bg], track_indices="stale")
    inserts = sim.run([{"a": 3, "bg": 3}] * 8 + [{"bg": 4}] * 12)
    by = {i[0]: i for i in inserts}
    assert by["bg"][1] == 256
    assert by["bg"][3] == by["bg"][1]


def test_no_overlap_single_slot(env, monkeypatch):
    _arm(monkeypatch)
    r, sched = _leg2(prompt=19707)
    sim = Sim([r], overlap=False)
    _assert_exact(sim.run(sched), [r])
    assert sim.extra_forwards == 0


def test_retract_at_plan_time_keeps_last_processed_round(env, monkeypatch):
    """Park/retract happens at planning: round k is in flight (its GPU write
    applied), result k not processed. The retention insert must donate the
    slot of round k-1 at round k-1's position, not the in-flight slot."""
    _arm(monkeypatch)
    r = Req("p", 30000, 10_000, live=6, pp=[40, 41])
    sim = Sim([r])
    s1, c1 = sim.launch({"p": 3})
    s2, c2 = sim.launch({"p": 5})  # overlap: launched before result 1
    sim.process(s1, c1)
    s3, c3 = sim.launch({"p": 2})
    sim.process(s2, c2)
    # planning round 4: in flight = round 3 (already written), retract now
    keep = sim.pool.get_mamba_ping_pong_keep_idx(r)
    slot = int(r.mamba_ping_pong_track_buffer[keep])
    assert r.mamba_last_track_seqlen == 30000 + 3 + 5
    assert sim.gpu.pos[slot] == r.mamba_last_track_seqlen
    inflight = int(r.mamba_ping_pong_track_buffer[1 - keep])
    assert sim.gpu.pos[inflight] == 30000 + 3 + 5 + 2  # round 3 wrote the other slot


# ---------------------------------------------------------------- mutants (positions)
def _expect_position_failure(env, monkeypatch, **mutate):
    _arm(monkeypatch)
    r, sched = _leg2()
    sim = Sim([r], mutate=mutate)
    inserts = sim.run(sched)
    with pytest.raises(AssertionError):
        _assert_exact(inserts, [r])


def test_mutant_naive_live_slot_donation_is_caught(env, monkeypatch):
    # the rejected fix: donate the live slot at kv_committed_len; the extra
    # forward moved it past the key
    _expect_position_failure(
        env, monkeypatch,
        donate=lambda sim, r: (int(r.mamba_pool_idx), r.kv_committed_len),
    )


def test_mutant_no_alternation_is_caught(env, monkeypatch):
    # write the same slot every round: the extra forward overwrites the keep
    _expect_position_failure(env, monkeypatch, other_idx=lambda w: w)


def test_mutant_stock_keep_rule_is_caught(env, monkeypatch):
    # forget the explicit keep: other(next) names the in-flight slot
    if ea is None:
        pytest.fail("l15_end_anchor missing (base commit)")
    monkeypatch.setattr(ea, "keep_override", lambda req: None)
    _expect_position_failure(env, monkeypatch)


def test_mutant_bonus_off_by_one_is_caught(env, monkeypatch):
    # key one token longer than the state (the unfed bonus counted)
    _expect_position_failure(
        env, monkeypatch,
        donate=lambda sim, r: (
            int(r.mamba_ping_pong_track_buffer[sim.pool.get_mamba_ping_pong_keep_idx(r)]),
            r.kv_committed_len + 1,
        ),
    )


def test_wiring_plan_before_verify_snapshot_and_steps_in_commit():
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(dfw.DFlashWorkerV2))
    assert "_l15_ea.plan_verify(batch)" in src
    plan_at = src.index("_l15_ea.plan_verify(batch)")
    prep_at = src.index("verify_input.prepare_for_verify(")
    assert plan_at < prep_at, "plan must run before the verify ForwardBatch snapshot"
    commit_src = textwrap.dedent(
        inspect.getsource(dfw.DFlashWorkerV2._update_target_mamba_state_after_verify)
    )
    assert "_l15_ea.steps_to_track(" in commit_src
    fbg = textwrap.dedent(inspect.getsource(dfw.DFlashWorkerV2.forward_batch_generation))
    # a stale plan of the previous round is dropped at the top of every forward
    assert fbg.index("batch.weg2_end_anchor = None") < fbg.index("if batch.forward_mode.is_extend()")
    ast.parse(src)


def test_forward_isolation_carries_the_plan_to_the_result(env, monkeypatch):
    """Metal 06:09 boot (..._1006_060930): _forward_isolation restored every
    ScheduleBatch dataclass field after the spec-v2 forward, the plan was None
    at batch.copy(), no anchor. The plan must survive; every OTHER field is
    still restored."""
    if ea is None:
        pytest.fail("l15_end_anchor missing (base commit)")
    _arm(monkeypatch)
    r = Req("iso", 60054, 48, live=5, pp=[10, 11])
    b = FakeSB(reqs=[r], req_to_token_pool=FakePool(),
               tree_cache=SimpleNamespace(page_size=1),
               spec_algorithm=SimpleNamespace(is_none=lambda: False))
    with _isolated(b):
        ea.plan_verify(b, rebuild=_stock_rebuild)
        assert b.mamba_track_indices is not None
    assert b.weg2_end_anchor == [0]
    assert b.weg2_end_anchor_mask is not None
    assert b.mamba_track_indices is None  # the rest is still undone
    # every new ScheduleBatch dataclass field of this feature is carried
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    new = {f.name for f in dataclasses.fields(ScheduleBatch) if "end_anchor" in f.name}
    assert new == set(sched_mod._FORWARD_ISOLATION_CARRY)


def test_armed_request_crossing_the_grid_is_exact(env, monkeypatch):
    """Point 3 of the metal diagnosis: an armed request that crosses 256 must
    not get a grid key paired with a round-end slot (pinned keep + stock flip)."""
    _arm(monkeypatch)
    r = Req("cross", 60100, 70, live=7, pp=[50, 51])  # 60160 crossed mid-run
    sim = Sim([r], track_indices="none")
    inserts = sim.run([{"cross": c} for c in (3, 5, 2, 8, 4, 6, 3, 2, 5, 7, 4, 3, 2, 6, 5, 4, 3)])
    _assert_exact(inserts, [r])
    assert inserts[0][1] > 60160
