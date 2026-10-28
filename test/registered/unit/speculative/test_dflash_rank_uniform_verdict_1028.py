# SPDX-License-Identifier: Apache-2.0
"""27b-rank-uniform-1028: the self-test verdicts of FORMHP-DRAFT-PLAN-SKIP and FORMHP-VPREP-EARLY are rank-uniform at TP>1.

CPU only. Two ranks run as two threads; their TP CPU group is a barrier stand-in (:class:`_Group`) that computes the element-wise MIN
like ``reduce_min_cpu`` and counts the calls per rank. A rank that calls the reduction a different number of times than its peer
breaks the barrier (timeout) instead of hanging -- the test then fails, which is the falsifier of the hang class
(memory: rank-lokaler-test-vor-kollektiv). One real two-process gloo run pins ``reduce_min_cpu`` itself.

Rules pinned: rank 0 PASS + rank 1 FAIL -> both keep the old path; both PASS -> both take the fast path in the same round; TP=1 ->
no reduction at all (``SpecTpSync.verdict_reducer`` is None, the local verdict is the old one). Red on 6ccc712279 (no reduce_min,
no verdict_reducer: the verdicts were rank-local)."""

import dataclasses
import multiprocessing as mp
import socket
import threading
import types

import pytest
import torch

from flliper.srt.speculative import dflash_plan_skip as ps
from flliper.srt.speculative import dflash_vprep_early as ve
from flliper.srt.speculative import spec_tp_sync as sts
from flliper.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

_TIMEOUT_S = 10.0


class _Group:
    """N ranks' CPU group: ``reducer(rank)`` returns that rank's ``reduce_min``; MIN over what every rank handed in."""

    def __init__(self, n: int):
        self.n = n
        self.calls = [0] * n
        self._vals = [None] * n
        self._out = None
        self._b = threading.Barrier(n, timeout=_TIMEOUT_S)

    def reducer(self, rank: int):
        def reduce_min(values):
            self.calls[rank] += 1
            self._vals[rank] = [int(v) for v in values]
            if self._b.wait() == 0:
                self._out = [min(col) for col in zip(*self._vals)]
            self._b.wait()
            return list(self._out)

        return reduce_min


def _ranks(fn, n=2):
    """Run ``fn(rank)`` on n threads; returns the results; re-raises a rank's error (a broken barrier = a would-be hang)."""
    out, err = [None] * n, [None] * n

    def run(r):
        try:
            out[r] = fn(r)
        except BaseException as e:  # noqa: BLE001
            err[r] = e

    ts = [threading.Thread(target=run, args=(r,), name=f"rank{r}") for r in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(4 * _TIMEOUT_S)
    for e in err:
        if e is not None:
            raise e
    return out


# ---------------------------------------------------------------- FORMHP-VPREP-EARLY (dflash_vprep_early.py) -----------------------


class _World:
    """A prep that writes what the real one writes, from the host bound it is given (cf. test_formhp_devdraft_vprep_1028)."""

    def __init__(self, metadata_reads_bound=False, prep_raises_at=()):
        cap = 4
        reg = types.SimpleNamespace(_slots={
            "seq_lens_cpu": types.SimpleNamespace(enabled=True, buffer=torch.full((cap,), 1, dtype=torch.int64)),
        })
        be = types.SimpleNamespace(forward_metadata=_Meta(torch.zeros(3, dtype=torch.int32)))
        self.runner = types.SimpleNamespace(buffer_registry=reg, attn_backend=be, raw_bs=2, raw_num_token=16, bs=2,
                                            _replay_graph_key="k")
        self.metadata_reads_bound = metadata_reads_bound
        self.prep_raises_at = set(prep_raises_at)
        self.exact = None
        self.calls = []
        self.n_prep = 0

    def prep(self):
        self.n_prep += 1
        if self.n_prep in self.prep_raises_at:
            raise RuntimeError("early prep boom")
        bound = self.exact + 8 if self.exact is not None else torch.tensor([30, 40])
        self.calls.append("prep_exact" if self.exact is not None else "prep_bound")
        fb = _FB(seq_lens_cpu=bound.clone(), seq_lens_sum=int(bound.sum()))
        self.runner.buffer_registry._slots["seq_lens_cpu"].buffer[:2].copy_(bound)
        add = int(bound[0]) if self.metadata_reads_bound else 0
        self.runner.attn_backend.forward_metadata.idx.copy_(torch.tensor([1, 2, 3], dtype=torch.int32) + add)
        return ("vi", fb, "pre", True)

    def wait(self):
        self.calls.append("wait")
        self.exact = torch.tensor([20, 25])

    def round(self, state):
        self.exact, self.calls = None, []
        return ve.staged_prep(state=state, eligible=True, prep=self.prep, marks=lambda: None, wait=self.wait,
                              exact_lens=lambda: self.exact, runner=self.runner, block_size=8, bs=2)



@dataclasses.dataclass
class _Meta:
    idx: torch.Tensor


@dataclasses.dataclass
class _FB:
    seq_lens_cpu: torch.Tensor = None
    seq_lens_sum: int = 0


_OLD = ["wait", "prep_exact"]
_SELFTEST = ["prep_bound", "wait", "prep_exact"]
_EARLY = ["prep_bound", "wait"]


@pytest.fixture(autouse=True)
def _switches_on(monkeypatch):
    monkeypatch.delenv(ve.ENV, raising=False)
    monkeypatch.delenv(ps.ENV, raising=False)


def test_vprep_rank0_pass_rank1_fail_both_keep_the_old_order():
    g = _Group(2)

    def rank(r):
        st = ve.VprepEarly(reduce_min=g.reducer(r))
        w = _World(metadata_reads_bound=(r == 1))  # rank 1's metadata reads the bound: its self-test differs
        w.round(st)
        first = (st.verdict, list(w.calls))
        for _ in range(ve.SELFTEST_ROUNDS + 2):
            w.round(st)
        return first, st.verdict, list(w.calls)

    res = _ranks(rank)
    for (first, verdict, last) in res:
        assert first == (False, _SELFTEST)  # decided in the same (first) round on both ranks
        assert verdict is False and last == _OLD
    assert g.calls == [1, 1]  # one reduction (the failed self-test round), none afterwards


def test_vprep_both_pass_arm_together():
    g = _Group(2)

    def rank(r):
        st = ve.VprepEarly(reduce_min=g.reducer(r))
        w = _World()
        seen = []
        for _ in range(ve.SELFTEST_ROUNDS):
            w.round(st)
            seen.append(st.verdict)
        w.round(st)
        return seen, list(w.calls)

    for seen, last in _ranks(rank):
        assert seen == [None] * (ve.SELFTEST_ROUNDS - 1) + [True]
        assert last == _EARLY
    assert g.calls == [ve.SELFTEST_ROUNDS] * 2


def test_vprep_selftest_error_on_one_rank_fails_both():
    g = _Group(2)

    def rank(r):
        st = ve.VprepEarly(reduce_min=g.reducer(r))
        w = _World(prep_raises_at=(1,) if r == 1 else ())
        w.round(st)
        return st.verdict

    assert _ranks(rank) == [False, False]
    assert g.calls == [1, 1]


def test_vprep_armed_error_disarms_all_ranks_together_at_the_round_counter():
    g = _Group(2)

    def rank(r):
        st = ve.VprepEarly(reduce_min=g.reducer(r))
        st.verdict = True
        w = _World()
        raise_at = None
        verdicts = []
        for i in range(ve.AGREE_EVERY):
            if r == 1 and i == 5:
                raise_at = w.n_prep + 1
                w.prep_raises_at = {raise_at}
            w.round(st)
            verdicts.append(st.verdict)
        return verdicts, st.counts

    (v0, c0), (v1, c1) = _ranks(rank)
    assert v0 == v1  # never apart: rank 1's error flips nobody on its own
    assert v0[:-1] == [True] * (ve.AGREE_EVERY - 1) and v0[-1] is False
    assert c1["redo"] == 1 and c0["redo"] == 0  # the erroring round prepared again (old way) on rank 1 only
    assert g.calls == [1, 1]  # the one periodic reduction


def test_vprep_static_flag_differs_between_ranks_fails_both(monkeypatch):
    g = _Group(2)
    local = threading.local()
    monkeypatch.setattr(ve, "switch_on", lambda: getattr(local, "on", True))

    def rank(r):
        local.on = r == 0  # the switch is on on rank 0 only
        st = ve.VprepEarly(reduce_min=g.reducer(r))
        st.agree_static(local_ok=True, why="FLLIPER_PDFLIP_D_EARLY_DRAFT")
        a = st.verdict
        st2 = ve.VprepEarly(reduce_min=g.reducer(r))
        local.on = True
        st2.agree_static(local_ok=(r == 1), why="FLLIPER_PDFLIP_D_EARLY_DRAFT")  # early-draft flag on rank 1 only
        b = st2.verdict
        st3 = ve.VprepEarly(reduce_min=g.reducer(r))
        st3.agree_static(local_ok=False, why="FLLIPER_PDFLIP_D_EARLY_DRAFT")  # off everywhere: uniform, nothing to fail
        return a, b, st3.verdict

    assert _ranks(rank) == [(False, False, None)] * 2


def test_vprep_tp1_has_no_reducer_and_keeps_the_local_verdict():
    st = ve.VprepEarly()
    assert st.reduce_min is None
    st.agree_static(local_ok=False, why="x")  # no-op on a single rank
    assert st.verdict is None
    w = _World(metadata_reads_bound=True)
    w.round(st)
    assert st.verdict is False and w.calls == _SELFTEST


# ---------------------------------------------------------------- FORMHP-DRAFT-PLAN-SKIP (dflash_plan_skip.py) ----------------------


def _updater():
    return types.SimpleNamespace(skip_wrapper_ids=frozenset(), probe_wrapper_ids=frozenset(),
                                 attn_backend=types.SimpleNamespace(num_wrappers=2))


def test_plan_skip_settle_rank0_pass_rank1_fail_both_plan_both_wrappers():
    g = _Group(2)

    def rank(r):
        d, u = ps.DraftPlanSkip(reduce_min=g.reducer(r)), _updater()
        d.settle(u, code=ps.PASS if r == 0 else ps.FAIL, why=f"rank {r}", ids=(1,))
        return d.verdict, u.skip_wrapper_ids

    assert _ranks(rank) == [(False, frozenset())] * 2


def test_plan_skip_settle_both_pass_skip_on_both():
    g = _Group(2)

    def rank(r):
        d, u = ps.DraftPlanSkip(reduce_min=g.reducer(r)), _updater()
        d.settle(u, code=ps.PASS, why="eq", ids=(1,))
        return d.verdict, u.skip_wrapper_ids

    assert _ranks(rank) == [(True, frozenset({1}))] * 2


def test_plan_skip_undecided_on_one_rank_keeps_both_undecided():
    g = _Group(2)

    def rank(r):
        d, u = ps.DraftPlanSkip(reduce_min=g.reducer(r)), _updater()
        d.settle(u, code=ps.PASS if r == 0 else ps.UNDECIDED, why="", ids=(1,))
        return d.verdict, d.undecided

    assert _ranks(rank) == [(None, 1)] * 2


def test_plan_skip_candidate_differs_between_ranks_no_rank_probes(monkeypatch):
    g = _Group(2)

    def rank(r):
        d, u = ps.DraftPlanSkip(reduce_min=g.reducer(r)), _updater()
        local_ids = (1,) if r == 0 else ()  # rank 1: switch off / static scan
        return d.agree_candidate(u, local_ids), d.verdict

    assert _ranks(rank) == [((), False)] * 2


def test_plan_skip_candidate_uniform_probes():
    g = _Group(2)
    assert _ranks(lambda r: ps.DraftPlanSkip(reduce_min=g.reducer(r)).agree_candidate(_updater(), (1,))) == [(1,), (1,)]


def _worker(updater, reduce_min, *, forward=None):
    w = object.__new__(DFlashWorkerV2)
    w.__dict__["_draft_plan_skip"] = ps.DraftPlanSkip(reduce_min=reduce_min)
    w.__dict__["block_size"] = 8
    w.__dict__["_draft_sampler"] = types.SimpleNamespace(out=torch.arange(14))
    w.__dict__["_draft_plan_skip_updater"] = lambda: updater
    if forward is not None:
        w.__dict__["draft_model_runner"] = types.SimpleNamespace(forward=forward)
    return w


def _out(hs):
    return types.SimpleNamespace(can_run_graph=True, logits_output=types.SimpleNamespace(hidden_states=hs))


def test_plan_skip_worker_judge_rank0_equal_rank1_differs_both_old_plan():
    g = _Group(2)
    hs = torch.randn(8, 16)

    def rank(r):
        u = _updater()
        w = _worker(u, g.reducer(r))
        snap = w._draft_plan_skip_snapshot(_out(hs), 2)
        ref = hs.clone()
        if r == 1:
            ref[3, 5] += 1.0  # rank 1's numerics differ
        w._draft_plan_skip_judge(snap, _out(ref), 2, (1,))
        return w.__dict__["_draft_plan_skip"].verdict, u.skip_wrapper_ids

    assert _ranks(rank) == [(False, frozenset())] * 2


def test_plan_skip_worker_probe_error_on_one_rank_one_reduction_each():
    """Rank 1's probe forward raises (settled in the probe, judge skipped), rank 0 judges: one outcome reduction per rank, both FAILED."""
    g = _Group(2)
    hs = torch.randn(8, 16)

    def rank(r):
        u = _updater()

        def fwd(_fb):
            if r == 1:
                raise RuntimeError("probe boom")
            return _out(hs)

        w = _worker(u, g.reducer(r), forward=fwd)
        snap = w._draft_plan_skip_probe(lambda: None, (1,), 2)
        if snap is not None:
            w._draft_plan_skip_judge(snap, _out(hs.clone()), 2, (1,))
        return w.__dict__["_draft_plan_skip"].verdict

    assert _ranks(rank) == [False, False]
    assert g.calls == [1, 1]


def test_plan_skip_worker_reducer_choice():
    """Solo placement: only the host drafts -> local verdict (a TP-group reduction would wait for shadows that never come);
    otherwise the TP verdict reducer."""
    marker = object()
    for solo, want in ((True, None), (False, marker)):
        w = object.__new__(DFlashWorkerV2)
        w.__dict__["_spec_solo_active"] = solo
        w.__dict__["_tp_sync"] = types.SimpleNamespace(verdict_reducer=lambda: marker)
        assert w._draft_plan_skip_state().reduce_min is want


def test_vprep_worker_state_takes_the_tp_reducer_and_agrees_statically():
    g = _Group(2)

    def rank(r):
        w = object.__new__(DFlashWorkerV2)
        w.__dict__["_tp_sync"] = types.SimpleNamespace(verdict_reducer=lambda: g.reducer(r))
        w.__dict__["_early_draft"] = r == 0
        return w._vprep_early_state().verdict

    assert _ranks(rank) == [False, False]
    assert g.calls == [1, 1]


# ---------------------------------------------------------------- SpecTpSync.verdict_reducer / reduce_min_cpu ----------------------


def _tp_group(world_size):
    return types.SimpleNamespace(world_size=world_size, rank_in_group=0, cpu_group=object())


def test_verdict_reducer_none_at_tp1_and_cpu_group_min_at_tp2(monkeypatch):
    seen = []

    def fake_all_reduce(t, op=None, group=None):
        seen.append((t.tolist(), op, group))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    assert sts.SpecTpSync(_tp_group(1)).verdict_reducer() is None
    assert seen == []
    grp = _tp_group(2)
    red = sts.SpecTpSync(grp).verdict_reducer()
    assert red([1, 0, 2]) == [1, 0, 2]
    assert seen == [([1, 0, 2], torch.distributed.ReduceOp.MIN, grp.cpu_group)]


def _gloo_rank(rank, port, values, q):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        q.put((rank, sts.reduce_min_cpu(values, cpu_group=dist.group.WORLD)))
    finally:
        dist.destroy_process_group()


def test_reduce_min_cpu_real_gloo_two_ranks():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    vals = {0: [2, 1, 1], 1: [0, 1, 2]}  # rank 0 PASS, rank 1 FAIL (+ two flag columns)
    ps_ = [ctx.Process(target=_gloo_rank, args=(r, port, vals[r], q)) for r in (0, 1)]
    for p in ps_:
        p.start()
    got = dict(q.get(timeout=60) for _ in ps_)
    for p in ps_:
        p.join(30)
    assert got == {0: [0, 1, 1], 1: [0, 1, 1]}
