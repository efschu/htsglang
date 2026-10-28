# SPDX-License-Identifier: Apache-2.0
"""formhp-devdraft-b2-1028: FORMHP-DRAFT-PLAN-SKIP -- the DFLASH draft's FlashInfer plan without the full wrapper no layer reads.

CPU only: the updater's wrapper loop with a recording call_begin_forward, the static scan, the bitwise compare, the arming, and the
worker's self-test judge on stand-ins. Red on cd1aa0a62 (no dflash_plan_skip module, no skip/probe ids on the updater)."""

import types

import pytest
import torch

from flliper.srt.layers.attention import flashinfer_backend as fb
from flliper.srt.speculative import dflash_plan_skip as ps
from flliper.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class _Attn(torch.nn.Module):
    def __init__(self, layer_id, sws):
        super().__init__()
        self.layer_id = layer_id
        self.sliding_window_size = sws


class _Model(torch.nn.Module):
    def __init__(self, windows):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Attn(i, w) for i, w in enumerate(windows)])


def _updater(window=2048):
    u = object.__new__(fb.FlashInferIndicesUpdaterPrefill)
    u.sliding_window_size = window
    u.prefill_wrapper_ragged = None
    u._swa_kv_pool = None
    u.kv_indptr = [torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)]
    u.qo_indptr = [torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)]
    u.attn_backend = types.SimpleNamespace(num_wrappers=2)
    u.skip_wrapper_ids = frozenset()
    u.probe_wrapper_ids = frozenset()
    u.update = u.update_sliding_window
    calls = []

    def _rec(ragged, wrapper, req_pool_indices, paged_kernel_lens, paged_kernel_lens_sum, seq_lens, prefix_lens, kv_start_idx,
             kv_indptr, qo_indptr, use_ragged, spec_info, **kw):
        calls.append((wrapper, paged_kernel_lens.clone(), int(paged_kernel_lens_sum), kv_start_idx.clone(),
                      None if kw.get("paged_kernel_lens_cpu") is None else kw["paged_kernel_lens_cpu"].clone()))

    u.call_begin_forward = _rec
    return u, calls


def _run(u):
    seq = torch.tensor([5000, 900], dtype=torch.int64)
    prefix = seq - 16
    u.update_sliding_window(
        req_pool_indices=torch.tensor([0, 1]), seq_lens=seq, seq_lens_cpu=seq.clone(), seq_lens_sum=int(seq.sum()),
        prefix_lens=prefix, prefill_wrappers=["W0", "W1"], use_ragged=False, encoder_lens=None, spec_info=None,
        extend_prefix_lens_cpu=prefix.clone())
    return seq


def test_updater_default_plans_both_wrappers():
    u, calls = _updater()
    _run(u)
    assert [c[0] for c in calls] == ["W0", "W1"]


def test_updater_skips_unread_wrapper_and_keeps_wrapper0_bytes():
    u, calls = _updater()
    _run(u)
    w0_old = calls[0]
    u2, calls2 = _updater()
    u2.skip_wrapper_ids = frozenset({1})
    _run(u2)
    assert [c[0] for c in calls2] == ["W0"]
    for a, b in zip(w0_old[1:], calls2[0][1:]):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        else:
            assert a == b


def test_updater_probe_plans_full_wrapper_on_one_key():
    u, calls = _updater()
    u.probe_wrapper_ids = frozenset({1})
    seq = _run(u)
    w1 = [c for c in calls if c[0] == "W1"][0]
    assert torch.equal(w1[1], torch.ones_like(seq))
    assert w1[2] == 2
    assert torch.equal(w1[3], seq - 1)
    assert torch.equal(w1[4], torch.ones_like(seq))
    # wrapper 0 identical to the normal plan
    u0, calls0 = _updater()
    _run(u0)
    assert torch.equal(calls[0][1], calls0[0][1]) and torch.equal(calls[0][3], calls0[0][3])


def test_static_scan():
    assert ps.unread_wrapper_ids(_Model([2048] * 5), 2, True) == (1,)
    assert ps.unread_wrapper_ids(_Model([2048, -1, 2048]), 2, True) == ()
    assert ps.unread_wrapper_ids(_Model([-1, -1]), 2, True) == (0,)
    assert ps.unread_wrapper_ids(_Model([]), 2, True) == ()
    assert ps.unread_wrapper_ids(_Model([2048]), 1, True) == ()
    assert ps.unread_wrapper_ids(_Model([2048]), 2, False) == ()


def test_outputs_equal_bitwise():
    a = torch.randn(4, 8).to(torch.bfloat16)
    ok, _ = ps.outputs_equal([a, None], [a.clone(), None])
    assert ok
    b = a.clone()
    b.view(torch.int16)[1, 3] ^= 1  # one bit
    ok, why = ps.outputs_equal([a], [b])
    assert not ok and "differ" in why
    assert not ps.outputs_equal([a], [None])[0]
    assert not ps.outputs_equal([a], [a.float()])[0]


def test_candidate_and_decide_arm_the_updater(monkeypatch):
    monkeypatch.delenv(ps.ENV, raising=False)
    u, _ = _updater()
    d = ps.DraftPlanSkip()
    assert d.candidate(u, _Model([2048] * 5)) == (1,)
    d.decide(u, True, "test", (1,))
    assert u.skip_wrapper_ids == frozenset({1}) and u.probe_wrapper_ids == frozenset()
    assert d.candidate(u, _Model([2048] * 5)) == ()  # decided once per process
    u2, _ = _updater()
    d2 = ps.DraftPlanSkip()
    d2.decide(u2, False, "test", (1,))
    assert u2.skip_wrapper_ids == frozenset()
    u3, _ = _updater()
    d3 = ps.DraftPlanSkip()
    assert d3.candidate(u3, _Model([2048, -1])) == () and d3.verdict is False
    monkeypatch.setenv(ps.ENV, "0")
    assert ps.DraftPlanSkip().candidate(_updater()[0], _Model([2048] * 5)) == ()


def _worker_stub(updater, sampler_out):
    w = object.__new__(DFlashWorkerV2)
    w.__dict__["_draft_plan_skip"] = ps.DraftPlanSkip()
    w.__dict__["block_size"] = 8
    w.__dict__["_draft_sampler"] = types.SimpleNamespace(out=sampler_out)
    w.__dict__["_draft_plan_skip_updater"] = lambda: updater
    return w


def _out(hs, graph=True):
    return types.SimpleNamespace(can_run_graph=graph, logits_output=types.SimpleNamespace(hidden_states=hs))


def test_judge_arms_on_equal_and_refuses_on_difference():
    u, _ = _updater()
    hs = torch.randn(8, 16)
    w = _worker_stub(u, torch.arange(14))
    snap = w._draft_plan_skip_snapshot(_out(hs), 2)
    w._draft_plan_skip_judge(snap, _out(hs.clone()), 2, (1,))
    assert w.__dict__["_draft_plan_skip"].verdict is True and u.skip_wrapper_ids == frozenset({1})

    u2, _ = _updater()
    w2 = _worker_stub(u2, torch.arange(14))
    snap2 = w2._draft_plan_skip_snapshot(_out(hs), 2)
    hs2 = hs.clone()
    hs2[3, 5] += 1.0
    w2._draft_plan_skip_judge(snap2, _out(hs2), 2, (1,))
    assert w2.__dict__["_draft_plan_skip"].verdict is False and u2.skip_wrapper_ids == frozenset()


def test_judge_eager_rounds_decide_nothing_then_refuse():
    u, _ = _updater()
    w = _worker_stub(u, torch.arange(14))
    hs = torch.randn(8, 16)
    for _ in range(3):
        w._draft_plan_skip_judge(w._draft_plan_skip_snapshot(_out(hs, False), 2), _out(hs, False), 2, (1,))
        assert w.__dict__["_draft_plan_skip"].verdict is None
    w._draft_plan_skip_judge(w._draft_plan_skip_snapshot(_out(hs, False), 2), _out(hs, False), 2, (1,))
    assert w.__dict__["_draft_plan_skip"].verdict is False


def test_probe_resets_probe_ids_even_on_error():
    u, _ = _updater()
    w = _worker_stub(u, torch.arange(14))

    def _boom(_fb):
        assert u.probe_wrapper_ids == frozenset({1})
        raise RuntimeError("x")

    w.__dict__["draft_model_runner"] = types.SimpleNamespace(forward=_boom)
    assert w._draft_plan_skip_probe(lambda: None, (1,), 2) is None
    assert u.probe_wrapper_ids == frozenset() and w.__dict__["_draft_plan_skip"].verdict is False
