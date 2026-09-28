"""H24c: several H24 skip-extends (E2) share ONE post-wake pass.

Hermetic (no CUDA). Metal rc12z26 (D log boot_...dauer09281752, TP0): after
the wake at 18:05:34 three resumes arrived with P's END state -- weg2-3-8,
weg2-2-7, weg2-3-9 -- and took three passes ('WEG2-POST-WAKE-PASS n=0/1/2
mode=EXTEND bs=1', gaps 217/82 ms), 18:06:34 two (weg2-6-13, weg2-6-14).
None of them runs a target forward; the passes were the adder closing the
batch behind the first skip ('weg2_skip_extend_taken' in budget_state and at
the top of add_one_req) although the worker already serves a batch of skips
(tail_adopt.skip_tokens returns one token per request, run_skip installs
each). 14 further requests in that boot lost their END state outright
('adopt=skipped:end_only:batch_not_empty') and ran a real extend.

What these cases pin:

* behind a skip the adder stays open (budget_state CONTINUE) to a request
  that takes the END state, and refuses every other one BEFORE any match or
  load-back side effect;
* the exact check (page prefix included) runs at the commit, and a batch
  holding only skips counts as empty for the next skip's plan_adopt;
* ``skip_joinable`` answers without taking the agreed entry (no pop);
* the worker's skip entry serves a batch of several skips (guard: green on
  the base too).
"""

import inspect
import logging
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers import schedule_policy as sp
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.weg2 import tail_adopt as ta
from sglang.srt.weg2 import tail_handoff as th

PAGE, RATIO = 64, 4
N = 241  # c = 240, page prefix 192
PREFIX = 192


def _ids(seed):
    return [(seed * 7919 + i) % 151000 for i in range(N)]


def _req(rid, seed, **kw):
    ids = _ids(seed)
    spp = SimpleNamespace(frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0,
                          min_new_tokens=0, ignore_eos=False)
    base = dict(rid=rid, origin_input_ids=ids, full_untruncated_fill_ids=list(ids), extra_key=None,
                prefix_indices=torch.arange(PREFIX, dtype=torch.int64), return_logprob=False,
                return_hidden_states=False, grammar=None, sampling_params=spp)
    base.update(kw)
    return SimpleNamespace(**base)


def _agreed(req, skip=True, agreed=True):
    spec = th.spec_for(req.rid, req.origin_input_ids, None, PAGE, RATIO)
    assert spec.page_prefix == PREFIX
    end = SimpleNamespace(key=th.tail_key(req.origin_input_ids, N, None))
    staged = SimpleNamespace(spec=spec, headers=[SimpleNamespace(end=end)], e1=True,
                             end_ok=True, drop_end=lambda: None)
    return ta.Agreed(staged=staged, agreed=agreed, skip=skip)


@pytest.fixture
def agreed(monkeypatch):
    box = {}
    monkeypatch.setattr(ta, "_AGREED", box)
    with envs.SGLANG_WEG2_TAIL_HANDOFF.override(True), envs.SGLANG_WEG2_TAIL_ADOPT.override(True):
        yield box


# ------------------------------------------------------------ skip_joinable
def test_joinable_names_the_skip_without_taking_it(agreed):
    a, b = _req("weg2-3-8", 1), _req("weg2-2-7", 2)
    agreed[a.rid], agreed[b.rid] = _agreed(a), _agreed(b)
    assert ta.skip_joinable(b)
    assert ta.skip_joinable(b, PREFIX)
    assert set(agreed) == {a.rid, b.rid}  # a peek: the commit still finds its entry
    assert not ta.skip_joinable(b, PREFIX + PAGE)  # another matched prefix -> not this skip


@pytest.mark.parametrize("what", ["e1", "unagreed", "none", "logprob", "penalty", "grammar", "key"])
def test_joinable_refuses_what_would_run_a_forward(agreed, what):
    kw = {"logprob": {"return_logprob": True}, "grammar": {"grammar": object()},
          "penalty": {"sampling_params": SimpleNamespace(frequency_penalty=0.5, presence_penalty=0.0,
                                                         repetition_penalty=1.0, min_new_tokens=0,
                                                         ignore_eos=False)}}.get(what, {})
    r = _req("weg2-3-9", 3, **kw)
    if what != "none":
        agreed[r.rid] = _agreed(r, skip=what != "e1", agreed=what != "unagreed")
    if what == "key":
        r.origin_input_ids = list(r.origin_input_ids[:-1]) + [r.origin_input_ids[-1] + 1]
        r.full_untruncated_fill_ids = list(r.origin_input_ids)
    assert not ta.skip_joinable(r)


def test_joinable_is_off_without_adopt(agreed):
    r = _req("weg2-3-9", 3)
    agreed[r.rid] = _agreed(r)
    with envs.SGLANG_WEG2_TAIL_ADOPT.override(False):
        assert not ta.skip_joinable(r)


# --------------------------------------------------------------- the adder
class _Adder(PrefillAdder):
    rem_total_tokens = 1 << 20
    cur_rem_tokens = 1 << 20


class _Delayer:
    def __init__(self):
        self.calls = 0

    def negotiate_should_allow_prefill(self, **_kw):
        self.calls += 1
        return False  # stop right behind the gate under test


def _adder(skip_taken=True):
    ad = object.__new__(_Adder)
    ad.prefill_spill_deep_taken = False
    ad.weg2_skip_extend_taken = skip_taken
    ad.is_hybrid_swa = False
    ad.rem_mamba_slots = None
    ad.rem_input_tokens = 1 << 20
    ad.rem_chunk_tokens = None
    ad.dllm_config = None
    ad.can_run_list = [object()] if skip_taken else []
    ad.prefill_delayer_single_pass = _Delayer()
    ad.running_batch = SimpleNamespace(batch_size=lambda: 0)
    ad.max_prefill_bs = ad.max_running_requests = ad.waiting_queue_len = 6
    return ad


def test_a_skip_leaves_the_batch_open_for_the_pass(agreed):
    """rc12z26 18:05:34: the adder returned OTHER right after the first skip,
    so the scheduler loop broke and the next skip waited a whole pass."""
    assert _adder().budget_state() == AddReqResult.CONTINUE


def test_a_second_skip_passes_the_gate_behind_a_skip(agreed):
    b = _req("weg2-2-7", 2)
    agreed[b.rid] = _agreed(b)
    ad = _adder()
    assert ad.add_one_req(b, truncation_align_size=None) == AddReqResult.OTHER
    assert ad.prefill_delayer_single_pass.calls == 1  # it got past the H24 gate
    assert b.rid in agreed  # nothing taken yet


@pytest.mark.parametrize("what", ["e1", "none"])
def test_a_request_needing_a_forward_waits_behind_a_skip(agreed, what):
    r = _req("weg2-6-14", 4)
    if what == "e1":
        agreed[r.rid] = _agreed(r, skip=False)
    ad = _adder()
    assert ad.add_one_req(r, truncation_align_size=None) == AddReqResult.OTHER
    assert ad.prefill_delayer_single_pass.calls == 0  # refused before any side effect


def test_the_spill_deep_batch_stays_closed(agreed):
    ad = _adder(skip_taken=False)
    ad.prefill_spill_deep_taken = True
    assert ad.budget_state() == AddReqResult.OTHER
    b = _req("weg2-2-7", 2)
    agreed[b.rid] = _agreed(b)
    assert ad.add_one_req(b, truncation_align_size=None) == AddReqResult.OTHER
    assert ad.prefill_delayer_single_pass.calls == 0


def test_commit_checks_the_matched_prefix_and_plans_as_empty():
    """Wiring: the exact gate (matched prefix, chunk/dllm paths) sits before
    the plan, and the plan of a skip behind skips sees an 'empty' batch."""
    src = inspect.getsource(PrefillAdder.add_one_req)
    gate = src.index("tail_adopt.skip_joinable(req, len(req.prefix_indices))")
    plan = src.index("_tail = tail_adopt.plan_adopt(")
    assert gate < plan
    assert "batch_empty=not self.can_run_list or self.weg2_skip_extend_taken" in src[plan:plan + 200]
    assert "input_tokens > self.rem_chunk_tokens" in src[src.rindex("if self.weg2_skip_extend_taken", 0, gate):gate]


# -------------------------------------------------------------- the worker
def test_the_worker_serves_a_batch_of_skips(monkeypatch, caplog):
    """Guard (green on the base as well): the skip entry of EAGLEWorkerV2
    already returns one token per request and installs each."""
    plans = {}
    monkeypatch.setattr(ta, "SKIP_PLANS", plans)
    reqs = [_req("weg2-3-8", 1), _req("weg2-2-7", 2), _req("weg2-3-9", 3)]
    for i, r in enumerate(reqs):
        spec = th.spec_for(r.rid, r.origin_input_ids, None, PAGE, RATIO)
        plans[r.rid] = ta.SkipPlan(spec=spec, first_token=100 + i, install=None, t0=0.0)
    batch = SimpleNamespace(reqs=reqs, hicache_consumer_index=-1)
    assert ta.skip_tokens(batch) == [100, 101, 102]
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.tail_adopt"):
        ta.run_skip(batch, counter=None)
    assert not plans
    assert caplog.text.count("WEG2-TAIL-SKIP-EXTEND rid=") == 3
