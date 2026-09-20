"""fnFA12/13/14 (20.09.): the sampler's rank-0 token sync (#622) must not run
on the Form A host -- the workers never sample, so the broadcast has no
receiver. NCCL op log fnFA14: host 195 all_reduce + 1 broadcast(1 int64) +
..., workers 195 all_reduce and no such broadcast; that one op wedged the
host's stream (fnFA13 watchdog) and aborted the Bar1 spin kernel (fnFA12)."""

import types

from sglang.srt.layers import sampler as sm
from sglang.srt.rank_role import HOST, WORKER, RankRolePlan, set_form_a_role_plan

FORM_A = RankRolePlan((HOST, WORKER, WORKER))


def _sampler(calls):
    s = sm.Sampler.__new__(sm.Sampler)
    s._tp_sync_coordinator = types.SimpleNamespace(world_size=3)
    s.tp_sync_group = object()
    return s


def test_form_a_host_skips_the_token_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(sm, "maybe_sync_sampled_tokens", lambda *a, **k: calls.append("bcast"))
    monkeypatch.setattr(sm, "SGLANG_SYNC_SAMPLED_TOKENS", True)
    s = _sampler(calls)
    info = types.SimpleNamespace(grammars=None)
    try:
        set_form_a_role_plan(FORM_A, rank=0)
        s._sync_token_ids_across_tp(object(), info)
        assert calls == []
    finally:
        set_form_a_role_plan(None)
    s._sync_token_ids_across_tp(object(), info)
    assert calls == ["bcast"]
