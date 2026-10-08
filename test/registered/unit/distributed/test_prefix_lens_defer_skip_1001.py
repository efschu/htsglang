"""nf-pd-post 01.10. (boot y6o): the #639 prefix-lens ballot of a skip-extend
batch is issued in ``prepare_for_extend`` but DECIDED after the batch's result.

Measured: D TP0's ``prepare_ms`` of the first post-wake pass is TP1's
``START-LOADING kv_issue_ms`` minus TP0's own, flip for flip (240 = 288 - 64,
244 = 280 - 47, 250 = 302 - 55, 273 = 324 - 54, 295 = 358 - 67): TP0 waited in
this ballot's all_reduce for the slowest worker's load-back issue before it
could stream P's token -- for a batch that runs NO target forward, i.e. enters
no collective whose shape the vector decides. The decision now follows the
result and precedes the next forward that can enter a collective.
"""
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.layers.dcp import prefix_lens_check as plc  # noqa: E402
from flliper.srt.layers.dcp.lockstep import (  # noqa: E402
    PrefixLensRankDivergence,
    prefix_lens_ballot,
)
from flliper.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from flliper.srt.managers.scheduler import Scheduler  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

DEGRADED = [[0], [2048], [2048]]


class _Work:
    def __init__(self, dist, t):
        self.dist, self.t, self.waited = dist, t, False

    def wait(self):
        self.waited = True
        self.dist.events.append("wait")
        self.dist._reduce(self.t)


class _FakeDist:
    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, per_rank):
        self.per_rank, self.events = per_rank, []

    def get_world_size(self, group=None):
        return len(self.per_rank)

    def _reduce(self, t):
        ballots = [prefix_lens_ballot(v) for v in self.per_rank]
        for i in range(t.numel()):
            t[i] = min(b[i] for b in ballots)

    def all_reduce(self, t, op=None, group=None, async_op=False):
        assert op is self.ReduceOp.MIN
        self.events.append("reduce_async" if async_op else "reduce")
        if async_op:
            return _Work(self, t)
        self._reduce(t)
        return None

    def all_gather_object(self, out, obj, group=None):
        self.events.append("gather")
        for i, v in enumerate(self.per_rank):
            out[i] = list(v)


@pytest.fixture
def dist(monkeypatch):
    plc._PENDING.clear()
    fake = _FakeDist([[0, 2048]] * 3)
    monkeypatch.setattr(torch, "distributed", fake)
    monkeypatch.setattr(plc, "_dcp_cpu_group", lambda: object())
    monkeypatch.setattr(plc, "_DEFER_ENABLED", True)
    yield fake
    plc._PENDING.clear()


def test_a_skip_ballot_is_issued_now_and_decided_later(dist):
    plc.assert_prefix_lens_rank_uniform([0, 2048], defer=True)
    assert dist.events == ["reduce_async"]          # issued at the vector, nobody waited
    assert plc.has_deferred()
    plc.resolve_deferred()
    assert dist.events == ["reduce_async", "wait"] and not plc.has_deferred()


def test_a_divergent_skip_ballot_still_refuses_on_every_rank(dist):
    for rank in range(3):
        dist.per_rank, dist.events = DEGRADED, []
        plc.assert_prefix_lens_rank_uniform(DEGRADED[rank], defer=True)   # no raise yet
        with pytest.raises(PrefixLensRankDivergence):
            plc.resolve_deferred()
        assert "gather" in dist.events
        assert not plc.has_deferred()


def test_an_ordinary_extend_decides_in_prepare_and_after_the_deferred_one(dist):
    plc.assert_prefix_lens_rank_uniform([0, 2048], defer=True)
    plc.assert_prefix_lens_rank_uniform([0, 2048])
    # the older ballot is waited for first, then the new one is reduced in place
    assert dist.events == ["reduce_async", "wait", "reduce"]
    assert not plc.has_deferred()


def test_the_switch_off_decides_in_prepare(dist, monkeypatch):
    monkeypatch.setattr(plc, "_DEFER_ENABLED", False)
    plc.assert_prefix_lens_rank_uniform([0, 2048], defer=True)
    assert dist.events == ["reduce"] and not plc.has_deferred()


def _sched_stub():
    import types

    return types.SimpleNamespace(_run_batch_forward=lambda b, p=None: "ran")


def test_run_batch_decides_before_a_forward_but_not_before_a_skip(dist):
    import types

    plc.assert_prefix_lens_rank_uniform([0, 2048], defer=True)
    skip = types.SimpleNamespace(pdflip_skip_extend=True)
    assert Scheduler.run_batch(_sched_stub(), skip) == "ran"
    assert plc.has_deferred()                        # the skip's own pass does not wait
    decode = types.SimpleNamespace(pdflip_skip_extend=False)
    assert Scheduler.run_batch(_sched_stub(), decode) == "ran"
    assert not plc.has_deferred() and dist.events[-1] == "wait"


def test_the_result_decides_and_prepare_defers_only_skips():
    src = inspect.getsource(Scheduler.process_batch_result)
    at = src.index("_prefix_lens_check.resolve_deferred()")
    assert src.index("process_batch_result_prefill(batch, result)") < at
    psrc = inspect.getsource(ScheduleBatch.prepare_for_extend)
    assert "assert_prefix_lens_rank_uniform(prefix_lens, defer=self.pdflip_skip_extend)" in psrc


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
