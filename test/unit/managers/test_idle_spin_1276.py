"""#1276: an idle rank waits instead of spinning.

Boot weg2sb4 (e4f1b9fcc6) ran 1,499.8 HICACHE-ROUND/s on group P and 303.1/s
on group D with an empty queue and a live front, for 34 idle minutes.  The
#547 blocking-poll ladder was fully built and simply never constructed: the
boot set neither --sleep-on-idle nor SGLANG_IDLE_BLOCKING_POLL, so
`Scheduler.idle_sleeper` was None on all six ranks.

These tests pin the three facts the fix rests on:
  (a) the ladder itself parks a genuinely quiet loop and costs nothing on the
      first rung (so leaving a loaded phase is byte-identical);
  (b) the weg-2 form now ARMS the ladder, on the request origin only;
  (c) the round census still runs its per-pass WORK every pass -- only the
      LOG LINE became state-change-driven.  (c) is the regression guard for
      the fix-1c / fix-2 per-pass semantics: a queued control object must
      still get its lap within one pass.
"""

import types

import pytest

from sglang.srt.managers.scheduler_components.idle_sleeper import (
    IDLE_POLL_CAP_MS,
    IdleSleeper,
    idle_poll_timeout_ms,
)


# ---------------------------------------------------------------- (a) ladder
def test_first_rung_does_not_poll_at_all():
    """Rung 0 must return 0 = 'skip the syscall', not 'poll with 0 timeout'."""
    assert idle_poll_timeout_ms(1) == 0
    assert idle_poll_timeout_ms(32) == 0


def test_ladder_steps_up_and_caps():
    assert idle_poll_timeout_ms(33) == 1
    assert idle_poll_timeout_ms(160) == 1
    assert idle_poll_timeout_ms(161) == 10
    assert idle_poll_timeout_ms(288) == 10
    assert idle_poll_timeout_ms(289) == IDLE_POLL_CAP_MS
    assert idle_poll_timeout_ms(10_000_000) == IDLE_POLL_CAP_MS


def test_cap_bounds_the_idle_round_rate():
    """The cap is what turns 1,499.8 rounds/s into ~20 rounds/s."""
    assert 1000.0 / IDLE_POLL_CAP_MS <= 25.0


def test_reset_returns_to_the_zero_poll_rung():
    slp = IdleSleeper.__new__(IdleSleeper)
    slp.idle_ticks = 5000
    IdleSleeper.reset(slp)
    assert slp.idle_ticks == 0
    assert idle_poll_timeout_ms(slp.idle_ticks + 1) == 0


# ------------------------------------------------------- (b) the weg-2 arming
def _scheduler_stub(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0):
    s = types.SimpleNamespace()
    s.ps = types.SimpleNamespace(
        pp_rank=pp_rank, attn_tp_rank=attn_tp_rank, attn_cp_rank=attn_cp_rank
    )
    s.server_args = types.SimpleNamespace(sleep_on_idle=False)
    s.ipc_channels = types.SimpleNamespace(
        recv_from_tokenizer=object(), recv_from_rpc=object()
    )
    return s


def _arms(monkeypatch, weg2_group, **rank):
    """Run the real init_idle_sleeper against a stub; return whether it armed."""
    from sglang.srt.managers.scheduler import Scheduler

    if weg2_group is None:
        monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_GROUP", weg2_group)
    s = _scheduler_stub(**rank)
    made = {}

    class _FakeSleeper:
        def __init__(self, sockets):
            made["sockets"] = sockets

    monkeypatch.setattr(
        "sglang.srt.managers.scheduler.IdleSleeper", _FakeSleeper, raising=True
    )
    Scheduler.init_idle_sleeper(s)
    return s.idle_sleeper is not None, made


def test_weg2_group_arms_the_idle_poll_on_the_origin(monkeypatch):
    """THE #1276 FIX. Before it this was False and the loop spun."""
    armed, made = _arms(monkeypatch, "P")
    assert armed
    # It must poll BOTH intake sockets: the flip RPC from the front arrives on
    # recv_from_rpc, and a rank parked on the tokenizer socket alone would
    # sleep through a flip.
    assert len(made["sockets"]) == 2


def test_without_the_weg2_group_the_stock_default_is_untouched(monkeypatch):
    armed, _ = _arms(monkeypatch, None)
    assert not armed


@pytest.mark.parametrize(
    "rank",
    [dict(pp_rank=1), dict(pp_rank=2), dict(attn_tp_rank=1), dict(attn_cp_rank=1)],
)
def test_followers_never_arm_their_own_sleeper(monkeypatch, rank):
    """Only the request ORIGIN owns the zmq sockets.  Followers take a blocking
    chain receive / broadcast and are driven at the origin's cadence, so
    parking the origin parks the group -- one wait point, not six."""
    armed, _ = _arms(monkeypatch, "P", **rank)
    assert not armed


# --------------------------------------------- (c) per-pass semantics intact
class _RoundStub:
    """Minimal stand-in for UnifiedRadixCache on the check_hicache_events path."""

    def __init__(self, prefetch=0):
        self.pp_rank, self.pp_size = 0, 3
        self.ongoing_prefetch = {i: None for i in range(prefetch)}
        self._pin_trace_every = 0
        self.enable_storage = True
        self.enable_storage_metrics = False
        self.storage_metrics_collector = None
        self.calls = {"drain_async": 0, "writing": 0, "loading": 0, "control": 0}

    def _attn_reduce_world(self):
        return 1

    def _drain_async_work(self):
        self.calls["drain_async"] += 1

    def writing_check(self):
        self.calls["writing"] += 1

    def loading_check(self):
        self.calls["loading"] += 1

    def drain_storage_control_queues(self):
        self.calls["control"] += 1


def _run(stub, n):
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    for _ in range(n):
        UnifiedRadixCache.check_hicache_events(stub)


def test_every_pass_still_does_its_work(caplog):
    """THE REGRESSION GUARD.  The logging change must not gate the work: a
    queued control object still gets its lap within one pass."""
    stub = _RoundStub()
    _run(stub, 100)
    assert stub.calls == {
        "drain_async": 100,
        "writing": 100,
        "loading": 100,
        "control": 100,
    }
    # the census counter itself is unchanged -- it still counts every round
    assert stub._1028_round == 100


def test_quiet_rounds_do_not_each_write_a_line(caplog):
    """Before #1276 this wrote a line every 25th round: 4 lines per 100."""
    caplog.set_level("INFO")
    stub = _RoundStub()
    _run(stub, 100)
    lines = [r for r in caplog.records if "HICACHE-ROUND" in r.getMessage()]
    assert len(lines) == 1, f"expected one line (the first round), got {len(lines)}"


def test_a_state_change_is_reported_immediately(caplog):
    caplog.set_level("INFO")
    stub = _RoundStub()
    _run(stub, 10)
    before = len([r for r in caplog.records if "HICACHE-ROUND" in r.getMessage()])
    stub.ongoing_prefetch = {0: None, 1: None}  # work arrived
    _run(stub, 1)
    after = [r for r in caplog.records if "HICACHE-ROUND" in r.getMessage()]
    assert len(after) == before + 1
    assert "ongoing_prefetch=2" in after[-1].getMessage()


def test_the_probe_can_never_break_the_round():
    """The census is wrapped: a broken attribute must not stop the pass."""
    stub = _RoundStub()
    stub._attn_reduce_world = lambda: 1 / 0  # noqa: E731
    _run(stub, 3)
    assert stub.calls["control"] == 3
