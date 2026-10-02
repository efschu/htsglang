"""DP-NACHLAUF 02.10.: the last decode rounds are written at the sleep, not
after the next wake.

N6i (f501e462d0) 27B D: rounds 285-331 of 18:14:46-47 reached the log at
18:15:28, after the next wake -- flush() is query-only and a sleeping group
runs no rounds or idle ticks. Pinned (red before): drain_blocking retires the
open round, syncs once and emits every pending round in order; nothing
pending = no sync; the switch off drains nothing; the sleep leg calls it after
the replay check.
"""
import inspect
import logging
import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import decode_round_log as D  # noqa: E402


class _Clock:
    def __init__(self):
        self.ready = False

    def harvest_round(self, span):
        if not self.ready:
            return None
        return SimpleNamespace(round_ms=10.0, wait_ms=1.0, families={}, split_refused=None)


def _log_with_pending(n=3):
    clk = _Clock()
    log = D.DecodeRoundLog(clock=clk)
    for i in range(n):
        log._pending[i] = SimpleNamespace(round_id=i, bs=1, rows=1, spans=[("s", "decode", True)],
                                          categories=[], wall=1790960000.0 + i, depth=None)
    return log, clk


def test_drain_syncs_once_and_emits_in_order(monkeypatch, caplog):
    monkeypatch.delenv(D.DecodeRoundLog.DRAIN_AT_SLEEP_ENV, raising=False)
    log, clk = _log_with_pending(3)
    log.flush()
    assert len(log._pending) == 3                          # query-only: not readable yet
    calls = []

    def sync():
        calls.append(1)
        clk.ready = True                                   # the device finished

    with caplog.at_level(logging.INFO):
        assert log.drain_blocking(sync=sync) == 3
    assert calls == [1] and not log._pending and log.cum_rounds == 3
    rounds = [r.getMessage() for r in caplog.records if "Decode rank batch" in r.getMessage()]
    assert [m.split("#round: ")[1].split(",")[0] for m in rounds] == ["0", "1", "2"]
    assert any("DECODE-ROUND-LOG drained n=3" in r.getMessage() for r in caplog.records)


def test_nothing_pending_no_sync_and_switch_off(monkeypatch):
    monkeypatch.delenv(D.DecodeRoundLog.DRAIN_AT_SLEEP_ENV, raising=False)
    log = D.DecodeRoundLog(clock=_Clock())
    calls = []
    assert log.drain_blocking(sync=lambda: calls.append(1)) == 0 and calls == []
    log, clk = _log_with_pending(2)
    monkeypatch.setenv(D.DecodeRoundLog.DRAIN_AT_SLEEP_ENV, "0")
    assert log.drain_blocking(sync=lambda: calls.append(1)) == 0 and calls == [] and len(log._pending) == 2


def test_sleep_leg_drains_after_the_replay_check():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu.WeightUpdater.release_memory_occupation) if hasattr(wu, "WeightUpdater") else inspect.getsource(wu)
    i_rep = src.index("if replay is not None:")
    i_drain = src.index("_drl.drain_blocking()")
    assert i_rep < i_drain < src.index("weg2_per_tag: Dict[str, List[float]] = {}")
