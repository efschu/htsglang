# SPDX-License-Identifier: Apache-2.0
"""#1580 ABORT-EXACT-RID (dual layout): an abort naming ``weg2-<epoch>-<n>`` matches that rid only.

The scheduler matched ``req.rid.startswith(abort.rid)``; the front's rids are a counter, so the abort
of ``weg2-0-1`` also hit the live ``weg2-0-10``..``weg2-0-19`` and ``weg2-0-100``.. (``weg2-0-8`` hit 80-89).
Boot b9h/b9i show 0 such hits in 89 abort events (deskq/done/1580-abort-prefix-match.md): the collision
is latent (an old rid aborted after a longer rid went live, i.e. at digit boundaries), not the B9b death
(that is the SAME-rid reuse family, Q-698 / POP-KEEPS-TWIN).

DANGER DIRECTIONS guarded here:
* default (switch off) = upstream prefix rule on every path, byte for byte;
* switch on but NOT a dual-layout process (flip / INT8 / NF forms) = prefix rule unchanged;
* switch on + dual layout: exact rid for the weg2 shape on every migrated site (chunked, waiting queue,
  grammar/retracted/disagg queues, anchor tails, intake-stall, p_intake, d_park, tail_adopt, tail_handoff,
  the follower's waiting-abort hold);
* a non-weg2 rid (client-chosen, batch sub-rids) keeps the prefix rule even when armed;
* abort_all and the empty rid are untouched (the callers guard them).
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import anchor_tails as AT  # noqa: E402
from sglang.srt.weg2 import abort_match as AM  # noqa: E402
from sglang.srt.weg2 import d_park_runtime as DP  # noqa: E402
from sglang.srt.weg2 import intake_stall as IS  # noqa: E402
from sglang.srt.weg2 import p_intake as PI  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ENV = "SGLANG_WEG2_ABORT_EXACT_RID"
ARMED = {"SGLANG_WEG2_DUAL_LAYOUT": "1", ENV: "1"}
NOT_DUAL = {"SGLANG_WEG2_DUAL_LAYOUT": "", ENV: "1"}
OFF = {"SGLANG_WEG2_DUAL_LAYOUT": "1", ENV: ""}


@pytest.fixture
def setenv(monkeypatch):
    def _set(env):
        for k, v in env.items():
            if v == "":
                monkeypatch.delenv(k, raising=False)
            else:
                monkeypatch.setenv(k, v)
        # the EnvBool reads os.environ lazily on every .get(); nothing cached here
    return _set


# -- the rule itself -------------------------------------------------------------------------------

def test_prefix_collision_is_real_by_default(setenv):
    """The bug: default (off) the abort of weg2-0-1 hits weg2-0-10, -19, -100."""
    setenv(OFF)
    for live in ("weg2-0-1", "weg2-0-10", "weg2-0-19", "weg2-0-100"):
        assert AM.rid_hit(live, "weg2-0-1") is True
    assert AM.rid_hit("weg2-0-20", "weg2-0-1") is False


def test_armed_matches_exactly(setenv):
    setenv(ARMED)
    assert AM.exact_armed() is True
    assert AM.rid_hit("weg2-0-1", "weg2-0-1") is True
    for live in ("weg2-0-10", "weg2-0-19", "weg2-0-100", "weg2-0-1000"):
        assert AM.rid_hit(live, "weg2-0-1") is False
    # the digit boundaries the report names: 8 vs 80-89, 9 vs 90-99
    assert AM.rid_hit("weg2-0-85", "weg2-0-8") is False
    assert AM.rid_hit("weg2-0-95", "weg2-0-9") is False
    assert AM.rid_hit("weg2-1-1", "weg2-0-1") is False        # another epoch


@pytest.mark.parametrize("env", [OFF, NOT_DUAL], ids=["switch-off", "not-dual-layout"])
def test_default_and_non_dual_keep_the_prefix_rule(setenv, env):
    setenv(env)
    assert AM.exact_armed() is False
    assert AM.rid_hit("weg2-0-19", "weg2-0-1") is True
    assert AM.rid_hit("abc123", "abc") is True
    assert AM.rid_hit("other", "abc") is False


def test_non_weg2_rids_keep_the_prefix_rule_even_armed(setenv):
    """Client-chosen rids and batch sub-rids (rid + suffix) are not the front's shape."""
    setenv(ARMED)
    assert AM.rid_hit("req-17_0", "req-17") is True
    assert AM.rid_hit("weg2-0-17-sub", "weg2-0-17") is False  # weg2 shape named -> exact, a sub-rid is not issued by the front
    assert AM.rid_hit("weg2-0-17", "weg2-0") is True           # not the full shape -> prefix, as before
    assert AM.rid_hit("anything", "") is True                  # empty rid: callers guard it; rule unchanged


def test_rank_uniform_pure_function(setenv):
    """No clock, no per-rank state: the same strings give the same answer on every call."""
    setenv(ARMED)
    answers = {AM.rid_hit("weg2-0-12", "weg2-0-1") for _ in range(50)}
    assert answers == {False}


# -- the migrated sites ----------------------------------------------------------------------------

def _tail(rid):
    return types.SimpleNamespace(rid=rid)


def test_anchor_tails_abort_targets(setenv):
    tails = [_tail("weg2-0-1"), _tail("weg2-0-10"), _tail("weg2-0-100")]
    setenv(OFF)
    assert [t.rid for t in AT.abort_targets(tails, rid="weg2-0-1", abort_all=False)] == [
        "weg2-0-1", "weg2-0-10", "weg2-0-100"]
    setenv(ARMED)
    assert [t.rid for t in AT.abort_targets(tails, rid="weg2-0-1", abort_all=False)] == ["weg2-0-1"]
    assert len(AT.abort_targets(tails, rid="weg2-0-1", abort_all=True)) == 3


def test_intake_stall_forget(setenv):
    for env, left in ((OFF, set()), (ARMED, {"weg2-0-10", "weg2-0-100"})):
        setenv(env)
        w = IS.IntakeStallWatch()
        w._reported = {"weg2-0-1", "weg2-0-10", "weg2-0-100"}
        w._rid = "weg2-0-10"
        w.forget("weg2-0-1")
        assert w._reported == left
        assert (w._rid is None) == (env is OFF)


def test_p_intake_forget_named(setenv):
    for env, left in ((OFF, set()), (ARMED, {"weg2-0-10", "weg2-0-100"})):
        setenv(env)
        sched = types.SimpleNamespace(_h91_intake_named={"weg2-0-1", "weg2-0-10", "weg2-0-100"})
        PI.forget(sched, "weg2-0-1")
        assert sched._h91_intake_named == left


def test_d_park_abort(setenv):
    class _R:
        def __init__(self, rid):
            self.rid = rid

    sent = []
    for env, expect in ((OFF, 3), (ARMED, 1)):
        setenv(env)
        sent.clear()
        ch = types.SimpleNamespace(send_to_tokenizer=types.SimpleNamespace(
            send_output=lambda out, req: sent.append(out.rid)))
        sched = types.SimpleNamespace(
            weg2_d_parked=[_R("weg2-0-1"), _R("weg2-0-10"), _R("weg2-0-100")],
            ipc_channels=ch, enable_hicache_storage=False)
        with mock.patch("sglang.srt.weg2.tail_handoff.park_end_enabled", return_value=False), \
                mock.patch.object(DP.d_park_draft, "drop_all"):
            n = DP.park_abort(sched, types.SimpleNamespace(rid="weg2-0-1", abort_all=False))
        assert n == expect
        assert sent[0] == "weg2-0-1"
        assert len(sched.weg2_d_parked) == 3 - expect


def test_scheduler_sites_all_migrated():
    """Every startswith(recv_req.rid) of _abort_request_now and the waiting-abort hold went through
    rid_hit: a new bare prefix match would silently reintroduce the collision."""
    import inspect

    from sglang.srt.managers import scheduler as SC

    src = inspect.getsource(SC.Scheduler._abort_request_now)
    assert ".startswith(recv_req.rid)" not in src
    assert src.count("_abort_match.rid_hit(") >= 7
    for name in ("_weg2_process_waiting_aborts", "_weg2_defer_waiting_abort"):
        fn = getattr(SC.Scheduler, name, None)
        if fn is not None:
            assert ".rid).startswith(rid)" not in inspect.getsource(fn)
