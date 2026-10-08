# SPDX-License-Identifier: Apache-2.0
"""#1420r DEFER-REARM: D-HANDBACK-DEFER re-arms a mark whose read was issued but landed
empty (P's tail write-through is asynchronous), instead of spending it on the second W31.

Metal B9 (desk 1420): 24/24 reroutes = 24/24 ``D-HANDBACK-DEFER state=refused``; the single
re-read landed empty at once, the second W31 spent the mark, the whole request went back over
P (~58 s). Rank agreement: the decision is pass-counted (W31 group verdict + retry's
pass-counted ``issued`` + the ``rearm`` counter), no wall clock.

DANGER DIRECTIONS (mutants run by hand on the module, both red: limit check removed -> the limit test
fails; default 0 -> 2 -> the default-off tests fail):
* default OFF (env unset / 0) = the old single-shot behaviour (second W31 spends the mark);
* ON: the re-arm engages until the limit and NOT beyond (mutant: no limit -> never spends);
* gate: no effect off the dual D (not armed), and none before a read was issued.
"""
from __future__ import annotations

import inspect
import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.srt.pdflip import dual_handback_defer as HB
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DUAL_D = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}
REARM = "FLLIPER_PDFLIP_DUAL_HANDBACK_DEFER_REARM"


def _req(rid="pdflip-0-5"):
    return types.SimpleNamespace(rid=rid, output_ids=[], kv_arrival_seq=1, _pp_store_presence_cache="neg",
                                 _pdflip_store_match_cache="neg")


def _sched(req, verdict="issued"):
    return types.SimpleNamespace(waiting_queue=[req], _prefetch_kvcache=lambda r: verdict)


@pytest.fixture()
def dual_d(monkeypatch):
    for k, v in DUAL_D.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv(REARM, raising=False)
    HB._REARM_LAST_LOG[0] = None


def _issue(req):
    """Drive retry() until the mark's read is issued (pass-counted back-off)."""
    for _ in range(40):
        HB.retry(_sched(req))
        if req._pdflip_hb_defer["issued"]:
            return
    raise AssertionError("read never issued")


def test_default_off_second_w31_spends(dual_d):
    r = _req()
    assert HB.begin(r, 3169) is True
    _issue(r)
    assert HB.begin(r, 3169) is False          # old behaviour: spent -> refused
    assert r._pdflip_hb_defer["spent"] is True
    assert r._pdflip_hb_defer.get("rearm", 0) == 0


def test_default_off_zero_and_garbage_are_off(dual_d, monkeypatch):
    for v in ("0", "", "abc", "-3"):
        monkeypatch.setenv(REARM, v)
        assert HB.rearm_max() == 0


def test_rearm_engages_until_limit_and_not_beyond(dual_d, monkeypatch, caplog):
    monkeypatch.setenv(REARM, "3")
    r = _req()
    assert HB.begin(r, 3169) is True
    with caplog.at_level(logging.WARNING):
        for k in range(1, 4):
            _issue(r)
            assert HB.begin(r, 3169) is True    # re-armed, not spent
            st = r._pdflip_hb_defer
            assert st["rearm"] == k and st["issued"] is False and not st["spent"]
            assert r._pp_store_presence_cache is None   # negative verdict dropped
            assert HB.pending(r, 1e9) is True           # the vote stands again
        _issue(r)
        assert HB.begin(r, 3169) is False       # limit 3 reached -> spent
    assert r._pdflip_hb_defer["spent"] is True and r._pdflip_hb_defer["rearm"] == 3
    lines = [m for m in caplog.messages if m.startswith("#1420r DEFER-REARM")]
    assert lines and "rearm=3 max=3" in lines[-1] and "rid=pdflip-0-5" in lines[-1]
    assert len(lines) <= 3


def test_rearm_log_is_rate_limited(dual_d, monkeypatch, caplog):
    monkeypatch.setenv(REARM, "50")
    r = _req()
    HB.begin(r, 10)
    clock = [100.0]
    with caplog.at_level(logging.WARNING):
        for _ in range(10):
            _issue(r)
            HB.begin(r, 10, now=lambda: clock[0])
    assert len([m for m in caplog.messages if m.startswith("#1420r DEFER-REARM")]) == 1


def test_gate_not_dual_d_no_effect(monkeypatch):
    monkeypatch.setenv(REARM, "5")
    for k in DUAL_D:
        monkeypatch.delenv(k, raising=False)
    r = _req()
    assert HB.begin(r, 3169) is False
    assert not hasattr(r, "_pdflip_hb_defer")
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert HB.begin(r, 3169) is False


def test_gate_no_rearm_before_read_issued_or_after_done(dual_d, monkeypatch):
    monkeypatch.setenv(REARM, "5")
    r = _req()
    HB.begin(r, 10)
    assert HB.begin(r, 10) is False             # never issued -> old path (spent)
    r2 = _req("pdflip-0-6")
    HB.begin(r2, 10)
    _issue(r2)
    HB.note_admit(r2)                           # admitted -> episode over
    assert HB.begin(r2, 10) is False


def test_rank_agreement_no_wallclock_in_decision():
    src = inspect.getsource(HB._rearm)
    decision = src.split("limit = rearm_max(env)")[1].split("st[\"rearm\"] = done + 1")[0]
    assert "now(" not in decision and "time." not in decision
