# SPDX-License-Identifier: Apache-2.0
"""L15-12c-SW: the refill kill switch SGLANG_WEG2_L15_REFILL (default 0).

The remote head must never carry an open TP0 gate without a switch
(operator order 01.10. ~19:10Z): with the switch OFF the cap-0 rank votes
None exactly as pre-E2, EVEN when every span has anchor_l2_slot >= 0; with
the switch ON the E2/E2b anchor gate decides, unchanged. The cap-0 wake
harness (fakes, env, spans) is reused from test_weg2_l15_cap0_wake_1001.py
by file-path import, so these tests run with or without the E2a commits.

Boot ladder (see the helper comment in weight_updater.py): boot 1 with
L15=1 and REFILL=0 (TP0 fallback), boot 2 with REFILL=1 only after a
green boot 1; the default flips to 1 after the metal proof.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))

_HERE = pathlib.Path(__file__).resolve().parent
_CAP0_TEST = _HERE / "test_weg2_l15_cap0_wake_1001.py"

_spec = importlib.util.spec_from_file_location(
    "test_weg2_l15_cap0_wake_1001", str(_CAP0_TEST)
)
cap0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap0)

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402

WU = wu.SchedulerWeightUpdaterManager
KEEP = cap0.KEEP
LOGGER = "sglang.srt.managers.scheduler_components.weight_updater"
REFILL = "SGLANG_WEG2_L15_REFILL"

_SPANS = cap0._SPANS  # rid "a", anchor_l2_slot=9 >= 0 -> E2b gate WOULD open


def _man():
    return cap0._fake_manifest(_SPANS)


def _run(monkeypatch, tmp_path, m, log_ctx=None, rank=0):
    host = cap0._HostPool({10: 5, 11: 5})
    mamba = cap0._MambaHostPool({9: 5})
    sched = cap0._Sched(host, mamba)
    fs = cap0._fake_self(sched, rank)
    cap0._gate_open(monkeypatch, m)
    if log_ctx is None:
        out = WU._l15_wake_hold_signal(fs)
    else:
        with log_ctx:
            out = WU._l15_wake_hold_signal(fs)
    return sched, fs, out


def test_refill_unset_votes_none_even_with_anchor_identity(
    monkeypatch, tmp_path, caplog
):
    cap0._env(monkeypatch, tmp_path, master=True, mib="c1=64")
    monkeypatch.delenv(REFILL, raising=False)
    ctx = caplog.at_level(logging.INFO, logger=LOGGER)
    sched, fs, out = _run(monkeypatch, tmp_path, _man(), log_ctx=ctx)
    assert out == (None, 0, 0, True), out
    assert fs._l15_wake_manifest is None          # not stashed
    assert not fs._l15_wake_refill                  # not a refill rank
    assert sched.token_to_kv_pool_allocator.clears == 0  # pools untouched
    msgs = [r.getMessage() for r in caplog.records]
    assert any("SGLANG_WEG2_L15_REFILL" in msg and "rank=0" in msg and "off" in msg
               for msg in msgs), msgs
    assert not any("anchors-missing" in msg for msg in msgs), msgs


def test_refill_zero_votes_none(monkeypatch, tmp_path):
    cap0._env(monkeypatch, tmp_path, master=True, mib="c1=64")
    monkeypatch.setenv(REFILL, "0")
    _sched, fs, out = _run(monkeypatch, tmp_path, _man())
    assert out == (None, 0, 0, True), out
    assert fs._l15_wake_manifest is None
    assert not fs._l15_wake_refill


def test_refill_one_opens_the_e2b_gate(monkeypatch, tmp_path):
    # switch ON: today's E2/E2b behaviour, the anchor gate decides.
    cap0._env(monkeypatch, tmp_path, master=True, mib="c1=64")
    monkeypatch.setenv(REFILL, "1")
    man = _man()
    _sched, fs, out = _run(monkeypatch, tmp_path, man)
    assert out == (man, 0, KEEP, True), out
    # the refill mark is set; the manifest is stashed later by the
    # hold-aware restore step (E2b pins that separately)
    assert fs._l15_wake_refill is True


def test_master_off_beats_refill(monkeypatch, tmp_path):
    # master off -> byte-identical pre-1.5 restore, whatever REFILL says
    cap0._env(monkeypatch, tmp_path, master=False, mib="c1=64")
    monkeypatch.setenv(REFILL, "1")
    _sched, fs, out = _run(monkeypatch, tmp_path, _man())
    assert out == (None, None, 0, False), out
    assert not fs._l15_wake_refill
