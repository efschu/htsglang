# SPDX-License-Identifier: Apache-2.0
"""QUIESCE-EMPTY-CACHE (02.10.): the quiesce flush's empty_cache, measured and switchable.

N6d (..._ec4d492f58_1002_170955): WEG2-SLEEP-SUB rpc_flush empty_cache=21 ms
per P flush. It is the only empty_cache before the flip legs on PP0 (PP0's
release flush is refused by the PENDING idle lap; sleep_complete's runs after
the legs and released 0 MiB on every N6d sleep), so it stays by default -- and
every Weg-2 flush now names what it gave back: 'WEG2-FLUSH-EMPTY-CACHE site=
ms= released_mib='. SGLANG_WEG2_QUIESCE_EMPTY_CACHE=0 skips it on the quiesce
RPC only. Hermetic, CPU.
"""
from __future__ import annotations

import inspect
import logging
import types
from unittest import mock

import pytest

from sglang.srt.managers import scheduler as S

LOG = "sglang.srt.managers.scheduler"


class _Dev:
    def __init__(self):
        self.reserved = 3 * 1024 ** 3
        self.calls = 0

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return 2 * 1024 ** 3

    def empty(self):
        self.calls += 1
        self.reserved -= 512 * 1024 ** 2


@pytest.fixture
def dev(monkeypatch):
    d = _Dev()
    monkeypatch.setattr(S.current_platform, "empty_cache", d.empty)
    monkeypatch.setattr(S.torch, "get_device_module", lambda *a: d)
    return d


def test_a_weg2_quiesce_flush_names_what_it_gave_back(dev, monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.delenv("SGLANG_WEG2_QUIESCE_EMPTY_CACHE", raising=False)
    with caplog.at_level(logging.INFO, logger=LOG):
        S._weg2_flush_empty_cache(None, True)
    assert dev.calls == 1
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("WEG2-FLUSH-EMPTY-CACHE")]
    assert line and "site=rpc" in line[0] and "released_mib=512.0" in line[0], line
    assert "reserved_mib=3072->2560" in line[0], line


def test_the_switch_skips_only_the_quiesce(dev, monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_WEG2_QUIESCE_EMPTY_CACHE", "0")
    with caplog.at_level(logging.INFO, logger=LOG):
        S._weg2_flush_empty_cache(None, True)       # the front's quiesce RPC
        assert dev.calls == 0
        S._weg2_flush_empty_cache(False, False)     # the kv release's flush
    assert dev.calls == 1
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("WEG2-FLUSH-EMPTY-CACHE")]
    assert any("site=rpc skipped" in m for m in msgs) and any("site=release ms=" in m for m in msgs), msgs


def test_a_stock_group_is_upstream_byte_for_byte(dev, monkeypatch, caplog):
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_QUIESCE_EMPTY_CACHE", "0")
    with caplog.at_level(logging.INFO, logger=LOG):
        S._weg2_flush_empty_cache(None, True)
    assert dev.calls == 1
    assert not [r for r in caplog.records if "WEG2-FLUSH-EMPTY-CACHE" in r.getMessage()]


def test_flush_cache_routes_its_empty_cache_through_the_helper():
    src = inspect.getsource(S.Scheduler.flush_cache)
    i = src.index("if empty_cache:")
    assert "_weg2_flush_empty_cache(zero_kv, tp_group_verdict)" in src[i:i + 120]
    assert "current_platform.empty_cache()" not in src[i:i + 120]
