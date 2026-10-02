# SPDX-License-Identifier: Apache-2.0
"""TREE-RESET-SUB (02.10.): one line per tree reset naming where it spends its time.

N6a (..._4dd76c122a_1002_164400) P sleeps: 'WEG2-SLEEP-SUB rpc_flush[passed
total=73 ... tree_reset=46 req_pool_clear=1 alloc_clear=0]' -- tree_reset
31-47 ms on every PP rank, the block of the P>D quiesce. Inside it: the H81
RESET-RELEASE (9.5-11.5 ms by its own line), the storage threads' stop/start
(#1068 RESET JOIN, joined_s=0.00), the host pool clear, the #1424g orphan
give-back and the at=reset holder census -- each unclocked. The marker
'WEG2-TREE-RESET-SUB total= unpin= release_host= tree_rebuild= queued_refs=
controller_reset= host_clear= orphans= census= tail= threads_start=' splits it.
Instrument only. Runs the real ``UnifiedRadixCache._reset_full`` on the
#1424g arena shell.
"""
from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_h1424g_tree_reset_sub", Path(__file__).with_name("test_arena_reset_orphans_1424g.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

arena = H.arena  # the fixture

STEPS = ["unpin", "release_host", "tree_rebuild", "queued_refs", "controller_reset",
         "host_clear", "orphans", "census", "tail"]


def test_the_reset_names_each_step_in_order_and_they_add_up(arena, caplog):
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    t.cache_controller._weg2_last_start_ms = 1.25
    with caplog.at_level(logging.INFO, logger="sglang.srt.mem_cache.unified_radix_cache"):
        t._reset_full()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("WEG2-TREE-RESET-SUB")]
    assert len(lines) == 1, lines
    line = lines[0]
    names = re.findall(r" (\w+)=([\d.]+)", line.split(" (ms per step")[0])
    keys = [k for k, _ in names]
    assert keys[0] == "total" and keys[1:-1] == STEPS and keys[-1] == "threads_start", keys
    vals = dict(names)
    assert abs(sum(float(vals[k]) for k in STEPS) - float(vals["total"])) < 0.5, line
    assert vals["threads_start"] == "1.2" or vals["threads_start"] == "1.3", line
    assert t.cache_controller.resets == 1, "the reset itself ran as before"


def test_no_controller_no_line(arena, caplog):
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    t.cache_controller = None
    with caplog.at_level(logging.INFO, logger="sglang.srt.mem_cache.unified_radix_cache"):
        t._reset_full()
    assert not [r for r in caplog.records if "WEG2-TREE-RESET-SUB" in r.getMessage()]


def test_the_controller_clocks_its_thread_restart():
    import inspect

    from sglang.srt.managers.cache_controller import HiCacheController

    src = inspect.getsource(HiCacheController.reset)
    i = src.index("self._start_storage_threads()")
    assert "_t_start = time.perf_counter()" in src[:i]
    assert "self._weg2_last_start_ms = (time.perf_counter() - _t_start) * 1000.0" in src[i:]
