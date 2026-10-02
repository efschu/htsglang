# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-NOHOLD-TREE (N4a D TP0 09:22:56Z, "#924 MAMBA SLOT ALIASING:
mamba_num_used=-2", free_and_cached=2): the cap-0 rank armed its (empty) keep
windows, so its sleep kept the held chains in the tree; at the wake it kept
NOTHING (REFILL off -> manifest consumed, votes None, verdict none) and the
plain restore cleared the pools but kept the tree -- the held anchors were
free AND cached, the idle leak check killed D. On the REAL tree and pools."""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker as IC,
)
from sglang.srt.weg2 import l15_p_adopt

WU = wu.SchedulerWeightUpdaterManager
_HERE = os.path.dirname(__file__)


def _fx():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_tree_rewrite_1001",
        os.path.join(_HERE, "test_weg2_l15_tree_rewrite_1001.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m._fixture()


def _wake(fx, retained, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_L15", "1")
    # the held chain the sleep kept on this rank (slots + anchor reserved)
    l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                      token_ids=list(range(900, 932)),
                      rows=list(range(1, 33)), anchor_row=2)
    sched = SimpleNamespace(req_to_token_pool=fx.pool,
                            token_to_kv_pool_allocator=fx.allocator,
                            tree_cache=fx.cache, draft_worker=None,
                            _l15_tree_retained=retained)
    fs = SimpleNamespace(scheduler=sched, _l15_wake_manifest=None)
    # the wake keeps nothing here: manifest consumed (REFILL off)
    fs._l15_wake_hold_signal = lambda: (None, 0, 0, True)
    fs._l15_clear_tms_keep_spans = lambda s: 0
    fs.flush_cache = lambda: False
    assert WU._weg2_wake_restore_pools(fs) is True
    return sched


def test_a_rank_that_retained_but_keeps_nothing_drops_its_chains(monkeypatch):
    _m, fx = _fx()
    sched = _wake(fx, True, monkeypatch)
    dup, _ids, shared = IC._mamba_double_claimed(fx.pool.mamba_allocator, fx.cache)
    assert dup == 0 and shared == 0, "held anchors free AND cached (#924)"
    assert sched._l15_tree_retained is False


def test_without_the_flag_the_tree_survives_the_restore_as_before(monkeypatch):
    """The #1455 restore keeps the tree on every rank that did not retain --
    and on this fixture that is exactly the aliasing the flag prevents."""
    _m, fx = _fx()
    _wake(fx, False, monkeypatch)
    _dup, _ids, shared = IC._mamba_double_claimed(fx.pool.mamba_allocator, fx.cache)
    assert shared > 0      # free_and_cached: the N4a death condition


def test_the_sleep_sets_the_flag_exactly_when_the_round_armed():
    import inspect

    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler)
    assert "self._l15_tree_retained = _l15_res is not None" in src
