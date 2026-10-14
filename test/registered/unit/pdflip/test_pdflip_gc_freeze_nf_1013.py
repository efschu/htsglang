# SPDX-License-Identifier: Apache-2.0
"""NF 13.10. GC-FREEZE (boot b60520f01d: 11 of 455 flips over 6 s, max 9.2 s; a gen-2
pass over ~1.12 M never-frozen boot objects took 4.4-5.4 s on PP0/PP1 after the cold heap
was swapped out, ``done/nf-flip-langsam-analyse-1013.md``).

Pinned here, without a GPU:

* the freeze runs once per rank, after boot, on P AND D (a fake ``gc`` records the calls),
  names itself ``GC-FREEZE frozen=<n> gen2_after=<m> rank=<r>``, and a second call is a no-op;
* the switch off = the old path (no collect, no freeze, the same return value as before);
* CPython 3.12's own immortals in the permanent generation (375 after a bare collect) are
  NOT "frozen by someone" (b25e417304), a real freeze (~800k) is;
* a later CUDA-graph capture does not unfreeze it (``base_cuda_graph_runner.freeze_gc`` --
  ``gc.unfreeze()`` would hand the boot heap back and collect it inside a flip);
* the launcher side: ``--p-gc-freeze``, the nextflash row's group defaults (freeze on P and
  D, GC warning on D), qwen27b untouched, a stated ``=0`` wins;
* the GC warning is armed on D (a real ``gc.callbacks`` entry).
"""

import argparse
import gc
import logging
import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import gc_instrument as G  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

NF, Q = F.PROFILE_NEXTFLASH, F.PROFILE_QWEN27B
ARGS0 = argparse.Namespace(gc_warning_threshold_secs=0.0)


class FakeGc:
    """Records what the freeze calls; ``freeze()`` moves the objects to the permanent generation."""

    def __init__(self, permanent=375, gen2=1_124_790):
        self.permanent = permanent
        self.gen2 = gen2
        self.calls = []
        self.callbacks = []

    def collect(self, *a):
        self.calls.append("collect")
        return 0

    def freeze(self):
        self.calls.append("freeze")
        self.permanent += self.gen2
        self.gen2 = 0

    def unfreeze(self):
        self.calls.append("unfreeze")
        self.gen2 += self.permanent
        self.permanent = 0

    def get_freeze_count(self):
        return self.permanent

    def get_objects(self, generation=None):
        return [None] * (self.gen2 if generation == 2 else 0)


@pytest.fixture
def fake(monkeypatch):
    f = FakeGc()
    monkeypatch.setattr(G, "gc", f)
    return f


def _group_env(profile, group):
    """The group env the launcher writes for ``profile`` (empty --env-p/--env-d)."""
    ns = argparse.Namespace(profile=profile, env_p="", env_d="")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FLLIPER_PDFLIP_GC_")}
    L.apply_profile_group_switch_defaults(ns, environ=clean)
    return L.parse_group_env(ns.env_p if group == "P" else ns.env_d)


# --------------------------------------------------------------------------- the freeze


@pytest.mark.parametrize("group,pp,tp", [("P", 0, 0), ("P", 1, 0), ("D", None, 0), ("D", None, 2)])
def test_freeze_runs_once_per_rank_after_boot_on_p_and_d(fake, caplog, group, pp, tp):
    env = _group_env(NF, group)
    with caplog.at_level(logging.INFO, logger=G.logger.name):
        out = G.arm_after_boot(ARGS0, tp, env=env, pp_rank=pp)
    assert out["freeze"] is True
    # one collect (nothing already garbage gets frozen), then exactly one freeze
    assert fake.calls == ["collect", "freeze"]
    rank = f"PP{pp}TP{tp}" if pp is not None else f"TP{tp}"
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("GC-FREEZE frozen=")]
    assert len(line) == 1, line
    assert f"frozen={375 + 1_124_790} gen2_after=0 rank={rank} " in line[0]
    # a second arm in the same process freezes nothing more (the permanent generation is populated)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=G.logger.name):
        out2 = G.arm_after_boot(ARGS0, tp, env=env, pp_rank=pp)
    assert out2["freeze"] is False
    assert fake.calls == ["collect", "freeze"]
    assert any(r.getMessage().startswith("GC-FREEZE skipped") for r in caplog.records)


def test_switch_off_is_the_old_path(fake):
    for env in ({}, {G.FREEZE_ENV: "0"}, {G.FREEZE_ENV: ""}):
        out = G.arm_after_boot(ARGS0, 0, env=env, pp_rank=0)
        assert out == {"warn": None, "freeze": False}
    assert fake.calls == []


def test_the_return_shape_is_unchanged():
    out = G.arm_after_boot(ARGS0, 0, env={}, pp_rank=None)
    assert set(out) == {"warn", "freeze"}


def test_3_12_immortals_are_not_a_freeze(fake):
    """b25e417304: CPython 3.12 parks its immortals in the permanent generation on every
    collection (375 measured after a bare gc.collect()); ``> 0`` would skip the freeze in
    every real process."""
    fake.permanent = 375
    assert G.permanent_generation_populated() is False
    assert G.arm_after_boot(ARGS0, 0, env={G.FREEZE_ENV: "1"})["freeze"] is True
    assert "freeze" in fake.calls


def test_a_real_freeze_is_left_alone(fake):
    fake.permanent = 800_000
    assert G.permanent_generation_populated() is True
    assert G.arm_after_boot(ARGS0, 0, env={G.FREEZE_ENV: "1"})["freeze"] is False
    assert fake.calls == []


def test_the_threshold_edge(fake):
    fake.permanent = G.FOREIGN_FREEZE_MIN - 1
    assert G.permanent_generation_populated() is False
    fake.permanent = G.FOREIGN_FREEZE_MIN
    assert G.permanent_generation_populated() is True


def test_a_real_interpreter_after_a_collect_is_not_populated():
    """The measured 3.12 behaviour itself (skipped where something already froze this process)."""
    if gc.get_freeze_count() >= G.FOREIGN_FREEZE_MIN:
        pytest.skip("this test process already carries a freeze")
    gc.collect()
    assert gc.get_freeze_count() < G.FOREIGN_FREEZE_MIN
    assert G.permanent_generation_populated() is False


def test_the_threshold_equals_the_loader_guard():
    load_gc = pytest.importorskip("flliper.srt.model_loader.load_gc")
    assert G.FOREIGN_FREEZE_MIN == load_gc._FOREIGN_FREEZE_MIN


def test_the_real_freeze_takes_the_heap_out_of_a_collection():
    """No fake: freeze_after_boot on the real gc moves the tracked objects to the permanent
    generation and the next gen-2 list is (nearly) empty; the test undoes it."""
    if gc.get_freeze_count() >= G.FOREIGN_FREEZE_MIN:
        pytest.skip("this test process already carries a freeze")
    keep = [[i] for i in range(30000)]  # tracked containers: enough to be a real freeze
    try:
        assert G.freeze_after_boot(0, 0) is True
        assert gc.get_freeze_count() >= G.FOREIGN_FREEZE_MIN
        assert len(gc.get_objects(2)) < 5000
    finally:
        gc.unfreeze()
        del keep


# --------------------------------------------------------------------------- the capture


def _runner_module():
    return pytest.importorskip("flliper.srt.model_executor.runner.base_cuda_graph_runner")


@pytest.mark.parametrize("permanent,enable_cudagraph_gc,want", [
    (375, False, ["collect", "freeze", "unfreeze", "collect"]),   # the old path (nothing frozen)
    (375, True, ["collect"]),                                      # upstream: gc stays on in capture
    (1_124_790, False, ["collect"]),                               # the boot heap is frozen: hands off
    (1_124_790, True, ["collect"]),
])
def test_a_capture_does_not_unfreeze_the_boot_heap(monkeypatch, permanent, enable_cudagraph_gc, want):
    mod = _runner_module()
    f = FakeGc(permanent=permanent)
    monkeypatch.setattr(mod, "gc", f)
    monkeypatch.setattr(G, "gc", f)
    with mod.freeze_gc(enable_cudagraph_gc):
        pass
    assert f.calls == want
    if permanent >= G.FOREIGN_FREEZE_MIN:
        assert f.permanent == permanent and "unfreeze" not in f.calls


def test_a_capture_after_the_boot_freeze_keeps_it(monkeypatch):
    """The whole chain on one fake: boot freeze, then a recapture inside a flip."""
    mod = _runner_module()
    f = FakeGc()
    monkeypatch.setattr(mod, "gc", f)
    monkeypatch.setattr(G, "gc", f)
    assert G.arm_after_boot(ARGS0, 0, env={G.FREEZE_ENV: "1"}, pp_rank=0)["freeze"] is True
    frozen = f.permanent
    with mod.freeze_gc(False):
        pass
    assert f.permanent == frozen and f.gen2 == 0 and "unfreeze" not in f.calls


# --------------------------------------------------------------------------- the launcher


def test_the_flag_and_its_env():
    p = L.build_parser()
    assert p.get_default("p_gc_freeze") == "off" == L.P_GC_FREEZE_DEFAULT
    assert L.p_gc_env(argparse.Namespace(p_gc_freeze="off")) == {}
    assert L.p_gc_env(argparse.Namespace()) == {}
    assert L.p_gc_env(argparse.Namespace(p_gc_freeze="on")) == {G.FREEZE_ENV: "1"}
    assert L.d_gc_env(argparse.Namespace(d_gc_freeze="off")) == {}
    assert L.d_gc_env(argparse.Namespace(d_gc_freeze="on")) == {G.FREEZE_ENV: "1"}


def test_nextflash_freezes_p_and_d_by_default_and_warns_on_d():
    p, d = _group_env(NF, "P"), _group_env(NF, "D")
    assert p[G.FREEZE_ENV] == "1" and d[G.FREEZE_ENV] == "1"
    assert d[G.WARN_ENV] == f"{G.P_GC_WARN_DEFAULT_S:g}"
    # P's warning stays the launcher's own default (apply_p_gc_warn_default), not a row entry
    assert G.WARN_ENV not in p


def test_qwen27b_states_nothing():
    row = F.PROFILES[Q]
    flat = str(dict(getattr(row, "group_switch_defaults", {}) or {}))
    assert G.FREEZE_ENV not in flat and G.WARN_ENV not in flat
    assert _group_env(Q, "P") == {} and _group_env(Q, "D") == {}
    # and no form-wide default reaches the 27B
    assert G.FREEZE_ENV not in F.PROFILE_SWITCH_DEFAULTS.get(Q, {})


@pytest.mark.parametrize("attr,group", [("env_p", "P"), ("env_d", "D")])
def test_a_stated_zero_wins_and_turns_it_off(fake, attr, group):
    ns = argparse.Namespace(profile=NF, env_p="", env_d="")
    setattr(ns, attr, f"{G.FREEZE_ENV}=0")
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FLLIPER_PDFLIP_GC_")}
    L.apply_profile_group_switch_defaults(ns, environ=clean)
    env = L.parse_group_env(getattr(ns, attr))
    assert env[G.FREEZE_ENV] == "0"
    assert G.arm_after_boot(ARGS0, 0, env=env, pp_rank=0)["freeze"] is False
    assert fake.calls == []


def test_the_nf_p_env_end_to_end_equals_flag_on(fake):
    """What the flag writes and what the row writes are the same switch."""
    assert L.p_gc_env(argparse.Namespace(p_gc_freeze="on")) == {G.FREEZE_ENV: _group_env(NF, "P")[G.FREEZE_ENV]}


# --------------------------------------------------------------------------- the D warning


def test_the_gc_warning_is_armed_on_d(monkeypatch, fake):
    armed = []
    monkeypatch.setattr("flliper.srt.utils.common.configure_gc_warning", armed.append)
    out = G.arm_after_boot(ARGS0, 0, env=_group_env(NF, "D"))
    assert out["warn"] == pytest.approx(0.5) and armed == [0.5]
    # the 27B's D arms nothing
    armed.clear()
    assert G.arm_after_boot(ARGS0, 0, env=_group_env(Q, "D"))["warn"] is None and armed == []
    # --env-d names it: the stated value wins (here: off)
    assert G.arm_after_boot(ARGS0, 0, env={G.WARN_ENV: "0"})["warn"] is None


def test_the_d_warning_is_a_real_gc_callback(fake):
    n = len(gc.callbacks)
    try:
        out = G.arm_after_boot(ARGS0, 1, env=_group_env(NF, "D"))
        assert out["warn"] == pytest.approx(0.5)
        assert len(gc.callbacks) == n + 1
    finally:
        del gc.callbacks[n:]


def test_the_scheduler_hands_its_pp_rank_to_the_arm():
    """The rank label of the marker needs pp_rank at the call site (AST: no torch import)."""
    import ast

    here = os.path.dirname(G.__file__)
    src = open(os.path.join(here, "..", "managers", "scheduler.py")).read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_pdflip_gc_arm"]
    assert len(calls) == 1
    assert [k.arg for k in calls[0].keywords] == ["pp_rank"]
    assert sys.version_info >= (3, 8)
