# SPDX-License-Identifier: Apache-2.0
"""RC2 review L4: `--d-hold-s 0` means OFF, exactly like unset.

The launcher refuses a negative hold with "both must be >= 0 (0 / unset =
off)", but 0 was not off: the launcher shipped `--d-hold-s 0.0` to the front,
the front kept a 0 s hold, and a small backlog FLIP-ECONOMICS holds was then
released the moment D went free -- a different policy from unset, where only
the fairness bound releases it. Now 0 is off on both sides: the front maps it
to None, the launcher ships the argv of unset and its IDLE POLICY line says
off. A tiny positive T still gives the old 0 behaviour, by name.

Hermetic: no GPU, no boot, no HTTP.
"""
from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import launcher as launcher_mod  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

X = 4096


def _front(**kw):
    return front_mod.Front("http://p", "http://d", "D", "hold0", "", 0, 0, {}, 45.0,
                           tp_prefill_max_tokens=X, **kw)


def _argv(*extra):
    ns = launcher_mod.build_parser().parse_args(["--tree", "/x", "--tag", "t", *extra])
    return launcher_mod.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 8, 8, X, X, "D")


# --------------------------------------------------------------------- front
@pytest.mark.parametrize("zero", [0, 0.0, -1.0])
def test_the_front_takes_a_zero_hold_as_off(zero):
    f = _front(d_hold_s=zero)
    assert f.d_hold_s is None
    assert "d_hold_s=off" in f.idle_policy_line()
    f._note_d_free(True, 100.0)
    assert f._d_hold_expired("idle", 100.0) is True, "idle: only min-dwell gates, as unset"
    assert f._d_hold_expired("backlog", 1000.0) is False, (
        "backlog: only the fairness bound releases it, as unset -- a 0 s hold released it at once")


def test_the_front_keeps_a_positive_hold():
    f = _front(d_hold_s=0.3)
    assert f.d_hold_s == pytest.approx(0.3)
    f._note_d_free(True, 100.0)
    assert f._d_hold_expired("backlog", 100.2) is False
    assert f._d_hold_expired("backlog", 100.5) is True


# ------------------------------------------------------------------ launcher
@pytest.mark.parametrize("value,on", [(None, False), (0, False), (0.0, False), (0.001, True), (10, True)])
def test_one_definition_of_a_hold_that_is_on(value, on):
    assert launcher_mod.d_hold_active(value) is on


def test_a_zero_hold_ships_the_argv_of_unset():
    assert _argv("--d-hold-s", "0") == _argv(), "0 = off must be byte-identical to unset"


def test_a_positive_hold_still_reaches_the_front():
    argv = _argv("--d-hold-s", "10")
    assert argv[argv.index("--d-hold-s") + 1] == "10.0"


def test_the_argv_and_the_policy_line_share_the_definition():
    assert "d_hold_active(" in inspect.getsource(launcher_mod.front_argv_for)
    src = inspect.getsource(launcher_mod.main)
    line = src[src.index("IDLE POLICY (27B, user order 2026-09-24)"):]
    line = line[:line.index("\n    log(") if "\n    log(" in line else 1200]
    assert "d_hold_active(ns.d_hold_s)" in line, "the IDLE POLICY line must call 0 off, as the argv does"
    assert "0 / unset = off" in src, "the >= 0 refusal keeps its promise, now true"
