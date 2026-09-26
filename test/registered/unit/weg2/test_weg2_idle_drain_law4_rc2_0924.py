# SPDX-License-Identifier: Apache-2.0
"""RC2 review L1: law 4 summed over a SHORT drain, independent of the flags.

`--d-short-drain-tokens N` (idle policy b) hands a queued SHORT-only backlog to
group D in ONE drain; D prefills the whole sum at once. Each request is <= X
by its own route verdict, but the sum was bounded by N alone and the launcher
accepted any N >= 0 -- so N > X (a flag combination) let one drain hand D more
than X uncached tokens, and so did a live X re-solve below the launch value.

Two riegel, both asserted here:
* the launcher refuses N > X by name (W153 Weg2ShortDrainAboveX), where X is
  the value it hands the front and D (resolve_x has applied the front floor);
* the front caps every drain at min(N, the X in force), whatever the flags.

Hermetic: no GPU, no boot, no HTTP.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import launcher as launcher_mod  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

X = 4096


def _front(n_max, x=X):
    return front_mod.Front("http://p", "http://d", "D", "law4", "", 0, 0, {}, 45.0,
                           tp_prefill_max_tokens=x, d_short_drain_tokens=n_max)


def _drain(f, sizes):
    """Queue SHORT (d_eligible) requests of the given uncached sizes and run
    one drain decision; returns (moved, rids still queued, rids handed to D)."""

    async def body():
        for i, n in enumerate(sizes):
            fut = asyncio.get_running_loop().create_future()
            f.queue.append(front_mod.Pending(f"s{i}", "/generate", {}, "x", time.time(), fut,
                                             est_prompt=n, est_uncached=n, d_eligible=True))
        moved = f._d_short_drain(time.time())
        return moved, [p.rid for p in f.queue], [p.rid for p in f._ready_for_d]

    return asyncio.run(body())


# ------------------------------------------------------------------ launcher
def test_the_launcher_refuses_n_above_x_by_name():
    with pytest.raises(SystemExit) as e:
        launcher_mod.refuse_short_drain_above_x(8192, X, "X=4096 source=flag (floor 4096)")
    msg = str(e.value)
    assert msg.startswith("W153 Weg2ShortDrainAboveX:"), msg
    for want in ("--d-short-drain-tokens 8192", "X=4096", "source=flag", "<= 4096", "0 for off"):
        assert want in msg, (want, msg)


@pytest.mark.parametrize("n,x", [(0, X), (1, X), (X, X), (8192, 8192), (0, 1)])
def test_the_launcher_passes_off_and_n_up_to_x(n, x):
    launcher_mod.refuse_short_drain_above_x(n, x, "prov")


def test_main_checks_n_against_the_x_it_hands_out():
    src = inspect.getsource(launcher_mod.main)
    seed = src.index("x_tokens, x_provenance = x_seed.tokens, x_seed.provenance")
    call = src.index("refuse_short_drain_above_x(int(ns.d_short_drain_tokens or 0), x_tokens, x_provenance)")
    assert seed < call, "the check must read the X resolve_x produced (flag after the front floor, or derived)"
    front_argv = inspect.getsource(launcher_mod.front_argv_for)
    assert "--d-short-drain-tokens" in front_argv, "the refused value would otherwise reach the front"


# --------------------------------------------------------------------- front
def test_the_front_caps_a_drain_at_x_when_n_is_above_it():
    # a front launched directly with N > X (no launcher in front of it)
    f = _front(n_max=8192)
    moved, queued, handed = _drain(f, [3000, 3000])  # each <= X, sum 6000 > X
    assert (moved, queued, handed) == (0, ["s0", "s1"], []), (moved, queued, handed)
    assert f.counters["d_short_drain_x_capped"] == 1


def test_a_live_x_below_the_launch_value_caps_the_drain():
    f = _front(n_max=8192, x=8192)   # a legal launch: N = X = 8192
    f.tp_prefill_max_tokens = 5000   # the live re-solve moved X down
    moved, queued, handed = _drain(f, [3000, 3000])  # each <= 5000, sum 6000 > 5000
    assert (moved, handed) == (0, []), (moved, queued, handed)
    assert f.counters["d_short_drain_x_capped"] == 1


def test_within_both_bounds_the_drain_still_runs():
    f = _front(n_max=X)
    moved, queued, handed = _drain(f, [1500, 2500])  # sum 4000 <= min(N, X)
    assert (moved, queued, handed) == (2, [], ["s0", "s1"])
    assert f.counters.get("d_short_drain_x_capped", 0) == 0


def test_above_n_is_not_counted_as_the_x_cap():
    f = _front(n_max=2048)
    moved, _, _ = _drain(f, [1500, 1500])  # sum 3000 > N (and <= X)
    assert moved == 0
    assert f.counters.get("d_short_drain_x_capped", 0) == 0, "N bound it, not X"
