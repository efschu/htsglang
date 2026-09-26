"""UNIFY S4: Live-X exactly once on the unified tree.

The NF line (H84/H85) and the 27B line (RC7-X) built the same Live-X twice:
the r_D probe from D's own prefill clock, the ceiling at D's W50 riegel, the
singleton band above the start X. On this tree each piece exists ONCE:

* launcher ``resolve_x_ceiling``: one def (the NF return shape, the 27B
  ``x_busy`` term with its W155 refusal);
* launcher ``resolve_x``: takes the calibration identity's ``accept`` filter,
  so a front log of another checkpoint/form/line never seeds X (NF inventory,
  risk (c));
* front: ONE ``_sample_r_d`` for both wire shapes, the body time (H84) first,
  else D's ``/get_server_info`` prefill clock (weg2/prefill_clock.py, H85);
* front: the X-SOLO band floor is ``--x-busy-tokens`` (27B X_busy), else the
  start X (H84) -- unset, byte-identical to the NF band.

Hermetic, CPU.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_x_live_ceiling_h84 import START_X, _band_front, _drive, _front  # noqa: E402

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

SRC = Path(L.__file__).resolve().parent


def _defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text())
    return sum(1 for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


# ------------------------------------------------------------- exactly once
@pytest.mark.parametrize("path,name", [
    ("launcher.py", "resolve_x"),
    ("launcher.py", "resolve_x_ceiling"),
    ("front.py", "d_prefill_seconds"),
    ("front.py", "r_d_probe"),
    ("front.py", "_x_solo_busy"),
    ("front.py", "_sample_r_d"),
    ("front.py", "_x_band_floor"),
])
def test_every_live_x_piece_is_defined_once(path, name):
    assert _defs(SRC / path, name) == 1


def test_one_prefill_clock_module_and_no_second_r_d_reader():
    assert (SRC / "prefill_clock.py").is_file()
    front_src = (SRC / "front.py").read_text()
    # the leg-2 wall never divides: the probe is the only producer of r_D samples
    assert front_src.count('note_x_sample("r_d"') == 1


# --------------------------------------------------------- the ceiling
def test_resolve_x_ceiling_without_x_busy_is_the_nf_form():
    d_x, front_c, line = L.resolve_x_ceiling(12288, START_X)
    assert (d_x, front_c) == (12288, 12288)
    assert "between 4096 and the live X goes to D only as a singleton" in line
    assert "--x-busy-tokens" not in line
    assert L.resolve_x_ceiling(0, START_X)[:2] == (START_X, 0)


def test_resolve_x_ceiling_names_x_busy_and_refuses_a_negative_one():
    _, _, line = L.resolve_x_ceiling(12288, START_X, 2048)
    assert "--x-busy-tokens 2048" in line and "between 2048 and the live X" in line
    with pytest.raises(SystemExit, match="W155 Weg2XBusyNegative"):
        L.resolve_x_ceiling(12288, START_X, -1)
    # 0 is legal: no D prefill while D decodes others
    L.resolve_x_ceiling(12288, START_X, 0)


def test_x_busy_reaches_the_front_only_when_set():
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
    assert ns.x_busy_tokens is None
    fa = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D")
    assert "--x-busy-tokens" not in fa
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--x-busy-tokens", "2048"])
    fa = L.front_argv_for("py", "/s", 1, 2, {}, [], ns, 0, 0, 1, 1, START_X, START_X, "D")
    assert fa[fa.index("--x-busy-tokens") + 1] == "2048"
    assert front_mod.main.__code__ is not None  # the front parser knows the flag:
    assert "--x-busy-tokens" in inspect.getsource(front_mod.main)


def test_main_resolves_x_with_the_calibration_identity():
    tree = ast.parse(inspect.getsource(L.main))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "resolve_x"]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("accept") == "calib_log_accept_of(ns)"


# --------------------------------------------------------- resolve_x accept
def _front_log(d: Path, name: str, flip_s=2.6, r_d=1036.0) -> Path:
    p = d / name
    p.write_text("")
    return p


def test_resolve_x_skips_logs_the_identity_refuses(tmp_path, monkeypatch):
    a = _front_log(tmp_path, "boot_weg2_a_1111111_0926_010203.front.log")
    b = _front_log(tmp_path, "boot_weg2_b_2222222_0926_010204.front.log")
    now = time.time()
    os.utime(a, (now - 10, now - 10))
    os.utime(b, (now, now))  # b is newer: without a filter it would seed X
    seen = []

    def measure(path, floor):
        seen.append(os.path.basename(path))
        return (2.6, 1036.0, 3649.0, 1, 1, 1)

    monkeypatch.setattr(L, "measure_x_inputs", measure)
    got = L.resolve_x(None, str(tmp_path), START_X)
    assert "source=boot:" + b.name in got.provenance and got.measured
    seen.clear()
    got = L.resolve_x(None, str(tmp_path), START_X, accept=lambda p: p.endswith(a.name))
    assert seen == [a.name]
    assert "source=boot:" + a.name in got.provenance
    assert "refused 1 front log(s) of another checkpoint/form/line" in got.provenance


def test_resolve_x_all_refused_falls_back_to_the_record_and_says_why(tmp_path, monkeypatch):
    _front_log(tmp_path, "boot_weg2_a_1111111_0926_010203.front.log")
    monkeypatch.setattr(L, "measure_x_inputs", lambda p, f: (2.6, 1036.0, 3649.0, 1, 1, 1))
    got = L.resolve_x(None, str(tmp_path), START_X, accept=lambda p: False)
    assert not got.measured and "source=recorded" in got.provenance
    assert "refused 1 front log(s)" in got.provenance


def test_resolve_x_flag_wins_whatever_the_filter(tmp_path):
    got = L.resolve_x(8192, str(tmp_path), START_X, accept=lambda p: False)
    assert got.tokens == 8192 and "source=flag" in got.provenance


# --------------------------------------------------------- the band floor
def test_band_floor_is_the_start_x_unless_x_busy_is_given():
    assert _front().x_busy_tokens is None
    f = _band_front()
    assert f._x_band_floor() == START_X
    f = _front(x_busy_tokens=2048)
    f.tp_prefill_max_tokens = 7500
    assert f._x_band_floor() == 2048
    f = _front(x_busy_tokens=9000)
    f.tp_prefill_max_tokens = 7500
    assert f._x_band_floor() == 7500  # never above the X in force


def test_x_busy_lowers_the_band_floor_a_busy_d_sends_the_band_to_p():
    """27B X_busy=2048: a 3000-token SHORT is a band request now; with D busy
    it routes LONG on the band floor instead of halting D's decode."""
    f = _band_front()
    f.x_busy_tokens = 2048
    f.groups["D"].outstanding["weg2-0-0"] = time.time()
    asyncio.run(_drive(f, [3000]))
    assert f.on_d == [] and f.counters["route_long"] == 1 and f.counters["x_solo_p"] == 1


def test_without_x_busy_the_same_request_takes_no_window():
    """Unset: 3000 <= start X is no band request -- SHORT at once, no X-SOLO
    decision (the NF H84 behaviour, byte for byte); with X_busy=2048 the same
    idle-D request is a singleton decided after the window."""
    f = _band_front()
    asyncio.run(_drive(f, [3000]))
    assert f.on_d == ["weg2-0-1"] and f.counters["x_solo_d"] == f.counters["x_solo_p"] == 0
    f = _band_front()
    f.x_busy_tokens = 2048
    asyncio.run(_drive(f, [3000]))
    assert f.on_d == ["weg2-0-1"] and f.counters["x_solo_d"] == 1
