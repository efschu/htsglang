"""BOOTZEIT 3 (29.09.): early D start -- the launcher half.

z30r3: D's init (30 s) starts only after P READY + sleep(P) + the D plan (8 s).
``--weg2-d-early-start on`` launches D with P, planned from the planner seat's
ONE expectation selector (``d_expect_dormant_other``, D-EXPECT 9f74489683),
and releases D's weight load only after sleep(P), iff every planned budget is
<= the measured one. 'off' (default) keeps the serial path.
"""

import argparse
import inspect

from sglang.srt.weg2 import launcher as L


def test_flag_defaults_off_and_help_renders():
    ap = L.build_parser()
    ns = ap.parse_known_args(["--tree", "t", "--tag", "x"])[0]
    assert ns.weg2_d_early_start == "off"
    assert not L._d_early_start_armed(ns)
    assert L._d_early_start_armed(argparse.Namespace(weg2_d_early_start="on"))
    assert not L._d_early_start_armed(argparse.Namespace())
    assert "--weg2-d-early-start" in ap.format_help()


def _main_src():
    return inspect.getsource(L.main)


def test_early_start_is_armed_only_by_the_flag_and_after_the_dry_branch():
    src = _main_src()
    i_armed = src.index("if _d_early_start_armed(ns):")
    assert src.index('log("DRY-RUN complete: nothing started, mounted, armed or written")') < i_armed
    assert i_armed < src.index('state.pids["P"] = spec_p.pid')


def test_early_plan_reads_the_planner_selector_no_second_source():
    src = _main_src()
    early = src[src.index("if _d_early_start_armed(ns):"):src.index("def _d_early_verdict(")]
    assert "d_expect_dormant_other(" in early and "_p_dormant_recs" in early
    assert "P_WINDOWS_MIB" not in early and "dc_expect_d[" not in early
    # the gate path goes to D, and only on the early plan
    assert "_des.GATE_ENV: _early_gate" in early


def test_serial_path_unchanged_when_off():
    src = _main_src()
    serial = src[src.index("    # 5. group D\n"):]
    serial = serial[:serial.index('state.pids["D"] = spec_d.pid')]
    assert "if _early_d is not None:" in serial
    assert 'spec_d, _ = _d_spec_from(dc_p, "D")' in serial
    assert "launch_group(spec_d, tree, log, dry)" in serial
    # the one D plan: the serial measured budgets and the gate's measured side
    # are built by the same budgets_from_dc call shape
    clo = src[src.index("def _d_spec_from("):src.index("def _d_early_verdict(")]
    assert "budgets_from_dc(\n            cards, dc_src, log, label, " in clo
    assert "env_d.update(extra_env or {})" in clo


def test_refuse_writes_the_gate_then_sweeps_the_group_and_restores_ns():
    src = _main_src()
    v = src[src.index("def _d_early_verdict("):src.index("    # 5. group D\n")]
    i_ref = v.index("_des.VERDICT_REFUSE")
    assert i_ref < v.index("spec.proc.wait(timeout=120)") < v.index("os.killpg(spec.proc.pid")
    assert v.index("os.killpg(spec.proc.pid") < v.index("ns.env_d, ns._d_kv_stage_written, ns._d_owner_solve = snap")
    # go only when budget_verdict holds AND the early D is still alive
    assert "spec.proc.poll()" in v and "_des.budget_verdict(" in v
