"""BOOTZEIT 3 (29.09.): early D start -- the launcher half.

z30r3: D's init (30 s) starts only after P READY + sleep(P) + the D plan (8 s).
``--weg2-d-early-start on`` launches D with P, planned from the planner seat's
ONE expectation selector (``d_expect_dormant_other``, D-EXPECT 9f74489683),
and releases D's weight load only after sleep(P), iff every planned budget is
<= the measured one. 'off' (default) keeps the serial path.
"""

import argparse
import inspect

import pytest

from sglang.srt.weg2 import d_early_start as des
from sglang.srt.weg2 import launcher as L


def test_flag_defaults_auto_and_help_renders():
    ap = L.build_parser()
    ns = ap.parse_known_args(["--tree", "t", "--tag", "x"])[0]
    assert ns.weg2_d_early_start == "auto"
    # the parser's default profile is the 27B: auto stays serial there
    assert not L._d_early_start_armed(ns)
    assert L._d_early_start_armed(argparse.Namespace(weg2_d_early_start="on"))
    assert not L._d_early_start_armed(argparse.Namespace())
    assert "--weg2-d-early-start" in ap.format_help()


def _main_src():
    return inspect.getsource(L.main)


def test_early_start_is_armed_only_by_the_flag_and_after_the_dry_branch():
    src = _main_src()
    i_armed = src.index("_early_d = None\n    if _d_early_start_armed(ns):")
    assert src.index('log("DRY-RUN complete: nothing started, mounted, armed or written")') < i_armed
    assert i_armed < src.index('state.pids["P"] = spec_p.pid')


def test_early_plan_reads_the_planner_selector_no_second_source():
    src = _main_src()
    early = src[src.index("_early_d = None\n    if _d_early_start_armed(ns):"):src.index("def _d_early_verdict(")]
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


# --- 27B review of cb98c3d94a: W185 and stage 0 ---


def _legacy_27b_row():
    import dataclasses
    from unittest import mock

    from sglang.srt.weg2 import form as F

    row = dataclasses.replace(F.PROFILES["qwen27b"], d_expect_from_p_records=False)
    return mock.patch.dict(F.PROFILES, {"qwen27b": row})


def test_w185_refuses_on_where_the_expectation_is_legacy_and_passes_next_flash():
    # 29.09. (step 1): W185 keys on d_expect_from_p_records; the 27B row set
    # it, so W185 is exercised on a legacy row (what the 27B row was)
    with _legacy_27b_row():
        with pytest.raises(L.Weg2DEarlyStartUnreviewed, match="W185"):
            L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="on", profile=None))
        with pytest.raises(L.Weg2DEarlyStartUnreviewed):
            L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="on", profile="qwen27b"))
    L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="on", profile="nextflash"))
    L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="on", profile="qwen27b"))
    L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="off", profile=None))


def test_w185_and_the_p_report_dir_come_before_group_p_starts():
    src = _main_src()
    i_p = src.index("launch_group(spec_p, tree, log, dry)")
    assert src.index("refuse_d_early_start_unreviewed(ns)") < i_p
    assert src.index("spec_p.env[_des.P_SIZED_DIR_ENV]") < i_p


def _watch(tmp_path, monkeypatch, samples, sized_after):
    """Drive d_early_stage0_watch on a fake clock: P rank k reports sized at
    tick sized_after[k]; samples[i] is the early D's VRAM reading at tick i."""
    monkeypatch.setattr(L, "boot_state_write", lambda *a, **k: None)
    gate = str(tmp_path / "s0.json")
    sized = tmp_path / "sized"
    sized.mkdir()
    tick = [0]

    def vram(_pids):
        for k, t in enumerate(sized_after):
            if t == tick[0]:
                (sized / f"p_sized.pp{k}tp0.json").write_text(
                    '{"pp_rank": %d, "tp_rank": 0, "used_by_me_mib": %d}' % (k, 1000 + k))
        return samples[min(tick[0], len(samples) - 1)]

    result, lines = {}, []
    L.d_early_stage0_watch(gate, str(sized), 3, lambda: {1}, lambda: True, lambda: True, vram,
                           {"u1": "nvml1"}, result, lines.append,
                           clock=lambda: float(tick[0]), sleep=lambda _s: tick.__setitem__(0, tick[0] + 1))
    return des.read_gate(gate), result, lines


def test_stage0_watch_goes_once_every_p_rank_is_sized(tmp_path, monkeypatch):
    doc, result, lines = _watch(tmp_path, monkeypatch, [{}], [1, 2, 4])
    assert doc["verdict"] == "go" and result["verdict"] == "go"
    assert result["p_used_by_me_mib"] == {"pp0tp0": 1000, "pp1tp0": 1001, "pp2tp0": 1002}
    assert any("GO after 4.0 s" in l for l in lines)


def test_stage0_watch_refuses_when_d_held_vram_before_p_was_sized(tmp_path, monkeypatch):
    doc, result, lines = _watch(tmp_path, monkeypatch, [{}, {"u1": 888}], [5, 5, 5])
    assert doc["verdict"] == "refuse" and result["d_held_mib"] == {"u1": 888}
    assert any("nvml1: D held 888 MiB before P was sized" in l for l in lines)


def test_journal_env_for_p_and_the_foreign_term_after_sleep():
    ap = L.build_parser()
    ns = ap.parse_known_args(["--tree", "t", "--tag", "x"])[0]
    assert ns.weg2_p_free_read_journal == "off"
    src = _main_src()
    i_p = src.index("launch_group(spec_p, tree, log, dry)")
    assert src.index("spec_p.env[_des.FREE_READ_JOURNAL_ENV]") < i_p
    i_dc = src.index("state.dc_measured_p = dc_p")
    tail = src[i_dc:i_dc + 1500]
    assert '"p_first_sleep_done"' in tail and "d_early_foreign_mib" in tail
    assert "nvml_process_mib(session_pids(_early_d[0].pid))" in tail  # per PID, measured


def test_load_gate_needs_stage0_go_and_d_gets_both_gates():
    src = _main_src()
    v = src[src.index("def _d_early_verdict("):src.index("    # 5. group D\n")]
    assert 'stage0.get("verdict") != _des.VERDICT_GO' in v
    early = src[src.index("_early_snap = ("):src.index("def _d_early_verdict(")]
    assert "_des.STAGE0_ENV: _early_stage0" in early and "d_early_stage0_watch" in early


# --- Default auto (Serie 29.09., dearly z30x2 424346f693: serving 205 s gegen frp 226 s) ---


def test_auto_arms_next_flash_and_leaves_the_27b_serial():
    assert L._d_early_start_armed(argparse.Namespace(weg2_d_early_start="auto", profile="nextflash"))
    assert not L._d_early_start_armed(argparse.Namespace(weg2_d_early_start="auto", profile="qwen27b"))
    assert not L._d_early_start_armed(argparse.Namespace(weg2_d_early_start="auto", profile=None))
    assert not L._d_early_start_armed(argparse.Namespace(weg2_d_early_start="off", profile="nextflash"))


def test_w185_refuses_only_an_explicit_on_never_auto():
    # auto resolves to off on the 27B -- a default must not refuse a 27B boot
    L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="auto", profile="qwen27b"))
    L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="auto", profile=None))
    L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="auto", profile="nextflash"))
    with _legacy_27b_row():
        L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="auto", profile="qwen27b"))
        with pytest.raises(L.Weg2DEarlyStartUnreviewed):
            L.refuse_d_early_start_unreviewed(argparse.Namespace(weg2_d_early_start="on", profile="qwen27b"))
