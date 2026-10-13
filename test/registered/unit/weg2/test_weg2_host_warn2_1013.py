# SPDX-License-Identifier: Apache-2.0
"""NF HOSTWARN2 (13.10.2026): the host-RAM refusals BEFORE and AT the boot WARN.

User rule 06.10. (verbatim, ``/root/.claude/CLAUDE.md`` "Host-RAM"): "Es gilt KEINE Verweigerung
wegen Host-RAM: Ledger (W20/W87), Launcher und Startskripte warnen nur ... Man laesst es darauf
ankommen, ob es OOM geht. VRAM-Verweigerungen bleiben unberuehrt."

HOSTWARN (12.10., ``test_weg2_host_warn_w98_1012``) turned the RUNTIME stops (W98 / W22 / W100 /
the launch guard) into ``HOST-RAM-WARN`` lines. This file is the other half, the gates in front of
the boot:

* the host ledger's ladder: W20 (no arm funds both moments), W21 (run peak above the hard bound),
  W97 (deviation cannot accept a peak at the watermark) -- ``host_ledger.choose(warn_only=True)``,
  which the launcher and the ledger CLI pass; W87 (the pinned arm) rides the same call;
* ``launcher.host_preflight``: the preflight script's RAM floor and the launcher's own
  ``MemAvailable < 40 GiB`` floor (code ``HOST-MEM``); the script's DISK verdict stays a refusal
  (new code ``HOST-DISK``, forceable like HOST-MEM was);
* ``pinned_host_budget.check_and_register_pinned_post`` and the three ``server_args`` parse-time
  guards: warn, register the post, the real allocation decides;
* the legacy WEG1 sizing and the turnkey host-headroom preflight.

Each test is RED on 2c8db9c8e5 (the refusal raised) and each refusal has a counter-test below
("VRAM / card / geometry stays a refusal").
"""

import ast
import logging
import os
import re
import stat
import textwrap

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import pinned_host_budget as phb
from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher, refusals
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

GIB = hl.GIB
WARN = hl.HOST_WARN_MARKER

# --- the #1360 specimen (same numbers as test_weg2_host_deviation_1360): predicted peak between
#     the hard bound and the reap watermark, so the arm is refused by the PEAK alone (W21).
_KW = dict(
    ring_bytes=int(30.4 * GIB), ring_span1_bytes=int(15.0 * GIB),
    cg_current_bytes=int(9.0 * GIB), reclaimable_bytes=int(3.0 * GIB),
    cg_ceiling_bytes=int(123.78 * GIB), s_gb_d=4,
    cg_ceiling_source="lxcfs MemTotal FALLBACK (CT999 specimen, no Docker cap)",
)


def _choose(**extra):
    return hl.choose(
        int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 150)],
        flip_ratchet=hl.resolve_flip_ratchet_gib(None), **_KW, **extra,
    )


# --- the dk5 box of test_weg2_store_priced_x_1317 for the launcher seam (W20 / W87)
DK5 = dict(memtotal=126_751_866_880, memavail=111_196_077_056, cg_current=22_719_148_032)
RING_KW = dict(ring_bytes=32964 * 1024 * 1024, ring_span1_bytes=29912 * 1024 * 1024)


def _box(tmp_path, memavail):
    mi = tmp_path / "meminfo"
    mi.write_text(f"MemTotal:       {DK5['memtotal'] // 1024} kB\nMemAvailable:   {int(memavail) // 1024} kB\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "memory.current").write_text(f"{DK5['cg_current']}\n")
    (cg / "memory.max").write_text("max\n")
    (cg / "memory.stat").write_text("anon 0\nfile 0\nslab_reclaimable 0\nslab_unreclaimable 0\nshmem 0\n")
    (cg / "memory.events").write_text("oom_kill 0\n")
    return str(mi), str(cg)


def _warn_lines(lines):
    return [ln for ln in lines if ln.startswith(WARN)]


# =========================================================================== host ledger: W21 / W97


def test_w21_run_peak_warns_and_boots_the_priced_arm():
    # red on 2c8db9c8e5: choose() has no warn_only and raises W21
    arm, _headroom, lines = _choose(warn_only=True)
    warns = _warn_lines(lines)
    assert any(" W21 HostRunPeakUnfunded -- " in w for w in warns), warns
    assert any(" W21 FrugalArmChosen -- " in w for w in warns), warns
    # the numbers are on the line, in full
    head = [w for w in warns if "HostRunPeakUnfunded" in w][0]
    assert "hard bound" in head and "GiB" in head
    # the arm that comes back is the ladder's own most frugal rung, priced as before
    assert (arm.s_gb, arm.m_mib) == (1, 150)
    assert not arm.fundable_moments or arm.predicted_run_peak_gib() is not None


def test_the_pure_verdict_still_raises_without_warn_only():
    # the default keeps the old contract: every pre-existing caller and test of the pure function
    with pytest.raises((hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)):
        _choose()
    with pytest.raises((hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)):
        _choose(warn_only=False)


def test_the_warn_line_does_not_wear_the_refusal_vocabulary():
    # death watchers / log greps read these strings as "the launcher refused the boot"
    _arm, _h, lines = _choose(warn_only=True)
    for w in _warn_lines(lines):
        head = w.split(" -- ", 1)[0]
        for old in ("Weg2HostLedgerRefused", "Weg2HostRunPeakRefused", "Weg2PinnedArmRefused",
                    "Weg2HostDeviationRefused", "WEG2-LAUNCH REFUSED"):
            assert old not in head, (old, head)


def test_warn_only_changes_no_price():
    # the table is the same with and without the verdict: same ARM lines, same terms
    with pytest.raises((hl.Weg2HostLedgerRefused, hl.Weg2HostRunPeakRefused)):
        _choose()
    arm, _h, lines = _choose(warn_only=True)
    arm_lines = [ln for ln in lines if ln.startswith("WEG2-HOST-LEDGER ARM S=")]
    assert len(arm_lines) == 1 and "refused (binding:" in arm_lines[0]
    assert arm.terms["host_ring_gib"] > 0


def test_w97_at_the_watermark_warns_and_accepts_the_deviation():
    # red on 2c8db9c8e5: Weg2HostDeviationRefused ("cannot accept a predicted run peak AT OR ABOVE the watermark")
    arm, _h, lines = hl.choose(
        int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 2400)],
        flip_ratchet=hl.FlipRatchet(per_flip_gib=40.0, flips_priced=1, source="test", from_record=False),
        deviation_reason="r", riegel_gib=10.0, warn_only=True, **_KW,
    )
    w97 = [w for w in _warn_lines(lines) if " W97 HostDeviationUnaccepted -- " in w]
    assert w97 and "reap watermark" in w97[0], lines
    assert arm.m_mib == 2400
    assert any(ln.startswith("WEG2-HOST-LEDGER DEVIATION") for ln in lines)
    # and the pure function keeps refusing
    with pytest.raises(hl.Weg2HostDeviationRefused):
        hl.choose(
            int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 2400)],
            flip_ratchet=hl.FlipRatchet(per_flip_gib=40.0, flips_priced=1, source="test", from_record=False),
            deviation_reason="r", riegel_gib=10.0, **_KW,
        )


def test_the_deviation_text_no_longer_claims_a_teardown():
    arm, _h, lines = hl.choose(
        int(123.78 * GIB), int(110.0 * GIB), arms=[(1, 150)],
        flip_ratchet=hl.resolve_flip_ratchet_gib(None), deviation_reason="r", riegel_gib=93.0, **_KW,
    )
    dev = [ln for ln in lines if ln.startswith("WEG2-HOST-LEDGER DEVIATION")]
    assert dev and "still tears down" not in dev[0] and "still warns at the latch" in dev[0]


def test_the_ledger_cli_warns_and_exits_zero(tmp_path, capsys):
    # host_ledger.main is a start script's preflight: red on 2c8db9c8e5 (rc 2 on W20)
    mi, cg = _box(tmp_path, 20 * GIB)
    rc = hl.main(["--meminfo", mi, "--cgroup", cg,
                  "--ring-bytes", str(RING_KW["ring_bytes"]), "--ring-span1-bytes", str(RING_KW["ring_span1_bytes"]),
                  "--ring-provenance", "test"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert f"{WARN} W20 HostLedgerUnfunded" in out and "WEG2_S_GB=1" in out


# =========================================================================== launcher seam: W20 / W87


def _price(tmp_path, memavail, **kw):
    mi, cg = _box(tmp_path, memavail)
    return launcher.choose_host_ledger(**RING_KW, meminfo_path=mi, cgroup_root=cg, **kw)


def test_w20_no_arm_funds_the_box_warns_with_the_full_ladder(tmp_path):
    # red on 2c8db9c8e5: Weg2HostLedgerRefused out of choose_host_ledger
    arm, _hd, lines, _cg = _price(tmp_path, 20 * GIB)
    assert not arm.fundable_moments
    assert (arm.s_gb, arm.m_mib) == (1, 600)            # the last rung of DEFAULT_ARMS
    w = _warn_lines(lines)
    assert any(" W20 HostLedgerUnfunded -- " in x for x in w), w
    assert any(" W20 FrugalArmChosen -- " in x and "launch_leftover=-" in x for x in w), w
    # the whole ladder is still printed (one ARM line per rung)
    assert len([ln for ln in lines if ln.startswith("WEG2-HOST-LEDGER ARM S=")]) >= 2


def test_a_fundable_box_prints_no_warning(tmp_path):
    arm, _hd, lines, _cg = _price(tmp_path, DK5["memavail"] - 8 * GIB + 0)
    assert arm.fundable_moments
    assert _warn_lines(lines) == []


def test_w87_an_unfundable_pinned_arm_warns_and_boots(tmp_path):
    # red on 2c8db9c8e5: "W87 Weg2PinnedArmRefused" (M=7200 is the first clearly unfundable rung on dk5)
    arm, _hd, lines, _cg = _price(tmp_path, DK5["memavail"], pin_m_mib=7200)
    assert (arm.s_gb, arm.m_mib) == (1, 7200)
    w = _warn_lines(lines)
    assert any(" W87 PinnedArmUnfunded -- " in x and "--pin-ledger-arm-m 7200" in x for x in w), w
    assert any(ln.startswith("WEG2-LEDGER ARM PINNED") for ln in lines)


def test_a_fundable_pinned_arm_is_unchanged(tmp_path):
    arm, _hd, lines, _cg = _price(tmp_path, DK5["memavail"], pin_m_mib=600)
    assert arm.fundable_moments and arm.m_mib == 600
    assert _warn_lines(lines) == []


def test_a_ledger_refusal_about_a_missing_number_still_refuses(tmp_path):
    # counter-test: warn_only covers the host-RAM VERDICTS, not a number that does not exist.
    # A half-cited prior boot (W105) is the ledger's own "no guessed half" refusal.
    with pytest.raises(hl.Weg2PriorCushionArgsIncomplete):
        _choose(warn_only=True, prior_cushion_min_gib=3.0)


# =========================================================================== host preflight


def _script(tmp_path, body, rc):
    p = tmp_path / "pre.sh"
    p.write_text("#!/bin/bash\n" + textwrap.dedent(body) + f"\nexit {rc}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


@pytest.fixture
def preflight_env(tmp_path, monkeypatch):
    if not os.path.exists("/sys/fs/cgroup/memory.events"):
        pytest.skip("host_preflight reads /sys/fs/cgroup/memory.events (not present here)")
    monkeypatch.setattr(launcher, "GPU_ARB", str(tmp_path))
    monkeypatch.setattr(hl, "swap_state_line", lambda *a, **k: ("swap: test", False))

    def meminfo(avail_gib):
        monkeypatch.setattr(hl, "read_meminfo", lambda *a, **k: {"MemTotal": 120 * GIB, "MemAvailable": int(avail_gib * GIB)})

    meminfo(100)
    return meminfo


def _run_preflight(tmp_path, monkeypatch, body, rc):
    monkeypatch.setattr(launcher, "HOST_PREFLIGHT", _script(tmp_path, body, rc))
    lines = []
    launcher.host_preflight(lines.append, "t", dry=False)
    return lines


RAM_REFUSAL = """
    echo "REFUSED: MemAvailable 9 GiB is under the 16 GiB floor (#721)." | tee -a "$1"
"""
DISK_REFUSAL = """
    echo "REFUSED: only 50 GiB free on /. The HiCache file backend writes" | tee -a "$1"
"""


def test_preflight_script_ram_floor_warns_and_goes_on(tmp_path, monkeypatch, preflight_env):
    # red on 2c8db9c8e5: refuse_value("HOST-MEM", "BOOT REFUSED by the host-ledger preflight")
    lines = _run_preflight(tmp_path, monkeypatch, RAM_REFUSAL, 1)
    w = _warn_lines(lines)
    assert len(w) == 1 and " HOST-MEM PreflightScriptFloor -- " in w[0] and "MemAvailable 9 GiB" in w[0], lines
    assert any("host preflight CONTINUED (host-RAM warnings=1)" in ln for ln in lines)


def test_preflight_own_40_gib_floor_warns_and_goes_on(tmp_path, monkeypatch, preflight_env):
    # red on 2c8db9c8e5: refuse_value("HOST-MEM", "free -g available 20.0 GiB < 40 GiB")
    preflight_env(20)
    lines = _run_preflight(tmp_path, monkeypatch, "true", 0)
    w = _warn_lines(lines)
    assert len(w) == 1 and " HOST-MEM Preflight40GiB -- " in w[0] and "20.0 GiB < 40 GiB" in w[0], lines
    assert "top RSS holders (NOT killed)" in w[0]


def test_preflight_both_floors_warn_twice(tmp_path, monkeypatch, preflight_env):
    preflight_env(20)
    lines = _run_preflight(tmp_path, monkeypatch, RAM_REFUSAL, 1)
    assert len(_warn_lines(lines)) == 2


def test_preflight_pass_is_unchanged(tmp_path, monkeypatch, preflight_env):
    lines = _run_preflight(tmp_path, monkeypatch, "true", 0)
    assert _warn_lines(lines) == []
    assert any(ln.startswith("host preflight PASS: MemAvailable 100.0 GiB") for ln in lines)


def test_preflight_disk_floor_stays_a_refusal(tmp_path, monkeypatch, preflight_env):
    # counter-test: the user rule frees host RAM, not disk; the script's one exit code carries both
    with pytest.raises(launcher.Weg2LaunchRefused) as cm:
        _run_preflight(tmp_path, monkeypatch, DISK_REFUSAL, 1)
    assert "disk floor" in str(cm.value) and "50 GiB free on /" in str(cm.value)


def test_preflight_mixed_verdict_refuses_on_the_disk_part(tmp_path, monkeypatch, preflight_env):
    with pytest.raises(launcher.Weg2LaunchRefused):
        _run_preflight(tmp_path, monkeypatch, RAM_REFUSAL + DISK_REFUSAL, 1)


def test_preflight_script_without_a_verdict_stays_a_refusal(tmp_path, monkeypatch, preflight_env):
    # a broken tool is not a host-RAM verdict
    with pytest.raises(launcher.Weg2LaunchRefused) as cm:
        _run_preflight(tmp_path, monkeypatch, "echo boom >&2", 3)
    assert "without a REFUSED verdict" in str(cm.value)


def test_preflight_disk_refusal_is_forceable_like_host_mem_was(tmp_path, monkeypatch, preflight_env):
    # --force semantics of the old (mislabelled) HOST-MEM exit are kept for the disk part
    refusals.arm(True)
    try:
        lines = _run_preflight(tmp_path, monkeypatch, DISK_REFUSAL, 1)
    finally:
        refusals.arm(False)
        os.environ.pop(refusals.ENV_FORCED_BOOT, None)
    assert any(ln.startswith("FORCED-PAST HOST-DISK ") for ln in lines)


def test_split_preflight_refusals_by_the_scripts_own_wording():
    ram, other = launcher.split_preflight_refusals(
        "noise\nREFUSED: MemAvailable 3 GiB is under\nREFUSED: only 5 GiB free on /\nREFUSED: something else\n")
    assert len(ram) == 1 and len(other) == 2


# =========================================================================== register / codes


def test_host_mem_is_a_warn_code_not_a_refusal():
    r = refusals.by_code("HOST-MEM")
    assert r.klass == refusals.CLASS_WARN and not r.forcebar
    assert refusals.warn_codes() == ["HOST-MEM"]
    assert "HOST-MEM" not in refusals.value_codes() and "HOST-MEM" not in refusals.hard_codes()
    pub = {x["code"]: x for x in refusals.public_register(["HOST-DISK"])}
    assert pub["HOST-MEM"]["wired"] is None and pub["HOST-MEM"]["klass_label"].startswith("warns")
    assert pub["HOST-DISK"]["wired"] is True


def test_the_launcher_no_longer_consults_the_switch_for_host_mem():
    src = open(launcher.__file__, encoding="utf-8").read()
    wired = refusals.wired_codes(src)
    assert "HOST-MEM" not in wired
    assert "HOST-DISK" in wired
    # an unarmed call of the old code is a plain raise (a warn code is never forceable)
    with pytest.raises(launcher.Weg2LaunchRefused):
        refusals.refuse_value("HOST-MEM", "x", launcher.Weg2LaunchRefused)


def test_the_force_help_says_host_ram_is_not_refused():
    src = open(launcher.__file__, encoding="utf-8").read()
    i = src.index('"--force", action="store_true"')
    h = src[i:i + 2500]
    assert "HOST RAM is not refused at all any more" in h and "(HW-COUNT, HW-UNCALIBRATED, HOST-DISK)" in h


# =========================================================================== pinned-host registry


@pytest.fixture
def pinned(monkeypatch, caplog):
    phb.clear_registered_posts()
    monkeypatch.setattr(phb, "pinned_host_memory_bytes", lambda: (100 * 10**9, 20 * 10**9))
    caplog.set_level(logging.WARNING, logger=phb.logger.name)
    yield caplog
    phb.clear_registered_posts()


def test_pinned_over_commit_warns_registers_and_lets_the_allocation_decide(pinned):
    # red on 2c8db9c8e5: ValueError("Pinned host RAM over-committed ...")
    phb.check_and_register_pinned_post("HiCache host tier", "--hicache-size", 50 * 10**9)
    msgs = [r.getMessage() for r in pinned.records]
    assert any(m.startswith(f"{WARN} PINNED HiCache host tier -- Pinned host RAM over-committed") for m in msgs), msgs
    assert "the allocation is attempted anyway" in msgs[-1]
    assert [p.name for p in phb.registered_posts()] == ["HiCache host tier"]


def test_pinned_second_post_sees_the_first_and_warns_again(pinned):
    phb.check_and_register_pinned_post("A", "--a", 50 * 10**9)
    phb.check_and_register_pinned_post("B", "--b", 10 * 10**9)
    assert sorted(p.name for p in phb.registered_posts()) == ["A", "B"]
    assert len([r for r in pinned.records if "PINNED" in r.getMessage()]) == 2


def test_pinned_fitting_post_is_silent(pinned):
    phb.check_and_register_pinned_post("A", "--a", 1 * 10**9)
    assert [r for r in pinned.records if WARN in r.getMessage()] == []


def test_the_pure_budget_function_still_returns_its_message():
    msg = phb.joint_pinned_host_error([phb.PinnedHostPost("A", "--a", 50 * 10**9)], 100 * 10**9, 20 * 10**9, 10 * 10**9)
    assert msg and "over-committed" in msg


def test_the_marker_is_one_string_in_both_modules():
    assert phb.HOST_WARN_MARKER == hl.HOST_WARN_MARKER == "HOST-RAM-WARN"


def test_a_failing_real_allocation_is_still_an_error_and_takes_its_post_back(pinned):
    # the warning does not swallow the real thing: when the allocation raises, the decorator undoes the
    # post it registered (#729) and the original error propagates untouched
    @phb.revert_pinned_posts_on_failure
    def _init():
        phb.check_and_register_pinned_post("P", "--p", 50 * 10**9)
        raise MemoryError("cudaHostAlloc failed")

    with pytest.raises(MemoryError):
        _init()
    assert phb.registered_posts() == ()


def _server_args_ast():
    path = os.path.join(os.path.dirname(phb.__file__), "..", "server_args.py")
    return ast.parse(open(path, encoding="utf-8").read())


def test_server_args_parse_time_pinned_guards_warn_instead_of_raising():
    # red on 2c8db9c8e5: three `raise ValueError(_err | err)` right after joint_pinned_host_error / host_ram_budget_error
    bad = []
    for n in ast.walk(_server_args_ast()):
        if (isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and getattr(n.exc.func, "id", "") == "ValueError"
                and len(n.exc.args) == 1 and isinstance(n.exc.args[0], ast.Name) and n.exc.args[0].id in ("err", "_err")):
            bad.append(n.lineno)
    assert bad == [], bad
    src = open(os.path.join(os.path.dirname(phb.__file__), "..", "server_args.py"), encoding="utf-8").read()
    assert src.count("warn_pinned_host(") == 3


# =========================================================================== legacy WEG1 sizing, turnkey


def test_weg1_ledger_without_a_kv_budget_warns_and_sizes_one_gb():
    from sglang.srt.planner import weg1_host_sizing as W

    r = W.size_host_pools(max_running_requests=8, chunked_prefill_size=1600, ranks=3, memavail_bytes=30 * 10**9,
                          cell_pp0_bytes=39365, prompt_max_tokens=60000, n_queue=8, mamba_host_mib=2000,
                          per_slot_rank0_mib=37.41, device_slots=20)
    assert r.s_ledger_gb == 1 and r.s_gb == 1
    w = [ln for ln in r.lines if ln.startswith(f"{WARN} WEG1 LedgerNoKvBudget")]
    assert len(w) == 1 and "30.00 GB" in w[0]


def test_weg1_geometry_floors_still_refuse():
    # counter-test: rows / anchors are pool geometry (the in-tree G10 floors), not host RAM
    from sglang.srt.planner import weg1_host_sizing as W

    with pytest.raises(W.SizingRefused):
        W.size_host_pools(max_running_requests=8, chunked_prefill_size=1600, ranks=3, memavail_bytes=119 * 10**9,
                          cell_pp0_bytes=39365, prompt_max_tokens=60000, n_queue=8, mamba_host_mib=10,
                          per_slot_rank0_mib=37.41, device_slots=20)
    with pytest.raises(W.SizingRefused):
        W.size_host_pools(max_running_requests=0, chunked_prefill_size=1600, ranks=3, memavail_bytes=119 * 10**9,
                          cell_pp0_bytes=39365, prompt_max_tokens=60000, n_queue=8, mamba_host_mib=2000,
                          per_slot_rank0_mib=37.41, device_slots=20)


def test_turnkey_host_headroom_warns_and_does_not_refuse(capsys):
    from sglang.srt.turnkey import preflight as PF

    class _Cfg:
        class preflight:
            host_headroom_gib = 15

    class _P:
        @staticmethod
        def mem_available_bytes():
            return 1 << 30

    assert PF.check_host_headroom(_Cfg, _P) is None
    err = capsys.readouterr().err
    assert f"{WARN} TURNKEY HostHeadroom" in err and "1.0 GiB < 15 GiB" in err


# =========================================================================== VRAM / card / topology: unchanged


def test_vram_and_card_codes_keep_their_classes():
    for c in ("D-BUDGET", "P-CARD", "WAKE-CREDIT", "PP-CUT", "HW-UNCALIBRATED", "HW-COUNT", "HW-BORROWED"):
        assert refusals.by_code(c).klass == refusals.CLASS_VALUE, c
    for c in ("KARTE-BELEGT", "HW-ARCH", "HW-TOPOLOGY", "SHM-BELEGT", "PORT-BELEGT"):
        assert refusals.by_code(c).klass == refusals.CLASS_HARD, c


def test_vram_refusal_sites_of_the_launcher_are_untouched():
    # counted on 2c8db9c8e5 (the base of this work): the launcher's refuse_value sites per VRAM / card code
    src = open(launcher.__file__, encoding="utf-8").read()
    counts = {c: len(re.findall(r'refuse_value\(\s*"%s"' % re.escape(c), src))
              for c in ("D-BUDGET", "P-CARD", "WAKE-CREDIT", "HW-COUNT", "HW-UNCALIBRATED", "HW-BORROWED")}
    # NF tree: only the gates BEFORE the flip path are wired to --force; D-BUDGET / P-CARD / WAKE-CREDIT are plain raises here
    assert counts == {"D-BUDGET": 0, "P-CARD": 0, "WAKE-CREDIT": 0, "HW-COUNT": 2, "HW-UNCALIBRATED": 1,
                      "HW-BORROWED": 1}, counts
    # 92 `raise Weg2LaunchRefused` on 2c8db9c8e5; HOSTWARN2 adds exactly one (the preflight script without a verdict)
    assert src.count("raise Weg2LaunchRefused") == 93


def test_a_foreign_process_on_a_card_still_refuses(monkeypatch):
    class _M:
        tenant_used_mib = 2000
        tenant_used_instrument = "test"
        reserved_mib = 500
        free_mib = 10
        allocatable_mib = 10

    card = launcher.Card(nvml_index=0, uuid="GPU-aaaa", name="RTX 3080", total_mib=20480)
    monkeypatch.setattr(launcher, "nvml_memory", lambda cards: {c.uuid: _M() for c in cards})
    with pytest.raises(launcher.Weg2LaunchRefused) as cm:
        launcher.cards_free_check([card], lambda m: None)
    assert "not free; not killing anything" in str(cm.value)


def test_an_unnamed_board_still_refuses_the_vram_residue_borrow():
    other = launcher.Card(nvml_index=3, uuid="GPU-ffffffff", name="NVIDIA L4", total_mib=24564)
    with pytest.raises(launcher.Weg2LaunchRefused) as cm:
        launcher.dc_measured_d_mib(other, launcher.WEIGHT_SOURCE_EXCHANGE)
    assert "W19" in str(cm.value)


def _do_stop_names(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "do_stop" and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
            out.append(n.args[0].value.split(" ", 1)[0])
    return sorted(out)


def test_the_fronts_vram_and_rank_death_stops_are_untouched():
    names = _do_stop_names(os.path.join(os.path.dirname(launcher.__file__), "front.py"))
    # the host-RAM stops are gone (HOSTWARN, 12.10.) ...
    assert "W22" not in names and "W98" not in names
    # ... the VRAM / rank-death ones are all still there
    for code, n in (("W17", 3), ("W19", 1), ("W4", 4), ("W2", 1), ("W3", 2), ("W9", 1)):
        assert names.count(code) == n, (code, names.count(code), names)
