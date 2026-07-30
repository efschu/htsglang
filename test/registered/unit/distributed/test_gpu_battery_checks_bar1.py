# SPDX-License-Identifier: Apache-2.0
"""Dry tests for the BAR1 block of the GPU battery (s10-s12).

Same contract as test_gpu_battery_checks.py: the checks are driven as
SUBPROCESSES against synthetic fixtures, because what the executor depends on
is "exactly one line on stdout and an exit code of 0/1/2", not the logic behind
it.

Two things get extra attention here, both because they are the ways a green
verdict could be wrong rather than merely absent:

  * THE LOG STRINGS ARE THE REAL ONES. The fixtures carry the lines
    parallel_state and htccl_bar1 actually emit, including the grep line-number
    prefix. A regex that silently stops matching would turn "no group reported
    a fallback" into "every group is fine", which is exactly the failure mode
    the ERREICHT/angefordert split was introduced to end.
  * THE BOLT HAS ITS OWN TEST. The all_gather coverage gap is the expected
    outcome of the current integration; it must produce a FAIL that names
    itself, not a generic crash verdict.

Hermetic and CPU-only: no card, no host, no ssh, no lock.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
CHECKS = os.path.join(BATTERY, "checks")

sys.path.insert(0, BATTERY)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bar1_marker_source as _src  # noqa: E402
from battery_steps import STEPS_BY_ID  # noqa: E402
from s10_bar1_driver import compose as s10_compose  # noqa: E402
from s10_bar1_driver import desired_state_reached  # noqa: E402
from s11_bar1_e2e import (  # noqa: E402
    compose,
    parse_graph_check,
    parse_log_evidence,
    parse_smoke,
)
from s12_prefill_kurve import tabelle, zusammenfassen  # noqa: E402

# The lines the code really writes -- built from the ACTUAL format strings in
# parallel_state.py / htccl.py / htccl_bar1.py / benchmark/bar1_graph_check.py
# via _bar1_marker_source.py, not retyped. #315: this file's fixtures used to
# be hand-typed and had drifted onto the German wording #295 moved the
# emitters away from -- both sides matched each other and nothing noticed.
# Deriving them from source instead makes that drift impossible: a wrong
# render here can only mean the source itself changed, which is exactly what
# should make this file (and test_bar1_marker_coupling.py) fail loudly.
#
# The "<lineno>:" prefix is grep -n's, not the emitter's; it stays hand-added.
LOG_GROUP_OK = "412:[2026-07-30 03:11:02] " + _src.render_group_ok_line(
    group="tp:0"
)
LOG_GROUP_OK_DCP = "418:[2026-07-30 03:11:02] " + _src.render_group_ok_line(
    group="dcp:0"
)
LOG_GROUP_FALLBACK_DCP = "418:[2026-07-30 03:11:02] " + _src.render_group_fallback_line(
    group="dcp:0", achieved="gloo", stage="setup",
    reason="Bar1Unavailable: the holder reports ENOMEM",
)
LOG_AUFBAU = "400:[2026-07-30 03:11:01] " + _src.render_setup_line(
    dauer_ms=27, peer_targets=2, region_mib=24.0, slots_desc="8 slots",
    slot_kib=512, payload_kib=20480, flags_bytes=256, export="dma-buf",
)
LOG_KASSE_TP = "401:[2026-07-30 03:11:01] " + _src.render_ledger_line(
    group="tp:0", balance="tp:0: 24.0 MiB",
)
LOG_KASSE_DCP = "409:[2026-07-30 03:11:01] " + _src.render_ledger_line(
    group="dcp:0", balance="tp:0: 24.0 MiB, dcp:0: 24.0 MiB",
)
LOG_RIEGEL = "902:RuntimeError: " + _src.render_riegel_message(
    op="all_gather", nbytes=10600448,
    grund="bar1 reports handles('all_gather', 10600448) -> False",
)

GRAPH_CHECK_OK = _src.render_graph_check_transcript()
GRAPH_CHECK_FALLEN = _src.render_graph_check_transcript(
    gate_cases=(
        ("einfach", True, True),
        ("two-graphs", True, False),
        ("wechselnde-form", True, True),
        ("reservation", True, True),
        ("gitter", False, True),
    )
)

SMOKE_TEXT = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20."
#: Die Fortsetzung, die /generate auf "1 2 3 4" liefern muss. Ab 5, weil die
#: ersten vier im Prompt stehen und nichts belegen.
SMOKE_FORTSETZUNG = " " + " ".join(str(i) for i in range(5, 26))


# ---------------------------------------------------------------------------
# harness (same contract as the base file)
# ---------------------------------------------------------------------------


def run_check(check: str, step_dir) -> tuple:
    proc = subprocess.run(
        [sys.executable, os.path.join(CHECKS, check), "--step-dir", str(step_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert (
        len(lines) == 1
    ), f"{check} gab {len(lines)} Zeilen aus, der Executor liest genau eine: {lines}"
    return proc.returncode, lines[0]


def assert_pass(check: str, step_dir, step_id: str) -> None:
    rc, line = run_check(check, step_dir)
    assert rc == 0, f"{check}: erwartet PASS, bekam rc={rc} / {line}"
    assert line == f"BATTERY-PASS {step_id}", line


def assert_fail(check: str, step_dir, step_id: str) -> str:
    rc, line = run_check(check, step_dir)
    assert rc == 1, f"{check}: erwartet FAIL, bekam rc={rc} / {line}"
    assert line.startswith(f"BATTERY-FAIL {step_id}: "), line
    return line


def assert_stop(check: str, step_dir, step_id: str) -> str:
    rc, line = run_check(check, step_dir)
    assert rc == 2, f"{check}: erwartet STOP, bekam rc={rc} / {line}"
    assert line.startswith(f"BATTERY-STOP {step_id}: "), line
    return line


def write_json(path, payload) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# the BAR1 block in the step table
# ---------------------------------------------------------------------------


class TestBar1StepTable:
    IDS = ("s10_bar1_driver", "s11_bar1_e2e", "s12_prefill_kurve")

    def test_the_three_steps_exist(self):
        for step_id in self.IDS:
            assert step_id in STEPS_BY_ID

    def test_they_are_a_chain(self):
        """Unlike s00-s09 these depend on each other: without the patched
        driver there is no direct path, and without a run that provably went
        over it there is nothing to put on a curve."""
        assert STEPS_BY_ID["s11_bar1_e2e"].deps == ("s10_bar1_driver",)
        assert STEPS_BY_ID["s12_prefill_kurve"].deps == ("s11_bar1_e2e",)

    def test_the_old_block_stays_independent(self):
        """s00-s09 must not gain a dependency on the BAR1 block -- the battery
        has to stay runnable on a stock driver."""
        for step_id, step in STEPS_BY_ID.items():
            if step_id in self.IDS:
                continue
            assert not any(d in self.IDS for d in step.deps), step_id

    def test_the_curve_is_not_retryable(self):
        """Eight boots. A retry is not an unattended decision."""
        assert not STEPS_BY_ID["s12_prefill_kurve"].retryable

    def test_container_side_locks_are_taken(self):
        """The host locks are the step's own job; the container ones are
        run_step.sh's, and both are needed because the two /tmp namespaces do
        not see each other."""
        for step_id in self.IDS:
            assert STEPS_BY_ID[step_id].locks == "battery", step_id
            assert STEPS_BY_ID[step_id].needs_cards, step_id

    def test_budgets_leave_room(self):
        for step_id in self.IDS:
            step = STEPS_BY_ID[step_id]
            assert step.timeout_s > step.expected_min * 60, step_id

    def test_the_two_arms_differ_in_the_htccl_variables_and_nothing_else(
        self, tmp_path
    ):
        """The comparison IS the difference between the arms. Generating both
        from one template makes that testable: anything but the arm label and
        the SGLANG_HTCCL* array showing up in the diff means the two boots
        differ in something nobody intended."""
        script = tmp_path / "gen.sh"
        script.write_text(
            "set -uo pipefail\n"
            f"cd {BATTERY}\n"
            "source ./battery_common.sh\n"
            "source ./battery_host.sh\n"
            "source ./_bar1_host_boot.sh\n"
            f"bar1_write_boot_script {tmp_path}/a.sh bar1 /l.log /p.pid 30030\n"
            f"bar1_write_boot_script {tmp_path}/b.sh grundlinie /l.log /p.pid 30030\n"
        )
        proc = subprocess.run(
            ["bash", str(script)], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        a = (tmp_path / "a.sh").read_text().splitlines()
        b = (tmp_path / "b.sh").read_text().splitlines()
        assert len(a) == len(b)
        unterschiede = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(unterschiede) == 2, unterschiede
        assert all(
            "arm" in x or "HTCCL_ENV" in x for x, _ in unterschiede
        ), unterschiede
        arm_a = next(x for x, _ in unterschiede if "HTCCL_ENV" in x)
        arm_b = next(y for _, y in unterschiede if "HTCCL_ENV" in y)
        for var in (
            "SGLANG_HTCCL=1",
            "SGLANG_HTCCL_TRANSPORT=bar1",
            "SGLANG_HTCCL_GRAPH_FREIGABE=1",
        ):
            assert var in arm_a
            assert var not in arm_b
        # Two traps from 04_BETRIEB / 05_FALLEN, in both arms.
        for line in a + b:
            assert "CUDA_DEVICE_ORDER" not in line
        assert any("CUDA_HOME=" in line for line in a)
        assert any("--rank-auto-reserve-mib 3000,2700,2700" in line for line in a)
        assert any("--rank-auto-reserve-mib 3000,2700,2700" in line for line in b)
        for generated in (tmp_path / "a.sh", tmp_path / "b.sh"):
            syn = subprocess.run(
                ["bash", "-n", str(generated)], capture_output=True, text=True
            )
            assert syn.returncode == 0, syn.stderr

    def test_host_plumbing_and_restore_exist(self):
        for name in (
            "battery_host.sh",
            "_bar1_host_boot.sh",
            "s10_restore.sh",
            "s12_prefill_kurve.py",
        ):
            assert os.path.exists(os.path.join(BATTERY, name)), name


# ---------------------------------------------------------------------------
# s10 driver state
# ---------------------------------------------------------------------------


def _driver(**over) -> dict:
    payload = {
        "kind": "bar1_driver_state",
        "schema_version": 1,
        "host": "192.168.0.1",
        "reachable": True,
        "reload_performed": False,
        "blocked": None,
        "viewers_blocking": [],
        "viewers_killed": [],
        "compute_apps": [],
        "regkey_expected": "RMSmallBarP2PPeerBar1=1",
        "regkey_line": 'RegistryDwords: "RMSmallBarP2PPeerBar1=1"',
        "regkey_present": True,
        "strings_smallbar": "37",
        "patch_level": "voll",
        "srcversion_loaded": "12526C5E114DB32BA29FA63",
        "srcversion_file": "12526C5E114DB32BA29FA63",
        "module_identity_matches": True,
        "modules": {
            "nvidia": "29",
            "nvidia_uvm": "8",
            "nvidia_modeset": "0",
            "dmabuf_holder": "0",
        },
        "dmabuf_holder_loaded": True,
        "dmabuf_dev_present": True,
        "cards": [
            {
                "nvml_index": "0",
                "name": "NVIDIA GeForce RTX 3080",
                "uuid": "GPU-a",
                "pci_bus_id": "00000000:05:00.0",
            },
            {
                "nvml_index": "1",
                "name": "NVIDIA GeForce RTX 5090",
                "uuid": "GPU-b",
                "pci_bus_id": "00000000:0a:00.0",
            },
            {
                "nvml_index": "2",
                "name": "NVIDIA GeForce RTX 3080",
                "uuid": "GPU-c",
                "pci_bus_id": "00000000:0b:00.0",
            },
        ],
    }
    payload.update(over)
    return payload


def _write_driver(tmp_path, payload=None):
    write_json(
        tmp_path / "driver_state.json", payload if payload is not None else _driver()
    )
    return tmp_path


class TestDriverCheck:
    CHECK, STEP = "check_s10_bar1_driver.py", "s10_bar1_driver"

    def test_pass(self, tmp_path):
        _write_driver(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_unreachable_host_is_stop(self, tmp_path):
        _write_driver(tmp_path, _driver(reachable=False))
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_viewer_blocking_is_stop(self, tmp_path):
        """nvtop holds the modules. Terminating it is a user decision, so the
        step reports instead of deciding."""
        _write_driver(
            tmp_path,
            _driver(blocked="Prozesse halten die Module:\n 709685 root nvtop nvtop"),
        )
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "BAR1_VIEWER_KILL_OK" in line

    def test_compute_apps_are_stop(self, tmp_path):
        _write_driver(tmp_path, _driver(compute_apps=["12345, python, 8000 MiB"]))
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_missing_regkey_is_fail(self, tmp_path):
        _write_driver(
            tmp_path, _driver(regkey_present=False, regkey_line='RegistryDwords: ""')
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Regkey" in line

    def test_minimal_patch_is_fail(self, tmp_path):
        _write_driver(tmp_path, _driver(patch_level="minimal", strings_smallbar="1"))
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "minimal" in line

    def test_loaded_module_is_not_the_patched_one_is_fail(self, tmp_path):
        """The regkey line proves a parameter, not an identity."""
        _write_driver(
            tmp_path,
            _driver(module_identity_matches=False, srcversion_loaded="AAA0000"),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "srcversion" in line

    def test_holder_module_missing_is_fail(self, tmp_path):
        _write_driver(tmp_path, _driver(dmabuf_holder_loaded=False))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_holder_device_missing_is_fail(self, tmp_path):
        _write_driver(tmp_path, _driver(dmabuf_dev_present=False))
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "/dev/dmabuf_holder" in line

    def test_card_lost_after_reset_is_fail(self, tmp_path):
        payload = _driver()
        payload["cards"] = payload["cards"][:2]
        _write_driver(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "PCI-Reset" in line

    def test_card_without_uuid_is_fail(self, tmp_path):
        payload = _driver()
        payload["cards"][1]["uuid"] = ""
        _write_driver(tmp_path, payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_uvm_missing_is_fail(self, tmp_path):
        payload = _driver()
        payload["modules"]["nvidia_uvm"] = ""
        _write_driver(tmp_path, payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_freshly_loaded_unused_module_still_passes(self, tmp_path):
        """lsmod's use count is "0" for a module nobody uses yet, which is
        exactly the state right after the reload. Reading that as "absent"
        would fail the step for having just succeeded."""
        payload = _driver(reload_performed=True)
        payload["modules"] = {
            "nvidia": "0",
            "nvidia_uvm": "0",
            "nvidia_modeset": "",
            "dmabuf_holder": "0",
        }
        _write_driver(tmp_path, payload)
        assert_pass(self.CHECK, tmp_path, self.STEP)


class TestDriverCompose:
    """The composer reads the raw host probe. These are the two decisions it
    makes on its own: patch level and module identity."""

    @staticmethod
    def _probe(tmp_path, **over):
        host = tmp_path / "host"
        os.makedirs(host, exist_ok=True)
        fields = {
            "kernel": "6.17.2-2-pve",
            "regkey_line": 'RegistryDwords: "RMSmallBarP2PPeerBar1=1"',
            "srcversion_loaded": "ABC",
            "srcversion_file": "ABC",
            "strings_smallbar": "37",
            "mod_nvidia": "29",
            "mod_nvidia_uvm": "8",
            "mod_dmabuf_holder": "0",
            "dmabuf_dev": "ja",
        }
        fields.update(over)
        (host / "state_before.txt").write_text(
            "".join(f"{k}={v}\n" for k, v in fields.items())
        )
        (host / "reach.txt").write_text("ok\n")
        (host / "cards_before.csv").write_text(
            "0, RTX 3080, GPU-a, 00000000:05:00.0, 20480, 12\n"
            "1, RTX 5090, GPU-b, 00000000:0a:00.0, 32768, 12\n"
            "2, RTX 3080, GPU-c, 00000000:0b:00.0, 20480, 12\n"
        )
        return tmp_path

    def test_desired_state_recognised(self, tmp_path):
        self._probe(tmp_path)
        payload = s10_compose(str(tmp_path), phase="before")
        ok, missing = desired_state_reached(payload)
        assert ok, missing
        assert payload["patch_level"] == "voll"
        assert len(payload["cards"]) == 3

    def test_minimal_patch_and_stale_module_are_both_caught(self, tmp_path):
        self._probe(tmp_path, strings_smallbar="1", srcversion_loaded="OTHER")
        payload = s10_compose(str(tmp_path), phase="before")
        ok, missing = desired_state_reached(payload)
        assert not ok
        assert any("Patch-Stand" in m for m in missing)
        assert any("srcversion" in m for m in missing)

    def test_empty_regkey_is_missing(self, tmp_path):
        self._probe(tmp_path, regkey_line='RegistryDwords: ""')
        payload = s10_compose(str(tmp_path), phase="before")
        assert not payload["regkey_present"]


# ---------------------------------------------------------------------------
# s11 end to end
# ---------------------------------------------------------------------------


def _e2e(**over) -> dict:
    payload = {
        "kind": "bar1_e2e",
        "schema_version": 5,
        "host": "192.168.0.1",
        "reachable": True,
        "integration_present": True,
        "port": 30030,
        "server_log_remote": "/root/battery-bar1/s11.server.log",
        "transport_angefordert": "bar1",
        "graph_freigabe": True,
        "graph_check": {
            "rc": 0,
            "cases": 5,
            "gate_cases": 4,
            "gefallen": [],
            "alle_bestanden": True,
        },
        "gruppen": [
            {"group": "dcp:0", "requested": "bar1", "achieved": "bar1"},
            {"group": "tp:0", "requested": "bar1", "achieved": "bar1"},
        ],
        "gruppen_bar1": ["dcp:0", "tp:0"],
        "gruppen_ausgewichen": [],
        "aufbau_gruppen": ["tp:0", "dcp:0"],
        "aufbau_lines": 6,
        "aufbau_ms": [27.0, 31.0],
        "riegel": None,
        "fatal": None,
        "log_quellen": ["htccl_lines.txt", "server.log"],
        "log_zeilen": 42,
        "smoke": {
            "vorhanden": True,
            "endpunkt": "generate",
            "content_prefix": SMOKE_FORTSETZUNG,
            "spec_accept_length": 2.9,
            "spec_verify_ct": 41,
            "finish_reason": "length",
            "zahlen_in_folge": 16,
            "zahlen_erwartet": 16,
            "anker_zahlen": 16,
            "anker_min": 4,
            "drift_zeichen": 0,
            "muell_befunde": [],
            "lm_intakt": True,
            "kohaerent": True,
            "unterprovisioniert": False,
            "error": None,
        },
    }
    payload.update(over)
    return payload


def _write_e2e(tmp_path, payload=None, log="alles gut\n"):
    write_json(tmp_path / "bar1_e2e.json", payload if payload is not None else _e2e())
    if log is not None:
        (tmp_path / "server.log").write_text(log)
    return tmp_path


class TestE2ECheck:
    CHECK, STEP = "check_s11_bar1_e2e.py", "s11_bar1_e2e"

    def test_pass(self, tmp_path):
        _write_e2e(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_without_log_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_with_log_is_fail(self, tmp_path):
        (tmp_path / "server.log").write_text("loading weights ...\n")
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_unreachable_is_stop(self, tmp_path):
        _write_e2e(tmp_path, _e2e(reachable=False))
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_missing_integration_is_stop(self, tmp_path):
        _write_e2e(tmp_path, _e2e(integration_present=False))
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "BAR1_HOST_WT" in line

    def test_blocked_step_is_stop_and_not_diagnosed_as_a_gate_failure(self, tmp_path):
        """A step that never got the cards leaves the same empty artifacts as a
        step whose gate never ran. The reason has to name the lock."""
        _write_e2e(
            tmp_path,
            _e2e(
                blocked="Host-Locks fremd gehalten -- nicht gebrochen",
                graph_check={
                    "rc": None,
                    "cases": 0,
                    "gate_cases": 0,
                    "gefallen": [],
                    "alle_bestanden": False,
                },
            ),
        )
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "blockiert" in line
        assert "Lock" in line

    def test_graph_gate_never_ran_is_stop(self, tmp_path):
        _write_e2e(
            tmp_path,
            _e2e(
                graph_check={
                    "rc": None,
                    "cases": 0,
                    "gate_cases": 0,
                    "gefallen": [],
                    "alle_bestanden": False,
                }
            ),
        )
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_graph_gate_fallen_is_fail(self, tmp_path):
        _write_e2e(
            tmp_path,
            _e2e(
                graph_check={
                    "rc": 1,
                    "cases": 5,
                    "gate_cases": 4,
                    "gefallen": ["two-graphs"],
                    "alle_bestanden": False,
                }
            ),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "two-graphs" in line

    def test_the_bolt_gets_its_own_verdict(self, tmp_path):
        """The all_gather coverage gap is the acceptance scenario of the
        parallel integration. It must be recognisable without opening a log."""
        _write_e2e(
            tmp_path,
            _e2e(
                riegel={
                    "op": "all_gather",
                    "bytes": 10600448,
                    "zeile": LOG_RIEGEL,
                }
            ),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "RIEGEL" in line
        assert "all_gather" in line
        assert "10600448" in line

    def test_only_one_group_on_bar1_is_fail(self, tmp_path):
        """tp over bar1, dcp over gloo: both log lines used to look the same,
        and the number that came out was half a gloo number."""
        _write_e2e(
            tmp_path,
            _e2e(
                gruppen=[
                    {"group": "dcp:0", "requested": "bar1", "achieved": "gloo"},
                    {"group": "tp:0", "requested": "bar1", "achieved": "bar1"},
                ],
                gruppen_bar1=["tp:0"],
                gruppen_ausgewichen=["dcp:0"],
            ),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "dcp:0" in line
        assert "gemischter Lauf" in line

    def test_no_erreicht_line_at_all_is_fail(self, tmp_path):
        _write_e2e(tmp_path, _e2e(gruppen=[], gruppen_bar1=[]))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_no_harvested_log_is_a_stop_not_a_fail(self, tmp_path):
        """"Nobody looked" is not "nothing was there".

        A step directory without a single harvested log line says nothing
        about the run. Reporting it as FAIL would blame the run for a gap in
        the harvest -- the same conflation the s12 fatal gate was split up
        for.
        """
        _write_e2e(
            tmp_path,
            _e2e(gruppen=[], gruppen_bar1=[], log_quellen=[], log_zeilen=0),
        )
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "niemand hat geschaut" in line

    def test_old_artifact_is_rejected_by_its_schema_version(self, tmp_path):
        """The producer that only ever read htccl_lines.txt is not readable.

        Its artifacts are empty on exactly the runs worth diagnosing, and an
        empty artifact from that producer is indistinguishable from a clean
        one. The schema bump makes that loud instead of plausible.
        """
        payload = _e2e()
        payload["schema_version"] = 1
        del payload["log_quellen"]
        _write_e2e(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "schema_version" in line

    def test_missing_log_quellen_at_the_right_schema_is_a_stop(self, tmp_path):
        """Belt and braces: the field is what the check reads, not the number."""
        payload = _e2e()
        del payload["log_quellen"]
        _write_e2e(tmp_path, payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "log_quellen" in line

    def test_a_dead_boot_is_reported_before_its_fallout(self, tmp_path):
        """Cause before consequence.

        A boot that died takes the ERREICHT lines, the setup lines and the
        smoke request with it. Whichever of those the check happens to reach
        first is the consequence; the traceback is the cause.
        """
        _write_e2e(
            tmp_path,
            _e2e(
                gruppen=[],
                gruppen_bar1=[],
                aufbau_lines=0,
                aufbau_gruppen=[],
                fatal="Traceback (most recent call last):",
            ),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Fatal im Serverlog" in line

    def test_missing_group_is_fail(self, tmp_path):
        _write_e2e(
            tmp_path,
            _e2e(
                gruppen=[{"group": "tp:0", "requested": "bar1", "achieved": "bar1"}],
                gruppen_bar1=["tp:0"],
                aufbau_gruppen=["tp:0"],
            ),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "dcp" in line

    def test_no_setup_line_is_fail(self, tmp_path):
        _write_e2e(tmp_path, _e2e(aufbau_lines=0, aufbau_gruppen=[]))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_setup_only_for_one_group_is_fail(self, tmp_path):
        _write_e2e(tmp_path, _e2e(aufbau_gruppen=["tp:0"]))
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "setup" in line

    def test_incoherent_smoke_is_fail(self, tmp_path):
        payload = _e2e()
        payload["smoke"].update(
            {"kohaerent": False, "zahlen_in_folge": 3, "content_prefix": "1 1 1 1"}
        )
        _write_e2e(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "inkohaerent" in line

    def test_under_provisioned_smoke_has_its_own_verdict(self, tmp_path):
        """It must never read as a transport finding.

        This is the 2026-07-30 outcome: 9x ACHIEVED=bar1, no bolt, a full
        boot -- and a smoke that spent its budget on a thinking preamble.
        Reporting that as "incoherent" invites the reading that bar1
        corrupted the output, which the byte proofs rule out.
        """
        payload = _e2e()
        payload["smoke"].update(
            {
                "kohaerent": False,
                "unterprovisioniert": True,
                "zahlen_in_folge": 3,
                "finish_reason": "length",
                "content_prefix": "Here's a thinking process: 1. Analyze ...",
            }
        )
        _write_e2e(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "UNTER-PROVISIONIERT" in line
        assert "NICHT ueber den Transport" in line
        assert "inkohaerent" not in line

    def test_a_chat_endpoint_artifact_is_a_stop(self, tmp_path):
        """Not decidable, so not decided.

        On the chat path the coherence number depends on the template and
        meta_info is opt-in -- a verdict from that artifact would be an
        opinion about the wrong thing.
        """
        payload = _e2e()
        payload["smoke"]["endpunkt"] = "chat"
        _write_e2e(tmp_path, payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "generate" in line

    def test_missing_accept_length_is_fail(self, tmp_path):
        payload = _e2e()
        payload["smoke"]["spec_accept_length"] = None
        _write_e2e(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "spec_accept_length" in line

    def test_fatal_in_log_is_fail(self, tmp_path):
        _write_e2e(tmp_path, _e2e(fatal="CUDA out of memory. Tried to allocate ..."))
        assert_fail(self.CHECK, tmp_path, self.STEP)


class TestE2EParsing:
    """The regexes against the strings the code really writes. A silently
    non-matching pattern would report a clean run for a mixed one."""

    def test_two_groups_on_bar1(self, tmp_path):
        (tmp_path / "htccl_lines.txt").write_text(
            "\n".join(
                [
                    LOG_AUFBAU,
                    LOG_KASSE_TP,
                    LOG_GROUP_OK,
                    LOG_KASSE_DCP,
                    LOG_GROUP_OK_DCP,
                ]
            )
        )
        out = parse_log_evidence(str(tmp_path))
        assert [g["group"] for g in out["gruppen"]] == ["dcp:0", "tp:0"]
        assert all(g["achieved"] == "bar1" for g in out["gruppen"])
        assert out["aufbau_gruppen"] == ["tp:0", "dcp:0"]
        assert out["aufbau_lines"] == 1
        assert out["riegel"] is None

    def test_fallback_is_visible_per_group(self, tmp_path):
        (tmp_path / "htccl_lines.txt").write_text(
            "\n".join([LOG_GROUP_OK, LOG_GROUP_FALLBACK_DCP])
        )
        out = parse_log_evidence(str(tmp_path))
        by_name = {g["group"]: g for g in out["gruppen"]}
        assert by_name["tp:0"]["achieved"] == "bar1"
        assert by_name["dcp:0"]["achieved"] == "gloo"
        assert by_name["dcp:0"]["requested"] == "bar1"

    def test_plain_traceback_is_a_fatal(self, tmp_path):
        """A boot killed by a traceback -- no OOM, no NCCL error -- is just as
        dead. The shell harvests the marker; FATAL_MARKERS used to drop it, so
        exactly that death passed the fatal gate."""
        (tmp_path / "htccl_lines.txt").write_text(
            LOG_GROUP_OK
            + "\n910:Traceback (most recent call last):\n"
            + '911:  File "server.py", line 1, in <module>\n'
        )
        out = parse_log_evidence(str(tmp_path))
        assert out["fatal"] is not None
        assert "Traceback" in out["fatal"]

    def test_clean_log_has_no_fatal(self, tmp_path):
        (tmp_path / "htccl_lines.txt").write_text(LOG_GROUP_OK + "\n")
        assert parse_log_evidence(str(tmp_path))["fatal"] is None

    def test_bolt_is_extracted_with_op_and_size(self, tmp_path):
        (tmp_path / "htccl_lines.txt").write_text(LOG_RIEGEL + "\n")
        out = parse_log_evidence(str(tmp_path))
        assert out["riegel"]["op"] == "all_gather"
        assert out["riegel"]["bytes"] == 10600448

    def test_server_log_alone_carries_the_evidence(self, tmp_path):
        """The file the shell writes on EVERY path, including the aborts.

        htccl_lines.txt is the grep result, harvested only after the server
        answered. On the run this test is built from it never existed, and
        the parser -- which read that file and nothing else -- reported an
        empty evidence list for a log that carried three ERREICHT lines, the
        bolt and the traceback.
        """
        (tmp_path / "server.log").write_text(
            "\n".join(
                [
                    LOG_GROUP_OK.split(":", 1)[1],
                    LOG_KASSE_TP.split(":", 1)[1],
                    LOG_AUFBAU.split(":", 1)[1],
                    LOG_GROUP_OK_DCP.split(":", 1)[1],
                    LOG_KASSE_DCP.split(":", 1)[1],
                ]
            )
            + "\n"
        )
        out = parse_log_evidence(str(tmp_path))
        assert out["log_quellen"] == ["server.log"]
        assert [g["group"] for g in out["gruppen"]] == ["dcp:0", "tp:0"]
        assert out["aufbau_lines"] == 1

    def test_no_log_file_at_all_is_visible_as_such(self, tmp_path):
        out = parse_log_evidence(str(tmp_path))
        assert out["log_quellen"] == []
        assert out["log_zeilen"] == 0
        assert out["gruppen"] == []

    def test_the_two_sources_do_not_double_count(self, tmp_path):
        """grep prefixes the line number, tail does not -- same line, twice."""
        (tmp_path / "htccl_lines.txt").write_text(LOG_AUFBAU + "\n")
        (tmp_path / "server.log").write_text(LOG_AUFBAU.split(":", 1)[1] + "\n")
        out = parse_log_evidence(str(tmp_path))
        assert out["log_quellen"] == ["htccl_lines.txt", "server.log"]
        assert out["aufbau_lines"] == 1


class TestAgainstTheRealS11Log:
    """The check driven against the ACTUAL log of the run that reported
    'no ERREICHT line at all'.

    The point of this class is that the fixture is not written from an idea
    of what the log looks like -- it is a shortened copy of
    gpu-battery-results/2026-07-30_bar1/s11_bar1_e2e/server.log, provenance
    in _fixture_provenance.json next to it. Against that file the check has
    to name the CAUSE (the capture aborted on an uncovered broadcast), not
    the consequence.
    """

    CHECK, STEP = "check_s11_bar1_e2e.py", "s11_bar1_e2e"
    FIXTURE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures", "gpu_battery", "s11_bar1_e2e",
    )

    def _step_dir(self, tmp_path):
        for name in ("server.log", "graph_check.txt", "graph_check_rc.txt"):
            shutil.copy(os.path.join(self.FIXTURE, name), str(tmp_path / name))
        return tmp_path

    def test_the_evidence_is_found_in_the_real_log(self, tmp_path):
        out = parse_log_evidence(str(self._step_dir(tmp_path)))
        assert out["log_quellen"] == ["server.log"]
        assert [g["group"] for g in out["gruppen"]] == [
            "dcp:0", "tp:0", "world:0",
        ]
        assert all(g["achieved"] == "bar1" for g in out["gruppen"])
        assert out["riegel"]["op"] == "broadcast"
        assert out["riegel"]["bytes"] == 128
        assert "Traceback" in out["fatal"]

    def test_the_check_fails_on_the_cause(self, tmp_path):
        step_dir = self._step_dir(tmp_path)
        payload = compose(str(step_dir), 30030, "/root/battery-bar1/s11.server.log")
        write_json(step_dir / "bar1_e2e.json", payload)
        line = assert_fail(self.CHECK, step_dir, self.STEP)
        assert "RIEGEL" in line
        assert "broadcast" in line
        # And explicitly NOT the consequence it used to report.
        assert "ERREICHT" not in line


class TestAgainstTheGreenTransportRun:
    """Attempt 3, on the card: the transport side really carried the run.

    The counterpart to the fixture above. Same day, same step, after the
    broadcast coverage landed: nine ACHIEVED=bar1 lines over world:0, tp:0
    and dcp:0, no bolt, and all SEVEN gate cases passed -- including
    `broadcast` and `broadcast-two-graphs`, which is the on-card proof
    that a broadcast survives capture and replay.

    What this class pins is the verdict SHAPE for that run: every transport
    check has to pass, and the one thing that is wrong has to be reported as
    a fault of the smoke. A run whose collectives are byte-proven must never
    come out sounding like a transport fault.
    """

    CHECK, STEP = "check_s11_bar1_e2e.py", "s11_bar1_e2e"
    FIXTURE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures", "gpu_battery", "s11_bar1_e2e_transport_gruen",
    )

    def _step_dir(self, tmp_path):
        for name in ("server.log", "graph_check.txt", "graph_check_rc.txt",
                     "smoke.json"):
            shutil.copy(os.path.join(self.FIXTURE, name), str(tmp_path / name))
        return tmp_path

    def _compose(self, tmp_path):
        step_dir = self._step_dir(tmp_path)
        payload = compose(str(step_dir), 30030, "/root/battery-bar1/s11.server.log")
        write_json(step_dir / "bar1_e2e.json", payload)
        return step_dir, payload

    def test_the_transport_side_is_clean(self, tmp_path):
        _, payload = self._compose(tmp_path)
        assert payload["riegel"] is None
        assert payload["fatal"] is None
        assert payload["gruppen_bar1"] == ["dcp:0", "tp:0", "world:0"]
        assert payload["gruppen_ausgewichen"] == []
        # Drei Raenge x drei Gruppen (world, tp, dcp) -- jeder Rang baut je
        # Gruppe eine eigene Region auf.
        assert payload["aufbau_lines"] == 9

    def test_all_seven_gate_cases_passed_including_broadcast(self, tmp_path):
        _, payload = self._compose(tmp_path)
        gate = payload["graph_check"]
        assert gate["gate_cases"] == 7
        assert gate["gefallen"] == []
        assert gate["alle_bestanden"] is True

    def test_the_only_complaint_is_about_the_smoke(self, tmp_path):
        step_dir, _ = self._compose(tmp_path)
        line = assert_stop(self.CHECK, step_dir, self.STEP)
        assert "/generate" in line
        # Kein Wort, das nach Transportfehler klingt.
        for verboten in ("RIEGEL", "ERREICHT", "Fatal", "gemischter Lauf"):
            assert verboten not in line, verboten

    def test_the_thinking_flip_is_visible_in_the_artifact(self, tmp_path):
        """Diagnosable without opening a log: the state is a field.

        The verdict says "repeat, do not judge"; the artifact still records
        WHY, so the next reader does not have to rediscover it.
        """
        _, payload = self._compose(tmp_path)
        smoke = payload["smoke"]
        assert smoke["endpunkt"] == "chat"
        assert smoke["finish_reason"] == "length"
        assert smoke["unterprovisioniert"] is True
        assert smoke["spec_accept_length"] is None
        assert "thinking process" in smoke["content_prefix"]

    def test_graph_gate_summary_is_read_case_by_case(self, tmp_path):
        """The exit code alone would hide WHICH case fell, and a gate whose
        cases nobody counted is a gate that could have run zero of them."""
        (tmp_path / "graph_check.txt").write_text(GRAPH_CHECK_OK)
        (tmp_path / "graph_check_rc.txt").write_text("0\n")
        out = parse_graph_check(str(tmp_path))
        assert out["gate_cases"] == 4
        assert out["cases"] == 5
        assert out["gefallen"] == []
        assert out["alle_bestanden"] is True

    def test_a_fallen_gate_case_is_named(self, tmp_path):
        (tmp_path / "graph_check.txt").write_text(GRAPH_CHECK_FALLEN)
        (tmp_path / "graph_check_rc.txt").write_text("1\n")
        out = parse_graph_check(str(tmp_path))
        assert out["gefallen"] == ["two-graphs"]
        assert out["alle_bestanden"] is False

    def test_gate_that_never_ran_reports_zero_cases(self, tmp_path):
        (tmp_path / "graph_check.txt").write_text("ImportError: no torch\n")
        (tmp_path / "graph_check_rc.txt").write_text("1\n")
        out = parse_graph_check(str(tmp_path))
        assert out["gate_cases"] == 0
        assert out["alle_bestanden"] is False

    def test_the_chat_shape_is_still_parseable(self, tmp_path):
        """Alte Artefakte bleiben LESBAR -- sie werden nur nicht bewertet.

        Die Chat-Form endet im Check bei einem STOP. Dass der Parser sie
        trotzdem versteht, ist der Unterschied zwischen "wir bewerten das
        nicht" und "wir koennen es nicht mehr oeffnen".
        """
        write_json(
            tmp_path / "smoke.json",
            {
                "choices": [
                    {
                        "message": {"content": SMOKE_TEXT},
                        "meta_info": {"spec_accept_length": 2.87},
                    }
                ]
            },
        )
        out = parse_smoke(str(tmp_path))
        assert out["endpunkt"] == "chat"
        assert out["zahlen_in_folge"] == 20
        assert out["spec_accept_length"] == 2.87

    def test_error_response_is_reported(self, tmp_path):
        write_json(tmp_path / "smoke.json", {"object": "error", "error": "no model"})
        out = parse_smoke(str(tmp_path))
        assert out["error"]


#: Abwechslungsreiche Prosa fuer die Negativkontrollen. KEIN wiederholter
#: Satz: eine Kontrolle, die aus "Satz " * 30 gebaut ist, faellt IMMER ueber
#: die Tokenschleifen-Pruefung und beweist damit nichts ueber die Bedingung,
#: die sie eigentlich isolieren wollte. Beim ersten Anlauf dieser Tests ist
#: genau das passiert.
_PROSA = " ".join(
    f"Der {w} Gedanke fuehrt uns zu einer weiteren Betrachtung ueber {z}."
    for w, z in zip(
        "erste zweite dritte vierte fuenfte sechste siebte achte neunte "
        "zehnte elfte zwoelfte".split(),
        "Netzwerke Kerne Puffer Register Karten Fenster Ringe Schlitze "
        "Flaggen Runden Sperren Belege".split(),
    )
)


class TestGenerateSmoke:
    """The criterion: is the LM intact -- not did it obey.

    Attempt 4 continued " 5 6 7 8 9 10" correctly at temperature 0 and then
    drifted into a coherent Russian forum post about wiring a three-phase
    motor. That is the characteristic of a raw continuation with no
    instruction, not damage: the old bar ("15 numbers in order") measured
    obedience and failed a healthy model. Real corruption does not produce
    six correct numbers and then well-formed prose.

    So: (a) an anchor of correct numbers IMMEDIATELY after the prompt, and
    (b) the text passes blunt garbage tests. Nothing here judges MEANING --
    an unbidden forum post is a perfectly intact language-model result.
    """

    def _schreibe(self, tmp_path, text, meta=None):
        write_json(
            tmp_path / "smoke.json",
            {"text": text, "meta_info": meta if meta is not None else {}},
        )
        return parse_smoke(str(tmp_path))

    def test_a_clean_continuation_is_intact(self, tmp_path):
        out = self._schreibe(
            tmp_path,
            SMOKE_FORTSETZUNG,
            {"spec_accept_length": 2.87, "spec_verify_ct": 41,
             "finish_reason": {"type": "length"}},
        )
        assert out["endpunkt"] == "generate"
        assert out["anker_zahlen"] >= 16
        assert out["lm_intakt"] is True
        assert out["muell_befunde"] == []
        assert out["spec_accept_length"] == 2.87
        assert out["spec_verify_ct"] == 41
        assert out["finish_reason"] == "length"

    def test_drift_after_a_good_anchor_passes(self, tmp_path):
        """The whole point of the rework, in miniature."""
        out = self._schreibe(
            tmp_path, " 5 6 7 8 9 10 " + _PROSA,
            {"finish_reason": {"type": "length"}},
        )
        assert out["anker_zahlen"] == 6
        assert out["lm_intakt"] is True
        assert out["unterprovisioniert"] is False

    def test_finish_reason_length_alone_is_not_a_failure(self, tmp_path):
        """A continuation prompt has no reason to stop -- `length` is normal."""
        out = self._schreibe(
            tmp_path, SMOKE_FORTSETZUNG, {"finish_reason": {"type": "length"}}
        )
        assert out["lm_intakt"] is True

    def test_the_anchor_must_be_immediate(self, tmp_path):
        """A 5 somewhere in the prose is not a continuation.

        The scattering counter would find 5..20 spread over any long text;
        that is how it reported 10 for attempt 4 and 3 for the thinking
        preamble. The anchor starts at character one.
        """
        out = self._schreibe(tmp_path, _PROSA + " 5 6 7 8 9 10")
        assert out["anker_zahlen"] == 0
        assert out["lm_intakt"] is False

    def test_a_wrong_number_breaks_the_anchor_immediately(self, tmp_path):
        """"0/10" right after the 10 must not pass as an 11.

        Verbatim from attempt 4: the drift begins with "10 0/10". A search
        that skips ahead would score that as a hit.
        """
        out = self._schreibe(tmp_path, " 5 6 7 8 9 10 0/10 " + _PROSA)
        assert out["anker_zahlen"] == 6

    def test_the_prompt_numbers_do_not_count(self, tmp_path):
        """Echoing "1 2 3 4" proves nothing -- the anchor starts at 5."""
        out = self._schreibe(tmp_path, "1 2 3 4 " + _PROSA)
        assert out["anker_zahlen"] == 0
        assert out["lm_intakt"] is False

    # -- Negativkontrollen: jede isoliert EINE Bedingung ------------------

    def test_a_token_loop_is_garbage_even_with_a_good_anchor(self, tmp_path):
        out = self._schreibe(tmp_path, " 5 6 7 8 9 10 " + "ja ja " * 100)
        assert out["anker_zahlen"] == 6
        assert out["lm_intakt"] is False
        assert any("Tokenschleife" in b for b in out["muell_befunde"])

    def test_unprintable_noise_is_garbage(self, tmp_path):
        """Isolated: varied, non-repeating, but not printable text."""
        import random

        rnd = random.Random(7)
        rauschen = "".join(chr(rnd.randrange(1, 32)) for _ in range(400))
        out = self._schreibe(tmp_path, " 5 6 7 8 9 10 " + rauschen)
        assert out["lm_intakt"] is False
        assert any("druckbare" in b for b in out["muell_befunde"])

    def test_low_vocabulary_is_garbage(self, tmp_path):
        """Isolated: printable, no immediate loop, but almost no vocabulary."""
        worte = ["alpha", "beta"] * 60
        # Umgestellt, damit sich keine kurze Einheit UNMITTELBAR wiederholt.
        text = " ".join(worte[::2] + worte[1::2])
        out = self._schreibe(tmp_path, " 5 6 7 8 9 10 " + text)
        assert out["lm_intakt"] is False
        assert any("Wortvielfalt" in b for b in out["muell_befunde"])

    def test_an_empty_answer_is_garbage(self, tmp_path):
        out = self._schreibe(tmp_path, "")
        assert out["lm_intakt"] is False
        assert out["muell_befunde"]

    def test_clean_prose_alone_is_not_garbage_but_has_no_anchor(self, tmp_path):
        """The control that keeps the garbage tests honest.

        Well-formed text with no continuation must fail on the ANCHOR and
        report no garbage -- otherwise the garbage tests are just a second,
        vaguer way of saying the same thing.
        """
        out = self._schreibe(tmp_path, _PROSA)
        assert out["muell_befunde"] == []
        assert out["anker_zahlen"] == 0
        assert out["lm_intakt"] is False

    def test_the_named_under_provisioned_state(self, tmp_path):
        """Coherent text, budget spent, the numbers never started."""
        out = self._schreibe(
            tmp_path, _PROSA, {"finish_reason": {"type": "length"}}
        )
        assert out["lm_intakt"] is False
        assert out["unterprovisioniert"] is True

    def test_garbage_is_never_called_under_provisioned(self, tmp_path):
        """The reassuring name must not cover a real fault."""
        out = self._schreibe(
            tmp_path, "ja ja " * 100, {"finish_reason": {"type": "length"}}
        )
        assert out["unterprovisioniert"] is False

    def test_a_good_anchor_is_never_called_under_provisioned(self, tmp_path):
        """Attempt 4 must not be filed under "budget went elsewhere"."""
        out = self._schreibe(
            tmp_path, " 5 6 7 8 9 10 " + _PROSA,
            {"finish_reason": {"type": "length"}},
        )
        assert out["unterprovisioniert"] is False

    def test_the_accept_length_comes_from_meta_info_only(self, tmp_path):
        """NOT spec_ema_accept_len -- a smoothed curve, not this request's
        acceptance length, and confusing the two is a known trap."""
        out = self._schreibe(
            tmp_path, SMOKE_FORTSETZUNG, {"spec_ema_accept_len": 3.4}
        )
        assert out["spec_accept_length"] is None

    def test_a_response_that_is_neither_shape_is_an_error(self, tmp_path):
        write_json(tmp_path / "smoke.json", {"unerwartet": True})
        out = parse_smoke(str(tmp_path))
        assert out["error"]
        assert "generate" in out["error"]


class TestAgainstTheRealDriftAnswer:
    """The artifact the old bar failed, driven verbatim.

    Not a reconstruction: this is smoke.json of attempt 4, text and
    meta_info byte for byte, provenance next to it. Every garbage threshold
    is calibrated against these numbers, so this test is what stops someone
    from tightening one until a healthy answer goes red again.
    """

    FIXTURE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures", "gpu_battery", "s11_bar1_e2e_generate_drift",
    )

    def _out(self, tmp_path):
        shutil.copy(os.path.join(self.FIXTURE, "smoke.json"),
                    str(tmp_path / "smoke.json"))
        return parse_smoke(str(tmp_path))

    def test_the_real_drift_answer_passes(self, tmp_path):
        out = self._out(tmp_path)
        assert out["endpunkt"] == "generate"
        assert out["anker_zahlen"] == 6
        assert out["muell_befunde"] == []
        assert out["lm_intakt"] is True
        assert out["unterprovisioniert"] is False

    def test_the_old_scattering_count_would_have_failed_it(self, tmp_path):
        """Kept as a number, so the reason stays visible in the artifact.

        10 of 16 -- six real, four digits out of the Russian prose. That is
        why the scattering count is a metric here and not the criterion.
        """
        out = self._out(tmp_path)
        assert out["zahlen_in_folge"] == 10
        assert out["zahlen_in_folge"] < 15

    def test_the_measured_numbers_match_the_thresholds(self, tmp_path):
        """The calibration, spelled out where a tightening would break it."""
        out = self._out(tmp_path)
        assert out["druckbar_anteil"] == 1.0
        assert out["wort_vielfalt"] > 0.4
        assert out["max_wiederholung"] == 3

    def test_the_spec_path_was_alive(self, tmp_path):
        out = self._out(tmp_path)
        assert out["spec_accept_length"] > 3.0
        assert out["spec_verify_ct"] == 165


class TestSmokeContractWithTheStepScript:
    """The parser's expected sequence must continue the prompt the shell sends.

    Both live in different languages and different files; a test is the only
    thing that can hold them together. Without it the parser could count a
    continuation nobody asked for -- and it would look green for whatever
    the model happened to say.
    """

    SHELL = os.path.join(BATTERY, "s11_bar1_e2e.sh")

    def _befehle(self) -> str:
        """Die Schrittdatei OHNE Kommentare.

        Der Kommentar ueber dem Request nennt /v1/chat/completions -- als
        Begruendung, warum es das nicht mehr ist. Ein Scan, der das nicht
        auseinanderhalten kann, faellt ueber seine eigene Erklaerung.
        """
        zeilen = [
            z for z in open(self.SHELL, encoding="utf-8").read().splitlines()
            if not z.lstrip().startswith("#")
        ]
        return "\n".join(zeilen)

    def test_the_shell_sends_the_prompt_the_parser_expects(self):
        from s11_bar1_e2e import SMOKE_PROMPT, ZAHLEN_VON

        text = self._befehle()
        assert f'\\"text\\": \\"{SMOKE_PROMPT}\\"' in text, (
            "der Fortsetzungs-Prompt in s11_bar1_e2e.sh und SMOKE_PROMPT "
            "sind auseinandergelaufen"
        )
        # Und die Zaehlung setzt genau dahinter an.
        letzte = int(SMOKE_PROMPT.split()[-1])
        assert ZAHLEN_VON == letzte + 1

    def test_the_shell_uses_generate_and_not_the_chat_endpoint(self):
        text = self._befehle()
        assert "/generate" in text
        assert "/v1/chat/completions" not in text

    def test_the_shell_gives_the_answer_room(self):
        """128 Token waren der Grund, warum die Zahlen nie drankamen."""
        import re

        text = self._befehle()
        treffer = re.search(r'max_new_tokens\\":\s*(\d+)', text)
        assert treffer, "kein max_new_tokens im Smoke-Request"
        assert int(treffer.group(1)) >= 512


# ---------------------------------------------------------------------------
# s12 prefill curve
# ---------------------------------------------------------------------------

BEKANNT = {1: 1190.7, 4: 1143.7, 8: 1105.0, 16: 1122.4}


def _kurve(**over) -> dict:
    plan = [1, 4, 8, 16]
    reihenfolge = []
    folge = 0
    for sessions in plan:
        for arm in ("bar1", "grundlinie"):
            folge += 1
            gruppen = (
                [
                    {"group": "tp:0", "requested": "bar1", "achieved": "bar1"},
                    {"group": "dcp:0", "requested": "bar1", "achieved": "bar1"},
                ]
                if arm == "bar1"
                else []
            )
            reihenfolge.append(
                {
                    "folge": folge,
                    "arm": arm,
                    "sessions": sessions,
                    "zeit": "2026-07-30T04:00:00",
                    "beleg_vorhanden": True,
                    "gruppen": gruppen,
                    "fatal_erhoben": True,
                    "fatal": None,
                }
            )
    payload = {
        "kind": "bar1_prefill_kurve",
        "schema_version": 2,
        "arme": ["bar1", "grundlinie"],
        "sessions_geplant": plan,
        "abbruch": None,
        "host_erreichbar": True,
        "integration_vorhanden": True,
        "punkte": 8,
        "reihenfolge": reihenfolge,
        "fatal": [],
        "fatal_ungeprueft": [],
        "kurven": {
            "bar1": {"1": 1310.0, "4": 1520.0, "8": 1690.0, "16": 1740.0},
            "grundlinie": {str(k): v for k, v in BEKANNT.items()},
        },
        "decode": [
            {
                "arm": arm,
                "batch": batch,
                "decode_tok_s": 88.0 if batch == 1 else 454.0,
                "ms_pro_token": 11.4,
                "spec_accept_length": 2.9,
                "output_sample": "Ein Rechner bearbeitet mehrere Aufgaben, indem er "
                "sie in kleine Abschnitte zerlegt und diese "
                "abwechselnd ausfuehrt.",
            }
            for arm in ("bar1", "grundlinie")
            for batch in (1, 16)
        ],
        "grundlinie_bekannt": {str(k): v for k, v in BEKANNT.items()},
        "grundlinie_quelle": "MESSUNG_PREFILL_ANTEIL.md",
        "grundlinie_abweichung_pct": {str(k): 0.0 for k in BEKANNT},
        "toleranz_pct": 5.0,
        "verhaeltnis_bar1_zu_grundlinie": {"1": 1.10, "4": 1.33, "8": 1.53, "16": 1.55},
        "output_samples": {
            "bar1": "Ein Rechner bearbeitet mehrere Aufgaben, indem er sie zerlegt "
            "und abwechselnd ausfuehrt.",
            "grundlinie": "Ein Rechner bearbeitet mehrere Aufgaben, indem er sie "
            "zerlegt und abwechselnd ausfuehrt.",
        },
    }
    payload.update(over)
    return payload


def _write_kurve(tmp_path, payload=None):
    write_json(
        tmp_path / "prefill_kurve.json", payload if payload is not None else _kurve()
    )
    return tmp_path


class TestPrefillKurveCheck:
    CHECK, STEP = "check_s12_prefill_kurve.py", "s12_prefill_kurve"

    def test_pass_with_a_rising_curve(self, tmp_path):
        _write_kurve(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_pass_with_a_flat_curve_too(self, tmp_path):
        """Flat is a result. A check that only passed a rising curve would be
        deciding the question the step exists to ask."""
        payload = _kurve()
        payload["kurven"]["bar1"] = {
            "1": 1195.0,
            "4": 1150.0,
            "8": 1110.0,
            "16": 1125.0,
        }
        payload["verhaeltnis_bar1_zu_grundlinie"] = {
            "1": 1.004,
            "4": 1.006,
            "8": 1.005,
            "16": 1.002,
        }
        _write_kurve(tmp_path, payload)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_unreachable_host_is_stop(self, tmp_path):
        _write_kurve(tmp_path, _kurve(host_erreichbar=False))
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_blocked_step_is_stop(self, tmp_path):
        payload = _kurve(blockiert="Host-Locks fremd gehalten -- nicht gebrochen")
        payload["kurven"] = {"bar1": {}, "grundlinie": {}}
        payload["reihenfolge"] = []
        _write_kurve(tmp_path, payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "blockiert" in line

    def test_half_a_pair_is_fail(self, tmp_path):
        payload = _kurve()
        del payload["kurven"]["grundlinie"]["8"]
        payload["reihenfolge"] = [
            r
            for r in payload["reihenfolge"]
            if not (r["arm"] == "grundlinie" and r["sessions"] == 8)
        ]
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "grundlinie" in line

    def test_aborted_run_names_the_reason(self, tmp_path):
        payload = _kurve(abbruch="Messung bar1/8 rc=1")
        payload["kurven"]["bar1"].pop("8")
        payload["kurven"]["bar1"].pop("16")
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "rc=1" in line

    def test_blockwise_measurement_is_fail(self, tmp_path):
        """A,A,A,A,B,B,B,B is two afternoons in one window, not a comparison."""
        payload = _kurve()
        payload["reihenfolge"] = sorted(
            payload["reihenfolge"], key=lambda r: (r["arm"], r["sessions"])
        )
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "blockweise" in line

    def test_bar1_point_with_a_gloo_group_is_fail(self, tmp_path):
        payload = _kurve()
        payload["reihenfolge"][0]["gruppen"][1]["achieved"] = "gloo"
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "mixed point" in line

    def test_baseline_point_with_htccl_is_fail(self, tmp_path):
        """The baseline arm differs in exactly three variables; if it sees
        HTCCL at all, the two arms are not the two arms."""
        payload = _kurve()
        payload["reihenfolge"][1]["gruppen"] = [
            {"group": "tp:0", "requested": "bar1", "achieved": "bar1"}
        ]
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Grundlinie" in line

    def test_missing_transport_evidence_is_fail(self, tmp_path):
        payload = _kurve()
        payload["reihenfolge"][0]["beleg_vorhanden"] = False
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Beleg" in line

    def test_boot_with_a_fatal_is_fail(self, tmp_path):
        """Eight boots that each died in a prefill OOM still hand in a
        throughput table, and without this gate it looks like a healthy one."""
        payload = _kurve()
        payload["fatal"] = [
            {
                "folge": 3,
                "arm": "bar1",
                "sessions": 4,
                "zeile": "logs/bar1_4.fatal.txt:1: torch.OutOfMemoryError: CUDA "
                "out of memory",
            }
        ]
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "OutOfMemoryError" in line

    def test_boot_without_a_fatal_harvest_is_fail(self, tmp_path):
        payload = _kurve()
        payload["fatal_ungeprueft"] = [{"folge": 2, "arm": "grundlinie", "sessions": 1}]
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "ohne Fatal-Ernte" in line

    def test_artifact_without_a_fatal_field_is_fail(self, tmp_path):
        """A producer that never harvested must not read like a clean run."""
        payload = _kurve()
        del payload["fatal"]
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "fatal" in line

    def test_missing_decode_point_is_fail(self, tmp_path):
        payload = _kurve()
        payload["decode"] = [d for d in payload["decode"] if d["batch"] != 16]
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "bs=16" in line

    def test_missing_output_sample_is_fail(self, tmp_path):
        payload = _kurve()
        payload["output_samples"]["bar1"] = None
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Sample" in line

    def test_baseline_that_does_not_reproduce_is_stop(self, tmp_path):
        payload = _kurve()
        payload["kurven"]["grundlinie"]["8"] = 900.0
        payload["grundlinie_abweichung_pct"]["8"] = -18.6
        _write_kurve(tmp_path, payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "reproduziert nicht" in line

    def test_small_baseline_drift_still_passes(self, tmp_path):
        payload = _kurve()
        payload["kurven"]["grundlinie"]["8"] = 1060.0
        payload["grundlinie_abweichung_pct"]["8"] = -4.1
        _write_kurve(tmp_path, payload)
        assert_pass(self.CHECK, tmp_path, self.STEP)


class TestPrefillKurveSummary:
    """punkte.jsonl -> summary -> live table, without a server."""

    @staticmethod
    def _punkte(tmp_path, punkte, belege=True, fatal=""):
        with open(tmp_path / "punkte.jsonl", "w") as f:
            for p in punkte:
                f.write(json.dumps(p) + "\n")
        if belege:
            os.makedirs(tmp_path / "belege", exist_ok=True)
            os.makedirs(tmp_path / "logs", exist_ok=True)
            for p in punkte:
                name = f"{p['folge']}_{p['arm']}_{p['sessions']}.txt"
                text = (
                    "\n".join([LOG_GROUP_OK, LOG_GROUP_OK_DCP])
                    if p["arm"] == "bar1"
                    else ""
                )
                (tmp_path / "belege" / name).write_text(text)
                # The shell's grep leaves an EMPTY file behind when it found
                # nothing -- that is the healthy case, not a missing harvest.
                (
                    tmp_path / "logs" / f"{p['arm']}_{p['sessions']}.fatal.txt"
                ).write_text(fatal)
        return tmp_path

    @staticmethod
    def _punkt(folge, arm, sessions, rate):
        return {
            "folge": folge,
            "arm": arm,
            "sessions": sessions,
            "zeit": "2026-07-30T04:00:00",
            "prefill": {"prefill_tok_s": rate, "requests": 9},
            "decode": [
                {
                    "batch": 1,
                    "decode_tok_s": 90.0,
                    "ms_pro_token": 11.1,
                    "spec_accept_length": 2.9,
                    "output_sample": "x" * 80,
                },
                {
                    "batch": 16,
                    "decode_tok_s": 460.0,
                    "ms_pro_token": 2.2,
                    "spec_accept_length": 2.8,
                    "output_sample": "x" * 80,
                },
            ],
        }

    def test_summary_joins_arms_and_reads_the_evidence(self, tmp_path):
        self._punkte(
            tmp_path,
            [
                self._punkt(1, "bar1", 1, 1310.0),
                self._punkt(2, "grundlinie", 1, 1190.0),
            ],
        )
        payload = zusammenfassen(str(tmp_path), 5.0, [1])
        assert payload["kurven"]["bar1"]["1"] == 1310.0
        assert payload["verhaeltnis_bar1_zu_grundlinie"]["1"] == pytest.approx(
            1310.0 / 1190.0
        )
        assert abs(payload["grundlinie_abweichung_pct"]["1"]) < 0.1
        assert payload["reihenfolge"][0]["gruppen"][0]["achieved"] == "bar1"
        assert payload["reihenfolge"][1]["gruppen"] == []
        assert payload["fatal"] == []
        assert payload["fatal_ungeprueft"] == []

    def test_summary_surfaces_a_boot_that_died(self, tmp_path):
        """The harvest existed on disk from the start; nothing read it, so
        eight boots that each died in a prefill OOM still handed in numbers."""
        self._punkte(
            tmp_path,
            [
                self._punkt(1, "bar1", 1, 1310.0),
                self._punkt(2, "grundlinie", 1, 1190.0),
            ],
            fatal="884:torch.OutOfMemoryError: CUDA out of memory. Tried 2.00 GiB\n",
        )
        payload = zusammenfassen(str(tmp_path), 5.0, [1])
        assert len(payload["fatal"]) == 2
        assert "OutOfMemoryError" in payload["fatal"][0]["zeile"]
        assert payload["fatal_ungeprueft"] == []

    def test_summary_marks_a_boot_without_a_harvest(self, tmp_path):
        """Nobody looked and nothing found must not read the same."""
        self._punkte(tmp_path, [self._punkt(1, "bar1", 1, 1310.0)], belege=False)
        payload = zusammenfassen(str(tmp_path), 5.0, [1])
        assert payload["fatal"] == []
        assert len(payload["fatal_ungeprueft"]) == 1

    def test_table_renders_both_arms_without_judging(self, tmp_path):
        self._punkte(
            tmp_path,
            [
                self._punkt(1, "bar1", 1, 1310.0),
                self._punkt(2, "grundlinie", 1, 1190.0),
            ],
        )
        payload = zusammenfassen(str(tmp_path), 5.0, [1])
        text = tabelle(payload)
        assert "| 1 | 1310.0 | 1190.0 | 1.101 |" in text
        for verdict in ("gut", "schlecht", "besser", "Gewinn", "!"):
            assert verdict not in text

    def test_incomplete_run_summarises_what_is_there(self, tmp_path):
        """The live table has to work after the FIRST point, or nobody can
        watch a run that takes over an hour."""
        self._punkte(tmp_path, [self._punkt(1, "bar1", 1, 1310.0)])
        payload = zusammenfassen(str(tmp_path), 5.0, [1, 4])
        assert payload["kurven"]["grundlinie"] == {}
        assert "| 1 | 1310.0 | None |" in tabelle(payload)


# ---------------------------------------------------------------------------
# the verdict contract, for the three new checks
# ---------------------------------------------------------------------------


class TestBar1VerdictContract:
    CHECKS = (
        ("check_s10_bar1_driver.py", "s10_bar1_driver"),
        ("check_s11_bar1_e2e.py", "s11_bar1_e2e"),
        ("check_s12_prefill_kurve.py", "s12_prefill_kurve"),
    )

    @pytest.mark.parametrize("check,step", CHECKS)
    def test_empty_dir_never_passes(self, check, step, tmp_path):
        rc, line = run_check(check, tmp_path)
        assert rc in (1, 2), line
        assert line.startswith(("BATTERY-FAIL ", "BATTERY-STOP ")), line

    @pytest.mark.parametrize("check,step", CHECKS)
    def test_garbage_artifact_is_a_verdict_not_a_traceback(self, check, step, tmp_path):
        for name in ("driver_state.json", "bar1_e2e.json", "prefill_kurve.json"):
            (tmp_path / name).write_text("{kein json")
        rc, line = run_check(check, tmp_path)
        assert rc in (1, 2), line
        assert line.startswith(("BATTERY-FAIL ", "BATTERY-STOP ")), line

    @pytest.mark.parametrize("check,step", CHECKS)
    def test_wrong_kind_is_a_verdict(self, check, step, tmp_path):
        for name in ("driver_state.json", "bar1_e2e.json", "prefill_kurve.json"):
            write_json(tmp_path / name, {"kind": "etwas_anderes", "schema_version": 1})
        rc, line = run_check(check, tmp_path)
        assert rc in (1, 2), line
        assert line.startswith(("BATTERY-FAIL ", "BATTERY-STOP ")), line
