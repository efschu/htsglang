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
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
CHECKS = os.path.join(BATTERY, "checks")

sys.path.insert(0, BATTERY)

from battery_steps import STEPS_BY_ID  # noqa: E402
from s10_bar1_driver import compose as s10_compose  # noqa: E402
from s10_bar1_driver import desired_state_reached  # noqa: E402
from s11_bar1_e2e import (  # noqa: E402
    parse_graph_check,
    parse_log_evidence,
    parse_smoke,
)
from s12_prefill_kurve import tabelle, zusammenfassen  # noqa: E402

# The lines the code really writes. Copied from
# parallel_state.py:638 / htccl.py:645 and htccl_bar1.py:1518 / :1530, with the
# "<lineno>:" prefix that grep -n puts in front of them.
LOG_GROUP_OK = (
    "412:[2026-07-30 03:11:02] HTCCL enabled for group 'tp:0': angefordert=bar1, "
    "ERREICHT=bar1. Every SGLANG_HTCCL* env must be identical on all ranks; the "
    "host-staged transports (shm/gloo/ucx) additionally require --disable-cuda-graph."
)
LOG_GROUP_OK_DCP = (
    "418:[2026-07-30 03:11:02] HTCCL enabled for group 'dcp:0': angefordert=bar1, "
    "ERREICHT=bar1. Every SGLANG_HTCCL* env must be identical on all ranks."
)
LOG_GROUP_FALLBACK_DCP = (
    "418:[2026-07-30 03:11:02] HTCCL group 'dcp:0': angefordert=bar1, ERREICHT=gloo "
    "(aufbau: Bar1Unverfuegbar: Halter meldet ENOMEM). Diese Gruppe laeuft NICHT "
    "ueber bar1. Ein Messwert aus diesem Lauf ist gemischt und darf nicht als "
    "bar1-Wert berichtet werden."
)
LOG_AUFBAU = (
    "400:[2026-07-30 03:11:01] HTCCL-BAR1: Aufbau in 27 ms, 2 Peer-Ziele, Region "
    "24.0 MiB je Rang (8 Schlitze), Schlitz 512 KiB, groesste Nutzlast 20480 KiB, "
    "Flaggen 256 Byte, Export ueber dma-buf. Ab hier wird im heissen Pfad nichts "
    "mehr gemappt."
)
LOG_KASSE_TP = (
    "401:[2026-07-30 03:11:01] HTCCL-BAR1: BAR1-Kasse dieser Karte nach Gruppe "
    "'tp:0': tp:0: 24.0 MiB."
)
LOG_KASSE_DCP = (
    "409:[2026-07-30 03:11:01] HTCCL-BAR1: BAR1-Kasse dieser Karte nach Gruppe "
    "'dcp:0': tp:0: 24.0 MiB, dcp:0: 24.0 MiB."
)
LOG_RIEGEL = (
    "902:RuntimeError: HTCCL: 'all_gather' mit 10600448 Byte waehrend einer "
    "CUDA-Graph-Aufzeichnung, aber bar1 meldet handles('all_gather', 10600448) -> "
    "False. Der Ausweichweg ist die host-gestaffelte gloo-Ebene."
)

GRAPH_CHECK_OK = """BAR1-Graph-Beleg: Geraete [0, 1, 2], 3 Raenge
--- Fall 'einfach' ----------------------------------------
    => BESTANDEN
==============================================================
Zusammenfassung
==============================================================
  BESTANDEN  [Gate]  einfach
  BESTANDEN  [Gate]  zwei-graphen
  BESTANDEN  [Gate]  wechselnde-form
  BESTANDEN  [Gate]  vorbehalt
  BESTANDEN  [Info]  gitter

Alle Gate-Faelle bestanden.
"""
GRAPH_CHECK_FALLEN = GRAPH_CHECK_OK.replace(
    "  BESTANDEN  [Gate]  zwei-graphen", "  GEFALLEN   [Gate]  zwei-graphen"
).replace("Alle Gate-Faelle bestanden.", "Gefallene Gate-Faelle: zwei-graphen.")

SMOKE_TEXT = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20."


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
        "schema_version": 1,
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
            {"gruppe": "dcp:0", "angefordert": "bar1", "erreicht": "bar1"},
            {"gruppe": "tp:0", "angefordert": "bar1", "erreicht": "bar1"},
        ],
        "gruppen_bar1": ["dcp:0", "tp:0"],
        "gruppen_ausgewichen": [],
        "aufbau_gruppen": ["tp:0", "dcp:0"],
        "aufbau_lines": 6,
        "aufbau_ms": [27.0, 31.0],
        "riegel": None,
        "fatal": None,
        "smoke": {
            "vorhanden": True,
            "content_prefix": SMOKE_TEXT,
            "spec_accept_length": 2.9,
            "zahlen_in_folge": 20,
            "kohaerent": True,
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
                    "gefallen": ["zwei-graphen"],
                    "alle_bestanden": False,
                }
            ),
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "zwei-graphen" in line

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
                    {"gruppe": "dcp:0", "angefordert": "bar1", "erreicht": "gloo"},
                    {"gruppe": "tp:0", "angefordert": "bar1", "erreicht": "bar1"},
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

    def test_missing_group_is_fail(self, tmp_path):
        _write_e2e(
            tmp_path,
            _e2e(
                gruppen=[{"gruppe": "tp:0", "angefordert": "bar1", "erreicht": "bar1"}],
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
        assert "Aufbau" in line

    def test_incoherent_smoke_is_fail(self, tmp_path):
        payload = _e2e()
        payload["smoke"].update(
            {"kohaerent": False, "zahlen_in_folge": 3, "content_prefix": "1 1 1 1"}
        )
        _write_e2e(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "inkohaerent" in line

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
        assert [g["gruppe"] for g in out["gruppen"]] == ["dcp:0", "tp:0"]
        assert all(g["erreicht"] == "bar1" for g in out["gruppen"])
        assert out["aufbau_gruppen"] == ["tp:0", "dcp:0"]
        assert out["aufbau_lines"] == 1
        assert out["riegel"] is None

    def test_fallback_is_visible_per_group(self, tmp_path):
        (tmp_path / "htccl_lines.txt").write_text(
            "\n".join([LOG_GROUP_OK, LOG_GROUP_FALLBACK_DCP])
        )
        out = parse_log_evidence(str(tmp_path))
        by_name = {g["gruppe"]: g for g in out["gruppen"]}
        assert by_name["tp:0"]["erreicht"] == "bar1"
        assert by_name["dcp:0"]["erreicht"] == "gloo"
        assert by_name["dcp:0"]["angefordert"] == "bar1"

    def test_bolt_is_extracted_with_op_and_size(self, tmp_path):
        (tmp_path / "htccl_lines.txt").write_text(LOG_RIEGEL + "\n")
        out = parse_log_evidence(str(tmp_path))
        assert out["riegel"]["op"] == "all_gather"
        assert out["riegel"]["bytes"] == 10600448

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
        assert out["gefallen"] == ["zwei-graphen"]
        assert out["alle_bestanden"] is False

    def test_gate_that_never_ran_reports_zero_cases(self, tmp_path):
        (tmp_path / "graph_check.txt").write_text("ImportError: no torch\n")
        (tmp_path / "graph_check_rc.txt").write_text("1\n")
        out = parse_graph_check(str(tmp_path))
        assert out["gate_cases"] == 0
        assert out["alle_bestanden"] is False

    def test_smoke_coherence_is_counted_not_judged(self, tmp_path):
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
        assert out["zahlen_in_folge"] == 20
        assert out["kohaerent"] is True
        assert out["spec_accept_length"] == 2.87

    def test_smoke_garbage_is_not_coherent(self, tmp_path):
        write_json(
            tmp_path / "smoke.json",
            {"choices": [{"message": {"content": "!!! !!! !!!"}, "meta_info": {}}]},
        )
        out = parse_smoke(str(tmp_path))
        assert out["kohaerent"] is False
        assert out["spec_accept_length"] is None

    def test_error_response_is_reported(self, tmp_path):
        write_json(tmp_path / "smoke.json", {"object": "error", "error": "no model"})
        out = parse_smoke(str(tmp_path))
        assert out["error"]


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
                    {"gruppe": "tp:0", "angefordert": "bar1", "erreicht": "bar1"},
                    {"gruppe": "dcp:0", "angefordert": "bar1", "erreicht": "bar1"},
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
                }
            )
    payload = {
        "kind": "bar1_prefill_kurve",
        "schema_version": 1,
        "arme": ["bar1", "grundlinie"],
        "sessions_geplant": plan,
        "abbruch": None,
        "host_erreichbar": True,
        "integration_vorhanden": True,
        "punkte": 8,
        "reihenfolge": reihenfolge,
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
        payload["reihenfolge"][0]["gruppen"][1]["erreicht"] = "gloo"
        _write_kurve(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "gemischter Punkt" in line

    def test_baseline_point_with_htccl_is_fail(self, tmp_path):
        """The baseline arm differs in exactly three variables; if it sees
        HTCCL at all, the two arms are not the two arms."""
        payload = _kurve()
        payload["reihenfolge"][1]["gruppen"] = [
            {"gruppe": "tp:0", "angefordert": "bar1", "erreicht": "bar1"}
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
    def _punkte(tmp_path, punkte, belege=True):
        with open(tmp_path / "punkte.jsonl", "w") as f:
            for p in punkte:
                f.write(json.dumps(p) + "\n")
        if belege:
            os.makedirs(tmp_path / "belege", exist_ok=True)
            for p in punkte:
                name = f"{p['folge']}_{p['arm']}_{p['sessions']}.txt"
                text = (
                    "\n".join([LOG_GROUP_OK, LOG_GROUP_OK_DCP])
                    if p["arm"] == "bar1"
                    else ""
                )
                (tmp_path / "belege" / name).write_text(text)
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
        assert payload["reihenfolge"][0]["gruppen"][0]["erreicht"] == "bar1"
        assert payload["reihenfolge"][1]["gruppen"] == []

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
