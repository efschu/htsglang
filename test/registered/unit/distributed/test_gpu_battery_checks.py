# SPDX-License-Identifier: Apache-2.0
"""Dry tests for the GPU battery (scripts/gpu_battery/).

The executor is a cheap model that does not read artifacts and does not form
opinions. It reads ONE line from a check script and acts on it. That makes the
check scripts the only thing standing between a broken run and a green report
-- so they are what gets tested here, against synthetic PASS / FAIL / STOP
fixtures.

Every check is driven as a SUBPROCESS, not as an imported function: the
contract the executor depends on is "exactly one line on stdout, and an exit
code of 0/1/2", and only a subprocess tests that contract rather than the
logic behind it.

Hermetic and CPU-only: no card, no server, no network, no /tmp lock is ever
taken. Fixtures are built in tmp_path so they cannot drift away from the
schemas they imitate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
CHECKS = os.path.join(BATTERY, "checks")
# Shortened, anonymised copies of real run artifacts -- see _copy_real_run.
FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "gpu_battery"
)

sys.path.insert(0, BATTERY)

from battery_state import plan, select_ids  # noqa: E402
from battery_steps import STEP_ORDER, STEPS, STEPS_BY_ID, resolve_ids  # noqa: E402

MIB = 1024 * 1024
PCI_A = "00000000:01:00.0"
PCI_B = "00000000:02:00.0"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def run_check(check: str, step_dir) -> tuple[int, str]:
    """Run a check exactly the way run_step.sh does."""
    proc = subprocess.run(
        [sys.executable, os.path.join(CHECKS, check), "--step-dir", str(step_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, (
        f"{check} gab {len(lines)} Zeilen aus, der Executor liest genau eine: {lines}"
    )
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
# the step table itself
# ---------------------------------------------------------------------------


class TestStepTable:
    def test_every_step_has_its_scripts_on_disk(self):
        """A step described in the table but missing on disk is a step the
        executor would STOP on halfway through the battery."""
        for step in STEPS:
            assert os.path.exists(os.path.join(BATTERY, step.script)), step.step_id
            assert os.path.exists(os.path.join(CHECKS, step.check)), step.step_id

    def test_dependencies_point_backwards(self):
        for step in STEPS:
            for dep in step.deps:
                assert dep in STEPS_BY_ID, (step.step_id, dep)
                assert STEP_ORDER.index(dep) < STEP_ORDER.index(step.step_id), (
                    f"{step.step_id} haengt an {dep}, das spaeter kommt"
                )

    def test_timeout_exceeds_expectation(self):
        """A budget at the expected duration turns every slow load into a
        false STOP."""
        for step in STEPS:
            assert step.timeout_s > step.expected_min * 60, step.step_id

    def test_boots_are_not_retryable(self):
        """A boot window comes out of a fixed budget; a failed boot is a
        result, not an accident."""
        for step in STEPS:
            if "boot" in step.step_id:
                assert not step.retryable, step.step_id

    def test_exactly_one_report_gate_and_it_is_boot_a(self):
        gates = [s.step_id for s in STEPS if s.report_gate]
        assert gates == ["s02_boot_a"], gates

    def test_lock_modes_are_known(self):
        for step in STEPS:
            assert step.locks in ("battery", "tool", "none"), step.step_id
            if not step.needs_cards:
                assert step.locks == "none", step.step_id

    def test_p2p_step_does_not_take_locks_itself(self):
        """run_all.sh acquires them; holding them here would make it abort on
        its own acquisition."""
        assert STEPS_BY_ID["s01_p2p_reprobe"].locks == "tool"

    def test_prefix_resolution(self):
        assert resolve_ids("s02") == ["s02_boot_a"]
        assert resolve_ids("s01,s06") == ["s01_p2p_reprobe", "s06_nccl_reference"]
        with pytest.raises(KeyError):
            resolve_ids("s99")


# ---------------------------------------------------------------------------
# resume / selection
# ---------------------------------------------------------------------------


class TestResumeAndSelection:
    @staticmethod
    def _state(**verdicts):
        return {
            "kind": "gpu_battery_state",
            "steps": {
                k: {"verdict": v, "finished": "2026-07-29T10:00:00"}
                for k, v in verdicts.items()
            },
        }

    def test_green_steps_are_skipped_by_default(self):
        state = self._state(s00_preflight="PASS")
        rows = plan(state, ["s00_preflight", "s01_p2p_reprobe"])
        assert rows[0][1] == "SKIP"
        assert rows[1][1] == "RUN"

    def test_force_reruns_a_green_step(self):
        state = self._state(s00_preflight="PASS")
        rows = plan(state, ["s00_preflight"], forced=["s00_preflight"])
        assert rows[0][1] == "RUN"
        assert "trotz PASS" in rows[0][2]

    def test_rerun_all_ignores_green(self):
        state = self._state(s00_preflight="PASS", s01_p2p_reprobe="PASS")
        rows = plan(state, ["s00_preflight", "s01_p2p_reprobe"], rerun_all=True)
        assert [r[1] for r in rows] == ["RUN", "RUN"]

    def test_failed_step_runs_again_without_force(self):
        state = self._state(s00_preflight="PASS", s01_p2p_reprobe="FAIL")
        rows = plan(state, ["s01_p2p_reprobe"])
        assert rows[0][1] == "RUN"

    def test_green_step_is_skipped_even_when_its_deps_are_not_green(self):
        """An already-green step has its artifacts on disk and is not going to
        run. Gating it on today's dependency state would report a finished step
        as BLOCKED and stall the resume on work that is done."""
        state = self._state(s02_boot_a="PASS")
        rows = plan(state, ["s02_boot_a"])
        assert rows[0][1] == "SKIP", rows

    def test_missing_dependency_blocks_rather_than_skips(self):
        """s08 consumes s01 and s06. Quietly passing over it would leave a hole
        that looks like a completed battery."""
        state = self._state(s00_preflight="PASS", s01_p2p_reprobe="PASS")
        rows = plan(state, ["s08_dispatcher_tables"])
        assert rows[0][1] == "BLOCKED"
        assert "s06_nccl_reference" in rows[0][2]

    def test_dependency_satisfied_within_the_same_plan(self):
        state = self._state(s00_preflight="PASS", s01_p2p_reprobe="PASS")
        rows = plan(state, ["s06_nccl_reference", "s08_dispatcher_tables"])
        assert [r[1] for r in rows] == ["RUN", "RUN"]

    def test_s08_is_resumable_alone_without_the_boots(self):
        """The whole point of artifact dependencies: no boot has to be
        repeated to re-run the CPU step weeks later."""
        state = self._state(s01_p2p_reprobe="PASS", s06_nccl_reference="PASS")
        rows = plan(state, ["s08_dispatcher_tables"])
        assert rows[0][1] == "RUN"

    def test_selection_operators(self):
        assert select_ids(only="s01,s06") == ["s01_p2p_reprobe", "s06_nccl_reference"]
        assert select_ids(from_id="s08") == [
            "s08_dispatcher_tables",
            "s09_sensor_smoke",
            "s10_bar1_driver",
            "s11_bar1_e2e",
            "s12_prefill_kurve",
        ]
        assert select_ids(to_id="s01") == ["s00_preflight", "s01_p2p_reprobe"]
        assert "s03_boot_b" not in select_ids(skip="s03,s04")


# ---------------------------------------------------------------------------
# s00 preflight
# ---------------------------------------------------------------------------


def _preflight(**over):
    payload = {
        "kind": "gpu_battery_preflight",
        "schema_version": 1,
        "min_free_mib": 400,
        "cards": [
            {
                "nvml_index": 0,
                "cuda_index": 0,
                "name": "RTX 5090",
                "uuid": "GPU-aaa",
                "pci_bus_id": PCI_A,
                "vram_total_mib": 32768,
                "vram_used_mib": 100,
                "vram_free_mib": 32668,
            },
            {
                "nvml_index": 1,
                "cuda_index": 1,
                "name": "RTX 3080",
                "uuid": "GPU-bbb",
                "pci_bus_id": PCI_B,
                "vram_total_mib": 20480,
                "vram_used_mib": 80,
                "vram_free_mib": 20400,
            },
        ],
        "inventory_errors": [],
        "driver": "580.00",
        "torch": "2.9.0",
        "nccl": "2.28.9",
        "locks_held": [],
        "required_files": {"/x": True},
        "tools": {"nvidia-smi": True, "py-spy": True, "curl": True},
        # s00_preflight.sh records the operator-level holder file; None means
        # /spinning/gpu-arb/holder does not exist.
        "arb_holder": None,
    }
    payload.update(over)
    return payload


# The real line shape of /spinning/gpu-arb/holder.
ARB_LINE = (
    "session={session}  cards=0,1,2  purpose=274-r7c-boot-b-dense-head  "
    "since=2026-07-29T23:32:20Z"
)


class TestPreflightCheck:
    CHECK, STEP = "check_s00_preflight.py", "s00_preflight"

    def test_pass(self, tmp_path):
        write_json(tmp_path / "preflight.json", _preflight())
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_red_corridor_is_stop(self, tmp_path):
        payload = _preflight()
        payload["cards"][1]["vram_free_mib"] = 120
        write_json(tmp_path / "preflight.json", payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "120" in line

    def test_broken_pci_join_is_stop(self, tmp_path):
        """Without the join every card index in every later report is
        ambiguous."""
        payload = _preflight()
        payload["cards"][0]["cuda_index"] = None
        write_json(tmp_path / "preflight.json", payload)
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_own_arbitration_window_passes(self, tmp_path):
        """The operator's own window is what authorises the battery."""
        payload = _preflight(arb_holder=ARB_LINE.format(session="operator"))
        write_json(tmp_path / "preflight.json", payload)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_foreign_arbitration_holder_is_stop(self, tmp_path):
        """The card locks only cover sessions that take them; the holder file
        is the cross-session window, and it was recorded and then read by
        nobody."""
        payload = _preflight(arb_holder=ARB_LINE.format(session="agent-lane-r7c"))
        write_json(tmp_path / "preflight.json", payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "agent-lane-r7c" in line

    def test_unreadable_arbitration_holder_is_stop(self, tmp_path):
        payload = _preflight(arb_holder="<unreadable>")
        write_json(tmp_path / "preflight.json", payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "session" in line

    def test_foreign_lock_is_stop(self, tmp_path):
        payload = _preflight(
            locks_held=[{"lock": "/tmp/gpu-card-0.lock", "info": "holder=fremd"}]
        )
        write_json(tmp_path / "preflight.json", payload)
        line = assert_stop(self.CHECK, tmp_path, self.STEP)
        assert "fremd" in line

    def test_missing_required_file_is_stop(self, tmp_path):
        write_json(
            tmp_path / "preflight.json", _preflight(required_files={"/nope": False})
        )
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_missing_pyspy_is_stop(self, tmp_path):
        payload = _preflight()
        payload["tools"]["py-spy"] = False
        write_json(tmp_path / "preflight.json", payload)
        assert_stop(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# s01 p2p re-probe
# ---------------------------------------------------------------------------


def _capability(**over):
    """Shaped after scripts/p2p_readiness/capability_matrix.py: rows live under
    "directed_pairs" and every row carries the DST card's BAR classification.
    The predecessor of this fixture invented a top-level "pairs" key, which no
    producer ever wrote -- the check built on it failed every real run."""

    def pair(src, dst, **kw):
        row = {
            "src_pci": src,
            "dst_pci": dst,
            "can_access_peer": True,
            "dst_bar1_nominal_bytes": 256 * MIB,
            "dst_bar1_classification": "windowed",
            "effective_max_single_copy_bytes": 240 * MIB,
            "effective_max_region_chunked_bytes": 250 * MIB,
            "probe_errors": [],
        }
        row.update(kw)
        return row

    def device(pci):
        return {
            "pci_bus_id": pci,
            "name": "NVIDIA GeForce RTX 3080",
            "uuid": f"GPU-{pci}",
            "nvml_index": 0,
            "cuda_index": 0,
            "vram_total_bytes": 20 * 1024 * MIB,
            "bar1_total_bytes": 256 * MIB,
            "bar1_classification": "windowed",
        }

    payload = {
        "schema_version": 3,
        "kind": "capability_matrix",
        "devices": [device(PCI_A), device(PCI_B)],
        "directed_pairs": [pair(PCI_A, PCI_B), pair(PCI_B, PCI_A)],
    }
    payload.update(over)
    return payload


def _d2d(**over):
    ladder = [64 * 1024, 1 * MIB, 255 * MIB, 256 * MIB, 257 * MIB]

    def row(src, dst, mode):
        return {
            "src": 0,
            "dst": 1,
            "src_pci": src,
            "dst_pci": dst,
            "mode": mode,
            "points": [
                {
                    "n": 40,
                    "median_s": s / 1e10,
                    "p95_s": s / 9e9,
                    "gib_per_s": 6.0,
                    "status": "ok",
                    "size_bytes": s,
                }
                for s in ladder
            ],
        }

    def leg(src, dst):
        return {
            "src": src,
            "dst": dst,
            "n": 22,
            "median_s": 0.09,
            "p95_s": 0.095,
            "gib_per_s": 2.0,
            "status": "ok",
        }

    payload = {
        "schema_version": 3,
        "kind": "d2d_bench",
        "pairs": [
            row(PCI_A, PCI_B, "direct"),
            row(PCI_B, PCI_A, "direct"),
            row(PCI_A, PCI_B, "staged"),
            row(PCI_B, PCI_A, "staged"),
        ],
        "arms": [
            {"kind": "bidir", "legs": [leg(0, 1), leg(1, 0)], "size_bytes": 192 * MIB},
        ],
    }
    payload.update(over)
    return payload


def _nccl_transport(**over):
    payload = {
        "schema_version": 3,
        "kind": "nccl_transport",
        "pairs": [
            {
                "cuda_pair": [0, 1],
                "pci_pair": [PCI_A, PCI_B],
                "status": "ok",
                "transports": [],
                "transport_summary": {"SHM": 2},
            }
        ],
    }
    payload.update(over)
    return payload


def _copy_real_run(tmp_path):
    """Copy the trimmed real-run artifacts into a step dir.

    Fixtures that were invented rather than derived are what let the schema
    drift sit undetected, so every artifact under FIXTURES is a shortened,
    anonymised copy of an actual run and keeps its structure byte-for-byte.
    """
    src = os.path.join(FIXTURES, "s01_p2p_reprobe", "results")
    dst = tmp_path / "results"
    shutil.copytree(src, dst)
    return dst


def _write_p2p(tmp_path, cap=None, d2d=None, nccl=None):
    results = tmp_path / "results"
    write_json(
        results / "capability_matrix.json", cap if cap is not None else _capability()
    )
    write_json(results / "d2d_bench.json", d2d if d2d is not None else _d2d())
    write_json(
        results / "nccl_transport.json", nccl if nccl is not None else _nccl_transport()
    )
    return results


class _FakeTensor:
    def __getitem__(self, idx):
        return self

    def item(self):
        return 2.0


class TestNcclTransportWorkerPinning:
    """The producer behind s01's third artifact, pinned without a card.

    Both ranks used to call set_device(0) while CUDA_VISIBLE_DEVICES listed
    BOTH cards of the pair, so two ranks landed on one device and NCCL refused
    the communicator outright ("Duplicate GPU detected"). The 2026-07-30 run
    recorded exactly that for all three pairs. torch is faked here, so the
    pinning is falsifiable on a machine with no GPU at all."""

    def _drive(self, rank):
        import types

        calls = []
        torch_stub = types.ModuleType("torch")
        torch_stub.float32 = "float32"
        torch_stub.cuda = types.SimpleNamespace(
            set_device=lambda dev: calls.append(("set_device", dev)),
            synchronize=lambda: calls.append(("synchronize", None)),
        )

        def ones(_n, dtype=None, device=None):
            calls.append(("ones", device))
            return _FakeTensor()

        torch_stub.ones = ones
        dist_stub = types.ModuleType("torch.distributed")
        dist_stub.init_process_group = lambda backend, rank, world_size: calls.append(
            ("init_process_group", rank)
        )
        dist_stub.all_reduce = lambda t: calls.append(("all_reduce", None))
        dist_stub.destroy_process_group = lambda: calls.append(("destroy", None))
        torch_stub.distributed = dist_stub

        p2p = os.path.join(REPO_ROOT, "scripts", "p2p_readiness")
        sys.path.insert(0, p2p)
        saved = {k: sys.modules.get(k) for k in ("torch", "torch.distributed")}
        sys.modules["torch"] = torch_stub
        sys.modules["torch.distributed"] = dist_stub
        try:
            import nccl_transport_check

            rc = nccl_transport_check.worker(rank)
        finally:
            for key, value in saved.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        return rc, calls

    @pytest.mark.parametrize("rank", (0, 1))
    def test_rank_pins_its_own_card(self, rank):
        rc, calls = self._drive(rank)
        assert rc == 0
        assert ("set_device", rank) in calls
        assert ("ones", f"cuda:{rank}") in calls

    def test_device_is_pinned_before_the_communicator_is_built(self):
        """NCCL reads the current device while building the communicator; a
        set_device afterwards is a set_device too late."""
        _, calls = self._drive(1)
        order = [name for name, _ in calls]
        assert order.index("set_device") < order.index("init_process_group")


class TestP2PCheck:
    CHECK, STEP = "check_s01_p2p_reprobe.py", "s01_p2p_reprobe"

    def test_pass(self, tmp_path):
        _write_p2p(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_no_results_dir_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_real_run_artifacts(self, tmp_path):
        """The check against a COPY OF THE REAL RUN (2026-07-30_bar1), not
        against a fixture someone imagined.

        This is the regression test for the defect that motivated the fix: the
        check asserted a top-level "pairs" key that scripts/p2p_readiness/
        never wrote, so it failed on a perfectly healthy measurement. Both the
        capability matrix and the d2d bench of that run must pass.

        The run's nccl_transport arm genuinely failed (both ranks landed on the
        same CUDA device), so the honest verdict on the artifact as a whole is
        FAIL -- and the reason must name the cause, not just the exit code."""
        results = _copy_real_run(tmp_path)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "nccl_transport" in line
        assert "Duplicate GPU detected" in line

        # ... and with only that arm healed, the real artifacts pass.
        nccl = json.loads((results / "nccl_transport.json").read_text())
        for row in nccl["pairs"]:
            row["status"] = "ok"
            row["transport_summary"] = {"SHM": 2}
        write_json(results / "nccl_transport.json", nccl)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_real_capability_matrix_reaches_the_279_loader(self, tmp_path):
        """The real matrix must not just parse -- it must be SEEN by the
        loader the dispatcher uses. Before the fix the loader read the same
        wrong key and returned zero rows, zero errors and zero skips: total
        silence, indistinguishable from a rig without P2P."""
        results = _copy_real_run(tmp_path)
        sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
        from sglang.srt.distributed.device_communicators.barlink_path_rates import (
            load_p2p_capability_matrix,
        )

        payload = json.loads((results / "capability_matrix.json").read_text())
        res = load_p2p_capability_matrix(payload)
        assert not res.errors
        assert len(res.skipped) == len(payload["directed_pairs"])

    def test_unmeasured_peer_flag_is_fail(self, tmp_path):
        """None is not False: it means the probe did not run."""
        cap = _capability()
        cap["directed_pairs"][0]["can_access_peer"] = None
        _write_p2p(tmp_path, cap=cap)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "can_access_peer" in line

    def test_legacy_pairs_key_is_fail(self, tmp_path):
        """A matrix that puts its rows anywhere but under the producer's key
        is a matrix nobody downstream will find."""
        cap = _capability()
        cap["pairs"] = cap.pop("directed_pairs")
        _write_p2p(tmp_path, cap=cap)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "directed_pairs" in line

    def test_missing_bar_classification_is_fail(self, tmp_path):
        cap = _capability()
        cap["directed_pairs"][0]["dst_bar1_classification"] = "unknown"
        _write_p2p(tmp_path, cap=cap)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "dst_bar1_classification" in line

    def test_classification_disagreeing_with_devices_is_fail(self, tmp_path):
        """Row and card survey must describe the same state."""
        cap = _capability()
        for row in cap["directed_pairs"]:
            row["dst_bar1_classification"] = "full"
            row["dst_bar1_nominal_bytes"] = 32 * 1024 * MIB
        _write_p2p(tmp_path, cap=cap)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "devices[]" in line

    def test_missing_pressure_arms_is_fail(self, tmp_path):
        """A ladder measured one copy at a time cannot see what two
        simultaneous copies do to the same window."""
        _write_p2p(tmp_path, d2d=_d2d(arms=[]))
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "arms" in line

    def test_failed_point_without_error_is_fail(self, tmp_path):
        d2d = _d2d()
        d2d["pairs"][0]["points"][-1] = {
            "size_bytes": 257 * MIB,
            "status": "failed",
        }
        _write_p2p(tmp_path, d2d=d2d)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "failed" in line

    def test_point_above_aperture_with_error_passes(self, tmp_path):
        """A copy above the effective aperture is a RESULT, not a defect."""
        d2d = _d2d()
        d2d["pairs"][0]["points"][-1] = {
            "size_bytes": 257 * MIB,
            "status": "failed",
            "error": "RuntimeError: mapping failure",
        }
        _write_p2p(tmp_path, d2d=d2d)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_p2p_without_effective_aperture_is_fail(self, tmp_path):
        """The failure mode the whole step exists to catch: p2p=True with a
        null effective aperture degrades every consumer to a placeholder while
        looking like a successful run."""
        cap = _capability()
        cap["directed_pairs"][0]["effective_max_single_copy_bytes"] = None
        _write_p2p(tmp_path, cap=cap)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "effective_max_single_copy_bytes" in line

    def test_no_p2p_anywhere_still_passes(self, tmp_path):
        """'No P2P' is a legitimate, fully recorded outcome, not a failure.
        This is what the real 2026-07-30 run looks like."""
        cap = _capability()
        for row in cap["directed_pairs"]:
            row["can_access_peer"] = False
            row["effective_max_single_copy_bytes"] = None
            row["effective_max_region_chunked_bytes"] = None
        _write_p2p(tmp_path, cap=cap)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_one_directional_matrix_is_fail(self, tmp_path):
        cap = _capability()
        cap["directed_pairs"] = cap["directed_pairs"][:1]
        _write_p2p(tmp_path, cap=cap)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_window_bracket_is_fail(self, tmp_path):
        """The knee at the window boundary IS the measurement; a ladder that
        steps over 255/256/257 MiB cannot show one."""
        d2d = _d2d()
        for row in d2d["pairs"]:
            row["points"] = [p for p in row["points"] if p["size_bytes"] != 256 * MIB]
        _write_p2p(tmp_path, d2d=d2d)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Klammer" in line

    def test_only_direct_mode_is_fail(self, tmp_path):
        d2d = _d2d()
        d2d["pairs"] = [r for r in d2d["pairs"] if r["mode"] == "direct"]
        _write_p2p(tmp_path, d2d=d2d)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_nccl_timeout_is_fail(self, tmp_path):
        nccl = _nccl_transport()
        nccl["pairs"][0]["status"] = "timeout after 120s"
        _write_p2p(tmp_path, nccl=nccl)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_wrong_envelope_is_fail(self, tmp_path):
        _write_p2p(tmp_path, cap=_capability(kind="something_else"))
        assert_fail(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# s02-s04 accept boots
# ---------------------------------------------------------------------------

PROMPTS = ("alphabet", "squares", "repeat", "code", "prose")


def _accept(**over):
    curve = {"0": 0.65, "1": 0.41, "2": 0.22}
    payload = {
        "tokens": 192,
        "steps": 3,
        "arms": [
            {
                "prompt": p,
                "prompt_tokens": 40,
                "serving": {
                    "accept_len_mean": 2.9,
                    "rounds": 60,
                    "curve": dict(curve),
                    "completion_tokens": 192,
                    "spec_accept_length": 2.88,
                },
            }
            for p in PROMPTS
        ],
    }
    payload.update(over)
    return payload


def _reference_column(accept):
    from check_common import (  # noqa: E402
        REFERENCE_ACCEPT,
        REFERENCE_COLUMN_KIND,
        REFERENCE_COLUMN_SCHEMA_VERSION,
        REFERENCE_SOURCE,
    )

    rows = []
    for arm in accept["arms"]:
        measured = arm["serving"]["accept_len_mean"]
        ref = REFERENCE_ACCEPT.get(arm["prompt"])
        rows.append(
            {
                "prompt": arm["prompt"],
                "measured": measured,
                "reference": ref,
                "ratio": round(measured / ref, 4) if ref and measured else None,
            }
        )
    return {
        "kind": REFERENCE_COLUMN_KIND,
        "schema_version": REFERENCE_COLUMN_SCHEMA_VERSION,
        "reference_source": REFERENCE_SOURCE,
        "rows": rows,
    }


VRAM_SUMMARY = (
    "NVML Karte                        Peak benutzt    Total  MIN frei\n"
    "   0 NVIDIA GeForce RTX 5090             30000    32768      2768\n"
    "   1 NVIDIA GeForce RTX 3080             18000    20480      2480\n"
)


def _write_boot(tmp_path, accept=None, log="Server started\n", extra_files=None):
    sys.path.insert(0, CHECKS)
    accept = accept if accept is not None else _accept()
    write_json(tmp_path / "accept.json", accept)
    write_json(tmp_path / "reference_column.json", _reference_column(accept))
    (tmp_path / "vram_summary.txt").write_text(VRAM_SUMMARY)
    if log is not None:
        (tmp_path / "server.log").write_text(log)
    for name, content in (extra_files or {}).items():
        (tmp_path / name).write_text(content)
    return tmp_path


class TestBootAcceptCheck:
    CHECK, STEP = "check_s02_boot_a.py", "s02_boot_a"

    def test_pass(self, tmp_path):
        _write_boot(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_falsifying_numbers_still_pass(self, tmp_path):
        """The falsifying outcome (accept ~1.5, position 0 at 24-45 %) is a
        RESULT. A check with an opinion about it would refuse the answer the
        boot was sent to collect."""
        accept = _accept()
        for arm in accept["arms"]:
            arm["serving"]["accept_len_mean"] = 1.42
            arm["serving"]["curve"] = {"0": 0.27, "1": 0.08, "2": 0.01}
        _write_boot(tmp_path, accept=accept)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_no_artifact_and_no_log_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_no_artifact_but_oom_in_log_is_fail(self, tmp_path):
        """The boot ran and said something -- that belongs to the fixer, not
        to the environment bucket."""
        (tmp_path / "server.log").write_text(
            "...\ntorch.OutOfMemoryError: CUDA out of memory\n"
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "out of memory" in line.lower()

    def test_no_artifact_but_log_present_is_fail(self, tmp_path):
        (tmp_path / "server.log").write_text("loading weights...\n")
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_none_accept_is_fail(self, tmp_path):
        accept = _accept()
        accept["arms"][2]["serving"]["accept_len_mean"] = None
        _write_boot(tmp_path, accept=accept)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "accept_len_mean" in line

    def test_missing_position_curve_is_fail(self, tmp_path):
        """A mean is structurally blind to a positional pathology; that
        blindness is how the round-7a defect survived."""
        accept = _accept()
        accept["arms"][0]["serving"]["curve"] = None
        _write_boot(tmp_path, accept=accept)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Positionskurve" in line

    def test_truncated_curve_is_fail(self, tmp_path):
        accept = _accept()
        accept["arms"][0]["serving"]["curve"] = {"0": 0.6}
        _write_boot(tmp_path, accept=accept)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_prompt_is_fail(self, tmp_path):
        accept = _accept()
        accept["arms"] = accept["arms"][:3]
        _write_boot(tmp_path, accept=accept)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_reference_column_is_stop(self, tmp_path):
        _write_boot(tmp_path)
        os.unlink(tmp_path / "reference_column.json")
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_wrong_reference_value_is_fail(self, tmp_path):
        """2.75-2.82 are two cells of a five-axis cross-algo battery and are
        not a comparison for a K=3 measurement."""
        accept = _accept()
        _write_boot(tmp_path, accept=accept)
        ref = _reference_column(accept)
        for row in ref["rows"]:
            if row["prompt"] == "prose":
                row["reference"] = 2.75
        write_json(tmp_path / "reference_column.json", ref)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Referenzwert" in line

    def test_missing_vram_summary_is_stop(self, tmp_path):
        _write_boot(tmp_path)
        os.unlink(tmp_path / "vram_summary.txt")
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_traceback_in_log_is_fail(self, tmp_path):
        _write_boot(tmp_path, log="ok\nTraceback (most recent call last):\n  boom\n")
        assert_fail(self.CHECK, tmp_path, self.STEP)


class TestBootCCheck:
    CHECK, STEP = "check_s04_boot_c.py", "s04_boot_c"

    def test_pass(self, tmp_path):
        _write_boot(
            tmp_path, extra_files={"loader_lines.txt": "loading dflash draft head\n"}
        )
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_incoherent_output_is_not_the_checks_business(self, tmp_path):
        """Boot C's own abort list says incoherent output is a RESULT."""
        accept = _accept()
        for arm in accept["arms"]:
            arm["serving"]["accept_len_mean"] = 0.4
        _write_boot(
            tmp_path,
            accept=accept,
            extra_files={"loader_lines.txt": "dflash drafter ready\n"},
        )
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_no_drafter_trace_is_fail(self, tmp_path):
        """A server that only came up on the target passes every accept
        assertion and has not answered Boot C's question."""
        _write_boot(tmp_path, extra_files={"loader_lines.txt": "gguf target loaded\n"})
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "draft" in line.lower()

    def test_empty_loader_lines_is_fail(self, tmp_path):
        _write_boot(tmp_path, extra_files={"loader_lines.txt": "  \n"})
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_loader_lines_is_stop(self, tmp_path):
        _write_boot(tmp_path)
        assert_stop(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# s05 boot D
# ---------------------------------------------------------------------------


def _reseed(**over):
    def arm(accept, reseed_forwards):
        return {
            "accept_len_mean": accept,
            # A LIST, the way boot_d_lane_reseed.sh copies it out of
            # accept_positions["rate"] -- not the dict this fixture used to
            # invent while no check looked at the field.
            "curve": [0.61, 0.38, 0.19],
            "decode_ms_mean": 41.2,
            "head_forwards": 100,
            "reseed_forwards": reseed_forwards,
            "spec_rounds": 60,
            "output_ids": [1, 2, 3],
        }

    payload = {
        "arms": [
            {
                "prompt": p,
                "prompt_tokens": 30,
                "arms": {"True": arm(1.4, 30), "False": arm(1.38, None)},
                "output_identical": True,
            }
            for p in ("squares", "code", "prose")
        ]
    }
    payload.update(over)
    return payload


def _write_boot_d(tmp_path, reseed=None, log="ok\n"):
    write_json(tmp_path / "reseed.json", reseed if reseed is not None else _reseed())
    (tmp_path / "vram_summary.txt").write_text(VRAM_SUMMARY)
    (tmp_path / "server.log").write_text(log)
    return tmp_path


class TestBootDCheck:
    CHECK, STEP = "check_s05_boot_d.py", "s05_boot_d"

    def test_pass(self, tmp_path):
        _write_boot_d(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_differing_outputs_are_not_a_failure(self, tmp_path):
        """output_identical == False IS the measurement."""
        reseed = _reseed()
        for row in reseed["arms"]:
            row["output_identical"] = False
        _write_boot_d(tmp_path, reseed=reseed)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_single_arm_is_fail(self, tmp_path):
        reseed = _reseed()
        del reseed["arms"][1]["arms"]["False"]
        _write_boot_d(tmp_path, reseed=reseed)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "A/B" in line

    def test_missing_decode_timing_is_fail(self, tmp_path):
        """The price of the re-seed is the point of the boot and it is read
        off decode_ms_mean."""
        reseed = _reseed()
        reseed["arms"][0]["arms"]["True"]["decode_ms_mean"] = None
        _write_boot_d(tmp_path, reseed=reseed)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_reseed_counter_is_fail(self, tmp_path):
        reseed = _reseed()
        reseed["arms"][0]["arms"]["True"]["reseed_forwards"] = None
        _write_boot_d(tmp_path, reseed=reseed)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_zero_spec_rounds_is_fail(self, tmp_path):
        """An accept mean over zero proposal rounds is not a small number, it
        is no number -- and the whole A/B is read off exactly that."""
        reseed = _reseed()
        reseed["arms"][0]["arms"]["False"]["spec_rounds"] = 0
        _write_boot_d(tmp_path, reseed=reseed)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "spec_rounds" in line

    def test_missing_curve_is_fail(self, tmp_path):
        reseed = _reseed()
        reseed["arms"][1]["arms"]["True"]["curve"] = None
        _write_boot_d(tmp_path, reseed=reseed)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Positionskurve" in line

    def test_short_curve_is_fail(self, tmp_path):
        """K=3 means three positions; two of them is a curve with a hole."""
        reseed = _reseed()
        reseed["arms"][0]["arms"]["True"]["curve"] = [0.6, 0.3]
        _write_boot_d(tmp_path, reseed=reseed)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Positionen" in line

    def test_dict_curve_still_accepted(self, tmp_path):
        """JSON string keys are what the other probes emit; both shapes are a
        curve."""
        reseed = _reseed()
        for row in reseed["arms"]:
            for arm in row["arms"].values():
                arm["curve"] = {"0": 0.6, "1": 0.35, "2": 0.18}
        _write_boot_d(tmp_path, reseed=reseed)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_without_log_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# s06 NCCL reference
# ---------------------------------------------------------------------------


def _nccl_rows():
    rows = []
    for load in ("idle", "host_stream_64mib"):
        for size in (64 * 1024, 1 * MIB):
            rows.append(
                {
                    "op": "all_reduce",
                    "transport": "SHM",
                    "world": 2,
                    "src_pci": PCI_A,
                    "dst_pci": PCI_B,
                    "size_bytes": size,
                    "iters": 50,
                    "p50_us": 120.0,
                    "p99_us": 190.0,
                    "load": load,
                }
            )
            for src, dst in ((PCI_A, PCI_B), (PCI_B, PCI_A)):
                rows.append(
                    {
                        "op": "send_recv",
                        "transport": "SHM",
                        "world": 2,
                        "src_pci": src,
                        "dst_pci": dst,
                        "size_bytes": size,
                        "iters": 50,
                        "p50_us": 95.0,
                        "p99_us": 150.0,
                        "load": load,
                    }
                )
    return rows


def _nccl_reference(**over):
    payload = {
        "schema_version": 1,
        "kind": "nccl_reference",
        "rows": _nccl_rows(),
        "pairs_status": [{"pci_pair": [PCI_A, PCI_B], "status": "ok", "rows": 12}],
    }
    payload.update(over)
    return payload


class TestNcclReferenceCheck:
    CHECK, STEP = "check_s06_nccl_reference.py", "s06_nccl_reference"

    def test_pass(self, tmp_path):
        write_json(tmp_path / "nccl_reference.json", _nccl_reference())
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_row_missing_a_mandatory_field_is_fail(self, tmp_path):
        """The loader drops such rows, so a file of them loads as empty."""
        payload = _nccl_reference()
        del payload["rows"][0]["p99_us"]
        write_json(tmp_path / "nccl_reference.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "p99_us" in line

    def test_p99_below_p50_is_fail(self, tmp_path):
        payload = _nccl_reference()
        payload["rows"][0]["p99_us"] = 1.0
        write_json(tmp_path / "nccl_reference.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_no_load_arm_is_fail(self, tmp_path):
        """The #278 lesson: a load axis is only worth taking symmetrically,
        and the schema makes it mandatory."""
        payload = _nccl_reference()
        payload["rows"] = [r for r in payload["rows"] if r["load"] == "idle"]
        write_json(tmp_path / "nccl_reference.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Last-Arm" in line

    def test_asymmetric_load_coverage_is_fail(self, tmp_path):
        payload = _nccl_reference()
        payload["rows"] = [
            r
            for r in payload["rows"]
            if not (r["load"] != "idle" and r["size_bytes"] == 1 * MIB)
        ]
        write_json(tmp_path / "nccl_reference.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "symmetrisch" in line

    def test_one_directional_send_recv_is_fail(self, tmp_path):
        """A symmetric-only table averages away exactly the asymmetry the rig
        was measured for."""
        payload = _nccl_reference()
        payload["rows"] = [
            r
            for r in payload["rows"]
            if not (r["op"] == "send_recv" and r["src_pci"] == PCI_B)
        ]
        write_json(tmp_path / "nccl_reference.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Richtung" in line

    def test_empty_transport_is_fail(self, tmp_path):
        payload = _nccl_reference()
        payload["rows"][0]["transport"] = None
        write_json(tmp_path / "nccl_reference.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_aborted_pair_is_fail(self, tmp_path):
        payload = _nccl_reference()
        payload["pairs_status"][0]["status"] = "timeout after 600s"
        write_json(tmp_path / "nccl_reference.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_wrong_schema_version_is_fail(self, tmp_path):
        write_json(tmp_path / "nccl_reference.json", _nccl_reference(schema_version=2))
        assert_fail(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# s06 producer: pair launch, timeout, logging
# ---------------------------------------------------------------------------


def _import_s06():
    sys.path.insert(0, BATTERY)
    import s06_nccl_reference

    return s06_nccl_reference


# A stand-in rank. It leaves a breadcrumb, then does what the argument says,
# so the parent's timeout, logging and kill paths are testable without a card.
STANDIN = """
import os, sys, time
frag, rank, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(f"{frag}.rank{rank}.progress", "w") as f:
    f.write("idle/all_reduce/65536B\\n")
    f.flush()
    os.fsync(f.fileno())
if mode == "flood":
    sys.stderr.write("MARKER\\n" + "x" * (512 * 1024) + "\\nSCHLUSS\\n")
    sys.stderr.flush()
    sys.exit(0)
sys.stderr.write("MARKER\\n")
sys.stderr.flush()
time.sleep(600)
"""


class TestS06PairLaunch:
    """The producer behind s06, driven with a stand-in child.

    The 2026-07-30 run wedged for 20 minutes and left a 0-byte NCCL log, an
    empty pid list and no result file. Every one of those four properties is
    pinned here, CPU-only.
    """

    def _run_pair(self, s06, tmp_path, mode, timeout_s, monkeypatch):
        frag = str(tmp_path / "ref.json.frag.0-1")
        monkeypatch.setattr(
            s06,
            "worker_argv",
            lambda rank, frag_path: [
                sys.executable,
                "-c",
                STANDIN,
                frag_path,
                str(rank),
                mode,
            ],
        )
        monkeypatch.setenv("BATTERY_STEP_DIR", str(tmp_path))
        (tmp_path / "pids").write_text("")
        return s06.run_pair(
            (0, 1),
            (PCI_A, PCI_B),
            frag,
            timeout_s,
            str(tmp_path / "seg"),
            (64 * 1024,),
            ("idle",),
        )

    def test_each_rank_sees_exactly_one_card(self, tmp_path, monkeypatch):
        """The hang itself: both ranks saw BOTH cards, so both payloads landed
        on the first one while barrier() fell back to rank % 2 and built its
        communicator on two different ones."""
        s06 = _import_s06()
        seen = []

        class _Fake:
            pid = os.getpid()
            returncode = 0

            def __init__(self, argv, **kw):
                seen.append(kw)

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return 0

        monkeypatch.setattr(s06.subprocess, "Popen", _Fake)
        monkeypatch.setattr(s06, "register_pid", lambda pid: None)
        s06.run_pair(
            (2, 5),
            (PCI_A, PCI_B),
            str(tmp_path / "frag"),
            10,
            str(tmp_path / "seg"),
            (64 * 1024,),
            ("idle",),
        )
        visible = [kw["env"]["CUDA_VISIBLE_DEVICES"] for kw in seen]
        assert visible == ["2", "5"], visible
        for kw in seen:
            assert kw["start_new_session"] is True
            assert kw["stdout"] is not s06.subprocess.PIPE
            assert kw["stderr"] is s06.subprocess.STDOUT

    def test_child_output_is_not_read_through_an_unattended_pipe(
        self, tmp_path, monkeypatch
    ):
        """A PIPE nobody reads holds 64 KiB and then blocks the writer. The
        previous version waited on rank 0 first, so rank 1 could sit in a full
        pipe for the whole timeout. Half a MiB of NCCL_DEBUG output must pass
        through without the child stalling."""
        s06 = _import_s06()
        status, detail, log = self._run_pair(s06, tmp_path, "flood", 60, monkeypatch)
        assert status == "ok", (status, detail)
        assert log.count("MARKER") == 2
        assert log.count("SCHLUSS") == 2
        assert len(log) > 1024 * 1024

    def test_timeout_binds_the_hang_and_names_where_it_stood(
        self, tmp_path, monkeypatch
    ):
        s06 = _import_s06()
        t0 = time.monotonic()
        status, detail, log = self._run_pair(s06, tmp_path, "hang", 3, monkeypatch)
        elapsed = time.monotonic() - t0
        assert elapsed < 60, elapsed
        assert status == "timeout nach 3s", status
        # honest partial result: which rank, and where it stood
        assert "Rang 0" in detail and "Rang 1" in detail
        assert "idle/all_reduce/65536B" in detail
        assert "MARKER" in log
        pids = [int(x) for x in (tmp_path / "pids").read_text().split() if x.strip()]
        assert len(pids) == 2
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            time.sleep(0.2)
        assert not any(_alive(p) for p in pids), "Kinder leben nach dem Timeout"

    def test_timeout_writes_a_pyspy_dump_before_the_kill(self, tmp_path, monkeypatch):
        s06 = _import_s06()
        self._run_pair(s06, tmp_path, "hang", 3, monkeypatch)
        for rank in (0, 1):
            dump = tmp_path / f"seg.rank{rank}.pyspy.txt"
            assert dump.exists(), f"kein Dump fuer Rang {rank}"
            assert dump.read_text().strip()

    def test_killpg_only_ever_targets_a_group_this_parent_created(self, monkeypatch):
        """Blast radius: a pgid that is not the child's own pid belongs to
        somebody else's session and must never receive a group signal."""
        s06 = _import_s06()
        killed = []
        monkeypatch.setattr(s06.os, "killpg", lambda pgid, sig: killed.append(pgid))
        monkeypatch.setattr(s06.os, "getpgid", lambda pid: 4242)

        class _Foreign:
            pid = 1234
            terminated = False

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                pass

        proc = _Foreign()
        s06.terminate_own_group(proc)
        assert killed == []
        assert proc.terminated

    def test_partial_payload_is_written_after_every_pair(self, tmp_path):
        s06 = _import_s06()
        out = tmp_path / "nccl_reference.json"
        payload = {
            "schema_version": 1,
            "kind": "nccl_reference",
            "pairs_status": [
                {
                    "pci_pair": [PCI_A, PCI_B],
                    "status": "timeout nach 600s",
                    "detail": "Rang 1 bei idle/all_reduce/65536B",
                    "rows": 0,
                }
            ],
        }
        s06.write_payload(payload, [], str(out))
        assert not (tmp_path / "nccl_reference.json.tmp").exists()
        # The honest fragment is a FAIL with a reason, not a missing artifact.
        line = assert_fail(
            "check_s06_nccl_reference.py", tmp_path, "s06_nccl_reference"
        )
        assert "timeout" in line


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestS06WorkerPreconditions:
    """Rank-local before collective: everything decidable without the peer is
    decided first, so a dead or wrongly pinned peer ends as a named error
    instead of a wedge inside a collective."""

    def _torch_stub(self, device_count, pci):
        import types

        torch_stub = types.ModuleType("torch")
        torch_stub.float32 = "float32"
        torch_stub.cuda = types.SimpleNamespace(
            device_count=lambda: device_count,
            set_device=lambda dev: None,
            synchronize=lambda: None,
            empty_cache=lambda: None,
        )

        class _T:
            def mul_(self, _v):
                return self

            def sum(self):
                return self

            def item(self):
                return 8192.0

        torch_stub.ones = lambda *a, **kw: _T()
        return torch_stub, pci

    def _drive(self, monkeypatch, device_count, own_pci, assigned_pci):
        s06 = _import_s06()
        torch_stub, _ = self._torch_stub(device_count, own_pci)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
        sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "p2p_readiness"))
        import p2p_common

        monkeypatch.setattr(p2p_common, "cuda_pci_bus_id", lambda dev: own_pci)
        monkeypatch.setenv("BATTERY_PCI_0", assigned_pci)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        return s06

    def test_two_visible_cards_is_a_named_error(self, tmp_path, monkeypatch):
        s06 = self._drive(monkeypatch, 2, PCI_A, PCI_A)
        with pytest.raises(s06.PreconditionFailed) as exc:
            s06._rank_local_checks(0, str(tmp_path / "p"))
        assert exc.value.rc == s06.RC_DEVICE_COUNT
        assert "erwartet genau 1" in str(exc.value)

    def test_wrong_card_is_a_named_error(self, tmp_path, monkeypatch):
        s06 = self._drive(monkeypatch, 1, PCI_B, PCI_A)
        with pytest.raises(s06.PreconditionFailed) as exc:
            s06._rank_local_checks(0, str(tmp_path / "p"))
        assert exc.value.rc == s06.RC_DEVICE_IDENTITY

    def test_correct_pinning_passes_and_leaves_a_breadcrumb(
        self, tmp_path, monkeypatch
    ):
        s06 = self._drive(monkeypatch, 1, PCI_A, PCI_A)
        progress = str(tmp_path / "p")
        assert s06._rank_local_checks(0, progress) == PCI_A
        assert "Kartenidentitaet" in open(progress).read()


# ---------------------------------------------------------------------------
# s07 offload register
# ---------------------------------------------------------------------------


def _offload(**over):
    def row(cls, route):
        return {
            "offload_class": cls,
            "route": route,
            "item_id": f"battery:{cls}",
            "size_bytes": 256 * MIB,
            "size_source_matches": True,
            "park_ms_p50": 61.0,
            "park_ms_p99": 70.0,
            "wave_in_ms_p50": 58.0,
            "wave_in_ms_p99": 66.0,
            "iters": 5,
            "wave_in_gb_per_s": 4.6,
            "state_sequence": ["resident", "parked", "resident"],
            "status": "ok",
        }

    payload = {
        "kind": "offload_register_gpu",
        "schema_version": 1,
        "device_ops": "CudaDeviceOps",
        "device": {"cuda_index": 0, "name": "RTX 5090", "pci_bus_id": PCI_A},
        "memory_saver": "real",
        "routes": {"tensor": "ok", "tag": "ok", "suspend": "ok"},
        "rows": [
            row("lane_workspaces", "tensor"),
            row("kv_shadow", "tensor"),
            row("graph_rungs", "tag"),
            row("gdn_state_sets", "tag"),
            row("cold_lane", "suspend"),
        ],
        "stats": {
            "parks": 25,
            "wave_ins": 25,
            "park_failures": 0,
            "wave_in_failures": 0,
        },
        "latency_term_ms": {"lane_workspaces": 58.0},
        "negative_control": {
            "policy": "auto",
            "sensor": "none",
            "refused": True,
            "error": (
                "OffloadRefused: park refused: class 'lane_workspaces' policy is "
                "'auto' and the saturation sensor reports no pressure"
            ),
        },
    }
    payload.update(over)
    return payload


class TestOffloadRegisterCheck:
    CHECK, STEP = "check_s07_offload_register_gpu.py", "s07_offload_register_gpu"

    def test_pass(self, tmp_path):
        write_json(tmp_path / "offload_register_gpu.json", _offload())
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_fake_device_ops_is_fail(self, tmp_path):
        """A validation with FakeDeviceOps validates nothing and would pass
        every latency assertion in microseconds."""
        write_json(
            tmp_path / "offload_register_gpu.json", _offload(device_ops="FakeDeviceOps")
        )
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "FakeDeviceOps" in line

    def test_unavailable_saver_is_stop(self, tmp_path):
        """Two of three routes untested -- a green verdict on one third of the
        register would be worse than none."""
        payload = _offload()
        payload["routes"] = {
            "tensor": "ok",
            "tag": "unavailable",
            "suspend": "unavailable",
        }
        write_json(tmp_path / "offload_register_gpu.json", payload)
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_route_error_is_fail(self, tmp_path):
        payload = _offload()
        payload["routes"]["tag"] = "error"
        write_json(tmp_path / "offload_register_gpu.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_silent_noop_park_is_fail(self, tmp_path):
        """A park that no-ops returns exactly what a working park returns;
        only the state sequence tells them apart."""
        payload = _offload()
        payload["rows"][0]["state_sequence"] = ["resident", "resident", "resident"]
        write_json(tmp_path / "offload_register_gpu.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "No-Op" in line

    def test_zero_latency_is_fail(self, tmp_path):
        payload = _offload()
        payload["rows"][1]["wave_in_ms_p50"] = 0.0
        write_json(tmp_path / "offload_register_gpu.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_size_mismatch_is_fail(self, tmp_path):
        payload = _offload()
        payload["rows"][0]["size_source_matches"] = False
        write_json(tmp_path / "offload_register_gpu.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_movement_failures_are_fail(self, tmp_path):
        payload = _offload()
        payload["stats"]["wave_in_failures"] = 2
        write_json(tmp_path / "offload_register_gpu.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_latency_term_is_fail(self, tmp_path):
        write_json(tmp_path / "offload_register_gpu.json", _offload(latency_term_ms={}))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_negative_control_is_stop(self, tmp_path):
        """Without the control the rows do not say whether the explicit class
        policy admitted the park or whether 'auto' stopped gating."""
        payload = _offload()
        payload.pop("negative_control")
        write_json(tmp_path / "offload_register_gpu.json", payload)
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_unrefused_negative_control_is_fail(self, tmp_path):
        payload = _offload()
        payload["negative_control"]["refused"] = False
        write_json(tmp_path / "offload_register_gpu.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Negativkontrolle" in line


class TestOffloadRegisterProbe:
    """The producer behind s07, driven without a card.

    The 2026-07-30 run measured nothing: the probe registered its items with
    the default 'auto' policy, which parks only under saturation pressure, so
    all six classes answered OffloadRefused in 8 s. The probe now asks for its
    parks through the register's own per-class knob -- and proves in the same
    run that the gate it stepped around is still closed for everyone else.
    """

    def _probe(self):
        sys.path.insert(0, BATTERY)
        import s07_offload_register_gpu

        return s07_offload_register_gpu

    def _dry_run(self, tmp_path):
        s07 = self._probe()
        out = tmp_path / "offload_register_gpu.json"
        rc = s07.main(["--dry-run", "--out", str(out), "--cycles", "4"])
        assert rc == 0
        return s07, json.loads(out.read_text())

    def test_dry_run_cycles_every_route_and_fills_the_rates(self, tmp_path):
        _, payload = self._dry_run(tmp_path)
        assert payload["routes"] == {"tensor": "ok", "tag": "ok", "suspend": "ok"}
        assert [r["offload_class"] for r in payload["rows"]] == [
            "lane_workspaces",
            "kv_shadow",
            "experts",
            "graph_rungs",
            "gdn_state_sets",
            "cold_lane",
        ]
        for row in payload["rows"]:
            assert row["status"] == "ok", row
            assert row["iters"] == 4
            assert row["wave_in_gb_per_s"] > 0, row
            assert row["park_ms_p50"] is not None
            assert "parked" in row["state_sequence"]
            assert "resident" in row["state_sequence"]
        assert payload["stats"]["parks"] == 24
        assert payload["stats"]["park_failures"] == 0
        assert payload["stats"]["wave_in_failures"] == 0

    def test_measured_classes_are_parked_by_an_explicit_policy(self, tmp_path):
        """'ram' is the register's documented explicit knob; 'auto' is the
        sensor-gated default the measurement must not depend on."""
        _, payload = self._dry_run(tmp_path)
        modes = {k: v["mode"] for k, v in payload["class_policies"].items()}
        for row in payload["rows"]:
            assert modes[row["offload_class"]] == "ram", modes
        assert modes["drafter_heads"] == "auto", "unbeteiligte Klassen bleiben auto"

    def test_negative_control_still_refuses_auto_without_pressure(self, tmp_path):
        _, payload = self._dry_run(tmp_path)
        control = payload["negative_control"]
        assert control["refused"] is True
        assert "OffloadRefused" in control["error"]
        assert "saturation sensor reports no pressure" in control["error"]

    def test_latency_term_is_the_measured_retrieval_not_a_constant(self, tmp_path):
        """The #279 dispatcher reads this number; registering 0.0 would hand
        it a guess dressed as a measurement."""
        _, payload = self._dry_run(tmp_path)
        assert payload["latency_term_ms"]
        for offload_class, term in payload["latency_term_ms"].items():
            assert term > 0, (offload_class, term)

    def test_dry_run_artifact_is_not_accepted_as_a_validation(self, tmp_path):
        """The plan phase writes a real envelope; only the device layer name
        keeps it from passing as a card run."""
        self._dry_run(tmp_path)
        assert_fail(
            "check_s07_offload_register_gpu.py", tmp_path, "s07_offload_register_gpu"
        )

    def test_plan_phase_needs_no_out(self, capsys):
        """The step's plan phase runs --dry-run without --out; a required
        --out turned that into an argparse error in the step log."""
        s07 = self._probe()
        assert s07.main(["--dry-run"]) == 0
        assert "Plan:" in capsys.readouterr().out

    def test_out_is_required_for_a_real_run(self):
        s07 = self._probe()
        with pytest.raises(SystemExit) as exc:
            s07.main([])
        assert exc.value.code == 2

    def test_preload_reexec_happens_once_and_carries_the_hook(self):
        """LD_PRELOAD is read by the linker at process start: the tag and
        suspend routes cannot be repaired in-process, only by re-exec."""
        s07 = self._probe()
        calls = []
        env = {}
        note = s07.maybe_reexec_for_memory_saver(
            env,
            ["probe.py", "--out", "x.json"],
            lambda exe, argv: calls.append((exe, argv)),
            path_fn=lambda: "/pkg/torch_memory_saver_cpp.so",
            libdir_fn=lambda: "/pkg/nvidia/cu13/lib",
        )
        assert note == "reexec"
        assert env["LD_PRELOAD"] == "/pkg/torch_memory_saver_cpp.so"
        # Without the runtime's directory the re-exec kills the interpreter
        # itself, not the probe: the hook NEEDs libcudart.so.<major>.
        assert env["LD_LIBRARY_PATH"] == "/pkg/nvidia/cu13/lib"
        assert len(calls) == 1
        assert calls[0][1][1:] == ["probe.py", "--out", "x.json"]
        # The guard the re-execed process sees: no second exec.
        assert (
            s07.maybe_reexec_for_memory_saver(
                env,
                [],
                lambda *a: calls.append(a),
                path_fn=lambda: "/pkg/x.so",
                libdir_fn=lambda: "/pkg/lib",
            )
            == "reexec"
        )
        assert len(calls) == 1

    def test_inherited_preload_is_left_alone(self):
        s07 = self._probe()
        env = {"LD_PRELOAD": "/other/torch_memory_saver_cpp_cu12.so"}
        note = s07.maybe_reexec_for_memory_saver(
            env,
            [],
            lambda *a: pytest.fail("kein exec noetig"),
            path_fn=lambda: "/x.so",
            libdir_fn=lambda: "/lib",
        )
        assert note == "inherited"

    def test_unresolvable_hook_is_reported_not_raised(self):
        """No memory saver = two routes untested; that is a STOP downstream,
        not a crash here."""
        s07 = self._probe()

        def boom():
            raise RuntimeError("kein Wheel")

        note = s07.maybe_reexec_for_memory_saver(
            {},
            [],
            lambda *a: pytest.fail("kein exec"),
            path_fn=boom,
            libdir_fn=lambda: "/lib",
        )
        assert note.startswith("unavailable: RuntimeError")

    def test_unresolvable_libcudart_does_not_preload(self):
        """A preload whose libcudart cannot be resolved does not fail the
        probe, it fails the interpreter (rc 127, no artifact at all)."""
        s07 = self._probe()
        env = {}
        note = s07.maybe_reexec_for_memory_saver(
            env,
            [],
            lambda *a: pytest.fail("kein exec ohne libcudart"),
            path_fn=lambda: "/pkg/hook.so",
            libdir_fn=lambda: None,
        )
        assert note.startswith("unavailable: libcudart")
        assert "LD_PRELOAD" not in env

    def test_libcudart_dir_comes_from_the_live_process_map(self, tmp_path):
        """This venv maps a cu12 AND a cu13 runtime; the hook is built for
        one major, so the directory is picked by that major, not by order."""
        s07 = self._probe()
        maps = tmp_path / "maps"
        maps.write_text(
            "7f00-7f01 r-xp 0 00:1f 1 /venv/nvidia/cuda_runtime/lib/libcudart.so.12\n"
            "7f02-7f03 r-xp 0 00:1f 2 /venv/nvidia/cu13/lib/libcudart.so.13\n"
        )
        torch = pytest.importorskip("torch")
        major = str(getattr(torch.version, "cuda", "") or "").split(".")[0]
        if major not in ("12", "13"):
            pytest.skip(f"CUDA-Major {major!r} nicht in der Fixture")
        expected = {
            "12": "/venv/nvidia/cuda_runtime/lib",
            "13": "/venv/nvidia/cu13/lib",
        }[major]
        assert s07.cuda_runtime_lib_dir(str(maps)) == expected


# ---------------------------------------------------------------------------
# s08 dispatcher tables
# ---------------------------------------------------------------------------


def _dispatcher(**over):
    payload = {
        "kind": "dispatcher_tables",
        "schema_version": 1,
        "sources": {
            "capability_matrix": {"path": "/x/capability_matrix.json", "exists": True},
            "d2d_bench": {"path": "/x/d2d_bench.json", "exists": True},
            "nccl_reference": {"path": "/x/nccl_reference.json", "exists": True},
            "gdr_matrix": {"path": None, "exists": False},
        },
        "profiles": {"measured": 12, "placeholder": 1, "measured_names": []},
        "peer_capable_pairs": 2,
        "apertures": {f"{PCI_A}->{PCI_B}": 240 * MIB},
        "errors": [],
        "skipped": [],
        "decisions": [
            {
                "message_class": "battery_contaminated",
                "nbytes": 1 * MIB,
                "protected": False,
                "path": "status_quo",
                "status_quo": True,
                "overflowed": False,
                "reason": "placeholder",
            },
            {
                "message_class": "battery_contaminated",
                "nbytes": 1 * MIB,
                "protected": True,
                "path": "status_quo",
                "status_quo": True,
                "overflowed": False,
                "reason": "placeholder",
            },
            {
                "message_class": "battery_measured",
                "nbytes": 1 * MIB,
                "protected": False,
                "path": "d2d_direct:a->b",
                "status_quo": False,
                "overflowed": False,
                "reason": "cheapest",
            },
        ],
        "neutrality_violations": [],
        "sensor_consulted_under_placeholder": False,
        "latency_consulted_under_placeholder": False,
        "measured_class_decided_paths": ["d2d_direct:a->b"],
    }
    payload.update(over)
    return payload


class TestDispatcherTablesCheck:
    CHECK, STEP = "check_s08_dispatcher_tables.py", "s08_dispatcher_tables"

    def test_pass(self, tmp_path):
        write_json(tmp_path / "dispatcher_tables.json", _dispatcher())
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_source_is_stop(self, tmp_path):
        payload = _dispatcher()
        payload["sources"]["nccl_reference"]["exists"] = False
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_silent_placeholder_degradation_is_fail(self, tmp_path):
        """The failure this step exists for: everything loaded as placeholder
        looks exactly like a successful run with no cards."""
        payload = _dispatcher()
        payload["profiles"]["measured"] = 0
        write_json(tmp_path / "dispatcher_tables.json", payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Platzhalter" in line

    def test_no_apertures_despite_peer_pairs_is_fail(self, tmp_path):
        payload = _dispatcher(apertures={})
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_no_apertures_on_a_rig_without_p2p_passes(self, tmp_path):
        """No peer-capable pair means no aperture, and that is the correct
        result rather than a gap."""
        payload = _dispatcher(apertures={}, peer_capable_pairs=0)
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_neutrality_violation_is_fail(self, tmp_path):
        payload = _dispatcher(
            neutrality_violations=[
                {"violation": "Klasse mit Platzhalter entschied echt"}
            ]
        )
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_contaminated_class_deciding_a_real_path_is_fail(self, tmp_path):
        payload = _dispatcher()
        payload["decisions"][1]["status_quo"] = False
        payload["decisions"][1]["path"] = "d2d_direct:a->b"
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_sensor_consulted_under_placeholder_is_fail(self, tmp_path):
        """Hard rule 1 says the sensor is not consulted at all, not that its
        answer is ignored."""
        payload = _dispatcher(sensor_consulted_under_placeholder=True)
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_latency_consulted_under_placeholder_is_fail(self, tmp_path):
        payload = _dispatcher(latency_consulted_under_placeholder=True)
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_tables_loaded_but_never_used_is_fail(self, tmp_path):
        payload = _dispatcher(measured_class_decided_paths=[])
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_loader_errors_are_fail(self, tmp_path):
        payload = _dispatcher(errors=["rows[3]: missing fields"])
        write_json(tmp_path / "dispatcher_tables.json", payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# s09 sensor smoke
# ---------------------------------------------------------------------------


def _smoke(**over):
    payload = {
        "kind": "sensor_smoke",
        "schema_version": 1,
        "flags": {
            "gdn_state_set_ladder": [4, 2, 1],
            "gdn_state_set_ladder_hysteresis": 2,
            "kv_pressure_ladder": "relief:dcp_ratio",
            "kv_pressure_pre_stage": True,
        },
        "token_capacity": 65536,
        "generation_identical": True,
        "generation_nonempty": True,
        "samples": 60,
        "occupancy_max": 0.41,
        "reading": {
            "samples": 60,
            "occupancy": 0.41,
            "trend_tokens_per_round": 33.0,
            "rounds_to_exhaustion": 1170.0,
            "verdict": "hold",
            "stage_verdict": "hold",
            "reason": "unter der Marke",
        },
        "reading_deterministic": True,
    }
    payload.update(over)
    return payload


def _write_smoke(tmp_path, payload=None, log="ok\n"):
    write_json(
        tmp_path / "sensor_smoke.json", payload if payload is not None else _smoke()
    )
    (tmp_path / "server.log").write_text(log)
    return tmp_path


class TestSensorSmokeProducer:
    """The probe against a REAL /get_server_info answer.

    server_info() splices ServerArgs and the scheduler snapshot together, and
    memory_usage sits under internal_states -- not at the top level. Reading
    the top level returned {} on every healthy server, so the step wrote
    "keine token_capacity" and returned 1 no matter how the server behaved:
    unpassable by construction, and invisible to a fixture that put the key
    where the reader expected it."""

    def _real_info(self):
        with open(os.path.join(FIXTURES, "s09_sensor_smoke", "server_info.json")) as f:
            return json.load(f)

    def _memory_usage(self):
        sys.path.insert(0, BATTERY)
        from s09_sensor_smoke import _memory_usage

        return _memory_usage

    def test_capacity_from_real_server_info(self):
        usage = self._memory_usage()(self._real_info())
        assert usage["token_capacity"] == 453632

    def test_top_level_still_read_as_fallback(self):
        usage = self._memory_usage()({"memory_usage": {"token_capacity": 7}})
        assert usage["token_capacity"] == 7

    def test_no_memory_block_stays_empty(self):
        assert self._memory_usage()({"internal_states": [{}]}) == {}


class TestSensorSmokeCheck:
    CHECK, STEP = "check_s09_sensor_smoke.py", "s09_sensor_smoke"

    def test_pass(self, tmp_path):
        _write_smoke(tmp_path)
        assert_pass(self.CHECK, tmp_path, self.STEP)

    def test_missing_artifact_without_log_is_stop(self, tmp_path):
        assert_stop(self.CHECK, tmp_path, self.STEP)

    def test_server_never_came_up_is_fail(self, tmp_path):
        (tmp_path / "server.log").write_text("loading...\n")
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_flag_lost_on_the_way_to_the_scheduler_is_fail(self, tmp_path):
        payload = _smoke()
        payload["flags"]["gdn_state_set_ladder"] = None
        _write_smoke(tmp_path, payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_pre_stage_flag_not_honoured_is_fail(self, tmp_path):
        payload = _smoke()
        payload["flags"]["kv_pressure_pre_stage"] = False
        _write_smoke(tmp_path, payload)
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_nondeterministic_generation_is_fail(self, tmp_path):
        """The ladders are supposed to be inert until something moves."""
        _write_smoke(tmp_path, _smoke(generation_identical=False))
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "inert" in line

    def test_flat_zero_occupancy_is_fail(self, tmp_path):
        """A zero line means the load never reached the pool and the sensor
        was fed nothing."""
        _write_smoke(tmp_path, _smoke(occupancy_max=0.0))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_too_few_samples_is_fail(self, tmp_path):
        _write_smoke(tmp_path, _smoke(samples=4))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_missing_trend_is_fail(self, tmp_path):
        payload = _smoke()
        payload["reading"]["trend_tokens_per_round"] = None
        _write_smoke(tmp_path, payload)
        line = assert_fail(self.CHECK, tmp_path, self.STEP)
        assert "Trend" in line

    def test_nondeterministic_reading_is_fail(self, tmp_path):
        _write_smoke(tmp_path, _smoke(reading_deterministic=False))
        assert_fail(self.CHECK, tmp_path, self.STEP)

    def test_probe_error_is_fail(self, tmp_path):
        _write_smoke(
            tmp_path, _smoke(error="get_server_info meldet keine token_capacity")
        )
        assert_fail(self.CHECK, tmp_path, self.STEP)


# ---------------------------------------------------------------------------
# the verdict contract itself
# ---------------------------------------------------------------------------


class TestVerdictContract:
    """The executor parses one line. These are the properties it relies on."""

    @pytest.mark.parametrize("step", STEPS, ids=[s.step_id for s in STEPS])
    def test_empty_step_dir_never_crashes_and_never_passes(self, step, tmp_path):
        rc, line = run_check(step.check, tmp_path)
        assert rc in (1, 2), f"{step.check} auf leerem Verzeichnis: rc={rc} / {line}"
        assert line.startswith(("BATTERY-FAIL ", "BATTERY-STOP ")), line

    @pytest.mark.parametrize("step", STEPS, ids=[s.step_id for s in STEPS])
    def test_garbage_artifact_is_a_verdict_not_a_traceback(self, step, tmp_path):
        """A check that crashes has judged nothing, and the executor must not
        be handed a traceback to interpret."""
        for name in (
            "preflight.json",
            "accept.json",
            "reseed.json",
            "nccl_reference.json",
            "offload_register_gpu.json",
            "dispatcher_tables.json",
            "sensor_smoke.json",
        ):
            (tmp_path / name).write_text("{not json at all")
        os.makedirs(tmp_path / "results", exist_ok=True)
        for name in ("capability_matrix.json", "d2d_bench.json", "nccl_transport.json"):
            (tmp_path / "results" / name).write_text("{not json at all")
        rc, line = run_check(step.check, tmp_path)
        assert rc in (1, 2), line
        assert line.startswith(("BATTERY-FAIL ", "BATTERY-STOP ")), line

    def test_reason_is_always_a_single_line(self, tmp_path):
        sys.path.insert(0, CHECKS)
        from check_common import CheckFail, run_check as rc_helper  # noqa: E402

        def raiser():
            raise CheckFail("erste Zeile\nzweite Zeile\n\ndritte")

        assert rc_helper("s99_probe", raiser) == 1


class TestFatalScanSkipsQuotedSubprocessLogs:
    """A boot is judged by its OWN lines (#303 part 4).

    An sglang server QUOTES the log of helper subprocesses it ran and recovered
    from -- above all the stage-0 hardware probe, whose failure is caught,
    named and worked around. When that probe is killed (which the #289 evidence
    run did on purpose, to stop it burning 600 s of card time), the server logs
    a warning with the probe's full traceback and then serves normally. The
    fatal scan used to see the quoted ``Traceback`` and score the boot as dead.

    What must still be caught, unchanged: a traceback on the server's own
    lines, and every other fatal marker.
    """

    KILLED_PROBE_LOG = (
        "[2026-07-30 19:29:24] auto-performance: running the stage-0 hardware "
        "micro-probe (no cached profile)...\n"
        "[2026-07-30 19:30:25] auto-performance: hardware probe failed (rc=1); "
        "the quoted lines below are the PROBE's log, not this server's:\n"
        "[probe-subprocess] Traceback (most recent call last):\n"
        '[probe-subprocess]   File "uneven_perf.py", line 983, in run_probe\n'
        "[probe-subprocess]     mp.spawn(_link_worker, ...)\n"
        "[probe-subprocess] ProcessExitedException: process 0 terminated with "
        "signal SIGTERM\n"
        "[2026-07-30 19:30:25] auto-performance: hardware profile unavailable "
        "-- keeping the plain VRAM-auto split.\n"
        "[2026-07-30 19:34:36] The server is fired up and ready to roll!\n"
    )

    # The same event as a log written BEFORE the marker existed: the quoted
    # block is recognised structurally, by the server timestamp that ends it.
    UNMARKED_KILLED_PROBE_LOG = (
        "[2026-07-30 19:30:25] auto-performance: hardware probe failed (rc=1):\n"
        "Traceback (most recent call last):\n"
        '  File "uneven_perf.py", line 983, in run_probe\n'
        "torch.multiprocessing.spawn.ProcessExitedException: process 0 "
        "terminated with signal SIGTERM\n"
        "[2026-07-30 19:30:25] Uneven DCP: auto-set dcp_size=3.\n"
        "[2026-07-30 19:34:36] The server is fired up and ready to roll!\n"
    )

    def _scan(self, tmp_path, text):
        sys.path.insert(0, CHECKS)
        from check_common import scan_log_for_fatals  # noqa: E402

        path = tmp_path / "server.log"
        path.write_text(text)
        return scan_log_for_fatals(str(path), "server.log")

    def test_killed_probe_traceback_is_not_a_boot_failure(self, tmp_path):
        assert self._scan(tmp_path, self.KILLED_PROBE_LOG) is None

    def test_unmarked_killed_probe_traceback_is_not_a_boot_failure(self, tmp_path):
        assert self._scan(tmp_path, self.UNMARKED_KILLED_PROBE_LOG) is None

    def test_real_boot_traceback_is_still_a_failure(self, tmp_path):
        log = (
            "[2026-07-30 19:29:24] Load weight begin.\n"
            "Traceback (most recent call last):\n"
            '  File "model_runner.py", line 1, in load_model\n'
            "RuntimeError: no kernel image is available\n"
        )
        found = self._scan(tmp_path, log)
        assert found is not None
        assert "Traceback" in found

    def test_a_real_fatal_after_a_quoted_block_is_still_found(self, tmp_path):
        log = self.KILLED_PROBE_LOG + (
            "[2026-07-30 19:35:00] Scheduler hit an exception\n"
            "torch.OutOfMemoryError: CUDA out of memory.\n"
        )
        found = self._scan(tmp_path, log)
        assert found is not None
        assert "out of memory" in found

    def test_oom_inside_a_quoted_probe_log_is_not_a_boot_failure(self, tmp_path):
        log = (
            "[2026-07-30 19:30:25] auto-performance: hardware probe failed "
            "(rc=1); the quoted lines below are the PROBE's log:\n"
            "[probe-subprocess] torch.OutOfMemoryError: CUDA out of memory.\n"
            "[2026-07-30 19:34:36] The server is fired up and ready to roll!\n"
        )
        assert self._scan(tmp_path, log) is None

    def test_the_two_prefix_literals_agree(self):
        """check_common and the emitter must name the same marker."""
        sys.path.insert(0, CHECKS)
        from check_common import QUOTED_SUBLOG_PREFIX  # noqa: E402

        from sglang.srt.uneven_perf import QUOTED_SUBLOG_PREFIX as EMITTED

        assert QUOTED_SUBLOG_PREFIX == EMITTED
