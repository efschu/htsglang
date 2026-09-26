# SPDX-License-Identifier: Apache-2.0
"""Docker plan, stage B (2026-09-24): the weg2 rig paths follow SGLANG_WEG2_*.

The launcher and two of its modules hard-code this rig's directories
(evidence tree, gpu-arb, HiCache L3 store, venv, PP calibration, corridor
sample) and the TMS preload build script its venv and output directory. A
container image has to point them elsewhere, and most of them have no CLI flag.

THE DANGER DIRECTION is the rig itself: with no variable set, every value must
be BYTE-IDENTICAL to the literal it replaced -- a boot on this rig may not
notice this commit at all. That is pinned first, against the literals spelled
out here (not read back from the module), and an EMPTY variable must behave
exactly like an unset one (an empty path would otherwise become "" and turn
`f"{GPU_ARB}/devtools/..."` into an absolute path under /).

Each probe imports the modules in a FRESH interpreter: the values are module
globals computed at import, so an in-process reload would test the reload.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

TREE = pathlib.Path(__file__).resolve().parents[4]
PYTHON_DIR = TREE / "python"
TMS_SCRIPT = TREE / "scripts" / "weg2" / "tms" / "build_tms_preload.sh"
#: the operator subdirectory under the gpu-arb root is a HOST path and keeps its name through the
#: rename (compat_shims.operator_dir); written split so the mechanical pass leaves it alone
_OP = "we" "g2"

#: The literals as they stood before stage B (launcher.py, host_ledger.py,
#: corridor_budget.py at 5a3f533f0f).
RIG_DEFAULTS = {
    "EVIDENCE_DIR": "/spinning/evidence-665-f1",
    "GPU_ARB": "/spinning/gpu-arb",
    "DUPLEX_PROBE_DEFAULT": "/spinning/gpu-arb/weg2/PROBE_RING_0907.md",
    "DEADMAN": "/spinning/gpu-arb/devtools/boot_deadman.sh",
    "MEMTS": "/spinning/gpu-arb/devtools/mem_timeseries.sh",
    "HOST_PREFLIGHT": "/spinning/gpu-arb/devtools/host_ledger_preflight.sh",
    "STORE_ROOT": "/spinning/hicache-weg2",
    "SHM_ARCHIVE_ROOT": "/spinning/gpu-arb/shm_residue",
    "VENV_DEFAULT": "/spinning/htsglang-gpu/.venv",
    "CALIB_DIR": "/spinning/gpu-arb/weg2/calib",
    "DEFAULT_SAMPLE_PATH": "/spinning/gpu-arb/weg2/corridor_budget_sample.json",
}

_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from sglang.srt.weg2 import corridor_budget, host_ledger, launcher
names = ("EVIDENCE_DIR", "GPU_ARB", "DUPLEX_PROBE_DEFAULT", "DEADMAN", "MEMTS",
         "HOST_PREFLIGHT", "STORE_ROOT", "SHM_ARCHIVE_ROOT", "VENV_DEFAULT")
out = {n: getattr(launcher, n) for n in names}
out["CALIB_DIR"] = host_ledger.CALIB_DIR
out["DEFAULT_SAMPLE_PATH"] = corridor_budget.DEFAULT_SAMPLE_PATH
print(json.dumps(out))
"""

_PATH_VARS = (
    "SGLANG_WEG2_EVIDENCE_DIR",
    "SGLANG_WEG2_GPU_ARB",
    "SGLANG_WEG2_DEVTOOLS_DIR",
    "SGLANG_WEG2_STORE_ROOT",
    "SGLANG_WEG2_VENV",
    "SGLANG_WEG2_TMS_OUT_DIR",
)


def _clean_env(**over):
    env = {k: v for k, v in os.environ.items() if k not in _PATH_VARS}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(over)
    return env


def _probe(**over) -> dict:
    r = subprocess.run(
        [sys.executable, "-c", _PROBE, str(PYTHON_DIR)],
        env=_clean_env(**over), capture_output=True, text=True, timeout=900,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_no_variable_is_byte_identical_to_the_rig_literals():
    assert _probe() == RIG_DEFAULTS


def test_empty_variables_count_as_unset():
    assert _probe(**{v: "" for v in _PATH_VARS}) == RIG_DEFAULTS


def test_every_path_follows_its_variable():
    got = _probe(
        SGLANG_WEG2_EVIDENCE_DIR="/img/evidence",
        SGLANG_WEG2_GPU_ARB="/img/arb",
        SGLANG_WEG2_DEVTOOLS_DIR="/opt/tools",
        SGLANG_WEG2_STORE_ROOT="/img/store",
        SGLANG_WEG2_VENV="/opt/venv",
    )
    assert got == {
        "EVIDENCE_DIR": "/img/evidence",
        "GPU_ARB": "/img/arb",
        "DUPLEX_PROBE_DEFAULT": "/img/arb/" + _OP + "/PROBE_RING_0907.md",
        "DEADMAN": "/opt/tools/boot_deadman.sh",
        "MEMTS": "/opt/tools/mem_timeseries.sh",
        "HOST_PREFLIGHT": "/opt/tools/host_ledger_preflight.sh",
        "STORE_ROOT": "/img/store",
        "SHM_ARCHIVE_ROOT": "/img/arb/shm_residue",
        "VENV_DEFAULT": "/opt/venv",
        "CALIB_DIR": "/img/arb/" + _OP + "/calib",
        "DEFAULT_SAMPLE_PATH": "/img/arb/" + _OP + "/corridor_budget_sample.json",
    }


def test_devtools_default_follows_gpu_arb():
    got = _probe(SGLANG_WEG2_GPU_ARB="/img/arb")
    assert got["DEADMAN"] == "/img/arb/devtools/boot_deadman.sh"
    assert got["HOST_PREFLIGHT"] == "/img/arb/devtools/host_ledger_preflight.sh"
    # Untouched variables keep the rig literal.
    assert got["EVIDENCE_DIR"] == RIG_DEFAULTS["EVIDENCE_DIR"]
    assert got["STORE_ROOT"] == RIG_DEFAULTS["STORE_ROOT"]
    assert got["VENV_DEFAULT"] == RIG_DEFAULTS["VENV_DEFAULT"]


def _tms_defaults(**over) -> tuple:
    """VENV/OUT_DIR exactly as the script's own two assignment lines resolve them."""
    lines = [
        ln for ln in TMS_SCRIPT.read_text().splitlines()
        if ln.startswith("VENV=") or ln.startswith("OUT_DIR=")
    ]
    assert len(lines) == 2, lines
    prog = "\n".join(lines) + '\nprintf "%s\\n%s\\n" "$VENV" "$OUT_DIR"\n'
    r = subprocess.run(["bash", "-c", prog], env=_clean_env(**over),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    venv, out_dir = r.stdout.splitlines()
    return venv, out_dir


def test_tms_script_defaults_unchanged_without_variables():
    assert _tms_defaults() == ("/spinning/htsglang-gpu/.venv", "/spinning/gpu-arb/weg2/tms")
    assert _tms_defaults(SGLANG_WEG2_VENV="", SGLANG_WEG2_TMS_OUT_DIR="",
                         SGLANG_WEG2_GPU_ARB="") == (
        "/spinning/htsglang-gpu/.venv", "/spinning/gpu-arb/weg2/tms")


def test_tms_script_follows_variables():
    assert _tms_defaults(SGLANG_WEG2_VENV="/opt/venv", SGLANG_WEG2_GPU_ARB="/img/arb") == (
        "/opt/venv", "/img/arb/" + _OP + "/tms")
    assert _tms_defaults(SGLANG_WEG2_GPU_ARB="/img/arb",
                         SGLANG_WEG2_TMS_OUT_DIR="/opt/tms")[1] == "/opt/tms"
