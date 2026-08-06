"""Fast, fixed-order evidence collector for GPU-rank wedge incidents.

When ranks wedge (deadlocked in NCCL / Bar1 spin-kernels), the spin loops
abort after ~3 minutes and the processes die.  This module races to collect
the signals that are still alive on disk / in process before they vanish.

Usage:
    python -m sglang.srt.debug_utils.wedge_triage --log run.log --out triage_$$
    python -m sglang.srt.debug_utils.wedge_triage --log run.log --out triage_$$ --pids 1234 5678
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import time
from typing import Optional


# -- Markers we grep for ----------------------------------------------------

_MARKERS = [
    "index out of bounds",
    "Bar1CollectiveAborted",
    "Health check failed",
    "abort flag snapshot",
    "collective history (rank",
    "collective census (rank",
]


# -- Helpers ----------------------------------------------------------------


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Return (returncode, stdout, stderr).  Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as exc:
        return -1, "", f"command not found: {exc}"
    except subprocess.TimeoutExpired:
        return -2, "", f"timeout after {timeout}s"
    except Exception as exc:
        return -3, "", f"unexpected error: {exc}"


def _discover_pids() -> list[int]:
    """Find sglang.launch_server PIDs via pgrep.  Returns empty list on any error."""
    rc, out, _err = _run(["pgrep", "-f", "sglang.launch_server"])
    if rc != 0:
        return []
    pids: list[int] = []
    for line in out.strip().splitlines():
        line = line.strip()
        if line:
            try:
                pids.append(int(line))
            except ValueError:
                pass
    return pids


# -- Core collector ---------------------------------------------------------


def collect(
    log_path: str,
    out_dir: str,
    pids: Optional[list[int]] = None,
    timeout_s: int = 40,
) -> dict:
    """Collect wedge evidence into *out_dir* and return a summary dict.

    Returns a dict with keys:
        timestamp,  files,  markers,  nvidia_smi,  py_spy,  errors
    """
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.datetime.now().isoformat()
    ts_file = os.path.join(out_dir, "timestamp.txt")
    with open(ts_file, "w") as fh:
        fh.write(ts + "\n")

    files: list[str] = [ts_file]
    errors: list[str] = []
    marker_counts: dict[str, int] = {m: 0 for m in _MARKERS}

    # -- 1. nvidia-smi snapshot ---------------------------------------------
    nsmi_path = os.path.join(out_dir, "nvidia_smi.csv")
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,power.draw,memory.used",
            "--format=csv,noheader",
        ],
        timeout=timeout_s,
    )
    try:
        with open(nsmi_path, "w") as fh:
            if rc == 0:
                fh.write(out)
            else:
                msg = f"nvidia-smi failed (rc={rc}): {err.strip()}"
                errors.append(msg)
                fh.write(msg + "\n")
    except OSError as exc:
        msg = f"write nvidia_smi.csv: {exc}"
        errors.append(msg)
    files.append(nsmi_path)

    # -- 2. py-spy dumps ----------------------------------------------------
    effective_pids: list[int] = []
    if pids is not None:
        effective_pids = list(pids)
    else:
        effective_pids = _discover_pids()

    py_spy_files: list[str] = []
    for pid in effective_pids:
        dump_name = f"py_spy_{pid}.png"
        dump_path = os.path.join(out_dir, dump_name)
        rc, _, err = _run(
            ["py-spy", "dump", "-o", dump_path, "--pid", str(pid)],
            timeout=timeout_s,
        )
        if rc == 0:
            py_spy_files.append(dump_path)
            files.append(dump_path)
        else:
            try:
                with open(dump_path + ".err", "w") as fh:
                    fh.write(f"py-spy dump pid={pid} rc={rc}: {err.strip()}\n")
                files.append(dump_path + ".err")
            except OSError:
                pass
            errors.append(f"py-spy pid={pid} rc={rc}: {err.strip()}")

    # -- 3. Log markers -----------------------------------------------------
    marker_path = os.path.join(out_dir, "log_markers.txt")
    try:
        with open(log_path, "r", errors="replace") as log_fh:
            log_lines = log_fh.readlines()
    except FileNotFoundError:
        log_lines = []
        errors.append(f"log file not found: {log_path}")
    except OSError as exc:
        log_lines = []
        errors.append(f"read log: {exc}")

    matched: dict[str, list[str]] = {m: [] for m in _MARKERS}
    for line in log_lines:
        lower_line = line.lower()
        for marker in _MARKERS:
            if marker.lower() in lower_line:
                matched[marker].append(line.rstrip("\n"))

    with open(marker_path, "w") as fh:
        fh.write(f"# Markers found in {log_path}\n")
        fh.write(f"# Timestamp: {ts}\n\n")
        any_match = False
        for marker in _MARKERS:
            lines = matched[marker]
            if lines:
                any_match = True
            fh.write(f"## {marker}  (count={len(lines)})\n")
            for ml in lines:
                fh.write(ml + "\n")
            fh.write("\n")
        if not any_match:
            fh.write("# No markers found.\n")

    files.append(marker_path)
    for marker in _MARKERS:
        marker_counts[marker] = len(matched[marker])

    return {
        "timestamp": ts,
        "files": files,
        "markers": marker_counts,
        "nvidia_smi": nsmi_path,
        "py_spy": py_spy_files,
        "errors": errors,
    }


# -- CLI --------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect evidence for a GPU-rank wedge incident."
    )
    parser.add_argument("--log", required=True, help="Path to the server log file.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--pids",
        nargs="*",
        type=int,
        default=None,
        help="PIDs to py-spy dump (auto-discovered when omitted).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="Per-command timeout in seconds (default 40).",
    )
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    result = collect(args.log, args.out, pids=args.pids, timeout_s=args.timeout)
    elapsed = time.monotonic() - t0

    print(f"wedge_triage  elapsed={elapsed:.1f}s  out={args.out}")
    print(f"  files collected: {len(result['files'])}")
    total_markers = sum(result["markers"].values())
    print(f"  log markers     : {total_markers}")
    if result["py_spy"]:
        print(f"  py-spy dumps    : {len(result['py_spy'])}")
    else:
        print("  py-spy dumps    : 0")
    if result["errors"]:
        print(f"  errors          : {len(result['errors'])}")
        for e in result["errors"]:
            print(f"    - {e}")


if __name__ == "__main__":
    main()
