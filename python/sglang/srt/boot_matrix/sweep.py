# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The standalone boot-matrix sweep (#349).

The pre-Docker manual sweep: boot each arm, collect its artifacts, check it,
print one verdict line per arm. Runnable as::

    python -m sglang.srt.boot_matrix.sweep --list          # card-less: the plan
    python -m sglang.srt.boot_matrix.sweep --dry-run A_default
    python -m sglang.srt.boot_matrix.sweep --run --model <path> --out <dir>

Only ``--run`` touches a card. ``--list`` and ``--dry-run`` are pure and are
what a reader (or a CI lint) exercises on a card-less host: they print the
composed command and the effective config each arm DECLARES, so the plan is
auditable before an hour of card time is spent on it.

The command COMPOSITION (``build_command``) is a pure function and is unit-
tested. The booting, probing and collection (:func:`run_arm`) is the thin GPU
layer; it reuses the check core so the sweep and the tenant reach identical
verdicts from identical artifacts.

BAND AND REFERENCE. The graded floor and the byte reference are not constants:
they are MEASURED from the baseline arm, A-vs-A, exactly as the #103 noise
floor and the #274 band are. ``A_default`` is booted first and its probes are
run twice; the byte reference is its first-run text and the graded floor is the
min score across the two runs. Every later arm is graded against that measured
band, never against a guessed number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.boot_matrix.arms import ARMS, BASE_ENV, BASE_FLAGS, Arm, arm_by_name
from sglang.srt.boot_matrix.check import Verdict, check_arm
from sglang.srt.boot_matrix.effective import READY_MARKER

# ---------------------------------------------------------------------------
# Standard probes. Byte tier stays SHORT (inside the GDN reproducibility
# window); graded tier is the two #274 forced continuations the house grader
# scores. Prompts are forced continuations so the right answer is mechanical.
# ---------------------------------------------------------------------------
PROBE_PROMPTS: Tuple[Dict[str, object], ...] = (
    {
        "name": "byte_count",
        "tier": "byte",
        "prompt": "Count up by ones, one number per line, no words:\n1\n2\n3\n",
        "max_new_tokens": 24,
    },
    {
        "name": "alphabet",
        "tier": "graded",
        "prompt": "Continue the sequence, one letter per line:\nt\nu\nv\n",
        "max_new_tokens": 32,
    },
    {
        "name": "squares",
        "tier": "graded",
        "prompt": "Continue: each line is n then n squared.\n10 100\n11 121\n",
        "max_new_tokens": 64,
    },
)


def build_command(
    arm: Arm, *, model_path: str, port: int
) -> Tuple[Dict[str, str], List[str]]:
    """Compose (env, argv) for one arm. Pure function -- the unit-tested core.

    Base recipe first, arm env/flags ADDED on top; a later flag wins under
    argparse, which is how an arm overrides e.g. ``--speculative-algorithm``
    (arm H turns spec off). The env is the process environment the arm needs;
    the argv is the launch line.
    """
    env: Dict[str, str] = dict(BASE_ENV)
    env.update(arm.env)

    argv: List[str] = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    argv.extend(BASE_FLAGS)
    argv.extend(arm.flags)
    return env, argv


@dataclass
class _CollectedArm:
    boot_status: str
    log_text: str
    probes: List[Dict[str, object]]


def _wait_for_boot(
    log_path: str, *, timeout_s: float, reject_markers: Sequence[str] = ()
) -> str:
    """Poll the log until ready, a refusal, a fatal, or the timeout. Returns a
    boot_status from arms.BOOT_STATUSES. GPU-path helper -- not unit-tested."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with open(log_path, errors="replace") as f:
                text = f.read()
        except OSError:
            text = ""
        if READY_MARKER in text:
            return "ready"
        if reject_markers and all(m in text for m in reject_markers):
            return "refused"
        if "ValueError" in text and reject_markers:
            return "refused"
        if "Traceback (most recent call last)" in text and not reject_markers:
            return "crashed"
        time.sleep(2.0)
    return "timeout"


def run_arm(
    arm: Arm,
    *,
    model_path: str,
    out_dir: str,
    port: int,
    reference_probes: Optional[Mapping[str, Dict[str, object]]] = None,
    band: Optional[Mapping[str, int]] = None,
) -> Verdict:
    """Boot one arm, collect artifacts, check it. The GPU layer.

    Deliberately thin: it composes the command, launches under setsid, waits,
    runs the probes for a boot arm, writes arm.json / server.log / probes.json,
    and hands the directory to :func:`check_arm`. Every judgement lives in the
    check, so this function has no verdict logic of its own -- the sweep and
    the tenant cannot disagree.
    """
    import subprocess

    arm_dir = os.path.join(out_dir, arm.name)
    os.makedirs(arm_dir, exist_ok=True)
    log_path = os.path.join(arm_dir, "server.log")
    env, argv = build_command(arm, model_path=model_path, port=port)

    proc_env = dict(os.environ)
    proc_env.update(env)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            argv, stdout=logf, stderr=subprocess.STDOUT, env=proc_env,
            start_new_session=True,
        )
    try:
        boot_status = _wait_for_boot(
            log_path,
            timeout_s=arm.expected_seconds + 300.0,
            reject_markers=arm.reject_markers,
        )
        probes: List[Dict[str, object]] = []
        if arm.kind == "boot" and boot_status == "ready" and arm.coherence != "none":
            probes = _run_probes(
                port=port,
                arm=arm,
                reference_probes=reference_probes or {},
                band=band or {},
            )
    finally:
        _terminate(proc)

    with open(log_path, errors="replace") as f:
        log_text = f.read()
    _write_artifacts(arm_dir, arm, boot_status, probes)
    return check_arm(arm, arm_dir)


def _run_probes(
    *,
    port: int,
    arm: Arm,
    reference_probes: Mapping[str, Dict[str, object]],
    band: Mapping[str, int],
) -> List[Dict[str, object]]:
    """Issue the standard probes and assemble probe dicts for grade_probes."""
    import requests  # local: the sweep's GPU path only

    out: List[Dict[str, object]] = []
    for spec in PROBE_PROMPTS:
        tier = str(spec["tier"])
        if arm.coherence == "graded_only" and tier == "byte":
            continue
        try:
            resp = requests.post(
                f"http://127.0.0.1:{port}/generate",
                json={
                    "text": spec["prompt"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": int(spec["max_new_tokens"]),
                    },
                },
                timeout=120,
            )
            text = resp.json().get("text", "")
        except Exception as exc:  # noqa: BLE001 - a probe failure is a probe fact
            text = ""
            spec = {**spec, "probe_error": str(exc)}
        name = str(spec["name"])
        probe: Dict[str, object] = {"name": name, "tier": tier, "text": text}
        if tier == "byte":
            ref = reference_probes.get(name, {})
            probe["ref_text"] = ref.get("text", "")
        else:
            probe["min_score"] = int(band.get(name, 0))
        out.append(probe)
    return out


def _write_artifacts(
    arm_dir: str, arm: Arm, boot_status: str, probes: List[Dict[str, object]]
) -> None:
    with open(os.path.join(arm_dir, "arm.json"), "w") as f:
        json.dump(
            {
                "name": arm.name,
                "kind": arm.kind,
                "axis": arm.axis,
                "boot_status": boot_status,
                "expected_seconds": arm.expected_seconds,
                "capture_note": arm.capture_note,
            },
            f,
            indent=2,
        )
    if probes:
        with open(os.path.join(arm_dir, "probes.json"), "w") as f:
            json.dump(probes, f, indent=2)


def _terminate(proc) -> None:
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for _ in range(15):
            if proc.poll() is not None:
                return
            time.sleep(1.0)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


# ---------------------------------------------------------------------------
# card-less surfaces
# ---------------------------------------------------------------------------
def render_plan() -> str:
    """The whole matrix as a table, plus the card-time estimate. Pure."""
    lines = ["boot-matrix arms (#349):", ""]
    total = 0.0
    for arm in ARMS:
        total += arm.expected_seconds
        kind = "REJECT" if arm.kind == "reject" else "boot  "
        note = f"  [{arm.capture_note.split(':')[0]}]" if arm.capture_note else ""
        lines.append(
            f"  {arm.name:24s} {kind} ~{int(arm.expected_seconds):4d}s  "
            f"{arm.axis}{note}"
        )
    lines.append("")
    lines.append(
        f"total estimated card time: {total/60.0:.0f} min over {len(ARMS)} arms "
        f"({sum(1 for a in ARMS if a.kind=='boot')} boot, "
        f"{sum(1 for a in ARMS if a.kind=='reject')} reject)."
    )
    lines.append(
        "reject arms are ~60 s each (arg-resolution refusal, no weight load); "
        "the boot arms dominate. K_bar1_graphs carries a cold-graph-cache "
        "capture caveat -- see its capture_note."
    )
    return "\n".join(lines)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Integration boot matrix sweep (#349)")
    p.add_argument("--list", action="store_true", help="print the plan and exit")
    p.add_argument(
        "--dry-run", metavar="ARM", help="print one arm's composed command and exit"
    )
    p.add_argument("--run", action="store_true", help="boot the matrix (needs cards)")
    p.add_argument("--model", default=os.environ.get("BOOT_MATRIX_MODEL", ""))
    p.add_argument("--out", default="/tmp/boot_matrix")
    p.add_argument("--port", type=int, default=31349)
    p.add_argument("--only", default="", help="comma-separated arm names to run")
    args = p.parse_args(argv)

    if args.list or (not args.dry_run and not args.run):
        print(render_plan())
        return 0

    if args.dry_run:
        arm = arm_by_name(args.dry_run)
        env, cmd = build_command(arm, model_path=args.model or "<MODEL>", port=args.port)
        print(f"# arm {arm.name} ({arm.kind}) -- {arm.axis}")
        print("# env:", " ".join(f"{k}={v}" for k, v in env.items()))
        print(" ".join(cmd))
        return 0

    if not args.model:
        print("--run needs --model (or BOOT_MATRIX_MODEL)", file=sys.stderr)
        return 2

    only = {n.strip() for n in args.only.split(",") if n.strip()}
    selected = [a for a in ARMS if not only or a.name in only]
    os.makedirs(args.out, exist_ok=True)
    verdicts: List[Verdict] = []
    for arm in selected:
        v = run_arm(arm, model_path=args.model, out_dir=args.out, port=args.port)
        print(v.render())
        verdicts.append(v)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump([v.to_json() for v in verdicts], f, indent=2)
    return 0 if all(v.status == 0 for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
