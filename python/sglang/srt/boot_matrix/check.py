# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``check_arm`` -- one arm's artifacts to one verdict (#349).

Hermetic and file-only, by the GPU-battery rule: a check never touches a card,
never starts a server, never needs the run alive. It reads what the arm's
runner collected and answers the one question the matrix asks -- did this
crossing stay sound. This is the part the unit tests exercise against synthetic
logs, so the whole judgement is provable on a card-less host.

The verdict vocabulary and the STOP/FAIL distinction are the battery's, for the
same reason (``scripts/gpu_battery/checks/check_common.py``):

* PASS -- the arm ran and the artifact says the crossing is sound. A reject
          arm that refused cleanly, at arg resolution, before weight load, is
          a PASS: that is the guard doing its job.
* FAIL -- the arm ran and the artifact shows a real cross-feature defect: a
          boot that should have come up crashed or hung; a boot that came up
          resolved to a configuration it did not declare (the #340 silent-env
          class); a configuration that must be refused booted anyway; a
          coherence score under the A-vs-A floor.
* STOP -- a precondition is wrong and nothing was learned: an artifact is
          missing, the grader is absent, a declared fact is not in the log, a
          byte probe was mis-designed past the reproducibility window.

THE ARTIFACT CONTRACT. One directory per arm run:
    arm.json     {"name","kind","boot_status", ...} -- boot_status is one of
                 arms.BOOT_STATUSES, written by the runner. The check never
                 infers liveness from an exit code (a hung boot and a clean
                 reject both exit non-zero).
    server.log   the boot log.
    probes.json  boot arms with a coherence tier: a list of probe dicts as
                 :func:`coherence.grade_probes` consumes them.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from sglang.srt.boot_matrix.arms import Arm
from sglang.srt.boot_matrix.coherence import CoherenceResult, grade_probes
from sglang.srt.boot_matrix.effective import (
    EffectiveConfig,
    first_refusal,
    report_effective,
)

PASS = 0
FAIL = 1
STOP = 2

# Mirrors scripts/gpu_battery/checks/check_common.FATAL_LOG_MARKERS. This is
# generic boot-liveness plumbing, not the grader, so it is kept local to make
# the package importable from a wheel (the tenant needs that); the grader is
# the one thing reused cross-tree, per the #349 instruction.
_FATAL_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "NCCL error",
    "Watchdog caught collective operation timeout",
    "Traceback (most recent call last)",
)
_SERVER_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^\]]*\]")
_QUOTED_PREFIX = "[probe-subprocess] "
_QUOTED_OPENERS = ("auto-performance: hardware probe failed",)


def _scan_fatals(log_text: str) -> Optional[str]:
    """First fatal marker with its line number, skipping quoted subprocess
    blocks (a handled probe failure is not the boot's failure -- the #303
    phantom-FAIL lesson). Mirrors check_common.scan_log_for_fatals."""
    in_quoted = False
    for lineno, line in enumerate(log_text.splitlines(), 1):
        if _QUOTED_PREFIX in line:
            continue
        if in_quoted:
            if not _SERVER_STAMP.match(line):
                continue
            in_quoted = False
        if any(opener in line for opener in _QUOTED_OPENERS):
            in_quoted = True
            continue
        for marker in _FATAL_MARKERS:
            if marker in line:
                return f"{lineno}: {' '.join(line.split())[:200]}"
    return None


@dataclass(frozen=True)
class Verdict:
    status: int  # PASS | FAIL | STOP
    arm: str
    reason: str
    effective: Optional[EffectiveConfig] = None
    coherence: Optional[CoherenceResult] = None
    #: The refusal line, for reject arms that passed.
    refusal: Optional[str] = None

    @property
    def label(self) -> str:
        return {PASS: "PASS", FAIL: "FAIL", STOP: "STOP"}[self.status]

    def render(self) -> str:
        head = f"BOOTMATRIX-{self.label} {self.arm}: {self.reason}"
        if self.effective is not None:
            head += f"\n    effective: {self.effective.render()}"
        if self.coherence is not None and self.coherence.grader_available:
            head += f"\n    {self.coherence.render()}"
        return head

    def to_json(self) -> dict:
        return {
            "arm": self.arm,
            "status": self.label,
            "reason": self.reason,
            "effective": self.effective.to_json() if self.effective else None,
            "coherence": self.coherence.to_json() if self.coherence else None,
            "refusal": self.refusal,
        }


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _mismatch_fields(arm: Arm, eff: EffectiveConfig) -> List[str]:
    """Declared vs resolved, field by field. Returns concrete disagreements
    only (a declared field that resolved to a DIFFERENT value). A declared
    field that resolved to None is handled separately as STOP: 'could not
    confirm' is not the same statement as 'resolved to the wrong thing'."""
    out: List[str] = []
    for key, want in arm.expect.items():
        got = getattr(eff, key, None)
        if got is not None and got != want:
            out.append(f"{key}: declared {want!r}, resolved {got!r}")
    return out


def _unconfirmed_fields(arm: Arm, eff: EffectiveConfig) -> List[str]:
    return [
        key
        for key, _ in arm.expect.items()
        if getattr(eff, key, None) is None
    ]


def check_arm(arm: Arm, artifact_dir: str) -> Verdict:
    """Turn one arm's collected artifacts into a verdict. Pure/file-only."""
    arm_json_path = os.path.join(artifact_dir, "arm.json")
    log_path = os.path.join(artifact_dir, "server.log")

    meta_text = _read_text(arm_json_path)
    if meta_text is None:
        return Verdict(STOP, arm.name, f"no arm.json in {artifact_dir}")
    try:
        meta = json.loads(meta_text)
    except json.JSONDecodeError as exc:
        return Verdict(STOP, arm.name, f"arm.json is not JSON: {exc}")
    boot_status = str(meta.get("boot_status", ""))

    log_text = _read_text(log_path)
    if log_text is None:
        return Verdict(STOP, arm.name, f"no server.log in {artifact_dir}")

    if arm.kind == "reject":
        return _check_reject(arm, log_text, boot_status)
    return _check_boot(arm, artifact_dir, log_text, boot_status)


def _check_reject(arm: Arm, log_text: str, boot_status: str) -> Verdict:
    eff = report_effective(log_text)
    if eff.ready:
        return Verdict(
            FAIL,
            arm.name,
            "a configuration that must be refused booted to ready instead",
            effective=eff,
        )
    refusal = first_refusal(log_text, list(arm.reject_markers))
    if refusal is not None:
        return Verdict(
            PASS,
            arm.name,
            "refused cleanly at boot with the named guard",
            refusal=refusal,
        )
    # Did not boot, but the expected guard text is absent.
    fatal = _scan_fatals(log_text)
    if fatal is not None:
        return Verdict(
            FAIL,
            arm.name,
            (
                "refused, but with an unexpected error rather than the named "
                f"guard (expected markers {list(arm.reject_markers)}); "
                f"first fatal at {fatal}"
            ),
        )
    return Verdict(
        STOP,
        arm.name,
        (
            f"neither the ready marker nor the refusal markers "
            f"{list(arm.reject_markers)} were found (boot_status={boot_status!r}); "
            "nothing was learned"
        ),
    )


def _check_boot(
    arm: Arm, artifact_dir: str, log_text: str, boot_status: str
) -> Verdict:
    eff = report_effective(log_text)

    if not eff.ready:
        fatal = _scan_fatals(log_text)
        if fatal is not None:
            return Verdict(
                FAIL,
                arm.name,
                f"boot that should have come up died; first fatal at {fatal}",
                effective=eff,
            )
        if boot_status in ("timeout", "crashed"):
            return Verdict(
                FAIL,
                arm.name,
                (
                    f"boot did not reach ready ({boot_status}) and left no "
                    "fatal marker -- the silent-hang class (#132 x weightless)"
                ),
                effective=eff,
            )
        return Verdict(
            STOP,
            arm.name,
            (
                f"boot did not reach ready (boot_status={boot_status!r}) and "
                "left no fatal marker; treat as environment until re-run"
            ),
            effective=eff,
        )

    # Came up. A fatal AFTER ready (e.g. an NCCL error on the first collective)
    # is still a real defect.
    fatal = _scan_fatals(log_text)
    if fatal is not None:
        return Verdict(
            FAIL,
            arm.name,
            f"came up but logged a fatal at {fatal}",
            effective=eff,
        )

    # The #340 catch: declared effective config vs resolved.
    mism = _mismatch_fields(arm, eff)
    if mism:
        return Verdict(
            FAIL,
            arm.name,
            "resolved a configuration it did not declare -- " + "; ".join(mism),
            effective=eff,
        )
    unconf = _unconfirmed_fields(arm, eff)
    if unconf:
        return Verdict(
            STOP,
            arm.name,
            (
                "could not confirm declared field(s) from the log: "
                + ", ".join(unconf)
                + " -- the report line is incomplete, not the config wrong"
            ),
            effective=eff,
        )

    if arm.coherence == "none":
        return Verdict(
            PASS, arm.name, "booted and resolved as declared", effective=eff
        )

    probes_text = _read_text(os.path.join(artifact_dir, "probes.json"))
    if probes_text is None:
        return Verdict(
            STOP,
            arm.name,
            "coherence tier requested but no probes.json was collected",
            effective=eff,
        )
    try:
        probes = json.loads(probes_text)
    except json.JSONDecodeError as exc:
        return Verdict(STOP, arm.name, f"probes.json is not JSON: {exc}", effective=eff)

    coh = grade_probes(probes)
    if not coh.grader_available:
        return Verdict(
            STOP,
            arm.name,
            "the #274 grader is not in this checkout; coherence not evaluated",
            effective=eff,
            coherence=coh,
        )
    if coh.byte_probe_too_long:
        return Verdict(
            STOP,
            arm.name,
            "a byte probe's reference is past the GDN reproducibility window "
            "-- fix the probe set, do not byte-gate there",
            effective=eff,
            coherence=coh,
        )
    if not coh.passed:
        return Verdict(
            FAIL,
            arm.name,
            "coherence gate red: " + coh.render(),
            effective=eff,
            coherence=coh,
        )
    return Verdict(
        PASS,
        arm.name,
        "booted, resolved as declared, coherence within the A-vs-A band",
        effective=eff,
        coherence=coh,
    )
