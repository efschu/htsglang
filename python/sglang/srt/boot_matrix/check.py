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
* VOID -- the arm RAN correctly and proved nothing, because its own subject
          never occurred. STOP is about the harness (an artifact is missing);
          VOID is about the LOAD (the artifacts are complete, the boot resolved
          as declared, the output is coherent -- and the mechanism under test
          never fired). The two must not be merged: STOP says "run it again the
          same way", VOID says "the load did not press hard enough, change the
          load". VOID can only ever replace a PASS, never a FAIL: a red arm
          stays red whether or not its trigger fired.

WHY VOID EXISTS. Every offload arm here declares kvso flags, and the check
confirmed they RESOLVED. Nothing confirmed a spill ever FIRED. A load that does
not press the KV pool spills zero times and is byte-coherent and resolves
exactly as declared -- so it reported PASS while proving nothing about the
spill path, which is a null soak dressed as a green light. The #550 window met
the same shape from the other side (``docs/dev/FEATURE_CATALOG.md:1793-1800``):
"one of the two WITH runs spilled and the other did not, so the arm measured
two different regimes and called the difference noise", and "The control never
spilled, so SPILL COST and HICACHE CONTENTION are confounded and neither is
isolated". ``Arm.require_any_markers`` is the treatment-side assertion,
``Arm.forbid_markers`` the control-side one.

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
    error_blocks,
    first_refusal,
    report_effective,
)

PASS = 0
FAIL = 1
STOP = 2
VOID = 3

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
#: Lines that OPEN a block the log itself frames as not-a-failure. Everything
#: until the next stamped server line belongs to that block and cannot be a
#: fatal, however many tracebacks it prints.
#:
#: "Ignore import error when loading" is sglang's own wording for an optional
#: import that did not resolve, and it prints the whole chained traceback
#: underneath. The htsglang Docker image ships torchcodec without a matching
#: ffmpeg, so every containerised boot emits two of these blocks -- sweep 1
#: scored K_bar1_graphs FAIL on one of them while the server was up and
#: serving correct output. A detector that cannot tell "ignored" from "fatal"
#: makes the whole Docker route unusable, and bar1 arms can only run there.
_QUOTED_OPENERS = (
    "auto-performance: hardware probe failed",
    "Ignore import error when loading",
)


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


def _first_error_line(log_text: str) -> Optional[str]:
    """The first raised-error line in the log, or None.

    Separate from :func:`_scan_fatals`: that one answers "did the boot die of
    something catastrophic", this one answers "did anything raise at all".
    A reject arm stopped by an argument-resolution ValueError leaves no fatal
    marker -- no traceback, no CUDA error -- and the difference between FAIL
    and STOP hangs on noticing it.
    """
    for _block, head in error_blocks(log_text):
        return head[:200]
    return None


@dataclass(frozen=True)
class Verdict:
    status: int  # PASS | FAIL | STOP | VOID
    arm: str
    reason: str
    effective: Optional[EffectiveConfig] = None
    coherence: Optional[CoherenceResult] = None
    #: The refusal line, for reject arms that passed.
    refusal: Optional[str] = None

    @property
    def label(self) -> str:
        return {PASS: "PASS", FAIL: "FAIL", STOP: "STOP", VOID: "VOID"}[self.status]

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
    """Declared vs resolved, field by field. Concrete disagreements only.

    A DECLARED ``None`` IS AN ASSERTION, NOT A MISSING ONE. It says "this axis
    must be absent from the resolved config" -- arm H declares
    ``spec_algorithm=None`` because it turns speculation off, and a server with
    speculation off legitimately prints no speculative line at all. So absence
    is what confirms it, and the presence of a value is the disagreement.

    Sweep 2 STOPped H on exactly this: it declared two axes absent, they were
    absent, and the check reported "could not confirm spec_algorithm,
    eagle_topk". Conflating "the log does not mention it" with "I could not
    tell" makes the one arm that isolates PS2 from the spec path unable to pass
    -- and it is a shape the matrix will meet again, because "feature off" is
    half of every crossing.
    """
    out: List[str] = []
    for key, want in arm.expect.items():
        got = getattr(eff, key, None)
        if want is None:
            if got is not None:
                out.append(
                    f"{key}: declared absent, resolved {got!r}"
                )
            continue
        if got is not None and got != want:
            out.append(f"{key}: declared {want!r}, resolved {got!r}")
    return out


def _unconfirmed_fields(arm: Arm, eff: EffectiveConfig) -> List[str]:
    """Declared axes the log did not let us decide either way.

    Only fields declared with a VALUE can be unconfirmed. A field declared
    ``None`` is decided by absence and is therefore always answerable -- see
    :func:`_mismatch_fields`.
    """
    return [
        key
        for key, want in arm.expect.items()
        if want is not None and getattr(eff, key, None) is None
    ]


def _load_precondition_held(arm: Arm, log_text: str) -> bool:
    """Did the arm's own subject occur at all?

    ANY of ``require_any_markers`` is enough -- they are alternative
    mechanisms for the same subject (decode-path spill vs PS2 born-spilled
    prefill), not a conjunction of requirements. An arm that declares none has
    no load precondition and is trivially satisfied: most arms prove a boot
    property, and demanding a runtime marker from them would invent a gate.
    """
    if not arm.require_any_markers:
        return True
    return any(marker in log_text for marker in arm.require_any_markers)


def _forbidden_hits(arm: Arm, log_text: str) -> List[str]:
    """Markers the arm declared absent that the log nevertheless contains.

    A control arm pinned spill-off is only a control if the pin is checked.
    Reported as a FAIL rather than a VOID on purpose: the arm declared a
    property of its own configuration and the log disproves it, which is the
    same class as a resolved config disagreeing with ``expect``.
    """
    return [marker for marker in arm.forbid_markers if marker in log_text]


def _void_reason(arm: Arm) -> str:
    return (
        "booted and resolved as declared, but the load precondition never "
        "held: none of "
        + ", ".join(repr(m) for m in arm.require_any_markers)
        + " appears in the log, so the mechanism this arm exists to test never "
        "fired. Nothing was proven -- press the KV pool harder (smaller "
        "--max-total-tokens, longer or more concurrent prompts) and re-run"
    )


def check_pairing(verdict: Verdict, control: Optional[Verdict]) -> Verdict:
    """Fold a declared control leg's verdict into a treatment arm's verdict.

    An arm whose result is a DELTA (``Arm.control_arm`` is set) has no result
    at all without its baseline, so a PASS whose control did not itself PASS is
    downgraded to VOID: the number exists, the comparison does not. Anything
    that is not a PASS is returned untouched -- a control cannot turn a red arm
    green, and a missing control cannot make a FAIL disappear.

    Pure and separate from :func:`check_arm` because the two legs are two runs
    with two artifact directories, and the pairing is a statement about the
    pair rather than about either run.
    """
    if verdict.status != PASS:
        return verdict
    if control is None:
        return Verdict(
            VOID,
            verdict.arm,
            (
                "declared a control leg but none was supplied, so its delta "
                "has no baseline -- the arm ran, the comparison did not"
            ),
            effective=verdict.effective,
            coherence=verdict.coherence,
        )
    if control.status != PASS:
        return Verdict(
            VOID,
            verdict.arm,
            (
                f"control leg {control.arm} is {control.label} "
                f"({control.reason}); the treatment ran but its delta has no "
                "usable baseline"
            ),
            effective=verdict.effective,
            coherence=verdict.coherence,
        )
    return verdict


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
    # Did not boot, and the arm's OWN guard did not fire. If anything else
    # raised, this arm stopped somewhere before the crossing it exists to
    # prove, and that is a FAIL rather than a STOP: the run is informative
    # (the arm is unrunnable as written, or an earlier guard shadows the one
    # under test) and a green light here is exactly the false pass sweep 1
    # handed out twice. STOP stays for the case where nothing at all is in the
    # log -- there, and only there, was nothing learned.
    fatal = _scan_fatals(log_text)
    other = _first_error_line(log_text)
    if fatal is not None or other is not None:
        return Verdict(
            FAIL,
            arm.name,
            (
                "refused, but with an unexpected error rather than the named "
                f"guard (expected markers {list(arm.reject_markers)}); the arm "
                f"never reached its own crossing -- {fatal or other}"
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

    # The control-side assertion. Placed with the #340 config checks because it
    # is one: an arm that declares "no spill may be admitted here" and then
    # logs a spill resolved to something it did not declare.
    forbidden = _forbidden_hits(arm, log_text)
    if forbidden:
        return Verdict(
            FAIL,
            arm.name,
            (
                "logged marker(s) the arm declared absent: "
                + ", ".join(repr(m) for m in forbidden)
                + " -- the pin did not hold, so this leg is contaminated and "
                "cannot serve as a control"
            ),
            effective=eff,
        )

    if arm.coherence == "none":
        if not _load_precondition_held(arm, log_text):
            return Verdict(
                VOID,
                arm.name,
                _void_reason(arm),
                effective=eff,
            )
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
    # LAST, so VOID can only ever replace a PASS. A coherence failure is a real
    # defect whether or not the arm's trigger fired, and must not be masked by
    # "the load was too light".
    if not _load_precondition_held(arm, log_text):
        return Verdict(
            VOID,
            arm.name,
            _void_reason(arm),
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
