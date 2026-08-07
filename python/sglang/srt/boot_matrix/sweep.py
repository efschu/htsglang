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
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.boot_matrix.arms import ARMS, BASE_ENV, BASE_FLAGS, Arm, arm_by_name
from sglang.srt.boot_matrix.check import Verdict, check_arm, check_pairing
from sglang.srt.boot_matrix.effective import READY_MARKER

logger = logging.getLogger(__name__)

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
    argparse, which is how an arm overrides e.g. ``--rank-tp-ratio``.

    ``arm.drop_flags`` REMOVES a base flag and its value beforehand. Appending
    can override a value but cannot unset one, and sweep 1 lost two arms to
    that gap: arm H wrote ``--speculative-algorithm none`` meaning "no spec"
    (there is no such algorithm -- it just named one nobody registered), and
    ``reject_dcp_offlane`` inherited ``--rank-auto-reserve-mib`` while pinning
    an explicit ``--rank-tp-ratio``, which the server refuses outright.
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
    argv.extend(_without(BASE_FLAGS, arm.drop_flags))
    argv.extend(arm.flags)
    return env, argv


def _without(flags: Sequence[str], drop: Sequence[str]) -> List[str]:
    """``flags`` minus each name in ``drop``, taking its value with it.

    A flag's value is the next token when that token is not itself a flag, so
    store-true options drop cleanly and valued options do not leave an orphan
    argument behind for argparse to misread as positional.
    """
    if not drop:
        return list(flags)
    unknown = [d for d in drop if d not in flags]
    if unknown:
        # A drop that matches nothing is a silent no-op, and a silent no-op in
        # the arm list is how an arm ends up running something other than what
        # it declares -- the whole failure class this matrix exists for.
        raise ValueError(
            f"drop_flags names {unknown}, which the base recipe does not "
            f"contain; the arm would silently keep the flag it meant to remove"
        )
    out: List[str] = []
    i = 0
    flags = list(flags)
    while i < len(flags):
        token = flags[i]
        if token in drop:
            i += 1
            if i < len(flags) and not flags[i].startswith("--"):
                i += 1
            continue
        out.append(token)
        i += 1
    return out


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


#: The arm the band is measured from. Named once so the measurement, the skip
#: in the loop and the tests cannot drift apart.
BAND_ARM = "A_default"


def _measure_band(
    selected: Sequence[Arm],
    *,
    model_path: str,
    out_dir: str,
    port: int,
    run: Optional[Callable[..., Verdict]] = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int], Optional[Verdict]]:
    """Boot ``A_default`` twice and derive the reference and the floor.

    This is the mechanism the module docstring describes, and it is the reason
    a later arm's coherence verdict means anything:

    * the BYTE reference is the baseline's FIRST-run text. Byte probes are
      short on purpose (inside the GDN reproducibility window), so the same
      prompt on the same build must reproduce it exactly; a later arm that
      does not is carrying a real difference.
    * the GRADED floor is the MINIMUM score across the two baseline runs. Two
      runs of the SAME configuration is an A-vs-A measurement -- whatever they
      differ by is this rig's noise, and an arm is only red below it. A floor
      taken from one run would call ordinary run-to-run variation a defect.

    Returns ``({}, {}, None)`` when the baseline is not in the selection, so
    ``--only`` on a single arm still runs; that arm is then graded against an
    empty band, which the check reports rather than hides.
    """
    runner = run or run_arm
    if not any(a.name == BAND_ARM for a in selected):
        return {}, {}, None
    arm = next(a for a in selected if a.name == BAND_ARM)

    first = runner(arm, model_path=model_path, out_dir=out_dir, port=port)
    reference: Dict[str, Dict[str, object]] = {}
    for probe in _probes_of(first, out_dir):
        reference[str(probe["name"])] = {"text": probe.get("text", "")}

    # The second run overwrites the first's artifacts on purpose: the arm
    # directory should hold the run the verdict was computed from, and that is
    # the second one (graded against the band the pair produced).
    second = runner(
        arm, model_path=model_path, out_dir=out_dir, port=port,
        reference_probes=reference, band={},
    )
    band: Dict[str, int] = {}
    for probe in _probes_of(second, out_dir):
        name = str(probe["name"])
        if str(probe.get("tier")) != "graded":
            continue
        scores = [
            int(p.get("score", 0))
            for p in (_probe_named(first, name, out_dir), probe)
            if p is not None and p.get("score") is not None
        ]
        if scores:
            band[name] = min(scores)
    # Re-check the baseline against the band it produced, so A_default is held
    # to the same standard as everything after it rather than exempted.
    graded = runner(
        arm, model_path=model_path, out_dir=out_dir, port=port,
        reference_probes=reference, band=band,
    )
    return reference, band, graded


def _probes_of(verdict: Verdict, out_dir: str) -> List[Dict[str, object]]:
    """Probe name, tier, TEXT and score for one finished baseline run.

    The two halves come from two places, and that is not an accident:

    * the SCORE is on the verdict's ``ProbeResult`` -- it is produced by the
      grader inside ``check_arm``;
    * the TEXT is only in ``probes.json`` on disk. ``ProbeResult`` has no
      ``text`` field, and the first cut of this function read
      ``getattr(p, "text", "")``, which is always "". That would have left the
      byte reference empty and reproduced the exact bug this repair exists to
      remove -- a band that measures nothing while the report says it did.

    It got caught because the run was checked against the artifact rather than
    the object; the unit test had been passing because its stub injected a
    ``text`` attribute the real class does not have. A stub that is more
    capable than the type it stands in for tests itself.
    """
    coh = getattr(verdict, "coherence", None)
    if coh is None:
        return []
    texts: Dict[str, str] = {}
    path = os.path.join(out_dir, getattr(verdict, "arm", BAND_ARM), "probes.json")
    try:
        with open(path) as f:
            for entry in json.load(f):
                texts[str(entry.get("name", ""))] = str(entry.get("text", ""))
    except (OSError, ValueError):
        # No artifact to read: the caller gets empty texts and the check
        # reports an empty band rather than a silently invented one.
        texts = {}
    out: List[Dict[str, object]] = []
    for p in getattr(coh, "probes", ()) or ():
        name = getattr(p, "name", "")
        out.append(
            {
                "name": name,
                "tier": getattr(p, "tier", ""),
                "text": texts.get(name, ""),
                "score": getattr(p, "score", None),
            }
        )
    return out


def _probe_named(
    verdict: Verdict, name: str, out_dir: str
) -> Optional[Dict[str, object]]:
    for p in _probes_of(verdict, out_dir):
        if p["name"] == name:
            return p
    return None


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
        # The ranks are not in the launcher's process group, so the terminate
        # above does not reach them; wait for the cards themselves. Placed
        # here rather than only in _measure_band because the hazard is
        # arm-to-arm -- the band merely hits it hardest, being the same large
        # model three times in a row.
        wait_for_vram_release()

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


#: A card with less than this in use counts as released. Not zero: the driver
#: keeps a few MiB of context around, and nvidia-smi reports it.
VRAM_IDLE_MIB = 512
#: Bound on the wait. Generous, because a 27B teardown across three ranks is
#: seconds and anything approaching this is a leak worth seeing in the log.
VRAM_RELEASE_TIMEOUT_S = 180.0


def wait_for_vram_release(
    timeout_s: float = VRAM_RELEASE_TIMEOUT_S,
    idle_mib: int = VRAM_IDLE_MIB,
    poll_s: float = 2.0,
) -> bool:
    """Block until every visible card is back near idle. Bounded; never raises.

    WHY THE PROCESS EXIT IS NOT ENOUGH. ``_terminate`` signals the launcher's
    process group and waits for the launcher. The TP scheduler ranks are NOT
    in that group -- ``run_arm`` starts the launcher with
    ``start_new_session=True`` and the launcher starts its ranks the same way,
    so they outlive both the signal and the wait, still holding their share of
    the model.

    Sweep 2 paid for this in the one place that boots the same arm three times
    in a row. A_default's third boot -- the one that regrades the baseline
    against the band it just produced -- died with "CUDA out of memory ...
    Process 1888263 has 18.87 GiB memory in use", and 1888263 was the SECOND
    boot's rank. The band itself was fine (it comes from boots one and two),
    but the arm that produces it could not pass, so the matrix reported its own
    baseline red.

    Returning False rather than raising is deliberate: a card that stays busy
    is a finding for the NEXT boot to report against a real log, not a reason
    to abort a sweep from inside a teardown helper.
    """
    import subprocess

    deadline = time.monotonic() + max(timeout_s, 0.0)
    last: Optional[int] = None
    while True:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            ).stdout
            used = [int(x.strip()) for x in out.splitlines() if x.strip()]
        except (OSError, ValueError, subprocess.SubprocessError):
            # No nvidia-smi (card-less host, unit tests): nothing to wait for.
            return True
        if not used:
            return True
        last = max(used)
        if last <= idle_mib:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "boot-matrix: cards still hold %d MiB after %.0fs of waiting "
                "for release; the next arm starts against that. A rank from "
                "the previous boot is still alive -- it is not in the "
                "launcher's process group, so the terminate did not reach it.",
                last, timeout_s,
            )
            return False
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# card-less surfaces
# ---------------------------------------------------------------------------
def _apply_pairings(verdicts: List[Verdict]) -> List[Verdict]:
    """Fold every declared control leg into its treatment arm's verdict.

    Pure over the verdict list, so it is unit-testable without a sweep. An arm
    that declares no ``control_arm`` is returned untouched, and a control leg
    that was not run in this sweep resolves to ``None`` -- which
    :func:`check_pairing` treats as "no baseline", i.e. VOID, not PASS. That
    asymmetry is the point: a sweep run with ``--only O_hicache_contention``
    must not be able to report a contention number.
    """
    by_name = {v.arm: v for v in verdicts}
    out: List[Verdict] = []
    for v in verdicts:
        try:
            arm = arm_by_name(v.arm)
        except KeyError:  # the A-vs-A band baseline is not a matrix arm
            out.append(v)
            continue
        if arm.control_arm is None:
            out.append(v)
            continue
        folded = check_pairing(v, by_name.get(arm.control_arm))
        if folded is not v:
            print(folded.render())
        out.append(folded)
    return out


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
    # The baseline is booted THREE times, not once: twice to measure the
    # A-vs-A band and once more graded against the band it produced. Two extra
    # boots is what a gate that can actually fail costs, and an estimate that
    # hid them would be wrong by the price of the thing that makes the run
    # worth doing.
    band_extra = 2 * next(
        (a.expected_seconds for a in ARMS if a.name == BAND_ARM), 0.0
    )
    lines.append(
        f"total estimated card time: {(total + band_extra)/60.0:.0f} min over "
        f"{len(ARMS)} arms ({sum(1 for a in ARMS if a.kind=='boot')} boot, "
        f"{sum(1 for a in ARMS if a.kind=='reject')} reject), including "
        f"{band_extra/60.0:.0f} min for the two extra {BAND_ARM} boots the "
        f"A-vs-A band is measured from."
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

    # MEASURE THE BAND FIRST. Until this existed the docstring above was a
    # promise and nothing else: run_arm was called without reference_probes,
    # so every arm carried ref_text="" and min_score=0 and the coherence half
    # of the gate could not fail. Sweep 1 reported "coherence within the
    # A-vs-A band" for arms that were never compared against anything.
    reference_probes, band, baseline = _measure_band(
        selected, model_path=args.model, out_dir=args.out, port=args.port
    )
    verdicts: List[Verdict] = []
    if baseline is not None:
        print(baseline.render())
        verdicts.append(baseline)
    for arm in selected:
        if baseline is not None and arm.name == BAND_ARM:
            continue
        v = run_arm(
            arm,
            model_path=args.model,
            out_dir=args.out,
            port=args.port,
            reference_probes=reference_probes,
            band=band,
        )
        print(v.render())
        verdicts.append(v)

    # PAIRING, AFTER every leg has run. An arm whose result is a DELTA has no
    # result without its baseline, so a treatment that passed while its
    # declared control did not -- or was not selected into this sweep at all --
    # is folded down to VOID here rather than printed as a green number. Done
    # in a second pass because the control may run AFTER the treatment in arm
    # order, and a first-pass fold would then read a verdict that does not
    # exist yet.
    verdicts = _apply_pairings(verdicts)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump([v.to_json() for v in verdicts], f, indent=2)
    return 0 if all(v.status == 0 for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
