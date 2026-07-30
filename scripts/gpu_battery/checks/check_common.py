#!/usr/bin/env python3
"""Shared verdict plumbing for the GPU battery checks.

Every check prints EXACTLY one line and nothing else:

    BATTERY-PASS <step>
    BATTERY-FAIL <step>: <one-sentence reason>
    BATTERY-STOP <step>: <one-sentence reason>

The distinction is the whole point of the executor protocol:

* STOP  the environment or a precondition is wrong -- a result file is
        missing, a card was busy, a tool is not installed, the step never
        really ran. Nothing was learned about the code under test.
* FAIL  the step ran and the artifact says the thing under test is broken.
        This is a real finding and goes to the bugfixer.

Checks look INSIDE the artifacts. An exit code says a process ended; it does
not say a boot produced an acceptance curve, that an aperture was measured, or
that a schema is loadable by the consumer that will have to load it. Those are
the questions here.

CPU-only and hermetic: a check never touches a card, never starts a server and
never needs the run to still be alive.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

PASS = 0
FAIL = 1
STOP = 2


class CheckFail(Exception):
    """The step ran and the artifact shows a real defect."""


class CheckStop(Exception):
    """A precondition or the environment is wrong; nothing was measured."""


def _one_line(text: object) -> str:
    """A verdict is one line. Newlines in a reason would break the executor's
    parsing, so they are folded rather than trusted. Takes the exception
    itself, not its str(), so no caller can forget the folding."""
    return " ".join(str(text).split())


def run_check(step: str, fn: Callable[[], Optional[str]]) -> int:
    """Run fn, print exactly one verdict line, return the exit code."""
    try:
        fn()
    except CheckStop as exc:
        print(f"BATTERY-STOP {step}: {_one_line(exc)}")
        return STOP
    except CheckFail as exc:
        print(f"BATTERY-FAIL {step}: {_one_line(exc)}")
        return FAIL
    except Exception as exc:  # a check that crashes has not judged anything
        print(
            f"BATTERY-STOP {step}: check itself raised {type(exc).__name__}: {_one_line(exc)}"
        )
        return STOP
    print(f"BATTERY-PASS {step}")
    return PASS


# ---------------------------------------------------------------------------
# artifact access -- a missing or unreadable artifact is always a STOP
# ---------------------------------------------------------------------------


def load_json(path: str, what: str) -> dict:
    if not os.path.exists(path):
        raise CheckStop(f"{what} missing ({path})")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckStop(f"{what} not readable ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckStop(f"{what} is not a JSON object ({path})")
    return payload


def read_text(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise CheckStop(f"{what} missing ({path})")
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError as exc:
        raise CheckStop(f"{what} not readable ({path}): {exc}") from exc


def require_envelope(
    payload: dict, kind: str, what: str, schema_version: Any = None
) -> None:
    if payload.get("kind") != kind:
        raise CheckFail(f"{what}: kind is {payload.get('kind')!r}, expected {kind!r}")
    if "schema_version" not in payload:
        raise CheckFail(f"{what}: schema_version missing")
    if schema_version is not None and payload["schema_version"] != schema_version:
        raise CheckFail(
            f"{what}: schema_version {payload['schema_version']!r}, "
            f"expected {schema_version!r}"
        )


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def require_number(value: Any, what: str, minimum: Optional[float] = None) -> float:
    if not is_number(value):
        raise CheckFail(f"{what} is {value!r}, not a number")
    if minimum is not None and value < minimum:
        raise CheckFail(f"{what} is {value}, expected >= {minimum}")
    return float(value)


def missing_fields(row: dict, fields: Iterable[str]) -> List[str]:
    return sorted(f for f in fields if f not in row)


# ---------------------------------------------------------------------------
# log scanning -- greps files, never pulls a server log into anyone's context
# ---------------------------------------------------------------------------

# Substrings that mean the run died, in the order they should be reported.
FATAL_LOG_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "NCCL error",
    "Watchdog caught collective operation timeout",
    "Traceback (most recent call last)",
)

# A server log also QUOTES the logs of helper subprocesses it ran, caught and
# recovered from -- the stage-0 hardware probe above all. Those quoted lines
# are evidence about a handled failure, not a symptom of the boot: a
# deliberately killed probe puts a full traceback into the server log while the
# server goes on to serve, and scoring that as a boot failure is how the #289
# evidence run produced a phantom BATTERY-FAIL (#303 part 4).
#
# Two ways a quoted block is recognised, because logs already on disk cannot be
# re-emitted:
#   1. the emitter's own marker, prefixed to every quoted line
#      (uneven_perf.QUOTED_SUBLOG_PREFIX -- keep the two literals in step);
#   2. structurally: sglang stamps its own lines with "[YYYY-MM-DD HH:MM:SS]",
#      so a block that starts at a line announcing a handled subprocess failure
#      and runs until the next stamped line is quoted material.
QUOTED_SUBLOG_PREFIX = "[probe-subprocess] "

#: Substrings that OPEN a quoted subprocess block in an sglang server log.
QUOTED_SUBLOG_OPENERS = ("auto-performance: hardware probe failed",)

_SERVER_LOG_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^\]]*\]")


def _is_server_line(line: str) -> bool:
    """Whether the line is the server's own, i.e. carries its timestamp."""
    return bool(_SERVER_LOG_STAMP.match(line))


def scan_log_for_fatals(
    path: str, what: str, markers: Sequence[str] = FATAL_LOG_MARKERS
) -> Optional[str]:
    """Return the first fatal marker found with its line number, or None.

    Lines belonging to a QUOTED subprocess log are skipped -- see
    ``QUOTED_SUBLOG_PREFIX``. A marker in there says a helper process died and
    the server reported it; the boot itself is judged by its own lines.

    The log itself is never returned -- only the one line that matters. The
    handoff quotes the path plus this line; the bugfixer greps the file.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, errors="replace") as f:
            in_quoted_block = False
            for lineno, line in enumerate(f, 1):
                if QUOTED_SUBLOG_PREFIX in line:
                    continue
                if in_quoted_block:
                    # The quoted block ends at the server's next own line.
                    if not _is_server_line(line):
                        continue
                    in_quoted_block = False
                if any(opener in line for opener in QUOTED_SUBLOG_OPENERS):
                    in_quoted_block = True
                    continue
                for marker in markers:
                    if marker in line:
                        return f"{path}:{lineno}: {_one_line(line)[:200]}"
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# the accept-curve contract, shared by boots A, B and C
# ---------------------------------------------------------------------------

# The reference column, per the r7c README: the same FP8 vehicle at pure NEXTN
# and K=3. Every accept number in these boots is reported against it. Kept here
# so the check and the emitter cannot drift apart.
REFERENCE_SOURCE = "docs/benchmarks/htsglang_tp3.json:87-90"
REFERENCE_ACCEPT = {"prose": 2.688, "code": 3.279}
REFERENCE_COLUMN_KIND = "r7c_reference_column"
REFERENCE_COLUMN_SCHEMA_VERSION = 1


def check_accept_artifact(
    step_dir: str,
    boot: str,
    prompts: Sequence[str],
    steps_k: int = 3,
) -> None:
    """Content contract for an r7c accept boot.

    Not "the recipe exited 0" -- that is true of a boot that produced five
    rows of None. What is required:

      1. every requested prompt has an arm,
      2. accept_len_mean is a real number (the recipe's own abort criterion:
         None after the first prompt means the spec path is not running),
      3. rounds > 0 (a spec path that never proposed measured nothing),
      4. the PER-POSITION curve exists and covers positions 0..K-1 -- a mean
         is structurally blind to a positional pathology, which is the entire
         reason the probe was written,
      5. the reference column is present for every prompt that has one,
      6. the VRAM summary has one row per card,
      7. the server log carries no OOM / NCCL / traceback marker.

    The MAGNITUDE of the accept numbers is deliberately not judged here. Both
    the reproducing and the falsifying outcome are results; deciding which one
    happened is the reader's job, not the executor's.
    """
    accept_path = os.path.join(step_dir, "accept.json")
    classify_missing_result(step_dir, boot, accept_path, "accept.json")
    report = load_json(accept_path, f"{boot}: accept.json")

    arms = report.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CheckFail(f"{boot}: accept.json has no arms")

    by_prompt = {a.get("prompt"): a for a in arms if isinstance(a, dict)}
    absent = [p for p in prompts if p not in by_prompt]
    if absent:
        raise CheckFail(f"{boot}: prompts without an arm: {','.join(absent)}")

    for prompt in prompts:
        arm = by_prompt[prompt]
        serving = arm.get("serving")
        if not isinstance(serving, dict):
            raise CheckFail(f"{boot}/{prompt}: no serving block")
        if serving.get("accept_len_mean") is None:
            raise CheckFail(
                f"{boot}/{prompt}: accept_len_mean is None -- the spec path is not "
                "running, or the probe is off"
            )
        require_number(serving["accept_len_mean"], f"{boot}/{prompt}: accept_len_mean")
        rounds = serving.get("rounds")
        if rounds is not None:
            require_number(rounds, f"{boot}/{prompt}: rounds", minimum=1)
        curve = serving.get("curve")
        positions = curve_positions(curve)
        if positions is None:
            raise CheckFail(
                f"{boot}/{prompt}: no per-position accept curve, no Positionskurve "
                f"(curve={curve!r}) -- the mean alone is blind to a positional "
                "pathology"
            )
        if len(positions) < steps_k:
            raise CheckFail(
                f"{boot}/{prompt}: Positionskurve covers {len(positions)} of "
                f"{steps_k} Positionen"
            )
        if 0 not in positions:
            raise CheckFail(f"{boot}/{prompt}: position 0 is missing from the curve")

    check_reference_column(step_dir, boot, prompts)
    check_vram_summary(step_dir, boot)

    log_path = os.path.join(step_dir, "server.log")
    fatal = scan_log_for_fatals(log_path, f"{boot}: server.log")
    if fatal:
        raise CheckFail(f"{boot}: Fatal im Serverlog -- {fatal}")


def classify_missing_result(
    step_dir: str, boot: str, result_path: str, what: str
) -> None:
    """A missing result file is not automatically a STOP.

    A boot whose server never came up, or that died on an OOM, DID run and DID
    tell us something -- that is a FAIL and it goes to the bugfixer. Only a
    boot that left no server log at all is a STOP, because then the recipe
    never got far enough to measure anything.

    Getting this backwards is expensive in both directions: a STOP sends the
    bugfixer nothing, and a FAIL on a busy card sends him chasing a defect
    that is not there.
    """
    if os.path.exists(result_path):
        return
    log_path = os.path.join(step_dir, "server.log")
    if not os.path.exists(log_path):
        raise CheckStop(
            f"{boot}: neither {what} nor server.log -- the recipe never ran"
        )
    fatal = scan_log_for_fatals(log_path, f"{boot}: server.log")
    if fatal:
        raise CheckFail(f"{boot}: no {what}, the server log names {fatal}")
    raise CheckFail(
        f"{boot}: no {what}, but a server.log ({log_path}) -- the server never came "
        "up, or the probe broke off"
    )


def curve_positions(curve: Any) -> Optional[Dict[int, float]]:
    """The probe reports the curve as {position: rate}; JSON turns the keys
    into strings. A list is accepted too. Returns None when there is no usable
    curve at all."""
    if curve is None:
        return None
    out: Dict[int, float] = {}
    if isinstance(curve, dict):
        for key, value in curve.items():
            try:
                pos = int(key)
            except (TypeError, ValueError):
                continue
            if is_number(value):
                out[pos] = float(value)
    elif isinstance(curve, list):
        for pos, value in enumerate(curve):
            if is_number(value):
                out[pos] = float(value)
    else:
        return None
    return out or None


def check_reference_column(step_dir: str, boot: str, prompts: Sequence[str]) -> None:
    """Obligation 7 of the r7c README: every accept number is reported AGAINST
    the reference, not on its own. The emitter writes the join; this verifies
    it exists, names its source and covers the prompts that have a reference."""
    path = os.path.join(step_dir, "reference_column.json")
    payload = load_json(path, f"{boot}: reference_column.json")
    require_envelope(
        payload,
        REFERENCE_COLUMN_KIND,
        f"{boot}: reference_column.json",
        REFERENCE_COLUMN_SCHEMA_VERSION,
    )
    if payload.get("reference_source") != REFERENCE_SOURCE:
        raise CheckFail(
            f"{boot}: reference_column names source "
            f"{payload.get('reference_source')!r}, expected {REFERENCE_SOURCE!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CheckFail(f"{boot}: reference_column.json has no rows")
    by_prompt = {r.get("prompt"): r for r in rows if isinstance(r, dict)}
    for prompt in prompts:
        if prompt not in by_prompt:
            raise CheckFail(f"{boot}: reference_column has no line for {prompt}")
        row = by_prompt[prompt]
        require_number(row.get("measured"), f"{boot}/{prompt}: measured")
        if prompt in REFERENCE_ACCEPT:
            expected = REFERENCE_ACCEPT[prompt]
            if row.get("reference") != expected:
                raise CheckFail(
                    f"{boot}/{prompt}: Referenzwert {row.get('reference')!r}, "
                    f"expected {expected}"
                )
            require_number(row.get("ratio"), f"{boot}/{prompt}: ratio")


def check_vram_summary(step_dir: str, boot: str) -> None:
    """Queue item 4: minimum free MiB per card over the whole boot. The number
    every multi-card placement decision has been missing, so a boot that did
    not produce it produced an incomplete result."""
    path = os.path.join(step_dir, "vram_summary.txt")
    text = read_text(path, f"{boot}: vram_summary.txt")
    rows = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) >= 4:
            rows += 1
    if rows < 1:
        raise CheckFail(
            f"{boot}: vram_summary.txt contains no card line "
            "(MIN free per card was never collected)"
        )


# ---------------------------------------------------------------------------
# import bootstrap for the repo-side consumers a check wants to prove against
# ---------------------------------------------------------------------------


def add_repo_to_path() -> None:
    """Put <worktree>/python on sys.path so a check can load a result with the
    SAME loader the production code will use. Proving an artifact parses with
    the real consumer beats re-implementing the schema in the check."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_python = os.path.abspath(os.path.join(here, "..", "..", "..", "python"))
    if os.path.isdir(repo_python) and repo_python not in sys.path:
        sys.path.insert(0, repo_python)
