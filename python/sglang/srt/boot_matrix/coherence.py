# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The coherence gate: short byte-exact + long graded (#349).

THE DESIGN CONSTRAINT, stated first because it decides everything else. Qwen
GDN prefill is NOT reproducible past ~109 tokens on any backend (a registered
upstream fact). So there is no such thing as a byte-identity gate on long
output -- it produces false reds, which is the exact failure mode a bug net
must not have. The gate is therefore two tiers:

* BYTE tier -- a SHORT forced continuation whose leading bytes must match a
  reference exactly. Valid only inside the reproducibility window, so a byte
  probe whose reference is too long is a mis-designed probe and the gate says
  STOP (environment), not FAIL (defect).

* GRADED tier -- anything longer is scored, not compared. This reuses the
  #274/#284 house grader verbatim (``scripts/dual_group/r12/graded.py``): the
  ``alphabet`` and ``squares`` continuations grade to an integer a near-tie
  flip cannot move, so a trajectory that diverged at a numeric tie and still
  emitted the correct sequence scores the same and the gate stays green. Text
  identity is reported as FRAMING only, never as the criterion -- the #360
  standard. The floor a score must clear is the empirical A-vs-A band, passed
  in per probe; this module does not invent it.

Everything here is a pure function of already-collected text. No server, no
card. The grader is imported from the checkout (it is import-clean and
server-free by its own design); if it cannot be found -- e.g. running from a
wheel with no scripts tree -- the result is flagged ``grader_available=False``
and the check turns that into STOP, never a FAIL.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

#: Proxy cap for "short enough to byte-gate". The registered limit is ~109
#: tokens; these forced continuations are mechanical (letters, "n n*n" lines),
#: so a generous character cap sits well inside the token window while staying
#: a pure-text check with no tokenizer dependency. A reference longer than this
#: is refused as a byte probe, not silently graded.
BYTE_TIER_MAX_CHARS = 320

#: Registered fact, carried here so the reason is next to the rule.
GDN_REPRO_LIMIT_TOKENS = 109


@dataclass(frozen=True)
class ProbeResult:
    name: str
    tier: str  # "byte" | "graded"
    ok: bool
    detail: str
    score: Optional[int] = None
    max_score: Optional[int] = None
    min_score: Optional[int] = None
    #: Byte-identity of the whole text vs its reference -- FRAMING only, printed
    #: so a reader can see a text difference, never itself a pass/fail input on
    #: the graded tier.
    byte_identical: Optional[bool] = None


@dataclass(frozen=True)
class CoherenceResult:
    passed: bool
    grader_available: bool
    #: A byte probe whose reference exceeds the reproducibility window: a probe
    #: design error, surfaced as STOP by the check rather than a false FAIL.
    byte_probe_too_long: bool = False
    probes: List[ProbeResult] = field(default_factory=list)

    def render(self) -> str:
        if not self.grader_available:
            return "coherence: SKIPPED (grader not found in checkout)"
        bits = [
            f"{p.name}[{p.tier}]="
            + (
                f"{p.score}/{p.max_score}>={p.min_score}"
                if p.tier == "graded"
                else ("byte-ok" if p.ok else "byte-MISMATCH")
            )
            for p in self.probes
        ]
        return f"coherence: {'PASS' if self.passed else 'FAIL'} [" + " ".join(bits) + "]"

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "grader_available": self.grader_available,
            "byte_probe_too_long": self.byte_probe_too_long,
            "probes": [
                {
                    "name": p.name,
                    "tier": p.tier,
                    "ok": p.ok,
                    "detail": p.detail,
                    "score": p.score,
                    "max_score": p.max_score,
                    "min_score": p.min_score,
                    "byte_identical": p.byte_identical,
                }
                for p in self.probes
            ],
        }


def _find_grader() -> Optional[Callable[[str, str], Dict[str, int]]]:
    """Import the canonical #274 grader from the checkout.

    Not vendored: reusing it is the instruction, and a second copy of the
    scorers is exactly the drift the battery's shared ``check_common`` exists
    to prevent. Located relative to this file's repo root; returns None if the
    scripts tree is absent (wheel install), which the caller renders as a
    grader-unavailable SKIP rather than a failure.
    """
    here = os.path.abspath(__file__)
    # python/sglang/srt/boot_matrix/coherence.py -> repo root is four up from
    # the sglang package dir.
    repo_root = here
    for _ in range(6):
        repo_root = os.path.dirname(repo_root)
        candidate = os.path.join(
            repo_root, "scripts", "dual_group", "r12", "graded.py"
        )
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location(
                "sglang_boot_matrix_r12_graded", candidate
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "score", None)
    return None


def grade_probes(
    probes: Sequence[Dict[str, object]],
    *,
    grader: Optional[Callable[[str, str], Dict[str, int]]] = None,
) -> CoherenceResult:
    """Grade a collected probe set. Pure function; hermetic.

    Each probe is a dict:
        name       scorer key ("alphabet"|"squares") or a free label
        tier       "byte" | "graded"
        text       the model's output for this probe
        ref_text   BYTE tier: the exact leading text expected
        min_score  GRADED tier: the A-vs-A floor the score must clear

    ``grader`` is injectable so the unit tests drive the exact scorers without
    a scripts tree; production passes None and the canonical grader is located.
    """
    resolved_grader = grader if grader is not None else _find_grader()
    if resolved_grader is None:
        return CoherenceResult(passed=False, grader_available=False)

    results: List[ProbeResult] = []
    too_long = False
    for probe in probes:
        name = str(probe.get("name", "?"))
        tier = str(probe.get("tier", "graded"))
        text = str(probe.get("text", "") or "")

        if tier == "byte":
            ref = str(probe.get("ref_text", "") or "")
            if len(ref) > BYTE_TIER_MAX_CHARS:
                too_long = True
                results.append(
                    ProbeResult(
                        name=name,
                        tier="byte",
                        ok=False,
                        detail=(
                            f"byte reference is {len(ref)} chars, over the "
                            f"{BYTE_TIER_MAX_CHARS}-char window: cannot byte-gate "
                            "past the GDN reproducibility limit"
                        ),
                    )
                )
                continue
            ok = text[: len(ref)] == ref
            results.append(
                ProbeResult(
                    name=name,
                    tier="byte",
                    ok=ok,
                    detail="leading bytes match" if ok else "leading bytes differ",
                    byte_identical=ok,
                )
            )
            continue

        # graded tier
        min_score = int(probe.get("min_score", 0) or 0)
        graded = resolved_grader(name, text)
        score = int(graded.get("score", -1))
        max_score = int(graded.get("max_score", -1))
        ref = probe.get("ref_text")
        byte_identical = (text == str(ref)) if ref is not None else None
        ok = score >= min_score and score >= 0
        detail = (
            f"score {score}/{max_score} vs A-vs-A floor {min_score}"
            if score >= 0
            else f"no scorer for probe {name!r}"
        )
        results.append(
            ProbeResult(
                name=name,
                tier="graded",
                ok=ok,
                detail=detail,
                score=score,
                max_score=max_score,
                min_score=min_score,
                byte_identical=byte_identical,
            )
        )

    passed = bool(results) and all(p.ok for p in results) and not too_long
    return CoherenceResult(
        passed=passed,
        grader_available=True,
        byte_probe_too_long=too_long,
        probes=results,
    )
