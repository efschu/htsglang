#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Turnkey runner for TICKET #398 gates A and A' -- native GGUF MXFP4 (type 39).

WHY THIS EXISTS. #398 is fully built: the kernel set is merged, the wheel that
carries it is installed (sha 67f03cfa), and the hermetic suites are green. The
only thing standing between "installed" and "proven" is fourteen CUDA tests
that report ``SKIPPED: no CUDA device`` off-GPU, on two architectures. That is
window work, and window work that has to be reconstructed from a prose ticket
at 3am is window work that gets done wrong or not at all. This script is the
ticket's sections 0, 1 and 1b made executable end to end.

It takes no window itself and boots no server. It runs tests and probes.

THREE HAZARDS IT ENCODES, each of which has already bitten this ticket once:

1. THE FALSIFIER'S SECOND ARM IS ``True / False / False``, NOT
   ``False / False / False`` (the #519 correction). The first value is
   ``hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")`` -- a property of
   which objects are on disk, i.e. of the WHEEL. ``SGLANG_GGUF_MXFP4_NATIVE``
   is read one level up in ``gguf.mxfp4_native()``, which returns False before
   it ever reaches the hasattr. So the lever flips the two DERIVED answers and
   must leave the marker alone. A runner that expected all three to flip would
   go red on a correct build -- the expectation would be the defect, not the
   code. That table is data here (:data:`FALSIFIER_ARMS`) so it can be read
   and argued with instead of being buried in an assertion.

2. ``sgl_kernel`` MUST BE IMPORTED BEFORE THE MARKER IS PROBED. torch registers
   ops when the extension ``.so`` loads, not when the namespace is touched. A
   probe that runs first answers "absent" on a wheel that carries the op, and
   if anything then defines a fake schema the real ``.so`` registers the same
   schema twice -- a C++ abort, not a catchable exception. That is not
   hypothetical: it aborted the interpreter under pytest and the gate looked
   like it had simply not run.

3. PHYSICAL INDEX != TORCH ORDINAL on this rig. Cards are resolved through the
   NVML IdentityMap and classified by their REAL compute capability, never by
   a hardcoded ordinal. The 5090 is not reliably ordinal 0, 1 or 2 across
   boots, and both arches are separate gates: sm120 already had an MXFP4 path
   (``Mxfp4MarlinMoEMethod``, safetensors), so an sm120-only green would not
   prove the GGUF path at all.

USAGE

    # hermetic, no GPU, no window -- proves this harness works and can fail
    python bench/398/run_398_gate_a.py --self-test

    # inside a claimed window (see /spinning/gpu-arb/, claim canon in
    # evidence-qwen38/claim_window.sh: heartbeat staleness AND no
    # launch_server mid-startup); this script does not claim for you
    python bench/398/run_398_gate_a.py --run --out /spinning/gpu-battery-results/<date>_398

Exit status: 0 = every gate green, 1 = a gate failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: The ggml quant type this ticket is about.
MXFP4_TYPE = 39

#: The GPU-only gate. Off-GPU these 14 report SKIPPED and prove nothing.
GATE_A_TESTS = "test/registered/unit/quantization/test_gguf_mxfp4_cuda.py"

#: Both are required. sm120 alone is not a pass: that card already had an
#: MXFP4 route before #398, so a green there does not isolate the GGUF path.
REQUIRED_ARCHES = ("sm86", "sm120")


@dataclass(frozen=True)
class Observation:
    """What one probe process reported."""

    marker: bool
    mxfp4_native: bool
    in_mmvq_set: bool

    def as_tuple(self) -> Tuple[bool, bool, bool]:
        return (self.marker, self.mxfp4_native, self.in_mmvq_set)


@dataclass(frozen=True)
class FalsifierArm:
    name: str
    env: Dict[str, str]
    expected: Tuple[bool, bool, bool]
    why: str


#: Gate A' as a table. See hazard 1 in the module docstring for why the second
#: arm's first value is True.
FALSIFIER_ARMS: Tuple[FalsifierArm, ...] = (
    FalsifierArm(
        name="native-default",
        env={},
        expected=(True, True, True),
        why="the installed wheel carries the kernels and nothing suppresses them",
    ),
    FalsifierArm(
        name="lever-off",
        env={"SGLANG_GGUF_MXFP4_NATIVE": "0"},
        expected=(True, False, False),
        why=(
            "the lever is read above the hasattr, so it flips the two derived "
            "answers and leaves the wheel marker alone (#519)"
        ),
    ),
)

#: Run in a subprocess, because the lever is read once at import time. Note the
#: import of sgl_kernel BEFORE the hasattr -- see hazard 2.
_PROBE_SRC = """
import json
import torch
import sgl_kernel  # noqa: F401  -- registers the ops; MUST precede the probe
marker = hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native")
from sglang.srt.layers.quantization.gguf import MXFP4_NATIVE, MMVQ_QUANT_TYPES
print("@@" + json.dumps({
    "marker": bool(marker),
    "mxfp4_native": bool(MXFP4_NATIVE),
    "in_mmvq_set": int(%d) in {int(t) for t in MMVQ_QUANT_TYPES},
}))
""" % MXFP4_TYPE


def classify_arch(capability: Tuple[int, int]) -> str:
    """``(8, 6) -> "sm86"``. The authority is the card, not its name."""
    return f"sm{capability[0]}{capability[1]}"


def select_gate_cards(
    cards: Sequence[object],
    capability_of: Dict[int, Tuple[int, int]],
) -> Dict[str, object]:
    """Pick one card per required arch, keyed by REAL capability.

    ``cards`` are CardIdentity records; ``capability_of`` maps cuda_ordinal to
    the compute capability torch reports for it. Cards torch cannot see
    (cuda_ordinal None) are not gate candidates -- a masked card cannot run a
    test.
    """
    chosen: Dict[str, object] = {}
    for card in cards:
        ordinal = getattr(card, "cuda_ordinal", None)
        if ordinal is None or ordinal not in capability_of:
            continue
        arch = classify_arch(capability_of[ordinal])
        if arch in REQUIRED_ARCHES and arch not in chosen:
            chosen[arch] = card
    return chosen


def check_arm(observed: Observation, arm: FalsifierArm) -> List[str]:
    """Return a list of human-readable mismatches; empty means green."""
    labels = ("wheel marker", "MXFP4_NATIVE", f"type {MXFP4_TYPE} in MMVQ set")
    out: List[str] = []
    for label, got, want in zip(labels, observed.as_tuple(), arm.expected):
        if got is not want:
            out.append(f"{arm.name}: {label} = {got}, expected {want}")
    return out


def parse_pytest_summary(text: str) -> Optional[str]:
    """The verbatim summary line. The ticket wants it quoted, not paraphrased."""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if any(
            f" {word}" in stripped or stripped.startswith(word)
            for word in ("passed", "failed", "error", "skipped", "no tests ran")
        ):
            return stripped
    return None


def _run(cmd: Sequence[str], env: Dict[str, str], timeout: int = 3600):
    merged = dict(os.environ)
    merged.update(env)
    return subprocess.run(
        list(cmd), env=merged, capture_output=True, text=True, timeout=timeout
    )


def probe(python: str, env: Dict[str, str], repo: str) -> Observation:
    done = _run([python, "-c", _PROBE_SRC], {**env, "PYTHONPATH": f"{repo}/python"}, 600)
    for line in done.stdout.splitlines():
        if line.startswith("@@"):
            return Observation(**json.loads(line[2:]))
    raise RuntimeError(
        "probe produced no result line.\n"
        f"stdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-2000:]}"
    )


@dataclass
class Report:
    arches: Dict[str, dict] = field(default_factory=dict)
    falsifier: Dict[str, dict] = field(default_factory=dict)
    green: bool = False

    def render(self) -> str:
        """The block to paste into TICKET_398 section 1 / 1b."""
        lines = ["## Gate A -- numerical correctness", ""]
        for arch in REQUIRED_ARCHES:
            got = self.arches.get(arch)
            if not got:
                lines.append(f"{arch:6s}: NOT RUN")
                continue
            lines.append(f"{arch:6s}: {got.get('summary', 'no summary parsed')}")
            lines.append(f"        card: {got.get('card', '?')}")
        lines += ["", "## Gate A' -- the falsifier on the real wheel", ""]
        for name, got in self.falsifier.items():
            lines.append(f"{name:16s}: {got.get('observed')}  {got.get('status')}")
        lines += ["", f"OVERALL: {'GREEN' if self.green else 'NOT GREEN'}"]
        return "\n".join(lines)


def self_test() -> int:
    """Hermetic proof that this harness runs, and that its gates can FAIL.

    Desk-written-never-executed is the failure mode being avoided: a runner
    nobody has executed is a runner that will not work in the window. Every
    assertion below runs off-GPU with no wheel, no card and no pytest.
    """
    failures: List[str] = []
    ran: List[str] = []
    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    # -- arch classification
    check("classify sm86", classify_arch((8, 6)) == "sm86")
    check("classify sm120", classify_arch((12, 0)) == "sm120")

    # -- card selection, on a rig whose NVML and CUDA orders DISAGREE, which is
    #    this rig's real shape. Selection must follow capability, not ordinal.
    @dataclass(frozen=True)
    class _Card:
        name: str
        nvml_index: int
        cuda_ordinal: Optional[int]

    cards = [
        _Card("NVIDIA GeForce RTX 3080", nvml_index=0, cuda_ordinal=1),
        _Card("NVIDIA GeForce RTX 5090", nvml_index=1, cuda_ordinal=0),
        _Card("NVIDIA GeForce RTX 3080", nvml_index=2, cuda_ordinal=2),
    ]
    caps = {0: (12, 0), 1: (8, 6), 2: (8, 6)}
    chosen = select_gate_cards(cards, caps)
    check("both arches selected", set(chosen) == {"sm86", "sm120"})
    check("sm120 is the 5090", "5090" in chosen["sm120"].name)
    check("sm120 took cuda ordinal 0", chosen["sm120"].cuda_ordinal == 0)
    check("sm86 is a 3080", "3080" in chosen["sm86"].name)

    # A card torch cannot see is not a gate candidate.
    masked = select_gate_cards([_Card("RTX 5090", 1, None)], {})
    check("masked card is not selectable", masked == {})

    # -- the falsifier table. THE point of the self-test: the corrected #519
    #    expectation must accept True/False/False and REJECT False/False/False.
    lever_off = FALSIFIER_ARMS[1]
    check(
        "lever-off accepts True/False/False",
        check_arm(Observation(True, False, False), lever_off) == [],
    )
    check(
        "lever-off REJECTS False/False/False (the pre-#519 expectation)",
        check_arm(Observation(False, False, False), lever_off) != [],
    )
    native = FALSIFIER_ARMS[0]
    check(
        "native-default accepts True/True/True",
        check_arm(Observation(True, True, True), native) == [],
    )
    check(
        "native-default REJECTS a wheel with no kernels",
        check_arm(Observation(False, False, False), native) != [],
    )
    # A silent regression to the repack on a native wheel must be caught.
    check(
        "native-default REJECTS type 39 missing from the MMVQ set",
        check_arm(Observation(True, True, False), native) != [],
    )

    # -- probe source carries the ordering hazard fix
    check(
        "probe imports sgl_kernel before probing the marker",
        _PROBE_SRC.index("import sgl_kernel") < _PROBE_SRC.index("hasattr"),
    )

    # -- summary parsing, including the shape that means "nothing ran"
    check(
        "parses a pass line",
        parse_pytest_summary("collecting...\n14 passed, 2 warnings in 3.4s")
        == "14 passed, 2 warnings in 3.4s",
    )
    check(
        "parses a skip line",
        "skipped" in (parse_pytest_summary("14 skipped in 0.2s") or ""),
    )
    check("no summary is None", parse_pytest_summary("") is None)

    # -- the report renders both arches even when one never ran
    rep = Report()
    rep.arches["sm86"] = {"summary": "14 passed", "card": "x"}
    check("renders a missing arch as NOT RUN", "NOT RUN" in rep.render())
    check("renders NOT GREEN by default", "NOT GREEN" in rep.render())

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    rejects = sum(1 for label in ran if "REJECT" in label or "not selectable" in label)
    print(
        f"self-test: OK ({len(ran)} checks, including {rejects} that assert a "
        "gate rejects bad input)"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="hermetic; no GPU")
    ap.add_argument("--run", action="store_true", help="real gates; needs cards")
    ap.add_argument("--out", default="/tmp/398_gate_a")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--python", default="/spinning/htsglang-gpu/.venv/bin/python3")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.run:
        ap.print_help()
        return 2

    try:
        import torch

        from sglang.srt.registry.nvml import identity_map
    except Exception as exc:  # pragma: no cover - environment
        print(f"cannot run: {exc}")
        return 2

    if not torch.cuda.is_available():
        print("cannot run: no CUDA device visible. This gate is GPU-only.")
        return 2

    os.makedirs(args.out, exist_ok=True)
    report = Report()

    cards = identity_map(allow_cuda_init=True).cards
    caps = {
        c.cuda_ordinal: torch.cuda.get_device_capability(c.cuda_ordinal)
        for c in cards
        if c.cuda_ordinal is not None
    }
    chosen = select_gate_cards(cards, caps)
    missing = [a for a in REQUIRED_ARCHES if a not in chosen]
    if missing:
        print(f"cannot run: no visible card for {missing}. Both arches are gates.")
        return 2

    # Gate A, per arch.
    for arch, card in chosen.items():
        done = _run(
            [args.python, "-m", "pytest", GATE_A_TESTS, "-q", "--color=no"],
            {
                "CUDA_VISIBLE_DEVICES": str(card.cuda_ordinal),
                "PYTHONPATH": f"{args.repo}/python",
            },
        )
        summary = parse_pytest_summary(done.stdout + "\n" + done.stderr)
        report.arches[arch] = {
            "summary": summary,
            "returncode": done.returncode,
            "card": card.describe() if hasattr(card, "describe") else str(card),
        }
        with open(os.path.join(args.out, f"gate_a_{arch}.log"), "w") as f:
            f.write(done.stdout + "\n" + done.stderr)
        print(f"[{arch}] {summary}")

    # Gate A', both arms.
    for arm in FALSIFIER_ARMS:
        observed = probe(args.python, arm.env, args.repo)
        problems = check_arm(observed, arm)
        report.falsifier[arm.name] = {
            "observed": observed.as_tuple(),
            "expected": arm.expected,
            "status": "OK" if not problems else "; ".join(problems),
        }
        print(f"[{arm.name}] {observed.as_tuple()} {'OK' if not problems else problems}")

    report.green = all(
        v.get("returncode") == 0 for v in report.arches.values()
    ) and all(v.get("status") == "OK" for v in report.falsifier.values())

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(
            {"arches": report.arches, "falsifier": report.falsifier, "green": report.green},
            f,
            indent=2,
        )
    with open(os.path.join(args.out, "TICKET_BLOCK.md"), "w") as f:
        f.write(report.render() + "\n")
    print()
    print(report.render())
    return 0 if report.green else 1


if __name__ == "__main__":
    raise SystemExit(main())
