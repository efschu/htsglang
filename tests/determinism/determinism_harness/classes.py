# SPDX-License-Identifier: Apache-2.0
"""The byte-identity CLASS registry -- the enforceable contract of #124.

Three classes, each with exact assertion semantics, distilled from the
fork's manually-validated gates (memory files ``weightless-kv-lane`` --
explicitly "auch die #124-Harness-Spec" --, ``moe-expert-offload``,
``marlin-offload-nicht-bit-identisch``, ``hetero-spec-determinismus``):

MACHINE_ZERO
    Bit-exact: identical token trajectory AND ``torch.equal`` logits
    (max |delta| = 0) vs. the reference. Applies to head-local prefill
    vs. TP=1 solo and to graph-weightless vs. eager-weightless (that lane
    has NO capture-gated dual-stream branches, so graph == eager exactly --
    unlike the MoE-offload capture path, see ``EXCLUDED_CASES``).

DECODE_CLASS
    Argmax-identical decode trajectory vs. solo/reference (0 flips over the
    gate window), divergence only within the fp-reassociation band (bounded
    per-step |delta|), and non-compounding (stationary noise, no drift)
    until a genuine near-tie. The discriminator is the argmax-clean
    trajectory over N tokens, NOT bit-exact logits.

SELF_DET_NEAR_TIE
    The offload class. Two obligations: (1) SELF-DETERMINISM -- run == run
    bit-identical at fixed config/seed; (2) vs. the no-offload reference,
    divergence ONLY at a genuine near-tie (reference top-margin to the
    flipped-to token <= near_tie_margin), with in-band deltas up to the
    fork; post-fork cascade is legitimate. Plus (GPU-window evidence, not a
    CPU assertion): quality-neutral, ppl within noise (#120).

    Explicitly NOT bit-exact to no-offload -- for BOTH offload paths.
    "FP8 offload is byte-identical" was an overclaim, retracted by its own
    author (``0fb3d8007``): the compacted resident+scratch layout changes
    the fp reduction order (same class as changing TP degree or batch
    size); a 256-token rerun matched only 118/256 with the first divergence
    at token 115 on a near-tie. FP8 and marlin differ only in magnitude
    (FP8 sub-ULP, marlin ~1e-2 from moe_align tiling). Do not re-encode the
    overclaim; ``tests/determinism/test_matrix.py`` guards against it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from .primitives import (
    check_argmax_clean_trajectory,
    check_delta_band,
    check_machine_zero,
    check_near_tie_only_divergence,
    check_non_compounding,
    check_self_determinism,
)
from .trajectory import Trajectory, Verdict, combine

__all__ = ["ByteIdentityClass", "ClassSpec", "CLASS_SPECS", "check_class"]


class ByteIdentityClass(enum.Enum):
    MACHINE_ZERO = "machine_zero"
    DECODE_CLASS = "decode_class"
    SELF_DET_NEAR_TIE = "self_det_near_tie"


@dataclass(frozen=True)
class ClassSpec:
    """Human- and machine-readable statement of a class's assertion semantics."""

    cls: ByteIdentityClass
    summary: str
    assertions: str
    #: which trajectories check_class needs: "ref+test" or "ref+test+rerun"
    required_inputs: str
    #: does check_class require band / near_tie_margin parameters?
    needs_band: bool
    needs_near_tie_margin: bool
    #: additional GPU-window evidence outside the CPU assertion (or None)
    gpu_evidence: Optional[str]
    provenance: str


CLASS_SPECS = {
    ByteIdentityClass.MACHINE_ZERO: ClassSpec(
        cls=ByteIdentityClass.MACHINE_ZERO,
        summary="bit-exact vs. reference: identical tokens, torch.equal logits, max|d|=0",
        assertions="check_machine_zero(ref, test)",
        required_inputs="ref+test",
        needs_band=False,
        needs_near_tie_margin=False,
        gpu_evidence=None,
        provenance=(
            "memory weightless-kv-lane: head-local single-shot prefill == TP=1 solo; "
            "graph-weightless == eager-weightless (#133 'Byte maschinen-null', "
            "B2 part-2 captured streaming max|d|=0)"
        ),
    ),
    ByteIdentityClass.DECODE_CLASS: ClassSpec(
        cls=ByteIdentityClass.DECODE_CLASS,
        summary=(
            "argmax-identical decode trajectory (0 flips over N tokens), per-step "
            "|d| within the fp-reassociation band, non-compounding"
        ),
        assertions=(
            "check_argmax_clean_trajectory(ref, test, near_tie_margin) AND "
            "check_delta_band(ref, test, band) AND "
            "check_non_compounding(ref, test, band)"
        ),
        required_inputs="ref+test",
        needs_band=True,
        needs_near_tie_margin=True,
        gpu_evidence=None,
        provenance=(
            "memory weightless-kv-lane: cross-rank-merge paths (decode + chunked "
            "prefill with prefix via cp_lse LSE merge) are argmax-identical to solo "
            "but not bit-exact (fp reassociation); B0 block=64 control 0 flips, "
            "B1 streaming-vs-resident 96/96 argmax-identical"
        ),
    ),
    ByteIdentityClass.SELF_DET_NEAR_TIE: ClassSpec(
        cls=ByteIdentityClass.SELF_DET_NEAR_TIE,
        summary=(
            "run==run bit-identical (self-determinism) AND vs. no-offload reference "
            "divergence only at genuine near-ties; NOT bit-exact to no-offload"
        ),
        assertions=(
            "check_self_determinism(test, rerun) AND "
            "check_near_tie_only_divergence(ref, test, band, near_tie_margin)"
        ),
        required_inputs="ref+test+rerun",
        needs_band=True,
        needs_near_tie_margin=True,
        gpu_evidence=(
            "quality-neutral: perplexity within noise vs. no-offload (#120 bar); "
            "coherence; self-det 5/5 across repeated boots"
        ),
        provenance=(
            "memory marlin-offload-nicht-bit-identisch + moe-expert-offload: BOTH "
            "offload paths (FP8 sub-ULP, marlin ~1e-2) are self-deterministic with "
            "near-tie-only divergence; 'FP8 byte-identical' retracted by author "
            "0fb3d8007 (256-token rerun 118/256, fork at token 115 near-tie)"
        ),
    ),
}


def check_class(
    cls: ByteIdentityClass,
    *,
    ref: Trajectory,
    test: Trajectory,
    rerun: Optional[Trajectory] = None,
    band: Optional[float] = None,
    near_tie_margin: Optional[float] = None,
) -> Verdict:
    """Run the full assertion bundle for a byte-identity class.

    ``ref``   -- the reference run (e.g. TP=1 solo, eager, no-offload).
    ``test``  -- the run under test.
    ``rerun`` -- a second run of the SAME test config (same boot, flags,
                 pinned seed); required by SELF_DET_NEAR_TIE.

    Missing band parameters for a class that needs them are a harness
    configuration error (raise), not a data verdict. A missing ``rerun`` for
    SELF_DET_NEAR_TIE is a failing verdict: without run==run evidence the
    class is simply not established.
    """
    spec = CLASS_SPECS[cls]
    if spec.needs_band and band is None:
        raise ValueError(f"{cls.value} requires an explicit fp band")
    if spec.needs_near_tie_margin and near_tie_margin is None:
        raise ValueError(f"{cls.value} requires an explicit near_tie_margin")

    if cls is ByteIdentityClass.MACHINE_ZERO:
        return check_machine_zero(ref, test)

    if cls is ByteIdentityClass.DECODE_CLASS:
        subs = [
            check_argmax_clean_trajectory(ref, test, near_tie_margin),
            check_delta_band(ref, test, band),
            check_non_compounding(ref, test, band),
        ]
        return combine("decode_class", subs)

    if cls is ByteIdentityClass.SELF_DET_NEAR_TIE:
        subs = []
        if rerun is None:
            subs.append(
                Verdict(
                    ok=False,
                    check="self_determinism",
                    message=(
                        "missing rerun trajectory: SELF_DET_NEAR_TIE requires two "
                        "runs of the identical config to establish run==run"
                    ),
                )
            )
        else:
            subs.append(check_self_determinism(test, rerun))
        subs.append(check_near_tie_only_divergence(ref, test, band, near_tie_margin))
        return combine("self_det_near_tie", subs)

    raise ValueError(f"unknown class {cls!r}")
