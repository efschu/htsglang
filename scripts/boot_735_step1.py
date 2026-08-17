#!/usr/bin/env python3
# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#735 step 1 -- contiguous-as-set. The turnkey boot ticket.

DESIGN_pp_layer_set.md section 9.1: the first boot must NOT be the family
layout. Addressing and transport are separated on purpose, because a failure in
a combined first boot would be ambiguous. Step 1 expresses the ORDINARY
contiguous split through the SET mechanism: ownership stays contiguous so the
existing stage-boundary transport carries it unchanged, while the whole new
path is exercised -- parser, ``make_layers`` set branch, ``local_slot``,
``owned_layer_ids``, pool sizing, sub-pool frames.

WHAT RUNS WHERE
---------------
The three deliberate refusal firings of section 9.2 are parser- and
guard-level. They need no GPU, so this script fires them on the spot and they
are NOT part of the window's work -- run ``--refusals-only`` any time. What
genuinely needs the window is the two-arm byte-identical comparison, and that
is all it needs.

THE SPLIT IS DERIVED, NOT TYPED
-------------------------------
DESIGN 9.1 shows a two-stage example (``"0-31;32-63"``, pp_size 2). The live
cut is THREE stages with ``--pp-stage-ratio 31,17,16`` and
``--pp-attn-stage-ratio 7,5,4``, so the example cannot be copied. Worse, the
ratio is not trivially the boundary set: ``derive_pp_layer_split`` snaps each
boundary so the requested number of FULL-ATTENTION layers lands on each side,
and a snap that moved a boundary would silently break byte-identity.

So this script calls the planner itself and derives the set from the same
function the server uses. On the current checkpoint that yields

    0-30;31-47;48-63        (31 / 17 / 16 layers, 7 / 5 / 4 FA)

which happens to match the plain cumulative reading -- but it is CHECKED here
rather than assumed, and if a future ratio snaps differently this script
follows it instead of booting a mismatched pair.

Device identity is resolved via NVML at run time and printed. Physical indices
are never hardcoded: NVML enumeration order can shift between boots.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))

#: The live cut, from scripts/route_a_631_prod_boot.sh.
PP_STAGE_RATIO = [31, 17, 16]
PP_ATTN_STAGE_RATIO = [7, 5, 4]
NUM_HIDDEN_LAYERS = 64
FULL_ATTENTION_INTERVAL = 4

LAYER_SET_ENV = "SGLANG_PP_LAYER_SET"


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def is_full_attention_mask(
    num_layers: int = NUM_HIDDEN_LAYERS, interval: int = FULL_ATTENTION_INTERVAL
) -> List[bool]:
    return [i % interval == interval - 1 for i in range(num_layers)]


def derive_layer_set(
    stage_ratio: Sequence[int] = PP_STAGE_RATIO,
    attn_ratio: Sequence[int] = PP_ATTN_STAGE_RATIO,
    num_layers: int = NUM_HIDDEN_LAYERS,
) -> Tuple[str, List[Tuple[int, int]], List[int]]:
    """``(env_value, [(lo, hi)...], per_stage_counts)`` from the LIVE planner.

    Uses ``derive_pp_layer_split`` -- the same function the server calls -- so
    the set cannot drift from the ratio it is supposed to reproduce.
    """
    from sglang.srt.distributed.utils import derive_pp_layer_split

    counts = derive_pp_layer_split(
        list(stage_ratio),
        is_full_attention=is_full_attention_mask(num_layers),
        num_hidden_layers=num_layers,
        attn_scores=list(attn_ratio),
    )
    ranges, base = [], 0
    for c in counts:
        ranges.append((base, base + c - 1))
        base += c
    return ";".join(f"{lo}-{hi}" for lo, hi in ranges), ranges, counts


def verify_split_is_contiguous_and_complete(
    ranges: Sequence[Tuple[int, int]], num_layers: int = NUM_HIDDEN_LAYERS
) -> None:
    """Step 1's whole premise: ownership must still be CONTIGUOUS.

    If this fails the run is not step 1 at all -- it is step 2 without the
    wire, and the existing stage-boundary transport would not carry it.
    """
    covered: List[int] = []
    for lo, hi in ranges:
        covered.extend(range(lo, hi + 1))
    if sorted(covered) != list(range(num_layers)):
        raise SystemExit(
            f"REFUSING: derived set does not cover [0, {num_layers}) exactly "
            f"once: {ranges}"
        )
    for (_, hi), (lo, _) in zip(ranges, ranges[1:]):
        if lo != hi + 1:
            raise SystemExit(
                f"REFUSING: derived set is NOT contiguous at {hi} -> {lo}. "
                f"Step 1 requires contiguous ownership; a gapped set is step 2 "
                f"and needs the per-layer wire."
            )


# ---------------------------------------------------------------------------
# Device identity
# ---------------------------------------------------------------------------


def nvml_inventory() -> List[Dict[str, object]]:
    """Physical index -> card, resolved at run time. Never hardcoded."""
    try:
        from sglang.srt.registry.nvml import identity_map

        return [
            {
                "nvml_index": c.nvml_index,
                "name": c.name,
                "uuid": c.uuid,
                "total_mib": c.total_mib,
            }
            for c in identity_map()
        ]
    except Exception as exc:  # noqa: BLE001 - reported, not fatal for --dry-run
        return [{"error": f"{type(exc).__name__}: {exc}"}]


# ---------------------------------------------------------------------------
# The three deliberate refusal firings (DESIGN 9.2) -- no GPU needed
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RefusalResult:
    name: str
    expected: str
    fired: bool
    message: str

    def line(self) -> str:
        mark = "FIRED " if self.fired else "SILENT"
        return f"  [{mark}] {self.name}: expected {self.expected}\n           {self.message}"


def fire_refusals() -> List[RefusalResult]:
    """Trigger each guard once, deliberately. A guard nobody has seen fire is
    not evidence (DESIGN 9.2)."""
    out: List[RefusalResult] = []

    # (1) any layer set + prefill/decode disaggregation -> NotImplementedError
    try:
        from sglang.srt.distributed.utils import refuse_noncontiguous_layer_descriptor

        refuse_noncontiguous_layer_descriptor({3: 0, 7: 1}, "boot_735_step1 probe")
        out.append(
            RefusalResult("layer-set + PD disagg", "NotImplementedError", False, "did NOT raise")
        )
    except NotImplementedError as exc:
        out.append(
            RefusalResult("layer-set + PD disagg", "NotImplementedError", True, str(exc)[:220])
        )
    except Exception as exc:  # noqa: BLE001
        out.append(
            RefusalResult(
                "layer-set + PD disagg",
                "NotImplementedError",
                False,
                f"WRONG TYPE {type(exc).__name__}: {exc}",
            )
        )

    # (2) layer set + an unconverted model loop -> RuntimeError naming the layer
    try:
        import torch

        from sglang.srt.layers.utils.common import PPMissingLayer

        placeholder = PPMissingLayer(unowned_layer_id=42)
        placeholder.forward(torch.zeros(1))
        out.append(
            RefusalResult("unconverted forward loop", "RuntimeError", False, "did NOT raise")
        )
    except RuntimeError as exc:
        out.append(
            RefusalResult("unconverted forward loop", "RuntimeError", True, str(exc)[:220])
        )
    except Exception as exc:  # noqa: BLE001
        out.append(
            RefusalResult(
                "unconverted forward loop",
                "RuntimeError",
                False,
                f"WRONG TYPE {type(exc).__name__}: {exc}",
            )
        )

    # (3) a malformed set -> PPLayerSetError naming the layer and stages
    try:
        from sglang.srt.distributed.utils import PPLayerSetError, parse_pp_layer_sets

        # Layer 30 omitted: the set covers 63 of 64 layers.
        parse_pp_layer_sets("0-29;31-47;48-63", NUM_HIDDEN_LAYERS, 3)
        out.append(RefusalResult("malformed layer set", "PPLayerSetError", False, "did NOT raise"))
    except PPLayerSetError as exc:
        out.append(RefusalResult("malformed layer set", "PPLayerSetError", True, str(exc)[:220]))
    except Exception as exc:  # noqa: BLE001
        out.append(
            RefusalResult(
                "malformed layer set",
                "PPLayerSetError",
                False,
                f"WRONG TYPE {type(exc).__name__}: {exc}",
            )
        )

    return out


# ---------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Arm:
    name: str
    env_overrides: Dict[str, str]

    def env(self, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        e = dict(base if base is not None else os.environ)
        for k, v in self.env_overrides.items():
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return e


def arms(layer_set: str) -> List[Arm]:
    """Baseline first. The set arm must differ ONLY in the env var."""
    return [
        Arm("baseline (env unset)", {LAYER_SET_ENV: None}),
        Arm("contiguous-as-set", {LAYER_SET_ENV: layer_set}),
    ]


#: Substrings that make an arm a FAILURE regardless of output equality. The
#: PPMissingLayer guard must stay SILENT in step 1: with contiguous ownership
#: no placeholder sits inside the iterated interval, so a firing guard means
#: the loop conversion is wrong -- not the layout (DESIGN 9.1).
FATAL_LOG_MARKERS = (
    "PPMissingLayer",
    "owned_layer_ids",
    "PPLayerSetError",
)


def scan_log_for_silent_guard(text: str) -> List[str]:
    return [m for m in FATAL_LOG_MARKERS if m in text]


def compare_arms(baseline_out: str, set_out: str) -> Tuple[bool, str]:
    if baseline_out == set_out:
        return True, "byte-identical"
    a, b = baseline_out.splitlines(), set_out.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"first divergence at line {i}:\n  baseline: {x!r}\n  set:      {y!r}"
    return False, f"length differs: baseline {len(a)} lines, set {len(b)} lines"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plan_report(layer_set: str, ranges, counts) -> str:
    fa = is_full_attention_mask()
    lines = [
        "#735 STEP 1 -- contiguous-as-set. BOOT TICKET",
        "",
        f"  {LAYER_SET_ENV}={layer_set!r}",
        f"  derived from --pp-stage-ratio {','.join(map(str, PP_STAGE_RATIO))}"
        f" and --pp-attn-stage-ratio {','.join(map(str, PP_ATTN_STAGE_RATIO))}",
        "",
        "  stage  layers        count  FA",
    ]
    for i, (lo, hi) in enumerate(ranges):
        nfa = sum(1 for L in range(lo, hi + 1) if fa[L])
        lines.append(f"    {i}    {lo:>2}-{hi:<3}       {counts[i]:>3}   {nfa}")
    lines += [
        "",
        "  ACCEPTANCE (all three required):",
        "    1. byte-identical generated output vs the same run with the env",
        "       var unset, same seed and prompt;",
        "    2. identical KV pool sizing in the startup log;",
        "    3. NO PPMissingLayer RuntimeError -- the guard must stay SILENT.",
        "       Ownership is contiguous here, so no placeholder sits inside the",
        "       iterated interval. A firing guard means the loop conversion is",
        "       wrong, not the layout.",
        "",
        "  Device identity (NVML, resolved now -- indices are not stable across",
        "  boots, so read this rather than assuming):",
    ]
    for card in nvml_inventory():
        lines.append(f"    {json.dumps(card)}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--refusals-only", action="store_true", help="fire the 9.2 guards and exit")
    p.add_argument("--json", action="store_true", help="machine-readable plan")
    args = p.parse_args(argv)

    layer_set, ranges, counts = derive_layer_set()
    verify_split_is_contiguous_and_complete(ranges)

    if args.json:
        print(
            json.dumps(
                {
                    "layer_set": layer_set,
                    "ranges": ranges,
                    "counts": counts,
                    "stage_ratio": PP_STAGE_RATIO,
                    "attn_stage_ratio": PP_ATTN_STAGE_RATIO,
                    "cards": nvml_inventory(),
                },
                indent=2,
            )
        )
        return 0

    if args.refusals_only:
        print("DESIGN 9.2 -- deliberate refusal firings (no GPU required):")
        results = fire_refusals()
        for r in results:
            print(r.line())
        ok = all(r.fired for r in results)
        print(f"\n{'ALL THREE FIRED' if ok else 'INCOMPLETE'}: {sum(r.fired for r in results)}/3")
        return 0 if ok else 1

    print(plan_report(layer_set, ranges, counts))
    if args.dry_run:
        print("\n  --dry-run: nothing launched.")
        return 0

    print(
        "\n  This script does not launch the server itself: the two arms need a\n"
        "  GPU window and the boot command lives in\n"
        "  scripts/route_a_631_prod_boot.sh. Run each arm with that script,\n"
        "  setting only the env var above for the second arm, then compare the\n"
        "  generated text and the KV pool sizing line.\n"
        "\n  Fire the 9.2 guards now with --refusals-only; they need no GPU."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
