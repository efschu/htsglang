# SPDX-License-Identifier: Apache-2.0
"""G4: the GGUF bytes per exchange TAG, for the W71 census (``xchg_residency``).

W71 prices the flip's VRAM peak from a census of bytes per ``(card, group, tag)``, MEASURED on a previous boot.
There is no measured census for a GGUF boot yet, and the expert bank is the one term of it that changes with the
checkpoint format: it holds ``R + C`` expert ROWS per layer, and a GGUF row is not an INT4 row -- the ggml type is
chosen per layer (unsloth UD-IQ4_XS: 2.2217 / 3.0029 / 3.3203 MiB against INT4's 2.4170 MiB), so the same VRAM
budget buys a different number of slots in every layer and the bytes of a tag are the sum over ITS layers of
``slots x that layer's row``, never ``layers x mean row``.

This module computes exactly that, from the GGUF header, and nothing else.  It does not touch the census file or
the solver: the numbers are the EXPERT-BANK terms to swap into a measured census (``swap_expert_bank``), with the
provenance printed beside them (``computed from the header, unmeasured``).  Pure python, stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MIB = float(1 << 20)

PROVENANCE = "computed from the GGUF header (row bytes per layer) and the stated slots; not measured"

#: the INT4 CompressedTensors W4 g128 row of Qwen3.8-Flash-Next, the formula of ``deskq/gguf-nf/calc.py``:
#: three 640x2560 projections, 4 bits per weight + one bf16 group scale per 128
INT4_ROW_FORMULA = "3 * (640 * 2560 / 2 + 640 * 2560 / 128 * 2)"


def int4_row_bytes(ffn: int = 640, hidden: int = 2560, group: int = 128) -> int:
    return int(3 * (ffn * hidden // 2 + ffn * hidden // group * 2))


def layer_row_bytes(files: Sequence[str]) -> Dict[int, int]:
    """``{layer: bytes of ONE expert, all projections}`` from the GGUF headers of ``files``."""
    from sglang.srt.layers.moe import gguf_layout as gl

    return {layer: sum(p.values()) for layer, p in sorted(gl.row_class_bytes(files).items())}


def row_classes(row_by_layer: Mapping[int, int]) -> Dict[int, List[int]]:
    """``{row bytes: [layers]}`` -- the classes (three for the unsloth UD-IQ4_XS export)."""
    out: Dict[int, List[int]] = {}
    for layer, b in sorted(row_by_layer.items()):
        out.setdefault(int(b), []).append(int(layer))
    return out


def tag_of_layer(layer: int, chunk_layers: int) -> str:
    """``weights_<layer // chunk_layers>`` -- the chunk tag of a layer band (``weight_chunk_tag``)."""
    return "weights_%d" % (int(layer) // int(chunk_layers))


def slots_for_budget(budget_mib: float, row_by_layer: Mapping[int, int], layers: Sequence[int]) -> float:
    """Slots PER LAYER a budget buys when every layer of the stage gets the same slot count: the budget over the
    sum of THAT stage's layer rows (``deskq/gguf-nf/calc.py``'s rule, and the reason a mean row is wrong)."""
    total = sum(int(row_by_layer[int(l)]) for l in layers)
    if total <= 0:
        return 0.0
    return float(budget_mib) * MIB / total


def bank_mib_by_tag(
    row_by_layer: Mapping[int, int],
    layers: Sequence[int],
    slots: float,
    chunk_layers: int,
) -> Dict[str, float]:
    """MiB of expert bank per chunk tag: ``slots`` rows in every layer of ``layers``, each at ITS row bytes.

    ``slots`` is the bank's row count (resident + scratch), an integer on a real rank; the fractional value is for
    the budget table."""
    out: Dict[str, float] = {}
    for l in layers:
        t = tag_of_layer(l, chunk_layers)
        out[t] = out.get(t, 0.0) + float(slots) * int(row_by_layer[int(l)]) / MIB
    return out


def swap_expert_bank(
    census_tags: Mapping[str, int],
    old_bank_mib: Mapping[str, float],
    new_bank_mib: Mapping[str, float],
) -> Dict[str, int]:
    """A measured group-on-card tag table (``tag -> MiB``) with its expert-bank term replaced: per tag
    ``measured - old_bank + new_bank``, rounded UP to whole MiB (W71 refuses on the high side only).  A tag the
    census lacks but the new bank names is added; a negative result is a named error -- the measurement did not
    contain the bank it is asked to remove."""
    out = {str(t): int(v) for t, v in census_tags.items()}
    for tag in sorted(set(old_bank_mib) | set(new_bank_mib)):
        have = out.get(tag, 0)
        old = float(old_bank_mib.get(tag, 0.0))
        new = float(new_bank_mib.get(tag, 0.0))
        v = have - old + new
        if v < -0.5:
            raise ValueError(
                "tag %s: the census holds %d MiB, the INT4 bank to remove is %.1f MiB -- not a bank of this census"
                % (tag, have, old)
            )
        out[tag] = int(math.ceil(max(0.0, v)))
    return out


@dataclass(frozen=True)
class Place:
    """One (group, card) holder of expert rows in a form: its layers and its expert budget."""

    label: str
    layers: Tuple[int, ...]
    budget_mib: float


def compare(
    places: Sequence[Place],
    gguf_row: Mapping[int, int],
    int4_row: int,
    chunk_layers: int,
) -> List[dict]:
    """One row per place: slots a budget buys with the INT4 row and with the GGUF rows, and the bank MiB per tag of
    the GGUF form -- the table of the G4 report.  ``int4_row`` is the same on every layer; ``gguf_row`` is per layer."""
    int4_by_layer = {l: int(int4_row) for p in places for l in p.layers}
    out = []
    for p in places:
        s_int4 = slots_for_budget(p.budget_mib, int4_by_layer, p.layers)
        s_gguf = slots_for_budget(p.budget_mib, gguf_row, p.layers)
        # a real bank has whole rows: the floor is what fits
        tags = bank_mib_by_tag(gguf_row, p.layers, math.floor(s_gguf), chunk_layers)
        out.append(
            {
                "place": p.label,
                "layers": len(p.layers),
                "budget_mib": float(p.budget_mib),
                "int4_row_mib": int4_row / MIB,
                "gguf_row_mib": sum(int(gguf_row[l]) for l in p.layers) / len(p.layers) / MIB,
                "slots_int4": s_int4,
                "slots_gguf": s_gguf,
                "tags_gguf_mib": tags,
                "bank_gguf_mib": sum(tags.values()),
            }
        )
    return out


def render_markdown(rows: Sequence[dict]) -> str:
    head = (
        "| place | layers | budget MiB | INT4 row MiB | GGUF mean row MiB | slots INT4 | slots GGUF (floor) "
        "| GGUF bank MiB | tags |\n|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    lines = [head]
    for r in rows:
        tags = ", ".join("%s %.0f" % (t, v) for t, v in sorted(r["tags_gguf_mib"].items(), key=lambda kv: int(kv[0].split("_")[1])))
        lines.append(
            "| %s | %d | %.0f | %.4f | %.4f | %.1f | %d | %.0f | %s |"
            % (
                r["place"], r["layers"], r["budget_mib"], r["int4_row_mib"], r["gguf_row_mib"],
                r["slots_int4"], math.floor(r["slots_gguf"]), r["bank_gguf_mib"], tags,
            )
        )
    return "\n".join(lines)
