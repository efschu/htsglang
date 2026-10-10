# SPDX-License-Identifier: Apache-2.0
"""NF-GGUF G5 (2026-10-09): the expert row PER LAYER where the planner used the mean.

THE GAP. A UD GGUF mixes quant types ACROSS layers: on the unsloth Qwen3.8-Flash-Next
UD-IQ4_XS three row classes (header, ``gguf-nf/calc.py``): 2 329 600 B per expert and
layer on 43 layers, 3 148 800 on layers 4, 30, 46 and 47, 3 481 600 on layer 2. Every
planner term that priced "stage layers x mean row" is right for a stage that holds ALL
layers (the D ranks: n x mean = the sum) and wrong for a P stage that holds a SLICE:
the cut 29/11/8 over- or under-prices a stage by the classes its slice happens to
contain (calc.py against the x177 budgets: PP0 +144 MiB, PP1 +71, PP2 -350).

THE FORM. A checkpoint whose layers carry unequal expert bytes hands the planner a
:class:`LayerRows`: a ``float`` whose VALUE is the mean (so every reader that was not
changed keeps its old number, bit for bit) plus the per-layer tuple. A reader that
prices a stage slice calls :func:`stage_total` / :func:`stage_mean`, which sum the slice
when the vector is there and compute ``stage_layers[s] x row`` otherwise. A checkpoint
whose layers are equal (every INT4/AWQ/FP8 release checkpoint) NEVER gets a vector --
:func:`rows_from_layer_bytes` returns a plain float -- so those paths are byte-identical
to before this module existed.

The stage layers are contiguous from layer 0: stage ``s`` holds layers
``[sum(stage_layers[:s]), sum(stage_layers[:s+1]))`` (the contiguous cut every PP stage
of this tree is).

Stdlib only; nothing here reads a file.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


class LayerRows(float):
    """Mean row (the float value) + the per-layer values (``per_layer``), same unit."""

    per_layer: Tuple[float, ...]

    def __new__(cls, per_layer: Sequence[float], mean: Optional[float] = None) -> "LayerRows":
        vals = tuple(float(x) for x in per_layer)
        if not vals:
            raise ValueError("LayerRows: no layers")
        obj = float.__new__(cls, (sum(vals) / len(vals)) if mean is None else float(mean))
        obj.per_layer = vals
        return obj

    def scaled(self, factor: float) -> "LayerRows":
        return LayerRows(tuple(v * float(factor) for v in self.per_layer),
                         mean=float(self) * float(factor))


def is_layer_vector(row: object) -> bool:
    return isinstance(row, LayerRows)


def nonuniform(per_layer: Sequence[float]) -> bool:
    """True when the layers do NOT all carry the same value."""
    vals = [float(x) for x in per_layer]
    return bool(vals) and (max(vals) != min(vals))


def rows_from_layer_bytes(per_layer: Sequence[float], *, scale: float = 1.0):
    """``per_layer * scale`` as a :class:`LayerRows` when the layers differ, else the
    plain mean float (every layer equal: the old scalar form, bit for bit)."""
    vals = [float(x) for x in per_layer]
    if not vals:
        return 0.0
    if nonuniform(vals):
        return LayerRows(vals).scaled(scale) if scale != 1.0 else LayerRows(vals)
    return (sum(vals) / len(vals)) * float(scale) if scale != 1.0 else sum(vals) / len(vals)


def stage_slice(stage_layers: Sequence[int], s: int) -> Tuple[int, int]:
    """``(first layer, one past the last)`` of stage ``s`` of a contiguous cut."""
    start = sum(int(x) for x in stage_layers[:s])
    return start, start + int(stage_layers[s])


def stage_total(row: object, stage_layers: Sequence[int], s: int) -> float:
    """What stage ``s`` pays for its layers' rows: the sum over its slice of the layer
    vector, or ``stage_layers[s] x row`` for a scalar (the pre-G5 product, exactly)."""
    if isinstance(row, LayerRows):
        a, b = stage_slice(stage_layers, s)
        per = row.per_layer
        if b <= len(per):
            return float(sum(per[a:b]))
        # a cut that names more layers than the vector holds: fall back to the mean
        return int(stage_layers[s]) * float(row)
    return int(stage_layers[s]) * float(row)


def stage_mean(row: object, stage_layers: Sequence[int], s: int) -> float:
    """:func:`stage_total` per layer of the stage (the per-layer cost a model that
    multiplies by the stage's layer count prices it at)."""
    n = max(1, int(stage_layers[s]))
    if isinstance(row, LayerRows):
        return stage_total(row, stage_layers, s) / n
    return float(row)


def stage_means(row: object, stage_layers: Sequence[int]) -> Tuple[float, ...]:
    return tuple(stage_mean(row, stage_layers, s) for s in range(len(stage_layers)))


def slot_of(expert_layer_rows: object, num_experts: int) -> object:
    """``expert_layer_rows / num_experts`` keeping the per-layer vector."""
    e = max(1, int(num_experts))
    if isinstance(expert_layer_rows, LayerRows):
        return LayerRows(tuple(v / e for v in expert_layer_rows.per_layer),
                         mean=float(expert_layer_rows) / e)
    return float(expert_layer_rows) / e


def describe(row: object) -> Optional[str]:
    """One short text for a log line, or None for a scalar."""
    if not isinstance(row, LayerRows):
        return None
    classes = {}
    for v in row.per_layer:
        classes[round(v, 4)] = classes.get(round(v, 4), 0) + 1
    return "per-layer %s" % ", ".join("%.4f x%d" % (k, n) for k, n in sorted(classes.items()))
