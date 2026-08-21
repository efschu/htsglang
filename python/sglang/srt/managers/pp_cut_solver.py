# SPDX-License-Identifier: Apache-2.0
"""#785/#702: what a PP cut is worth, in tokens, before booting it.

WHY THIS CAN EXIST NOW. The PP-phase KV pool is a min-reduce over the ranks,
and every term that decides a rank's share is a function of the CUT:

* its weight layout -- ``layout_pp``, which #785 makes computable from a
  meta-device model without loading a byte;
* its arming floor, because the floor carries the arena tail and the tail is
  ``max(0, layout_pp - layout_tp)``;
* its per-token cell, because full attention sits on every Nth layer, so a
  contiguous cut DETERMINES how many attention layers a stage holds.

Until the tail could be derived, the middle term could only be learned by
booting, which is why cut selection was a hardware experiment. It is now a
desk computation, and a boot is what CHECKS it rather than what performs it.

NO RIG CONSTANTS LIVE HERE. Every number that describes a machine -- card
budgets, the per-layer weight mass, the residual per-rank overhead -- is a
field on :class:`CutModel`, calibrated from the instruments of a boot that
actually ran on that machine. A solver carrying one rig's fitted vector would
be exactly the transfer this corpus forbids; the shape is general, the numbers
are measured, and they are measured per rig.

WHAT IT DOES NOT MODEL, STATED SO IT IS NOT MISTAKEN FOR COMPLETE. The
residual term is affine in the layer and attention counts. That is a fit, not
a derivation: it lumps mamba/GDN state, draft weights, activation buffers and
graph memory into two coefficients. It reproduces the boot it was fitted to
almost exactly, and it is the part of this model most likely to be wrong when
extrapolated far -- so a chosen cut is booted and re-read, never shipped on
the model's word.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

MIB = 1048576.0


@dataclasses.dataclass(frozen=True)
class RankOutcome:
    layout_pp_mib: float
    arena_tail_mib: float
    arming_floor_mib: float
    residual_mib: float
    kv_available_mib: float
    cell_bytes: int
    tokens: int


@dataclasses.dataclass(frozen=True)
class CutModel:
    """A rig, described in the terms a cut moves.

    ``budget_mib``      per-rank VRAM budget (``--rank-gpu-memory-mib``).
    ``layout_tp_mib``   per-rank TP weight layout; a function of the FLIP
                        VECTOR, not of the cut, so it is constant here.
    ``per_layer_mib``   weight mass of one transformer layer.
    ``embed_mib``       weight mass carried only by the FIRST stage.
    ``lm_head_mib``     weight mass carried only by the LAST stage.
    ``bytes_per_token_per_attn_layer``
                        the KV cell of a single full-attention layer, so a
                        stage's cell is this times its attention count.
    ``attn_layer_ids``  indices of the full-attention layers, which is what
                        makes the attention split implied rather than chosen.
    ``floor_base_mib``  the arming floor a rank holds with NO arena tail.
    ``floor_over_tail_mib``
                        what the floor adds ON TOP of a tail (corridor law +
                        arming margin); the floor is the larger of this plus
                        the tail, and ``floor_base_mib``.
    ``residual_*``      the affine fit described in the module docstring.
    """

    budget_mib: Tuple[float, ...]
    layout_tp_mib: Tuple[float, ...]
    per_layer_mib: float
    embed_mib: float
    lm_head_mib: float
    bytes_per_token_per_attn_layer: int
    attn_layer_ids: Tuple[int, ...]
    floor_base_mib: float
    floor_over_tail_mib: float
    residual_const_mib: float
    residual_per_layer_mib: float
    residual_per_attn_mib: float

    @property
    def n_ranks(self) -> int:
        return len(self.budget_mib)

    @property
    def n_layers(self) -> int:
        return max(self.attn_layer_ids) + 1 if self.attn_layer_ids else 0

    def attn_counts(self, cut: Sequence[int]) -> List[int]:
        """How many full-attention layers each stage holds.

        NOT A FREE PARAMETER. Attention sits at fixed model positions, so a
        contiguous cut fixes this. Treating it as independent invents cuts
        that cannot exist -- a 16-layer stage holding 3 attention layers when
        every 4th layer is attention.
        """
        edges, running = [], 0
        for width in cut[:-1]:
            running += int(width)
            edges.append(running)
        counts = [0] * len(cut)
        for layer in self.attn_layer_ids:
            stage = 0
            while stage < len(edges) and layer >= edges[stage]:
                stage += 1
            counts[stage] += 1
        return counts

    def layout_pp_mib(self, cut: Sequence[int]) -> List[float]:
        out = [int(w) * self.per_layer_mib for w in cut]
        out[0] += self.embed_mib
        out[-1] += self.lm_head_mib
        return out

    def evaluate(self, cut: Sequence[int]) -> Optional[List[RankOutcome]]:
        """Per-rank outcome, or ``None`` for a cut that cannot be built."""
        if len(cut) != self.n_ranks or any(int(w) < 1 for w in cut):
            return None
        if sum(int(w) for w in cut) != self.n_layers:
            return None
        counts = self.attn_counts(cut)
        if min(counts) < 1:
            # A stage with no full-attention layer has no KV cell at all; the
            # min-reduce cannot be expressed over it.
            return None
        layouts = self.layout_pp_mib(cut)
        out = []
        for rank in range(self.n_ranks):
            tail = max(0.0, layouts[rank] - self.layout_tp_mib[rank])
            floor = max(self.floor_base_mib, self.floor_over_tail_mib + tail)
            residual = (
                self.residual_const_mib
                + self.residual_per_layer_mib * int(cut[rank])
                + self.residual_per_attn_mib * counts[rank]
            )
            available = self.budget_mib[rank] - layouts[rank] - floor - residual
            cell = self.bytes_per_token_per_attn_layer * counts[rank]
            tokens = int(available * MIB / cell) if available > 0 else 0
            out.append(
                RankOutcome(
                    layout_pp_mib=layouts[rank],
                    arena_tail_mib=tail,
                    arming_floor_mib=floor,
                    residual_mib=residual,
                    kv_available_mib=available,
                    cell_bytes=cell,
                    tokens=tokens,
                )
            )
        return out

    def pool_tokens(self, cut: Sequence[int]) -> int:
        """The global pool: a min-reduce, which is why balance beats total.

        A cut that hands one rank an enormous share and starves another is
        worth exactly what the starved rank can fund. That is the whole reason
        the shipped cut sat at 161378 tokens while two of its three ranks could
        each have funded four times that.
        """
        outcomes = self.evaluate(cut)
        return 0 if outcomes is None else min(o.tokens for o in outcomes)

    def rank_cuts(self, limit: int = 10) -> List[Tuple[int, List[int]]]:
        """Every buildable cut, best first. Exhaustive: the space is tiny."""
        found: List[Tuple[int, List[int]]] = []
        for cut in _compositions(self.n_layers, self.n_ranks):
            tokens = self.pool_tokens(cut)
            if tokens:
                found.append((tokens, list(cut)))
        found.sort(key=lambda row: (-row[0], row[1]))
        return found[:limit]


def _compositions(total: int, parts: int):
    if parts == 1:
        yield [total]
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield [first] + rest


def calibrate_residuals(
    model: CutModel, cut: Sequence[int], measured_available_mib: Sequence[float]
) -> CutModel:
    """Fit the residual term to a boot's own KV-available column.

    Three ranks give three equations for three coefficients, so this is exactly
    determined -- calibration, not validation. It is stated that way rather
    than dressed up: what checks the fit is booting a cut it did not see.
    """
    counts = model.attn_counts(cut)
    layouts = model.layout_pp_mib(cut)
    rows, targets = [], []
    for rank in range(model.n_ranks):
        tail = max(0.0, layouts[rank] - model.layout_tp_mib[rank])
        floor = max(model.floor_base_mib, model.floor_over_tail_mib + tail)
        rows.append([1.0, float(cut[rank]), float(counts[rank])])
        targets.append(
            model.budget_mib[rank]
            - layouts[rank]
            - floor
            - float(measured_available_mib[rank])
        )
    const, per_layer, per_attn = _solve3(rows, targets)
    return dataclasses.replace(
        model,
        residual_const_mib=const,
        residual_per_layer_mib=per_layer,
        residual_per_attn_mib=per_attn,
    )


def _solve3(
    rows: List[List[float]], targets: List[float]
) -> Tuple[float, float, float]:
    """Gaussian elimination on a 3x3. Refuses a singular system."""
    m = [list(rows[i]) + [targets[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-9:
            raise ValueError(
                "the calibration system is singular: the three ranks do not "
                "differ enough in layer and attention count to separate the "
                "residual terms"
            )
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= factor * m[col][c]
    return tuple(m[i][3] / m[i][i] for i in range(3))  # type: ignore[return-value]


def describe(model: CutModel, cut: Sequence[int]) -> Dict[str, object]:
    """A cut's outcome in one flat dict, for logs and for tests."""
    outcomes = model.evaluate(cut)
    if outcomes is None:
        return {"cut": list(cut), "buildable": False}
    return {
        "cut": list(cut),
        "buildable": True,
        "attn": model.attn_counts(cut),
        "layout_pp_mib": [round(o.layout_pp_mib, 2) for o in outcomes],
        "arena_tail_mib": [round(o.arena_tail_mib, 2) for o in outcomes],
        "arming_floor_mib": [round(o.arming_floor_mib, 2) for o in outcomes],
        "kv_available_mib": [round(o.kv_available_mib, 1) for o in outcomes],
        "cell_bytes": [o.cell_bytes for o in outcomes],
        "tokens": [o.tokens for o in outcomes],
        "pool_tokens": min(o.tokens for o in outcomes),
    }
