"""#707: the PP-phase holdback in closed form (Slot-2's derivation).

The per-rank holdback measured at the instrument boot — 45.143 / 44.074 /
60.258 % — is not a reserve and not an empirical fudge. It is a **cap**, from
``phase_flip_seam_reserve.floor_allowed_tokens``:

    allowed_tokens = id_space + (free_at_measure - arming_floor - margin) / cell
    holdback_frac  = 1 - allowed_tokens / (profiled_bytes / cell)

Because it is a cap rather than a subtraction, the resting free column still
*holds* the arming floor; nothing is set aside twice. The PP phase pays it
because a flip must ARM out of that column, and the TP pass measures 0.000 % on
every rank for the matching reason: there is no flip to arm from.

**Why the binder holds back most, which is structural and not a coincidence.**
The binding rank is the one whose free column has run down to its arming floor,
so its bracket ``(free_at_measure - arming_floor - margin)`` is ~zero and its
allowed tokens collapse to ``id_space``. At the instrument boot PP2's bracket is
**0.0 MiB** against PP0's 2164.8 and PP1's 260.2. Simultaneously PP2 has the
smallest cell (4 attention layers), hence the largest *raw* token capacity — so
the largest holdback fraction and the binding role co-occur by construction.
Reading the big holdback as PP2 being wasteful is exactly backwards.

Provenance, which this module keeps explicit: ``cell`` is config-derived,
``id_space`` / ``free_at_measure`` / ``margin`` come from the seam record and
``arming_floor`` from #676 — all exact for a booted layout. Only the layout
shift of ``free_at_measure`` to an unbooted cut is extrapolated, and it is
arithmetic on weights and GDN state rather than a fit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

MIB = 1024.0 * 1024.0


class SeamHoldbackError(ValueError):
    """A holdback question that cannot be answered as posed."""


@dataclasses.dataclass(frozen=True)
class SeamRecord:
    """Per-rank seam facts for ONE booted layout. All exact."""

    id_space_tokens: float
    #: ``free_at_measure - arming_floor - margin`` per rank, in MiB. Held as the
    #: combined bracket because that is what the cap consumes; splitting it adds
    #: no information the formula uses.
    bracket_mib: tuple[float, ...]
    cell_bytes: tuple[int, ...]

    def allowed_tokens(self) -> tuple[float, ...]:
        return tuple(
            self.id_space_tokens + (b * MIB) / c
            for b, c in zip(self.bracket_mib, self.cell_bytes)
        )


@dataclasses.dataclass(frozen=True)
class WeightTerms:
    """Per-family checkpoint terms. Census-measured, not fitted."""

    attn_mib_per_layer: float = 374.2
    linear_mib_per_layer: float = 476.2
    mamba_mib_per_linear_layer: float = 51.20


#: Census terms for this checkpoint family. A module-level singleton rather than
#: a default-argument call, so every caller shares one definition.
CENSUS_TERMS = WeightTerms()


def holdback_fraction(
    allowed_tokens: float, profiled_bytes: float, cell_bytes: int
) -> float:
    """``1 - allowed / (profiled / cell)``. Slot-2's closed form, verbatim."""
    if cell_bytes <= 0:
        raise SeamHoldbackError("cell must be positive.")
    profiled_tokens = float(profiled_bytes) / float(cell_bytes)
    if profiled_tokens <= 0.0:
        raise SeamHoldbackError("profiled capacity must be positive.")
    return 1.0 - float(allowed_tokens) / profiled_tokens


def shift_bracket(
    booted_bracket_mib: Sequence[float],
    booted_counts: Sequence[int],
    booted_attn: Sequence[int],
    counts: Sequence[int],
    attn: Sequence[int],
    terms: WeightTerms | None = None,
) -> tuple[float, ...]:
    """Move each rank's bracket to another layout. THE ONLY EXTRAPOLATED STEP.

    ``free_at_measure(cut) = free_at_measure(booted) - dweights - dmamba``, with
    weights charged per FAMILY (attention layers and linear layers cost
    different bytes) and GDN state charged per linear layer. ``arming_floor``
    and ``margin`` are carried across unchanged; if #676 resolves a different
    floor for the target layout, pass its bracket directly instead.
    """
    terms = terms or CENSUS_TERMS
    n = len(booted_bracket_mib)
    for name, seq in (
        ("booted_counts", booted_counts),
        ("booted_attn", booted_attn),
        ("counts", counts),
        ("attn", attn),
    ):
        if len(seq) != n:
            raise SeamHoldbackError(f"{name} covers {len(seq)} ranks, expected {n}.")
    out: list[float] = []
    for i in range(n):
        d_attn = int(attn[i]) - int(booted_attn[i])
        d_lin = (int(counts[i]) - int(attn[i])) - (
            int(booted_counts[i]) - int(booted_attn[i])
        )
        d_weights = (
            d_attn * terms.attn_mib_per_layer + d_lin * terms.linear_mib_per_layer
        )
        d_mamba = d_lin * terms.mamba_mib_per_linear_layer
        out.append(float(booted_bracket_mib[i]) - d_weights - d_mamba)
    return tuple(out)


def available_bytes_for_cut(
    record: SeamRecord,
    booted_counts: Sequence[int],
    booted_attn: Sequence[int],
    counts: Sequence[int],
    attn: Sequence[int],
    terms: WeightTerms | None = None,
) -> tuple[float, ...]:
    """Per-rank ``available_bytes`` under a candidate cut.

    ``available_bytes == adjusted == allowed_tokens * cell``, confirmed against
    the instrument boot to the capture's 0.1 MiB print precision. The cell moves
    with the cut because it is ``attention_layers x kv_cell``.
    """
    terms = terms or CENSUS_TERMS
    if len(attn) != len(record.cell_bytes):
        raise SeamHoldbackError("attention counts and cells disagree on rank count.")
    per_attn = record.cell_bytes[0] / max(1, int(booted_attn[0]))
    bracket = shift_bracket(
        record.bracket_mib, booted_counts, booted_attn, counts, attn, terms
    )
    out: list[float] = []
    for i, a in enumerate(attn):
        if int(a) <= 0:
            raise SeamHoldbackError(
                f"rank{i} holds no attention layer, so it has no cell and no "
                "token capacity to cap."
            )
        cell = int(a) * per_attn
        allowed = record.id_space_tokens + (bracket[i] * MIB) / cell
        if allowed <= 0.0:
            raise SeamHoldbackError(
                f"rank{i} caps to {allowed:,.0f} tokens under cut {tuple(counts)}: "
                "its free column no longer clears the arming floor, so the "
                "layout cannot arm a flip."
            )
        out.append(allowed * cell)
    return tuple(out)
