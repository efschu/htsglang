"""#707: the holdback closed form, checked against the instrument boot.

Calibration data from boot c3e94878ff, PP-phase sizing at 22:11:08:

    PP0 profiled=14819.5 MiB adjusted=8129.5 holdback=6690.1 (45.143%)
    PP1 profiled= 8083.4     adjusted=4520.7 holdback=3562.7 (44.074%)
    PP2 profiled= 8576.3     adjusted=3408.4 holdback=5167.9 (60.258%)
    available_bytes 8524386304 / 4740280320 / 3573989376
    cell 14336 / 10240 / 8192, binder PP2 @ 436278

The TP-stack pass at 22:13:27 measured holdback = 0.000% on every rank.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.seam_holdback import (
    SeamHoldbackError,
    SeamRecord,
    WeightTerms,
    available_bytes_for_cut,
    holdback_fraction,
    shift_bracket,
)

MIB = 1024 * 1024
PROFILED_MIB = (14819.5, 8083.4, 8576.3)
ADJUSTED_MIB = (8129.5, 4520.7, 3408.4)
REPORTED_PCT = (45.143, 44.074, 60.258)
CELL = (14336, 10240, 8192)
INCUMBENT = (28, 20, 16)
INCUMBENT_ATTN = (7, 5, 4)
BINDER_TOKENS = 436_278.0


def _allowed():
    return tuple(ADJUSTED_MIB[i] * MIB / CELL[i] for i in range(3))


def _record():
    allowed = _allowed()
    id_space = min(allowed)
    return SeamRecord(
        id_space_tokens=id_space,
        bracket_mib=tuple((allowed[i] - id_space) * CELL[i] / MIB for i in range(3)),
        cell_bytes=CELL,
    )


def test_the_closed_form_reproduces_every_reported_holdback():
    """The verification that closes #707, to 0.01 pp."""
    allowed = _allowed()
    for i in range(3):
        got = 100.0 * holdback_fraction(allowed[i], PROFILED_MIB[i] * MIB, CELL[i])
        assert got == pytest.approx(REPORTED_PCT[i], abs=0.01)


def test_the_cell_is_attention_layers_times_the_kv_cell():
    """Third independent confirmation of the #702 divisor, from a live sizer."""
    for attn, cell in zip(INCUMBENT_ATTN, CELL):
        assert attn * 2048 == cell


def test_the_binder_holds_back_most_BECAUSE_it_binds():
    """The structural insight, pinned so it is not read backwards.

    The binding rank is the one whose free column has run down to its arming
    floor, so its bracket is ~zero and its allowed tokens collapse to id_space.
    It simultaneously has the smallest cell, hence the largest RAW capacity and
    therefore the largest holdback FRACTION. Those co-occur by construction; a
    big holdback is not waste.
    """
    rec = _record()
    assert rec.bracket_mib[2] == pytest.approx(0.0, abs=1.0), "PP2 sits at its floor"
    assert rec.bracket_mib[0] > 2000.0, "PP0 has headroom"
    assert CELL[2] == min(CELL)
    assert REPORTED_PCT[2] == max(REPORTED_PCT)
    # And its allowed tokens ARE the id space, i.e. it sets the pool.
    assert rec.allowed_tokens()[2] == pytest.approx(BINDER_TOKENS, rel=1e-4)


def test_the_booted_layout_reproduces_its_own_available_bytes():
    """Round trip: feeding the booted cut back must return the metal numbers."""
    got = available_bytes_for_cut(
        _record(), INCUMBENT, INCUMBENT_ATTN, INCUMBENT, INCUMBENT_ATTN
    )
    for g, want in zip(got, (8_524_386_304, 4_740_280_320, 3_573_989_376)):
        assert g == pytest.approx(want, rel=1e-4)


def test_only_the_bracket_shift_is_extrapolated_and_it_is_per_family():
    """Weights are charged per FAMILY, not at a flat per-layer rate.

    An attention layer and a linear layer cost different bytes (374.2 vs 476.2
    MiB), and only linear layers carry GDN state (51.20 MiB). A flat rate would
    mis-price exactly the cuts that move the attention boundary.
    """
    terms = WeightTerms()
    base = (1000.0, 1000.0, 1000.0)
    # Move one LINEAR layer from rank1 to rank0: rank0 pays weights + GDN state.
    got = shift_bracket(base, INCUMBENT, INCUMBENT_ATTN, (29, 19, 16), (7, 5, 4), terms)
    assert got[0] == pytest.approx(
        1000.0 - terms.linear_mib_per_layer - terms.mamba_mib_per_linear_layer
    )
    assert got[1] == pytest.approx(
        1000.0 + terms.linear_mib_per_layer + terms.mamba_mib_per_linear_layer
    )
    assert got[2] == 1000.0

    # Converting a LINEAR layer into an ATTENTION one at constant layer count
    # FREES memory: attention layers are the cheaper family AND carry no GDN
    # state. So the bracket grows by (476.2 - 374.2) + 51.20 = 153.2 MiB.
    # It is not a free win, though -- more attention layers means a bigger cell,
    # hence fewer tokens per byte. The two effects pull opposite ways, which is
    # exactly why the frontier prices cell and bracket separately.
    got2 = shift_bracket(
        base, INCUMBENT, INCUMBENT_ATTN, (28, 20, 16), (8, 4, 4), terms
    )
    freed = (
        terms.linear_mib_per_layer
        - terms.attn_mib_per_layer
        + terms.mamba_mib_per_linear_layer
    )
    assert got2[0] == pytest.approx(1000.0 + freed)
    assert freed == pytest.approx(153.2, abs=0.05)
    assert terms.attn_mib_per_layer < terms.linear_mib_per_layer


def test_a_layout_that_cannot_arm_a_flip_is_refused():
    """A cap that goes non-positive means the free column no longer clears the
    arming floor -- the layout cannot arm, which is a refusal not a small pool."""
    rec = SeamRecord(
        id_space_tokens=0.0, bracket_mib=(-9e9, 100.0, 100.0), cell_bytes=CELL
    )
    with pytest.raises(SeamHoldbackError, match="arming floor"):
        available_bytes_for_cut(
            rec, INCUMBENT, INCUMBENT_ATTN, INCUMBENT, INCUMBENT_ATTN
        )


def test_a_stage_without_attention_layers_has_no_cell_to_cap():
    with pytest.raises(SeamHoldbackError, match="no attention layer"):
        available_bytes_for_cut(
            _record(), INCUMBENT, INCUMBENT_ATTN, (3, 45, 16), (0, 12, 4)
        )


def test_malformed_inputs_are_refused():
    with pytest.raises(SeamHoldbackError, match="ranks"):
        shift_bracket(
            (1.0, 2.0, 3.0), (28, 20), INCUMBENT_ATTN, INCUMBENT, INCUMBENT_ATTN
        )
    with pytest.raises(SeamHoldbackError, match="cell must be positive"):
        holdback_fraction(1.0, 1.0, 0)
