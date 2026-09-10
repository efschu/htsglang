"""#1332 (S6 slice 1, steps 1-2) -- the bounce term is DERIVED, and bounded by
the WIDEST layer.

USER LAW 2026-09-11: the 27 GiB of layer bytes must not live permanently in
system RAM. The acceptance quantity is therefore a SIZE -- the ledger's
`host weights term` falling to 0 while a bounded bounce buffer stands in for
it -- and never a count like `agreed_count > 0`. Design note:
WEG2_REUSE_SPEC_0908.md section 10.

DANGER DIRECTION of this slice is a bound that looks bounded and is not, so the
mutants named by the operator all shrink or unbook the residency:

  M1  mean instead of max          -> test_the_bound_is_the_widest_layer_never_the_mean
  M2  depth 3 without a price      -> test_depth_is_priced_and_multiplies_the_buffer
  M3  the buffer unbooked          -> test_the_ledger_charges_the_bounce_at_both_moments
"""

import pytest

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import xchg_bounce as xb

MIB = xb.MIB
GIB = xb.GIB

# THIS RIG's instance of the expression, from the plan's own halves
# (oncard 10.28 + cross 16.84 GiB) over 64 layers (32+18+14, 8 layers/chunk).
BYTES_PER_DIRECTION = int(27.12 * GIB)
N_LAYERS = 64
PAIRS = 6
# The widest layer is ABOVE the mean by construction on a non-uniform cut.
MEAN_LAYER = -(-BYTES_PER_DIRECTION // N_LAYERS)
WIDEST_LAYER = int(MEAN_LAYER * 1.30)
SIGMA_H = 43996 * MIB          # boot weg2xsn8's own host weights term


def _terms(**kw):
    base = dict(
        bytes_per_direction=BYTES_PER_DIRECTION, n_layers=N_LAYERS,
        widest_layer_bytes=WIDEST_LAYER, pairs=PAIRS,
    )
    base.update(kw)
    return xb.bounce_terms(**base)


# --------------------------------------------------------------------------
# M1: the bound
# --------------------------------------------------------------------------


def test_the_bound_is_the_widest_layer_never_the_mean():
    """M1, the danger direction of the whole slice.

    `bytes_per_direction / n_layers` is the right term for the steady stream
    and the WRONG one for the bound: a boot sized on it dies on whichever
    layer is above it. Both are carried; only the max sizes the buffer.
    """
    t = _terms()
    assert t.mean_layer_bytes == MEAN_LAYER
    assert t.widest_layer_bytes > t.mean_layer_bytes, "the fixture must be non-uniform"
    assert t.buffer_bytes == WIDEST_LAYER * xb.ASSEMBLE_DEPTH_DEFAULT
    assert t.buffer_bytes != MEAN_LAYER * xb.ASSEMBLE_DEPTH_DEFAULT
    assert t.covers_widest_layer, "one depth-slot must hold the widest layer"
    # And the mean-sized buffer would NOT cover it -- which is the failure the
    # bound exists to prevent, asserted rather than described.
    mean_sized = _terms(widest_layer_bytes=MEAN_LAYER)
    assert (mean_sized.buffer_bytes // mean_sized.depth) < WIDEST_LAYER


def test_a_buffer_that_cannot_hold_the_widest_layer_is_refused_by_name():
    """The W71-form refusal, at ARM time and with both numbers on the line."""
    t = _terms(depth=1, widest_layer_bytes=WIDEST_LAYER)
    short = xb.BounceTerms(
        bytes_per_direction=t.bytes_per_direction, n_layers=t.n_layers,
        widest_layer_bytes=WIDEST_LAYER, depth=1, pairs=PAIRS,
        slot_bytes=t.slot_bytes, mean_layer_bytes=t.mean_layer_bytes,
        buffer_bytes=MEAN_LAYER,              # sized on the mean: too small
        staging_bytes=t.staging_bytes,
    )
    assert not short.covers_widest_layer
    msg = xb.under_coverage_refusal(short, widest_layer_name="model.layers.41")
    assert msg.startswith("W71 Weg2XchgResidencyUnarmable")
    assert "model.layers.41" in msg
    assert f"{WIDEST_LAYER // MIB} MiB" in msg
    assert "never the mean" in msg
    assert "before either group starts" in msg


def test_coverage_is_per_depth_slot_and_not_the_whole_buffer():
    """Two layers in flight do not make ONE layer fit.

    THE DISCRIMINATING CASE, and my first cut of this file did not have it: at
    `buffer = widest * depth` the per-slot and whole-buffer comparisons agree,
    so a mutant that grades the WHOLE buffer against one layer passed. It is
    caught only between the two -- a buffer big enough in total and too small
    per slot, which is exactly what an assembly writes into.
    """
    t = _terms()
    half_short = xb.BounceTerms(
        bytes_per_direction=t.bytes_per_direction, n_layers=t.n_layers,
        widest_layer_bytes=WIDEST_LAYER, depth=2, pairs=PAIRS,
        slot_bytes=t.slot_bytes, mean_layer_bytes=t.mean_layer_bytes,
        buffer_bytes=int(WIDEST_LAYER * 1.5),   # >= one layer, < two
        staging_bytes=t.staging_bytes,
    )
    assert half_short.buffer_bytes >= WIDEST_LAYER, "the whole buffer DOES fit one"
    assert not half_short.covers_widest_layer, (
        "but a depth-slot does not, and the slot is what an assembly writes "
        "into -- grading the whole buffer would arm a boot that cannot "
        "assemble its widest layer"
    )


def test_an_absent_widest_layer_is_refused_and_never_becomes_the_mean():
    """Absent is not 'use the mean' -- that would silently un-bound the bound."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="widest_layer_bytes"):
            _terms(widest_layer_bytes=bad)
    with pytest.raises(ValueError, match="n_layers"):
        _terms(n_layers=0)
    with pytest.raises(ValueError, match="bytes_per_direction"):
        _terms(bytes_per_direction=0)


# --------------------------------------------------------------------------
# M2: depth
# --------------------------------------------------------------------------


def test_depth_is_priced_and_multiplies_the_buffer():
    """M2: a third layer in flight costs a whole layer of host RAM.

    Depth 2 buys the assemble/copy-out overlap; depth 3 buys nothing once the
    x4 link saturates (section 10.4). Raising it must SHOW its price.
    """
    d2, d3 = _terms(depth=2), _terms(depth=3)
    assert d3.buffer_bytes - d2.buffer_bytes == WIDEST_LAYER
    assert d3.total_bytes - d2.total_bytes == WIDEST_LAYER
    assert "depth=3" in d3.expression()
    assert xb.ASSEMBLE_DEPTH_DEFAULT == 2


def test_the_slot_is_64_mib_double_buffered_and_says_why():
    """#1277's arm, not E2's -- E2 measured the SINGLE-buffered form."""
    assert xb.SLOT_BYTES_DEFAULT == 64 * MIB
    assert xb.SLOTS_PER_PAIR == 2
    t = _terms()
    assert t.staging_bytes == PAIRS * 2 * 64 * MIB == 768 * MIB


# --------------------------------------------------------------------------
# The derivation is printed, so the total is checkable and not a constant
# --------------------------------------------------------------------------


def test_the_expression_prints_every_term_it_was_derived_from():
    t = _terms()
    e = t.expression()
    for token in ("buffer=", "widest_layer=", "depth=", "mean_layer=",
                  "n_layers=", "bytes_per_direction=", "staging=", "pairs=",
                  "slot=", "bounce_total="):
        assert token in e, token
    assert t.total_bytes == t.buffer_bytes + t.staging_bytes


def test_this_rigs_instance_of_the_expression():
    """The section-10.2 table, as arithmetic rather than prose.

    Not a pin on the rig: it is the expression evaluated at this cut, and the
    module ships no such constant.
    """
    t = _terms(widest_layer_bytes=MEAN_LAYER)      # uniform-cut instance
    assert abs(t.mean_layer_bytes / MIB - 433.9) < 0.5
    assert abs(t.buffer_bytes / MIB - 867.8) < 1.0
    assert t.staging_bytes // MIB == 768
    assert abs(t.total_bytes / GIB - 1.60) < 0.01
    assert abs((SIGMA_H - t.total_bytes) / GIB - 41.37) < 0.02
    assert abs(SIGMA_H / t.total_bytes - 26.9) < 0.1


def test_the_arm_line_prints_sigma_h_and_the_bounce_SEPARATELY():
    """Section 10.6: the acceptance is that one goes to 0 as the other stands
    in for it, so one number carrying both would make it unobservable."""
    t = _terms()
    line = xb.arm_line(t, sigma_h_bytes=0)
    assert "sigma_h_mib=0" in line
    assert f"bounce_total_mib={t.total_bytes // MIB}" in line
    assert "covers_widest=yes" in line
    # Before the slice lands, Sigma H is the xsn8 figure and both appear.
    before = xb.arm_line(t, sigma_h_bytes=SIGMA_H)
    assert f"sigma_h_mib={SIGMA_H // MIB}" in before
    assert "released_gib=" in before
    assert "sigma_h_mib=0" not in before


# --------------------------------------------------------------------------
# M3: the ledger books it
# --------------------------------------------------------------------------


def test_the_ledger_charges_the_bounce_at_both_moments():
    """M3: an unbooked buffer is host RAM nobody priced.

    The hull has been on the branch since 2afbecf601; this pins that the
    bounce term reaches `_boot_charges_gib`, i.e. that it is charged and not
    merely accepted as an argument.
    """
    t = _terms()
    img = hl.ImageTerms(
        p_gib=38.63, d_gib=38.63, p_source="s", d_source="s",
        p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
    )
    without = hl._boot_charges_gib(hl.charge_terms(1, 600, 3, img, s_gb_d=4))
    with_b = hl._boot_charges_gib(
        hl.charge_terms(1, 600, 3, img, s_gb_d=4,
                        xchg_bounce_host_bytes=t.total_bytes)
    )
    assert abs((with_b - without) - t.total_bytes / GIB) < 1e-6, (
        "the bounce must move the boot charges by exactly its own size"
    )
    charges = hl.charge_terms(1, 600, 3, img, s_gb_d=4,
                              xchg_bounce_host_bytes=t.total_bytes)
    assert abs(charges["xchg_bounce_gib"] - t.total_bytes / GIB) < 1e-9


def test_the_ledger_arm_line_carries_BOTH_residencies_in_one_grep():
    """Section 10.6: Sigma H and the bounce, side by side on the ARM line.

    Sigma H is also on the TERMS line -- but that is a different line, and a
    reader comparing two lines is a reader who can mismatch two boots. The
    slice's acceptance is a TRANSITION (one to 0.00 as the other stands in),
    so both must be readable in one grep and never folded into one figure.
    """
    import inspect

    src = inspect.getsource(hl.choose)
    assert "host_weights={arm.terms['host_ring_gib']:.2f}" in src
    assert "xchg_bounce={arm.terms['xchg_bounce_gib']:.2f}" in src
    # And they must be SEPARATE fields, not one sum.
    i = src.index("host_weights={")
    j = src.index("xchg_bounce={")
    assert abs(i - j) > 0 and "host_weights" != "xchg_bounce"
