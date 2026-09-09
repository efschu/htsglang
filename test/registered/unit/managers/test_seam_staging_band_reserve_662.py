# SPDX-License-Identifier: Apache-2.0
"""#662: the seam may spend into the band's tolerance; ordinary work may not.

MEASURED, GATE C, 2026-08-15, device 0:

    staging 1059 MiB needed but only 1000 MiB is spendable
    (driver free 2024 MiB, allocator cache 100 MiB, reserve 1024 MiB kept free)
    ... tp_to_pp REFUSED 8 times consecutively and is being treated as
    unfundable

Refused by 59 MiB, then the direction latched and the instance held in TP with
50k+ tokens pending at 1000-1600 tok/s where the PP layout does 4000-7000 --
the exact outcome this whole strand exists to remove.

The corridor is a band: 1024 MiB +-20 %, and the verdict is the continuous
minimum against the FLOOR. A cutover that dips to 819 MiB for the length of a
wave walk is lawful. Holding the CENTRE as a hard reserve during staging
spends that tolerance on nothing and refuses flips the law permits; against
the floor the same instant offers 2024 - 819 = 1205 MiB and it funds.

SCOPE IS THE POINT. This is the STAGING reserve, not the corridor law. Every
ordinary allocation is still judged against the centre by the guard; only the
cutover -- bounded, unanimous, over in seconds -- reaches into the tolerance.

#1257c RE-EXPRESSED THE INPUTS OF THIS FILE (2026-09-09); the arithmetic is
untouched.

When #662 was written, ``--rank-user-reserve-mib`` WAS the corridor law: it
defaulted to 1024 and ``_seam_staging_reserve_bytes`` read
``rank_user_reserve_mib or 1024``. The user decision of 2026-09-09 separates
the two quantities that literal conflated -- "die 1024er grenze ... existiert
ja nur weil du den wahren vram verbrauch nicht bepreisen konntest UND weil ich
manchmal noch vram fuer andere prozesse brauche" -- so the law is the awake
group's measured transient (or the named 1024 fallback) and the reserve is an
ADDITIONAL per-card term on top of it, defaulting to 0.

GATE C's instant is therefore now stated as "law 1024, user reserve 0", which
is what it always physically was; ``_args(1024)`` now means law 1024 PLUS a
1024 MiB reserve, a different scenario. Every number below is the same number
as before -- only the input that produces it is named correctly.

AND THE STUB NOW EXPOSES WHAT THE CODE READS. ``_args`` supplied a bare
``rank_user_reserve_mib`` field, which the shipped function stopped reading
when the reserve became a per-card quantity, so these tests were green without
the reserve reaching the code at all.
"""

import types

import pytest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.managers import phase_flip_runtime as pfr

MIB = 1 << 20


@pytest.fixture(autouse=True)
def _no_ambient_corridor_env(monkeypatch):
    """A leaked law override or per-card reserve would move every number."""
    monkeypatch.delenv(cg.LAW_ENV, raising=False)
    monkeypatch.delenv(cg.USER_RESERVE_ENV, raising=False)


def _args(reserve_mib=0):
    """A ServerArgs stub shaped like the one the shipped function reads."""
    return types.SimpleNamespace(
        rank_user_reserve_mib=reserve_mib,
        user_reserve_mib_scalar=lambda: reserve_mib,
    )


def test_the_seam_reserves_the_band_floor_not_the_centre():
    assert pfr._seam_staging_reserve_bytes(_args(0)) == 819 * MIB


def test_it_tracks_the_declared_band_rather_than_a_private_constant():
    law = cg.CORRIDOR_LAW_MIB
    expected = int(law - law * cg.CORRIDOR_BAND_FRACTION)
    assert pfr._seam_staging_reserve_bytes(_args(0)) == expected * MIB


def test_the_measured_refusal_now_funds():
    """The GATE C instant, replayed as arithmetic."""
    driver_free, staging_needed = 2024 * MIB, 1059 * MIB
    centre_reserve = 1024 * MIB
    floor_reserve = pfr._seam_staging_reserve_bytes(_args(0))

    assert driver_free - centre_reserve < staging_needed, "what was refused"
    assert driver_free - floor_reserve >= staging_needed, "what now funds"
    # And by how much, so a future change that eats this margin is visible.
    assert (driver_free - floor_reserve - staging_needed) // MIB == 146


def test_a_bigger_user_reserve_still_scales_with_the_band():
    """1024 MiB of ADDITIONAL user reserve on top of the 1024 MiB law."""
    assert pfr._seam_staging_reserve_bytes(_args(1024)) == 1638 * MIB


def test_an_absent_reserve_falls_back_to_the_shipped_default():
    assert pfr._seam_staging_reserve_bytes(types.SimpleNamespace()) == 819 * MIB


def test_the_tolerance_never_makes_the_reserve_LARGER():
    """A tolerance that grew the reserve would be a refusal machine.

    #1257c restates the BOUND because the quantity moved. It used to read
    "never more than the user asked for", which held only while the user's
    reserve WAS the whole law; with the default reserve at 0 that bound would
    now demand a seam reserve of zero -- the ``cuMemCreate failed:
    CUDA_ERROR_OUT_OF_MEMORY`` this module exists to avoid. The property that
    carries #662's intent is the one about the BAND: the seam may reach INTO
    the tolerance and never above the floor it is measured from.
    """
    for reserve in (0, 256, 1024, 4096):
        floor = cg.corridor_floor_mib("", user_reserve_mib=reserve)
        got = pfr._seam_staging_reserve_bytes(_args(reserve))
        assert got <= floor.mib * MIB
        assert got == floor.verdict_floor_mib * MIB


def test_the_corridor_law_itself_is_untouched():
    """The scope check. The band moved the SEAM's reserve; the law the guard
    judges ordinary allocations by is still the centre."""
    assert cg.corridor_law_mib() == 1024
    assert cg.corridor_band_floor_mib() == 819


# ---------------------------------------------------------------------------
# The JOINT constraint: pool-fill and seam fundability are one solve.
#
# They were maximised as separate criteria and collided by 59 MiB. Causally,
# on this rig: deriving the arming floor from the band recovered ~205 MiB/rank,
# that went into pool, device 0's under-load free fell ~2.5 -> ~1.9-2.0 GiB,
# and the seam's 1059 MiB staging missed. The earlier boot flipped BECAUSE its
# pool was smaller. Sizing must therefore reserve band_floor + seam_draw, from
# the SAME number the runtime staging reserve uses.
# ---------------------------------------------------------------------------

from sglang.srt.managers import phase_flip_seam_reserve as sr


def test_sizing_reserves_the_band_floor_not_the_centre():
    law = 1024 * MIB
    assert sr._band_floor_bytes(law) == 819 * MIB


def test_sizing_and_the_runtime_seam_reserve_are_the_same_number():
    """The disagreement WAS the defect: sizing kept 1024 while the gate spent
    to 819, so a pool gain could eat the seam's margin unnoticed."""
    law_mib = 1024
    sizing = sr._band_floor_bytes(law_mib * MIB)
    runtime = pfr._seam_staging_reserve_bytes(_args(law_mib))
    assert sizing == runtime == 819 * MIB


def test_the_solve_leaves_the_floor_plus_the_seam_draw():
    """At the solved id space, resting free is floor + draw -- so the cutover
    dip bottoms out AT the floor, which is lawful, and no tighter."""
    law = 1024 * MIB
    floor = sr._band_floor_bytes(law)
    seam_draw = 1059 * MIB  # the measured GATE C staging demand
    resting_free = floor + seam_draw
    assert resting_free - seam_draw == floor
    assert (resting_free - floor) >= seam_draw, "the seam always fits"


def test_a_pool_gain_is_taken_after_the_seam_demand_not_out_of_it():
    """The regression this exists to prevent: free memory recovered by any
    future change is spendable only above floor + draw."""
    law = 1024 * MIB
    floor = sr._band_floor_bytes(law)
    seam_draw = 1059 * MIB
    recovered = 205 * MIB  # what the arming-floor derivation gave back
    resting_free = floor + seam_draw + recovered
    assert resting_free - seam_draw >= floor, "the seam still funds after the gain"


# ---------------------------------------------------------------------------
# The C20 entry-margin arithmetic, and what it actually compares.
#
# Reported from metal as "want 2251 with free 3206 fails a 512 MiB margin --
# the arming floor looks double-counted". It is neither. The predicate is
#
#     law_ok = margin_bytes > 0 and (free_after - staging >= law_floor)
#
# `margin_bytes` is an ENABLE FLAG and appears in no comparison; the arming
# floor appears nowhere at all. The binding number was the corridor CENTRE:
# 3206 - 2251 = 955, which is 69 MiB under 1024 -- and 136 MiB CLEAR of the
# band floor at 819. So the flip was legal and was delayed anyway, and the log
# named 512, a number the arithmetic never touches.
# ---------------------------------------------------------------------------

from sglang.srt.managers import phase_flip_runtime as _pfr


def test_the_seam_transient_floor_is_the_bands_lower_edge():
    assert _pfr._seam_transient_floor_bytes(1024 * MIB) == 819 * MIB


def test_the_reported_instant_is_legal_against_the_band_and_was_delayed():
    free_after, staging = 3206 * MIB, 2251 * MIB
    remaining = free_after - staging
    assert remaining == 955 * MIB
    assert remaining < 1024 * MIB, "what the centre-based check saw"
    assert remaining >= _pfr._seam_transient_floor_bytes(1024 * MIB), (
        "and what the band says: legal, with 136 MiB to spare"
    )


def test_the_512_margin_is_not_in_the_comparison_at_all():
    """It is an enable flag. Naming it in the refusal is what made this read
    as a double-counted arming floor from outside."""
    free_after, staging, margin = 3206 * MIB, 2251 * MIB, 512 * MIB
    # The margin neither adds to the requirement nor is subtracted from it.
    assert (free_after - staging) != margin
    assert (free_after - staging - margin) != 0


def test_the_fallback_delays_rather_than_enters():
    """If the band cannot be read the floor stays where it was: delaying a
    legal flip costs throughput, entering an illegal one costs the law."""
    assert _pfr._seam_transient_floor_bytes(0) == 0
