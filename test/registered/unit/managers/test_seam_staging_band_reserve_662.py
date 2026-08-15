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
"""

import types

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.managers import phase_flip_runtime as pfr

MIB = 1 << 20


def _args(reserve_mib=1024):
    return types.SimpleNamespace(rank_user_reserve_mib=reserve_mib)


def test_the_seam_reserves_the_band_floor_not_the_centre():
    assert pfr._seam_staging_reserve_bytes(_args(1024)) == 819 * MIB


def test_it_tracks_the_declared_band_rather_than_a_private_constant():
    reserve = 1024
    expected = int(round(reserve * (1.0 - cg.CORRIDOR_BAND_FRACTION)))
    assert pfr._seam_staging_reserve_bytes(_args(reserve)) == expected * MIB


def test_the_measured_refusal_now_funds():
    """The GATE C instant, replayed as arithmetic."""
    driver_free, staging_needed = 2024 * MIB, 1059 * MIB
    centre_reserve = 1024 * MIB
    floor_reserve = pfr._seam_staging_reserve_bytes(_args(1024))

    assert driver_free - centre_reserve < staging_needed, "what was refused"
    assert driver_free - floor_reserve >= staging_needed, "what now funds"
    # And by how much, so a future change that eats this margin is visible.
    assert (driver_free - floor_reserve - staging_needed) // MIB == 146


def test_a_bigger_user_reserve_still_scales_with_the_band():
    assert pfr._seam_staging_reserve_bytes(_args(2048)) == 1638 * MIB


def test_an_absent_reserve_falls_back_to_the_shipped_default():
    assert pfr._seam_staging_reserve_bytes(types.SimpleNamespace()) == 819 * MIB


def test_the_seam_never_reserves_MORE_than_the_user_asked_for():
    """A tolerance that made the reserve larger would be a refusal machine."""
    for mib in (256, 1024, 4096):
        assert pfr._seam_staging_reserve_bytes(_args(mib)) <= mib * MIB


def test_the_corridor_law_itself_is_untouched():
    """The scope check. The band moved the SEAM's reserve; the law the guard
    judges ordinary allocations by is still the centre."""
    assert cg.corridor_law_mib() == 1024
    assert cg.corridor_band_floor_mib() == 819
