"""W2 pre-flight -- can this configuration EVER arm inside the corridor band?

PURPOSE: a hermetic, no-GPU gate that must pass before a flip window is spent.
boot_827_review_0823_0910c burned a window discovering at runtime what this
predicate answers at the desk.

THE PREDICATE, stated once:

    band_floor + max(seam_entry_reserve, measured_seam_draw) + arming_margin
        <= band_ceiling

WHY THE `measured_seam_draw` TERM IS NOT OPTIONAL, and this is the whole point
of the file. The obvious W2 assertion is "with the flag on, 819 + 218 + 192 =
1229 <= 1229, satisfiable". That is TRUE and it is a FALSE GREEN, because it
silently assumes a measured draw of zero. `arming_floor_target_bytes` takes
`max(resolved_reserve, measured_draw_mib)`, so the draw dominates whenever it
exceeds the solved reserve -- and on metal it did, immediately.

MEASURED ON boot_827_review_0823_0910c, one boot, arming floor observed at
FOUR distinct values:

    1037 MiB   = 819 + 218   the solved reserve, at boot, all three ranks
                             ("#826 arming floor 1037 MiB, solver-derived")
    1206 MiB   = 819 + 387   a measured seam draw
    1255 MiB   = 819 + 436   a measured seam draw   (11 occurrences)
    1726 MiB   = 819 + 907   a measured seam draw

Only the first fits. 1255 + 192 = 1447 > 1229, and that is the value the flip
actually abandoned against:

    PHASE-FLIP FLIP ABANDONED (pool too small for the live set): pp_to_tp.
    corridor gate refused the seam staging: want 1746 MiB, free 1436 ...
    arming floor 1255 MiB, corridor law 1024 MiB

So the honest pre-flight answer for W2 is NOT "pass". It is: pass only while
the seam's measured draw stays at or below the solved reserve, and the
measured draws on this rig do not.
"""

import pytest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.managers.phase_flip_seam_reserve import DEFAULT_ARMING_MARGIN_MIB

# Seam draws measured on boot_827_review_0823_0910c, in MiB.
MEASURED_DRAWS_0910C = (162, 387, 436, 907)


def arming_is_reachable(*, reserve_mib: int, measured_draw_mib: int) -> bool:
    """The reachability predicate, as a pure function.

    Mirrors `arming_floor_target_bytes`'s own arithmetic: the floor is the
    band floor plus the LARGER of the configured reserve and the measured
    draw, and the gate additionally wants `DEFAULT_ARMING_MARGIN_MIB` free on
    top of it.
    """
    floor = cg.corridor_band_floor_mib() + max(int(reserve_mib), int(measured_draw_mib))
    return floor + DEFAULT_ARMING_MARGIN_MIB <= cg.corridor_band_ceiling_mib()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(cg.SOLVED_FLOOR_ENV, raising=False)
    cg._reset_arming_floor_provenance()
    yield
    cg._reset_arming_floor_provenance()


def test_the_shipped_default_can_never_arm():
    """RED SIDE: 819 + 512 + 192 = 1523 against a ceiling of 1229."""
    reserve = cg.seam_entry_reserve_mib_resolved()
    assert reserve == 512
    assert not arming_is_reachable(reserve_mib=reserve, measured_draw_mib=0)
    floor = cg.corridor_band_floor_mib() + reserve
    assert floor == 1331
    assert floor + DEFAULT_ARMING_MARGIN_MIB == 1523 > cg.corridor_band_ceiling_mib()


def test_the_solved_reserve_is_reachable_at_zero_draw(monkeypatch):
    """GREEN SIDE, and exactly as narrow as it really is."""
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    reserve = cg.seam_entry_reserve_mib_resolved()
    assert reserve == 218
    assert arming_is_reachable(reserve_mib=reserve, measured_draw_mib=0)
    floor = cg.corridor_band_floor_mib() + reserve
    assert floor == 1037
    assert floor + DEFAULT_ARMING_MARGIN_MIB == 1229 == cg.corridor_band_ceiling_mib()


def test_the_measured_draws_from_boot_827_are_NOT_reachable(monkeypatch):
    """THE ARM THAT STOPS THE FALSE GREEN.

    Three of the four seam draws measured on boot_827 put the floor back over
    the ceiling even with the solved reserve adopted. A pre-flight that
    asserted only the zero-draw case would have certified that boot.
    """
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    reserve = cg.seam_entry_reserve_mib_resolved()
    verdicts = {
        d: arming_is_reachable(reserve_mib=reserve, measured_draw_mib=d)
        for d in MEASURED_DRAWS_0910C
    }
    assert verdicts == {162: True, 387: False, 436: False, 907: False}, verdicts
    # The value the flip actually abandoned against.
    assert cg.corridor_band_floor_mib() + 436 == 1255


def test_the_largest_tolerable_draw_is_the_solved_reserve(monkeypatch):
    """Boundary, pinned: the predicate turns over exactly at 218."""
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    assert arming_is_reachable(reserve_mib=218, measured_draw_mib=218)
    assert not arming_is_reachable(reserve_mib=218, measured_draw_mib=219)


def test_preflight_verdict_for_w2_is_conditional_not_pass(monkeypatch):
    """The single assertion the window queue should read.

    W2 does NOT flip to preflight_pass=Y on this rig's measured draws. It is
    conditional: Y only while the seam draw stays <= the solved reserve.
    """
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    reserve = cg.seam_entry_reserve_mib_resolved()
    worst = max(MEASURED_DRAWS_0910C)
    assert not arming_is_reachable(reserve_mib=reserve, measured_draw_mib=worst)
