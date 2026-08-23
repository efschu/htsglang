"""#826 -- adopt the solved arming floor, or refuse the boot BY NAME.

#770 delivered `solve_arming_floor` as an OBSERVER: its only consumer
(`phase_flip_runtime.py:6435`) turns it into advice text and sets no floor, so
`DEFAULT_SEAM_ENTRY_RESERVE_MIB` stayed at the shipped 512 and
`arming_floor_mib()` stayed at 1331 against a band ceiling of 1229 -- 294 MiB
short BY CONSTRUCTION, on every card, on every boot.

The solver's own docstring names this file's job:

    "this function SOLVES and REPORTS -- it returns the reachable floor and
     the reserve it implies -- and the caller decides whether to adopt it or
     refuse at boot. What must never happen again is the third option the
     tree shipped: neither adopting nor refusing, and instead advising at
     runtime that the flip 'is retried when occupancy drops', which describes
     a state the corridor law forbids."

SCOPE LIMIT, deliberate and narrow. This adopts the floor VALUE only. It does
NOT revive the withdrawn floor clamp (`3b2bbde3ad`, a cap under live rows --
`test_residency_cap_flip_levelling_792` is that watchdog) and it takes no
kv-slack draws. Those are a separate metal ticket.
"""


import pytest

from sglang.srt.managers import corridor_guard as cg
from sglang.srt.managers.phase_flip_seam_reserve import DEFAULT_ARMING_MARGIN_MIB


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(cg.SOLVED_FLOOR_ENV, raising=False)
    cg._reset_arming_floor_provenance()
    yield
    cg._reset_arming_floor_provenance()


# --------------------------------------------------------------------------
# The default path must be byte-identical.
# --------------------------------------------------------------------------


def test_without_the_flag_nothing_moves():
    """The shipped path is the regression baseline: 819 + 512 = 1331."""
    assert cg.seam_entry_reserve_mib() == cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB == 512
    assert cg.arming_floor_mib() == cg.corridor_band_floor_mib() + 512
    assert cg.arming_floor_mib() == 1331


def test_an_explicit_reserve_argument_still_wins():
    """The parameter was always the override point; making the DEFAULT
    resolvable must not take that away."""
    assert cg.arming_floor_mib(seam_entry_reserve_mib=100) == (
        cg.corridor_band_floor_mib() + 100
    )


# --------------------------------------------------------------------------
# The flag adopts the solved value.
# --------------------------------------------------------------------------


def test_the_flag_adopts_the_solved_reserve_and_the_floor_fits(monkeypatch):
    """THE POINT. With the flag on, the arming floor plus its margin must fit
    UNDER the band ceiling -- which is exactly what 1331 could never do."""
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    reserve = cg.seam_entry_reserve_mib()
    floor = cg.arming_floor_mib()
    ceiling = cg.corridor_band_ceiling_mib()
    assert reserve == 218, reserve
    assert floor == cg.corridor_band_floor_mib() + 218 == 1037
    # The whole purpose: reachable from inside the acceptance band.
    assert floor + DEFAULT_ARMING_MARGIN_MIB <= ceiling
    # And strictly better than the shipped configuration, which is not.
    assert 1331 + DEFAULT_ARMING_MARGIN_MIB > ceiling


def test_the_adopted_floor_never_drops_below_the_band_floor(monkeypatch):
    """The band is a HARD user rule the solver never touches. An arming floor
    below the band floor would let the gate bless an allocation the corridor
    law forbids -- the failure `DEFAULT_SEAM_ENTRY_RESERVE_MIB`'s own comment
    describes ("an arming floor BELOW the law ... is refused")."""
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    assert cg.arming_floor_mib() >= cg.corridor_band_floor_mib()


def test_provenance_is_logged_once_with_the_numbers(monkeypatch, caplog):
    """A derived number that does not say where it came from is a hand-pin
    with extra steps."""
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    with caplog.at_level("WARNING"):
        cg.seam_entry_reserve_mib()
        cg.seam_entry_reserve_mib()
        cg.seam_entry_reserve_mib()
    lines = [r for r in caplog.records if "arming floor" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in lines]
    msg = lines[0].getMessage()
    assert "solver-derived" in msg
    assert "218" in msg and "1037" in msg and "1229" in msg


# --------------------------------------------------------------------------
# Unsatisfiable must REFUSE AT BOOT, never livelock at runtime.
# --------------------------------------------------------------------------


def test_an_unsatisfiable_solve_refuses_by_name_at_boot(monkeypatch):
    """THE DANGER DIRECTION.

    If no reserve fits under the ceiling, the honest answers are a named boot
    refusal or nothing -- never a floor adopted anyway, and never the shipped
    third option of advising at runtime that the operator should wait for
    occupancy to drop, which is a state the corridor law forbids. 18f measured
    that wait: draining the load did not lift the lock.
    """
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    # Push the margin past the whole band so nothing can fit.
    monkeypatch.setattr(cg, "_arming_margin_mib_for_solver", lambda: 10_000)
    with pytest.raises(cg.ArmingFloorUnsatisfiable) as e:
        cg.seam_entry_reserve_mib()
    detail = str(e.value)
    assert "arming floor" in detail.lower()
    # It must name the numbers, so the refusal is evidence not an assertion.
    assert str(cg.corridor_band_ceiling_mib()) in detail


def test_the_refusal_does_not_fire_on_the_default_path(monkeypatch):
    """An unsatisfiable solve must not take down a boot that never asked for
    the solved floor. Opt-in means opt-in."""
    monkeypatch.setattr(cg, "_arming_margin_mib_for_solver", lambda: 10_000)
    assert cg.seam_entry_reserve_mib() == 512  # no raise


def test_flag_off_values_are_accepted_as_off(monkeypatch):
    for off in ("0", "", "false", "no"):
        monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, off)
        cg._reset_arming_floor_provenance()
        assert cg.seam_entry_reserve_mib() == 512, off


# --------------------------------------------------------------------------
# THE ANTI-INERTNESS ARM.
# --------------------------------------------------------------------------


def test_the_production_arming_path_honours_the_flag(monkeypatch):
    """THE MOST IMPORTANT TEST IN THIS FILE.

    #770's solver was correct and inert: nothing applied it. This actuator
    could have repeated that exactly, because BOTH production consumers pass
    `seam_entry_reserve_mib` EXPLICITLY -- so a flag that only changed the
    DEFAULT ARGUMENT would move nothing on the path it was built for. Caught
    by reading the call sites before booting, not by a test that assumed the
    default was reached.

    This drives the real consumer and requires the floor to move.
    """
    from sglang.srt.managers import phase_flip_seam_reserve as sr

    monkeypatch.setattr(sr, "_arming_margin_bytes", lambda: 0)

    monkeypatch.delenv(cg.SOLVED_FLOOR_ENV, raising=False)
    cg._reset_arming_floor_provenance()
    shipped = sr.arming_floor_target_bytes(measured_draw_mib=0, configured_mib=0) >> 20

    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    cg._reset_arming_floor_provenance()
    solved = sr.arming_floor_target_bytes(measured_draw_mib=0, configured_mib=0) >> 20

    assert shipped == 1331, shipped
    assert solved == 1037, solved
    assert solved < shipped, "the flag is inert on the production path"


def test_a_measured_draw_above_the_solved_reserve_still_raises_the_floor(monkeypatch):
    """The max() with the measured draw is deliberately kept. A seam that
    demonstrably draws more than the solved reserve must still raise the
    floor -- lowering it to fit a band would be trading a correctness
    invariant for a funding win, which is what 3b2bbde3ad withdrew."""
    from sglang.srt.managers import phase_flip_seam_reserve as sr

    monkeypatch.setattr(sr, "_arming_margin_bytes", lambda: 0)
    monkeypatch.setenv(cg.SOLVED_FLOOR_ENV, "1")
    cg._reset_arming_floor_provenance()
    got = sr.arming_floor_target_bytes(measured_draw_mib=400, configured_mib=0) >> 20
    assert got == cg.corridor_band_floor_mib() + 400 == 1219
