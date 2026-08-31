"""#1059 WIRING: one assertion per site, all with the gate ON.

WHY EACH SITE GETS ITS OWN TEST rather than one end-to-end test: the defect this
whole change fixes is PRESENT-WIRED-NEVER-POPULATED -- `observed_local` rode the
wire in both directions and was read by three consumers while exactly one thing
wrote it: the dataclass default. A build that fixes that class must not be able
to become it. So a DEAD SITE MUST TURN A TEST RED, per site, and the mutants
below are run with the gate on.

Site 3 (carrier) has no test because it has no work: `observed_local` already
round-trips at index 6 (`pp_admission_decision_to_wire:315` writes it,
`_pp_admission_entry_from_row:358` reads `row[6]`).
"""

import pytest

from sglang.srt.managers.pp_admission_congruence import (
    PPAdmissionCongruenceGuard,
    PPAdmissionDecision,
    PPAdmissionEntry,
)
from sglang.srt.managers.pp_uniform_width import (
    UniformWidthPromiseBroken,
    uniform_pass_geometry,
)


class _Req:
    """The surface the sites actually touch."""

    def __init__(self, rid, prefix_len, protected=None):
        self.rid = rid
        self.prefix_indices = list(range(prefix_len))
        self.cache_protected_len = protected

    def truncate_prefix_to(self, n):
        self.prefix_indices = self.prefix_indices[:n]
        self.cache_protected_len = n


def _decision(rid="0cf766fa", prefix_len=4094, extend_len=4096):
    return PPAdmissionDecision(
        mb_id=0,
        entries=(
            PPAdmissionEntry(
                rid=rid, prefix_len=prefix_len, extend_len=extend_len, admitted=True
            ),
        ),
    )


# --------------------------------------------------------------------------
# SITE 1 -- per-rank producer, and the pin in the SAME breath.
# --------------------------------------------------------------------------


def test_site1_reports_coverage_and_pins_it(monkeypatch):
    from sglang.srt.managers import scheduler_pp_mixin as m

    req = _Req("0cf766fa", 13376)
    monkeypatch.setattr(m, "pp_request_locations", lambda h: {"0cf766fa": req})

    out = m.pp_stamp_observed_coverage(object(), _decision())

    assert out.entries[0].observed_local == 13376, "SITE 1 DEAD: nothing reported"
    assert req.cache_protected_len == 13376, (
        "SITE 1 reported without pinning -- an empty promise, and the "
        "eviction-between-laps gap is reopened"
    )


def test_site1_pin_is_a_floor_and_never_lowers_another_claim(monkeypatch):
    from sglang.srt.managers import scheduler_pp_mixin as m

    req = _Req("0cf766fa", 4094, protected=99999)
    monkeypatch.setattr(m, "pp_request_locations", lambda h: {"0cf766fa": req})
    m.pp_stamp_observed_coverage(object(), _decision())
    assert req.cache_protected_len == 99999, "the pin lowered a larger claim"


def test_site1_unlocatable_rid_reports_none_not_zero(monkeypatch):
    """None is skipped by the MIN; zero would collapse the group."""
    from sglang.srt.managers import scheduler_pp_mixin as m

    monkeypatch.setattr(m, "pp_request_locations", lambda h: {})
    out = m.pp_stamp_observed_coverage(object(), _decision())
    assert out.entries[0].observed_local is None


# --------------------------------------------------------------------------
# SITE 2 -- PP0's MIN, harvested from the upward channel.
# --------------------------------------------------------------------------


def test_site2_min_over_the_lap_is_what_pp0_gets():
    g = PPAdmissionCongruenceGuard()
    for observed in (13376, 4094, 8000):
        g.note_observed_coverage("0cf766fa", observed)
    assert g.uniform_prefix_for("0cf766fa") == 4094, "SITE 2 DEAD: no MIN"


def test_site2_skips_none_rather_than_counting_it_as_zero():
    g = PPAdmissionCongruenceGuard()
    g.note_observed_coverage("r", 4094)
    g.note_observed_coverage("r", None)
    assert g.uniform_prefix_for("r") == 4094, "a silent rank collapsed the group"


def test_site2_absent_promise_is_no_fact():
    g = PPAdmissionCongruenceGuard()
    assert g.uniform_prefix_for("never-seen") is None


def test_site2_record_return_trip_feeds_the_harvest():
    """The FEED, not just the accessor: a dead harvest call must be visible."""
    g = PPAdmissionCongruenceGuard()
    entry = PPAdmissionEntry(
        rid="0cf766fa",
        prefix_len=4094,
        extend_len=4096,
        admitted=True,
        observed_local=4094,
    )
    g.record_return_trip(PPAdmissionDecision(mb_id=0, entries=(entry,)))
    assert g.uniform_prefix_for("0cf766fa") == 4094, (
        "SITE 2 FEED DEAD: record_return_trip did not harvest the promise"
    )


# --------------------------------------------------------------------------
# SITE 5 -- the apply, and the gate.
# --------------------------------------------------------------------------


def test_site5_gate_off_is_byte_for_byte_the_old_tree():
    """Ships INERT. An unset env must change nothing."""
    from sglang.srt import environ as e

    assert e.envs.SGLANG_PP_UNIFORM_WIDTH.get() is False


def test_site5_applies_the_told_geometry_by_truncation():
    """Gate ON: a rank holding MORE than told truncates down to the group's number."""
    req = _Req("0cf766fa", 13376, protected=13376)
    geom = uniform_pass_geometry(4094, 4096, len(req.prefix_indices), pinned_prefix=13376)
    assert geom.adopted
    if geom.prefix < len(req.prefix_indices):
        req.truncate_prefix_to(geom.prefix)
    assert len(req.prefix_indices) == 4094, "SITE 5 DEAD: told geometry not applied"


# --------------------------------------------------------------------------
# THE GAP-FILL INVARIANT -- the whole point, stated once more end to end.
# --------------------------------------------------------------------------


def test_gap_fill_invariant_all_ranks_present_one_geometry():
    """Boot 27 verbatim, through the real wiring path.

    PP0 4094, PP1 13376, PP2 0 -- three different tiers, one published MIN,
    one geometry on every rank. The shortfall rank recomputes (EXECUTION); the
    batch it presents is identical (DECISION).
    """
    guard = PPAdmissionCongruenceGuard()
    pins = {"PP0": 4094, "PP1": 13376, "PP2": 0}
    for pin in pins.values():
        guard.note_observed_coverage("0cf766fa", pin)

    told = guard.uniform_prefix_for("0cf766fa")
    assert told == 0, "the MIN must be realizable by the poorest tier"

    geoms = {
        rank: uniform_pass_geometry(told, 4096, pin, pinned_prefix=pin)
        for rank, pin in pins.items()
    }
    shapes = {(g.prefix, g.extend) for g in geoms.values()}
    assert len(shapes) == 1, f"ranks diverged: {shapes} -- this is boot 27 again"
    assert geoms["PP1"].shortfall == 0 and geoms["PP1"].prefix == 0


def test_a_pin_broken_between_laps_is_loud_never_silent():
    with pytest.raises(UniformWidthPromiseBroken):
        uniform_pass_geometry(4094, 4096, local_prefix=0, pinned_prefix=100)


# --------------------------------------------------------------------------
# DEAD-SITE ASSERTIONS -- the present-wired-never-populated guard.
#
# Sites 2-apply and 5 sit inside functions that need a live scheduler to
# exercise, so a behavioural test cannot reach them here and a DELETED call
# would stay green -- which is exactly the class this build exists to fix. An
# AST check is the honest instrument for that: it asserts the wiring EXISTS at
# the named site, and says so rather than pretending to test behaviour.
# --------------------------------------------------------------------------

import ast  # noqa: E402
import inspect  # noqa: E402


def _calls_in(module, func_name):
    src = inspect.getsource(module)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return {
                (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None))
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
            }
    raise AssertionError(f"{func_name} not found in {module.__name__}")


def test_site1_is_wired_into_the_upward_channel():
    from sglang.srt.managers import scheduler_pp_mixin as m

    assert "pp_stamp_observed_coverage" in _calls_in(
        m, "pp_output_payload_with_return_trip"
    ), "SITE 1 DEAD: the producer is not called on the return trip"


def test_site2_harvest_is_wired_into_record_return_trip():
    from sglang.srt.managers import pp_admission_congruence as c

    assert "note_observed_coverage" in _calls_in(c, "record_return_trip"), (
        "SITE 2 FEED DEAD: promises arrive and are never harvested"
    )


def test_site2_min_is_wired_into_pp0s_build():
    from sglang.srt.managers import pp_admission_congruence as c

    assert "uniform_prefix_for" in _calls_in(c, "build_pp_admission_decision"), (
        "SITE 2 APPLY DEAD: PP0 publishes its own number, not the group MIN "
        "-- this is option 2, which has no mechanism behind it"
    )


def test_site5_is_wired_into_the_readmission_consult():
    from sglang.srt.managers import schedule_batch as sb

    calls = _calls_in(sb, "init_next_round_input")
    assert "apply_uniform_pass_geometry_1059" in calls, (
        "SITE 5 DEAD: the told geometry is never applied and every rank keeps "
        "deriving its own width -- boot 27"
    )


def test_site5_actually_moves_the_prefix_not_merely_calls_the_helper():
    """BEHAVIOURAL, because the AST check alone was a FALSE GREEN.

    Measured on my own build: a mutant that kept the call and discarded its
    result (`None and uniform_pass_geometry(...)`) passed all 34 tests. A site
    whose deadness cannot be detected is the very shape this change removes, so
    site 5 is now asserted by EFFECT.
    """
    from sglang.srt.managers.schedule_batch import Req

    req = Req.__new__(Req)
    req.prefix_indices = list(range(13376))
    req.cache_protected_len = 13376
    req._1059_told_prefix = 4094
    req._1059_told_extend = 4096

    assert req.apply_uniform_pass_geometry_1059() is True
    assert len(req.prefix_indices) == 4094, "SITE 5 DEAD: the prefix did not move"
    assert req._1059_applied.extend == 4096


def test_site5_no_told_fact_is_a_no_op():
    from sglang.srt.managers.schedule_batch import Req

    req = Req.__new__(Req)
    req.prefix_indices = list(range(777))
    req.cache_protected_len = 777
    assert req.apply_uniform_pass_geometry_1059() is False
    assert len(req.prefix_indices) == 777


def test_site4_stamps_the_told_geometry_onto_the_req():
    """The carrier of the fact from the receive point to the apply point."""
    from sglang.srt.managers import scheduler_pp_mixin as m

    src = inspect.getsource(m)
    assert "_1059_told_prefix" in src and "_1059_told_extend" in src, (
        "SITE 4 DEAD: the told geometry never reaches the Req"
    )
