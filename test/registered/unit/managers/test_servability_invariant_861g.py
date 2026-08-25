# SPDX-License-Identifier: Apache-2.0
"""#861g: no request state may be vetoed in EVERY layout. The desk check.

TWO DEADLOCKS IN ONE NIGHT, both from independently-correct vetoes:

  #858    strict purity x #856 no-carry -> 150 flips, ZERO decode batches
  W37-E   #861e decode-hold x #861d seam premise x strict purity
          -> 7 requests unservable in either layout, GPU 0 % for 198 s

Two is a class, and each instance cost a GPU window. Servability is a property
of the CONJUNCTION of the gates, and nothing else in the tree checks the
conjunction -- every term has its own tests and passes them.

THE INVARIANT: for every request state, at least one layout can serve it with
ALL gates evaluated together. A state vetoed everywhere is a latent deadlock.
"""

import pytest

from sglang.srt.managers.servability_matrix import (
    PP,
    REQUEST_STATES,
    TP,
    VETOES_861J,
    VETOES_AS_SHIPPED,
    VETOES_FIXED,
    deadlocked_cells,
    decode_reachable,
    orbit_deadlocked_cells,
    render_table,
    seam_retract_map,
    served_in,
    servable,
)


def test_the_invariant_holds_with_the_root_fix():
    """THE GATE. Empty means every state has a route to service."""
    bad = deadlocked_cells(VETOES_FIXED)
    assert not bad, "latent deadlock(s):\n" + render_table(VETOES_FIXED)


def test_the_matrix_reproduces_the_W37E_deadlock_as_shipped():
    """RED FIXTURE 1. The harness must independently find the deadlock that
    cost a window -- otherwise it is a green light measuring nothing."""
    bad = dict(deadlocked_cells(VETOES_AS_SHIPPED))
    assert "retracted-with-output" in bad, render_table(VETOES_AS_SHIPPED)
    blamed = set(bad["retracted-with-output"])
    assert {"decode-work-hold", "seam-transport-premise", "strict-purity"} <= blamed


def test_only_that_one_cell_was_closed_as_shipped():
    """The fix must close the deadlock without opening another: exactly one
    cell differs between the two veto sets."""
    before = {n for n, _ in deadlocked_cells(VETOES_AS_SHIPPED)}
    after = {n for n, _ in deadlocked_cells(VETOES_FIXED)}
    assert before - after == {"retracted-with-output"}
    assert after == set()


def test_858_pair_strict_purity_alone_does_not_strand_prefill():
    """RED FIXTURE 2, the #858 shape: strict purity forbids prefill in TP, so
    a queued request MUST remain servable in PP or the pair deadlocks."""
    queued = next(s for s in REQUEST_STATES if s.name == "queued-never-started")
    assert served_in(queued, TP, VETOES_FIXED) is False, "purity still forbids it in tp"
    assert served_in(queued, PP, VETOES_FIXED) is True, "and pp must still take it"


@pytest.mark.parametrize("state", REQUEST_STATES, ids=lambda s: s.name)
def test_every_state_is_servable_somewhere(state):
    assert servable(state, VETOES_FIXED), (
        f"{state.name} is vetoed in BOTH layouts -- a latent deadlock of the "
        f"#858 / W37-E class"
    )


def test_exactly_one_side_claims_a_retracted_request():
    """THE FIX PRINCIPLE, pinned. W37-E's root was that the decode side CLAIMED
    a retracted-with-output request ("decode work in flight") while only the
    prefill side could SERVE it -- so both vetoed and neither served. Exactly
    one side must claim it."""
    r = next(s for s in REQUEST_STATES if s.name == "retracted-with-output")
    assert r.needs_prefill is True, "it is prefill work"
    assert r.resident_decoding is False, "it is NOT decode work"


def test_the_orbit_axis_rediscovers_the_W37F_specimen():
    """#861j: the INPUT-MODEL gap, closed. The flat matrix passed at the desk
    (VETOES_FIXED has no deadlocked cell) while three metal boots produced
    ZERO decode batches -- because 'servable in PP' is not 'ever decodes'.
    The runnability axis walks the flip orbit under the #856 retraction map
    and must find at the desk what the boots found on metal: with the
    as-shipped premise, NOTHING ever decodes -- not even a fresh request,
    exactly as `Decode batch phase=` == 0 measured on w37d4/w37f/w37f2."""
    assert not deadlocked_cells(VETOES_FIXED), "the flat matrix is green..."
    dead = dict(orbit_deadlocked_cells(VETOES_FIXED))
    assert "retracted-with-output" in dead, "...and the orbit must still fail"
    assert "queued-never-started" in dead, (
        "on metal even fresh requests never decoded: prefilled in PP, "
        "retracted at the flip, refused re-materialization in TP"
    )
    assert "seam-transport-premise" in dead["retracted-with-output"]


@pytest.mark.parametrize("state", REQUEST_STATES, ids=lambda s: s.name)
def test_decode_is_reachable_for_every_state_with_861j(state):
    """THE NEW GATE: with both W37-F doors fixed, every state's orbit reaches
    a decode step in TP within a bounded number of flips."""
    assert decode_reachable(state, VETOES_861J), (
        f"{state.name} never decodes on the flip orbit -- the W37-F shape: "
        f"servable somewhere every round, and never once served in the "
        f"decode layout"
    )


def test_cold_transport_is_still_refused_in_tp_under_861j():
    """CAN-FAIL, the dangerous direction: a request retracted before any fill
    carries no restore evidence, and admitting it in TP would be real work in
    the wrong layout (W37-D). Its route stays PP-first."""
    cold = next(s for s in REQUEST_STATES if s.name == "retracted-without-output")
    assert served_in(cold, TP, VETOES_861J) is False
    assert decode_reachable(cold, VETOES_861J), "but its orbit still decodes via PP"


def test_the_seam_map_is_the_856_retraction():
    resident = next(s for s in REQUEST_STATES if s.name == "decoding-resident")
    queued = next(s for s in REQUEST_STATES if s.name == "queued-never-started")
    assert seam_retract_map(resident).name == "retracted-with-output"
    assert seam_retract_map(queued) is queued


def test_the_state_list_covers_the_states_the_night_produced():
    names = {s.name for s in REQUEST_STATES}
    for required in (
        "queued-never-started",
        "chunk-prefilling",
        "decoding-resident",
        "retracted-with-output",
        "retracted-without-output",
        "flip-carried-parked",
        "spilled-awaiting-resume",
    ):
        assert required in names, required
