# SPDX-License-Identifier: Apache-2.0
"""#861f: COMBINATION falsifiers -- terms that pass alone and deadlock together.

THE CEILING THIS ANSWERS. Three windows in a row the sweep was incomplete, and
W37-E found the reason: every fix carried BOTH its own symmetry pins and still
deadlocked, because the pins asked "does THIS term fire correctly" and never
"does this term fire while the OTHER terms are silent for their own reasons".

W37-E's specimen, all three legs true at once, every leg a fix of mine:

  #861e   7 retracted-WITH-output -> decode_work_bs 7 -> demand SILENT
  #861d   no cached prefix -> SEAM TRANSPORT REFUSED -> no read-through in tp
  purity  prefill cannot run in tp by law

Result: not servable in EITHER layout. Flips frozen at 9, GPU 0 % on three
cards, 7 queued, no first token for 198 s.

THE RULE THIS FILE ENFORCES: for every policy term, drive the OTHER terms'
silence conditions simultaneously and assert the system still has an exit. A
term is not correct because it is correct alone.
"""


import pytest

from sglang.srt.managers.phase_policy import (
    MIN_DECODE_STEPS_PER_PHASE,
    PhasePolicyInputs,
)


def mk(**kw):
    base = dict(phase="tp", pending_prefill_tokens=0, running_bs=0, now=0.0)
    base.update(kw)
    return PhasePolicyInputs(**base)


# ------------------------------------------------------- THE DEADLOCK FIXTURE


def w37e_deadlock_inputs():
    """The metal state, field for field.

    7 requests retracted by the cutover, each having produced >=1 output token
    (OUTTRACE showed n=1), all sitting in the waiting queue. Because they
    produced output, `reset_for_retract` credited them their ENTIRE prompt as
    cached, so every backlog counter that subtracts that credit reads 0.
    Nothing is resident.
    """
    return mk(
        phase="tp",
        pending_prefill_tokens=0,      # full retract credit -> reads 0
        admissible_prefill_tokens=0,   # same, via cache_protected_len
        running_bs=0,                  # nothing resident
        retracted_unfinished_bs=7,     # the 7, parked in the queue
        decode_steps_this_phase=0,
    )


def test_the_deadlock_fixture_has_an_exit():
    """MUST FAIL on 02bd70681c and pass here.

    On the old formulation `decode_work_bs()` returned 7 (running 0 + retracted
    7), the demand term returned 0, and nothing armed -- the wedge. The exit is
    that a retracted-unfinished request is PREFILL work, so no bundle is
    mid-flight and the demand is free to fire.
    """
    inp = w37e_deadlock_inputs()
    assert inp.decode_work_bs() == 0, (
        "a retracted request is not decode work: counting it as such is what "
        "left 7 requests unservable in either layout"
    )
    assert inp.bundle_is_mid_flight() is False, (
        "nothing is resident, so nothing can be chopped by a flip"
    )


def test_the_deadlock_fixture_still_needs_a_backlog_signal():
    """HONEST SCOPE. The exit above unblocks the HOLD; it does not by itself
    make the demand fire, because both backlog counters still read 0 for the
    full-credit reason. This pins that the remaining gap is the CREDIT
    SEMANTICS, not the hold -- so a future reader cannot mistake this fixture
    for a claim that the whole specimen is solved by the hold fix alone.
    """
    inp = w37e_deadlock_inputs()
    assert inp.demand_prefill_tokens() == 0
    # ...and with an honest backlog signal the exit is complete:
    fixed = mk(
        admissible_prefill_tokens=18586,
        running_bs=0,
        retracted_unfinished_bs=7,
        decode_steps_this_phase=0,
    )
    assert fixed.demand_prefill_tokens() == 18586


# --------------------------------------------- the three prior specimens hold


def test_d4_thrash_a_live_bundle_is_still_protected():
    """d4: residents decoding, few steps done -> hold. The protection moves
    from "requests exist somewhere" to "this phase owes the bundle steps"."""
    inp = mk(admissible_prefill_tokens=18586, running_bs=2, decode_steps_this_phase=1)
    assert inp.bundle_is_mid_flight() is True
    assert inp.demand_prefill_tokens() == 0


def test_d4_after_a_fair_share_of_steps_the_flip_is_allowed():
    """AND IT TERMINATES. The d4 hold must not become the W37-E wedge: once the
    bundle has had its steps, queued work wins."""
    inp = mk(
        admissible_prefill_tokens=18586,
        running_bs=2,
        decode_steps_this_phase=MIN_DECODE_STEPS_PER_PHASE,
    )
    assert inp.bundle_is_mid_flight() is False
    assert inp.demand_prefill_tokens() == 18586


def test_d2_wedge_still_fires():
    inp = mk(admissible_prefill_tokens=5988, running_bs=0, retracted_unfinished_bs=0)
    assert inp.demand_prefill_tokens() == 5988


def test_d3_pingpong_genuine_decoders_still_silence_the_demand():
    inp = mk(admissible_prefill_tokens=5988, running_bs=2, decode_steps_this_phase=0)
    assert inp.demand_prefill_tokens() == 0


def test_nothing_anywhere_is_silent():
    assert mk().demand_prefill_tokens() == 0


# ------------------------------------------------- the combination rule itself


COMBINATION_MATRIX = [
    # (name, inputs, must_have_exit)
    ("w37e deadlock", w37e_deadlock_inputs(), True),
    (
        "d4 live bundle, steps owed",
        mk(admissible_prefill_tokens=1, running_bs=2, decode_steps_this_phase=0),
        False,
    ),
    (
        "d4 bundle served its steps",
        mk(
            admissible_prefill_tokens=1,
            running_bs=2,
            decode_steps_this_phase=MIN_DECODE_STEPS_PER_PHASE + 1,
        ),
        True,
    ),
    ("d2 wedge", mk(admissible_prefill_tokens=1, running_bs=0), True),
]


@pytest.mark.parametrize("name,inp,must_exit", COMBINATION_MATRIX, ids=lambda x: x if isinstance(x, str) else "")
def test_combination_matrix(name, inp, must_exit):
    """Every combination either has an exit or is deliberately holding a live
    bundle. A state with no exit and no live bundle is a deadlock by
    definition -- that is the invariant W37-E violated."""
    has_exit = not inp.bundle_is_mid_flight()
    assert has_exit is must_exit, name


def test_no_state_holds_without_something_resident():
    """THE INVARIANT, stated once. The policy may hold the layout ONLY while
    something is genuinely resident. Holding with nothing resident is the
    W37-E shape and can never be correct."""
    for bs in (0,):
        for steps in (0, 1, MIN_DECODE_STEPS_PER_PHASE, 99):
            for retracted in (0, 7):
                inp = mk(
                    running_bs=bs,
                    decode_steps_this_phase=steps,
                    retracted_unfinished_bs=retracted,
                )
                assert inp.bundle_is_mid_flight() is False, (bs, steps, retracted)


# ------------------------------------- #861i: EVERY INPUT MUST HAVE A PRODUCER


def test_every_policy_input_field_is_written_at_the_build_site():
    """#861i ZUKUNFTS-CHECK for the desk-written-never-executed class.

    W37-F: `decode_steps_this_phase` was DECLARED on PhasePolicyInputs, READ by
    `bundle_is_mid_flight()`, and WRITTEN NOWHERE. It defaulted to 0 for ever,
    so the anti-chop floor could only ever be decided by `running_bs` -- the
    value the cutover manufactures -- and the d4 thrash returned: 48 flips, 30
    decode batches, ZERO completions, GPU 0 % on all three cards.

    A guard whose input has no producer is not a weak guard, it is an ABSENT
    one that reads as present. The sibling sweep at the time found this was the
    only field of seven without a writer; this test keeps it that way.

    Deliberately checks the BUILD SITE rather than "anywhere in the tree": a
    field written only in tests is exactly the shape that shipped here.
    """
    import dataclasses
    import inspect

    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.managers.phase_policy import PhasePolicyInputs

    src = inspect.getsource(sched_mod)
    missing = [
        f.name
        for f in dataclasses.fields(PhasePolicyInputs)
        if f"{f.name}=" not in src
    ]
    assert not missing, (
        f"PhasePolicyInputs field(s) with no producer at the build site: "
        f"{missing}. They will read their default for ever, and any guard that "
        f"consults them is absent while looking present (#861i)."
    )
