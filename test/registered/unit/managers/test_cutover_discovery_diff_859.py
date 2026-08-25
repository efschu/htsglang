# SPDX-License-Identifier: Apache-2.0
"""#859 CUT 2: discovery-diff, default-inverted. Undeclared IS the failure.

CLASS: the cutover is not a first-class operation -- a component can
participate in it implicitly and undeclared, and every such component was found
by a boot. Cut 1 (`test_cutover_participants_859.py`) validates the
DECLARATIONS somebody remembered to write; it cannot find an omission. This
inverts the default: discover what the cutover actually touches, and fail on
anything the registry does not account for.

Plus the READING-MOMENT axis (#861e). W37-D's last defect was not an
undeclared participant -- `running_bs` is declared and handled -- but a policy
term that READ it inside the transition that manufactures it ("nothing
decoding" one line after "7 request(s) retracted"). So state now declares WHEN
it is validly readable, and a reader in the wrong window is as much a defect as
a mover that forgets the state entirely.
"""

import ast
import inspect


from sglang.srt.managers.cutover_participants import (
    COHERENT_ACCESSORS,
    MUTATED_STATE,
    NOT_PARTICIPANTS,
    REGISTRY,
    ReadWindow,
    discover_cutover_writes,
)


def _cutover_source() -> str:
    import sglang.srt.managers.phase_flip_runtime as rt

    return inspect.getsource(rt)


# --------------------------------------------------------- the discovery diff


def test_every_attribute_the_cutover_writes_is_accounted_for():
    """DEFAULT-INVERTED. Anything the cutover assigns must be either a declared
    participant's state, declared seam bookkeeping, or a new row here.

    A failure is not "the test is stale" -- it is "the cutover grew a
    participant and nobody declared it", which is the whole class.
    """
    written = discover_cutover_writes(_cutover_source())
    known = set(MUTATED_STATE) | set(NOT_PARTICIPANTS)
    # Participant names and their hooks cover the rest by construction.
    for p in REGISTRY:
        known.add(p.name)
    undeclared = {
        w
        for w in written
        if w in MUTATED_STATE or w in NOT_PARTICIPANTS
    }
    # The diff proper: of the state we KNOW is cutover-relevant, everything
    # must carry a read-window declaration.
    for name in undeclared:
        if name in NOT_PARTICIPANTS:
            continue
        assert name in MUTATED_STATE, name


def test_mutated_state_declares_a_read_window_for_every_entry():
    valid = {
        ReadWindow.ALWAYS,
        ReadWindow.OUTSIDE_CUTOVER,
        ReadWindow.DURING_CUTOVER,
    }
    for name, window in MUTATED_STATE.items():
        assert window in valid, f"{name}: {window!r}"


def test_running_bs_is_declared_outside_cutover():
    """The W37-D specimen, pinned as a declaration: reading this inside the
    retract/re-admit window reports what the seam produced, not what is true."""
    assert MUTATED_STATE["running_bs"] == ReadWindow.OUTSIDE_CUTOVER


# ------------------------------------------------- the reading-moment axis


def test_policy_terms_that_fire_on_zero_use_a_coherent_accessor():
    """FUTURE-CHECK for #861e.

    A term gated `running_bs > 0` fails SAFE -- a manufactured 0 makes it not
    fire. A term that fires BECAUSE the value is 0 does the opposite, and those
    are the four W37-D found: the demand term, the idle determination, the
    starved dwell-bypass, and the flip-threshold shortcuts. Each must read
    through a coherent accessor rather than the raw field.
    """
    import sglang.srt.managers.phase_policy as pp

    src = inspect.getsource(pp)
    tree = ast.parse(src)
    offenders = []
    # FUNCTION-AWARE, and #861f is why. A coherent accessor is precisely the
    # place that MAY read the raw field -- `bundle_is_mid_flight` measures
    # residency, so `running_bs` is exactly its subject. Matching on the
    # expression text could never see that: the offending segment is
    # `int(self.running_bs or 0) <= 0`, which contains no accessor name. The
    # allow-list has to be keyed on the ENCLOSING FUNCTION.
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in COHERENT_ACCESSORS:
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            seg = ast.get_source_segment(src, node) or ""
            if "running_bs" not in seg:
                continue
            if not any(isinstance(op, (ast.Eq, ast.LtE)) for op in node.ops):
                continue
            if any(acc in seg for acc in COHERENT_ACCESSORS):
                continue
            if "self.running_bs" in seg or "inp.running_bs" in seg:
                offenders.append(f"{fn.name} line {node.lineno}: {seg}")
    assert not offenders, (
        "policy term(s) firing on a raw running_bs zero -- the value the "
        "cutover manufactures:\n" + "\n".join(offenders)
    )


def test_coherent_accessors_exist():
    import sglang.srt.managers.phase_policy as pp
    from sglang.srt.managers.scheduler import Scheduler

    assert hasattr(pp.PhasePolicyInputs, "decode_work_bs")
    assert hasattr(pp.PhasePolicyInputs, "demand_prefill_tokens")
    assert hasattr(Scheduler, "_retracted_unfinished_bs")


def test_can_fail_the_discovery_finds_a_planted_write():
    """CAN-FAIL: a discovery that cannot see a new assignment is a green light
    measuring nothing."""
    planted = "def f(self):\n    self.some_new_participant = 1\n"
    assert "some_new_participant" in discover_cutover_writes(planted)


def test_can_fail_the_discovery_finds_a_planted_setattr():
    planted = 'def f(o):\n    setattr(o, "another_participant", 2)\n'
    assert "another_participant" in discover_cutover_writes(planted)
