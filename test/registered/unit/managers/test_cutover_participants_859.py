# SPDX-License-Identifier: Apache-2.0
"""#859 SEED: the cutover's participant registry is checkable, not remembered.

CLASS: "the cutover is not a first-class operation" -- it is a sequence of
steps that each move the piece they know about, and every piece nobody
remembered is found by a boot. Seven blockers in one night were all this class.

FUTURE-CHECK: this file. It cannot find a participant nobody has thought of --
no test can -- but it makes three things impossible without a test going red:

  1. registering a participant without naming WHO handles it at the seam;
  2. registering one without naming HOW you would know the hook ran (the #719
     lesson: W36 built a heartbeat so "clean" and "blind" could not look
     identical, W37-C logged `checked=0` eighteen times, and the mechanism was
     still blind -- a hook without a probe is a hook you cannot prove ran);
  3. letting a named hook or probe symbol be renamed or deleted out from under
     the registry, which is how a registry rots into a document.

Gaps are ALLOWED and must be EXPLAINED. An unbuilt obligation carries
``hook=None``/``probe=None`` plus a ``gap`` naming what is missing, so the
backlog is enumerable from code (``participants_with_gaps()``) rather than
living in institutional memory.
"""

import importlib

import pytest

from sglang.srt.managers.cutover_participants import (
    LOG,
    REGISTRY,
    Participant,
    participants_found_by_boot,
    participants_with_gaps,
)


def _resolve(path: str):
    """Import a dotted symbol path, or raise with the path in the message."""
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:cut])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        obj = module
        for attr in parts[cut:]:
            obj = getattr(obj, attr)  # AttributeError names the missing piece
        return obj
    raise ImportError(f"no importable module prefix in {path!r}")


def test_registry_is_not_empty():
    assert len(REGISTRY) >= 10, (
        "a registry small enough to hold in your head is a registry that does "
        "not need to exist"
    )


@pytest.mark.parametrize("p", REGISTRY, ids=lambda p: p.name)
def test_every_participant_names_its_obligations_or_its_gap(p: Participant):
    """Rule 1 and 2: a hook and a probe, or an explained gap."""
    assert p.what and p.ticket, p.name
    if p.hook is None or p.probe is None:
        assert p.gap, (
            f"{p.name} has an unbuilt obligation and no `gap` explaining what "
            f"is missing. An unexplained gap is indistinguishable from an "
            f"oversight, which is the whole failure mode this registry exists "
            f"to remove."
        )


@pytest.mark.parametrize(
    "p", [p for p in REGISTRY if p.hook], ids=lambda p: p.name
)
def test_every_named_hook_exists(p: Participant):
    """Rule 3: a registry whose symbols have drifted is a document."""
    _resolve(p.hook)


@pytest.mark.parametrize(
    "p", [p for p in REGISTRY if p.probe and not p.probe.startswith(LOG)],
    ids=lambda p: p.name,
)
def test_every_named_symbol_probe_exists(p: Participant):
    _resolve(p.probe)


def test_log_probes_are_marked_as_such():
    """A probe that is a log substring must SAY so, so the test above does not
    try to import it and so a reader can tell the two kinds apart."""
    for p in REGISTRY:
        if p.probe and not p.probe.startswith(LOG):
            assert "." in p.probe, f"{p.name}: {p.probe!r} is neither a dotted symbol nor a {LOG} marker"


def test_the_backlog_is_enumerable_from_code():
    """`participants_with_gaps()` IS the #859 backlog. A backlog nobody can
    enumerate is a backlog rediscovered by a boot."""
    gaps = participants_with_gaps()
    assert all(p.gap for p in gaps)
    # The still-open ones this night surfaced must be on it BY NAME. Both are
    # MISSING PROBES, which is the #719 shape: a hook exists and nothing proves
    # it ran, so "handled nothing because nothing needed it" and "handled
    # nothing because the reach missed it" are byte-identical.
    open_names = {p.name for p in gaps}
    assert "carried_batch_spec_algorithm" in open_names
    assert "carried_batch_spec_info" in open_names


def test_the_boot_paid_column_is_the_argument_for_this_file():
    """Every entry marked found_by='boot' cost a GPU window to discover."""
    paid = participants_found_by_boot()
    assert len(paid) >= 8, (
        f"only {len(paid)} participants recorded as boot-discovered; the "
        f"registry's justification is exactly this count"
    )


def test_names_are_unique():
    names = [p.name for p in REGISTRY]
    assert len(names) == len(set(names))


def test_can_fail_a_participant_missing_both_obligations_and_a_gap():
    """CAN-FAIL: the shape the registry must refuse."""
    bad = Participant(
        name="x", what="y", hook=None, probe=None, ticket="#0", gap=None
    )
    with pytest.raises(AssertionError):
        test_every_participant_names_its_obligations_or_its_gap(bad)


def test_can_fail_a_hook_that_does_not_exist():
    bad = Participant(
        name="x",
        what="y",
        hook="sglang.srt.managers.cutover_participants.no_such_symbol",
        probe="log:whatever",
        ticket="#0",
    )
    with pytest.raises(AttributeError):
        test_every_named_hook_exists(bad)
