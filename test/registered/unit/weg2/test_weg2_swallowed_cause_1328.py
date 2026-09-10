"""#1328 -- a swallowed exception that reports only its TYPE is not a finding.

BOOT weg2xsn6 (@ 2afbecf601) LEFT THIS AND NOTHING MORE, on 24 of 24 legs in
both groups:

    WEG2-XCHG-SHADOW leg=7 ... ran=no why=no-plan classes=0 pieces=0 ...
    manifest=manifest-failed:AttributeError

The shadow's observer arms may never raise into a flip leg, so they catch
`BaseException` and return a reason string -- and that string was
`type(exc).__name__` alone. So the boot recorded that *an* AttributeError
happened *somewhere*, and named neither the attribute nor the site.

THE COST IS MEASURED, NOT ASSERTED: three separate hypotheses were built on
that one word and each was refuted --
  * `region.boot_hash` -- refuted: `XchgRegion.__init__` sets it unconditionally
    (weight_exchange_region.py:538), so every region has it;
  * `agreed.theirs` -- refuted: `AgreedPieces` carries `theirs: int`, and the
    adapter guards the None case;
  * `derive_card_manifest` -- refuted by local reproduction: its failures are
    CLEAN RETURNS (`no-model`, `no-ring-layout:...`), and `reconcile_card_manifest`
    plus `manifest_state_message` run green in isolation and print `peer-absent`.
An instrument built to swallow the one fact that would have ended this in a
single read is the defect; the attribute is still unnamed and that is the point.

WHAT THIS SLICE DELIBERATELY DOES **NOT** DO: allocate W85. The briefing asked
for "a refusal with a reason, W85 is free", on the premise that
`ran=no why=no-plan` printed as a normal exit. Measured against the boot log,
it does not: `W79 Weg2XchgShadowRankLocalSkip` fired **27x on P and 24x on D**,
already carries `reason=` and `detail=`, and the manifest reason already rides
into it (`manifest=` 24 hits). The refusal HAS a name. A second code for the
same refusal is the second bookkeeping this tree deletes on sight, and the
W-code census law is one name per refusal, used exactly once. What was missing
was never the name -- it was the CAUSE.

DANGER DIRECTION is an instrument that looks informative and is not, so the
mutants must make the suite red by removing information while keeping shape:

  M1  the note must carry the MESSAGE, not just the type
      -> test_the_note_names_the_message
  M2  the note must carry the SITE that raised
      -> test_the_note_names_the_site_that_raised
  M3  describing a failure may never raise
      -> test_the_note_survives_a_hostile_exception
  M4  the note must stay ONE grep-able field
      -> test_the_note_is_one_bounded_newline_free_field
  M5  the adapters must actually use it
      -> test_both_observer_arms_report_the_note_not_the_bare_type
"""

import pytest

from sglang.srt.managers.scheduler_components import weight_updater as wu


def _raised(fn):
    """Return the exception with a REAL traceback, as the arms receive it."""
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001
        return exc
    raise AssertionError("the fixture did not raise")


# --------------------------------------------------------------------------
# M1 / M2: the two facts a bare type cannot carry
# --------------------------------------------------------------------------


def test_the_note_names_the_message():
    """M1: `AttributeError` alone cost a desk pass; the attribute ends it."""
    exc = _raised(lambda: None.some_missing_attribute)
    note = wu._weg2_exc_note(exc)
    assert note.startswith("AttributeError")
    assert "some_missing_attribute" in note, (
        "the attribute name is the whole reason this exists -- xsn6 reported "
        "`manifest-failed:AttributeError` and three hypotheses were built on it"
    )


def test_the_note_names_the_site_that_raised():
    """M2: the LAST frame of the exception's own traceback, file:line.

    The site is the one fact a type cannot carry, and it is what turns the
    next boot's line into a `sed -n` instead of a hypothesis.
    """
    exc = _raised(lambda: None.another_missing_one)
    note = wu._weg2_exc_note(exc)
    assert " @ " in note, note
    tail = note.split(" @ ", 1)[1]
    assert tail.startswith(__file__.rsplit("/", 1)[-1]), tail
    assert ":" in tail and tail.rsplit(":", 1)[1].isdigit(), tail
    # The basename, never the absolute path -- the field rides an existing
    # line and a 90-character path would push the rest of it out of view.
    assert "/" not in tail


# --------------------------------------------------------------------------
# M3 / M4: the instrument's own failure modes
# --------------------------------------------------------------------------


def test_the_note_survives_a_hostile_exception():
    """M3: an instrument that raises while describing a failure replaces the
    finding with its own -- the worst outcome available to it."""

    class Hostile(RuntimeError):
        def __str__(self):
            raise ValueError("I refuse to be described")

    exc = _raised(lambda: (_ for _ in ()).throw(Hostile()))
    note = wu._weg2_exc_note(exc)
    assert "Hostile" in note, note


def test_the_note_is_one_bounded_newline_free_field():
    """M4: it rides an EXISTING line, so it must not break the grep."""
    exc = _raised(lambda: None.x)
    try:
        raise ValueError("line one\nline two\twith  runs   of space" + "z" * 400)
    except ValueError as e:
        wide = wu._weg2_exc_note(e)
    assert "\n" not in wide and "\t" not in wide
    assert len(wide) <= 120, len(wide)
    assert "  " not in wide, "collapsed whitespace keeps the field one token-run"
    # And the limit is a parameter, not a literal buried in the body.
    assert len(wu._weg2_exc_note(exc, limit=20)) <= 20


# --------------------------------------------------------------------------
# M5: wired, not merely present
# --------------------------------------------------------------------------


def test_both_observer_arms_report_the_note_not_the_bare_type():
    """M5: PRESENT-AND-WIRED, not PRESENT-BUT-UNWIRED.

    A source pin, matched to the error class: the defect was a formatting
    choice at two `except BaseException` sites, and what must be true is that
    neither still formats the bare type.
    """
    import inspect

    src = inspect.getsource(wu)
    assert 'f"manifest-failed:{_weg2_exc_note(exc)}"' in src
    assert 'f"derivation-failed:{_weg2_exc_note(exc)}"' in src
    assert 'f"manifest-failed:{type(exc).__name__}"' not in src
    assert 'f"derivation-failed:{type(exc).__name__}"' not in src


def test_no_second_w_code_was_allocated_for_a_refusal_that_has_one():
    """The decision, pinned so it is not silently reversed.

    `W79 Weg2XchgShadowRankLocalSkip` is the shadow's rank-local refusal and it
    fired 27x/24x on weg2xsn6 with `reason=` and `detail=` already populated.
    W85 stays FREE: one name per refusal is the census law, and a second name
    for the same event is what makes a boot log ungrepable.
    """
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    assert sh.RANK_LOCAL_SKIP_MARKER == "W79 Weg2XchgShadowRankLocalSkip"
    msg = sh.rank_local_skip_message(
        reason="no-plan", rank=0, leg=0, epoch="e",
        detail="manifest=manifest-failed:AttributeError: 'X' has no 'y' @ f.py:1")
    assert msg.startswith("W79 Weg2XchgShadowRankLocalSkip")
    assert "reason=no-plan" in msg
    assert "manifest-failed:AttributeError" in msg, (
        "the cause must reach the named refusal's own line -- that is the "
        "chain #1328 completes, and it is why no new code is needed"
    )
    for mod in (wu, sh):
        assert "W85" not in inspect_source(mod), "W85 must stay free"


def inspect_source(mod):
    import inspect

    return inspect.getsource(mod)
