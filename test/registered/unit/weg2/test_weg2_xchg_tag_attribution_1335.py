# SPDX-License-Identifier: Apache-2.0
"""#1335 (B4r): the COVER line stops mixing two books in one number.

OPERATOR DECISION 2026-09-11, recorded here because the alternative was
considered and REFUSED rather than forgotten: **Gate 0 is NOT wired.**
``weight_exchange_region.gate0_check`` carries the per-tag comparison of the
plan's claim against the saver's ledger and has zero production references, and
wiring it would add collectives to the wake seam whose re-entry HANG
(``:1367-1375``) the plan of record already names as a danger direction.  The
measurement that made that decision: **no bytes are missing.**  Summed over
every weights-family tag the saver is LARGER on every rank of weg2xsn19
(+72.9 MiB on P rank 0, +188.9 / +169.0 / +184.5 on D ranks 0/1/2), so a
negative per-tag figure is a BOOKING ARTEFACT, not a deficit.  A hang at the
cutover costs a boot; a misattributed tag boundary costs a diagnosis.

SO THE FIX IS THE LINE, AND THIS IS THE DEFECT IT REMOVES.  ``slack_bytes`` was
``tms_bytes - planned_bytes - buffers_bytes``: a SILENT SUBTRACTION ACROSS TWO
BOOKS.  ``planned``/``buffers`` come from ``walk_live_tensors``' region+name
attribution; ``tms_bytes`` is the SAVER's own per-tag ledger
(``model_runner._weg2_xchg_tag_bytes`` -> ``memory_saver_adapter.tag_bytes``).
Printed as one number called "slack", its negative side read as "the tag is too
small" -- and it cost four boots of that reading.

WHICH AUTHORITY MAY ANSWER "WHICH TAG DO THESE BYTES BELONG TO": THE SAVER, and
the argument is ownership, not convenience.  ``torch_memory_saver_adapter``
exposes ``pause(tag)``, ``resume(tag)`` and ``tag_bytes(tag)`` on ONE object,
keyed by the SAME tag string -- the tag that gets paused IS the saver's tag, and
its ledger is what decides which pages a flip releases and restores.  The walk's
region+name attribution never moves a page; it answers a DIFFERENT question
("which tag does this NAME suggest"), and conflating the two is the defect.
So the saver's book is the authority for tag membership, and the walk's total is
a CLAIM against it.

THE DANGER DIRECTION, stated because this change is print-only and could
otherwise look like cosmetics: this edit allocates nothing and cannot move a tag
boundary.  What it changes is which book a reader is invited to size from -- and
that is where the direction bites.  A tag sized from the WALK's book could be
SMALLER than what the saver actually holds for it, and a tag too small
CORRUPTS (the flip's wave would restore fewer pages than the tag owns); a tag
too large merely costs VRAM.  Hence the authority ruling above points at the
saver, the expensive-but-safe side, and hence ``slack_mib`` may never again
carry a negative number a reader could size down from.

THE MEASURED ROWS BELOW ARE weg2xsn19'S OWN, from the four ``WEG2-XCHG-COVER``
lines of ``tag=weights_draft`` plus two agreeing rows for contrast.  The
fixture is built from the PRINTED MiB figures, so every assertion is to 0.1 MiB
-- the precision the line itself carries.  No invented numbers.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

MIB = wx.MIB

#: weg2xsn19, `WEG2-XCHG-COVER ... tag=weights_draft`, verbatim fields.
#: (label, rank, planned_mib, buffers_mib, tms_mib, old_printed_slack_mib)
DRAFT_ROWS = (
    ("D rank2", 2, 1331.0, 64.1, 1280.0, -115.2),
    ("D rank1", 1, 1331.2, 64.1, 1280.0, -115.4),
    ("D rank0", 0, 1483.6, 64.1, 1440.0, -107.7),
    ("P pp2",   0, 1618.7, 64.1, 1572.0, -110.9),
)

#: The same boot's AGREEING rows, so the fix is not tested only on the defect.
#: `weights_0` is the cross-check that fixes the mechanism as attribution and
#: not measurement: it carries one extra ~64 MiB buffer (21 buffers / 64.2 MiB
#: against 20 / 0.2 on every other chunk tag) and THERE the saver agrees.
AGREE_ROWS = (
    ("weights_0 D rank1", "weights_0", 516.9, 64.2, 614.0, +33.0),
    ("weights_1 D rank1", "weights_1", 516.9, 0.2, 550.0, +33.0),
)


def _row(*, tag="weights_draft", rank=1, planned_mib, buffers_mib, tms_mib):
    """A ``TagCoverage`` from the PRINTED MiB figures of a real boot line."""
    return wx.TagCoverage(
        rank=rank,
        tag=tag,
        mode="exchange",
        planned_bytes=int(round(planned_mib * MIB)),
        buffers_bytes=int(round(buffers_mib * MIB)),
        tms_bytes=int(round(tms_mib * MIB)),
        uncovered=(),
        short=(),
        missing=(),
        n_parameters=20,
        n_buffers=8,
        n_attributes=0,
    )


def _fields(line: str) -> dict:
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


# ===========================================================================
# (1) THE WORD "slack" MAY NEVER AGAIN CARRY A NEGATIVE NUMBER.
# ===========================================================================


@pytest.mark.parametrize("label,rank,planned,buffers,tms,old", DRAFT_ROWS)
def test_a_diverging_row_prints_no_negative_slack(label, rank, planned, buffers, tms, old):
    """RED at 04cd920add: it prints exactly ``old``, and that is the reading
    that cost four boots.  ``slack`` means FREE ROOM; a cross-book difference is
    not free room in either sign, so the word is withheld rather than reused.
    """
    f = _fields(_row(rank=rank, planned_mib=planned, buffers_mib=buffers,
                     tms_mib=tms).cover_line())
    assert f["slack_mib"] == "n/a", (label, f["slack_mib"])
    assert not f["slack_mib"].startswith("-"), label


@pytest.mark.parametrize("label,rank,planned,buffers,tms,old", DRAFT_ROWS)
def test_the_cross_book_difference_gets_its_OWN_named_term(label, rank, planned, buffers, tms, old):
    """The #1026 form: the remainder BY CONSTRUCTION, with a name, never as an
    implicit difference a reader has to reconstruct.  Its value reproduces the
    old figure to the printed precision -- the ARITHMETIC was never wrong, only
    its name and the inference it invited.
    """
    f = _fields(_row(rank=rank, planned_mib=planned, buffers_mib=buffers,
                     tms_mib=tms).cover_line())
    assert "attribution_delta_mib" in f, f
    assert float(f["attribution_delta_mib"]) == pytest.approx(old, abs=0.15), (label, f)
    # And the WALK's book is printed as ONE named total, not left to be summed.
    assert float(f["walk_mib"]) == pytest.approx(planned + buffers, abs=0.15)


@pytest.mark.parametrize("label,rank,planned,buffers,tms,old", DRAFT_ROWS)
def test_the_line_NAMES_BOTH_AUTHORITIES(label, rank, planned, buffers, tms, old):
    """A delta between two books is unreadable without both books' names."""
    f = _fields(_row(rank=rank, planned_mib=planned, buffers_mib=buffers,
                     tms_mib=tms).cover_line())
    assert "tms_tag_bytes" in f["attribution"], f["attribution"]
    assert "region" in f["attribution"], f["attribution"]


@pytest.mark.parametrize("label,rank,planned,buffers,tms,old", DRAFT_ROWS)
def test_a_diverging_row_carries_a_GREPPABLE_verdict(label, rank, planned, buffers, tms, old):
    """The fix must make the finding MORE visible, not less.

    Withholding the negative number without a word in its place would trade a
    wrong reading for no reading at all -- so the state is named, and named so
    a boot record can count it.
    """
    f = _fields(_row(rank=rank, planned_mib=planned, buffers_mib=buffers,
                     tms_mib=tms).cover_line())
    assert f["attribution_verdict"] == "SAVER-BOOKS-LESS-THAN-WALK", f


# ===========================================================================
# (2) THE AGREEING SIDE, so the fix is not only tested where it fires.
# ===========================================================================


@pytest.mark.parametrize("label,tag,planned,buffers,tms,delta", AGREE_ROWS)
def test_an_agreeing_row_still_reports_slack_as_the_overhang(label, tag, planned, buffers, tms, delta):
    """Where the saver holds MORE than the walk accounts for, the difference IS
    free room and ``slack_mib`` is the right word.  +0.08..+0.58 GiB/rank is
    measured and normal, so this side must keep reading as it always has.
    """
    f = _fields(_row(tag=tag, planned_mib=planned, buffers_mib=buffers,
                     tms_mib=tms).cover_line())
    assert float(f["slack_mib"]) == pytest.approx(delta, abs=0.15), (label, f)
    assert float(f["attribution_delta_mib"]) == pytest.approx(delta, abs=0.15)
    assert f["attribution_verdict"] == "OVERHANG", f


# ===========================================================================
# (3) AN ABSENT SAVER ANSWER IS NOT A DEFICIT -- the documented trap, closed.
# ===========================================================================


def test_no_saver_answer_is_its_own_state_and_forms_no_delta():
    """The old code's own docstring named this and still printed a number.

    ``tms_bytes == 0`` means THE SAVER COULD NOT ANSWER (no hook, no symbol).
    Subtracting the walk's book from an absence produced a large negative
    "slack" indistinguishable from a real divergence -- an absence read as a
    deficit, which is the shape this whole ticket is about.  The existing test
    ``test_slack_is_printed_never_asserted`` asserted that negative number; it
    is REPLACED here rather than deleted, with its argument attached.
    """
    f = _fields(_row(planned_mib=1331.2, buffers_mib=64.1, tms_mib=0.0).cover_line())
    assert f["tms_answered"] == "no", f
    assert f["slack_mib"] == "n/a", f
    assert f["attribution_delta_mib"] == "n/a", (
        "a delta against an absence is not a measurement"
    )
    assert f["attribution_verdict"] == "NO-SAVER-ANSWER", f


# ===========================================================================
# (4) ONE AUTHORITY PER QUANTITY -- the #1333 shape, applied here.
# ===========================================================================


def test_the_two_book_property_is_GONE():
    """``slack_bytes`` computed the cross-book number and called it slack.

    Deleted, not renamed in place: the two readings now have two names
    (``overhang_bytes``, ``attribution_delta_bytes``) and the walk's book has
    one (``walk_bytes``), so no single attribute can mix the two again.
    """
    row = _row(planned_mib=1331.2, buffers_mib=64.1, tms_mib=1280.0)
    assert not hasattr(row, "slack_bytes")
    assert row.walk_bytes == row.planned_bytes + row.buffers_bytes
    assert row.attribution_delta_bytes == row.tms_bytes - row.walk_bytes
    # OVERHANG is None exactly when it is not free room.
    assert row.overhang_bytes is None
    agree = _row(tag="weights_0", planned_mib=516.9, buffers_mib=64.2, tms_mib=614.0)
    assert agree.overhang_bytes == agree.attribution_delta_bytes


def test_the_field_ORDER_keeps_tms_before_slack():
    """``test_weg2_xchg_cover_1273`` pins this ordering; the new fields are
    APPENDED around it rather than interleaved, the same rule ``cover_line``
    already states for everything after ``uncovered=``."""
    line = _row(planned_mib=1331.2, buffers_mib=64.1, tms_mib=1280.0).cover_line()
    assert line.index("tms_mib=") < line.index("slack_mib=")
    assert line.index("planned_mib=") < line.index("walk_mib=")


def test_a_reader_cannot_conclude_missing_bytes_from_this_line():
    """THE ACCEPTANCE, as the operator stated it.

    On the measured diverging row there is no negative number anywhere on the
    line, and the one word that could be read as a shortfall is absent.  The
    only signed figure is the NAMED cross-book delta, next to both book names.
    """
    line = _row(planned_mib=1331.2, buffers_mib=64.1, tms_mib=1280.0).cover_line()
    f = _fields(line)
    signed = {k: v for k, v in f.items() if v.startswith("-")}
    assert set(signed) == {"attribution_delta_mib"}, signed
    assert "attribution" in f and "attribution_verdict" in f
