# SPDX-License-Identifier: Apache-2.0
"""The seam registry checks itself against the tree (slice 6).

Why this file exists, measured rather than supposed: in slice 5 the F4 entry
still cited `distributed/utils.py:1430` and `:1180`, and both had become
other code -- the function had moved to :1478. A seam list whose file:line
have drifted is WORSE than no seam list, because it is believed: the next
reader opens the wrong line, finds nothing, and concludes the seam was
imagined.

So every seam carries `anchors` -- the same claim as its prose `where`, but
in a form a test can falsify -- and this file falsifies them. It is the one
test in the Form A set that fails when the CODE moves rather than when the
code breaks, which is exactly its job.
"""

import pathlib

import pytest

from sglang.srt.rank_role import SEAMS

SRT = pathlib.Path(__file__).resolve().parents[2] / "python" / "sglang" / "srt"
#: How far an anchor may drift before it must be re-pinned. Small on purpose:
#: a window wide enough to absorb real drift is a window wide enough to match
#: the wrong thing.
WINDOW = 4


def _anchor_cases():
    for sid, seam in SEAMS.items():
        for path, line, needle in seam.anchors:
            yield pytest.param(sid, path, line, needle, id=f"{sid}-{path}:{line}")


def test_every_seam_carries_at_least_one_checkable_anchor():
    """Prose alone is not a citation -- it cannot go stale loudly."""
    missing = [sid for sid, seam in SEAMS.items() if not seam.anchors]
    assert not missing, f"seams with no checkable anchor: {missing}"


@pytest.mark.parametrize("sid,path,line,needle", list(_anchor_cases()))
def test_the_anchor_still_points_at_what_it_claims(sid, path, line, needle):
    f = SRT / path
    assert f.is_file(), f"{sid}: {path} does not exist (moved or renamed?)"
    lines = f.read_text(errors="replace").splitlines()
    assert line <= len(lines), (
        f"{sid}: {path} has {len(lines)} lines but the anchor cites {line} -- "
        "the file shrank under the citation."
    )
    lo = max(0, line - 1 - WINDOW)
    hi = min(len(lines), line + WINDOW)
    window = "\n".join(lines[lo:hi])
    assert needle in window, (
        f"{sid}: {path}:{line} no longer contains {needle!r} within "
        f"+-{WINDOW} lines. Re-pin the anchor (and check the prose in "
        f"`where` while you are there); do NOT widen the window."
    )


def test_the_prose_and_the_anchors_name_the_same_files():
    """`where` is what a human reads; `anchors` is what the test checks. If
    they name different files, one of them is lying and there is no way to
    tell which -- so they are pinned to agree."""
    for sid, seam in SEAMS.items():
        for path, _line, _needle in seam.anchors:
            stem = path.split("/")[-1]
            assert stem in seam.where, (
                f"{sid}: anchor file {stem} is not mentioned in the prose "
                f"`where` ({seam.where!r})."
            )


def test_a_wired_seam_anchors_the_code_that_wires_it():
    """A seam marked wired must point at something that exists, not merely
    at the place where the work WOULD go. Guards against a seam being
    flipped to wired in the registry and nowhere else."""
    for sid, seam in SEAMS.items():
        if seam.wired:
            assert seam.anchors, f"{sid} claims wired but anchors nothing"
