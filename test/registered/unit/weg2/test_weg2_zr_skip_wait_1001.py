"""ZR-3 (01.10.): an END-state request in front of a forward batch waits a pass.

Metal y6h (D log ...dauer10011531, TP0, 15:37:17): weg2-4-14 (resume, its
read short) took the pass with a 3612-token extend; weg2-4-15 (N=41464, P's
END state agreed, parts=3/3) came next, met a non-empty batch and dropped its
END state -- 'adopt=skipped:end_only:batch_not_empty', extend 56 tokens from
the page anchor 41408. skip_first.order put both at the head (joinable by the
page prefix before the match), and only the commit found weg2-4-14's matched
prefix short. The mirror of H24c: behind a forward, a skip waits a pass with
its agreed entry intact; the next pass leads with it.
"""

import inspect

import pytest

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.weg2 import tail_adopt as ta

from test_weg2_tail_skip_batch_h24c_0928 import PREFIX, _agreed, _req, agreed  # noqa: F401


def test_skip_waits_behind_a_forward(agreed):
    r = _req("weg2-4-15", 5)
    agreed[r.rid] = _agreed(r)
    assert ta.skip_waits(r, PREFIX, skip_taken=False, batch_nonempty=True)
    assert r.rid in agreed  # the END state is kept for the next pass


def test_skip_into_an_empty_batch_does_not_wait(agreed):
    r = _req("weg2-4-15", 5)
    agreed[r.rid] = _agreed(r)
    assert not ta.skip_waits(r, PREFIX, skip_taken=False, batch_nonempty=False)


def test_skip_behind_skips_does_not_wait(agreed):
    # H24c: a batch holding only skips counts as empty for the next skip
    r = _req("weg2-4-15", 5)
    agreed[r.rid] = _agreed(r)
    assert not ta.skip_waits(r, PREFIX, skip_taken=True, batch_nonempty=True)


@pytest.mark.parametrize("what", ["e1", "none", "prefix"])
def test_no_wait_for_what_takes_no_end_state(agreed, what):
    r = _req("weg2-4-14", 4)
    if what != "none":
        agreed[r.rid] = _agreed(r, skip=what != "e1")
    at = PREFIX + 64 if what == "prefix" else PREFIX
    assert not ta.skip_waits(r, at, skip_taken=False, batch_nonempty=True)


def test_wired_before_the_plan():
    src = inspect.getsource(PrefillAdder.add_one_req)
    gate = src.index("tail_adopt.skip_waits(")
    assert gate < src.index("_tail = tail_adopt.plan_adopt(")
    assert "skip_taken=self.weg2_skip_extend_taken" in src[gate:gate + 200]
    assert "batch_nonempty=bool(self.can_run_list)" in src[gate:gate + 250]
