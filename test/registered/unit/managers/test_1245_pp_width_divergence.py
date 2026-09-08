"""#1245: two PP stages must not build different widths for the same pass.

REPRODUCES BOOT weg2rg3 (tip 5b015ad139, group P, 2026-09-08 04:23:45/46Z),
which died on ``W27 PPWidthDivergenceRefused``: 4096 rows delivered into a
100-token batch, same rid, same slot=1, same ``fwd_ct=37``, same pass ``n=38``.

    [04:23:45 PP0] #969 EXTENT n=38 reqs=[('bc39b188',15400,15401,15400,1),
                                          ('0cf64713', 0, 4095, 0, 4095)]
    [04:23:46 PP1] #969 EXTENT n=38 reqs=[('bc39b188',15400,15401,15400,1),
                                          ('0cf64713', 6009, 6108, 6009, 99)]

The ONE difference between the ranks is an asynchronous event: PP1's HiCache
prefetch for ``0cf64713`` completed at 04:23:45 and PP0's at 04:23:46, one
second after PP0 had already launched that pass at prefix 0. Everything else --
membership, order, slot, forward counter -- agreed.

WHAT THIS TEST IS AND IS NOT. It is a ring DOUBLE of the admission station, not
of the wire: it drives the SHIPPED predicate (``pp_row_carrier_present``), the
SHIPPED extent reader (``_pp_load_back_extent``) and the SHIPPED transition
(``clear_state_aligned_extent_undistributable``) across three stage objects, and
asks the one question the wire later asks -- do the three widths agree. It does
NOT exercise gloo, CUDA, or ``ForwardBatch``; a divergence that arises purely in
the proxy transport is out of its reach and is the width guard's job.

RED/GREEN IS IN ONE FILE ON PURPOSE. ``apply_1245=False`` is not a story about
the old tip, it is the old tip's code path: the same three stages with the drop
bypassed. So the RED arm is a reproduction and the GREEN arm is the fix, with
one set of inputs between them.
"""

import os
import sys

import pytest

_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))),
    "python",
)
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from sglang.srt.managers.pp_admission_congruence import (  # noqa: E402
    LOAD_BACK_EXTENT_ATTR,
    clear_state_aligned_extent_undistributable,
    pp_row_carrier_present,
)

# The measured pass, verbatim from the boot log.
CHUNK = 4096
RID_RESIDENT = "bc39b188"  # 15400 -> 15401, its last token
RID_FRESH = "0cf64713"  # 6108-token prompt, host hit 6009 when prefetched
FRESH_LEN = 6108
FRESH_HOST_HIT = 6009


class _Req:
    """Only what the station touches."""

    def __init__(self, rid, total_len, prefix_len, extent):
        self.rid = rid
        self.total_len = total_len
        self.prefix_len = prefix_len
        setattr(self, LOAD_BACK_EXTENT_ATTR, extent)


class _PS:
    def __init__(self, pp_rank, pp_size):
        self.pp_rank = pp_rank
        self.pp_size = pp_size


class _Stage:
    """One PP stage. ``carrier`` is the ONLY knob: it is what #1233 S0 turned
    off for Weg-2 group P by hardcoding ``pp_flip_counters = None``."""

    def __init__(self, pp_rank, pp_size=3, carrier=False):
        self.ps = _PS(pp_rank, pp_size)
        self.pp_flip_counters = object() if carrier else None


def _load_back_extent(req):
    # The shipped reader, without importing schedule_policy's torch surface:
    # `_pp_load_back_extent` is `getattr(req, LOAD_BACK_EXTENT_ATTR, None)`.
    return getattr(req, LOAD_BACK_EXTENT_ATTR, None)


def _admit_width(stage, reqs, apply_1245: bool) -> int:
    """The station: per request, maybe adopt the async host hit as prefix, then
    build one chunked-prefill batch and return its ROW COUNT -- the number the
    #631/#1233 width guard compares across stages."""
    width = 0
    for req in reqs:
        if stage.ps.pp_size > 1 and apply_1245:
            if not pp_row_carrier_present(stage):
                clear_state_aligned_extent_undistributable(req)
        extent = _load_back_extent(req)
        prefix = req.prefix_len + (extent or 0)
        remaining = req.total_len - prefix
        width += max(0, min(remaining, CHUNK - width))
    return width


def _pass_n38(prefetch_done: bool):
    """The two requests of pass n=38. ``prefetch_done`` is the rank-local,
    wall-clock fact that differed between PP0 and PP1 at 04:23:45."""
    return [
        _Req(RID_RESIDENT, 15401, 15400, None),
        _Req(RID_FRESH, FRESH_LEN, 0, FRESH_HOST_HIT if prefetch_done else None),
    ]


# PP0's prefetch had NOT completed, PP1's and PP2's had. That skew is the input,
# not the defect; the defect is what the station does with it.
_RING = [
    (0, False),
    (1, True),
    (2, True),
]


def test_red_carrierless_ring_reproduces_the_w27_width_divergence():
    """Without #1245 the three stages build 4096 / 100 / 100 -- the boot log's
    numbers -- and the group dies at the width guard."""
    widths = [
        _admit_width(_Stage(r, carrier=False), _pass_n38(done), apply_1245=False)
        for r, done in _RING
    ]
    assert widths == [4096, 100, 100], widths
    assert len(set(widths)) > 1, (
        "the RED arm must actually diverge, or the GREEN arm proves nothing"
    )


def test_green_carrierless_ring_agrees_after_1245():
    """With #1245 no stage adopts a hit it cannot tell its peers, so all three
    build the same width and the width guard has nothing to refuse."""
    widths = [
        _admit_width(_Stage(r, carrier=False), _pass_n38(done), apply_1245=True)
        for r, done in _RING
    ]
    assert len(set(widths)) == 1, widths
    assert widths == [4096, 4096, 4096], widths


def test_pp0_runs_the_same_rule_so_this_is_not_a_follower_compensation():
    """RAENGE-NIE-UNEINS: the drop must fire on PP0 too. A rule only followers
    obey is a compensation, and compensations are what this campaign deletes."""
    pp0 = _Stage(0, carrier=False)
    req = _Req(RID_FRESH, FRESH_LEN, 0, FRESH_HOST_HIT)
    assert not pp_row_carrier_present(pp0)
    assert clear_state_aligned_extent_undistributable(req) == FRESH_HOST_HIT
    assert getattr(req, LOAD_BACK_EXTENT_ATTR) is None


def test_single_stage_is_untouched():
    """pp_size == 1 (Weg-2 group D, and every non-PP boot) never enters the
    block: upstream's path stays byte-for-byte."""
    solo = _Stage(0, pp_size=1, carrier=False)
    reqs = _pass_n38(prefetch_done=True)
    assert _admit_width(solo, reqs, apply_1245=True) == 100
    assert getattr(reqs[1], LOAD_BACK_EXTENT_ATTR) == FRESH_HOST_HIT


def test_gate_lifts_itself_when_a_carrier_exists():
    """The predicate is the whole self-restoring property: with a row carrier
    PP0's published prefix governs and the drop must NOT fire."""
    withc = _Stage(1, carrier=True)
    assert pp_row_carrier_present(withc)
    reqs = _pass_n38(prefetch_done=True)
    assert _admit_width(withc, reqs, apply_1245=True) == 100
    assert getattr(reqs[1], LOAD_BACK_EXTENT_ATTR) == FRESH_HOST_HIT


def test_the_instrument_counts_events_not_visits():
    """DENOMINATOR LAW: a visit that carried no extent is not an event, and
    counting it would make the boot-log occurrence number meaningless."""
    from sglang.srt.managers import pp_admission_congruence as pac

    before = pac._1042_LIFECYCLE["undistributable"]
    empty = _Req(RID_FRESH, FRESH_LEN, 0, None)
    assert clear_state_aligned_extent_undistributable(empty) is None
    assert pac._1042_LIFECYCLE["undistributable"] == before


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
