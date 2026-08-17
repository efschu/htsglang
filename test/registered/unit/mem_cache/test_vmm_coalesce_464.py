"""#464: coalesce the graph-state VMM resume to one handle per contiguous run.

`KvVmmArena.commit_range` splits a gap into `self._chunk`-sized extents (the
#330 dial), so a ~1 GiB resume becomes ~500 x 2 MiB extents and therefore ~500
map + ~500 setAccess driver calls. When the run is CONTIGUOUS those extents
describe one VA region and one handle suffices, taking the resume to ~3 calls
(map, setAccess, memset).

Contiguity is a PRECONDITION, not an assumption: `decommit_span` can leave an
interior hole (the #631 note in `commit_range`), so a non-contiguous plan is a
legitimate state. The coalescer refuses it loudly rather than merging across
the hole or splitting into an unpredictable shape.

DEFAULT OFF: the 40-85 ms band is unmeasured. The flag exists so the
measurement can be taken, not because the win is assumed — that measurement is
a window item, not part of this build.

Hermetic: pure arithmetic over a plan plus a mocked driver, no CUDA.
"""

import pytest
from sglang.srt.mem_cache.kv_vmm_backing import (
    VmmCoalesceRefused,
    coalesce_commit_plan,
)

CHUNK = 2 * 1024 * 1024  # the #330 dial as deployed


def _contiguous(n, chunk=CHUNK):
    return [(i * chunk, chunk) for i in range(n)]


class _MockDriver:
    """Counts the driver calls a plan would issue."""

    def __init__(self):
        self.maps = 0
        self.set_access = 0
        self.memsets = 0

    def apply(self, plan):
        for _pos, _step in plan:
            self.maps += 1
            self.set_access += 1
        self.memsets += 1  # one per resume, regardless of extent count

    @property
    def total(self):
        return self.maps + self.set_access + self.memsets


def test_default_off_reproduces_the_per_chunk_plan_exactly():
    """Off must be byte-for-byte the pre-#464 behaviour."""
    plan = _contiguous(500)
    out, rep = coalesce_commit_plan(plan, enabled=False)
    assert out == plan
    assert rep.coalesced is False
    assert rep.driver_calls_saved == 0
    assert "disabled" in rep.reason


def test_a_1_gib_contiguous_resume_drops_to_three_driver_calls():
    """THE claim: ~500 x 2 MiB becomes map + setAccess + memset."""
    plan = _contiguous(500)  # ~1 GiB
    before = _MockDriver()
    before.apply(plan)
    assert before.total == 1001  # 500 + 500 + 1

    out, rep = coalesce_commit_plan(plan, enabled=True)
    after = _MockDriver()
    after.apply(out)

    assert len(out) == 1
    assert after.total == 3, "map + setAccess + memset"
    assert rep.coalesced is True
    assert rep.driver_calls_saved == 998


def test_the_coalesced_extent_covers_exactly_the_same_bytes():
    """A resume that maps a different range is a bug, not an optimisation."""
    plan = _contiguous(8)
    out, rep = coalesce_commit_plan(plan, enabled=True)
    assert out[0][0] == plan[0][0], "same start"
    assert sum(s for _, s in out) == sum(s for _, s in plan), "same total bytes"
    assert rep.bytes_total == 8 * CHUNK


def test_a_hole_REFUSES_to_coalesce_rather_than_merging_across_it():
    """The precondition. decommit_span can leave an interior hole, so this is
    a legitimate state — and merging across it would map pages nobody asked
    for."""
    plan = [(0, CHUNK), (CHUNK, CHUNK), (3 * CHUNK, CHUNK)]  # gap at 2*CHUNK
    out, rep = coalesce_commit_plan(plan, enabled=True)
    assert out == plan, "falls back to the per-chunk plan, unchanged"
    assert rep.coalesced is False
    assert "not contiguous" in rep.reason
    assert "hole" in rep.reason
    # And it names WHERE, so the caller can act on it.
    assert str(2 * CHUNK) in rep.reason


def test_require_contiguous_raises_for_a_caller_that_asserted_it():
    plan = [(0, CHUNK), (3 * CHUNK, CHUNK)]
    with pytest.raises(VmmCoalesceRefused, match="not contiguous"):
        coalesce_commit_plan(plan, enabled=True, require_contiguous=True)


def test_multiple_holes_are_all_counted_and_the_first_is_named():
    plan = [(0, CHUNK), (2 * CHUNK, CHUNK), (5 * CHUNK, CHUNK)]
    _out, rep = coalesce_commit_plan(plan, enabled=True)
    assert "2 hole(s)" in rep.reason
    assert str(CHUNK) in rep.reason  # first hole starts at CHUNK


def test_a_single_extent_is_left_alone():
    plan = [(0, CHUNK)]
    out, rep = coalesce_commit_plan(plan, enabled=True)
    assert out == plan
    assert rep.coalesced is False
    assert "nothing to coalesce" in rep.reason


def test_an_empty_plan_is_not_an_error():
    out, rep = coalesce_commit_plan([], enabled=True)
    assert out == []
    assert rep.bytes_total == 0
    assert rep.coalesced is False


def test_uneven_extents_still_coalesce_when_contiguous():
    """The last chunk of a run is short; that must not block coalescing."""
    plan = [(0, CHUNK), (CHUNK, CHUNK), (2 * CHUNK, CHUNK // 4)]
    out, rep = coalesce_commit_plan(plan, enabled=True)
    assert out == [(0, 2 * CHUNK + CHUNK // 4)]
    assert rep.coalesced is True


def test_driver_calls_saved_is_two_per_removed_extent():
    """One map and one setAccess per extent that no longer exists."""
    _out, rep = coalesce_commit_plan(_contiguous(10), enabled=True)
    assert rep.extents_before == 10 and rep.extents_after == 1
    assert rep.driver_calls_saved == 18
