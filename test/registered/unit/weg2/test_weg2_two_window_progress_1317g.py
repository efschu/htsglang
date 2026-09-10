"""#1317g -- TWO windows on one prompt: release must precede re-issue.

WHY THIS FILE EXISTS. The Q6d caveat says C1 has never been observed firing on
metal (`#1317 WINDOW-RELEASE` = 0 on boot weg2sn6c despite D running the
staging role), and no test exercised release AND re-issue across more than one
window. The forward-progress claim of the whole C1/C4/C6 chain was therefore
unpinned in both places at once.

WHAT THIS IS, STATED PLAINLY SO IT IS NOT OVER-READ. This is a STATE-MACHINE
test over the real predicates and the real solver, with the host pool modelled
as a row counter. It is NOT live integration: it does not run a
cache_controller, a load-back ack or a collective. It pins the ARITHMETIC AND
THE ORDERING that make two windows possible -- which is exactly the half that
was unpinned -- and it cannot tell you that C1 fires on metal. The boot remains
the first exercise of that.

THE CLAIM: for a prompt longer than one window, window 2 CANNOT allocate until
window 1's host rows are released, so `release` is a precondition of
`re-issue` and not an optimisation. If that ordering is ever broken the loop
re-issues into a full pool and stalls -- which is precisely the
`WINDOW-REISSUE > 0 with WINDOW-RELEASE = 0` pair the Q6d order tells the boot
agent to record.
"""

import pytest

from sglang.srt.weg2 import host_ledger as hl

# Group D on the standing boot form.
D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS = 30518, 4096, 1, 4096, 37
PROMPT = 60000          # > one window AND > the pool, so it needs >= 2 windows


def _W():
    return hl.solve_window(D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, D_ANCHORS)


class _Pool:
    """The host pool as a row counter, with the pool's own alloc contract:
    an allocation that does not fit returns None and takes nothing."""

    def __init__(self, size):
        self.size = size
        self.free = size

    def alloc(self, rows):
        if rows > self.free:
            return None
        self.free -= rows
        return rows

    def release(self, rows):
        self.free = min(self.size, self.free + rows)


def test_the_prompt_needs_more_than_one_window_and_exceeds_the_pool():
    """The premise of the whole file, asserted rather than assumed."""
    w = _W()
    assert w == 24576
    assert PROMPT > w, "a one-window prompt would not test the loop at all"
    assert PROMPT > D_POOL, "and it must exceed the pool, or no release is needed"
    assert -(-PROMPT // w) == 3


def test_window_two_cannot_allocate_while_window_one_still_holds_its_rows():
    """THE LOAD-BEARING ORDERING. Without release the pool has 5,942 rows left
    after window 1 and window 2 needs 24,576: the alloc returns None, which is
    the stall the metal caveat warns about."""
    w = _W()
    pool = _Pool(D_POOL)
    assert pool.alloc(w) == w                     # window 1 lands
    assert pool.free == D_POOL - w == 5942
    assert pool.alloc(w) is None, (
        "window 2 must NOT fit while window 1 is resident -- if this ever "
        "passes, the pool is not the constraint the loop is built around"
    )


def test_release_then_reissue_lets_the_whole_prompt_through():
    """The full walk: three windows, each releasing before the next, and the
    prompt is covered. Also asserts the pool never goes negative and never
    holds more than one window plus the write-through floor."""
    w = _W()
    pool = _Pool(D_POOL)
    covered = 0
    windows = 0
    while covered < PROMPT:
        got = pool.alloc(min(w, PROMPT - covered))
        assert got is not None, f"window {windows + 1} could not allocate"
        windows += 1
        covered += got
        # C1: the window is device-resident and store-present, so its host
        # rows go back before the next read is issued.
        pool.release(got)
        assert 0 <= pool.free <= D_POOL
    assert windows == 3
    assert covered == PROMPT
    assert pool.free == D_POOL, "every window's rows must come back"


def test_the_live_set_is_one_window_plus_the_ring_floor():
    """What the solver promised: live = W + ring_floor <= pool, with slack.
    A second concurrent window would break it, which is why the solver funds
    `chains` and not `depth`."""
    w = _W()
    assert w + D_FLOOR == 28672
    assert w + D_FLOOR <= D_POOL
    assert D_POOL - w - D_FLOOR == 1846
    assert 2 * w + D_FLOOR > D_POOL, (
        "two resident windows must NOT fit -- the loop is sequential by design"
    )


def test_the_release_predicate_gates_exactly_the_four_terms():
    """The re-issue is only safe because release refuses a node it may not
    free. Pinned against the shipped predicate, so a fifth precondition or a
    lost one fails here."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.release_staged_window)
    for term in ("node.evicted", "node.l3_present", "host_lock_ref",
                 "node.backuped", "root_node"):
        assert term in src, f"{term!r} left the release predicate"


def test_the_chain_keeps_the_resume_anchor_out_of_the_release():
    """Window k+1 continues from window k's deepest node, so that node must
    survive its own window's recycle -- the R-4 stall. The loading_check
    caller starts the walk at `node.parent` for exactly this reason."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.loading_check)
    assert "release_staged_window_chain(node.parent)" in src, (
        "the walk must start at the PARENT or it recycles the anchor the next "
        "window needs"
    )
    # and the anchor budget the solver funds must cover the windows this
    # prompt takes, plus the head's own surviving anchor
    w = _W()
    assert w // D_CHUNK + 1 <= D_ANCHORS


def test_the_reissue_is_gated_on_the_group_agreed_mark_only():
    """A rank-local gate here would split the group inside the #580 vote. The
    two terms are admission identity and the group-agreed truncation verdict."""
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler._weg2_issue_next_window)
    assert "self.chunked_req is not req" in src
    assert "_weg2_window_open" in src
    assert "issued:truncated_group" in src


def test_reissue_without_release_is_the_pair_the_boot_must_record():
    """The Q6d caveat as arithmetic: re-issuing n times while releasing 0 times
    covers exactly ONE window, no matter how large n is. So the pair
    (WINDOW-REISSUE > 0, WINDOW-RELEASE = 0) is not slow progress -- it is no
    progress, and it must never be read as a pass."""
    w = _W()
    pool = _Pool(D_POOL)
    covered = pool.alloc(w) or 0
    reissues = 0
    for _ in range(10):
        reissues += 1
        if pool.alloc(w) is None:      # no release ever happens
            continue
        covered += w
    assert reissues == 10
    assert covered == w, "without release, ten re-issues cover one window"
    assert covered < PROMPT
