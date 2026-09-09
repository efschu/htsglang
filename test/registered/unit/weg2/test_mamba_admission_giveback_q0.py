"""Q0: the X-gate (W50) refusal exit must give back the slot the match acquired.

SPECIMEN, boot weg2sn5s @ fd9244056d, all three D ranks, fleet-shape load
(shared ~16k anchored prefix + growing tails):

    22:27:51 #924D station=alloc_cow rid=664901103e55 mamba_slot=[7] site=finalize_match_result
    22:27:51 WEG2 X-GATE  rid=664901103e55 uncached=17236 X=8742 verdict=W31
    22:27:51 W50 Weg2TpPrefillExceeded rid=664901103e55 uncached=17236 X=8742
    on_idle  [mamba] total=30, available=4, evictable=25, leaked_mamba_pages={7}
             mamba_leak_owners=[slot=7 slot_used=True last_event=ALLOC@seq6158
                                releaser=none-recorded]

ONE station line for the entire request, and ZERO `#991` give-back lines in the
whole log. A prefix match drew a COW resume slot speculatively; the X gate then
refused the request; the refusal exit returned it to the front WITHOUT the slot.
The request never reaches `alloc` or `cache_finished_req`, so no later station
can release it -- the refusal exit is the only owner of that give-back.

WHY IT LOOKED INTERMITTENT. Both conditions must hold: `uncached > X` (so W50
fires -- long prompts only) AND a prefix match carrying mamba state (so the COW
acquire happens at all). A synthetic load with a unique leading nonce satisfies
the first and never the second, which is why 4/4 sequential 41k prompts ran
clean and the fleet's shared-prefix shape leaked within two passes.

THE CLASS: three admission-refusal exits, the give-back open-coded in two of
them and absent from the third. Fixed by lifting it into ONE function
(`release_admission_acquired_mamba_slot`) called from all three, rather than
adding a third copy.
"""

import types

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.common import release_admission_acquired_mamba_slot

POOL = 8


class FakeReq:
    def __init__(self, rid="specimen", session=None):
        self.rid = rid
        self.session = session
        self.mamba_pool_idx = None
        self.mamba_slot_acquired_this_admission = False
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.mamba_loadback_anchor_adopted = False


def make_tree():
    pool = types.SimpleNamespace(mamba_allocator=MambaSlotAllocator(POOL, device="cpu"))
    return types.SimpleNamespace(req_to_token_pool=pool)


def partition(tree, tree_owned, live_reqs):
    """The FULL partition, both directions -- the assertion the first suite lacked.

    expected == free U tree U live, and the three are pairwise disjoint. The
    orphan direction is `expected - free - tree - live`; the aliasing direction
    is any non-empty pairwise intersection (a slot both free and owned is what
    makes alloc() hand a live anchor's state to the next request).
    """
    alloc = tree.req_to_token_pool.mamba_allocator
    free = set(alloc.free_slots.tolist())
    owned = {int(v) for v in tree_owned}
    live = {int(r.mamba_pool_idx) for r in live_reqs if r.mamba_pool_idx is not None}
    expected = set(range(1, alloc.size + 1))
    return {
        "orphans": expected - free - owned - live,
        "free_and_tree": free & owned,
        "free_and_live": free & live,
        "tree_and_live": owned & live,
    }


def assert_balanced(tree, tree_owned=(), live_reqs=()):
    p = partition(tree, tree_owned, live_reqs)
    assert p["orphans"] == set(), f"orphaned slots: {p['orphans']}"
    assert p["free_and_tree"] == set(), f"free AND tree-held (aliasing): {p['free_and_tree']}"
    assert p["free_and_live"] == set(), f"free AND live-req (aliasing): {p['free_and_live']}"
    assert p["tree_and_live"] == set(), f"tree AND live-req (aliasing): {p['tree_and_live']}"


def cow_acquire(tree, req):
    """What `finalize_match_result` does on a prefix match with mamba state."""
    slot = tree.req_to_token_pool.mamba_allocator.alloc(1)
    assert slot is not None
    req.mamba_pool_idx = slot[0]
    req.mamba_slot_acquired_this_admission = True
    req.mamba_cow_src_index = torch.tensor([1])
    return slot


# --------------------------------------------------------------------------
# the specimen: match -> COW acquire -> X-gate refusal -> on_idle
# --------------------------------------------------------------------------


def test_x_gate_refusal_gives_the_cow_slot_back():
    """RED on fd9244056d: the refusal exit had no give-back at all."""
    tree = make_tree()
    req = FakeReq("664901103e55")
    cow_acquire(tree, req)
    # the X gate refuses (uncached 17236 > X 8742) and the exit runs
    released = release_admission_acquired_mamba_slot(req, tree, site="weg2_x_refusal")
    assert released is True
    assert req.mamba_pool_idx is None
    assert_balanced(tree)


def test_refusal_clears_the_admission_carry_overs():
    """A refused request must not carry a resume anchor into its next try."""
    tree = make_tree()
    req = FakeReq()
    cow_acquire(tree, req)
    req.mamba_loadback_anchor_adopted = True
    req.mamba_needs_clear = True
    release_admission_acquired_mamba_slot(req, tree, site="weg2_x_refusal")
    assert req.mamba_slot_acquired_this_admission is False
    assert req.mamba_cow_src_index is None
    assert req.mamba_needs_clear is False
    assert req.mamba_loadback_anchor_adopted is False


def test_repeated_refusals_do_not_drain_the_pool():
    """The measured rate was one slot per refused long request."""
    tree = make_tree()
    for i in range(POOL):
        req = FakeReq(f"r{i}")
        cow_acquire(tree, req)
        release_admission_acquired_mamba_slot(req, tree, site="weg2_x_refusal")
        assert_balanced(tree)


# --------------------------------------------------------------------------
# the OTHER direction: the give-back must not free what it does not own
# --------------------------------------------------------------------------


def test_batch_owned_slot_is_NOT_given_back():
    """Without the stamp the slot has a different releaser; freeing it here is
    the #1051 double-owned defect (boot weg2sn5o: free_and_cached=1)."""
    tree = make_tree()
    req = FakeReq()
    slot = tree.req_to_token_pool.mamba_allocator.alloc(1)
    req.mamba_pool_idx = slot[0]
    req.mamba_slot_acquired_this_admission = False   # batch-owned
    assert release_admission_acquired_mamba_slot(req, tree, site="x") is False
    assert req.mamba_pool_idx is not None
    assert_balanced(tree, live_reqs=[req])


def test_session_held_slot_is_NOT_given_back():
    tree = make_tree()
    req = FakeReq(session=object())
    cow_acquire(tree, req)
    assert release_admission_acquired_mamba_slot(req, tree, site="x") is False
    assert_balanced(tree, live_reqs=[req])


def test_no_slot_is_a_noop():
    tree = make_tree()
    assert release_admission_acquired_mamba_slot(FakeReq(), tree, site="x") is False
    assert_balanced(tree)


def test_give_back_is_idempotent():
    """A second exit on the same request must not double-free."""
    tree = make_tree()
    req = FakeReq()
    cow_acquire(tree, req)
    assert release_admission_acquired_mamba_slot(req, tree, site="x") is True
    assert release_admission_acquired_mamba_slot(req, tree, site="x") is False
    assert_balanced(tree)


# --------------------------------------------------------------------------
# the SERVED direction stays balanced (the fix must not break the happy path)
# --------------------------------------------------------------------------


def test_served_request_keeps_its_slot_through_admission():
    """Admitted, not refused: the slot stays with the live request."""
    tree = make_tree()
    req = FakeReq()
    cow_acquire(tree, req)
    # admitted -> the batch owns it now, the stamp dies (HybridReqToTokenPool.alloc)
    req.mamba_slot_acquired_this_admission = False
    assert release_admission_acquired_mamba_slot(req, tree, site="x") is False
    assert_balanced(tree, live_reqs=[req])


def test_all_three_exits_call_the_one_helper():
    """#Q0: ONE mechanism, not a third copy."""
    import inspect

    from sglang.srt.managers import schedule_policy, scheduler

    sched = inspect.getsource(scheduler)
    pol = inspect.getsource(schedule_policy)
    assert sched.count("release_admission_acquired_mamba_slot(") >= 2, "both scheduler exits"
    assert 'site="weg2_x_refusal"' in sched, "the X-gate exit is wired"
    assert 'site="admission_revert"' in sched, "the admission revert exit is wired"
    assert 'site="pp_schedule_refused"' in pol, "the PP-refusal exit is wired"
    # and the open-coded twins are gone
    assert "mamba_allocator.free(\n                                req.mamba_pool_idx.unsqueeze(-1)\n                            )" not in pol
