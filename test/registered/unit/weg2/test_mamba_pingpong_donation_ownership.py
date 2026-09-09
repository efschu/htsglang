"""Q0: the ping-pong donation orphans one Mamba slot per cached long prefill.

THE SPECIMEN. Boot weg2sn5n @ e873a65644, 2026-09-09T21:19:33Z, all three D
ranks, in ``on_idle``:

    ValueError: pool memory leak detected!
    [full]  total=758528, available=707585, evictable=50943, withheld=0
    [mamba] total=30, available=11, evictable=18, withheld=0,
            leaked_mamba_pages={26}

``[full]`` balances exactly (707585 + 50943 == 758528); the deficit is one
MAMBA slot. ``TREE CENSUS`` was self-consistent on every rank
(``MAMBA: tracked_evictable=18 recomputed_evictable=18``) and
``_all_component_values_flatten`` walks every node from the root regardless of
tier, so this is NOT the #935/#936 census false positive. It is world (b) of
the invariant checker's own docstring: "the tree holds NOTHING and the rows
were orphaned outside it -- a missing free / a lost owner". Same shape as boot
ARM 3 (``leaked_mamba_pages={16}`` after ONE 24k direct prefill).

THE ROOT, and why only long NEW prefixes trip it. This boot runs
``--mamba-radix-cache-strategy extra_buffer`` with overlap, so each request
holds THREE slots: one active + two ping-pong track slots. On the finish path
the unified component donates the tracked state to the radix tree:

  * ``mamba_component.py`` ``prepare_for_caching_req`` (the
    ``elif self.enable_mamba_extra_buffer:`` branch) allocates a FRESH
    ``new_slot`` and calls ``donate_mamba_ping_pong_slot(req, new_slot)``.
  * ``memory_pool.py`` ``donate_mamba_ping_pong_slot`` returns the OLD slot at
    ``donate_idx`` (that is what the tree takes) and REPLACES the buffer entry
    in place: ``buf[donate_idx] = new_slot``.
  * ``mamba_component.py`` ``cleanup_after_caching_req`` then recomputes
    ``keep_idx = get_mamba_ping_pong_keep_idx(req)`` -- the SAME index -- and
    passes it as ``mamba_ping_pong_track_buffer_to_keep``, so
    ``free_mamba_cache`` frees only ``buf[1 - keep_idx]``.

``buf[keep_idx]`` is ``new_slot``, which the REQUEST owns; the tree owns the
OLD slot. Keeping it hands it to nobody: not on the free list, not in a node.
Exactly one orphaned slot per finished request whose donation the tree
accepted.

WHY THE SIBLING IS THE PROOF THIS IS THE DEFECT AND NOT THE DESIGN.
``mamba_radix_cache.py:746-761`` -- the implementation the unified component
replaced -- takes the tree's value from the buffer IN PLACE and allocates
nothing:

    keep_idx  = get_mamba_ping_pong_keep_idx(req)
    src_active = req.mamba_ping_pong_track_buffer[keep_idx].unsqueeze(-1)
    mamba_value = src_active.clone()          # the tree owns buf[keep_idx]

There, "keep ``buf[keep_idx]``" is exactly right, because the tree really does
own that slot. The unified port added a fresh allocation and an in-buffer
replacement while KEEPING the sibling's ``keep_idx`` semantics, which the
replacement had just made false. The term was not lost, as in #1051 -- it was
carried across a change that invalidated it.

THE SECOND LEAK ON THE SAME BRANCH (L2 below): when the tree REFUSES the
donation (``mamba_exist=True``), ``keep_idx`` is None and both buffer entries
go back -- but the OLD donated slot is no longer in the buffer and nothing
frees it either. The int8 branch directly above handles its twin of this case
(``insert_value_unused -> _free_mamba_value``) and the plain branch below it
does too (#929). Only this branch handles neither.

WHAT THESE TESTS EXERCISE. The two REAL methods under test --
``donate_mamba_ping_pong_slot`` and ``free_mamba_cache`` -- are bound to a
minimal harness carrying only the attributes they touch, driving a REAL
``MambaSlotAllocator`` on CPU. That is deliberate: the defect is the ownership
arithmetic between those two methods and the allocator, and a full
``HybridReqToTokenPool`` + tree + GPU would add scaffolding without adding
coverage of the transaction. The partition asserted after every station is the
invariant checker's OWN arithmetic (``expected - free - tree``), so a green
test here is the same statement the on-idle ledger makes on metal.
"""

import types

import pytest
import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
    MambaComponent,
)

POOL_SIZE = 8


class FakeReq:
    def __init__(self, rid="specimen"):
        self.rid = rid
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_pingpong_clear_indices = None
        self.mamba_slot_acquired_this_admission = False
        self.req_pool_idx = 0
        self.last_node = None
        self.session = None


def make_pool(lazy=False, track_size=2):
    """A harness carrying exactly what the two real methods touch."""
    pool = types.SimpleNamespace()
    pool.mamba_allocator = MambaSlotAllocator(POOL_SIZE, device="cpu")
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = lazy
    pool.mamba_ping_pong_track_buffer_size = track_size
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = {}
    # Bind the REAL implementations under test.
    for name in (
        "free_mamba_cache",
        "donate_mamba_ping_pong_slot",
        "set_mamba_ping_pong_slot",
        "get_mamba_ping_pong_keep_idx",
        "get_mamba_ping_pong_other_idx",
    ):
        setattr(pool, name, getattr(HybridReqToTokenPool, name).__get__(pool))
    return pool


def make_component(pool):
    """The REAL ``cleanup_after_caching_req`` under test, minimally bound.

    This is the site the fix changes, so the test must drive IT rather than a
    local restatement of its decision -- a test that recomputed ``keep_idx``
    itself would stay red after the fix and prove nothing about the component.
    """
    comp = types.SimpleNamespace()
    comp.cache = types.SimpleNamespace(req_to_token_pool=pool)
    comp.int8_ckpt_pool = None
    comp.enable_mamba_extra_buffer = True
    for name in ("cleanup_after_caching_req", "_free_mamba_value"):
        setattr(comp, name, getattr(MambaComponent, name).__get__(comp))
    return comp


def finish(comp, req, *, tree_took, donated):
    """The finish station: what the scheduler calls when the request ends."""
    insert_result = types.SimpleNamespace(mamba_exist=not tree_took)
    insert_params = types.SimpleNamespace(mamba_value=donated)
    comp.cleanup_after_caching_req(
        req, is_finished=True, insert_result=insert_result, insert_params=insert_params
    )


def partition(pool, tree_values):
    """The invariant checker's own arithmetic, verbatim.

    ``invariant_checker.py``:
        expected = set(range(1, allocator.size + 1))
        leaked   = expected - set(free_slots) - set(tree values)
    """
    free = set(pool.mamba_allocator.free_slots.tolist())
    tree = {int(v) for v in tree_values}
    expected = set(range(1, pool.mamba_allocator.size + 1))
    return expected - free - tree


def admit(pool, req, n_track=2):
    """Give the request its active slot + ping-pong buffer, as alloc() does."""
    active = pool.mamba_allocator.alloc(1)
    assert active is not None
    req.mamba_pool_idx = active[0]
    slots = pool.mamba_allocator.alloc(n_track)
    assert slots is not None
    buf = torch.full((pool.mamba_ping_pong_track_buffer_size,), -1, dtype=slots.dtype)
    buf[:n_track] = slots
    req.mamba_ping_pong_track_buffer = buf
    req.mamba_next_track_idx = 0
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx] = buf
    return req


# --------------------------------------------------------------------------
# L1 -- the specimen: the tree ACCEPTS the donation
# --------------------------------------------------------------------------


def test_L1_accepted_donation_leaves_no_orphan():
    """One long NEW prefix: prefill -> donate -> tree inserts -> finish.

    RED before the fix with exactly the specimen shape: one request, one slot
    lost, everything else balanced.
    """
    pool = make_pool()
    req = admit(pool, FakeReq())

    # The component's extra_buffer donation: a FRESH slot replaces the entry.
    new_slot = pool.mamba_allocator.alloc(1)
    assert new_slot is not None
    donated = pool.donate_mamba_ping_pong_slot(req, new_slot)

    # The tree took it (mamba_exist == False -> mamba_value_inserted True).
    tree_values = [int(donated.item())]
    finish(make_component(pool), req, tree_took=True, donated=donated)

    orphans = partition(pool, tree_values)
    assert orphans == set(), (
        f"orphaned mamba slots after one cached prefill: {orphans} -- "
        "the tree owns the OLD slot, the request's fresh replacement was kept "
        "and handed to nobody"
    )


def test_L1_orphan_count_is_exactly_one_per_request():
    """The specimen's rate: {16} after one request, {26} after another."""
    pool = make_pool()
    tree_values = []
    for i in range(2):
        req = admit(pool, FakeReq(f"r{i}"))
        new_slot = pool.mamba_allocator.alloc(1)
        donated = pool.donate_mamba_ping_pong_slot(req, new_slot)
        tree_values.append(int(donated.item()))
        finish(make_component(pool), req, tree_took=True, donated=donated)
    assert partition(pool, tree_values) == set()


# --------------------------------------------------------------------------
# L2 -- the same branch, the other direction: the tree REFUSES
# --------------------------------------------------------------------------


def test_L2_refused_donation_leaves_no_orphan():
    """``mamba_exist=True``: keep_idx is None, both buffer entries go back --
    but the OLD donated slot left the buffer and nothing frees it."""
    pool = make_pool()
    req = admit(pool, FakeReq())

    new_slot = pool.mamba_allocator.alloc(1)
    donated = pool.donate_mamba_ping_pong_slot(req, new_slot)

    # The tree already had mamba state for this node: donation unused. The
    # component owes it back, exactly as the int8 and plain branches do.
    finish(make_component(pool), req, tree_took=False, donated=donated)

    assert partition(pool, []) == set(), (
        "the refused donation left the buffer and nothing freed it"
    )


# --------------------------------------------------------------------------
# the transaction's own postconditions
# --------------------------------------------------------------------------


def test_donation_hands_the_tree_the_old_slot_and_the_req_the_new_one():
    """Pins the ownership semantics the fix depends on."""
    pool = make_pool()
    req = admit(pool, FakeReq())
    before = req.mamba_ping_pong_track_buffer.clone()
    donate_idx = pool.get_mamba_ping_pong_keep_idx(req)

    new_slot = pool.mamba_allocator.alloc(1)
    donated = pool.donate_mamba_ping_pong_slot(req, new_slot)

    assert int(donated.item()) == int(before[donate_idx]), "tree gets the OLD slot"
    assert int(req.mamba_ping_pong_track_buffer[donate_idx]) == int(new_slot[0]), (
        "the buffer entry is REPLACED in place -- which is what makes the "
        "sibling's 'keep buf[keep_idx]' semantics false here"
    )


def test_no_double_free_on_the_finish_path():
    """The fix must not swing into the opposite defect (#1051's direction).

    ``MambaSlotAllocator.free`` refuses a double free by name, so a fix that
    frees the tree's slot as well fails here rather than silently corrupting.
    """
    pool = make_pool()
    req = admit(pool, FakeReq())
    new_slot = pool.mamba_allocator.alloc(1)
    donated = pool.donate_mamba_ping_pong_slot(req, new_slot)
    finish(make_component(pool), req, tree_took=True, donated=donated)
    # the tree's slot must still be OUT of the free list
    assert int(donated.item()) not in set(
        pool.mamba_allocator.free_slots.tolist()
    ), "the tree's slot was returned to the pool -- that is the #1051 defect"


def test_request_state_is_cleared_after_finish():
    pool = make_pool()
    req = admit(pool, FakeReq())
    new_slot = pool.mamba_allocator.alloc(1)
    donated = pool.donate_mamba_ping_pong_slot(req, new_slot)
    finish(make_component(pool), req, tree_took=True, donated=donated)
    assert req.mamba_pool_idx is None
    assert req.mamba_ping_pong_track_buffer is None
    assert req.mamba_next_track_idx is None


@pytest.mark.parametrize("lazy", [False, True])
def test_both_pingpong_modes(lazy):
    """``get_mamba_ping_pong_keep_idx`` differs between lazy and normal mode;
    the ownership arithmetic must not."""
    pool = make_pool(lazy=lazy)
    req = admit(pool, FakeReq())
    new_slot = pool.mamba_allocator.alloc(1)
    donated = pool.donate_mamba_ping_pong_slot(req, new_slot)
    finish(make_component(pool), req, tree_took=True, donated=donated)
    assert partition(pool, [int(donated.item())]) == set()
