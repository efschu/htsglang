"""#929: the donated Mamba slot that nobody frees.

MEASURED, window 2g boot 2 (`boot_2f_698cd396ce_0827_0713.log`, pin
`698cd396ce`): one rank of three lost exactly one Mamba slot of twenty and the
on-idle ledger named it --

    [07:22:13 PP0] ... [mamba] total=20, available=19, evictable=0,
      protected=0, session_held=0, uncached=0, withheld=0, double_owned=0,
      double_owned_src=census, ..., leaked_mamba_pages={11}

``leaked_mamba_pages`` is ``expected - free - cached``
(``invariant_checker.py:551-558``), so slot 11 was on NEITHER the allocator's
free list NOR in any tree node. Not double-booked -- unreachable. That
QUIESCENT end state, against peers reading 20/20 at the same instant, is the
evidence.

RETRACTED, and named because it was in the first draft of this analysis: the
per-rung "PP0 +1" reading taken from ``mamba usage`` on prefill lines is NOT
evidence of this leak and must not be cited as such. Those samples are
mid-flight, and PP0 runs a stage ahead of its peers in the replicated schedule
(admits first, releases last), so a phase offset produces exactly that
divergence with nothing leaked. At every quiescent point of the 2f boot the
three ranks' ``#912-OVERLAY`` read-outs agree exactly (8 samples). An indicator
is only a finding once it is shown to measure what it claims; this one was not,
so the "one booking" unification of the two readings is withdrawn. What remains
-- and what these tests pin -- is the end-state leak alone.

THE ASYMMETRY THIS PINS. ``_donate_mamba_value`` allocates a FRESH slot for the
donation (``mamba_component.py:902-903``) and parks it on
``insert_params.mamba_value`` -- NOT on ``req.mamba_pool_idx``. Three sibling
paths then clean up, and only two of them release that donation when the tree
did not take it:

    int8 path          (:936-945)  frees it via ``insert_value_unused``
    not-finished path  (:960-964)  frees it
    plain finished     (:958-959)  does NOT -- it only frees ``req``

``free_mamba_cache(req)`` releases ``req.mamba_pool_idx``. Nothing releases the
donation, so it leaks -- one slot, once, on the rank that took the path.

WHY A RE-ADMITTED REQUEST TAKES IT. ``mamba_value_inserted`` is
``insert_result is not None and not insert_result.mamba_exist`` (:931-933), and
``mamba_exist=True`` is what ``insert`` returns for ``len(key) == 0``
(``unified_radix_cache.py:1569``, ``:1668``) -- the whole key already in the
tree, nothing to add. That is exactly a resident the cutover retracted and
re-admitted: its prompt was just donated, the insert adds nothing, and the slot
it allocated for the donation is dropped on the floor.

THESE TESTS DRIVE THE REAL FUNCTION against a real ``MambaSlotAllocator`` on
CPU and assert on the free list, deliberately NOT on the source text. The
sibling paths are the pin: they are green before and after the fix, so a change
that "fixes" the plain path by breaking them cannot pass.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator


class _Pool:
    """The members ``cleanup_after_caching_req`` reaches through the pool.

    ``mamba_ckpt_pool`` lives here rather than on the component because
    ``MambaComponent.int8_ckpt_pool`` is a PROPERTY reading
    ``self.cache.req_to_token_pool.mamba_ckpt_pool`` (:676-677) -- setting the
    attribute on the component raises, and a test that set it would be
    exercising a fake instead of the real dispatch.
    """

    def __init__(self, allocator, ckpt_pool=None):
        self.mamba_allocator = allocator
        self.mamba_ckpt_pool = ckpt_pool
        self.freed_reqs = []

    def free_mamba_cache(self, req, mamba_ping_pong_track_buffer_to_keep=None):
        # Mirrors production: releases the REQUEST's own slot, nothing else.
        self.freed_reqs.append(req)
        idx = getattr(req, "mamba_pool_idx", None)
        if idx is not None:
            self.mamba_allocator.free(idx)
            req.mamba_pool_idx = None

    def get_mamba_ping_pong_keep_idx(self, req):
        return None


def _component(allocator, *, int8=None, extra_buffer=False):
    """A MambaComponent with only the attributes this method reads.

    Built by ``__new__`` so the test needs no CUDA pool, no radix cache and no
    engine -- the method under test reaches exactly four attributes.
    """
    from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
        MambaComponent,
    )

    comp = MambaComponent.__new__(MambaComponent)
    comp.enable_mamba_extra_buffer = extra_buffer
    comp.cache = SimpleNamespace(req_to_token_pool=_Pool(allocator, int8))
    return comp


def _setup(**kw):
    alloc = MambaSlotAllocator(size=20, device="cpu")
    alloc.clear()
    comp = _component(alloc, **kw)
    req_slot = alloc.alloc(1)          # the request's own slot
    donation = alloc.alloc(1)          # what :902-903 allocates for the donation
    req = SimpleNamespace(mamba_pool_idx=req_slot, mamba_last_track_seqlen=7)
    params = SimpleNamespace(mamba_value=donation)
    return alloc, comp, req, params, req_slot, donation


def _free_ids(alloc):
    return sorted(int(x) for x in alloc.free_slots.tolist())


class TheDonationMustNotLeak(unittest.TestCase):
    """RED before the fix."""

    def test_the_unused_donation_returns_to_the_free_list(self):
        alloc, comp, req, params, req_slot, donation = _setup()
        before = alloc.available_size()

        # mamba_exist=True  ->  the tree already had the whole key, nothing was
        # inserted, so the donation is unused. This is the re-admission case.
        comp.cleanup_after_caching_req(
            req,
            True,
            insert_result=SimpleNamespace(mamba_exist=True),
            insert_params=params,
        )

        self.assertIn(
            int(donation[0]),
            _free_ids(alloc),
            "the donated slot was allocated at mamba_component.py:902-903 and "
            "the tree did not take it, so nothing owns it -- it must be back "
            "on the free list. This is leaked_mamba_pages={11} in miniature.",
        )
        self.assertEqual(
            alloc.available_size(),
            before + 2,
            "both the request's slot AND the unused donation must come back",
        )

    def test_the_leak_is_exactly_one_slot_per_call(self):
        """The measured shape: one slot, not a drift."""
        alloc, comp, req, params, req_slot, donation = _setup()
        comp.cleanup_after_caching_req(
            req, True, SimpleNamespace(mamba_exist=True), params
        )
        self.assertEqual(
            20 - alloc.available_size(),
            0,
            "after cleanup nothing may still be held; a residue of exactly 1 is "
            "the 2g-2 signature (available=19 on a 20-slot pool)",
        )


class TheSiblingPathsMustStayGreen(unittest.TestCase):
    """The pin. These pass before AND after the fix -- the fix must not work by
    changing them, and a regression in them is not a fix."""

    def test_the_int8_path_already_frees_the_unused_donation(self):
        alloc, comp, req, params, req_slot, donation = _setup(int8=object())
        comp._free_mamba_value = lambda v: alloc.free(v)
        comp.cleanup_after_caching_req(
            req, True, SimpleNamespace(mamba_exist=True), params
        )
        self.assertIn(int(donation[0]), _free_ids(alloc))
        self.assertEqual(alloc.available_size(), 20)

    def test_the_not_finished_path_already_frees_the_unused_donation(self):
        alloc, comp, req, params, req_slot, donation = _setup()
        comp._free_mamba_value = lambda v: alloc.free(v)
        comp.cleanup_after_caching_req(
            req, False, SimpleNamespace(mamba_exist=True), params
        )
        self.assertIn(
            int(donation[0]),
            _free_ids(alloc),
            "the not-finished branch (:960-964) frees the donation already",
        )


class TheDangerDirection(unittest.TestCase):
    """The opposite sign, and the worse one.

    Freeing a donation the tree DID take hands a live tree node's state slot
    back to the allocator, which then reissues it -- two requests sharing one
    Mamba state, a wrong answer that never crashes (#924's own reasoning). A
    leak costs a slot; this costs correctness. It must be impossible to pass
    these tests by freeing unconditionally.
    """

    def test_an_inserted_donation_is_never_freed(self):
        alloc, comp, req, params, req_slot, donation = _setup()
        comp._free_mamba_value = lambda v: alloc.free(v)

        # mamba_exist=False -> the tree TOOK the donation; it is live state.
        comp.cleanup_after_caching_req(
            req, True, SimpleNamespace(mamba_exist=False), params
        )

        self.assertNotIn(
            int(donation[0]),
            _free_ids(alloc),
            "the tree owns this slot now; returning it to the allocator would "
            "let alloc() hand one Mamba state to two requests",
        )

    def test_an_inserted_donation_is_never_freed_on_the_int8_path(self):
        alloc, comp, req, params, req_slot, donation = _setup(int8=object())
        comp._free_mamba_value = lambda v: alloc.free(v)
        comp.cleanup_after_caching_req(
            req, True, SimpleNamespace(mamba_exist=False), params
        )
        self.assertNotIn(int(donation[0]), _free_ids(alloc))

    def test_a_missing_insert_params_does_not_crash_the_cleanup(self):
        """``insert_params`` is Optional in the signature; the finished path
        must tolerate None rather than raise inside a teardown."""
        alloc, comp, req, params, req_slot, donation = _setup()
        comp._free_mamba_value = lambda v: alloc.free(v)
        comp.cleanup_after_caching_req(
            req, True, SimpleNamespace(mamba_exist=True), None
        )
        self.assertIsNone(req.mamba_pool_idx)


if __name__ == "__main__":
    unittest.main()
