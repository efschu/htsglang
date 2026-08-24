# SPDX-License-Identifier: Apache-2.0
"""#657 item 16: allocation steering, and the limits of what it can steer.

Every test here is hermetic (CPU tensors, no server) and several are
CAN-FAIL PROOFS rather than assertions of intent: a gate nobody has watched
refuse is indistinguishable from a gate that is never reached.
"""

import unittest

import torch

from sglang.srt.managers.corridor_steering import (
    _NO_PROPOSAL,
    AllocationSteering,
    absorbing_card,
    owner_class_of,
)

_MIB = 1024 * 1024


class _FakeAllocator:
    """The head-of-free-list contract, without a KV cache behind it."""

    def __init__(self, size: int):
        self.page_size = 1
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self.is_not_in_free_group = True
        self._owner_bias = None

    # The two methods the steer uses, copied in behaviour from
    # PagedTokenToKVPoolAllocator so the policy can be tested without CUDA.
    def set_owner_bias(self, bias):
        if bias is not None:
            mod, lo, hi = (int(v) for v in bias)
            if mod <= 0 or not (0 <= lo < hi <= mod):
                raise ValueError(f"bad class {bias!r}")
        self._owner_bias = bias
        return self._apply_owner_bias()

    def _apply_owner_bias(self):
        if self._owner_bias is None:
            return 0
        mod, lo, hi = self._owner_bias
        pages = self.free_pages
        res = pages % mod
        m = (res >= lo) & (res < hi)
        n = int(m.sum())
        if n and n != pages.numel():
            self.free_pages = torch.cat((pages[m], pages[~m]))
        return n

    def alloc(self, n: int):
        out = self.free_pages[:n]
        self.free_pages = self.free_pages[n:]
        return out


class _FakeScheduler:
    def __init__(self, alloc):
        self.token_to_kv_pool_allocator = alloc


def _min_reduce(payloads):
    """The group's MIN channel, given every rank's payload."""
    return [min(vals) for vals in zip(*payloads)]


class TestOwnerClass(unittest.TestCase):
    def test_classes_tile_the_block_exactly(self):
        ratios = [14, 10, 8]
        got = [owner_class_of(ratios, r) for r in range(3)]
        self.assertEqual(got, [(32, 0, 14), (32, 14, 24), (32, 24, 32)])
        # Every residue belongs to exactly one rank: the owner rule is a
        # partition, so a steer can never make a slot ambiguous.
        seen = set()
        for _, lo, hi in got:
            seen |= set(range(lo, hi))
        self.assertEqual(seen, set(range(32)))

    def test_a_rank_outside_the_vector_is_refused(self):
        with self.assertRaises(ValueError):
            owner_class_of([14, 10, 8], 3)


class TestAbsorbingCard(unittest.TestCase):
    def test_picks_the_most_free_card_in_absolute_bytes(self):
        col = [2369 * _MIB, 5030 * _MIB, 2715 * _MIB]
        self.assertEqual(absorbing_card(col, 256 * _MIB), 1)

    def test_stands_down_when_the_column_is_level(self):
        col = [2000 * _MIB, 2100 * _MIB, 2050 * _MIB]
        self.assertIsNone(absorbing_card(col, 256 * _MIB))

    def test_level_is_a_band_the_caller_sets(self):
        col = [2000 * _MIB, 2300 * _MIB, 2050 * _MIB]
        self.assertIsNone(absorbing_card(col, 512 * _MIB))
        self.assertEqual(absorbing_card(col, 128 * _MIB), 1)


class TestFreeListPartition(unittest.TestCase):
    def test_the_bias_promotes_exactly_the_class_and_keeps_order(self):
        a = _FakeAllocator(64)
        promoted = a.set_owner_bias((32, 14, 24))  # rank 1's class
        head = a.free_pages[:promoted]
        self.assertTrue(bool(((head % 32 >= 14) & (head % 32 < 24)).all()))
        # Stable: within the promoted half and within the remainder, the
        # original ascending order survives. Two ranks partitioning the same
        # list must produce the SAME list, not merely an equivalent one.
        self.assertEqual(head.tolist(), sorted(head.tolist()))
        tail = a.free_pages[promoted:]
        self.assertEqual(tail.tolist(), sorted(tail.tolist()))

    def test_the_partition_is_a_permutation_and_loses_nothing(self):
        a = _FakeAllocator(200)
        before = set(a.free_pages.tolist())
        a.set_owner_bias((32, 0, 14))
        self.assertEqual(set(a.free_pages.tolist()), before)
        self.assertEqual(a.free_pages.numel(), 200)

    def test_allocations_then_land_on_the_steered_rank(self):
        a = _FakeAllocator(320)
        a.set_owner_bias((32, 24, 32))  # rank 2
        got = a.alloc(40)
        res = got % 32
        self.assertTrue(bool(((res >= 24) & (res < 32)).all()))

    def test_the_steer_cannot_starve_an_allocation(self):
        # Only 80 slots belong to the biased class; asking for 200 still
        # succeeds, because the tail is the rest of the free list.
        a = _FakeAllocator(320)
        a.set_owner_bias((32, 24, 32))
        got = a.alloc(200)
        self.assertEqual(got.numel(), 200)


class TestSteeringDecision(unittest.TestCase):
    def _steer(self, rank, nvml_index, size=320):
        alloc = _FakeAllocator(size)
        return AllocationSteering(
            _FakeScheduler(alloc),
            ratios=[14, 10, 8],
            rank=rank,
            nvml_index=nvml_index,
        )

    def _round(self, steers, column):
        """One seam: every rank builds a payload, the group MINs them."""
        payloads = []
        for s in steers:
            n = len(s.ratios)
            payload = [_NO_PROPOSAL] * (n + 4)
            if s.nvml_index is not None:
                payload[s.rank] = s.nvml_index
            proposal = s._propose(column)
            payload[n] = proposal
            payload[n + 1] = -proposal
            c = s._free_list_checksum()
            payload[n + 2] = c
            payload[n + 3] = -c
            payloads.append(payload)
        reduced = _min_reduce(payloads)
        return [s.decide(lambda _p, r=reduced: r, column) for s in steers]

    def test_the_permutation_is_learned_not_assumed(self):
        # THE RIG'S ACTUAL PERMUTATION: rank 0 is the 5090 at nvidia-smi 1.
        # A steer that assumed rank == card would push bytes onto the
        # binding card, so the mapping is resolved per rank and exchanged.
        steers = [self._steer(0, 1), self._steer(1, 0), self._steer(2, 2)]
        column = [2369 * _MIB, 5030 * _MIB, 2715 * _MIB]
        self._round(steers, column)  # first seam only learns the permutation
        for s in steers:
            self.assertEqual(s.state.rank_to_nvml, (1, 0, 2))
        # Second seam: the fullest card is nvidia-smi 1, which is RANK 0.
        got = self._round(steers, column)
        self.assertEqual(got, [0, 0, 0])

    def test_every_rank_reaches_the_same_verdict_when_readings_disagree(self):
        steers = [self._steer(0, 1), self._steer(1, 0), self._steer(2, 2)]
        base = [2369 * _MIB, 5030 * _MIB, 2715 * _MIB]
        self._round(steers, base)
        # Each rank now reads its own slightly different column, and two of
        # them name different cards. The verdict must still be identical.
        cols = [
            [2369 * _MIB, 5030 * _MIB, 2715 * _MIB],  # -> rank 0
            [2369 * _MIB, 2500 * _MIB, 5030 * _MIB],  # -> rank 2
            [2369 * _MIB, 5030 * _MIB, 2715 * _MIB],  # -> rank 0
        ]
        payloads = []
        for s, col in zip(steers, cols):
            n = len(s.ratios)
            p = [_NO_PROPOSAL] * (n + 4)
            p[s.rank] = s.nvml_index
            prop = s._propose(col)
            p[n], p[n + 1] = prop, -prop
            c = s._free_list_checksum()
            p[n + 2], p[n + 3] = c, -c
            payloads.append(p)
        reduced = _min_reduce(payloads)
        verdicts = [s.decide(lambda _p, r=reduced: r, c) for s, c in zip(steers, cols)]
        self.assertEqual(len(set(verdicts)), 1, "the ranks disagreed on the steer")
        self.assertTrue(all(s.state.disagreements == 1 for s in steers))

    def test_a_divergent_free_list_DISARMS_the_steer(self):
        """CAN-FAIL PROOF for the premise this mechanism rests on."""
        steers = [self._steer(0, 1), self._steer(1, 0), self._steer(2, 2)]
        column = [2369 * _MIB, 5030 * _MIB, 2715 * _MIB]
        self._round(steers, column)
        # Rank 1's free list is no longer the others': the replication
        # assumption has failed on metal. Steering must stop, not continue.
        bad = steers[1].scheduler.token_to_kv_pool_allocator
        bad.free_pages = torch.flip(bad.free_pages, dims=(0,))
        self._round(steers, column)
        for s in steers:
            self.assertFalse(s.state.armed)
            self.assertIn("NOT replicated", s.state.disarmed_reason)
            self.assertIsNone(s.scheduler.token_to_kv_pool_allocator._owner_bias)

    def test_an_unresolved_permutation_DISARMS_rather_than_guesses(self):
        """The FIRST BOOT's actual failure, pinned as a regression.

        Every rank reported ``rank 0`` because the scheduler's topology
        snapshot describes the current phase and this instance boots in PP3,
        where ``tp_size == 1`` and every rank's ``tp_rank`` is 0. All three
        wrote their NVML column into slot 0, the permutation came back
        ``(0, 1048576, 1048576)``, and the steer refused to run. Fixed by
        indexing on the world rank; kept as a test because a steer that
        guessed a column would push bytes ONTO the binding card.
        """
        steers = [self._steer(0, 1), self._steer(0, 0), self._steer(0, 2)]
        column = [2369 * _MIB, 5030 * _MIB, 2715 * _MIB]
        self._round(steers, column)
        for s in steers:
            self.assertFalse(s.state.armed)
            self.assertIn("permutation did not resolve", s.state.disarmed_reason)

    def test_a_level_column_steers_nothing(self):
        steers = [self._steer(0, 1), self._steer(1, 0), self._steer(2, 2)]
        level = [2400 * _MIB, 2450 * _MIB, 2500 * _MIB]
        self._round(steers, level)
        got = self._round(steers, level)
        self.assertEqual(got, [None, None, None])
        for s in steers:
            self.assertIsNone(s.state.bias_rank)
            self.assertTrue(s.state.armed)

    def test_the_checksum_is_order_sensitive(self):
        # A set-based fingerprint would be blind to exactly the failure the
        # check exists for, so this pins the property rather than the value.
        s = self._steer(0, 1)
        alloc = s.scheduler.token_to_kv_pool_allocator
        first = s._free_list_checksum()
        alloc.free_pages = torch.cat(
            (alloc.free_pages[5:6], alloc.free_pages[:5], alloc.free_pages[6:])
        )
        self.assertNotEqual(first, s._free_list_checksum())


class TestTheRealAllocatorMethods(unittest.TestCase):
    """The same properties against the SHIPPED class, not the fake.

    The fake exists so the policy can be tested without a KV cache; it would
    be worthless if the real methods behaved differently, and a desk-written
    method that no test ever executed is exactly what this chain has shipped
    before. Built with ``__new__`` and the four attributes the two methods
    touch -- they are pure tensor code and need nothing else.
    """

    def test_the_allocator_this_rig_actually_uses_carries_the_bias(self):
        """THE TRAP THAT COST A BOOT, pinned so it cannot come back.

        The scheduler keeps ONE allocator for process life -- the PP stack's
        -- and the PP layout has ``dcp_size == 1``, so at ``page_size == 1``
        the chooser builds a plain ``TokenToKVPoolAllocator``, NOT the paged
        one (``model_runner_kv_cache_mixin.py:4083``). The first boot of this
        mechanism armed and steered nothing, reporting "the active allocator
        has no owner bias" nine times, because the methods lived on the paged
        subclass. They belong on the shared base, and this test fails if they
        ever move back down.
        """
        from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        for cls in (
            BaseTokenToKVPoolAllocator,
            TokenToKVPoolAllocator,
            PagedTokenToKVPoolAllocator,
        ):
            self.assertTrue(hasattr(cls, "set_owner_bias"), cls.__name__)
            self.assertTrue(hasattr(cls, "_apply_owner_bias"), cls.__name__)

    def test_the_unpaged_allocator_partitions_identically(self):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        a = TokenToKVPoolAllocator.__new__(TokenToKVPoolAllocator)
        a.page_size = 1
        a.free_pages = torch.arange(1, 321, dtype=torch.int64)
        a.release_pages = torch.empty((0,), dtype=torch.int64)
        a.is_not_in_free_group = True
        a._owner_bias = None
        a.device = "cpu"
        a.need_sort = False
        fake = _FakeAllocator(320)
        self.assertEqual(
            a.set_owner_bias((32, 14, 24)), fake.set_owner_bias((32, 14, 24))
        )
        self.assertEqual(a.free_pages.tolist(), fake.free_pages.tolist())
        # And its own alloc path takes the head, which is the whole premise.
        got = a.alloc(40)
        res = got % 32
        self.assertTrue(bool(((res >= 14) & (res < 24)).all()))

    def _real(self, size=320):
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        a = PagedTokenToKVPoolAllocator.__new__(PagedTokenToKVPoolAllocator)
        a.page_size = 1
        a.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        a.release_pages = torch.empty((0,), dtype=torch.int64)
        a.is_not_in_free_group = True
        a._owner_bias = None
        a.device = "cpu"  # merge_and_sort_free rebuilds release_pages on it
        return a

    def test_real_partition_matches_the_fake(self):
        real, fake = self._real(), _FakeAllocator(320)
        self.assertEqual(
            real.set_owner_bias((32, 14, 24)), fake.set_owner_bias((32, 14, 24))
        )
        self.assertEqual(real.free_pages.tolist(), fake.free_pages.tolist())

    def test_real_bias_is_cleared_by_none(self):
        real = self._real()
        real.set_owner_bias((32, 14, 24))
        real.set_owner_bias(None)
        self.assertIsNone(real._owner_bias)
        self.assertEqual(real._apply_owner_bias(), 0)

    def test_real_refuses_a_paged_layout(self):
        # CAN-FAIL PROOF: with page_size > 1 a page spans several residues,
        # so a page id does not name an owner and the steer must refuse
        # rather than reorder something it has mis-read.
        real = self._real()
        real.page_size = 64
        with self.assertRaises(ValueError):
            real.set_owner_bias((32, 0, 14))

    def test_real_refuses_a_malformed_class(self):
        real = self._real()
        for bad in [(0, 0, 1), (32, 14, 14), (32, -1, 4), (32, 20, 40)]:
            with self.assertRaises(ValueError):
                real.set_owner_bias(bad)

    def test_the_default_path_is_unchanged_when_nothing_is_steered(self):
        """PROTECT THE SHIP CONFIG. The bias lives on the shared base, so its
        code runs on every boot; with no bias set, `merge_and_sort_free` must
        produce exactly the sorted list it always did. Also covers the
        allocators that do NOT call `super().__init__` (SWA, hisparse) and so
        never get the attribute at all -- every read of it is a `getattr`
        with a default, and this fails if one stops being.
        """
        real = self._real()
        del real._owner_bias  # the SWA/hisparse shape: attribute never set
        real.free_pages = torch.tensor([9, 3, 7], dtype=torch.int64)
        real.release_pages = torch.tensor([5, 1], dtype=torch.int64)
        real.merge_and_sort_free()
        self.assertEqual(real.free_pages.tolist(), [1, 3, 5, 7, 9])
        self.assertEqual(real.release_pages.numel(), 0)
        self.assertEqual(real._apply_owner_bias(), 0)

    def test_real_partition_holds_through_a_merge(self):
        # merge_and_sort_free re-sorts, which undoes the partition; the
        # override must put it back or the steer lives only until the first
        # refill.
        real = self._real()
        real.set_owner_bias((32, 24, 32))
        real.release_pages = torch.arange(400, 420, dtype=torch.int64)
        real.merge_and_sort_free()
        head = real.free_pages[:8]
        self.assertTrue(bool(((head % 32 >= 24) & (head % 32 < 32)).all()))
        self.assertEqual(real.release_pages.numel(), 0)


class TestSteeringCannotMoveResidency(unittest.TestCase):
    """The structural finding, pinned so it is not re-discovered by a boot.

    The scheduler holds ONE allocator for process life -- the PP stack's --
    and in that layout a slot id IS the row on every rank. So a residue-class
    steer changes WHICH ids are handed out, not how many rows each card
    commits, and the KV rung's floor is a function of the MAXIMUM live id.
    A bias therefore cannot lower that floor, and can raise it.
    """

    def test_a_class_bias_raises_the_maximum_live_id(self):
        plain = _FakeAllocator(320)
        biased = _FakeAllocator(320)
        biased.set_owner_bias((32, 24, 32))  # rank 2's class: ids 24..31 mod 32
        n = 40
        plain_max = int(plain.alloc(n).max())
        biased_max = int(biased.alloc(n).max())
        self.assertLess(plain_max, biased_max)
        # The number matters, not just the sign: the KV rung's floor is
        # max_live + 1 + reserve, and every row between the two ceilings
        # stays committed on all three cards for nothing.
        self.assertGreaterEqual(biased_max - plain_max, n)


class TestTheAgreementIdiomHasOneOwner(unittest.TestCase):
    """#856 F7: the (x, -x) MIN-pair ballot is not re-derived here.

    `decide` hand-rolled the idiom TWICE -- once for the absorbing-rank
    proposal, once for the free-list checksum -- while
    `managers/tree_congruence.py` already owned it as `digest_pair` /
    `agreement`. Three copies of "did the ranks agree?" is three places a
    defect in that question has to be found.

    This asserts IDENTITY, not equivalent behaviour, deliberately. A test
    that only checked the answers would pass against a fourth private copy
    that happens to agree today, which is exactly the state being retired.
    """

    def test_the_primitives_are_the_shared_ones(self):
        from sglang.srt.managers import corridor_steering, tree_congruence

        self.assertIs(corridor_steering.digest_pair, tree_congruence.digest_pair)
        self.assertIs(corridor_steering.agreement, tree_congruence.agreement)

    def test_the_shared_agreement_still_decides_both_ways(self):
        # The can-fail direction for the import above: the shared primitive
        # must actually discriminate, or pointing at it proves nothing.
        from sglang.srt.managers.tree_congruence import agreement, digest_pair

        lo, neg = digest_pair(7)
        self.assertTrue(agreement(lo, neg))
        # Two ranks proposing 7 and 9: MIN gives (7, -9), which is disagreement.
        self.assertFalse(agreement(min(7, 9), min(-7, -9)))


if __name__ == "__main__":
    unittest.main()
