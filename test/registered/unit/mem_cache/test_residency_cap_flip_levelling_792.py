# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#792: the phase flip's recovery levelling caps the pool below its live set.

#790 made the under-delivery VISIBLE and added a rung that peels past the
confiscated region. Boot ``instr12`` (2026-08-21, tree /spinning/boot12tree at
d59537cd9d) shows why that rung cannot win, and names the actuator that has to
change::

    05:28:17 PP1  KV-BACKING released 160 MiB by backing 137233 rows instead
                  of 161792 (highest live row 136720, 24145 ids withheld)
    05:28:21 PP1  KV-BACKING cap agreement: exposed rows 137233 -> 40960
                  (group level 40960, backed 49152)
    05:28:21 PP1  PHASE-FLIP-SPILL KV recovery levelled to the group: ... the
                  group's poorest backs 40960, so the allocator is capped at
                  40960 (-96344 exposed rows)
    05:29:28      RuntimeError: Out of memory. Try to allocate 512 tokens.
                  Available full tokens: 67935 (full_available_size=261 +
                  full_evictable_size_=67674)
                  EVICTION UNDER-DELIVERED: asked for 512 tokens, the pool
                  received 94 ... A RESIDENCY CAP IS ENGAGED and is holding
                  63641 slot ids out of the allocator's free list

**63641 is the number that identifies the culprit.** Above the shrink's cap of
137233 the id space contains 161792 - 137233 = 24559 ids IN TOTAL, so a cap at
that level can never withhold 63641 of them. Above the levelled cap of 40960 it
contains 120832, and 63641 fits. The cap that confiscated the pool to death was
not the shrink's -- which sits one page above the live set, by the #717 safety
net in ``_shrink_to`` -- but the one the tp_to_pp post-cutover levelling
installed 95760 rows BELOW the highest live row.

THE MECHANISM, in one sentence. ``phase_flip_spill.recover_kv_backing`` reduces
``[backed, -backed]`` and levels every rank's allocator to the group's MINIMUM
BACKED rows (phase_flip_spill.py:1165-1173) through
``KvBackingRelief.level_recovery_to`` -> ``reconcile_to``
(kv_backing_relief.py:2202+), which engaged the cap with no reference to the
live set at all -- so the cap lands under the rows the radix tree is holding,
``KvRowCap._apply`` (:490-512) confiscates every id the peel frees above it,
and the pool can never be paid again.

The seam's OTHER half already states the law this path was missing:
``collective_cap_target`` returns None when the group's MAX floor is above the
MIN capable, because "the poorest rank cannot expose the rows a peer's live set
requires" and "declining leaves the divergence for the flip's frame ballot to
refuse, which costs a flip and never a rank". The recovery levelling is the one
place in the chain that did not make that trade.

WHY NOT THE OTHER TWO CANDIDATE FIXES.

* RELEASING the cap at the cutover re-admits ids over pages the shrink has
  genuinely unmapped. That is the fault the cap exists to prevent, and it has
  been measured: 69054 rows of backing under a highest live row of 233289, and
  the next access above the cap was an illegal address (the crash that reverted
  c4e557963e, quoted in ``_shrink_to``). A blanket release trades an OOM that
  raises for a fault that kills every rank without raising.
* REFUSING the carry above an engaged cap is not available: the carried slots
  are not seeded by the flip, they are already live, and dropping them is a
  live request losing its KV at the seam. The order is wrong as well -- on this
  specimen the cap is engaged four seconds AFTER the cutover, by the recovery
  hook, so there is no carry left to refuse.

Hermetic: no CUDA, no distributed. The allocator and the residency cap are the
production classes -- ``_Pool`` stands in only for the VMM arena, which has no
CPU implementation, and it supplies none of the guarantees under test.
"""

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as kbr
from sglang.srt.managers import phase_flip_spill as pfs
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import alloc_token_slots, payable_size
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MIB = 1024 * 1024

#: The boot's numbers, scaled by 1/39.5 so a CPU test can hold them and the
#: shape survives: an id space, a live set covering most of it, and a group
#: minimum backing at a quarter of it.
#:
#:   metal 161792 id rows / 136720 highest live row / 40960 group minimum
POOL = 4096
LIVE_TOP = 3400
GROUP_MIN = 1024
RESERVE = 16
CHUNK = 256

#: What the shrink's own cap would be: one page and the admission reserve above
#: the live set, which is what ``_floor_rows`` computes and what the #717 net in
#: ``_shrink_to`` raises a too-deep target to.
SHRINK_CAP = LIVE_TOP + 1 + 1 + RESERVE


class _Card:
    def __init__(self, free_mib):
        self.free = int(free_mib) * MIB

    def probe(self):
        return self.free


class _Pool:
    """The VMM arena, reduced to the two numbers this path reads.

    Reservation (``size``) and backing (``full_pool_backed_rows``) are separate,
    as they are on the real pool. Nothing here decides whether a freed id
    reaches the allocator -- that is the production ``KvRowCap`` on a production
    ``TokenToKVPoolAllocator``, which is the guarantee this file is about.
    """

    def __init__(self, rows: int, backed: int, card: _Card):
        self.size = rows
        self.page_size = 1
        self._backed = backed
        self._card = card
        self.supports_backing_spans = True
        self.backing_commit_chunk_bytes = 0
        self.calls = []

    @property
    def full_pool_backed_rows(self) -> int:
        return self._backed

    def runtime_set_backing_rows(self, rows: int) -> int:
        self.calls.append(int(rows))
        self._backed = int(rows)
        return 0


def _allocator(size: int = POOL) -> TokenToKVPoolAllocator:
    """The production allocator on the CPU, with no KV tensors behind it.

    ``kvcache`` is dereferenced only by ``get_cpu_copy``/``load_cpu_copy``,
    which nothing on the allocation path calls; ``alloc``/``free``/
    ``available_size`` and the free-listener protocol are the real ones.
    """
    return TokenToKVPoolAllocator(
        size=size, dtype=torch.int64, device="cpu", kvcache=None, need_sort=False
    )


def _rank(*, backed: int, live_top: int = LIVE_TOP, capped_at=SHRINK_CAP):
    """One rank after a shrink, at the tp_to_pp post-cutover hook.

    ``_rows_at_boot`` is left None on purpose, which makes ``recover()`` return
    0 at its first line. That is instr12's state to the letter -- every rank
    logged "recovery deferred ... the pool stays at 40960 of 40960 rows" -- so
    the levelling below is the only actor, exactly as it was on metal.
    """
    card = _Card(4096)
    pool = _Pool(POOL, backed, card)
    alloc = _allocator()
    relief = kbr.KvBackingRelief(
        pool,
        alloc,
        live_slots_fn=lambda: torch.tensor([1, live_top], dtype=torch.int64),
        bytes_per_row=MIB,
        probe=card.probe,
        law_floor_bytes=1024 * MIB,
        admission_reserve_rows=RESERVE,
    )
    if capped_at is not None:
        relief._cap.engage(capped_at)
    return relief


def _scheduler(relief):
    sched = type("S", (), {})()
    setattr(sched, pfs.KV_BACKING_RELIEF_ATTR, relief)
    return sched


def _group_channel(fleet):
    """The seam's element-wise MIN channel over a simulated fleet.

    Every rank's own vector takes part -- ``min(mine, group)`` is what an
    all-reduce returns -- and the group half is gathered from the same public
    readings the production payload is built from, before any rank moves.
    """
    gathered = [
        [r.backed_rows(), -r.backed_rows(), -r.live_floor_rows()] for r in fleet
    ]
    group = [min(vals) for vals in zip(*gathered)]

    def reduce_fn(vals):
        return [min(mine, theirs) for mine, theirs in zip(vals, group)]

    return reduce_fn


def _level_the_fleet(fleet):
    """Run the tp_to_pp post-cutover recovery hook on every rank."""
    reduce_fn = _group_channel(fleet)
    for relief in fleet:
        pfs.recover_kv_backing(_scheduler(relief), reduce_fn=reduce_fn)


def _fleet():
    """instr12's shape: two ranks with the memory, one corridor-bound.

    Every rank holds the SAME token rows -- that is what pure PP means, and it
    is why the group's live floor is one number -- while the poorest rank's
    arena has come back at a quarter of the id space.
    """
    return [_rank(backed=POOL), _rank(backed=GROUP_MIN), _rank(backed=POOL)]


class CarriedTreeCache:
    """A tree that frees its leaves into the allocator and counts what it freed.

    That IS ``FullComponent.evict_component``: free the leaf's value, add
    ``len(value)`` to the tracker, subtract it from the evictable size. It does
    NOT check whether the pool received the slots, because the class it stands
    in for does not check either -- see
    ``test_residency_cap_eviction_790.py``, whose stand-in this one follows.

    The leaves come out oldest-first, and the oldest ones hold the high slot
    ids the flip carried across: the LRU frontier of the specimen.
    """

    def __init__(self, allocator, leaves):
        self.token_to_kv_pool_allocator = allocator
        self._leaves = [torch.tensor(ids, dtype=torch.int64) for ids in leaves]
        self._evictable = sum(int(t.numel()) for t in self._leaves)
        self.uniform_avail_floor = None
        self.evict_calls = []

    def evictable_size(self):
        return self._evictable

    def full_evictable_size(self):
        return self._evictable

    def is_chunk_cache(self):
        return False

    def is_tree_cache(self):
        return True

    def evict(self, params):
        asked = int(params.num_tokens)
        self.evict_calls.append(asked)
        counted = 0
        while counted < asked and self._leaves:
            leaf = self._leaves.pop(0)
            self.token_to_kv_pool_allocator.free(leaf)
            counted += int(leaf.numel())
            self._evictable -= int(leaf.numel())
        return type("R", (), {"num_tokens_evicted": counted})()

    def pretty_print(self):
        return ""

    def available_and_evictable_str(self):
        a = int(self.token_to_kv_pool_allocator.available_size())
        return (
            f"Available full tokens: {a + self._evictable} "
            f"(full_available_size={a} + full_evictable_size_={self._evictable})\n"
        )


def _carried_tree(relief):
    """The radix tree the flip left behind, on a drained pool.

    Slots run up to the live high-water mark, which is far above the level the
    levelling wants to cap at; one low leaf survives so the eviction pays
    SOMETHING and the specimen's "the pool received 94" is reproduced rather
    than a total zero.

    Thirty high leaves, deliberately more than the ask needs, so the run shows
    what #790's relief rung does here: it peels the tree EMPTY -- "the tree
    still reports 0 evictable tokens" in the raise below -- and the pool is
    still 192 tokens short. Peeling is the wrong actuator against a
    confiscator that takes everything above its line; the cap is.
    """
    alloc = relief._alloc
    payable = int(alloc.available_size())
    if payable:
        alloc.alloc(payable)
    leaf = CHUNK // 4
    high = [
        list(range(LIVE_TOP - (i + 1) * leaf, LIVE_TOP - i * leaf)) for i in range(30)
    ]
    low = [list(range(1, 1 + leaf))]
    return CarriedTreeCache(alloc, high + low)


class TheLevellingMustNotCapBelowTheLiveSet(unittest.TestCase):
    """The root, on the production cap and the production allocator."""

    def test_the_recovery_levelling_leaves_every_rank_able_to_pay(self):
        fleet = _fleet()
        floors = [r.live_floor_rows() for r in fleet]
        self.assertEqual(
            len(set(floors)), 1, "fixture: pure PP means one group live floor"
        )
        self.assertGreater(
            floors[0], GROUP_MIN, "fixture: the group minimum must be below the floor"
        )

        _level_the_fleet(fleet)

        for i, relief in enumerate(fleet):
            self.assertGreaterEqual(
                relief.exposed_rows(),
                floors[i],
                f"rank {i} was capped at {relief.exposed_rows()} with its live "
                f"set needing {floors[i]}: every id the radix tree is holding "
                f"is now above the cap, so eviction frees the TREE and pays "
                f"the POOL nothing",
            )

    def test_a_freed_slot_still_reaches_the_pool_after_the_cutover(self):
        """The delivery itself, measured the way #790 measures it."""
        fleet = _fleet()
        _level_the_fleet(fleet)

        rich = fleet[0]
        alloc = rich._alloc
        alloc.alloc(int(alloc.available_size()))
        before = payable_size(alloc)
        alloc.free(torch.arange(LIVE_TOP - CHUNK, LIVE_TOP, dtype=torch.int64))
        self.assertEqual(
            payable_size(alloc) - before,
            CHUNK,
            "the pool was paid nothing for a leaf the tree gave up: the cap "
            "the cutover installed sits below these slot ids and its free "
            "listener takes them straight back",
        )

    def test_the_next_prefill_survives_the_cutover(self):
        """End to end: the specimen's raise, or its absence."""
        fleet = _fleet()
        _level_the_fleet(fleet)

        rich = fleet[0]
        tree = _carried_tree(rich)
        got = alloc_token_slots(tree, CHUNK)
        self.assertIsNotNone(got)
        self.assertEqual(len(got), CHUNK)
        self.assertLessEqual(
            int(max(got)),
            rich.exposed_rows(),
            "a slot above this rank's exposed level names a page it may not "
            "hand out",
        )

    def test_the_levelling_reports_the_numbers_when_it_declines(self):
        """A refusal nobody can read is the #790 silence again."""
        fleet = _fleet()
        with self.assertLogs(pfs.logger, level="ERROR") as captured:
            _level_the_fleet(fleet)
        joined = "\n".join(captured.output)
        self.assertIn("DECLINED to level the recovery", joined)
        self.assertIn(str(GROUP_MIN), joined)
        self.assertIn(str(fleet[0].live_floor_rows()), joined)


class TheLevellingStillHappensWhenItIsHonest(unittest.TestCase):
    """The guard is a limit, not an off switch.

    If this class goes red the fix has disabled #656 C22-e -- the divergence
    that wedged boot ``boot_m1`` for 33 minutes -- instead of bounding it.
    """

    def _quiet_fleet(self):
        """The same fleet with an idle live set, which is the ordinary case."""
        return [
            _rank(backed=POOL, live_top=8, capped_at=None),
            _rank(backed=GROUP_MIN, live_top=8, capped_at=None),
            _rank(backed=POOL, live_top=8, capped_at=None),
        ]

    def test_a_group_whose_live_set_fits_is_levelled_as_before(self):
        fleet = self._quiet_fleet()
        _level_the_fleet(fleet)
        self.assertEqual(
            {r.exposed_rows() for r in fleet},
            {GROUP_MIN},
            "every rank must expose the SAME id space when the level is "
            "honest; one rank above the group is #656 C22-e's whole defect",
        )

    def test_the_levelling_releases_no_pages(self):
        fleet = self._quiet_fleet()
        backed = [r.backed_rows() for r in fleet]
        _level_the_fleet(fleet)
        self.assertEqual([r.backed_rows() for r in fleet], backed)


class TheActuatorRefusesWhoeverAsks(unittest.TestCase):
    """The invariant belongs to the actuator, not to one caller.

    ``reconcile_to`` is reached from the seam agreement as well, and a future
    third caller must not be able to recreate the state either.
    """

    def test_reconcile_to_declines_a_target_below_the_live_floor(self):
        relief = _rank(backed=POOL)
        before = relief.exposed_rows()
        moved = relief.reconcile_to(GROUP_MIN)
        self.assertEqual(moved, 0)
        self.assertEqual(
            relief.exposed_rows(),
            before,
            "the actuator moved the cap under the live set for a caller that "
            "asked it to; the group check is a second line of defence, not "
            "the only one",
        )

    def test_level_recovery_to_declines_it_too(self):
        relief = _rank(backed=POOL)
        self.assertEqual(relief.level_recovery_to(GROUP_MIN), 0)
        self.assertEqual(relief.exposed_rows(), SHRINK_CAP)

    def test_a_target_that_clears_the_floor_is_still_applied(self):
        relief = _rank(backed=POOL, capped_at=None)
        floor = relief.live_floor_rows()
        relief.reconcile_to(floor)
        self.assertEqual(relief.exposed_rows(), floor)

    def test_an_unreadable_live_set_declines_rather_than_guessing(self):
        """An unknown live set is not an empty one -- ``_shrink_to``'s reading,
        applied to the id decision."""

        def boom():
            raise RuntimeError("live set unavailable")

        relief = _rank(backed=POOL)
        relief._live_slots_fn = boom
        self.assertEqual(relief.live_floor_rows(), kbr._UNBOUNDED_ROWS)
        self.assertEqual(relief.reconcile_to(GROUP_MIN), 0)


class TheDeclineIsAGroupDecision(unittest.TestCase):
    """Every rank must reach the same verdict, or the ranks part company.

    The verdict is read off the REDUCED vector, so it is a function of a value
    every rank computes identically -- the same construction
    ``collective_cap_target`` uses, and the reason a rank whose own live set is
    tiny still declines when a peer's is not.
    """

    def test_a_rank_with_a_small_live_set_declines_with_the_group(self):
        loaded = _rank(backed=POOL)
        quiet = _rank(backed=POOL, live_top=8)
        poor = _rank(backed=GROUP_MIN, live_top=8)
        fleet = [loaded, quiet, poor]
        _level_the_fleet(fleet)
        self.assertEqual(
            {r.exposed_rows() for r in fleet},
            {SHRINK_CAP},
            "the quiet ranks levelled while the loaded one refused; the group "
            "now exposes two different id spaces, which is the divergence "
            "this whole mechanism exists to prevent",
        )

    def test_a_truncated_channel_declines(self):
        """A payload that came back short leaves the floor unknown."""
        fleet = _fleet()
        backed = [r.backed_rows() for r in fleet]

        def short_reduce(vals):
            return [min(backed), -max(backed)]

        for relief in fleet:
            pfs.recover_kv_backing(_scheduler(relief), reduce_fn=short_reduce)
        self.assertEqual(
            {r.exposed_rows() for r in fleet},
            {SHRINK_CAP},
            "an absent floor field was read as a low floor, so a peer on an "
            "older tree could still cap this rank below its live set",
        )


if __name__ == "__main__":
    unittest.main()
