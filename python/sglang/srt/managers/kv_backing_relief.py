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
"""#656 spec item 12: KV residency follows the load, so KV is a relief provider.

    "THERE IS NO FIXED MAX KV: KV is itself a spill class into system RAM
     ... what sits in VRAM at any moment is EXACTLY what has to be there right
     then, the rest in system RAM."

(Translated from the user's German; the original wording is in the commit that
introduced this module.)

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
The KV pool already sits on a VA reservation: ``swappable_backing=True`` is
passed whenever the phase flip is on, so the pool's addresses are fixed at
boot and only the PHYSICAL pages underneath move. That is the property spec
item 13 needs -- a residency change cannot invalidate a captured CUDA graph,
because nothing the graph baked in has moved.

So the machinery to unmap KV pages already existed
(``runtime_set_backing_rows`` -> ``KvVmmBufferOwner.shrink`` -> ``cuMemUnmap``
+ ``cuMemRelease``). What did NOT exist is the thing that makes using it under
load safe, and it is the whole content of this module:

**THE ALLOCATOR CAP.** ``shrink`` states its precondition plainly -- "rows
above the new span must be dead" -- and nothing in the tree computed a safe
shrink point from the live set. The one existing shrink path, the #330 vram
dial, sidesteps the problem by DESTROYING the live set first
(``tree_cache.reset()``, ``req_to_token_pool.clear()``,
``allocator.resize()``), which is fine for a dial turned between runs and
impossible under serving load. Without a cap, the allocator goes on believing
it may hand out every id up to ``size``; the next allocation above the
watermark writes to unmapped VA, and that is ``cudaErrorIllegalAddress`` --
a FAULT that kills every rank, not an exception someone catches.

:class:`KvRowCap` closes that hole non-destructively. It never touches a live
allocation: it withholds the high ids from the FREE LIST, which is the only
place unallocated capacity lives. ``available_size()`` then falls out correct
without being told, because it is derived from the free list, and the
scheduler simply admits less work -- which is the intended behaviour under
pressure, and infinitely better than a fault.

THREE PLACES A CAP LEAKS, AND WHY EACH IS A TEST
-------------------------------------------------
1. **Eviction does not compact.** A freed id keeps its value, so a high id
   freed after the cap was applied walks straight back onto the free list. The
   cap therefore subscribes to the allocator's free listener and re-applies
   itself on every free.
2. **``clear()`` rebuilds ``arange(1, size+1)``**, silently re-admitting every
   id above the watermark while the backing is still unmapped. The cap
   re-applies on clear for the same reason.
3. **A cap that bought nothing is worse than no cap**, because it costs
   capacity and returns no bytes. If the driver did not move, the cap comes
   straight back off.

THE RETURNED BYTES ARE MEASURED, NEVER BELIEVED
-----------------------------------------------
``runtime_set_backing_rows`` returns bytes UNMAPPED. Under
``SGLANG_FLIP_SEAM_RETAIN_HANDLES`` the arena parks the physical handle
instead of releasing it, so those bytes are address space and NVML's free
column never moves. The corridor law is stated in NVML's free column and the
ledger law says price a payload from what the driver actually gave back, so
this provider probes free memory before and after and reports the DIFFERENCE.
That makes it immune to retention rather than dependent on a flag, and it is
the same discipline that caught the drafter estimate, the idle mamba slots and
kvso -- three payloads in this chain that freed nothing the driver could see.

WHAT IS NOT HERE YET
--------------------
This rung releases backing that NO row occupies -- the slack between the live
high-water mark and the pool's reservation. It moves no data anywhere, so it
is the cheapest half of item 12 and the correct one to build first. Lowering
the watermark FURTHER requires evicting cached prefix entries (data discarded,
recomputable) and then spilling live sessions to kvso's pinned host pool (data
moved, restorable). Both lower ``max_live`` and then reuse exactly this code
path; they are separate providers at higher cost, not changes to this one.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-BACKING"

_MIB = 1024 * 1024

#: Report EVERY proposal, not only the ones that change the deficit's sign.
KV_RELIEF_TRACE_ENV = "SGLANG_KV_RELIEF_TRACE"

#: A desire no reduction can lower: the neutral element of an element-wise MIN.
_UNBOUNDED_ROWS = 1 << 40

#: Rows this rung refuses to give up ON TOP of the live high-water mark, so a
#: shrink leaves a pool that can still ADMIT. See :meth:`KvBackingRelief._floor_rows`.
KV_ADMISSION_RESERVE_ENV = "SGLANG_KV_ADMISSION_RESERVE_ROWS"

#: #662: may this rung lower the high-water mark by EVICTING recomputable
#: prefix cache, rather than only releasing the slack above it?
#:
#: Defaults ON, because with it OFF the seam can only be funded by VRAM held
#: free at rest -- which is the corridor-law breach this flag exists to end.
#: It is kept as a flag purely so the CAN-FAIL PROOF is runnable on metal: at
#: 0 the guard must refuse a flip from a corridor-filled pool, and at 1 the
#: same flip must clear. A relief mechanism that has never been observed to
#: change an outcome is indistinguishable from one that is never reached.
KV_RADIX_EVICT_ENV = "SGLANG_KV_RADIX_EVICT_RELIEF"

#: One chunked prefill on the shipped configuration (``chunked_prefill_size``
#: 512), which is the exact allocation that raised when the floor reserved
#: nothing: "Try to allocate 512 tokens. Available full tokens: 0". The factory
#: derives the real default from the scheduler and falls back to this.
DEFAULT_ADMISSION_RESERVE_ROWS = 512

#: The proposal of a rank that cannot take part in a shrink at all -- no relief
#: object, no VA reservation, or a live set it could not read. Its third field
#: (``current``) is 0, and :func:`collective_kv_target` declines whenever the
#: group's minimum ``current`` is not positive, so ONE abstention cancels the
#: whole decision.
#:
#: That is the correct direction and the expensive lesson of HANDOFF_675 §1a:
#: the danger is never "nobody capped", it is "some capped and some did not".
#: An abstaining rank cannot cap, so its peers must not either.
ABSTAIN = (_UNBOUNDED_ROWS, 0, 0, 0)


def collective_kv_target(reduced):
    """The group's shared row target, from an element-wise MIN of proposals.

    ``reduced`` is what a MIN all-reduce returns over the four-field proposals
    :meth:`KvBackingRelief.propose` produces::

        [ desire, -floor, current, -current ]
          |         |       |        |
          |         |       |        `-- max current pool rows (diagnostic)
          |         |       `-- MIN current pool rows: the shared reference
          |         `-- negated, so MIN yields the MAX floor across the group
          `-- MIN desire: the most-pressed rank sets the ambition

    Two independent quantities, and getting their relationship backwards is a
    fault rather than an inefficiency:

    * the AMBITION is a minimum -- relief is driven by whichever rank is
      closest to the corridor floor, because that is the rank the flip will
      otherwise be refused on;
    * the LIMIT is a maximum -- the target must clear the highest live row on
      EVERY rank, because the target is an absolute row id and the shrink
      unmaps physical pages under it. A target below a peer's live set is
      ``cudaErrorIllegalAddress``, which kills every rank rather than raising.

    **The limit wins.** Returns None when there is nothing every rank can give
    up, which is a normal outcome and not an error.
    """
    if len(reduced) < 3:
        return None
    desire = int(reduced[0])
    max_floor = -int(reduced[1])
    min_current = int(reduced[2])
    if min_current <= 0:
        # An abstention, or a rank with no pool. Not a shrink the group can
        # make uniformly, so it is not a shrink the group makes.
        return None
    target = max(desire, max_floor)
    if target >= min_current:
        return None
    return int(target)


#: The proposal of a rank that cannot take part in the CAP AGREEMENT below.
#: Its first field (``capable``) is 0, and :func:`collective_cap_target`
#: declines whenever the group's minimum is not positive, so one abstention
#: cancels the levelling for everyone -- the same direction as :data:`ABSTAIN`,
#: and for the same reason.
CAP_ABSTAIN = (0, 0, 0, 0)


def collective_cap_target(reduced):
    """The ONE row level every rank's allocator exposes, from a MIN reduction.

    #656 C22, and it is the recovery half of the law
    :func:`collective_kv_target` states for the shrink: *a refusal may be
    decided locally, a CAPACITY may not*.

    ``reduced`` is what a MIN all-reduce returns over the four-field proposals
    :meth:`KvBackingRelief.cap_proposal` produces::

        [ capable, -floor, exposed, -exposed ]
            |         |       |        |
            |         |       |        `-- MAX exposed level (is the group level?)
            |         |       `-- MIN exposed level right now
            |         `-- negated, so MIN yields the MAX floor across the group
            `-- MIN capable: the poorest rank sets the level the group lives at

    **The poorest rank sets the level, and the most-loaded rank sets the
    limit.** ``capable`` is how far a rank could expose rows *without breaching
    its own corridor law*, so the MIN is a level every rank can honour with
    memory it has actually mapped. ``floor`` is the highest live row plus the
    admission reserve, so a target below it would withhold ids a request is
    using -- the limit therefore wins, exactly as it does for the shrink.

    Returns the level unconditionally when one exists -- INCLUDING when the
    group is already at it, because "no change needed" is a property of each
    rank's own state and :meth:`KvBackingRelief.reconcile_to` is a no-op in
    that case. A None here means there is no honest level at all: a rank
    abstained, or -- the one asymmetry against the shrink -- the MAX floor is
    ABOVE the MIN capable. That last case cannot be answered: the poorest rank
    cannot expose the rows a peer's live set requires, and forcing it would
    hand out unmapped memory. Declining leaves the divergence for the flip's
    frame ballot to refuse, which costs a flip and never a rank.
    """
    if len(reduced) < 4:
        return None
    capable = int(reduced[0])
    max_floor = -int(reduced[1])
    if capable <= 0:
        # An abstention, or a rank with nothing to expose.
        return None
    if max_floor > capable:
        return None
    min_exposed = int(reduced[2])
    max_exposed = -int(reduced[3])
    if min_exposed == max_exposed == capable:
        # ALREADY LEVEL, AND THAT IS NOT THE SAME AS "NOTHING TO DO ON THIS
        # RANK". When the group is NOT level, every rank -- including the ones
        # already at the target -- has to run the same normalisation, because
        # the cap's release path SORTS the free list while its apply path
        # preserves eviction order. A rank that skipped would hand out
        # different row ids from the ones that moved, which is a divergent
        # live slot set and therefore a divergent wire frame. Measured on
        # metal: boot_v2, six abandoned rounds, pool census identical on all
        # three ranks. So the "nothing to do" decision is taken HERE, from
        # the reduced view of the whole group, and never per rank.
        return None
    return int(capable)


#: The live-slot half of the widened rung payload, for a rank that has no
#: relief object to read (no pool, a stub runtime, a hermetic test). The
#: digest pair is ``(0, 0)`` so it cannot make the group's digests disagree by
#: itself, the row extent is ``-1`` (contributes nothing to the MAX), and the
#: backing is :data:`_UNBOUNDED_ROWS` so the MIN is decided by whichever rank
#: actually knows its backing.
SLOT_ABSTAIN = (0, 0, -1, _UNBOUNDED_ROWS)


def slot_proposal(digest: int, max_live_row: int, backed_rows: int):
    """This rank's four-field proposal for the group's LIVE SLOT agreement.

    #656 C22-d. The pairing is the same one the fit verdict and the frame
    ballot already use, so it costs four integers on a reduction the rung
    runs anyway and no second collective::

        [ digest, -digest, -max_live_row, backed_rows ]
            |        |          |             `-- MIN backed rows: the highest
            |        |          |                 row id EVERY rank has mapped
            |        |          `-- negated, so MIN yields the group's MAX
            |        |              live row id
            |        `-- negated, so MIN yields the MAX digest
            `-- MIN digest: with the pair above, ``min == -max`` answers
                "do the ranks enumerate the SAME live slot set"

    ``digest`` must be a function of the SET and nothing else -- the caller
    feeds it a sorted, deduplicated enumeration -- or this ballot would
    disagree on a reordering that is not a divergence at all.
    """
    return (int(digest), -int(digest), -int(max_live_row), int(backed_rows))


def collective_slot_ballot(reduced):
    """Decode the live-slot half of the reduced rung payload.

    Returns ``None`` when the payload is too short (a peer on an older tree,
    or a channel that truncated), which the caller must treat as "no verdict"
    and not as agreement -- an absent ballot leaves today's behaviour, which
    is the frame ballot refusing the flip.
    """
    if reduced is None or len(reduced) < 4:
        return None
    lo = int(reduced[0])
    hi = -int(reduced[1])
    return {
        "agree": lo == hi,
        "digest_lo": lo,
        "digest_hi": hi,
        # The group's highest live row id: the id space a union has to span.
        "max_live_row": -int(reduced[2]),
        # The highest row id EVERY rank has physically backed. A union may not
        # contain a row at or above it: on the rank whose backing ends there,
        # the mover would read unmapped memory.
        "min_backed_rows": int(reduced[3]),
    }


class KvRowCap:
    """Withhold slot ids above ``cap`` from the allocator's free list.

    Non-destructive by construction: live allocations are not enumerated, not
    moved and not touched. Only unallocated ids are held back, so engaging a
    cap can never invalidate a row a request is using.
    """

    def __init__(self, allocator: Any) -> None:
        self._alloc = allocator
        self._cap: Optional[int] = None
        self._withheld = None
        self._subscribed = False

    @property
    def engaged(self) -> bool:
        return self._cap is not None

    @property
    def cap(self) -> Optional[int]:
        return self._cap

    @property
    def withheld(self) -> int:
        return 0 if self._withheld is None else int(self._withheld.numel())

    def _publish(self) -> None:
        """Tell the allocator how much capacity is out of circulation.

        The scheduler's idle invariant checks
        ``available + evictable + protected + session_held + uncached ==
        total``. Withheld capacity is in none of those buckets, so without a
        term of its own it reads as a LEAK -- and it is a fatal one: the first
        boot that exercised the cap died at the first idle check with
        "pool memory leak detected! [full] total=500000, available=419745".

        Published in the unit ``available_size()`` reports, which is TOKENS:
        the paged allocator holds pages in its free list and multiplies by
        ``page_size``, so a raw id count would be wrong by that factor on
        every paged lane.
        """
        page = max(1, int(getattr(self._alloc, "page_size", 1) or 1))
        try:
            self._alloc.residency_withheld_slots = self.withheld * page
        except Exception:  # pragma: no cover - exotic allocator objects
            pass

    def engage(self, cap: int) -> int:
        """Hold back every free id above ``cap``. Returns the count withheld."""
        import torch

        self._cap = int(cap)
        if not self._subscribed:
            # Both hooks exist for the same reason: an id above the cap that
            # re-enters the free list is an id the next allocation may hand to
            # a kernel writing into unmapped memory.
            register = getattr(self._alloc, "register_free_listener", None)
            if register is not None:
                # THE TWO HOOKS ARE NOT THE SAME FUNCTION (#485). On a FREE,
                # ids above the cap re-enter the list one batch at a time and
                # must be ADDED to what is already withheld. On a CLEAR the
                # allocator rebuilds ``free_pages`` as ``arange(1, size+1)``,
                # so nothing is outstanding any more and the withheld set has
                # to be recomputed from scratch. Wiring the accumulating
                # ``_apply`` to both made a clear double-book its own ids:
                # measured 2026-08-12, ``available=267217 withheld=25566``
                # against ``total=280000`` -- withheld exactly 2x the true
                # count -- which the idle invariant reports as a pool memory
                # leak and which puts DUPLICATE ids into the free list on the
                # next ``release()``.
                register(lambda _idx: self._apply(), self._on_clear)
                self._subscribed = True
            else:
                logger.warning(
                    "%s allocator has no free listener; a freed high id can "
                    "re-enter the free list above the backed watermark",
                    LOG_PREFIX,
                )
        self._apply()
        if self._withheld is None:
            self._withheld = torch.empty((0,), dtype=torch.int64)
        self._publish()
        return self.withheld

    def release(self) -> int:
        """Return every withheld id. Returns the count restored."""
        import torch

        self._cap = None
        n = self.withheld
        if self._withheld is not None and n:
            for name in ("free_pages", "release_pages"):
                pages = getattr(self._alloc, name, None)
                if pages is not None:
                    # Sorted, because the allocator takes from the FRONT and
                    # the high-water mark this rung prices itself against only
                    # tracks occupancy while low ids are reused first.
                    #
                    # THROUGH THE HOST, for the corridor reason spelled out in
                    # ``sort_free_lists``: the merge genuinely changes the
                    # tensor's SIZE, so one device allocation is unavoidable --
                    # but doing the cat and the sort on the device would take
                    # three more, and this path now runs on every recovery
                    # levelling rather than only on the rare cap agreement.
                    host = torch.cat(
                        (
                            pages.detach().to("cpu", torch.int64),
                            self._withheld.to("cpu", torch.int64),
                        )
                    )
                    ordered = torch.sort(host).values
                    setattr(
                        self._alloc,
                        name,
                        ordered.to(device=pages.device, dtype=pages.dtype),
                    )
                    break
        self._withheld = None
        self._publish()
        return n

    def sort_free_lists(self) -> None:
        """Put every free list in ascending id order on THIS rank.

        Called by the group cap agreement so that the order is a function of
        MEMBERSHIP and nothing else. Two ranks with the same free ids must
        hand the next request the same row id, or their live slot sets -- and
        therefore the lengths of the payloads they frame -- part company.

        IT SORTS THROUGH THE HOST, AND THAT IS A CORRIDOR DECISION (#656
        C22-e). The free list is a device tensor of one int64 per row --
        4.7 MiB at this rig's 586642 rows -- and ``torch.sort`` on it
        allocates BOTH a values and an indices tensor, so the obvious
        in-place-looking ``setattr(..., torch.sort(pages).values)`` costs
        ~14 MiB of transient DEVICE memory. That was invisible while this ran
        only on the rare rounds the cap agreement moved a rank. C22-d made it
        run every seam round on every rank, and the seam is exactly where this
        rig's corridor is tightest: measured 2026-08-14, gpu0's continuous
        minimum fell from 1028/1084 MiB (0 samples below the 1024 law across
        159212) to 978/990 MiB with 2 and 4 samples BELOW it. The law is a
        hard user limit, so the sort may not be paid for in device memory.

        Sorting on the host and writing back with ``copy_`` allocates NOTHING
        on the device: the storage is reused. The host pays ~10 MiB and one
        round trip of a few milliseconds, once per seam, against a flip
        cadence measured in tens of seconds.

        The equality guard is not an optimisation for its own sake -- it skips
        the write-back entirely on the common round where the list is already
        ascending, which is every round after the first one that had nothing
        to reorder.
        """
        import torch

        for name in ("free_pages", "release_pages"):
            pages = getattr(self._alloc, name, None)
            if pages is None or pages.numel() < 2:
                continue
            host = pages.detach().to("cpu", torch.int64, copy=True)
            ordered = torch.sort(host).values
            if torch.equal(host, ordered):
                continue
            pages.copy_(ordered.to(pages.dtype))

    def _on_clear(self) -> None:
        """Re-apply the cap after the allocator rebuilt its free list.

        ``clear()`` replaces ``free_pages`` with ``arange(1, size + 1)``, so
        every id the cap was holding is back in the list and NONE of them is
        outstanding. The withheld set is therefore stale in full, not stale in
        part: it is dropped and recomputed. Adding to it instead is what
        published twice the true count and put duplicate ids into the free
        list on the following ``release()``.
        """
        self._withheld = None
        self._apply()
        # A cap that was engaged stays published even when the rebuilt list
        # happens to hold nothing above it, so the counter never keeps a value
        # the free list no longer supports.
        self._publish()

    def _apply(self) -> None:
        """Move ids above the cap out of every free list.

        Accumulates: on a free, ids above the cap re-enter one batch at a
        time and each batch adds to what is held. That is correct for the
        free path and WRONG for a clear, which is why the clear has its own
        hook (``_on_clear``) rather than sharing this one.
        """
        import torch

        if self._cap is None:
            return
        for name in ("free_pages", "release_pages"):
            pages = getattr(self._alloc, name, None)
            if pages is None or pages.numel() == 0:
                continue
            over = pages > self._cap
            if not bool(over.any()):
                continue
            taken = pages[over].to("cpu", torch.int64)
            setattr(self._alloc, name, pages[~over])
            self._withheld = (
                taken if self._withheld is None else torch.cat((self._withheld, taken))
            )
            # A ``torch.unique`` belt here was written and REMOVED on purpose.
            # It also makes the symptom disappear -- and that is the problem:
            # with it in place the regression tests below pass whether or not
            # ``_on_clear`` exists, so the instrument could no longer fail and
            # would have certified the wrong fix. Duplicates are prevented by
            # having the clear rebuild its set, which is the actual invariant;
            # a dedupe would have hidden the next path that books twice.
            self._publish()


class KvBackingRelief:
    """A corridor-guard provider that returns UNOCCUPIED KV backing.

    ``free_up_to(nbytes)`` lowers the pool's physical backing to just above
    the highest live row, releasing at most the rows the ask needs, and
    returns the bytes NVML says it got back.
    """

    def __init__(
        self,
        pool: Any,
        allocator: Any,
        *,
        live_slots_fn: Callable[[], Any],
        bytes_per_row: int,
        probe: Optional[Callable[[], int]] = None,
        device_index: int = 0,
        margin_rows: int = 0,
        buffers: int = 0,
        law_floor_bytes: int = 1024 * 1024 * 1024,
        admission_reserve_rows: int = DEFAULT_ADMISSION_RESERVE_ROWS,
        tree_cache_fn: Optional[Callable[[], Any]] = None,
        pool_fn: Optional[Callable[[], Any]] = None,
        flip_pending_fn: Optional[Callable[[], Any]] = None,
        flip_armed_fn: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._pool = pool
        #: THE ID-SPACE OWNER, and it never moves. The scheduler holds ONE
        #: allocator for the life of the process -- a single id space is what
        #: makes a row identifiable across the flip at all -- so the cap, the
        #: reservation and everything the collective cap agreement reads stay
        #: anchored on the pool this object was built with, whatever pool the
        #: backing calls are currently acting on.
        self._id_space_pool = pool
        #: #662: WHICH POOL HOLDS THE PAGES RIGHT NOW. The flip has two
        #: layouts and two arenas, and only one of them is backed at a time.
        #: Resolved per call rather than captured, for the reason the tree
        #: cache is: a reference taken once is a reference to whichever layout
        #: happened to be active when this object was built, and on the
        #: tp_to_pp leg that is the EMPTY one. See :meth:`_rebind`.
        self._pool_fn = pool_fn
        #: #744: THE PARKED EXTENT. A phase flip quiesces its requests before
        #: it packs them, and a quiesced request is in none of the batch
        #: structures ``_live_reqs`` enumerates -- so ``req_rows`` reads 0
        #: while its rows are still allocated and about to be read by
        #: ``_pack_outgoing``. Both the trigger (``_nothing_resident``) and
        #: the safety net (``_shrink_to`` via ``_max_live_row``) go through
        #: that one enumeration, so they were blind together. This side
        #: channel is what they see the parked rows through.
        #:
        #: Returns ``(rows, max_row_id)``; ``(-1, -1)`` means UNKNOWN, which
        #: is treated as blocking, never as empty. ``None`` leaves the rung
        #: exactly as it was.
        self._flip_pending_fn = flip_pending_fn
        #: #744 second line: refuse to evict at all while a flip is armed.
        #: Independent of the extent above on purpose -- the failure is a
        #: SILENT eviction followed by a delayed illegal access, so one
        #: mechanism is not enough. Gated on ARMED only: the rung must stay
        #: fully alive outside flips (#688's evict-rung funding depends on
        #: it), and ``test_744`` pins that it is not dead.
        self._flip_armed_fn = flip_armed_fn
        #: Per-pool backing state, keyed by pool identity. Geometry, the boot
        #: reservation and the exhaustion marker are all facts about ONE
        #: arena and must not follow the rung across a rebind.
        self._pool_state: dict = {}
        self._alloc = allocator
        self._live_slots_fn = live_slots_fn
        #: #662: the id-targeted evictor that lowers ``max_live`` itself.
        #: Without it this rung can only release backing NO row occupies --
        #: the slack above the high-water mark -- which is precisely why the
        #: seam had to be funded from VRAM held free at rest. Resolved per
        #: call rather than captured: the tree cache is rebuilt on a flush,
        #: and a stale reference would evict into a tree nobody reads.
        self._tree_cache_fn = tree_cache_fn
        #: Rows given up to the watermark actuator, cumulative, for the log.
        self.evicted_rows_total = 0
        self.evict_count = 0
        self._bytes_per_row = int(bytes_per_row)
        #: Number of arena buffers (2 x layer_num). The release granularity is
        #: one commit chunk in EACH of them, not one chunk overall.
        self._buffers = int(buffers)
        self._probe = probe
        self._device_index = int(device_index)
        self._margin_rows = int(margin_rows)
        #: The USER'S CORRIDOR LAW, and deliberately not the guard's arming
        #: floor. Recovery is bounded by this; a proof run that raises the
        #: arming floor must make the gate work earlier, never make the pool
        #: permanently smaller.
        self._law_floor_bytes = int(law_floor_bytes)
        #: THE ADMISSION RESERVE (#656 register C20, residual 1). Rows kept
        #: ABOVE the live high-water mark, so what survives a shrink is a pool
        #: that can still take work rather than only hold it. Zero restores the
        #: pre-2026-08-11 floor exactly, as a value of the same term.
        self._admission_reserve_rows = max(0, int(admission_reserve_rows))
        self._cap = KvRowCap(allocator)
        #: The row count to restore to. Latched on the FIRST shrink and never
        #: overwritten by a second one, so a two-step relief still recovers to
        #: the boot reservation rather than to the intermediate step.
        self._rows_at_boot: Optional[int] = None
        #: THE BACKING LEVEL AT WHICH A SHRINK RETURNED NO DRIVER BYTES, or
        #: None while this rank is willing to be asked. Read through the
        #: :attr:`_exhausted` property, which compares it against the CURRENT
        #: level -- so exhaustion expires the moment the backing moves.
        #:
        #: #662-F4: THIS USED TO BE A BOOL, AND IT LATCHED FOR THE LIFE OF THE
        #: PROCESS. Measured on metal 2026-08-15: one shrink of a pool the
        #: phase flip had already emptied returned zero bytes (it could not
        #: have returned anything -- see :meth:`_current_rows`), the flag
        #: latched, and from that instant the rung declined every ask on BOTH
        #: legs while reporting ``slack=170368`` rows. The tp_to_pp seam then
        #: abandoned nine times for want of ~500 MiB and the instance never
        #: reached the prefill layout again.
        #:
        #: One shrink at one backing level is evidence about THAT level. It is
        #: not evidence about the next one, and the cost of treating it as
        #: permanent is the entire prefill layout. A retry is cheap now that a
        #: failed shrink is never undone (:meth:`_shrink_to`): it engages a cap
        #: and calls the dial in the SHRINK direction, which never allocates.
        self._exhausted_at_rows: Optional[int] = None
        #: The target whose shrink returned nothing. A DEEPER ask re-arms.
        self._exhausted_target_rows: Optional[int] = None
        self.shrink_count = 0
        self.recover_count = 0
        self.released_total = 0
        #: -1 means "nothing reported yet", so the FIRST proposal always logs
        #: and a run can never be silent about this rung again.
        self._last_deficit_sign = -1
        #: The cause of the last ABSTAIN, or None when this rank is taking
        #: part. Edge-triggers the abstain warning and gates the recovery line.
        self._last_abstain_reason: Optional[str] = None
        self._abstain_count = 0
        self._trace_all = os.environ.get(KV_RELIEF_TRACE_ENV, "") == "1"

    # -- plumbing --------------------------------------------------------

    def _rebind(self) -> None:
        """Point the backing calls at the layout that actually holds pages.

        THE RUNG WAS BOUND TO ONE POOL AND THE FLIP HAS TWO. The scheduler's
        pool is the PP layout's, so on the pp_to_tp leg the rung is looking at
        the source -- backed, with slack above the live set, able to pay. On
        the tp_to_pp leg the SAME pool is the destination, and the seam emptied
        it a phase ago: no extents, nothing to release, and every proposal it
        makes is arithmetic over memory that is not there. That is why the leg
        into the prefill layout had no funder even after the exclusion in
        ``collective_kv_backing_relief`` was lifted.

        The money on that leg is in the TP layout's pool: it is the SOURCE, it
        is fully backed, and the rows above its live high-water mark hold
        nothing. Releasing them early hands the gate exactly the bytes it is
        about to refuse for -- and the seam was going to release that whole
        layout at the cutover anyway, so this is the same memory arriving in
        time to be useful rather than one gate too late.

        WHAT DOES NOT MOVE: the cap and the id space. Both layouts index the
        same allocator, so a target is a row id and means the same thing in
        either pool; :meth:`_reservation_rows` and everything the collective
        cap agreement reads stay on ``_id_space_pool``. Only geometry, the
        boot reservation and the exhaustion marker are per-arena, and those
        are carried in ``_pool_state``.
        """
        if self._pool_fn is None:
            return
        try:
            pool = self._pool_fn()
        except Exception as e:
            logger.warning(
                "%s could not resolve the active layout's pool (%s); staying "
                "on the pool this rung was built with",
                LOG_PREFIX,
                e,
            )
            return
        if pool is None or pool is self._pool:
            return
        # Park the state of the pool we are leaving.
        self._pool_state[id(self._pool)] = {
            "bytes_per_row": self._bytes_per_row,
            "buffers": self._buffers,
            "rows_at_boot": self._rows_at_boot,
            "exhausted_at_rows": self._exhausted_at_rows,
            "exhausted_target_rows": getattr(self, "_exhausted_target_rows", None),
            "pool": self._pool,
        }
        state = self._pool_state.get(id(pool))
        if state is None:
            row_bytes, n_buffers = row_geometry(pool)
            state = {
                "bytes_per_row": int(row_bytes),
                "buffers": int(n_buffers),
                "rows_at_boot": None,
                "exhausted_at_rows": None,
                "pool": pool,
            }
            logger.info(
                "%s now funding from the ACTIVE layout's pool on device %s: "
                "%d B/row over %d arena buffers. The pool this rung was built "
                "with is the other layout's and is unbacked while that layout "
                "is inactive, so a proposal against it would be arithmetic "
                "over memory that is not mapped.",
                LOG_PREFIX,
                self._device_index,
                int(row_bytes),
                int(n_buffers),
            )
        self._pool = pool
        self._bytes_per_row = int(state["bytes_per_row"])
        self._buffers = int(state["buffers"])
        self._rows_at_boot = state["rows_at_boot"]
        self._exhausted_at_rows = state["exhausted_at_rows"]
        self._exhausted_target_rows = state.get("exhausted_target_rows")

    @property
    def _exhausted(self) -> bool:
        """Is this rank declining to be asked RIGHT NOW?

        True only while the backing still stands exactly where the failed
        shrink left it. Any movement -- a recovery, a grow, the phase flip
        restoring this layout -- re-arms the rung, because the arena that
        could not pay at one level is a different proposition at another.
        """
        if self._exhausted_at_rows is None:
            return False
        # Read the level FIRST: ``_current_rows`` retires the marker when it
        # sees an emptied layout, so the marker must be re-read afterwards
        # rather than captured across the call.
        current = int(self._current_rows())
        marker = self._exhausted_at_rows
        return marker is not None and current == int(marker)

    def _declines_target(self, target: int) -> bool:
        """Is this rank still declining, GIVEN what is being asked of it?

        Exhaustion holds only while the ask is no deeper than the one that
        failed. A target at least one release granularity below the failed one
        is a different question and gets a different answer -- which is what
        stops the marker from being a deadlock (see :meth:`_mark_exhausted`).
        """
        if not self._exhausted:
            return False
        failed = getattr(self, "_exhausted_target_rows", None)
        if failed is None:
            return True
        granularity = max(1, self._min_release_rows())
        if int(target) <= int(failed) - granularity:
            # Deeper than the ask that failed: a different question, because
            # release is extent-granular.
            return False
        # SLACK OVERRIDES THE MARKER, and this is the rule the original brief
        # asked for and I twice failed to implement: "never per process, when
        # slack >> need".
        #
        # Keying on the target alone refuses every SHALLOWER ask after a deep
        # one failed -- and the asks that follow are always shallower, because
        # the deficit only ever asks for what it needs. Measured 12:26:41: PP1
        # held 112,126 rows of slack above its floor, priced a real +1009 MiB
        # deficit, and still declined, because a deep shrink had failed
        # earlier from a different level. Several GiB sat releasable behind a
        # marker.
        #
        # One failed shrink is weak evidence and this is where it stops being
        # decisive: when the slack in front of the rung dwarfs what is being
        # asked for, the cost of trying is one cap and one dial call that
        # cannot allocate, and the cost of not trying is the prefill layout.
        chunked = self._min_release_rows()
        if chunked <= 0:
            # No commit chunk means no extent can clear at ANY depth, so slack
            # is not evidence of anything and the marker stands. (Such a pool
            # is disqualified from the rung entirely at construction; this is
            # the belt.)
            return True
        current = self._current_rows()
        slack = max(0, current - int(target))
        return slack < 2 * chunked

    @_exhausted.setter
    def _exhausted(self, value: bool) -> None:
        """Keep the boolean spelling, now meaning "exhausted AT THIS LEVEL".

        Setting True latches against the backing as it stands right now, so
        the statement stays true exactly as long as the evidence for it does.
        """
        if value:
            self._mark_exhausted()
        else:
            self._exhausted_at_rows = None

    def last_proposal_summary(self) -> str:
        """One line describing this rung's most recent decision, or why none.

        For the caller that REFUSES: at that moment the reader needs to know
        whether the rung declined, abstained, or was never reached, and those
        three have very different fixes. Returns a sentence rather than a
        dict, because it is going straight into a refusal message.
        """
        t = getattr(self, "_last_proposal_terms", None)
        if t is None:
            return (
                "the KV rung produced NO proposal this round -- it was not "
                "reached, which is a different defect from declining"
            )
        verdict = (
            f"SHRINK to {t['desire']}" if t["desire"] < t["current"] else "no change"
        )
        why = t["skipped"] or (
            "the cheaper tier covered the gap"
            if t["deficit"] <= 0
            else "KV capacity is the funder"
        )
        # #714: a floor ABOVE the cap is not a tight round, it is an
        # impossible one, and "slack=0" alone cannot tell them apart.
        #
        # Measured on 0b61699cc3: current=137216, floor=398471. The floor
        # formula is right -- margin_rows defaults to 0 and is never passed,
        # and the admission reserve is chunked_prefill_size (512) -- so
        # floor = max_live + 513 and max_live was 397,958.
        #
        # CORRECTED (#717, F4-r4 c4e557963e): I first read that as a stale id
        # outliving a pool shrink. It is not. 397,958 is a VALID id in the
        # ~437k id space while only 137,216 rows are backed -- the live set is
        # SPARSE, so a high-water id above the backed-row count is normal. The
        # actual root there was _resident_ceiling encoding "none" and "unknown"
        # as the same -1 sentinel, so an idle box read as an unreadable split
        # and eviction was never priced. This guard stays because the CONDITION
        # it reports is real and disabling: whatever the cause, a floor above
        # the cap pins slack to 0 and the rung cannot fund.
        # slack is max(0, ...), so it pins to 0 for as long as that holds and
        # the rung can never propose a shrink. The evict-rung funding path is
        # then permanently unavailable and every flip falls back on the raw
        # seam fund alone -- which is why that boot abandoned three times over
        # a 55 MiB shortfall instead of funding it from KV once.
        unreachable = ""
        if t["floor_rows"] > t["current"]:
            gap = int(t["floor_rows"]) - int(t["current"])
            unreachable = (
                f" -- FLOOR UNREACHABLE: it exceeds the current cap by {gap} "
                "rows, so this rung can never fund and every flip depends on "
                "the raw seam fund alone. The floor is max_live + 1 + margin + "
                "admission reserve over a SPARSE live set: max_live is a "
                "high-water ID in the id space, not a count of backed rows, so "
                "it can legitimately exceed the number of rows backed. State "
                "the fact, do not infer the cause."
            )
        return (
            f"KV rung: current={t['current']} rows, floor={t['floor_rows']}, "
            f"slack={max(0, t['current'] - t['floor_rows'])}, deficit="
            f"{t['deficit'] / _MIB:+.0f} MiB -> {verdict} ({why}){unreachable}"
        )

    def _mark_exhausted(self, target: Optional[int] = None) -> None:
        """Record the level AND the target at which the arena returned nothing.

        BOTH, because the level alone is self-locking. A shrink that releases
        nothing leaves the physical level exactly where it was, so a marker
        keyed only to the level marks the level the rung is stuck at -- and the
        only thing that could move it is a successful shrink, which the marker
        now prevents. Measured on this rig 2026-08-15: a shrink to 94955 rows
        returned no driver bytes at 12:16:00, and 47 seconds later the rung was
        still declining with 72981 rows of slack in front of it, at the same
        level, for ever.

        The target is what makes the evidence falsifiable. "A shrink to X
        returned nothing" says nothing about a shrink to something deeper than
        X -- release is extent-granular, so a deeper ask clears extents a
        shallower one could not touch. That is the same argument the granularity
        rounding in :meth:`free_up_to` already makes.
        """
        self._exhausted_at_rows = int(self._current_rows())
        self._exhausted_target_rows = None if target is None else int(target)

    def _free_bytes(self) -> int:
        if self._probe is not None:
            return int(self._probe())
        import torch

        return int(torch.cuda.mem_get_info(self._device_index)[0])

    def _supported(self) -> bool:
        return callable(getattr(self._pool, "runtime_set_backing_rows", None))

    def _current_rows(self) -> int:
        """Rows PHYSICALLY BACKED right now -- never the reservation.

        ``pool.size`` is the logical row count and it does not move when the
        backing does: ``initial_backing_rows`` states plainly that it "does
        NOT touch self.size", because on this pool family ``size`` keeps the
        stock semantics of the non-VMM constructor. The committed span lives
        in ``full_pool_backed_rows``.

        Reading ``size`` instead cost a boot on 2026-08-11. After the first
        shrink to 347161 rows, ``size`` still said 500000, so the next ask was
        computed against 500000 and produced a target of 379067 -- ABOVE the
        committed span. ``runtime_set_backing_rows`` converges the backing to
        its argument in BOTH directions, so that was a grow:
        ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY``, from inside relief,
        on the card that needed relieving.

        #662-F4: AND ``full_pool_backed_rows`` IS NOT PHYSICAL EITHER. Its name
        promises a measurement; it returns ``full_kv_pool.size``, a CONFIGURED
        row count. That was harmless while the #330 dial was the only thing
        moving the backing, because the dial writes ``size`` on every step. The
        phase flip does not: ``release_backing`` / ``restore_backing`` unmap and
        remap this pool's pages and say so in their own comment -- "SIZING IS
        NOT TOUCHED". So for the whole of the TP phase the PP layout's pool
        holds NO committed extents while ``size`` still reports its pre-flip
        count.

        Measured on metal 2026-08-15, tp_to_pp gate, all three ranks:

            KV-BACKING proposal ... rows current=407051 floor=1157
              (max_live=644 + admission reserve 512, slack=405894)
            KV-BACKING shrink to 222081 rows reported 0 MiB but the driver's
              free column did not move

        Those 405894 rows of "slack" did not exist. The pool had been emptied
        by the pp_to_tp cutover eighteen seconds earlier, so the shrink could
        not have returned a byte -- and the zero it returned was then read as
        evidence that the ARENA was exhausted, which latched the rung off for
        the rest of the process (see :attr:`_exhausted_at_rows`).

        So ask the arena. ``backed_bytes`` is the number the boot's own
        exclusive-backing pin asserts on, and it cannot report backing that is
        not mapped.
        """
        backed = self._physical_backed_rows()
        if backed is None:
            rows = getattr(self._pool, "full_pool_backed_rows", None)
            backed = (
                int(rows) if rows is not None else int(getattr(self._pool, "size", 0))
            )
        if backed <= 0:
            # SEEING AN EMPTY LAYOUT RETIRES THE EXHAUSTION MARKER, and this
            # is the one place every caller passes through, which is why the
            # invalidation lives here rather than in the property.
            #
            # A layout the flip has emptied carries no evidence about an
            # arena: its extents went back to the driver, and the pages that
            # return on the next restore are different handles. Worse, the
            # level it comes back at can equal the level the failed shrink
            # left behind -- so a marker compared only by level would survive
            # a whole phase and go on declining. That is the process-lifetime
            # latch wearing a level-shaped disguise, and it is the defect this
            # work exists to remove.
            self._exhausted_at_rows = None
        return backed

    def _physical_backed_rows(self) -> Optional[int]:
        """Rows the ARENA has committed, or None when it cannot be measured.

        Release is extent-granular, so this can exceed the exact row span by
        less than one commit chunk per buffer. That overshoot is bounded and
        in the safe direction: it never invents backing that is not there,
        which is the only error mode that matters here.

        None -- never 0 -- when the pool does not expose the reading, so a pool
        that never flips keeps exactly its previous behaviour.
        """
        if self._bytes_per_row <= 0:
            return None
        # THE MINIMUM ACROSS BUFFERS, not the average, and the difference is
        # the whole point. ``backed_bytes`` is a SUM, so dividing it by the
        # all-buffers per-row size gives an AVERAGE depth -- true only when the
        # backing is uniform, which the waved seam guarantees it is not.
        # ``decommit_range`` frees extents lying wholly above the keep point
        # PER BUFFER, so a target derived from the average sits above the
        # shallowest buffer's watermark and the shrink returns nothing while
        # looking like a large one.
        #
        # Measured 2026-08-15, the 2048-chunk boot: read 591872 from the
        # average, asked for 320217 and 352067, got 0 MiB nine times. The six
        # shrinks that PAID were the ones whose target was below every buffer
        # (73345 from 149504). Same defect class as reading the configured
        # size, one level down: a number that is not what the shrink acts on.
        uniform = getattr(self._pool, "uniform_backed_rows", None)
        if uniform is not None:
            try:
                rows = int(uniform)
            except (TypeError, ValueError):
                rows = -1
            if rows >= 0:
                return rows
        raw = getattr(self._pool, "backed_bytes", None)
        if raw is None:
            return None
        try:
            backed = int(raw)
        except (TypeError, ValueError):
            return None
        if backed < 0:
            return None
        return backed // self._bytes_per_row

    def _reserved_rows(self) -> Optional[int]:
        """The arena's immutable row ceiling, or None when unreadable (#684).

        None -- never 0 -- when the pool exposes no reservation, so a pool
        without an arena keeps exactly its previous behaviour and the clamp
        simply does not engage. 0 from the pool means the same thing: no
        arena, hence nothing to clamp against, NOT a ceiling of zero.

        NOT :meth:`_reservation_rows`, and the two must not be conflated. That
        one is the ALLOCATOR's id space, read from the id-space owner, and it
        feeds ``exposed_rows`` and the collective cap agreement. This one is
        the ARENA's VA span on whichever layout the backing calls currently
        point at -- the bound ``_check_final`` enforces on a grow.

        RANK-LOCAL BY NATURE, and that is why reading it needs no collective.
        A reservation is one card's VA span; under uneven TP the ranks hold
        different ones (190596 / 136140 / 108912 on the 2026-08-16 boot), so
        there is no group number here to agree on. What recovery changes is
        this rank's own physical backing, and the module's collective -- the
        cap agreement -- is on the SHRINK target, which is unchanged.
        """
        raw = getattr(self._pool, "reserved_backing_rows", None)
        if raw is None:
            return None
        try:
            rows = int(raw)
        except (TypeError, ValueError):
            return None
        return rows if rows > 0 else None

    def _min_release_rows(self) -> int:
        """Rows that must be given up before ANY extent can clear.

        One commit chunk in EVERY buffer, expressed in rows. Below this the
        release is arithmetically guaranteed to be zero, so attempting it can
        only waste a cap and exhaust the provider.
        """
        chunk = int(getattr(self._pool, "backing_commit_chunk_bytes", 0) or 0)
        if chunk <= 0 or self._buffers <= 0:
            return 0
        return int(math.ceil(chunk * self._buffers / self._bytes_per_row))

    def _describe_live_split(self, max_live: int) -> str:
        """One clause naming WHAT pins the ceiling, or '' when unknown.

        The clause exists because the two sources have different futures.
        Rows held by resident requests are the floor's irreducible half. Rows
        held only by the radix tree are evictable by the cache's own policy,
        so a ceiling pinned by the TREE is a floor that could be lowered
        without giving up a single live token -- at the price of prefix-cache
        hits, which is a price this instance's own traffic can be measured
        against rather than assumed.

        NOTHING ACTS ON THIS YET, deliberately. Its whole purpose is to say
        how much the unbuilt actuator would be worth before anyone builds it,
        because five shifts of this chain have built relief for payloads that
        turned out to be empty.
        """
        split = getattr(self, "_last_live_split", None)
        if not split:
            return ""
        tree_max = int(split.get("tree_max", -1))
        req_max = int(split.get("req_max", -1))
        if tree_max < 0 and req_max < 0:
            return ""
        pinned_by = "the radix tree" if tree_max >= req_max else "resident requests"
        # What the floor would be if the tree stopped pinning it: the
        # resident half, which is what an eviction could not touch.
        headroom_rows = max(0, tree_max - req_max)
        return (
            f" | ceiling pinned by {pinned_by} (tree_max={tree_max} over "
            f"{int(split.get('tree_rows', 0))} rows, req_max={req_max} over "
            f"{int(split.get('req_rows', 0))} rows); an id-targeted eviction "
            f"could lower max_live={int(max_live)} by at most {headroom_rows} "
            f"rows"
        )

    def _max_live_row(self) -> int:
        try:
            live = self._live_slots_fn()
        except Exception as e:
            # An unknown live set is not an empty one. Refusing to shrink is
            # the only safe reading, because the number this decides is the
            # point below which memory gets unmapped.
            logger.warning("%s live-set probe failed: %s", LOG_PREFIX, e)
            return -1
        # #657: who PINS the ceiling. The floor this number produces is what
        # keeps backing committed on every card, and its two sources are
        # priced completely differently: a resident request's row cannot be
        # given up, a radix-tree row is evictable by the cache's own policy.
        # Read from the live-set function's own side channel -- enumerating
        # is the expensive half and it has just been done.
        self._last_live_split = getattr(self._live_slots_fn, "last_split", None)
        # #744: THE SAFETY NET READS THIS FUNCTION TOO. ``_shrink_to``
        # re-measures through here to turn the cap from a hope into a fact,
        # but it re-measures the SAME enumeration that missed the parked
        # request -- so trigger and net were blind together, and fixing only
        # the trigger would leave the net equally blind to any other caller
        # that shrinks during a park. Folding the parked extent in HERE is
        # what makes both see it from one source.
        pending_rows, pending_top = self._flip_pending()
        if pending_rows < 0:
            # Unknown parked extent: refuse to shrink, same reading as an
            # unknown live set above.
            logger.warning(
                "%s parked flip extent unreadable -- refusing to shrink",
                LOG_PREFIX,
            )
            return -1
        if live is None or int(getattr(live, "numel", lambda: 0)()) == 0:
            return max(0, pending_top)
        return max(int(live.max()), pending_top)

    def _floor_rows(self, max_live: int) -> int:
        """The lowest row count this rank may be capped to, in rows.

        The shrink precondition stated as a number: every row at or below the
        high-water mark must stay backed, plus one page of slack so the very
        next allocation does not immediately re-arm.

        PLUS THE ADMISSION RESERVE, and that term is the whole of register
        C20's residual 1. The precondition above protects the rows that EXIST.
        It reserves nothing to admit new work with, so a caller that asks this
        rung for more than the card can fund drives it here on every seam and
        the pool arrives at a state where ``available_size()`` is 0 while every
        live row is perfectly safe. Measured on metal 2026-08-11 under
        ``SGLANG_SEAM_ENTRY_MARGIN_MIB=8192``: 42 cutovers after the first
        delay, three ranks raised
        ``Out of memory. Try to allocate 512 tokens. Available full tokens: 0``
        inside ``_get_new_batch_prefill_raw`` -- in the scheduler loop, which
        is fatal, and two minutes AFTER the branches that were suspected.

        The reserve sits above ``max_live`` rather than being carved out of it
        because that is the only range whose freeness is guaranteed: every id
        above the high-water mark is unallocated by definition, while ids below
        it may all be in use. So a cap of ``max_live + 1 + margin + reserve``
        leaves at least ``reserve`` ALLOCATABLE ids, which is the quantity
        admission actually spends.

        GROUP-UNIFORM WITHOUT A NEW COLLECTIVE. The reduction already takes the
        MAX floor across the group (``collective_kv_target``), so a target that
        clears this rank's reserve clears every rank's.
        """
        page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
        floor = max(
            page,
            int(max_live) + 1 + self._margin_rows + self._admission_reserve_rows,
        )
        return int(math.ceil(floor / page) * page)

    # -- #662: the watermark actuator -------------------------------------

    def _flip_pending(self) -> tuple:
        """``(rows, max_row_id)`` the flip has parked. ``(-1, -1)`` = unknown.

        UNKNOWN IS NOT EMPTY. An unreadable probe returns ``(-1, -1)`` and
        every caller below treats that as "something may be parked", because
        the cost of guessing wrong is an unmapped row under a live read.
        """
        # getattr, not attribute access: these methods are invoked UNBOUND
        # against stubs by the #717 suites (constructing a real rung needs a
        # pool, an allocator and a live-set function), so an object without
        # the channel must degrade to "nothing parked", not raise.
        fn = getattr(self, "_flip_pending_fn", None)
        if fn is None:
            return (0, -1)
        try:
            rows, top = fn()
            return (int(rows), int(top))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s flip-pending probe failed: %s", LOG_PREFIX, e)
            return (-1, -1)

    def _flip_armed(self) -> bool:
        """True when a phase flip is armed on this rank (or unreadable)."""
        fn = getattr(self, "_flip_armed_fn", None)
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s flip-armed probe failed: %s", LOG_PREFIX, e)
            return True

    def _evict_enabled(self) -> bool:
        return os.environ.get(KV_RADIX_EVICT_ENV, "1") not in ("0", "false", "False")

    def _tree_cache(self):
        if self._tree_cache_fn is None:
            return None
        try:
            return self._tree_cache_fn()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s tree cache unavailable: %s", LOG_PREFIX, e)
            return None

    def _resident_ceiling(self) -> int:
        """Highest row a RESIDENT REQUEST pins, or -1 when none/unknown.

        This is the half of the live set eviction cannot touch, and it is
        therefore the true floor of the watermark. Read from the live-set
        function's side channel (``last_split``), which the flip's own
        enumeration already populates -- see build_flip_live_slots_fn.

        AN UNREADABLE SPLIT RETURNS -1 AND THE CALLER MUST TREAT THAT AS
        "DO NOT EVICT". Defaulting an unknown resident ceiling to 0 would
        say "every row is evictable", which is the one wrong answer that
        unmaps memory a live request is reading.
        """
        split = getattr(self, "_last_live_split", None)
        if not split:
            return -1
        try:
            return int(split.get("req_max", -1))
        except Exception:  # pragma: no cover - defensive
            return -1

    def _nothing_resident(self) -> bool:
        """True when the split is READABLE and reports zero resident rows.

        #717. ``_resident_ceiling`` returns -1 for two opposite states and the
        caller cannot tell them apart:

          * the split is unreadable -- evict NOTHING, because unmapping a row
            a live request is reading is the one unrecoverable error;
          * there are no resident requests at all -- every live row is held by
            the radix tree alone, so the set is recomputable.

        ``build_flip_live_slots_fn`` sets ``req_max`` to -1 when it has no
        request parts, so the second state is encoded exactly like the first,
        and the rung took the conservative branch precisely when it had the
        most to win.

        WHAT THIS PREDICATE DOES NOT MEAN, and the first attempt at #717 read
        it this way: it does not mean there are no live rows. The tree's rows
        are live and addressable. It means only that they are RECOMPUTABLE --
        that an eviction is permitted to try. Whether the eviction actually
        succeeded is a separate question, answered after the fact in
        ``_shrink_to``, never assumed here.

        The split records ``req_rows`` beside ``req_max``, so the two states
        are distinguishable from data already on hand -- no new enumeration.
        """
        # #744: a request PARKED for a phase flip is in none of the batch
        # structures the split enumerates, so ``req_rows == 0`` is also what a
        # mid-flip quiesce looks like. On 2026-08-17 19:30:40 that read
        # evicted 127,731 rows the flip was about to pack and the next access
        # above the new cap was an illegal address, 24 log lines later. The
        # parked extent is consulted FIRST because it is the one state in
        # which the rest of this predicate is confidently wrong.
        pending_rows, _ = self._flip_pending()
        if pending_rows != 0:
            return False
        split = getattr(self, "_last_live_split", None)
        if not split:
            return False
        try:
            return int(split.get("req_rows", -1)) == 0
        except Exception:  # pragma: no cover - defensive
            return False

    def _evict_floor_rows(self, max_live: int) -> Tuple[int, int]:
        """``(floor_rows, evictable_rows)`` if the mark were lowered.

        The pair the proposal needs: how low this rank could cap once the
        recomputable half of the live set is given up, and what that would
        cost in rows. When eviction is unavailable, or the ceiling is
        already pinned by resident requests, this degrades EXACTLY to
        ``_floor_rows(max_live)`` with zero cost -- so a rank that cannot
        evict proposes precisely what it proposed before this existed.
        """
        plain = self._floor_rows(max_live)
        if not self._evict_enabled():
            return plain, 0
        # #744 second line, independent of the extent probe: while a flip is
        # ARMED its requests are being quiesced, so every read of "what is
        # resident" is in motion. Refuse for the duration. Gated on ARMED
        # only -- outside a flip this rung stays fully live, which #688's
        # funding depends on.
        if self._flip_armed():
            return plain, 0
        tree = self._tree_cache()
        if tree is None:
            return plain, 0
        req_max = self._resident_ceiling()
        if req_max < 0:
            if not self._nothing_resident():
                # Unknown resident half: refuse to price an eviction at all.
                return plain, 0
            # #717: NOTHING RESIDENT is not "unknown". No request pins any
            # row, so nothing is above the reserve that an eviction may not
            # touch, and _floor_rows(-1) is the reserve alone. Treating this
            # as unknown is what pinned slack to 0 on an idle box and left
            # every flip funded by the raw seam budget.
            #
            # This is a PRICE, not a promise. It says what could be won if the
            # eviction succeeds; _shrink_to re-reads the live set afterwards
            # and raises the cap if it did not. Pricing optimistically here is
            # only safe BECAUSE of that check -- the first attempt at #717
            # made this same change without it and unmapped live rows.
            req_max = -1  # _floor_rows(-1) == the reserve, nothing above it
        if req_max >= int(max_live):
            # The mark is pinned by work in flight; nothing to win here.
            return plain, 0
        floor = self._floor_rows(req_max)
        if floor >= plain:
            return plain, 0
        try:
            from sglang.srt.managers.kv_radix_watermark import evictable_rows_above

            rows, _nodes = evictable_rows_above(tree, max(0, floor - 1))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s could not price the watermark rung: %s", LOG_PREFIX, e)
            return plain, 0
        if rows <= 0:
            return plain, 0
        return floor, int(rows)

    def _lower_watermark_to(self, target: int) -> int:
        """Evict every recomputable row at or above ``target``. Rows freed.

        Called on the SHRINK path only, immediately before the cap, so the
        rows the cap is about to withhold are genuinely unoccupied by the
        time it withholds them.
        """
        if not self._evict_enabled():
            return 0
        tree = self._tree_cache()
        if tree is None:
            return 0
        # #744: same refusal on the collecting half. Both sides must agree on
        # what an armed flip means, for the reason the comment below already
        # gives about the branch: a disagreement becomes an illegal address.
        if self._flip_armed():
            return 0
        req_max = self._resident_ceiling()
        if req_max < 0 and not self._nothing_resident():
            return 0
        # #717, THE HALF THE FIRST ATTEMPT MISSED. It opened PRICING on the
        # nothing-resident branch and left this refusal in place, so the rung
        # priced a win it then declined to collect: the target dropped to the
        # reserve, this method returned 0 without evicting anything, and the
        # cap engaged over a full live set. Both sides must agree on what the
        # branch means, or the disagreement becomes an illegal address.
        #
        # req_max stays -1 here, which `evict_rows_above` reads as "no
        # resident row pins anything" and therefore does not refuse -- the
        # correct reading when the split has told us there are zero resident
        # rows.
        try:
            from sglang.srt.managers.kv_radix_watermark import evict_rows_above

            # A cap of ``target`` rows admits ids strictly below ``target``,
            # so the last id that may survive is ``target - 1``.
            freed = int(
                evict_rows_above(
                    tree, max(0, int(target) - 1), resident_ceiling=req_max
                )
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s watermark eviction failed: %s", LOG_PREFIX, e)
            return 0
        if freed > 0:
            self.evicted_rows_total += freed
            self.evict_count += 1
            logger.info(
                "%s EVICTED %d recomputable row(s) to bring the high-water "
                "mark below %d on device %d (resident ceiling %d, %.1f MiB of "
                "prefix cache given up, %d row(s) over %d seam(s) so far). "
                "This is the seam's fund: content, not empty VRAM.",
                LOG_PREFIX,
                freed,
                int(target),
                self._device_index,
                req_max,
                freed * self._bytes_per_row / _MIB,
                self.evicted_rows_total,
                self.evict_count,
            )
        return freed

    def fundable_bytes(self) -> int:
        """Bytes this rank could return WITHOUT crossing its admission floor.

        The bound a caller needs to ask this rung for a DISCRETIONARY amount
        honestly. A rank that cannot take part answers 0, which reads as "ask
        me for nothing extra" -- the safe direction, since the mandatory part
        of an ask is never bounded by this.

        Pure: it reads the live set and the backing and computes. It is on the
        gate's unconditional path, so it must not touch residency.
        """
        self._rebind()
        if not self._supported() or self._bytes_per_row <= 0 or self._exhausted:
            return 0
        max_live = self._max_live_row()
        if max_live < 0:
            return 0
        current = self._current_rows()
        if current <= 0:
            return 0
        # #662: quote the floor this rank could actually REACH, which
        # includes the recomputable half of the live set. Quoting the plain
        # floor here while the shrink path can go lower would understate the
        # rung to its only caller, and an honest bound is the whole purpose
        # of this method.
        floor, _evictable = self._evict_floor_rows(max_live)
        return max(0, current - floor) * self._bytes_per_row

    # -- the collective decision -----------------------------------------

    def propose(
        self,
        *,
        want_bytes: int,
        floor_bytes: int,
        delta_bytes: int,
        cheap_relief_bytes: int = 0,
    ):
        """This rank's four-field proposal for the group's shrink target.

        PURE: it reads free memory and the live set and computes; it changes
        no residency and touches no allocator. That is what lets every rank
        call it unconditionally, which is the property the reduction needs --
        a collective reached by only some ranks is a hang, and putting one
        behind the guard's rank-local arm condition would turn the capacity
        desync of HANDOFF_675 §1a into something strictly worse.

        ``cheap_relief_bytes`` is what the tiers BELOW this one could still
        return (on this rig: torch's allocator cache). Counting it here is how
        the tier law survives the rung's move out of the guard's ladder --
        free money is spent before KV capacity is. The estimate may overstate
        what those tiers really return, and that is the safe direction:
        overstating cheap relief understates this ask, and under-shrinking is
        recoverable (the guard refuses, the flip is retried next round) while
        over-shrinking costs admission capacity that bought nothing.
        """
        self._rebind()
        if not self._supported():
            return self._abstain(
                "the pool has no runtime_set_backing_rows entry point, so this "
                "rank cannot change its backing at all"
            )
        if self._bytes_per_row <= 0:
            return self._abstain(
                f"bytes_per_row is {self._bytes_per_row}, so no row count can "
                "be computed from a byte deficit"
            )
        current = self._current_rows()
        if current <= 0:
            return self._abstain(
                f"the pool reports {current} backed rows, so there is no "
                "backing to give up"
            )
        max_live = self._max_live_row()
        if max_live < 0:
            # An unknown live set is not an empty one, and this is the number
            # that decides where memory gets unmapped. Abstain, and take the
            # group with us -- see ABSTAIN.
            return self._abstain(
                "the live set could not be read, and an unknown live set is "
                "not an empty one -- it is the row below which unmapping is a "
                "FAULT"
            )
        self._clear_abstain()
        # #662: THE FLOOR IS NOW A CHOICE, NOT A READING. Without the
        # watermark actuator the floor is wherever the cache happens to
        # have left its highest id, and on a corridor-filled pool that is
        # at or above ``current`` -- the rung then reports "no slack above
        # the live set" and funds nothing, which is exactly why the seam
        # had to be paid for in VRAM held free at rest. With it, the floor
        # is the RESIDENT half of the live set, and the difference is
        # recomputable prefix cache this rung may spend.
        floor_rows, evictable_rows = self._evict_floor_rows(max_live)
        self._last_evictable_rows = int(evictable_rows)
        desire = current
        # Hoisted so the diagnostic below can report the terms on the path
        # where the rung declines, which is the ONLY path it has ever taken.
        free_now = -1
        deficit = 0
        # WHY THE RUNG DECLINED, as a distinct fact from the deficit's sign.
        # The first version of this trace printed "the cheaper tier covers the
        # whole gap" on EVERY non-shrinking path, including the two where no
        # gap is ever computed -- a diagnostic that states a false cause is
        # worse than one that states none, because the next reader stops
        # looking.
        skipped = ""
        if floor_rows >= current:
            skipped = (
                f"no slack above the live set: floor_rows {floor_rows} >= "
                f"current {current}, so there is nothing this rung may give up"
            )
        # THE DEFICIT IS COMPUTED BEFORE EXHAUSTION IS CONSULTED, and the order
        # is the fix. Exhaustion used to short-circuit here, which meant the
        # rung could not tell how DEEP an ask it was refusing -- and since a
        # shrink that releases nothing leaves the level unchanged, the marker
        # keyed to that level could never expire. Measured: 47 s of declining
        # with 72981 rows of slack in front of it, at a level nothing could
        # move. Pricing first costs one free-memory probe and makes the refusal
        # answerable: a target deeper than the one that failed is new evidence.
        if floor_rows < current:
            free_now = self._free_bytes()
            need = int(floor_bytes) + int(delta_bytes) + max(0, int(want_bytes))
            deficit = need - free_now - max(0, int(cheap_relief_bytes))
            if deficit > 0:
                rows = int(math.ceil(deficit / self._bytes_per_row))
                # Rounded UP to the arena's release granularity: below it the
                # release is arithmetically guaranteed to be zero (§1d).
                rows = max(rows, self._min_release_rows())
                page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
                desire = max(floor_rows, current - rows)
                desire = int(math.ceil(desire / page) * page)
                if self._declines_target(desire):
                    failed = getattr(self, "_exhausted_target_rows", None)
                    skipped = (
                        "this rank's arena returned no driver bytes at a "
                        f"shrink to {failed}, and this ask ({desire}) is not "
                        "deeper than that by a release granularity, so it is "
                        "the same question and gets the same answer"
                    )
                    desire = current
        self._trace_proposal(
            current=current,
            floor_rows=floor_rows,
            max_live=max_live,
            want_bytes=int(want_bytes),
            floor_bytes=int(floor_bytes),
            delta_bytes=int(delta_bytes),
            free_now=free_now,
            cheap_relief_bytes=int(cheap_relief_bytes),
            deficit=deficit,
            desire=desire,
            skipped=skipped,
        )
        return (int(desire), -int(floor_rows), int(current), -int(current))

    def _abstain(self, reason: str):
        """Return ABSTAIN and say so. Never silently.

        WHY THIS IS LOUDER THAN A DECLINE. A decline is this rank's arithmetic
        saying the cheap tier already covers the gap -- the tier law working.
        An ABSTAIN is this rank saying it cannot take part, and
        :func:`collective_kv_target` then cancels the decision for EVERY rank,
        because the danger was never "nobody capped", it is "some capped and
        some did not" (HANDOFF_675 §1a). So one rank's local defect turns spec
        item 12 off node-wide, and from the outside that is indistinguishable
        from a rung whose deficit never crossed zero -- the exact confusion
        that cost five shifts on this mechanism.

        EDGE-TRIGGERED ON THE REASON, not on every call: the first line dates
        the failure and a per-call repeat would bury it. A different
        precondition failing re-arms the edge, because that is new information.
        The count rides along so "still abstaining" stays legible without a
        line per proposal.
        """
        self._abstain_count += 1
        if reason != self._last_abstain_reason:
            self._last_abstain_reason = reason
            logger.warning(
                "%s ABSTAIN on device %s (#%d): %s. This CANCELS THE SHRINK "
                "FOR THE WHOLE GROUP -- the min-reduce declines whenever any "
                "rank's current row count is not positive, so no rank will "
                "cap this round, and spec item 12 is inert node-wide until "
                "this rank recovers.",
                LOG_PREFIX,
                self._device_index,
                self._abstain_count,
                reason,
            )
        elif self._trace_all:
            logger.warning(
                "%s ABSTAIN on device %s (#%d, unchanged): %s",
                LOG_PREFIX,
                self._device_index,
                self._abstain_count,
                reason,
            )
        # A proposal was NOT made, so the deficit's sign carries no meaning
        # this round. Reset it, or the first real proposal after an abstain
        # can match a stale sign and be swallowed by the edge trigger -- which
        # would restore the very silence this method exists to end.
        self._last_deficit_sign = -1
        return ABSTAIN

    def _clear_abstain(self) -> None:
        """Announce the return, so a WARNING is not this rung's last word."""
        if self._last_abstain_reason is None:
            return
        logger.info(
            "%s device %s is no longer abstaining after %d abstained "
            "proposal(s) (last cause: %s); the group's shrink decision can be "
            "reached again.",
            LOG_PREFIX,
            self._device_index,
            self._abstain_count,
            self._last_abstain_reason,
        )
        self._last_abstain_reason = None

    def _trace_proposal(self, **t) -> None:
        """Say why this rung did or did not propose a shrink.

        WHY THIS EXISTS. Spec item 12's rung declined on every one of ~324
        seam legs across two acceptance runs and emitted NOT ONE LINE while
        doing it, because the only logging on this path was inside the
        ``deficit > 0`` branch -- i.e. only on the path that already works.
        A mechanism whose decline is silent is indistinguishable from a
        mechanism that is never reached, and this chain has now shipped
        several of those.

        The decisive term was invisible for the same reason. Reconstructed
        afterwards from 93 gate lines, the deficit was NEGATIVE on 100% of
        arms, and dropping ``cheap_relief_bytes`` alone flipped every one of
        them positive (+260..+832 MiB): the cheap tier's estimate
        (``reserved - allocated``, which deliberately overstates because it
        counts intra-segment fragmentation ``empty_cache`` cannot return) is
        larger than the gap it is subtracted from. That is the tier law
        working as written -- free money before KV capacity -- but nothing
        said so out loud.

        EDGE-TRIGGERED at INFO on a change in the sign of the deficit, so an
        acceptance run keeps the signal without an env var, and it cannot
        flood: ``propose`` runs on the pp_to_tp leg only, measured at 1.4-3
        calls per minute per rank. ``SGLANG_KV_RELIEF_TRACE=1`` makes every
        call report.
        """
        deficit_mib = t["deficit"] / _MIB
        sign = 1 if t["deficit"] > 0 else 0
        edge = sign != self._last_deficit_sign
        self._last_deficit_sign = sign
        # RETAINED EVEN WHEN NOT LOGGED, so a REFUSAL can print the terms.
        #
        # The edge trigger keeps a steady state quiet, which is right, but a
        # refusal is not an edge -- and at a refusal the silence is exactly
        # the ambiguity this method's own docstring warns about. Measured the
        # hard way on 2026-08-15: the seam was refused by 59 MiB, this rung
        # had emitted nothing for five minutes, and I read that as "the rung
        # was never consulted" and went looking for a missing call. It was
        # consulted every gate and had simply declined quietly.
        self._last_proposal_terms = dict(t)
        if not (edge or self._trace_all):
            return
        logger.info(
            "%s proposal on device %s: rows current=%d floor=%d (max_live=%d "
            "+ admission reserve %d, slack=%d) | need = floor %.0f + delta "
            "%.0f + want %.0f = %.0f MiB "
            "against free %.0f MiB and cheap relief %.0f MiB -> deficit "
            "%+.0f MiB -> %s. %s",
            LOG_PREFIX,
            self._device_index,
            t["current"],
            t["floor_rows"],
            t["max_live"],
            self._admission_reserve_rows,
            max(0, t["current"] - t["floor_rows"]),
            t["floor_bytes"] / _MIB,
            t["delta_bytes"] / _MIB,
            t["want_bytes"] / _MIB,
            (t["floor_bytes"] + t["delta_bytes"] + t["want_bytes"]) / _MIB,
            t["free_now"] / _MIB,
            t["cheap_relief_bytes"] / _MIB,
            deficit_mib,
            (f"SHRINK to {t['desire']}" if t["desire"] < t["current"] else "no change")
            + self._describe_live_split(t["max_live"]),
            (
                t["skipped"]
                if t["skipped"]
                else (
                    "the cheaper tier covers the whole gap, so KV capacity is "
                    "not spent -- this is the tier law, not a broken rung"
                    if t["deficit"] <= 0
                    else "the cheap tier cannot cover it; KV capacity is the funder"
                )
            ),
        )

    def apply_target(self, target_rows: Optional[int]) -> int:
        """Cap and shrink to EXACTLY the row count the group agreed on.

        Deliberately does NOT consult ``self._exhausted``. Exhaustion is
        evidence about THIS rank's arena, and it is a reason to stop ASKING
        (:meth:`propose` honours it) -- never a reason to stay uncapped while
        peers cap, which is the admission disagreement that wedged the group.
        A rank that pays nothing and caps anyway loses admission capacity for
        no bytes; a rank that pays nothing and stays uncapped loses the group.
        """
        self._rebind()
        if target_rows is None:
            return 0
        target = int(target_rows)
        if not self._supported() or self._bytes_per_row <= 0 or target <= 0:
            return 0
        current = self._current_rows()
        if target >= current:
            return 0
        return self._shrink_to(target, current)

    # -- the provider ----------------------------------------------------

    def free_up_to(self, nbytes: int) -> int:
        """Rank-local relief. NOT registered with the corridor guard.

        Kept as the single-rank primitive the unit tests pin the watermark,
        granularity and accounting laws against. In a distributed instance the
        target comes from :func:`collective_kv_target` instead, because a
        capacity may not be decided locally.
        """
        self._rebind()
        if not self._supported() or self._bytes_per_row <= 0 or self._exhausted:
            return 0
        max_live = self._max_live_row()
        if max_live < 0:
            return 0
        current = self._current_rows()
        page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
        floor, _evictable = self._evict_floor_rows(max_live)
        rows_wanted = int(math.ceil(max(0, int(nbytes)) / self._bytes_per_row))
        # RELEASE IS EXTENT-GRANULAR, PER BUFFER, AND THAT IS COARSE.
        #
        # The arena holds each of the 2*layer_num buffers at its own offset and
        # ``decommit_range`` frees only extents lying WHOLLY above the keep
        # point. A shrink is therefore split across every buffer, so a request
        # for N bytes moves only N/n_buffers in each one -- and if that is less
        # than one commit chunk, NOTHING is released anywhere.
        #
        # Measured 2026-08-11 with a 256 MiB chunk: a 78262-row shrink asked
        # about 40 MiB of each of ~28 buffers, cleared no extent in any of
        # them, and returned 0 while the log looked like a working rung.
        #
        # So round the ask UP to the granularity instead of attempting a
        # no-op. Over-delivering is not a failure -- the guard re-probes the
        # driver and stops asking once the target is met -- whereas
        # under-delivering is silent and costs a wasted cap.
        rows_wanted = max(rows_wanted, self._min_release_rows())
        target = max(floor, current - rows_wanted)
        target = int(math.ceil(target / page) * page)
        if target >= current:
            return 0
        return self._shrink_to(target, current)

    def _shrink_to(self, target: int, current: int) -> int:
        """Cap to ``target`` rows, unmap above it, and report DRIVER bytes."""
        before = self._free_bytes()
        if self._rows_at_boot is None:
            self._rows_at_boot = current
        # #662: EVICT FIRST, AND ONLY THEN CAP. The rows this cap is about
        # to withhold must be unoccupied by the time it withholds them, and
        # on a corridor-filled pool they are occupied by recomputable prefix
        # cache. Evicting here rather than in the caller keeps the whole
        # order -- evict, cap, unmap -- in one place, because it is the
        # ORDER that is the safety property and splitting it across two
        # modules is how it would come apart.
        self._lower_watermark_to(target)
        # #717: THE FLOOR FOLLOWS COMPLETION, NOT INTENTION.
        #
        # ``target`` was priced on the assumption that the eviction above
        # would bring the high-water mark under it. That assumption is not
        # self-enforcing, and when it failed the pool capped below rows that
        # were still mapped: 69,054 rows of backing under a highest live row
        # of 233,289, and the next access to a row above the cap was an
        # illegal address (the crash that reverted c4e557963e).
        #
        # It fails for ordinary reasons, not exotic ones.
        # ``_lower_watermark_to`` REFUSES and returns 0 whenever
        # ``_resident_ceiling()`` is negative, and this call site used to
        # discard that return value; ``evict_rows_above`` likewise refuses
        # outright when a resident request pins a row above the target; and a
        # pass over the tree can free less than asked. In every one of those
        # cases the intention was recorded and the completion was not.
        #
        # "No running requests" is NOT "no live rows" -- the radix tree's
        # cached rows are live and addressable -- so the only statistic this
        # may trust is the live set AFTER the eviction ran. Re-measuring
        # costs one enumeration on the shrink path, which is the seam, and it
        # is the price of the cap being a fact rather than a hope.
        post_live = self._max_live_row()
        if post_live < 0:
            # An unknown live set is not an empty one, and this is the last
            # point at which refusing is still free.
            logger.warning(
                "%s ABANDONED the shrink to %d rows: the live set could not "
                "be re-read after the eviction, so no cap can be shown safe.",
                LOG_PREFIX,
                target,
            )
            return 0
        safe_floor = self._floor_rows(post_live)
        if target < safe_floor:
            page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
            raised = int(math.ceil(safe_floor / page) * page)
            logger.warning(
                "%s the eviction did not deliver the mark this shrink was "
                "priced against: target %d rows sits below the highest live "
                "row %d, whose floor is %d. RAISING the cap to %d -- capping "
                "as asked would unmap rows that are still addressable. The "
                "rung wins less than it priced; that is the correct outcome, "
                "not a failure to work around.",
                LOG_PREFIX,
                target,
                post_live,
                safe_floor,
                raised,
            )
            target = raised
            if target >= current:
                # Nothing left to win. Capping at `current` would spend the
                # seam's one attempt for zero bytes.
                return 0
        # ORDER IS THE SAFETY PROPERTY: cap FIRST, unmap SECOND. Reversed,
        # there is a window in which the allocator may hand out an id whose
        # pages have already gone back to the driver.
        self._cap.engage(target)
        try:
            claimed = int(self._pool.runtime_set_backing_rows(target))
        except Exception as e:
            logger.error(
                "%s runtime_set_backing_rows(%d) failed: %s; releasing the cap",
                LOG_PREFIX,
                target,
                e,
            )
            self._cap.release()
            return 0
        measured = max(0, self._free_bytes() - before)
        if measured <= 0:
            # A SHRINK THAT FREED NOTHING MUST NOT BE UNDONE HERE, and getting
            # this wrong cost 2.5 GiB per gate arm on metal (2026-08-11).
            #
            # ``recover()`` GROWS the pool, and growing calls ``finalize``,
            # which calls ``cuMemCreate``. Undoing a failed shrink therefore
            # ALLOCATES -- inside a gate that armed precisely because memory
            # was short. Measured: free 3040 -> 460 MiB across one refusal
            # whose own detail line claimed it had "reclaimed 428 MiB", and
            # eventually ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` on
            # the way back. The relief provider was the biggest consumer on
            # the card.
            #
            # So: KEEP THE CAP (it is free, and it is the invariant that
            # nothing is handed out above the watermark), do not re-commit,
            # and stop trying. Recovery happens on the tp->pp leg, at an
            # idle boundary, where an allocation is affordable and survivable.
            #
            # EXHAUSTION IS ONLY EVIDENCE WHEN THE ASK WAS BIG ENOUGH TO PAY.
            # Since the target became collective, a rank can be handed a
            # target shallower than ITS OWN release granularity -- one commit
            # chunk in every one of its buffers, and the three PP stages here
            # hold 28 / 20 / 16 of them. Releasing nothing under such a target
            # says nothing about the arena; it says the group agreed on a
            # number smaller than this rank can act on. Marking it exhausted
            # would silence a voice that has real bytes to offer at a deeper
            # target -- a slow leak of the group's ambition, and invisible,
            # because every log line would still look correct.
            asked = int(current) - int(target)
            granularity = self._min_release_rows()
            if granularity <= 0 or asked >= granularity:
                self._mark_exhausted(target)
            logger.warning(
                "%s shrink to %d rows reported %.0f MiB but the driver's free "
                "column did not move, so this pool cannot pay: the arena has "
                "no commit chunk, or its handles are retained "
                "(SGLANG_FLIP_SEAM_RETAIN_HANDLES), and unmapping without "
                "releasing yields address space rather than memory. The cap "
                "STAYS ON -- undoing it here would re-commit pages inside a "
                "gate that armed because memory was short. No further attempt "
                "will be made until the next recovery.",
                LOG_PREFIX,
                target,
                claimed / (1024 * 1024),
            )
            return 0
        self.shrink_count += 1
        self.released_total += measured
        logger.info(
            "%s released %.0f MiB by backing %d rows instead of %d "
            "(highest live row %d, pool claimed %.0f MiB, %d ids withheld "
            "from the allocator)",
            LOG_PREFIX,
            measured / (1024 * 1024),
            target,
            current,
            self._max_live_row(),
            claimed / (1024 * 1024),
            self._cap.withheld,
        )
        return measured

    def recover(self) -> int:
        """Re-back the pool toward its boot reservation, as far as the
        corridor law allows, and lift the cap to whatever it reached.

        RECOVERY IS AN ALLOCATION, AND IT MUST OBEY THE SAME LAW THE SHRINK
        WAS SERVING. The first metal boot of this rung recovered straight to
        the boot rows with no reference to free memory, on the leg where the
        PP pool becomes active again, and drove rank 1 to **6 MiB free** --
        with a ``cuMemCreate`` OOM on the way. The design had called that leg
        "an idle boundary, where an allocation is affordable"; measured, it is
        not, because the pool being re-committed is exactly as large as the
        relief that was taken.

        So the grow is BOUNDED by this card's distance from the corridor law
        (never by the gate's proof-time arming floor, which would cripple
        recovery for an instrument). What it cannot re-commit stays capped:
        that is an admission-capacity loss, which is recoverable on any later
        leg, rather than a breach or a fault, which are not.

        Restore order is the mirror of the shrink: pages FIRST, cap SECOND.
        Lifting the cap before the memory exists would re-admit ids that are
        still unmapped, which is the very fault the cap prevents.
        """
        self._rebind()
        if self._rows_at_boot is None:
            return 0
        boot_rows = int(self._rows_at_boot)
        was = self._current_rows()
        rows = boot_rows
        if self._supported() and was < boot_rows:
            if self._bytes_per_row > 0:
                headroom = self._free_bytes() - self._law_floor_bytes
                affordable = int(headroom // self._bytes_per_row)
                rows = min(boot_rows, was + max(0, affordable))
                page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
                rows = int(rows // page * page)
            # #684: CLAMP TO WHAT THE ACTUATOR CAN ACCEPT, NOT TO WHAT WE
            # REMEMBER. The reservation is fixed at construction from the
            # pool's size AT THAT MOMENT and never moves again, while `size`
            # itself is mutable -- the #330 dial writes it on every step. So a
            # target derived from a remembered row count can sit above the
            # ceiling, and `_check_final` refuses it identically every time:
            # measured 59 times in 20 minutes on three ranks, 02:15:24 to
            # 02:35:26, `recovery to 270646 rows failed: ... reserved=190596`.
            #
            # Recovery is what LIFTS the cap, so those 59 refusals meant the
            # cap never lifted, the pool stayed shrunk, and every later
            # `free_up_to` found the backing already at its target and claimed
            # 0 MiB -- which the shrink path then reported as an exhausted
            # ARENA. One unsatisfiable number, and the corridor guard's only
            # rung above the allocator cache was dead for the whole boot.
            #
            # DERIVATION IS NOT TRUSTED; THE BOUND IS ASKED. This is the same
            # correction as #681 and #682: validate against what the actuator
            # can pay rather than against the count that proposed it. The
            # clamp is therefore deliberately not conditional on knowing WHY
            # the remembered number is stale.
            ceiling = self._reserved_rows()
            if ceiling is not None and rows > ceiling:
                logger.warning(
                    "%s recovery target %d rows exceeds the pool's immutable "
                    "reservation of %d; clamping and correcting the remembered "
                    "boot row count, which was stale (#684).",
                    LOG_PREFIX,
                    rows,
                    ceiling,
                )
                rows = int(ceiling // page * page) if page > 0 else int(ceiling)
                # AND CORRECT THE MEMORY, or the clamp only converts a loud
                # failure into a quiet one: `_rows_at_boot` would still name an
                # impossible level and every later recovery would re-clamp to
                # the same place while believing it had further to go. The
                # pool's ceiling is what it can hold, so that is what "fully
                # recovered" means for this arena.
                self._rows_at_boot = max(int(rows), int(was))
                boot_rows = self._rows_at_boot
            if rows <= was:
                logger.warning(
                    "%s recovery deferred: %d MiB free leaves nothing above "
                    "the %d MiB corridor law to re-commit with, so the pool "
                    "stays at %d of %d rows. Admission capacity is reduced "
                    "until a later leg -- which is a capacity loss, never a "
                    "breach.",
                    LOG_PREFIX,
                    self._free_bytes() // (1024 * 1024),
                    self._law_floor_bytes // (1024 * 1024),
                    was,
                    boot_rows,
                )
                return 0
            try:
                self._pool.runtime_set_backing_rows(rows)
            except Exception as e:
                # Growing commits pages, so it can fail for want of memory.
                # THE CAP STAYS ON when it does: the invariant is "nothing is
                # handed out above what is backed", and a failed grow leaves
                # the watermark exactly where it was.
                logger.error(
                    "%s recovery to %d rows failed: %s. The cap stays engaged, "
                    "so admission capacity remains reduced -- a capacity loss, "
                    "never a fault.",
                    LOG_PREFIX,
                    rows,
                    e,
                )
                return 0
        now = self._current_rows()
        # Pages first, cap second -- and the cap comes back at the level the
        # pages actually reached, not at the level they were aiming for.
        self._cap.release()
        if now < boot_rows:
            self._cap.engage(now)
            logger.info(
                "%s recovered to %d of %d rows (corridor-bounded); the cap "
                "stays at that level and the boot reservation is remembered "
                "for a later leg",
                LOG_PREFIX,
                now,
                boot_rows,
            )
        else:
            self._rows_at_boot = None
            self._exhausted_at_rows = None
        self.recover_count += 1
        return max(0, now - was)

    # -- the cap agreement (#656 C22) -------------------------------------

    def _reservation_rows(self) -> int:
        """Rows the allocator's id space spans -- the reservation, not the
        backing. This is what an UNCAPPED allocator will hand out.

        READ FROM THE ID-SPACE OWNER, never from whichever layout the backing
        calls are currently pointed at. This number feeds ``exposed_rows``,
        which feeds the collective cap agreement, and the two layouts have
        different row counts -- letting it follow a rebind would make the
        group's agreed id space depend on which phase each rank happened to be
        in, which is the capacity desync that wedged this instance once
        already (HANDOFF_675 1a)."""
        return int(getattr(self._id_space_pool, "size", 0) or self._current_rows())

    def exposed_rows(self) -> int:
        """The highest row id this rank's allocator may hand out.

        The cap when one is engaged, the whole reservation otherwise. This is
        the quantity that has to be identical across the group: it decides
        ``available_size()``, which feeds ADMISSION, and it decides which ids
        the flip's live-slot enumeration can encounter, which decides the
        length of the payload each rank frames.
        """
        if self._cap.engaged and self._cap.cap is not None:
            return int(self._cap.cap)
        return self._reservation_rows()

    def backed_rows(self) -> int:
        """Rows this rank has PHYSICALLY BACKED, as a public reading.

        #656 C22-d: the seam's live-slot agreement needs it. A row id above
        this number is not mapped on this rank, so a framed set containing one
        would have the mover read unmapped memory -- a
        ``cudaErrorIllegalAddress`` that kills every rank rather than raising.
        The agreement therefore bounds the group's union by the MIN of this
        value across ranks, and that minimum has to come from a public reading
        rather than from a private one the caller reaches around for.
        """
        self._rebind()
        return int(self._current_rows())

    def level_recovery_to(self, target: int) -> int:
        """#656 C22-e: cap this rank's ID SPACE to the group's, after a grow.

        :meth:`reconcile_to` with one addition that matters: it REMEMBERS the
        reservation. ``reconcile_to`` clears ``_rows_at_boot`` whenever the
        level it lands on reaches the ceiling it knows about, and a rank that
        recovered fully has already cleared it -- so levelling such a rank down
        with ``reconcile_to`` alone would cap its allocator AND destroy the
        only record that it owes itself a recovery. :meth:`recover` returns 0
        immediately when ``_rows_at_boot`` is None, so that rank would never
        climb back and the level would be a ratchet. It is explicitly not one:
        the level rises again as soon as the poorest rank can fund it.

        Returns the change in this rank's exposed level (signed).
        """
        target = int(target)
        if self._rows_at_boot is None and target < self._reservation_rows():
            # Remember what this rank is entitled to before capping below it.
            self._rows_at_boot = self._reservation_rows()
            self._exhausted_at_rows = None
        return int(self.reconcile_to(target))

    def normalize_free_lists(self) -> None:
        """Put this rank's free lists in ascending id order, unconditionally.

        #656 C22-d, and it is the SOURCE half of the live-slot divergence the
        agreement below repairs after the fact.

        ``reconcile_to`` already ends with this sort, but it only ever runs
        when :func:`collective_cap_target` returns a level -- and that function
        returns ``None`` precisely when the group's exposed counts already
        AGREE. So the one state in which nothing normalises the order is the
        state in which the counts are equal, which is exactly the state the
        metal wedged in: rank PP1 had taken a corridor-bounded ``recover()``
        (``KvRowCap.release`` SORTS, ``engage`` preserves eviction order) while
        its peers, which never shrank, had never sorted at all. Identical
        membership, identical counts, different ORDER -- and the allocator
        takes from the FRONT, so the next request got a different physical row
        id on PP1 than on PP0/PP2. From there the live slot sets part company
        with nothing in the pool census to show for it.

        Called on every rank on every seam round, from the one point every
        rank reaches unconditionally. It is a pure ordering of ids: no bytes
        move, no capacity changes, and doing it when it was already sorted is
        free.
        """
        self._cap.sort_free_lists()

    def cap_proposal(self):
        """This rank's four-field proposal for the group's exposed row level.

        PURE: reads free memory, the backing and the live set, and computes.
        It is on the seam round's unconditional path, so it must not change
        residency and must not raise -- an unreadable live set abstains, which
        makes the whole group decline.
        """
        max_live = self._max_live_row()
        if max_live < 0:
            return CAP_ABSTAIN
        backed = self._current_rows()
        if backed <= 0:
            return CAP_ABSTAIN
        # WHAT THIS RANK CAN EXPOSE IS WHAT IS ALREADY BACKED. NOT ONE ROW
        # MORE, and the missing term is the one that had to be measured to be
        # believed. The first metal boot of this agreement proposed
        # ``backed + (free - law) / bytes_per_row`` -- what ``recover`` would
        # be allowed to commit -- and the levelling then tried to GROW on the
        # pp->tp leg, i.e. to hand back the very rows the collective shrink
        # had just taken to fund the seam. Measured 2026-08-13 15:40:23Z:
        # ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY`` on all three ranks,
        # rank 0 driven to 3 MiB free (1021 MiB below the law), the seam then
        # unfundable, and the instance parked in TP with a 9-token prefill it
        # could not run.
        #
        # So the agreement is STRICTLY NON-ALLOCATING. Growing has exactly one
        # owner -- ``recover``, on the leg the pool becomes active again, with
        # its own corridor bound -- and this decides only which of the backed
        # rows the group agrees to hand out. The two never fight, and the
        # level still rises: a rank that recovers raises its own proposal, and
        # its peers follow by RELEASING a cap over pages they never gave up.
        capable = backed
        page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
        capable = int(capable // page * page)
        if capable <= 0:
            return CAP_ABSTAIN
        exposed = self.exposed_rows()
        floor = self._floor_rows(max_live)
        return (int(capable), -int(floor), int(exposed), -int(exposed))

    def reconcile_to(self, target: int) -> int:
        """Bring this rank's exposed row level to exactly ``target``.

        IT COMMITS NOTHING, EVER, and it releases nothing either. This is an
        ID decision: the pages stay exactly as they are and only the
        allocator's free list moves. Growing the backing has ONE owner --
        :meth:`recover`, on the leg the pool becomes active again, with its
        own corridor bound -- and an agreement that also grew would hand back
        the rows the collective shrink had just taken to fund the seam. That
        is not hypothetical; it OOM'd all three ranks on the first metal boot
        of this mechanism (see :meth:`cap_proposal`).

        ``target`` is the group MIN of what every rank has BACKED, so it is
        never above this rank's own backing; the ``min`` below is a
        belt-and-braces reading rather than a clamp that does work.

        Levelling a rank DOWN costs it no real capacity: under pure PP every
        rank holds the same token rows, so rows above the group minimum could
        never have been admitted against anyway. What it buys is that the
        ranks cannot disagree about ``available_size()`` or about which ids a
        live-slot enumeration may encounter.

        Returns the change in this rank's exposed level (signed).
        """
        target = int(target)
        before = self.exposed_rows()
        backed = self._current_rows()
        # NO EARLY RETURN FOR "I AM ALREADY THERE". The caller only reaches
        # this when the GROUP is not level (``collective_cap_target`` decides
        # that from the reduced view), and every rank must then run the same
        # release-and-re-engage so that every rank's free list ends in the
        # same ORDER. Skipping here is what left one rank sorted and another
        # in eviction order on boot_v2, and a different order hands the next
        # request different row ids -- a divergent live slot set, and a
        # divergent wire frame, with the pool census identical on every rank.
        level = min(target, backed)
        ceiling = (
            int(self._rows_at_boot)
            if self._rows_at_boot is not None
            else self._reservation_rows()
        )
        self._cap.release()
        if level < self._reservation_rows():
            self._cap.engage(level)
        # AND MAKE THE ORDER A FUNCTION OF MEMBERSHIP ALONE.
        #
        # ``release`` sorts only when it actually had ids withheld, and
        # ``engage``'s filter preserves whatever order it found, so after the
        # pair above a rank that HAD a cap is sorted and a rank that did not
        # is still in eviction order -- with identical membership. The
        # allocator takes from the FRONT, so those two ranks hand the next
        # request different row ids, and the live slot sets part company with
        # nothing in the pool census to show for it. That is boot_v2's second
        # divergence, and it is why the sort is explicit and unconditional
        # here rather than a side effect of one branch.
        self._cap.sort_free_lists()
        if self._rows_at_boot is not None and level >= ceiling:
            self._rows_at_boot = None
            self._exhausted_at_rows = None
        after = self.exposed_rows()
        if after != before:
            logger.info(
                "%s cap agreement: exposed rows %d -> %d (group level %d, "
                "backed %d). The group's ranks hold the same token rows, so "
                "the level every rank can fund is the level the group has",
                LOG_PREFIX,
                before,
                after,
                target,
                backed,
            )
        return int(after - before)


def row_geometry(pool: Any):
    """``(bytes_per_row, n_buffers)`` for the pool's arena, or ``(0, 0)``.

    Both numbers come from the arena's own buffer descriptors, because that is
    the geometry ``shrink`` actually prices against. The buffer COUNT matters
    as much as the row size: release is extent-granular per buffer, so the
    smallest release that can return anything is one commit chunk times the
    number of buffers.
    """
    return _bytes_per_row(pool), _buffer_count(pool)


def _buffer_count(pool: Any) -> int:
    full = getattr(pool, "full_kv_pool", pool)
    owner = getattr(full, "_post_capture_owner", None)
    specs = getattr(owner, "_specs", None) if owner is not None else None
    return len(specs) if specs else 0


def _bytes_per_row(pool: Any) -> int:
    """Bytes of physical backing one KV row costs across every buffer.

    Derived from the arena's own buffer descriptors when they exist, because
    that is the geometry ``shrink`` actually prices against -- K and V, every
    layer, whatever the layout's rows-per-token happens to be. Anything
    reconstructed from head counts would be a second source of truth for a
    number that decides how much memory gets unmapped.

    Returns 0 when the geometry cannot be read, which makes the provider inert
    rather than wrong: a bad row size would shrink the pool by the wrong
    amount in a direction that faults.
    """
    full = getattr(pool, "full_kv_pool", pool)
    owner = getattr(full, "_post_capture_owner", None)
    specs = getattr(owner, "_specs", None) if owner is not None else None
    if not specs:
        return 0
    total = 0
    for spec in specs:
        desc = getattr(spec, "desc", None)
        if desc is None:
            return 0
        row_bytes = int(getattr(desc, "row_bytes", 0))
        per_row = max(1, int(getattr(desc, "tokens_per_row", 1) or 1))
        total += row_bytes // per_row
    return int(total)


def rung_can_pay(scheduler: Any) -> bool:
    """Will this boot have a KV rung able to return bytes at the seam?

    THE SAME DISQUALIFIERS :func:`kv_backing_provider` APPLIES, asked without
    building anything. The seam reserve has to price the rung while sizing the
    pool, and at that point the relief object does not exist yet -- it is
    installed at the first corridor gate, which is later than both the pool
    sizing and the seam measurement. Re-deriving the conditions there would be
    a second source of truth for "can this rung pay", and the two would drift;
    this is the one place they are written.

    A predicate, never an amount. What the rung may cover is decided by the
    caller and bounded there.
    """
    if os.environ.get("SGLANG_KV_BACKING_RELIEF", "1") not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return False
    get_kvcache = getattr(allocator, "get_kvcache", None)
    pool = get_kvcache() if callable(get_kvcache) else None
    if pool is None or not callable(getattr(pool, "runtime_set_backing_rows", None)):
        return False
    if not bool(getattr(pool, "supports_backing_spans", False)):
        # A chunkless arena cannot return anything to the driver, so a pool
        # sized as if it could would be sized on a promise nothing keeps.
        return False
    row_bytes, _buffers = row_geometry(pool)
    return row_bytes > 0


def kv_backing_provider(
    scheduler: Any,
    *,
    device_index: int,
    probe: Optional[Callable[[], int]] = None,
    law_floor_bytes: int = 1024 * 1024 * 1024,
) -> Optional[KvBackingRelief]:
    """Build the relief for a scheduler's KV pool, or None when unavailable.

    Returns None rather than an inert callable when the pool is not on a VA
    reservation: a provider that is registered but can never pay makes the
    guard's spend order read as if a tier were funded when it is not, and this
    chain has shipped three of those.
    """
    # ON BY DEFAULT SINCE THE TARGET BECAME COLLECTIVE (2026-08-11).
    #
    # It was opt-in for one shift, and the reason was a wedge rather than
    # caution: the cap changes ``available_size()``, which feeds ADMISSION,
    # and each rank used to size its own shrink from its own free memory and
    # its own live set. Three ranks capped to 449039 / 451037 / 175225 /
    # 145734 rows in one boot, the group stopped agreeing about how much work
    # it could take, and the scheduler stopped heartbeating while every rank
    # was alive and logging.
    #
    # The target is now agreed by one MIN all-reduce at a point every rank
    # reaches unconditionally (``collective_kv_backing_relief``), and the same
    # uniformity was then measured on metal: 347161 rows on all three ranks,
    # then 94017 on all three, health 200 throughout, flips continuing in both
    # directions. So the switch turns OFF a rung that works rather than ON one
    # that might not, which is the direction an escape hatch should face.
    if os.environ.get("SGLANG_KV_BACKING_RELIEF", "1") not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        logger.warning(
            "%s relief is DISABLED by SGLANG_KV_BACKING_RELIEF. Spec item 12's "
            "device half is off: the KV pool keeps its full backing in both "
            "phases and the pp->tp leg loses its only funder.",
            LOG_PREFIX,
        )
        return None
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if allocator is None:
        return None
    get_kvcache = getattr(allocator, "get_kvcache", None)
    pool = get_kvcache() if callable(get_kvcache) else None
    if pool is None or not callable(getattr(pool, "runtime_set_backing_rows", None)):
        return None
    if not bool(getattr(pool, "supports_backing_spans", False)):
        # A CHUNKLESS ARENA CANNOT PAY, AND TRYING COSTS REAL MEMORY.
        #
        # Without a commit chunk the arena holds one extent per buffer, and
        # ``decommit_range`` releases only extents lying WHOLLY above the keep
        # point -- so a shrink to any watermark inside that extent releases
        # exactly zero while still lowering ``pool.size``. The pool then looks
        # smaller than its backing, and the way back re-commits.
        #
        # Measured on metal 2026-08-11: registered against a chunkless pool,
        # this provider drove device 0 from 3040 MiB free to 460 and ended in
        # ``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY``. An inert provider
        # would have been merely useless; this one was the biggest consumer on
        # the card. So the missing chunk is a DISQUALIFIER, not a warning.
        logger.warning(
            "%s the KV pool's arena has NO COMMIT CHUNK, so a partial release "
            "cannot return anything to the driver. Backing relief is NOT "
            "registered. Boot with SGLANG_FLIP_SEAM_CHUNK_MIB set to enable "
            "chunked commits and this rung with it.",
            LOG_PREFIX,
        )
        return None
    row_bytes, n_buffers = row_geometry(pool)
    if row_bytes <= 0:
        logger.warning(
            "%s could not read the pool's row geometry; KV backing relief is "
            "NOT registered (an inert provider would misreport the ladder as "
            "funded)",
            LOG_PREFIX,
        )
        return None
    from sglang.srt.managers.phase_flip_runtime import build_flip_live_slots_fn

    live_fn = build_flip_live_slots_fn(scheduler)

    def _flip_armed() -> bool:
        """#744 line 2. Unreadable is treated as ARMED, never as idle."""
        rt = getattr(scheduler, "phase_flip_runtime", None)
        if rt is None:
            return False
        try:
            return bool(rt.is_armed())
        except Exception:  # noqa: BLE001 - an unreadable flip is not an idle one
            return True

    def _flip_pending():
        """#744 line 1: ``(rows, max_row_id)`` the flip has parked.

        Consulted only while a flip is armed, which is what makes the sticky
        value on ``live_fn`` safe: outside a flip this answers "nothing
        parked" unconditionally, so the rung stays fully live and #688's
        funding path is untouched. While armed and with no enumeration on
        record yet, the honest answer is UNKNOWN -- which blocks.
        """
        if not _flip_armed():
            return (0, -1)
        return getattr(live_fn, "last_req_extent", None) or (-1, -1)

    return KvBackingRelief(
        pool,
        allocator,
        live_slots_fn=live_fn,
        flip_armed_fn=_flip_armed,
        flip_pending_fn=_flip_pending,
        bytes_per_row=row_bytes,
        probe=probe,
        device_index=device_index,
        buffers=n_buffers,
        law_floor_bytes=law_floor_bytes,
        admission_reserve_rows=_admission_reserve_rows(scheduler),
        # #662: RESOLVED PER CALL, NEVER CAPTURED. The tree cache object is
        # replaced on a flush (flush_cache builds a new one), and a rung
        # holding the old reference would evict into a tree the scheduler no
        # longer reads -- freeing rows the allocator still believes are
        # cached. Reading it through the scheduler each time is the only
        # form that cannot go stale.
        tree_cache_fn=lambda: getattr(scheduler, "tree_cache", None),
        # #662-F4: and the POOL is resolved per call for the same reason, with
        # a sharper edge. The scheduler's pool is the PP layout's; during the
        # TP phase the seam has released it and it holds no pages at all. A
        # rung captured on it can only ever fund the pp_to_tp leg.
        pool_fn=lambda: _active_layout_pool(scheduler, pool),
    )


def _active_layout_pool(scheduler: Any, fallback: Any):
    """The KV pool of the layout that is RESIDENT right now.

    The flip's two layouts are two pools with two arenas and only one is
    backed at a time. ``scheduler.phase_flip_active_stack`` says which, and it
    is set at the cutover, so it is already correct by the time the next gate
    runs.

    Falls back to the scheduler's own pool whenever the answer is not
    unambiguous -- no stacks installed, an unrecognised phase, a missing
    worker. That reproduces the previous behaviour exactly, which is the right
    direction for a resolution that decides where memory gets unmapped.
    """
    stacks = getattr(scheduler, "phase_flip_stacks", None)
    if stacks is None:
        return fallback
    phase = getattr(scheduler, "phase_flip_active_stack", None)
    if str(phase) != "tp":
        # PP resident (or unknown): the scheduler's own pool IS that layout's.
        return fallback
    worker = getattr(stacks, "tp_worker", None)
    runner = getattr(worker, "model_runner", None) if worker is not None else None
    tp_pool = getattr(runner, "token_to_kv_pool", None) if runner is not None else None
    if tp_pool is None or not callable(
        getattr(tp_pool, "runtime_set_backing_rows", None)
    ):
        return fallback
    return tp_pool


def _admission_reserve_rows(scheduler: Any) -> int:
    """Rows the rung keeps allocatable, from the scheduler's own admission size.

    THE RESERVE IS NOT A SAFETY FACTOR, it is the largest single admission the
    scheduler can attempt while this rung is the thing that shrank the pool. On
    the shipped configuration that is ``chunked_prefill_size`` (512), which is
    the number the failure quoted back: "Try to allocate 512 tokens".

    Derived rather than constant because the two move together -- a boot with a
    larger prefill chunk needs a larger reserve to make the same progress -- and
    a constant sized once against one configuration is exactly the shape this
    corpus keeps having to retract. ``SGLANG_KV_ADMISSION_RESERVE_ROWS``
    overrides it, and 0 restores the pre-C20-residual floor.
    """
    raw = os.environ.get(KV_ADMISSION_RESERVE_ENV, "")
    if raw.strip():
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning(
                "%s %s=%r is not an integer; using the derived reserve",
                LOG_PREFIX,
                KV_ADMISSION_RESERVE_ENV,
                raw,
            )
    args = getattr(scheduler, "server_args", None)
    for holder, name in (
        (scheduler, "chunked_prefill_size"),
        (args, "chunked_prefill_size"),
    ):
        size = getattr(holder, name, None) if holder is not None else None
        try:
            size = int(size)
        except (TypeError, ValueError):
            continue
        # A negative or zero chunk means "unchunked", which says nothing about
        # the reserve; fall through to the default rather than reserve nothing.
        if size > 0:
            return size
    return DEFAULT_ADMISSION_RESERVE_ROWS
