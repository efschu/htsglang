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

    "ES GIBT KEIN FESTES MAX KV: KV selbst ist Spill-Klasse in den System-RAM
     ... im VRAM liegt zu jedem Zeitpunkt GENAU das, was gerade dort liegen
     muss, der Rest im System-RAM."

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
from typing import Any, Callable, Optional

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
                    back = self._withheld.to(pages.device, pages.dtype)
                    merged = torch.cat((pages, back))
                    # Sorted, because the allocator takes from the FRONT and
                    # the high-water mark this rung prices itself against only
                    # tracks occupancy while low ids are reused first.
                    setattr(self._alloc, name, torch.sort(merged).values)
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
        """
        import torch

        for name in ("free_pages", "release_pages"):
            pages = getattr(self._alloc, name, None)
            if pages is None or pages.numel() < 2:
                continue
            setattr(self._alloc, name, torch.sort(pages).values)

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
    ) -> None:
        self._pool = pool
        self._alloc = allocator
        self._live_slots_fn = live_slots_fn
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
        #: Set when a shrink returned no driver bytes. One such attempt is
        #: evidence about the ARENA, not about this moment, so repeating it
        #: can only cost time and risk -- and on metal it cost 2.5 GiB.
        self._exhausted = False
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
        """
        rows = getattr(self._pool, "full_pool_backed_rows", None)
        if rows is None:
            return int(getattr(self._pool, "size", 0))
        return int(rows)

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
        if live is None or int(getattr(live, "numel", lambda: 0)()) == 0:
            return 0
        return int(live.max())

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

    def fundable_bytes(self) -> int:
        """Bytes this rank could return WITHOUT crossing its admission floor.

        The bound a caller needs to ask this rung for a DISCRETIONARY amount
        honestly. A rank that cannot take part answers 0, which reads as "ask
        me for nothing extra" -- the safe direction, since the mandatory part
        of an ask is never bounded by this.

        Pure: it reads the live set and the backing and computes. It is on the
        gate's unconditional path, so it must not touch residency.
        """
        if not self._supported() or self._bytes_per_row <= 0 or self._exhausted:
            return 0
        max_live = self._max_live_row()
        if max_live < 0:
            return 0
        current = self._current_rows()
        if current <= 0:
            return 0
        return max(0, current - self._floor_rows(max_live)) * self._bytes_per_row

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
        floor_rows = self._floor_rows(max_live)
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
        if self._exhausted:
            skipped = (
                "this rank's arena is EXHAUSTED (a previous shrink returned no "
                "driver bytes), so it stops asking; the deficit is not computed"
            )
        elif floor_rows >= current:
            skipped = (
                f"no slack above the live set: floor_rows {floor_rows} >= "
                f"current {current}, so there is nothing this rung may give up"
            )
        if floor_rows < current and not self._exhausted:
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
        if not self._supported() or self._bytes_per_row <= 0 or self._exhausted:
            return 0
        max_live = self._max_live_row()
        if max_live < 0:
            return 0
        current = self._current_rows()
        page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
        floor = self._floor_rows(max_live)
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
                self._exhausted = True
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
            self._exhausted = False
        self.recover_count += 1
        return max(0, now - was)

    # -- the cap agreement (#656 C22) -------------------------------------

    def _reservation_rows(self) -> int:
        """Rows the allocator's id space spans -- the reservation, not the
        backing. This is what an UNCAPPED allocator will hand out."""
        return int(getattr(self._pool, "size", 0) or self._current_rows())

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
        return int(self._current_rows())

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
            self._exhausted = False
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

    return KvBackingRelief(
        pool,
        allocator,
        live_slots_fn=build_flip_live_slots_fn(scheduler),
        bytes_per_row=row_bytes,
        probe=probe,
        device_index=device_index,
        buffers=n_buffers,
        law_floor_bytes=law_floor_bytes,
        admission_reserve_rows=_admission_reserve_rows(scheduler),
    )


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
