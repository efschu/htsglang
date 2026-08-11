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

#: A desire no reduction can lower: the neutral element of an element-wise MIN.
_UNBOUNDED_ROWS = 1 << 40

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
                register(lambda _idx: self._apply(), self._apply)
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

    def _apply(self) -> None:
        """Move ids above the cap out of every free list, idempotently."""
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

    # -- plumbing --------------------------------------------------------

    def _free_bytes(self) -> int:
        if self._probe is not None:
            return int(self._probe())
        import torch

        return int(torch.cuda.mem_get_info(self._device_index)[0])

    def _supported(self) -> bool:
        return callable(getattr(self._pool, "runtime_set_backing_rows", None))

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

    def _max_live_row(self) -> int:
        try:
            live = self._live_slots_fn()
        except Exception as e:
            # An unknown live set is not an empty one. Refusing to shrink is
            # the only safe reading, because the number this decides is the
            # point below which memory gets unmapped.
            logger.warning("%s live-set probe failed: %s", LOG_PREFIX, e)
            return -1
        if live is None or int(getattr(live, "numel", lambda: 0)()) == 0:
            return 0
        return int(live.max())

    def _floor_rows(self, max_live: int) -> int:
        """The lowest row count this rank may be capped to, in rows.

        The shrink precondition stated as a number: every row at or below the
        high-water mark must stay backed, plus one page of slack so the very
        next allocation does not immediately re-arm.
        """
        page = max(1, int(getattr(self._pool, "page_size", 1) or 1))
        floor = max(page, int(max_live) + 1 + self._margin_rows)
        return int(math.ceil(floor / page) * page)

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
        if not self._supported() or self._bytes_per_row <= 0:
            return ABSTAIN
        current = int(getattr(self._pool, "size", 0))
        if current <= 0:
            return ABSTAIN
        max_live = self._max_live_row()
        if max_live < 0:
            # An unknown live set is not an empty one, and this is the number
            # that decides where memory gets unmapped. Abstain, and take the
            # group with us -- see ABSTAIN.
            return ABSTAIN
        floor_rows = self._floor_rows(max_live)
        desire = current
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
        return (int(desire), -int(floor_rows), int(current), -int(current))

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
        current = int(getattr(self._pool, "size", 0))
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
        current = int(getattr(self._pool, "size", 0))
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
        """Re-back the pool to its boot reservation and lift the cap.

        Restore order is the mirror of the shrink: pages FIRST, cap SECOND.
        Lifting the cap before the memory exists would re-admit ids that are
        still unmapped, which is the very fault the cap prevents.
        """
        if self._rows_at_boot is None:
            return 0
        rows = int(self._rows_at_boot)
        was = int(getattr(self._pool, "size", 0))
        if self._supported() and was < rows:
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
        self._cap.release()
        self._rows_at_boot = None
        self._exhausted = False
        self.recover_count += 1
        return max(0, int(getattr(self._pool, "size", rows)) - was)


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
) -> Optional[KvBackingRelief]:
    """Build the relief for a scheduler's KV pool, or None when unavailable.

    Returns None rather than an inert callable when the pool is not on a VA
    reservation: a provider that is registered but can never pay makes the
    guard's spend order read as if a tier were funded when it is not, and this
    chain has shipped three of those.
    """
    # OFF BY DEFAULT, AND THE REASON IS A WEDGE, NOT CAUTION.
    #
    # The cap changes ``available_size()``, which feeds ADMISSION. Each rank
    # sizes its own shrink from its own free memory and its own live set, so
    # the three ranks capped to 449039 / 451037 / 175225 / 145734 rows in one
    # boot -- i.e. they no longer agreed on how much work the group could
    # take. A PP group whose ranks disagree about admission desyncs, and this
    # one did: the scheduler stopped heartbeating and /health reported
    # "couldn't get a response from detokenizer" while every rank was alive.
    #
    # The bytes half is PROVEN on metal (want 208 MiB, free 4 -> 1844 MiB,
    # reclaimed 1840 MiB from [kv-backing]). What is missing is agreement: the
    # target must be a COLLECTIVE MINIMUM across ranks, the way the seam's
    # abandon already rides ``_collective_min``, so every rank caps to the
    # same row count. Until that lands, the rung is opt-in and the ship config
    # runs without it rather than with a group-desync hazard.
    if os.environ.get("SGLANG_KV_BACKING_RELIEF", "") not in ("1", "true", "yes", "on"):
        logger.info(
            "%s relief is OFF (set SGLANG_KV_BACKING_RELIEF=1 to enable). The "
            "device half is proven, but the shrink target is still rank-local "
            "and ranks that disagree about admission desync the group.",
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
    )
