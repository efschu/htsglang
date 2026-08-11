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
        law_floor_bytes: int = 1024 * 1024 * 1024,
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
        current = self._current_rows()
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
        # Hoisted so the diagnostic below can report the terms on the path
        # where the rung declines, which is the ONLY path it has ever taken.
        free_now = -1
        deficit = 0
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
        )
        return (int(desire), -int(floor_rows), int(current), -int(current))

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
            "%s proposal on device %s: rows current=%d floor=%d (max_live=%d, "
            "slack=%d) | need = floor %.0f + delta %.0f + want %.0f = %.0f MiB "
            "against free %.0f MiB and cheap relief %.0f MiB -> deficit "
            "%+.0f MiB -> %s. %s",
            LOG_PREFIX,
            self._device_index,
            t["current"],
            t["floor_rows"],
            t["max_live"],
            max(0, t["current"] - t["floor_rows"]),
            t["floor_bytes"] / _MIB,
            t["delta_bytes"] / _MIB,
            t["want_bytes"] / _MIB,
            (t["floor_bytes"] + t["delta_bytes"] + t["want_bytes"]) / _MIB,
            t["free_now"] / _MIB,
            t["cheap_relief_bytes"] / _MIB,
            deficit_mib,
            (
                f"SHRINK to {t['desire']}"
                if t["desire"] < t["current"]
                else "no change"
            ),
            (
                "the cheaper tier covers the whole gap, so KV capacity is not "
                "spent -- this is the tier law, not a broken rung"
                if t["deficit"] <= 0
                else "the cheap tier cannot cover it; KV capacity is the funder"
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
    if os.environ.get("SGLANG_KV_BACKING_RELIEF", "1") not in ("1", "true", "yes", "on"):
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
    )
