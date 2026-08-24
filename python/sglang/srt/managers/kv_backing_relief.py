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
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-BACKING"

_MIB = 1024 * 1024

#: Report EVERY proposal, not only the ones that change the deficit's sign.
KV_RELIEF_TRACE_ENV = "SGLANG_KV_RELIEF_TRACE"

#: A desire no reduction can lower: the neutral element of an element-wise MIN.
_UNBOUNDED_ROWS = 1 << 40

#: #796: THE GROUP'S SHRINK IS AGREED AS A PROPORTION, NOT AS A ROW ID.
#:
#: Parts per million of each rank's own current cap, so 1000000 means "no
#: change" and is the true neutral element of the MIN below.
#:
#: WHY THE CURRENCY HAD TO CHANGE, measured on metal 2026-08-22
#: (boot_798_0822_0629.log, 114 identical declines under live load). The ranks
#: of this fork do NOT have equally sized pools -- that is the whole point of
#: uneven TP shards, uneven DCP tokens and the KV ratio -- and on that boot they
#: were 204800 / 135168 / 112640 rows. The seam's funder was the one axis still
#: agreeing an ABSOLUTE row id, and an absolute row id cannot express a shrink
#: across pools of different sizes. Two faults stacked:
#:
#: 1. ``propose`` encoded "no change" as ``desire = current``. PP2 needed
#:    nothing (deficit -55 MiB) and proposed 112640, which -- because its pool
#:    is the SMALLEST -- was the smallest desire in the group and therefore won
#:    the MIN that this module documents as "the most-pressed rank sets the
#:    ambition". It does the opposite: the least-pressed rank with the smallest
#:    pool sets it.
#: 2. Even repaired, PP0's genuine ambition of 154376 lies ABOVE PP2's entire
#:    pool of 112640, so it can never fall below ``min_current`` and the group
#:    concludes nobody asked for anything.
#:
#: So PP0 sat 295 MiB short of seam staging while its own rung offered 1576 MiB
#: from 185271 rows of slack, the flip abandoned, the decode bundle had already
#: been drained to arm it, and the instance idled with 473499 tokens waiting.
#:
#: A PROPORTION IS THE CONGRUENT CURRENCY, and not merely a convenient one. The
#: uneven DCP token vector is calibrated against the ranks' capacity RATIO, so a
#: shrink that preserves the ratio leaves admission congruent, while shrinking
#: one rank alone would invalidate the vector -- which is HANDOFF_675 §1a's
#: desync in a new dress. On an even fleet this degrades exactly to the previous
#: behaviour, which the pp_to_tp legs of that same boot exercise as a built-in
#: regression (all three ranks at 450560, GRANTED).
_SHRINK_SCALE = 1_000_000


def _shrink_ppm(desire_rows: int, current_rows: int) -> int:
    """This rank's ambition as a proportion of its own cap.

    ``desire >= current`` means NO CHANGE, and it must map to exactly
    ``_SHRINK_SCALE`` -- the neutral element of the MIN. Encoding it as the
    rank's own row count is the #796 defect: on an uneven fleet the smallest
    pool's "no change" is the smallest number in the group and silently wins.
    """
    if current_rows <= 0:
        return _SHRINK_SCALE
    if desire_rows >= current_rows:
        return _SHRINK_SCALE
    # Rounded DOWN, so the proportion never asks for less than the rank did.
    return max(1, (int(desire_rows) * _SHRINK_SCALE) // int(current_rows))


def floor_exceeds_local_cap(floor_rows: int, current_rows: int) -> bool:
    """#770/#812: is this rank's floor ABOVE its own cap? Then it is a defect.

    THE TWO CASES ``_floor_ppm`` USED TO COLLAPSE, and why collapsing them
    silently froze the group. Both returned ``_SHRINK_SCALE``:

        floor == current   HEALTHY. The rank is exactly at its floor and has
                           no slack to give. A true fact about a full pool.
        floor >  current   DEFECT. The rank's live set plus its admission
                           reserve does not FIT the rows it has backed -- it
                           is under-backed. Nothing about the group.

    Measured, boot_816_core_0823_0608.log 06:32:05, all three ranks carrying
    the SAME floor 128549 (correct: under PP a request's tokens occupy KV on
    every stage, so the live set is genuinely replicated) against caps that
    differ by design::

        PP0  backed 212992   floor 128549   ->  60.4% of its own cap
        PP1  backed 124928   floor 128549   -> 102.9%   <- THIS
        PP2  backed 133120   floor 128549   ->  96.6%

    PP1's 102.9% clamps to 100%, the group MAX takes it, ``explain_kv_target``
    computes ``target = max(desire, max_floor)`` and NOBODY shrinks -- vetoing
    PP0's fully fundable 84443-row plan on the very rank that needed it.

    ``_shrink_ppm`` already documents this exact trap for the ambition side
    ("on an uneven fleet the smallest pool's 'no change' is the smallest
    number in the group and silently wins"). The lesson was never applied to
    the floor side. Under the standing kein-bindender-rang law a floor above
    a rank's own cap is a DEFECT REPORT about that rank's backing, never a
    capacity verdict for its peers.
    """
    return int(current_rows) > 0 and int(floor_rows) > int(current_rows)


def _floor_ppm(floor_rows: int, current_rows: int) -> int:
    """This rank's floor as a proportion, rounded UP so the limit never slips.

    A floor at or above the cap still returns the neutral element -- under a
    PROPORTIONAL agreement a rank with no slack genuinely cannot be shrunk,
    and answering otherwise would drive it below its own live set. What
    changed is that the DEFECT case is no longer indistinguishable from the
    healthy one: see :func:`floor_exceeds_local_cap`, which the callers use to
    name an under-backed rank instead of reporting a group-wide veto.
    """
    if current_rows <= 0:
        return _SHRINK_SCALE
    if floor_rows >= current_rows:
        return _SHRINK_SCALE
    return min(
        _SHRINK_SCALE,
        -((-int(floor_rows) * _SHRINK_SCALE) // int(current_rows)),
    )


def _rows_for_ppm(ppm: int, current_rows: int) -> int:
    """The group's proportion as THIS rank's row count, rounded UP.

    Up, because rounding a shrink target down would take the rank slightly
    deeper than the group agreed, and the safe direction for a target that
    unmaps pages is always the shallower one.
    """
    if current_rows <= 0:
        return 0
    return min(
        int(current_rows),
        -((-int(ppm) * int(current_rows)) // _SHRINK_SCALE),
    )


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


def collective_kv_shrink_ppm(reduced):
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
    # #796: the comparison is against the NO-CHANGE proportion, not against the
    # group's smallest cap. Against ``min_current`` this asked "does any rank
    # want to go below the SMALLEST rank's pool", which on an uneven fleet is a
    # question no healthy rank ever answers yes to -- PP0 wanting its own
    # 204800-row pool cut to 154376 is not below PP2's 112640-row cap, so the
    # group read it as nobody asking for anything at all.
    if target >= _SHRINK_SCALE:
        return None
    return int(target)


def collective_kv_target(reduced, current_rows=None):
    """The group's agreed target as a ROW COUNT for a pool of ``current_rows``.

    #796: the group agrees a PROPORTION, because the pools are uneven by design
    and a row id is meaningless to a peer. Every rank still needs a row count in
    the end, and this is where the proportion becomes one.

    ``current_rows`` defaults to the group's smallest cap, which is field 2 of
    the reduction. On an EVEN fleet that IS every rank's own cap, so the default
    reproduces the pre-#796 return value exactly -- which is why the suites that
    predate this change still read as they did. A rank on an uneven fleet must
    pass its OWN cap, and :meth:`KvBackingRelief.apply_shrink_ppm` does.
    """
    ppm = collective_kv_shrink_ppm(reduced)
    if ppm is None:
        return None
    if current_rows is None:
        current_rows = int(reduced[2])
    return _rows_for_ppm(int(ppm), int(current_rows))


def explain_kv_target(reduced) -> str:
    """WHY :func:`collective_kv_target` returned what it returned, in one line.

    THE SILENCE THIS ENDS, measured on metal 2026-08-22
    (boot_798_0822_0543.log). PP0 was short of seam staging with a fundable
    plan in hand -- ``current=204800 rows, floor=115681, slack=89119,
    deficit=+1740 MiB -> SHRINK to 149126`` -- and no shrink ever ran. Across
    that whole boot there is not one occurrence of ``runtime_set_backing_rows``,
    of "the eviction did not deliver the mark", or of "ABSTAIN on device":
    this function's caller returned ``None`` and ``apply_target`` was never
    reached. Eight flips abandoned, phase purity yielded, and the instance
    prefilled in the TP layout.

    The decisive term was a PEER's floor, and it is the one number no log
    carries. ``target = max(desire, max_floor)`` must clear EVERY rank's live
    set, so a rank under no memory pressure at all -- PP2, sitting on 2693 MiB
    spendable and reporting ``fundable_bytes() == 0`` on all eight of its asks,
    which by construction means its floor was at or above its own cap -- vetoes
    the shrink that the pressed rank needs. Every rank computes its floor,
    every rank retains it in ``_last_proposal_terms``, and a rank that FITS
    prints nothing, because the only caller that prints these terms is the one
    that refuses.

    A bare ``Optional[int]`` cannot carry that. The caller was left to report
    "returned NOTHING" and list three possible causes without saying which one
    held -- and the three want opposite responses: an abstention is repaired on
    the abstaining rank, a peer-floor veto is answered by lowering that peer's
    floor, and a cheap-tier decline is the tier law working and wants nothing
    at all. Naming the cause is the difference between a reading and an
    inference, and this chain has now paid for that difference more than once.

    PURE, and deliberately separate from the decision itself: it re-derives the
    same three terms from the same reduced tuple rather than being threaded
    through the decision as an out-parameter, so it cannot change what the
    group decides. A diagnostic that can alter the verdict it reports is worse
    than none.
    """
    if len(reduced) < 3:
        return (
            "the reduced proposal is MALFORMED (fewer than three fields), so no "
            "target could be read at all -- this is a payload-length defect, not "
            "a capacity verdict"
        )
    desire = int(reduced[0])
    max_floor = -int(reduced[1])
    min_current = int(reduced[2])
    if min_current <= 0:
        return (
            f"DECLINED because a rank ABSTAINED (the group's smallest current row "
            f"count is {min_current}). One abstention cancels the shrink for every "
            f"rank, because the danger was never 'nobody capped' -- it is 'some "
            f"capped and some did not'. Repair the abstaining rank; its own "
            f"ABSTAIN line names the precondition it failed."
        )
    target = max(desire, max_floor)
    if target >= _SHRINK_SCALE:
        if max_floor >= _SHRINK_SCALE:
            return (
                f"DECLINED by a PEER FLOOR: the deepest proportion any rank "
                f"asked for is {desire / 10000:.1f}% of its own cap, but the "
                f"group's highest floor is {max_floor / 10000:.1f}% -- at or "
                f"above the no-change proportion -- so NO shrink clears every "
                f"rank's live set. A rank under no pressure can veto the shrink "
                f"a pressed rank needs. The answer is on the FLOOR side (lower "
                f"the highest floor by evicting its recomputable prefix), never "
                f"on the ambition side."
            )
        return (
            f"DECLINED because no rank asked to shrink at all (deepest "
            f"proportion {desire / 10000:.1f}% of its own cap, smallest pool "
            f"{min_current} rows): the cheaper tiers covered every rank's gap, "
            f"which is the tier law working."
        )
    if max_floor > desire:
        return (
            f"GRANTED {target / 10000:.1f}% of each rank's own cap, RAISED from "
            f"the deepest ask of {desire / 10000:.1f}% by the group's highest "
            f"floor {max_floor / 10000:.1f}%: the pressed rank wins less than it "
            f"priced, because a peer's live set sits above what it asked for."
        )
    return (
        f"GRANTED {target / 10000:.1f}% of each rank's own cap: the most-pressed "
        f"rank set the ambition ({desire / 10000:.1f}%), the group's highest floor "
        f"({max_floor / 10000:.1f}%) permits it, and the smallest pool in the "
        f"group is {min_current} rows."
    )


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


def exposure_over_backing(exposed_rows: int, backed_rows: int) -> int:
    """Rows the allocator may hand out that have NO COMMITTED PAGE behind them.

    #816. The one-line statement of an invariant this module maintained
    everywhere except where it released the cap::

        exposed_rows() <= committed backing rows

    Pure and module-level so the decision can be tested without a pool, an
    arena or a boot -- and so there is exactly ONE definition of "over-exposed"
    for every caller, which is the "one consumer never got the treatment"
    lesson #345, #352 and #355 each paid for separately.

    ZERO IS THE ONLY SAFE ANSWER. A positive result means the allocator can
    hand out an id whose pages went back to the driver, and the first write to
    it is a device-side assert in the KV writer's bound check
    (memory_pool.py:4978) -- the crash this function exists to make
    impossible.
    """
    return max(0, int(exposed_rows) - int(backed_rows))


#: A group backing floor that has not been observed yet. Distinct from 0, which
#: is a real floor meaning "no rank has anything backed"; see
#: :func:`group_exposure_ceiling` for why the two must not collapse.
GROUP_FLOOR_UNKNOWN = -1


#: #839-METAL v2: THE NAMED EXITS OF THE FLOOR-NEED PATH.
#:
#: WHY THIS EXISTS AT ALL. Window 6 booted a floor-need actuator that did
#: nothing: 570 flip arms, 0 tp_to_pp, the floor rank's backing never moved,
#: and NEITHER branch of the fix printed -- no grow and no refusal, though
#: those two were supposed to be exhaustive. The log could not say which of
#: five silent ``return 0`` fired, so a whole window was spent proving only
#: that the callsite was reached.
#:
#: THAT IS THE THIRD INSTANCE OF ONE FORM on this tree:
#:   1. WEDGE-RECOVERY 2026-08-22 -- six exits returned None for six causes
#:      and the log line asserted ONE of them;
#:   2. ``publish_group_exposure`` -- five exits, one ``if moved:`` line, which
#:      made "0 seam-ballot publications" read as "the path is unreachable"
#:      when it had in fact executed on all 153 arms and declined;
#:   3. this path, which I built AFTER filing (2) as a defect. Naming the
#:      exits is therefore not decoration here, it is the fix.
#:
#: THE RULE, and it is cheap: every exit carries a name, every name is
#: COUNTED unconditionally so a desk can assert on it, and each distinct exit
#: is LOGGED ONCE rather than every round -- window 5's 1368 repeated
#: GROW-DEBT-UNPAID lines are the reason "log everything" is not the answer.
FLOOR_NEED_NO_GROUP_VERDICT = "NO-GROUP-VERDICT"
FLOOR_NEED_STALE_ARENA = "STALE-ARENA"
FLOOR_NEED_GROUP_FITS = "GROUP-FITS"
FLOOR_NEED_NOT_THE_FLOOR = "NOT-THE-FLOOR"
FLOOR_NEED_GAP = "GAP"
FLOOR_NEED_NO_GAP = "NO-GAP"
FLOOR_NEED_POOL_CANNOT_GROW = "POOL-CANNOT-GROW"
FLOOR_NEED_COMMIT_RAISED = "COMMIT-RAISED"
#: The window-6 root: the setter returned WITHOUT RAISING and the pool did not
#: reach the target anyway -- it clamped. v1 treated a non-raising call as
#: success, computed ``grown`` as 0, cleared nothing, logged nothing, and
#: returned 0. Indistinguishable from "there was no gap".
FLOOR_NEED_COMMIT_CLAMPED = "COMMIT-CLAMPED"
FLOOR_NEED_GROWN = "GROWN"

FLOOR_NEED_EXITS = (
    FLOOR_NEED_NO_GROUP_VERDICT,
    FLOOR_NEED_STALE_ARENA,
    FLOOR_NEED_GROUP_FITS,
    FLOOR_NEED_NOT_THE_FLOOR,
    FLOOR_NEED_GAP,
    FLOOR_NEED_NO_GAP,
    FLOOR_NEED_POOL_CANNOT_GROW,
    FLOOR_NEED_COMMIT_RAISED,
    FLOOR_NEED_COMMIT_CLAMPED,
    FLOOR_NEED_GROWN,
)

#: Marker every named exit line carries, so one grep finds the whole family.
FLOOR_NEED_LOG_MARKER = "[#839-METAL] floor-need"


def group_exposure_ceiling(local_backed_rows: int, group_backed_floor: int) -> int:
    """The id ceiling that keeps the group's exposure IDENTICAL across ranks.

    #833, and it is the half of #816 that fix could not see from one rank.

    ``exposed_rows`` states the contract in its own docstring: the exposed id
    space "has to be identical across the group ... it decides which ids the
    flip's live-slot enumeration can encounter". #816 closed a real crash by
    capping exposure at each rank's OWN committed backing -- and under the
    mandated uneven configuration (uneven token vector, uneven TP vector,
    uneven DCP) the ranks' backings differ BY DESIGN, so a per-rank cap makes
    the exposures differ too. #816's own acceptance evidence records exactly
    that, and nobody read it as a second defect::

        boot 0516  PP2 exposed 449306  committed 126976  withdrew 322330
                   PP1 exposed 449306  committed 120832  withdrew 328474
                   PP0 exposed 449306  committed 204800  withdrew 244506

    Three ranks enter at ONE exposure and leave at THREE. From that moment the
    widest rank hands out ids the narrowest rank has no page for, and
    ``_agree_live_slots`` (phase_flip_runtime.py) must refuse the union the
    moment the live id space grows past the narrowest backing -- permanently,
    because sustained load keeps re-issuing high ids faster than they drain.

    MEASURED, boot_window3_0823_1733: reservations 204334 / 119782 / 126828
    rows on PP0 / PP1 / PP2; seven cutovers completed in the first two minutes
    while the id space was still small, then every ``tp_to_pp`` from 17:43:12
    onward abandoned, all twelve lines naming PP1's 120832 as "the poorest
    rank has only 120832 rows BACKED". The instance served single-phase for the
    remaining 22 minutes. W6 PASSED on the previous window only because that
    boot died in 75 s, before the id space could reach the floor.

    WHY THIS IS A DEFECT AND NOT A CAPACITY VERDICT. Per the standing law, a
    finding of the form "rank A has surplus but it is unreachable because rank
    B binds" is ALWAYS a defect report and NEVER a capacity verdict. The
    surplus is real -- PP0's 204800 rows exist -- and stranding it is a cost
    this function makes VISIBLE (the caller logs it) rather than silent. The
    federated fix that would spend it is the standing #795 debt; what is
    unacceptable in the meantime is issuing ids that no consumer can honour.

    ``GROUP_FLOOR_UNKNOWN`` (or any negative value) means no group verdict has
    been observed, and the answer is then the local backing -- i.e. exactly
    #816's behaviour, unchanged. That default matters: this runs on paths that
    have no collective (a rank-local recovery, a stub, a hermetic test), and
    guessing a floor there would strand rows for no reason.

    A floor of 0 is a REAL floor and is honoured as one. Collapsing it into
    "unknown" is the sentinel collision #714 paid for, where "none" and
    "unknown" shared ``-1`` and eviction pricing silently skipped.
    """
    local = max(0, int(local_backed_rows))
    floor = int(group_backed_floor)
    if floor < 0:
        return local
    return min(local, floor)


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
        #: #833: the group's MIN backed rows, as last decoded from the rung's
        #: own reduction. Every rank decodes the SAME reduced value in the same
        #: round, so storing it here cannot make the ranks disagree -- which is
        #: the entire point, since the quantity it bounds
        #: (:meth:`exposed_rows`) is documented as having to be identical
        #: across the group. ``GROUP_FLOOR_UNKNOWN`` until a ballot is seen, so
        #: a rank with no collective behaves exactly as it did before #833.
        self._group_backed_floor: int = GROUP_FLOOR_UNKNOWN
        #: #839 A: WHICH ARENA THE FLOOR ABOVE WAS MEASURED IN.
        #:
        #: The rung serves TWO layouts (:meth:`_rebind`) whose row counts are
        #: different numbers for the same allocator. The floor arrives from the
        #: seam ballot, which reads whichever layout was active THEN; the
        #: exposure clamp compares it against whichever layout is active NOW.
        #: A row count from the other arena is not comparable, and when the
        #: stale reading is the WIDER one ``min(local, floor)`` stops binding
        #: and every rank publishes its own backing.
        #:
        #: Measured, boot_window4A_0823_2059: three group-uniform rounds at
        #: 122880 (floor narrow, clamp wide -- the floor bound), then at
        #: 21:06:11-12 PP1/PP2/PP0 published 122880/131072/210944, each rank's
        #: own backing to the row (floor wide, clamp narrow -- nothing bound).
        #: The last completed flip is at 21:06:13; every ``tp_to_pp`` after it
        #: abandoned, twelve of them naming PP0's 210944 against a group floor
        #: of 122880.
        self._group_floor_arena: Optional[int] = None
        #: #839 A: the exposure ceiling this rank last PUBLISHED, PER ARENA.
        #:
        #: THE PUBLICATION RULE THIS EXISTS FOR: exposure may be LOWERED at any
        #: time by any rank-local reading, and may be RAISED only by a group
        #: verdict measured in the arena it is raised in. A rank-local grow may
        #: commit pages whenever it likes -- what it may not do is announce
        #: them.
        #:
        #: KEYED BY ARENA FOR THE SAME REASON THE FLOOR IS STAMPED WITH ONE: a
        #: ceiling is a row id, and a row id from the other layout is not a
        #: smaller or larger ceiling, it is a different quantity. A single
        #: cross-arena "last published" would defend the wide layout's number
        #: while the narrow layout is active, which caps nothing at all -- the
        #: exact shape of the defect this closes.
        self._published_exposure: Dict[int, int] = {}
        #: Rows this rank has backed but may not expose, because a peer cannot
        #: map them. Reported, never hidden: it is the #795 federation debt
        #: made countable rather than a silent capacity loss.
        self._stranded_by_group_floor = 0
        #: #839-METAL: the group's HIGHEST LIVE ROW, and the arena it was
        #: measured in. Decoded beside the floor out of the SAME reduction
        #: (``collective_slot_ballot``) and, until this ticket, dropped.
        #:
        #: Without it the group can see that it is stuck and not by how much:
        #: window 5 held ``min_backed_rows`` at 126976 for two whole boots
        #: while the need climbed 131048 -> 131051 -> 131073, and the 4097-row
        #: difference -- the entire defect -- was never a quantity anything
        #: held.
        self._group_live_need: int = GROUP_FLOOR_UNKNOWN
        self._group_need_arena: Optional[int] = None
        #: #839-METAL: the standing "the floor rank cannot fund the live set"
        #: refusal, or ``None``. Readable rather than only logged so the
        #: condition can be asserted at a desk instead of grepped off metal.
        self._floor_need_refusal: Optional[dict] = None
        #: #839-METAL v2: the NAMED EXIT census of the floor-need path.
        #: Counted unconditionally so a desk can assert which exit fired
        #: without booting; window 6 had to spend a whole GPU window
        #: establishing only that the callsite was reached.
        self._floor_need_exit_counts: Dict[str, int] = {}
        #: Dedup key set for the once-per-distinct logging. Keyed by arena too,
        #: because the same reason in the other layout is a different event.
        self._floor_need_said: set = set()
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
        # #796: A RUNG WITH NO SLACK IS A GROUP-WIDE VETO, so it owes the
        # reader the branch that put its floor where it is. The agreed shrink
        # target must clear the HIGHEST floor in the group, so this rank --
        # which may be under no memory pressure at all -- can cancel a peer's
        # fully fundable plan. Measured 2026-08-22: it did, eight arms running,
        # and the eight branches of _evict_floor_rows were indistinguishable
        # from the outside. Printed only when slack is 0, which is exactly when
        # the floor is capable of being the binding term.
        vetoing = ""
        if int(t["floor_rows"]) >= int(t["current"]):
            reason = getattr(self, "_last_evict_floor_reason", None)
            if reason:
                vetoing = (
                    f" -- NO SLACK, so this rank's floor can VETO the group's "
                    f"shrink even if a peer has a fundable plan; the floor is "
                    f"where it is because {reason}"
                )
        return (
            f"KV rung: current={t['current']} rows, floor={t['floor_rows']}, "
            f"slack={max(0, t['current'] - t['floor_rows'])}, deficit="
            f"{t['deficit'] / _MIB:+.0f} MiB -> {verdict} ({why})"
            f"{unreachable}{vetoing}"
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

    def _retained_bytes_clause(self) -> str:
        """Retained arena bytes for this device, or an honest 'unknown'.

        #796: retention is the one candidate the old refusal message named
        that could actually be true, and it was named as an env var rather
        than as a number. ``arena_census`` already keeps it -- read-only,
        allocation-free, and it never raises by contract -- so there is no
        reason to make the reader go and check a variable.

        A census that cannot be read must say so. Reporting an unreadable
        census as ``0 MiB`` would be inventing the measurement that the whole
        point of this change is to stop inventing.
        """
        try:
            from sglang.srt.mem_cache.kv_vmm_backing import arena_census

            census = arena_census() or {}
            row = census.get(int(self._device_index))
            if row is None and len(census) == 1:
                # CUDA_VISIBLE_DEVICES isolation can make the arena's device id
                # disagree with this rung's index. One arena set on the process
                # is unambiguous regardless of which id it filed itself under.
                row = next(iter(census.values()))
            if row is None:
                return "unknown (no arena for this device in the census)"
            return "%.0f MiB across %d arena(s)" % (
                int(row.get("retained", 0)) / (1024 * 1024),
                int(row.get("arenas", 0)),
            )
        except Exception as e:  # pragma: no cover - a census must not fail a log
            return "unknown (census unreadable: %s)" % (e,)

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

    def release_rows_after_floor(relief, rows_wanted, current, floor, page) -> int:
        """Apply the granularity round-up, then RE-CHECK it after the floor clamp.

        THE CLAMP UNDOES THE ROUND-UP, and until 2026-08-22 nothing noticed.
        The caller rounds the ask up to one release granularity because a shrink
        smaller than one commit chunk per buffer clears no extent anywhere. The
        next line then clamps the target to the eviction floor -- and when the
        floor binds, the surviving distance is below one granule again.

        THE NUMBERS THIS DOCSTRING FIRST CARRIED WERE NOT MEASURED, and the
        correction matters more than the guard. It read "PP2's shape:
        current=126976, floor=88945, granule=229376 ... the granule EXCEEDS
        the whole pool, so no shrink on that rank can ever pay at this chunk
        size", attributed to boot_798_0822_0737.log. The string "229376" does
        not occur in that log. It is 28 * 8192 and it is the fixture constant
        from test_shrink_cannot_pay_reason_796.py (256 MiB * 28 / 32 KiB),
        carried across as if it had been read off metal.

        What that boot actually recorded for PP2:

            :1325  32768 B/row over 32 arena buffers
            :16    flip_seam_chunk_mib=8, enable_vram_dial=False
                   -> arena commit chunk 8 MiB
            :3382  current=126976 floor=88945 slack=38031

            _min_release_rows() = ceil(8 MiB * 32 / 32 KiB) = 8192 rows

        The granule is 8192, which is 6.4% of the pool, and PP2's post-floor
        distance of 38031 rows is 4.64 granules. So the clamp does NOT defeat
        the granule on that rank, the granule does not exceed any pool seen so
        far, and chunk sizing (--flip-seam-chunk-mib) is not the lever the old
        text sent the next reader after.

        WHAT IS STILL UNEXPLAINED: all 15 zero-byte shrinks in that boot asked
        at least three whole granules deep, so granularity cannot account for
        any of them. The guard below is correct and cheap, but it would not
        have prevented one of them. The open question is why
        runtime_set_backing_rows reported zero released bytes at that depth.

        The guard stays because the shape it refuses is real arithmetic: when
        the floor clamp does leave less than one granule, attempting the shrink
        costs a cap and returns nothing. It is simply not what PP2 hit.

        DIRECTION OF SAFETY, which is why this cannot revive #717: the guard only
        ever turns a shrink into NO shrink. It never deepens one, so it cannot
        pull backing below the highest live row -- the fault that reverted
        c4e557963e and killed boots.
        """
        rows_wanted = max(int(rows_wanted), int(relief._min_release_rows()))
        target = max(int(floor), int(current) - rows_wanted)
        target = int(math.ceil(target / page) * page)
        if target >= int(current):
            return 0
        granularity = int(relief._min_release_rows())
        if granularity > 0 and (int(current) - target) < granularity:
            # Below one granule after the clamp: decommit_range would clear no
            # extent in any buffer and return address space rather than memory.
            # Refuse BEFORE engaging the cap -- discovering this by attempting it
            # costs the rank its capacity and returns nothing.
            return 0
        return relief._shrink_to(target, int(current))

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
        floor = int(math.ceil(floor / page) * page)
        # #770/#812: DETECT AND NAME AN UNDER-BACKED RANK. DO NOT "FIX" IT HERE.
        #
        # A FIRST VERSION OF THIS CLAMPED THE FLOOR DOWN TO THE CAP, and that
        # was WRONG in the dangerous direction. The floor is `live set + 1 +
        # margin + reserve`; lowering it to a cap that sits BELOW that value
        # does not make the rank able to shrink, it authorises a cap below rows
        # that are still in use. `test_residency_cap_flip_levelling_792
        # ::TheLevellingMustNotCapBelowTheLiveSet` exists for exactly this and
        # failed on the clamp while passing on the unclamped code -- the tree
        # already knew, and the invariant is older than this ticket.
        #
        # So the floor STAYS. What changes is that the condition stops being
        # invisible: a floor above this rank's own backing means the live set
        # plus the admission reserve does not FIT the rows it has, which is a
        # defect about THIS rank's backing whose only real answers are to grow
        # the backing or to change what the group agrees on. Neither is
        # available to a local floor computation, and pretending otherwise is
        # how a correctness invariant gets traded for a funding win.
        #
        # It must still never propagate as a capacity verdict for peers
        # (kein-bindender-rang): see floor_exceeds_local_cap and the caller
        # that reports it.
        #
        # The measured shape, 06:32:05: floor 128549 on all three ranks --
        # correct, the live set is replicated under PP -- against backed rows
        # 212992 / 124928 / 133120, i.e. 102.9% of PP1's own cap.
        #
        cap = 0
        try:
            cap = int(self._current_rows())
        except Exception:  # noqa: BLE001 - a floor must not raise
            cap = 0
        if floor_exceeds_local_cap(floor, cap):
            logger.warning(
                "%s UNDER-BACKED RANK: floor %d rows exceeds the %d rows this "
                "rank has backed (%.1f%% of its own cap). The live set plus the "
                "admission reserve does not fit the backing. The floor is NOT "
                "lowered -- capping below the live set would authorise a cap "
                "over rows still in use -- so this rank simply has nothing to "
                "give. That is a defect about THIS rank's backing, whose answer "
                "is to grow it; it must never be read as a capacity verdict for "
                "peers, which under a proportional agreement is exactly how it "
                "would otherwise freeze the group.",
                LOG_PREFIX,
                floor,
                cap,
                100.0 * floor / cap,
            )
        return int(floor)

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

    def _parked_ceiling(self) -> int:
        """#748: the parked extent as an EXCLUSION SET, not a gate.

        Returns the highest row id the flip has parked, which the rung must
        keep mapped, or -1 when nothing is parked. ``-2`` means the extent is
        UNKNOWN while a flip is armed -- the one case that still refuses
        wholesale, because there is no boundary to name.

        WHY THIS REPLACED A GATE. #744 refused the rung outright while a flip
        was armed. That stopped the 21:18 eviction crash and also strangled the
        flip's own funder: seam staging asks the rung to evict recomputable
        prefix rows, which is what "KV capacity is the funder" means, so a
        wholesale refusal produced 35 refused tp_to_pp flips and an IDLE-LOCK
        with 407,622 tokens pending and nothing resident (#748, 21:46:32).

        The extent is exactly the information needed to be selective: rows
        INSIDE it are the ones the flip is about to pack and may not be
        touched; every row above it is recomputable prefix and is precisely
        what the funding wants. So the extent pins the ceiling instead of
        closing the rung.
        """
        rows, top = self._flip_pending()
        if rows < 0:
            # UNKNOWN. Only refuses while a flip is actually armed; outside
            # one there is nothing to protect. #746 confined this case to a
            # flip whose ARM-TIME extent measurement itself failed -- the
            # snapshot otherwise exists from arm to exit.
            return -2 if self._flip_armed() else -1
        if rows == 0:
            return -1
        return int(top)

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
        # #748: one definition of "is something parked", shared with the
        # exclusion ceiling. -1 means nothing is parked and the extent is
        # irrelevant -- including an UNKNOWN probe outside a flip, which must
        # not close the rung: there is nothing to protect when no flip is
        # armed.
        if self._parked_ceiling() != -1:
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

        #796: EVERY RETURN RECORDS WHY. Over a sparse live set the plain floor
        is routinely ABOVE the cap (#714: ``max_live`` is a high-water ID, not
        a count of backed rows), so all eight of the branches below produce one
        observable -- ``floor >= current``, slack 0 -- and that observable
        cancels the group's shrink for EVERY rank, because the agreed target
        must clear the highest floor in the group. On 2026-08-22 a rank under
        no memory pressure vetoed a peer's fully fundable +1740 MiB plan that
        way, eight arms in a row, and nothing in the log said which branch had
        done it. The eight want different answers -- three are healthy, one is
        a setting, the rest are defects -- so the reason is recorded here and
        printed by ``last_proposal_summary`` whenever the rung has no slack.
        """
        plain = self._floor_rows(max_live)
        if not self._evict_enabled():
            self._last_evict_floor_reason = (
                "the evict rung is DISABLED by configuration, so this rank can "
                "only offer the slack above its plain floor"
            )
            return plain, 0
        # #748: the parked extent EXCLUDES rows, it does not close the rung.
        # An UNKNOWN extent under an armed flip is the one case with no
        # boundary to name, so it still refuses.
        parked = self._parked_ceiling()
        if parked == -2:
            self._last_evict_floor_reason = (
                "the PARKED flip extent is unreadable, and an unknown extent "
                "has no boundary to evict up to -- a DEFECT if it persists"
            )
            return plain, 0
        tree = self._tree_cache()
        if tree is None:
            self._last_evict_floor_reason = (
                "there is no radix TREE cache on this rank, so no recomputable "
                "prefix can be priced -- a DEFECT if the pool is meant to cache"
            )
            return plain, 0
        req_max = self._resident_ceiling()
        if parked >= 0:
            # The parked rows pin the ceiling. Everything above them is
            # recomputable prefix and stays evictable -- which is the funding
            # the flip itself is waiting on.
            req_max = max(req_max, parked)
        if req_max < 0:
            if not self._nothing_resident():
                # Unknown resident half: refuse to price an eviction at all.
                self._last_evict_floor_reason = (
                    "the RESIDENT half of the live set could not be read while "
                    "rows are resident, and an unknown split is not an empty "
                    "one -- a DEFECT if it persists"
                )
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
            self._last_evict_floor_reason = (
                f"the mark is PINNED by work in flight: the resident/parked "
                f"ceiling {req_max} is at or above the high-water row "
                f"{int(max_live)}, so there is no recomputable prefix above it "
                f"-- healthy, the pool is genuinely live"
            )
            return plain, 0
        floor = self._floor_rows(req_max)
        if floor >= plain:
            self._last_evict_floor_reason = (
                f"evicting would not lower this rank's floor: the priced floor "
                f"{floor} is no better than the plain floor {plain} -- healthy"
            )
            return plain, 0
        try:
            from sglang.srt.managers.kv_radix_watermark import evictable_rows_above

            rows, _nodes = evictable_rows_above(tree, max(0, floor - 1))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("%s could not price the watermark rung: %s", LOG_PREFIX, e)
            self._last_evict_floor_reason = (
                f"pricing the watermark rung RAISED ({e}) -- a DEFECT; the "
                f"floor fell back to the plain {plain}"
            )
            return plain, 0
        if rows <= 0:
            # TWO DIFFERENT CONDITIONS REACH THIS ZERO, and until 2026-08-22
            # both were reported as the first one. Branch 8 claimed "the pool is
            # genuinely live" without having measured liveness anywhere.
            #
            # When the priced floor sits ABOVE the high-water row, the query
            # just asked for evictable rows in a region that does not exist:
            # the tree cannot hold rows above the pool's own high-water mark, so
            # the zero is a TAUTOLOGY and carries no information about what is
            # live. This is #714 arriving here rather than at the pool cap --
            # `_floor_rows(x) == x + 1 + margin + reserve`, so a resident
            # ceiling within (margin + reserve) rows of the high-water lifts the
            # priced floor past it. Measured on metal (boot_798_0822_0737.log):
            # priced floor 167440 against high-water row 164055.
            #
            # The guards above cover req_max >= max_live (PINNED) and
            # floor >= plain, but neither covers this, so it fell through to the
            # pricing call and was mislabelled by it.
            #
            # The return is unchanged in BOTH cases -- this distinguishes the
            # message only, so the ladder's behaviour is byte-identical.
            if floor > int(max_live):
                self._last_evict_floor_reason = (
                    f"the priced floor {floor} is ABOVE the high-water row "
                    f"{int(max_live)} (plain floor {plain}), so the eviction "
                    f"query covered an EMPTY region and its zero says nothing "
                    f"about what is live -- #714 at the high-water mark; the "
                    f"floor still vetoes the group's shrink"
                )
            else:
                self._last_evict_floor_reason = (
                    f"NO EVICTABLE rows above the priced floor {floor} (plain "
                    f"floor {plain}, high-water row {int(max_live)}) -- the "
                    f"radix tree prices nothing in a REAL band of "
                    f"{int(max_live) - floor + 1} rows above the resident "
                    f"ceiling. Healthy if those rows are genuinely live; a "
                    f"DEFECT if they are unaccounted, which the POOL CENSUS "
                    f"line is the place to check. Either way it vetoes the "
                    f"group's shrink"
                )
            return plain, 0
        self._last_evict_floor_reason = (
            f"PRICED an eviction: floor {floor} instead of the plain {plain}, "
            f"funded by {int(rows)} evictable rows above it"
        )
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
        # #748: both sides must agree on what the parked extent means, for the
        # reason the comment below already gives about the branch -- a
        # disagreement becomes an illegal address. Here that means the SAME
        # exclusion ceiling, not the same wholesale refusal.
        parked = self._parked_ceiling()
        if parked == -2:
            return 0
        req_max = self._resident_ceiling()
        if parked >= 0:
            req_max = max(req_max, parked)
        elif req_max < 0 and not self._nothing_resident():
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
        # #796: the first two fields leave in PARTS PER MILLION OF THIS RANK'S
        # OWN CAP. Rows are this rank's private unit -- the pools are uneven by
        # design -- so a row id is meaningless to a peer, while a proportion is
        # the same statement on every rank. ``current`` stays in rows in field
        # 2 because its only job is to detect an abstention (a non-positive
        # value), and field 3 stays its negation for the diagnostic maximum.
        return (
            _shrink_ppm(desire, current),
            -_floor_ppm(floor_rows, current),
            int(current),
            -int(current),
        )

    def preview_shrink_ppm(self, ppm: int) -> tuple:
        """What ``apply_shrink_ppm`` WILL do, without doing it.

        #796: the verdict line is logged before the shrink is applied, so it
        can only report the outcome by predicting it. This is that prediction,
        and ``apply_shrink_ppm`` is implemented on top of it so the two cannot
        disagree -- a diagnostic that drifts from the behaviour it describes is
        the defect this method exists to prevent, not a style preference.

        MEASURED, boot_798_0822_0646.log 06:50:59Z: the verdict reused the
        rung's PROPOSAL summary, so PP1 -- whose deficit was already covered,
        and which therefore proposed the neutral element -- printed "no change"
        in the same round in which it unmapped 702 MiB. Under a proportional
        agreement the proposal and the applied action diverge BY DESIGN: a rank
        that asked for nothing still pays its share, because shrinking only the
        pressed rank would change the capacity ratio the uneven DCP token
        vector is calibrated against.

        Returns ``(current_rows, target_rows)``. ``target_rows >= current_rows``
        means this rank yields nothing this round.
        """
        current = self._current_rows()
        if current <= 0:
            return 0, 0
        rows = _rows_for_ppm(int(ppm), current)
        max_live = self._max_live_row()
        floor_rows, _evictable = (
            self._evict_floor_rows(max_live) if max_live >= 0 else (current, 0)
        )
        return int(current), max(int(floor_rows), int(rows))

    def explain_shrink_ppm(self, ppm: int) -> str:
        """One clause naming what THIS rank does with the group's proportion.

        Replaces ``last_proposal_summary()`` on the GRANTED path. That method
        stays correct for the caller it was written for -- the one that refuses
        -- but on a granted seam it reports the one number guaranteed to be
        wrong for any rank that was not the most-pressed.
        """
        current, target = self.preview_shrink_ppm(ppm)
        if current <= 0:
            return "this rank has no backed pool, so it yields nothing"
        if target >= current:
            return (
                f"KV rung: current={current} rows -> no shrink applied; this "
                f"rank's own floor ({target}) already sits at or above the "
                "group's proportion, so it yields nothing this round"
            )
        freed_mib = ((current - target) * self._bytes_per_row) // (1024 * 1024)
        return (
            f"KV rung: current={current} rows -> APPLIED shrink to {target} "
            f"rows ({freed_mib} MiB returned). Every rank pays the group's "
            "proportion, including one under no pressure of its own: the "
            "capacity ratio across ranks is what the uneven token vector is "
            "calibrated against."
        )

    def apply_shrink_ppm(self, ppm: int) -> int:
        """Apply the group's agreed PROPORTION to this rank's own pool.

        The group agrees the proportion; each rank converts it against its own
        cap and its own floor. That is the split #796 turned on: the DECISION
        is uniform, the ROW COUNT is rank-local, and the floor -- which protects
        this rank's live set from an unmap -- can only ever be applied here.
        """
        current, target = self.preview_shrink_ppm(ppm)
        if current <= 0:
            return 0
        if target >= current:
            # This rank's own floor already sits at or above the group's
            # proportion, so it wins nothing this round. Not an error, and not
            # a reason to deny the ranks that can pay -- which is precisely the
            # veto #796 removed.
            return 0
        return int(self.apply_target(target))

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
        return self.release_rows_after_floor(rows_wanted, current, floor, page)

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
            # #816: the shrink failed, so the backing is wherever it already
            # was -- which may be BELOW the id space this release just
            # re-exposed. Undoing the cap must not hand out ids the failed
            # shrink never backed.
            self.clamp_exposure_to_backing("after a failed shrink")
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
            # #796: REPORT WHAT WAS MEASURED, NOT TWO CANDIDATE CAUSES.
            #
            # The previous wording offered "the arena has no commit chunk, or
            # its handles are retained", separated the two in neither code nor
            # fact, and on 2026-08-22 (boot_798_0822_0810.log) was wrong about
            # both across all 24 refusals. A chunkless arena cannot reach this
            # line at all -- registration refuses one outright at the
            # ``supports_backing_spans`` gate -- and retention is a number
            # ``arena_census()`` already keeps, so naming an env var the reader
            # then has to go and check is strictly worse than printing it.
            chunk_bytes = int(getattr(self._pool, "backing_commit_chunk_bytes", 0) or 0)
            retained = self._retained_bytes_clause()
            if claimed <= 0:
                # THE POOL AND THE DRIVER AGREE, so there is nothing to
                # reconcile and no unmap to reason about. This is the branch
                # that actually fired on metal, 24 times out of 24, while 15
                # shrinks on the SAME boot released 256/512 MiB -- which is why
                # no standing property of the arena can be the explanation.
                # ``runtime_set_backing_rows`` returns BYTES RELEASED TO THE
                # DRIVER, so claimed=0 is the pool declining upstream of any
                # measurement this rung makes.
                logger.warning(
                    "%s shrink to %d rows released NOTHING and the pool agrees: "
                    "it returned claimed=%d bytes, so the backing never moved "
                    "and there is no pool-versus-driver divergence to explain. "
                    "Measured state: asked %d rows against a release "
                    "granularity of %d rows (commit chunk %.0f MiB across %d "
                    "buffers, %.0f KiB per row), arena retained %s. The cap "
                    "STAYS ON -- undoing it here would re-commit pages inside a "
                    "gate that armed because memory was short. No further "
                    "attempt will be made until the next recovery.",
                    LOG_PREFIX,
                    target,
                    int(claimed),
                    asked,
                    granularity,
                    chunk_bytes / (1024 * 1024),
                    self._buffers,
                    self._bytes_per_row / 1024,
                    retained,
                )
            else:
                # THE REAL DIVERGENCE, kept because it is a genuine failure
                # mode -- it simply was not the one being hit.
                logger.warning(
                    "%s shrink to %d rows: the pool reported %.0f MiB released "
                    "but the driver's free column did not move. That "
                    "disagreement is the unmap-without-release signature -- "
                    "extents unmapped while this process still owns their "
                    "handles yield address space rather than memory. Measured "
                    "state: arena retained %s, commit chunk %.0f MiB across %d "
                    "buffers, asked %d rows against a granularity of %d. The "
                    "cap STAYS ON -- undoing it here would re-commit pages "
                    "inside a gate that armed because memory was short. No "
                    "further attempt will be made until the next recovery.",
                    LOG_PREFIX,
                    target,
                    claimed / (1024 * 1024),
                    retained,
                    chunk_bytes / (1024 * 1024),
                    self._buffers,
                    asked,
                    granularity,
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
        #
        # #839 A: AND IT COMES BACK BOUNDED BY THE GROUP, IN THE SAME BREATH.
        # This used to re-engage at ``now`` -- this rank's own new backing --
        # and leave the group bound to ``clamp_exposure_to_backing`` two
        # statements later. The end state was the same, but the invariant
        # "exposed <= min(local committed, group bound)" was false in between,
        # and the ``else`` leg below left the WHOLE reservation exposed for
        # that span. The invariant is meant to hold at every point, not only
        # at the end of the function, so the ceiling is computed BEFORE the
        # release and applied as part of the same release-and-re-engage.
        ceiling = self._exposure_ceiling(now)
        self._cap.release()
        if ceiling < self._reservation_rows():
            self._cap.engage(ceiling)
        self._record_published(ceiling)
        # AND SAY WHAT WAS PUBLISHED, ON EVERY RANK, ON EVERY RECOVERY.
        # The old sequence announced the ceiling only when the clamp below
        # found something to withdraw, so the group-uniformity of the exposure
        # was observable only in the rounds where it was already being
        # corrected. Window 4's whole verdict rests on nine such lines. This
        # one fires unconditionally, carries the group bound that decided it,
        # and keeps the "Capped at" marker the existing checks grep for.
        logger.warning(
            "%s [#839] exposure published after recovery: %d rows committed "
            "on this rank, group bound %s (%s). Capped at %d. Every rank "
            "publishes the group bound in the same round, so three ranks "
            "printing three different levels here is the #839 A divergence "
            "and not a property of the pools.",
            LOG_PREFIX,
            now,
            (
                "unknown -- no ballot seen"
                if self._group_backed_floor < 0
                else int(self._group_backed_floor)
            ),
            (
                "measured in this arena"
                if self._group_floor_arena == self._arena_key()
                else "measured in the OTHER arena, so it may lower but not "
                "raise; this arena's last published level %s also holds"
                % (self._published_exposure.get(self._arena_key()),)
            ),
            ceiling,
        )
        if now < boot_rows:
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
        # #816: the branch above compares against ``boot_rows`` -- the
        # REMEMBERED recovery target -- not against the allocator's id space.
        # When ``boot_rows <= now < _reservation_rows()`` it takes the else
        # leg, engages nothing, and the release two lines up has just exposed
        # the WHOLE id space over ``now`` committed rows. That is the #816
        # crash, and it is why the clamp is unconditional rather than part of
        # the condition: the condition is the thing that was wrong.
        self.clamp_exposure_to_backing("after recovery")
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

    def note_group_backing_floor(self, rows: int) -> None:
        """Record the group's MIN backed rows, as the rung just reduced it.

        #833. Called with the value ``collective_slot_ballot`` decoded, which
        every rank reads identically out of the same reduction -- so this can
        never be a source of divergence, and it is the only value in this
        object that is allowed to bound :meth:`exposed_rows`.

        A negative argument is discarded rather than stored: the abstain
        sentinel and a truncated payload both arrive that way, and treating
        either as a floor of zero would withdraw the entire id space on a rank
        whose peer merely failed to report.
        """
        floor = int(rows)
        if floor < 0:
            return
        self._group_backed_floor = floor
        # #839 A: AND STAMP THE ARENA IT WAS MEASURED IN. The ballot read
        # ``backed_rows()``, which rebinds first, so this floor is a row count
        # in whichever layout is active at THIS instant. Recording the value
        # without recording which arena it counts is what let a wide reading
        # be compared against a narrow backing -- see ``_group_floor_arena``.
        self._group_floor_arena = self._arena_key()

    def note_group_live_need(self, max_live_row: int) -> None:
        """Record the group's HIGHEST LIVE ROW, from the same ballot payload.

        #839-METAL. ``collective_slot_ballot`` decodes this beside
        ``min_backed_rows`` out of one reduction (:func:`collective_slot_ballot`,
        the ``max_live_row`` key) and until now only the floor was consumed. The
        need is the number that says HOW FAR the floor has to rise; without it
        the group knows it is stuck and not by how much.

        MEASURED, window 5, integ/round5, BOTH boots to the row::

            live set needs 131073   PP1 backs 126976   <- the floor, and short
            live set needs 131073   PP2 backs 133120
            live set needs 131073   PP0 backs 215040

        PP1's backing never moved from 126976 in either boot -- 30 minutes on
        5b -- and every one of 153 flip arms abandoned on the 4097-row gap.

        Negative arrives from the abstain sentinel and from a truncated
        payload, and is discarded for the same reason a negative floor is: a
        peer that failed to report is not a group that needs nothing.
        """
        row = int(max_live_row)
        if row < 0:
            # The abstain sentinel and a truncated payload both arrive as -1.
            # Discarding is right; SILENTLY discarding was half of why v1 was
            # unreadable on metal, so the caller's next verdict names
            # NO-GROUP-VERDICT rather than returning a bare 0.
            return
        # SPAN, NOT ROW ID. ``max_live_row`` is the HIGHEST LIVE ROW; the
        # number of rows the union has to span is that plus one, which is
        # exactly how the abandon path reads it:
        # ``span = int(ballot.get("max_live_row", -1)) + 1``
        # (phase_flip_runtime.py:8754). v1 compared the raw row id against a
        # ROW COUNT, so it asked for one row less than the flip needs -- on
        # window 6's numbers 131072 against a floor of 126976 instead of
        # 131073, which is the difference between a flip that fits and one
        # that abandons by a single row.
        self._group_live_need = row + 1
        self._group_need_arena = self._arena_key()

    def _note_floor_need_exit(self, reason: str, **facts) -> None:
        """Count EVERY exit, log each DISTINCT one once. #839-METAL v2.

        Counting is unconditional so a desk can assert on it without a boot;
        logging is deduplicated on (arena, reason, facts) so the steady-state
        exit -- ``GROUP-FITS``, which is the healthy answer on most rounds --
        cannot flood a log the way window 5's 1368 repeated GROW-DEBT-UNPAID
        lines did. Both halves are needed: a counter nobody prints is invisible
        on metal, and a line printed every round is noise nobody reads.
        """
        self._floor_need_exit_counts[reason] = (
            self._floor_need_exit_counts.get(reason, 0) + 1
        )
        key = (self._arena_key(), reason, tuple(sorted(facts.items())))
        if key in self._floor_need_said:
            return
        self._floor_need_said.add(key)
        logger.warning(
            "%s %s exit=%s %s",
            LOG_PREFIX,
            FLOOR_NEED_LOG_MARKER,
            reason,
            " ".join(f"{k}={v}" for k, v in sorted(facts.items())),
        )

    def floor_need_exits(self) -> dict:
        """The exit census, readable without a boot. #839-METAL v2."""
        return dict(self._floor_need_exit_counts)

    def floor_need_verdict(self) -> tuple:
        """``(gap, reason)`` -- the gap AND which exit produced it.

        EVERY return here is named. That is the whole point of v2: window 6
        could not tell "no group verdict yet" from "this rank is not the floor"
        from "the group already fits", because all three were ``return 0``.
        """
        floor = int(self._group_backed_floor)
        need = int(self._group_live_need)
        if floor < 0 or need < 0:
            self._note_floor_need_exit(
                FLOOR_NEED_NO_GROUP_VERDICT, floor=floor, need=need
            )
            return 0, FLOOR_NEED_NO_GROUP_VERDICT
        # REBIND BEFORE READING THE ARENA, NOT AFTER. ``_arena_key`` is
        # ``id(self._pool)`` and ``self._pool`` only follows ``pool_fn`` when
        # ``_rebind`` runs, so asking which arena is active before rebinding
        # answers with the PREVIOUS one -- a stale answer compares row counts
        # from two layouts, the exact defect #839 A closes. The v1 draft had
        # these two lines the other way round and a guard test caught it.
        self._rebind()
        arena = self._arena_key()
        if self._group_floor_arena != arena or self._group_need_arena != arena:
            self._note_floor_need_exit(
                FLOOR_NEED_STALE_ARENA,
                floor_arena_ok=int(self._group_floor_arena == arena),
                need_arena_ok=int(self._group_need_arena == arena),
            )
            return 0, FLOOR_NEED_STALE_ARENA
        if need <= floor:
            self._note_floor_need_exit(
                FLOOR_NEED_GROUP_FITS, floor=floor, need=need
            )
            return 0, FLOOR_NEED_GROUP_FITS
        local = int(self._current_rows())
        if local > floor:
            self._note_floor_need_exit(
                FLOOR_NEED_NOT_THE_FLOOR, local=local, floor=floor, need=need
            )
            return 0, FLOOR_NEED_NOT_THE_FLOOR
        gap = need - floor
        self._note_floor_need_exit(
            FLOOR_NEED_GAP, floor=floor, need=need, gap=gap
        )
        return gap, FLOOR_NEED_GAP

    def floor_need_gap(self) -> int:
        """Rows THIS rank must commit for the group floor to cover the live set.

        Thin wrapper over :meth:`floor_need_verdict`, kept because callers and
        tests already read a bare int. The REASON is the thing v2 adds; read it
        with ``floor_need_verdict()`` or ``floor_need_exits()``.
        """
        return self.floor_need_verdict()[0]

    def close_floor_need_gap(self) -> int:
        """Commit the pages the group floor is short of the live set. #839-METAL.

        RANK-LOCAL, AND THAT IS WHY IT MAY RUN HERE. This reaches
        ``runtime_set_backing_rows`` and nothing else; ``kv_backing_relief.py``
        enters no collective anywhere, the same property that lets
        ``grow_kv_backing_local`` run at a rank-local cadence.

        IT COMMITS PAGES AND ANNOUNCES NOTHING. The exposed id space still
        moves only in :meth:`publish_group_exposure`, only on a group verdict,
        only in the arena that verdict was measured in.

        THE WINDOW-6 ROOT, and the reason v2 exists: a setter that returns
        WITHOUT RAISING is not proof that the pool grew. If it clamps to the
        rank's budget, v1 computed ``grown`` as 0, recorded no refusal, logged
        nothing and returned 0 -- byte-identical to "there was no gap". So the
        one outcome that must reach an operator, "this rank cannot fund the
        group's live set", was the one outcome that was silent. v2 VERIFIES THE
        COMMIT against the target and names the clamp.

        Returns rows actually committed. Every other outcome is named.
        """
        gap, reason = self.floor_need_verdict()
        if gap <= 0:
            self._note_floor_need_exit(FLOOR_NEED_NO_GAP, because=reason)
            return 0
        floor = int(self._group_backed_floor)
        target = floor + gap
        setter = getattr(self._pool, "runtime_set_backing_rows", None)
        if not callable(setter):
            self._record_floor_need_refusal(
                floor, target, "pool has no runtime_set_backing_rows"
            )
            self._note_floor_need_exit(
                FLOOR_NEED_POOL_CANNOT_GROW, floor=floor, target=target
            )
            return 0
        try:
            setter(target)
        except Exception as e:  # noqa: BLE001 -- MemoryError and driver errors alike
            # NOTHING IS HALF-DONE that needs undoing: the setter either
            # committed the span or it did not, and the exposed id space was
            # never moved, so the group is exactly where it was.
            self._record_floor_need_refusal(floor, target, repr(e))
            self._note_floor_need_exit(
                FLOOR_NEED_COMMIT_RAISED, floor=floor, target=target
            )
            return 0
        # VERIFY. A non-raising setter is a claim, not evidence.
        reached = int(self._current_rows())
        if reached < target:
            # PARTIAL PROGRESS IS STILL PROGRESS, and it is reported as the
            # number it is. Returning 0 here would under-report rows that were
            # genuinely committed -- the defect being fixed is the SILENCE, not
            # the partiality. The refusal carries the shortfall; the return
            # carries the work done. A caller that sees a positive return AND a
            # standing refusal is being told the truth: it grew, and not enough.
            self._record_floor_need_refusal(
                floor,
                target,
                f"the pool CLAMPED: commit returned without error but the "
                f"backing reached {reached} of {target} rows",
            )
            self._note_floor_need_exit(
                FLOOR_NEED_COMMIT_CLAMPED,
                floor=floor,
                target=target,
                reached=reached,
            )
            return max(0, reached - floor)
        self._floor_need_refusal = None
        grown = reached - floor
        self._note_floor_need_exit(
            FLOOR_NEED_GROWN, floor=floor, target=target, grown=grown
        )
        return max(0, grown)

    def _record_floor_need_refusal(self, binding: int, target: int, why: str) -> None:
        """Say it ONCE per (arena, target), and keep it readable.

        Once, because window 5's lesson is that an alarm repeated 1368 times is
        not more informative than one and does crowd out the log. Readable
        rather than only logged, because a criterion that can only be checked
        by grepping a boot log cannot be checked at a desk.
        """
        key = (self._arena_key(), int(target))
        said = self._floor_need_refusal
        if said is not None and said.get("key") == key:
            return
        self._floor_need_refusal = {
            "key": key,
            "binding_rows": int(binding),
            "need": int(target),
            "short": int(target) - int(binding),
            "why": str(why),
        }
        logger.warning(
            "%s [#839-METAL] GROUP FLOOR CANNOT FUND THE LIVE SET: this rank "
            "holds the group floor at %d rows and the group's live set needs "
            "%d, so the group is short %d rows and EVERY tp_to_pp will abandon "
            "on the union bound until that changes (%s). This is the binding "
            "rank -- the backed-but-unexposed surplus on the WIDER ranks is "
            "not payable against it, because the agreed level is the group MIN "
            "and growing a rank that is not the floor moves it by zero "
            "(#834 crit 13's counter is blind to this rank by construction: "
            "its backed-but-unexposed count is 0 precisely because it IS the "
            "floor). Size this rank's budget up, or accept single-phase.",
            LOG_PREFIX,
            binding,
            target,
            int(target) - int(binding),
            why,
        )

    def floor_need_refusal(self) -> Optional[dict]:
        """The standing refusal, or ``None`` when the floor can fund the need."""
        return self._floor_need_refusal

    def _arena_key(self):
        """Identity of the layout the backing calls are currently pointed at.

        ``id(self._pool)`` is already the key ``_rebind`` parks per-arena state
        under, so this is the same notion of "which layout", named once rather
        than spelled out at each comparison.
        """
        return id(self._pool)

    def _record_published(self, level: int) -> None:
        """Remember the ceiling that was just put on the wire. #839 A.

        Every path that sets this rank's exposed id space ends here, so the
        "never raise without a fresh in-arena verdict" rule has one number to
        defend rather than one per caller.
        """
        self._published_exposure[self._arena_key()] = int(level)

    def published_exposure(self) -> Optional[int]:
        """The level a group verdict last put this rank at, in THIS arena.

        #839 B: a public reading, because the deferred grow needs to know which
        level it must not clamp back below and had no way to ask.

        NOT ``exposed_rows()``. That returns the whole reservation whenever no
        cap is engaged, and the reservation is a property of this rank's
        allocator rather than anything the group agreed to -- using it as a
        level would clamp every rank to its own id space, which is the #839 A
        divergence arriving through the payment path instead of the clamp.
        ``None`` means nothing has been published in this arena yet.
        """
        return self._published_exposure.get(self._arena_key())

    def _exposure_ceiling(self, backed: int) -> int:
        """The highest id this rank may EXPOSE right now. #839 A.

        ONE RULE, IN ONE SENTENCE: exposure may be LOWERED by any reading at
        any time, and may be RAISED only by a group verdict measured in the
        arena the raise happens in.

        1. NO GROUP VERDICT HAS EVER BEEN SEEN -- the answer is the local
           backing, which is #816's behaviour and is what a single-rank shape,
           a stub rung and every hermetic test get. Unchanged on purpose:
           guessing a floor where no collective exists strands rows for no
           reason (``group_exposure_ceiling`` states this).

        2. A VERDICT EXISTS FOR THIS ARENA -- ``min(local, floor)``, exactly as
           #833 shipped it, and this is the one case that may RAISE. It is
           uniform by construction: the floor IS the group minimum of the
           ranks' own backings in this same reduction, so no rank's local
           reading is below it and every rank computes the same number.

        3. A VERDICT EXISTS BUT WAS MEASURED IN THE OTHER ARENA. Its row count
           is not comparable with this arena's, so ``min(local, floor)`` is no
           longer a bound anyone agreed to -- and the failure is silent,
           because when the stale reading is the WIDER one the ``min`` simply
           stops binding and each rank publishes its own backing. So a stale
           floor may still LOWER (it can only be wrong toward too little
           exposure, which is a capacity loss and not an id a peer cannot map)
           and may never RAISE: the level last published IN THIS ARENA caps
           it. Before anything has been published in an arena there is nothing
           to hold, and the ceiling is then ``min(local, floor)`` as before --
           one round of the old behaviour at most, because the next seam
           ballot stamps this arena and the raise becomes legal again.

           Window 4 segment A is case 3 in both directions and shows why the
           asymmetry is the whole fix. Rounds 1-3: floor 122880 measured
           narrow, backing read wide -- the stale floor bound, and the group
           was uniform. Round 4: floor measured wide, backing read narrow --
           the same ``min`` bound nothing, and 122880/131072/210944 went out.
           Capping by the last published ceiling holds round 4 at 122880,
           which is where rounds 1-3 already were.
        """
        local = max(0, int(backed))
        floor = int(self._group_backed_floor)
        if floor < 0:
            return local
        arena = self._arena_key()
        ceiling = group_exposure_ceiling(local, floor)
        if self._group_floor_arena == arena:
            return ceiling
        held = self._published_exposure.get(arena)
        if held is not None:
            ceiling = min(ceiling, int(held))
        return ceiling

    def publish_group_exposure(self, why: str) -> int:
        """Move this rank's exposed id space to the group's agreed level.

        #839 A+B: THE SINGLE PUBLICATION POINT, and the two halves of the
        window-4 pair are the two directions of this one call.

        DOWN (#839 A) -- no rank may expose an id above the group's poorest
        backing, so a rank that grew locally is brought back to the floor.

        UP (#839 B) -- and this is the direction that did not exist. #834
        splits the grow from the levelling so the expensive half can leave the
        no-return window; the rows it backs then wait for "a later collective
        to raise the level". The only collective that ever raised it ran inside
        the seam's ``tp_to_pp`` cutover, which is downstream of the exposure it
        gates: the pool was too small to flip, so no flip ran, so no levelling
        ran, so the pool stayed too small. Measured, boot_window4B_0823_2116:
        GROW DEFERRED 3 / GROW PAID 2 and 588 ``GROW-DEBT-UNPAID`` lines
        standing at 83968 backed-but-unexposed rows for 32 rounds, with all 207
        abandons reading "pool too small for the live set".

        The ballot this is called from already carries the group's MIN backed
        rows and already runs on every rank on every seam round, on the one
        path every rank reaches unconditionally. So the debt is settled with a
        reduction that is already on the wire: NO NEW COLLECTIVE IS ENTERED,
        which is the constraint that ruled out moving the levelling itself
        (the 2026-08-08 boots 9/10 wedge -- a blocking reduction at a local
        cadence pairing with a peer blocked in a pipeline recv).

        WHY RAISING HERE IS SAFE. ``floor`` is the group minimum of the ranks'
        own ``backed_rows()`` in this same reduction, so every rank has the
        pages behind every id at or below it, and every rank computes the same
        number. It commits nothing: ``reconcile_to`` is an id decision and the
        pages do not move.

        Returns the signed change in exposed rows (0 when nothing moved).
        """
        floor = int(self._group_backed_floor)
        if floor < 0:
            return 0
        self._rebind()
        level = self._exposure_ceiling(self._current_rows())
        if level <= 0:
            return 0
        if self._published_exposure.get(self._arena_key()) == level:
            # NOTHING TO PUBLISH, and skipping is safe HERE specifically.
            # ``reconcile_to`` documents that it has no early return because
            # every rank must end with the same free-list ORDER -- but this
            # call site follows ``normalize_free_lists()``, which sorts on
            # every rank on every seam round whatever this decides. The order
            # invariant is therefore already held by someone else, and re-doing
            # a release-and-re-engage of the whole free list every round to
            # re-establish it would be paying twice.
            return 0
        moved = int(self.level_recovery_to(level))
        self._record_published(level)
        self._group_floor_arena = self._arena_key()
        if moved:
            logger.warning(
                "%s [#839] exposure published (%s): this rank moves %+d "
                "exposed rows to the group's agreed level %d (its own backing "
                "is %d). Exposure is only ever RAISED by a group verdict "
                "measured in the arena it is raised in -- a rank-local grow "
                "may commit pages, never announce them (#839 A) -- and this "
                "is also the payment for a deferred grow, which previously "
                "had no creditor outside the seam (#839 B, #834 crit 13).",
                LOG_PREFIX,
                why,
                moved,
                level,
                int(self._current_rows()),
            )
        return moved

    def clamp_exposure_to_backing(self, why: str) -> int:
        """Never leave the allocator exposing an id with no page behind it.

        #816, and it is the MIRROR of #717/#722. That one put the cap BELOW
        rows that were still live and the next read was an illegal address;
        this one leaves ids exposed ABOVE the rows that are still committed and
        the next write is a device-side assert in the KV writer's bound check.
        Same invariant, two directions:

            highest live row  <=  committed backing  >=  exposed id space

        MEASURED ON METAL, 2026-08-23 00:33:54, the crash this closes::

            PP-ADMISSION verdict=ADMIT ... avail=97385 evictable=320465
            Assertion `index >= 105414 (out of range): set_kv_buffer (MHA)'

        105414 is ``self.size + page_size``, i.e. 105413 committed rows, while
        admission was pricing against 97385 + 320465 = 417850 reachable ones.
        312437 rows of pure exposure, and the first prefill whose tail landed
        up there took the assert.

        WHY A CLAMP AND NOT A WIDER BOUND. The writer's bound is deliberately
        graph-stable (``graph_safe_store_bound``, memory_pool.py:131) and
        widening it would re-admit the silent-corruption band it exists to
        exclude. The id space is the thing that is wrong, so the id space is
        what gets corrected.

        WHY IT CANNOT RE-CREATE #722. It only ever LOWERS exposure toward
        ``_current_rows()`` -- a MEASURED committed count, never a remembered
        one (the #684 lesson) -- and it never lowers the BACKING. If the
        backing already sits below the live set, that is the #722 state and it
        is reported here rather than papered over: capping cannot repair it,
        only a grow can, so this logs and leaves the decision to ``recover``.

        Returns the number of over-exposed rows it withdrew (0 when the state
        was already sound), so callers and tests can assert on the action
        rather than on the absence of a symptom.
        """
        exposed = self.exposed_rows()
        backed = self._current_rows()
        # #833: THE CEILING IS A GROUP QUANTITY, NOT A RANK-LOCAL ONE.
        # ``exposed_rows`` documents that this id space "has to be identical
        # across the group". Capping at the local backing alone -- #816's
        # behaviour -- makes it differ by exactly the amount the ranks' pools
        # differ, which under the mandated uneven vectors is by design and
        # never zero. See ``group_exposure_ceiling`` for the measured cost.
        # #839 A: AND THE FLOOR IT IS COMPARED AGAINST MUST BE A READING OF
        # THE SAME ARENA. ``group_exposure_ceiling`` is still the arithmetic;
        # ``_exposure_ceiling`` is the question of whether the floor in hand is
        # allowed to answer for this layout at all. Comparing a row count from
        # the other layout is how three group-uniform rounds became
        # 122880/131072/210944 in one round on metal.
        ceiling = self._exposure_ceiling(backed)
        stranded = max(0, int(backed) - int(ceiling))
        if stranded != self._stranded_by_group_floor:
            self._stranded_by_group_floor = stranded
            if stranded:
                # NOT a capacity verdict -- a defect report with a price on it.
                logger.warning(
                    "%s group exposure floor (%s): this rank has %d rows "
                    "BACKED but the group's poorest rank has only %d, so %d "
                    "rows are backed-but-unexposable on this rank (#839: the "
                    "second number is the GROUP bound in force, which is the "
                    "ballot's floor when it was measured in this arena and the "
                    "last agreed level when it was not). They are "
                    "withheld because an id above the group floor is one a "
                    "peer cannot map, and the flip's live-slot union would "
                    "have to refuse it -- which is how a single narrow rank "
                    "silently ends every cutover for the rest of a boot "
                    "(#833). This surplus is REAL and reaching it is the "
                    "standing #795 federation debt, not a property of the "
                    "hardware: no rank 'binds' a pool.",
                    LOG_PREFIX,
                    why,
                    int(backed),
                    int(ceiling),
                    stranded,
                )
        over = exposure_over_backing(exposed, ceiling)
        if not over:
            return 0
        live = self._max_live_row()
        # The #722 test stays on the COMMITTED backing, never on the group
        # ceiling: a live row between the ceiling and this rank's own backing
        # is mapped here and is not the #722 crash. Testing it against the
        # ceiling would raise a false unmapped-live-rows alarm on every rank
        # wider than the group's poorest one -- i.e. on almost every rank.
        if live >= 0 and live >= backed:
            # The #722 shape, and it is ALREADY true before this function acts.
            # Say so loudly and separately: clamping to ``backed`` here is
            # still strictly better than leaving 312k rows exposed, but it does
            # not make those live rows addressable again.
            logger.error(
                "%s exposure clamp (%s) found the #722 state underneath: the "
                "highest live row is %d but only %d rows are committed, so "
                "live rows are already unmapped. Clamping exposure to %d "
                "anyway -- it stops NEW ids escaping, it cannot repair the "
                "ones already handed out. A grow, not a cap, is what fixes "
                "this.",
                LOG_PREFIX,
                why,
                live,
                backed,
                backed,
            )
        self._cap.engage(ceiling)
        self._record_published(ceiling)
        logger.warning(
            "%s exposure clamp (%s): the allocator could hand out %d rows "
            "while only %d are committed, so %d rows had no page behind them. "
            "Capped at %d. Leaving them exposed is the #816 crash: the first "
            "write above the backing is a device-side assert in "
            "masked_set_kv_buffer (memory_pool.py:4978), which cost this "
            "instance four boots on 2026-08-22/23.",
            LOG_PREFIX,
            why,
            exposed,
            backed,
            over,
            ceiling,
        )
        return over

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

    def live_floor_rows(self) -> int:
        """The lowest row level this rank's allocator may be capped to.

        #792. The same number ``cap_proposal`` puts on the wire as its second
        field, as a public reading, because the RECOVERY levelling needs it
        too and had no way to ask for it. Returns :data:`_UNBOUNDED_ROWS` when
        the live set cannot be read: an unknown live set is not an empty one,
        and every caller of this reading must decline rather than guess -- the
        same direction ``_shrink_to`` takes when ``_max_live_row`` comes back
        negative.

        IT COSTS ONE LIVE-SET ENUMERATION, which is the same price
        ``cap_proposal`` already pays on every seam round. The callers are the
        cutover legs, once each, so this at most doubles a cost the seam
        already carries -- and the alternative is a cap engaged below the live
        set, which cannot be paid off at any price.
        """
        max_live = self._max_live_row()
        if max_live < 0:
            return _UNBOUNDED_ROWS
        return int(self._floor_rows(max_live))

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
        # #792: AND THE LIMIT WINS HERE TOO, WHICH IT DID NOT.
        #
        # ``collective_cap_target`` already states this law for the seam's
        # agreement -- it returns None when the group's MAX floor is above the
        # MIN capable, because "the poorest rank cannot expose the rows a
        # peer's live set requires". The RECOVERY levelling
        # (``phase_flip_spill.recover_kv_backing``) reaches this same actuator
        # through :meth:`level_recovery_to` and reduced only ``[backed,
        # -backed]``, so it could hand a target the live set forbids -- and
        # this method engaged it without a word.
        #
        # Measured, boot instr12 2026-08-21, on all three ranks::
        #
        #   05:28:17  KV-BACKING released 160 MiB by backing 137233 rows
        #             instead of 161792 (highest live row 136720,
        #             24145 ids withheld from the allocator)
        #   05:28:21  KV-BACKING cap agreement: exposed rows 137233 -> 40960
        #             (group level 40960, backed 49152)
        #   05:28:21  PHASE-FLIP-SPILL KV recovery levelled to the group ...
        #   05:29:28  RuntimeError: Out of memory. Try to allocate 512 tokens.
        #             Available full tokens: 67935 (full_available_size=261
        #             + full_evictable_size_=67674)
        #             EVICTION UNDER-DELIVERED: asked for 512 tokens, the pool
        #             received 94 ... A RESIDENCY CAP IS ENGAGED and is
        #             holding 63641 slot ids
        #
        # 63641 withheld ids is the proof that the cap at the death was the
        # LEVELLED one and not the shrink's: above a cap of 137233 the id
        # space holds only 161792 - 137233 = 24559 ids in total, so 63641 is
        # arithmetically impossible there and perfectly possible above 40960.
        # The levelling had put the cap 95760 rows BELOW the highest live row,
        # and from that instant every id a peel freed above 40960 went
        # straight into ``_withheld``. The tree emptied and the pool was never
        # paid.
        #
        # A cap below the live set is not a conservative cap -- it is an
        # unpayable pool, because the ids the tree is holding are exactly the
        # ids the cap confiscates. Declining costs the group a levelled id
        # space, which the flip's frame ballot then refuses: a lost flip,
        # never a rank, and precisely the trade ``collective_cap_target``
        # already makes one function above.
        #
        # Nothing is touched on the decline -- not the cap, not the free list
        # order. The order stays a function of membership because
        # ``normalize_free_lists`` runs on every rank on every seam round
        # regardless of what this method decides.
        #
        # ONE enumeration, read for both the verdict and the line that
        # explains it: asking twice would double the seam's live-set cost for
        # a log field.
        max_live = self._max_live_row()
        floor = _UNBOUNDED_ROWS if max_live < 0 else int(self._floor_rows(max_live))
        if level < floor:
            logger.error(
                "%s DECLINED to level this rank to %d rows: its live set "
                "needs %d (highest live row %d + margin + admission reserve). "
                "Capping below the live set would withhold every id the radix "
                "tree is holding, so eviction would free the TREE and pay the "
                "POOL nothing -- the pool becomes unpayable and the next "
                "prefill raises out-of-memory with a full evictable tree "
                "(#792). The ranks stay on different id spaces and the seam's "
                "frame ballot refuses the flip until a recovery closes the "
                "gap: a lost flip, never a dead rank.",
                LOG_PREFIX,
                level,
                floor,
                max_live,
            )
            return 0
        ceiling = (
            int(self._rows_at_boot)
            if self._rows_at_boot is not None
            else self._reservation_rows()
        )
        self._cap.release()
        if level < self._reservation_rows():
            self._cap.engage(level)
        # #839 A: AND IT DELIBERATELY DOES NOT RECORD A PUBLISHED LEVEL.
        #
        # ``_record_published`` feeds the "never raise without a fresh in-arena
        # verdict" rule, and that rule is only worth anything if what it
        # defends is a level THE GROUP agreed to. This method is the raw
        # actuator: ``level_recovery_to`` reaches it with a group target, but
        # ``recover``'s own re-engage and every rank-local caller reach it with
        # a rank-local one, and the two are indistinguishable from here.
        # Recording here was written and REMOVED after it turned the window-4
        # reproduction red again -- a rank-local ``reconcile_to(backed)`` had
        # recorded 210944 as "published", which then licensed the very raise
        # the rule exists to refuse. The recording therefore stays with the
        # callers that know the level came from a verdict:
        # ``publish_group_exposure`` and ``_exposure_ceiling``'s own consumers.
        #
        # RESIDUAL, stated rather than assumed: ``apply_cap_agreement`` reaches
        # this method every seam round with ``collective_cap_target``'s level,
        # which is the same group MIN over backed rows the ballot publishes, so
        # the two agree. Where it can still differ is that ``cap_proposal``
        # reads ``_current_rows()`` without rebinding first -- the same
        # arena-blindness #839 closes at the clamp. It can only under-expose
        # from here (``min(target, backed)`` never raises), so it costs
        # capacity and cannot issue an id a peer has not backed. Named in W14b
        # as the next thing to look at if a boot shows one level per rank with
        # the clamp lines uniform.
        # #816: ``level`` is the GROUP minimum and the comparison is against
        # the id-space span, so the no-cap leg exposes every id this rank has
        # -- backed or not. The group level decides what the ranks AGREE on;
        # this rank's own committed backing decides what it may physically
        # write. Both bind, and the tighter one wins.
        self.clamp_exposure_to_backing("after the cap agreement")
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


def flip_pending_from_live_fn(live_fn, armed_fn, active_layout_fn=None) -> tuple:
    """``(rows, max_row_id)`` the flip has parked. ``(-1, -1)`` = UNKNOWN.

    #748: THE IDLE BOX IS NOT UNKNOWN, and reading it as unknown is what
    wedged comp4 (2026-08-18 06:36Z: "no KV provider" x9, "IDLE-LOCK" x5,
    pp_to_tp relief "returned NOTHING ... evicted 0 rows over 0 shrinks").

    ``phase_flip_runtime`` sets ``last_req_extent`` only when
    ``split["req_rows"] > 0``. On an idle box no enumeration ever sees a
    resident request, so the attribute is never set -- and the previous
    version read that absence as UNKNOWN, which made the ceiling refuse
    wholesale. It therefore refused hardest in exactly the case with nothing
    parked and nothing to protect: 407k pending, 0 running, no funding, wedge.

    ``req_rows == 0`` is the #717 ambiguity one level out -- two opposite
    states with one encoding:

    * nothing has EVER been resident (an idle box): nothing to protect;
    * resident but quiesced for the flip (parked): protect it.

    They are separable from data already on hand, which is why this needs no
    new enumeration. If any enumeration ever saw a resident request the sticky
    extent EXISTS and answers. If an enumeration has merely RUN, the split is
    readable and its ``req_rows == 0`` is positive evidence rather than
    absence of it. Only a split that was never readable at all is genuinely
    unknown, and that alone still refuses.
    """
    try:
        if not armed_fn():
            return (0, -1)
    except Exception:  # noqa: BLE001 - an unreadable flip is not an idle one
        return (-1, -1)

    extent = getattr(live_fn, "last_req_extent", None)
    if extent:
        # #802: an extent is a row id, and a row id only means anything in the
        # pool it was enumerated in. The extent survives the cutover (one
        # writer, no clearer), so one enumerated in the PP phase is still here
        # when tp_to_pp arms -- describing a pool the seam has since released.
        #
        # ONLY a POSITIVE mismatch discards it. An untagged extent (old
        # writer, or a layout that would not read) and an unreadable current
        # layout both keep the pre-#802 behaviour of protecting, because the
        # cost of being wrong here is unmapping a row the flip is about to
        # read (#722/#744). Discarding is allowed exactly when both sides are
        # known AND different, which is the one case where the extent
        # provably describes the layout that is NOT backed.
        tag = extent[2] if len(extent) > 2 else None
        active = None
        if active_layout_fn is not None:
            try:
                active = active_layout_fn()
            except Exception:  # noqa: BLE001 - unreadable stays protective
                active = None
        if tag is not None and active is not None and str(tag) != str(active):
            return (0, -1)
        return (int(extent[0]), int(extent[1]))

    split = getattr(live_fn, "last_split", None)
    if split is not None:
        try:
            if int(split.get("req_rows", -1)) == 0:
                # An enumeration ran and found nothing resident. Nothing is
                # parked, so the extent excludes nothing and the rung is free
                # to fund the flip it is armed for.
                return (0, -1)
        except Exception:  # noqa: BLE001 - a malformed split is unknown
            return (-1, -1)

    return (-1, -1)


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

        #746: answered from the controller's ARM-TIME SNAPSHOT
        (``PhaseFlipRuntime.parked_extent``), which is exact -- "the rows
        this flip will pack" is fixed at arm, where the flip measures it.
        The sticky last-enumeration value this replaced was stale by
        construction and absent for a flip that armed before any
        enumeration ran.

        Outside a flip this answers "nothing parked" unconditionally, so
        the rung stays fully live and #688's funding path is untouched.

        While armed with no readable snapshot (the arm-time measurement
        failed), the honest answer is UNKNOWN -- which #748's exclusion
        ceiling turns into its one remaining wholesale refusal.

        #808 MERGE RESOLUTION, and the reasoning is recorded because a
        first attempt got it wrong. #746 and #802 fixed the SAME wholesale
        refusal from opposite ends, on branches that never met:

          * #802 made a STALE extent survivable, by discarding one that
            provably belongs to the RELEASED layout -- the case that priced
            a floor above the resident pool's whole cap (348106 against
            212992 rows; measured again 2026-08-22 as 354774 against
            204800) and left the seam permanently unfundable.
          * #746 removes the staleness AT THE SOURCE: the extent is
            measured at ARM, by the controller that decides to arm, so it
            cannot describe a layout the flip has already left.

        The tempting resolution -- snapshot first, layout-tagged sticky
        value as fallback -- was tried and is WRONG. #746 asserts
        structurally that the sticky channel is gone from the writer, and a
        fallback re-opens the stale path it exists to close. It also
        misreads #808: that defect was never "the snapshot was unreadable",
        it was "the sticky value was authoritative and described the other
        pool". Keeping the sticky value to soften an error case would have
        re-introduced the actual failure mode to guard a rarer one.

        RESIDUAL, NAMED RATHER THAN PATCHED: an arm-time measurement that
        fails still yields UNKNOWN, and UNKNOWN-while-armed still closes
        the rung wholesale (``_parked_ceiling`` returns -2). That path is
        now the only remaining route into #808's shape. It is not softened
        here, because softening it means trusting an extent nobody
        measured; if it is ever observed on metal the fix belongs at the
        snapshot, not at this reader.
        """
        if not _flip_armed():
            return (0, -1)
        rt = getattr(scheduler, "phase_flip_runtime", None)
        snap = None
        if rt is not None:
            try:
                snap = rt.parked_extent()
            except Exception:  # noqa: BLE001 - unreadable is UNKNOWN, not empty
                snap = None
        if snap is None:
            return (-1, -1)
        return (int(snap[0]), int(snap[1]))

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
