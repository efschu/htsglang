"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


# ---- #924D: the per-request slot trail that decides #1190 ----------------
#
# #1190 is "a long-context probe answered with an OLDER probe's answer" (O-1
# FAIL on boots 8 and 10). Two carriers can produce it and the boot logs cannot
# tell them apart: mamba slot ALIASING (#924 -- the tree resumes from a slot
# alloc() has already handed to somebody else) or a host backup filed under the
# wrong key. The discriminator is the slot ID, printed once per request at each
# station of its state's life. If the B probe's decode slot equals the slot
# A1's tree node still names, #1190 IS the #924 sibling; if it does not, #1190
# is a separate carrier and stays open.
#
# BOUNDED BY CONSTRUCTION -- per (rid, station), never per step, with a global
# cap. The denominator is printed with the cap line so a missing rid can never
# be read as "that station was not reached".
#: #1467: THE SLOT TRAIL SYNCHRONISES THE GPU.  `note_924d` and
#: `_note_slot_event` call `.tolist()` on a CUDA index tensor, and `free()`
#: indexed with a boolean mask -- each is a device sync in the SCHEDULER
#: thread, which then waits for every prefill kernel already queued.
#: MEASURED (py-spy, boot weg2xsn223, PP0, 90 s over three 100k prefills):
#: 284 of 3516 samples inside `_do_alloc` at the trail line, reached from
#: stash_chunked_request -> cache_unfinished_req -> _alloc_mamba_slot; the
#: #1466 PASS-STALL stamp read schedule_ms=350-430 three times per request
#: change (the whole seconds without a forward on all P ranks).  The trail
#: is opt-in (SGLANG_MAMBA_SLOT_TRAIL=1); the double-free refusal stays,
#: evaluated through a CUDA event once the mask is complete, never by
#: waiting for it.
SLOT_TRAIL_ENV = "SGLANG_MAMBA_SLOT_TRAIL"
_SLOT_TRAIL = os.environ.get(SLOT_TRAIL_ENV, "") == "1"
_924D_SEEN: set = set()
_924D_SEQ = 0
_924D_CAP = 8192
_924D_SUPPRESSED = 0


def note_924d(station: str, *, rid=None, slot=None, node_id=None, extra: str = "", dedup: bool = True) -> None:
    """One ``#924D`` line per (subject, station). Never raises.

    THE SUBJECT IS THE RID WHERE THERE IS ONE AND THE NODE OTHERWISE, and that
    is not a convenience: ``write_backup`` carries no request (it is driven off
    the tree, not off a batch), so keying the backup station on the rid would
    collapse EVERY backup in the process into a single ``rid=None`` line and
    the trail would silently lose the station that names which node holds which
    slot -- the very join #1190 needs.
    """
    global _924D_SUPPRESSED
    if not _SLOT_TRAIL:
        return  # #1467: opt-in -- `.tolist()` on a CUDA slot is a device sync
    try:
        subject = str(rid)[:12] if rid is not None else f"node{node_id}"
        # #Q0-trail: LIFECYCLE STATIONS MUST NOT DEDUP. The (subject, station)
        # key prints ONE line per pair, which is right for a census and WRONG
        # for a trail: a request that allocates a ping-pong buffer TWICE shows
        # one alloc line, and the second -- the one that strands the first
        # pair -- is silently dropped. Measured on boot weg2sn5r: the trail
        # read as complete (alloc [27,28] ... free [9,8]) while the write that
        # swapped them was suppressed by this very key.
        global _924D_SEQ
        if dedup:
            key = (subject, station)
        else:
            _924D_SEQ += 1
            key = (subject, station, _924D_SEQ)
        if key in _924D_SEEN:
            return
        if len(_924D_SEEN) >= _924D_CAP:
            _924D_SUPPRESSED += 1
            if _924D_SUPPRESSED in (1, 100) or _924D_SUPPRESSED % 10000 == 0:
                logger.warning(
                    "#924D CAP REACHED: %d station event(s) suppressed after "
                    "%d distinct (rid, station) pairs. Absence of a rid below "
                    "this line is the cap, not the station.",
                    _924D_SUPPRESSED,
                    _924D_CAP,
                )
            return
        _924D_SEEN.add(key)
        if slot is not None and hasattr(slot, "reshape"):
            slot = [int(x) for x in slot.reshape(-1).tolist()]
        logger.info(
            "#924D station=%s rid=%s mamba_slot=%s node=%s%s",
            station,
            subject,
            slot,
            node_id,
            f" {extra}" if extra else "",
        )
    except Exception:  # noqa: BLE001 - an instrument may never break the pool
        pass


class MambaSlotDoubleFree(RuntimeError):
    """A Mamba state slot was returned to the free list twice.

    #924. Named rather than generic because the free list is a bare
    ``torch.cat``: a duplicate is REPRESENTABLE, and the next ``alloc`` hands
    the same state slot to two requests. Two requests sharing a Mamba state is
    a wrong answer that never crashes -- the class the on-idle ledger only ever
    saw minutes later, as an unattributable ``available=23`` on a 20-slot pool,
    with the duplicate already collapsed by the diagnosis's own ``set()``.
    """


class MambaSlotAllocator:
    """Manages the free-list of Mamba pool slot indices.

    Unlike ``BaseTokenToKVPoolAllocator`` which is designed for per-token KV
    pages, Mamba slots are request-level (typically 1 slot per request).
    We keep the interface minimal and do NOT inherit the KV base class.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        # Active preallocated batch for `alloc_group_begin` / `alloc_group_end`.
        # When non-None, `alloc(1)` consumes the next slot from this iterator
        # instead of calling `_do_alloc(1)` per request. Reset to None outside
        # a group window so `alloc` falls through to the per-call path.
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Identity to ``available_size`` for the
        static pool (slot-count and byte-coordinated views coincide); the shared
        ``UnifiedMambaSlotAllocator`` overrides it with the byte-coordinated view.
        Lets ``alloc_req_slots`` call it uniformly without a getattr fallback."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        # #1051b (side finding, own class -- a LEAK, not a double free):
        # dropping a still-open group's iterator discards its un-consumed
        # slots WITHOUT returning them. They stay `slot_used=True` and are on
        # no free list, i.e. exactly the `leaked_mamba_pages` shape the on-idle
        # ledger reports minutes later with nothing left to attribute it to.
        # The single production caller (scheduler.py:10002) pairs begin/end
        # under a carried refusal (#791), so this is unreachable today -- but
        # an exception escaping between the two makes it reachable without any
        # edit to this file, and the correct answer is not "assume the pair".
        self.alloc_group_end()
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        self._drain_double_free_checks()  # #1467: read completed answers, never wait
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        self.slot_used[select_index] = True
        self._note_slot_event(select_index, "ALLOC")
        return select_index

    # ---- #1033b: OWNERSHIP PROVENANCE, the half #924's instrument was missing.
    #
    # `_refuse_double_free` names its stack "Releasing caller" -- and that is the
    # SECOND releaser, the one that merely arrived last. The FIRST releaser, who
    # actually let go of a slot it did not own (or failed to clear the reference
    # that let someone else let go again), was recorded NOWHERE. Six boots of
    # this defect (1046cut 21, 1048fix 24, 1049n9 12, 1050fix 6, 1050dev 42,
    # 1033edge 3) produced 108 events and not one of them names the offender.
    # Same class as the health-check line that named the detokenizer: the
    # instrument names the visible party, not the responsible one.
    #
    # AND IT MUST DISTINGUISH TWO DIFFERENT DEFECTS THAT LOOK IDENTICAL HERE:
    #   * plain double free  -- freed twice with no alloc in between; the second
    #     releaser holds a reference the first should have cleared.
    #   * USE-AFTER-RECYCLE  -- freed, RE-ALLOCATED to someone else, then freed
    #     again by the first owner's stale reference. This one is far worse: the
    #     slot is live for its new owner at the moment it is handed back to the
    #     free list, so alloc() will shortly serve it to a THIRD request while
    #     the second is still reading it. Distinguished by whether an ALLOC event
    #     sits between the two releases, which is why alloc is recorded too.
    #
    # Strings and ints only -- no tensors, no device work. The mamba pool is ~10
    # slots, so this is a dict of ~10 short entries; it cannot grow with load.
    _PROV_FRAMES = 9

    def _note_slot_event(self, index, kind: str) -> None:
        if not _SLOT_TRAIL:
            return  # #1467: opt-in -- `index.tolist()` is a device sync
        try:
            book = getattr(self, "_slot_provenance", None)
            if book is None:
                book = {}
                self._slot_provenance = book
            seq = getattr(self, "_slot_event_seq", 0) + 1
            self._slot_event_seq = seq
            if kind == "FREE":
                # #1429 (xsn190 py-spy): `traceback.format_stack` reads source
                # lines through linecache on EVERY free, and the idle PP loop
                # frees a probe slot ~1000x/s (alloc_group_end) -- the main
                # thread spent its idle time here. A raw frame walk records the
                # same provenance (file:line function) at microsecond cost.
                frames = []
                f = sys._getframe(1)
                while f is not None and len(frames) < self._PROV_FRAMES:
                    frames.append(f"  {f.f_code.co_filename}:{f.f_lineno} {f.f_code.co_name}")
                    f = f.f_back
                where = "\n".join(reversed(frames))
            else:
                where = ""
            for slot in index.tolist() if hasattr(index, "tolist") else [index]:
                slot = int(slot)
                prev = book.get(slot)
                book[slot] = {
                    "kind": kind,
                    "seq": seq,
                    "where": where,
                    "prev_kind": None if prev is None else prev["kind"],
                    "prev_seq": None if prev is None else prev["seq"],
                    "prev_where": None if prev is None else prev["where"],
                }
        except Exception:  # noqa: BLE001 - provenance may never break the pool
            pass

    def _describe_slot_history(self, slot: int) -> str:
        book = getattr(self, "_slot_provenance", None)
        if not book or int(slot) not in book:
            return (
                "  slot %s: NO PROVENANCE RECORDED. It was neither allocated nor "
                "freed through this allocator since the last clear() -- so the "
                "reference the second releaser holds did not come from here."
                % slot
            )
        e = book[int(slot)]
        verdict = (
            "USE-AFTER-RECYCLE"
            if e["kind"] == "ALLOC"
            else ("DOUBLE FREE (no alloc in between)" if e["kind"] == "FREE" else "?")
        )
        head = (
            "  slot %s: %s. Last event before this release was %s (#%s); the "
            "event before that was %s (#%s)."
            % (slot, verdict, e["kind"], e["seq"], e["prev_kind"], e["prev_seq"])
        )
        if e["kind"] == "FREE" and e["where"]:
            return head + "\n  FIRST RELEASER (this is the offender):\n" + e["where"]
        if e["kind"] == "ALLOC" and e["prev_kind"] == "FREE" and e["prev_where"]:
            return (
                head
                + "\n  The slot was RE-ALLOCATED after that free and is LIVE for "
                "its new owner right now. Releaser BEFORE the re-alloc:\n"
                + e["prev_where"]
            )
        return head

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        # Ids outside the ledger's span (the ``-1`` sentinel the ping-pong
        # track buffer carries, filtered at memory_pool.py:2213 but not
        # everywhere) would WRAP a bool tensor and answer the membership
        # question about the wrong slot. They are excluded from the ledger
        # rather than answered wrongly; the free list itself is unchanged.
        in_ledger = (free_index >= 0) & (free_index < self.slot_used.numel())
        # #1467: no boolean-mask indexing here (a device sync); out-of-ledger
        # ids are routed to slot 0, the unused dummy row (free_slots starts
        # at 1), and masked out of the double-free question.
        safe = torch.where(in_ledger, free_index, torch.zeros_like(free_index))
        self._defer_double_free_check(safe, in_ledger)
        self.slot_used[safe] = False
        # #1033b: recorded AFTER the ledger update, so the book holds only
        # releases that genuinely flipped a slot True->False. A refused release
        # never becomes somebody's "first releaser".
        self._note_slot_event(safe, "FREE")
        if getattr(self, "_phase_limit", None) is not None:
            # H95c: a slot above the phase's limit goes back to the withheld
            # set, never to the free list (unreachable while the limit holds:
            # nothing above it is handed out)
            self._split_phase_withheld(torch.cat((self.free_slots, free_index)))
            return
        self.free_slots = torch.cat((self.free_slots, free_index))

    # ---- H95c: the D phase's slot limit (weg2/d_seat_vram.py) ---------------
    #
    # In a D phase of n < --d-bs seats only slots 1..L(n) have pages behind
    # them (the saver's span map). The allocator is what makes that safe: it
    # hands out nothing above L(n), so no kernel can index an unmapped slot.
    # The limit is a function of replicated inputs (n, the cap, this pool's
    # size and its ownership ledger), so every rank withholds the same slots.

    def _split_phase_withheld(self, candidates: torch.Tensor) -> None:
        limit = int(self._phase_limit)
        prev = getattr(self, "_phase_withheld", None)
        if prev is not None and prev.numel():
            candidates = torch.cat((candidates, prev))
        keep = candidates <= limit
        self.free_slots = candidates[keep]
        self._phase_withheld = candidates[~keep]

    def set_phase_limit(self, limit: Optional[int]) -> bool:
        """Hand out only slots ``1..limit`` (None or >= size: all of them).
        False -- and nothing changed -- when a slot above ``limit`` is in use:
        its state is live and must stay reachable."""
        if limit is None or int(limit) >= int(self.size):
            withheld = getattr(self, "_phase_withheld", None)
            if withheld is not None and withheld.numel():
                self.free_slots = torch.cat((self.free_slots, withheld))
            self._phase_withheld = None
            self._phase_limit = None
            return True
        limit = max(1, int(limit))
        self._drain_double_free_checks(wait=True)
        if bool(self.slot_used[limit + 1:].any()):
            return False
        self._phase_limit = limit
        self._split_phase_withheld(self.free_slots)
        return True

    @property
    def phase_limit(self) -> Optional[int]:
        return getattr(self, "_phase_limit", None)

    @property
    def phase_withheld_slots(self) -> int:
        """H95d: slot ids the phase's limit holds out of the free list.

        They are free (no owner, ``slot_used`` False -- ``set_phase_limit``
        refuses while one above the limit is live) but not ``available``: they
        have no pages. Without a term of their own the idle ledger reads them
        as a leak -- boot fnFL2h91bb2 died on it at the first idle after the
        first n=1 wake, all three ranks, ``[mamba] total=38, available=7``,
        ``leaked_mamba_pages`` = exactly 8..38. The same NAMED POSTEN as the
        KV pool's ``residency_withheld_slots`` (#656/#790). Replicated: a
        function of the limit and of this pool's ledger, like the limit."""
        withheld = getattr(self, "_phase_withheld", None)
        return 0 if withheld is None else int(withheld.numel())

    def phase_withheld_ids(self) -> list:
        """The ids behind ``phase_withheld_slots`` (a list, duplicates kept --
        the idle census asks the double-free question over them too)."""
        withheld = getattr(self, "_phase_withheld", None)
        return [] if withheld is None else [int(v) for v in withheld.tolist()]

    def _defer_double_free_check(self, safe: torch.Tensor, in_ledger: torch.Tensor) -> None:
        """#1467: the #924 double-free question, asked WITHOUT waiting.

        The mask is computed on the device now; its answer is read only once a
        CUDA event recorded behind it reports complete (`query()`), i.e. never
        by stalling the scheduler thread behind queued prefill kernels.  Every
        `free()`/`alloc()` first drains the completed entries, so a refusal is
        delayed by at most the passes the mask spent in flight -- still ONE
        boot's attribution with the releaser's stack, recorded at release time.
        On a CPU allocator (tests) the answer is read immediately.
        """
        already_free = self.slot_used[safe].logical_not() & in_ledger
        stack = "".join(traceback.format_stack(limit=12)[:-2])
        pend = getattr(self, "_1467_pending", None)
        if pend is None:
            pend = self._1467_pending = []
        if already_free.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
            pend.append((ev, already_free, safe, stack))
        else:
            pend.append((None, already_free, safe, stack))
        self._drain_double_free_checks()

    def _drain_double_free_checks(self, wait: bool = False) -> None:
        pend = getattr(self, "_1467_pending", None)
        if not pend:
            return
        keep = []
        for ev, already_free, safe, stack in pend:
            if ev is not None and not wait and not ev.query():
                keep.append((ev, already_free, safe, stack))
                continue
            if ev is not None and wait:
                ev.synchronize()
            if bool(already_free.any()):
                self._refuse_double_free(safe[already_free], stack=stack)
        self._1467_pending = keep

    def _refuse_double_free(self, free_index: torch.Tensor, stack: Optional[str] = None) -> None:
        """#924: say WHO returned a slot that is already free, then raise.

        WHY NOT A SILENT DEDUP. Dropping the duplicate here would keep the
        free list sound and leave the second releaser in place -- the surplus
        would stop being observable while the ownership defect that produced it
        went on running. That is symptom treatment, and the on-idle ledger,
        which is the only thing that ever noticed, would go quiet.

        WHY THE CALLER'S STACK. There are twenty-odd call sites that reach
        ``mamba_allocator.free`` (regular finish, radix eviction, the flip's
        slot union, the HiCache ack drain, abort/retraction, the streaming
        session). The 2026-08-27 specimen inflated by ONE slot at a time and
        the death came minutes later at ``on_idle``, by which point nothing in
        the process still knew which of those it was. The first offender's
        stack is recorded, so ONE boot attributes it instead of three.

        WHY IT RAISES. The sibling with the same job -- ``HostKVCache.free``
        (``pool_host/base.py:359-380``, the #905 hardening) -- logs the same
        diagnosis and then asserts. This one is worse if allowed to continue:
        a doubly-freed KV row is caught by the row-ownership authority, while a
        doubly-freed Mamba slot is handed to two requests and answers both.
        """
        # #1033b NAMED GAP -- THIS GUARD SEES ONLY HALF OF ITS OWN SUBJECT, and
        # the half it misses is the worse one. Measured at the desk 2026-08-31
        # (devtools/check_1033b_mamba_provenance.py case 2):
        #
        #   alloc slot 1 -> owner A frees it -> the pool RE-ALLOCATES slot 1 to
        #   owner B -> A's stale reference frees it again.
        #
        # At that second free `slot_used[1]` is True, because B legitimately
        # holds it. `already_free.any()` is therefore False, this function
        # returns without a word, and the slot goes back on the free list WHILE
        # B IS STILL READING IT. alloc() then hands it to C. That is exactly the
        # outcome the docstring below says this guard exists to prevent -- "a
        # doubly-freed Mamba slot is handed to two requests and answers both" --
        # and it happens silently, with no log line and no raise.
        #
        # CONSEQUENCE FOR EVERY #924 COUNT EVER REPORTED: the events in the boot
        # logs (108 across six boots) are only the subset where nothing was
        # re-allocated in between. They are a LOWER BOUND on the ownership
        # defect, never a measure of it. Do not read "#924 = 0" as "no double
        # free"; read it as "none of the visible shape".
        #
        # WHY IT IS NOT CLOSED HERE: catching it needs the RELEASER'S identity
        # at the call site -- a per-slot generation token handed out by alloc()
        # and presented at free() -- which touches all eleven callers of this
        # method. That is a real design change, not a guard tweak, and it is
        # FILED rather than guessed at. What IS built is the provenance book
        # above, which names the first releaser for the visible shape; rooting
        # that is the prerequisite for deciding whether the token is needed.
        if free_index.numel() == 0:
            return
        if stack is None:  # direct call (tests, old callers): the ids passed are the offenders
            already_free = self.slot_used[free_index].logical_not()
            if not bool(already_free.any()):
                return
            free_index = free_index[already_free]
        offenders = free_index.tolist()
        n = getattr(self, "_double_free_count", 0) + 1
        self._double_free_count = n
        trace = stack if stack is not None else "".join(traceback.format_stack(limit=12)[:-2])
        if getattr(self, "_first_double_free_trace", None) is None:
            self._first_double_free_trace = trace
        logger.error(
            "#924 MAMBA SLOT DOUBLE FREE: slot(s) %s are already on the free "
            "list and were returned again. The free list is a bare torch.cat, "
            "so a second release is representable and inflates "
            "available_size() by exactly the duplicate count -- which is how "
            "this surfaced, as available=%d on a %d-slot pool at an on_idle "
            "check minutes later, with the duplicate already collapsed by the "
            "diagnosis's own set(). (%d so far.)\n"
            "#1033b OWNERSHIP PROVENANCE -- BOTH holders, because naming only "
            "the second one is what made this unrootable for six boots:\n%s\n"
            "SECOND RELEASER (the one that merely arrived last -- this is what "
            "the old 'Releasing caller:' line showed, and it is NOT normally "
            "the offender):\n%s",
            offenders,
            len(self.free_slots) + len(offenders),
            self.size,
            n,
            "\n".join(self._describe_slot_history(s) for s in offenders),
            trace,
        )
        raise MambaSlotDoubleFree(
            f"mamba slot(s) {offenders} were already free; a second release "
            f"would let alloc() hand one state slot to two requests"
        )

    def clear(self):
        # #1051b: a flush/reset invalidates every outstanding slot id, so an
        # open group iterator must die with the generation that minted it.
        # Surviving it would let `alloc()` hand out an id from the OLD
        # free-list generation and `alloc_group_end` return it into the FRESH
        # one -- a duplicate on a free list that is a bare `torch.cat` and
        # cannot represent uniqueness. `UnifiedMambaSlotAllocator.clear`
        # already does this; the static allocator only did it in `__init__`.
        self._alloc_iter = None
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        # #1033b: the provenance book describes THIS generation of the pool. A
        # clear() resets ownership wholesale, so a pre-clear releaser must never
        # be reported as the first releaser of a post-clear slot.
        self._slot_provenance = {}
        self._slot_event_seq = 0
        # #924 ownership ledger, the shape ``HostKVCache`` already carries:
        # ``free_slots`` alone cannot answer "is this slot already free?"
        # without an O(n) membership scan, and the answer is what separates a
        # legitimate release from a double free. Sized ``size + 1`` because
        # slot 0 is reserved and never handed out; it stays False forever.
        self.slot_used = torch.zeros(
            self.size + 1, dtype=torch.bool, device=self.device
        )
        # H95c: a flush inside a D phase keeps the phase's slot limit -- the
        # slots above it still have no pages.
        if getattr(self, "_phase_limit", None) is not None:
            self._phase_withheld = None
            self._split_phase_withheld(self.free_slots)
