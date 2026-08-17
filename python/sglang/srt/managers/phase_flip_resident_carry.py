# Copyright 2023-2024 SGLang Team
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
"""#631 defect J.3: carry the RESIDENT DECODE SET across the phase flip.

THE DEFECT, and it is one line of omission
------------------------------------------
The cutover swaps the stack and the scheduler topology, and its step 6
calls ``Scheduler.init_pp_loop_state()`` to re-initialise the PP loop
arrays for the new ``pp_size``. That function REBINDS ``running_mbs`` to a
fresh list of empty ``ScheduleBatch`` objects. Under ``event_loop_pp``,
``running_mbs`` IS the rank's resident decode set -- ``running_batch`` and
``last_batch`` are per-slot aliases rebound at the top of every slot
iteration (#631 J.1). So step 6 drops every resident request on the floor:
the ``Req`` objects become unreachable, and with them go

  * their KV rows, which stay allocated because nothing ever completes or
    frees them -- the "leaked_full_pages" the idle invariant checker
    reports one pass later; and
  * their mamba/GDN slot locks, which stay held -- ``x_lru
    .full_lock_ref=1`` with the tree idle, which trips the Mamba Radix
    sanity check and SIGQUITs the group.

TWO SYMPTOMS, ONE OMISSION. Measured at a real cutover, 2026-08-09
02:36:05-07Z, with the idle leak check demoted to WARN so the accounting
crash could not mask the result:

    POOL CENSUS at-arm        cur_slot_reqs=1 resident_reqs=1 slots=[0]
    POOL CENSUS pre-cutover   cur_slot_reqs=1 resident_reqs=1 slots=[0]
    POOL CENSUS post-cutover  cur_slot_reqs=0 resident_reqs=0 slots=[]

Present and enumerated right up to the cutover, gone immediately after.
The KV MOVE is innocent -- it reported balanced cells and (since J.1)
enumerates every resident slot. What was missing is that nothing carried
the REQUESTS.

WHY THE FIX LIVES IN ``init_pp_loop_state`` AND NOT IN THE CUTOVER
-----------------------------------------------------------------
Because the cutover is not the only place that calls it.
``event_loop_pp`` calls ``init_pp_loop_state()`` at its own entry, and the
TP->PP leg re-dispatches INTO that loop right after the cutover. A carry
installed only in the cutover would therefore be wiped a few microseconds
later by the loop it was installed for. Patching both call sites is two
chances to be wrong; the honest form is one rule at the one function that
destroys the state:

    INIT MUST NEVER DESTROY A RESIDENT REQUEST.

Harvest before the rebind, re-seed after it. At boot nothing is resident,
so the rule is a no-op and the default path is bit-for-bit unchanged --
which is also why it is safe to state unconditionally rather than as a
special case for the flip.

WHAT DOES *NOT* HAVE TO BE CARRIED, AND WHY
-------------------------------------------
No KV row, no ``req_to_token`` row, and no mamba slot id is rewritten
here, and that is not an oversight: ``phase_flip_boot`` step 5a REBINDS
the two stacks' ``req_to_token`` and ``req_index_to_mamba_index_mapping``
to the SAME tensors, and both layouts key on the same global slot ids.
Every per-request handle a carried ``Req`` holds -- ``req_pool_idx``, its
KV rows, its mamba slot -- therefore stays valid across the layout swap by
construction. The bytes behind those ids are what the KV and GDN movers
relocate. Rebuilding the batch from its requests would re-derive state
that is already correct, and would install a second source of truth for
it (the J.2 lesson: enumeration and invariant must share one basis).

WHY IT IS SAFE TO MOVE A BATCH AT THIS INSTANT
----------------------------------------------
The flip only commits at a QUIESCENT boundary: no forward in flight, no
half-written chunk, the previous batch drained (see
``build_flip_quiescence_fn``). Every resident request is therefore
SETTLED -- its last forward's result applied, its seq_lens and committed
KV consistent -- which is exactly the state in which a ``ScheduleBatch``
may be re-homed. Outside that boundary this would be unsound, so the
carry deliberately offers no other entry point.

THE ONE MERGE, AND THE DUPLICATION HAZARD IT AVOIDS
---------------------------------------------------
``pp_loop_size`` differs per phase (``pp_size + pp_async_batch_depth``),
so a PP phase's three resident slot batches have nowhere to go in a TP
phase's single-slot array: the resident batches must be MERGED. Merging
is done exactly once, here, with the scheduler's own ``merge_batch``
primitive -- and merging is NOT idempotent, because ``merge_batch``
extends ``self.reqs`` in place. A second merge of the same list would
enter the same ``Req`` twice, which is duplicate rows, a double free and
silently corrupt context. Two defences, both cheap:

  * harvest dedupes by batch IDENTITY (``id(batch)``), so the same object
    reached through ``running_mbs[0]`` and ``running_batch`` is one entry;
  * harvest REFUSES LOUDLY if one ``Req`` appears in two DISTINCT
    harvested batches, rather than merging them into a duplicate.

``last_mbs`` is deliberately NOT a harvest source. At quiescence it can
hold nothing that ``running_mbs`` does not: a slot's ``last_mbs`` entry is
set only right after that slot's batch result is processed, and the same
slot's ``mbs`` entry is then still non-empty -- which the quiescence
predicate refuses. It is checked as an INVARIANT instead (see
``assert_no_orphan_resident_reqs``): a request visible only through
``last_mbs`` at a cutover means the quiescence predicate is wrong, and
that must be loud, not silently absorbed.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-CARRY"

#: How far above ``max_running_requests`` the resident set may legally sit
#: (#682). Not slack, and not a fudge factor: the scheduler MAINTAINS this
#: bound, and the guard used to assert a tighter one than the code it was
#: watching.
#:
#: ``Scheduler._get_new_batch_prefill_raw`` suspends the running-request cap
#: for as long as a chunked prefill is in flight, and names the reason:
#:
#:     # Ignore the check if self.chunked_req is not None.
#:     # In PP case, chunked requests (or dllm requests) can start in one
#:     # microbatch and end in another microbatch, so the
#:     # max_running_requests per microbatch should not be strict. Instead,
#:     # we should always allow chunked requests to be added, otherwise,
#:     # there will be a memory leak.
#:
#: EXACTLY ONE, because the scheduler holds exactly one: ``self.chunked_req``
#: is a single slot, asserted empty (``assert self.chunked_req is None``)
#: before a new one is stashed. So ``cap + 1`` is the true bound and
#: ``cap + 2`` is still a corrupted resident set -- which is what keeps this
#: a repair of defect M's ceiling rather than a removal of it.
#:
#: ADDITIVE, NOT PROPORTIONAL: ``--max-running-requests 1`` is a real
#: single-stream configuration and its chunked prefill is precisely the case
#: the scheduler's bypass exists for.
#:
#: UNCONDITIONAL, AND THAT IS THE GROUP-UNIFORMITY ARGUMENT. Gating this on
#: ``scheduler.chunked_req is not None`` would be tighter and would be a
#: hang: that flag is per-rank scheduler state, the PP ranks sit at
#: different pipeline positions, and at one cutover instant a peer can hold
#: it while this rank has just cleared it -- so the same resident set would
#: be legal on one rank and fatal on another. Deriving the ceiling from
#: ``max_running_requests`` alone keeps the verdict replicated:
#: ``Scheduler.init_admission_limiter`` documents that value as "uniform
#: across ranks by construction: every input to `ceiling` is min-reduced
#: before it gets here". No collective is added.
IN_FLIGHT_CHUNKED_ALLOWANCE = 1


class ResidentCarryError(RuntimeError):
    """A resident request would have been dropped, duplicated or orphaned.

    Always raised BEFORE the damage: every caller checks on the way into
    or out of the topology swap, where the alternative is a stranded KV
    page and a stranded mamba lock discovered a pass later, with the
    evidence already stale (which is precisely how J.3 cost three boots).
    """


# No batch on this server holds more requests than max_running_requests plus
# IN_FLIGHT_CHUNKED_ALLOWANCE (5 on the 4-request recipe), and no plausible
# configuration approaches this number. It exists only to separate "a request
# list" from "an object that is not one", cheaply, without trusting the caller
# to pass a scheduler. The scheduler-aware, much tighter bound lives in
# ``harvest_resident_batches``.
IMPLAUSIBLE_RESIDENT_REQS = 65536


def _reqs_of(batch) -> List:
    """The requests resident in ``batch``, or a loud refusal.

    DEFECT M (2026-08-09 21:47:29Z) -- the reason this is not a one-liner
    -------------------------------------------------------------------
    The body used to be ``list(getattr(batch, "reqs", []) or [])``. On
    metal it once returned 10485760 entries, and once 12582912. Those are
    not request counts: they are exactly 10 MiB and 12 MiB expressed in
    bytes (10 x 2^20, 12 x 2^20). Something that is not a request list had
    been left reachable as a batch's ``reqs``.

    The same wrong number then did two separate kinds of damage, which is
    why the guard belongs HERE and not at either consumer:

      * ``Scheduler.maybe_arm_phase_policy`` sums these lengths into
        ``running_bs``, so a genuinely IDLE instance armed a pp_to_tp
        flip; and
      * the cutover handed the same resident set to
        ``arm_draft_bootstrap``, whose ``committed_slots`` allocates one
        tensor PER REQUEST. Against a claimed 10.5 M requests the rank
        never came back: host RAM reached 112 GiB of 120 GB, the cgroup
        OOM killer took rank 0 (exit code -9), and ranks 1 and 2 spun on
        a broken TCPStore pipe until the instance was gone.

    THE ORDER OF THE CHECKS IS THE FIX. The hang happened INSIDE that
    ``list()`` call -- materialising ten million elements is itself the
    damage -- so any validation written after it is validation that never
    runs. This checks the TYPE first, and then a length that is O(1) to
    read, and only then materialises.

    NOT A CLAMP, deliberately. Substituting a plausible count for a wrong
    one would let the flip proceed against a resident set nobody can name,
    leaving the leaking handle in place and moving the next symptom
    further from its cause. This names the offending object's type and
    refuses.
    """
    reqs = getattr(batch, "reqs", None)
    if reqs is None:
        return []
    if isinstance(reqs, (list, tuple)):
        return list(reqs)

    # Anything else is the defect-M shape. Report what it actually is --
    # the type and, when it is cheap and safe, the length -- because the
    # whole point of this refusal is to name the object that the next
    # reader has to go and fix.
    try:
        size = len(reqs)  # type: ignore[arg-type]
        size_note = f"len={size}"
        if size and size % (1 << 20) == 0:
            size_note += f" (= {size // (1 << 20)} MiB in bytes)"
    except TypeError:
        size_note = "no len()"
    raise ResidentCarryError(
        f"{LOG_PREFIX} batch {type(batch).__name__} has a 'reqs' attribute "
        f"that is not a request list: type={type(reqs).__name__}, "
        f"{size_note}. Refusing to iterate it. This is defect M: the "
        f"resident harvest must never materialise an object it cannot "
        f"identify, because the cutover allocates one tensor per resident "
        f"request."
    )


def _is_resident(batch) -> bool:
    return batch is not None and len(_reqs_of(batch)) > 0


def harvest_resident_batches(scheduler) -> List:
    """Every batch object holding resident requests on this rank.

    Sources, in authority order: the PP slot array ``running_mbs`` (the
    resident set under ``event_loop_pp``) and ``running_batch`` (the
    resident handle of every non-PP loop). Deduplicated by object
    identity, because ``running_batch`` is normally an ALIAS of one of the
    slots and merging an object into itself would double its requests.
    """
    out: List = []
    seen_batches: Set[int] = set()

    # The tight, scheduler-aware half of the defect-M guard. ``_reqs_of``
    # can only ask "is this a request list at all"; here the real ceiling
    # is known, so a batch claiming more requests than the server is ever
    # allowed to run concurrently is refused BY SLOT NAME. A list of the
    # right type but an impossible length is still a corrupted resident
    # set, and it reaches ``committed_slots`` just the same.
    cap = getattr(scheduler, "max_running_requests", None)
    try:
        cap = int(cap) if cap else None
    except (TypeError, ValueError):
        cap = None
    ceiling = None if cap is None else cap + IN_FLIGHT_CHUNKED_ALLOWANCE

    reported_excursion = [False]

    def _take(batch, origin: str) -> None:
        if not _is_resident(batch):
            return
        if ceiling is not None:
            n = len(_reqs_of(batch))
            if cap is not None and n > cap and not reported_excursion[0]:
                # #682 RECEIPT, AND IT IS ONE BECAUSE THE ATTRIBUTION IS AN
                # INFERENCE. That the fifth resident on 2026-08-16 02:07:22
                # was the in-flight chunked prefill is established by
                # ELIMINATION -- the scheduler's cap bypass is the only
                # documented route past `max_running_requests`, and
                # `AdmissionLimiter`'s ceiling "can never be exceeded" -- not
                # by observation: no chunked request was visible in that log,
                # and the nearest #679 park was three minutes earlier.
                #
                # So say what was actually true at the moment the allowance
                # was spent. If a future excursion reports chunked_req=CLEAR,
                # the elimination has a hole and this widening is covering
                # something else.
                #
                # A LOG, NEVER A DECISION: the verdict above reads only
                # `max_running_requests`, which is rank-uniform, so this line
                # cannot make two ranks disagree no matter what it prints.
                reported_excursion[0] = True
                logger.info(
                    "%s %s holds %d resident request(s), %d above "
                    "max_running_requests=%d -- the in-flight chunked prefill "
                    "allowance (#682). chunked_req=%s on this rank. Carrying.",
                    LOG_PREFIX,
                    origin,
                    n,
                    n - cap,
                    cap,
                    (
                        "SET"
                        if getattr(scheduler, "chunked_req", None) is not None
                        else "CLEAR"
                    ),
                )
            if n > ceiling:
                raise ResidentCarryError(
                    f"{LOG_PREFIX} {origin} claims {n} resident request(s), "
                    f"above the {ceiling} this scheduler can hold "
                    f"(max_running_requests={cap} plus "
                    f"{IN_FLIGHT_CHUNKED_ALLOWANCE} for an in-flight chunked "
                    f"prefill). Refusing to carry it. This is defect M: a "
                    f"corrupted resident set arms a flip that no load "
                    f"justifies and then makes the cutover allocate one "
                    f"tensor per claimed request."
                )
        if id(batch) in seen_batches:
            return
        seen_batches.add(id(batch))
        out.append(batch)

    for slot, mb in enumerate(getattr(scheduler, "running_mbs", []) or []):
        _take(mb, f"running_mbs[{slot}]")
    _take(getattr(scheduler, "running_batch", None), "running_batch")

    # Refuse a request reachable through two DISTINCT batches. Merging
    # those would enter the same Req twice -- duplicate KV rows and a
    # double free -- and the failure would be silent until the pool
    # arithmetic drifted. A request lives in exactly one microbatch slot;
    # if that stops being true, this must be the thing that says so.
    return out


def duplicate_resident_reqs(batches: Sequence, waiting_queue: Sequence = ()) -> List[str]:
    """Requests reachable through MORE THAN ONE of these batches, or ALSO QUEUED.

    TWO SPECIMENS, and the second is why the universe grew.

    #731, 2026-08-17: a cutover left a request RESIDENT AND QUEUED. This
    function could not see it -- it compared batches against each other and
    never consulted ``waiting_queue`` -- so the duplication was silent while
    ``_pending_prefill_tokens`` billed the same prompt twice (51,369 ->
    102,307 tokens, ~2x) and the inflated backlog drove six cutovers that
    served nothing. A guard whose universe excludes half the places a request
    can live reports "no duplicates" and means "none of the kind I look for".
    ``waiting_queue`` is optional so existing callers keep working, and the
    resident-vs-queued class is reported with a ``queued:`` prefix so the two
    shapes stay distinguishable at the call site.

    MEASURED ON METAL, 2026-08-09 04:55:43Z, and it falsified the
    assumption this used to assert. The first version of the carry RAISED
    here, on the reasoning that "a request belongs to exactly one
    microbatch slot". Under a real agentic multi-turn load that refusal
    fired and took the instance down:

        ResidentCarryError: request b0446d8c... is resident in TWO
        distinct batches at once

    So the assumption was wrong, and the loud refusal is what proved it --
    which is the right outcome for the check and the wrong outcome for the
    server. The duplication is real and must be RESOLVED rather than
    refused: ``merge_batch`` extends ``reqs`` in place, so carrying the
    same Req twice is duplicate rows and a double free.
    """
    seen: Set[int] = set()
    dups: List[str] = []
    for batch in batches:
        for req in _reqs_of(batch):
            if id(req) in seen:
                dups.append(str(getattr(req, "rid", "?")))
            seen.add(id(req))
    for req in waiting_queue or ():
        if id(req) in seen:
            dups.append(f"queued:{getattr(req, 'rid', '?')}")
    return dups


def describe_resident_slots(scheduler) -> str:
    """One forensic line naming every slot's three handles. Read-only.

    #631 DEFECT R cost two boots to attribute because the refusal named a
    COUNT and nothing else: "running_mbs[0] claims 5". A count cannot
    distinguish the three ways the number can be wrong -- duplicated Reqs
    inside one batch, one batch reachable twice, or a genuinely oversized
    admission -- and those have different fixes. This prints the whole
    slot row instead, so the next occurrence is attributable from the log
    without a second boot:

      * ``reqs``/``uniq`` differing is duplication INSIDE the batch
        (defect R's signature);
      * ``mbs=None`` next to a resident ``run`` is the purity-idle slot
        that makes a stale ``last`` reachable; and
      * ``last`` naming a distinct EXTEND batch is the object that gets
        re-merged.

    Never raises: it runs on an error path, where a second exception would
    destroy the evidence it exists to capture. ``_reqs_of`` refuses a
    defect-M ``reqs``, so that call is guarded too.
    """

    def _shape(batch) -> str:
        if batch is None:
            return "None"
        try:
            reqs = _reqs_of(batch)
        except ResidentCarryError as exc:
            return f"<unreadable: {exc.__class__.__name__}>"
        mode = getattr(batch, "forward_mode", None)
        try:
            mode_s = "extend" if mode is not None and mode.is_extend() else str(mode)
        except Exception:  # noqa: BLE001 - forensic path, never mask evidence
            mode_s = "?"
        return (
            f"id={id(batch) & 0xFFFFFF:06x} reqs={len(reqs)} "
            f"uniq={len({id(r) for r in reqs})} mode={mode_s}"
        )

    slots = list(getattr(scheduler, "running_mbs", []) or [])
    last = list(getattr(scheduler, "last_mbs", []) or [])
    mbs = list(getattr(scheduler, "mbs", []) or [])
    rows = []
    for i in range(len(slots)):
        rows.append(
            f"[{i}] run({_shape(slots[i])}) "
            f"last({_shape(last[i] if i < len(last) else None)}) "
            f"mbs({_shape(mbs[i] if i < len(mbs) else None)})"
        )
    rows.append(f"running_batch({_shape(getattr(scheduler, 'running_batch', None))})")
    return " | ".join(rows)


def repair_duplicate_resident_reqs(scheduler) -> int:
    """Drop Reqs that appear MORE THAN ONCE inside a single resident batch.

    #631 DEFECT R, the containment half. The leak itself is fixed at its
    source (``_pp_record_slot_last_batch``); this exists because the
    guard's refusal was, on its own, a DEADLOCK:

        under strict purity no decode runs in the PP layout, so only a
        flip to TP can drain the resident set -- and the guard refused to
        evaluate the flip policy precisely while the set looked corrupt.
        The refusal therefore blocked the one action that clears the
        condition it detects. 1115 flips before, zero after, and the
        instance never recovered.

    This is the SECOND time in this chain that "a detector that only
    declines to act is not containment" was paid for, so the rule is now
    stated rather than rediscovered: a refusal must not be able to outlive
    the condition it names. Either repair the set or permit the drain --
    never spin.

    Repair, not clamp. A Req reachable twice through ONE batch's ``reqs``
    is unambiguously wrong at any count: the batch's per-request tensor
    rows and its ``reqs`` list are one basis, so a duplicate row is a
    double free waiting to happen. Removing the later occurrences restores
    the batch to a state the allocator can describe, which is more than
    "small enough to pass the ceiling" -- a clamp would leave a set nobody
    can name and move the next symptom further from its cause (the defect-M
    lesson, ``_reqs_of``).

    ``filter_batch(keep_indices=...)`` is the scheduler's own primitive and
    the same one ``merge_resident_batches`` uses to resolve partial
    overlap, so the batch's tensors stay consistent with its ``reqs``
    rather than acquiring a second convention.

    Returns the number of duplicate entries removed, and logs loudly per
    batch: a repair that runs is a bug report, not a routine event. With
    defect R fixed this should never fire; if it does, the log line names
    the slot to start from.
    """
    removed = 0
    slots = list(enumerate(getattr(scheduler, "running_mbs", []) or []))
    candidates: List[Tuple[str, object]] = [
        (f"running_mbs[{i}]", b) for i, b in slots
    ]
    running = getattr(scheduler, "running_batch", None)
    if not any(b is running for _, b in candidates):
        candidates.append(("running_batch", running))

    for origin, batch in candidates:
        if batch is None:
            continue
        reqs = _reqs_of(batch)
        seen: Set[int] = set()
        keep = []
        for idx, req in enumerate(reqs):
            if id(req) in seen:
                continue
            seen.add(id(req))
            keep.append(idx)
        if len(keep) == len(reqs):
            continue
        dropped = len(reqs) - len(keep)
        logger.error(
            "%s REPAIR: %s held %d entries for %d distinct request(s); "
            "dropping %d duplicate entrie(s). This is #631 defect R -- a "
            "stale last_batch was re-merged into a resident batch. The "
            "flip policy may evaluate again after this repair.",
            LOG_PREFIX,
            origin,
            len(reqs),
            len(keep),
            dropped,
        )
        batch.filter_batch(keep_indices=keep)
        removed += dropped
    return removed


def resident_req_identity(scheduler) -> List[Tuple]:
    """The carried set as comparable identity, for the before/after pin.

    ``(rid, req_pool_idx)`` rather than object ids: it survives logging,
    reads in an error message, and names the row the allocator charges.
    Sorted so two readings of the same set compare equal regardless of
    slot order (the slot ARRANGEMENT legitimately changes at a flip; the
    membership may not).

    UNALLOCATED IS -1, AND THE DEFAULT ARGUMENT DID NOT DELIVER THAT.
    ``getattr(req, "req_pool_idx", -1)`` returns the -1 only when the
    attribute is ABSENT. It is not: ``schedule_batch`` declares
    ``req_pool_idx: Optional[int] = None``, and a Req is visible in
    ``last_batch`` / ``running_mbs`` / ``chunked_req`` from the moment it
    is admitted, which is BEFORE its slot is allocated. So the attribute
    is present and None, the default never fires, and ``int(None)`` raised

        TypeError: int() argument must be ... not 'NoneType'

    taking all three ranks down at 2026-08-09 20:59:45Z, mid-cutover, on
    the very first line of ``_cutover``. The same Optional-None trap the
    live-slot enumeration documents at length (a request without a pool
    slot owns no rows); this function simply never got the memo, and only
    started reaching that state once the armed park began letting flips
    commit between prefill chunks.

    -1 is the right identity for "admitted, not yet allocated": it is
    stable across the cutover (a cutover allocates nothing), it compares
    equal to itself in the before/after pin, and it is visibly not a row.
    """
    ident: List[Tuple] = []
    for batch in harvest_resident_batches(scheduler):
        for req in _reqs_of(batch):
            idx = getattr(req, "req_pool_idx", None)
            ident.append(
                (str(getattr(req, "rid", "?")), -1 if idx is None else int(idx))
            )
    return sorted(ident)


def orphan_resident_reqs(scheduler) -> List[str]:
    """Requests reachable ONLY through ``last_mbs``/``last_batch``.

    THIS IS ALSO THE QUIESCENCE TERM (#631 defect L), which is why it is a
    query and not only an assertion. ``last_batch`` names the batch that
    ran in the previous iteration; in steady decode it holds exactly the
    requests ``running_batch`` holds, and refusing to flip on "last_batch
    is not empty" therefore refuses because requests EXIST -- the same
    category error that made ``_pp_microbatches_drained`` block every
    automatic flip. What actually matters is narrower and is this: is
    every live request reachable through the handle the carry harvests?

    Right after a prefill iteration the answer is briefly no -- the new
    requests are still only in ``last_batch`` and get merged into the
    running batch by the next ``get_next_batch_to_run``. That is a real
    reason to wait, it clears itself in one iteration, and it is bounded.
    """
    carried: Set[int] = set()
    for batch in harvest_resident_batches(scheduler):
        for req in _reqs_of(batch):
            carried.add(id(req))
    orphans: List[str] = []
    sources: List = list(getattr(scheduler, "last_mbs", []) or [])
    sources.append(getattr(scheduler, "last_batch", None))
    for batch in sources:
        for req in _reqs_of(batch):
            if id(req) not in carried:
                orphans.append(str(getattr(req, "rid", "?")))
    return orphans


def assert_no_orphan_resident_reqs(scheduler) -> None:
    """The cutover's form of the check above: by then it must be empty.

    The quiescence predicate gates on the same query, so reaching a
    cutover with an orphan means the predicate did not hold -- a bug to
    raise, not a carry to widen. Silently widening the harvest here would
    hide a broken predicate behind a carry that appears to work.
    """
    orphans = orphan_resident_reqs(scheduler)
    if orphans:
        raise ResidentCarryError(
            f"{LOG_PREFIX} {len(orphans)} request(s) are reachable only "
            f"through last_mbs/last_batch at the cutover: {orphans[:8]}. At "
            f"a quiescent boundary a slot's last_* entry can hold nothing "
            f"its running_mbs entry does not, so this means the quiescence "
            f"predicate admitted a boundary that is not quiescent. Refusing "
            f"rather than widening the harvest around a broken predicate."
        )


def merge_resident_batches(batches: Sequence) -> Optional[object]:
    """Fold resident batches into ONE with the scheduler's own primitive.

    ``merge_batch`` mutates the accumulator in place, so this is called
    exactly once per flip (module header). Returns ``None`` when nothing
    is resident -- the boot case, and the case every flip so far ran in.
    """
    live = [b for b in batches if _is_resident(b)]
    if not live:
        return None
    base = live[0]
    carried: Set[int] = {id(r) for r in _reqs_of(base)}
    for other in live[1:]:
        reqs = _reqs_of(other)
        keep = [i for i, r in enumerate(reqs) if id(r) not in carried]
        if not keep:
            # Every request in this batch is already carried through
            # another one: merging it would duplicate all of them.
            logger.warning(
                "%s a resident batch was entirely duplicate (%d req(s) "
                "already carried); not merged",
                LOG_PREFIX,
                len(reqs),
            )
            continue
        if len(keep) != len(reqs):
            # PARTIAL overlap, resolved with the scheduler's own primitive
            # so the batch's tensors stay consistent with its reqs. Raising
            # here is what took the instance down at 04:55:43Z; the
            # duplication is real, so it is resolved, not refused.
            logger.warning(
                "%s a resident batch shared %d req(s) with another; "
                "filtering the duplicates before the merge",
                LOG_PREFIX,
                len(reqs) - len(keep),
            )
            other.filter_batch(keep_indices=keep)
        carried.update(id(r) for r in _reqs_of(other))
        base.merge_batch(other)
    return base


def install_resident_set(scheduler, batches: Sequence, to_tp: bool) -> Optional[object]:
    """Re-home the carried resident set into the target phase's handle.

    * TP phase: one ``running_batch``; the PP slot arrays are left empty,
      and deliberately so -- a stale slot batch still referencing the
      carried requests would be a second, ageing view of the resident set,
      and the NEXT flip's harvest would resurrect requests that had since
      finished.
    * PP phase: slot 0 of ``running_mbs``; the remaining slots stay empty
      and refill naturally as the pipeline schedules.

    ``last_batch``/``last_mbs`` are cleared in both directions: they name
    a batch whose result was already processed in the OTHER topology, and
    the destination loop merges whatever they hold into its running batch.
    """
    merged = merge_resident_batches(batches)
    if to_tp:
        # Empty the slot arrays first, so nothing references the merged
        # batch twice even transiently.
        n = len(getattr(scheduler, "running_mbs", []) or [])
        if n:
            scheduler.running_mbs = [_empty_batch_like(scheduler) for _ in range(n)]
            scheduler.last_mbs = [None] * n
        if merged is not None:
            scheduler.running_batch = merged
        scheduler.last_batch = None
    else:
        slots = getattr(scheduler, "running_mbs", None)
        if merged is not None:
            if not slots:
                raise ResidentCarryError(
                    f"{LOG_PREFIX} cannot re-home {len(_reqs_of(merged))} "
                    f"resident request(s) into the PP phase: running_mbs is "
                    f"empty, so the loop arrays were not initialised for "
                    f"this topology."
                )
            slots[0] = merged
            scheduler.running_batch = merged
        scheduler.last_mbs = [None] * len(slots or [])
        scheduler.last_batch = None
    consumed = _consume_carried_from_waiting_queue(scheduler, merged)
    if merged is not None:
        logger.warning(
            "%s carried %d resident request(s) across the cutover into the "
            "%s phase%s",
            LOG_PREFIX,
            len(_reqs_of(merged)),
            "tp" if to_tp else "pp",
            f" ({consumed} also removed from the waiting queue)" if consumed else "",
        )
    return merged


def _consume_carried_from_waiting_queue(scheduler, merged) -> int:
    """Take out of the queue what the carry just made resident. Never raises.

    THE DEFECT THIS CLOSES (#731). The carry re-homes requests into
    ``running_batch``/``running_mbs[0]`` and used to leave ``waiting_queue``
    untouched, so a request queued at cutover time ended up existing TWICE --
    once resident and invisible to the policy as runnable, once queued and
    counted. ``_pending_prefill_tokens`` sums the waiting queue and the
    resident set without excluding their intersection, so the same prompt was
    billed twice: measured 2026-08-17, 51,369 -> 102,307 tokens across one
    cutover, within rounding of exactly 2x.

    The inflated backlog then drove the flip policy past its threshold, which
    is why that boot showed six cutovers and served nothing. THE FLIP CHURN WAS
    A SYMPTOM, not a defect of its own -- do not "fix" it separately.

    OWNERSHIP IN ONE PLACE: after a cutover the carried request is RESIDENT, and
    the resident set owns it. The queue must not also claim it.

    Two edges, both deliberate:
    * a request that is ONLY queued (never resident) is untouched -- this
      removes exactly what it re-homed, nothing else;
    * running it twice removes nothing the second time, so a carry over an
      already-consumed queue is a no-op rather than a corruption.
    """
    if merged is None:
        return 0
    try:
        queue = getattr(scheduler, "waiting_queue", None)
        if not queue:
            return 0
        carried = {id(req) for req in _reqs_of(merged)}
        if not carried:
            return 0
        kept = [req for req in queue if id(req) not in carried]
        removed = len(queue) - len(kept)
        if removed:
            scheduler.waiting_queue = kept
        return removed
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not kill a cutover
        logger.warning("%s could not consume carried queue entries: %s", LOG_PREFIX, exc)
        return 0


def reseed_decode_input_relay(scheduler) -> int:
    """Hand the PP phase its OWN last token, not a buffer nobody maintained.

    THE DEFECT THIS CLOSES, and why it is one-directional
    -----------------------------------------------------
    A plain-decode round does not carry its input token on the batch. The
    last spec round ends with ``batch.input_ids = None``
    (``scheduler.py``, "rebuilt next iter from draft_token"), and the next
    round's ``resolve_forward_inputs`` gathers it out of
    ``future_map.output_tokens_buf`` -- a pool-indexed relay whose
    contract is that the round which READS a row was preceded by a round
    that WROTE it.

    The TP speculative phase breaks that contract while it runs, and it is
    entitled to: the NON-OVERLAP V2 path drives the worker synchronously
    and installs ``next_draft_input`` directly, so it never calls
    ``_relay_forward_payload`` and never stashes. Inside that phase this
    costs nothing, because the next round rebuilds ``input_ids`` from the
    draft tokens instead of gathering. At the ``tp_to_pp`` cutover the
    reader changes: the first PP decode gathers, and what it gathers is
    the token the PREVIOUS PP phase stashed -- stale by the entire TP
    phase, hundreds of tokens. The model then appends a foreign token to a
    correct prefix and answers from it, which surfaces as ONE wrong token
    at the cutover and a coherent continuation afterwards, per request,
    only for the requests whose stale row happens to differ.

    The other leg is clean by construction and always was: the PP phase
    stashes on EVERY pass (``scheduler_pp_mixin``), so a batch entering the
    TP phase finds the relay fresh. That asymmetry is the whole direction
    dependence of the defect.

    THE RULE, which is why the fix lives at the handover
    ----------------------------------------------------
        A VALUE HANDED ACROSS THE CUTOVER MUST BE RE-DERIVED FROM THE
        TRUTH AT THE HANDOVER, NEVER INHERITED FROM A BUFFER THE SOURCE
        PHASE STOPPED MAINTAINING.

    ``req.output_ids[-1]`` is that truth: it is what the request has
    admitted to having produced, and it is what the KV is one short of
    (``kv == seen - 1``, measured on both legs). Re-stashing it is the same
    primitive the batch-rebuild path already uses before handing a batch to
    a gathering round, so this introduces no second convention.

    Returns the number of requests re-seeded. Logs the stale value against
    the truth for every request, which is the falsifier: if they agree and
    the answers still break, the defect is not here.
    """
    import torch

    from sglang.srt.managers.overlap_utils import RelayPayload

    future_map = getattr(scheduler, "future_map", None)
    if future_map is None:
        return 0
    batches = []
    for slot in getattr(scheduler, "running_mbs", None) or []:
        if _is_resident(slot):
            batches.append(slot)
    running = getattr(scheduler, "running_batch", None)
    if _is_resident(running) and not any(b is running for b in batches):
        batches.append(running)

    total = 0
    for batch in batches:
        reqs = [r for r in _reqs_of(batch) if getattr(r, "output_ids", None)]
        # A REQUEST CAN HAVE OUTPUT AND NO SLOT, and the cutover must survive
        # it (#656 successor 36). ``req_pool_idx`` is None for a request whose
        # pool slot has been released or was never assigned, and the tensor
        # build below turns that into
        #     TypeError: 'NoneType' object cannot be interpreted as an integer
        # inside ``_cutover``, i.e. inside the no-return region -- so all three
        # ranks abort together and serving dies. Measured 2026-08-11 16:51:50,
        # every rank, 14.5 minutes into a run; the trace is
        # phase_flip_resident_carry.py:674 <- _cutover:1176 <- _execute:4349.
        #
        # Skipping them is CORRECT and not merely defensive: the relay reseeds
        # a slot-indexed buffer, so a request with no slot has nothing to
        # reseed. It is logged rather than dropped quietly, because a resident
        # request losing its slot across a cutover is a fact worth seeing.
        slotless = [r for r in reqs if getattr(r, "req_pool_idx", None) is None]
        if slotless:
            reqs = [r for r in reqs if getattr(r, "req_pool_idx", None) is not None]
            logger.warning(
                "%s decode-input relay: %d carried request(s) have output but "
                "no pool slot (rids %s); they cannot be reseeded because the "
                "relay is slot-indexed, so they are skipped. Before this was "
                "handled, the tensor build raised inside the cutover and took "
                "every rank down with it.",
                LOG_PREFIX,
                len(slotless),
                ", ".join(str(getattr(r, "rid", "?"))[:8] for r in slotless[:8]),
            )
        if not reqs:
            continue
        indices = torch.tensor(
            [r.req_pool_idx for r in reqs],
            dtype=torch.int64,
            device=future_map.output_tokens_buf.device,
        )
        truth = torch.tensor(
            [r.output_ids[-1] for r in reqs],
            dtype=torch.int64,
            device=future_map.output_tokens_buf.device,
        )
        # THE FALSIFIER, read BEFORE the write: what the first PP decode
        # would have gathered, next to what the request actually last
        # produced. One D2H per cutover, not per round.
        stale = future_map.output_tokens_buf[indices].tolist()
        want = truth.tolist()
        mismatched = sum(1 for a, b in zip(stale, want) if a != b)
        logger.warning(
            "%s decode-input relay at cutover: %d/%d request(s) would have "
            "gathered a STALE token -- %s",
            LOG_PREFIX,
            mismatched,
            len(reqs),
            " | ".join(
                "%s relay=%s last=%s%s"
                % (str(getattr(r, "rid", "?"))[:8], a, b, "" if a == b else " MISMATCH")
                for r, a, b in zip(reqs, stale, want)
            ),
        )
        future_map.stash(indices, RelayPayload(bonus_tokens=truth))
        # And make the gather the ONLY source: a leftover ``input_ids``
        # from the speculative phase names draft tokens that the PP phase
        # has no drafter to interpret.
        batch.input_ids = None
        total += len(reqs)
    return total


def _empty_batch_like(scheduler):
    """A fresh empty slot batch, built the way the loop builds them."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    return ScheduleBatch(reqs=[], batch_is_full=False)


def carry_across_pp_loop_init(scheduler, harvested: Sequence) -> None:
    """The ``init_pp_loop_state`` seam: re-seed what the rebind destroyed.

    Called with the batches harvested BEFORE the arrays were rebound. The
    destination is always the PP phase here -- ``init_pp_loop_state``
    exists for that loop -- and the cutover promotes slot 0 into
    ``running_batch`` afterwards when the target phase is TP.
    """
    if not harvested:
        return
    install_resident_set(scheduler, harvested, to_tp=False)


def promote_slot_zero_to_running_batch(scheduler) -> Optional[object]:
    """The TP leg's second half: the slot array is not the TP loop's handle.

    ``init_pp_loop_state`` re-seeds into ``running_mbs[0]`` because that is
    where the PP loop reads its resident set. The non-PP loops read
    ``running_batch``, so the TP leg moves it -- MOVE, not copy: the slots
    are emptied, or the carried requests would remain visible through a
    view the TP loop never updates.
    """
    slots = list(getattr(scheduler, "running_mbs", []) or [])
    resident = [b for b in slots if _is_resident(b)]
    if not resident:
        scheduler.last_batch = None
        return None
    return install_resident_set(scheduler, resident, to_tp=True)
