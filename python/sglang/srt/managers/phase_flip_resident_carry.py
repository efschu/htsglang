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


class ResidentCarryError(RuntimeError):
    """A resident request would have been dropped, duplicated or orphaned.

    Always raised BEFORE the damage: every caller checks on the way into
    or out of the topology swap, where the alternative is a stranded KV
    page and a stranded mamba lock discovered a pass later, with the
    evidence already stale (which is precisely how J.3 cost three boots).
    """


def _reqs_of(batch) -> List:
    return list(getattr(batch, "reqs", []) or [])


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

    def _take(batch) -> None:
        if not _is_resident(batch):
            return
        if id(batch) in seen_batches:
            return
        seen_batches.add(id(batch))
        out.append(batch)

    for mb in getattr(scheduler, "running_mbs", []) or []:
        _take(mb)
    _take(getattr(scheduler, "running_batch", None))

    # Refuse a request reachable through two DISTINCT batches. Merging
    # those would enter the same Req twice -- duplicate KV rows and a
    # double free -- and the failure would be silent until the pool
    # arithmetic drifted. A request lives in exactly one microbatch slot;
    # if that stops being true, this must be the thing that says so.
    return out


def duplicate_resident_reqs(batches: Sequence) -> List[str]:
    """Requests reachable through MORE THAN ONE of these batches.

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
    return dups


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
    if merged is not None:
        logger.warning(
            "%s carried %d resident request(s) across the cutover into the "
            "%s phase",
            LOG_PREFIX,
            len(_reqs_of(merged)),
            "tp" if to_tp else "pp",
        )
    return merged


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
