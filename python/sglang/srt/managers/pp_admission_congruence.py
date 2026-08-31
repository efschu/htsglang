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
"""#791: PP ranks agree on a batch by DECISION, not by luck.

THE DEFECT (measured, see #788's `_trace_pp_admission_verdict` and the
uniformity-floor scope note next to it in scheduler.py). Under Pipeline
Parallelism every stage is an independent scheduler that re-derives its own
admission verdict from its OWN local radix/eviction state
(`_get_new_batch_prefill_raw`). The #616g rank-uniformity floors that already
solve this for Tensor Parallelism are scoped to `tp_cpu_group` and are a
no-op whenever that group has one member -- true on every rank of a
TP=1/PP=N boot. Requests are chain-forwarded to every stage unconditionally,
but nothing forwards the ADMISSION DECISION alongside them, so two stages can
disagree about which requests are in the batch, or agree on membership but
disagree on how much prefix each request reuses. Either divergence corrupts
the cross-stage activation tensor: `ScheduleBatch.prepare_for_extend` derives
`extend_num_tokens` (the tensor's row count) directly from `len(prefix_indices)`
per request (schedule_batch.py), so a prefix-length disagreement is a SHAPE
disagreement, not an accounting one.

WHAT CROSSES THE WIRE. Never `req.prefix_indices` -- those are literal KV-pool
slot pointers into THIS rank's own pool (schedule_batch.py, `match_prefix`);
shipping them off-rank is meaningless and would itself be a form of the
corruption this module exists to prevent. What crosses is the DECISION: an
ordered list of `(rid, prefix_len, extend_len, admitted)`, built once by the
rank that owns admission truth for the request stream (PP0, the sole rank
that reads from the tokenizer socket -- see
`SchedulerRequestReceiver.recv_requests`, request_receiver.py:117-121,143) and
carried forward through the chain by whichever mechanism the caller wires in
(see CARRIER below). This module has no wire code of its own; it is the pure
decision/reconcile logic, deliberately kept out of `scheduler_pp_mixin.py`
(another strand owns that file's receive path for #789 -- see the module-level
SCOPE FENCE note below) and out of the hot admission path in scheduler.py.

CARRIER (design note, not wired here). The typed tensor-dict/proxy channel
(`pp_typed_channel.py`) is the preferred carrier: it is already keyed per
`(src, kind)` and already documents that "a non-tensor entry travelling in
this dict is established practice on this channel, not a new risk"
(pp_typed_channel.py, module docstring). `kind` is a plain string there, not a
closed enum (`stash_typed`/`take_typed` take any `str`), so a new kind such as
`"admission_decision"` can ride alongside the `"proxy"` tensors for the same
`mb_id` without modifying `pp_typed_channel.py`. It also travels with the
exact activation tensor whose shape the decision determines, which is the
property that matters: a decision and its tensor can never be observed
out of sync. A separate per-pass pyobj message (modelled on the disagg
loops' consensus sends, `_pp_send_pyobj_to_next_stage` /
`point_to_point_pyobj`, scheduler_pp_mixin.py:721-736,902-910) was considered
and rejected for this reason -- it is a second, independently-timed channel,
and nothing pins it to arrive paired with the tensor it describes.

TWO FAILURE SHAPES, ASYMMETRIC (peer-established, HANDOFF quality).
  local match >= told  -> truncate to told. SAFE. This discards some of this
      rank's own legitimate local reuse; it is the identical slack trade
      #616g already makes on the TP axis (MIN-over-ranks), taken here on the
      PP axis instead. `reconcile_pp_admission_decision` takes this path
      silently -- it is not an anomaly, it is the expected common case
      whenever a downstream rank's cache is, or ever was, warmer than PP0's.
  local match <  told   -> UNSAFE, and it is physically un-fixable in the
      SAME pass. `told` was computed by an upstream rank BEFORE this rank's
      shortfall was knowable; the activation tensor already in flight was
      already sized against `told`, and only the FIRST stage does the
      embedding lookup, so a downstream rank that lacks KV for
      `[local, told)` has no token-level input from which to backfill it --
      the hidden states for that range were never computed by any stage,
      because the upstream rank that owns them decided they were reusable
      and skipped their forward entirely. There is no in-pass recomputation
      that closes this gap without an upstream redo, and an upstream redo
      mid-pass is exactly the blocking round-trip
      (scheduler.py:6391-6407's 2026-08-17 deadlock family) this module is
      built to avoid adding to the admission path. So the request cannot be
      honoured THIS pass. `reconcile_pp_admission_decision` therefore never
      fabricates a prefix length for it: it marks the request RETRACTED
      (removed from `effective`, so no caller can accidentally admit it with
      a corrupt length), emits exactly one bounded WARNING naming rank, rid,
      told and local, and carries the retraction forward in the amended
      decision so every remaining downstream rank makes the SAME membership
      decision about that one rid -- membership, not per-token shape, is
      what changes, which is a change PP already has a mechanism for
      (retraction, used today by the disagg decode loop's
      `send_retract_work`) rather than a new kind of failure. The request is
      not lost: it is expected to be re-queued and re-admitted on a LATER
      pass, at which point PP0 builds a fresh decision from current state
      and `told` for it starts at 0 (full recompute, correct, merely
      slower). This module does not implement the re-queue itself -- that is
      scheduler-loop wiring, out of this module's and this phase's scope --
      it only guarantees that what it hands back never lets a caller treat
      an unhonourable length as safe.

THE CONGRUENCE GUARD IS WHAT MAKES THE DEGRADE RARE, NOT WHAT REPLACES IT.
Bounding `told` at admission time against a downstream rank's ACTUAL current
local match would need that rank's state at the moment of the decision, which
is exactly the blocking collective this module must not add
(scheduler.py:6391-6407). A non-blocking, previously-published floor (each
rank's local match from its last completed pass, piggybacked on the existing
output-tensor return trip) narrows the window in which a downstream rank's
cache can still have moved between "last published" and "this pass" -- but it
cannot close that window to zero, so the degrade path above is not optional
scaffolding; it is the thing that makes the guard's staleness survivable
instead of silently wrong. Wiring that publish/consult loop is future work
(also out of scope for scheduler_pp_mixin.py under the current #789 scope
fence); this module's contract holds with or without it.

NO COLLECTIVE. This module performs no `torch.distributed` call of any kind,
by construction -- see `test_reconcile_never_touches_torch_distributed` in
the paired test file for a source-level pin on that property. Every function
here is a pure, rank-local computation over already-local inputs.

DEFAULT PATH. `pp_size <= 1` (today's only shipped configuration) must be
byte-identical to not having this module at all. `reconcile_pp_admission_decision`
and `build_pp_admission_decision` both take `pp_size` and short-circuit to an
identity pass-through when it is `<= 1`; see
`test_pp_size_one_is_a_pure_pass_through` for the pin.

NO HAND-PINNED NUMBERS. There are none in this module: every comparison below
is between two already-materialised local integers (`told` vs `local`), never
a heuristic constant.

#630: THE RETRY LIVELOCK, AND WHY THE DEGRADE ABOVE IS NOT BY ITSELF ENOUGH.
The degrade in "TWO FAILURE SHAPES" excludes an unhonourable request and
expects it to be re-queued and re-admitted on a LATER pass. Nothing in that
sentence forces the later pass to ask for less. Left alone: PP0 re-derives
`told` from its own unchanged local state, tells the downstream rank the
same too-long length, that rank retracts again -- a deterministic cycle with
no forward progress. That IS #630's family (see
`test_pp_disk_hicache_guard_630.py`'s history of this exact codebase's prior
livelock: "a bounded/degrading path that makes no forward progress is the
livelock defect," rooted there in a bounded wait that polled a REPORT
(`is_completed()`) instead of ever consuming a DRIVING signal, so two
polling peers never advanced). "It degrades gracefully" is not sufficient;
the degrade must strictly advance the request's state. `PPAdmissionCongruenceGuard`
below closes this: it remembers each retraction's ACTUAL observed local
match (`observed_local`, always `< told` by construction, since a retraction
only fires on that branch) and clamps the NEXT `told` PP0 offers for that
rid to at most that remembered value, so the second pass cannot repeat the
identical failure -- it can only repeat a STRICTER one, and a strictly
decreasing sequence of non-negative integers is well-founded. It is
rank-local state, mutated only from the same `amended_decision` values
`reconcile_pp_admission_decision` already produces for forwarding, so it
adds no new send/recv and no new synchronisation point -- see NO COLLECTIVE
above, which continues to hold with this class in play.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def offered_prefix_key(
    fill_ids: Optional[Sequence[int]], prefix_len: int
) -> Optional[int]:
    """#963: a stable identity for the PREFIX an offer is made over.

    Returns None when the tokens are not available, which leaves the guard on
    its pre-#963 rid-scoped path -- the caller must never invent a key, because
    two different prefixes sharing one key would clamp a prefix no rank ever
    reported short (cache loss on every rank, silent and permanent).

    STABLE ACROSS PROCESSES, and that is not a detail. The ranks are separate
    processes and `hash()` is salted per process by PYTHONHASHSEED, so a
    built-in hash here would disagree between the very ranks this feature
    exists to keep congruent -- and would do so only on multi-process runs,
    i.e. never in a unit test. `tree_congruence` learned this as its
    constraint 3 and chose blake2b; the same choice, for the same reason.

    LENGTH IS MIXED IN so that a prefix and its extension cannot collide: two
    requests matching 1250 and 2500 tokens of the same text are different
    cache states and must not share a floor.

    ONE `struct.pack` rather than a per-token join: this runs once per offered
    request per pass, on the admission path, and the prefix can be tens of
    thousands of tokens.
    """
    if fill_ids is None or prefix_len <= 0:
        return None
    ids = list(fill_ids[:prefix_len])
    if len(ids) != prefix_len:
        # A prefix longer than the tokens we hold is not a prefix we can name.
        return None
    payload = struct.pack("<Q", prefix_len) + struct.pack(f"<{len(ids)}q", *ids)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

logger = logging.getLogger(__name__)


#: #944: "this rank could not FIND the request", which is not the same fact as
#: "this rank has no prefix for it" and must never be spelled the same way.
#:
#: THE CLASS, and this is its THIRD instance. `_pp_reconcile_incoming_admission`
#: resolves a rid through a chain of lookups and, on a total miss, used to write
#: 0 into `local_match_lens`. The consumer below then read that 0 as a
#: MEASUREMENT and voided the pass. #797c fixed the chunked_req miss, #798 fixed
#: the wrong-slot miss -- two symptom patches of one class, each adding a lookup
#: while the miss kept answering with a number. #944 is the running-batch miss,
#: measured: 2106 unhonourable-prefix events, 2107 voided passes and a
#: three-rank hang in 35 s under real agent load.
#:
#: The idiom is not invented here. `tp_head_congruence._ABSENT_MATCH` (-1)
#: already distinguishes "does not hold this rid" from a length, one file over,
#: and MIN-reduces to itself for exactly this reason.
#:
#: SCOPE: this value lives INSIDE the resolution path (`local_match_lens` and
#: the reader immediately below it) and never crosses the wire. A rank reports
#: a miss to its peers as `PPAdmissionEntry.unresolved`, a boolean, because
#: `observed_local` has readers that treat it as a length -- see that field.
UNKNOWN_MATCH = -1

#: #944: how many consecutive rounds a rid may be DEFERRED for being
#: unresolvable before PP0 stops asking, refuses loudly, and pins the next
#: offer to `told=0`.
#:
#: WHY A CAP IS LOAD-BEARING AND NOT HOUSEKEEPING. `_learned_floor` is what
#: damped the re-offer before, and it is fed from `observed_local` -- which a
#: lookup miss must not set, because nothing was measured. So `UNKNOWN_MATCH`
#: ALONE makes the measured 2106-void loop WORSE than the old 0 did: the 0 was
#: a false measurement, but it at least clamped. The cap replaces that
#: accidental clamp with a bound that terminates for a stated reason.
#:
#: 3, because a defer exists to let rank-local state SETTLE (a request landing
#: in the running batch, a slot's chunked req being restored), and that settles
#: within one ring lap; `pp_size - 1` rounds is the longest a decision takes to
#: visit every rank, so three gives that a full round trip plus one. A larger
#: number does not buy a better outcome -- it buys more voided passes before
#: the same escape.
UNRESOLVED_DEFER_CAP = 3


class PPScheduleRefused(Exception):
    """#791 CORE: this rank cannot EXECUTE the forwarded pass geometry.

    Raised out of the admission adder and caught by the scheduler, which
    voids the pass through #797's existing machinery. It is deliberately an
    exception and not a return code: every `AddReqResult` a caller can return
    means "build a batch without this request", and building a batch without a
    request the upstream already launched hidden states for is precisely the
    corruption this refusal exists to prevent. There is no narrower batch to
    fall back to.
    """


#: #987: the widest fill divergence a decision may carry across the seam.
#:
#: The defect this closes is ONE token wide and cannot be wider for the reason
#: it exists: a single sampled token held on one side of one `tp_to_pp` hop
#: (#631 OUTTRACE, boot 7 -- PP0 `n=1 off=0 tail=[25]`, followers `n=0`). The
#: cap is 8 rather than 1 so a two- or three-token variant of the same seam
#: shape is carried rather than deadlocked, and it is SMALL rather than
#: generous on purpose: past it, the two ranks are not one token out of step
#: on a seam, they hold different requests, and shipping an arbitrary suffix
#: of one across the wire would convert a loud, correct refusal into a silent
#: fabrication. Beyond the cap the entry is not adopted and the existing
#: refusal stands, with both numbers named (`#987 FILL-REFUSE`).
FILL_CARRY_TAIL_CAP = 8


@dataclass(frozen=True)
class PPAdmissionEntry:
    """One request's admission decision, as it crosses the wire.

    Deliberately does NOT carry `prefix_indices`, `last_node`, or any other
    rank-local pool handle -- see the module docstring's "WHAT CROSSES THE
    WIRE" section. `prefix_len`/`extend_len` are plain integers; a receiver
    reconstructs its own local `prefix_indices` from `prefix_len` against its
    OWN pool (or, if it cannot honour `prefix_len`, retracts -- it never
    borrows the sender's pointers).
    """

    rid: str
    prefix_len: int
    extend_len: int
    admitted: bool = True
    retracted: bool = False
    retracted_by_rank: Optional[int] = None
    observed_local: Optional[int] = None
    """Set only on a newly-retracted entry (see `reconcile_pp_admission_decision`):
    the retracting rank's ACTUAL local match length, i.e. the value that made
    `told` unhonourable (`observed_local < prefix_len` always holds when this
    is set -- that inequality is exactly what triggered the retraction).
    `None` for every non-retracted entry, and for an entry that was already
    retracted by an EARLIER rank (that rank's `observed_local` is preserved
    unchanged; a later rank must not overwrite it with its own, unrelated,
    local state -- see `reconcile_pp_admission_decision`'s pass-through
    branch). This is the signal `PPAdmissionCongruenceGuard` (#630) learns
    from -- never a guess, never a timeout, always a real observed shortfall."""

    unresolved: bool = False
    """#944: this rank could not LOCATE the request at all, so it measured
    nothing and `observed_local` is None.

    A SEPARATE FIELD, NOT A SPECIAL VALUE OF `observed_local`, and that is the
    whole point of this ticket. "I looked and found N tokens" and "I could not
    find the request" are different facts with opposite correct responses -- the
    first must void the pass, the second must let the group retry -- and #797c,
    #798 and #944 are three instances of one class in which they shared a
    spelling and no reader could tell them apart. Encoding the miss as a
    sentinel `observed_local` would be that same class one level up: THIS
    field's readers treat it as a LENGTH and feed it to `_learned_floor`, which
    clamps the next round's offer, and a floor learned from a number nobody
    measured is exactly the defect.

    ON THE WIRE because the rank that OBSERVES the miss is never the rank that
    can ACT on it: only PP0 chooses `told`. Downstream ranks report; PP0 owns
    the count and the escalation (`UNRESOLVED_DEFER_CAP`). A defer that only
    one rank takes IS the next divergence, so no rank takes one alone."""

    fill_len: Optional[int] = None
    """#987: the UPSTREAM's own `len(full_untruncated_fill_ids)` for this rid.

    `None` on every entry built before this field existed and on every entry
    from a rank that does not set it -- and `None` means "say nothing", not
    "zero". A receiver that reads `None` behaves exactly as it did before this
    field was added.

    WHY A LENGTH TRAVELS AT ALL, when the module docstring's WHAT CROSSES THE
    WIRE section is otherwise strict that a receiver reconstructs local
    quantities locally. `prefix_len` is a quantity each rank may legitimately
    hold differently -- it is a statement about a CACHE, and a rank that
    cannot honour it retracts. `fill_len` is not: it is the length of the
    REQUEST, origin tokens plus tokens already generated for it, and two ranks
    disagreeing about it is the third clause of `schedule_refusal_reason`
    saying, correctly, that they "disagree about the request itself, not
    merely about its cache". That disagreement has exactly one true answer,
    the upstream holds it, and until #987 no rank could ask.

    MEASURED, R9 census over boots 6-7: 506 of 513 void-causing refusals are
    ONE rid, off by ONE token. Rank 0 held an output token (id 25) that never
    crossed the `tp_to_pp` seam, so rank 0 read 8447 and its followers 8446,
    and the third clause vetoed every pass for two minutes."""

    fill_tail: Tuple[int, ...] = ()
    """#987: the trailing output token ids the upstream holds beyond a
    follower's own fill, newest last. Empty when nothing is carried.

    THE TAIL AND THE LENGTH TRAVEL TOGETHER OR NOT AT ALL, and pairing them is
    a correctness requirement rather than a convenience. A follower that
    adopted `fill_len` alone would agree about the SIZE of the request while
    still not holding its last token: every subsequent index into the fill --
    the extend range, the radix match key, the KV row count -- would name a
    position it has no token for. That is a worse defect than the refusal it
    would silence, so the adopt is defined only over the pair.

    BOUNDED AT `FILL_CARRY_TAIL_CAP`. The seam divergence this closes is one
    token wide by construction (one sampled token, held on one side of one
    hop). A divergence wider than the cap is not this shape and must not be
    papered over by shipping an arbitrary suffix of a request across the
    wire -- it is refused loudly instead, with both numbers named."""

    last_chunk: Optional[bool] = None
    """#996: the DECIDING rank's own verdict on whether this extent finishes
    the request. `None` = "say nothing" (a legacy sender, or a stand-in that
    published no fill), and a receiver reading `None` falls back to the
    pre-#996 local derivation exactly.

    WHY THIS TRAVELS, and it is the same argument `fill_len` above makes, one
    step further. Until #996 the receiver RE-DERIVED this verdict at
    `schedule_policy.py:_add_scheduled_req` as

        last_chunk = prefix_len + extend_len >= len(full_untruncated_fill_ids)

    over a comment claiming it was "the schedule's to say ... arithmetic on
    forwarded integers". Two of those three integers are forwarded; the third
    is not. `len(full_untruncated_fill_ids)` is rebuilt from THIS rank's
    `origin_input_ids + output_ids (+ carried tail)` on every pass
    (`Req._refresh_fill_ids`, schedule_batch.py:1326 -- unconditional, ahead of
    the `tree_cache` gate at :1355), so the verdict was a rank-local quantity
    wearing a forwarded one's clothes.

    THE GUARD ONLY COVERED HALF THE DISAGREEMENT. `schedule_refusal_reason`'s
    third clause refuses `prefix + extend > local_fill_len` -- the decision
    asking for more than this rank holds. The opposite skew, this rank holding
    MORE than the rank that decided, passes every clause and silently flips the
    verdict from "last chunk" to "mint a continuation". `adopt_carried_fill`
    does not close it either: it only ever APPENDS (:1628-1633), so it lifts a
    short follower up to the decider and can never bring a long one down.

    MEASURED, boot 16 (996fbf4aca, 2026-08-28 22:21:48,
    boot_943bx_996fbf4aca_0828_221614.log): PP1 and PP2 both logged
    `#987 FILL-ADOPT rid=da614e20... local=8446 -> upstream=8447`, and PP1 then
    died at scheduler_pp_mixin.py:2147 with `#631 PROXY LEFTOVER REFUSED: a
    proxy stamped mb_id=1 seq=17 rows=4096 epoch=2 arrived while this rank is
    on mb_id=2 in flip epoch 2` -- the same-epoch branch, i.e. a proxy for a
    pass this rank had already left. All three ranks down 68 s after first
    load. The continuation nobody decided on is what put that proxy on the
    wire.

    CARRIED, NOT RE-CHECKED. The fix is not a fourth clause in
    `schedule_refusal_reason` -- tightening that to `!=` would refuse every
    legitimate middle chunk, where `prefix + extend < local_fill_len` is the
    normal case, and void every pass. The verdict simply stops being derived
    from a local length."""

    load_back_len: Optional[int] = None
    """#968/#1035: PP0's host load-back EXTENT for this rid, in tokens.

    `None` = "say nothing" (a legacy sender, `pp_size <= 1`, or a pass on
    which PP0 offered no load-back), and a receiver reading `None` performs
    NO load-back at all -- never a rank-local substitute. That asymmetry is
    the whole point: an absent fact is an honest miss costing recompute,
    while a rank-local second derivation is the shape divergence boot
    1815081d46 died in.

    WHY AN EXTENT TRAVELS WHERE `prefix_len` ALREADY DOES. `prefix_len` is an
    upper bound the receiver TRUNCATES to (`Req.truncate_prefix_to`), and its
    contract -- `told <= this rank's local match` -- is what
    `reconcile_pp_admission_decision` enforces. A host load-back GROWS the
    prefix beyond the local device match, so folding it into `prefix_len`
    would make every load-back row violate that contract on arrival and be
    retracted as unhonourable. The two quantities move in opposite
    directions; they cannot share a field.

    DECIDED AT PASS N, APPLIED AT PASS N+1, ON EVERY RANK INCLUDING PP0. The
    per-pass proxy frame is received at `scheduler_pp_mixin.py:4476`, AFTER
    that rank has already planned its batch at `:4292` -- so a fact stamped
    on the frame for pass N cannot drive pass N's own admission. PP0
    therefore does not apply its own offer either: it records the extent, the
    row makes one lap, and every rank first applies it on the pass after it
    holds it. Uniformity is then CONSTRUCTIVE -- the number comes from one
    rank -- rather than a bet that two ranks measured the same thing.

    THE COST, STATED HONESTLY -- AND AN EARLIER VERSION OF THIS PARAGRAPH WAS
    WRONG. It claimed "the first chunk computes without the prefix and every
    later chunk honours it". THAT IS NOT WHAT THE CODE DOES: the only consumer
    of this fact is in `PrefillAdder.add_one_req`; `add_chunked_req` has NO
    load-back site, so the later chunks of an already-admitted request do not
    consume it. The claim described an intention, not a mechanism, and it is
    corrected here rather than left to be read as a guarantee.

    WHAT IS ACTUALLY PAID: the request that DISCOVERS an extent never benefits
    from it -- it defers, publishes, and runs its whole prefill uncached. The
    benefit lands on the next request over the same content, which is a FRESH
    admission through `add_one_req` and therefore does consume it. Since a warm
    hit is by definition a different request carrying the same prompt, that is
    exactly the case this exists for; what is genuinely lost is the remaining
    chunks of the FIRST request to see a given prefix.

    NOT MIRRORED INTO `add_chunked_req` DELIBERATELY. That would put a second
    prefix mutation on the chunked path, which is the seam #965/#988 record as
    having already cost a boot (`extend_range.start == len(prefix_indices)`
    re-derived at the mutation, not at the exits). One consumer, one mutation
    site.

    MEASURED, boot_855_939reread_0840f82601_0830_232150: 15/15 `#1035
    RANK-LOCAL LOAD-BACK REFUSED` lines, 5 rids x 3 ranks, host hit
    rank-IDENTICAL for every rid (1278, 350, 4618, 4618, 350). That
    uniformity is a HAZARD READING, never a licence: it says the offered
    extent will normally be honourable, NOT that a rank may derive one
    locally when the fact is absent."""


@dataclass(frozen=True)
class PPAdmissionDecision:
    """An ordered, per-pass admission decision for one PP microbatch slot.

    `mb_id` matches the microbatch-slot keying already used by the typed
    proxy channel (`pp_typed_channel.py`'s `(src, kind)` demux, and
    `_pp_proxy_stamp`'s per-`mb_id` stamping in scheduler_pp_mixin.py) --
    carried here so a caller that wires this onto that channel has the same
    key available, without this module importing anything from it.
    """

    mb_id: int
    entries: Tuple[PPAdmissionEntry, ...]

    def by_rid(self) -> Dict[str, PPAdmissionEntry]:
        return {e.rid: e for e in self.entries}


class PPAdmissionCongruenceGuard:
    """PP0-side memory of downstream shortfalls -- closes the #630 retry
    livelock without a collective. See the module docstring's "#630: THE
    RETRY LIVELOCK" section for the defect and why this shape was chosen
    over a one-shot `told=0` pin.

    WHY "PP0 LEARNS" OVER "ONE-SHOT told=0 PIN" (the two shapes the task
    required evaluating). A pin that forces `told=0` on the very next retry
    also terminates the loop -- an empty prefix is always `<= any local
    match`, so a second retraction for that rid becomes impossible -- but it
    throws away every byte of that rid's reuse, on every rank, forever after
    a single one-token shortfall. Learning the shortfall's ACTUAL value
    (`observed_local`, never zero) and clamping `told` to it is the same
    shape of state (one int per outstanding rid) and no harder to build; the
    difference is entirely in what it buys: the degrade stays RARE (a rid
    that was one token short of `told` loses exactly one token of reuse, the
    next pass, not all of it) instead of merely TERMINATING. Chosen:
    PP0-learns-coverage.

    RID-SCOPED, ONE-SHOT, CLEARS ON SUCCESS. `_learned_floor[rid]` exists
    only while `rid` has an outstanding, unresolved shortfall. The instant a
    pass admits `rid` with NO retraction anywhere in the chain -- observed
    via `record_return_trip` on the fully chain-reconciled decision, the same
    value already threaded rank-to-rank as `amended_decision` -- the entry is
    deleted. A rid that was once short is not permanently capped once its
    cache state (on whichever rank was short) has moved on; this is the
    "clears once served" requirement the one-shot-pin shape would also have
    had to satisfy, met here for a learned value instead of a boolean.

    NO COLLECTIVE. Rank-local mutable state, read and written only by the
    rank that calls `build_pp_admission_decision` (PP0) and fed only by
    decisions that already crossed the wire through the EXISTING retraction
    mechanism. No new send, no new recv, no new synchronisation point -- see
    NO COLLECTIVE in the module docstring.

    TERMINATION (well-founded, not merely bounded). A retraction only fires
    when `local < told` (by construction of `reconcile_pp_admission_decision`),
    so every NEW retraction for `rid` sets `_learned_floor[rid]` to a value
    strictly less than the `told` that just failed. The next cycle's `told`
    for that rid is clamped to `<= _learned_floor[rid]`, so it is strictly
    less than the `told` that just failed -- a strictly decreasing sequence
    of non-negative integers, which cannot cycle and cannot decrease forever.
    It terminates in at most `told_initial` cycles, and in practice within at
    most `pp_size - 1` (each cycle can only discover a floor from a rank that
    has not yet reconciled this rid this pass).
    """

    def __init__(self, unresolved_defer_cap: int = UNRESOLVED_DEFER_CAP) -> None:
        self._learned_floor: Dict[str, int] = {}
        #: #963: the same shortfall, scoped to the PREFIX it was measured
        #: against instead of to the request that happened to carry it.
        #:
        #: WHY THE RID-SCOPED TABLE ABOVE CANNOT REACH THIS DEFECT. A rank's
        #: shortfall is a fact about that rank's TREE, not about the request.
        #: When the trees diverge -- one rank admitted a chunk its peers were
        #: still prefetching, reached the unconditional stash
        #: (`scheduler.py:7010-7011`) and inserted a prefix the others never
        #: received -- every FRESH rid over that prefix starts from an
        #: unclamped `told` and buys its own voided pass. This class's
        #: termination argument is per rid and is silent about the
        #: population, which is exactly the gap: measured window-958 boot 2,
        #: `_learned_floor` RAN and LOWERED on PP0 and still never bound,
        #: because all six offers were six DISTINCT rids over ONE 1250-token
        #: prefix. Keyed here, the first voided pass teaches every later
        #: request sharing that prefix.
        #:
        #: The key is an opaque, caller-supplied fingerprint of the offered
        #: prefix TOKENS. It must be stable across processes -- `hash()` is
        #: PYTHONHASHSEED-salted and would disagree across the very ranks this
        #: exists to keep congruent (`tree_congruence`'s constraint 3, learned
        #: the same way).
        self._prefix_floor: Dict[int, int] = {}
        #: rid -> the prefix key its most recent offer was made over, so a
        #: retraction arriving on the return trip can be attributed to the
        #: prefix that was actually offered rather than to whatever the
        #: request matches by the time it comes home.
        self._offer_prefix_key: Dict[str, int] = {}
        #: #944: consecutive rounds `rid` came back UNRESOLVED (no rank could
        #: locate it). Distinct from `_learned_floor` because it counts a
        #: different population -- see `PPAdmissionEntry.unresolved`.
        self._unresolved_rounds: Dict[str, int] = {}
        #: rids whose loud refusal has already been emitted, so the escalation
        #: is one line and not one line per pass. A bounded refusal that
        #: re-logs every pass is the 2106-line log this ticket is about, in a
        #: different colour.
        self._escalated: set = set()
        #: <= 0 disables the #944 bound entirely, restoring the pre-cap
        #: behaviour exactly -- the same neuterable escape `#552`'s
        #: `defer_limit` carries, so a can-fail proof can take the cap down on
        #: its own without taking the sentinel down with it.
        self._unresolved_defer_cap = int(unresolved_defer_cap)
        #: #944b THE LAP-FREE BOUND: rid -> (told last offered, how many
        #: CONSECUTIVE passes that identical told has been offered).
        #:
        #: WHY THIS EXISTS AND `_unresolved_rounds` DOES NOT SUFFICE, measured
        #: on the live rig 2026-08-27: `_unresolved_rounds` is fed by
        #: `record_return_trip`, which is fed by a RING LAP -- the void output
        #: carrying the chain-reconciled decision home
        #: (`scheduler_pp_mixin.py:6357` -> `_pp_void_output_payload:6394` ->
        #: `pp_output_payload_with_return_trip:1060`, absorbed at
        #: `_pp_absorb_void_output:6414` -> `pp_absorb_admission_return:6473`).
        #: THE VOID THAT MUST BE COUNTED IS THE SAME EVENT THAT BLOCKS THE LAP:
        #: the voided pass parks a middle rank in `_pp_drain_voided_proxy`
        #: (blocking `_pp_recv_typed_dict`), so the output never completes the
        #: ring, PP0 never absorbs, the guard never learns, and the cap never
        #: arms. Measured: 4010 UNRESOLVED, 0 UNRESOLVABLE, ONE rid, 8023 lines,
        #: `told=8192` on every one of them -- the offer never moved.
        #:
        #: So the bound may not depend on ANY downstream fact. This one is
        #: derived entirely from what PP0 itself does: it re-offered the same
        #: rid the same length again. That observation needs no peer, no lap and
        #: no collective, which is exactly why it survives the failure it is
        #: meant to bound.
        self._offer_streak: Dict[str, Tuple[int, int]] = {}
        #: #987 rids whose LAST OFFERED PASS came back VOIDED, i.e. rids this
        #: rank offered and that were demonstrably NOT SERVED.
        #:
        #: WHY THE STREAK NEEDED A SECOND KEY, measured on boots 6-7 (R9, the
        #: 506/513 census). `_offer_streak` above bounds the loop by asking
        #: "did the OFFER stop moving?", and #944b's reasoning for that key is
        #: sound for the failure it was written against (one rid, `told=8192`,
        #: 8023 identical lines). It is blind to the failure actually on the
        #: rig: rank 0 ALTERNATES between two offers for the same rid --
        #: `told=7939` when it re-offers it as a waiting-queue member and
        #: `told=0` when it re-offers it as `chunked_req` -- so the offer moves
        #: on every single pass, the streak resets on every single pass, and
        #: the cap armed ONCE in 506 laps. A moving offer was read as progress;
        #: it was two spellings of the same standstill.
        #:
        #: SO THE KEY IS THE REFUSAL, NOT THE OFFER. What this set records is
        #: the one fact the alternation cannot fake: the pass carrying that
        #: offer came home VOID. It is PP0-local in exactly the sense #944b
        #: requires -- `_pp_absorb_void_output` is rank 0's own consumption of
        #: its own launched batch, needs no peer to answer and no lap to
        #: complete, and runs precisely on the passes that fail. It is NOT
        #: `_unresolved_rounds`: that one is fed by the chain-reconciled
        #: decision, which `_pp_refuse_forwarded_schedule` EMPTIES
        #: (`void_pp_admission_decision`) before it laps home, so the refused
        #: rid is not in it -- the reason R9 measured `UNRESOLVABLE=0` beside a
        #: 506-refusal census.
        #:
        #: CONSUMED BY `note_offer`, one mark per offer: a mark set by pass N's
        #: void is read by pass N+1's offer and cleared there, so the set never
        #: grows past the rids currently in flight and a rid that starts being
        #: served resets by simply not being marked again.
        self._refused_since_offer: Set[str] = set()
        #: #955 rid -> the `told` at which the RECOMPUTE TERMINATOR was spent.
        #:
        #: WHAT IT BUYS: the standing law allows a dead-premise recompute as a
        #: NAMED emergency exit, once per request, with its discard count. It
        #: does not allow a stream of them. Presence here means "this rid has
        #: already had its one exit", and `note_offer` will clamp the offer
        #: without re-arming the escalation that drives the exit.
        #:
        #: DELIBERATELY CLEARED ONLY BY AN ADMITTED LAP (`record_return_trip`),
        #: which is the very lap-gating #955 removes from `_escalated` -- and
        #: the asymmetry is the point, because the DIRECTION of an unreachable
        #: clear is opposite here. An unreachable clear on `_escalated` re-arms
        #: an actuator for ever (a loop); an unreachable clear here can only
        #: withhold a SECOND recompute, which is what the law asks for anyway.
        #: A lifecycle end whose failure mode is "the emergency exit is not
        #: offered twice" needs no lap-free escape hatch. Being served is also
        #: the only honest proof that the premise genuinely changed: the offer
        #: moving to 0 is this guard's OWN clamp, not independent evidence, so
        #: it must not re-open the exit.
        self._terminator_spent: Dict[str, int] = {}

    def terminator_spent(self, rid: str) -> Optional[int]:
        """#955 diagnostic/test hook: the `told` this rid's terminator was
        spent at, or None if it still has its one exit."""
        return self._terminator_spent.get(rid)

    def note_terminator_spent(self, rid: Optional[str]) -> None:
        """#955 THE ESCALATION'S OWN CONSEQUENCE ENDS IT.

        Called by the recompute terminator (`scheduler_pp_mixin.
        pp_apply_dead_premise_at_chunk_boundary`) at the moment it spends
        itself, from the same branch that deletes the request-side marks.

        WHY THE TERMINATOR AND NOT THE RING. `_escalated` was discarded at
        exactly one site -- the `elif entry.admitted:` branch of
        `record_return_trip` -- i.e. only after a completed PP ring round-trip.
        The escalation's own consequence (mark dead -> decline the re-fetch ->
        spend the terminator -> void the pass) is what PREVENTS that
        round-trip, so the only operation able to clear the flag was
        unreachable from the path the flag creates. Measured on metal
        (window-951-boot, both boots byte-identical): 87 `#946 PREMISE
        RECOMPUTE`, 85 of them one rid, 8192 tokens each -- 696,320 tokens
        re-prefilled for a single request in 14 s, and `told=8192` unchanged
        across 344 offers.
        #944b learned this for the INCREMENT and made `_offer_streak` lap-free.
        This is the same lesson applied to the CLEAR, which the comment at the
        admitted branch below had recorded as a deliberate asymmetry -- true
        only while a pass that serves the rid is reachable, which is precisely
        what the escalation prevents.

        TAKES NO `told`: it reads this guard's OWN last offer for the rid.
        Passing the value in from the scheduler would create a second
        expression for a quantity that already has one, which is the defect
        `_executed_extent` exists to avoid one level up.
        """
        if rid is None:
            return
        prev_told, _streak = self._offer_streak.get(rid, (None, 0))
        if prev_told is not None:
            self._terminator_spent[rid] = int(prev_told)
        else:
            # No offer on record (a stand-in, or a rid escalated by a path that
            # never counted). The exit was still spent, so it must still be
            # recorded; -1 names "spent, at an offer this guard never saw".
            self._terminator_spent.setdefault(rid, -1)
        self._escalated.discard(rid)

    def unresolved_rounds(self, rid: str) -> int:
        """#944 diagnostic/test hook: consecutive unresolved rounds for `rid`.

        LAP-FED, so it reads 0 exactly when the ring is broken -- which is when
        it would matter most. Kept because it NAMES the population correctly
        when laps do arrive; it is no longer what bounds anything. See
        `offer_streak`.
        """
        return self._unresolved_rounds.get(rid, 0)

    def note_offer(self, rid: str, told: int) -> bool:
        """#944c RECORD ONE OFFER. True iff `rid` has now exceeded the cap.

        COUNTING IS NOT CLAMPING, and splitting them is the whole of #944c.
        The bound needs to SEE every offer production makes; it may only ACT on
        the offers it is allowed to rewrite. Those are different sets, and
        conflating them is what put the counter on one branch of a two-branch
        function.

        THE TWO BRANCHES, measured. `build_pp_admission_decision` constructs an
        offer in two places: the EXECUTED branch (`_executed_extent` returned a
        geometry -- an already-admitted, mid-chunked-prefill request being
        re-offered out of `can_run_list`, which is the production case) and the
        FALLBACK branch (no executed geometry). Until #944c only the fallback
        consulted the guard, so on the rig the streak stayed at 0 while two
        rids were re-offered `told=8192` thousands of times and the cap never
        armed. This method is what both branches now call.

        It never rewrites anything, so the EXECUTED branch can call it safely:
        that branch must REPORT the prefix the rank actually used, and a value
        derived from anything else names a pass no rank ran (the instr21
        defect, and the reason this class's docstring forbids re-applying the
        clamp there). See `prefix_len_for` for the acting half.
        """
        if self._unresolved_defer_cap <= 0:
            return False
        told = int(told)
        prev_told, streak = self._offer_streak.get(rid, (None, 0))
        # #987 THE REFUSAL KEY, read and spent in the same breath. One mark per
        # offer: pass N's void sets it, pass N+1's offer consumes it. See
        # `_refused_since_offer` for why the OFFER moving is not evidence that
        # anything moved, and why this is.
        refused = rid in self._refused_since_offer
        self._refused_since_offer.discard(rid)
        if prev_told == told or refused:
            streak = streak + 1
        else:
            # #955 THE LAP-FREE END OF THE ESCALATION, and the exact
            # counterpart of the lap-free INCREMENT #944b built two tickets
            # ago. The offer moving is the same observation the streak reset
            # already trusts -- PP0's own, needing no peer, no lap and no
            # collective -- so an escalation raised because the offer STOPPED
            # moving must end when it moves again. Leaving it set was what let
            # `is_escalated` (read at scheduler.py:9325) re-write the
            # dead-premise mark every pass from state the terminator never
            # touches.
            #
            # #987 NARROWED, NOT REMOVED, and the narrowing is one word: this
            # branch now requires that the offer moved AND that the last pass
            # carrying it was not refused. #955's argument survives intact for
            # the case it was written about (an offer that moves on a ring that
            # turns); what it never covered is an offer that moves on a ring
            # that voids, which is the alternation R9 measured. Ending an
            # escalation on evidence the failure itself manufactures is the
            # #939 class -- a compensator answered by the defect it compensates
            # -- so the end is gated on the one fact the defect cannot forge.
            streak = 1
            self._escalated.discard(rid)
        self._offer_streak[rid] = (told, streak)
        # `told <= 0` is already the terminator, so a rid sitting there is
        # progressing by definition and is never "over the cap".
        #
        # #987 "BY DEFINITION" HELD ONLY WHILE told=0 MEANT PROGRESS. On the
        # rig it is half of the alternation: rank 0 offers the SAME rid
        # `told=0` on every pass it re-offers it as `chunked_req`, and 168 of
        # R9's 506 refusals are that offer being refused (`names prefix 0,
        # holds 7939`). A rid whose told=0 pass came home VOID is not
        # progressing -- it is not being served at all -- so the exemption is
        # withdrawn for exactly that rid and kept for every other. This cannot
        # re-arm the recompute: `_terminator_spent` below still answers the
        # second escalation with a clamp.
        if (told <= 0 and not refused) or streak <= self._unresolved_defer_cap:
            return False
        if rid in self._terminator_spent:
            # #955 CLAMP, BUT DO NOT RE-ARM. The rid has already had its one
            # named recompute. The offer is still pinned to `told=0` by the
            # return value below -- that is the FORWARD exit, admitted
            # unconditionally by `reconcile_pp_admission_decision`, and it
            # costs nothing to keep making it. What must not happen again is
            # the ESCALATION, because that is what re-arms the actuator and
            # buys a second full re-prefill of the same request.
            #
            # Counting is not clamping (#944c) and clamping is not escalating
            # (#955): three separate questions that used to share one boolean.
            return True
        if rid not in self._escalated:
            self._escalated.add(rid)
            logger.error(
                "#944 PP-ADMISSION UNRESOLVABLE rid=%s: this rank has offered "
                "the SAME prefix length (told=%d) for %d consecutive passes "
                "without the request ever being served. Reported unresolved "
                "rounds for it: %d -- if that is 0 while this streak is not, "
                "the return trip is not completing, which is the ring being "
                "blocked by the very void this bound exists to end. No rank "
                "could locate the request in any of the four places one can "
                "live -- the waiting queue, `chunked_req`, the named slot's "
                "chunked req, or the running batch -- so no rank measured a "
                "prefix, no floor can be learned, and the offer cannot be "
                "damped by measurement. Where this rank is allowed to rewrite "
                "the offer it now offers told=0, which is honourable without a "
                "measurement; where it is only allowed to REPORT what already "
                "executed it leaves the geometry alone and this line is the "
                "whole of its response. If this repeats for many rids, the "
                "lookup chain has a FIFTH gap and that is the bug -- not this "
                "refusal.",
                rid,
                told,
                streak,
                self._unresolved_rounds.get(rid, 0),
            )
        return True

    def note_pass_refused(self, rids: Iterable[str]) -> int:
        """#987: these rids were offered on a pass that came home VOID.

        The write side of `_refused_since_offer`. Called by PP0 as it absorbs
        a void output (`scheduler_pp_mixin._pp_absorb_void_output`), over the
        members of the batch that void names -- so the argument is rank 0's
        own launched batch, not a downstream report, and no lap has to
        complete for it to be true.

        WHY NOT `record_return_trip`. That is the other half of the same
        message and it is fed the CHAIN-RECONCILED decision, which
        `_pp_refuse_forwarded_schedule` empties with
        `void_pp_admission_decision` before the void starts home. A schedule
        refusal therefore arrives as a void carrying ZERO entries: the pass is
        known to have failed and the rid that failed it is not in the payload.
        That is the exact shape R9 measured (506 refusals, `UNRESOLVABLE=0`,
        `_unresolved_rounds` never incremented), and it is why the refusal has
        to be read off the batch rather than off the decision.

        IDEMPOTENT AND BOUNDED. A set, marked once per void and spent by the
        next `note_offer` for that rid; several voids between two offers leave
        one mark, which is correct -- the streak counts OFFERS that were not
        served, not voids. Returns how many rids were newly marked, for the
        caller's instrument line. Never raises on a member with no rid.
        """
        # BOUNDED. A mark is normally spent by the next offer for that rid, but
        # a rid that voids and is then finished or aborted is never offered
        # again and its mark would sit here for the life of the process. The
        # bound is generous relative to the rids that can be in flight at once,
        # so clearing it can only ever cost a streak one lap of counting -- it
        # cannot make the bound unreachable, because the void that would refill
        # it repeats every pass.
        if len(self._refused_since_offer) > 4096:
            self._refused_since_offer.clear()
        marked = 0
        for rid in rids:
            if not rid:
                continue
            rid = str(rid)
            if rid not in self._refused_since_offer:
                marked += 1
            self._refused_since_offer.add(rid)
        return marked

    def refused_since_offer(self, rid: str) -> bool:
        """#987 diagnostic/test hook: is this rid carrying an unspent refusal
        mark? Reads False for a rid that has never been refused and for one
        whose mark the next offer already spent."""
        return rid in self._refused_since_offer

    def is_escalated(self, rid: str) -> bool:
        """#946: has this rid's prefix premise been declared dead?

        The read side of `note_offer`'s escalation, exposed so the SCHEDULER
        can act on it at the one point where acting is legal (the chunk
        boundary). This class must not reach into the scheduler itself -- it is
        the pure half of the feature and importing upward would be the cycle
        the module docstring's NO COLLECTIVE section is careful about.
        """
        return rid in self._escalated

    def offer_streak(self, rid: str) -> int:
        """#944b diagnostic/test hook: consecutive identical re-offers of `rid`.

        The LAP-FREE counter that actually bounds the loop. Reads 0 for a rid
        that has never been offered or whose offer just changed.
        """
        return self._offer_streak.get(rid, (None, 0))[1]

    def learned_floor(self, rid: str) -> Optional[int]:
        """Diagnostic/test hook: the outstanding floor for `rid`, or None.

        The SIBLING of `unresolved_rounds` above, and both exist so a test can
        tell the two populations apart from the outside -- reading them off one
        accessor, or off `prefix_len_for`, would fold them back together at the
        exact seam this ticket is about.
        """
        return self._learned_floor.get(rid)

    #: #963: cap on `_prefix_floor` / `_offer_prefix_key`. The rid-scoped
    #: tables are bounded by the live request population and clear on serve;
    #: a prefix-scoped one is bounded by nothing, and an unbounded dict on the
    #: admission path is a leak that only shows up after hours. Oldest entry
    #: evicted first -- losing a floor is safe in the direction that matters
    #: (the prefix is re-offered, one rank retracts, the floor is re-learned),
    #: whereas losing a RECENT one would reopen the livelock.
    PREFIX_FLOOR_SLOTS = 512

    def _remember_prefix_floor(self, key: int, observed: int) -> None:
        """Tighten (never widen) the floor for one prefix, under the cap."""
        existing = self._prefix_floor.get(key)
        self._prefix_floor[key] = (
            int(observed) if existing is None else min(int(existing), int(observed))
        )
        while len(self._prefix_floor) > self.PREFIX_FLOOR_SLOTS:
            self._prefix_floor.pop(next(iter(self._prefix_floor)))

    def prefix_len_for(
        self, rid: str, candidate_prefix_len: int, *, prefix_key: Optional[int] = None
    ) -> int:
        """What PP0 should tell downstream ranks for `rid` this pass.

        `candidate_prefix_len` is PP0's own fresh local match -- what
        `build_pp_admission_decision` would tell without this guard.
        Clamped to the learned floor if one is outstanding for `rid`;
        returned unchanged otherwise (a rid with no retraction history is
        not constrained by this guard at all).

        #944 ESCALATION, AND IT IS DECIDED HERE BECAUSE ONLY PP0 CAN ACT ON IT.
        A rid PP0 has re-offered the SAME `told` for more than
        `UNRESOLVED_DEFER_CAP` consecutive passes is pinned to `told=0` and
        refused loudly, once. Downstream ranks observe the miss but cannot fix
        it -- they do not choose `told`, and a rank that rewrote the geometry
        mid-ring would be the instr20 crash -- so the single decision is here.

        #944b THE TRIGGER IS PP0'S OWN RE-OFFER, NOT A REPORTED ROUND COUNT,
        and that correction is the whole lesson of the 2026-08-27 acceptance
        boot. The first version counted `_unresolved_rounds`, fed by
        `record_return_trip`, fed by a RING LAP. But the lap is carried by the
        void output, and the void parks a middle rank in
        `_pp_drain_voided_proxy` -- so the event that must be counted is the
        same event that stops the counting. Measured: 4010 UNRESOLVED lines, 0
        escalations, one rid, `told=8192` on all 8023 of its lines. The bound
        was dead code on precisely the path it was built for, which is the
        #939 class ("a compensator made unreachable by the refusal it exists to
        compensate") for the second time.

        A RE-OFFER IS OBSERVABLE WITHOUT ANY PEER. PP0 already knows what it
        told last pass and what it is about to tell now; if those are equal
        again and again, the loop is not progressing, and that is true whoever
        broke the ring and whether or not anything ever comes back. Any bound
        that needs a downstream fact needs a lap, and the lap is the thing the
        defect breaks -- so a lap-free trigger is not a convenience here, it is
        the only kind that can work.

        IT DOES NOT MISFIRE ON THE #630 SHORTFALL PATH. A genuine shortfall
        learns a floor, and `_learned_floor` makes the next `told` STRICTLY
        SMALLER (see this class's termination argument above), so the streak
        resets on every healthy retraction and never reaches the cap. Only a
        `told` that stops moving -- which is exactly the non-progressing case --
        can accumulate. `_unresolved_rounds` is kept as the population NAME and
        as corroboration when laps do arrive; it no longer bounds anything.

        WHY told=0 WHEN THIS CLASS'S OWN DOCSTRING REJECTS A told=0 PIN. It
        rejects it as the GENERAL policy for an ordinary shortfall, where
        learning the real observed value costs the same and keeps the degrade
        rare. That argument does not reach here: a lookup miss measured
        NOTHING, so there is no value to learn, and `told=0` is the only offer
        that is honourable without a measurement (`reconcile_pp_admission_
        decision` admits it unconditionally). It is therefore not the rejected
        shape creeping back in -- it is the only terminator available once the
        measurable one is gone. The cost is one request's prefix reuse, once,
        after three voided rounds; the alternative is the unbounded re-offer,
        which is the #858 livelock shape.
        """
        # #963: the offer is clamped by the poorest of the two scopes. The rid
        # floor is what THIS request was already told it could not have; the
        # prefix floor is what ANY request over this prefix was measured
        # short of. Both are honest observations from a rank that looked, and
        # the group has to live with the poorest of them.
        #
        # `prefix_key=None` -- a caller that cannot name the prefix -- leaves
        # the behaviour byte-identical to the pre-#963 rid-scoped path, so
        # this cannot change a call site it was never reasoned about on.
        floor = self._learned_floor.get(rid)
        if prefix_key is not None:
            self._offer_prefix_key[rid] = prefix_key
            while len(self._offer_prefix_key) > self.PREFIX_FLOOR_SLOTS:
                self._offer_prefix_key.pop(next(iter(self._offer_prefix_key)))
            prefix_floor = self._prefix_floor.get(prefix_key)
            if prefix_floor is not None:
                floor = prefix_floor if floor is None else min(floor, prefix_floor)
        told = (
            candidate_prefix_len if floor is None else min(candidate_prefix_len, floor)
        )
        # #944c COUNT, THEN ACT -- and the counting half is shared with the
        # EXECUTED branch, which may count but must never be clamped.
        return 0 if self.note_offer(rid, told) else told

    def record_return_trip(self, decision: PPAdmissionDecision) -> None:
        """Consume a fully chain-reconciled decision on its way back to PP0.

        For every entry:
          * retracted this pass -> learn `_learned_floor[rid] = observed_local`,
            tightening (never widening) any floor already outstanding -- a
            stricter rank's finding must never be overwritten by a looser one.
          * admitted, not retracted -> the rid was served this pass with no
            shortfall anywhere in the chain; clear any outstanding floor.
          * neither (excluded before this module ever saw it, e.g. by PP0's
            own local admission control) -> not this guard's concern, left
            untouched.

        Performs no distributed communication and never blocks: this is a
        plain dict update over already-local data, the same
        `amended_decision` value the caller already has in hand for
        forwarding.
        """
        for entry in decision.entries:
            if entry.retracted:
                if entry.unresolved:
                    # #944 THE OTHER POPULATION, COUNTED SEPARATELY. No rank
                    # could locate this rid, so there is nothing to learn a
                    # floor from -- but the round still happened and still cost
                    # a voided pass, and something has to bound how many times
                    # that may repeat. Counted here rather than at the reporting
                    # rank because the count has to be the GROUP's: every rank
                    # reports independently, only PP0 acts (`prefix_len_for`).
                    #
                    # INCREMENT-ONLY ON THIS PATH, cleared only by a pass that
                    # actually served the rid (below). A defer that resets its
                    # own counter makes the bound unreachable, which is #552's
                    # measured lesson: "the bug itself wearing a fix".
                    self._unresolved_rounds[entry.rid] = (
                        self._unresolved_rounds.get(entry.rid, 0) + 1
                    )
                    continue
                observed = entry.observed_local
                if observed is None:
                    # A retraction without an observed value cannot safely
                    # tighten a floor -- leave any existing floor as-is rather
                    # than guess.
                    #
                    # #944 MADE THIS AN EXPECTED PATH, and the comment is
                    # corrected rather than left to mislead. It used to read
                    # "every retraction this module itself produces always sets
                    # observed_local, so this branch guards a malformed/foreign
                    # decision, not an expected path". The #944 UNRESOLVED
                    # branch now retracts with `observed_local=None` on purpose:
                    # a LOOKUP MISS taught this pass nothing about the rank's
                    # prefix, so there is no lesson to learn, and inventing one
                    # (0, or the -1 sentinel) would clamp the next round's offer
                    # from a number that was never measured. The guard was
                    # already the correct behaviour; only its reachability
                    # changed.
                    continue
                existing = self._learned_floor.get(entry.rid)
                self._learned_floor[entry.rid] = (
                    observed if existing is None else min(existing, observed)
                )
                # #963: THE SAME LESSON, SCOPED TO THE PREFIX. The retracting
                # rank measured its own tree against the prefix this offer was
                # made over; that measurement is true of every other request
                # over the same prefix until the trees change. Learning it here
                # is what turns "one voided pass per rid, for ever" into "one
                # voided pass, once".
                offered_key = self._offer_prefix_key.get(entry.rid)
                if offered_key is not None:
                    self._remember_prefix_floor(offered_key, observed)
            elif entry.admitted:
                self._learned_floor.pop(entry.rid, None)
                # #963: SERVED means every rank admitted this pass, ran it and
                # reached its own `cache_unfinished_req` -- so the trees agree
                # on this prefix again and the floor must go. Holding it would
                # convert a transient divergence into permanent cache loss on a
                # prefix every rank holds, which is the one direction this fix
                # must never fail in.
                served_key = self._offer_prefix_key.pop(entry.rid, None)
                if served_key is not None:
                    self._prefix_floor.pop(served_key, None)
                # #944: the rid was SERVED, so both kinds of outstanding state
                # are stale -- the learned floor and the unresolved streak. The
                # streak must clear here and nowhere else: one bad minute of
                # lookups must not cap a request's reuse for the rest of its
                # life, and the escalation must be able to fire again if the
                # rid genuinely becomes unresolvable a second time.
                self._unresolved_rounds.pop(entry.rid, None)
                self._escalated.discard(entry.rid)
                # #955: the rid was SERVED, so its one emergency exit is
                # restored along with everything else. This is the ONLY site
                # that restores it, and being served is the only honest proof
                # that the premise genuinely changed -- see
                # `note_terminator_spent` for why an unreachable clear is the
                # safe direction here and was the fatal one for `_escalated`.
                self._terminator_spent.pop(entry.rid, None)
                # #944b The lap-free streak clears here TOO, and only here.
                # The asymmetry is deliberate and is the point: the INCREMENT
                # must survive a broken ring (so it is lap-free), the CLEAR
                # only has to work when the ring turns -- and a pass that
                # actually served the rid is, by construction, a pass on which
                # it turned.
                self._offer_streak.pop(entry.rid, None)
                # #987: and so does the refusal mark, on the same argument. A
                # rid that reached here was served, so any mark still standing
                # for it describes a pass that is now history; leaving it would
                # let one old void count against the first offer of a rid that
                # is demonstrably being served again.
                self._refused_since_offer.discard(entry.rid)

    def outstanding_rids(self) -> Tuple[str, ...]:
        """Diagnostic/test hook: rids currently carrying a learned floor."""
        return tuple(self._learned_floor)


def fill_carry_for(req) -> Tuple[Optional[int], Tuple[int, ...]]:
    """#987: `(fill_len, fill_tail)` this rank should PUBLISH for `req`.

    Read off the request as it stands on the deciding rank: the length of its
    `full_untruncated_fill_ids` and the last `FILL_CARRY_TAIL_CAP` OUTPUT
    tokens of it. `(None, ())` when the request carries no readable fill,
    which is the pre-#987 wire content exactly.

    ONLY OUTPUT TOKENS ARE ELIGIBLE, and the bound is not cosmetic. The tail
    is capped at `len(output_ids)` as well as at `FILL_CARRY_TAIL_CAP`, so
    what this publishes can only ever be tokens this rank GENERATED. The
    prompt half of the fill is `origin_input_ids`, which every rank receives
    from the tokenizer over its own channel and which no rank may amend from a
    peer -- carrying prompt tokens here would be the double-prefill law's
    forbidden direction (a request's input rewritten by another rank), and the
    seam defect never produces one: `tp_to_pp` strands a SAMPLED token.

    NO SYNCHRONISATION AND NO TENSOR (#790, #796). `full_untruncated_fill_ids`
    and `output_ids` are `array("q")` host arrays (schedule_batch.py:742,746);
    `len()` and a small trailing slice of them touch no device.
    """
    fill_ids = getattr(req, "full_untruncated_fill_ids", None)
    if fill_ids is None:
        return (None, ())
    try:
        fill_len = int(len(fill_ids))
    except Exception:  # noqa: BLE001 - an unreadable fill names no fill
        return (None, ())
    output_ids = getattr(req, "output_ids", None)
    n_output = 0 if output_ids is None else len(output_ids)
    k = min(FILL_CARRY_TAIL_CAP, int(n_output), fill_len)
    if k <= 0:
        return (fill_len, ())
    try:
        tail = tuple(int(t) for t in fill_ids[fill_len - k :])
    except Exception:  # noqa: BLE001 - an unreadable tail names no tail
        return (fill_len, ())
    return (fill_len, tail)


def _last_chunk_verdict(
    prefix_len: int, extend_len: int, fill_len: Optional[int]
) -> Optional[bool]:
    """#996: does `prefix_len + extend_len` finish a request of `fill_len`?

    `None` when the fill is unreadable -- the same "say nothing" that
    `fill_carry_for` returns for that case, and for the same reason: a verdict
    derived from a length nobody could read is worse than no verdict, because
    the receiver's fallback is at least honest about being local.

    ONE EXPRESSION, ON THE DECIDING RANK. This is deliberately the same
    comparison the receiver used to make for itself
    (`schedule_policy.py:_add_scheduled_req`); what changes is WHOSE
    `fill_len` it is asked against. Keeping the arithmetic identical and
    moving only the authority is what makes this a carry rather than a new
    rule -- there is no second definition of "last chunk" to drift.
    """
    if fill_len is None:
        return None
    return int(prefix_len) + int(extend_len) >= int(fill_len)


#: #1040 counters. Kept apart on purpose: a shortfall whose anchor is simply
#: deeper in the tree than the host tier reaches is an ordinary alignment, while
#: a MATCH THAT FOUND NO ANCHOR AT ALL is the #1039 population -- the tree kept
#: the node's shape and lost its recurrent payload. Summing them would produce
#: one number that cannot answer either question.
#:
#: AND THE LOSS STATISTIC HAS ITS OWN DENOMINATOR, separate from the total
#: refusals. A round-down with an anchor present gives back PART of the hit --
#: that is the population the user's "at most one HiCache chunk" grant is about,
#: and the number that has to be reported against it. An extent of 0 because
#: there was NO anchor gives back the WHOLE hit, which is a different event with
#: a different fix; pooling the two inflates the loss median and max with values
#: that have nothing to do with rounding. Measured while building this: the
#: pooled form reported loss_max=300 on a fixture whose only genuine round-down
#: lost 200.
_1040_ALIGN = {
    "n": 0,  # extents chosen
    "rounded": 0,  # extents an anchor actually moved (anchor > 0)
    "loss_sum": 0,  # tokens given back BY ROUNDING (anchor > 0 only)
    "loss_max": 0,
    "anchor_absent": 0,  # anchor gone while the KEY still matched
    "absent_forgone": 0,  # tokens given back because there was NO anchor
    "below_first_anchor": 0,  # no anchor because the match is short: legitimate
}


#: #1041: the field the WRITER-SIDE choice lands in. Read by the row builder,
#: never re-derived there.
LOAD_BACK_EXTENT_ATTR = "pp_load_back_extent"


def stamp_state_aligned_extent(req) -> Optional[int]:
    """#1041: CHOOSE THE EXTENT WHERE THE FACT IS BORN, not where a batch is.

    THE DEFECT THIS CLOSES, measured on boot 8 (1040round). The extent used to
    be derived inside `build_pp_admission_decision`, which is reached only from
    `_get_new_batch_prefill_raw`'s tail -- AFTER an admission loop with eight
    `continue` branches and an `if len(can_run_list) == 0: return` above it. The
    request that HELD the host hit was skipped at scheduler.py:10170
    (`prefetch_pending`) before the adder ever saw it, the list came back empty,
    the method returned, and no decision was built at all: three ranks logged
    `#788 PP-ADMISSION verdict=DECLINE ... reason=loop_skips(prefetch_pending=
    1(first=f7f997c0fafc451e))` at 09:13:52 and the same rid logged
    `#968 LOAD-BACK DEFERRED ... holds no PP0 extent for it yet` two seconds
    later. The chain fed itself: hit present -> prefetch pending -> skipped ->
    empty list -> no decision -> no extent -> deferral.

    WHY THE WRITER IS THE RIGHT PLACE, and why this is not one more per-path
    patch. `Req.host_hit_length` has exactly TWO live writers -- this call's two
    sites, `Req.init_next_round_input` and `match_prefix_for_req` -- both
    unpacking one `match_prefix` result; `Req.__init__` only zeroes it, and
    `truncate_prefix_to` is dead code (0 callers, 0 name reads). A request
    therefore CANNOT carry a nonzero host hit without executing one of them.
    Every `can_run_list` filler -- `add_one_req`, `add_chunked_req`,
    `add_one_req_ignore_eos`, the dllm pair, `_add_scheduled_req` -- must match
    before it is executable, so the match dominates all of them. That is the
    dominator argument, and it is made over the WRITERS on purpose: the call
    graph cannot carry it (`call_path add_chunked_req -> match_prefix_for_req`
    walks past 156 unresolved edges and finds nothing, which is a bounded
    negative and no proof of anything).

    Rank-local by design and harmless: every rank stamps its own, but only PP0's
    is ever published (`scheduler.py`'s `pp_rank == 0` gate), and every other
    rank reads the told value off the row. Uniformity still comes from ONE rank
    choosing, exactly as before -- what changes is only WHEN it chooses.
    """
    extent = state_aligned_load_back_len(req)
    try:
        setattr(req, LOAD_BACK_EXTENT_ATTR, extent)
    except Exception:  # noqa: BLE001 - never break a match walk
        pass
    return extent


def state_aligned_load_back_len(req) -> Optional[int]:
    """#1040: PP0's load-back extent, rounded DOWN to a state-bearing boundary.

    THE USER'S GRANT, IN ONE EXPRESSION: up to one HiCache chunk may be
    re-prefilled, and it is re-prefilled FROM THE KV POSITION THAT MATCHES THE
    MOST CURRENT RECURRENT STATE. KV is divisible -- any prefix length is a
    legal place to stop -- while the mamba/GDN state is pointlike and cannot be
    trimmed to a length it does not belong to. Where the two disagree the KV
    yields, because giving back tokens costs recompute while keeping them costs
    correctness (`schedule_policy.py`'s FIX-3 comment spells the same hazard out
    at the receiving end).

    The anchor is NOT recomputed here. `req.state_anchor_depth` is the depth of
    the node the match walk's own validators accepted -- mamba's being
    `is_resume_candidate` itself -- so this function only clamps against a
    decision that was already made, once, in the one place that makes it. A
    second anchor rule here would be the second bookkeeping the upstream-minimal
    law rejects, and #747 records what two anchor lineages do to each other.

    THREE ANSWERS, AND THEY ARE NOT THE SAME ANSWER:

      ``None`` from `state_anchor_depth` -- this cache has NO state-bearing
        component (pure KV). Nothing to align to, every length is valid, the
        extent is returned untouched. That is what keeps `pp_size <= 1` and
        every upstream configuration byte-for-byte what they were.
      ``0`` extent -- the match reached no acceptable anchor. "Load back
        nothing" is then the CORRECT verdict, not a failure: under leaf-only
        mamba data a split nulls the parent's state, so the candidate set along
        one path is typically one point or none. It is counted, because a zero
        anchor under a LONG key match is the #1039 symptom (the anchor died with
        an evicted node) and a zero anchor under a SHORT one is just a request
        that has not reached the first checkpoint yet. The counter separates
        them; the action is the same either way.
      a SHORTENED extent -- the anchor sits below the host hit's end. The
        difference is the loss, in tokens, and it is summed and maxed here so a
        boot can state the user's "a few thousand tokens" expectation as a
        MEASURED number instead of an assumption.

    NOTE ON THE EXPECTED SIZE OF THE LOSS: with `--mamba-checkpoint-interval`
    set, anchors sit on a grid and the loss is bounded by the interval. With the
    interval OFF the anchor is wherever traffic last committed one, so there is
    no bound to compare against and the distribution measured here IS the
    finding. Do not read a large loss as a defect without first reading the
    interval the boot ran with.
    """
    kv = int(getattr(req, "host_hit_length", 0) or 0)
    if kv <= 0:
        return None
    anchor = getattr(req, "state_anchor_depth", None)
    if anchor is None:
        # Pure-KV cache: no state to align to. Upstream's number, unmodified.
        return kv or None

    # `state_anchor_depth` is ABSOLUTE (from the root) while the row carries a
    # DELTA (how many host tokens to pull in beyond what the device already
    # holds), so the anchor has to be expressed in the same coordinate before
    # it can clamp anything. `prefix_indices` is the device-resident half; its
    # length is read with `len()`, never a boolean context, because it is a
    # tensor of pool pointers (#796).
    prefix_indices = getattr(req, "prefix_indices", None)
    device_len = 0 if prefix_indices is None else len(prefix_indices)
    room = int(anchor) - int(device_len)
    extent = kv if room >= kv else max(0, room)

    _1040_ALIGN["n"] += 1
    loss = kv - extent
    key_depth = getattr(req, "key_match_depth", None)
    absent_class = None
    if int(anchor) <= 0:
        # No anchor at all -- the WHOLE hit is given back. Which of the two
        # worlds is it? A long key match with no surviving anchor is the #1039
        # symptom (the tree kept the node's shape and lost its recurrent
        # payload); a short one is simply a request that has not reached a
        # checkpoint yet. Same action, different finding.
        _1040_ALIGN["absent_forgone"] += loss
        if key_depth is not None and int(key_depth) > 0:
            _1040_ALIGN["anchor_absent"] += 1
            absent_class = "ANCHOR-ABSENT-ON-MATCH"
        else:
            _1040_ALIGN["below_first_anchor"] += 1
            absent_class = "below-first-anchor"
    elif loss > 0:
        # THE ROUNDING PROPER, and the only population the user's "at most one
        # HiCache chunk" bound is a statement about.
        _1040_ALIGN["rounded"] += 1
        _1040_ALIGN["loss_sum"] += loss
        if loss > _1040_ALIGN["loss_max"]:
            _1040_ALIGN["loss_max"] = loss

    n = _1040_ALIGN["n"]
    if loss > 0 or absent_class == "ANCHOR-ABSENT-ON-MATCH" or n <= 5 or n % 64 == 0:
        logger.info(
            "#1040 EXTENT STATE-ALIGN rid=%s kv=%d extent=%d loss=%d "
            "anchor_depth=%s device_len=%d key_match_depth=%s class=%s -- "
            "n=%d rounded=%d loss_sum=%d loss_max=%d anchor_absent=%d "
            "absent_forgone=%d below_first_anchor=%d",
            getattr(req, "rid", None),
            kv,
            extent,
            loss,
            anchor,
            device_len,
            key_depth,
            absent_class or "aligned",
            n,
            _1040_ALIGN["rounded"],
            _1040_ALIGN["loss_sum"],
            _1040_ALIGN["loss_max"],
            _1040_ALIGN["anchor_absent"],
            _1040_ALIGN["absent_forgone"],
            _1040_ALIGN["below_first_anchor"],
        )
    return extent or None


def forwarded_last_chunk(
    decision: Optional[PPAdmissionDecision],
) -> Dict[str, bool]:
    """#996: the `rid -> last_chunk` map a receiving rank EXECUTES.

    The third projection of the same decision object, alongside
    `forwarded_fill_carry` and `scheduler_pp_mixin._pp_forwarded_schedule_from`,
    and gated identically -- one fact read three times, never three facts to
    keep in step. A voided decision has no entries and yields `{}`.

    Entries with no `last_chunk` (a legacy sender, or one whose fill was
    unreadable) are ABSENT from the map, so an absent rid and a rid with
    nothing to say are the same thing to the reader: fall back to the local
    derivation. That is what keeps a mixed-version group behaving exactly as
    it did before this field existed.
    """
    if decision is None:
        return {}
    out: Dict[str, bool] = {}
    for entry in getattr(decision, "entries", ()) or ():
        verdict = getattr(entry, "last_chunk", None)
        if verdict is None:
            continue
        out[entry.rid] = bool(verdict)
    return out


def forwarded_fill_carry(
    decision: Optional[PPAdmissionDecision],
) -> Dict[str, Tuple[int, Tuple[int, ...]]]:
    """#987: the `rid -> (fill_len, fill_tail)` map a receiving rank adopts from.

    The exact twin of `scheduler_pp_mixin._pp_forwarded_schedule_from`: a
    second projection of the SAME decision object, so the two maps can never
    name different rid sets or different passes. A voided decision has no
    entries and therefore yields `{}`, which is the same emptying the schedule
    map gets on that path -- one fact, read twice, never two facts to keep in
    step.

    Entries with no `fill_len` (a legacy sender, or a rank that published
    nothing) are absent from the map, so an absent rid and a rid with nothing
    to say are the same thing to every reader: no adopt.
    """
    if decision is None:
        return {}
    out: Dict[str, Tuple[int, Tuple[int, ...]]] = {}
    for entry in getattr(decision, "entries", ()) or ():
        fill_len = getattr(entry, "fill_len", None)
        if fill_len is None:
            continue
        out[entry.rid] = (
            int(fill_len),
            tuple(int(t) for t in (getattr(entry, "fill_tail", ()) or ())),
        )
    return out


def _executed_extent(req) -> Optional[Tuple[int, int]]:
    """`(start, length)` of the extend range this rank ACTUALLY BUILT, or None.

    THE BATCH IS THE SCHEDULE -- this reads the very field
    `ScheduleBatch.prepare_for_extend` sizes the cross-stage tensor from
    (`extend_range.start` == `len(prefix_indices)`, `extend_range.length`;
    schedule_batch.py:2261-2281), rather than recomputing a value that ought
    to agree with it. Two expressions that must agree is the defect one level
    up, so there is exactly one expression.

    `None` ONLY for a request the adder never touched. Every one of the eight
    `can_run_list.append(req)` sites in schedule_policy.py is immediately
    preceded by a `set_extend_range` (:1168/:1170, :1199/:1200, :1297/:1298,
    :1420/:1421, :1535/:1538, :1555/:1558, :1742/:1745, :1785/:1789), so a
    member of a real `can_run_list` always has one and the fallback in
    `build_pp_admission_decision` is reachable only from a bare test stand-in.
    That was worth establishing rather than assuming: a `getattr` default
    that silently produced 0, or silently produced the full length, would be
    the instr21 defect in new clothing.

    A ZERO-LENGTH RANGE IS REPORTED, NOT SUPPRESSED. `add_chunked_req`'s #679
    park (schedule_policy.py:1399) sets `extend_range(prefix, prefix)` and
    returns WITHOUT appending, so a parked chunk is not in `can_run_list` at
    all; but a chunk that lands exactly on its last token can be appended with
    `new_len == 0` (:1420-1421). If the first rank ran zero rows for a
    request, every rank must run zero rows for it -- reporting that faithfully
    is the contract, and inventing a length for it would be the same defect
    again. Downstream, `schedule_refusal_reason` therefore refuses a NEGATIVE
    extend and executes a zero verbatim.
    """
    extend_range = getattr(req, "extend_range", None)
    if extend_range is None:
        return None
    start = getattr(extend_range, "start", None)
    end = getattr(extend_range, "end", None)
    if start is None or end is None:
        return None
    start, end = int(start), int(end)
    return start, max(0, end - start)


def build_pp_admission_decision(
    mb_id: int,
    reqs: Sequence,
    *,
    pp_size: int,
    guard: Optional[PPAdmissionCongruenceGuard] = None,
    require_executed_geometry: bool = False,
    fact_only_reqs: Sequence = (),
) -> PPAdmissionDecision:
    """PP0's (or, under `pp_size<=1`, the only rank's) committed decision.

    Reads `req.rid`, `len(req.prefix_indices)`, and the request's own extend
    length (`req.extend_input_len` if present, else derived from
    `full_untruncated_fill_ids` minus the prefix) -- all values this rank
    already computed locally while building its batch. Emits only integers;
    `prefix_indices` itself never leaves this function.

    `guard`, when given and `pp_size > 1`: clamps `prefix_len` to any
    learned floor outstanding for that rid (#630 -- see
    `PPAdmissionCongruenceGuard`), and correspondingly INCREASES `extend_len`
    by the same amount so `prefix_len + extend_len` still equals the
    request's true total length. This is the same physical token count
    either way; clamping `told` down does not shorten the request, it only
    reclassifies some of its tokens from "reused prefix" to "freshly
    computed extend" -- exactly the same accounting
    `reconcile_pp_admission_decision`'s safe-truncate branch already relies
    on implicitly (there, the caller's own downstream scheduling absorbs the
    difference; here, the wire decision must state it explicitly, since it
    is computed before any downstream rank sees it).

    `guard=None` (the default) or `pp_size<=1`: behaviour is byte-identical
    to before this parameter existed -- see DEFAULT PATH above, still true
    with a guard object in play as long as it is never passed a `pp_size<=1`
    caller's decisions to learn from.

    #791 CORE: THE DECISION REPORTS WHAT THIS RANK RAN, NOT WHAT IT WAS
    OFFERED -- and until boot instr21 it reported neither.

    `req.extend_input_len` DOES NOT EXIST. There is no assignment to it
    anywhere in the tree; it survives only as a doc comment
    (schedule_batch.py:1847) and two stale references (:2379-2380). So the
    `getattr` below has ALWAYS returned `None` and the fallback has ALWAYS
    run -- and the fallback computes `len(full_untruncated_fill_ids) -
    prefix`, i.e. the WHOLE remaining prompt. Under chunked prefill that is
    never the batch: `PrefillAdder.add_one_req` caps the chunk at
    `rem_chunk_tokens` AFTER this value would have been read
    (schedule_policy.py:1738-1766), and `add_chunked_req` does the same
    (:1415-1419).

    Nothing noticed for as long as nothing read `extend_len`. The first boot
    that executed it died in 37 seconds (instr21, PP1 10:42:12): a ~17000-
    token drive prompt, one 512-row chunk on the wire, and

        ValueError: #631 PP proxy/batch mismatch: received hidden_states with
                    512 row(s) for a 1 batch of 16983 token(s)

    16983 is exactly `len(fill_ids) - prefix`. The disagreement had moved
    INSIDE the schedule: it named a pass no rank had run.

    `req.extend_range` is the executed geometry. `PrefillAdder` writes it via
    `set_extend_range` on every path that appends to `can_run_list`
    (schedule_policy.py:1719, :1743, :1762, :1202, :1428, and this feature's
    own :1287), and this function is called from scheduler.py AFTER that loop,
    over `can_run_list` itself -- so the value is always already there, and
    the producer needs no move. `prepare_for_extend` derives the tensor's row
    count from the same pair (`extend_range.start` == `len(prefix_indices)`,
    `extend_range.length`; schedule_batch.py:2261-2281), which is what makes
    reporting it -- rather than recomputing an equivalent -- the point.

    THE GUARD IS NOT RE-APPLIED ON THAT PATH, and the old comment claiming it
    was idempotent is now false. The admission loop clamps `prefix_indices` to
    the learned floor BEFORE `add_one_req` (scheduler.py), and
    `add_one_req`'s host load-back then GROWS it again
    (schedule_policy.py:1707-1717) -- so re-running `prefix_len_for` here
    would clamp a second time and move the difference into `extend_len`,
    manufacturing a third geometry nobody ran. The guard's job is upstream, on
    the request, where it belongs; this function's job is to report.

    `require_executed_geometry` (True from scheduler.py, the one production
    call site): a request with NO `extend_range` is a LOUD REFUSAL naming the
    rid, never a default. A silent 0 or a silent full length would be the
    instr21 defect in new clothing, and `None` is REACHABLE here --
    `Req.reset_for_retract` sets `extend_range = None`
    (schedule_batch.py:1588), which is how boot instr19 died at
    scheduler.py:5572 with `AttributeError: 'NoneType' object has no
    attribute 'end'`. Refusing here reaches that same state three frames
    earlier and names the request; the alternative is not "no refusal", it is
    the AttributeError we have already lost a boot to.

    WHEN THE None CASE AND THE RETRACTION CASE CO-OCCUR -- and they can, since
    `reset_for_retract` is ON the retraction path, which this rebuild makes
    rare but cannot eliminate (physical impossibility is real): the refusal
    fires FIRST, on the first rank, before any decision is sent. That is
    strictly the better order. The pass is voided everywhere by the ordinary
    mechanism (the first rank builds no batch, so `launched` is False and the
    empty decision admits nothing downstream), and no rank ever sees a
    geometry for a request whose geometry had been torn down.

    `require_executed_geometry=False` (the default) keeps the pre-#791-core
    arithmetic for a stand-in that carries no adder output -- the reqs in
    test_pp_admission_retry_livelock_630.py and
    test_pp_admission_prefix_indices_tensor_796.py, which exist to pin the
    guard and the #796 tensor read and never reach a real batch.
    """
    entries = []
    # #1041 FACT CARRIERS: entries that deliver an extent and NOTHING else.
    #
    # These are requests this rank SAW this pass and did not admit -- skipped by
    # one of the admission loop's eight `continue` branches, or refused by the
    # adder's budget. Boot 8 proved the cost of leaving them silent: the request
    # holding the host hit was skipped at `prefetch_pending` before the adder,
    # `can_run_list` came back empty, the method returned above this call, and
    # the extent was never published for anyone.
    #
    # `admitted=False` is what makes them safe, and it is enforced by
    # CONSTRUCTION rather than by care at each reader:
    #   * `forwarded_schedule` filters `e.admitted and not e.retracted`, so a
    #     carrier can never enter `_pp_scheduled_extents` and therefore never
    #     reaches the #791 membership comparison (scheduler.py:10508/:10522) --
    #     the PPScheduleRefused-storm direction is closed at the source, not by
    #     a second test here.
    #   * `reconcile_pp_admission_decision` passes a non-admitted entry through
    #     verbatim and never schedules it.
    #   * `forwarded_last_chunk` / `forwarded_fill_carry` key on `fill_len`,
    #     which a carrier does not carry, so it is absent from both maps.
    #   * `apply_pp_load_back_row` iterates ALL entries and stamps by rid, which
    #     is exactly what a carrier is for; a rid the receiver does not hold is
    #     already a counted no-op there, never a refusal (the rid may legitimately
    #     live only on PP0).
    # A carrier with no extent would be pure noise on the wire, so it is not
    # emitted at all.
    for req in fact_only_reqs or ():
        _extent = getattr(req, LOAD_BACK_EXTENT_ATTR, None)
        if not _extent:
            continue
        entries.append(
            PPAdmissionEntry(
                rid=req.rid,
                prefix_len=0,
                extend_len=0,
                admitted=False,
                load_back_len=int(_extent),
            )
        )
    for req in reqs:
        executed = _executed_extent(req)
        if executed is not None:
            # #944c THE OFFER THE BOUND USED TO MISS. This is the PRODUCTION
            # case -- an already-admitted, mid-chunked-prefill request being
            # re-offered out of `can_run_list` -- and until #944c it returned
            # here without the guard ever seeing it. Measured on the rig: two
            # rids, `told=8192` on every line, streak 0, cap never armed.
            #
            # COUNTED, NEVER CLAMPED, and the asymmetry is required rather
            # than cautious: this branch REPORTS the prefix the rank actually
            # executed, so rewriting it would name a pass no rank ran -- the
            # instr21 defect, and exactly what this function's docstring
            # forbids. `note_offer` returns whether the cap is exceeded; here
            # the loud refusal it emits IS the whole response, and the acting
            # half belongs upstream where the prefix is still choosable.
            if guard is not None and pp_size > 1:
                guard.note_offer(req.rid, executed[0])
            # #987: the request's own length and its trailing output tokens
            # ride the decision on BOTH branches. This one is the production
            # case (an already-admitted, mid-chunked-prefill request), i.e.
            # exactly the shape the R9 census found refused 506 times.
            executed_fill_len, executed_fill_tail = fill_carry_for(req)
            entries.append(
                PPAdmissionEntry(
                    rid=req.rid,
                    prefix_len=executed[0],
                    extend_len=executed[1],
                    admitted=True,
                    fill_len=executed_fill_len,
                    fill_tail=executed_fill_tail,
                    # #996: the last-chunk verdict is decided HERE, against
                    # this rank's own fill, and travels with the geometry it
                    # belongs to. Same reader as `fill_len` above, so the two
                    # can never describe different fills.
                    last_chunk=_last_chunk_verdict(
                        executed[0], executed[1], executed_fill_len
                    ),
                    # #968/#1035: the load-back extent PP0 OFFERS for this rid.
                    # Read straight off the request, where the load-back site
                    # recorded it this pass; never re-derived here, because a
                    # second derivation of the quantity whose disagreement IS
                    # the defect is how #995's first version manufactured false
                    # refusals.
                    # ROW-COLLAPSE: read the LIVE hit off the request at
                    # build time. The `pp_load_back_offer` field existed only
                    # to carry this number from the load-back site to here,
                    # within one pass on one rank -- a private hand-off that
                    # never needed to be request state. `host_hit_length` is
                    # the same number, set by `init_next_round_input` earlier
                    # in this same pass, and it is what the deferral line
                    # prints.
                    # #1040: STATE-ALIGNED, not raw. The extent PP0 publishes is
                    # rounded DOWN to the deepest boundary that carries a
                    # recurrent state, because the KV half of a prefix can stop
                    # anywhere and the GDN half cannot.
                    # #1041: READ, never re-derive. The choice was made at the
                    # match (`stamp_state_aligned_extent`), which is the one
                    # site every executable request must pass. Deriving it here
                    # made the fact depend on this pass admitting the request,
                    # and boot 8 measured what that costs.
                    load_back_len=getattr(req, LOAD_BACK_EXTENT_ATTR, None),
                )
            )
            continue
        if require_executed_geometry:
            raise PPScheduleRefused(
                f"#791 SCHEDULE UNBUILDABLE for rid={getattr(req, 'rid', '?')}: "
                f"this rank admitted the request but its extend_range is "
                f"{getattr(req, 'extend_range', None)!r}, so there is no "
                f"executed geometry to forward. Reporting a length derived "
                f"from anything else would name a pass no rank ran. "
                f"`reset_for_retract` is the known producer of this state "
                f"(schedule_batch.py:1588); boot instr19 met it one frame "
                f"later as an AttributeError at scheduler.py:5572."
            )
        # #796: `prefix_indices` is a TENSOR of KV-pool slot pointers, so it
        # must never reach a boolean context. `x or []` evaluates `bool(x)`,
        # and torch raises on both ends of the range that matters here: an
        # EMPTY tensor gives "Boolean value of Tensor with no values is
        # ambiguous" and a multi-element one gives the "more than one
        # element" variant. Only the single-element case would have gone
        # through silently -- so the original spelling was broken for very
        # nearly every request, and it took the boot that first got this far
        # to find out (PP0 aborted here on its first real prefill, boot
        # instr5, once the #796 send-handle fix stopped the ring wedging at
        # idle before any request was ever admitted).
        #
        # `len()` reads the shape only and does not synchronise, which is
        # what #790 requires of anything on this path; it is also the
        # spelling the rest of this feature already uses
        # (scheduler_pp_mixin.py:2652, :3508).
        prefix_indices = getattr(req, "prefix_indices", None)
        raw_prefix_len = 0 if prefix_indices is None else len(prefix_indices)
        raw_extend_len = getattr(req, "extend_input_len", None)
        if raw_extend_len is None:
            fill_ids = getattr(req, "full_untruncated_fill_ids", None)
            raw_extend_len = (
                max(0, len(fill_ids) - raw_prefix_len) if fill_ids is not None else 0
            )
        raw_prefix_len = int(raw_prefix_len)
        raw_extend_len = int(raw_extend_len)

        if guard is not None and pp_size > 1:
            # #963: name the PREFIX this offer is made over, so a downstream
            # rank's shortfall is learned against the prefix rather than
            # against this one request. The rid-scoped floor alone cannot
            # bind when the divergence is a property of the tree: measured
            # window-958 boot 2, six DISTINCT rids were offered the same
            # 1250-token prefix and each bought its own voided pass. `None`
            # (tokens unavailable) keeps the pre-#963 behaviour exactly.
            told = guard.prefix_len_for(
                req.rid,
                raw_prefix_len,
                prefix_key=offered_prefix_key(
                    getattr(req, "full_untruncated_fill_ids", None), raw_prefix_len
                ),
            )
        else:
            told = raw_prefix_len

        # told <= raw_prefix_len always (prefix_len_for only ever clamps
        # down): the shortfall moves from "prefix" to "extend", the total
        # (the only quantity the request's own token count constrains)
        # stays put.
        extend_len = raw_extend_len + (raw_prefix_len - told)

        # #987: same publication on the fallback branch, from the same reader.
        fallback_fill_len, fallback_fill_tail = fill_carry_for(req)
        entries.append(
            PPAdmissionEntry(
                rid=req.rid,
                prefix_len=told,
                extend_len=extend_len,
                admitted=True,
                fill_len=fallback_fill_len,
                fill_tail=fallback_fill_tail,
                # #996: same verdict, same rule, on the branch that derives
                # its extent instead of reading it back. `told + extend_len`
                # is invariant under the #630 guard clamp (which only moves
                # tokens from prefix to extend), so the verdict does not move
                # with it either.
                last_chunk=_last_chunk_verdict(told, extend_len, fallback_fill_len),
                # #968/#1035: same offer, same reader, on the fallback branch.
                # ROW-COLLAPSE: see the sibling constructor above.
                # #1040/#1041: the SAME stamped field, not a second copy of the
                # arithmetic -- one chooser at the writer, so the two branches
                # cannot publish extents chosen by different rules.
                load_back_len=getattr(req, LOAD_BACK_EXTENT_ATTR, None),
            )
        )
    return PPAdmissionDecision(mb_id=mb_id, entries=tuple(entries))


def reconcile_pp_admission_decision(
    decision: PPAdmissionDecision,
    local_match_lens: Dict[str, int],
    *,
    rank: int,
    pp_size: int,
    log: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, int], PPAdmissionDecision]:
    """A downstream rank's rank-local reconciliation of a received decision.

    Returns `(effective_prefix_len_by_rid, amended_decision)`:
      * `effective_prefix_len_by_rid` contains exactly the rids this rank may
        safely admit THIS pass, each mapped to the prefix length it must use
        (always `<= told`, and `<= this rank's own local match`). A retracted
        rid is simply ABSENT here -- callers must never default a missing rid
        to 0-is-safe-to-proceed-with; absence means "do not admit this pass".
      * `amended_decision` is what this rank forwards to the next stage: the
        same entries, except any newly-retracted rid has `admitted=False`,
        `retracted=True`, `retracted_by_rank=rank`. An already-retracted
        entry (set by an earlier rank) is passed through unchanged and does
        NOT get re-logged or re-evaluated -- see the "exactly one WARNING"
        pin in the paired test.

    `pp_size<=1`: pure identity pass-through (every `told` becomes
    `effective` unconditionally, no entry is ever retracted, nothing is
    logged) -- see DEFAULT PATH above.

    Raises nothing. An unhonourable entry is data, not an exception -- see
    the module docstring's "TWO FAILURE SHAPES" section for why a raise here
    would turn an ordinary, expected cache-topology fact (a downstream rank's
    cache is colder than PP0's) into a crash on every such admission.
    """
    log = log or logger
    if pp_size <= 1:
        effective = {e.rid: e.prefix_len for e in decision.entries if e.admitted}
        return effective, decision

    effective: Dict[str, int] = {}
    amended: List[PPAdmissionEntry] = []
    for entry in decision.entries:
        if not entry.admitted or entry.retracted:
            # Already excluded upstream (by PP0's own verdict, or by an
            # earlier rank's shortfall). Pass through verbatim: do not
            # re-derive, do not re-log, do not resurrect it.
            amended.append(entry)
            continue

        raw_local = local_match_lens.get(entry.rid, UNKNOWN_MATCH)
        local = int(raw_local)
        told = entry.prefix_len
        unknown = local == UNKNOWN_MATCH

        if told <= 0:
            # #944 A ZERO OFFER DEMANDS NOTHING, so no lookup result -- not
            # even a failed one -- can make it unhonourable. The rank is being
            # asked to reuse no prefix at all and to compute the request from
            # its first token; that is executable whether or not the rank has
            # ever heard of the rid.
            #
            # BEFORE THE SENTINEL THIS FELL OUT OF THE ARITHMETIC and needed no
            # branch: a miss answered 0 and `0 >= 0` held. `UNKNOWN_MATCH` is
            # -1, so it stops falling out, and leaving it implicit retracts the
            # FIRST, congruent round of every request -- strictly worse than
            # the defect being fixed here, and measured as exactly that by
            # `test_the_park_alone_leaves_a_flat_livelock`.
            #
            # It is also load-bearing for the bound: `UNRESOLVED_DEFER_CAP`'s
            # escape offers told=0 precisely because this branch honours it
            # without a measurement. Remove this and the cap stops terminating.
            #
            # A NO-OP for every rank that DID resolve (local >= 0 >= told), so
            # the known path below is unchanged, not merely equivalent.
            effective[entry.rid] = told
            amended.append(entry)
            continue

        if local >= told:
            # SAFE: truncate any extra local reuse to `told`. Same slack
            # trade #616g already makes on the TP axis -- take it.
            effective[entry.rid] = told
            amended.append(entry)
            continue

        if unknown:
            # #944: A MISS IS NOT A ZERO, AND IT IS NOT A VERDICT EITHER.
            #
            # The producer could not locate this request in any of the places it
            # knows to look. That is a statement about the LOOKUP, not about
            # this rank's cache, and the two have opposite correct responses:
            # a genuine shortfall must void the pass, whereas a miss on a
            # request this rank is very likely holding must not.
            #
            # Reported distinctly -- in the log AND in a field of its own on
            # the wire -- so the two populations can never again be read as
            # one, which is the whole reason #797c and #798 each looked like a
            # fresh defect instead of the same one twice.
            #
            # THE GROUP DEFERS, NOT THIS RANK. Excluding the rid from
            # `effective` and retracting the entry puts the pass through the
            # existing #797 void, and that void is already group-uniform:
            # `pp_pass_should_void` ORs the incoming flag and never clears it,
            # so a retraction anywhere on the ring stops every rank. This rank
            # therefore does not decide anything -- it reports. Whether the
            # group defers AGAIN or gives up is decided once, by PP0, from the
            # count `unresolved` feeds (`PPAdmissionCongruenceGuard`,
            # `UNRESOLVED_DEFER_CAP`). A defer that only one rank takes IS the
            # next divergence.
            log.warning(
                "#944 PP-ADMISSION UNRESOLVED prefix on rank %d: rid=%s "
                "told=%d local=UNKNOWN -- the reconcile could not find this "
                "request in the waiting queue, in chunked_req, in the slot's "
                "chunked req, or in the running batch. This is a LOOKUP MISS "
                "reported as such, NOT a measured zero: the rank may well hold "
                "the prefix. Excluded from this pass.",
                rank,
                entry.rid,
                told,
            )
            # #944 THE SENTINEL MUST NOT REACH THE WIRE, and a test caught
            # this: `observed_local` feeds `_learned_floor[rid]` (:293), which
            # clamps the NEXT round's `told`, and this field is documented at
            # :237 as "never zero". Writing -1 there makes the learned floor
            # -1 and changes the offer sequence -- which is exactly what
            # `test_the_park_alone_leaves_a_flat_livelock` went red on, on
            # BEHAVIOUR rather than on a number.
            #
            # The field is Optional[int], so None is the honest value: this
            # pass learned NOTHING about the rank's prefix, which is a
            # different statement from "it has -1 of it". The sentinel stays
            # where it belongs, inside the resolution map.
            #
            # Same discipline the sentinel exists for, one level up: a value's
            # meaning is a property of its READERS, and `observed_local` has
            # readers `local_match_lens` does not.
            amended.append(
                replace(
                    entry,
                    admitted=False,
                    retracted=True,
                    retracted_by_rank=rank,
                    observed_local=None,
                    unresolved=True,
                )
            )
            continue

        # UNSAFE and physically un-fixable this pass (see module docstring).
        # Fail loudly and boundedly: exactly one WARNING, never a raise, and
        # the request is excluded from `effective` rather than handed a
        # length it cannot honour. The BOOT and every OTHER request in this
        # decision are unaffected.
        log.warning(
            "#791 PP-ADMISSION unhonourable prefix on rank %d: rid=%s "
            "told=%d local=%d -- serving this request without prefix reuse "
            "on a later pass instead of corrupting the cross-stage tensor",
            rank,
            entry.rid,
            told,
            local,
        )
        amended.append(
            replace(
                entry,
                admitted=False,
                retracted=True,
                retracted_by_rank=rank,
                observed_local=local,
            )
        )
        # Deliberately absent from `effective`: see the docstring above on
        # why "missing" must mean "do not admit", not "assume 0 is safe".

    return effective, PPAdmissionDecision(mb_id=decision.mb_id, entries=tuple(amended))


def forwarded_schedule(
    decision: Optional[PPAdmissionDecision],
) -> Dict[str, Tuple[int, int]]:
    """#791 CORE: the pass geometry a downstream rank must EXECUTE.

    `rid -> (prefix_len, extend_len)` for exactly the entries that are still
    admitted and not retracted -- i.e. exactly the rids
    `reconcile_pp_admission_decision` put in `effective`, with the SECOND
    number that function drops on the floor.

    WHY THIS EXISTS AT ALL, AND IT IS NOT A NEW DATUM. `extend_len` has
    crossed the wire since #791's first commit (see `PPAdmissionEntry` above
    and the module docstring's "WHAT CROSSES THE WIRE"), and until this
    function nothing read it. `reconcile_pp_admission_decision` returns
    `Dict[rid, prefix_len]`, so a downstream rank received the chunk length
    the first rank committed and then RE-DERIVED its own from
    `PrefillAdder.add_one_req`'s rank-local `rem_chunk_tokens` /
    `rem_total_tokens` / host-load-back. That re-derivation is the boot
    instr20 crash, in full:

        PP0  verdict=ADMIT rid=6cbe2733 prefix_lens=0  chunked=1  ->  512 rows
        PP1  verdict=ADMIT rid=6cbe2733 prefix_lens=512 chunked=0  ->  333 tokens

    -- one 845-token prompt, two different splits of it, taken in the same
    second on the same forwarded decision. PP0 was told nothing new; PP1
    simply asked its own state a question the schedule had already answered.
    A HiCache prefetch had landed on PP1 (and PP2) between the retraction and
    the re-admit, `needs_host_load_back()` went true there and stayed false on
    PP0, and `add_one_req`'s load-back put the 512 prefix tokens back on a
    `prefix_indices` the admission loop had just clamped to the schedule's 0.

    So the fix is not to make the local derivation agree -- it is to stop
    deriving. A rank holding a forwarded schedule reads BOTH numbers off it.

    `None`, or a decision with nothing admitted (a #797 void), yields an empty
    mapping, which every consumer reads as "no schedule is being executed this
    pass" -- byte-identically to the behaviour that shipped before this
    function existed. Pure; no rank-local input of any kind.
    """
    if decision is None:
        return {}
    return {
        e.rid: (int(e.prefix_len), int(e.extend_len))
        for e in decision.entries
        if e.admitted and not e.retracted
    }


def order_batch_by_schedule(reqs: Sequence, schedule: Dict[str, Tuple[int, int]]):
    """#791 CORE: put this rank's batch into the FORWARDED order.

    ORDER IS GEOMETRY, and it is the one divergence every width check on this
    branch is blind to. `ScheduleBatch.prepare_for_extend` concatenates each
    request's tokens in `can_run_list` order (schedule_batch.py:2261-2262), so
    the same rid set in a different order gives the SAME row count and a
    different meaning for every row. `model_runner.py`'s `_hs.shape[0] !=
    _want` counts rows; the #757 stamp counts rows; neither can see a
    permutation. Only an identity can.

    The two orders are independently derived and there is no reason for them
    to agree: `can_run_list` follows THIS rank's `waiting_queue`, the schedule
    follows the first rank's, and the queues are fed by separate
    chain-forward arrivals.

    SORTED, NOT MERELY CHECKED. The decision is an ORDERED list --
    `build_pp_admission_decision` iterates the first rank's own
    `can_run_list`, so `decision.entries` IS that rank's batch order, and
    `forwarded_schedule` preserves it (dicts keep insertion order). A caller
    reaching this function has already proven membership identical, so the
    permutation is total and loses nothing; refusing would void a pass that
    is perfectly runnable once the rows line up.

    Returns a NEW list. An empty schedule returns the input order unchanged,
    which is the default path (no forwarded geometry) byte for byte.
    """
    if not schedule:
        return list(reqs)
    order = {rid: i for i, rid in enumerate(schedule)}
    return sorted(reqs, key=lambda req: order[req.rid])


#: #987 rid -> how many times its adopt has been reported. The adopt repeats
#: every pass until the follower produces the token itself, so the LINE is
#: rate-limited while the ACT is not: the first occurrence and every 64th
#: after it. A no-op adopt (nothing carried, or the fill already agrees) is
#: never printed at all -- it is the healthy state and printing it would bury
#: the occurrence that matters.
#:
#: BOUNDED, because it is keyed by rid and rids are unbounded over a run. A
#: log gate that grows one entry per request for the life of the process is a
#: leak wearing an instrument's clothes; past the bound the whole table is
#: dropped, which costs one extra line per surviving rid and nothing else.
_FILL_ADOPT_SEEN: Dict[str, int] = {}
_FILL_ADOPT_LOG_EVERY = 64
_FILL_ADOPT_SEEN_CAP = 4096


def adopt_carried_fill(
    req, carried: Optional[Tuple[int, Tuple[int, ...]]]
) -> Optional[int]:
    """#987: materialise the upstream's fill on this rank. Tokens appended, or None.

    THE DEFECT, measured (R9 census, boots 6-7). Rank 0 holds one sampled
    output token that never crossed the `tp_to_pp` seam -- #631 OUTTRACE:
    `PP0 n=1 off=0 tail=[25]`, followers `n=0`. So rank 0's
    `len(full_untruncated_fill_ids)` is 8447 and every follower's is 8446, and
    `schedule_refusal_reason`'s third clause reads `7939 + 508 > 8446` and
    vetoes -- correctly, on its own terms: the two ranks really do disagree
    about the request. 506 of 513 void-causing refusals are that one token.

    THE FIX IS TO END THE DISAGREEMENT, NOT TO WEAKEN THE CLAUSE. The clause
    is right and stays exactly as it is; what changes is that the upstream now
    SAYS what it holds (`PPAdmissionEntry.fill_len` / `fill_tail`) and this
    rank materialises it before the clause is asked. Loosening the inequality
    by one instead would have left the follower executing an extend range over
    a token it does not have, which is the same corruption in the other
    direction.

    OUTPUT TAIL ONLY, NEVER THE PREFIX (Kein-Doppel-Prefill). Nothing here
    touches `prefix_indices`, `last_node`, `origin_input_ids` or any pool
    handle: not one cached token is discarded and not one prompt token is
    rewritten by a peer. What is adopted is the trailing GENERATED tokens the
    upstream already holds -- see `fill_carry_for` for why only those are
    eligible to be published in the first place.

    AND NOT `output_ids` EITHER, which is the shape this function's first
    draft would have taken. A consumer sweep of this tree (2026-08-28) shows
    `req.output_ids` is read by the client stream and by the finish check with
    NO PP-rank guard on either: `output_streamer.py:383-384` slices
    `output_ids[send_token_offset:]` into the payload that
    `output_streamer.py:166` sends, and the socket that carries it belongs to
    `pp_rank == 0` (`ipc_channels.py:36-72`), which under this feature is a
    RECEIVING rank, not the sampler; and `schedule_batch.py:1584` compares
    `len(output_ids) >= max_new_tokens` on every rank independently, so one
    extra local token lets one rank declare a request finished a step before
    its peers -- the divergence class `tp_worker.py:679-694` records as a
    permanent cross-rank hang. A token this rank did not sample must therefore
    never enter `output_ids`. It goes into a SHADOW pair
    (`Req.pp_carried_fill_len` / `Req.pp_carried_fill_tail`) that exactly one
    reader honours, `Req._refresh_fill_ids`, which is the definition of the
    fill and the only quantity in dispute. Same discipline as #944's
    `unresolved`: a distinct fact gets a distinct field, never a second
    meaning bolted onto an existing one.

    SELF-CANCELLING, hence idempotent across passes. The deficit is recomputed
    every call against `len(origin_input_ids) + len(output_ids)`, so once this
    rank generates the token itself the deficit is 0, the shadow is cleared,
    and the fill is back to being made of nothing but this rank's own state.
    Re-adopting an unchanged deficit rewrites the same shadow and is a no-op.

    TWO LOUD REFUSALS, neither of which raises. Both leave the fill untouched,
    which hands the decision straight back to `schedule_refusal_reason` -- the
    refusal is delivered by the clause that already owns it, and this line
    only names the two numbers the clause cannot see:
      * the divergence is WIDER than what is carried (`FILL_CARRY_TAIL_CAP`,
        or a tail shorter than the gap). Not the seam shape; adopting an
        arbitrary suffix would fabricate a request.
      * this rank is AHEAD of the upstream. Never truncate: a fill is not
        shortened to match a peer, and the honest response is to say so and
        let the geometry check answer.

    Returns the number of tokens now carried (0 when the fill already agreed
    and the shadow was cleared), or None when there was nothing to adopt.
    """
    if carried is None:
        return None
    carried_fill_len, carried_tail = carried
    if carried_fill_len is None:
        return None
    rid = getattr(req, "rid", "?")
    origin = getattr(req, "origin_input_ids", None)
    output = getattr(req, "output_ids", None)
    if origin is None or output is None:
        return None
    natural_len = len(origin) + len(output)
    deficit = int(carried_fill_len) - int(natural_len)

    if deficit == 0:
        # Agreed. Drop any shadow a previous pass left standing -- this rank
        # has caught up on its own and the carried tokens are now its own.
        if getattr(req, "pp_carried_fill_len", None) is not None:
            _clear_carried_fill(req)
        return 0

    if deficit < 0:
        logger.error(
            "#987 FILL-REFUSE rid=%s: this rank holds %d fill token(s), the "
            "upstream names only %d -- this rank is AHEAD by %d. A fill is "
            "never truncated to match a peer, so nothing is adopted and the "
            "forwarded geometry is answered by the ordinary check. This is "
            "not the #631 seam shape (there the UPSTREAM holds the extra "
            "token); a receiver running ahead of its decider means the "
            "decision was built from state older than this rank's own.",
            rid,
            natural_len,
            int(carried_fill_len),
            -deficit,
        )
        return None

    if deficit > FILL_CARRY_TAIL_CAP or deficit > len(carried_tail):
        logger.error(
            "#987 FILL-REFUSE rid=%s: the upstream names %d fill token(s), "
            "this rank holds %d -- a gap of %d, against a carried tail of %d "
            "token(s) and a cap of %d. NOT ADOPTED: past the cap the two "
            "ranks are not one seam token out of step, they hold different "
            "requests, and shipping an arbitrary suffix of one across the "
            "wire would replace a correct refusal with a fabrication. The "
            "geometry check refuses this pass on its own terms; the #631 "
            "seam is the thing to fix, not this bound.",
            rid,
            int(carried_fill_len),
            natural_len,
            deficit,
            len(carried_tail),
            FILL_CARRY_TAIL_CAP,
        )
        return None

    tail = tuple(int(t) for t in carried_tail[len(carried_tail) - deficit :])
    req.pp_carried_fill_len = int(carried_fill_len)
    req.pp_carried_fill_tail = tail
    # The shadow is only a promise until the fill is rebuilt from it; this is
    # the line that makes `len(full_untruncated_fill_ids)` -- the quantity the
    # third clause reads -- actually equal the upstream's.
    req._refresh_fill_ids()

    if len(_FILL_ADOPT_SEEN) > _FILL_ADOPT_SEEN_CAP:
        _FILL_ADOPT_SEEN.clear()
    seen = _FILL_ADOPT_SEEN.get(rid, 0)
    _FILL_ADOPT_SEEN[rid] = seen + 1
    if seen == 0 or seen % _FILL_ADOPT_LOG_EVERY == 0:
        logger.warning(
            "#987 FILL-ADOPT rid=%s local=%d -> upstream=%d appended=%d "
            "tail=%s (seam #631, seen=%d). The upstream holds output token(s) "
            "this rank never received across tp_to_pp; they are carried on "
            "the admission decision and materialised HERE, in the fill only "
            "-- output_ids, prefix_indices and the cached prefix are "
            "untouched, so no token is recomputed and none is emitted twice.",
            rid,
            natural_len,
            int(carried_fill_len),
            deficit,
            list(tail),
            seen + 1,
        )
    return deficit


def _clear_carried_fill(req) -> None:
    """#987: drop the shadow and rebuild the fill from this rank's own state."""
    req.pp_carried_fill_len = None
    req.pp_carried_fill_tail = ()
    req._refresh_fill_ids()


def schedule_refusal_reason(
    *,
    rid: str,
    scheduled_prefix_len: int,
    scheduled_extend_len: int,
    local_prefix_len: int,
    local_fill_len: int,
) -> Optional[str]:
    """`None` when this rank can execute the forwarded geometry verbatim;
    otherwise the LOUD REFUSAL text, quoting the forwarded decision.

    THE THREE WAYS A FORWARDED GEOMETRY CAN BE LOCALLY IMPOSSIBLE, and none
    of them may be answered by adjusting the geometry:

      * the rank's own prefix is not the scheduled one. The admission loop
        clamps `prefix_indices` to the schedule before the adder ever sees the
        request, and `reconcile_pp_admission_decision` retracts anything it
        cannot reach, so this is unreachable on a healthy pass -- which is
        exactly why it is checked. A silent re-clamp here would be the same
        local narrowing this module exists to abolish.
      * the request does not have the tokens. `prefix + extend` past
        `full_untruncated_fill_ids` cannot be filled from anything this rank
        holds.
      * a NEGATIVE extend. Not merely unrunnable -- unrepresentable, so it can
        only mean a malformed decision.

    A ZERO EXTEND IS EXECUTABLE, and the change of mind is #791-core's own
    doing. Before the producer reported the EXECUTED geometry, a zero could
    only have been a fabrication; now it is a faithful report of a first rank
    that ran zero rows for that request (a chunk landing exactly on its last
    token, schedule_policy.py:1420-1421). Refusing it would void a pass the
    upstream ran perfectly well -- and substituting a length of this rank's
    own is precisely what this module abolishes. See `_executed_extent`.

    The caller turns this into a voided pass (the #797 path), never into a
    narrower batch. Pure: five integers in, a string or None out.
    """
    if scheduled_extend_len < 0:
        return (
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={rid}: the decision "
            f"names extend_len={scheduled_extend_len}, which is not a length. "
            f"A downstream rank may not substitute one of its own."
        )
    if local_prefix_len != scheduled_prefix_len:
        return (
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={rid}: the decision "
            f"names prefix_len={scheduled_prefix_len}, this rank holds "
            f"{local_prefix_len}. The batch geometry is the upstream's to "
            f"decide; narrowing or widening it here is what pairs one "
            f"microbatch's hidden states with another's metadata."
        )
    if scheduled_prefix_len + scheduled_extend_len > local_fill_len:
        return (
            f"#791 FORWARDED SCHEDULE UNEXECUTABLE for rid={rid}: the decision "
            f"names prefix_len={scheduled_prefix_len} + "
            f"extend_len={scheduled_extend_len}, past this rank's "
            f"{local_fill_len} fill token(s). The two ranks disagree about the "
            f"request itself, not merely about its cache."
        )
    return None


def entries_retracted_by_rank(
    decision: Optional[PPAdmissionDecision], rank: int
) -> Tuple[PPAdmissionEntry, ...]:
    """The entries THIS rank newly retracted from a decision it received.

    #791c. `reconcile_pp_admission_decision` above is the only writer of
    `retracted_by_rank`, and it writes it ONLY on the hop that first finds
    the prefix unhonourable -- an entry another rank already retracted takes
    the pass-through branch and keeps that rank's number. So a non-empty
    result here means exactly one thing, with no inference: THIS RANK'S BATCH
    FOR THIS PASS IS NARROWER THAN ITS UPSTREAM'S, because the upstream built
    and forwarded its own batch from the decision as it stood BEFORE this
    retraction, and nothing can amend a batch that is already in flight.

    That is the whole content of the 2026-08-21 07:12:49 mispair (boot
    instr17): PP0 admitted two requests (126 extend tokens), PP1 found
    rid=5e744c29's told=16896 prefix unhonourable against its own local=0,
    excluded it from `effective`, and built a one-request batch of 22 tokens
    -- then paired PP0's 126 rows of hidden states with it. Every identity
    the proxy carries (slot, sequence, flip epoch) was CORRECT: the message
    was the current pass's, from the current ring, in the current epoch. Only
    the WIDTH disagreed, and the width disagreed because of this function's
    subject.

    PURE, AND KEPT IN THIS MODULE ON PURPOSE. The consumer is the PP proxy
    receive guard in scheduler_pp_mixin.py, but the fact is a property of a
    decision, and this module already owns every other such property
    (`congruent_rids` below is the same shape). `None` -- a caller with no
    decision recorded for the slot -- yields an empty tuple, which every
    consumer reads as "nothing known against this pass", i.e. exactly the
    behaviour that shipped before this function existed.
    """
    if decision is None:
        return ()
    return tuple(
        e
        for e in decision.entries
        if e.retracted
        and e.retracted_by_rank is not None
        and int(e.retracted_by_rank) == int(rank)
    )


def void_pp_admission_decision(
    decision: PPAdmissionDecision,
) -> PPAdmissionDecision:
    """#797: drop every STILL-ADMITTED entry, leaving the retractions alone.

    THE PASS, NOT THE REQUEST, IS WHAT A RETRACTION COSTS. `reconcile_pp_
    admission_decision` above narrows a decision by ONE rid and leaves the
    rest admitted, which is correct as a statement about that rid and wrong
    as a plan for the pass: the upstream has already built and launched a
    batch containing BOTH rids, so a batch of "the rest" is a strict subset
    of the one whose hidden states are already on the wire. Boot instr17
    computed exactly that subset -- 22 tokens against 126 rows -- and boots
    instr15/16/17 between them logged 661, 1651 and 1718 of these narrowings
    while the width check caught ONE. The other ~4000 were same-width
    pairings, which are silent wrong output. See `entries_retracted_by_rank`.

    So the retracting rank drops the whole pass instead, and this is the
    membership half of that: every entry that is still admitted becomes
    `admitted=False`, which `reconcile_pp_admission_decision`'s own
    pass-through branch then carries to every rank after this one, and which
    scheduler.py's admission loop reads as "not named this pass" -- the
    existing requeue-for-free path, not a new one.

    admitted=False AND retracted=False, WHICH IS THE THIRD STATE AND NOT AN
    OVERSIGHT. `PPAdmissionCongruenceGuard.record_return_trip` reads exactly
    these two flags: `retracted` teaches a floor, `admitted` clears one, and
    neither ("excluded before this module ever saw it") leaves the rid
    untouched. A collaterally-dropped rid must land in that third state: it
    suffered no shortfall, so it must teach no floor, and it was never
    served, so it must clear none either. Marking it `retracted` would clamp
    a prefix that was perfectly honourable; leaving it `admitted` would clear
    a floor another rank had just paid to learn.

    Pure and idempotent: a decision with nothing left to drop is returned
    with equal content, and a `retracted` entry is never touched.
    """
    if not any(e.admitted and not e.retracted for e in decision.entries):
        return decision
    return PPAdmissionDecision(
        mb_id=decision.mb_id,
        entries=tuple(
            replace(e, admitted=False) if e.admitted and not e.retracted else e
            for e in decision.entries
        ),
    )


def congruent_rids(decisions: Iterable[PPAdmissionDecision]) -> bool:
    """True iff every decision in `decisions` agrees on membership AND on
    every admitted rid's `(prefix_len, extend_len)`. Test/diagnostic helper:
    the property this whole module exists to make true across PP ranks."""
    decisions = list(decisions)
    if len(decisions) <= 1:
        return True
    reference = {
        e.rid: (e.prefix_len, e.extend_len)
        for e in decisions[0].entries
        if e.admitted and not e.retracted
    }
    for other in decisions[1:]:
        seen = {
            e.rid: (e.prefix_len, e.extend_len)
            for e in other.entries
            if e.admitted and not e.retracted
        }
        if seen != reference:
            return False
    return True


# How many consecutive VACUOUS admission verdicts the #788 trace swallows
# before it emits a roll-up line. Chosen against the measured idle rate of
# boot instr10 (146023 trace lines per rank), which puts a roll-up roughly
# every minute or two of idling: frequent enough that a reader never has to
# wonder whether the instrument is still alive, rare enough that three hours
# of idling costs kilobytes instead of the 5.9 GB boot instr11 wrote.
#
# It is a COUNT, never a duration. A wall-clock cadence would make the
# emission points rank-local and destroy the congruence property below.
PP_ADMISSION_VACUOUS_ROLLUP_EVERY = 1024


def pp_admission_verdict_is_vacuous(
    n_reqs: int, queue: int, running: int, chunked: int
) -> bool:
    """True iff this admission verdict carries no congruence evidence.

    A pass with nothing admitted, nothing queued, nothing running and nothing
    chunked was taken over an empty scheduler. Two ranks cannot disagree
    about an empty scheduler, so the line cannot show the divergence the #788
    trace exists to catch -- and at idle it is essentially every line: boot
    instr11 wrote 5.9 GB in three hours, of which the informative fraction
    was a rounding error.

    THE PROPERTY THIS FUNCTION MUST KEEP, and the reason it takes four plain
    ints and nothing else. The acceptance gate for rank congruence
    (evidence-665-f1/verdict_790.sh, step 3) diffs the emitted trace payloads
    across PP0/PP1/PP2 and requires ONE group per event. That diff is only
    meaningful if every rank decides to speak or stay silent from data every
    rank has identically. So the predicate is a pure function of the
    CONGRUENT payload fields -- `n_reqs`, `queue`, `running`, `chunked` --
    and must never consult anything rank-local: not wall-clock time, not a
    per-rank log volume, not a random sample, and not `avail`/`evictable`
    (which are this rank's own pool accounting and legitimately differ). If
    rank 0 suppressed a line rank 1 emitted, the gate would report a
    divergence that never happened, and an instrument that manufactures
    false positives is worse than no instrument.
    """
    return n_reqs == 0 and queue == 0 and running == 0 and chunked == 0
