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
"""#888b: a resident the phase FORBIDS to run must be able to yield its seat.

THE MEASUREMENT THIS EXISTS FOR, and it is not the one the ticket predicted.
W38 rerun, boot_w38rerun_0826_1304.log, 13:11:00-13:13:14 -- 156 seconds in
which the PP phase held with prefill pending and admitted nothing, ended by
the 173.6s decode-starvation cap rather than by a drain::

    PHASE-POLICY holding in pp: prefilling in pp (33 tok pending)
                                (pending prefill 33 tok, running bs 8)
    #788 PP-ADMISSION verdict=DECLINE n_reqs=0 avail=12876 evictable=0
         queue=2 running=4 chunked=0
         reason=gate=batch_full_or_empty_queue(batch_is_full=1,queue=2)

TWO FACTS IN THAT LINE FALSIFY THE OBVIOUS READING.

* ``avail=12876``. Twelve thousand free KV tokens against 33 pending. The
  KV pool was NOT the binder, so no amount of KV-token relief could have
  admitted this prefill. The pool-utilisation figure that framed the ticket
  (0.97) is true and irrelevant: 3% of a large pool is still 400x the ask.
* ``reason=gate=batch_full_or_empty_queue``. Sixty of sixty emitted declines
  inside the stall name that gate, and ZERO name ``no_allocatable_reqs``. The
  refusal never reached a memory test at all.

THE BINDER IS THE REQUEST SEAT. ``max_num_reqs = self.max_running_requests``
(model_runner_kv_cache_mixin.py:3393), so the ``req_to_token_pool`` has
exactly as many seats as the concurrency cap -- eight on this recipe. Eight
carriers were resident, so ``req_to_token_pool.available_size()`` was 0, and
``get_num_allocatable_reqs`` min()s against it AFTER #677's carrier discount
has already done its work. Discounting a parked carrier against
``max_running`` cannot free a seat the same carrier is still sitting in.

AND THE SECOND DEFECT IS WHAT HID THE FIRST. ``batch_is_full`` is a latch
(scheduler.py, "a PERSISTENT flag carried on running_batch"): it is set by
the admission gate itself and its clear sites all live on the decode path --
``update_running_batch`` and the finish paths. Under strict purity the PP
phase MAY NOT DECODE, so in that phase the flag has no reachable clear site
at all. One pass with a full seat table latches it; every later pass returns
at the flag, above the seat test, above the carrier discount, above every
relief. The instance then reports "batch is full" for 156 seconds without
ever re-deriving whether it still is. The class is already named --
``cutover_participants.py`` registers ``latched_batch_flags`` -- but its hook
runs at the SEAM, and this stall happens in the middle of a residency.

SO THE PHASE WAITED FOR AN EVENT ITS OWN OCCUPANCY FORBIDS: it holds while
prefill is pending; the pending prefill needs a seat; every seat is held by a
carrier it may not run to completion; and nothing takes a seat back. That is
the same shape #677 measured on 2026-08-16 and answered with arithmetic. The
arithmetic was correct and insufficient: it removed the cap that was not
binding and left the pool that was.

WHAT THIS MODULE IS
-------------------
The DECISION, and only the decision. Whether one resident carrier must yield
its seat this pass, and WHICH RESOURCE is actually binding when it does. It
allocates nothing, frees nothing, imports no torch and holds no scheduler
reference; the actuator it decides for already exists and is already proven
(``Scheduler._retract_decode_and_requeue``, itself extracted for #679 so this
kind of caller would not own a second copy of a retraction).

Splitting it this way is not tidiness. The stall above was mis-attributed
from a pool-utilisation reading, and the correction is a decision that NAMES
THE BINDER it measured rather than one that assumes it. A verdict carrying
``binder="req_slot"`` is falsifiable against a boot log; "memory pressure" is
not.

THE DANGER DIRECTION, STATED FIRST
----------------------------------
Yielding a seat destroys a request's decode progress: it is retracted and
re-queued, and it re-prefills from its prompt. Wrongly yielding is therefore
expensive and, in one direction, silently so -- a carrier retracted while its
phase could have run it loses work for nothing and the log shows only a
successful admission afterwards. Every HOLD rule below exists for that
direction, and :func:`carrier_relief_verdict` refuses on ANY of them rather
than weighing them.

Above all: **the phase must forbid decode.** A resident the phase permits to
run is not stuck, it is merely waiting its turn, and taking its seat is a
regression dressed as a fix. The decode-OOM branch owns retraction under
memory pressure; this owns it only under a phase prohibition.

ONE VICTIM PER PASS, AND THE BOUND IS THE UNIFORMITY ARGUMENT
-------------------------------------------------------------
The verdict is for exactly one carrier. Not a tuning choice: #583 is the case
where the entry decision was group-uniform and the LOOP BOUND was not, so
ranks entered retraction together and popped different numbers of victims.
A constant bound of one is uniform by construction. Freeing one seat is also
exactly what admission needs -- ``get_num_allocatable_reqs`` is re-read after
the yield, and a pass that still cannot admit simply declines and re-decides
next round, which is the same self-clearing shape as #679's park.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "BINDER_KV_TOKEN",
    "BINDER_MAMBA_SLOT",
    "BINDER_NONE",
    "BINDER_REQ_SLOT",
    "CarrierReliefVerdict",
    "ENV_PARKED_CARRIER_RELIEF",
    "carrier_relief_verdict",
    "latched_flag_must_be_rederived",
    "name_the_binder",
    "parked_carrier_relief_enabled",
]


#: The seat in ``req_to_token_pool``. Sized to ``max_running_requests``, held
#: by a resident whether or not it is stepping. THE MEASURED BINDER.
BINDER_REQ_SLOT = "req_slot"
#: Device KV rows. The binder the ticket predicted and the log refuted.
BINDER_KV_TOKEN = "kv_token"
#: A mamba/GDN state slot. Named for completeness; measured 35% free.
BINDER_MAMBA_SLOT = "mamba_slot"
#: Nothing is short. A verdict carrying this must never yield.
BINDER_NONE = "none"


#: KILL SWITCH, not an opt-in, and the direction is deliberate.
#:
#: #679's ladder is off by default and was off on the boot that produced this
#: stall, so it could not have helped and did not. A relief that is armed only
#: when an operator remembers it is the "present but inert" failure this tree
#: keeps finding -- it would ship a fix for a wedge and leave the wedge.
#:
#: The arming condition is STRUCTURAL instead: the verdict yields only when the
#: phase forbids decode, which is exactly the state this exists for and is
#: false on every boot without strict phase purity. Such a boot is byte-
#: identical whether this flag is set or not. The flag is here so an operator
#: who meets a defect in it can take it out of the path without a rebuild.
ENV_PARKED_CARRIER_RELIEF = "SGLANG_PARKED_CARRIER_RELIEF"


def parked_carrier_relief_enabled() -> bool:
    return os.environ.get(ENV_PARKED_CARRIER_RELIEF, "1") not in (
        "0",
        "",
        "false",
        "False",
    )


@dataclass(frozen=True)
class CarrierReliefVerdict:
    """One pass's answer, with the quantity it was decided from.

    ``binder`` is meaningful on a yield and on a hold alike: a hold that
    reports ``BINDER_NONE`` says admission is blocked by something this
    module does not measure, which is a finding rather than a silence.
    """

    #: Retract exactly one carrier from the running batch this pass.
    yield_carrier: bool
    #: Which resource admission is actually short of. Never assumed.
    binder: str
    #: Human-readable, ends up in the receipt verbatim.
    reason: str

    def describe(self) -> str:
        verb = "YIELD one carrier" if self.yield_carrier else "HOLD"
        return f"{verb} (binder={self.binder}): {self.reason}"


def name_the_binder(
    *,
    req_slots_free: int,
    kv_avail_tokens: int,
    kv_need_tokens: int,
    mamba_slots_free: int,
) -> str:
    """Which resource is short, measured rather than assumed.

    ORDER IS BY WHAT ADMISSION TESTS FIRST, not by cost. ``req_to_token_pool
    .available_size()`` is a term of ``get_num_allocatable_reqs``; the KV and
    mamba budgets are tested later, inside ``PrefillAdder.budget_state``. A
    pass that never gets a seat never reaches the budget, so reporting a KV
    shortfall while the seat table is empty would reproduce exactly the
    mis-attribution that framed this ticket.

    ``kv_need_tokens <= 0`` means the caller did not measure a KV ask; it is
    NOT the same as "KV is fine" and must not be reported as a KV shortfall.
    """
    if int(req_slots_free) <= 0:
        return BINDER_REQ_SLOT
    if int(kv_need_tokens) > 0 and int(kv_avail_tokens) < int(kv_need_tokens):
        return BINDER_KV_TOKEN
    if int(mamba_slots_free) <= 0:
        return BINDER_MAMBA_SLOT
    return BINDER_NONE


def latched_flag_must_be_rederived(
    *, decode_forbidden: bool, flag_is_latched: bool
) -> bool:
    """May this pass trust a latched ``batch_is_full``?

    NO, and only, when the phase forbids decode -- because that is precisely
    the phase in which every clear site of the flag is unreachable
    (``update_running_batch`` and the finish paths all sit on the decode
    path). A latch whose clear site the phase forbids is not a latch, it is a
    reading that can only get staler.

    In a phase that permits decode the flag keeps its stock meaning and its
    stock owners, and this returns False. That is what keeps a non-purity boot
    byte-identical: nothing here fires unless a phase prohibition is active.
    """
    return bool(decode_forbidden) and bool(flag_is_latched)


def carrier_relief_verdict(
    *,
    decode_forbidden: bool,
    pending_prefill_tokens: int,
    queue_len: int,
    allocatable_reqs: int,
    resident_bs: int,
    parked_count: int,
    chunk_in_flight: bool,
    req_slots_free: int,
    kv_avail_tokens: int,
    kv_need_tokens: int = 0,
    mamba_slots_free: int = 0,
) -> CarrierReliefVerdict:
    """Must one carrier yield its seat this pass? Pure; nothing is mutated.

    Every rule below is a HOLD, and the yield is what is left when none of
    them fires. Written that way on purpose: the expensive error is a yield
    that should not have happened, so the default has to be the safe one and
    each exception has to be argued rather than the reverse.
    """
    # -- the prohibition itself. Without it this is just retraction. --------
    if not decode_forbidden:
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            "the phase permits decode, so a resident is waiting its turn "
            "rather than stuck; the decode-OOM branch owns retraction here",
        )

    # -- is there anything to relieve FOR? ----------------------------------
    # A phase with no pending prefill is not starving, it is idle, and an
    # idle phase that retracts a carrier destroys work to admit nothing.
    if int(queue_len) <= 0 or int(pending_prefill_tokens) <= 0:
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            f"no prefill is pending (queue={int(queue_len)}, "
            f"pending={int(pending_prefill_tokens)} tok); nothing to admit",
        )

    # -- is admission actually blocked? -------------------------------------
    # Read from the same expression the gate uses, not from a proxy. If a
    # seat is already available the pass admits unaided and a yield here
    # would be pure loss.
    if int(allocatable_reqs) > 0:
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            f"admission is not blocked ({int(allocatable_reqs)} allocatable); "
            "the pass proceeds unaided",
        )

    # -- #679 composition: one admission authority per pass -----------------
    # A chunk in flight is the chunked door's pass, and the relief ladder
    # already runs there. Two reliefs on one pass would retract for a
    # shortfall the other has just paid, and the chunked request itself is
    # the one resident whose partial prefill a retraction throws away.
    if chunk_in_flight:
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            "a chunked prefill is in flight; that pass belongs to the #679 "
            "ladder and its request must not be retracted underneath it",
        )

    # -- only a carrier the phase forbids to run may be taken ---------------
    # The parked set is the named record of exactly that. An empty set with
    # the prohibition active means the record has not been reconciled this
    # residency, and acting on an unreconciled record is how a running
    # request gets retracted.
    if int(parked_count) <= 0:
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            "no carrier is recorded as parked, so no resident is provably "
            "unable to run; refusing to guess which one is idle",
        )

    # -- never the last one -------------------------------------------------
    # ``retract_decode`` keeps at least one request by construction, so a
    # request for a victim here would free nothing and the receipt would
    # claim a relief that did not happen.
    if int(resident_bs) <= 1:
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            f"only {int(resident_bs)} resident; retraction keeps the last "
            "request, so this pass would free nothing",
        )

    binder = name_the_binder(
        req_slots_free=req_slots_free,
        kv_avail_tokens=kv_avail_tokens,
        kv_need_tokens=kv_need_tokens,
        mamba_slots_free=mamba_slots_free,
    )
    if binder == BINDER_NONE:
        # Admission is blocked and nothing this module measures is short.
        # SAY SO rather than yielding on the strength of the block alone:
        # retracting against an unmeasured binder is how a relief becomes a
        # ritual that costs a request per pass and frees the wrong thing.
        return CarrierReliefVerdict(
            False,
            BINDER_NONE,
            "admission is blocked but no measured resource is short "
            f"(seats {int(req_slots_free)}, kv {int(kv_avail_tokens)}, "
            f"mamba {int(mamba_slots_free)}); the binder is elsewhere",
        )

    return CarrierReliefVerdict(
        True,
        binder,
        f"{int(parked_count)} of {int(resident_bs)} resident carrier(s) "
        f"cannot run in this phase and {int(pending_prefill_tokens)} tok of "
        f"prefill is pending behind them; binder is {binder} "
        f"(seats free {int(req_slots_free)}, kv avail {int(kv_avail_tokens)}, "
        f"mamba free {int(mamba_slots_free)}). Yielding ONE seat",
    )


def relief_receipt(
    verdict: CarrierReliefVerdict,
    *,
    seats_before: int,
    seats_after: int,
    tokens_gained: int,
    victims: int,
) -> str:
    """What the boot log must be able to be grepped for afterwards.

    NAMES THE SEAT DELTA, not only the token delta. ``_retract_decode_and_
    requeue`` returns tokens, and tokens are the quantity that was never
    short here; a receipt reporting only those would be the same
    mis-attribution one layer down.
    """
    return (
        f"PARKED-CARRIER-RELIEF {verdict.describe()} -- retracted {victims} "
        f"request(s), request seats {seats_before} -> {seats_after}, "
        f"KV tokens gained {tokens_gained}. The victim is re-queued and "
        f"re-prefills; admission is re-read and the gate still decides."
    )


def hold_receipt(verdict: CarrierReliefVerdict) -> Optional[str]:
    """A hold worth logging, or None for the ones that are just quiet.

    Only the holds that mean something happened. "The phase permits decode"
    is every pass of every ordinary boot and must never reach a log.
    """
    if verdict.yield_carrier:
        return None
    if verdict.binder == BINDER_NONE and "permits decode" in verdict.reason:
        return None
    if "no prefill is pending" in verdict.reason:
        return None
    if "admission is not blocked" in verdict.reason:
        return None
    return f"PARKED-CARRIER-RELIEF {verdict.describe()}"
