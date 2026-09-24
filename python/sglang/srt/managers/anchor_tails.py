"""fnFL2 H42 (Task #118): END-ANCHOR tails of SEVERAL requests per P forward.

THE ROOT. Group P arms the END-OF-PREFILL ANCHOR (#1233,
``SGLANG_WEG2_END_ANCHOR``): a prompt of N tokens that fits the chunk whole is
still split into a body ``[prefix, cut)`` and a tail ``[cut, N)`` with
``cut = floor_grain(N - 1)``, so the recurrent state at N-1 is published. The
held tail was scheduled by the ONE chunked-request field
(``Scheduler.chunked_req``), which admits one continuation per pass
(#959/#995/#996). The second whole-fit prompt of a pass was therefore refused
(``[#967] SECOND CONTINUATION REFUSED ... add_one_req/end-anchor``) and the
admission loop broke: on P at most ONE request reached its end per forward,
and every 1-4 token tail ran as a forward of its own (x144: ``#new-token: 1``
at 30-200 ms per stage).

THE DESIGN. An anchor body is not a general continuation: its remainder is
the tail, at most one grain long, and the tail is FINAL by construction. So
it does not need the single field. Armed
(``SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS`` and the END-ANCHOR), every
anchor body becomes an ANCHOR TAIL of its own request:

* the adder mints it into ``PrefillAdder.new_anchor_tails`` (no #996 assert,
  no #959/#967 refusal -- those guard the single field, which a tail never
  occupies);
* the scheduler keeps them in ``Scheduler.anchor_tails``: excluded from the
  running-batch merge and stashed exactly like ``chunked_req`` (the stash is
  where each request's own anchor node, MAMBA-ARENA claim and tail-handoff
  capture happen -- per request, keyed by rid);
* the next pass re-adds ALL tails through ``add_chunked_req`` -- the same
  method, budget charge, forwarded-schedule execution (#791/#996) and
  ``carried_chunk`` lock discipline the single continuation uses -- before the
  admission loop, which then fills the rest of the chunk budget with new
  bodies;
* PP followers mint the same tails from the same forwarded geometry
  (``_add_scheduled_req``: carried last-chunk verdict False AND the extent
  ends at the anchor cut of the carried fill) and re-add them per rid.

Unarmed, every function here is either not called or returns the stock
value, and the stock path is byte-identical.

Narrow arguments only (large-class-style §1.6): nothing here reads or writes
the scheduler; the scheduler threads results back onto its own fields.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: Log cadence of the per-pass census line (first N, then every M-th).
_LOG_FIRST = 16
_LOG_EVERY = 256
_pass_n = 0


def multi_anchor_tails_armed(end_anchor_armed: bool) -> bool:
    """Armed only where the END-ANCHOR splits (group P). Read per call so a
    test's ``override`` and a boot's environment are the same reading."""
    return bool(end_anchor_armed) and bool(
        envs.SGLANG_WEG2_ENABLE_P_MULTI_ANCHOR_TAILS.get()
    )


def anchor_cut(fill_len: int, grain: int) -> int:
    """Where the END-ANCHOR puts a prompt's last chunk boundary (#1233;
    fnFL2x14: on a QSA pool the grain is the page)."""
    grain = max(1, int(grain))
    return (int(fill_len) - 1) // grain * grain


def is_anchor_body(*, fill_len: int, start: int, end: int, grain: int) -> bool:
    """Is the extent ``[start, end)`` an END-ANCHOR BODY of a ``fill_len``
    prompt, i.e. does it stop exactly at the anchor cut so that only the final
    tail ``[cut, fill_len)`` remains? The ONE rule every mint site answers
    (PP0's two fresh-request branches, a follower's forwarded extent), so all
    ranks classify the same rid the same way."""
    fill_len, start, end = int(fill_len), int(start), int(end)
    if fill_len < 2 or end >= fill_len:
        return False
    cut = anchor_cut(fill_len, grain)
    return end == cut and cut > start


class TailReadd(NamedTuple):
    #: tails that remain tails after this pass (not named, or a
    #: budget-truncated re-add)
    kept: List[Any]
    #: tails that are in this pass's can_run_list
    readded: List[Any]
    #: tails a forwarded schedule did not name (kept, resumed later)
    not_named: List[Any]


def readd_anchor_tails(
    tails: Sequence[Any],
    adder: Any,
    *,
    incoming: Optional[Mapping[str, int]],
) -> TailReadd:
    """Re-add every carried tail, in order, before the admission loop.

    Per tail the same steps the scheduler takes for ``chunked_req``: refresh
    the round input; on a rank executing a forwarded decision skip a rid the
    decision does not name (#992: kept, resumed on the first pass that names
    it) and adopt a told prefix that differs (#994); then
    ``adder.add_chunked_req`` -- which executes a scheduled extent under the
    carried last-chunk verdict, or derives the final extent locally. A tail is
    final unless the chunk budget truncated it; then it stays a tail.
    """
    kept: List[Any] = []
    readded: List[Any] = []
    not_named: List[Any] = []
    for tail in tails:
        tail.init_next_round_input()
        if incoming is not None:
            told = incoming.get(tail.rid)
            if told is None:
                not_named.append(tail)
                kept.append(tail)
                continue
            if told != len(tail.prefix_indices):
                tail.truncate_prefix_to(told)
        rest = adder.add_chunked_req(tail)
        # the #679 zero-budget park returns the tail WITHOUT batching it
        if any(tail is r for r in adder.can_run_list):
            readded.append(tail)
        if rest is not None:
            kept.append(tail)
    return TailReadd(kept=kept, readded=readded, not_named=not_named)


def adopt_anchor_tails(
    kept: Sequence[Any], minted: Sequence[Any], can_run_list: Sequence[Any]
) -> List[Any]:
    """The scheduler's tail list after this pass: the kept tails, then this
    pass's mints. Every tail IN this pass's batch takes one
    ``inflight_middle_chunks`` (its result is a middle chunk: no output, no
    finish) -- the increment the scheduler takes for ``chunked_req``."""
    tails: List[Any] = []
    seen = set()
    for req in list(kept) + list(minted):
        if id(req) in seen:
            continue
        seen.add(id(req))
        tails.append(req)
    in_batch = {id(r) for r in can_run_list}
    for req in tails:
        if id(req) in in_batch:
            req.inflight_middle_chunks += 1
    return tails


def stash_due(tails: Iterable[Any]) -> List[Any]:
    """Tails whose last chunk produced KV beyond the cached prefix -- the
    ``chunked_req`` stash condition (a parked chunk is a no-op)."""
    due = []
    for tail in tails:
        rng = getattr(tail, "extend_range", None)
        if rng is not None and rng.end > len(tail.prefix_indices):
            due.append(tail)
    return due


def bodies_in_batch(tails: Sequence[Any], can_run_list: Sequence[Any]) -> Tuple[Any, ...]:
    """The members of this pass's batch that continue as tails. Excluded from
    the running-batch merge when this batch comes back as ``last_batch`` -- a
    PP slot sees it ``pp_size`` passes later, after the tail ran in another
    slot, which is the reason ``last_batch.chunked_req`` is excluded."""
    in_batch = {id(r) for r in can_run_list}
    return tuple(t for t in tails if id(t) in in_batch)


def contains_last_prefill_chunk(
    can_run_list: Sequence[Any], chunked_req: Any, tails: Sequence[Any]
) -> bool:
    """``ScheduleBatch.contains_last_prefill_chunk``. With no tail it is the
    stock expression unchanged; with tails, a one-request batch whose only
    member continues (the chunked request OR a tail) carries no last chunk."""
    if not tails:
        return chunked_req is None or len(can_run_list) != 1
    if len(can_run_list) != 1:
        return True
    only = can_run_list[0]
    return not (only is chunked_req or any(only is t for t in tails))


class VoidRestore(NamedTuple):
    #: the scheduler's tail list after the void: the pre-admission list
    tails: List[Any]
    #: (tail, inflight_taken) for every pre-admission tail: its chunk never
    #: runs and is parked; the inflight increment is given back only if this
    #: pass took one (a truncated re-add) -- never for a final re-add, whose
    #: body chunk's increment is still owed to the body's own result
    parks: List[Tuple[Any, bool]]
    #: tails minted by the voided pass: plain batch members now, parked back
    #: into the waiting queue by the void with their give-backs (#984)
    dropped: List[Any]


def restore_after_void(*, before: Sequence[Any], current: Sequence[Any]) -> VoidRestore:
    """#797d for tails: undo the voided pass's tail admission."""
    cur = {id(t) for t in current}
    prev = {id(t) for t in before}
    parks = [(t, id(t) in cur) for t in before]
    dropped = [t for t in current if id(t) not in prev]
    return VoidRestore(tails=list(before), parks=parks, dropped=dropped)


def abort_targets(tails: Iterable[Any], *, rid: str, abort_all: bool) -> List[Any]:
    """Tails an AbortReq names. They are FINISHED WITH ABORT after their final
    forward (``to_finish``, the running-request route): the tail was decided by
    PP0 and runs on every stage, so dropping it on one rank would split the
    ring; letting its 1-4 tokens run keeps all ranks on one schedule."""
    return [
        t
        for t in tails
        if abort_all or str(getattr(t, "rid", "")).startswith(str(rid))
    ]


def log_pass(
    *,
    minted: Sequence[Any],
    readd: Optional[TailReadd],
    tails: Sequence[Any],
    pp_rank: int,
) -> None:
    """The boot's reading, once per pass with tail activity (rate-limited):
    how many anchor bodies this pass minted and how many tails it ended."""
    global _pass_n
    readded = list(readd.readded) if readd is not None else []
    if not minted and not readded:
        return
    _pass_n += 1
    if _pass_n > _LOG_FIRST and _pass_n % _LOG_EVERY:
        return
    kept_ids = {id(k) for k in (readd.kept if readd is not None else [])}
    ended = [t for t in readded if id(t) not in kept_ids]
    logger.info(
        "WEG2 ANCHOR-TAILS n=%d pp_rank=%d minted=%d tails_ended=%d kept=%d "
        "not_named=%d open=%d minted_rids=%s ended_rids=%s",
        _pass_n,
        int(pp_rank),
        len(minted),
        len(ended),
        len(kept_ids),
        len(readd.not_named) if readd is not None else 0,
        len(tails),
        # H42b: 16 chars -- x153b printed `weg2-8-1,weg2-8-1` for rids 11 and 14
        ",".join(str(getattr(r, "rid", "?"))[:16] for r in minted) or "-",
        ",".join(str(getattr(r, "rid", "?"))[:16] for r in ended) or "-",
    )


# ---------------------------------------------------------------- H42b
# The count arm's own fix (carried continuations already hold their seats,
# x153b fwd17) lives at the arm in scheduler.py: `+ _carried_n`.
class BurstVerdict(NamedTuple):
    hold: bool
    reason: str
    ready: int
    ready_tokens: int
    pending: int
    oldest_ms: float


def burst_quiet_ms(window_ms: float) -> float:
    """How long no new rid may have arrived before an assembled burst counts
    as complete: a quarter of the window, at least 20 ms (the tokenizer hands
    a burst over one request at a time)."""
    return max(20.0, float(window_ms) / 4.0)


def burst_hold_verdict(
    *,
    window_ms: float,
    now: float,
    carried: int,
    ready_arrivals: Sequence[float],
    ready_tokens: int,
    pending: int,
    last_arrival: Optional[float],
    budget_tokens: Optional[int],
    seat_cap: int,
) -> BurstVerdict:
    """fnFL2 H42b: hold THIS pass's fresh admissions for the rest of a burst?

    Only a pass that would carry NOTHING but new bodies is ever held: a
    forward that runs anyway (carried tails, the chunked request) takes the
    ready bodies along. Admit at once when the chunk budget or the seats are
    already full (nothing more could join), when the oldest ready request has
    waited the window, or when the queue is quiet (no rid seen within
    ``burst_quiet_ms`` and no #1400 store verdict outstanding). ``now`` and the
    arrivals are ``time.monotonic()`` seconds.
    """
    n = len(ready_arrivals)
    oldest_ms = (now - min(ready_arrivals)) * 1000.0 if n else 0.0

    def _v(hold: bool, reason: str) -> BurstVerdict:
        return BurstVerdict(hold, reason, n, int(ready_tokens), int(pending), oldest_ms)

    if window_ms <= 0:
        return _v(False, "off")
    if carried > 0:
        return _v(False, "carried")
    if n == 0:
        return _v(False, "nothing-ready")
    if budget_tokens is None:
        return _v(False, "no-chunk-budget")
    if ready_tokens >= budget_tokens:
        return _v(False, "budget-full")
    if n >= seat_cap:
        return _v(False, "seats-full")
    if oldest_ms >= window_ms:
        return _v(False, "window")
    quiet = last_arrival is None or (now - last_arrival) * 1000.0 >= burst_quiet_ms(window_ms)
    if pending == 0 and quiet:
        return _v(False, "quiet")
    return _v(True, "assembling")
