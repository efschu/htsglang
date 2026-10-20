"""S82 (19.10.2026): PP0 parks for its next set, not blind in the output receive.

THE SPECIMENS (rc18 7737b78948, P log ..._1019_200330; vorboot ..._1019_181004).

* 20:42:22-30 (the largest COMMIT-STALL of the boot, PP0 ``#1466 PASS-STALL ... commit_ms=4153``):
  PP0 admits pdflip-28-233 (16117 tokens) into slot 0 and runs its forward (host-blocking, MoE expert
  waves, ~3.1 s). Its fork twin pdflip-28-232 (17984 shared tokens) is held by TW and released only at
  the top of the NEXT PP0 pass (``#TW TWIN-REGISTER`` 20:42:26, after the forward); its store read takes
  102 ms. PP0 then runs slots 1 and 2 empty and parks in the output receive of slot 0 for 4153 ms while
  PP1/PP2 compute the same 16117-token chunk -- with the twin's read landed after 0.1 s
  (``PDFLIP-LOAD-DEVICE req=pdflip-28-232 ... read_ms=102 harvest_ms=4206``). The twin needed 30 tokens.
* 20:16:35-47 (monitor P-PAUSE 4.7 s, queue 4): the same shape with four TW twins of a vision request;
  their reads land at 20:16:38.8, PP0 sits in the output receive until 20:16:40.1 (commit_ms=1668), reaps
  them at 20:16:40.2 (harvest_ms 1388-1451).
* Both boots, bounded exactly (PASS-TAIL commit intervals x PDFLIP-LOAD-DEVICE harvest): a held request's
  read landed while PP0 sat in the output receive for 28.4-38.6 s (181004) and 7.9-14.3 s (rc18).

WHY A READ THAT LANDED IS NOT ENOUGH. On NF the #1400 told is PACED with the PF read acks (#1416e,
``PF TOLD-ACKED``): PP0 publishes the read-ahead at a pass top, every follower early-settles and acks it
at once ('#1400 FOLLOWER-EARLY-SETTLE ... at=ack' in the same second), and PP0 admits only at a LATER pass
top once the acks are in. While PP0 sits in the output receive it has no pass top, so the acks that
landed within milliseconds wait for the output of a set PP1/PP2 are still computing ('PF TOLD-ACKED
rid=pdflip-0-14 ... after=2.28s' with both acks sent at 20:07:02).

THE RULE. At the top of a PP0 pass (``pdflip_store_told.pp0_publish``, BEFORE the told objects go on the
wire -- the P-MINIFWD place), PP0 parks when

* the output receive at the end of THIS pass is going to block: the next slot holds an in-flight batch
  whose output is owed and not yet posted (the same counters ``_pp_wait_for_dict_readiness`` polls);
* no queued request is admissible already, none is decidable now, and no chunked request is carried
  (the plan would run it anyway);
* a queued request waits on an event that resolves WITHOUT PP0: its own store read is open (held,
  no deferral) or its paced Admit waits for the followers' acks;
* the pass carries no flip/flush/abort control request and NO new request (B1: its intake and store read
  run right after this pass top, as before S82 -- never behind the park).

It waits for the FIRST of: an ack set completes, an open read can terminate, the owed output is posted
or the probe can no longer tell (or the readiness budget runs out). No new clock: the park and the pass's
output receive share ONE deadline (review A1: ``_s82_receive_deadline``, the receive's readiness gate takes
only the rest), so a wedge is named after one budget as before.

PRIO = ALTER (review A3, user order 14.10.). An Admit decided in the park would run the younger set on PP0
before PP0 receives the OLDER set's output. It is admitted in THIS pass only when its MEASURED PP0 price
(START-LOADING + forward, :func:`price_ms`) ends before that output is due (:func:`down_ms` from the older
set's launch end); unmeasured, too late, or any other park ending -> the park's rids are HELD for this pass
(``_s82_hold``) and follow at the next pass top, after the older output -- the pre-S82 order. Instrument:
``S82 PARK-AGE`` per decision, ``S82 OUTPUT-LATE`` for the older output after a park admit. Metal check:
20:42 twin (30 tokens) 1754 ms against 3746 ms left -> admitted; 20:16 four younger (2431 on 409472 cached)
3338 ms against 1754 ms left -> held.

RANK-UNIFORM BY CONSTRUCTION (R11, RAENGE-NIE-UNEINS). Only PP0 waits, only BEFORE its tolds/Admits go
on the wire; the followers see the same objects on the request chain they always see, one pass of PP0
later or earlier, exactly as with P-MINIFWD. Nothing is decided off PP0, no follower sees a clock.

S82 TWIN-AT-PLAN (``twin_at_plan_on``): TW resolves a provisional promise at the source's first plan
(p_twin_defer rule 5), but asked only at the next pass TOP -- after PP0's own host-blocking forward. PP0
now asks ``release_due`` once more right after its plan, before the forward, so the twin's store read
runs DURING that forward. Same entry point, same decisions, only earlier; p_twin_defer itself is not
changed (S79 works there).

Switches (default ON, user order 14.10.): ``FLLIPER_PDFLIP_S82_COMMIT_PARK``,
``FLLIPER_PDFLIP_S82_TWIN_AT_PLAN``; 0 = the pre-S82 pass.

Narrow arguments: :func:`decide` and :func:`wait` are pure; the scheduler-facing adapter lives in
``pdflip_store_told``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_S82_COMMIT_PARK"
ENV_TWIN_AT_PLAN = "FLLIPER_PDFLIP_S82_TWIN_AT_PLAN"
#: Host poll of the park (the reads run on the prefetch thread, the acks arrive on the PF stream, the
#: output on the PP wire) -- the same step P-MINIFWD polls with.
POLL_S = 0.002
_LOG_FIRST = 16
_LOG_EVERY = 64

_STATE = {"n": 0, "acks": 0, "read": 0, "output": 0, "bound": 0, "error": 0, "waited_ms": 0.0,
          "twin_at_plan": 0}


def _env_on(name: str, env=None) -> bool:
    e = os.environ if env is None else env
    return str(e.get(name, "1")).strip().lower() not in ("0", "false", "no", "off", "")


def on(env=None) -> bool:
    return _env_on(ENV, env)


def twin_at_plan_on(env=None) -> bool:
    return _env_on(ENV_TWIN_AT_PLAN, env)


def reset() -> None:
    """Tests: forget the counters."""
    _STATE.clear()
    _STATE.update({"n": 0, "acks": 0, "read": 0, "output": 0, "bound": 0, "error": 0, "waited_ms": 0.0,
                   "twin_at_plan": 0})


def stats() -> dict:
    return dict(_STATE)


#: S82 B1 (Nach-Review 20.10.): request kinds whose intake registers a store read (#1400 intake). A pass that
#: carries one never parks -- its intake and read run right after the pass top, as before S82, not after the park.
ARRIVAL_SUFFIXES = ("GenerateReqInput", "EmbeddingReqInput")


def is_arrival(req) -> bool:
    """A new request whose intake (and store read) this pass's ``process_input_requests`` runs."""
    return type(req).__name__.endswith(ARRIVAL_SUFFIXES)


class Verdict(NamedTuple):
    park: bool
    reason: str
    reads: tuple
    acks: tuple


def decide(
    *,
    output_pending: Optional[bool],
    control: bool,
    carried: bool,
    arrivals: bool = False,
    admissible_any: bool,
    decidable_any: bool,
    open_reads: Sequence[str],
    ack_waits: Sequence[str],
) -> Verdict:
    """Park this PP0 pass top? ``output_pending`` True = the output receive at the end of this pass
    will block (owed, not posted); False = it will not; None = cannot be told (no park).
    ``admissible_any`` = a queued rid joins the plan now (told/Admit stored); ``decidable_any`` = a
    paced Admit this pass decides anyway; ``open_reads`` / ``ack_waits`` = the waiters."""
    if output_pending is None:
        return Verdict(False, "commit-unknown", (), ())
    if not output_pending:
        return Verdict(False, "no-blocking-commit", (), ())
    if control:
        return Verdict(False, "control", (), ())
    if arrivals:
        # B1: a new request's intake + store read must never wait behind the park (before S82 its read ran
        # while PP0 sat in the receive) -- the class 'P waits for a read' this module exists to end
        return Verdict(False, "arrivals", (), ())
    if carried:
        return Verdict(False, "carried-chunk", (), ())
    if admissible_any or decidable_any:
        return Verdict(False, "joins-anyway", (), ())
    reads, acks = tuple(open_reads), tuple(ack_waits)
    if not reads and not acks:
        return Verdict(False, "no-waiter", (), ())
    return Verdict(True, "park", reads, acks)


def wait(
    verdict: Verdict,
    *,
    output_pending: Callable[[], Optional[bool]],
    read_done: Callable[[str], bool],
    acks_done: Callable[[str], bool],
    harvest: Callable[[], None],
    bound_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple:
    """Poll until an ack set completes, an open read can terminate, the owed output is posted, or
    ``bound_s`` (the readiness budget of the receive this replaces) runs out.
    Returns ``(outcome, rid or None, waited_s)``; outcome 'acks' | 'read' | 'output' | 'bound' | 'error'.
    A broken probe ends the park ('error'), never the pass."""
    t0 = clock()
    while True:
        try:
            harvest()
            for rid in verdict.acks:
                if acks_done(rid):
                    return "acks", rid, clock() - t0
            for rid in verdict.reads:
                if read_done(rid):
                    return "read", rid, clock() - t0
            if output_pending() is not True:
                return "output", None, clock() - t0
        except Exception as exc:  # noqa: BLE001 -- the park is an optimisation, the pass must run
            logger.warning("S82 COMMIT-PARK probe failed (%s: %s)", type(exc).__name__, exc)
            return "error", None, clock() - t0
        waited = clock() - t0
        if waited >= bound_s:
            return "bound", None, waited
        # floor: a remainder that rounds to ~0 must still move the clock (no spin on a float tail)
        sleep(min(POLL_S, max(POLL_S / 10.0, bound_s - waited)))


def log_park(verdict: Verdict, outcome: str, rid: Optional[str], waited_s: float, slot) -> None:
    _STATE["n"] += 1
    n = _STATE["n"]
    _STATE[outcome] = _STATE.get(outcome, 0) + 1
    _STATE["waited_ms"] += waited_s * 1000.0
    if n <= _LOG_FIRST or n % _LOG_EVERY == 0 or outcome in ("bound", "error"):
        logger.info(
            "S82 COMMIT-PARK n=%d outcome=%s rid=%s waited_ms=%.1f slot=%s reads=%d acks=%d "
            "(acks=%d read=%d output=%d bound=%d waited_ms_sum=%.0f): PP0 waited at its pass top for the "
            "first of {ack set complete, read terminable, owed output posted} instead of inside the "
            "output receive",
            n, outcome, rid, waited_s * 1000.0, slot, len(verdict.reads), len(verdict.acks),
            _STATE["acks"], _STATE["read"], _STATE["output"], _STATE["bound"], _STATE["waited_ms"],
        )


# ---------------------------------------------------------------------------
# S82 A3 (review nf-review-s82-1019, user order 'Prio = Alter' 14.10.): an Admit decided in the park may
# NOT make the OLDER set wait. The new set runs on PP0 (START-LOADING + forward, host-blocking) before PP0
# receives the older set's output, so it is admitted from the park only when its MEASURED PP0 price ends
# before the older output is due:
#
#     now + load(cached_new) + fwd(n_new, cached_new)  <=  t_launch_old + down(n_old, cached_old)
#
# * fwd  -- PP0's own cost line ``ms = a + b*n + c*n*p_k`` (phase_policy.fit_cost_line, the X-COST-LINE fit)
#           over PP0's ``Prefill rank batch`` records (prefill_clock cost ring, this process);
# * load -- PP0's measured START-LOADING rate (ms per loaded token, median of this process's records);
# * down -- the same line fitted over PP0's EXACT downstream samples: launch end -> output received, taken
#           only from passes whose output was provably NOT posted when PP0 entered the receive (so the
#           receive end is the arrival, not PP0's lateness).
# Any of them unmeasured -> no Admit from the park (the older set first). Otherwise -- outcome 'output',
# 'bound', 'error', or the rule says no -- the park's ack rids are HELD for this pass: the older output is
# received and processed first, the younger Admit follows at the next pass top, exactly the pre-S82 order.

#: rings of measured samples (per process; only PP0 decides with them)
RING_MAX = 64
_LOAD: List[Tuple[int, float]] = []          # (loaded tokens, START-LOADING ms)
_DOWN: List[Tuple[int, int, float]] = []     # (new tokens, cached tokens, launch end -> output ms)
#: START-LOADING records below this many tokens carry the fixed cost, not the rate
LOAD_MIN_TOKENS = 8192


def _push(ring: list, item) -> None:
    ring.append(item)
    if len(ring) > RING_MAX:
        del ring[: len(ring) - RING_MAX]


def note_start_loading(tokens: int, ms: float) -> None:
    """``PDFLIP-START-LOADING`` of this rank: one measured load (scheduler-thread ms)."""
    try:
        n, v = int(tokens), float(ms)
    except (TypeError, ValueError):
        return
    if n >= LOAD_MIN_TOKENS and v > 0.0:
        _push(_LOAD, (n, v))


def note_downstream(new_tokens: int, cached_tokens: int, ms: float) -> None:
    """One EXACT downstream sample of PP0 (launch end -> output received, output not posted at entry)."""
    try:
        n, p, v = int(new_tokens), int(cached_tokens), float(ms)
    except (TypeError, ValueError):
        return
    if n > 0 and v > 0.0:
        _push(_DOWN, (n, max(0, p), v))


def reset_rings() -> None:
    """Tests: forget the measured samples."""
    del _LOAD[:]
    del _DOWN[:]


def _line_ms(line: Optional[dict], n: int, cached: int) -> Optional[float]:
    if not line:
        return None
    return float(line["a_ms"]) + float(line["b_ms"]) * n + float(line["c_ms"]) * n * (cached / 1000.0)


def _fit(rows) -> Optional[dict]:
    from flliper.srt.pdflip.phase_policy import fit_cost_line

    line, _why = fit_cost_line(list(rows), min_tokens=1, min_samples=3)
    return line


def fwd_rows() -> list:
    """PP0's own forward records (prefill_clock cost ring: one per flushed ``Prefill rank batch``)."""
    from flliper.srt.pdflip import prefill_clock as _pfc

    return [(int(r["n"]), int(r["cached"]), float(r["ms"])) for r in _pfc.cost_snapshot()["recent"]
            if int(r.get("chunks", 1) or 1) == 1]


def price_ms(n_new: int, cached_new: int, *, fwd: Optional[Sequence] = None,
             load: Optional[Sequence] = None) -> Optional[float]:
    """The new set's measured PP0 price (START-LOADING + forward), or None (unmeasured)."""
    line = _fit(fwd_rows() if fwd is None else fwd)
    f = _line_ms(line, int(n_new), int(cached_new))
    rates = sorted(v / n for n, v in (_LOAD if load is None else load) if n > 0)
    if f is None or not rates:
        return None
    rate = rates[len(rates) // 2]
    return f + rate * max(0, int(cached_new))


def down_ms(n_old: int, cached_old: int, *, down: Optional[Sequence] = None) -> Optional[float]:
    """The older set's measured downstream time (launch end -> output), or None (unmeasured)."""
    return _line_ms(_fit(_DOWN if down is None else down), int(n_old), int(cached_old))


class AgeVerdict(NamedTuple):
    admit: bool
    reason: str
    price_ms: Optional[float]
    due_in_ms: Optional[float]


def age_rule(*, now: float, price: Optional[float], launch_old: Optional[float],
             down: Optional[float]) -> AgeVerdict:
    """Prio = Alter: admit the younger set from the park only if its PP0 price ends before the older
    set's output is due (all times monotonic seconds, prices ms). Unmeasured -> no."""
    if price is None:
        return AgeVerdict(False, "price-unmeasured", None, None)
    if launch_old is None or down is None:
        return AgeVerdict(False, "due-unmeasured", price, None)
    due_in = (float(launch_old) + float(down) / 1000.0 - float(now)) * 1000.0
    if price <= due_in:
        return AgeVerdict(True, "before-due", price, due_in)
    return AgeVerdict(False, "older-first", price, due_in)


def log_age(v: AgeVerdict, rids: Sequence[str]) -> None:
    k = "age_admit" if v.admit else "age_hold"
    _STATE[k] = _STATE.get(k, 0) + 1
    n = _STATE[k]
    if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
        logger.info(
            "S82 PARK-AGE %s rids=%s reason=%s price_ms=%s due_in_ms=%s (n=%d): Prio = Alter -- the younger set "
            "is admitted from the park only if its measured PP0 price ends before the older set's output is due",
            "ADMIT" if v.admit else "HOLD", list(rids), v.reason,
            None if v.price_ms is None else round(v.price_ms), None if v.due_in_ms is None else round(v.due_in_ms), n,
        )


def log_output_late(slot, late_max_ms: Optional[float], predicted_late_ms: Optional[float]) -> None:
    """A3 instrument: the older set's output after a park that admitted the younger one -- ``late_max_ms``
    = receive entry minus park end when the output was already posted at entry (it was posted inside that
    interval; 0 = not yet posted at entry: no lateness); ``predicted_late_ms`` = receive entry minus the
    due time the rule used."""
    _STATE["late_n"] = _STATE.get("late_n", 0) + 1
    _STATE["late_max_ms_sum"] = _STATE.get("late_max_ms_sum", 0.0) + float(late_max_ms or 0.0)
    n = _STATE["late_n"]
    if n <= _LOG_FIRST or n % _LOG_EVERY == 0 or (late_max_ms or 0.0) > 0.0:
        logger.info(
            "S82 OUTPUT-LATE n=%d slot=%s output_late_ms<=%s predicted_late_ms=%s late_max_ms_sum=%.0f: how much "
            "later the OLDER set's output was received than posted, after the park admitted the younger set",
            n, slot, None if late_max_ms is None else round(late_max_ms),
            None if predicted_late_ms is None else round(predicted_late_ms), _STATE["late_max_ms_sum"],
        )


def log_twin_at_plan(rids: Sequence[str]) -> None:
    _STATE["twin_at_plan"] += len(rids)
    n = _STATE["twin_at_plan"]
    if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
        logger.info(
            "S82 TWIN-AT-PLAN rids=%s (n=%d): the held twin's wait resolved at the source's plan -- its store "
            "read is registered BEFORE PP0's forward, not at the next pass top", list(rids), n,
        )
