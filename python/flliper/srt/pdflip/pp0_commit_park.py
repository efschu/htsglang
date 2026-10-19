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
* the pass carries no flip/flush/abort control request.

It waits for the FIRST of: an ack set completes, an open read can terminate, the owed output is posted
(or the readiness budget of that very receive runs out). No new clock: the wait ends no later than the
receive it replaces would have; then the pass runs exactly as before, with the Admit / told decided on
facts that are already true. An Admit decided in the park is admitted in THIS pass (the set runs on PP0
while PP1/PP2 still compute the previous one -- upstream's launch-before-exchange order, sglang
PR #40499), instead of one lap later.

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
from typing import Callable, NamedTuple, Optional, Sequence

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
    _STATE.update({"n": 0, "acks": 0, "read": 0, "output": 0, "bound": 0, "error": 0, "waited_ms": 0.0,
                   "twin_at_plan": 0})


def stats() -> dict:
    return dict(_STATE)


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


def log_twin_at_plan(rids: Sequence[str]) -> None:
    _STATE["twin_at_plan"] += len(rids)
    n = _STATE["twin_at_plan"]
    if n <= _LOG_FIRST or n % _LOG_EVERY == 0:
        logger.info(
            "S82 TWIN-AT-PLAN rids=%s (n=%d): the held twin's wait resolved at the source's plan -- its store "
            "read is registered BEFORE PP0's forward, not at the next pass top", list(rids), n,
        )
