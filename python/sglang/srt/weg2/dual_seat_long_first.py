"""D-SEAT-LONG-FIRST (#1986, dual front, ENV switch, default OFF): the next free D seat goes first to a
request whose LONG P leg is already finished.

Metal pt2 (boot ...fs10051150, 11:54-12:03Z): weg2-0-10 (249962 tokens) finished P leg 1 at 11:59:08
(174 s of P work) and never got a D seat before the probe client's 300 s timeout dropped it at 11:59:51
(``WEG2-CLIENT-GONE state=handoff action=dropped-before-d``). All six D seats stood full for 209 s
(11:56:23 -> 11:59:52). The seat ORDER was not the cause there (0-10 was the oldest arrival, so already
the head of ``_ready_for_d`` under SEAT-AGE) -- but behind a contended queue the order is the only lever
the front has: the seat-wait of leg-1-done requests was 8-36 s each in that boot (median 14 s over 37), and
a 174 s P leg thrown away for a 30 s seat wait is the expensive loss.

THE RULE. In the D admitter's waiting line (``Front._ready_for_d``, leg 1 done, no seat yet) a request whose
P leg COMPUTED at least ``SGLANG_WEG2_DUAL_D_SEAT_LONG_P_TOKENS`` tokens moves in front of the requests that
did not (long ones among themselves in arrival-age order; the rest keep the order they had). Only the
order of the WAITING changes, only at the moment a seat is taken:

* no running decode is touched, nothing is parked or displaced (a seat that is held stays held; D never
  pauses for this);
* the number of seats, the KV/token budget gates and the SEAT-AGE backfill (an older head that does not fit
  D's free KV lets the first younger one that fits run) are unchanged -- a long head that does not fit is
  still passed by one that does;
* it decides the front's seat hand-out alone: one process, no collective, no wall clock (the key is the
  integer ``leg1_prompt_tokens - leg1_cached_tokens`` the front read from P's usage when leg 1 returned,
  and the rid's arrival counter). The D ranks' own queue order (age by rid, SEAT-AGE displacement on D) is
  untouched and is NOT consulted -- so every D rank sees the same requests in the same order whatever the
  switch says.

"Computed", not "prompt": ``0-21`` had prompt_tokens=51122 but cached_tokens=47360 -- 6.3 s of P work. The
cost the rule protects is the P work that would be redone, which is the uncached part.

Inert unless ``Front.dual_layout`` and the switch > 0 (default 0 = byte for byte as before).

RAW-PROMPT MEASURE (#1998, ENV ``SGLANG_WEG2_DUAL_D_SEAT_LONG_USE_RAW_PROMPT``, default OFF = the measure above,
byte for byte). pt4 (fs10051332): the 250k needle (weg2-0-61, prompt 249921) was paused 3x and resumed from L2;
its LAST leg read ``cached_tokens=241664`` and computed 8257 < 32768, so it did not count as long and the rule
did not protect the very request it was built for (the 65k foreign request went before it). With the switch on
the measure is the request's RAW prompt length (``leg1_prompt_tokens`` = P's usage prompt_tokens, the whole
prompt incl. the cached head; ``est_prompt`` when leg 1 reported none), whatever the cache or earlier legs did.
The threshold stays ``SGLANG_WEG2_DUAL_D_SEAT_LONG_P_TOKENS`` (metal: 131072, so a 40-65k request does not get
precedence). Same properties: order of the waiting only, front-side integers, no clock, no collective."""
from __future__ import annotations

from typing import Any, Callable, Iterable, List

from sglang.srt.environ import envs
from sglang.srt.weg2.seat_age import rid_age

ENV = "SGLANG_WEG2_DUAL_D_SEAT_LONG_P_TOKENS"
ENV_RAW_PROMPT = "SGLANG_WEG2_DUAL_D_SEAT_LONG_USE_RAW_PROMPT"


def threshold() -> int:
    """Tokens of P work from which a finished leg 1 is 'long'; 0 = off."""
    try:
        return max(0, int(envs.SGLANG_WEG2_DUAL_D_SEAT_LONG_P_TOKENS.get()))
    except Exception:  # noqa: BLE001 -- a bad value is OFF, never a crash in the admitter
        return 0


def p_work_tokens(p: Any) -> int:
    """The P work of this request's finished leg 1: prompt minus what P read from the cache."""
    pt = int(getattr(p, "leg1_prompt_tokens", 0) or 0)
    ct = int(getattr(p, "leg1_cached_tokens", 0) or 0)
    return max(0, pt - ct)


def raw_prompt_tokens(p: Any) -> int:
    """The RAW prompt length of the request (cached head included): leg 1's usage prompt_tokens, the front's
    ``est_prompt`` when leg 1 reported none (same fallback as ``_hs.lost_terms`` in the front)."""
    pt = int(getattr(p, "leg1_prompt_tokens", 0) or 0)
    return pt or int(getattr(p, "est_prompt", 0) or 0)


def use_raw_prompt() -> bool:
    """#1998 switch: measure the raw prompt instead of the computed P work (default OFF)."""
    try:
        return bool(envs.SGLANG_WEG2_DUAL_D_SEAT_LONG_USE_RAW_PROMPT.get())
    except Exception:  # noqa: BLE001 -- a bad value is OFF
        return False


def measure() -> Callable[[Any], int]:
    """The 'how long is this request' function the rule compares with the threshold."""
    return raw_prompt_tokens if use_raw_prompt() else p_work_tokens


def long_first(items: Iterable[Any], min_tokens: int,
               work_of: Callable[[Any], int] = p_work_tokens,
               rid_of: Callable[[Any], Any] = lambda x: getattr(x, "rid", x)) -> List[Any]:
    """``items`` with the long-P-leg ones first (oldest arrival first among them), the others after them in
    their given order. ``min_tokens`` <= 0 returns the given order unchanged."""
    seq = list(items)
    if int(min_tokens) <= 0:
        return seq
    longs = [x for x in seq if work_of(x) >= int(min_tokens)]
    if not longs:
        return seq
    longs.sort(key=lambda x: rid_age(rid_of(x)))   # stable
    ids = {id(x) for x in longs}
    return longs + [x for x in seq if id(x) not in ids]
