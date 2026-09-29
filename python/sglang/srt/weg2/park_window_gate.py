"""PARK-WINDOW-GATE (F22 / 27B decision 29.09. ~13:55Z): D admits no extend whose
cost reaches past the open collect window.

MEASURED (F22 marker audit): the park RPC in front of a D->P flip waits for the
D pass that is running when the collect window fires -- NF z30w-park median
0.40 s, p90 2.32 s (10 of 54 > 1 s; 09:01:38-43 a resume extend of 2564
tokens, eager expert pass 5.1 s, then the park 6.07 s); 27B z30j median
0.66 s, p90 1.15 s, max 3.31 s. D's own handler is 285 ms. A running forward
cannot be parked, so the only lever is not starting the long one.

ONE decision site stays the front's PARK-COLLECT-WINDOW (98b596db36 /
741bdfcef4): while it HOLDs it sends D its deadline (``left_ms``) and D's
measured cost line (X-COST-LINE ``ms = a + b*n + c*n*p_k``, 94e238791a); D
only applies it -- an extend whose forward (``a`` once per forward plus every
admitted request's marginal cost) would end after the deadline waits in the
queue. The window's own price, rent and caps decide when the park comes; this
gate adds no second rule.

Group-uniform without a new collective: the window arrives as one control
request, broadcast to every rank of D at the same scheduler pass, and every
term of the verdict (``left_ms`` as received, the line, the request's
uncached extent and prefix, the pass's admitted set, D's running count) is
replicated -- no clock is read at admission.

Never idles D: with nothing running the gate admits (the window fires at once
on an idle D anyway, 'd-idle'). The window clears at the park, at the wake and
when the front sends ``left_ms < 0``.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: the window in force on this scheduler: dict(epoch, left_ms, a_ms, b_ms, c_ms) or None
STATE_ATTR = "_weg2_park_window"
#: the marginal cost (ms) of the extends this gate admitted in the current pass
PASS_ATTR = "_weg2_park_window_pass_ms"
#: per request: how often the gate held it back, and since when (instrument only)
REQ_DEFERS_ATTR = "_weg2_pw_defers"
REQ_SINCE_ATTR = "_weg2_pw_since"
#: rankstats: the longest a held request waited before its admission (ms)
HOLD_MAX_ATTR = "_weg2_park_window_hold_max_ms"


def extend_ms(line: dict, new_tokens: int, prefix_tokens: int) -> float:
    """X-COST-LINE's marginal cost of one request in a forward (``a`` is the
    forward's, counted once): ``(b + c * p_k) * n`` ms."""
    return (float(line["b_ms"]) + float(line["c_ms"]) * max(0, int(prefix_tokens)) / 1000.0) \
        * max(0, int(new_tokens))


def admits(window: Optional[dict], *, new_tokens: int, prefix_tokens: int,
           batch_ms: float, running_n: int) -> tuple:
    """``(admit, cost_ms)``: the forward with this request (``a`` + the pass's
    admitted marginal costs ``batch_ms`` + this one) against ``left_ms``. Pure."""
    if window is None or int(running_n) <= 0:
        return True, 0.0
    marg = extend_ms(window, new_tokens, prefix_tokens)
    cost = float(window["a_ms"]) + float(batch_ms) + marg
    return cost <= float(window["left_ms"]), cost


#: the front re-sends an open window only when its deadline moved this much
#: (or the epoch / the line changed) -- one control request per step, not per poll
RESEND_MS = 250
PATH = "/weg2/park_window"


def front_message(sent: Optional[tuple], *, epoch: int, left_ms: int,
                  line: Optional[dict]) -> tuple:
    """The front's side, pure: ``(body or None, new_sent)``. ``left_ms < 0`` =
    the window is not open (clear once if one was sent); ``sent`` is
    ``(epoch, left_ms, line_key)`` of the last window sent."""
    if int(left_ms) < 0:
        if sent is None:
            return None, None
        return {"epoch": int(epoch), "left_ms": -1}, None
    if line is None:
        return None, sent
    key = (round(float(line["a_ms"]), 1), round(float(line["b_ms"]), 4), round(float(line["c_ms"]), 6))
    if (sent is not None and sent[0] == int(epoch) and sent[2] == key
            and int(sent[1]) - int(left_ms) < RESEND_MS):
        return None, sent
    # 27B review (c): D reads no clock, the window it applies is at most one
    # resend step old -- so the front sends it pessimistic by that step
    body = {"epoch": int(epoch), "left_ms": max(0, int(left_ms) - RESEND_MS),
            "a_ms": key[0], "b_ms": key[1], "c_ms": key[2]}
    return body, (int(epoch), int(left_ms), key)


def note(sched, recv_req) -> None:
    """The front's ``/weg2/park_window``: set (``left_ms >= 0``) or clear."""
    left = int(getattr(recv_req, "left_ms", -1))
    if left < 0:
        clear(sched, "front")
        return
    win = {"epoch": int(recv_req.epoch), "left_ms": left, "a_ms": float(recv_req.a_ms),
           "b_ms": float(recv_req.b_ms), "c_ms": float(recv_req.c_ms)}
    prev = getattr(sched, STATE_ATTR, None)
    setattr(sched, STATE_ATTR, win)
    n = getattr(sched, "_weg2_park_window_n", 0) + 1
    sched._weg2_park_window_n = n
    if prev is None or n <= 8 or n % 64 == 0:
        logger.info("WEG2 PARK-WINDOW-GATE set epoch=%d left_ms=%d line a=%.0f b=%.3f c=%.5f (n=%d): "
                    "D admits no extend whose forward ends after the collect window",
                    win["epoch"], left, win["a_ms"], win["b_ms"], win["c_ms"], n)


def clear(sched, why: str) -> None:
    if getattr(sched, STATE_ATTR, None) is not None:
        setattr(sched, STATE_ATTR, None)
        logger.info("WEG2 PARK-WINDOW-GATE clear why=%s defers_total=%d hold_max_ms=%d", why,
                    int(getattr(sched, "_weg2_park_window_defer_n", 0) or 0),
                    int(getattr(sched, HOLD_MAX_ATTR, 0) or 0))


def _released(sched, req) -> None:
    """A request the gate held before is admitted: name its defers and its
    hold (27B review (b): hunger is accepted -- after FIRE comes P, the resume
    stands first after the wake; the hard bound is the existing 60-s wait
    bound). The clock is the instrument's, never the verdict's."""
    n = int(getattr(req, REQ_DEFERS_ATTR, 0) or 0)
    if not n:
        return
    import time as _time

    held_ms = (_time.monotonic() - float(getattr(req, REQ_SINCE_ATTR, _time.monotonic()))) * 1000.0
    setattr(sched, HOLD_MAX_ATTR, max(float(getattr(sched, HOLD_MAX_ATTR, 0.0) or 0.0), held_ms))
    logger.info("WEG2 PARK-WINDOW-GATE released rid=%s defers=%d held_ms=%.0f window=%s",
                str(getattr(req, "rid", "?"))[:16], n, held_ms,
                "open" if getattr(sched, STATE_ATTR, None) is not None else "cleared")
    setattr(req, REQ_DEFERS_ATTR, 0)


def defers(sched, req, *, uncached: int, prefix_tokens: int, batch_empty: bool,
           running_n: int) -> bool:
    """The admission loop's verdict for ``req`` (True = it waits this pass).
    Inert without a window -- the stock admission, byte for byte."""
    window = getattr(sched, STATE_ATTR, None)
    if window is None:
        _released(sched, req)
        return False
    if batch_empty:
        setattr(sched, PASS_ATTR, 0.0)
    batch_ms = float(getattr(sched, PASS_ATTR, 0.0) or 0.0)
    if int(uncached) <= 0:
        # 27B review (a): no extend forward (TAIL-READY / E2 skip, nothing
        # uncached) -- no a, no marginal cost; it never delays the park
        _released(sched, req)
        return False
    ok, cost = admits(window, new_tokens=uncached, prefix_tokens=prefix_tokens,
                      batch_ms=batch_ms, running_n=running_n)
    if ok:
        setattr(sched, PASS_ATTR, batch_ms + extend_ms(window, uncached, prefix_tokens))
        _released(sched, req)
        return False
    if not getattr(req, REQ_DEFERS_ATTR, 0):
        import time as _time

        try:
            setattr(req, REQ_SINCE_ATTR, _time.monotonic())
        except AttributeError:  # a slotted stand-in: no instrument
            pass
    try:
        setattr(req, REQ_DEFERS_ATTR, int(getattr(req, REQ_DEFERS_ATTR, 0) or 0) + 1)
    except AttributeError:
        pass
    n = getattr(sched, "_weg2_park_window_defer_n", 0) + 1
    sched._weg2_park_window_defer_n = n
    if n <= 8 or n % 64 == 0:
        logger.info("WEG2 PARK-WINDOW-GATE defer rid=%s uncached=%d prefix=%d cost_ms=%.0f left_ms=%d "
                    "running=%d (n=%d): the forward would end after the collect window -- the park "
                    "must not wait for it", str(getattr(req, "rid", "?"))[:16], int(uncached),
                    int(prefix_tokens), cost, int(window["left_ms"]), int(running_n), n)
    return True
