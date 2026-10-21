"""W2C C2/C3 (20.10.2026): PP0 announces its next request chain in the proxy, the followers absorb it BEFORE the row.

THE SERIALISATION (rc27 893d09fa3f, P log ..._1020_145331, review ``done/nf-review-s103-w2c-entwurf-1020.md``).
P runs PP3 with a slot ring (``#969N ADMIT slot=0/1/2``), so a younger set CAN run on PP0 while PP1/PP2 compute the older
row. It does not, because the younger set is admitted only after every follower acked its told (EB before Admit, #1416e
paced / PF): PP0 publishes the read-ahead at a pass top, a follower acks it when it ABSORBS the chain at its own pass head,
PP0 admits once the acks are in. A follower holding the older row's proxy with no chain message pending computes the row
first (#1460 FOLLOWER-GATE); PP0's chain message of the next pass lands a few ms after the proxy and is absorbed only after
that host-blocking forward. The ack then comes about when the older output does, and PP0's S82 park ends on ``output``:
rc27 16:37:27 ``S82 COMMIT-PARK n=16 outcome=output ... waited_ms=743.6`` (pdflip-34-635), rc25 12:23:40 (pdflip-0-3,
2,3 s per the W2C design table).

THE RULE (C2/C3 with the review's V1, V2, V4):

* **C2, PP0 announces.** At the launch end of slot s (its proxy send), PP0 stamps ``__w2c_chain_next__ = k`` into the
  frame, k = the count of request-chain messages PP0 will have posted after its NEXT chain send (``local_sent(req) + 1``).
  Only when (a) a held, queued rid's store read is terminable now -- the next pass top publishes its read-ahead -- and
  (b) no other slot is in flight and no chunked request is carried (the pass end's receive cannot block, chain k follows
  within PP0's pass-top time). Otherwise the frame is byte-identical to before (no key).
* **V1, PP0 waits for nothing before chain k.** The pass that sends chain k skips every wait before its chain send (S82
  COMMIT-PARK, P-MINIFWD hold): a follower waiting for chain k cannot send the acks or the output those would wait for.
  The admit follows in a later pass's S82 park (``outcome=acks``), as before.
* **C3, the follower absorbs chain k before the row.** ``_pp_proxy_frame_pending`` leaves an announced frame in the inbox
  while this rank has absorbed fewer than k chain messages (``local_consumed(req) - len(inbox)``); the loop turns, the
  counter-gated chain receive takes chain k as soon as PP0 posted it, and the frame is delivered on the next probe. No new
  blocking receive, no new thread: the same defer the #791T STORE-TOLD hop uses.
* **V2, PP1 relays.** A middle rank stamps its own frame for the same slot with ``local_sent(req)`` -- every chain message
  it has forwarded, chain k among them (it absorbed and forwarded k before planning this row) -- so PP2 too absorbs k
  before its row.
* **V4, bounded and named.** A follower whose announced chain does not arrive within the proxy readiness budget stops
  with ``#789 W2C ANNOUNCED-CHAIN TIMEOUT`` (RAENGE-NIE-UNEINS), never a silent park.

RANK-UNIFORM. PP0 alone decides the announcement; the followers act on the number in the frame. Absorbing chain k before
the older row is the order an idle follower already runs today (rc27 16:37:21 PP1: ``#1460 CHAIN-RECV blocked 2820 ms``
returned the chain PP0 posted AFTER the 633 frame, before PP1 ran that row), so no new ordering is introduced (V3).

No switch, no timer, no reserve (user orders 14.10. / 15.10.): the wait is on an event (chain k posted), the bound is the
existing readiness budget.
"""

from __future__ import annotations

import os
import logging
import time
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

#: the frame key (a non-tensor dict entry, popped by the receiver before anything maps over the dict -- the
#: ``__stamp__`` / admission-row precedent)
KEY = "__w2c_chain_next__"

#: the module clock (tests drive a step clock through it)
_clock = time.monotonic

_LOG_FIRST = 24
_LOG_EVERY = 256
_N: Dict[str, int] = {"announce": 0, "sent": 0, "hold": 0, "release": 0, "relay": 0, "skip_waits": 0, "mismatch": 0}


def stats() -> dict:
    return dict(_N)


def reset() -> None:
    for k in _N:
        _N[k] = 0
    _WOULD.update({"n": 0, "qual": 0})


def _say(key: str) -> bool:
    _N[key] = _N.get(key, 0) + 1
    n = _N[key]
    return n <= _LOG_FIRST or n % _LOG_EVERY == 0


# ---------------------------------------------------------------------------
# pure verdicts


def pp0_should_announce(*, armed: bool, others_in_flight: bool, carried: bool, publishable: Iterable[str]) -> bool:
    """C2: announce chain k in this frame? Only when the next pass top publishes a read-ahead the followers must ack AND
    nothing else is in flight (the pass end's receive cannot block, so chain k follows within PP0's pass-top time)."""
    return bool(armed) and not others_in_flight and not carried and any(True for _ in publishable)


def announce_value(local_sent: int) -> int:
    """C2: the chain count after PP0's next send (one chain message per PP0 pass, unconditionally)."""
    return int(local_sent) + 1


def absorbed(local_consumed: int, inbox_len: int) -> int:
    """Chain messages this rank has taken out of its receiver (absorbed by a pass head)."""
    return int(local_consumed) - int(inbox_len)


def owed(announced: Optional[int], absorbed_n: int) -> bool:
    """C3: does an announced frame wait for its chain? ``None`` = not announced (today's path)."""
    return announced is not None and int(absorbed_n) < int(announced)


def relay_value(upstream_announced: Optional[int], local_sent: int) -> Optional[int]:
    """V2: a middle rank's own announcement for the slot -- everything it has forwarded (chain k included)."""
    return None if upstream_announced is None else int(local_sent)


def with_key(base: Optional[dict], value: Optional[int]) -> Optional[dict]:
    """The frame extras: ``base`` unchanged without an announcement, else a COPY with the key (``base`` is the relayed
    admission row and must not keep a stale announcement for a later pass)."""
    if value is None:
        return base
    out = dict(base) if base else {}
    out[KEY] = int(value)
    return out


class AnnouncedChainTimeout(RuntimeError):
    """V4: the announced chain never reached this rank within the readiness budget."""


def check_deadline(since: float, now: float, budget_s: float, *, slot: int, k: int, absorbed_n: int) -> None:
    if budget_s > 0 and now - since > budget_s:
        raise AnnouncedChainTimeout(
            f"#789 W2C ANNOUNCED-CHAIN TIMEOUT slot={slot} k={k} absorbed={absorbed_n} waited_s={now - since:.1f} "
            f"budget_s={budget_s:.1f}: the upstream announced its next request chain in this frame (PP0 W2C C2) and "
            f"has not posted it -- the frame is NOT computed on without it (RAENGE-NIE-UNEINS)")


# ---------------------------------------------------------------------------
# scheduler-facing adapters (getattr-based: stand-ins in the tests carry only what they touch)


def _counters(scheduler):
    return getattr(scheduler, "pp_flip_counters", None)


def _chan_req() -> str:
    from flliper.srt.managers.pp_wire_channels import CHAN_REQ

    return CHAN_REQ


def _local_sent(scheduler) -> Optional[int]:
    c = _counters(scheduler)
    if c is None:
        return None
    try:
        return int(c.local_sent(_chan_req()))
    except Exception:  # noqa: BLE001 - no counter, no announcement
        return None


def _absorbed_now(scheduler) -> Optional[int]:
    c = _counters(scheduler)
    receiver = getattr(scheduler, "pp_chain_receiver", None)
    if c is None or receiver is None:
        return None
    try:
        return absorbed(int(c.local_consumed(_chan_req())), len(getattr(receiver, "inbox", ()) or ()))
    except Exception:  # noqa: BLE001
        return None


def _pp_rank_size(scheduler):
    ps = getattr(scheduler, "ps", None)
    return int(getattr(ps, "pp_rank", 0) or 0), int(getattr(ps, "pp_size", 1) or 1)


def _armed_pp0(scheduler) -> bool:
    from flliper.srt.managers import pdflip_store_told as _told

    rank, size = _pp_rank_size(scheduler)
    if rank != 0 or size <= 1:
        return False
    if str(os.environ.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "")).strip() == "1":
        return False  # 27B DUAL (port 20.10.): the dual P keeps its PP protocol; PP0 announces nothing, followers follow
    if not bool(getattr(scheduler, "_pdflip_told_paced_on", False)):
        return False
    if getattr(scheduler, "pdflip_dormant", False):
        return False
    try:
        return bool(_told.armed(scheduler))
    except Exception:  # noqa: BLE001
        return False


def proxy_extra(scheduler, mb_id: int, base: Optional[dict]) -> Optional[dict]:
    """The frame extras for slot ``mb_id`` at the launch end: PP0 announces (C2), a middle rank relays (V2).
    Never raises into the send; without a verdict the frame is the pre-W2C frame (``base``)."""
    try:
        rank, size = _pp_rank_size(scheduler)
        if size <= 1:
            return base
        if rank == 0:
            k = _pp0_announce(scheduler, mb_id)
            if k is None:
                note_would_announce(scheduler, mb_id)
            return with_key(base, k)
        up = (getattr(scheduler, "_w2c_upstream_by_slot", None) or {}).pop(int(mb_id), None)
        if up is None:
            return base
        sent = _local_sent(scheduler)
        own = relay_value(up, sent) if sent is not None else None
        if own is not None and _say("relay"):
            logger.info("W2C RELAY slot=%d k_up=%d k_own=%d (n=%d): this rank absorbed and forwarded the announced "
                        "chain before its row; the next stage absorbs it before ITS row (V2)",
                        int(mb_id), int(up), own, _N["relay"])
        return with_key(base, own)
    except Exception as exc:  # noqa: BLE001 - an announcement is an optimisation, the frame must go
        logger.warning("W2C proxy_extra skipped (%s: %s)", type(exc).__name__, exc)
        return base


def _pp0_announce(scheduler, mb_id: int) -> Optional[int]:
    if not _armed_pp0(scheduler):
        return None
    mbs = getattr(scheduler, "mbs", None)
    if not isinstance(mbs, (list, tuple)):
        return None
    others = any(b is not None for j, b in enumerate(mbs) if j != int(mb_id))
    carried = getattr(scheduler, "chunked_req", None) is not None
    from flliper.srt.managers import pdflip_store_told as _told

    rids = _told.w2c_publishable(scheduler)   # the verdict below is the single place that weighs the slots
    if not pp0_should_announce(armed=True, others_in_flight=others, carried=carried, publishable=rids):
        return None
    sent = _local_sent(scheduler)
    if sent is None:
        return None
    k = announce_value(sent)
    scheduler._w2c_announced = (k, _clock())
    if _say("announce"):
        logger.info("W2C CHAIN-ANNOUNCE slot=%d k=%d rids=%s (n=%d): PP0's next chain carries a read-ahead the "
                    "followers ack before computing this row (C2)", int(mb_id), k, list(rids)[:4], _N["announce"])
    return k


# ---------------------------------------------------------------------------
# P-COMMIT MESSTOR (1): W2C WOULD-ANNOUNCE (review nf-rc30-pcommit-review-1021 §2.5; instrument only, no frame change)
#
# The class "arrival during PP0's host-blocking launch" (rc30: 28 drains / 40.6 s) would need C2 to announce on a
# READABLE request socket instead of a terminable held read. Before anything is built: at the launch end PP0 asks the
# socket (``request_receiver.control_message_pending``: a zero-timeout zmq poll, nothing consumed) and C2(b) (no other
# slot in flight, no carried chunk); at the NEXT pass top -- where the frames are received anyway -- it names what they
# were: generate/embedding arrivals vs control requests (W2C review V3: only a pure-arrival chain may be announced).
# ``qualifies`` = readable AND C2(b) AND arrivals only. The qualifying rids are kept for #PDRAIN, which sums the drain
# wait they sat behind (``w2c_would_ms_sum``) -- the per-boot seconds the build gate (~20 s/boot) is read from.
_W2C_GEN_SUFFIXES = ("GenerateReqInput", "EmbeddingReqInput")
_W2C_INTERNAL = ("PdFlipVisionVerdict",)
_WOULD: Dict[str, Any] = {"n": 0, "qual": 0}


def note_would_announce(scheduler, mb_id: int) -> None:
    """PP0 at a launch end whose frame carries NO announcement: is the request socket readable, is C2(b) met?"""
    try:
        if not _armed_pp0(scheduler):
            return
        rr = getattr(scheduler, "request_receiver", None)
        probe = getattr(rr, "control_message_pending", None)
        if not callable(probe) or not probe():
            return
        mbs = getattr(scheduler, "mbs", None)
        others = isinstance(mbs, (list, tuple)) and any(b is not None for j, b in enumerate(mbs) if j != int(mb_id))
        carried = getattr(scheduler, "chunked_req", None) is not None
        scheduler._w2c_would = {"t": _clock(), "slot": int(mb_id), "others": bool(others), "carried": bool(carried)}
    except Exception:  # noqa: BLE001 -- an instrument
        pass


def classify_kinds(reqs) -> Dict[str, Any]:
    gen, ctrl, other, rids = 0, 0, 0, []
    for r in reqs or ():
        name = type(r).__name__
        if name.endswith(_W2C_GEN_SUFFIXES):
            gen += 1
            rid = getattr(r, "rid", None)
            if isinstance(rid, str):
                rids.append(rid)
        elif name in _W2C_INTERNAL:
            other += 1
        else:
            ctrl += 1
    return {"gen": gen, "ctrl": ctrl, "other": other, "rids": rids}


def pp0_would_at_top(scheduler, recv_reqs) -> None:
    """PP0 pass top (``pdflip_store_told.pp0_publish``), the frames this pass received: close the launch end's probe."""
    w = getattr(scheduler, "_w2c_would", None)
    if w is None:
        return
    scheduler._w2c_would = None
    try:
        k = classify_kinds(recv_reqs)
        c2b = not w["others"] and not w["carried"]
        qualifies = c2b and k["gen"] > 0 and k["ctrl"] == 0
        _WOULD["n"] += 1
        if qualifies:
            _WOULD["qual"] += 1
            d = getattr(scheduler, "_w2c_would_rids", None)
            if d is None:
                d = scheduler._w2c_would_rids = {}
            for rid in k["rids"]:
                d[rid] = _clock()
            while len(d) > 256:
                d.pop(next(iter(d)))
        logger.info(
            "W2C WOULD-ANNOUNCE slot=%d readable_at_launch_end=1 others_in_flight=%d carried=%d c2b=%d gen=%d ctrl=%d "
            "other=%d qualifies=%d rids=%s after_ms=%.0f (n=%d qual=%d): instrument only -- the frame carried no "
            "announcement; a socket trigger would have announced this chain",
            w["slot"], int(w["others"]), int(w["carried"]), int(c2b), k["gen"], k["ctrl"], k["other"],
            int(qualifies), k["rids"][:4], (_clock() - w["t"]) * 1000.0, _WOULD["n"], _WOULD["qual"],
        )
    except Exception:  # noqa: BLE001
        pass


def would_stats() -> dict:
    return dict(_WOULD)


def pp0_take_announced(scheduler) -> Optional[int]:
    """V1, PP0 pass top (``pp0_publish``): the chain this pass sends was announced -> no wait before it. One-shot: the
    flag is set at the previous pass's proxy send and taken by the very next pass top (one chain send per PP0 pass)."""
    a = getattr(scheduler, "_w2c_announced", None)
    if a is None:
        return None
    scheduler._w2c_announced = None
    scheduler._w2c_sending = a
    if _say("skip_waits"):
        logger.info("W2C PASS-K k=%d (n=%d): announced pass -- no S82 park / MINIFWD hold before the chain send (V1)",
                    int(a[0]), _N["skip_waits"])
    return int(a[0])


def pp0_after_chain_send(scheduler) -> None:
    """PP0, right after the chain send of a pass: the announced k must be the count just reached."""
    a = getattr(scheduler, "_w2c_sending", None)
    if a is None:
        return
    scheduler._w2c_sending = None
    try:
        k, t0 = int(a[0]), float(a[1])
        sent = _local_sent(scheduler)
        if sent is not None and sent != k:
            _N["mismatch"] += 1
            logger.warning("W2C CHAIN-K MISMATCH k=%d sent=%d (n=%d): the announced count was not the next send -- "
                           "the followers wait until the count reaches k (PP0 sends one chain per pass)",
                           k, sent, _N["mismatch"])
        elif _say("sent"):
            logger.info("W2C CHAIN-K-SENT k=%d after_ms=%.1f (n=%d): the announced chain is on the wire",
                        k, (_clock() - t0) * 1000.0, _N["sent"])
    except Exception:  # noqa: BLE001
        pass


def note_frame(scheduler, mb_id: int, raw) -> Optional[int]:
    """Receiver: pop the announcement off a frame (BEFORE anything maps over the dict) and keep it for this rank's
    relay of the same slot (V2)."""
    if not isinstance(raw, dict):
        return None
    value = raw.pop(KEY, None)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    d = getattr(scheduler, "_w2c_upstream_by_slot", None)
    if d is None:
        d = scheduler._w2c_upstream_by_slot = {}
    d[int(mb_id)] = value
    w = getattr(scheduler, "_w2c_wait", None) or {}
    since = w.pop((int(mb_id), value), None)
    if since is not None and _say("release"):
        logger.info("W2C FRAME-RELEASE slot=%d k=%d waited_ms=%.1f (n=%d): the announced chain was absorbed, the "
                    "row runs now (C3)", int(mb_id), value, (_clock() - since) * 1000.0, _N["release"])
    return value


def follower_defer(scheduler, mb_id: int, head, budget_s: Optional[float] = None) -> bool:
    """C3 at the frame probe: True = leave the announced frame in the inbox (this rank has not absorbed chain k yet).
    The loop keeps turning; the counter-gated chain receive takes k when PP0 posted it. V4: past the readiness budget
    a named stop."""
    if not isinstance(head, dict):
        return False
    k = head.get(KEY)
    if k is None:
        return False
    n = _absorbed_now(scheduler)
    if n is None:
        return False
    k = int(k)
    w = getattr(scheduler, "_w2c_wait", None)
    if w is None:
        w = scheduler._w2c_wait = {}
    if not owed(k, n):
        return False
    now = _clock()
    since = w.get((int(mb_id), k))
    if since is None:
        if len(w) >= 64:   # a held frame later dropped as stale never releases its entry: keep the map bounded
            for key in sorted(w, key=w.get)[:32]:
                w.pop(key, None)
        w[(int(mb_id), k)] = now
        if _say("hold"):
            logger.info("W2C FRAME-HOLD slot=%d k=%d absorbed=%d (n=%d): the frame announces chain k -- absorbed "
                        "before this row so the told's ack is not behind the forward (C3)", int(mb_id), k, n, _N["hold"])
        return True
    if budget_s is None:
        from flliper.srt.managers.scheduler_pp_mixin import _pp_proxy_readiness_budget_s

        budget_s = _pp_proxy_readiness_budget_s()
    check_deadline(since, now, float(budget_s), slot=int(mb_id), k=k, absorbed_n=n)
    return True


__all__ = [
    "KEY", "AnnouncedChainTimeout", "pp0_should_announce", "announce_value", "absorbed", "owed", "relay_value",
    "with_key", "check_deadline", "proxy_extra", "pp0_take_announced", "pp0_after_chain_send", "note_frame",
    "follower_defer", "stats", "reset",
]
