"""RESUME-VIA-P (27B rc12k27 b23, 27.09.; NF parks alike): a request D refuses
with W50 AFTER its stream has started is not aborted -- P prefills its full
context and D continues the SAME stream.

THE DEATH (b23, three client streams, 10:23:16 / 10:23:38 / 10:24:16). Every
one was a running, streaming request that the immediate park parked and the
wake resumed; at the resume D found its prefix gone (``PHASE-PURITY ...
state=cold host_hit=0``), the X gate priced the whole context
(``X-GATE uncached=80508 X=12288 verdict=W31``) and ``_weg2_answer_x_refusals``
aborted it. The abort reaches the client as an in-band error after the first
byte -- the front's re-route (X-REQUEUE, which works for fresh requests: 4x on
b23) cannot apply, the client already holds text (``W50 ... (STREAM, served)
... re-route impossible``). Law 4 forbids the only other way out (D prefills
over X; memory kein-d-direct-prefill-ueber-x).

THE PATH (switch ``SGLANG_WEG2_RESUME_VIA_P``, default on; ``0`` = the abort,
byte for byte):

* D (every rank, the same queue mutation; only TP0 writes): a STREAMED request
  with generated tokens that the X gate refuses is kept -- moved to the park
  list (``weg2_d_parked``, park site flip), its park-read cap and #1324 stamp
  cleared -- and its exact context (``origin + output`` ids) is written to
  ``<arena>/needs-p/<rid>.json`` with the refused extent. Bounded: the
  ``MAX_ATTEMPTS``-th refusal of one rid falls back to the named abort.
* Front: each controller pass takes the files (one scandir of a small dir),
  logs ``WEG2 W50-REROUTE ... path=midstream`` and queues a P-only leg 1 --
  ``/generate`` with those input_ids, ``max_new_tokens=1``, the same rid. That
  is an over-X request like any LONG: the immediate park and the flip follow,
  P prefills and publishes (its hand-off keys under the rid), the flip back
  wakes D, the parked request's hold read finds the context, the X gate
  admits the tail and D continues the stream it never closed.
  ``RESUME-VIA-P done rid p_ms d_resume_ms`` when the stream moves again.

No bytes are added to any client stream (no SSE comment): the signal travels
through the shared arena directory, which every group and the front can see.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_RESUME_VIA_P"
SUBDIR = "needs-p"
#: the number of P legs one rid may get (a loop through P that never lands
#: must end by name, not spin).
#:
#: KRIT3 (NF rc12s 17:33-17:35, weg2-3-16 / weg2-3-18): the bound counted
#: REFUSALS, and a parked request is re-offered to the X gate on every awake
#: pass (park_tick's awake re-queue) -- weg2-3-16 was refused at 17:33:33 (n=1)
#: and 17:33:36 (n=2) in the SAME D phase, before any P leg ran (the front
#: dropped the second needs-p: one path per rid), then once after its one P
#: leg (17:35:47, n=3) and once more in that same pass: exhausted, in-band W50,
#: an answerless 200. Now an attempt is a P LEG: a refusal in the same D wake
#: as the previous one is the same hold (no new attempt, still held), and the
#: bound is 2 -- the first P leg, then EXACTLY ONE more sequential P leg
#: (attempt=2); a refusal after the second leg ends by name.
MAX_ATTEMPTS = 2
ATTEMPTS_ATTR = "_weg2_rvp_n"
SINCE_ATTR = "_weg2_rvp_since"
#: the D wake (see :func:`note_wake`) in which this rid's last attempt began.
WAKE_ATTR = "_weg2_rvp_wake"
#: scheduler attribute: D's wake count (every rank, the same resume requests).
SCHED_WAKE_ATTR = "_weg2_rvp_wake_n"
ENV_WAKE_ATTEMPTS = "SGLANG_WEG2_RVP_ATTEMPT_PER_WAKE"


def wake_attempts_enabled(env=None) -> bool:
    """KRIT3: ``SGLANG_WEG2_RVP_ATTEMPT_PER_WAKE`` (default on; 0 = every
    refusal is an attempt, bound 3, as before)."""
    e = os.environ if env is None else env
    raw = (e.get(ENV_WAKE_ATTEMPTS, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def max_attempts(env=None) -> int:
    return MAX_ATTEMPTS if wake_attempts_enabled(env) else 3


def note_wake(sched) -> None:
    """KRIT3: a resume of D's memory is a wake -- P ran in between. Called on
    every rank from the same resume request (replicated, no collective)."""
    if sched is None:
        return
    try:
        setattr(sched, SCHED_WAKE_ATTR, int(getattr(sched, SCHED_WAKE_ATTR, 0) or 0) + 1)
    except Exception:  # noqa: BLE001 -- a partial scheduler keeps the old count
        pass


def _same_hold(sched, req) -> bool:
    """KRIT3: this refusal falls in the same D wake as ``req``'s last attempt
    (no P leg can have run for it in between)."""
    if sched is None or not wake_attempts_enabled():
        return False
    last = getattr(req, WAKE_ATTR, None)
    return last is not None and last == int(getattr(sched, SCHED_WAKE_ATTR, 0) or 0)


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def arena_dir(tag: str = "", env=None) -> str:
    e = os.environ if env is None else env
    base = (e.get("SGLANG_HICACHE_ARENA_DIR", "") or "").strip()
    if not base and tag:
        base = f"/dev/shm/weg2-arena-{tag}"
    return base


def needs_p_dir(tag: str = "", env=None) -> str:
    base = arena_dir(tag, env)
    return os.path.join(base, SUBDIR) if base else ""


#: ROS (NF rc12m-dpr 12:16:12, weg2-16-59 / weg2-18-64): the hold covers EVERY
#: streamed request, output or not. D cannot know whether the front already
#: sent the client a byte (message_start + 5 s pings: the front's lookahead
#: commits after 8 chunks, ~35 s, with output=0), and a per-rank read of a
#: front marker would be a rank-local verdict. So D holds every streamed
#: refusal (the replicated ``stream`` flag decides) and the FRONT, which knows
#: whether it committed, answers: committed -> P-only leg 1 (RESUME-VIA-P);
#: not committed -> X-REQUEUE exactly as before (closing D's leg aborts the
#: parked request, d_park_runtime.park_abort). ``0`` = the output>0 rule.
ENV_OPEN_STREAM = "SGLANG_WEG2_RESUME_OPEN_STREAM"


def open_stream_enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV_OPEN_STREAM, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


#: the needs-p reason of a VISION hold (D cannot cover an image of a request
#: that has already streamed): the front's P leg sends the ORIGINAL body.
REASON_VISION = "vision_not_in_prefix"


def eligible(req, env=None, sched=None, vision: bool = False) -> bool:
    """A refusal D may turn into RESUME-VIA-P: switch on, group D, a streamed
    request, under the attempt bound. With ROS off only a request that has
    generated tokens (the old rule); with ROS on every streamed request -- the
    front decides between the resume and X-REQUEUE (see ENV_OPEN_STREAM). A
    non-stream request keeps the abort (X-REQUEUE, nothing reached the client)."""
    e = os.environ if env is None else env
    if not enabled(e):
        return False
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
        return False
    if not bool(getattr(req, "stream", False)):
        return False
    if getattr(req, "multimodal_inputs", None) is not None and not vision:
        # VISION-D x ROS: the P-only leg 1 carries input_ids alone -- P would
        # prefill an image's placeholder ids as text and D would continue on a
        # wrong prefix. An image request keeps the named W50 (X-REQUEUE sends
        # the original request, image included, before the first byte). The
        # VISION hold (``vision=True``) is different: its P leg is the ORIGINAL
        # body (image included), see ``REASON_VISION`` and the front.
        return False
    if not open_stream_enabled(e):
        try:
            if len(getattr(req, "output_ids", None) or ()) <= 0:
                return False
        except TypeError:
            return False
    if _same_hold(sched, req):
        return True  # KRIT3: the same hold -- the front already has its P leg
    return int(getattr(req, ATTEMPTS_ATTR, 0) or 0) < max_attempts(e)


def held_refusal_body(rec: dict) -> bytes:
    """ROS: the refusal text the front's X-REQUEUE parses, for a request D held
    (the needs-p record) instead of refusing in-band -- same marker, same
    extent sentence as D's W50 message."""
    return (f"W50 Weg2TpPrefillExceeded (held on D, not yet committed to the client): "
            f"this request's extent after prefix matching is {int(rec.get('d_extent') or 0)}. "
            f"X={int(rec.get('x') or 0)}").encode()


def context_ids(req) -> List[int]:
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if ids is None:
        ids = list(getattr(req, "origin_input_ids", None) or ()) + list(getattr(req, "output_ids", None) or ())
    return [int(t) for t in ids]


def write_request(rid: str, ids: Iterable[int], d_extent: int, x: int, reason: str,
                  directory: Optional[str] = None, attempt: int = 1,
                  extra_key: Optional[str] = None) -> str:
    """Atomically publish one needs-P request; returns the path ('' = not written).
    ``extra_key`` (Q-460): the request's KV namespace, so P prefills the
    context under the key D reads it back with."""
    d = directory if directory is not None else needs_p_dir()
    if not d or not rid:
        return ""
    try:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{rid}.json")
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            rec = {"rid": rid, "input_ids": list(ids), "d_extent": int(d_extent), "x": int(x),
                   "reason": reason, "attempt": int(attempt), "t": time.time()}
            if extra_key is not None:
                rec["extra_key"] = str(extra_key)
            json.dump(rec, f)
        os.replace(tmp, p)
        return p
    except Exception:  # noqa: BLE001 - an unwritten request falls back to the abort
        logger.warning("RESUME-VIA-P write failed for rid=%s", rid, exc_info=True)
        return ""


def take_requests(directory: str) -> List[dict]:
    """Front: consume every published request (read, then remove). A file that
    cannot be read is removed and skipped by name."""
    out: List[dict] = []
    if not directory:
        return out
    try:
        entries = [e for e in os.scandir(directory) if e.name.endswith(".json")]
    except FileNotFoundError:
        return out
    except OSError as exc:
        logger.warning("RESUME-VIA-P scandir %s failed: %r", directory, exc)
        return out
    for e in sorted(entries, key=lambda x: x.name):
        try:
            with open(e.path) as f:
                out.append(json.load(f))
        except Exception as exc:  # noqa: BLE001
            logger.warning("RESUME-VIA-P unreadable request %s skipped: %r", e.name, exc)
        try:
            os.remove(e.path)
        except OSError:
            pass
    return out


#: #248h: first capacity park of this rid (monotonic, rank-local; the group
#: verdict is MIN-reduced) and the last one (the re-read cadence).
CAPPARK_SINCE_ATTR = "_weg2_rvp_cappark_since"
CAPPARK_AT_ATTR = "_weg2_rvp_cappark_at"
CAPPARK_MARK = "#248h RESUME-VIA-P CAPACITY-PARK"
#: the shortest interval between two capacity re-reads of one rid (the #1456
#: re-read timer's value): a read that ends short again is not re-issued in a
#: loop of passes.
CAPPARK_REREAD_S = 2.0


def capacity_park_bound_s() -> float:
    try:
        from sglang.srt.environ import envs

        return float(envs.SGLANG_WEG2_RVP_CAPACITY_PARK_S.get() or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def capacity_park_on() -> bool:
    return capacity_park_bound_s() > 0 and enabled()


def capacity_park_precondition(req) -> bool:
    """#248h: the REPLICATED half of the candidate test (switch, group D, a
    stream without an image, released by the W88 store-short bound) -- it
    decides whether the group takes the MIN vote at all, so it may read no
    rank-local term. Never raises."""
    try:
        if not capacity_park_on():
            return False
        if (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
            return False
        if not bool(getattr(req, "stream", False)):
            return False
        if getattr(req, "multimodal_inputs", None) is not None:
            return False
        return bool(getattr(req, "_weg2_store_short_fallback", False))
    except Exception:  # noqa: BLE001
        return False


def capacity_park_candidate(req, x: int, sched=None, now: Optional[float] = None) -> bool:
    """#248h (30.09., NF y4b D 03:58:57, weg2-32-72): this rank's vote that
    the X refusal of ``req`` is a CAPACITY short read, not a missing context.

    True when: the RESUME-VIA-P preconditions hold (switch, group D, a stream,
    no image); its last store read was released by the store-short bound
    (``_weg2_store_short_fallback``, the W88 over-X re-route); it delivered
    less than the store holds (``delivered < deliverable``: pages lost to a
    full arena / host budget, not unwritten); what the store holds covers the
    context up to X (a whole read would be admitted -- P already wrote it,
    another P leg cannot add anything); and the park bound has not lapsed.
    y4b: delivered 39168, deliverable 95104, total 95137, X 12288 -> True
    three times; the base spent two P legs (7.7 s / 7.5 s, flips included)
    and ended the client's stream with W50 at the third refusal. Never raises
    (a vote of the group MIN)."""
    try:
        if not capacity_park_precondition(req):
            return False
        delivered = getattr(req, "_weg2_store_delivered", None)
        deliverable = getattr(req, "_weg2_store_deliverable", None)
        if delivered is None or deliverable is None or int(delivered) >= int(deliverable):
            return False
        ids = getattr(req, "full_untruncated_fill_ids", None)
        total = len(ids) if ids is not None else len(context_ids(req))
        if x <= 0 or total - int(deliverable) > int(x):
            return False
        since = getattr(req, CAPPARK_SINCE_ATTR, None)
        now = time.monotonic() if now is None else float(now)
        return since is None or (now - float(since)) < capacity_park_bound_s()
    except Exception:  # noqa: BLE001 -- a vote never breaks the MIN
        return False


def park_for_capacity(sched, req, d_extent: int, x: int) -> None:
    """#248h: D keeps ``req`` parked (flip park, every rank the same mutation)
    for a re-read of its whole context -- no needs-p file, no attempt spent,
    nothing to the client. The park re-queues it when the arena has room
    (``d_park_runtime.park_tick`` -> :func:`capacity_requeue_due`) or at the
    next wake's hold read (#248f: oldest first)."""
    from sglang.srt.weg2 import d_park_read, d_park_runtime, d_seats

    now = time.monotonic()
    if getattr(req, CAPPARK_SINCE_ATTR, None) is None:
        setattr(req, CAPPARK_SINCE_ATTR, now)
    setattr(req, CAPPARK_AT_ATTR, now)
    delivered = getattr(req, "_weg2_store_delivered", None)
    deliverable = getattr(req, "_weg2_store_deliverable", None)
    setattr(req, d_park_read.CAP_ATTR, None)
    d_park_read.clear_read_cycle(req)
    d_seats.mark_parked(req, d_seats.SITE_FLIP, epoch=None, now=now)
    parked = d_park_runtime.parked_list(sched)
    if not any(p is req for p in parked):
        parked.append(req)
    logger.warning(
        "%s rid=%s d_extent=%d X=%d delivered=%s deliverable=%s held_s=%.1f bound_s=%.0f "
        "attempts=%d -- the store holds the context and D's read of it ended short on its own "
        "capacity: D keeps the stream parked and re-reads it when the arena has room (no P leg, "
        "no attempt, no W50 to the client)",
        CAPPARK_MARK, str(req.rid)[:24], int(d_extent), int(x), delivered, deliverable,
        now - float(getattr(req, CAPPARK_SINCE_ATTR)), capacity_park_bound_s(),
        int(getattr(req, ATTEMPTS_ATTR, 0) or 0))


def clear_capacity_park(req) -> None:
    """#248h: the X gate admitted ``req`` -- its capacity park is over (a later
    one starts a fresh bound)."""
    if getattr(req, CAPPARK_SINCE_ATTR, None) is not None or getattr(req, CAPPARK_AT_ATTR, None) is not None:
        setattr(req, CAPPARK_SINCE_ATTR, None)
        setattr(req, CAPPARK_AT_ATTR, None)


def capacity_requeue_due(sched, parked, now: Optional[float] = None) -> list:
    """#248h: the capacity-parked requests of ``parked`` whose re-read may run
    now -- in park order, as far as the KV arena holds them beside this wake's
    reads still holding it (#248f accounting), at most one re-read per
    :data:`CAPPARK_REREAD_S`. Rank-local clock: the caller MIN-reduces."""
    from sglang.srt.weg2 import park_l3

    now = time.monotonic() if now is None else float(now)
    cands = [r for r in parked or () if getattr(r, CAPPARK_AT_ATTR, None) is not None]
    if not cands:
        return []
    cap = park_l3.arena_capacity(sched) if park_l3.arena_gate_on() else None
    in_flight = park_l3._in_flight_pages(sched, getattr(sched, "weg2_post_wake_settle", None) or ())
    out = []
    for r in cands:
        if now - float(getattr(r, CAPPARK_AT_ATTR)) < CAPPARK_REREAD_S:
            break
        need = park_l3.read_pages(sched, r)
        if cap is not None and in_flight > 0 and in_flight + need > cap:
            break
        in_flight += need
        out.append(r)
    return out


def keep_on_d(sched, req, d_extent: int, x: int, reason: str = "x_refusal_midstream") -> bool:
    """D, inside the W31 answer: keep ``req`` parked for P instead of the abort.
    Every rank runs it (the same queue mutation); only TP rank 0 writes the
    file. Returns whether the file was written (rank > 0: True); the caller
    keeps the request either way."""
    from sglang.srt.weg2 import d_park_read, d_park_runtime, d_seats

    same = _same_hold(sched, req)
    n = int(getattr(req, ATTEMPTS_ATTR, 0) or 0) + (0 if same else 1)
    setattr(req, ATTEMPTS_ATTR, n)
    setattr(req, WAKE_ATTR, int(getattr(sched, SCHED_WAKE_ATTR, 0) or 0))
    setattr(req, SINCE_ATTR, time.monotonic())
    # the read after P is the WHOLE context: the park's retained-span cap and
    # the previous read's stamp describe a cycle that is over
    setattr(req, d_park_read.CAP_ATTR, None)
    d_park_read.clear_read_cycle(req)  # W88 CYCLE: stamp, witness, bound
    # The decision is eligible() -- replicated on every rank. The file is TP0's
    # side effect only: a failed write never splits the ranks (the request
    # stays parked on all of them; park_tick's awake re-queue brings it back to
    # this gate, and MAX_ATTEMPTS ends it by name).
    # the Scheduler keeps its rank on ``ps`` (``self.tp_rank`` does not exist
    # there -- scheduler.py's own notes at the #10516/#18042 sites)
    rank0 = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0) == 0
    ok = bool(write_request(str(req.rid), context_ids(req), d_extent, x, reason,
                            attempt=n, extra_key=getattr(req, "extra_key", None))) if rank0 else True
    d_seats.mark_parked(req, d_seats.SITE_FLIP, epoch=None, now=time.monotonic())
    parked = d_park_runtime.parked_list(sched)
    if not any(p is req for p in parked):
        parked.append(req)
    logger.warning(
        "WEG2 W50-REROUTE rid=%s d_extent=%d X=%d reason=%s path=midstream n=%d "
        "attempt=%d/%d hold=%s written=%s -- D keeps the stream open and parks the request; P "
        "prefills its %d-token context, D resumes after the flip back (no abort, no bytes to the "
        "client)",
        str(req.rid)[:24], int(d_extent), int(x), reason, n, n, max_attempts(),
        "same" if same else "new", ok if rank0 else "rank>0", len(context_ids(req)))
    return ok
