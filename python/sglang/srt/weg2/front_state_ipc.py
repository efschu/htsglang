"""The front's writes into the boot's state directory (IPC §2.2, writer ``front``).

User law 2026-09-28 ~20:45Z: control and IPC never over log lines -- state.json,
events.jsonl, stop_request.json; logs are for humans only. Every write goes
through the ONE writer, ``weg2/state_file.py``; nothing here defines a second
format.

* ``publish_front_stop`` (W98-STOP-FINAL, z30u 07:14:10): a front STOP is a
  named, controlled teardown. It lands as the event ``front_stop`` and -- while
  the boot is live -- as the A5 stop request (``stop_request.json``, origin
  ``front``). The host writer turns that request into ``dead`` with this cause
  and ``cause.rc = 24`` at the boot's end (``state_file finish``); the Phase-2
  healthcheck (``state_file health``) reads it as dead at once. The lifecycle
  itself is NOT written from here: ``stopping``/``dead`` are host states, and
  both make the arms' group watchers stand down (27B ``gwatch`` returns on
  them BEFORE its ``docker stop``), so a front-written lifecycle would turn a
  torn-down boot into one that holds until its window ends.
* ``publish_flip_cushion``: one ``flip_cushion`` event per completed flip --
  the host cushion's course without waiting for a W98.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from sglang.srt.weg2 import state_file

#: lifecycle states in which a front STOP writes NO stop request: the end is
#: already running (stopping) or done (terminal) -- the first cause wins.
NO_STOP_REQUEST = ("stopping",) + tuple(state_file.TERMINAL)


def stop_cause_code(name: str) -> str:
    """``"W98 Weg2HostRateLatched"`` -> ``"W98_Weg2HostRateLatched"``: W-number
    plus name, never the bare W-number (IPC §2.2 cause.code, 27B requirement 6)."""
    return re.sub(r"\s+", "_", str(name).strip()) or "FRONT_STOP"


def publish_front_stop(d: str, *, name: str, detail: str, state_before: Optional[str],
                       epoch: Optional[int], awake: Optional[str]) -> dict:
    """Event ``front_stop`` always; ``stop_request.json`` while the boot is live
    and no earlier stop request stands (a watcher's or the deadman's cause is
    never overwritten). Returns what was written, for the human line.

    ``StateFileError`` is a ``SystemExit`` (the CLI's exit path) -- it is turned
    into a named result here, so a foreign schema can never end the front."""
    try:
        return _publish_front_stop(d, name=name, detail=detail, state_before=state_before,
                                   epoch=epoch, awake=awake)
    except state_file.StateFileError as e:
        return {"written": False, "stop_request": False, "lifecycle": None,
                "why": f"StateFileError: {e}"}


def _publish_front_stop(d: str, *, name: str, detail: str, state_before: Optional[str],
                        epoch: Optional[int], awake: Optional[str]) -> dict:
    st = state_file.read(d)
    if not st:
        return {"written": False, "stop_request": False, "lifecycle": None,
                "why": f"{d}/state.json missing -- the host writer creates it"}
    code = stop_cause_code(name)
    cur = (st.get("lifecycle") or {}).get("state")
    state_file.add_event(
        d, "front_stop",
        {"name": str(name), "detail_full": str(detail), "front_state_before": state_before,
         "epoch": epoch, "awake": awake, "lifecycle": cur},
        writer="front", code=code)
    path = os.path.join(d, "stop_request.json")
    req = False
    if cur not in NO_STOP_REQUEST and not os.path.exists(path):
        state_file.write_json_atomic(path, {
            "code": code, "origin": "front", "group": None, "rank": None,
            "detail_full": f"{name}: {detail}"})
        req = True
    return {"written": True, "stop_request": req, "lifecycle": cur, "code": code}


def publish_flip_cushion(d: str, rec: dict) -> bool:
    """One ``flip_cushion`` event (writer front). False = no state.json."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return False
    try:
        state_file.add_event(d, "flip_cushion", dict(rec), writer="front")
    except state_file.StateFileError:
        return False
    return True


# ---------------- DASHBOARD-AUS-IPC (a)/(b), 29.09. ----------------
# User order (via 27B): the dashboard is fed from the IPC, not from log lines.
# The front writes the flip as events and its own keys under state.json
# `front` -- through state_file (writer front), never by a direct open, and
# never a lifecycle field (those are the host's and the launcher's).

#: the host mirrors these five keys of `front` from /weg2/state (state_file beat);
#: the front never writes them, so every key has one writer.
HOST_FRONT_KEYS = ("state", "epoch", "awake", "queue", "outstanding")
#: kept out of a flip_done event: the chunk list (a line over state_file.EVENT_MAX
#: would be cut into invalid JSON).
FLIP_DONE_DROP = ("chunks",)


def publish_event(d: str, typ: str, data: dict) -> bool:
    """One event of the front into the boot's events.jsonl (writer front).
    False = no state.json (no state dir, or the host has not created it)."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return False
    try:
        state_file.add_event(d, typ, dict(data), writer="front")
    except state_file.StateFileError:
        return False
    return True


def publish_front_fields(d: str, fields: dict) -> bool:
    """The front's own keys under state.json `front`, dotted, so the host's mirror
    keys (:data:`HOST_FRONT_KEYS`) stay the host's. A host key here is a ValueError."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return False
    bad = [k for k in fields if k in HOST_FRONT_KEYS]
    if bad:
        raise ValueError(f"front_state_ipc: {bad} are the host's mirror keys of `front`")
    try:
        state_file.transition(d, None, fields={f"front.{k}": v for k, v in fields.items()},
                              writer="front")
    except state_file.StateFileError:
        return False
    return True


def flip_done_payload(rec: dict, flip_begin_ts: float) -> dict:
    """The ``flip_done`` event: the front's flip_log record (drain/sleep/wake/flip_ms,
    legs, overlap, critical path) without the chunk list, plus the begin stamp."""
    out = {k: v for k, v in rec.items() if k not in FLIP_DONE_DROP}
    out["chunks_n"] = len(rec.get("chunks") or ())
    out["flip_begin_ts"] = round(float(flip_begin_ts), 3)
    return out


#: A14: the group-health verdicts on whose CHANGE the front looks for rank stops.
RANK_STOP_VERDICTS = ("failing", "held", "dead")


def rank_stops_of(rankstate_dir: str, group: str) -> List[Tuple[str, dict]]:
    """``(rank_key, stop)`` for every stop the group's ranks recorded in their
    rankstats files (weg2/rankstats.py ``stops.last``). A file of another
    schema or a torn one is skipped, never raised."""
    from sglang.srt.weg2 import rankstats

    out: List[Tuple[str, dict]] = []
    for p in sorted(glob.glob(os.path.join(rankstate_dir, f"{group}.*{rankstats.SUFFIX}"))):
        try:
            with open(p) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("schema") != rankstats.SCHEMA:
            continue
        rk = f"tp{rec.get('tp_rank')}pp{rec.get('pp_rank')}"
        for s in ((rec.get("stops") or {}).get("last") or ()):
            if isinstance(s, dict):
                out.append((rk, dict(s, pid=rec.get("pid"))))
    return out


def publish_rank_stops(d: str, group: str, seen: set) -> int:
    """A14: every stop of ``group``'s ranks not yet published -> one event
    ``rank_stop`` (writer front; envelope group/rank/code, data t/reason/code/
    exc/ticket/text/pid). The rank directory is the launcher's
    ``groups.<G>.rankstate_dir``. ``seen`` belongs to the caller's ONE IPC
    thread. Returns the number published; no state dir = 0."""
    if not d or not os.path.exists(os.path.join(d, "state.json")):
        return 0
    try:
        st = state_file.read(d)
    except state_file.StateFileError:
        return 0
    rs_dir = ((st.get("groups") or {}).get(group) or {}).get("rankstate_dir")
    if not rs_dir:
        return 0
    n = 0
    for rk, s in rank_stops_of(rs_dir, group):
        key = (group, rk, s.get("t"), s.get("code"))
        if key in seen:
            continue
        try:
            state_file.add_event(d, "rank_stop", dict(s, group=group, rank=rk),
                                 writer="front", group=group, rank=rk, code=s.get("code"))
        except state_file.StateFileError:
            return n
        seen.add(key)
        n += 1
    return n


def group_health_verdict(http_ok: bool, alive: bool, streak: int, held: bool) -> str:
    """One word per FH poll (front_health GroupFacts): dead (process gone),
    held (#1223 DEBUG-HOLD), failing (a counted /health failure), busy (a slow
    /health while the group computes, FP beacon: not counted), ok."""
    if not alive:
        return "dead"
    if held:
        return "held"
    if int(streak or 0) > 0:
        return "failing"
    return "ok" if http_ok else "busy"


#: the front's IPC queue bound (27B review of b02cec1ad5): a writer blocked on the
#: disk or on the state lock must never make the front pile up anon RAM.
IPC_QUEUE_MAX = 1024


class BoundedWriter:
    """ONE writer thread over a BOUNDED FIFO. On overflow the OLDEST entry is
    dropped and counted (``dropped``) -- a newer record supersedes it for the
    dashboard, and the front never waits here. A failed write is counted
    (``failed``) and handed to ``on_error``, never raised into the front."""

    def __init__(self, maxlen: int = IPC_QUEUE_MAX, name: str = "weg2-ipc", on_error=None) -> None:
        import collections
        import threading

        self.maxlen = int(maxlen)
        self.dropped = 0
        self.failed = 0
        self._q = collections.deque()
        self._cv = threading.Condition()
        self._on_error = on_error
        self._t = threading.Thread(target=self._run, name=name, daemon=True)
        self._t.start()

    def depth(self) -> int:
        with self._cv:
            return len(self._q)

    def submit(self, fn, *args) -> None:
        with self._cv:
            if len(self._q) >= self.maxlen:
                self._q.popleft()
                self.dropped += 1
            self._q.append((fn, args))
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q:
                    self._cv.wait()
                fn, args = self._q.popleft()
            try:
                fn(*args)
            except Exception as e:  # noqa: BLE001 -- a lost record never stops the front
                self.failed += 1
                if self._on_error is not None:
                    self._on_error(fn, e)


class OutstandingBook:
    """(Ported 1:1 from the NF front, desk/nf-y6d-anchor-pin-1001; the 27B front
    uses its arrival stamp for weg2_ttft_seconds, TSDB 01.10.)

    Every open request of the front: its arrival and its last token
    (state.json ``front.oldest_outstanding_*`` / ``front.outstanding_stalest``).

    y4y 17:05:40Z: two burst requests of the GROW probe ran 300 s without a
    single token into their timeout while an anchor stream beside them was
    served -- the progress watcher reads only the served TOTALS, so it stayed
    silent (Nutzer 30.09.: "outstanding 3, niemand merkts"). One row per
    request answers it: how old, where, and how long since its last token.

    Data the front already has, in the event loop, no sync: the arrival at the
    rid's birth, a token = a chunk D streamed for the rid (a non-streamed
    request shows its first token when it ends), the where from the front's
    own structures (queue / P / D outstanding / parked, ``flip`` while a flip
    runs). The rid leaves at the handler's end (the one rid-end site)."""

    #: a row no structure names and whose handler end never came (a test
    #: double calling the handler unwrapped) is dropped after this long
    ORPHAN_S = 3600.0

    def __init__(self) -> None:
        self.arrival: dict = {}
        self.last_tok: dict = {}
        #: y5c (30.09., weg2-0-2, 113k non-stream, 235 s): the rids whose client
        #: asked no stream -- the front forwards D's answer whole at its end,
        #: so it sees no token in between and has no per-rid IPC source for
        #: one. Their rows say so (``stream`` 0, ``no_token_s`` None) instead of
        #: reading "no token for N s" out of a blindness.
        self.nonstream: set = set()

    def arrive(self, rid, now: float) -> None:
        self.arrival.setdefault(str(rid), float(now))

    def stream(self, rid, is_stream: bool) -> None:
        if is_stream:
            self.nonstream.discard(str(rid))
        else:
            self.nonstream.add(str(rid))

    def token(self, rid, now: float) -> None:
        self.last_tok[str(rid)] = float(now)

    def end(self, rid) -> None:
        self.arrival.pop(str(rid), None)
        self.last_tok.pop(str(rid), None)
        self.nonstream.discard(str(rid))

    def block(self, now: float, queued, p_out, d_out, parked, flipping: bool, top: int = 8) -> dict:
        """``queued``: (rid, t_arrive) of the front's queue; ``p_out``/``d_out``:
        the groups' outstanding maps (rid -> leg start); ``parked``: D's parked
        rids. A booked rid in none of them is between two (routing, a seat
        wait): ``queue``."""
        rows = {}
        for rid in list(self.arrival):
            rows[rid] = ["queue", self.arrival[rid]]
        for rid, t in queued:
            r = str(rid)
            rows[r] = ["flip" if flipping else "queue", self.arrival.get(r, float(t))]
        for rid, t in list((p_out or {}).items()):
            r = str(rid)
            rows[r] = ["P", self.arrival.get(r, float(t))]
        park = {str(x) for x in (parked or ())}
        for rid, t in list((d_out or {}).items()):
            r = str(rid)
            rows[r] = ["parked" if r in park else "D", self.arrival.get(r, float(t))]
        placed = {str(r) for r, _t in queued} | {str(r) for r in (p_out or {})} | {str(r) for r in (d_out or {})}
        for rid in [r for r in rows if r not in placed and now - rows[r][1] > self.ORPHAN_S]:
            rows.pop(rid)
            self.end(rid)
        entries = []
        for rid, (where, t) in rows.items():
            lt = self.last_tok.get(rid)
            blind = rid in self.nonstream
            entries.append({"rid": rid, "where": where, "age_s": round(max(0.0, now - t), 1),
                            "stream": 0 if blind else 1,
                            "last_token_s": None if lt is None else round(max(0.0, now - lt), 1),
                            # non-stream: the front cannot see D's tokens -- None, not a stall
                            "no_token_s": (None if blind else
                                           round(max(0.0, now - (lt if lt is not None else t)), 1))})
        oldest = max(entries, key=lambda e: e["age_s"]) if entries else None
        return {
            "outstanding_n": len(entries),
            "outstanding_nonstream_n": sum(1 for e in entries if not e["stream"]),
            "oldest_outstanding_age_s": oldest["age_s"] if oldest else None,
            "oldest_outstanding_first_token_s": oldest["last_token_s"] if oldest else None,
            "oldest_outstanding_rid": oldest["rid"] if oldest else None,
            "oldest_outstanding_where": oldest["where"] if oldest else None,
            "oldest_outstanding_stream": oldest["stream"] if oldest else None,
            "outstanding_stalest": sorted(
                entries, key=lambda e: -(e["no_token_s"] if e["no_token_s"] is not None else -1.0)
            )[:max(1, int(top))],
        }


class DpFlipClock:
    """D->P flip time in the USER's definition (FLIPZEIT-VERLAUF-0929.md, Folgepunkt
    30.09.): Decode-Ende -> P-Prefill-Start, i.e. the last D decode round (the park
    RPC's send when D's decodes were parked for this flip, else D's last served
    leg 2), no earlier than the arrival of the oldest waiter -> the start of P's
    first prefill (leg 1's end minus P's own prefill time ``weg2_prefill_s``; the
    leg-1 dispatch when P's body does not carry it -- named as the source).

    ``flip_first_work`` stays as it is (flip begin -> first dispatch); this is
    ONE ``flip_user_time`` event per D->P flip that reached a first leg 1, with
    the parts: ``park_rpc_ms`` (the park RPC before the begin), ``pre_begin_ms``
    (decode end -> flip begin), ``legs_ms`` (begin -> done), ``first_chunk_ms``
    (done -> P prefill start). One clock: time.time() of the front."""

    def __init__(self) -> None:
        self._park: Optional[dict] = None
        self._last_d_served: Optional[float] = None
        self._armed: Optional[dict] = None

    def note_park(self, epoch: int, t_sent: Optional[float], rpc_ms: Optional[float]) -> None:
        self._park = {"epoch": int(epoch), "t_sent": t_sent, "rpc_ms": rpc_ms}

    def note_d_served(self, now: float) -> None:
        self._last_d_served = float(now)

    def begin(self, epoch_before: int, flip_begin_ts: float, oldest_waiter_ts: Optional[float]) -> None:
        """A D->P flip begins (``epoch_before`` = the D phase that ends)."""
        park = self._park if (self._park is not None and self._park["epoch"] == int(epoch_before)) else None
        if park is not None and park.get("t_sent") is not None:
            end, src = float(park["t_sent"]), "park_rpc_sent"
        elif self._last_d_served is not None and self._last_d_served <= flip_begin_ts:
            end, src = self._last_d_served, "last_d_served"
        else:
            end, src = float(flip_begin_ts), "flip_begin"
        if oldest_waiter_ts is not None and float(oldest_waiter_ts) > end:
            end, src = float(oldest_waiter_ts), "oldest_waiter_arrival"
        # IDLE-FLIP (z30y14 epoch 5, 37.4 s): a flip with no waiter at its begin and
        # no park before it is the idle-layout swap -- nobody waits for it, the first
        # request arrives later. Its user clock then starts at that request's dispatch, never at
        # the decode end, else the idle gap reads as flip time.
        self._armed = {"epoch": int(epoch_before) + 1, "start_ts": end, "start_source": src,
                       "idle_flip": oldest_waiter_ts is None and park is None,
                       "flip_begin_ts": float(flip_begin_ts),
                       "park_rpc_ms": (None if park is None else park.get("rpc_ms")), "done_ts": None}
        self._park = None

    def done(self, now: float, beacon_snap: Optional[Dict[int, Tuple[int, int, int]]] = None) -> None:
        """``beacon_snap`` (PDFLIP-E): P's progress beacons at the flip's done,
        ``{pid: (forward_ct, t_start_ns, t_done_ns)}``; None = no beacon reading."""
        if self._armed is not None and self._armed["done_ts"] is None:
            self._armed["done_ts"] = float(now)
            # PDFLIP-E3 (N5f first D->P 13:30:08): P had never run a forward since
            # the boot, so it had NO beacon file at done -- an empty reading is a
            # baseline of 0 (every file that appears is that rank's first forward),
            # not "no beacon". None = beacon off / unreadable.
            self._armed["beacon_snap"] = None if beacon_snap is None else dict(beacon_snap)
            self._armed["beacon_first"] = {}
            # Y8P-PPFWD-ARM-RACE (NF y8p 03.10. 08:52:05 / 08:53:41 / 08:57:00; port 490): P's
            # first forward can start BEFORE the front logs ``done`` (the wake RPC returns, P
            # admits at once; 0-70 ms ahead). It is then already inside the done snapshot, the
            # rise this clock waits for is the SECOND chunk's start -- a false Nachlauf of one
            # whole chunk (2.5-2.9 s on NF). P sleeps for the whole D phase, so a rank whose
            # ``t_start`` lies at or after the flip's begin ran its first forward of the woken
            # group: count it from the snapshot itself. Never raises.
            try:
                fb = float(self._armed.get("flip_begin_ts") or 0.0)
                snap = self._armed["beacon_snap"]
                if fb > 0.0 and snap:
                    for pid, row in snap.items():
                        if int(row[0]) > 0 and float(row[1]) / 1e9 >= fb:
                            self._armed["beacon_first"][pid] = (float(row[1]) / 1e9, False)
                    if self._armed["beacon_first"]:
                        self.note_beacon(snap)       # derives pp_first_ts / pp_last_ts from them
            except Exception:  # noqa: BLE001 -- an instrument never breaks a flip
                pass

    def waits_for_beacon(self) -> bool:
        """Until every P rank rose (the end is known at the first; the last
        stage's start completes the decomposition)."""
        a = self._armed
        if a is None or a.get("beacon_snap") is None:
            return False
        if not a["beacon_snap"]:                    # empty baseline: the end is all we can know
            return a.get("pp_first_ts") is None
        return a.get("pp_last_ts") is None

    def first_forward_ts(self) -> Optional[float]:
        """PDFLIP-E2: the D->P end -- the FIRST P rank's first forward after the
        flip (PP0 starts the prefill batch; the followers' starts are pipeline
        fill, i.e. prefill compute). None until one rank rose."""
        a = self._armed
        return None if a is None else a.get("pp_first_ts")

    def note_beacon(self, cur: Dict[int, Tuple[int, int, int]]) -> Optional[float]:
        """PDFLIP-E2 (user: D->P ends at the "PREFILL BATCH BEGINN"; NF y7l: the
        span to the LAST stage's first forward is pipeline fill = prefill
        compute, PP0 chunk 1 3.0-3.5 s, then PP1, then PP2): fold one beacon
        reading in. A rank whose forward_ct rose past its done snapshot began
        its first forward after the flip at ``t_start`` (exact while the rise is
        1; a wider rise between two readings is marked approximate). The
        EARLIEST such start = PP0's first forward = the D->P end
        (``pp_first_ts``); once every rank rose, the latest = ``pp_last_ts``,
        kept for the decomposition only. Returns ``pp_first_ts`` once known."""
        a = self._armed
        if a is None or a.get("beacon_snap") is None:
            return None
        if a.get("pp_last_ts") is not None:
            return a.get("pp_first_ts")
        first = a.setdefault("beacon_first", {})
        snap = a["beacon_snap"]
        # PDFLIP-E3: a rank absent from the done reading has forward_ct 0 there
        for pid in set(snap) | (set(cur) if not snap else set()):
            ct0 = int(snap.get(pid, (0, 0, 0))[0])
            if pid in first:
                continue
            row = cur.get(pid)
            if row is None or int(row[0]) <= ct0:
                continue
            first[pid] = (float(row[1]) / 1e9, int(row[0]) - ct0 > 1)
        if first:
            t0, ap0 = min(first.values(), key=lambda x: x[0])
            if a.get("pp_first_ts") is None or t0 < a["pp_first_ts"]:
                a["pp_first_ts"], a["pp_first_approx"] = t0, ap0
        if snap and len(first) == len(snap):
            a["pp_last_ts"] = max(t for t, _ in first.values())
        return a.get("pp_first_ts")

    def first_prefill(self, rid: Optional[str], t_dispatch: float, t_end: float,
                      p_prefill_s: Optional[float]) -> Optional[dict]:
        """The first leg 1 after the flip finished: the event, or None.

        PDFLIP-E2: P's prefill start is the FIRST P rank's (PP0's) first forward
        (``note_beacon``) -- ``prefill_start_source="pp_first_forward"``; the
        last stage's start rides as ``pp_last_start_ts`` (pipeline fill counts
        as prefill, not as flip). Without
        that reading the end is MISSING (``prefill_start_ts`` None, no
        ``flip_user_ms``): never the leg-1 dispatch (user: "DIE FALSCHE, ZU
        KLEINE ZAHL MUSS UEBERALL WEG")."""
        a = self._armed
        if a is None:
            return None
        self._armed = None
        if a.get("pp_first_ts") is not None:
            start = max(float(a["pp_first_ts"]), float(a["done_ts"] or a["pp_first_ts"]))
            src = "pp_first_forward" + ("_approx" if a.get("pp_first_approx") else "")
        else:
            start, src = None, "missing"
        pp_last = a.get("pp_last_ts")
        done = a["done_ts"] if a["done_ts"] is not None else start
        idle = bool(a.get("idle_flip"))
        pre_begin_start = a["start_ts"]
        if idle and float(t_dispatch) > a["start_ts"]:
            a["start_ts"], a["start_source"] = float(t_dispatch), "first_dispatch_after_idle_flip"
            pre_begin_start = None  # the user arrived after the flip: no pre-begin wait

        def ms(x, y):
            return None if x is None or y is None else round((float(y) - float(x)) * 1000.0)
        return {"epoch": a["epoch"], "dir": "D>P", "rid": rid, "idle_flip": idle,
                "start_ts": round(a["start_ts"], 3), "start_source": a["start_source"],
                "prefill_start_ts": None if start is None else round(start, 3),
                "prefill_start_source": src,
                "leg1_dispatch_ts": round(float(t_dispatch), 3),
                "pp_last_start_ts": None if pp_last is None else round(float(pp_last), 3),
                "flip_user_ms": ms(a["start_ts"], start),
                "parts": {"park_rpc_ms": (None if a["park_rpc_ms"] is None else round(float(a["park_rpc_ms"]))),
                          "pre_begin_ms": ms(pre_begin_start, a["flip_begin_ts"]),
                          "legs_ms": ms(a["flip_begin_ts"], done),
                          "first_chunk_ms": ms(done, start)},
                "definition": "Decode-Ende -> P-Prefill-Start", "clock": "time.time front"}


class FirstWorkClock:
    """Flip time from ONE clock (time.time() of the front's own process): the flip's
    begin stamp and the woken group's first work are both taken here.

    P->D: the first content D streams after the flip = the first decode token (the
    user's flip time, P end -> first decode token; a P->D flip begins at P's end).
    D->P: the first leg 1 the front dispatches to P after the flip. Armed at
    ``WEG2-FLIP done``, fired once; the next flip re-arms it, so a flip whose woken
    group never worked before the next one simply never fires. No model switch:
    the 27B flip-time tile reads the same event."""

    #: DASHBOARD-AUS-IPC (30.09., Inventar FEHLT 3; 27B 11 flip_done / 3
    #: flip_first_work, NF 32 / 30): EVERY flip that reached ``done`` gets
    #: exactly one ``flip_first_work`` -- its first work, or ``what: "none"``
    #: with the time to the flip's end and the reason no work came (the next
    #: flip began first, or the front stopped). A ``none`` event has NO flip
    #: time (user 29.09.: P end -> first decode token / decode end -> first
    #: prefill, never flip_total): ``flip_time_ms`` null, the begin -> done span
    #: only as ``flip_total_ms`` -- the dashboard's flip-time tile and history
    #: marks take every non-null ``flip_time_ms``.
    NONE = "none"

    # DASHBOARD-IPC 01.10. (NF + 27B, 3-12 per boot): a P->D ``decode_token``
    # fired 0.02-0.99 s after ``flip_begin`` -- long before D was awake --
    # because ANY chunk of ANY open D stream counted, e.g. a wait-bound-parked
    # stream of an earlier D phase (NF 01.10. 05:27:47.638 epoch 26:
    # weg2-18-137, dispatched in epoch 18, fired 153 ms after the begin; done
    # came at +2.2 s). D content before ``done`` now counts only for a leg 2
    # dispatched at or after this flip's begin -- the hand-off, which the
    # dormant admit sends DURING the flip (05:08:13: weg2-3-10 D-ADMIT 64 ms
    # after the begin, first token 2.9 s after P's end; a rule "dispatched
    # after done" would drop exactly the flip it defines). From ``done`` on,
    # D is awake and any content is its work.

    def __init__(self) -> None:
        self._armed: Optional[dict] = None
        #: the end of P's last leg 1 and the last flip's done (front clock)
        self._p_end: Optional[float] = None
        self._last_done: float = float("-inf")
        #: D chunks refused as a flip's first work (stale, before done), boot total
        self.stale_skipped = 0

    def waits_for(self, group: str) -> bool:
        """An armed flip whose woken group is ``group`` (cheap: per D chunk)."""
        a = self._armed
        return a is not None and a["wake"] == group

    def note_p_end(self, now: float) -> None:
        """P served a leg 1 (its end is P's end when it was the phase's last)."""
        self._p_end = float(now)
        a = self._armed
        if a is not None and a["wake"] == "D" and a.get("done_ts") is None:
            a["p_end_ts"] = max(float(a.get("p_end_ts") or now), float(now))

    def arm(self, epoch: int, sleep: str, wake: str, flip_begin_ts: float) -> Optional[dict]:
        """Arm the new flip; returns the ``none`` event of the previous flip when
        it reached ``done`` and its woken group never worked (publish it)."""
        prev = self.flush("next_flip_before_work")
        self._armed = {"epoch": int(epoch), "dir": f"{sleep}>{wake}", "wake": wake,
                       "flip_begin_ts": float(flip_begin_ts)}
        if wake == "D" and self._p_end is not None and self._p_end >= self._last_done:
            # P's last leg 1 in the P phase this flip ends
            self._armed["p_end_ts"] = self._p_end
        return prev

    #: PDFLIP-E (NF rule, 02.10.): D cannot emit before its layers are back; a
    #: D content up to this long before the kv wake answered still counts
    AWAKE_SLACK_S = 0.3

    def note_awake(self, ts: float) -> None:
        """PDFLIP-E: the woken D's kv wake answered (its layers are back) --
        from here D content before ``done`` is this flip's first decode token
        (NF y7l: D decodes during wake-kv/dc, before done)."""
        a = self._armed
        if a is not None and a["wake"] == "D":
            a["awake_ts"] = float(ts)

    def done(self, now: float) -> None:
        """The armed flip reached ``done`` (its ``flip_done`` was published)."""
        self._last_done = float(now)
        if self._armed is not None:
            self._armed["done_ts"] = float(now)

    def flush(self, reason: str) -> Optional[dict]:
        """The armed flip's ``none`` event (only for a flip that reached ``done``;
        a flip that never finished has no ``flip_done`` to pair), disarmed."""
        a = self._armed
        self._armed = None
        if a is None or a.get("done_ts") is None:
            return None
        return {"epoch": a["epoch"], "dir": a["dir"], "flip_begin_ts": round(a["flip_begin_ts"], 3),
                "first_work_ts": None, "flip_time_ms": None,
                "flip_total_ms": round((a["done_ts"] - a["flip_begin_ts"]) * 1000.0),
                "what": self.NONE, "reason": reason, "rid": None, "clock": "time.time front"}

    def seen(self, group: str, what: str, rid: Optional[str], now: float,
             leg2_dispatch_ts: Optional[float] = None) -> Optional[dict]:
        """The woken group worked: the event, or None. For D, ``leg2_dispatch_ts``
        is the rid's leg-2 dispatch (front clock; None = unknown): content
        before ``done`` counts only for a leg 2 dispatched in this flip."""
        a = self._armed
        if a is None or a["wake"] != group:
            return None
        _awake = a.get("awake_ts")
        if group == "D" and a.get("done_ts") is None and (
                # PDFLIP-E (NF rule): with D's kv wake known, its content counts from the
                # wake on (minus a slack) whatever leg 2 it belongs to, never before;
                (float(now) < float(_awake) - self.AWAKE_SLACK_S) if _awake is not None
                # without it, the 01.10. rule: only a leg 2 dispatched in this flip
                else (leg2_dispatch_ts is None or float(leg2_dispatch_ts) < a["flip_begin_ts"])):
            a["stale_skipped"] = int(a.get("stale_skipped", 0)) + 1
            self.stale_skipped += 1
            return None
        self._armed = None
        ev = {"epoch": a["epoch"], "dir": a["dir"], "flip_begin_ts": round(a["flip_begin_ts"], 3),
              "first_work_ts": round(float(now), 3),
              "flip_time_ms": round((float(now) - a["flip_begin_ts"]) * 1000.0),
              "what": what, "rid": rid, "clock": "time.time front",
              "done_ts": None if a.get("done_ts") is None else round(a["done_ts"], 3),
              "before_done": a.get("done_ts") is None,
              "stale_skipped": int(a.get("stale_skipped", 0))}
        if group == "D":
            # the user's P->D flip time (29.09.): P end -> first decode token
            p_end = a.get("p_end_ts")
            start = float(p_end) if p_end is not None else a["flip_begin_ts"]
            ev.update({"leg2_dispatch_ts": (None if leg2_dispatch_ts is None
                                            else round(float(leg2_dispatch_ts), 3)),
                       "p_end_ts": None if p_end is None else round(float(p_end), 3),
                       "p_end_source": "p_leg1_end" if p_end is not None else "flip_begin",
                       "flip_user_ms": round((float(now) - start) * 1000.0)})
        return ev
