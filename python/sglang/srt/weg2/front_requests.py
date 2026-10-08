"""DASHBOARD-IPC 01.10. (dashboard redesign, user 01.10. ~07:20Z/~07:40Z): the
front's per-request and per-flip facts as IPC records -- never a log line.

User wishes this module serves (rigdash reads each field as it appears in
state.json / events.jsonl / VictoriaMetrics):

* the current phase live -- ``front.flip`` (:class:`FlipPhase`) and
  ``front.d_activity`` (:meth:`RequestBook.activity_block`);
* the flip time in the USER's definition (P end -> first decode token,
  decode end -> first prefill), split into Vorlauf / Layer / Nachlauf;
* TTFT per route (``via``: after_p / d_direct / d_single) with its
  decomposition (``front.ttft_by_via``);
* a session view -- one ``request_done`` event per finished request and one
  ``park`` event per park episode (events.jsonl), optionally the Influx point
  ``weg2_req`` (TSDB-DELTA-27B-1001.md 1c).

Shape rules (operator 01.10.): nothing here waits, locks or syncs. Every
method is a handful of dict operations on the event loop; the records are
handed to the front's ONE BoundedWriter thread for the disk. Every table is
bounded (oldest out). One clock: ``time.time()`` of the front's process.

Thread note: the live writer (BoundedWriter thread) reads :attr:`FlipPhase.snap`
and :attr:`RequestBook.activity` -- both are REPLACED (never mutated) on the
loop, so a reader always holds one consistent snapshot.
"""
from __future__ import annotations

import collections
import math
from typing import Any, Dict, Iterable, Optional, Tuple

#: the routes a request reaches its first token by (front.py LEG2-FIRST-CONTENT)
VIAS = ("after_p", "d_direct", "d_single")
#: the TTFT decomposition, contiguous: queue + p_prefill + flip_wait +
#: d_first_token + other == ttft (other = a re-route's extra legs, else 0)
TTFT_PARTS = ("queue_ms", "p_prefill_ms", "flip_wait_ms", "d_first_token_ms", "other_ms")
#: front.flip phases
PHASES = ("vorlauf", "layer", "nachlauf")
#: front.d_activity values
D_ACTIVITY = ("decode", "prefill", "idle")
#: an events.jsonl line is cut at state_file.EVENT_MAX (4000 bytes): a
#: ``request_done`` record has a fixed shape (no list grows with the request),
#: ~1.2 kB -- the test pins it below EVENT_MAX - 500


def _ms(a: Optional[float], b: Optional[float]) -> Optional[int]:
    if a is None or b is None:
        return None
    return int(round((float(b) - float(a)) * 1000.0))


def _r3(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), 3)


#: FlipPhase.first_work: "the work began now"
_NOW = object()


class FlipPhase:
    """``front.flip`` = {phase, dir, since_ts, begin_ts, reason, ...}.

    * ``vorlauf``: from the flip decision (the D->P park RPC, else the entry
      into ``flip()``) to ``WEG2-FLIP begin``;
    * ``layer``: begin -> ``flip_done`` (the layer swap);
    * ``nachlauf``: done -> the woken group's first work (P->D: the first
      decode token; D->P: the begin of the first forward on the last P
      pipeline stage after ``done``, 02.10.), the ``flip_first_work`` event;
    * ``null``: no flip open. ``last`` keeps the previous flip's three spans.

    A first work that comes before ``done`` (the D wake leg ended before the
    P sleep leg) closes the flip at ``done`` with a zero Nachlauf."""

    def __init__(self) -> None:
        self.snap: Dict[str, Any] = {"phase": None, "dir": None, "since_ts": None, "begin_ts": None,
                                     "reason": None, "decision_ts": None, "done_ts": None, "last": None}
        #: USAGE-DETAILS (02.10.): every flip's window on the loop -- {dir,
        #: start (decision), begin, done (None = open), aborted}; bounded.
        #: Read by the front's per-request usage details only.
        self.hist: "collections.deque[Dict[str, Any]]" = collections.deque(maxlen=256)

    def _hist_close(self, now: float, aborted: bool) -> None:
        if self.hist and self.hist[-1]["done"] is None:
            self.hist[-1]["done"] = float(now)
            self.hist[-1]["aborted"] = bool(aborted)

    def windows(self, since: float, now: float):
        """[(start, end)] of the flips whose window reaches ``since`` or later
        (an open one ends ``now``)."""
        out = []
        for f in self.hist:
            end = f["done"] if f["done"] is not None else now
            if end >= since:
                out.append((f["start"], end))
        return out

    def begun(self, since: float, until: float) -> int:
        """Flips (not aborted) whose begin lies in ``[since, until]``."""
        return sum(1 for f in self.hist if not f["aborted"] and since <= f["begin"] <= until)

    def _set(self, **kw: Any) -> None:
        s = dict(self.snap)
        s.update(kw)
        self.snap = s  # replaced, never mutated (the live writer reads it off-loop)

    def vorlauf(self, direction: str, reason: str, now: float) -> bool:
        """The flip is decided. A flip already open keeps its own decision."""
        if self.snap["phase"] in ("vorlauf", "layer"):
            return False
        self._set(phase="vorlauf", dir=direction, since_ts=_r3(now), decision_ts=_r3(now),
                  begin_ts=None, done_ts=None, reason=reason, first_work_ts=None, first_work_seen=False)
        return True

    def layer(self, direction: str, now: float, reason: str) -> None:
        s = self.snap
        same = s["phase"] == "vorlauf" and s["dir"] == direction
        self._set(phase="layer", dir=direction, since_ts=_r3(now), begin_ts=_r3(now),
                  decision_ts=s["decision_ts"] if same else _r3(now),
                  reason=(s["reason"] if same and s["reason"] else reason),
                  done_ts=None, first_work_ts=None, first_work_seen=False)
        self._hist_close(now, aborted=True)  # a flip never closed ends at the next one
        start = s["decision_ts"] if same and s["decision_ts"] is not None else now
        self.hist.append({"dir": direction, "start": float(min(start, now)), "begin": float(now),
                          "done": None, "aborted": False})

    def done(self, now: float) -> None:
        self._hist_close(now, aborted=False)
        if self.snap["phase"] != "layer":
            return
        if self.snap.get("first_work_seen") or self.snap.get("first_work_ts") is not None:
            self._close(self.snap.get("first_work_ts"), now, now)
            return
        self._set(phase="nachlauf", since_ts=_r3(now), done_ts=_r3(now))

    def first_work(self, now: float, what: Optional[str] = None, at: Any = _NOW) -> None:
        """The woken group worked. ``at`` = when the work began, if not ``now``
        (D->P, 02.10.: the begin of the first forward on the last P pipeline
        stage); ``at=None`` = that reading is missing -- the flip closes with
        no Nachlauf value, never with the time this call came."""
        fw = now if at is _NOW else at
        ph = self.snap["phase"]
        if ph == "layer":
            self._set(first_work_ts=_r3(fw), first_work_seen=True, first_work_what=what)
        elif ph == "nachlauf":
            self._set(first_work_what=what)
            self._close(fw, self.snap["done_ts"], now)

    def abort(self, now: float, why: str) -> None:
        """A flip refused / a STOP: no flip open any more (``last`` says why)."""
        s = self.snap
        if s["phase"] is None:
            return
        if s["phase"] == "layer":
            self._hist_close(now, aborted=True)
        last = {"dir": s["dir"], "reason": s["reason"], "aborted": why, "phase_at_abort": s["phase"],
                "decision_ts": s["decision_ts"], "begin_ts": s["begin_ts"], "done_ts": s["done_ts"]}
        self.snap = {"phase": None, "dir": None, "since_ts": _r3(now), "begin_ts": None, "reason": None,
                     "decision_ts": None, "done_ts": None, "last": last}

    def _close(self, fw: Optional[float], done: Optional[float], now: float) -> None:
        s = self.snap
        last = {"dir": s["dir"], "reason": s["reason"], "decision_ts": s["decision_ts"],
                "begin_ts": s["begin_ts"], "done_ts": _r3(done), "first_work_ts": _r3(fw),
                "first_work_what": s.get("first_work_what"),
                "vorlauf_ms": _ms(s["decision_ts"], s["begin_ts"]),
                "layer_ms": _ms(s["begin_ts"], done),
                "nachlauf_ms": max(0, _ms(done, fw)) if done is not None and fw is not None else None,
                "decision_to_first_work_ms": _ms(s["decision_ts"], fw)}
        self.snap = {"phase": None, "dir": None, "since_ts": _r3(now), "begin_ts": None, "reason": None,
                     "decision_ts": None, "done_ts": None, "last": last}


def _via_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {"n": 0, "ms_sum": 0.0, "ms_max": 0.0, "last_ms": None, "last_ts": None}
    for p in TTFT_PARTS:
        row[p + "_sum"] = 0.0
    return row


class RequestBook:
    """Every request's timeline at the front, for ``request_done`` / ``park``
    events, ``front.ttft_by_via`` and ``front.d_activity``.

    Fed from the front's existing seams (arrival, SESSION-TRACE, leg 1
    dispatch/served, leg 2 dispatch/first content/served/end, park/resume,
    the handler's end); bounded at :attr:`MAX_ROWS` rows (oldest out)."""

    MAX_ROWS = 4096
    MAX_SESSIONS = 4096

    def __init__(self, now: float = 0.0) -> None:
        self.rows: "collections.OrderedDict[str, Dict[str, Any]]" = collections.OrderedDict()
        self.turns: "collections.OrderedDict[str, int]" = collections.OrderedDict()
        self.ttft_by_via: Dict[str, Dict[str, Any]] = {v: _via_row() for v in VIAS}
        #: rids with a live leg 2 on D
        self.d_live: set = set()
        #: rids with an open park (the flip-to-D resume walks only these)
        self.parked_open: set = set()
        #: (value, since_ts, counts) -- replaced on change, read off-loop
        self.activity: Tuple[str, Optional[float], Dict[str, int]] = (
            "idle", _r3(now) if now else None, {"prefill": 0, "decode": 0, "parked": 0})
        #: D's KV page size when the front has seen it (hand-off seam status)
        self.page_size: Optional[int] = None
        self.dropped = 0

    # ---------------- rows ----------------
    def _row(self, rid: Any) -> Optional[Dict[str, Any]]:
        return self.rows.get(str(rid))

    def arrive(self, rid: Any, now: float, epoch: int) -> None:
        r = str(rid)
        self.rows[r] = {"rid": r, "arrival_ts": float(now), "epoch0": int(epoch), "stream": None,
                        "sess": None, "turn": None, "leg1_n": 0, "leg2_n": 0, "parks": 0, "resumes": 0,
                        "park_ms": 0.0, "park_open": None}
        while len(self.rows) > self.MAX_ROWS:
            old, _ = self.rows.popitem(last=False)
            self.parked_open.discard(old)
            self.d_live.discard(old)
            self.dropped += 1

    def stream(self, rid: Any, is_stream: bool) -> None:
        row = self._row(rid)
        if row is not None:
            row["stream"] = bool(is_stream)

    def session(self, rid: Any, sess: str) -> None:
        """SESSION-TRACE's hash; ``turn`` = this rid's ordinal in its session
        as this front saw it (1 = the session's first request here)."""
        row = self._row(rid)
        if row is None or not sess:
            return
        n = self.turns.pop(sess, 0) + 1
        self.turns[sess] = n
        while len(self.turns) > self.MAX_SESSIONS:
            self.turns.popitem(last=False)
        row["sess"], row["turn"] = sess, n

    def client(self, rid: Any, ip: Optional[str]) -> None:
        """Origin address of the request (session_trace.client_ip), for the dashboard's session overview."""
        row = self._row(rid)
        if row is not None and ip:
            row["client_ip"] = str(ip)

    def est_prompt(self, rid: Any, n: Optional[int]) -> None:
        row = self._row(rid)
        if row is not None and n:
            row["est_prompt"] = int(n)

    # ---------------- legs ----------------
    def leg1_dispatch(self, rid: Any, now: float) -> None:
        row = self._row(rid)
        if row is None:
            return
        row["leg1_n"] += 1
        row.setdefault("first_dispatch_ts", float(now))
        row.setdefault("leg1_dispatch_ts", float(now))

    def leg1_done(self, rid: Any, now: float, prompt: int, cached: int,
                  tiers: Optional[Dict[str, int]], prefill_s: Optional[float]) -> None:
        row = self._row(rid)
        if row is None:
            return
        row["leg1_end_ts"] = float(now)
        p = row.setdefault("p", {"prompt": 0, "cached": 0, "prefill_s": 0.0, "prefill_s_n": 0})
        p["prompt"], p["cached"] = int(prompt or 0), int(cached or 0)
        if prefill_s is not None and prefill_s > 0:
            p["prefill_s"] += float(prefill_s)
            p["prefill_s_n"] += 1
        if tiers is not None:
            p["tiers"] = dict(tiers)

    def leg1_end_of(self, rid: Any) -> Optional[float]:
        row = self._row(rid)
        return None if row is None else row.get("leg1_end_ts")

    def leg2_dispatch(self, rid: Any, now: float) -> None:
        row = self._row(rid)
        if row is None:
            return
        row["leg2_n"] += 1
        row.setdefault("first_dispatch_ts", float(now))
        row["leg2_dispatch_ts"] = float(now)
        self.d_live.add(row["rid"])
        self._activity(now)

    def leg2_dispatch_of(self, rid: Any) -> Optional[float]:
        row = self._row(rid)
        return None if row is None else row.get("leg2_dispatch_ts")

    def first_token(self, rid: Any, now: float, via: str) -> Optional[Dict[str, Any]]:
        """D's first content for the rid (a STREAM): the TTFT and its parts into
        ``ttft_by_via``. Once per rid; returns the parts or None."""
        row = self._row(rid)
        if row is None or row.get("first_token_ts") is not None:
            return None
        row["first_token_ts"] = float(now)
        row["via"] = via
        parts = self.ttft_parts(row)
        row["ttft"] = parts
        v = self.ttft_by_via.get(via)
        if v is not None and parts.get("ttft_ms") is not None:
            ms = float(parts["ttft_ms"])
            v["n"] += 1
            v["ms_sum"] = round(v["ms_sum"] + ms, 1)
            v["ms_max"] = max(v["ms_max"], ms)
            v["last_ms"] = ms
            v["last_ts"] = _r3(now)
            for p in TTFT_PARTS:
                v[p + "_sum"] = round(v[p + "_sum"] + float(parts.get(p) or 0.0), 1)
        self._activity(now)
        return parts

    @staticmethod
    def ttft_parts(row: Dict[str, Any]) -> Dict[str, Any]:
        """Contiguous: queue (arrival -> first dispatch), p_prefill (leg 1's wall),
        flip_wait (leg 1's end -> the last leg 2 dispatch), d_first_token (that
        dispatch -> D's first content); ``other`` closes the sum (a re-route)."""
        arr, ft = row.get("arrival_ts"), row.get("first_token_ts")
        q = _ms(arr, row.get("first_dispatch_ts"))
        l1d, l1e, l2d = row.get("leg1_dispatch_ts"), row.get("leg1_end_ts"), row.get("leg2_dispatch_ts")
        p = _ms(l1d, l1e) if (l1d is not None and l1e is not None) else 0
        fw = _ms(l1e, l2d) if (l1e is not None and l2d is not None and l2d >= l1e) else 0
        d = _ms(l2d, ft)
        ttft = _ms(arr, ft)
        known = [x for x in (q, p, fw, d) if x is not None]
        other = (ttft - sum(known)) if ttft is not None else None
        return {"ttft_ms": ttft, "queue_ms": q, "p_prefill_ms": p, "flip_wait_ms": fw,
                "d_first_token_ms": d, "other_ms": other}

    def d_served(self, rid: Any, prompt: int, cached: int, completion: int,
                 tiers: Optional[Dict[str, int]], handoff: bool) -> None:
        row = self._row(rid)
        if row is None:
            return
        row["d"] = {"prompt": int(prompt or 0), "cached": int(cached or 0),
                    "completion": int(completion or 0), "tiers": dict(tiers) if tiers else None,
                    "handoff": bool(handoff)}

    def d_prefill_s(self, rid: Any, prefill_s: Optional[float]) -> None:
        row = self._row(rid)
        if row is not None and prefill_s is not None:
            row["d_prefill_s"] = float(prefill_s)

    def leg2_end(self, rid: Any, now: float) -> None:
        r = str(rid)
        if r in self.d_live:
            self.d_live.discard(r)
            row = self._row(r)
            if row is not None:
                row["leg2_end_ts"] = float(now)
            self._activity(now)

    # ---------------- park ----------------
    def park(self, rid: Any, now: float, reason: str, epoch: int) -> bool:
        row = self._row(rid)
        if row is None or row["park_open"] is not None:
            return False
        row["park_open"] = (float(now), str(reason), int(epoch))
        self.parked_open.add(row["rid"])
        row["parks"] += 1
        self._activity(now)
        return True

    def parked(self) -> Iterable[str]:
        return list(self.parked_open)

    def park_windows_of(self, rid: Any, now: float):
        """USAGE-DETAILS: [(start, end)] of every park of ``rid`` (an open one
        ends ``now``)."""
        row = self._row(rid)
        if row is None:
            return []
        out = list(row.get("park_windows") or ())
        if row["park_open"] is not None:
            out.append((float(row["park_open"][0]), float(now)))
        return out

    def resume(self, rid: Any, now: float, why: str, epoch: int) -> Optional[Dict[str, Any]]:
        """The rid's open park ends: the ``park`` event, or None (none open)."""
        row = self._row(rid)
        if row is None or row["park_open"] is None:
            return None
        t, reason, ep = row["park_open"]
        row["park_open"] = None
        self.parked_open.discard(row["rid"])
        row["resumes"] += 1
        row["park_ms"] += max(0.0, (float(now) - t) * 1000.0)
        wins = row.setdefault("park_windows", [])  # USAGE-DETAILS: the rid's park windows
        if len(wins) < 64:
            wins.append((float(t), float(now)))
        self._activity(now)
        return self._park_event(row, t, float(now), reason, ep, why, epoch)

    def _park_event(self, row: Dict[str, Any], t: float, resume: Optional[float], reason: str,
                    ep: int, why: str, epoch: int) -> Dict[str, Any]:
        ctx = self._context_tokens(row)
        return {"rid": row["rid"], "session_id": row.get("sess"), "park_ts": _r3(t), "resume_ts": _r3(resume),
                "park_ms": _ms(t, resume), "reason": reason, "end": why, "epoch_park": ep,
                "epoch_resume": int(epoch), "pages": self._pages(ctx), "context_tokens_est": ctx,
                "pages_src": ("est_context/page_size" if self._pages(ctx) is not None else None)}

    # ---------------- d_activity ----------------
    def _activity(self, now: float) -> None:
        pre = dec = par = 0
        for r in self.d_live:
            row = self.rows.get(r)
            if row is None:
                continue
            if row["park_open"] is not None:
                par += 1
            elif row.get("first_token_ts") is not None:
                dec += 1
            else:
                pre += 1  # stream before its first content, or a non-stream leg (no token seen)
        val = "prefill" if pre else "decode" if dec else "idle"
        counts = {"prefill": pre, "decode": dec, "parked": par}
        cur = self.activity
        if cur[0] != val:
            self.activity = (val, _r3(now), counts)
        elif cur[2] != counts:
            self.activity = (val, cur[1], counts)

    def activity_block(self) -> Dict[str, Any]:
        val, since, counts = self.activity
        return {"value": val, "since_ts": since, "n": dict(counts)}

    def ttft_block(self) -> Dict[str, Any]:
        return {v: dict(row) for v, row in self.ttft_by_via.items()}

    # ---------------- end ----------------
    def _context_tokens(self, row: Dict[str, Any]) -> Optional[int]:
        d = row.get("d")
        if d and d.get("prompt"):
            return int(d["prompt"]) + int(d.get("completion") or 0)
        p = row.get("p")
        if p and p.get("prompt"):
            return int(p["prompt"])
        return row.get("est_prompt")

    def _pages(self, ctx: Optional[int]) -> Optional[int]:
        if not ctx or not self.page_size:
            return None
        return int(math.ceil(int(ctx) / int(self.page_size)))

    def done(self, rid: Any, now: float, status: Any, epoch: int,
             common: Optional[Tuple[str, int]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """The handler's end: ``(request_done record, park event of a park still
        open)``; (None, None) for a rid this book never saw (or dropped)."""
        r = str(rid)
        row = self.rows.pop(r, None)
        self.parked_open.discard(r)
        if r in self.d_live:
            self.d_live.discard(r)
            self._activity(now)
        if row is None:
            return None, None
        park_ev = None
        if row["park_open"] is not None:
            t, reason, ep = row["park_open"]
            row["park_ms"] += max(0.0, (float(now) - t) * 1000.0)
            park_ev = self._park_event(row, t, None, reason, ep, "request_end", epoch)
        return self.record(row, now, status, epoch, common), park_ev

    def record(self, row: Dict[str, Any], now: float, status: Any, epoch: int,
               common: Optional[Tuple[str, int]] = None) -> Dict[str, Any]:
        t = row.get("ttft") or (self.ttft_parts(row) if row.get("first_token_ts") else {})
        p, d = row.get("p"), row.get("d")
        via = row.get("via") or ("after_p" if row["leg1_n"] else ("d_direct" if row["leg2_n"] else None))
        prefill: Dict[str, Any] = {}
        if p is not None:
            l1 = _ms(row.get("leg1_dispatch_ts"), row.get("leg1_end_ts"))
            prefill["P"] = {"ms": (round(p["prefill_s"] * 1000.0) if p["prefill_s_n"] else l1),
                            "ms_src": "weg2_prefill_s" if p["prefill_s_n"] else "leg1_wall",
                            "wall_ms": l1, "tokens": max(0, p["prompt"] - p["cached"]),
                            "prompt": p["prompt"], "cached": p["cached"]}
        if d is not None:
            dps = row.get("d_prefill_s")
            prefill["D"] = {"ms": None if dps is None else round(dps * 1000.0),
                            "ms_src": None if dps is None else "weg2_prefill_s",
                            "tokens": max(0, d["prompt"] - d["cached"]),
                            "prompt": d["prompt"], "cached": d["cached"]}
        cached = None
        if d is not None:
            tiers = d.get("tiers") or {}
            cached = {"total": d["cached"], "device": tiers.get("device"), "host": tiers.get("host"),
                      "storage": tiers.get("storage"),
                      # the P->D hand-off: D's cached share of a rid whose leg 1 ran on P
                      "told": d["cached"] if d.get("handoff") else 0}
        ft, end = row.get("first_token_ts"), float(now)
        ctx = self._context_tokens(row)
        rec = {
            "rid": row["rid"], "session_id": row.get("sess"), "turn": row.get("turn"), "client_ip": row.get("client_ip"), "via": via,
            "stream": row.get("stream"), "status": status,
            "arrival_ts": _r3(row["arrival_ts"]), "first_token_ts": _r3(ft), "end_ts": _r3(end),
            "ttft_ms": t.get("ttft_ms"), "queue_ms": t.get("queue_ms") if t else _ms(row["arrival_ts"], row.get("first_dispatch_ts")),
            "p_prefill_ms": t.get("p_prefill_ms"), "flip_wait_ms": t.get("flip_wait_ms"),
            "d_first_token_ms": t.get("d_first_token_ms"), "other_ms": t.get("other_ms"),
            "prefill": prefill, "cached": cached,
            "common_prefix": None if common is None else int(common[1]),
            "prev_rid": None if common is None else common[0],
            "decode_tokens": None if d is None else d["completion"],
            "decode_ms": _ms(ft, end) if ft is not None else None,
            "context_tokens": ctx, "kv_pages": self._pages(ctx),
            "parks": row["parks"], "resumes": row["resumes"], "park_ms": int(round(row["park_ms"])),
            "flip_epochs": max(0, int(epoch) - int(row["epoch0"])),
            "legs": {"p": row["leg1_n"], "d": row["leg2_n"]},
            "wall_s": round(end - float(row["arrival_ts"]), 3),
        }
        return rec


def influx_req_fields(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(tags, fields)`` of the ``weg2_req`` point (TSDB-DELTA 1c): low-cardinality
    tags only (via, status); rid and session are FIELDS."""
    pf = rec.get("prefill") or {}
    c = rec.get("cached") or {}
    tags = {"via": rec.get("via"), "status": rec.get("status")}
    fields = {"rid": rec.get("rid"), "session_id": rec.get("session_id"), "turn": rec.get("turn"),
              "ttft_ms": rec.get("ttft_ms"), "queue_ms": rec.get("queue_ms"),
              "p_prefill_ms": rec.get("p_prefill_ms"), "flip_wait_ms": rec.get("flip_wait_ms"),
              "d_first_token_ms": rec.get("d_first_token_ms"),
              "leg2_ms": rec.get("d_first_token_ms"),
              "wall_s": rec.get("wall_s"), "decode_ms": rec.get("decode_ms"),
              "prompt": (pf.get("D") or pf.get("P") or {}).get("prompt"),
              "cached": c.get("total"), "cached_device": c.get("device"), "cached_host": c.get("host"),
              "cached_storage": c.get("storage"), "cached_told": c.get("told"),
              "completion": rec.get("decode_tokens"), "context_tokens": rec.get("context_tokens"),
              "p_prefill_tokens": (pf.get("P") or {}).get("tokens"),
              "d_prefill_tokens": (pf.get("D") or {}).get("tokens"),
              "common_prefix": rec.get("common_prefix"), "parks": rec.get("parks"),
              "park_ms": rec.get("park_ms"), "flip_epochs": rec.get("flip_epochs")}
    return tags, fields
