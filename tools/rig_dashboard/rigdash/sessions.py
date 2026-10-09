"""Sessions overview of a boot out of the front's ``request_done`` events (events.jsonl, one per finished request).

User 08.10.: "bei sessions soll die IP der herkunftssession angegeben werden und die aktuelle 'groesse' also laenge an
kontext".  Per session (the front's 10-hex hash of X-Claude-Code-Session-Id / metadata.user_id, ``session_id``):

* ``ctx`` = ``context_tokens`` of the session's newest request -- the context the session has now (prompt + answer),
* ``ip`` = ``client_ip`` of the newest request that carries one.  The front does NOT write this field today (it records
  the session hash, never the peer; and behind the router 30099 the peer it sees is the router).  Without it the column
  says so instead of showing a value: nothing is guessed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

LIMIT = 40


def _ctx(r: dict) -> Optional[int]:
    v = r.get("context_tokens")
    return int(v) if isinstance(v, (int, float)) and v >= 0 else None


def sessions_view(req_done: List[dict], limit: int = LIMIT) -> dict:
    """``{"rows": [...], "n_requests": N, "ip_known": bool}``, newest session first.  A request without a session id
    (no header, no metadata) forms the one row ``session`` = None."""
    by: Dict[Optional[str], List[dict]] = {}
    for r in req_done or ():
        by.setdefault(r.get("session_id") or None, []).append(r)
    rows = []
    for sess, rs in by.items():
        rs.sort(key=lambda r: r.get("end_ts") or 0.0)
        last = rs[-1]
        ips = [r["client_ip"] for r in rs if r.get("client_ip")]
        ctxs = [c for c in (_ctx(r) for r in rs) if c is not None]
        pre = ((last.get("prefill") or {}).get("D") or (last.get("prefill") or {}).get("P") or {})
        rows.append({"session": sess, "n": len(rs), "turn": max([r.get("turn") or 0 for r in rs] or [0]) or None,
                     "first_ts": rs[0].get("arrival_ts"), "last_ts": last.get("end_ts"),
                     "ctx": _ctx(last), "ctx_max": max(ctxs) if ctxs else None,
                     "prompt": pre.get("prompt"), "cached": (last.get("cached") or {}).get("total"),
                     "via": last.get("via"), "ip": ips[-1] if ips else None,
                     "ips": sorted(set(ips))})
    rows.sort(key=lambda x: -(x["last_ts"] or 0.0))
    return {"rows": rows[:limit], "n_sessions": len(rows), "n_requests": sum(x["n"] for x in rows),
            "ip_known": any(r.get("client_ip") for r in req_done or ())}
