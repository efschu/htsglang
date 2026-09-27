"""SESSION-TRACE (HS 27.09., P-HiCache-read order): which agent session a rid
belongs to, and how far its prompt shares the previous prompt of that session.

WHY. The front log has no session id, so the question "does P's store hit end
before pages that ARE in the follow-up prompt?" could only be answered by a
heuristic (NF p_cached_ketten.py: previous turn = the last finished D leg with
prompt+completion <= this prompt). The load driver is Claude Code agents through
the router on 30099, which forwards the client's headers (hop-by-hop dropped),
and Claude Code sends ``X-Claude-Code-Session-Id``. Fallback: the session part
of ``metadata.user_id`` in the Anthropic body.

WHAT IS LOGGED. Never the id itself, never text: a 10-hex sha1 of it
(``sess=``). With the front tokenizer's ids (X-EXACT counts them anyway) the
common token prefix with the SAME session's previous prompt:

  WEG2 SESSION-PREFIX rid sess prev_rid common prompt prev_prompt

``common`` is where this prompt leaves the previous one: a P hit that ends
there reached everything the prompt shares with its predecessor; a hit that
ends earlier left shared pages unread.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, Mapping, Optional, Tuple

HEADER = "X-Claude-Code-Session-Id"
_META_SESSION = re.compile(r"session_([0-9A-Za-z-]{8,})")
#: sessions whose last prompt ids are kept (a prompt is <= ~262k int32 ids,
#: ~1 MiB; 64 sessions bound it well below the front's footprint).
MAX_SESSIONS = 64


def session_raw(headers: Optional[Mapping[str, str]], payload: Any) -> Tuple[str, str]:
    """``(raw id, source)``; ``("", "none")`` when neither carries one."""
    try:
        v = (headers or {}).get(HEADER) or (headers or {}).get(HEADER.lower())
    except Exception:  # noqa: BLE001
        v = None
    if v and str(v).strip():
        return str(v).strip(), "header"
    try:
        uid = str(((payload or {}).get("metadata") or {}).get("user_id") or "")
    except Exception:  # noqa: BLE001
        uid = ""
    m = _META_SESSION.search(uid)
    if m:
        return m.group(1), "metadata"
    return "", "none"


def short(raw: str) -> str:
    return hashlib.sha1(raw.encode(errors="replace")).hexdigest()[:10] if raw else ""


def common_prefix(a, b) -> int:
    """Length of the common prefix of two id sequences (numpy or lists)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    try:
        import numpy as np

        x = np.asarray(a[:n])
        y = np.asarray(b[:n])
        diff = np.nonzero(x != y)[0]
        return int(diff[0]) if diff.size else n
    except Exception:  # noqa: BLE001 - plain lists without numpy
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n


class SessionPrefixes:
    """Per session: the last prompt's ids and rid (bounded LRU)."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self.max = int(max_sessions)
        self._last: "OrderedDict[str, Tuple[str, Any]]" = OrderedDict()

    def note(self, sess: str, rid: str, ids) -> Optional[Tuple[str, int, int]]:
        """Record ``ids`` as ``sess``'s latest prompt; return ``(prev_rid,
        common, prev_len)`` against the previous one, or None (first seen)."""
        if not sess or ids is None:
            return None
        try:
            import numpy as np

            cur = np.asarray(ids, dtype=np.int64).copy()
        except Exception:  # noqa: BLE001
            cur = list(ids)
        prev = self._last.pop(sess, None)
        self._last[sess] = (rid, cur)
        while len(self._last) > self.max:
            self._last.popitem(last=False)
        if prev is None:
            return None
        prev_rid, prev_ids = prev
        return prev_rid, common_prefix(prev_ids, cur), len(prev_ids)
