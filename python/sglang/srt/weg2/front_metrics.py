"""TSDB push (user order 01.10. ~07:40Z, spec docs/TSDB-DELTA-27B-1001.md 1c):
the Influx-line protocol and the bounded pusher the weg2 front uses to write
its ``weg2_req`` points to VictoriaMetrics (``SGLANG_WEG2_METRICS_PUSH_URL``,
default OFF).

This is the PUSH SUBSET of the 27B front's ``front_metrics.py`` (desk/27b-
tsdb-metrics-1001 30c4c22c3f), byte-identical in ``influx_line`` /
``InfluxPusher`` / ``_http_post``, so the full module (the front's own
``/metrics`` route and ``weg2_*`` families) supersedes this file when it is
ported to the NF line -- one implementation, not two.

Rules: the HTTP write runs in the front's BoundedWriter thread (``submit``),
never on the event loop or the flip; a full buffer drops its oldest points
(counted); a failure is counted, never raised. Never ``rid`` as a tag.
"""
from __future__ import annotations

import collections
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

#: the push bundles points for about this long before one write
PUSH_EVERY_S = 2.0
#: points kept while the writer is slow or the target unreachable (oldest dropped)
PUSH_BUFFER_MAX = 20000
PUSH_TIMEOUT_S = 2.0


# ---------------------------------------------------------------------------
# Influx line protocol
# ---------------------------------------------------------------------------

def _esc_key(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def _field_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v}i"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return repr(float(v))
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def influx_line(measurement: str, tags: Dict[str, Any], fields: Dict[str, Any],
                ts_ns: Optional[int] = None) -> Optional[str]:
    """One Influx line; None when no field has a value (a line needs one)."""
    fs = []
    for k, v in fields.items():
        fv = _field_value(v)
        if fv is not None:
            fs.append(f"{_esc_key(k)}={fv}")
    if not fs:
        return None
    head = _esc_key(measurement)
    for k in sorted(tags):
        v = tags[k]
        if v is None or str(v) == "":
            continue
        head += f",{_esc_key(k)}={_esc_key(v)}"
    ts = int(ts_ns if ts_ns is not None else time.time_ns())
    return f"{head} {','.join(fs)} {ts}"


class InfluxPusher:
    """Points into a bounded buffer (any thread); ``maybe_flush`` hands ONE
    write to ``submit`` (the front's BoundedWriter) at most every
    ``every_s`` -- the HTTP write runs there, never in the caller."""

    def __init__(self, url: str, *, submit: Callable[..., None], every_s: float = PUSH_EVERY_S,
                 maxlen: int = PUSH_BUFFER_MAX, timeout_s: float = PUSH_TIMEOUT_S,
                 post: Optional[Callable[[str, bytes, float], None]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.url = url
        self._submit = submit
        self.every_s = float(every_s)
        self.timeout_s = float(timeout_s)
        self._buf: collections.deque = collections.deque()
        self.maxlen = int(maxlen)
        self._lock = threading.Lock()
        self._inflight = False
        self._last = clock()
        self._clock = clock
        self._post = post or _http_post
        self.points = 0
        self.dropped = 0
        self.errors = 0
        self.writes = 0

    def add(self, line: Optional[str]) -> None:
        if not line:
            return
        with self._lock:
            if len(self._buf) >= self.maxlen:
                self._buf.popleft()
                self.dropped += 1
            self._buf.append(line)
            self.points += 1

    def maybe_flush(self, force: bool = False) -> bool:
        """Submit one write when due; True when a write was handed off."""
        now = self._clock()
        with self._lock:
            if self._inflight or not self._buf:
                return False
            if not force and now - self._last < self.every_s:
                return False
            lines = list(self._buf)
            self._buf.clear()
            self._inflight = True
            self._last = now
        try:
            self._submit(self._write, lines)
        except Exception:  # noqa: BLE001 - a refused hand-off loses this batch only
            with self._lock:
                self._inflight = False
            self.errors += 1
            return False
        return True

    def _write(self, lines: List[str]) -> None:
        try:
            self._post(self.url, ("\n".join(lines) + "\n").encode(), self.timeout_s)
            self.writes += 1
        except Exception:  # noqa: BLE001 - counted, never raised (the writer counts too)
            self.errors += 1
        finally:
            with self._lock:
                self._inflight = False


def _http_post(url: str, body: bytes, timeout_s: float) -> None:
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "text/plain; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        r.read()
