"""TSDB (user order 01.10. ~07:40Z, spec docs/TSDB-DELTA-27B-1001.md): the
pdflip front's own Prometheus metrics, the P/D aggregation of ``/metrics`` and
the optional Influx-line push to VictoriaMetrics.

Rules this module is shaped around (operator 01.10.):

* READ-ONLY instruments: an observation is an in-process counter/histogram
  update, nothing else. They are always on; they never change routing.
* Everything that is not a pure read is gated by env, default OFF: the push
  (``FLLIPER_PDFLIP_METRICS_PUSH_URL``) only. It runs in the front's BoundedWriter
  thread (a ``submit`` callable the front hands in) -- never on the event loop,
  never on the flip path; a full buffer drops its oldest points (counted).
* A metric failure is COUNTED (``pdflip_metrics_errors_total{where}``), never
  raised into the front.
* Never ``rid`` as a label (cardinality); the push carries it as a FIELD.

The registry is the module's own (a ``CollectorRegistry``, not the process
global): the front is one process, and its families must not mix with an
imported library's defaults. Without ``prometheus_client`` every method is a
no-op and ``render()`` says so in a comment line.
"""
from __future__ import annotations

import collections
import logging
import os
import re
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PUSH_URL_ENV = "FLLIPER_PDFLIP_METRICS_PUSH_URL"
MODEL_ENV = "FLLIPER_PDFLIP_METRICS_MODEL"
#: the push bundles points for about this long before one write
PUSH_EVERY_S = 2.0
#: points kept while the writer is slow or the target unreachable (oldest dropped)
PUSH_BUFFER_MAX = 20000
PUSH_TIMEOUT_S = 2.0
#: one group's /metrics scrape bound (spec: 2 s)
GROUP_SCRAPE_TIMEOUT_S = 2.0
GROUP_LABEL = "pdflip_group"

_LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0)
_FLIP_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0)
_RPC_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

VIAS = ("d_direct", "after_p", "d_single")


# imported HERE, at the front module's import (front.py imports this module at
# top level): a first import inside the event loop is the weg2rc2 class
# (test_pdflip_loop_import_keepalive_rc2_0925 -- 5.36 s of loop lost there)
try:
    import prometheus_client as _PROMETHEUS_CLIENT
except Exception:  # noqa: BLE001 - no client: the instruments are no-ops
    _PROMETHEUS_CLIENT = None


def _prom():
    return _PROMETHEUS_CLIENT


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


# ---------------------------------------------------------------------------
# P/D /metrics aggregation
# ---------------------------------------------------------------------------

_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?(\s.*)$")


def relabel_group(text: str, group: str, seen_meta: set) -> str:
    """Every sample of a group server's exposition gets ``pdflip_group="<group>"``;
    a ``# HELP``/``# TYPE`` line of a family already emitted (by the front or
    the other group) is dropped, so the joint text names each family once."""
    out: List[str] = []
    tag = f'{GROUP_LABEL}="{group}"'
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                key = (parts[1], parts[2])
                if key in seen_meta:
                    continue
                seen_meta.add(key)
            out.append(line)
            continue
        m = _SAMPLE_RE.match(line)
        if m is None:
            continue  # not a sample line we understand: never forward garbage
        name, labels, rest = m.group(1), m.group(2), m.group(3)
        if labels and labels != "{}":
            labels = "{" + tag + "," + labels[1:]
        else:
            labels = "{" + tag + "}"
        out.append(f"{name}{labels}{rest}")
    return "\n".join(out) + ("\n" if out else "")


def meta_of(text: str) -> set:
    seen = set()
    for line in text.splitlines():
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                seen.add((parts[1], parts[2]))
    return seen


# ---------------------------------------------------------------------------
# the front's metrics
# ---------------------------------------------------------------------------

class FrontMetrics:
    def __init__(self, *, submit: Optional[Callable[..., None]] = None,
                 environ: Optional[Dict[str, str]] = None) -> None:
        if environ is None:
            # the registered switches (environ.py: one switch, overridable in tests)
            from flliper.srt.environ import envs

            model_raw = envs.FLLIPER_PDFLIP_METRICS_MODEL.get()
            url_raw = envs.FLLIPER_PDFLIP_METRICS_PUSH_URL.get()
        else:
            model_raw, url_raw = environ.get(MODEL_ENV), environ.get(PUSH_URL_ENV)
        self.model = (model_raw or "").strip() or None
        self.prom = _prom()
        self.errors: Dict[str, int] = collections.Counter()
        self._req: Dict[str, Dict[str, float]] = {}
        self.registry = None
        url = (url_raw or "").strip()
        self.pusher: Optional[InfluxPusher] = (
            InfluxPusher(url, submit=submit) if url and submit is not None else None)
        if self.prom is None:
            return
        p = self.prom
        r = self.registry = p.CollectorRegistry(auto_describe=True)
        self.ttft = p.Histogram("pdflip_ttft_seconds", "Arrival at the front to D's first content",
                                ["via"], buckets=_LAT_BUCKETS, registry=r)
        self.leg2_first = p.Histogram("pdflip_leg2_first_content_seconds",
                                      "Leg-2 dispatch to D's first content", ["via"],
                                      buckets=_LAT_BUCKETS, registry=r)
        self.request = p.Histogram("pdflip_request_seconds", "Wall time of a served leg",
                                   ["group"], buckets=_LAT_BUCKETS, registry=r)
        self.served = p.Counter("pdflip_served", "Served legs", ["group"], registry=r)
        self.tokens = p.Counter("pdflip_tokens", "Tokens of served legs", ["group", "kind"], registry=r)
        self.flip = p.Histogram("pdflip_flip_seconds", "Flip times (layer = the flip_log flip_ms, "
                                "first_work = begin to the woken group's first work, user = "
                                "decode end to P prefill start)", ["dir", "kind"],
                                buckets=_FLIP_BUCKETS, registry=r)
        self.flips = p.Counter("pdflip_flips", "Flips done", ["dir"], registry=r)
        self.park_rpc = p.Histogram("pdflip_park_rpc_seconds", "D park RPC wall time",
                                    buckets=_RPC_BUCKETS, registry=r)
        self.queue_len = p.Gauge("pdflip_queue_len", "Front queue length", registry=r)
        self.outstanding = p.Gauge("pdflip_outstanding", "Open requests of the front", registry=r)
        self.d_seats = p.Gauge("pdflip_d_seats", "D seats in force", registry=r)
        self.d_parked = p.Gauge("pdflip_d_parked", "D parked requests", registry=r)
        self.awake = p.Gauge("pdflip_awake", "1 for the awake group", ["group"], registry=r)
        self.scrape_ok = p.Gauge("pdflip_group_scrape_ok", "1 when the group's /metrics answered",
                                 [GROUP_LABEL], registry=r)
        self.err_c = p.Counter("pdflip_metrics_errors", "Metric/push failures (never raised)",
                               ["where"], registry=r)
        self.push_c = p.Gauge("pdflip_metrics_push", "Push state (points, dropped, errors, writes)",
                              ["what"], registry=r)

    # -- error discipline -------------------------------------------------
    def _err(self, where: str, exc: BaseException) -> None:
        self.errors[where] += 1
        try:
            if self.registry is not None:
                self.err_c.labels(where=where).inc()
        except Exception:  # noqa: BLE001
            pass
        if self.errors[where] <= 3:
            logger.warning("PDFLIP-METRICS %s failed (%s: %s) -- counted, never raised",
                           where, type(exc).__name__, exc)

    def _tags(self, **kw) -> Dict[str, Any]:
        t = {"model": self.model}
        t.update(kw)
        return t

    def _push(self, measurement: str, tags: Dict[str, Any], fields: Dict[str, Any]) -> None:
        if self.pusher is None:
            return
        try:
            self.pusher.add(influx_line(measurement, self._tags(**tags), fields))
            self.pusher.maybe_flush()
        except Exception as e:  # noqa: BLE001
            self._err("push", e)

    # -- request path -----------------------------------------------------
    def leg2_first_content(self, rid: str, via: str, leg2_s: float, arrival_ts: Optional[float],
                           now: Optional[float] = None) -> None:
        try:
            now = time.time() if now is None else float(now)
            rec = self._req.setdefault(str(rid), {})
            rec["leg2_ms"] = round(float(leg2_s) * 1000.0, 1)
            if self.registry is not None:
                self.leg2_first.labels(via=via).observe(max(0.0, float(leg2_s)))
            if arrival_ts is not None:
                ttft = max(0.0, now - float(arrival_ts))
                rec["ttft_ms"] = round(ttft * 1000.0, 1)
                if self.registry is not None:
                    self.ttft.labels(via=via).observe(ttft)
            rec["via"] = via
            while len(self._req) > 4096:
                self._req.pop(next(iter(self._req)))
        except Exception as e:  # noqa: BLE001
            self._err("leg2_first_content", e)

    def served_leg(self, group: str, rid: str, wall_s: float, prompt: int, cached: int,
                   completion: int = 0) -> None:
        try:
            if self.registry is not None:
                self.served.labels(group=group).inc()
                self.request.labels(group=group).observe(max(0.0, float(wall_s)))
                for kind, v in (("prompt", prompt), ("cached", cached), ("completion", completion)):
                    if v:
                        self.tokens.labels(group=group, kind=kind).inc(max(0, int(v)))
            if group == "D":
                # the pdflip_req POINT is the front's request_done (DASHBOARD-IPC, NF
                # front_requests.influx_req_fields) -- one point per finished request,
                # written through this module's pusher; nothing pushed per leg here
                self._req.pop(str(rid), None)
        except Exception as e:  # noqa: BLE001
            self._err("served_leg", e)

    def forget(self, rid: str) -> None:
        self._req.pop(str(rid), None)

    # -- flip events (the front's own IPC events, tapped) -----------------
    def on_event(self, typ: str, data: Dict[str, Any]) -> None:
        try:
            if typ == "flip_done":
                d = f"{data.get('sleep')}>{data.get('wake')}"
                fm = data.get("flip_ms")
                if self.registry is not None:
                    self.flips.labels(dir=d).inc()
                    if fm is not None:
                        self.flip.labels(dir=d, kind="layer").observe(float(fm) / 1000.0)
                self._push("pdflip_flip", {"dir": d}, {"epoch": data.get("epoch"), "flip_ms": fm,
                                                     "drain_quiesce_ms": data.get("drain_quiesce_ms")})
            elif typ == "flip_first_work":
                v = data.get("flip_time_ms")
                if v is not None:
                    d = str(data.get("dir") or "?")
                    if self.registry is not None:
                        self.flip.labels(dir=d, kind="first_work").observe(float(v) / 1000.0)
                    self._push("pdflip_flip", {"dir": d}, {"epoch": data.get("epoch"), "first_work_ms": v})
            elif typ == "flip_user_time":
                v = data.get("flip_user_ms")
                d = str(data.get("dir") or "D>P")
                parts = data.get("parts") or {}
                if v is not None and self.registry is not None:
                    self.flip.labels(dir=d, kind="user").observe(float(v) / 1000.0)
                self._push("pdflip_flip", {"dir": d}, {"epoch": data.get("epoch"), "user_ms": v,
                                                     "park_rpc_ms": parts.get("park_rpc_ms")})
        except Exception as e:  # noqa: BLE001
            self._err("on_event", e)

    def park_rpc_done(self, seconds: float) -> None:
        try:
            if self.registry is not None:
                self.park_rpc.observe(max(0.0, float(seconds)))
        except Exception as e:  # noqa: BLE001
            self._err("park_rpc", e)

    # -- scrape -----------------------------------------------------------
    def set_gauges(self, *, queue_len: int, outstanding: int, d_seats: Optional[int], d_parked: int,
                   awake: Optional[str], groups: Iterable[str] = ("P", "D")) -> None:
        if self.registry is None:
            return
        try:
            self.queue_len.set(int(queue_len))
            self.outstanding.set(int(outstanding))
            if d_seats is not None:
                self.d_seats.set(int(d_seats))
            self.d_parked.set(int(d_parked))
            for g in groups:
                self.awake.labels(group=g).set(1 if g == awake else 0)
            if self.pusher is not None:
                for k in ("points", "dropped", "errors", "writes"):
                    self.push_c.labels(what=k).set(getattr(self.pusher, k))
        except Exception as e:  # noqa: BLE001
            self._err("gauges", e)

    def set_scrape_ok(self, group: str, ok: bool) -> None:
        if self.registry is None:
            return
        try:
            self.scrape_ok.labels(**{GROUP_LABEL: group}).set(1 if ok else 0)
        except Exception as e:  # noqa: BLE001
            self._err("scrape_ok", e)

    def render(self) -> str:
        if self.registry is None:
            return "# pdflip front metrics: prometheus_client not importable\n"
        try:
            return self.prom.generate_latest(self.registry).decode()
        except Exception as e:  # noqa: BLE001
            self._err("render", e)
            return ""

    def aggregate(self, group_texts: List[Tuple[str, Optional[str]]]) -> str:
        """The front's own text plus every group's, relabelled; a group whose
        text is None (scrape failed) contributes only ``pdflip_group_scrape_ok 0``."""
        for g, t in group_texts:
            self.set_scrape_ok(g, t is not None)
        own = self.render()
        seen = meta_of(own)
        parts = [own]
        for g, t in group_texts:
            if t is None:
                continue
            try:
                parts.append(relabel_group(t, g, seen))
            except Exception as e:  # noqa: BLE001
                self._err("relabel", e)
        return "".join(parts)


__all__ = ["FrontMetrics", "InfluxPusher", "influx_line", "relabel_group", "meta_of",
           "PUSH_URL_ENV", "MODEL_ENV", "GROUP_LABEL", "GROUP_SCRAPE_TIMEOUT_S"]
