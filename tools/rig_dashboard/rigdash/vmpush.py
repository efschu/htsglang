"""IPC -> VictoriaMetrics (Nutzer-Order 01.10. ~07:40Z: Server- und Performance-Daten dauerhaft in einer
bestehenden Zeitreihen-DB, das Dashboard und spätere Auswertungen lesen nur daraus; kein grep, kein
Log-Parsing als Datenquelle; wo unsere Anbindung nicht passt, machen wir UNSERE Seite kompatibel).

Bis die Front und die Ränge ihre Marker selbst nach VictoriaMetrics schreiben (Liste an 27B, 01.10.),
übersetzt der Probennehmer die IPC, die er ohnehin jede Sekunde liest, in Prometheus-Metriken:

  state.json front       weg2_front_*  (served, served_tokens, arrival_seat.ttft_* , Warteschlange ...)
  rankstate/*.rankstats  weg2_rank_*   (kumulative Zähler und Pegel je Rang)
  events.jsonl           weg2_flip_time_ms / weg2_flip_user_ms als Punkte mit dem Zeitstempel des Flips

Kein Log wird geöffnet.  Labels: ``model`` (27B|NF), ``boot`` (Kurzform der Boot-ID, ein Wert je Boot),
``group``/``rank``/``dir``/``kind`` -- nie eine rid (Nutzer: keine Labels mit hoher Kardinalität).
Geschrieben wird per ``/api/v1/import/prometheus`` (Text-Exposition mit Zeitstempel in ms).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_URL = "http://127.0.0.1:8428"
PUSH_S = 5.0

#: rankstats fields -> metric suffix (all cumulative counters unless named in RANK_GAUGES)
RANK_FIELDS = (
    ("prefill", "new_tokens", "prefill_new_tokens_total"),
    ("prefill", "cached_tokens", "prefill_cached_tokens_total"),
    ("prefill", "chunks", "prefill_chunks_total"),
    ("prefill", "compute_ms", "prefill_compute_ms_total"),
    ("decode", "tokens", "decode_tokens_total"),
    ("decode", "rounds", "decode_rounds_total"),
    ("decode", "gpu_ms", "decode_gpu_ms_total"),
    ("decode", "running", "decode_running"),
    ("decode", "last_bs", "decode_last_bs"),
    ("decode", "accept_len_ewma", "decode_accept_len"),
    ("sched", "full_token_usage", "kv_usage_ratio"),
    ("sched", "queue_req", "queue_requests"),
    ("sched", "pending_tokens", "pending_tokens"),
    ("cap", "kv_tokens", "kv_capacity_tokens"),
    ("cap", "seats", "seats_capacity"),
    ("cache", "loadback_n", "l2_loadback_total"),
    ("cache", "loadback_tok", "l2_loadback_tokens_total"),
    ("cache", "mamba_resume_n", "mamba_resume_total"),
    ("cache", "store_incomplete_n", "l3_incomplete_total"),
)


def short_boot(boot_id: Optional[str]) -> str:
    """``nfh91…dauer-boot-20261001T064831Z-2025`` -> ``boot-20261001T064831Z-2025`` (one value per boot)."""
    b = boot_id or "?"
    i = b.rfind("-boot-")
    return b[i + 1:] if i >= 0 else b[-40:]


def _num(x) -> Optional[float]:
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _lbl(d: Dict[str, str]) -> str:
    return "{" + ",".join('%s="%s"' % (k, str(v).replace("\\", "\\\\").replace('"', '\\"')) for k, v in sorted(d.items())) + "}"


def lines_for_boot(ipc: dict, rankstats: Dict[str, dict], model: str, now_ms: int) -> List[str]:
    """Exposition lines (``name{labels} value ts_ms``) for one live boot -- pure, unit-tested."""
    base = {"model": model, "boot": short_boot(ipc.get("boot_id") or ipc.get("dir"))}
    out: List[str] = []

    def put(name: str, v, extra: Optional[Dict[str, str]] = None, ts: Optional[int] = None):
        x = _num(v)
        if x is None:
            return
        out.append("%s%s %s %d" % (name, _lbl(dict(base, **(extra or {}))), repr(x), ts if ts is not None else now_ms))

    fr = ipc.get("front") or {}
    put("weg2_front_up", 0.0 if ipc.get("terminal") else 1.0)
    for g, n in (fr.get("served") or {}).items():
        put("weg2_front_served_total", n, {"group": g})
    for g, st in (fr.get("served_tokens") or {}).items():
        if isinstance(st, dict):
            for k in ("prompt", "cached", "completion", "n"):
                put("weg2_front_served_tokens_total", st.get(k), {"group": g, "kind": k})
    a = fr.get("arrival_seat") or {}
    # TTFT der Nutzer: Ankunft an der Front -> erster Inhalt von D (front.py LEG2-FIRST-CONTENT, IPC-Zähler)
    put("weg2_front_ttft_count", a.get("ttft_n"))
    put("weg2_front_ttft_ms_sum", a.get("ttft_ms_sum"))
    put("weg2_front_ttft_ms_max", a.get("ttft_ms_max"))
    put("weg2_front_verdict_count", a.get("verdict_n"))
    put("weg2_front_verdict_ms_sum", a.get("verdict_ms_sum"))
    for k in ("outstanding_n", "d_phase_n", "d_parked_n", "epoch", "oldest_outstanding_age_s"):
        put("weg2_front_" + k, fr.get(k))
    q = fr.get("queue")
    put("weg2_front_queue", len(q) if isinstance(q, list) else q)
    aw = fr.get("awake")
    if aw in ("P", "D"):
        for g in ("P", "D"):
            put("weg2_front_awake", 1.0 if aw == g else 0.0, {"group": g})
    for rk, rec in sorted((rankstats or {}).items()):
        g, _, r = rk.partition(".")
        ex = {"group": g, "rank": r or "?"}
        for sec, field, name in RANK_FIELDS:
            blk = rec.get(sec) if isinstance(rec.get(sec), dict) else {}
            put("weg2_rank_" + name, blk.get(field), ex)
        cache = rec.get("cache") if isinstance(rec.get("cache"), dict) else {}
        for k in ("store_incomplete_delivered", "store_incomplete_deliverable", "mamba_tok"):
            put("weg2_rank_cache_%s_total" % k, cache.get(k), ex)
        pf = cache.get("prefetch") if isinstance(cache.get("prefetch"), dict) else {}
        for k, v in pf.items():          # L3 prefetch census (attempted/issued/landed/refused/expired/timeout ...)
            put("weg2_rank_l3_prefetch_total", v, dict(ex, outcome=k))
    return out


def flip_points(ipc: dict, model: str, since_ts: float) -> Tuple[List[str], float]:
    """One point per flip newer than ``since_ts`` at the flip's own time (events.jsonl, front clock):
    weg2_flip_time_ms{dir} = flip_first_work.flip_time_ms (P>D Flipzeit, what="none" gives none),
    weg2_flip_user_ms{dir="D>P"} = flip_user_time.flip_user_ms (D>P Flipzeit nach Nutzerdefinition)."""
    base = {"model": model, "boot": short_boot(ipc.get("boot_id") or ipc.get("dir"))}
    out, newest = [], since_ts
    for d in ipc.get("flip_first_work") or []:
        t = d.get("flip_begin_ts") or d.get("t")
        v = d.get("flip_time_ms")
        if t is None or t <= since_ts or d.get("dir") not in ("P>D", "D>P"):
            continue
        newest = max(newest, float(t))
        if v is not None and d.get("what") != "none":
            out.append("weg2_flip_time_ms%s %s %d" % (_lbl(dict(base, dir=d["dir"])), repr(float(v)), int(float(t) * 1000)))
    for u in ipc.get("flip_user_time") or []:
        t, v = u.get("start_ts"), u.get("flip_user_ms")
        if t is None or v is None or t <= since_ts:
            continue
        newest = max(newest, float(t))
        out.append("weg2_flip_user_ms%s %s %d" % (_lbl(dict(base, dir="D>P")), repr(float(v)), int(float(t) * 1000)))
    return out, newest


def push(lines: Iterable[str], url: str = DEFAULT_URL, timeout: float = 4.0) -> int:
    body = ("\n".join(lines) + "\n").encode()
    if len(body) <= 1:
        return 0
    req = urllib.request.Request(url.rstrip("/") + "/api/v1/import/prometheus", data=body, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return body.count(b"\n")


class Bridge:
    """The sampler's push loop: every PUSH_S the live boots' IPC into VictoriaMetrics."""

    def __init__(self, boots, url: str = DEFAULT_URL):
        self.boots, self.url = boots, url
        self.flip_seen: Dict[str, float] = {}
        self.view_done: set = set()
        self.last: dict = {"t": None, "lines": 0, "error": None}

    def tick(self, now: Optional[float] = None) -> int:
        from . import history, ipcstate
        now = now or time.time()
        b = self.boots
        with b.ipc.lock:
            items = [(d, st) for d, st in b.ipc._st.items() if st.get("kind") == "boot"]
        lines: List[str] = []
        for d, st in items:
            ipc = ipcstate.boot_view(d, st, b.ipc._ev.get(d), now)
            key = ipc.get("boot_id") or d
            model = history.model_of_ipc(ipc)
            if key not in self.flip_seen:
                # first sight (also after a sampler restart): every flip still in the events, VM dedups
                self.flip_seen[key] = 0.0
            pts, self.flip_seen[key] = flip_points(ipc, model, self.flip_seen[key])
            lines += pts
            try:      # Flipzeit in Nutzersicht (Vorlauf/Layer/Nachlauf) aus den Rang-Segmenten des Rings
                from . import ipcboot
                m = b.model(key)
                if m is not None and m.ring:
                    segs = ipcboot.timeline_view(m, not ipc.get("terminal"), None, now, ipcboot.boot_start(ipc))["segs"]
                    lines += flip_view_points(ipcboot.flip_views(segs, ipc, now), model, short_boot(key), self.view_done)
            except Exception as e:  # noqa: BLE001 -- one boot's view never stops the push
                self.last["flip_view_error"] = "%s: %s" % (type(e).__name__, e)
            if ipc.get("terminal"):
                continue
            with b.lock:
                rank = (b.rank.get(key) or {}).get("rankstats") or {}
            lines += lines_for_boot(ipc, rank, model, int(now * 1000))
        n = push(lines, self.url)
        self.last = {"t": now, "lines": n, "error": None}
        return n

    def run_forever(self, stop) -> None:
        while not stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 -- VM down must not stop the sampler
                self.last = dict(self.last, error="%s: %s" % (type(e).__name__, e), t=time.time())
            stop.wait(PUSH_S)


# ----------------------------------------------------------------------------- reading (PromQL)

class VmClient:
    """rigdash reads VictoriaMetrics per PromQL HTTP API (``/api/v1/query``)."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 2.0):
        self.url, self.timeout = url.rstrip("/"), timeout

    def query(self, promql: str, t: Optional[float] = None) -> List[dict]:
        from urllib.parse import urlencode
        q = {"query": promql}
        if t is not None:
            q["time"] = "%.3f" % t
        with urllib.request.urlopen(self.url + "/api/v1/query?" + urlencode(q), timeout=self.timeout) as r:
            d = json.loads(r.read())
        if d.get("status") != "success":
            raise RuntimeError(d.get("error") or "VM query failed")
        return d["data"]["result"]

    def query_range(self, promql: str, start: float, end: float, step: float) -> Dict[int, float]:
        """{unix_s: value} of the (summed) result of a range query."""
        from urllib.parse import urlencode
        q = {"query": promql, "start": "%.3f" % start, "end": "%.3f" % end, "step": "%ds" % max(1, int(step))}
        with urllib.request.urlopen(self.url + "/api/v1/query_range?" + urlencode(q), timeout=self.timeout) as r:
            d = json.loads(r.read())
        if d.get("status") != "success":
            raise RuntimeError(d.get("error") or "VM query_range failed")
        out: Dict[int, float] = {}
        for s in d["data"]["result"]:
            for t, v in s.get("values") or []:
                try:
                    out[int(round(float(t)))] = float(v)
                except (TypeError, ValueError):
                    pass
        return out

    def query_range_by(self, promql: str, start: float, end: float, step: float, label: str) -> Dict[str, Dict[int, float]]:
        """{label value: {unix_s: value}} of a range query."""
        from urllib.parse import urlencode
        q = {"query": promql, "start": "%.3f" % start, "end": "%.3f" % end, "step": "%ds" % max(1, int(step))}
        with urllib.request.urlopen(self.url + "/api/v1/query_range?" + urlencode(q), timeout=self.timeout) as r:
            d = json.loads(r.read())
        if d.get("status") != "success":
            raise RuntimeError(d.get("error") or "VM query_range failed")
        out: Dict[str, Dict[int, float]] = {}
        for ser in d["data"]["result"]:
            k = (ser.get("metric") or {}).get(label, "")
            for t, v in ser.get("values") or []:
                try:
                    out.setdefault(k, {})[int(round(float(t)))] = float(v)
                except (TypeError, ValueError):
                    pass
        return out

    def scalar_by(self, promql: str, label: str) -> Dict[str, float]:
        out = {}
        for s in self.query(promql):
            try:
                out[s["metric"].get(label, "")] = float(s["value"][1])
            except (KeyError, ValueError, TypeError):
                pass
        return out


#: the first tiles out of VictoriaMetrics (Nutzer-Order 01.10.): TTFT of the users per model, the cards' power
TILE_QUERIES = {
    "ttft_mean_5m_ms": ("sum by (model) (increase(weg2_front_ttft_ms_sum[5m])) / "
                        "sum by (model) (increase(weg2_front_ttft_count[5m]))", "model"),
    "ttft_n_5m": ("sum by (model) (increase(weg2_front_ttft_count[5m]))", "model"),
    # instant = only the boots the sampler still pushes (live); a stopped boot goes stale
    "ttft_mean_boot_ms": ("sum by (model) (weg2_front_ttft_ms_sum) / sum by (model) (weg2_front_ttft_count)", "model"),
    "ttft_max_boot_ms": ("max by (model) (weg2_front_ttft_ms_max)", "model"),
    "power_sum_w": ("sum(nvidia_smi_power_draw_watts)", ""),
}


def tiles(client: VmClient) -> dict:
    out: dict = {"src": "VictoriaMetrics " + client.url, "error": None}
    try:
        for k, (q, lab) in TILE_QUERIES.items():
            out[k] = client.scalar_by(q, lab) if lab else next(iter(client.scalar_by(q, "").values()), None)
        out["ttft_last"] = ttft_last(client)
    except Exception as e:  # noqa: BLE001 -- the page says so instead of a number
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def ttft_series(client: VmClient, model: str, ts: List[int], step: int) -> dict:
    """TTFT der Nutzer je Eimer des Verlaufs (history.view ts/step) aus VictoriaMetrics: Mittel der Anfragen,
    deren erstes Token im Eimer [t, t+step) kam (increase ueber das Fenster, ausgewertet an t+step), und ihre
    Zahl. Ein Eimer ohne Anfrage ist eine Luecke."""
    if not ts:
        return {"mean_ms": [], "n": [], "error": None}
    # window = bucket (no overlap: MetricsQL increase() also uses the last sample before the window); the
    # bridge pushes every 5 s, so a finer bucket (zoom) reads a 5-s window
    w = max(int(step), 5)
    sel = 'model="%s"' % model
    try:
        s = client.query_range("sum(increase(weg2_front_ttft_ms_sum{%s}[%ds]))" % (sel, w), ts[0] + step, ts[-1] + step, step)
        n = client.query_range("sum(increase(weg2_front_ttft_count{%s}[%ds]))" % (sel, w), ts[0] + step, ts[-1] + step, step)
    except Exception as e:  # noqa: BLE001 -- the chart says so
        return {"mean_ms": [None] * len(ts), "n": [None] * len(ts), "error": "%s: %s" % (type(e).__name__, e)}
    mean, cnt = [], []
    for t in ts:
        k = t + step
        nn, ss = n.get(k), s.get(k)
        ok = nn is not None and ss is not None and nn >= 0.5
        mean.append(ss / nn if ok else None)
        cnt.append(nn if nn is not None and nn >= 0.5 else None)
    return {"mean_ms": mean, "n": cnt, "error": None, "window_s": w,
            "src": "VictoriaMetrics weg2_front_ttft_* (state.json front.arrival_seat, LEG2-FIRST-CONTENT)"}


def flip_view_points(views: List[dict], model: str, boot: str, done_keys: set) -> List[str]:
    """Flipzeit in Nutzersicht (ipcboot.flip_views) als Punkte zum flip_begin: weg2_flip_user_view_ms{dir,part}
    mit part = total|vorlauf|layer|nachlauf|d_extend; nur gemessene Flips (kind ok), jeder einmal."""
    out = []
    for x in views:
        if x.get("kind") != "ok" or x.get("total_ms") is None or x.get("begin") is None:
            continue
        key = (boot, round(float(x["begin"]), 3))
        if key in done_keys:
            continue
        done_keys.add(key)
        ts = int(float(x["begin"]) * 1000)
        for part, k in (("total", "total_ms"), ("vorlauf", "vorlauf_ms"), ("layer", "layer_ms"),
                        ("nachlauf", "nachlauf_ms"), ("d_extend", "nachlauf_d_extend_ms")):
            v = x.get(k)
            if v is not None:
                out.append("weg2_flip_user_view_ms%s %s %d" % (_lbl({"model": model, "boot": boot, "dir": x["dir"], "part": part}),
                                                              repr(float(v)), ts))
    return out


def ttft_last(client: "VmClient", now: Optional[float] = None, span_s: int = 900) -> Dict[str, dict]:
    """Der LETZTE TTFT-Wert je Modell aus den 5-s-Proben der Bruecke: der juengste Takt mit Zuwachs von
    ttft_count; bei genau einer Anfrage im Takt ist es ihr exakter Wert, sonst das Mittel dieser n (gesagt)."""
    import time as _t
    now = now or _t.time()
    s = client.query_range_by("sum by (model) (weg2_front_ttft_ms_sum)", now - span_s, now, 5, "model")
    n = client.query_range_by("sum by (model) (weg2_front_ttft_count)", now - span_s, now, 5, "model")
    out = {}
    for m, cs in n.items():
        ss = s.get(m) or {}
        ts = sorted(cs)
        for a, b in zip(reversed(ts[:-1]), reversed(ts[1:])):
            dn = cs[b] - cs[a]
            if dn >= 0.5 and a in ss and b in ss:
                out[m] = {"ms": (ss[b] - ss[a]) / dn, "n": int(round(dn)), "t": b, "exact": round(dn) == 1}
                break
    return out
