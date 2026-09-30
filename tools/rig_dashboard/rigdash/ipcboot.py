"""The boot cards from IPC only -- no boot log is opened (Nutzer 30.09.: "keine IPC über Logs",
NF-Operator 30.09.: "Log-Rückfall für ALLE Werte entfernen. Keine Anzeige liest mehr ein Boot-Log").

Replaces live.LiveLogs as the source of /api/live ``boots``.  Per boot state dir
(``/spinning/docker-acceptance/<line>/state/<boot_id>/``):

  state.json          lifecycle, cause, front mirror (awake, queue, served, served_tokens, d_phase_n,
                      groups, errors), groups.<G>.launch/form, tag, rev, profile, image, container
  events.jsonl        flip_begin / flip_done / flip_first_work / group_health / rank_stop / group_ready ...
  rankstate/<G>/*.rankstats   weg2.rankstats/1 per rank, timer-written: cumulative counters

Every second a sample of every non-terminal boot's rank counters goes into a 16-min ring; the rates,
windows, bursts, 15-min curves and the phase bar are deltas over that ring.  The view keeps the keys
the page already reads (prefill/decode/totals/cache/series/timeline/flip_times/...), so the card
renders unchanged; a value without an IPC source is None and the page says "fehlt in IPC" with the
writer that would have to write it (ipcfields.MISSING_WRITER).  Pure view functions, unit-tested.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from . import ipcfields, ipcstate, stops

SAMPLE_S = 1.0
RING_S = 16 * 60.0
WINDOW_S = 60.0
BUCKET_S = 5.0
SPAN_S = 900.0
STALE_S = 20.0          # a rank file older than this: the rank gives no sign of life
BURST_GAP_S = 5.0       # idle stretches up to this long do not end a burst
MAX3_S = 3.0
MAX3_WIN_S = 120.0
FIRST_TOKEN_HEADLINE_FOR_27B = False     # same rule as the log reader had (README: Flipzeit)
INSTRUMENTS = {"first_token": "P-Ende→erstes Token (flip_first_work)", "flip_total": "flip_ms (flip_done)"}


def _n(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _grp(key: str) -> str:
    return key.split(".", 1)[0]


def compact(rec: dict) -> dict:
    """The counters of one rankstats record the view needs (cumulative ones and levels)."""
    pre = rec.get("prefill") if isinstance(rec.get("prefill"), dict) else {}
    dec = rec.get("decode") if isinstance(rec.get("decode"), dict) else {}
    tok = rec.get("tokens") if isinstance(rec.get("tokens"), dict) else {}
    sch = rec.get("sched") if isinstance(rec.get("sched"), dict) else {}
    cache = rec.get("cache") if isinstance(rec.get("cache"), dict) else {}
    pf = cache.get("prefetch") if isinstance(cache.get("prefetch"), dict) else {}
    cap = rec.get("cap") if isinstance(rec.get("cap"), dict) else {}
    spec = rec.get("spec") if isinstance(rec.get("spec"), dict) else {}
    work = rec.get("work") if isinstance(rec.get("work"), dict) else {}
    first = lambda *xs: next((x for x in xs if x is not None), None)  # noqa: E731
    return {
        "ts": _n(rec.get("ts")),
        "pnew": first(_n(pre.get("new_tokens")), _n(tok.get("prefill_total"))),
        "pcached": _n(pre.get("cached_tokens")),
        "pchunks": _n(pre.get("chunks")),
        "pcomp": first(_n(pre.get("compute_ms")), _n(pre.get("gpu_ms"))),
        "plast_t": _n((pre.get("last") or {}).get("t")) if isinstance(pre.get("last"), dict) else None,
        "dtok": first(_n(dec.get("tokens")), _n(tok.get("decode_total"))),
        "rounds": _n(dec.get("rounds")),
        "dgpu": _n(dec.get("gpu_ms")),
        "running": first(_n(dec.get("running")), _n(sch.get("running_req"))),
        "acc_len": first(_n(dec.get("accept_len_ewma")),
                         (_n(spec.get("accept_tokens_total")) / _n(spec.get("forward_ct_total")))
                         if _n(spec.get("forward_ct_total")) else None),
        "acc_rate": _n(dec.get("accept_rate_ewma")),
        "cuda_graph": dec.get("cuda_graph"),
        "by_bs": dec.get("gpu_ms_by_bs") if isinstance(dec.get("gpu_ms_by_bs"), dict) else None,
        "kv": _n(sch.get("full_token_usage")),
        "queue": _n(sch.get("queue_req")),
        "pending": _n(sch.get("pending_tokens")),
        "fwd": _n(work.get("forward_ct")),
        "lb_n": _n(cache.get("loadback_n")), "lb_tok": _n(cache.get("loadback_tok")),
        "mamba_n": _n(cache.get("mamba_resume_n")), "l3inc_n": _n(cache.get("store_incomplete_n")),
        "pf_refused": _n(pf.get("refused")), "pf_timeout": _n(pf.get("timeout")),
        "cap_kv": _n(cap.get("kv_tokens")), "cap_seats": _n(cap.get("seats")),
    }


def _front_small(front: dict) -> dict:
    keep = ("state", "awake", "queue", "outstanding", "served", "served_tokens", "d_phase_n", "d_parked_n",
            "d_seats", "epoch", "ts")
    return {k: front.get(k) for k in keep if k in (front or {})}


def first_key(keys, g: str) -> Optional[str]:
    k = "%s.tp0pp0" % g
    if k in keys:
        return k
    ks = sorted(x for x in keys if _grp(x) == g)
    return ks[0] if ks else None


def _d(b: dict, a: dict, f: str) -> Optional[float]:
    x, y = b.get(f), a.get(f)
    if x is None or y is None:
        return None
    return x - y if x >= y else x          # a counter that went backwards restarted


def pairs(ring, key: str) -> List[Tuple[float, float, dict, dict]]:
    """(t_a, t_b, rec_a, rec_b) over consecutive samples of one rank whose file advanced; t = the
    dashboard's sample clock (the rank's own ts is the rate denominator, see _rate)."""
    out, prev = [], None
    for s in ring:
        r = s["r"].get(key)
        if r is None or r.get("ts") is None:
            continue
        if prev is not None and r["ts"] > prev[1]["ts"]:
            out.append((prev[0], s["t"], prev[1], r))
        if prev is None or r["ts"] > prev[1]["ts"]:
            prev = (s["t"], r)
    return out


def _dts(p) -> float:
    return p[3]["ts"] - p[2]["ts"]


def _sum(ps, f):
    return sum((_d(p[3], p[2], f) or 0.0) for p in ps)


def burst_of(ps, f: str) -> Optional[List]:
    """The newest active stretch of pairs (f advanced), idle gaps up to BURST_GAP_S inside."""
    out, gap = [], 0.0
    for p in reversed(ps):
        if (_d(p[3], p[2], f) or 0) > 0:
            out.append(p)
            gap = 0.0
        elif out:
            gap += _dts(p)
            if gap > BURST_GAP_S:
                break
    return list(reversed(out)) or None


def max3(ps, f: str, now: float) -> Optional[float]:
    best, tok, dt = None, 0.0, 0.0
    for p in ps:
        if p[1] < now - MAX3_WIN_S:
            continue
        tok += _d(p[3], p[2], f) or 0.0
        dt += _dts(p)
        if dt >= MAX3_S:
            r = tok / dt
            best = r if best is None or r > best else best
            tok, dt = 0.0, 0.0
    return best


def one_s(ps, f: str, now: float) -> float:
    if not ps or ps[-1][1] < now - 3 * SAMPLE_S:
        return 0.0
    p = ps[-1]
    return (_d(p[3], p[2], f) or 0.0) / _dts(p)


# ----------------------------------------------------------------------------- the view parts

def _prefill_burst(ring, keys, g, ps) -> Optional[dict]:
    if not ps:
        return None
    t0, t1 = ps[0][0], ps[-1][1]
    tok, chunks, cached = _sum(ps, "pnew"), _sum(ps, "pchunks"), _sum(ps, "pcached")
    wall = sum(_dts(p) for p in ps)
    ranks = {}
    for k in sorted(x for x in keys if _grp(x) == g):
        rp = [p for p in pairs(ring, k) if p[0] >= t0 - 0.5 and p[1] <= t1 + 0.5]
        n, ms = _sum(rp, "pnew"), _sum(rp, "pcomp")
        if ms > 0:
            ranks[k.split(".", 1)[1]] = {"tps": n / (ms / 1000.0)}
    gpu = min((r["tps"] for r in ranks.values()), default=None)
    return {"tokens": tok, "chunks": chunks, "cached": cached, "mean_chunk": (tok / chunks) if chunks else None,
            "wall_s": wall, "wall_tps": (tok / wall) if wall > 0 else None, "ranks": ranks, "tps_gpu": gpu,
            "tps": gpu, "wall_confounded_tps": None}


def prefill_view(ring, keys, g, now) -> Optional[dict]:
    k = first_key(keys, g)
    if k is None:
        return None
    ps = pairs(ring, k)
    win = [p for p in ps if p[1] >= now - WINDOW_S]
    act = [p for p in win if (_d(p[3], p[2], "pnew") or 0) > 0]
    last = ring[-1]["r"].get(k) or {}
    return {"window_s": WINDOW_S, "one_s": one_s(ps, "pnew", now), "max3s_120": max3(ps, "pnew", now),
            "now": _prefill_burst(ring, keys, g, act) if act else None,
            "last_burst": _prefill_burst(ring, keys, g, burst_of(ps, "pnew")),
            "last_t": last.get("plast_t"), "queue": last.get("queue"), "pending_tok": last.get("pending")}


def decode_view(ring, keys, g, front, now) -> Optional[dict]:
    k = first_key(keys, g)
    if k is None:
        return None
    ps = pairs(ring, k)
    last = ring[-1]["r"].get(k) or {}
    if not ps and not last.get("dtok"):
        return None
    win = [p for p in ps if p[1] >= now - WINDOW_S]
    act = [p for p in win if (_d(p[3], p[2], "dtok") or 0) > 0]
    rate = lambda xs: (_sum(xs, "dtok") / sum(_dts(p) for p in xs)) if xs else None  # noqa: E731
    gms = _sum(act, "dgpu")
    lb = burst_of(ps, "dtok")
    by_bs = {str(bs): {"median_ms": round(v[1] / v[0], 1), "n": int(v[0]), "stat": "Mittel"}
             for bs, v in sorted((last.get("by_bs") or {}).items(), key=lambda x: int(x[0]))
             if isinstance(v, (list, tuple)) and len(v) == 2 and v[0]}
    seats = (front or {}).get("d_seats")
    if not seats and (front or {}).get("d_phase_n") is not None:
        seats = {"n": front.get("d_phase_n"), "parked_n": front.get("d_parked_n"), "cap": last.get("cap_seats")}
    return {"window_s": WINDOW_S, "gen_tps": rate(act), "gen_tps_last": rate(lb) if lb else None,
            "rows": len(act), "last_t": lb[-1][1] if lb else None, "one_s": one_s(ps, "dtok", now),
            "max3s_120": max3(ps, "dtok", now), "running": last.get("running"), "accept_len": last.get("acc_len"),
            "accept_rate": last.get("acc_rate"), "cuda_graph": last.get("cuda_graph"), "full_use": last.get("kv"),
            "queue_req": last.get("queue"), "max_running": last.get("cap_seats"),
            "max_total_tokens": last.get("cap_kv"), "seats": seats or None, "round_bs": None,
            "compute_tps": (_sum(act, "dtok") / (gms / 1000.0)) if gms > 0 else None, "round_ms_by_bs": by_bs}


def series_view(ring, keys, groups, now) -> dict:
    n = int(SPAN_S // BUCKET_S)
    t0 = (now // BUCKET_S) * BUCKET_S - (n - 1) * BUCKET_S
    ts = [t0 + i * BUCKET_S for i in range(n)]
    start = ring[0]["t"] if ring else now
    out = {"t": ts}
    for g in groups:
        for f, name in (("pnew", "prefill"), ("dtok", "decode")):
            k = first_key(keys, g)
            if k is None or (name == "decode" and g == "P"):
                continue
            acc = [None if t + BUCKET_S <= start else 0.0 for t in ts]
            for p in pairs(ring, k):
                i = int((p[1] - t0) // BUCKET_S)
                if 0 <= i < n and acc[i] is not None:
                    acc[i] += _d(p[3], p[2], f) or 0.0
            out["%s_%s_tps" % (g, name)] = [None if v is None else v / BUCKET_S for v in acc]
    return out


def timeline_view(ring, keys, groups, flip_done, live, now) -> dict:
    """Phase bar from the samples: per sample step the class that worked (P prefill, D prefill,
    decode) or idle (with the awake group); flips painted from flip_done events (begin -> done)."""
    kP = first_key(keys, "P") or first_key(keys, "single")
    kD = first_key(keys, "D")
    kdec = kD or first_key(keys, "single")
    by_t = {}
    for key, cls, f in ((kP, "P", "pnew"), (kD, "D", "pnew"), (kdec, "dec", "dtok")):
        if key is None:
            continue
        for p in pairs(ring, key):
            d = _d(p[3], p[2], f) or 0.0
            if d > 0:
                cur = by_t.setdefault((p[0], p[1]), {})
                cur[cls] = cur.get(cls, 0.0) + d
    segs: List[dict] = []
    samples = list(ring)
    for a, b in zip(samples, samples[1:]):
        if b["t"] < now - SPAN_S:
            continue
        work = by_t.get((a["t"], b["t"]), {})
        k = "P" if work.get("P") else "D" if work.get("D") else "dec" if work.get("dec") else "idle"
        awake = (b.get("front") or {}).get("awake")
        tok = work.get(k, 0.0)
        if segs and segs[-1]["k"] == k and abs(segs[-1]["e"] - a["t"]) < 3 * SAMPLE_S and \
                (k != "idle" or segs[-1].get("awake") == awake):
            segs[-1]["e"] = b["t"]
            segs[-1]["tok"] += tok
            segs[-1]["n"] += 1
        else:
            segs.append({"s": a["t"], "e": b["t"], "k": k, "tok": tok, "n": 1, "awake": awake if k == "idle" else None})
    for fd in flip_done or []:
        s, e = fd.get("flip_begin_ts"), fd.get("t")
        if s is None or e is None or e < now - SPAN_S:
            continue
        out = []
        for x in segs:
            if x["e"] <= s or x["s"] >= e:
                out.append(x)
                continue
            if x["s"] < s:
                out.append(dict(x, e=s))
            if x["e"] > e:
                out.append(dict(x, s=e))
        out.append({"s": s, "e": e, "k": "flip", "total_ms": fd.get("flip_ms"), "drain_ms": fd.get("drain_quiesce_ms"),
                    "slept": fd.get("sleep"), "woke": fd.get("wake"), "n": 0, "tok": 0})
        segs = sorted(out, key=lambda x: x["s"])
    for x in segs:
        dur = max(1e-6, x["e"] - x["s"])
        if x["k"] in ("P", "D", "dec"):
            x["tps"] = x["tok"] / dur
            if x["k"] == "dec":
                x["rounds"] = None
    if segs and live and segs[-1]["k"] != "flip":
        segs[-1]["running"] = True
    t1 = now if live or not ring else ring[-1]["t"]
    return {"segs": segs, "span_s": SPAN_S, "t1": t1}


def _q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def flip_times_view(first_work: List[dict], flip_done: List[dict], is_27b: bool) -> dict:
    out = {"instruments": INSTRUMENTS,
           "headline": "flip_total" if is_27b and not FIRST_TOKEN_HEADLINE_FOR_27B else "first_token"}
    for d in ("P>D", "D>P"):
        fw = sorted((x for x in first_work if x.get("dir") == d and x.get("flip_time_ms") is not None),
                    key=lambda x: x.get("flip_begin_ts") or 0)
        sl, wk = d.split(">")
        fd = [x for x in flip_done if x.get("sleep") == sl and x.get("wake") == wk and x.get("flip_ms") is not None]
        vals = [float(x["flip_time_ms"]) for x in fw]
        lay = [float(x["flip_ms"]) for x in fd]
        out[d] = {"n": len(vals), "last": vals[-1] if vals else None, "last_t": fw[-1].get("flip_begin_ts") if fw else None,
                  "median": _q(vals, 0.5), "p90": _q(vals, 0.9), "resolution_s": 0.001, "open": False,
                  "src": "IPC flip_first_work (%s)" % (fw[-1].get("what") or "?") if fw else None,
                  "layer_n": len(lay), "layer_newest": lay[-1] if lay else None,
                  "layer_median": _q(lay, 0.5), "layer_p90": _q(lay, 0.9)}
    done_by_begin = {round(x.get("flip_begin_ts") or 0, 1): x for x in flip_done}
    recent = []
    for x in sorted(first_work, key=lambda x: x.get("flip_begin_ts") or 0)[-24:]:
        fd = done_by_begin.get(round(x.get("flip_begin_ts") or 0, 1)) or {}
        recent.append({"t": x.get("flip_begin_ts"), "dir": x.get("dir"), "ms": x.get("flip_time_ms"),
                       "layer_ms": fd.get("flip_ms"), "state": "ok", "src": "ipc"})
    out["recent"] = recent
    return out


def totals_view(ring, keys, front, boot_t0, now) -> dict:
    last = ring[-1]["r"] if ring else {}
    wall = max(1.0, now - boot_t0) if boot_t0 else None

    def grp(g, f, comp):
        k = first_key(last.keys(), g)
        if k is None:
            return None, None, None, None
        tok = last[k].get(f)
        rates = []
        for kk, r in last.items():
            if _grp(kk) == g and r.get(f) and r.get(comp):
                rates.append(r[f] / (r[comp] / 1000.0))
        chunks = last[k].get("pchunks") if f == "pnew" else None
        return tok, chunks, (min(rates) if rates else None), ((tok / wall) if tok is not None and wall else None)

    pg = "P" if first_key(last.keys(), "P") else "single"
    p_new, p_chunks, p_gpu, p_wall = grp(pg, "pnew", "pcomp")
    d_new, d_chunks, d_gpu, d_wall = grp("D", "pnew", "pcomp")
    dg = "D" if first_key(last.keys(), "D") else "single"
    dec, _, dec_gpu, dec_wall = grp(dg, "dtok", "dgpu")
    served = (front or {}).get("served") or {}
    return {"p_new": p_new, "p_chunks": p_chunks, "p_rate_gpu": p_gpu, "p_rate_wall": p_wall,
            "d_new": d_new, "d_chunks": d_chunks, "d_rate_gpu": d_gpu, "d_rate_wall": d_wall,
            "decoded": dec, "dec_rate_gpu": dec_gpu, "dec_rate_wall": dec_wall,
            "served_requests": served.get("D") if isinstance(served, dict) else None,
            "boot_wall_s": wall, "read_progress": 1.0}


def cache_view(ring, keys, now) -> dict:
    out = {}
    last = ring[-1] if ring else None
    if not last:
        return out
    for g in ("P", "D", "single"):
        k = first_key(last["r"].keys(), g)
        if k is None:
            continue
        ps = [p for p in pairs(ring, k) if p[1] >= now - WINDOW_S]
        cached, new = _sum(ps, "pcached"), _sum(ps, "pnew")
        lbt = _sum(ps, "lb_tok")
        w = {"chunks": _sum(ps, "pchunks"), "cached": cached, "new": new,
             "hit_share": cached / (cached + new) if cached + new else None,
             "loadback_n": _sum(ps, "lb_n"), "loadback_tok": lbt,
             "l2_share": lbt / (cached + new) if cached + new and lbt else None}
        r = last["r"][k]
        bc, bn = r.get("pcached") or 0.0, r.get("pnew") or 0.0
        b = {"chunks": r.get("pchunks"), "cached": bc, "new": bn, "hit_share": bc / (bc + bn) if bc + bn else None,
             "loadback_n": r.get("lb_n"), "loadback_tok": r.get("lb_tok"),
             "l2_share": (r.get("lb_tok") or 0) / (bc + bn) if bc + bn and r.get("lb_tok") else None,
             "mamba_n": r.get("mamba_n"), "l3inc_n": r.get("l3inc_n"), "l3inc_share": None,
             "prefetch_refused": r.get("pf_refused"), "prefetch_timeout": r.get("pf_timeout")}
        out[g] = {"window": w, "boot": b}
    st_now = ((last.get("front") or {}).get("served_tokens")) or {}
    old = next((s for s in ring if s["t"] >= now - WINDOW_S), None)
    st_old = ((old or {}).get("front") or {}).get("served_tokens") or {}
    for g in ("P", "D"):
        row = st_now.get(g)
        if not isinstance(row, dict):
            continue
        o = st_old.get(g) if isinstance(st_old.get(g), dict) else {}
        pr, ca = float(row.get("prompt") or 0), float(row.get("cached") or 0)
        wp, wc = pr - float(o.get("prompt") or 0), ca - float(o.get("cached") or 0)
        out["served_" + g] = {"boot": {"n": row.get("n"), "prompt": pr, "cached": ca, "req_hit_share": ca / pr if pr else None},
                              "window": {"prompt": wp, "cached": wc, "req_hit_share": wc / wp if wp > 0 else None}}
    return out


def boot_start(ipc: dict) -> Optional[float]:
    for part in (ipc.get("boot_id") or "").split("-"):
        if len(part) == 16 and part[8] == "T" and part.endswith("Z"):
            try:
                import datetime
                return datetime.datetime.strptime(part, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=datetime.timezone.utc).timestamp()
            except ValueError:
                return None
    return ipc.get("serving_since_ts")


def ring_rates(ring) -> dict:
    """C7: per rank the rates of its newest sample step (Δ over the rank's own clock), from the ring."""
    out = {}
    keys = set()
    for s in ring:
        keys.update(s["r"].keys())
    for k in keys:
        ps = pairs(ring, k)
        if not ps:
            continue
        p = ps[-1]
        dt = _dts(p)
        r = {"dt_s": round(dt, 3)}
        for name, f in (("prefill_tps", "pnew"), ("decode_tps", "dtok")):
            d = _d(p[3], p[2], f)
            if d is not None:
                r[name] = round(d / dt, 1)
        dn, dc = _d(p[3], p[2], "pnew"), _d(p[3], p[2], "pcomp")
        if dn is not None and dc:
            r["prefill_tps_gpu"] = round(dn / (dc / 1000.0), 1)
        out[k] = r
    return out


FLIP_KEYS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "D3", "F3")


def build_view(ipc: dict, ring, rank: dict, rates: dict, now: float) -> dict:
    """One boot card from its IPC alone.  Pure over its inputs."""
    ring = list(ring or [])
    keys = set()
    for s in ring:
        keys.update(s["r"].keys())
    last = ring[-1] if ring else {"t": now, "r": {}, "front": {}}
    front = dict(ipc.get("front") or {})
    groups = sorted({_grp(k) for k in keys})
    fresh = [r["ts"] for r in last["r"].values() if r.get("ts")]
    newest = max(fresh) if fresh else None
    hb = ipc.get("heartbeat_age_s")
    sig = newest if newest is not None else ((now - hb) if hb is not None else None)
    live = not ipc.get("terminal") and sig is not None and now - sig < STALE_S
    evs = ipc.get("ipc_events") or []
    flip_done = [dict(e.get("data") or {}, t=(e.get("data") or {}).get("t") or e.get("ts"))
                 for e in evs if e.get("type") == "flip_done"]
    fw = list(ipc.get("flip_first_work") or [])
    model = ipc.get("model") or ""
    tag = ipc.get("tag") or ""
    is27 = "27b" in (model + " " + tag + " " + (ipc.get("dir") or "")).lower()
    fields = ipcfields.resolve(ipc, rank, {}, rates or ring_rates(ring))
    if not any(e.get("type") in ("flip_begin", "flip_done", "flip_first_work") for e in (ipc.get("ipc_events") or [])) \
            and not ipc.get("flip_first_work"):
        # no flip yet in this boot: the flip fields are empty, not missing a writer
        for k in FLIP_KEYS:
            if fields[k]["src"] != "ipc":
                fields[k]["empty"] = "noch kein Flip in diesem Boot"
    if fields["C7"]["src"] != "ipc" and (ipc.get("terminal") or len(ring) < 2):
        fields["C7"]["empty"] = "keine Raten: Boot beendet" if ipc.get("terminal") else "erste Probe, Raten ab der zweiten"
    last_act = {}
    for g in groups:
        ts = [p[1] for k in keys if _grp(k) == g for p in pairs(ring, k)
              if any((_d(p[3], p[2], f) or 0) > 0 for f in ("pnew", "dtok", "fwd"))]
        last_act[g] = max(ts) if ts else None
    a12 = (fields.get("A12") or {}).get("value") or {}
    health = {g: {"t": v.get("ts"), "alive": v.get("alive"), "http_ok": v.get("http_ok"), "streak": v.get("streak"),
                  "group": g, "src": "front.groups"} for g, v in a12.items() if isinstance(v, dict) and v.get("ts")}
    a13 = (fields.get("A13") or {}).get("value") or {}
    a14 = (fields.get("A14") or {}).get("value") or []
    b6 = (fields.get("B6") or {}).get("value") or {}
    t0 = boot_start(ipc)
    v = {
        "stem": ipc.get("boot_id") or ipc.get("dir"),
        "meta": {"tag": tag, "model": model, "topology": ipc.get("topology"), "sha": ipc.get("rev"),
                 "form": " | ".join("%s: %s" % (g, (f.get("describe") or f) if isinstance(f, dict) else f)
                                    for g, f in sorted((ipc.get("forms") or {}).items())) or None},
        "age_s": round(now - sig, 1) if sig is not None else None,
        "last_log_t": sig,              # name kept for the readers: here the newest IPC sign of life
        "first_t": t0,
        "live": live,
        "awake": {"awake": front.get("awake")} if front.get("awake") else None,
        "queue": None,
        "flip_open": bool(b6.get("open")),
        "flips": flip_done[-12:],
        "flip_count": len(flip_done),
        "flip_times": flip_times_view(fw, flip_done, is27),
        "health": health,
        "errors": [{"t": x.get("t"), "group": x.get("group"), "text": x.get("text") or x.get("exc") or ""}
                   for x in (a13.get("last") or [])],
        "error_count": a13.get("n"),
        "stops": [{"t": x.get("t") or x.get("ts"), "group": x.get("group"), "src": "IPC",
                   "text": " ".join(str(x.get(k)) for k in ("code", "reason", "exc", "text") if x.get(k))} for x in a14],
        "stop_count": len(a14),
        "last_activity": last_act,
        "last_activity_any": max([t for t in last_act.values() if t] or [0]) or None,
        "prefill": {g: pv for g in groups if g in ("P", "D", "single")
                    for pv in [prefill_view(ring, keys, g, now)] if pv},
        "decode": {g: dv for g in groups if g in ("D", "single")
                   for dv in [decode_view(ring, keys, g, front, now)] if dv},
        "totals": totals_view(ring, keys, front, t0, now),
        "cache": cache_view(ring, keys, now),
        "series": series_view(ring, keys, [g for g in groups if g in ("P", "D", "single")], now),
        "timeline": timeline_view(ring, keys, groups, flip_done, live, now),
        "fields": ipcfields.for_page(fields),
        "fields_summary": ipcfields.summary(fields),
        "end": stops.classify_ipc(ipc),
    }
    view_ipc = {k: x for k, x in ipc.items() if k not in ("ipc_events", "flip_first_work")}
    v["ipc"] = view_ipc
    return v


class IpcBoots:
    """Samples every boot state dir once a second (rank counters + front mirror) into a ring."""

    def __init__(self, roots=ipcstate.STATE_ROOTS):
        self.ipc = ipcstate.IpcStates(roots)
        self.lock = threading.Lock()
        self.rings: Dict[str, deque] = {}
        self.rank: Dict[str, dict] = {}
        self.final: set = set()
        self.rates = ipcfields.Rates()
        self.last_error: Optional[str] = None

    def poll(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.ipc.poll(now)
        seen = set()
        for v in self.ipc.boots(now):
            key = v.get("boot_id") or v.get("dir")
            seen.add(key)
            if key in self.final:
                continue
            rank = ipcfields.read_rank_files(ipcfields.state_rankstate_dirs(v))
            samp = {"t": now, "r": {k: compact(r) for k, r in rank["rankstats"].items()},
                    "front": _front_small(v.get("front") or {})}
            with self.lock:
                ring = self.rings.setdefault(key, deque())
                ring.append(samp)
                while ring and ring[0]["t"] < now - RING_S:
                    ring.popleft()
                self.rank[key] = rank
                if v.get("terminal"):
                    self.final.add(key)
        with self.lock:
            for k in list(self.rings):
                if k not in seen:
                    self.rings.pop(k, None)
                    self.rank.pop(k, None)
                    self.final.discard(k)

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            t0 = time.time()
            try:
                self.poll(t0)
                self.last_error = None
            except Exception as e:  # keep sampling alive; visible in /api/live
                self.last_error = "%s: %s" % (type(e).__name__, e)
            stop.wait(max(0.05, SAMPLE_S - (time.time() - t0)))

    def snapshot(self, now: Optional[float] = None, max_boots: int = 10) -> List[dict]:
        now = now or time.time()
        with self.ipc.lock:
            items = [(d, st) for d, st in self.ipc._st.items() if st.get("kind") == "boot"]
            mts = dict(self.ipc._mt)
        items.sort(key=lambda x: -mts.get(x[0], 0))
        views = []
        newest_line = {}
        for d, st in items[:max_boots]:
            ipc = ipcstate.boot_view(d, st, self.ipc._ev.get(d), now)
            key = ipc.get("boot_id") or d
            with self.lock:
                ring = list(self.rings.get(key) or ())
                rank = self.rank.get(key) or {"rankstats": {}, "rankstate": {}}
            v = build_view(ipc, ring, rank, None, now)
            line = "27b" if "/27b/" in d else "nf"
            v["primary"] = v["live"] or line not in newest_line
            newest_line.setdefault(line, key)
            views.append(v)
        views.sort(key=lambda v: (not v["live"], not v["primary"], v["age_s"] if v["age_s"] is not None else 1e12))
        return views

    def activity(self, key: str, start: float, n: int, bs: float) -> List[dict]:
        """Per bucket which class computed and how many tokens (energy.EnergyBook), from the ring."""
        out = [{"P": False, "D": False, "dec": False, "P_tok": 0, "D_tok": 0, "dec_tok": 0} for _ in range(n)]
        with self.lock:
            ring = list(self.rings.get(key) or ())
        keys = set()
        for s in ring:
            keys.update(s["r"].keys())
        for g, cls, f in (("P", "P", "pnew"), ("single", "P", "pnew"), ("D", "D", "pnew"),
                          ("D", "dec", "dtok"), ("single", "dec", "dtok")):
            k = first_key(keys, g)
            if k is None:
                continue
            for p in pairs(ring, k):
                i = int((p[1] - start) // bs)
                d = _d(p[3], p[2], f) or 0.0
                if 0 <= i < n and d > 0:
                    out[i][cls] = True
                    out[i][cls + "_tok"] += int(d)
        return out
