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

from . import activity, ipcfields, ipcstate, stops

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
        "plast_gpu": _n((pre.get("last") or {}).get("gpu_ms")) if isinstance(pre.get("last"), dict) else None,
        # FEHLT 7 (d086ac9a71, ab Build y5a): the chunk's own time and pure compute, without the wait
        # behind the previous chunk
        "plast_new": _n((pre.get("last") or {}).get("new")) if isinstance(pre.get("last"), dict) else None,
        "plast_own": _n((pre.get("last") or {}).get("own_ms")) if isinstance(pre.get("last"), dict) else None,
        "plast_conly": _n((pre.get("last") or {}).get("compute_only_ms")) if isinstance(pre.get("last"), dict) else None,
        "last_bs": _n(dec.get("last_bs")),
        "si_del": _n(cache.get("store_incomplete_delivered")), "si_deliv": _n(cache.get("store_incomplete_deliverable")),
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
            "d_seats", "d_cached_tokens", "epoch", "ts")
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

LIVE_BURST_S = 6.0      # a prefill burst whose last chunk ended this recently is "running"


def _rank_gpu_rates(ring, keys, g, t0, t1) -> Dict[str, dict]:
    """Rate per rank over the chunks that ended in [t0, t1].  With FEHLT 7 (build y5a):
    Σ last.new / Σ last.compute_only_ms of the chunks read one by one -- the rank's pure compute
    rate.  Without it: Δnew / Δcompute_ms, which on PP0 includes the wait behind the previous
    chunk -- a lower bound (``exact`` False)."""
    out = {}
    for k in sorted(x for x in keys if _grp(x) == g):
        n = ms = cn = cms = 0.0
        for a, b in activity.rank_pairs(ring, k):
            e = b.get("plast_t") or b["ts"]
            if not (t0 - 0.5 <= e <= t1 + 0.5):
                continue
            n += _d(b, a, "pnew") or 0.0
            ms += _d(b, a, "pcomp") or 0.0
            if (_d(b, a, "pchunks") or 0) == 1 and b.get("plast_conly") and b.get("plast_new"):
                cn += b["plast_new"]
                cms += b["plast_conly"]
        if cms > 0 and cn > 0:
            out[k.split(".", 1)[1]] = {"tps": cn / (cms / 1000.0), "exact": True}
        elif ms > 0 and n > 0:
            out[k.split(".", 1)[1]] = {"tps": n / (ms / 1000.0), "exact": False}
    return out


def _burst_dict(ring, keys, g, b) -> dict:
    ranks = _rank_gpu_rates(ring, keys, g, b["s"], b["e"])
    gpu = min((r["tps"] for r in ranks.values()), default=None) if b["tok"] >= activity.MIN_RATE_TOK else None
    exact = bool(ranks) and all(r.get("exact") for r in ranks.values())
    return {"tokens": b["tok"], "chunks": b["n"], "cached": b["cached"], "mean_chunk": (b["tok"] / b["n"]) if b["n"] else None,
            "wall_s": b["wall_s"], "wall_tps": b["rate"], "ranks": ranks, "tps_gpu": gpu, "tps_gpu_exact": exact,
            "tps": b["rate"],
            "t0": b["s"], "t1": b["e"]}


def prefill_view(m: "activity.Model", g: str, now: float) -> Optional[dict]:
    """Prefill tile.  Rates only over the time the chunks ran (activity.py): the running or last burst
    (P-Ende wall clock: first chunk's start on the first stage -> last chunk's end on the last stage),
    the best burst of the ring, and the chunks that ended in the last 60 s."""
    cs = m.pchunks.get(g)
    if cs is None:
        return None
    bs = activity.bursts(cs)
    last = bs[-1] if bs else None
    running = last is not None and now - last["e"] <= LIVE_BURST_S
    win = [c for c in cs if c["e0"] >= now - WINDOW_S]
    wb = activity.bursts(win)
    wtok = sum(b["tok"] for b in wb)
    wwall = sum(b["wall_s"] for b in wb)
    rated = [b["rate"] for b in bs if b["rate"] is not None]
    k0, _ = activity.stage_keys(m.keys, g)
    lastrec = (m.ring[-1]["r"].get(k0) if m.ring and k0 else None) or {}
    return {"window_s": WINDOW_S,
            "one_s": (last["rate"] or 0.0) if running else 0.0,
            "max3s_120": max(rated) if rated else None,
            "now": ({"tokens": wtok, "chunks": sum(b["n"] for b in wb), "cached": sum(b["cached"] for b in wb),
                     "mean_chunk": wtok / max(1, sum(b["n"] for b in wb)), "wall_s": wwall,
                     "wall_tps": (wtok / wwall) if wwall > 0 and wtok >= activity.MIN_RATE_TOK else None,
                     "ranks": _rank_gpu_rates(m.ring, m.keys, g, now - WINDOW_S, now),
                     "tps_gpu": min((r["tps"] for r in _rank_gpu_rates(m.ring, m.keys, g, now - WINDOW_S, now).values()),
                                    default=None) if wtok >= activity.MIN_RATE_TOK else None}
                    if wb else None),
            "last_burst": _burst_dict(m.ring, m.keys, g, last) if last else None,
            "last_t": last["e"] if last else None, "queue": lastrec.get("queue"), "pending_tok": lastrec.get("pending"),
            "instrument": "Chunks zur Rechenzeit (prefill.last t/gpu_ms), Schub = erster Chunk-Start PP0 bis letztes Chunk-Ende letzte Stufe"}


def decode_view(m: "activity.Model", g: str, front, now: float) -> Optional[dict]:
    if g != m.dec_group:
        return None
    k, _ = activity.stage_keys(m.keys, g)
    last = (m.ring[-1]["r"].get(k) if m.ring else None) or {}
    iv = m.dec
    if not iv and not last.get("dtok"):
        return None
    win = [x for x in iv if x["e"] >= now - WINDOW_S]
    # the rate while decoding: steady intervals only (decode before and after) -- the first and the
    # last interval of a stretch hold idle time the sample cannot place
    sw = [x for x in win if x["steady"]] or win
    tok, dur = sum(x["tok"] for x in sw), sum(x["dur"] for x in sw)
    st = activity.decode_stretches(iv)
    lastst = st[-1] if st else None
    streams = [x["stream"] for x in win if x["stream"] is not None]
    # per stream and seats with the same denominator: tokens / seat-seconds, seat-seconds / decode
    # seconds -- only over the time D decoded (Nutzer 30.09. ~21:10Z: sitze mitführen)
    seat_s = sum(x["seat_s"] for x in sw if x.get("seat_s"))
    busy_s = sum(x["busy"] for x in sw if x.get("seat_s"))
    tok_s = sum(x["tok"] for x in sw if x.get("seat_s"))
    bmin = min((x["bs_min"] for x in sw if x.get("bs_min") is not None), default=None)
    bmax = max((x["bs_max"] for x in sw if x.get("bs_max") is not None), default=None)
    cur = iv[-1] if iv and now - iv[-1]["e"] <= 3 * SAMPLE_S else None
    best, acc_t, acc_d = None, 0.0, 0.0
    for x in iv:                       # best 3-s stretch of the last 120 s, over decode time only
        if x["e"] < now - MAX3_WIN_S:
            continue
        acc_t += x["tok"]
        acc_d += x["dur"]
        if acc_d >= MAX3_S:
            r = acc_t / acc_d
            best = r if best is None or r > best else best
            acc_t = acc_d = 0.0
    gms = sum((x.get("gpu_ms") or 0.0) for x in sw)
    by_bs = {str(bs): {"median_ms": round(v[1] / v[0], 1), "n": int(v[0]), "stat": "Mittel"}
             for bs, v in sorted((last.get("by_bs") or {}).items(), key=lambda x: int(x[0]))
             if isinstance(v, (list, tuple)) and len(v) == 2 and v[0]}
    seats = (front or {}).get("d_seats")
    if not seats and (front or {}).get("d_phase_n") is not None:
        seats = {"n": front.get("d_phase_n"), "parked_n": front.get("d_parked_n"), "cap": last.get("cap_seats")}
    return {"window_s": WINDOW_S, "gen_tps": (tok / dur) if dur >= 2.0 else None,
            "gen_tps_last": lastst["rate"] if lastst else None, "rows": len(win),
            "last_t": lastst["e"] if lastst else None,
            "one_s": (cur["tok"] / cur["busy"]) if cur else 0.0, "max3s_120": best,
            "per_stream": (tok_s / seat_s) if seat_s > 0 else ((sum(streams) / len(streams)) if streams else None),
            "per_stream_n": len(streams),
            "seats_mean": (seat_s / busy_s) if busy_s > 0 else None, "seats_min": bmin, "seats_max": bmax,
            "seats_src": ("rankstats decode.gpu_ms_by_bs (Δ je Probe, nach Rundenzeit gewichtet)"
                          if any(x.get("bs_src") == "by_bs" for x in sw) else
                          "rankstats decode.running" if any(x.get("bs_src") == "running" for x in sw) else None),
            "running": last.get("running"), "last_bs": last.get("last_bs"), "accept_len": last.get("acc_len"),
            "accept_rate": last.get("acc_rate"), "cuda_graph": last.get("cuda_graph"), "full_use": last.get("kv"),
            "queue_req": last.get("queue"), "max_running": last.get("cap_seats"),
            "max_total_tokens": last.get("cap_kv"), "seats": seats or None,
            "round_bs": {"bs": last.get("last_bs")} if last.get("last_bs") is not None else None,
            "compute_tps": (tok / (gms / 1000.0)) if gms > 0 and dur >= 2.0 else None, "round_ms_by_bs": by_bs}


ZOOM_STEPS = (1.0, 2.0, 5.0)
ZOOM_POINTS = 240


def zoom_bucket(span_s: float) -> float:
    """The finest bucket that keeps a zoomed stretch under ZOOM_POINTS (the ring holds 1-s samples)."""
    return next((s for s in ZOOM_STEPS if span_s / s <= ZOOM_POINTS), BUCKET_S)


def series_view(m: "activity.Model", now: float, zoom: Optional[Tuple[float, float]] = None) -> dict:
    """15-min curves of the boot card.  ``*_tps`` = tokens / bucket (wall; the tok/s/W and energy
    figures), ``*_rate`` = tokens / the time the phase worked in the bucket (the curve drawn; Nutzer
    30.09. ~21:05Z: "decode durchsatz ... nicht durchgehend sondern extrem sprunghaft" -- a D-extend
    of 2 s in a 5-s bucket halved the wall rate although decode ran at full speed), ``*_seats`` =
    mean decode seats over the decode time.  ``zoom`` = (t0, t1): that stretch at 1/2/5-s buckets."""
    if zoom is not None:
        # the ring holds RING_S: a longer zoomed stretch (dragged in the days-long history) is cut to it
        zoom = (max(zoom[0], now - RING_S), min(zoom[1], now))
        if zoom[1] - zoom[0] < 1.0:
            zoom = (now - 1.0, now)
        step = zoom_bucket(zoom[1] - zoom[0])
        t0 = (zoom[0] // step) * step
        n = max(1, int(math.ceil((min(zoom[1], now) - t0) / step)))
    else:
        step = BUCKET_S
        n = int(SPAN_S // BUCKET_S)
        t0 = (now // BUCKET_S) * BUCKET_S - (n - 1) * BUCKET_S
    b = m.buckets(t0, n, step)
    out = {"t": [t0 + i * step for i in range(n)], "step": step}
    if zoom is not None:
        out["zoom"] = [zoom[0], zoom[1]]
    if "P" in m.pchunks or "single" in m.pchunks:
        g = "P" if "P" in m.pchunks else "single"
        out[g + "_prefill_tps"] = b["p_tps"]
        out[g + "_prefill_rate"] = b["p_rate"]
    if "D" in m.pchunks:
        out["D_prefill_tps"] = b["d_tps"]
        out["D_prefill_rate"] = b["d_rate"]
    if m.dec_group:
        out[m.dec_group + "_decode_tps"] = b["dec_tps"]
        out[m.dec_group + "_decode_rate"] = b["dec_rate"]
        out[m.dec_group + "_decode_seats"] = b["seats"]
        out[m.dec_group + "_decode_stream"] = b["stream_tps"]
    return out


def timeline_view(m: "activity.Model", live: bool, awake_now, now: float, boot_t0: Optional[float] = None) -> dict:
    """Phase bar = the data-backed segments of activity.Model (Nutzer 30.09.: idle, flip and tail must
    be told apart).  Time inside the window that no sample covers is "unknown" (rigdash did not watch),
    never idle and never flip; before the boot's start nothing is drawn."""
    end = now if live else (m.ring[-1]["t"] if m.ring else now)
    lo = max(now - SPAN_S, boot_t0 or (now - SPAN_S))
    segs = [dict(x) for x in m.segments(end) if x["e"] > lo]
    src = {"P": m.pchunks.get("P", []) + m.pchunks.get("single", []), "D": m.pchunks.get("D", []), "dec": m.dec}
    out, prev = [], lo
    for x in segs:
        x["s"] = max(x["s"], lo)
        if x["s"] > prev + 0.05:
            out.append({"s": prev, "e": x["s"], "k": "unknown", "why": "keine IPC-Probe (rigdash sah den Boot nicht)"})
        if x["k"] in src:
            tok = activity.spread(src[x["k"]], x["s"], 1, max(1e-3, x["e"] - x["s"]))[0]
            x["tok"] = tok
            x["tps"] = tok / max(1e-3, x["e"] - x["s"])
            x["n"] = 1
        if x["k"] == "idle":
            x["awake"] = m.awake_at(x["s"], awake_now)
        out.append(x)
        prev = max(prev, x["e"])
    if end > prev + 0.05:
        out.append({"s": prev, "e": end, "k": "unknown", "why": "noch keine zweite IPC-Probe"})
    if live and out and out[-1]["k"] in ("P", "D", "dec"):
        out[-1]["running"] = True
    if live:
        for x in out[-2:]:
            if x["k"] == "unknown" and x["e"] >= end - 1.5 and (x.get("why") or "").startswith("Rang-Zähler"):
                x["why"] = "Arbeit läuft gerade, der Chunk ist noch nicht fertig -- die Zuordnung folgt mit seinem Ende"
    return {"segs": out, "span_s": SPAN_S, "t1": now if live else end, "overlap": m.overlap_s(),
            "states": list(activity.STATES)}


def _q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def flip_times_view(first_work: List[dict], flip_done: List[dict], is_27b: bool,
                    user_time: Optional[List[dict]] = None) -> dict:
    """Flipzeit per direction, never flip_total (NF-Operator 30.09.):
    P>D = flip_first_work.flip_time_ms (P-Ende -> erstes Decode-Token), what="none" never counted;
    D>P = flip_user_time.flip_user_ms (Decode-Ende -> P-Prefill-Start, ab Build y4z 53977b2b67) --
    no fallback on flip_first_work: without the event the D>P value is "fehlt in IPC"."""
    out = {"instruments": INSTRUMENTS,
           "headline": "flip_total" if is_27b and not FIRST_TOKEN_HEADLINE_FOR_27B else "first_token"}
    ut = sorted((x for x in (user_time or []) if x.get("flip_user_ms") is not None),
                key=lambda x: x.get("start_ts") or 0)
    for d in ("P>D", "D>P"):
        mine = [x for x in first_work if x.get("dir") == d]
        no_work = sum(1 for x in mine if x.get("what") == "none")
        if d == "P>D":
            fw = sorted((x for x in mine if x.get("what") != "none" and x.get("flip_time_ms") is not None),
                        key=lambda x: x.get("flip_begin_ts") or 0)
            vals = [float(x["flip_time_ms"]) for x in fw]
            last_t = fw[-1].get("flip_begin_ts") if fw else None
            src = "IPC flip_first_work (%s)" % (fw[-1].get("what") or "?") if fw else None
            missing = None
        else:
            vals = [float(x["flip_user_ms"]) for x in ut]
            last_t = ut[-1].get("start_ts") if ut else None
            src = ("IPC flip_user_time%s" % (" (Ende = nur Leg-1-Dispatch)"
                                             if ut[-1].get("prefill_start_source") == "leg1_dispatch" else "")) if ut else None
            missing = None if ut else "flip_user_time (ab Build y4z)"
        sl, wk = d.split(">")
        fd = [x for x in flip_done if x.get("sleep") == sl and x.get("wake") == wk and x.get("flip_ms") is not None]
        lay = [float(x["flip_ms"]) for x in fd]
        out[d] = {"n": len(vals), "last": vals[-1] if vals else None, "last_t": last_t,
                  "median": _q(vals, 0.5), "p90": _q(vals, 0.9), "resolution_s": 0.001, "open": False,
                  "src": src, "missing": missing, "no_work": no_work, "layer_n": len(lay),
                  "layer_newest": lay[-1] if lay else None, "layer_median": _q(lay, 0.5), "layer_p90": _q(lay, 0.9)}
    done_by_begin = {round(x.get("flip_begin_ts") or 0, 1): x for x in flip_done}
    recent = []
    for x in sorted((x for x in first_work if x.get("what") != "none"), key=lambda x: x.get("flip_begin_ts") or 0)[-24:]:
        fd = done_by_begin.get(round(x.get("flip_begin_ts") or 0, 1)) or {}
        if x.get("dir") == "D>P":
            u = next((y for y in ut if y.get("parts") is not None and abs((y.get("start_ts") or 0) - (x.get("flip_begin_ts") or 0)) < 15
                      and (y.get("prefill_start_ts") or 0) >= (x.get("flip_begin_ts") or 0)), None)
            ms = u.get("flip_user_ms") if u else None
        else:
            ms = x.get("flip_time_ms")
        recent.append({"t": x.get("flip_begin_ts"), "dir": x.get("dir"), "ms": ms,
                       "layer_ms": fd.get("flip_ms"), "state": "ok", "src": "ipc"})
    out["recent"] = recent
    return out


# ----------------------------------------------------------------------------- Flipzeit in Nutzersicht
# Nutzer 01.10. ~08:20Z (Definition 29.09.): Flipzeit = volle Nutzersicht, nie flip_total.
#   P->D: P-Ende (letzter Prefill-Chunk) -> erstes Decode-Token auf D
#   D->P: Decode-Ende -> erster P-Prefill-Start
# zerlegt in Vorlauf (Arbeitsende -> flip_begin), Layer-Tausch (flip_begin -> flip_done), Nachlauf (flip_done ->
# erste Arbeit).  Gemessen am 01.10. (NF y6d): das Front-Ereignis flip_first_work (decode_token) kam bei P->D in
# 3-12 Flips je Boot VOR flip_done (+0,02..+0,99 s nach flip_begin), waehrend die Rang-Zaehler das erste Decode
# erst nach flip_done zeigen -- daher die Werte "unter 1 s".  P->D misst rigdash deshalb aus den Rang-Segmenten
# (activity.Model: P-Chunk-Ende aus prefill.last, Decode-Beginn aus decode.tokens/gpu_ms) und nennt das
# Front-Ereignis nur zum Vergleich.  D->P kommt aus flip_user_time (Front, mit parts).
FLIP_IDLE_S = 120.0          # no work this long after flip_done (or before the next flip): Leerlauf-Flip


def _seg_first(segs, kinds, t_from, t_to):
    for x in segs:
        if x["k"] in kinds and x["e"] > t_from and x["s"] < t_to:
            return max(x["s"], t_from)
    return None


def _seg_last_end(segs, kinds, t_to, t_from):
    best = None
    for x in segs:
        if x["k"] in kinds and x["s"] < t_to and x["e"] > t_from:
            best = min(x["e"], t_to) if best is None else max(best, min(x["e"], t_to))
    return best


def flip_views(segs: List[dict], ipc: dict, now: float) -> List[dict]:
    """One row per flip of the ring window, newest last: dir, begin, total_ms (Nutzersicht), parts, kind
    (ok | leerlauf | offen | ohne_daten), front_ms (the front's own number) and front_early (front event before
    flip_done).  Pure over its inputs (unit-tested)."""
    evs = ipc.get("ipc_events") or []
    begins = sorted(((e.get("data") or {}).get("flip_begin_ts") or e.get("ts"), e.get("data") or {})
                    for e in evs if e.get("type") == "flip_begin")
    done = {round(float((e.get("data") or {}).get("flip_begin_ts") or 0), 2): e.get("data") or {}
            for e in evs if e.get("type") == "flip_done"}
    fw = {round(float(x.get("flip_begin_ts") or 0), 2): x for x in ipc.get("flip_first_work") or []}
    ut = list(ipc.get("flip_user_time") or [])
    lo = segs[0]["s"] if segs else now
    out = []
    for i, (b, bd) in enumerate(begins):
        if b is None or b < lo:
            continue
        nxt = begins[i + 1][0] if i + 1 < len(begins) else None
        key = round(float(b), 2)
        fd = done.get(key) or {}
        sl, wk = bd.get("sleep"), bd.get("wake")
        d = "%s>%s" % (sl, wk)
        t_done = fd.get("t")
        row = {"dir": d, "begin": b, "done": t_done, "layer_ms": fd.get("flip_ms"),
               "vorlauf_ms": None, "nachlauf_ms": None, "total_ms": None, "kind": "offen", "nachlauf_d_extend_ms": None}
        f = fw.get(key) or {}
        if f.get("flip_time_ms") is not None and f.get("what") != "none":
            row["front_ms"] = f.get("flip_time_ms")
            row["front_what"] = f.get("what")
            row["front_early"] = bool(t_done and f.get("first_work_ts") and f["first_work_ts"] < t_done)
        if t_done is None:
            row["kind"] = "offen" if nxt is None else "ohne_daten"
            out.append(row)
            continue
        horizon = min(nxt if nxt is not None else now, t_done + FLIP_IDLE_S)
        if d == "P>D":
            p_end = _seg_last_end(segs, ("P", "single"), b + 0.3, b - FLIP_IDLE_S)
            first = _seg_first(segs, ("dec",), t_done - 0.3, horizon)
            if p_end is not None:
                row["vorlauf_ms"] = max(0.0, (b - p_end) * 1000.0)
            if first is not None:
                row["nachlauf_ms"] = max(0.0, (first - t_done) * 1000.0)
                ext = sum(max(0.0, min(x["e"], first) - max(x["s"], t_done)) for x in segs
                          if x["k"] == "D" and x["e"] > t_done and x["s"] < first)
                row["nachlauf_d_extend_ms"] = ext * 1000.0 if ext > 0 else None
            if p_end is None and first is None:
                row["kind"] = "leerlauf"           # no P work before, no decode after: nothing waited
            elif first is None:
                row["kind"] = "offen" if horizon >= now - 0.5 and nxt is None else "leerlauf"
            elif p_end is None:
                row["kind"] = "ok"                 # the flip was not ended by a prefill (D's queue woke D)
                row["total_ms"] = (first - b) * 1000.0
                row["start_src"] = "flip_begin (kein P-Chunk davor im Ring)"
            else:
                row["kind"] = "ok"
                row["total_ms"] = (first - p_end) * 1000.0
                row["start_src"] = "P-Chunk-Ende (rankstats prefill.last)"
            row["src"] = "Rang-Segmente (rankstats) + events flip_begin/flip_done"
        else:
            u = next((x for x in ut if fd.get("epoch") is not None and x.get("epoch") == fd.get("epoch")), None) or \
                next((x for x in ut if (x.get("prefill_start_ts") or 0) >= b and (nxt is None or (x.get("prefill_start_ts") or 0) < nxt)), None)
            if u is not None and u.get("flip_user_ms") is not None:
                p = u.get("parts") or {}
                row.update(total_ms=float(u["flip_user_ms"]), vorlauf_ms=p.get("pre_begin_ms"),
                           nachlauf_ms=p.get("first_chunk_ms"), legs_ms=p.get("legs_ms"), park_rpc_ms=p.get("park_rpc_ms"),
                           kind="leerlauf" if u.get("idle_flip") else "ok",
                           src="events flip_user_time (%s -> %s)" % (u.get("start_source") or "?", u.get("prefill_start_source") or "?"))
            else:
                first = _seg_first(segs, ("P", "single"), t_done - 0.3, horizon)
                row["kind"] = "leerlauf" if first is None and (nxt is not None or horizon < now - 0.5) else ("offen" if first is None else "ohne_daten")
                row["src"] = "kein flip_user_time zu diesem Flip"
        out.append(row)
    return out


def flip_last(views: List[dict]) -> dict:
    """Per direction: the newest measured flip (bold on the page), the newest idle flip, median/p90 of the
    measured ones in the ring, and how many front events came before flip_done."""
    out = {}
    for d in ("P>D", "D>P"):
        mine = [x for x in views if x["dir"] == d]
        ok = [x for x in mine if x["kind"] == "ok" and x.get("total_ms") is not None]
        idle = [x for x in mine if x["kind"] == "leerlauf"]
        vals = [x["total_ms"] for x in ok]
        out[d] = {"last": ok[-1] if ok else None, "n": len(ok), "median": _q(vals, 0.5), "p90": _q(vals, 0.9),
                  "idle_n": len(idle), "idle_last": idle[-1] if idle else None,
                  "open": next((x for x in reversed(mine) if x["kind"] == "offen"), None),
                  "front_early_n": sum(1 for x in mine if x.get("front_early"))}
    return out


PHASE_LABEL = {"P": "P aktiv (Prefill)", "single": "Prefill", "D": "D aktiv: Prefill", "dec": "D aktiv: Decode"}


def phase_now(segs: List[dict], ipc: dict, front: dict, views: List[dict], live: bool, now: float) -> Optional[dict]:
    """The phase NOW for the overview (Nutzer 01.10. ~08:20Z: "gerade beim Flippen sieht der Server aus, als
    wuerde nichts passieren"): {k, label, sub, since, flip_since, dir}.  Flip first (events, the front's clock),
    then the rank segments (activity.Model).  The Vorlauf of a flip is only known once flip_begin came; before
    that the page shows the working or idle group (front field for a live Vorlauf: see DASHBOARD-REDESIGN-1001)."""
    if not live:
        return None
    last = views[-1] if views else None
    if last and last.get("done") is None and last.get("kind") == "offen":
        return {"k": "flip", "dir": last["dir"], "label": "FLIP " + last["dir"].replace(">", "→"),
                "sub": "Layer-Tausch", "since": last["begin"], "flip_since": last["begin"]}
    if last and last.get("done") is not None and last.get("kind") == "offen" and now - last["done"] < FLIP_IDLE_S:
        return {"k": "flip", "dir": last["dir"], "label": "FLIP " + last["dir"].replace(">", "→"),
                "sub": "Nachlauf (bis zur ersten Arbeit)", "since": last["done"], "flip_since": last["begin"]}
    if (front or {}).get("state") == "flipping":
        return {"k": "flip", "dir": None, "label": "FLIP", "sub": "Vorlauf (Drain/Park, vor flip_begin)",
                "since": (front or {}).get("ts") or now, "flip_since": None}
    work = [x for x in segs if x["k"] != "unknown"]
    if not work:
        return {"k": "unknown", "label": "unbekannt", "sub": "noch keine Probe", "since": now}
    cur = work[-1]
    if segs and segs[-1]["k"] == "unknown" and now - cur["e"] > 3.0:
        return {"k": "unknown", "label": "unbekannt", "sub": segs[-1].get("why") or "", "since": segs[-1]["s"]}
    k = cur["k"]
    since = cur["s"]
    for x in reversed(work[:-1]):            # the run of the same state, across sample-sized pieces
        if x["k"] != k or since - x["e"] > 1.5:
            break
        since = x["s"]
    if k in PHASE_LABEL:
        return {"k": k, "label": PHASE_LABEL[k], "sub": "", "since": since}
    if k == "idle":
        aw = cur.get("awake") or (front or {}).get("awake")
        return {"k": "idle", "label": "%s inaktiv" % (aw or "?"), "sub": "wach, keine Arbeit", "since": since, "awake": aw}
    if k in ("flip_pd", "flip_dp"):
        return {"k": "flip", "dir": "P>D" if k == "flip_pd" else "D>P", "label": "FLIP " + ("P→D" if k == "flip_pd" else "D→P"),
                "sub": "Layer-Tausch", "since": since, "flip_since": since}
    if k == "flip_tail":
        return {"k": "flip", "dir": None, "label": "FLIP", "sub": "Nachlauf (bis zur ersten Arbeit)", "since": since}
    return {"k": k, "label": k, "sub": cur.get("why") or "", "since": since}


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


def identical_bursts(ring, group: str = "P") -> List[dict]:
    """Bursts of identical prompts (NF-Operator 30.09.: dmatrix bs stages send k identical prompts at
    once, e.g. 6 x 65602 tokens within 15 ms; in one P pass none can wait for the first, so all are
    cached = 0 -- a bench pattern, not a broken cache).  From the front's served_tokens mirror
    (state.json front, IPC): one mirror step whose Δn >= 2 legs all ended together, whose Δprompt
    splits into Δn equal lengths and which read nothing from cache.  (A per-leg record would make
    it exact; the mirror carries sums.)"""
    out, prev = [], None
    for smp in ring:
        row = (((smp.get("front") or {}).get("served_tokens")) or {}).get(group)
        if not isinstance(row, dict):
            continue
        cur = (float(row.get("n") or 0), float(row.get("prompt") or 0), float(row.get("cached") or 0))
        if prev is not None and cur != prev:
            dn, dp, dc = cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]
            if dn >= 2 and dp > 0 and dc == 0 and abs(dp / dn - round(dp / dn)) < 1e-9 and dp / dn >= 1024:
                out.append({"t": smp["t"], "n": int(dn), "len": int(round(dp / dn)), "prompt": dp})
        prev = cur
    return out


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
             "mamba_n": r.get("mamba_n"), "l3inc_n": r.get("l3inc_n"),
             "l3inc_share": (r["si_del"] / r["si_deliv"]) if r.get("si_del") is not None and r.get("si_deliv") else None,
             "l3inc_delivered": r.get("si_del"), "l3inc_deliverable": r.get("si_deliv"),
             "prefetch_refused": r.get("pf_refused"), "prefetch_timeout": r.get("pf_timeout")}
        out[g] = {"window": w, "boot": b}
    dct = (last.get("front") or {}).get("d_cached_tokens")
    if isinstance(dct, dict) and "D" in out:
        out["D"]["boot"]["handoff"] = dct.get("handoff")
        out["D"]["boot"]["d_prefix_hit"] = dct.get("d_prefix_hit")
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
        # hit rate without bursts of identical prompts (bench), over the window and over the ring
        bu = identical_bursts(ring, g)
        if bu:
            first = next((x for x in ring if isinstance((((x.get("front") or {}).get("served_tokens")) or {}).get(g), dict)), None)
            r0 = ((first or {}).get("front") or {}).get("served_tokens", {}).get(g) or {}
            rp, rc = pr - float(r0.get("prompt") or 0), ca - float(r0.get("cached") or 0)
            bw = sum(x["prompt"] for x in bu if x["t"] >= now - WINDOW_S)
            ba = sum(x["prompt"] for x in bu)
            out["served_" + g]["bursts"] = bu[-6:]
            out["served_" + g]["bursts_n"] = len(bu)
            out["served_" + g]["window"]["hit_no_burst"] = (wc / (wp - bw)) if wp - bw > 0 else None
            out["served_" + g]["ring"] = {"prompt": rp, "cached": rc, "hit": rc / rp if rp > 0 else None,
                                          "hit_no_burst": rc / (rp - ba) if rp - ba > 0 else None, "span_s": RING_S}
    return out


def prefill_route(front: dict) -> dict:
    """Where the prefills ran (Nutzer 30.09.: "wird jeder Prefill zu P geflippt?").  Exact only with
    the front field ``front.routes`` {d_direct, via_p, d_drain, reroute_midstream} (proposed, writer
    weg2/front.py at the X-EXACT-ERR via classification); until then derived from ``front.served``:
    every request through P has one P leg and one D leg, a D-direct request only a D leg, so
    d_direct = D legs - P legs (mirror every 5 s, reroutes invisible) -- labelled as derived."""
    r = (front or {}).get("routes")
    if isinstance(r, dict):
        return dict(r, src="front.routes", exact=True)
    sv = (front or {}).get("served")
    if not isinstance(sv, dict) or sv.get("P") is None or sv.get("D") is None:
        return {"src": None, "exact": False, "missing": "front.routes"}
    return {"via_p": sv["P"], "d_direct": max(0, sv["D"] - sv["P"]), "src": "front.served (D − P, abgeleitet)",
            "exact": False, "missing": "front.routes"}


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


def _awake(m: "activity.Model", front: dict):
    """Awake group now: the woken group of the newest flip_done (events), else the front mirror."""
    return (front or {}).get("awake")


def model_of(ipc: dict, ring) -> "activity.Model":
    """activity.Model with everything the phase states need from the boot's IPC."""
    evs = ipc.get("ipc_events") or []
    begins = [dict(e.get("data") or {}, flip_begin_ts=(e.get("data") or {}).get("flip_begin_ts") or e.get("ts"))
              for e in evs if e.get("type") == "flip_begin"]
    life = {"serving_since": ipc.get("serving_since_ts"),
            "terminal_since": ipc.get("lifecycle_since") if ipc.get("terminal") else None,
            "terminal_state": ipc.get("lifecycle") if ipc.get("terminal") else None}
    return activity.Model(ring, flip_done_of(ipc), list(ipc.get("flip_first_work") or []), begins,
                          list(ipc.get("flip_user_time") or []), life)


def life_view(ipc: dict, t0: Optional[float], sig: Optional[float]) -> dict:
    """boot_s = Boot-Start (boot_id) -> serving_since_ts; dur_s = Boot-Start -> Ende (lifecycle_since eines
    terminalen Boots, sonst das juengste IPC-Lebenszeichen).  None, wo eine Uhr fehlt."""
    ready = ipc.get("serving_since_ts")
    end_t = ipc.get("lifecycle_since") if ipc.get("terminal") else sig
    end_t = end_t or sig
    return {"boot_s": round(ready - t0, 1) if t0 and ready and ready >= t0 else None,
            "dur_s": round(end_t - t0, 1) if t0 and end_t and end_t >= t0 else None}


def flip_done_of(ipc: dict) -> List[dict]:
    return [dict(e.get("data") or {}, t=(e.get("data") or {}).get("t") or e.get("ts"))
            for e in (ipc.get("ipc_events") or []) if e.get("type") == "flip_done"]


def build_view(ipc: dict, ring, rank: dict, rates: dict, now: float,
               zoom: Optional[Tuple[float, float]] = None) -> dict:
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
    m = model_of(ipc, ring)
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
        "flip_times": flip_times_view(fw, flip_done, is27, ipc.get("flip_user_time")),
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
                    for pv in [prefill_view(m, g, now)] if pv},
        "decode": {g: dv for g in groups if g in ("D", "single")
                   for dv in [decode_view(m, g, front, now)] if dv},
        "totals": totals_view(ring, keys, front, t0, now),
        "prefill_route": prefill_route(front),
        "cache": cache_view(ring, keys, now),
        "series": series_view(m, now),
        # the zoomed stretch as its own curves; "series" stays the 15 min the tiles' 60-s figures read
        "series_zoom": series_view(m, now, zoom) if zoom else None,
        "timeline": timeline_view(m, live, _awake(m, front), now, t0),
        "fields": ipcfields.for_page(fields),
        "phase_now": None,
        "fields_summary": ipcfields.summary(fields),
        "end": stops.classify_ipc(ipc),
    }
    # Nutzer 01.10.: Dauer und Bootzeit je Boot in "Letzte Boots"
    v.update(life_view(ipc, t0, sig))
    # Nutzer 01.10. ~08:20Z: Flipzeit in Nutzersicht mit Vorlauf/Layer-Tausch/Nachlauf, und die Phase jetzt
    fv = flip_views(v["timeline"]["segs"], ipc, now)
    v["flip_views"] = fv[-24:]
    v["flip_last"] = flip_last(fv)
    v["phase_now"] = phase_now(v["timeline"]["segs"], ipc, front, fv, live, now)
    view_ipc = {k: x for k, x in ipc.items() if k not in ("ipc_events", "flip_first_work", "flip_user_time")}
    v["ipc"] = view_ipc
    return v


class IpcBoots:
    """Samples every boot state dir once a second (rank counters + front mirror) into a ring.

    ``role`` (30.09. ~21:40Z, sampler in its own process): "both" = sample and serve in one process
    (tests, a dev server without state dir); "sampler" = sample and write each sample to ``store``
    (sampler.RingStore); "reader" = the web server: reads state.json/events and the rank files for the
    fields, but the ring comes from ``store`` -- the reading moment is never on the web server's GIL."""

    def __init__(self, roots=ipcstate.STATE_ROOTS, store=None, role: str = "both"):
        self.store, self.role = store, role
        self._synced_t = 0.0
        self._extent_t = 0.0
        self.ipc = ipcstate.IpcStates(roots)
        self.lock = threading.Lock()
        self.rings: Dict[str, deque] = {}
        self.rank: Dict[str, dict] = {}
        self.final: set = set()
        self.rates = ipcfields.Rates()
        self.last_error: Optional[str] = None

    def warm_from_store(self, now: Optional[float] = None) -> int:
        """Sampler start (01.10.: every deploy restarts the sampler, and its first append deleted the 16-min ring
        the store still held -- the phase bar showed 10 min "keine IPC-Probe" after each deploy): take the
        store's samples of the last RING_S back into the writer's ring before the first poll."""
        if self.store is None or self.role != "sampler":
            return 0
        now = now or time.time()
        n = 0
        with self.lock:
            for k, t, samp in self.store.since(now - RING_S):
                self.rings.setdefault(k, deque()).append(samp)
                n += 1
        return n

    def poll(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.ipc.poll(now)
        seen = set()
        new = []
        for v in self.ipc.boots(now):
            key = v.get("boot_id") or v.get("dir")
            seen.add(key)
            if key in self.final:
                continue
            if self.role == "reader":
                continue                   # the rank files are the sampler's (store.ranks() in sync)
            rank = ipcfields.read_rank_files(ipcfields.state_rankstate_dirs(v))
            samp = {"t": now, "r": {k: compact(r) for k, r in rank["rankstats"].items()},
                    "front": _front_small(v.get("front") or {})}
            with self.lock:
                if self.role != "reader":
                    ring = self.rings.setdefault(key, deque())
                    ring.append(samp)
                    while ring and ring[0]["t"] < now - RING_S:
                        ring.popleft()
                    new.append((key, now, samp))
                self.rank[key] = rank
                if v.get("terminal"):
                    self.final.add(key)
        with self.lock:
            for k in list(self.rings) + list(self.rank):
                if k not in seen:
                    self.rings.pop(k, None)
                    self.rank.pop(k, None)
                    self.final.discard(k)
            keep = {k: r[0]["t"] for k, r in self.rings.items() if r}
        if self.role == "sampler" and self.store is not None:
            self.store.append(new, keep)
            with self.lock:
                ranks = dict(self.rank)
            self.store.set_ranks({k: r for k, r in ranks.items() if k not in self.final or k in keep})
        if self.role == "reader" and self.store is not None:
            self.sync(now)

    def sync(self, now: Optional[float] = None) -> None:
        """Reader: the samples the sampler process wrote since the last sync, into the in-memory ring;
        every 10 s the ring is trimmed to what the store still holds."""
        rows = self.store.since(self._synced_t)
        ranks = self.store.ranks()
        with self.lock:
            self.rank = ranks
            for k, t, samp in rows:
                self.rings.setdefault(k, deque()).append(samp)
                self._synced_t = max(self._synced_t, t)
        if (now or time.time()) - self._extent_t >= 10.0:
            self._extent_t = now or time.time()
            ext = self.store.extent()
            with self.lock:
                for k in list(self.rings):
                    if k not in ext:
                        self.rings.pop(k, None)
                        continue
                    r = self.rings[k]
                    while r and r[0]["t"] < ext[k]:
                        r.popleft()

    def run_forever(self, stop: threading.Event) -> None:
        # on the tick (x.5 s), not "1 s after the last poll": a slow poll delays one reading, the grid stays
        phase = 0.5
        nxt = time.time() // SAMPLE_S * SAMPLE_S + SAMPLE_S + phase
        while not stop.is_set():
            stop.wait(max(0.0, nxt - time.time()))
            if stop.is_set():
                break
            t0 = time.time()
            try:
                self.poll(t0)
                self.last_error = None
            except Exception as e:  # keep sampling alive; visible in /api/live
                self.last_error = "%s: %s" % (type(e).__name__, e)
            nxt += SAMPLE_S
            if nxt < time.time():
                nxt = time.time() // SAMPLE_S * SAMPLE_S + SAMPLE_S + phase

    def snapshot(self, now: Optional[float] = None, max_boots: int = 10,
                 zoom: Optional[Tuple[float, float]] = None) -> List[dict]:
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
            v = build_view(ipc, ring, rank, None, now, zoom)
            line = "27b" if "/27b/" in d else "nf"
            v["primary"] = v["live"] or line not in newest_line
            newest_line.setdefault(line, key)
            views.append(v)
        views.sort(key=lambda v: (not v["live"], not v["primary"], v["age_s"] if v["age_s"] is not None else 1e12))
        return views

    def model(self, key: str) -> Optional["activity.Model"]:
        with self.lock:
            ring = list(self.rings.get(key) or ())
        if not ring:
            return None
        with self.ipc.lock:
            d = next((d for d, st in self.ipc._st.items() if (st.get("boot_id") or d) == key), None)
            st = self.ipc._st.get(d) if d else None
        ipc = ipcstate.boot_view(d, st, self.ipc._ev.get(d), time.time()) if st else {}
        return model_of(ipc, ring)

    def activity(self, key: str, start: float, n: int, bs: float) -> List[dict]:
        """Per bucket which class computed and how many tokens (energy.EnergyBook): the work at the
        time it was done (activity.Model), not at the sample that saw the counter move."""
        out = [{"P": False, "D": False, "dec": False, "P_tok": 0, "D_tok": 0, "dec_tok": 0} for _ in range(n)]
        m = self.model(key)
        if m is None:
            return out
        b = m.buckets(start, n, bs)
        for i in range(n):
            for cls, k in (("P", "p_tps"), ("D", "d_tps"), ("dec", "dec_tps")):
                v = b[k][i] or 0.0
                if v > 0:
                    out[i][cls] = True
                    out[i][cls + "_tok"] = int(round(v * bs))
        return out
