"""WHEN a group worked and HOW MUCH, from the 1-s rankstats samples (ipcboot ring) -- no log.

Nutzer 30.09. ~16:45Z: "die echtzeitwerte sind völliger blödsinn die ganze zeit 16k/s prefill".
Root cause (found in the production history of NF y4x, 30.09. 16:19-16:36Z): the P rank writes
``prefill.new_tokens`` as a cumulative counter that jumps by a WHOLE chunk (16384 tokens) the moment
the chunk finishes.  A rate taken as Δcounter / Δ(sample clock) puts the whole chunk into the one
sample (1 s -> "16k/s") or 5-s bucket (3275 / 6550 / 9814 tok/s = 1/2/3 chunks per 5 s) in which the
counter moved, and shows the ~3 s the chunk really computed as idle.  The same aliasing made P and D
look active at the same time and gave 1-token "bursts" of 1 s.

The instrument here puts every piece of work at the time it was DONE:

* prefill chunk: ``prefill.last {t, gpu_ms, new}`` is the rank's own record of its newest chunk;
  the chunk ran from ``t - gpu_ms`` on the first stage (PP0/TP0) to its ``t`` on the LAST pipeline
  stage (same chunk count).  Its tokens are spread over exactly that interval.  Several chunks inside
  one sample (short extends) start at the previous sample's clock.
* decode: ``decode.tokens`` moves every round (tens of ms), so the Δ between two samples belongs to
  the interval between the two rank clocks -- minus any flip window inside it.
* flips: events ``flip_begin`` .. ``flip_done`` (the front's clock, ms); the stretch from flip_done
  to the first work after it (``flip_first_work``) is the flip tail.

Rates are then tokens / the time the work took: a prefill burst (chunks that follow each other
within CHUNK_GAP_S) is tokens / (end of its last chunk on the last stage - start of its first chunk
on the first stage) -- the P-Ende wall clock; decode per stream only over intervals with decode
before AND after (steady), tokens / wall / running.  Pure functions, unit-tested.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

CHUNK_GAP_S = 1.5      # chunks closer than this belong to one burst
MIN_RATE_TOK = 512     # a burst smaller than this has no meaningful rate (a 1-token re-extend)
DEC_MERGE_S = 1.5      # decode intervals closer than this form one decode stretch
#: a prefill chunk needs this many new tokens on average to count as a real prefill.  On D every request
#: admits with a 1-token extend after the P->D hand-off (y8vb D.log: 3029 of 3051 prefill batches had
#: #new-token 1; the stock "input throughput" of such a line is 1 token / wall seconds since the previous line,
#: 5,8 tok/s).  They are admits, not prefill: counted apart, never rated (Auftrag 880, Nutzer 03.10. "6 token/s
#: prefill in D???").
WIDE_MIN_TOK = 64


def is_wide(tok, n) -> bool:
    """True when a chunk record (``n`` chunks, ``tok`` new tokens) has real prefill width (mean >= WIDE_MIN_TOK)."""
    tok = tok or 0
    n = n or 0
    return tok >= WIDE_MIN_TOK * n if n > 0 else tok >= WIDE_MIN_TOK


def _grp(key: str) -> str:
    return key.split(".", 1)[0]


def _tp_pp(key: str) -> Tuple[int, int]:
    part = key.split(".", 1)[1] if "." in key else ""
    try:
        i = part.index("pp")
        return int(part[2:i]), int(part[i + 2:])
    except ValueError:
        return 0, 0


def stage_keys(keys, g: str) -> Tuple[Optional[str], Optional[str]]:
    """(first stage TP0/PP0, last pipeline stage TP0/PPmax) of a group."""
    ks = sorted(k for k in keys if _grp(k) == g)
    if not ks:
        return None, None
    tp0 = [k for k in ks if _tp_pp(k)[0] == 0] or ks
    first = min(tp0, key=lambda k: _tp_pp(k)[1])
    last = max(tp0, key=lambda k: _tp_pp(k)[1])
    return first, last


def _d(b: dict, a: dict, f: str) -> Optional[float]:
    x, y = b.get(f), a.get(f)
    if x is None or y is None:
        return None
    return x - y if x >= y else x


def rank_pairs(ring, key: str):
    """Consecutive samples of one rank whose file advanced: (a, b) compact records."""
    out, prev = [], None
    for s in ring:
        r = s["r"].get(key)
        if r is None or r.get("ts") is None:
            continue
        if prev is not None and r["ts"] > prev["ts"]:
            out.append((prev, r))
        if prev is None or r["ts"] > prev["ts"]:
            prev = r
    return out


def first_rise(ring, key: Optional[str], fields, t_from: float, t_to: float):
    """The first rise of any of ``fields`` on rank ``key`` whose rank clock lies after ``t_from``:
    (seen_ts, lo) -- the work began in (lo, seen_ts], lo = max(previous rank clock, t_from).  The rank file
    is written every FLLIPER_PDFLIP_RANKSTATS_PERIOD_S (1 s), so seen_ts - lo is the instrument's resolution.
    None when the ring holds no such rise up to ``t_to``."""
    if not ring or key is None:
        return None
    for a, b in rank_pairs(ring, key):
        if b["ts"] <= t_from:
            continue
        if a["ts"] > t_to:
            break
        if any((b.get(f) is not None and a.get(f) is not None and b[f] > a[f]) for f in fields):
            return b["ts"], max(a["ts"], t_from)
    return None


#: the front's stamp of the D>P end (Nutzer 02.10. ~13:05Z: "... prefill batch beginn"): the first forward on
#: P's FIRST stage after the wake; pp_last_start_ts rides along (the pipeline fill PP0 -> PP-last is prefill)
DP_END_SOURCE = "pp_first_forward"


def dp_prefill_start(ring, keys, u: Optional[dict], t_from: float, t_to: float):
    """D>P end = Beginn des ersten Prefill-Forwards auf P (erste PP-Stufe) nach dem Wake -- (end, lo, src):
    the front's flip_user_time.prefill_start_ts when it is a ``pp_first_forward`` stamp (exact, lo = end),
    else the first rise of rankstats work.forward_ct of P's first stage (TP0/PP0) after ``t_from``, bounded by
    the rank clock (lo = previous rank clock).  Never the leg-1 DISPATCH.  None when neither is there."""
    if u is not None and u.get("prefill_start_source") == DP_END_SOURCE and u.get("prefill_start_ts") is not None:
        t = float(u["prefill_start_ts"])
        return t, t, "front flip_user_time.prefill_start_ts (%s)" % DP_END_SOURCE
    k0 = stage_keys(keys, "P")[0] if keys else None
    r = first_rise(ring, k0, ("fwd",), t_from, t_to)
    if r is None:
        return None
    return r[0], r[1], "rankstats %s work.forward_ct (first prefill forward on P, rank cycle)" % k0


def chunks(ring, keys, g: str, kind: Optional[str] = None) -> List[dict]:
    """Prefill chunks of group g with the time they really ran.

    Measured at NF y4x (30.09. ~17:00Z, raw rankstats every 0.5 s): the first stage's chunk ends
    ``prefill.last.t`` rise monotonically (one chunk per ~2.4 s on PP0 for 16k chunks), but a chunk's
    ``gpu_ms`` can reach back before the previous chunk's end (it was queued behind it), and the LAST
    stage's counter often jumps by 2-4 chunks between two samples (PP2 is fast).  So:

    * burst = chunks whose PP0 start (t - gpu_ms) lies within CHUNK_GAP_S of the burst's end; the burst
      runs from the first chunk's PP0 start to the end of its last chunk on the last stage (the first
      last-stage record whose counter reached that chunk) -- the P-Ende wall clock;
    * the curve spreads a burst's tokens evenly over the burst: never more than the burst rate, every
      token once (per chunk gave 10k spikes with 27B's 1k-token chunks, several per sample).

    ``kind``: None = every chunk; "wide" = only chunks with a mean width >= WIDE_MIN_TOK new tokens (real
    prefill); "admit" = only the narrower ones (1-token admit extends).  The filter runs BEFORE the bursts
    are formed, so a chain of admits never stretches the wall time of a real prefill burst.
    """
    k0, kl = stage_keys(keys, g)
    if k0 is None:
        return []
    last_recs: List[Tuple[float, float]] = []            # (counter, last.t) of the last stage
    if kl != k0:
        for a, b in rank_pairs(ring, kl):
            if (_d(b, a, "pchunks") or 0) > 0 and b.get("plast_t") is not None:
                last_recs.append((b["pchunks"], b["plast_t"]))
    out = []
    for a, b in rank_pairs(ring, k0):
        n = _d(b, a, "pchunks") or 0
        tok = _d(b, a, "pnew") or 0
        if n <= 0 and tok <= 0:
            continue
        if kind is not None and is_wide(tok, n) != (kind == "wide"):
            continue
        e0 = b.get("plast_t") if b.get("plast_t") is not None else b["ts"]
        # FEHLT 7 (build y5a): own_ms = the chunk's own time after the previous one left -- the
        # chunk's real start; without it gpu_ms, which reaches back into the wait behind the previous
        own = b.get("plast_own") if n == 1 else None
        s = e0 - (own if own is not None else (b.get("plast_gpu") or 0.0)) / 1000.0
        if n > 1:
            # several chunks between two samples: they ran back to back for their summed compute time
            s = min(s, e0 - (_d(b, a, "pcomp") or 0.0) / 1000.0)
            if k0 == kl:
                # one stage (D): no pipeline queue in front of a chunk, its wait is expert streaming and
                # collectives -- the chunks ran for their summed gpu_ms (02.10.: 2944+475 tokens took
                # 4766+1961 ms, compute alone 1323 ms drew a 5-s hole before the prefill)
                s = min(s, e0 - (_d(b, a, "pgpu") or 0.0) / 1000.0)
        cnt = b.get("pchunks")
        el = next((t for c, t in last_recs if cnt is not None and c >= cnt), None) if kl != k0 else e0
        out.append({"s": s, "e0": e0, "e": max(e0, el if el is not None else e0), "tok": tok,
                    "cached": _d(b, a, "pcached") or 0.0, "n": n, "comp_ms": _d(b, a, "pcomp") or 0.0, "g": g,
                    "ext": b.get("plast_ext")})
    out.sort(key=lambda c: c["e0"])
    # bursts and the spread intervals
    burst: List[dict] = []
    for c in out + [None]:
        if c is not None and burst and c["s"] <= max(x["e"] for x in burst) + CHUNK_GAP_S:
            burst.append(c)
            continue
        if burst:
            # the curve: a burst's tokens spread evenly over the burst (its P-Ende wall clock).  Per
            # chunk is not honest at 1-s rows: with ~1k-token chunks (27B) several end inside one
            # sample, their end stamps bunch, and [prev end, end] gave 10k tok/s spikes (18:30Z).
            start = min(x["s"] for x in burst)
            end = max(x["e"] for x in burst)
            for i, x in enumerate(burst):
                x["parts"] = [(start, max(end, start + 1e-3))]
                x["e"] = end if i == len(burst) - 1 else x["e0"]
        burst = [c] if c is not None else []
    return out


def bursts(cs: List[dict]) -> List[dict]:
    out: List[dict] = []
    for c in sorted(cs, key=lambda x: x["s"]):
        if out and c["s"] <= out[-1]["e"] + CHUNK_GAP_S:
            b = out[-1]
            b["e"] = max(b["e"], c["e"])
            b["tok"] += c["tok"]
            b["n"] += c["n"]
            b["cached"] += c["cached"]
            b["cs"].append(c)
        else:
            out.append({"s": c["s"], "e": c["e"], "tok": c["tok"], "n": c["n"], "cached": c["cached"], "cs": [c]})
    for b in out:
        b["wall_s"] = b["e"] - b["s"]
        b["rate"] = b["tok"] / b["wall_s"] if b["tok"] >= MIN_RATE_TOK and b["wall_s"] > 0 else None
    return out


# ----------------------------------------------------------------------------- Tiefe je Prefill / Decode
# Nutzer 02.10. ~12:04Z: "nicht nur wie viele token sondern (token x - y, tok new)" -- the P prefill rate falls with
# the context depth (27B N5a: 73k tokens from depth 4k ran 10,8k -> 4,1k tok/s, 130k tokens from depth 77824 only
# 4,3k -> 2,4k tok/s), so every prefill shows x = start depth (prefix), y = end depth, n = new tokens, and the
# rate at its start and end.  Sources, best first:
#   1. rankstats prefill.last.ext [[rid, start, end]] per chunk (the #969 EXTENT; port seat, 02.10.);
#   2. events request_done prefill.<G> {cached, prompt, tokens} of the requests whose prefill ended in the burst
#      (exact per rid, but only once the request is DONE);
#   3. rankstats prefill.cached_tokens: #cached-token is the prefix of a request's FIRST chunk (later chunks of a
#      chunked prefill add 0, schedule_policy add_chunked_req), so the burst's delta is the start depth of the
#      request(s) that began in it.
EDGE_SHARE = 0.15      # tok/s at the start / end of a prefill: over the first / last 15 % of its tokens
REQ_END_TOL_S = (0.5, 3.0)   # a request_done prefill end may lie this far before / after the burst


def edge_rates(b: dict) -> Tuple[Optional[float], Optional[float]]:
    """(tok/s over the first EDGE_SHARE of the burst's tokens, tok/s over the last EDGE_SHARE), on the first
    stage's chunk clock (prefill.last t): start = tokens / (end of the k-th chunk - start of the first),
    end = tokens / (end of the last chunk - end of the chunk before the last group).  None below three chunk
    records or MIN_RATE_TOK tokens (no slope to read)."""
    cs = sorted(b.get("cs") or [], key=lambda c: c["e0"])
    n = b.get("tok") or 0.0
    if len(cs) < 3 or n < MIN_RATE_TOK:
        return None, None
    want = max(EDGE_SHARE * n, 1.0)
    acc, k = 0.0, 0
    while k < len(cs) - 1 and acc < want:
        acc += cs[k]["tok"]
        k += 1
    t0 = min(c["s"] for c in cs)
    start = acc / (cs[k - 1]["e0"] - t0) if cs[k - 1]["e0"] > t0 else None
    acc2, j = 0.0, len(cs)
    while j > 1 and acc2 < want:
        j -= 1
        acc2 += cs[j]["tok"]
    span = cs[-1]["e0"] - cs[j - 1]["e0"]
    end = acc2 / span if span > 0 else None
    return start, end


def _req_prefill(r: dict, g: str) -> Tuple[Optional[float], Optional[dict]]:
    """(time the request's prefill on group g ended, request_done prefill.<G>) or (None, None).  P: the
    front's leg-1 end (p_leg1_end_ts, else arrival + queue + p_prefill = the leg-1 window, front_requests
    ttft_parts); D: the first content (D's prefill ends with the first token)."""
    pre = r.get("prefill") if isinstance(r.get("prefill"), dict) else {}
    keys = ("P",) if g == "P" else ("D",) if g == "D" else ("P", "D")
    for k in keys:
        x = pre.get(k)
        if not isinstance(x, dict) or not x.get("tokens") or x.get("prompt") is None:
            continue
        if k == "P":
            t = r.get("p_leg1_end_ts")
            if t is None and r.get("arrival_ts") is not None and r.get("p_prefill_ms") is not None:
                t = float(r["arrival_ts"]) + float(r.get("queue_ms") or 0) / 1000.0 + float(r["p_prefill_ms"]) / 1000.0
        else:
            t = r.get("first_token_ts")
        if t is not None:
            return float(t), x
    return None, None


def prefill_depth(b: dict, done: Optional[List[dict]] = None, g: str = "P") -> dict:
    """Token x-y (n neu) of one prefill burst, with the rate at its start and end (module note above).
    ``reqs`` = per request {rid, x, y, n} where the source names requests; ``exact`` = x/y are the requests'
    own depths (sources 1 and 2 with matching token sums), not the burst's summed prefix."""
    n = int(round(b.get("tok") or 0))
    out = {"n": n, "x": None, "y": None, "src": None, "exact": False, "reqs": []}
    ts, te = edge_rates(b)
    out["tps_start"], out["tps_end"] = ts, te
    ext: Dict[str, List[int]] = {}
    for c in sorted(b.get("cs") or [], key=lambda c: c["e0"]):
        for rid, s0, e0 in (c.get("ext") or ()):
            lo_hi = ext.setdefault(rid, [s0, e0])
            lo_hi[0], lo_hi[1] = min(lo_hi[0], s0), max(lo_hi[1], e0)
    if ext:
        reqs = [{"rid": r, "x": v[0], "y": v[1], "n": v[1] - v[0]} for r, v in ext.items()]
        if len(reqs) == 1 and n > 0:
            # a sample shows only the NEWEST chunk's extent: with one request the burst's own token count
            # reaches back to its true start
            r = reqs[0]
            r["x"] = max(0, min(r["x"], r["y"] - n))
            r["n"] = r["y"] - r["x"]
        out.update(x=min(r["x"] for r in reqs), y=max(r["y"] for r in reqs), reqs=reqs, exact=True,
                   src="rankstats prefill.last.ext (#969 EXTENT per chunk)")
        return out
    lo, hi = b["s"] - REQ_END_TOL_S[0], b["e"] + REQ_END_TOL_S[1]
    reqs = []
    for r in done or ():
        t, x = _req_prefill(r, g)
        if t is not None and lo <= t <= hi:
            reqs.append({"rid": r.get("rid"), "x": int(x.get("cached") or 0), "y": int(x["prompt"]),
                         "n": int(x.get("tokens") or 0), "t": t})
    if reqs:
        reqs.sort(key=lambda r: r["t"])
        tot = sum(r["n"] for r in reqs)
        out.update(x=min(r["x"] for r in reqs), y=max(r["y"] for r in reqs), reqs=reqs,
                   exact=abs(tot - n) <= max(64, 0.02 * n),
                   src="events request_done (prefill.%s cached/prompt/tokens)" % ("D" if g == "D" else "P"))
        return out
    c = int(round(b.get("cached") or 0))
    out.update(x=c, y=c + n, src="rankstats prefill.cached_tokens (prefix at the first chunk of the request)")
    return out


def decode_by_bs(iv: List[dict], s: float, e: float) -> List[dict]:
    """Decode of [s, e] per batch size: tokens / decode seconds at that bs (the group's tok/s) and per seat.
    Only sample intervals whose rounds all ran at one bs (decode.gpu_ms_by_bs delta, else decode.running) are
    told apart; the rest is "mix" (their mean bs named).  Each interval counts with its share inside [s, e]."""
    acc: Dict[object, Dict[str, float]] = {}
    for x in iv:
        ov = sum(max(0.0, min(y, e) - max(a, s)) for a, y in x.get("parts") or [(x["s"], x["e"])])
        if ov <= 0 or x["dur"] <= 0:
            continue
        f = min(1.0, ov / x["dur"])
        pure = x.get("bs_min") is not None and x.get("bs_min") == x.get("bs_max")
        key = int(x["bs_min"]) if pure else "mix"
        a = acc.setdefault(key, {"tok": 0.0, "busy": 0.0, "seat_s": 0.0})
        a["tok"] += x["tok"] * f
        a["busy"] += x["busy"] * f
        a["seat_s"] += (x.get("seat_s") or 0.0) * f
    out = []
    for k, a in acc.items():
        if a["busy"] <= 0 or a["tok"] <= 0:
            continue
        tps = a["tok"] / a["busy"]
        out.append({"bs": k, "tps": tps, "per_slot": (a["tok"] / a["seat_s"]) if a["seat_s"] > 0 else None,
                    "bs_mean": (a["seat_s"] / a["busy"]) if a["seat_s"] > 0 else None,
                    "busy_s": a["busy"], "tok": a["tok"]})
    out.sort(key=lambda r: (r["bs"] == "mix", r["bs"] if r["bs"] != "mix" else 0))
    return out


def decode_rounds(iv: List[dict], s: float, e: float, solo: Optional[List[dict]] = None) -> dict:
    """Round time of D's decode inside [s, e] (dual: while P prefills): {"ms", "n", "by_bs": [{bs, ms, n, solo_ms}],
    "solo_ms"} from the delta of ``decode.gpu_ms_by_bs`` {bs: [rounds, ms]} of the sample intervals, each counted with
    its share inside [s, e].  ``solo`` = intervals in which P did not work: their mean round ms per bs is the
    reference (solo_ms per bs, and "solo_ms" = the same bs mix weighted by the rounds of [s, e])."""
    acc: Dict[str, List[float]] = {}
    for x in iv:
        ov = sum(max(0.0, min(y, e) - max(a, s)) for a, y in x.get("parts") or [(x["s"], x["e"])])
        if ov <= 0 or x["dur"] <= 0:
            continue
        f = min(1.0, ov / x["dur"])
        for bs, (dn, dms) in (x.get("rnd") or {}).items():
            a = acc.setdefault(bs, [0.0, 0.0])
            a[0] += dn * f
            a[1] += dms * f
    ref: Dict[str, List[float]] = {}
    for x in solo or ():
        for bs, (dn, dms) in (x.get("rnd") or {}).items():
            a = ref.setdefault(bs, [0.0, 0.0])
            a[0] += dn
            a[1] += dms
    rows, tn, tms, wn, wms = [], 0.0, 0.0, 0.0, 0.0
    for bs, (dn, dms) in sorted(acc.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        if dn <= 0:
            continue
        r = ref.get(bs)
        sm = (r[1] / r[0]) if r and r[0] >= SOLO_MIN_ROUNDS else None
        rows.append({"bs": int(bs) if bs.isdigit() else bs, "ms": dms / dn, "n": dn, "solo_ms": sm})
        tn += dn
        tms += dms
        if sm is not None:
            wn += dn
            wms += dn * sm
    return {"ms": (tms / tn) if tn > 0 else None, "n": tn, "by_bs": rows,
            "solo_ms": (wms / wn) if wn > 0 and wn >= 0.5 * tn else None}


#: a dual D prefill chunk with at most this many NEW tokens (on average per chunk) and a cached prefix is a
#: resume extend (#988 LOADBACK + 1-token extend: P's finished request taken over by D), not prefill work
RESUME_NEW_TOK = 64
#: round-time reference per bs needs this many rounds without P (else no reference is shown)
SOLO_MIN_ROUNDS = 20


def admit_in(chunks: List[dict], s: float, e: float) -> dict:
    """The 1-token admit extends (chunks(kind="admit")) that ended inside [s, e]: {n, tok}."""
    hit = [c for c in chunks if s - 0.05 <= c["e0"] <= e + 0.05]
    return {"n": int(sum(c.get("n") or 1 for c in hit)), "tok": int(sum(c["tok"] for c in hit))}


def co_extends(chunks: List[dict], s: float, e: float) -> dict:
    """What D's prefill chunks inside [s, e] are in a dual boot (Nutzer 03.10.: "D 1-30 tok/s ... falsch"): the
    chunks (``n``), their cached prefix tokens (``cached``, spread over the burst like the new tokens) and new
    tokens (``new``).  ``resume`` = every chunk carries a cached prefix and at most RESUME_NEW_TOK new tokens:
    that is a hand-over from P (loadback + 1-token extend), whose tok/s say nothing about prefill speed."""
    hit = [c for c in chunks if min(c["e"], e) - max(c["s"], s) > -0.05]
    if not hit:     # a slice of the burst between two chunk windows: the burst's chunks (they spread over it)
        hit = [c for c in chunks if any(min(y, e) > max(x, s) for x, y in c.get("parts") or ())]
    n = int(sum(c.get("n") or 1 for c in hit))
    resume = bool(hit) and all((c.get("cached") or 0) > 0 and (c.get("tok") or 0) <= RESUME_NEW_TOK * max(1, c.get("n") or 1)
                               for c in hit)
    return {"n": n, "cached": spread(chunks, s, 1, max(1e-3, e - s), "cached")[0],
            "new": spread(chunks, s, 1, max(1e-3, e - s), "tok")[0], "resume": resume}


def decode_reqs(ring, key: Optional[str], s: float, e: float, done: Optional[List[dict]] = None) -> Tuple[List[dict], Optional[str]]:
    """Per request in the decode stretch [s, e]: Token x-y (n neu) and its tok/s.  (rows, src).
    1. rankstats decode.reqs [[rid, prompt, out]] of the samples around [s, e] (exact; port seat 02.10.):
       x = prompt + out at the segment's start, y at its end, n = the out delta, tok/s = n / rank-clock span;
    2. else events request_done of the requests whose decode [first_token_ts, end_ts] overlaps [s, e]:
       the request's own mean rate (decode_tokens / its decode wall time, parks included) laid linearly over
       its decode -- an estimate, and a request still running is not there yet."""
    seen: Dict[str, List[Tuple[float, int, int]]] = {}
    seen_field = False
    if key is not None:
        for smp in ring or ():
            r = smp["r"].get(key)
            if not r or r.get("ts") is None or r.get("dreqs") is None:
                continue
            seen_field = True
            if not (s - 1.5 <= r["ts"] <= e + 1.5):
                continue
            for rid, p, o in r["dreqs"]:
                seen.setdefault(rid, []).append((r["ts"], p, o))
    if seen_field:
        rows = []
        for rid, xs in seen.items():
            # the depth at the segment's edges: the newest sample at or before s, the first at or after e
            # (the 1-s raster), else the first / last sample inside
            t0, p0, o0 = max((x for x in xs if x[0] <= s), default=xs[0], key=lambda x: x[0])
            t1, p1, o1 = min((x for x in xs if x[0] >= e), default=xs[-1], key=lambda x: x[0])
            n = max(0, o1 - o0)
            rows.append({"rid": rid, "x": p0 + o0, "y": p1 + o1, "n": n,
                         "tps": (n / (t1 - t0)) if t1 - t0 >= 0.5 else None, "est": False})
        rows.sort(key=lambda r: -r["y"])
        return rows, "rankstats decode.reqs (je Probe)"
    rows = []
    for r in done or ():
        ft, end, dt = r.get("first_token_ts"), r.get("end_ts"), r.get("decode_tokens")
        if ft is None or end is None or not dt or end <= ft or end <= s or ft >= e:
            continue
        ft, end, dt = float(ft), float(end), int(dt)
        ctx = r.get("context_tokens")
        p0 = int(ctx) - dt if ctx else None
        if p0 is None or p0 < 0:
            pre = r.get("prefill") if isinstance(r.get("prefill"), dict) else {}
            px = pre.get("D") or pre.get("P") or {}
            p0 = int(px.get("prompt") or 0)
        rate = dt / (end - ft)
        x = p0 + rate * (max(s, ft) - ft)
        y = p0 + rate * (min(e, end) - ft)
        rows.append({"rid": r.get("rid"), "x": int(round(x)), "y": int(round(y)), "n": int(round(y - x)),
                     "tps": rate, "est": True})
    rows.sort(key=lambda r: -r["y"])
    return rows, ("events request_done (avg of the request, linear over its decode time)" if rows else None)


def flip_windows(flip_done: List[dict]) -> List[Tuple[float, float]]:
    return sorted((x["flip_begin_ts"], x["t"]) for x in flip_done or ()
                  if x.get("flip_begin_ts") is not None and x.get("t") is not None and x["t"] >= x["flip_begin_ts"])


def open_flips(begins: Optional[List[dict]], flips: List[Tuple[float, float]]) -> List[dict]:
    """flip_begin events without a flip_done.  Only the newest begin can still be in flight: one front runs
    one flip at a time, so a begin that a later begin follows was abandoned (NF y8c 0cf3 19:33:06, the boot's
    first D>P begun, then begun anew at 19:33:20 with the same epoch_before -- drawn as open to "now" it
    painted every later P phase as FLIP D->P).  Such a begin ends at the next begin (``open_end``); the newest
    one has ``open_end`` None = still open."""
    done_b = [s for s, _ in flips]
    starts = sorted({float(x["flip_begin_ts"]) for x in begins or () if x.get("flip_begin_ts") is not None}
                    | set(done_b))
    out = []
    for x in begins or ():
        b = x.get("flip_begin_ts")
        if b is None or any(abs(b - d) < 1.0 for d in done_b):
            continue
        nxt = next((t for t in starts if t > b + 1.0), None)
        out.append(dict(x, open_end=nxt))
    return out


def _minus(s: float, e: float, wins) -> List[Tuple[float, float]]:
    """[s, e] without the windows."""
    parts = [(s, e)]
    for a, b in wins:
        nxt = []
        for x, y in parts:
            if b <= x or a >= y:
                nxt.append((x, y))
                continue
            if x < a:
                nxt.append((x, a))
            if b < y:
                nxt.append((b, y))
        parts = nxt
    return [(x, y) for x, y in parts if y - x > 1e-3]


def _bs_of(a: dict, b: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(time-weighted mean, min, max) batch size of the decode rounds between two records of one rank,
    from the Δ of ``decode.gpu_ms_by_bs`` {bs: [rounds, ms]} -- every round of the interval by its own
    duration.  (None, None, None) without the counter (then the caller falls back to decode.running)."""
    x, y = b.get("by_bs"), a.get("by_bs") or {}
    if not isinstance(x, dict):
        return None, None, None
    num = den = 0.0
    seen = []
    for k, v in x.items():
        try:
            bs = float(k)
            n1, ms1 = float(v[0]), float(v[1])
            p = y.get(k) or (0, 0.0)
            n0, ms0 = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        dn, dms = n1 - n0, ms1 - ms0
        if dn < 0 or dms < 0:              # the counter restarted
            dn, dms = n1, ms1
        if dn > 0 and dms > 0:
            num += bs * dms
            den += dms
            seen.append(bs)
    if den <= 0:
        return None, None, None
    return num / den, min(seen), max(seen)


def _rounds_of(a: dict, b: dict) -> Dict[str, Tuple[float, float]]:
    """{bs: (rounds, ms)} of the decode rounds between two records of one rank (Δ of ``decode.gpu_ms_by_bs``,
    the counter's restart handled as in _bs_of).  {} without the counter."""
    x, y = b.get("by_bs"), a.get("by_bs") or {}
    out: Dict[str, Tuple[float, float]] = {}
    if not isinstance(x, dict):
        return out
    for k, v in x.items():
        try:
            n1, ms1 = float(v[0]), float(v[1])
            p = y.get(k) or (0, 0.0)
            n0, ms0 = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        dn, dms = n1 - n0, ms1 - ms0
        if dn < 0 or dms < 0:
            dn, dms = n1, ms1
        if dn > 0 and dms > 0:
            out[str(k)] = (dn, dms)
    return out


def decode_intervals(ring, keys, g: str, excl) -> List[dict]:
    """Decode work between two samples of the group's first rank, flip windows cut out.  ``steady``
    = decode also in the sample before and after (a full-speed interval, the per-stream instrument).

    The honest denominator (Nutzer 30.09. ~21:05Z: "der decode durchsatz ... extrem sprunghaft"):
    ``busy`` = the seconds of the interval D really decoded.  A steady interval decoded all of its
    time; the first and the last interval of a stretch hold time before the first / after the last
    round, so there ``busy`` = the rounds' own time (Δdecode.gpu_ms) scaled by the boot's measured
    wall/GPU ratio of the steady intervals, capped at the interval.  ``seat_s`` = busy x the mean
    batch size of its rounds (Δgpu_ms_by_bs, time-weighted; else decode.running) -- the seat-seconds
    the tokens were produced in, so tokens / seat_s is the rate per stream and seat_s / busy the mean
    number of seats, both only over the time D decoded (sleep and flips never count as 0 seats)."""
    k0, _ = stage_keys(keys, g)
    if k0 is None:
        return []
    raw = []
    for a, b in rank_pairs(ring, k0):
        dk = _d(b, a, "dtok")
        raw.append((a, b, dk if dk and dk > 0 else 0.0))
    out = []
    for i, (a, b, dk) in enumerate(raw):
        if dk <= 0:
            continue
        parts = _minus(a["ts"], b["ts"], excl)
        dur = sum(y - x for x, y in parts)
        if dur <= 0:
            continue
        steady = (0 < i < len(raw) - 1 and raw[i - 1][2] > 0 and raw[i + 1][2] > 0
                  and dur >= 0.8 * (b["ts"] - a["ts"]))
        run = b.get("running") or a.get("running")
        bs, bmin, bmax = _bs_of(a, b)
        bsrc = "by_bs" if bs is not None else None
        if bs is None and run:
            bs = bmin = bmax = float(run)
            bsrc = "running"
        out.append({"s": parts[0][0], "e": parts[-1][1], "parts": parts, "dur": dur, "tok": dk,
                    "gpu_ms": _d(b, a, "dgpu"), "rnd": _rounds_of(a, b),
                    "run": run, "bs": b.get("last_bs"), "steady": steady,
                    "bs_mean": bs, "bs_min": bmin, "bs_max": bmax, "bs_src": bsrc,
                    "stream": (dk / dur / run) if steady and run else None})
    # the wall/GPU ratio of steady decoding: a round's gpu_ms misses the host time between rounds
    sd = sum(x["dur"] for x in out if x["steady"] and (x["gpu_ms"] or 0) > 0)
    sg = sum(x["gpu_ms"] / 1000.0 for x in out if x["steady"] and (x["gpu_ms"] or 0) > 0)
    ratio = min(3.0, max(1.0, sd / sg)) if sg > 0 else 1.0
    for x in out:
        if x["steady"] or not (x["gpu_ms"] or 0) > 0:
            x["busy"] = x["dur"]
        else:
            x["busy"] = max(1e-3, min(x["dur"], ratio * x["gpu_ms"] / 1000.0))
        x["seat_s"] = x["busy"] * x["bs_mean"] if x["bs_mean"] else None
    return out


def decode_stretches(iv: List[dict]) -> List[dict]:
    """Decode stretches; their rate over the steady intervals only (no rate below 2 s of them)."""
    out: List[dict] = []
    for x in iv:
        if out and x["s"] <= out[-1]["e"] + DEC_MERGE_S:
            o = out[-1]
            o["e"] = x["e"]
        else:
            out.append({"s": x["s"], "e": x["e"], "tok": 0.0, "dur": 0.0, "all_tok": 0.0})
            o = out[-1]
        o["all_tok"] += x["tok"]
        if x["steady"]:
            o["tok"] += x["tok"]
            o["dur"] += x["dur"]
    for o in out:
        o["rate"] = o["tok"] / o["dur"] if o["dur"] >= 2.0 else None
    return out


def _busy_spans(cs: List[dict]) -> List[dict]:
    """The time prefill ran: the unique spans its chunks' tokens are spread over (one per burst), so
    tokens / busy is the burst rate with exactly the denominator the curve used."""
    seen = sorted({(a, b) for c in cs for a, b in (c.get("parts") or [(c["s"], c["e"])]) if b > a})
    return [{"s": a, "e": b, "parts": [(a, b)], "w": b - a} for a, b in seen]


def prefill_requests(cs: List[dict], is_d: bool = False) -> List[dict]:
    """The requests behind a group's prefill chunk records (user 08.10.: "input tokens (from cache)": a jump to the
    cached prefix, then the newly prefilled tokens rise on top of it).  rankstats count a request's cached prefix
    ONCE, at its first chunk (``prefill.cached_tokens``, later chunks add 0), so a record with cached > 0 starts a
    request, a record without it extends the running one while it follows within CHUNK_GAP_S.  ``pts`` = (time, new
    tokens so far): 0 at the start, the running sum at the end of each chunk record; ``cached`` stays level from
    the start.  Without the per-chunk extent (prefill.last.ext) this is the finest honest split."""
    out: List[dict] = []
    cur = None
    for c in sorted(cs, key=lambda x: x["e0"]):
        if cur is None or (c.get("cached") or 0) > 0 or c["s"] > cur["e"] + CHUNK_GAP_S:
            cur = {"s": c["s"], "e": c["e0"], "cached": float(c.get("cached") or 0.0), "d": is_d,
                   "pts": [(c["s"], 0.0)], "new": 0.0}
            out.append(cur)
        cur["new"] += c.get("tok") or 0.0
        cur["pts"].append((max(c["e0"], cur["pts"][-1][0]), cur["new"]))
        cur["e"] = max(cur["e"], c["e0"])
    return out


def _new_at(pts: List[Tuple[float, float]], t: float) -> float:
    """New tokens of a request by time t: linear between its chunk ends, 0 before the start, all after the end."""
    if t <= pts[0][0]:
        return 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * ((t - t0) / (t1 - t0) if t1 > t0 else 1.0)
    return pts[-1][1]


def prefill_levels(reqs: List[dict], lo: float, n: int, step: float) -> Dict[str, List[Optional[float]]]:
    """Per bucket the prefill in progress at the END of the bucket (the request that finished its newest chunk last
    among those that began before the bucket's end): ``pf_cached`` = its cached prefix (level from its start),
    ``pf_new`` = the tokens it has newly prefilled by then, ``pf_isd`` = 1 for a D request.  None = no prefill in
    progress (a gap, never 0): the curve starts at the cached prefix and climbs, it does not come up from zero."""
    cached: List[Optional[float]] = [None] * n
    new: List[Optional[float]] = [None] * n
    isd: List[Optional[float]] = [None] * n
    for r in sorted(reqs, key=lambda x: (x["e"], x["s"])):
        i0, i1 = int((r["s"] - lo) // step), int((r["e"] - lo - 1e-9) // step)
        for i in range(max(0, i0), min(n - 1, i1) + 1):
            t = min(lo + (i + 1) * step, r["e"])
            cached[i], new[i], isd[i] = r["cached"], _new_at(r["pts"], t), 1.0 if r["d"] else 0.0
    return {"pf_cached": cached, "pf_new": new, "pf_isd": isd}


def spread(items, lo: float, n: int, step: float, key: str = "tok", excl=None) -> List[float]:
    """Tokens of each interval spread uniformly over the time it ran (flip windows cut out), per bucket."""
    acc = [0.0] * n
    hi = lo + n * step
    for it in items:
        parts = it.get("parts") or _minus(it["s"], it["e"], excl or ())
        tot = sum(y - x for x, y in parts)
        if tot <= 0 or not it.get(key):
            continue
        dens = it[key] / tot
        for x, y in parts:
            x, y = max(x, lo), min(y, hi)
            while x < y:
                i = int((x - lo) // step)
                z = min(y, lo + (i + 1) * step)
                if 0 <= i < n:
                    acc[i] += dens * (z - x)
                x = z
    return acc


#: batch sizes the history tells apart in the decode throughput (user 08.10.: "bs1 bs2 bs3 bs4 bs5 bs6")
BS_CLASSES = (1, 2, 3, 4, 5, 6)
#: the states of the phase bar (Nutzer 30.09. ~18Z: "idle sieht aus wie flip ... das muss eindeutig
#: unterscheidbar sein").  Each comes from data; what no data explains is "unknown", never idle or flip.
STATES = ("P", "D", "dec", "flip_pd", "flip_dp", "flip_tail", "vis_load", "vis_enc", "vis_unload",
          "idle", "off", "unknown")
#: Nutzer 02.10. ~11:00Z: "der visiontower laden rechnen entladen auch mit in die phasenliste" -- the
#: transient tower stage on P's PP0 (rankstats ``vision``, legs with wall clock): laden = build + reserve
#: + load, rechnen = encode + attach, entladen = teardown.  It runs in P's admission after the wake, so it
#: sits inside the flip tail and before the P prefill: it wins over both.
VIS_LEGS = {"build": "vis_load", "reserve": "vis_load", "load": "vis_load",
            "encode": "vis_enc", "attach": "vis_enc", "teardown": "vis_unload"}
VIS_PRIO = 0.5
#: Nutzer 03.10. ~17:50Z: "beim 27b dual nvfp4 die prefill/decode schraffiert übereinanderliegen und auch im
#: tooltip beim hovern beides angezeigt werden in solch einer phase".  In the dual layout P (PP) and D (TP) run
#: AT THE SAME TIME on their own cards (no flip): a P prefill segment in which D placed work too (its decode
#: rounds or its prefill/extend chunks, the same token model) carries ``co`` = "dec" / "D".  Kept as the P
#: segment (the phase states, the Flipzeit and every flip line stay as they are); the bar draws it hatched in
#: both colours and the hover names both.  Bucket shares of it: ph_<CO_STATES> (only for a dual boot).
CO_STATES = {"Pdec": "dec", "PD": "D"}
SAMPLE_GAP_S = 3.0     # two dashboard samples further apart: the time between is unobserved (unknown)
RANK_GAP_S = 30.0       # two records of one rank further apart: the rank stalled or was gone -- not observed


def vision_runs(ring) -> Tuple[List[dict], Optional[dict]]:
    """Tower stages from the ranks' rankstats ``vision`` (compacted by ipcboot.vision_compact): every
    finished run once (a rank keeps the newest runs, so one run sits in many samples), oldest first;
    and the leg a stage is in right now (the newest sample's ``live``), else None."""
    seen: Dict[Tuple, dict] = {}
    live = None
    for s in ring:
        for k, r in (s.get("r") or {}).items():
            v = r.get("vis") if isinstance(r, dict) else None
            if not isinstance(v, dict):
                continue
            for run in v.get("recent") or []:
                if run.get("t0") is None or not run.get("legs"):
                    continue
                seen[(_grp(k), run.get("run"), round(float(run["t0"]), 2))] = dict(run, group=_grp(k))
    if ring:
        for k, r in (ring[-1].get("r") or {}).items():
            v = r.get("vis") if isinstance(r, dict) else None
            if isinstance(v, dict) and isinstance(v.get("live"), dict):
                live = dict(v["live"], group=_grp(k))
    return sorted(seen.values(), key=lambda x: x["t0"]), live


def _outstanding(front: dict) -> Optional[float]:
    o = (front or {}).get("outstanding")
    if isinstance(o, dict):
        vals = [v for v in o.values() if isinstance(v, (int, float))]
        return float(sum(vals)) if vals else None
    return float(o) if isinstance(o, (int, float)) else None


class Model:
    """All activity of one boot from its ring + events; the views and the history read this.

    ``begins`` = flip_begin event data (an open flip has no flip_done yet), ``user_time`` =
    flip_user_time data (D>P: end of the flip tail = P prefill start), ``life`` = {serving_since,
    terminal_since, terminal_state} from state.json, ``dual`` = the boot runs the dual layout (P and D
    at the same time, ipcboot.is_dual): only then a P segment names D's simultaneous work (``co``)."""

    def __init__(self, ring, flip_done: List[dict], first_work: List[dict], begins: Optional[List[dict]] = None,
                 user_time: Optional[List[dict]] = None, life: Optional[dict] = None, dual: bool = False):
        self.dual = bool(dual)
        self.ring = list(ring or ())
        self.keys = set()
        for s in self.ring:
            self.keys.update(s["r"].keys())
        self.groups = sorted({_grp(k) for k in self.keys})
        self.flip_done = [x for x in flip_done or () if x.get("flip_begin_ts") is not None and x.get("t") is not None]
        self.flips = flip_windows(flip_done)
        # who is awake when: the woken group from flip_done on, the slept group before the first flip
        self.wakes = sorted((x["t"], x.get("wake"), x.get("sleep")) for x in flip_done or ()
                            if x.get("t") is not None and x.get("wake"))
        self.first_work = [x for x in first_work or () if x.get("flip_begin_ts") is not None]
        self.user_time = [x for x in user_time or () if x.get("prefill_start_ts") is not None]
        self.life = life or {}
        self.open_flips = open_flips(begins, self.flips)
        self.pchunks = {g: chunks(self.ring, self.keys, g) for g in self.groups if g in ("P", "D", "single")}
        # D: real prefill (wide) and the 1-token admit extends apart (WIDE_MIN_TOK); the phase bar keeps pchunks
        self.dwide = chunks(self.ring, self.keys, "D", "wide") if "D" in self.pchunks else []
        self.dadmit = chunks(self.ring, self.keys, "D", "admit") if "D" in self.pchunks else []
        dg = "D" if "D" in self.groups else ("single" if "single" in self.groups else None)
        self.dec_group = dg
        excl = list(self.flips) + [(s, e) for s, e, _ in self.tails()] + \
            [(c["s"], c["e"]) for c in self.pchunks.get("D", [])]
        self.dec = decode_intervals(self.ring, self.keys, dg, sorted(excl)) if dg else []
        self.vis_runs, self.vis_live = vision_runs(self.ring)

    # --- phases -----------------------------------------------------------
    def vision_spans(self, now: Optional[float] = None) -> List[Tuple[float, float, str, dict]]:
        """(s, e, vis_*, run) per tower-stage leg; the running stage (``live``) up to ``now``."""
        out = []
        for r in self.vis_runs:
            for leg, (s, e) in sorted(r["legs"].items(), key=lambda kv: kv[1][0]):
                k = VIS_LEGS.get(leg)
                if k is not None and e >= s:
                    out.append((s, max(e, s + 1e-3), k, r))
        lv = self.vis_live
        end = now if now is not None else (self.ring[-1]["t"] if self.ring else None)
        if lv and lv.get("since") is not None and end is not None and end > lv["since"] \
                and not any(r.get("run") == lv.get("run") for r in self.vis_runs):
            k = VIS_LEGS.get(lv.get("leg") or "")
            if k is not None:
                out.append((lv["since"], end, k, dict(lv, live=True)))
        return out

    def _dp_user_time(self, b: float, done: float) -> Optional[dict]:
        return next((u for u in self.user_time if done - 30 <= u["prefill_start_ts"] and u["prefill_start_ts"] >= b
                     and abs((u.get("start_ts") or b) - b) < 30), None)

    def _dp_ends(self) -> List[Tuple[float, float, Optional[float]]]:
        """(done, end, first chunk start) per D>P flip: end = dp_prefill_start (first forward on P's first
        stage, Nutzer 02.10.), the first chunk start of the burst after it (the rank prefill.last record),
        which lies later on PP > 1 -- the stretch between is P's pipeline fill, prefill, not flip."""
        out = []
        pcs = self.pchunks.get("P", []) + self.pchunks.get("single", [])
        for x in self.first_work:
            if x.get("what") == "none" or x.get("dir") != "D>P":
                continue
            b = x["flip_begin_ts"]
            done = next((e for s, e in self.flips if abs(s - b) < 1.0), None)
            if done is None:
                continue
            u = self._dp_user_time(b, done)
            # the first forward starts after flip_begin and after P's leg-1 dispatch
            t_from = float(b)
            if u is not None and u.get("prefill_start_source") != DP_END_SOURCE:
                t_from = max(t_from, float(u.get("prefill_start_ts") or 0.0))
            r = dp_prefill_start(self.ring, self.keys, u, t_from, done + 120.0)
            if r is None:
                continue
            first = min((c["s"] for c in pcs if done - 0.5 <= c["s"] <= r[0] + 30.0), default=None)
            out.append((done, r[0], first))
        return out

    def tails(self) -> List[Tuple[float, float, str]]:
        """Flip tail = flip_done -> first work after it: P>D the first decode token (flip_first_work), D>P the
        first prefill forward on P's first stage (dp_prefill_start -- never the leg-1 dispatch); what="none"
        has none."""
        out = []
        for x in self.first_work:
            if x.get("what") == "none" or x.get("dir") == "D>P":
                continue
            b = x["flip_begin_ts"]
            done = next((e for s, e in self.flips if abs(s - b) < 1.0), None)
            if done is None:
                continue
            end = float(x["first_work_ts"]) if x.get("first_work_ts") is not None else (
                b + float(x["flip_time_ms"]) / 1000.0 if x.get("flip_time_ms") is not None else None)
            if end is not None and end > done:
                out.append((done, end, x.get("dir") or ""))
        for done, end, _ in self._dp_ends():
            if end > done:
                out.append((done, end, "D>P"))
        return out

    def pipeline_fills(self) -> List[Tuple[float, float]]:
        """P's pipeline fill after a D>P flip: first forward on PP0 -> start of the first chunk record (on PP > 1
        the chunk passes PP0, PP1, ... before the last stage starts) -- prefill, drawn as P."""
        return [(end, first) for _, end, first in self._dp_ends() if first is not None and first > end]

    def _flip_kind(self, sleep, wake) -> str:
        return "flip_dp" if (sleep, wake) == ("D", "P") else "flip_pd"

    def segments(self, now: Optional[float] = None) -> List[dict]:
        """Non-overlapping segments, each from data (priority left to right):
        flip (flip_begin..flip_done, an open flip up to now) > flip tail > P prefill > D prefill/extend >
        decode > aus/lädt/tot (before serving_since / after the terminal lifecycle) > idle (a sample
        pair with no rank counter moving AND the front's queue = 0 and outstanding = 0) > unknown
        (a sample pair with work queued but no rank moving, a counter moving without tokens, or
        no sample at all)."""
        raw = []
        for x in self.flip_done:
            raw.append((x["flip_begin_ts"], x["t"], self._flip_kind(x.get("sleep"), x.get("wake")), 0))
        end_all = now if now is not None else (self.ring[-1]["t"] if self.ring else None)
        for x in self.open_flips:
            end = x["open_end"] if x["open_end"] is not None else end_all
            if end is not None and end > x["flip_begin_ts"]:
                raw.append((x["flip_begin_ts"], end, self._flip_kind(x.get("sleep"), x.get("wake")), 0))
        for s, e, _ in self.tails():
            raw.append((s, e, "flip_tail", 1))
        for s, e in self.pipeline_fills():
            raw.append((s, e, "P", 2))
        for s, e, k, _ in self.vision_spans(now):
            raw.append((s, e, k, VIS_PRIO))
        for g, cs in self.pchunks.items():
            k = "D" if g == "D" else "P"
            # a burst is one prefill phase: between two chunks of it the pipeline's other stages work
            for c in bursts(cs):
                raw.append((c["s"], c["e"], k, 2 if k == "P" else 3))
        for d in self.dec:
            for x, y in d["parts"]:
                raw.append((x, y, "dec", 4))
        ss, ts = self.life.get("serving_since"), self.life.get("terminal_since")
        if self.ring:
            t0 = self.ring[0]["t"]
            if ss is not None and t0 < ss:
                raw.append((t0, ss, "off", 5))
            if ts is not None:
                raw.append((ts, max(ts, end_all or ts), "off", 5))
        work = [r for r in raw if r[2] in ("P", "D", "dec")]
        for a, b in zip(self.ring, self.ring[1:]):
            if b["t"] - a["t"] > SAMPLE_GAP_S:
                raw.append((a["t"], b["t"], "unknown", 7))
                continue
            moved = any((_d(b["r"].get(k) or {}, a["r"].get(k) or {}, f) or 0) > 0
                        for k in self.keys for f in ("pnew", "dtok", "fwd", "pchunks"))
            q = (b.get("front") or {}).get("queue")
            o = _outstanding(b.get("front") or {})
            if moved:
                # a counter moved in this sample interval: the work placed by the token model that
                # overlaps it covers the rest of it (rank clocks and the sample clock differ by < 1 s);
                # without any placed work it is work we cannot name
                near = [w for w in work if w[0] < b["t"] and w[1] > a["t"]]
                if near:
                    w = max(near, key=lambda w: min(w[1], b["t"]) - max(w[0], a["t"]))
                    raw.append((a["t"], b["t"], w[2], w[3] + 0.5))
                else:
                    raw.append((a["t"], b["t"], "unknown", 7))
            elif q == 0 and o == 0:
                raw.append((a["t"], b["t"], "idle", 6))
            else:
                raw.append((a["t"], b["t"], "unknown", 7))
        cuts = sorted({t for s, e, _, _ in raw for t in (s, e)})
        segs: List[dict] = []
        for a, b in zip(cuts, cuts[1:]):
            live = [r for r in raw if r[0] <= a and r[1] >= b]
            if not live:
                continue
            top = min(live, key=lambda r: r[3])
            k, co = top[2], None
            if self.dual and k == "P" and top[3] == 2:
                # dual layout: P's placed prefill and D's placed work (its own token model: decode rounds
                # prio 4, prefill/extend chunks prio 3 -- never a sample fill-in) in the same stretch
                dw = [r for r in live if r[2] in ("D", "dec") and r[3] in (3, 4)]
                if dw:
                    co = min(dw, key=lambda r: r[3])[2]
            if segs and segs[-1]["k"] == k and segs[-1].get("co") == co and abs(segs[-1]["e"] - a) < 1e-6:
                segs[-1]["e"] = b
            else:
                segs.append({"s": a, "e": b, "k": k})
                if co:
                    segs[-1]["co"] = co
        for x in segs:
            if x["k"] == "unknown":
                x["why"] = self._why_unknown(x)
            if x["k"] == "off":
                x["why"] = "loading (before serving)" if ss is not None and x["e"] <= ss + 1e-6 else \
                    "off/dead (%s)" % (self.life.get("terminal_state") or "beendet")
        return segs

    def _why_unknown(self, x) -> str:
        return self._why_unknown_base(x)

    def _why_unknown_base(self, x) -> str:
        pairs = [(a, b) for a, b in zip(self.ring, self.ring[1:]) if a["t"] < x["e"] and b["t"] > x["s"]]
        if not pairs or any(b["t"] - a["t"] > SAMPLE_GAP_S for a, b in pairs):
            return "no IPC sample in this time"
        if any(any((_d(b["r"].get(k) or {}, a["r"].get(k) or {}, f) or 0) > 0 for k in self.keys for f in ("fwd", "pchunks"))
               for a, b in pairs):
            return "Rank counter moves, but without assignable tokens"
        return "Requests open (queue/outstanding > 0), but no rank is working"

    def overlap_s(self) -> Dict[str, float]:
        """Seconds in which P prefill and D work (prefill or decode) ran at the same time -- outside
        the dual layout that is a measuring error; the audit reads this."""
        p = [(c["s"], c["e"]) for g, cs in self.pchunks.items() if g in ("P", "single") for c in cs]
        d = [(c["s"], c["e"]) for c in self.pchunks.get("D", [])] + [xy for x in self.dec for xy in x["parts"]]
        tot = 0.0
        for a, b in p:
            for x, y in d:
                tot += max(0.0, min(b, y) - max(a, x))
        return {"p_vs_d_s": round(tot, 3)}

    def awake_at(self, t: float, default=None):
        cur = self.wakes[0][2] if self.wakes else default
        for tt, wake, _ in self.wakes:
            if tt <= t:
                cur = wake
            else:
                break
        return cur or default

    # --- buckets ------------------------------------------------------------
    def coverage(self, lo: float, n: int, step: float) -> List[float]:
        """Seconds of each bucket the dashboard watched: the stretch between two consecutive samples
        at most SAMPLE_GAP_S apart.  The sampler's loop slips (measured on the live service 30.09.
        ~21:10Z: ~28 % of the 1-s rows held no sample, because a poll took > 1 s under /api/live load),
        but the work between two samples is placed by the rank clocks (spread), so such a second is
        observed, not missing -- it was drawn as a gap and broke every curve into pieces."""
        cov = [0.0] * n
        hi = lo + n * step
        spans = [(a["t"], b["t"]) for a, b in zip(self.ring, self.ring[1:]) if b["t"] - a["t"] <= SAMPLE_GAP_S]
        # counters by the RANK's clock (Nutzer 30.09. ~21:40Z: "der probenehmer sollte doch nicht an zu
        # viel last scheitern"): two records of one rank bound an interval whose Δcounters are complete,
        # however late the dashboard read the second one -- that interval is observed, up to RANK_GAP_S
        for k in self.keys:
            spans += [(a["ts"], b["ts"]) for a, b in rank_pairs(self.ring, k) if b["ts"] - a["ts"] <= RANK_GAP_S]
        for sa, sb in spans:
            x, y = max(sa, lo), min(sb, hi)
            while x < y:
                i = int((x - lo) // step)
                z = min(y, lo + (i + 1) * step)
                if 0 <= i < n:
                    cov[i] += z - x
                x = z
        for s in self.ring:                      # a lone sample still marks its bucket as seen
            i = int((s["t"] - lo) // step)
            if 0 <= i < n and cov[i] <= 0:
                cov[i] = 1e-3
        return cov

    def buckets(self, lo: float, n: int, step: float) -> Dict[str, List[Optional[float]]]:
        """Rates (tok/s) and levels per bucket; None where the dashboard did not watch the boot.

        Next to the wall rates (tokens / bucket, the energy and token accounting) the honest
        denominators (Nutzer 30.09. ~21:05Z): ``dec_busy`` / ``p_busy`` / ``d_busy`` = the share of the
        bucket the phase really worked, ``dec_seat`` = seat-seconds per second, ``dec_bs_min/max`` =
        the smallest / largest batch of its rounds.  Tokens / busy is the rate WHILE working, which
        stays level across the D-extend and flip pauses instead of dipping to their share of the bucket;
        stored as shares, the division stays right at every history tier."""
        cov = self.coverage(lo, n, step)
        have = [c > 0 for c in cov]
        sampled = [False] * n
        kv = [[0.0, 0] for _ in range(n)]
        kvp = [[0.0, 0] for _ in range(n)]
        kD, _ = stage_keys(self.keys, self.dec_group) if self.dec_group else (None, None)
        kP, _ = stage_keys(self.keys, "P")
        for s in self.ring:
            i = int((s["t"] - lo) // step)
            if not 0 <= i < n:
                continue
            have[i] = sampled[i] = True
            # flip layout (Nutzer 01.10.): the group that sleeps holds no KV on its cards, so its
            # curve drops to 0 while the other layout runs; a sleeping rank's rankstats still carry
            # its last value.  The dual layout has no flips (no wakes), both groups stay as read.
            awake = self.awake_at(s["t"]) if self.wakes else None
            for key, acc, grp in ((kD, kv, self.dec_group), (kP, kvp, "P")):
                r = s["r"].get(key) if key else None
                if awake is not None and grp != awake and key:
                    acc[i][1] += 1
                    continue
                if r and r.get("kv") is not None:
                    acc[i][0] += 100.0 * r["kv"]
                    acc[i][1] += 1
        pc = self.pchunks
        pcs = pc.get("P", []) + pc.get("single", [])
        p_tok = spread(pcs, lo, n, step)
        d_tok = spread(pc.get("D", []), lo, n, step)
        dec = spread(self.dec, lo, n, step)
        cache_p = spread(pcs, lo, n, step, key="cached")
        cache_d = spread(pc.get("D", []), lo, n, step, key="cached")
        p_busy = spread(_busy_spans(pcs), lo, n, step, key="w")
        d_busy = spread(_busy_spans(pc.get("D", [])), lo, n, step, key="w")
        dw_tok = spread(self.dwide, lo, n, step)
        dw_busy = spread(_busy_spans(self.dwide), lo, n, step, key="w")
        dec_busy = spread(self.dec, lo, n, step, key="busy")
        dec_seat = spread(self.dec, lo, n, step, key="seat_s")
        bmin: List[Optional[float]] = [None] * n
        bmax: List[Optional[float]] = [None] * n
        for x in self.dec:
            if x.get("bs_min") is None:
                continue
            for a, b in x["parts"]:
                i0, i1 = int((max(a, lo) - lo) // step), int((min(b, lo + n * step) - lo - 1e-9) // step)
                for i in range(max(0, i0), min(n - 1, i1) + 1):
                    bmin[i] = x["bs_min"] if bmin[i] is None else min(bmin[i], x["bs_min"])
                    bmax[i] = x["bs_max"] if bmax[i] is None else max(bmax[i], x["bs_max"])
        out: Dict[str, List[Optional[float]]] = {}
        g = lambda arr: [(v / step) if have[i] else None for i, v in enumerate(arr)]  # noqa: E731
        out["p_tps"], out["d_tps"], out["dec_tps"] = g(p_tok), g(d_tok), g(dec)
        out["tok_comp_p"], out["tok_comp_d"] = out["p_tps"], out["d_tps"]
        out["tok_cache"], out["tok_dcached"] = g(cache_p), g(cache_d)
        out["p_busy"], out["d_busy"], out["dec_busy"] = g(p_busy), g(d_busy), g(dec_busy)
        out["dec_seat"] = g(dec_seat)
        lv = prefill_levels(prefill_requests(pcs) + prefill_requests(pc.get("D", []), True), lo, n, step)
        for k, arr in lv.items():
            out[k] = [v if have[i] else None for i, v in enumerate(arr)]
        out["dec_bs_min"], out["dec_bs_max"] = bmin, bmax
        # decode per batch size (user 08.10.): only the intervals whose rounds ALL ran at exactly bs = k count
        # for class k (tokens and busy seconds), so tokens / busy of a class is the group's tok/s at that bs
        for k in BS_CLASSES:
            pure = [x for x in self.dec if x.get("bs_min") is not None and x["bs_min"] == x["bs_max"] == k]
            tk, bk = spread(pure, lo, n, step), spread(pure, lo, n, step, key="busy")
            used = [have[i] and bk[i] > 1e-9 for i in range(n)]
            out["dec_bs%d_tps" % k] = [(tk[i] / step) if used[i] else None for i in range(n)]
            out["dec_bs%d_busy" % k] = [(bk[i] / step) if used[i] else None for i in range(n)]
        # rates while working (None = the phase did not work in this bucket: a real gap)
        rate = lambda tok, busy: [(t / b) if have[i] and b > 1e-6 else None  # noqa: E731
                                  for i, (t, b) in enumerate(zip(tok, busy))]
        out["dec_rate"], out["p_rate"], out["d_rate"] = rate(dec, dec_busy), rate(p_tok, p_busy), rate(dw_tok, dw_busy)
        out["seats"] = rate(dec_seat, dec_busy)
        out["stream_tps"] = rate(dec, dec_seat)
        # levels: the mean of the samples in the bucket; a watched bucket without its own sample
        # holds the previous one (sample-and-hold, never a gap inside a watched stretch)
        for name, acc in (("kv_pct", kv), ("kv_p_pct", kvp)):
            arr, last = [], None
            for i, (a, c) in enumerate(acc):
                v = (a / c) if c else (last if have[i] else None)
                arr.append(v)
                if c:
                    last = v
                elif not have[i]:
                    last = None
            out[name] = arr
        out["ipc"] = [1.0 if h else None for h in have]
        # a watched bucket without its own sample: its levels (KV) were held, not read (the "held" count)
        out["held"] = [1.0 if h and not sm else None for h, sm in zip(have, sampled)]
        # phase share per bucket (0..1 per state): averages stay right at every history tier
        frac = {k: [0.0] * n for k in STATES + tuple(CO_STATES)}
        co_of = {v: c for c, v in CO_STATES.items()}
        for x in self.segments():
            a, b = max(x["s"], lo), min(x["e"], lo + n * step)
            cok = co_of.get(x.get("co"))
            while a < b:
                i = int((a - lo) // step)
                z = min(b, lo + (i + 1) * step)
                if 0 <= i < n:
                    frac[x["k"]][i] += (z - a) / step
                    if cok:
                        frac[cok][i] += (z - a) / step
                a = z
        for k in STATES:
            out["ph_" + k] = [(min(1.0, frac[k][i]) if have[i] else None) for i in range(n)]
        if self.dual:
            # the part of ph_P in which D worked at the same time (dual layout only; a flip boot writes none)
            for k in CO_STATES:
                out["ph_" + k] = [(min(1.0, frac[k][i]) if have[i] else None) for i in range(n)]
        return out
