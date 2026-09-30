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


def chunks(ring, keys, g: str) -> List[dict]:
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
        e0 = b.get("plast_t") if b.get("plast_t") is not None else b["ts"]
        # FEHLT 7 (build y5a): own_ms = the chunk's own time after the previous one left -- the
        # chunk's real start; without it gpu_ms, which reaches back into the wait behind the previous
        own = b.get("plast_own") if n == 1 else None
        s = e0 - (own if own is not None else (b.get("plast_gpu") or 0.0)) / 1000.0
        if n > 1:
            # several chunks between two samples: they ran back to back for their summed compute time
            s = min(s, e0 - (_d(b, a, "pcomp") or 0.0) / 1000.0)
        cnt = b.get("pchunks")
        el = next((t for c, t in last_recs if cnt is not None and c >= cnt), None) if kl != k0 else e0
        out.append({"s": s, "e0": e0, "e": max(e0, el if el is not None else e0), "tok": tok,
                    "cached": _d(b, a, "pcached") or 0.0, "n": n, "comp_ms": _d(b, a, "pcomp") or 0.0, "g": g})
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
        else:
            out.append({"s": c["s"], "e": c["e"], "tok": c["tok"], "n": c["n"], "cached": c["cached"]})
    for b in out:
        b["wall_s"] = b["e"] - b["s"]
        b["rate"] = b["tok"] / b["wall_s"] if b["tok"] >= MIN_RATE_TOK and b["wall_s"] > 0 else None
    return out


def flip_windows(flip_done: List[dict]) -> List[Tuple[float, float]]:
    return sorted((x["flip_begin_ts"], x["t"]) for x in flip_done or ()
                  if x.get("flip_begin_ts") is not None and x.get("t") is not None and x["t"] >= x["flip_begin_ts"])


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


def decode_intervals(ring, keys, g: str, excl) -> List[dict]:
    """Decode work between two samples of the group's first rank, flip windows cut out.  ``steady``
    = decode also in the sample before and after (a full-speed interval, the per-stream instrument)."""
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
        out.append({"s": parts[0][0], "e": parts[-1][1], "parts": parts, "dur": dur, "tok": dk,
                    "gpu_ms": _d(b, a, "dgpu"),
                    "run": run, "bs": b.get("last_bs"), "steady": steady,
                    "stream": (dk / dur / run) if steady and run else None})
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


#: the states of the phase bar (Nutzer 30.09. ~18Z: "idle sieht aus wie flip ... das muss eindeutig
#: unterscheidbar sein").  Each comes from data; what no data explains is "unknown", never idle or flip.
STATES = ("P", "D", "dec", "flip_pd", "flip_dp", "flip_tail", "idle", "off", "unknown")
SAMPLE_GAP_S = 3.0      # two dashboard samples further apart: the time between is unobserved (unknown)


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
    terminal_since, terminal_state} from state.json."""

    def __init__(self, ring, flip_done: List[dict], first_work: List[dict], begins: Optional[List[dict]] = None,
                 user_time: Optional[List[dict]] = None, life: Optional[dict] = None):
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
        done_b = [s for s, _ in self.flips]
        self.open_flips = [x for x in begins or () if x.get("flip_begin_ts") is not None
                           and not any(abs(x["flip_begin_ts"] - b) < 1.0 for b in done_b)]
        self.pchunks = {g: chunks(self.ring, self.keys, g) for g in self.groups if g in ("P", "D", "single")}
        dg = "D" if "D" in self.groups else ("single" if "single" in self.groups else None)
        self.dec_group = dg
        excl = list(self.flips) + [(s, e) for s, e, _ in self.tails()] + \
            [(c["s"], c["e"]) for c in self.pchunks.get("D", [])]
        self.dec = decode_intervals(self.ring, self.keys, dg, sorted(excl)) if dg else []

    # --- phases -----------------------------------------------------------
    def tails(self) -> List[Tuple[float, float, str]]:
        """Flip tail = flip_done -> first work after it: P>D the first decode token (flip_first_work),
        D>P the P prefill start (flip_user_time, else flip_first_work's first work); what="none" has none."""
        out = []
        for x in self.first_work:
            if x.get("what") == "none":
                continue
            b = x["flip_begin_ts"]
            done = next((e for s, e in self.flips if abs(s - b) < 1.0), None)
            if done is None:
                continue
            end = None
            if x.get("dir") == "D>P":
                u = next((u for u in self.user_time if done - 30 <= u["prefill_start_ts"] and u["prefill_start_ts"] >= b
                          and abs((u.get("start_ts") or b) - b) < 30), None)
                if u is not None:
                    end = u["prefill_start_ts"]
            if end is None and x.get("first_work_ts") is not None:
                end = float(x["first_work_ts"])
            if end is None and x.get("flip_time_ms") is not None:
                end = b + float(x["flip_time_ms"]) / 1000.0
            if x.get("dir") == "D>P" and end is not None:
                # the front's first-work stamp for D>P is the leg-1 DISPATCH (flip_user_time
                # prefill_start_source "leg1_dispatch" on y4z too); P's prefill really starts with its
                # first chunk (rank prefill.last) -- measured NF y4z 30.09.: up to 5 s later
                pcs = self.pchunks.get("P", []) + self.pchunks.get("single", [])
                first = min((c["s"] for c in pcs if done - 0.5 <= c["s"] <= done + 20.0), default=None)
                if first is not None and first > end:
                    end = first
            if end is not None and end > done:
                out.append((done, end, x.get("dir") or ""))
        return out

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
            if end_all is not None and end_all > x["flip_begin_ts"]:
                raw.append((x["flip_begin_ts"], end_all, self._flip_kind(x.get("sleep"), x.get("wake")), 0))
        for s, e, _ in self.tails():
            raw.append((s, e, "flip_tail", 1))
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
            k = min(live, key=lambda r: r[3])[2]
            if segs and segs[-1]["k"] == k and abs(segs[-1]["e"] - a) < 1e-6:
                segs[-1]["e"] = b
            else:
                segs.append({"s": a, "e": b, "k": k})
        for x in segs:
            if x["k"] == "unknown":
                x["why"] = self._why_unknown(x)
            if x["k"] == "off":
                x["why"] = "lädt (vor serving)" if ss is not None and x["e"] <= ss + 1e-6 else \
                    "aus/tot (%s)" % (self.life.get("terminal_state") or "beendet")
        return segs

    def _why_unknown(self, x) -> str:
        return self._why_unknown_base(x)

    def _why_unknown_base(self, x) -> str:
        pairs = [(a, b) for a, b in zip(self.ring, self.ring[1:]) if a["t"] < x["e"] and b["t"] > x["s"]]
        if not pairs or any(b["t"] - a["t"] > SAMPLE_GAP_S for a, b in pairs):
            return "keine IPC-Probe in dieser Zeit"
        if any(any((_d(b["r"].get(k) or {}, a["r"].get(k) or {}, f) or 0) > 0 for k in self.keys for f in ("fwd", "pchunks"))
               for a, b in pairs):
            return "Rang-Zähler bewegt sich, aber ohne zuordenbare Tokens"
        return "Anfragen offen (queue/outstanding > 0), aber kein Rang arbeitet"

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
    def buckets(self, lo: float, n: int, step: float) -> Dict[str, List[Optional[float]]]:
        """Rates (tok/s) and levels per bucket; None where no sample of the boot lies in the bucket."""
        have = [False] * n
        kv = [[0.0, 0] for _ in range(n)]
        kvp = [[0.0, 0] for _ in range(n)]
        kD, _ = stage_keys(self.keys, self.dec_group) if self.dec_group else (None, None)
        kP, _ = stage_keys(self.keys, "P")
        for s in self.ring:
            i = int((s["t"] - lo) // step)
            if not 0 <= i < n:
                continue
            have[i] = True
            for key, acc in ((kD, kv), (kP, kvp)):
                r = s["r"].get(key) if key else None
                if r and r.get("kv") is not None:
                    acc[i][0] += 100.0 * r["kv"]
                    acc[i][1] += 1
        pc = self.pchunks
        p_tok = spread(pc.get("P", []) + pc.get("single", []), lo, n, step)
        d_tok = spread(pc.get("D", []), lo, n, step)
        dec = spread(self.dec, lo, n, step)
        cache_p = spread(pc.get("P", []) + pc.get("single", []), lo, n, step, key="cached")
        cache_d = spread(pc.get("D", []), lo, n, step, key="cached")
        sw = [[0.0, 0.0] for _ in range(n)]
        for x in self.dec:
            if x["stream"] is None:
                continue
            for a, b in x["parts"]:
                while a < b:
                    i = int((a - lo) // step)
                    z = min(b, lo + (i + 1) * step)
                    if 0 <= i < n:
                        sw[i][0] += x["stream"] * (z - a)
                        sw[i][1] += z - a
                    a = z
        out: Dict[str, List[Optional[float]]] = {}
        g = lambda arr: [(v / step) if have[i] else None for i, v in enumerate(arr)]  # noqa: E731
        out["p_tps"], out["d_tps"], out["dec_tps"] = g(p_tok), g(d_tok), g(dec)
        out["tok_comp_p"], out["tok_comp_d"] = out["p_tps"], out["d_tps"]
        out["tok_cache"], out["tok_dcached"] = g(cache_p), g(cache_d)
        out["stream_tps"] = [(a / b) if b > 0 else None for a, b in sw]
        out["kv_pct"] = [(a / c) if c else None for a, c in kv]
        out["kv_p_pct"] = [(a / c) if c else None for a, c in kvp]
        out["ipc"] = [1.0 if h else None for h in have]
        # phase share per bucket (0..1 per state): averages stay right at every history tier
        frac = {k: [0.0] * n for k in STATES}
        for x in self.segments():
            a, b = max(x["s"], lo), min(x["e"], lo + n * step)
            while a < b:
                i = int((a - lo) // step)
                z = min(b, lo + (i + 1) * step)
                if 0 <= i < n:
                    frac[x["k"]][i] += (z - a) / step
                a = z
        for k in STATES:
            out["ph_" + k] = [(min(1.0, frac[k][i]) if have[i] else None) for i in range(n)]
        return out
