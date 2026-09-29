"""One reader per dashboard field (DASHBOARD-AUS-IPC-INVENTAR-0929, the 25 "Übergang" rows).

User 29.09. ~12:00Z (via 27B): "und warum sind die ganzen werte im dashboard noch aus log.
warum arbeitet da niemand dran?"  -- so every field that still came from a log line gets its
IPC reader NOW, before the producers are in a running image.  The switch is per field and by
presence: when the boot's IPC carries the source, the field is read from there
(``src="ipc"``); when it does not, the log value stays and is labelled ``aus Log (Übergang)``.
No deploy has to wait for a boot: the day a producer lands in an image, its field switches.

Sources (read-only; the writers are weg2/state_file.py, weg2/front.py, weg2/rankstats.py):

  state.json      groups.<G>.form, front.{groups,errors,served_tokens,d_phase_n,d_parked_n,d_seats},
                  lifecycle, serving_since_ts
  events.jsonl    flip_begin, flip_done, flip_first_work, group_health, rank_stop,
                  post_wake_pass, group_ready
  <log>.rankstate/<G>.tp<t>pp<p>.rankstats   weg2.rankstats/1: work, tokens, spec, sched,
                  errors, last_post_wake and (plan §3) prefill, decode, cache
  <log>.rankstate/<G>.tp<t>pp<p>.json        RankState (schema 1/2), kv{holds_kv,kv_tokens,share}, seats

Rates (C7 and the headline tok/s of C1/C3) come from the deltas of the monotone rankstats
counters between two samples of the same file (the reader keeps the previous sample); a
counter that went backwards is a restart, never a negative rate.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

LOG_LABEL = "aus Log (Übergang)"
RANKSTATS_SCHEMA = "weg2.rankstats/1"
RANKSTATE_SCHEMAS = (1, 2)

#: the 25 rows of the inventory that were "Übergang" (order = the page's order)
KEYS = ("A1", "A4", "A12", "A13", "A14", "A15",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7",
        "D1", "D2", "D3", "D4",
        "E1", "E2", "E3",
        "F3", "F4", "F6")


def field(key: str, ipc_value, ipc_src: Optional[str], log_value, from_log: Optional[List[str]] = None) -> dict:
    """The switch: IPC when present, else the log value with its label.  ``from_log`` names the
    sub-fields the IPC record carries as null (RANKSTATS-S3-SCHEMA: null = the source is missing
    on this rank, never 0) -- those stay on the log with the label."""
    if ipc_value is not None:
        f = {"key": key, "src": "ipc", "ipc_src": ipc_src, "value": ipc_value, "label": None}
        if from_log:
            f["from_log"] = sorted(set(from_log))
            f["from_log_label"] = LOG_LABEL
        return f
    return {"key": key, "src": "log", "ipc_src": None, "value": log_value, "label": LOG_LABEL}


def _null_leaves(rows: dict) -> List[str]:
    """Dotted leaves (one nesting level) that are null in every rank row of a block --
    RANKSTATS-S3-SCHEMA: null = the source is missing on this rank, never 0.  Generic over the
    block, no per-name exception list: a producer that fills a leaf switches it by itself."""
    keys: Dict[str, list] = {}
    for r in rows.values():
        for k, v in (r or {}).items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    keys.setdefault(k + "." + kk, []).append(vv)
            else:
                keys.setdefault(k, []).append(v)
    return sorted(k for k, vs in keys.items() if all(v is None for v in vs))


def _non_null(d) -> bool:
    return isinstance(d, dict) and any(v is not None for v in d.values())


# ----------------------------------------------------------------------------- rank files

def rankstate_dirs(files: dict) -> List[str]:
    """The rank state dirs of a log-discovered boot: next to each group log, ``<log>.rankstate``."""
    out = []
    for g, f in (files or {}).items():
        p = (f or {}).get("path") if isinstance(f, dict) else f
        if g != "front" and p and os.path.isdir(p + ".rankstate"):
            out.append(p + ".rankstate")
    return out


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def read_rank_files(dirs: List[str]) -> dict:
    """{"rankstats": {"D.tp0pp0": rec}, "rankstate": {"D.tp0pp0": rec}} -- schema-checked."""
    stats, state = {}, {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.rankstats"))):
            rec = _load(p)
            if rec and rec.get("schema") == RANKSTATS_SCHEMA:
                stats[os.path.basename(p)[: -len(".rankstats")]] = rec
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            rec = _load(p)
            if rec and rec.get("schema") in RANKSTATE_SCHEMAS:
                state[os.path.basename(p)[: -len(".json")]] = rec
    return {"rankstats": stats, "rankstate": state}


def _grp(key: str) -> str:
    return key.split(".", 1)[0]


def _first_rank(stats: dict, group: str) -> Optional[dict]:
    """TP0/PP0 of a group -- the rank the headline numbers are read from."""
    for k in (f"{group}.tp0pp0",):
        if k in stats:
            return stats[k]
    ks = sorted(k for k in stats if _grp(k) == group)
    return stats[ks[0]] if ks else None


class Rates:
    """Deltas of monotone rankstats counters between two samples of one rank file."""

    def __init__(self):
        self.prev: Dict[str, dict] = {}

    def update(self, boot: str, stats: dict) -> dict:
        out = {}
        for k, rec in stats.items():
            pk = boot + "/" + k
            p = self.prev.get(pk)
            ts = float(rec.get("ts") or 0)
            pre = rec.get("prefill") or {}
            # RANKSTATS-S3-SCHEMA-0929: the compute-honest rate is Δnew_tokens / Δgpu_ms
            cur = {"ts": ts, "prefill": _num((rec.get("tokens") or {}).get("prefill_total")),
                   "decode": _num((rec.get("tokens") or {}).get("decode_total")),
                   "pnew": _num(pre.get("new_tokens")),
                   "pcomp": _num(pre.get("gpu_ms") if pre.get("gpu_ms") is not None else pre.get("compute_ms"))}
            if p is None or ts <= p["ts"]:
                if p is None or ts != p["ts"]:
                    self.prev[pk] = cur
                if p is not None and "rates" in p:
                    out[k] = p["rates"]
                continue
            dt = ts - p["ts"]
            r = {"dt_s": round(dt, 3)}
            for name, a, b in (("prefill_tps", cur["prefill"], p["prefill"]),
                               ("decode_tps", cur["decode"], p["decode"])):
                if a is not None and b is not None and a >= b:
                    r[name] = round((a - b) / dt, 1)
            if None not in (cur["pnew"], p["pnew"], cur["pcomp"], p["pcomp"]) and cur["pcomp"] > p["pcomp"]:
                r["prefill_tps_gpu"] = round((cur["pnew"] - p["pnew"]) / ((cur["pcomp"] - p["pcomp"]) / 1000.0), 1)
            cur["rates"] = r
            self.prev[pk] = cur
            out[k] = r
        return out


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- per field

def _ev(ipc: Optional[dict], typ: str) -> List[dict]:
    return [e for e in ((ipc or {}).get("ipc_events") or []) if e.get("type") == typ]


def _data(e: dict) -> dict:
    return e.get("data") or {}


def _front(ipc: Optional[dict]) -> dict:
    return (ipc or {}).get("front") or {}


def _flip_first_work(ipc, direction: str):
    rows = [_data(e) for e in _ev(ipc, "flip_first_work") if _data(e).get("dir") == direction]
    if not rows:
        return None
    ms = [float(r["flip_time_ms"]) for r in rows if r.get("flip_time_ms") is not None]
    ms_sorted = sorted(ms)
    return {"n": len(ms), "last_ms": ms[-1] if ms else None,
            "median_ms": ms_sorted[len(ms_sorted) // 2] if ms_sorted else None,
            "rows": rows[-24:]}


def _flip_done(ipc):
    rows = [_data(e) for e in _ev(ipc, "flip_done")]
    if not rows:
        return None
    keep = ("epoch", "sleep", "wake", "drain_quiesce_ms", "sleep_ms", "wake_ms", "flip_ms", "legs_wall_ms",
            "overlap_ms", "critical_path", "flip_begin_ts", "t")
    return [{k: r.get(k) for k in keep if k in r} for r in rows]


def resolve(ipc: Optional[dict], rank: Optional[dict], logv: dict, rates: Optional[dict] = None) -> dict:
    """All fields of one boot.  ``ipc`` = ipcstate.boot_view (+ ``ipc_events``), ``rank`` =
    read_rank_files(), ``logv`` = the log view of the same boot (live.Boot.view), ``rates`` =
    Rates.update() of this poll.  Pure over its inputs."""
    rank = rank or {}
    stats = rank.get("rankstats") or {}
    rstate = rank.get("rankstate") or {}
    rates = rates or {}
    front = _front(ipc)
    f: Dict[str, dict] = {}

    # A1 boot list: the state dir of this tag
    a1 = ({"boot_id": ipc.get("boot_id"), "lifecycle": ipc.get("lifecycle")} if ipc else None)
    f["A1"] = field("A1", a1, "state.json", {"stem": logv.get("stem"), "live": logv.get("live")})

    # A4 form: groups.<G>.form written by the launcher
    forms = {g: v for g, v in ((ipc or {}).get("forms") or {}).items() if v}
    f["A4"] = field("A4", forms or None, "state.json groups.form", (logv.get("meta") or {}).get("form"))

    # A12 group health: front.groups (live) + group_health events (changes)
    gh = front.get("groups")
    gh_ev = {}
    for e in _ev(ipc, "group_health"):
        gh_ev[_data(e).get("group")] = _data(e)
    a12 = None
    if isinstance(gh, dict) and gh:
        a12 = {g: dict(v or {}, verdict=(gh_ev.get(g) or {}).get("verdict")) for g, v in gh.items()}
    elif gh_ev:
        a12 = gh_ev
    f["A12"] = field("A12", a12, "state.json front.groups + group_health", logv.get("health"))

    # A13 error lines: rankstats.errors (per rank) + front.errors
    errs, n_err, have = [], 0, False
    for k, rec in sorted(stats.items()):
        e = rec.get("errors")
        if isinstance(e, dict):
            have = True
            n_err += int(e.get("n") or 0)
            errs += [dict(x, group=_grp(k), rank=k) for x in (e.get("last") or [])]
    if isinstance(front.get("errors"), dict):
        have = True
        n_err += int(front["errors"].get("n") or 0)
        errs += [dict(x, group="front") for x in (front["errors"].get("last") or [])]
    a13 = {"n": n_err, "last": sorted(errs, key=lambda x: x.get("t") or 0)[-8:]} if have else None
    f["A13"] = field("A13", a13, "rankstats.errors + front.errors",
                     {"n": logv.get("error_count"), "last": logv.get("errors")})

    # A14 named stops: the front's rank_stop events (writer front, envelope group/rank/code,
    # data {t, reason, code, exc, ticket, text, pid, group, rank}); before the front has
    # published them, the ranks' own rankstats.stops {n, last[8]} (written on the death path)
    stops_ev = [dict(_data(e), ts=e.get("ts"), group=e.get("group") or _data(e).get("group"),
                     rank=e.get("rank") or _data(e).get("rank"), code=e.get("code") or _data(e).get("code"))
                for e in _ev(ipc, "rank_stop")]
    a14_src = "events rank_stop"
    if not stops_ev:
        for k, rec in sorted(stats.items()):
            for x in ((rec.get("stops") or {}).get("last") or []):
                stops_ev.append(dict(x, group=_grp(k), rank=k.split(".", 1)[-1]))
        stops_ev.sort(key=lambda x: x.get("t") or 0)
        a14_src = "rankstats.stops"
    f["A14"] = field("A14", stops_ev[-12:] or None, a14_src,
                     {"n": logv.get("stop_count"), "last": logv.get("stops")})

    # A15 alarm dead/hung: the per-rank heartbeat = rankstats ts + forward counter
    a15 = {k: {"ts": rec.get("ts"), "forward_ct": (rec.get("work") or {}).get("forward_ct"),
               "seq": rec.get("seq")} for k, rec in stats.items()} or None
    f["A15"] = field("A15", a15, "rankstats.ts + work.forward_ct", logv.get("last_activity"))

    # B1/B2 flip time from the front's one clock
    lft = logv.get("flip_times") or {}
    f["B1"] = field("B1", _flip_first_work(ipc, "P>D"), "events flip_first_work P>D", lft.get("p2d") or lft)
    f["B2"] = field("B2", _flip_first_work(ipc, "D>P"), "events flip_first_work D>P", lft.get("d2p") or lft)

    # B3 layer exchange, B4 count/last, B5 bars
    done = _flip_done(ipc)
    f["B3"] = field("B3", done[-1] if done else None, "events flip_done", (logv.get("flips") or [None])[-1])
    f["B4"] = field("B4", {"n": len(done), "last": done[-1]} if done else None, "events flip_done",
                    {"n": logv.get("flip_count"), "last": (logv.get("flips") or [None])[-1]})
    f["B5"] = field("B5", done[-24:] if done else None, "events flip_done", logv.get("flips"))

    # B6 flip open: a flip_begin whose epoch has no flip_done yet
    begins = [_data(e) for e in _ev(ipc, "flip_begin")]
    b6 = None
    if begins:
        last_b = begins[-1]
        done_epochs = {r.get("epoch") for r in (done or [])}
        ep = last_b.get("epoch_before")
        is_open = not any(x in done_epochs for x in (ep, (ep + 1) if isinstance(ep, int) else None))
        b6 = {"open": is_open, "since_ts": last_b.get("flip_begin_ts") if is_open else None,
              "sleep": last_b.get("sleep"), "wake": last_b.get("wake")}
    f["B6"] = field("B6", b6, "events flip_begin/flip_done", logv.get("flip_open"))

    # B7 D post-wake pass: rankstats.last_post_wake of D TP0 (or the post_wake_pass event)
    pw = [_data(e) for e in _ev(ipc, "post_wake_pass")]
    d0 = _first_rank(stats, "D")
    b7 = pw[-1] if pw else ((d0 or {}).get("last_post_wake") or None)
    f["B7"] = field("B7", b7, "rankstats.last_post_wake / events post_wake_pass", None)

    # C1 prefill per rank (plan §3 rankstats.prefill), rate from deltas
    c1 = {k: dict(rec["prefill"], **{kk: v for kk, v in (rates.get(k) or {}).items() if kk.startswith("prefill")})
          for k, rec in stats.items() if _non_null(rec.get("prefill"))} or None
    f["C1"] = field("C1", c1, "rankstats.prefill", logv.get("prefill"))

    # C2 scheduler queue: rankstats.sched (waiting/running; §3 queue_req, running_req, pending_tokens,
    # full_token_usage); a leaf null on every rank stays on the log with the label
    c2 = {k: rec["sched"] for k, rec in stats.items() if isinstance(rec.get("sched"), dict)} or None
    f["C2"] = field("C2", c2, "rankstats.sched", logv.get("queue_log"),
                    from_log=_null_leaves(c2) if c2 else None)

    # C3 decode: rankstats.decode (§3; running/accept/cuda_graph only on the stats rank TP0,
    # null elsewhere) or tokens.decode_total + spec (2188e1bd98)
    c3 = {}
    for k, rec in stats.items():
        dec = rec.get("decode") if isinstance(rec.get("decode"), dict) else {}
        spec = rec.get("spec") or {}
        row = {kk: v for kk, v in dec.items() if v is not None}
        fct, acc = spec.get("forward_ct_total"), spec.get("accept_tokens_total")
        if fct:
            row.setdefault("accept_len_mean", round(float(acc or 0) / float(fct), 3))
        if "decode_tps" in (rates.get(k) or {}):
            row["gen_tps"] = rates[k]["decode_tps"]
        if any(v not in (None, 0) for kk, v in row.items() if kk != "tokens") or row.get("tokens"):
            c3[k] = row
    f["C3"] = field("C3", c3 or None, "rankstats.decode/spec + Deltas", logv.get("decode"))

    # C4 decode round time per bs (§3)
    c4 = {k: rec["decode"]["gpu_ms_by_bs"] for k, rec in stats.items()
          if isinstance(rec.get("decode"), dict) and rec["decode"].get("gpu_ms_by_bs")} or None
    f["C4"] = field("C4", c4, "rankstats.decode.gpu_ms_by_bs", {g: (d or {}).get("round_ms_by_bs")
                                                                  for g, d in (logv.get("decode") or {}).items()})

    # C5 KV tokens / seats: the RankState record's kv {holds_kv, kv_tokens, share} + seats
    # (rank_state.note_capacity, both optional -- absent = the field stays on the log); the
    # rankstats cap {kv_tokens, seats} carries the same numbers between two RankState rewrites
    c5 = {}
    for k, rec in rstate.items():
        kv = rec.get("kv") if isinstance(rec.get("kv"), dict) else {}
        if kv.get("kv_tokens") is not None or rec.get("seats") is not None:
            c5[k] = {"holds_kv": kv.get("holds_kv"), "kv_tokens": kv.get("kv_tokens"), "share": kv.get("share"),
                     "seats": rec.get("seats"), "src": "RankState"}
    for k, rec in stats.items():
        cap = rec.get("cap") if isinstance(rec.get("cap"), dict) else {}
        if k not in c5 and (cap.get("kv_tokens") is not None or cap.get("seats") is not None):
            c5[k] = {"holds_kv": None, "kv_tokens": cap.get("kv_tokens"), "share": None,
                     "seats": cap.get("seats"), "src": "rankstats.cap"}
    f["C5"] = field("C5", c5 or None, "RankState kv/seats + rankstats.cap", None)

    # C6 D seats: front.d_seats, or the front's d_phase_n / d_parked_n
    c6 = front.get("d_seats") or ({"n": front.get("d_phase_n"), "parked_n": front.get("d_parked_n")}
                                  if front.get("d_phase_n") is not None else None)
    f["C6"] = field("C6", c6, "state.json front.d_seats", None)

    # C7 rates from deltas
    f["C7"] = field("C7", rates or None, "rankstats Deltas", logv.get("series"))

    # D1/D2 work per forward: rankstats.work.spans when a producer keeps them, else the §3
    # prefill.last {t, gpu_ms} (the last timed forward: end t, start t - gpu_ms) + chunks as the
    # forward sequence.  P ranks -> D1 (prefill), D ranks -> D2 (the extend of a D phase).
    work = {}
    for k, rec in stats.items():
        sp = (rec.get("work") or {}).get("spans")
        last = (rec.get("prefill") or {}).get("last") or {}
        if sp:
            work[k] = sp
        elif last.get("t") is not None and last.get("gpu_ms") is not None:
            t1 = float(last["t"])
            work[k] = [{"t0": round(t1 - float(last["gpu_ms"]) / 1000.0, 3), "t1": t1,
                        "kind": "extend" if _grp(k) == "D" else "prefill", "new": last.get("new"),
                        "seq": (rec.get("prefill") or {}).get("chunks")}]
    d1 = {k: [s for s in v if s.get("kind") != "extend"] for k, v in work.items() if _grp(k) != "D"}
    d2 = {k: [s for s in v if s.get("kind") == "extend"] for k, v in work.items() if _grp(k) == "D"}
    d1 = {k: v for k, v in d1.items() if v} or None
    d2 = {k: v for k, v in d2.items() if v} or None
    f["D1"] = field("D1", d1, "rankstats.work.spans / prefill.last", (logv.get("timeline") or {}).get("P"))
    f["D2"] = field("D2", d2, "rankstats.work.spans / prefill.last (D)", (logv.get("timeline") or {}).get("D"))

    # D3 flip grey / tail class: B1 + B3 + B7 together
    d3 = ({"first_work": f["B1"]["value"], "done": f["B3"]["value"], "post_wake": f["B7"]["value"]}
          if f["B1"]["src"] == "ipc" and f["B3"]["src"] == "ipc" else None)
    f["D3"] = field("D3", d3, "events flip_first_work + flip_done + post_wake", logv.get("flip_times"))

    # D4 boot loading before first work: lifecycle + group_ready + serving_since_ts
    ready = {_data(e).get("group") or e.get("group"): _data(e).get("after_s") for e in _ev(ipc, "group_ready")}
    d4 = ({"lifecycle": ipc.get("lifecycle"), "serving_since_ts": ipc.get("serving_since_ts"),
           "group_ready_after_s": ready} if ipc and (ipc.get("serving_since_ts") or ready) else None)
    f["D4"] = field("D4", d4, "state.json lifecycle + group_ready", {"first_work_t": logv.get("first_work_t")})

    # E1 cached tokens (§3 prefill.cached_tokens)
    e1 = {k: rec["prefill"]["cached_tokens"] for k, rec in stats.items()
          if isinstance(rec.get("prefill"), dict) and rec["prefill"].get("cached_tokens") is not None} or None
    f["E1"] = field("E1", e1, "rankstats.prefill.cached_tokens", logv.get("cache"))

    # E2 loadback / mamba resume / store incomplete / prefetch (§3 rankstats.cache)
    # (§3 + Nachzug ff643c9010: prefetch {attempted, issued, landed, deferred (#1068 DEFERRED),
    # defer_refused, expired, refused (#915 REFUSED), timeout (#1157 REAPED)}; a leaf null on every
    # rank -- e.g. timeout on a rank without tree_cache -- stays on the log with the label)
    e2 = {k: rec["cache"] for k, rec in stats.items() if isinstance(rec.get("cache"), dict)} or None
    f["E2"] = field("E2", e2, "rankstats.cache", logv.get("cache"), from_log=_null_leaves(e2) if e2 else None)

    # E3 served tokens per leg
    f["E3"] = field("E3", front.get("served_tokens") or None, "state.json front.served_tokens",
                    (logv.get("totals") or {}).get("served"))

    # F3/F4/F6: the feature cards reuse B1, A4 and C1-C6
    f["F3"] = field("F3", f["B1"]["value"] if f["B1"]["src"] == "ipc" else None, f["B1"]["ipc_src"],
                    f["B1"]["value"])
    f["F4"] = field("F4", f["A4"]["value"] if f["A4"]["src"] == "ipc" else None, f["A4"]["ipc_src"],
                    f["A4"]["value"])
    c = {k: f[k]["value"] for k in ("C1", "C2", "C3", "C4", "C5", "C6") if f[k]["src"] == "ipc"}
    f["F6"] = field("F6", c or None, "C1-C6 aus IPC", None)
    return f


def for_page(fields: dict) -> dict:
    """What goes to the page: the IPC value, or only the label -- the log value is already
    in the boot's view under its old key and is not sent twice."""
    return {k: ({kk: vv for kk, vv in fv.items()} if fv["src"] == "ipc"
                else {kk: vv for kk, vv in fv.items() if kk != "value"}) for k, fv in fields.items()}


def summary(fields: dict) -> dict:
    """How many of the fields read the IPC now (the page shows it in the footer)."""
    n_ipc = sum(1 for v in fields.values() if v.get("src") == "ipc")
    return {"ipc": n_ipc, "log": len(fields) - n_ipc, "n": len(fields),
            "log_keys": [k for k, v in fields.items() if v.get("src") != "ipc"]}
