"""Builds the frozen fixture of NF boot ...1019_230333, P run epoch 181 (D->P 06:26:10.094Z, P->D 06:26:23.464Z).

Sources (all real): events.jsonl of /spinning/docker-acceptance/nf/state/<boot>/ (flip_begin/done/first_work/
user_time/request_done of that window) and the P log lines "Prefill rank batch" (06:26:17 / :21 / :23, PP0..PP2).
The 1-s rankstats ring of that minute no longer exists (the ring keeps 16 min), so the ring is REBUILT from the
log: the log carries whole seconds only, the flush instants below are therefore inferred -- PP2's from the front's
pp_last_start_ts (16.759) + its gpu-ms 668.4 = 17.43, the others from the bubble_ms chain; documented in the test.
Run:  python3 make_fixture.py <state_dir>   (state_dir only for events/state; the ring needs no ring.sqlite)
"""
import gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = 1792477570.0          # 06:26:10Z
BOOT = "nfint4h6ablxcw3spmlrhlvwfldauer-boot-20261019T230301Z-0d0f"
# (stage, flush instant, new, cached, gpu_ms, bubble_ms) per chunk, from the P log (gpu-ms/bubble exact, flush inferred)
CHUNKS = {
    0: [(T0 + 7.45, 7903, 99328, 2642.1, 25023.2), (T0 + 11.30, 2294, 78848, 2021.4, 3082.7), (T0 + 13.42, 185, 49536, 1196.4, 1740.1)],
    1: [(T0 + 7.45, 7903, 99328, 1660.1, 25701.7), (T0 + 11.30, 2294, 78848, 935.5, 3464.8), (T0 + 13.42, 185, 49536, 514.1, 1897.9)],
    2: [(T0 + 7.43, 7903, 99328, 668.4, 26395.0), (T0 + 11.28, 2294, 78848, 222.1, 3633.8), (T0 + 13.42, 185, 49536, 44.6, 2098.2)],
}


def rec(stage, ts):
    done = [c for c in CHUNKS[stage] if c[0] <= ts]
    last = done[-1] if done else None
    return {"schema": "pdflip.rankstats/1", "ts": round(ts, 3), "group": "P", "tp_rank": 0, "pp_rank": stage,
            "prefill": {"chunks": 400 + len(done), "new_tokens": 1000000 + sum(c[1] for c in done),
                        "cached_tokens": 20000000 + sum(c[2] for c in done),
                        "gpu_ms": 800000.0 + sum(c[3] for c in done), "compute_ms": 800000.0 + sum(c[3] for c in done),
                        "bubble_ms": 20000000.0 + sum(c[4] for c in done),
                        "own_ms": 800000.0 + sum(c[3] for c in done), "own_n": 400 + len(done),
                        "last": ({"t": round(last[0], 3), "new": last[1], "gpu_ms": last[3], "own_ms": last[3]} if last else
                                 {"t": T0 - 40.0, "new": 3, "gpu_ms": 100.0, "own_ms": 100.0})},
            "decode": {"rounds": 0, "gpu_ms": 0.0, "tokens": 0, "running": None},
            "sched": {"full_token_usage": 0.2}}


def build(state_dir):
    from rigdash import ipcboot  # noqa: E402
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", ".."))
    ring, t = [], T0 - 25.0 + 0.3
    while t <= T0 + 50.0:
        r = {"P.tp0pp%d" % st: ipcboot.compact(rec(st, t - 0.1)) for st in range(3)}
        r["D.tp0pp0"] = ipcboot.compact({"schema": "pdflip.rankstats/1", "ts": t - 0.1, "group": "D",
                                         "prefill": {"chunks": 900, "new_tokens": 400000, "cached_tokens": 9e7, "gpu_ms": 5e5,
                                                     "compute_ms": 5e5, "last": {"t": T0 - 30, "new": 1, "gpu_ms": 50.0}},
                                         "decode": {"rounds": 0, "gpu_ms": 0.0, "tokens": 0, "running": 0}})
        ring.append({"t": round(t, 4), "r": r, "front": {"awake": "D" if t < T0 + 1.6 else ("P" if t < T0 + 16.5 else "D")}})
        t += 1.0
    with gzip.open(os.path.join(HERE, "ring.jsonl.gz"), "wt") as fh:
        for row in ring:
            fh.write(json.dumps(row) + "\n")
    keep = ("flip_begin", "flip_done", "flip_first_work", "flip_user_time", "lifecycle", "request_done", "park")
    out = []
    for l in open(os.path.join(state_dir, "events.jsonl")):
        try:
            e = json.loads(l)
        except ValueError:
            continue
        d = e.get("data") or {}
        ts = e.get("ts") or 0
        if e.get("type") in keep and T0 - 30 < ts < T0 + 45 or (e.get("type") == "request_done" and d.get("rid") in
                                                                  ("pdflip-180-3125", "pdflip-180-3126")):
            out.append(l)
    with open(os.path.join(HERE, "events.jsonl"), "w") as fh:
        fh.writelines(out)
    st = json.load(open(os.path.join(state_dir, "state.json")))
    mini = {k: st[k] for k in ("boot_id", "schema", "kind", "line", "tag", "rev", "image", "profile", "container", "cause",
                               "serving_since_ts", "seq")}
    mini["lifecycle"] = {"prev": "flipping", "since_ts": T0 + 21.6, "state": "serving"}
    mini["front"] = {"awake": "D"}
    with open(os.path.join(HERE, "state.json"), "w") as fh:
        json.dump(mini, fh, indent=1)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", ".."))
    build(sys.argv[1])
