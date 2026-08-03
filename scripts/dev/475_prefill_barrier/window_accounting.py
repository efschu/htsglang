"""#475: prefill-window accounting per probe point, cluster-based.

CollectiveClock semantics (utils/collective_clock.py): per rank the armed
prefill forward reports T = compute + wait, wait = device time INSIDE
collectives. Ranks run lockstep, so T is common. Therefore
    max_r compute_r = T - min_r wait_r   (the critical rank's own work)
    min_r wait_r                          (the part no weight re-split removes)
Batches are grouped into probe CLUSTERS by a >20 s idle gap and matched to the
punkte.jsonl records in order.
"""
import json
import re
import datetime
from pathlib import Path

LINE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^\]]*TP(\d+)\] Prefill rank batch, "
    r"#new-token: (\d+), #cached-token: (\d+), #chunks: (\d+), "
    r"gpu-ms: ([\d.]+) \(compute ([\d.]+), wait ([\d.]+)\)")
MIN_NEW = 1500   # full 2048-chunk batches only; tails are not comparable

def parse(p):
    rows = []
    for i, ln in enumerate(open(p, errors="replace")):
        m = LINE.search(ln)
        if m and int(m.group(3)) >= MIN_NEW:
            rows.append(dict(lineno=i + 1,
                t=datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                tp=int(m.group(2)), new=int(m.group(3)),
                T=float(m.group(6)), c=float(m.group(7)), w=float(m.group(8))))
    return rows

def group(rows):
    out, pend = [], []
    for r in rows:
        for g in pend:
            if (r["new"] == g[0]["new"] and abs(r["T"] - g[0]["T"]) < 4.0
                    and r["tp"] not in [x["tp"] for x in g]
                    and abs((r["t"] - g[0]["t"]).total_seconds()) < 5):
                g.append(r)
                break
        else:
            pend.append([r])
        keep = []
        for g in pend:
            if len(g) == 3:
                out.append(dict(t=min(x["t"] for x in g), new=g[0]["new"],
                                lineno=min(x["lineno"] for x in g),
                                T=max(x["T"] for x in g),
                                c=[x["c"] for x in sorted(g, key=lambda y: y["tp"])],
                                w=[x["w"] for x in sorted(g, key=lambda y: y["tp"])]))
            else:
                keep.append(g)
        pend = keep
    out.sort(key=lambda x: x["t"])
    return out

def clusters(bs, gap=10):
    cs, cur = [], []
    for b in bs:
        if cur and (b["t"] - cur[-1]["t"]).total_seconds() > gap:
            cs.append(cur)
            cur = []
        cur.append(b)
    if cur:
        cs.append(cur)
    return cs

def report(bs, tag, extra=""):
    tok = sum(b["new"] for b in bs)
    k = 1000.0 / tok
    T = sum(b["T"] for b in bs)
    mc = sum(max(b["c"]) for b in bs)
    co = sum(min(b["w"]) for b in bs)
    c = [sum(b["c"][i] for b in bs) for i in range(3)]
    print(f"  {tag:34s} n={len(bs):3d} | per-1k-tok ms: T={T*k:6.1f} "
          f"crit-compute={mc*k:6.1f} ({mc/T*100:4.1f}%) collective={co*k:6.1f} "
          f"({co/T*100:4.1f}%) | c/rank={[round(x*k,1) for x in c]} "
          f"| L{bs[0]['lineno']}-{bs[-1]['lineno']} {extra}")
    return dict(T=T*k, mc=mc*k, co=co*k, n=len(bs))

if __name__ == "__main__":
    BAT = {"424": "/spinning/gpu-battery-results/2026-08-02_424_phase_record_bench",
           "433": "/spinning/gpu-battery-results/2026-08-02_433_int8_prefill",
           "435": "/spinning/gpu-battery-results/2026-08-02_435_coupling_fp8bar1"}
    for bat, d in BAT.items():
        d = Path(d)
        recs = [json.loads(x) for x in open(d / "raw/punkte.jsonl")]
        print(f"########## battery {bat}")
        for p in sorted((d / "raw").glob("server_*.log")):
            base = p.name[7:-4]
            mine = [r for r in recs if re.sub(r"_floorP\d$", "", r["arm"]) == base]
            cs = clusters(group(parse(p)))
            print(f" -- {base}: {len(cs)} clusters vs {len(mine)} probe records")
            for i, cl in enumerate(cs):
                r = mine[i] if i < len(mine) else None
                tag = f"{r['arm']} s={r['sessions']}" if r else f"cluster{i}"
                pr = f"probe={r['prefill']['prefill_tok_s']:7.1f}tok/s" if r else ""
                report(cl, tag, pr)
