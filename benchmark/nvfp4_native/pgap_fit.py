import re, sys, statistics as st
from collections import defaultdict
pb = re.compile(r'\[(\S+ \S+) PP(\d)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),.*#pending-token: (\d+)')
pg = re.compile(r'\[(\S+ \S+) PP(\d)\] #PGAP pp_rank=(\d) fwd=(\d+) tokens=(\d+) gpu_gap_ms=(\S+) gpu_fwd_ms=([\d.]+) host\[plan=(\d+) proxy_recv=(\d+) launch=(\d+) fi_plan=(\d+)')
def parse(path):
    pend = defaultdict(list)  # rank -> list of (new, cached, pending)
    rows = defaultdict(list)
    state = {}
    for line in open(path, errors='replace'):
        m = pb.search(line)
        if m:
            r = int(m.group(2)); new, cached, pending = int(m.group(4)), int(m.group(5)), int(m.group(6))
            pend[r].append((new, cached, pending)); continue
        m = pg.search(line)
        if m:
            r = int(m.group(3)); fwd = int(m.group(4)); tok = int(m.group(5)); ms = float(m.group(7))
            rows[r].append((fwd, tok, ms, int(m.group(8)), int(m.group(9)), int(m.group(10)), int(m.group(11))))
    out = {}
    for r in rows:
        # pair i-th PGAP with i-th Prefill batch line of that rank
        P = pend[r]; R = rows[r]
        n = min(len(P), len(R)); pos_list = []
        total = None; seen = 0
        for i in range(n):
            new, cached, pending = P[i]
            if total is None or pending + new > (total - seen):  # new request
                total = pending + new + cached; seen = cached
            start = seen; seen += new
            pos_list.append((start, new, R[i][2], R[i]))
        out[r] = pos_list
    return out
def fit(pts):
    xs = [p[0] for p in pts]; ys = [p[2] for p in pts]
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x-mx)**2 for x in xs) or 1
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / sxx
    return my - b*mx, b
for path in sys.argv[1:]:
    d = parse(path)
    print("==", path.split('/')[-1])
    for r in sorted(d):
        pts = [p for p in d[r] if p[1] == 512]
        if len(pts) < 10: continue
        ys = sorted(p[2] for p in pts)
        a, b = fit([p for p in pts if p[0] <= 65536])
        near0 = [p[2] for p in pts if p[0] < 1024]
        bands = {}
        for lo, hi in ((0,2048),(2048,8192),(8192,16384),(16384,32768),(32768,65536),(65536,1<<30)):
            v = [p[2] for p in pts if lo <= p[0] < hi]
            if v: bands[f"{lo//1024}-{hi//1024 if hi<1<<30 else 'inf'}k"] = (len(v), round(st.median(v),1))
        print(f" PP{r}: n={len(pts)} median={st.median(ys):.1f} ms  fit gpu_fwd = {a:.1f} + {b*1000:.3f} ms/1k-ctx  bands(n,median)={bands}")
