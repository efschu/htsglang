#!/usr/bin/env python3
"""#655: one row per load ATTEMPT, joining the backend log's own memory
milestones with the host memory time series sampled through the load."""
import glob
import os
import re
import sys

SERIES = "/root/651-p2/logs/kv655_series.tsv"


def load_series():
    rows = []
    if not os.path.exists(SERIES):
        return rows
    for ln in open(SERIES):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        try:
            rows.append((p[0], [int(x) for x in p[1:7]]))
        except ValueError:
            continue
    return rows


def at(series, hhmmss):
    """Series sample nearest the given UTC clock time."""
    def secs(t):
        h, m, s = (int(x) for x in t.split(":"))
        return h * 3600 + m * 60 + s
    if not series:
        return None
    tgt = secs(hhmmss)
    best = min(series, key=lambda r: abs(secs(r[0]) - tgt))
    return best if abs(secs(best[0]) - tgt) <= 6 else None


def main():
    series = load_series()
    logs = sorted(glob.glob("/root/651-p2/logs/backend_*.log"), key=os.path.getmtime)[-int(sys.argv[1] if len(sys.argv) > 1 else 20):]
    hdr = ["log", "dtype", "mamba_slots", "pre_GB", "postW_GB", "mamba_GB",
           "scratch", "KVtok", "KV_GB", "MemFree_MB", "Cached_MB", "GTT_GB", "result"]
    print("\t".join(hdr))
    for lg in logs:
        txt = open(lg, "rb").read().decode("utf-8", "replace")
        g = lambda p: (re.search(p, txt) or [None, None])[1] if re.search(p, txt) else ""
        pre = g(r"Load weight begin\. avail mem=([\d.]+) GB")
        postw = g(r"Load weight end\..*?avail mem=([\d.]+) GB")
        # local wall clock of the sizing step, converted to the series' UTC
        m = re.search(r"\[[\d-]+ (\d\d:\d\d:\d\d)\] (?:Mamba Cache is allocated|GGUF dequant scratch)", txt)
        ssm = g(r"ssm_state size: ([\d.]+)GB")
        conv = g(r"conv_state size: ([\d.]+)GB")
        slots = g(r"max_mamba_cache_size: (\d+)")
        scr = g(r"reserving [\d.]+ GiB.*?\(([-\d.]+ -> [-\d.]+) GiB\)")
        kvt = g(r"KV Cache is allocated\. dtype: \S+ #tokens: (\d+)")
        if not kvt:
            kvt = g(r"KV Cache is allocated\..*?#tokens: (\d+)")
        kvd = g(r"KV Cache is allocated\. dtype: (\S+),")
        ksz = g(r"K size: ([\d.]+) GB")
        vsz = g(r"V size: ([\d.]+) GB")
        cli = "fp8" if "kv_cache_dtype='fp8" in txt else "auto"
        res = "OK" if kvt else ("FAIL:no-KV-mem" if "leave no GPU memory" in txt else "FAIL:other")
        free = cach = gtt = ""
        if m:
            h, mi, s = (int(x) for x in m.group(1).split(":"))
            utc = "%02d:%02d:%02d" % ((h - 2) % 24, mi, s)  # host is UTC+2
            row = at(series, utc)
            if row:
                free = "%.0f" % (row[1][0] / 1024)
                cach = "%.0f" % (row[1][2] / 1024)
                gtt = "%.2f" % (row[1][4] / 1073741824)
        kvgb = ""
        if ksz and vsz:
            kvgb = "%.3f" % (float(ksz) + float(vsz))
        mam = ""
        if ssm:
            mam = "%.2f" % (float(ssm) + float(conv or 0))
        print("\t".join([os.path.basename(lg)[8:14], (kvd or cli), slots or "auto",
                         pre or "", postw or "", mam, scr or "", kvt or "", kvgb,
                         free, cach, gtt, res]))


main()
