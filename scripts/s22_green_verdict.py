#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631/#656 green-run verdict: every acceptance axis in one place.

Reads the corridor time series and the server log and prints the axes the
run is judged on, so the verdict is not assembled by hand from three
terminals. All per-card triples are printed in BOTH orders, because this
corpus has one convention (nvidia-smi index) and the ranks are in another,
and every misreading of that has cost a successor a wrong conclusion.
"""
import csv
import re
import subprocess
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "/spinning/evidence-631/s22/green/corridor.csv"
LOG = sys.argv[2] if len(sys.argv) > 2 else "/spinning/serving-30030.boot.log"
FLOOR = 1024
# --chunked-prefill-size in the boot under test; the dynamic arm must be
# judged against exceeding THIS, not against variety.
STATIC_CHUNK = 2048


def grep_count(pat):
    r = subprocess.run(["grep", "-c", pat, LOG], capture_output=True, text=True, timeout=120)
    return int((r.stdout or "0").strip() or 0)


def grep_all(pat, limit=200000):
    r = subprocess.run(["grep", "-oE", pat, LOG], capture_output=True, text=True, timeout=180)
    return (r.stdout or "").splitlines()[:limit]


rows = []
with open(CSV) as fh:
    rd = csv.reader(fh)
    next(rd, None)
    for row in rd:
        if len(row) >= 4:
            try:
                rows.append([float(row[0])] + [int(x) for x in row[1:4]])
            except ValueError:
                pass

print("=" * 74)
print("GREEN RUN VERDICT")
print("=" * 74)
if rows:
    span = (rows[-1][0] - rows[0][0]) / 1000.0
    mins = [min(r[i] for r in rows) for i in (1, 2, 3)]
    print(f"corridor samples   : {len(rows)} over {span/60:.1f} min ({span:.0f}s)")
    print(f"minimum, idx order : {mins[0]}, {mins[1]}, {mins[2]}  (nvidia-smi 0,1,2)")
    print(f"minimum, RANK order: rank0(5090)={mins[1]}  rank1(3080)={mins[0]}  rank2(3080)={mins[2]}")
    worst = min(mins)
    print(f"floor {FLOOR} MiB     : {'HELD' if worst >= FLOOR else 'BREACHED'} "
          f"(worst {worst}, margin {worst - FLOOR:+d})")
    print(f"surplus above floor: {sum(m - FLOOR for m in mins)} MiB total "
          f"(the OTHER half of the corridor law -- free should be NEAR 1024)")
else:
    print("no corridor samples")

flips = grep_all(r"PHASE-FLIP DONE (pp_to_tp|tp_to_pp)")
pp2tp = sum(1 for f in flips if "pp_to_tp" in f)
tp2pp = sum(1 for f in flips if "tp_to_pp" in f)
print(f"\nflips              : {len(flips)}  ({pp2tp} pp_to_tp / {tp2pp} tp_to_pp)")
print(f"FLIP ABANDONED     : {grep_count('FLIP ABANDONED')}   <- livelock signature")
print(f"tracebacks         : {grep_count('Traceback')}")
print(f"spill rung fires   : {grep_count('SPILL rung')}")

pre = grep_all(r"Prefill batch.{0,220}")
graphed = sum(1 for p in pre if "cuda graph: True" in p)
print(f"\nPURITY")
print(f"  prefill batches  : {len(pre)}, with a CUDA graph: {graphed} "
      f"({'PURE -- prefill is eager, i.e. PP only' if graphed == 0 else 'IMPURE'})")
print(f"  purity refusals  : {grep_count('cannot run in tp')} "
      f"('prefill cannot run in tp' -- the gate acting)")
dec = grep_all(r"Decode batch.{0,220}")
withacc = sum(1 for d in dec if "accept" in d)
print(f"  decode batches   : {len(dec)}, carrying accept len: {withacc}")
if not dec:
    print("    NOTE: zero decode lines does NOT mean zero decode. Decode logging is")
    print("    gated on decode_log_interval (default 40 iterations); short generations")
    print("    never reach it. Do not read decode batch size off an empty set.")

acc = [float(m) for m in grep_all(r"accept len: [0-9.]+")
       for m in re.findall(r"[0-9.]+", m)]
if acc:
    print(f"  accept len       : mean {sum(acc)/len(acc):.2f} over {len(acc)} samples")

bs = {}
for d in dec:
    m = re.search(r"#running-req: (\d+)", d)
    if m:
        bs[m.group(1)] = bs.get(m.group(1), 0) + 1
if bs:
    print(f"  decode #running-req histogram: {dict(sorted(bs.items()))}")

chunks = {}
for p in pre:
    m = re.search(r"#new-token: (\d+)", p)
    if m:
        chunks[int(m.group(1))] = chunks.get(int(m.group(1)), 0) + 1
if chunks:
    top = sorted(chunks.items(), key=lambda kv: -kv[1])[:6]
    print(f"\nprefill chunk sizes (#new-token -> count): {top}")
    # THE DISCRIMINATOR IS "LARGER THAN THE STATIC SIZE", NOT "DIFFERENT".
    # A static run already shows many distinct sizes: every request ends in
    # a remainder chunk and short prompts never fill one. Counting distinct
    # values reports "varying" for a server with the flag OFF, which is
    # exactly the built-but-inert misreading this run exists to avoid.
    # Dynamic chunking can only be doing something if a chunk EXCEEDS
    # chunked_prefill_size (it grows chunks toward max_prefill_tokens).
    static = max(chunks)
    over = {k: v for k, v in chunks.items() if k > STATIC_CHUNK}
    print(f"  distinct sizes   : {len(chunks)} (not a dynamic-chunking signal by itself)")
    print(f"  chunks > {STATIC_CHUNK}    : {sum(over.values())} {dict(sorted(over.items())[:6])}")
    print(f"  verdict          : "
          f"{'DYNAMIC observable' if over else 'STATIC -- no chunk exceeded the configured size'}")
    print(f"  largest chunk    : {static}")

print("\nHOST RAM")
try:
    with open("/sys/fs/cgroup/memory.events") as fh:
        ev = dict(l.split() for l in fh.read().splitlines())
    print(f"  oom_kill         : {ev.get('oom_kill')}  (precedent: 9 before this session)")
except OSError:
    pass
try:
    with open("/sys/fs/cgroup/memory.peak") as fh:
        print(f"  memory.peak      : {int(fh.read().strip())/2**30:.1f} GiB")
except OSError:
    pass
print("=" * 74)
