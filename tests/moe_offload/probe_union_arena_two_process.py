"""Two-process metal probe for the union arena (evidence, not a gate).

Run by hand on a free card, never from the hermetic suite -- it needs a GPU
and two processes:

    D=/tmp/uprobe; rm -rf $D; mkdir -p $D
    python probe_union_arena_two_process.py owner 0 $D &   # then, after ~20 s
    python probe_union_arena_two_process.py peer  0 $D

MEASURED 2026-09-21 on the 5090 (CUDA device 0), 2 GiB arena:
  OWNER-SEES-PEER-WRITE=YES   PEER-SEES-OWNER-PATTERN=YES   OWNER-PATTERN-INTACT=YES
  card free with the peer ATTACHED          28.23 GiB of 31.34
  card free with the peer holding its OWN   26.12 GiB   (control, probe_copy_control)
i.e. the second process's view of the same weights costs ZERO card bytes.
"""

import os, sys, time
import torch

from sglang.srt.weg2.union_arena_vmm import (
    UnionRendezvousServer, UnionVmmArena, fetch_union, socket_path,
)

MODE = sys.argv[1]
DEV = int(sys.argv[2])
DIR = sys.argv[3]
NBYTES = 2 * 1024**3
CARD = "PROBE-CARD-0001"
OFF_OWNER = 1 << 20
OFF_PEER = 2 << 20
N = 4096
path = socket_path(DIR, CARD)
done = os.path.join(DIR, "peer.done")

owner_pat = torch.arange(N, dtype=torch.uint8).repeat(4)[:N]
peer_pat = torch.full((N,), 0x5A, dtype=torch.uint8)

if MODE == "owner":
    if os.path.exists(done):
        os.unlink(done)
    a = UnionVmmArena.create(DEV, NBYTES)
    t = a.tensor
    t.zero_()
    t[OFF_OWNER:OFF_OWNER + N] = owner_pat.cuda(DEV)
    torch.cuda.synchronize(DEV)
    fds = a.export_fds()
    srv = UnionRendezvousServer(path, "MANIFEST-PROBE", fds)
    print(f"OWNER ready bytes={a.total_bytes} base=0x{a.base:x} fds={len(fds)} sock={path}", flush=True)
    t0 = time.time()
    while not os.path.exists(done) and time.time() - t0 < 180:
        time.sleep(0.2)
    got = t[OFF_PEER:OFF_PEER + N].cpu()
    same = bool((got == peer_pat).all())
    print(f"OWNER-SEES-PEER-WRITE={'YES' if same else 'NO'} "
          f"(first bytes {got[:8].tolist()})", flush=True)
    # the owner's own pattern must be untouched
    mine = t[OFF_OWNER:OFF_OWNER + N].cpu()
    print(f"OWNER-PATTERN-INTACT={'YES' if bool((mine == owner_pat).all()) else 'NO'}", flush=True)
    srv.stop()
    print("OWNER done", flush=True)
else:
    text, fds = fetch_union(path, timeout_s=120)
    print(f"PEER got manifest={text!r} fds={len(fds)}", flush=True)
    a = UnionVmmArena.attach(DEV, NBYTES, fds)
    t = a.tensor
    got = t[OFF_OWNER:OFF_OWNER + N].cpu()
    same = bool((got == owner_pat).all())
    print(f"PEER-SEES-OWNER-PATTERN={'YES' if same else 'NO'} "
          f"(first bytes {got[:8].tolist()})", flush=True)
    t[OFF_PEER:OFF_PEER + N] = peer_pat.cuda(DEV)
    torch.cuda.synchronize(DEV)
    free, total = torch.cuda.mem_get_info(DEV)
    print(f"PEER card free={free/2**30:.2f} GiB of {total/2**30:.2f}", flush=True)
    open(done, "w").write("1")
    print("PEER done", flush=True)
    time.sleep(3)
