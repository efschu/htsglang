"""Two-process metal probe of the BIND seam (evidence, not a gate).

    D=/tmp/bprobe; rm -rf $D; mkdir -p $D
    python probe_union_bind_two_process.py owner 0 $D &   # then, after ~50 s
    python probe_union_bind_two_process.py peer  0 $D

MEASURED 2026-09-21 on the 5090, 1.5 GiB parameter set:
  PEER shared=1.50 GiB kept=0.004 GiB
  PEER-VALUES-INTACT=YES (541.6194 -> 541.6194)
  PEER-REBOUND=YES (0x7ad80c000000 -> 0x502000000)   <- the owner's arena base
  PEER-CARD-FREED=+1.50 GiB (27.11 -> 28.61)
The peer's parameters point into the owner's pages, hold the right values,
and the card gets the whole shared amount back.
"""

import os, sys, time
import torch, torch.nn as nn

from sglang.srt.weg2.union_arena_bind import bind_image, own_image

MODE, DEV, DIR = sys.argv[1], int(sys.argv[2]), sys.argv[3]
CARD = str(torch.cuda.get_device_properties(DEV).uuid)
done = os.path.join(DIR, "bind.done")
N = 24  # 24 x 64 MiB = 1.5 GiB


class Toy(nn.Module):
    def __init__(self, private: bool):
        super().__init__()
        g = torch.Generator().manual_seed(4711)
        for i in range(N):
            t = torch.empty(16 * 1024 * 1024, dtype=torch.float32)
            t.normal_(generator=g)
            self.register_parameter(f"w{i:02d}", nn.Parameter(t.cuda(DEV), False))
        if private:
            self.register_parameter(
                "private", nn.Parameter(torch.zeros(1 << 20, device=f"cuda:{DEV}"), False)
            )


def free_gib():
    return torch.cuda.mem_get_info(DEV)[0] / 2**30


if MODE == "owner":
    if os.path.exists(done):
        os.unlink(done)
    m = Toy(private=False)
    own_image(m, union_dir=DIR, tag="probe", card=CARD, phase="P", device=DEV)
    ptr = m.w00.data_ptr()
    print(f"OWNER published, w00 at 0x{ptr:x}, free={free_gib():.2f} GiB", flush=True)
    t0 = time.time()
    while not os.path.exists(done) and time.time() - t0 < 300:
        time.sleep(0.2)
    print(f"OWNER w00 sum={float(m.w00.sum()):.4f} free_end={free_gib():.2f}", flush=True)
else:
    m = Toy(private=True)
    want = float(m.w00.sum())
    before_ptr = m.w00.data_ptr()
    before = free_gib()
    shared, kept = bind_image(m, union_dir=DIR, card=CARD, phase="D", device=DEV)
    after = free_gib()
    got = float(m.w00.sum())
    print(f"PEER shared={shared/2**30:.2f} GiB kept={kept/2**30:.3f} GiB", flush=True)
    print(f"PEER-VALUES-INTACT={'YES' if abs(got-want) < 1e-3 else 'NO'} "
          f"({want:.4f} -> {got:.4f})", flush=True)
    print(f"PEER-REBOUND={'YES' if m.w00.data_ptr() != before_ptr else 'NO'} "
          f"(0x{before_ptr:x} -> 0x{m.w00.data_ptr():x})", flush=True)
    print(f"PEER-CARD-FREED={after-before:+.2f} GiB (before {before:.2f} after {after:.2f})",
          flush=True)
    open(done, "w").write("1")
    time.sleep(3)
