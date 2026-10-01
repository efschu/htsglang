"""#651: GPU keep-busy daemon — closes the idle windows in which the gfx1103
errata class corrupts kernels and hard-faults contexts (attribution: system
suspend, runtime PM and GFXOFF each falsified; the empirical invariant is
sustained-load = clean, idle/bursty = wobble). A tiny matmul every 50 ms keeps
the GFX core continuously fed at negligible cost (~1% util)."""
import time
import torch

a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
i = 0
while True:
    a = (a @ a).clamp(-1, 1)
    torch.cuda.synchronize()
    i += 1
    if i % 1200 == 0:
        print(f"keepbusy alive, {i} ticks", flush=True)
    time.sleep(0.05)
