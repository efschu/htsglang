#!/usr/bin/env python3
"""#655: expose --max-mamba-cache-size. Atomic rename so a concurrent retry
never reads a half-written script."""
import os
import sys

P = "/root/651-p2/scripts/boot_ondemand.sh"
src = open(P).read()
if "MAMBASLOTS" in src:
    print("already patched")
    sys.exit(0)

anchor = "KVDTYPE=${KVDTYPE:-auto}\n"
assert anchor in src

block = '''# Mamba state slots. Auto-sizing gives this checkpoint 15 slots (0.96 GB of
# ssm_state) out of a post-weights budget of ~2.3 GB, while the service runs
# --max-running-requests 1. Everything above the hard demand floor
# (1 active + 1 ping-pong + 1 donation + 1 pinned checkpoint = 4 slots at this
# concurrency, see mamba_pool_floor.py) is reuse cache, and on this machine it
# is cache bought with the entire KV pool: the pool is what survives after the
# mamba reservation and the GGUF dequant scratch are taken out, which is why it
# has been landing in the low thousands of tokens. Empty = auto, i.e. unchanged.
MAMBASLOTS=${MAMBASLOTS:-}
MAMBA_ARGS=()
if [ -n "$MAMBASLOTS" ]; then
  MAMBA_ARGS=(--max-mamba-cache-size "$MAMBASLOTS")
fi

'''
src = src.replace(anchor, block + anchor, 1)

a2 = '  "${KV_ARGS[@]}" \\\n'
assert a2 in src
src = src.replace(a2, a2 + '  "${MAMBA_ARGS[@]}" \\\n', 1)

tmp = P + ".tmp655"
open(tmp, "w").write(src)
os.chmod(tmp, 0o755)
os.replace(tmp, P)
print("patched ok")
