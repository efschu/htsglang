#!/usr/bin/env python3
"""#655: add KVDTYPE opt-in + an in-script memory snapshot at the exact moment
the launch begins. Idempotent; makes a .655bak once."""
import os
import shutil
import sys

P = "/root/651-p2/scripts/boot_ondemand.sh"
BAK = P + ".655bak"
src = open(P).read()

if "SNAP655" in src:
    print("already patched")
    sys.exit(0)

if not os.path.exists(BAK):
    shutil.copy2(P, BAK)

# --- A) snapshot immediately after the page cache is dropped -----------------
anchor_a = 'echo "note: could not drop caches (not root?); load margin will be tighter"\n'
assert anchor_a in src, "drop_caches anchor not found"
snap = anchor_a + '''
# SNAP655: #655 instrumentation. The KV pool on this APU is the thin residual
# left after the weights, so what the pool ends up being is decided by host
# memory state at THIS instant -- after the cache drop, before the 22.7 GiB
# allocation. Recording it here rather than from outside is the difference
# between correlating against the real starting condition and correlating
# against a snapshot taken some seconds and one cache drop earlier.
if [ -n "${SNAP655_FILE:-}" ]; then
  /root/651-p2/scripts/memsnap655.sh "postdrop" >> "$SNAP655_FILE" 2>/dev/null || true
fi
'''
src = src.replace(anchor_a, snap, 1)

# --- B) opt-in KV cache dtype ------------------------------------------------
anchor_b = "exec python -m sglang.launch_server \\\n"
assert anchor_b in src, "exec anchor not found"
kvblock = '''# KV cache storage dtype. Default "auto" keeps the model dtype and appends
# NOTHING to the command line, so the established operating point is bit-for-bit
# the same command it was before this variable existed. fp8_e5m2 / fp8_e4m3
# halve the per-token cell, which on this machine is the only lever that changes
# the pool arithmetic at all: the pool is available_bytes // cell_size, and
# available_bytes here is a ~1.2 GiB residual that no tuning has been able to
# enlarge. Halving the divisor is worth twice the context from the same residual
# -- at the cost of a lossy KV, so it stays opt-in.
KVDTYPE=${KVDTYPE:-auto}
KV_ARGS=()
if [ "$KVDTYPE" != "auto" ]; then
  KV_ARGS=(--kv-cache-dtype "$KVDTYPE")
fi

'''
src = src.replace(anchor_b, kvblock + anchor_b, 1)

anchor_c = '  --mem-fraction-static "$MEMFRAC" \\\n'
assert anchor_c in src, "memfrac anchor not found"
src = src.replace(anchor_c, anchor_c + '  "${KV_ARGS[@]}" \\\n', 1)

open(P, "w").write(src)
print("patched ok")
