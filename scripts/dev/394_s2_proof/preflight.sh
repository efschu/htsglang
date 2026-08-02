#!/usr/bin/env bash
# #394 slice 2 proof window -- PREFLIGHT. Run this BEFORE booting either arm.
#
# Four things, all cheap, all expensive to discover late:
#   1. /dev/shm capacity. The reference rig was remounted 63 -> 96 GiB on
#      2026-08-02 to fit an ~88 GiB cold tier, and THAT REMOUNT IS NOT
#      RESTART-PERSISTENT. A container or host restart silently puts the cap
#      back, and the failure lands mid-load, ~7 minutes in.
#   2. Leftover segments from a previous launch. They are refused at read time
#      by the sealed header, but they still occupy the tmpfs budget checked
#      in (1), so a stale set can make a perfectly sized tier not fit.
#   3. The rank -> card mapping, resolved via NVML, printed as a table. Torch
#      order is not NVML order on this rig; every positional shortcut breaks
#      here, and the arm's --rank-gpu-id has to be derived from this output
#      rather than assumed.
#   4. The measured H2D provenance. #394 refuses to weight a split on a
#      nameplate when asked to, and the proportional arm is only meaningful
#      if the card probe cache is present and current.
#
# Exits non-zero on the first hard failure. One PASS/WARN/FAIL line per check.

set -uo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT or provide /root/rig-env.sh}"
VENV="${VENV:?set VENV}"
WT="${WT:-$(dirname "$REPO_ROOT")/wt-394-s2}"
# Cold-tier bytes the run will want. Default sized for V4-Flash UD-IQ3_XXS at
# the 0.485/0.42/0.42 resident fractions of the 2026-08-02 battery; override
# for any other recipe rather than trusting this number.
NEED_GIB="${NEED_GIB:-88}"

fail=0
say() { printf '%-6s %s\n' "$1" "$2"; [ "$1" = FAIL ] && fail=1; return 0; }

echo "== #394 slice 2 preflight =="

# --- 1. tmpfs capacity ------------------------------------------------------
df -h /dev/shm | tail -1
cap_gib=$(python3 - <<'PY'
import os
st = os.statvfs("/dev/shm")
print(f"{st.f_blocks * st.f_frsize / 2**30:.1f}")
PY
)
free_gib=$(python3 - <<'PY'
import os
st = os.statvfs("/dev/shm")
print(f"{st.f_bavail * st.f_frsize / 2**30:.1f}")
PY
)
if awk "BEGIN{exit !($cap_gib >= $NEED_GIB)}"; then
  say PASS "/dev/shm capacity ${cap_gib} GiB >= ${NEED_GIB} GiB needed"
else
  say FAIL "/dev/shm capacity ${cap_gib} GiB < ${NEED_GIB} GiB needed -- remount:
       mount -o remount,size=96G /dev/shm     (NOT restart-persistent)"
fi
if awk "BEGIN{exit !($free_gib >= $NEED_GIB)}"; then
  say PASS "/dev/shm free ${free_gib} GiB"
else
  say FAIL "/dev/shm free ${free_gib} GiB < ${NEED_GIB} GiB -- see stale segments below"
fi

# The same check the runtime does, through the runtime's own function, so a
# drift between this script and cold_tier_shm.preflight is impossible.
PYTHONPATH="$WT/python" "$VENV/bin/python" - "$NEED_GIB" <<'PY'
import sys

from sglang.srt.layers.moe.cold_tier_shm import preflight

need = int(float(sys.argv[1]) * 2**30)
print("       cold_tier_shm.preflight:", preflight(need))
PY

# --- 2. stale segments ------------------------------------------------------
stale=$(find /dev/shm -maxdepth 1 -name 'sgl-cold-*' 2>/dev/null | wc -l)
if [ "$stale" -eq 0 ]; then
  say PASS "no leftover sgl-cold-* segments"
else
  bytes=$(du -sh --apparent-size /dev/shm/sgl-cold-* 2>/dev/null | tail -1 | cut -f1)
  say WARN "$stale leftover sgl-cold-* entries (~$bytes). They are REFUSED at
       read time by the sealed header, so they cannot corrupt a run -- but they
       occupy the budget above. Remove only if no server is running:
         ls /dev/shm/sgl-cold-* && rm /dev/shm/sgl-cold-*"
fi

# --- 3. rank -> card, via NVML ---------------------------------------------
PYTHONPATH="$WT/python" "$VENV/bin/python" - <<'PY'
from sglang.srt.registry.nvml import identity_map

imap = identity_map()
print("       NVML index -> card (derive --rank-gpu-id from THIS, not from a guess)")
for card in sorted(imap.cards, key=lambda c: c.nvml_index):
    print(
        f"       nvml={card.nvml_index}  cuda={card.cuda_ordinal}  "
        f"{card.pci_bus_id}  {card.name}  {card.total_bytes / 2**30:.0f} GiB"
    )
PY

# --- 4. measured H2D provenance --------------------------------------------
PYTHONPATH="$WT/python" "$VENV/bin/python" - <<'PY'
from sglang.srt.layers.moe.expert_offload import (
    HOST_SHARD_SOURCE_PROBE,
    resolve_host_shard_ratio,
)
from sglang.srt.registry.nvml import identity_map

uuids = tuple(
    c.uuid for c in sorted(identity_map().cards, key=lambda c: c.cuda_ordinal)
)
ratio = resolve_host_shard_ratio(len(uuids), uuids)
verdict = "PASS" if ratio.source == HOST_SHARD_SOURCE_PROBE else "WARN"
print(f"{verdict:<6} host-shard ratio: {ratio.describe()}")
if verdict != "PASS":
    print(
        "       the proportional arm would be weighted by the NVML nameplate. "
        "Run the rigmon card probe first, or accept an ESTIMATE arm and say so."
    )
PY

echo
if [ "$fail" -eq 0 ]; then
  echo "PREFLIGHT OK -- proceed to boot_ab.sh"
else
  echo "PREFLIGHT FAILED -- fix the FAIL lines before spending a boot"
fi
exit "$fail"
