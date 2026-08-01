#!/bin/bash
# Release the page cache before a DeepSeek V4 Flash GGUF boot (#391).
#
# WHY THIS IS A STEP AND NOT A COMMENT
# ---------------------------------------------------------------------------
# Boot 9 (2026-08-01) started with ~20 GiB of page cache belonging to OTHER
# work on this box. `memory.current` -- which is what rammon.sh guards and what
# the OOM killer compares against the limit -- counts it, so the guard spent
# the whole run 20 GiB closer to firing than the load itself justified. The
# cache is reclaimable, but reclaimable is not reclaimable IN TIME: boot 8 died
# with the kernel reclaiming at full tilt. Starting from a clean baseline costs
# one command and removes the term.
#
# It is OPTIONAL. Nothing in the load depends on it; it only makes the headroom
# figure in ram.log mean what it says. The GGUF page cache of an earlier
# attempt of the SAME checkpoint needs no reset either way -- the loader pages
# those out as it re-consumes them (runbook 4.5.5).
#
# WHAT ACTUALLY WORKS WHERE (measured on this rig, 2026-08-01)
# ---------------------------------------------------------------------------
#   * `/proc/sys/vm/drop_caches` is the textbook answer and is NOT available in
#     this LXC container: the file is owned by the unmapped `nobody` and
#     opening it for write returns EACCES even as container root
#     (CAP_SYS_ADMIN is in the bounding set, the userns ownership is what
#     refuses). It is also host-GLOBAL -- on a shared Proxmox box it would
#     throw away every other container's cache to fix ours.
#   * `/sys/fs/cgroup/memory.reclaim` (cgroup v2, kernel >= 5.19) IS writable
#     here and is the better instrument anyway: it reclaims from THIS cgroup
#     only, which is exactly the accounting the guard watches. Measured:
#     writing `512M` moved `file` 52.57 -> 52.06 GiB, i.e. it released what it
#     was asked for and nothing else.
#
# So the ladder is memory.reclaim first, drop_caches second, and an honest
# report of "neither was available" third -- never a silent no-op.
set -u

TARGET_GIB=${1:-24}
# Overridable so the "neither mechanism is available" branch can be exercised
# without a container that actually lacks them.
CG=${DSV4_CGROUP_ROOT:-/sys/fs/cgroup}

file_gib() {
  awk '/^file /{printf "%.2f", $2/1073741824}' "$CG/memory.stat" 2>/dev/null \
    || echo "?"
}

before=$(file_gib)
echo "preboot cache reset: file=${before} GiB before, asking for ${TARGET_GIB} GiB"

if [ -w "$CG/memory.reclaim" ] && printf '%sG' "$TARGET_GIB" \
    > "$CG/memory.reclaim" 2>/dev/null; then
  echo "  cgroup memory.reclaim: released down to file=$(file_gib) GiB"
  exit 0
fi

# A partial reclaim reports EAGAIN and still frees what it could, so re-read
# rather than trusting the exit status alone.
after=$(file_gib)
if [ "$after" != "$before" ]; then
  echo "  cgroup memory.reclaim: partial, file=${after} GiB (EAGAIN is normal"
  echo "  when the ask exceeds what is reclaimable right now)"
  exit 0
fi

if [ -w /proc/sys/vm/drop_caches ] \
    && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
  echo "  /proc/sys/vm/drop_caches: dropped host-wide, file=$(file_gib) GiB"
  exit 0
fi

cat <<'EOF'
  NEITHER mechanism was available in this container.
  memory.reclaim is not writable and /proc/sys/vm/drop_caches refuses even
  container root (userns ownership). Run this on the PVE host instead, before
  launching, and note in the run directory that it was done there:

      sync && echo 3 > /proc/sys/vm/drop_caches

  Do NOT skip the note: without it a later reader cannot tell a clean baseline
  from a 20 GiB foreign one, and the headroom in ram.log means nothing.
EOF
exit 1
