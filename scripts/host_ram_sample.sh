#!/usr/bin/env bash
# Host-RAM sampler for #631 Route A.
#
# Corpse M (2026-08-09 21:47:29Z): the green run died at minute 5 because
# rank 0's scheduler took exit code -9 -- a kernel SIGKILL. The cgroup
# recorded oom_kill and a 112 GiB memory.peak against 120 GB of host RAM.
# Nothing in the phase-flip path is supposed to hold host RAM: the boot ran
# with enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=
# False, cpu_offload_gb=0, kv_session_offload_host_ram_gib=0 and
# enable_hierarchical_cache=False.
#
# This sampler exists to tell two very different stories apart:
#
#   (a) page cache high-water. Reading a 27 GB checkpoint three times
#       fills the file-backed pages of the cgroup. That inflates
#       memory.current and memory.peak but is reclaimable and CANNOT by
#       itself trigger an OOM kill.
#   (b) anonymous growth. Memory that is not backed by a file cannot be
#       reclaimed. If this climbs across flips, the cutover seam leaks and
#       every long run is on a timer.
#
# So the load-bearing columns are anon and file SEPARATELY, never their
# sum, plus oom_kill as a monotonic counter that answers "did the kernel
# kill anything during THIS window" without depending on dmesg (the kernel
# ring buffer is not readable in this container).
#
# Usage: host_ram_sample.sh <seconds> <out.csv> [interval_s]
set -euo pipefail

DURATION="${1:-3600}"
OUT="${2:-/spinning/evidence-631/host_ram.csv}"
INTERVAL="${3:-1}"

CG=/sys/fs/cgroup

read_stat() { awk -v k="$1" '$1==k {print $2; found=1} END {if (!found) print 0}' "$CG/memory.stat"; }
read_oom() { awk '$1=="oom_kill" {print $2; found=1} END {if (!found) print 0}' "$CG/memory.events"; }

echo "ts,current_b,anon_b,file_b,slab_b,oom_kill,rank0_rss_kb,rank1_rss_kb,rank2_rss_kb,total_sched_rss_kb" > "$OUT"

END=$(( $(date +%s) + DURATION ))
while [ "$(date +%s)" -lt "$END" ]; do
  ts=$(date -u +%FT%T.%3NZ)
  cur=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
  anon=$(read_stat anon)
  file=$(read_stat file)
  slab=$(read_stat slab)
  oom=$(read_oom)

  # Per-rank scheduler RSS. The schedulers are the processes that would
  # carry a seam leak; rank order follows the pid order they were spawned
  # in, which is rank 0,1,2 for this boot recipe.
  rss=($(ps -eo rss,cmd --no-headers 2>/dev/null \
          | grep -F "sglang::scheduler" | grep -v grep \
          | awk '{print $1}' | head -3))
  if [ "${#rss[@]}" -lt 3 ]; then
    # Fall back to any launch_server-descended python holding a CUDA ctx.
    rss=($(ps -eo rss,cmd --no-headers 2>/dev/null \
            | grep -E "sglang(\.launch_server|::)" | grep -v grep \
            | sort -rn | awk '{print $1}' | head -3))
  fi
  r0=${rss[0]:-0}; r1=${rss[1]:-0}; r2=${rss[2]:-0}
  tot=$(( r0 + r1 + r2 ))

  echo "$ts,$cur,$anon,$file,$slab,$oom,$r0,$r1,$r2,$tot" >> "$OUT"
  sleep "$INTERVAL"
done
