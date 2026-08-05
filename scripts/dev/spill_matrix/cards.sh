#!/bin/bash
# Card identity and VRAM-corridor sampler for the spill matrix window.
#
# Two jobs, deliberately in one place so a result file always carries the
# identity of the cards it was measured on:
#
#   identity  -- print the NVML index -> name -> UUID -> PCI bus map. The rig's
#                torch order and NVML order DIVERGE (cuda:0 is the 5090 at NVML
#                index 1), so no recipe may hardcode an index; derive it here.
#   corridor  -- sample free VRAM on every card at 1 Hz into a TSV. The rule is
#                >= 400 MiB free on ALL cards for the whole load; a violation is
#                a result, not a nuisance, so the sampler records rather than
#                intervenes.
#
# Usage:
#   cards.sh identity
#   cards.sh corridor <out.tsv> [seconds]     # default 600 s
#   cards.sh verdict  <out.tsv>               # min free per card + PASS/FAIL
set -u

FLOOR_MIB=400

usage() { echo "usage: $0 {identity|corridor <out.tsv> [seconds]|verdict <out.tsv>}" >&2; exit 2; }

cmd_identity() {
    echo "# NVML index -> name -> UUID -> PCI bus (resolve, never assume)"
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total \
               --format=csv,noheader
}

cmd_corridor() {
    local out=$1 secs=${2:-600} i=0
    : > "$out"
    printf 'epoch\tidx\tfree_mib\tused_mib\n' >> "$out"
    while [ "$i" -lt "$secs" ]; do
        # One nvidia-smi call per second, all cards at once: cheap and keeps the
        # samples of one instant on one clock.
        nvidia-smi --query-gpu=index,memory.free,memory.used \
                   --format=csv,noheader,nounits \
        | while IFS=', ' read -r idx free used; do
            printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$idx" "$free" "$used" >> "$out"
          done
        i=$((i + 1))
        sleep 1
    done
}

cmd_verdict() {
    local out=$1
    [ -s "$out" ] || { echo "corridor file $out is empty -- NOT-EXAMINED"; exit 3; }
    # Minimum free per card over the whole sampled window.
    awk -v floor="$FLOOR_MIB" '
        NR > 1 { if (!(($2) in mn) || $3 < mn[$2]) mn[$2] = $3; n++ }
        END {
            if (n == 0) { print "no samples -- NOT-EXAMINED"; exit 3 }
            bad = 0
            for (idx in mn) {
                printf "card %s: min free %s MiB\n", idx, mn[idx]
                if (mn[idx] < floor) bad = 1
            }
            printf "samples=%d floor=%d MiB -> %s\n", n, floor, (bad ? "CORRIDOR-RED" : "CORRIDOR-GREEN")
            exit (bad ? 1 : 0)
        }' "$out"
}

case "${1:-}" in
    identity) cmd_identity ;;
    corridor) [ $# -ge 2 ] || usage; cmd_corridor "$2" "${3:-600}" ;;
    verdict)  [ $# -ge 2 ] || usage; cmd_verdict "$2" ;;
    *) usage ;;
esac
