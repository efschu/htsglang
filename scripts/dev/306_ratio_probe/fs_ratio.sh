#!/bin/sh
# #306 step 1 -- how much compression the FILESYSTEM is already doing.
#
# The verdict in ANALYSE_306 §6.2 turns on this: a dataset with transparent
# compression has already taken the disk-tier win, so an application-level
# codec on top is worth only the marginal difference. `du` reports ALLOCATED
# blocks and `stat -c %s` the apparent size; their quotient is the ratio the
# storage layer is delivering for free, per file, with no probe of its own.
#
# Run it against any candidate cold-tier asset before pricing a codec for it.
set -eu

for f in "$@"; do
    [ -f "$f" ] || { printf '%s: not a file\n' "$f" >&2; continue; }
    apparent=$(stat -c %s "$f")
    allocated=$(du -B1 "$f" | cut -f1)
    awk -v a="$apparent" -v b="$allocated" -v n="$f" \
        'BEGIN { printf "%-60s apparent=%-14d allocated=%-14d fs_ratio=%.4f\n", n, a, b, a/b }'
done
