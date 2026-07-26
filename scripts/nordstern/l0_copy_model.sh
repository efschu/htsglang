#!/bin/bash
# Copy the L0 checkpoint to the second host. ~25 GB at a measured 112 MiB/s
# over the 1 GbE link == roughly 4 minutes. No GPU involved.
set -eu
SRC=${SRC:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8/}
DST=${DST:-/root/models/qwen3.6-27b-fp8/}
SECOND=${SECOND:-192.168.0.89}
KEY=${KEY:-/root/.ssh/id_ed25519_192.168.0.89}
ssh -i "$KEY" -o IdentitiesOnly=yes root@$SECOND "mkdir -p $DST"
rsync -aL --info=progress2 -e "ssh -i $KEY -o IdentitiesOnly=yes" "$SRC" "root@$SECOND:$DST"
echo "copied; verifying file count"
ssh -i "$KEY" -o IdentitiesOnly=yes root@$SECOND "ls $DST | wc -l"
