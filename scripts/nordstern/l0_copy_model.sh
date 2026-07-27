#!/bin/bash
# Copy the L0 checkpoint to the second host. ~25 GB at a measured 112 MiB/s
# over the 1 GbE link == roughly 4 minutes. No GPU involved.
#
# Paths, host and key come from the environment (MODEL_ROOT, RIG2_MODEL_DIR,
# RIG2_HOST, RIG2_KEY); source your local rig env file first. Unset variables
# fall back to placeholders so an unsourced run fails instead of copying to
# some other machine.
set -eu
SRC=${SRC:-${MODEL_ROOT:-<MODEL_ROOT>}/Qwen3.6-27B-FP8/}
DST=${DST:-${RIG2_MODEL_DIR:-<RIG2_MODEL_DIR>}/qwen3.6-27b-fp8/}
SECOND=${SECOND:-${RIG2_HOST:-<RIG2_IP>}}
KEY=${KEY:-${RIG2_KEY:-<RIG2_SSH_KEY>}}
ssh -i "$KEY" -o IdentitiesOnly=yes root@$SECOND "mkdir -p $DST"
rsync -aL --info=progress2 -e "ssh -i $KEY -o IdentitiesOnly=yes" "$SRC" "root@$SECOND:$DST"
echo "copied; verifying file count"
ssh -i "$KEY" -o IdentitiesOnly=yes root@$SECOND "ls $DST | wc -l"
