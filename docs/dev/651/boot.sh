#!/bin/bash
# Task #651: parameterized boot for Qwen3.6-35B-A3B-UD-Q4_K_XL GGUF.
#
# Staging is deliberate -- each stage adds exactly one variable, so a failure
# names its own cause instead of leaving three suspects:
#   STAGE=a  TP=1, no spec              -> loader / kernels / checkpoint
#   STAGE=b  TP=1, NEXTN spec           -> the #647 router-gate fix on-card
#   STAGE=c  TP=2, NEXTN spec           -> tensor parallelism
#   STAGE=d  PP=3, NEXTN spec           -> pipeline parallelism
#
# Env knobs: STAGE, TP, PP, SPEC=0|1, DEVICES=<uuid,uuid>, PORT, GRAPHS=0|1
set -u

WT=/spinning/wt-gguf-q4-651
VENV=/spinning/htsglang-gpu/.venv
MODEL_DIR=/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
GGUF="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

STAGE=${STAGE:-a}
TP=${TP:-1}
PP=${PP:-1}
SPEC=${SPEC:-0}
GRAPHS=${GRAPHS:-0}
PORT=${PORT:-30040}
CTX=${CTX:-8192}
MEMFRAC=${MEMFRAC:-0.90}
LOG=${BOOT_LOG:-/spinning/651-gguf-q4/stage_${STAGE}.boot.log}

# Devices are resolved by NVML UUID, never by index: torch's enumeration and
# NVML's diverge, and NVML order itself can shift across boots/driver states.
# DEVICES=all -> every card, fastest first (5090 leads). DEVICES=5090 -> just it.
DEVSEL=${DEVICES:-5090}
RESOLVED="$("$VENV/bin/python" - "$DEVSEL" <<'PY'
import sys
import pynvml as N
sel = sys.argv[1]
N.nvmlInit()
cards = []
for i in range(N.nvmlDeviceGetCount()):
    h = N.nvmlDeviceGetHandleByIndex(i)
    name = N.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    uuid = N.nvmlDeviceGetUUID(h)
    uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
    free = N.nvmlDeviceGetMemoryInfo(h).free // 1024 // 1024
    cards.append((name, uuid, free))
if sel == "5090":
    pick = [c for c in cards if "5090" in c[0]]
elif sel == "all":
    # fastest first: 5090 ahead of the 3080s, matching the rig convention
    pick = sorted(cards, key=lambda c: 0 if "5090" in c[0] else 1)
else:
    pick = [c for c in cards if c[1] in sel.split(",")]
print(",".join(c[1] for c in pick))
print(";".join(f"{c[0]}|{c[2]}" for c in pick))
PY
)"
DEV_UUIDS="$(echo "$RESOLVED" | sed -n 1p)"
DEV_INFO="$(echo "$RESOLVED" | sed -n 2p)"
if [ -z "$DEV_UUIDS" ]; then
  echo "REFUSE: device selection '$DEVSEL' resolved to nothing via NVML." >&2
  exit 1
fi
echo "devices: $DEV_INFO"

# Arbitration safety net: the hardware always wins over the files.
echo "$DEV_INFO" | tr ';' '\n' | while IFS='|' read -r nm fr; do
  if [ "${fr:-0}" -lt 15000 ]; then
    echo "WARN: $nm has only ${fr} MiB free" >&2
  fi
done

export CUDA_VISIBLE_DEVICES="$DEV_UUIDS"
export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

ARGS=(
  --model-path "$GGUF"
  --tokenizer-path "$MODEL_DIR"
  --served-model-name Qwen3.6-35B-A3B-Q4
  --load-format gguf
  --quantization gguf
  --tp-size "$TP"
  --pp-size "$PP"
  --context-length "$CTX"
  --max-running-requests 1
  --mem-fraction-static "$MEMFRAC"
  --trust-remote-code
  --host 127.0.0.1 --port "$PORT"
)
[ "$GRAPHS" = "1" ] || ARGS+=(--disable-cuda-graph)
# The two 3080s have no P2P on this rig; the custom all-reduce path needs it.
[ "$TP" -gt 1 ] && ARGS+=(--disable-custom-all-reduce)
if [ "$SPEC" = "1" ]; then
  # Qwen3.5/3.6 NEXTN drafts from the SAME GGUF (blk.40 is the MTP block);
  # the draft path is mandatory for this family, there is no auto-default.
  ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-draft-model-path "$GGUF"
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

: > "$LOG"
{
  printf '=== #651 STAGE %s %s ===\n' "$STAGE" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'tree=%s commit=%s branch=%s dirty=%s\n' "$WT" \
         "$(git -C "$WT" rev-parse --short HEAD)" \
         "$(git -C "$WT" branch --show-current)" \
         "$(git -C "$WT" status --porcelain | wc -l)"
  printf 'devices=%s\n' "$DEV_INFO"
  printf 'TP=%s PP=%s SPEC=%s GRAPHS=%s CTX=%s\n' "$TP" "$PP" "$SPEC" "$GRAPHS" "$CTX"
} >> "$LOG"

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}" >> "$LOG" 2>&1 &
echo "stage $STAGE pgid $!  log $LOG  port $PORT"
