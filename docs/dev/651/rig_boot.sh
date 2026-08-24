#!/bin/bash
# Task #651 -- RIG boot for Qwen3.6-35B-A3B-UD-Q4_K_XL GGUF (CUDA, 3 cards).
#
# Successor to docs/dev/651/boot.sh, which is unusable as written: it pins
# WT=/spinning/wt-gguf-q4-651 (a tree 1509 commits behind the current
# integration line, which BOOT NUR NEUESTER STAND forbids booting) and a
# MODEL_DIR that does not exist on this box. This script runs the newest tree
# and resolves the checkpoint instead of asserting it.
#
# Staging is deliberate -- each stage adds exactly one variable, so a failure
# names its own cause instead of leaving three suspects:
#   STAGE=a  TP=1, no spec      -> loader / kernels / checkpoint
#   STAGE=b  TP=1, NEXTN spec   -> the #647 router-gate fix on-card (blk.40)
#   STAGE=c  TP=2, NEXTN spec   -> tensor parallelism
#   STAGE=d  PP=3, NO spec      -> pipeline parallelism, alone
#   STAGE=e  PP prefill + TP decode + spec, via --enable-phase-flip
#
# The d/e split is forced by the tree, not by taste. server_args.py:19303
# (reached from entrypoints/engine.py:889 via check_server_args) asserts:
#
#     assert self.speculative_algorithm is None or self.enable_phase_flip
#
# PP and speculation cannot run in the same phase: no draft worker exists in a
# PP phase, and the draft constructors take no pp_rank, so it is enforced by
# construction rather than by refusing a flag combination. The predecessor's
# boot.sh STAGE=d asked for PP=3 + NEXTN together and would have died on this
# assert -- it was never run, so nobody found out. Verified at desk 2026-08-24:
# PP=3+NEXTN REFUSED, PP=3 no-spec ACCEPTED, TP=2+NEXTN ACCEPTED.
#
# STAGE=e is therefore the configuration that actually delivers "PP and TP and
# speculation" on this tree: #631 Route A, one instance that runs PP for prefill
# and flips to TP for decode on the same ranks, with the drafter armed on the TP
# side at cutover.
#
# Env knobs: STAGE, TP, PP, SPEC=0|1, DEVICES=<sel>, PORT, GRAPHS=0|1, MODEL,
#            CTX, MEMFRAC, BOOT_LOG, FLIP_TP_VECTOR
set -u

WT=${WT:-/spinning/wt-651-rig}
VENV=/spinning/htsglang-gpu/.venv

# ---------------------------------------------------------------- checkpoint
# Resolved, not asserted. The predecessor hardcoded a models-cache path that is
# absent here; a boot that dies on a missing file after CUDA init has spent a
# GPU window to learn something `test -f` answers for free.
STAGE=${STAGE:-a}
case "$STAGE" in
  a|b|c|d|e) ;;
  *) echo "REFUSE: unknown STAGE '$STAGE' (expected a|b|c|d|e)" >&2; exit 2 ;;
esac

MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  for cand in \
    /spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    /spinning/llm_stuff/*/Qwen3.6-35B-A3B*UD-Q4_K_XL.gguf \
    /spinning/llm_stuff/club-3090/models-cache/*/Qwen3.6-35B-A3B*Q4_K_XL.gguf ; do
    [ -f "$cand" ] && { MODEL="$cand"; break; }
  done
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "REFUSE: no Qwen3.6-35B-A3B UD-Q4_K_XL GGUF found." >&2
  echo "Set MODEL=/abs/path/to.gguf explicitly." >&2
  exit 2
fi
MODEL_DIR="$(dirname "$MODEL")"
# The tokenizer must come from a sibling directory, not the .gguf: the fork
# loads tokenizer.json beside the checkpoint (8f713f6bd4 recorded this trap for
# the Gemma-4 adapter -- passing the gguf as --tokenizer-path re-parses the
# whole weight stream to find a vocab it already has).
TOKENIZER=${TOKENIZER:-$MODEL_DIR}

SPEC=${SPEC:-0}
GRAPHS=${GRAPHS:-0}
PORT=${PORT:-30040}
CTX=${CTX:-8192}
MEMFRAC=${MEMFRAC:-0.90}
# DRYRUN=1 prints the resolved cmdline and exits before launching. Everything
# above the launch -- checkpoint resolution, NVML device selection, the
# card-count check -- still runs, so this exercises the whole script at desk
# without spending a GPU window or claiming a card.
DRYRUN=${DRYRUN:-0}

# Stage presets. Explicit TP/PP/SPEC still win if exported.
case "$STAGE" in
  a) TP=${TP:-1}; PP=${PP:-1}; SPEC=${SPEC:-0}; DEVICES=${DEVICES:-5090} ;;
  b) TP=${TP:-1}; PP=${PP:-1}; SPEC=1;          DEVICES=${DEVICES:-5090} ;;
  c) TP=${TP:-2}; PP=${PP:-1}; SPEC=1;          DEVICES=${DEVICES:-all}  ;;
  # d: PP alone. SPEC is forced OFF -- see the assert quoted in the header.
  d) TP=${TP:-1}; PP=${PP:-3}; SPEC=0;          DEVICES=${DEVICES:-all}  ;;
  # e: PP prefill -> TP decode, drafter armed on the TP side at cutover.
  e) TP=${TP:-1}; PP=${PP:-3}; SPEC=1; FLIP=1;  DEVICES=${DEVICES:-all}  ;;
esac
FLIP=${FLIP:-0}
# No default exists for the flip's TP decode layout and the tree refuses without
# it ("there is no default because pool sizing derives from it"). 32,16,16 is the
# weighting the 2026-08-24 window-7 boot ran on these same three cards, i.e.
# roughly 5090 : 3080 : 3080; it is a starting point to be measured, not a
# derived optimum for this model.
FLIP_TP_VECTOR=${FLIP_TP_VECTOR:-32,16,16}
LOG=${BOOT_LOG:-/spinning/651-gguf-q4/stage_${STAGE}.boot.log}
mkdir -p "$(dirname "$LOG")"

# ------------------------------------------------------------------- devices
# Devices are resolved by NVML UUID, never by index: torch's enumeration and
# NVML's diverge, and NVML order itself can shift across boots/driver states.
# DEVICES=all -> every card, fastest first (5090 leads). DEVICES=5090 -> just it.
RESOLVED="$("$VENV/bin/python" - "$DEVICES" <<'PY'
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
    mem = N.nvmlDeviceGetMemoryInfo(h)
    cards.append((name, uuid, mem.free // 1024 // 1024, mem.total // 1024 // 1024))
if sel == "5090":
    pick = [c for c in cards if "5090" in c[0]]
elif sel == "all":
    pick = sorted(cards, key=lambda c: 0 if "5090" in c[0] else 1)
else:
    pick = [c for c in cards if c[1] in sel.split(",")]
print(",".join(c[1] for c in pick))
print(";".join(f"{c[0]}|{c[2]}|{c[3]}" for c in pick))
PY
)"
DEV_UUIDS="$(echo "$RESOLVED" | sed -n 1p)"
DEV_INFO="$(echo "$RESOLVED" | sed -n 2p)"
if [ -z "$DEV_UUIDS" ]; then
  echo "REFUSE: device selection '$DEVICES' resolved to nothing via NVML." >&2
  exit 1
fi
NDEV=$(echo "$DEV_UUIDS" | tr ',' '\n' | grep -c .)
NEED=$(( TP * PP ))
if [ "$NDEV" -lt "$NEED" ]; then
  echo "REFUSE: STAGE $STAGE needs $NEED card(s) (TP=$TP x PP=$PP), NVML resolved $NDEV." >&2
  exit 1
fi
echo "devices: $DEV_INFO"

# Arbitration safety net: the hardware always wins over the holder files.
echo "$DEV_INFO" | tr ';' '\n' | while IFS='|' read -r nm fr tot; do
  [ "${fr:-0}" -lt 15000 ] && echo "WARN: $nm has only ${fr} MiB free (total ${tot})" >&2
done

export CUDA_VISIBLE_DEVICES="$DEV_UUIDS"
export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
# NOT set on purpose: the barlink BAR1 transport. Rig window 1 (5598241c65)
# read the 5090 as 19.58 GiB CUDA-visible and concluded Q4_K_XL "cannot fit
# TP=1"; that ceiling was the BAR1 aperture env of that boot, not the card.
# Bare, NVML reports 32607 MiB total and the 2026-08-24 window-7 boot budgeted
# it 31800 MiB. Stage a therefore fits and the old verdict does not carry.

ARGS=(
  --model-path "$MODEL"
  --tokenizer-path "$TOKENIZER"
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
# PP requires the synchronous scheduler; the tree warns and forces it anyway,
# but passing it explicitly keeps the recorded cmdline honest about what ran.
[ "$PP" -gt 1 ] && ARGS+=(--disable-overlap-schedule)
if [ "$FLIP" = "1" ]; then
  ARGS+=(--enable-phase-flip --phase-flip-tp-vector "$FLIP_TP_VECTOR")
fi
# The two 3080s have no P2P on this rig (all PHB); the custom all-reduce path
# needs it, so TP collectives fall back to NCCL.
[ "$TP" -gt 1 ] && ARGS+=(--disable-custom-all-reduce)
if [ "$SPEC" = "1" ]; then
  # Qwen3.5/3.6 NEXTN drafts come from the SAME GGUF (blk.40 is the MTP block);
  # the draft path is mandatory for this family, there is no auto-default.
  # Passing the .gguf FILE here is also what arms the #290 fix in
  # model_config.py:698 (check_gguf_file -> quantization="gguf"), without which
  # the drafter is built dense and silently proposes noise at accept ~1.005.
  ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-draft-model-path "$MODEL"
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
fi

if [ "$DRYRUN" = "1" ]; then
  printf 'DRYRUN stage=%s TP=%s PP=%s SPEC=%s GRAPHS=%s\n' \
         "$STAGE" "$TP" "$PP" "$SPEC" "$GRAPHS"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'PYTHONPATH=%s\n' "$PYTHONPATH"
  printf '%s -m sglang.launch_server' "$VENV/bin/python"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

: > "$LOG"
{
  printf '=== #651 STAGE %s %s ===\n' "$STAGE" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'tree=%s commit=%s branch=%s dirty=%s\n' "$WT" \
         "$(git -C "$WT" rev-parse --short HEAD)" \
         "$(git -C "$WT" branch --show-current)" \
         "$(git -C "$WT" status --porcelain | wc -l)"
  printf 'model=%s\n' "$MODEL"
  printf 'devices=%s\n' "$DEV_INFO"
  printf 'TP=%s PP=%s SPEC=%s GRAPHS=%s CTX=%s MEMFRAC=%s\n' \
         "$TP" "$PP" "$SPEC" "$GRAPHS" "$CTX" "$MEMFRAC"
} >> "$LOG"

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}" >> "$LOG" 2>&1 &
echo "stage $STAGE pgid $!  log $LOG  port $PORT"
