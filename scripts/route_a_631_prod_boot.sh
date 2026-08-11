#!/usr/bin/env bash
# #631 Route A PRODUCTION boot on port 30030.
#
# This is route_a_631_flip_boot.sh (the accepted acceptance recipe, commit
# 89bd1c569f / FINAL_631.md) plus the serving-facing surface and the
# environment carried over from the previous production recipe
# /root/bin/start-serving-30030.sh.
#
# ONE instance, THREE ranks: PP=3 prefill (stage ratio 2,1,1) -> live flip
# -> TP=3 decode (vector 30,17,17) with NEXTN speculation. The flip is
# armed by POST /phase_flip; there is no automatic cadence.
#
# DEVICE SELECTION IS BY UUID, NEVER BY NVML INDEX (measured drift on this
# rig moved the 5090 between indices across boots). The 5090 is ordered
# FIRST in CUDA_VISIBLE_DEVICES; --rank-gpu-id indexes INTO the visible
# set, which is ordering-independent.
#
# CORRIDOR: RANK_MIB 16150,10550,10550 is the corridor-confirmed budget
# (FINAL_631 §1, boot 23: minimum NVML free 2338/1221/1130 MiB at 100 ms
# sampling through boot, flip and a speculating decode). The higher
# 16400,10750,10750 used for the acceptance ladder breached the 1024 MiB
# corridor at the boot peak. Do not raise without re-sampling.
set -euo pipefail

WT="${WT:-/spinning/wt-631-routea}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8}"
PORT="${PORT:-30030}"
# Bind ALL interfaces so the server is reachable from the local network,
# not just from this box (operator request 2026-08-08). It previously bound
# 127.0.0.1, which made it localhost-only: every LAN client -- laptop,
# phone, the hetero hosts -- got connection refused with no indication why.
#
# NOTE, deliberately not hidden: 0.0.0.0 exposes the OpenAI-compatible API
# to everything that can route to this host, and this server runs without
# --api-key. That is acceptable on this private rig LAN and is the
# requested behaviour; put an --api-key (or a firewall rule) in front of it
# before this box ever sits on an untrusted network.
HOST="${HOST:-0.0.0.0}"
# Shipped 2026-08-08 after exclusive KV backing (89572e996d). Each rank
# sits just under the per-rank PHYSICAL-availability ceiling that
# _profile_available_bytes enforces at PP sizing time (~22.4 GiB rank 0,
# ~14.9 GiB the 3080s) -- that check, not the corridor, is what now caps
# the pool: corridor acceptance on the binding 32768-token prefill rung
# still measured 2198/4033/2292 MiB free against a floor of 1024.
# CORRIDOR-CONFIRMED UNDER THE ACCEPTANCE LOAD (#631, 2026-08-09), which
# is the only load a corridor number may be quoted against. Sampled at
# 100 ms for 390 s through boot, 23 policy-driven flips and a speculating
# decode with CUDA graphs on:
#
#   per-card MINIMUM free 1034 / 2824 / 1228 MiB, floor 1024, 0 breaches
#   on every card. 87/87 requests ok, 0 aborted.
#
# Derived, not guessed: each rank's budget was moved by that card's
# measured distance from the floor under this same load, twice. Note the
# 5090 still shows ~1800 MiB of headroom -- rank 0 can go up, but raising
# it alone does not raise max_total_num_tokens while rank 1 is the
# min-reducing rank (see SGLANG_UNEVEN_TOKEN_VECTOR below).
#
# DO NOT raise any entry without re-sampling under the acceptance load,
# and do not try to buy corridor by LOWERING these: measured on this rig,
# cutting ~1 GiB of budget moved the free minimum by +690 / +250 / +4 MiB
# on the three cards. The allocator, not the budget, governs the floor --
# which is why PYTORCH_CUDA_ALLOC_CONF is set below.
RANK_MIB="${RANK_MIB:-22700,11920,11970}"
CTX="${CTX:-262144}"
MAX_RUNNING="${MAX_RUNNING:-4}"
# Measured, not inherited -- see the ladder at the flag site below.
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-512}"
MAMBA_SLOTS="${MAMBA_SLOTS:-20}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-500000}"
# OFF by default and not a preference: --enable-phase-flip hard-refuses
# --enable-hierarchical-cache at argument time (#630). A default of 1 made
# the script unbootable without an explicit HICACHE=0.
# Who decides when to flip. 'manual' reproduces every boot before the
# policy existed: nothing flips unless a POST /phase_flip says so, which
# is why an unattended instance rested in whatever layout the last call
# left. 'auto' installs the #631 phase policy. The policy REFUSES to boot
# without a threshold, so PHASE_POLICY_TP_TOK_S (the measured TP-phase
# prefill throughput at the 8k rung) or PHASE_POLICY_FLIP_TOKENS (an
# explicit N) must accompany POLICY=auto -- a threshold is a measurement,
# not a default.
POLICY="${POLICY:-manual}"
# SPEC=off boots the TP decode phase WITHOUT the NEXTN draft worker.
#
# Not a convenience knob: it isolates the resident-request carry (#631
# J.3) from the draft state question. A request that prefills in the PP
# phase has no draft KV, because the PP phase carries no draft worker at
# all -- so a carried request entering a SPECULATING TP phase raises a
# second, independent question on top of the carry. SPEC=off answers the
# carry alone; the speculating boot is the rung after it. The cutover
# already treats a draft-less instance as a first-class shape
# ("bit-for-bit the state an instance without speculation has").
SPEC="${SPEC:-on}"
PHASE_POLICY_TP_TOK_S="${PHASE_POLICY_TP_TOK_S:-}"
PHASE_POLICY_FLIP_TOKENS="${PHASE_POLICY_FLIP_TOKENS:-}"
# MEASURED, NOT GUESSED (2026-08-09, 32768-token prompts, SPEC=off):
# the minimum dwell is paid IN PREFILL THROUGHPUT, because a prompt that
# arrives while the instance is in TP keeps prefilling there until the
# flip is allowed to arm.
#   min dwell 15s -> policy could not arm for 15s; ~14300 of 32768 tokens
#                    were already computed in TP; 1485 tok/s
#   min dwell  2s -> policy arms after the FIRST chunk (30720 tok still
#                    pending); 4524 tok/s, i.e. the full PP-layout rate
# 3s keeps a thrash bound while leaving the flip inside one chunk. The
# structural thrash protection is the hysteresis band around N, not this.
PHASE_POLICY_MIN_DWELL_S="${PHASE_POLICY_MIN_DWELL_S:-3}"
# STRICT PHASE PURITY (user rule, 2026-08-09) and the two windows that make
# it progress. min_dwell above is a THRASH bound and cannot schedule the
# alternation: it gates how SOON a flip may arm, never how LONG a phase is
# allowed to keep the other side waiting. Under purity that difference is
# the whole game -- the PP window is what stops the instance parking in the
# prefill layout with decode work it is forbidden to run (measured defect
# K: 508 decode records in PP, 0 % CUDA graphs, 35 tok/s), and the TP floor
# is what stops the always-present backlog dragging the instance back out
# of the decode layout after one min_dwell.
PHASE_POLICY_PP_WINDOW_S="${PHASE_POLICY_PP_WINDOW_S:-15}"
PHASE_POLICY_TP_DECODE_FLOOR_S="${PHASE_POLICY_TP_DECODE_FLOOR_S:-10}"
PHASE_FLIP_PURITY="${PHASE_FLIP_PURITY:-strict}"
# #656 spec item 6: the spill ladder. 'cache' is the server's own default and
# is restated here so an A/B can set it to 'none' without editing the script.
# Do NOT read the default off this line -- server_args owns it; this only
# makes the knob reachable.
PHASE_FLIP_SPILL_DEPTH="${PHASE_FLIP_SPILL_DEPTH:-cache}"
# PP stage ratio = the PP layout's LAYER split, and therefore also its weight
# shard split and its per-rank KV layer count. 2,1,1 (=16/8/8 of 32) is the
# shipped recipe. It is reachable here because rank 0's PP KV pool is what
# binds the 5090 at large pools: at 600000 tokens the PP pool is 8.53 GiB on
# rank 0 against 6.45 GiB for the TP pool, so the 5090's ceiling is set by a
# split the TP vector cannot touch.
PP_STAGE_RATIO="${PP_STAGE_RATIO:-2,1,1}"
PHASE_POLICY_IDLE_DWELL_S="${PHASE_POLICY_IDLE_DWELL_S:-}"
PHASE_IDLE_STATE="${PHASE_IDLE_STATE:-}"
HICACHE="${HICACHE:-0}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"
# --kv-pressure-ladder auto REFUSES on this rig: it cannot map ranks to
# cards when the node holds different card models (3080 20480 + 5090
# 32607) without an explicit rank->card vector. Off by default here; pass
# KV_LADDER=<vector> deliberately if you have one.
KV_LADDER="${KV_LADDER:-}"
if [ "$HICACHE" = "1" ]; then
  HICACHE_FLAGS="--enable-hierarchical-cache --hicache-ratio $HICACHE_RATIO --hicache-write-policy write_through --hicache-mem-layout page_first_direct --hicache-io-backend direct"
else
  HICACHE_FLAGS=""
fi
if [ "$SPEC" = "off" ]; then
  SPEC_FLAGS=""
else
  SPEC_FLAGS="--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
fi
LOGDIR="${LOGDIR:-/tmp/route-a-631}"
LOG="${SERVING_LOG:-/spinning/serving-30030.boot.log}"
mkdir -p "$LOGDIR"

# The production stop guard. Same contract as start-serving-30030.sh: while
# the marker exists, production must not be restored.
if [ -f /spinning/PRODUCTION_STOPPED ]; then
  echo "REFUSED: production is stopped by operator order." >&2
  cat /spinning/PRODUCTION_STOPPED >&2
  exit 3
fi

# Single-instance guard (the 19:44 Bar1Failed was a second boot racing a
# live instance for the aperture).
if pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null 2>&1; then
  echo "REFUSE: a serving instance for port $PORT is already running." >&2
  echo "Read /spinning/gpu-arb/holder for the owner; stop it with" >&2
  echo "kill -TERM -- -<pgid> before rebooting." >&2
  exit 1
fi

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="/spinning/htsglang-gpu/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# THE CORRIDOR KNOB, and it is not --rank-gpu-memory-mib (#631, measured
# 2026-08-09). torch's caching allocator expands into whatever the KV pool
# does not take and, with the default segment allocator, never returns
# those pages to the driver -- so the free-VRAM minimum on a card doing
# heavy prefill converges toward zero no matter how the per-rank budget is
# set. Two shifts iterated the budget against this. Same load, same
# budget, only this variable changed:
#
#   card    MIN free default -> expandable_segments    breaches
#   3080a        54 ->  146                             2356 -> 2447
#   5090        804 -> 1672                             1400 ->    0
#   3080b       292 ->  448                             2355 ->  703
#
# The 5090 goes from breaching to HOLDING the 1024 MiB floor outright.
# 100/100 requests ok, 0 aborted, and the post-idle 32768-token probe
# still prefills in PP at 4507 tok/s, so nothing was paid for it here.
# Overridable because it interacts with CUDA graph capture and the VMM
# arena, and an A/B needs the old behaviour reachable on purpose.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
# The flip's TP phase token-shards KV under the WEIGHTED owner rule; the
# boot builder REFUSES without this pair (phase_flip_boot).
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
# Mixed 5090+3080 group is the CONFIGURATION, not a symptom.
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

# #656 SPEC ITEM 12: THE KV ARENA'S COMMIT CHUNK. A PRECONDITION, NOT A KNOB.
#
# Without a commit chunk the arena holds ONE extent per buffer, and
# decommit_range releases only extents lying WHOLLY above the keep point --
# so a partial shrink returns exactly zero to the driver while still lowering
# the pool's idea of its size. Every row-range primitive in kv_vmm_backing is
# inert, the KV relief rung refuses to register (correctly), and the seam's
# own waved span release is skipped too.
#
# THE VALUE COMES FROM A FORMULA, NOT FROM INTUITION ABOUT PAGE SIZES:
#
#     min_release_rows = commit_chunk_bytes * n_buffers / bytes_per_row
#
# Release is extent-granular PER BUFFER and the arena holds 2*layer_num of
# them, so the smallest release that can return anything is one chunk in
# EVERY buffer. At 256 MiB that is 256 MiB * 28 / 15 KiB = 489132 rows --
# more than the whole 500000-row pool, i.e. structurally unable to pay. At
# 8 MiB it is ~15285 rows (~230 MiB/rank), the right size against a gate
# asking a few hundred MiB. Measured on metal 2026-08-11 at exactly that:
# 2016 / 1440 / 1152 MiB released on the three ranks in one shrink.
export SGLANG_FLIP_SEAM_CHUNK_MIB="${SGLANG_FLIP_SEAM_CHUNK_MIB:-8}"

# #631: the phase-flip presence rendezvous tag. Set ONCE here so every
# rank of this boot inherits the SAME value, and so a LATER boot gets a
# different one. Both properties are load-bearing: identical across ranks
# because the flags are a rendezvous, unique per boot because a colliding
# tag let boot 15 read boot 14's leftover markers and open the entry gate
# on peers that were not there.
export SGLANG_PHASE_FLIP_INSTANCE="${SGLANG_PHASE_FLIP_INSTANCE:-$(date +%s)-$$}"

# #631 phase policy tuning. Exported only when set, so an unset knob keeps
# the module default rather than exporting an empty string that the
# parser would have to special-case.
[ -n "$PHASE_POLICY_TP_TOK_S" ] && export SGLANG_PHASE_POLICY_TP_TOK_S="$PHASE_POLICY_TP_TOK_S"
[ -n "$PHASE_POLICY_FLIP_TOKENS" ] && export SGLANG_PHASE_POLICY_FLIP_TOKENS="$PHASE_POLICY_FLIP_TOKENS"
[ -n "$PHASE_POLICY_MIN_DWELL_S" ] && export SGLANG_PHASE_POLICY_MIN_DWELL_S="$PHASE_POLICY_MIN_DWELL_S"
[ -n "$PHASE_POLICY_PP_WINDOW_S" ] && export SGLANG_PHASE_POLICY_PP_WINDOW_S="$PHASE_POLICY_PP_WINDOW_S"
[ -n "$PHASE_POLICY_TP_DECODE_FLOOR_S" ] && export SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S="$PHASE_POLICY_TP_DECODE_FLOOR_S"
[ -n "$PHASE_POLICY_IDLE_DWELL_S" ] && export SGLANG_PHASE_POLICY_IDLE_DWELL_S="$PHASE_POLICY_IDLE_DWELL_S"
[ -n "$PHASE_IDLE_STATE" ] && export HTSGLANG_PHASE_IDLE_STATE="$PHASE_IDLE_STATE"

# --- decode-phase KV token split -------------------------------------------
# The flip's TP vector 30,17,17 drives BOTH the weight shard plan and, by
# default, the KV token split. Those two want different ratios: the weight
# shard follows compute, while the token split must follow each rank's
# REMAINING memory after its weights are placed. Sizing KV with the compute
# vector makes the most compute-loaded rank the binding one and min-reduces
# the whole pool to its unit:
#
#   rank 0 local capacity 12750 tok / ratio 30 = unit 425  <- binds
#   rank 1 local capacity 68646 tok / ratio 17 = unit 4038
#   rank 2 local capacity 30515 tok / ratio 17 = unit 1795
#   -> global max_total_num_tokens 27200, with ranks 1 and 2 left idle.
#
# The server computes the token-proportional vector itself and prints it as
# a restart hint; 7,39,18 lifts the same physical memory to ~108480 tokens
# (4.0x) with no extra VRAM, by giving each rank a share of the token
# vector matched to what it can actually hold.
export SGLANG_UNEVEN_TOKEN_VECTOR="${SGLANG_UNEVEN_TOKEN_VECTOR:-28,26,20}"

# --- transport ---------------------------------------------------------------
# barlink is the standing default transport: defects are fixed forward,
# never worked around by falling back to NCCL. Note that the #631
# acceptance boots ran with SGLANG_BARLINK UNSET, i.e. on NCCL -- the
# barlink build-window lines in those logs come from build_window_enabled()
# which defaults True independently of the transport, so they do not
# evidence barlink. Production therefore runs a transport the acceptance
# ladder did not exercise; that is deliberate (parity with the previous
# production decode optimum) and is the reason BARLINK=0 must stay
# explicitly reachable for an A/B rather than by accident.
BARLINK="${BARLINK:-1}"
if [ "$BARLINK" = "1" ]; then
  if [ ! -e /dev/dmabuf_holder ]; then
    echo "REFUSE: SGLANG_BARLINK_TRANSPORT=bar1 requested but" >&2
    echo "/dev/dmabuf_holder is missing. Run /root/smallbar_reload.sh on the" >&2
    echo "Proxmox host (never silent-degrade to the device sub-transport)." >&2
    exit 1
  fi
  export SGLANG_BARLINK=1
  export SGLANG_BARLINK_TRANSPORT=bar1
  # Interim #603b mitigation: 5x collective deadline covers a full cold
  # flashinfer sampling JIT build. The abort check stays armed.
  export SGLANG_BARLINK_BAR1_CAP_CYCLES=300000000000

  # --- BAR1 aperture budget (REQUIRED with --enable-phase-flip) ------------
  # The phase flip DOUBLES the number of wire-carrying process groups: on
  # top of world:0 and pp:0 it builds flip_tp:0 and flip_dcp:0 (groups of
  # world_size 1 -- here tp:0/dcp:0 in the PP phase and flip_pp:0 in the TP
  # phase -- short-circuit and take no window). The stock per-group default
  # of 96 MiB was sized for the non-flip topology and does not fit four
  # groups into this card's aperture:
  #
  #   BAR1 free per NVML 248 MiB - reserve 32 MiB = 216 MiB usable, versus
  #   4 x 96 = 384 MiB requested. Measured consequence (boot 15:26:40Z):
  #   world:0 96, pp:0 96, flip_tp:0 clipped to 24, flip_dcp:0 clipped to
  #   0 -> Bar1Failed, all three ranks dead at group build.
  #
  # The 3080s' 256 MiB gross aperture is the binding constraint (window_for
  # takes the minimum across the group), not the 5090's.
  #
  # So the windows are budgeted by the payload each group actually carries,
  # rather than left on a one-size default:
  #   pp:0       chunked prefill activations, chunked_prefill_size 2048 x
  #              hidden x 2 B ~ 20 MiB per hop -- the LARGEST payload on the
  #              rig, and PP send/recv is the prefill hot path. Keeps the
  #              full 96 MiB (largest payload ~24 MiB at that window).
  #   flip_tp:0  TP decode all_reduce. Decode payloads are bs x hidden x
  #              2 B, i.e. tens of KiB at --max-running-requests 4; 48 MiB
  #              is already far above need and leaves room for a wider bs.
  #   flip_dcp:0 decode-phase KV all_to_all under the weighted owner rule,
  #              also decode-sized.
  #   world:0    control-plane broadcasts (default below).
  # Total 24 + 96 + 48 + 32 = 200 MiB against 216 MiB usable.
  #
  # NOTE these are EXPLICIT windows, so a shortfall REFUSES at group build
  # with the arithmetic in hand instead of silently clipping to a number
  # nobody chose. That is the intended contract; if a future topology adds
  # a group, this budget must be recomputed, not widened by reflex.
  export SGLANG_BARLINK_BAR1_WINDOW_MIB="${SGLANG_BARLINK_BAR1_WINDOW_MIB:-24}"
  export SGLANG_BARLINK_BAR1_WINDOW_MIB_PP_0="${SGLANG_BARLINK_BAR1_WINDOW_MIB_PP_0:-96}"
  export SGLANG_BARLINK_BAR1_WINDOW_MIB_FLIP_TP_0="${SGLANG_BARLINK_BAR1_WINDOW_MIB_FLIP_TP_0:-48}"
  export SGLANG_BARLINK_BAR1_WINDOW_MIB_FLIP_DCP_0="${SGLANG_BARLINK_BAR1_WINDOW_MIB_FLIP_DCP_0:-32}"
else
  export SGLANG_BARLINK=0
  unset SGLANG_BARLINK_TRANSPORT || true
  echo "NOTE: barlink DISABLED for this boot (NCCL transport), BARLINK=0." >&2
fi

# Census cadence back to the near-free default (the cadence-1 diagnostic
# window is closed, #650).
export SGLANG_COLLECTIVE_CENSUS_INTERVAL="${SGLANG_COLLECTIVE_CENSUS_INTERVAL:-50}"
# #581 standing falsifier: ack_write must stay ~0 in idle.
export SGLANG_MAMBA_PIN_TRACE="${SGLANG_MAMBA_PIN_TRACE:-50}"
# #605 VRAM flight-recorder marks are permanently armed (budgets computed,
# not guessed).
export SGLANG_VRAM_FLIGHT_DIR="${SGLANG_VRAM_FLIGHT_DIR:-/spinning/flight_605}"

# --- resolve cards by NAME -> UUID, 5090 FIRST -------------------------------
BIG_UUID=""; SMALL_UUID=()
while IFS=, read -r _idx name uuid _tot; do
    name="$(echo "$name" | xargs)"; uuid="$(echo "$uuid" | xargs)"
    case "$name" in *5090*) BIG_UUID="$uuid" ;; *3080*) SMALL_UUID+=("$uuid") ;; esac
done < <(nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader)
[[ -n "$BIG_UUID" && ${#SMALL_UUID[@]} -ge 2 ]] || {
    echo "FATAL: expected one 5090 and two 3080s from NVML" >&2; exit 1; }

# --- boot provenance ---------------------------------------------------------
# ROTATE, NEVER TRUNCATE. This was ": > $LOG", and it cost the boot-18
# diagnosis outright: the wedge was investigated from a py-spy that lived
# only in an agent's context, and the serving log that would have carried
# the presence/gate timeline was overwritten by the NEXT boot four minutes
# later. Rank 2's state is now permanently unknown, so the fix for that
# wedge had to rest on inference. Keeping the previous log costs a few MiB.
if [ -s "$LOG" ]; then
  PREV_STAMP="$(date -u -r "$LOG" '+%Y%m%dT%H%M%SZ' 2>/dev/null || date -u '+%Y%m%dT%H%M%SZ')"
  mv "$LOG" "${LOG%.log}.${PREV_STAMP}.log" 2>/dev/null || true
  # Keep the last 15 boots; that spans a full debugging session.
  ls -1t "${LOG%.log}".*.log 2>/dev/null | tail -n +16 | xargs -r rm -f
fi
: > "$LOG"
BOOT_COMMIT="$(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
BOOT_BRANCH="$(git -C "$WT" branch --show-current 2>/dev/null || echo detached)"
BOOT_DIRTY="$(git -C "$WT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
export SGLANG_BOOT_COMMIT="$BOOT_COMMIT"
{
  printf '=== BOOT PROVENANCE %s ===\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'tree=%s commit=%s branch=%s dirty_files=%s\n' \
         "$WT" "$BOOT_COMMIT" "$BOOT_BRANCH" "$BOOT_DIRTY"
  printf 'barlink=%s transport=%s rank_mib=%s ctx=%s max_running=%s\n' \
         "$BARLINK" "${SGLANG_BARLINK_TRANSPORT:-nccl}" "$RANK_MIB" "$CTX" "$MAX_RUNNING"
  printf 'cuda:0=5090=%s\ncuda:1=3080a=%s\ncuda:2=3080b=%s\n' \
         "$BIG_UUID" "${SMALL_UUID[0]}" "${SMALL_UUID[1]}"
} >> "$LOG"

cd "$WT"
CUDA_VISIBLE_DEVICES="$BIG_UUID,${SMALL_UUID[0]},${SMALL_UUID[1]}" \
setsid "$PY" -m sglang.launch_server \
    --model-path "$MODEL" --trust-remote-code \
    --served-model-name Qwen3.6-27B \
    --tp-size 1 --pp-size 3 \
    --pp-stage-ratio "$PP_STAGE_RATIO" \
    --rank-gpu-id 0,1,2 \
    --rank-gpu-memory-mib "$RANK_MIB" \
    --enable-phase-flip \
    --phase-flip-tp-vector 30,17,17 \
    --phase-flip-policy "$POLICY" \
    --phase-flip-purity "$PHASE_FLIP_PURITY" \
    --phase-flip-spill-depth "$PHASE_FLIP_SPILL_DEPTH" \
    --disable-overlap-schedule \
    --kv-cache-dtype fp8_e4m3 --context-length "$CTX" \
    `# CHUNK: MEASURED, not inherited (#656 item 8, 2026-08-11). The default` \
    `# 2048 was never chosen by anyone and was not even present in argv.` \
    `# Ladder at 22.8k prompts, one boot per arm, A-vs-A floor 0.2-2.6%:` \
    `#   256 2579 | 512 2666 | 1024 2597 | 2048 2533 | 4096 2377 |` \
    `#   8192 2156 tok/s | 16384 CUDA OOM + gloo death.` \
    `# 512 beats BOTH neighbours by more than the floor, so it is an` \
    `# interior optimum and not just the smallest value tried. Bigger buys` \
    `# activation memory with corridor headroom -- slower AND tighter, and` \
    `# fatal at 16384. At concurrency the memory term arrives sooner, so the` \
    `# safe direction from here is DOWN, never up.` \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --max-running-requests "$MAX_RUNNING" \
    `# The GDN/mamba state pool pins CONCURRENCY, not token capacity: the` \
    `# auto-sized pool gave 5 slots and the admission ratio (radix cache` \
    `# on) needs 5 per request, so max_running_requests was silently` \
    `# reduced 4 -> 1 and decode ran bs=1. Size it for the real bs.` \
    --max-mamba-cache-size "$MAMBA_SLOTS" \
    `# The TP layout can only ever address ids the scheduler's allocator` \
    `# issues, and that allocator is the PP stack's. Left uncapped the TP` \
    `# pool sizes itself to its own budget -- measured 788026 against an id` \
    `# space of 367704 -- and hoards the VRAM the PP pool needs. Capping` \
    `# costs nothing (the surplus was unaddressable) and is what let the id` \
    `# space grow. Keep it comfortably ABOVE the id space: the boot guard` \
    `# refuses a TP capacity below it.` \
    --max-total-tokens "$MAX_TOTAL_TOKENS" \
    $SPEC_FLAGS \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --chat-template-default-kwargs '{"preserve_thinking": true}' \
    --enable-cache-report \
    `# SPILL OFFLOAD (standard stack): the device KV pool is the working` \
    `# set, not the ceiling. HiCache keeps a host-RAM tier behind it at` \
    `# --hicache-ratio x the device pool, so KV pressure spills to host` \
    `# instead of evicting or refusing. The file/disk backend stays OFF:` \
    `# it crashed the 2026-08-05 proof run (rank desync on the DCP group` \
    `# via the storage-prefetch collective) and that root is not closed.` \
    $HICACHE_FLAGS \
    `# Graduated response to KV pressure rather than a cliff (opt-in:` \
    `# 'auto' cannot map ranks to cards on a mixed-model node).` \
    ${KV_LADDER:+--kv-pressure-ladder "$KV_LADDER"} \
    --enable-metrics --host "$HOST" --port "$PORT" \
    "$@" >> "$LOG" 2>&1 &
echo "serving pgid $!  log $LOG"
