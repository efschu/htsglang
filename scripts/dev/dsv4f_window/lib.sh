#!/usr/bin/env bash
# Shared helpers for the 2026-08-04 DSV4F window (#478 / #470 / #462).
#
# Sourced by every boot script in this directory. Defines functions only; it
# never boots anything and never sets -e on the caller.
#
# DESK-WRITTEN, NEVER EXECUTED against a GPU. Only `bash -n` has run.
#
# Ground rules encoded here (/spinning/htsglang/CLAUDE.md):
#   - bounded waits only (curl -m, capped loops), never an unbounded wait;
#   - own PIDs via pidfiles, py-spy dump before any kill, never pkill;
#   - device identity via NVML UUID / PCI-BDF, never a bare index;
#   - every boot carries --enable-metrics (asserted by `assert_metrics_flag`).
#
# ===========================================================================
# INDEX SPACES -- read this before touching any per-rank quantity.
# ===========================================================================
# NVML order and CUDA order are DIFFERENT on this rig, permanently, not as
# boot-to-boot drift. nvidia-smi/NVML enumerates by PCI bus; torch defaults to
# CUDA_DEVICE_ORDER=FASTEST_FIRST and puts the 5090 first:
#
#   NVML 0 = RTX 3080 (05:00.0)      CUDA 0 = RTX 5090 (0A:00.0)
#   NVML 1 = RTX 5090 (0A:00.0)      CUDA 1 = RTX 3080 (05:00.0)
#   NVML 2 = RTX 3080 (0B:00.0)      CUDA 2 = RTX 3080 (0B:00.0)
#
# Which space each flag takes:
#   --rank-gpu-id                CUDA ordinals (server_args.py:8476-8477,
#                                gpu_id_for_rank returns rank_gpu_id[world_rank])
#   --rank-auto-reserve-mib      positionally zipped against --rank-gpu-id
#                                (server_args.py:9111) -- i.e. per RANK
#   --rank-moe-resident-fraction per RANK, same positional zip
#   --speculative-draft-gpu      CUDA ordinals. server_args.py:3581-3589 says
#                                verbatim "CUDA device index (torch.cuda
#                                order, same space as --rank-gpu-id)".
#   nvidia-smi --query-gpu=...   NVML indices. NEVER line these up against a
#                                rank without going through the UUID bridge
#                                (`nvml_index_for_rank`).
#
# So the proven recipe's `--rank-gpu-id 0,1,2` is correct as written: rank 0
# -> cuda:0 -> the 5090, which is why rank 0 carries the largest reserve and
# resident fraction (and is the clock rank, #439). And the correct value for
# --speculative-draft-gpu on this rig is the 5090's CUDA ordinal (0), NOT its
# NVML index (1). Passing 1 does NOT error -- rank 1 legitimately maps to
# cuda:1 -- it silently places the DSpark draft head on a 3080, where the
# MXFP4 Marlin path does not exist (SM90/SM120 only). Silent wrongness, so it
# is asserted rather than assumed: see `assert_draft_gpu_is_5090`.

# ---------------------------------------------------------------------------
# Paths and defaults. All overridable from the environment.
# ---------------------------------------------------------------------------
WT="${WT:-/spinning/wt-dsv4f-window}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
PY="${PY:-$VENV/bin/python}"
RUN="${RUN:-/spinning/gpu-battery-results/2026-08-04_dsv4f_window}"
ARB="${ARB:-/spinning/gpu-arb}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
GGUF_ROOT="${GGUF_ROOT:-$MODEL_ROOT/DeepSeek-V4-Flash-0731-GGUF}"
DSPARK_HEAD="${DSPARK_HEAD:-$MODEL_ROOT/DeepSeek-V4-Flash-0731-dspark-head-filtered}"
SCRIPT_DIR="${SCRIPT_DIR:-$WT/scripts/dev/dsv4f_window}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/dsv4f_chat_template.jinja}"

# The window's arbitration identity. The heartbeat rewrites this LINE; never a
# bare `touch` (the reaper reads content, and a bare touch on a foreign holder
# would silently claim someone else's window).
ARB_SESSION="${ARB_SESSION:-agent-dsv4f-window}"

# MemAvailable floor. Derived, not desk-picked: the GGUF streaming loader is
# told to hold up to SGLANG_GGUF_STREAM_TRIM_SOFT_GIB (88) GiB of page cache
# and trim back to _TARGET_GIB (78); below SOFT + 8 GiB of headroom the load
# thrashes instead of streaming. There is no swap on this host, so a shortfall
# cannot be absorbed.
MEM_AVAIL_FLOOR_GIB="${MEM_AVAIL_FLOOR_GIB:-96}"

# #493 corridor rule: free >= 400 MiB absolute on ALL cards, judged at peak.
VRAM_FREE_FLOOR_MIB="${VRAM_FREE_FLOOR_MIB:-400}"

# ---------------------------------------------------------------------------
# Logging / failure
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die()  { printf 'REFUSED: %s\n' "$*" >&2; exit 1; }
utc()  { date -u +%Y-%m-%dT%H:%M:%SZ; }

# ---------------------------------------------------------------------------
# preflight -- every precondition, each failure naming the offending number.
#
# The corridor check is PER CARD, so it is correct in NVML space and needs no
# bridge. Anything PER RANK must go through `nvml_index_for_rank` instead.
# ---------------------------------------------------------------------------
preflight() {
    local arm="${1:?preflight needs an arm name}"
    mkdir -p "$RUN" || die "cannot create RUN dir $RUN"

    # --- swap must be 0 (an anon shortfall has nowhere to go) --------------
    local swap_kib
    swap_kib="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
    [ "$swap_kib" = "0" ] || die "SwapTotal is ${swap_kib} kB, expected 0. This host is configured swapless; a non-zero value means the machine is not the one these budgets were derived on."

    # --- MemAvailable floor ------------------------------------------------
    local avail_kib floor_kib avail_gib
    avail_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    floor_kib=$(( MEM_AVAIL_FLOOR_GIB * 1024 * 1024 ))
    avail_gib=$(( avail_kib / 1024 / 1024 ))
    [ "$avail_kib" -ge "$floor_kib" ] || die "MemAvailable is ${avail_gib} GiB, floor is ${MEM_AVAIL_FLOOR_GIB} GiB (SGLANG_GGUF_STREAM_TRIM_SOFT_GIB=88 + 8 GiB headroom, no swap on this host). Free page cache or lower MEM_AVAIL_FLOOR_GIB deliberately."

    # --- no compute process on ANY card -----------------------------------
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
    [ -z "$apps" ] || die "GPUs are NOT idle -- nvidia-smi reports compute PIDs: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'). This window requires exclusive cards. Never kill a foreign process; coordinate through $ARB/holder."

    # --- free VRAM corridor, per CARD, keyed by UUID (#493) ----------------
    local nvml_idx uuid name free
    while IFS=, read -r nvml_idx uuid name free; do
        nvml_idx="${nvml_idx// /}"; uuid="${uuid// /}"; free="${free// /}"
        name="$(printf '%s' "$name" | sed 's/^ *//; s/ *$//')"
        [ -n "$nvml_idx" ] || continue
        [ "$free" -ge "$VRAM_FREE_FLOOR_MIB" ] || die "card ${uuid} (${name}, NVML index ${nvml_idx}) has ${free} MiB free, corridor floor is ${VRAM_FREE_FLOOR_MIB} MiB (#493)."
        log "preflight: NVML ${nvml_idx} ${name} ${uuid} free=${free} MiB"
    done < <(nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader,nounits)

    # --- we must HOLD the arbitration ------------------------------------
    [ -e "$ARB/holder" ] || die "$ARB/holder does not exist -- this window never claimed the cards. Claim it (arb_claim) before any boot."
    grep -q "session=${ARB_SESSION}" "$ARB/holder" || die "$ARB/holder does not name session=${ARB_SESSION}; it currently reads: $(cat "$ARB/holder"). Another session holds the cards -- coordinate, do not displace."
    local age
    age=$(( $(date +%s) - $(stat -c %Y "$ARB/holder") ))
    [ "$age" -le 300 ] || die "$ARB/holder is ours but stale (${age}s > 300s) -- the heartbeat is dead, so the reaper may take the cards mid-arm. Restart arb_claim before booting."

    log "preflight OK for arm=${arm}: swap 0, MemAvailable ${avail_gib} GiB, no compute apps, corridor clear, holder fresh (${age}s)"
}

# ---------------------------------------------------------------------------
# Arbitration
# ---------------------------------------------------------------------------
ARB_HB_PID="${ARB_HB_PID:-$RUN/arb_heartbeat.pid}"
ARB_HB_STOP="${ARB_HB_STOP:-$RUN/arb_heartbeat.stop}"

arb_owner_line() {
    printf 'session=%s  cards=0,1,2  purpose=%s  since=%s\n' \
        "$ARB_SESSION" "${1:-DSV4F window 2026-08-04 (#478/#470/#462)}" "$(utc)"
}

arb_claim() {
    local purpose="${1:-DSV4F window 2026-08-04 (#478/#470/#462)}"
    mkdir -p "$RUN"
    # Preserve the incumbent SERVING holder line VERBATIM so restore_serving.sh
    # can put it back byte for byte.
    if [ -e "$ARB/holder" ] && [ ! -e "$RUN/holder_serving_original.txt" ]; then
        cp -a "$ARB/holder" "$RUN/holder_serving_original.txt"
        log "arb: incumbent holder preserved verbatim -> $RUN/holder_serving_original.txt"
    fi
    if [ -e "$ARB/holder" ] && ! grep -q "session=${ARB_SESSION}" "$ARB/holder"; then
        local age; age=$(( $(date +%s) - $(stat -c %Y "$ARB/holder") ))
        [ "$age" -gt 300 ] || die "arb holder is live (age ${age}s) and not ours: $(cat "$ARB/holder")"
        log "arb: foreign holder is stale (${age}s); taking over per the README orphan rule"
    fi
    arb_owner_line "$purpose" > "$ARB/holder"
    rm -f "$ARB_HB_STOP"
    local line; line="$(arb_owner_line "$purpose")"
    setsid bash -c 'while [ ! -e "$1" ]; do printf "%s\n" "$2" > "$3"; sleep 30; done' \
        _ "$ARB_HB_STOP" "$line" "$ARB/holder" </dev/null >/dev/null 2>&1 &
    echo $! > "$ARB_HB_PID"
    printf '%s  %s  ACQUIRE cards=0,1,2 -- %s\n' "$(utc)" "$ARB_SESSION" "$purpose" >> "$ARB/log"
    log "arb: claimed, heartbeat pid $(cat "$ARB_HB_PID")"
}

# Stop the heartbeat BEFORE any release. Standing rule, no exceptions.
arb_heartbeat_stop() {
    [ -e "$ARB_HB_PID" ] || { log "arb: no heartbeat pidfile, nothing to stop"; return 0; }
    : > "$ARB_HB_STOP"
    local pid i; pid="$(cat "$ARB_HB_PID")"
    for i in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || { log "arb: heartbeat stopped after ${i}s"; rm -f "$ARB_HB_PID"; return 0; }
        sleep 1
    done
    kill -TERM "$pid" 2>/dev/null || true
    rm -f "$ARB_HB_PID"
    log "arb: heartbeat TERMed after the 20s bounded wait"
}

# ---------------------------------------------------------------------------
# power_tag -- MANDATORY at the start AND end of every arm.
#
# The user lowered every card's power target on 2026-08-03, so every number
# measured after that carries its power state or it compares to nothing. Old
# full-power baselines are dead.
#
# DELIBERATE DEVIATION from the briefing: it asks for `powerstate_<arm>.json`
# written at start AND end. Writing one filename twice destroys the start
# reading -- the exact value the rule exists to preserve. So each call APPENDS
# one record to `powerstate_<arm>.jsonl` and rewrites `powerstate_<arm>.json`
# as the JSON array of all records. The named artifact still exists and
# nothing is lost.
# ---------------------------------------------------------------------------
power_tag() {
    local arm="${1:?power_tag needs an arm name}"
    local phase="${2:?power_tag needs a phase (start|end)}"
    mkdir -p "$RUN"
    nvidia-smi --query-gpu=index,name,uuid,power.limit,power.default_limit,power.max_limit,clocks.max.sm,temperature.gpu,memory.total \
        --format=csv 2>/dev/null \
    | PT_ARM="$arm" PT_PHASE="$phase" PT_UTC="$(utc)" \
      PT_JSONL="$RUN/powerstate_${arm}.jsonl" PT_JSON="$RUN/powerstate_${arm}.json" \
      "$PY" -c '
import csv, json, os, sys
rows = list(csv.DictReader(sys.stdin))
rec = {"arm": os.environ["PT_ARM"], "phase": os.environ["PT_PHASE"],
       "utc": os.environ["PT_UTC"],
       "index_space": "NVML (nvidia-smi). Not CUDA order -- see lib.sh header.",
       "gpus": [{k.strip(): (v.strip() if v is not None else None)
                 for k, v in r.items()} for r in rows]}
if not rec["gpus"]:
    print("power_tag: nvidia-smi returned no rows", file=sys.stderr)
    raise SystemExit(3)
with open(os.environ["PT_JSONL"], "a") as fh:
    fh.write(json.dumps(rec) + "\n")
recs = [json.loads(x) for x in open(os.environ["PT_JSONL"]) if x.strip()]
with open(os.environ["PT_JSON"], "w") as fh:
    json.dump(recs, fh, indent=1)
print("power_tag %s/%s: %d GPUs recorded" % (rec["arm"], rec["phase"], len(rec["gpus"])))
' || die "power_tag failed for arm=${arm} phase=${phase}; no arm may be quoted without its power state"
}

# ---------------------------------------------------------------------------
# resolve_cards -- BOTH orderings, bridged by PCI BDF / UUID, written to
# $RUN/device_order.json (and a per-arm copy) so every artifact in this window
# records which index space it used.
#
# Exports:
#   CARD_5090_NVML / CARD_5090_CUDA / CARD_5090_UUID / CARD_5090_TOTAL_MIB
#   DEVICE_ORDER_JSON (path)
#
# The CUDA side needs a CUDA context (registry/nvml.py:588-604:
# get_device_properties goes through _lazy_init and costs a few hundred MiB on
# every visible card). That is why this runs ONCE, in its own short-lived
# process, BEFORE the server boots -- never inside the launcher.
#
# The bridge is over the PCI bus id, never over index equality between pynvml
# and torch (registry/nvml.py:590-591: "Bridged over the bus id, never over
# the index: that identity is the whole point"). torch's own
# get_device_properties().uuid / .name are recorded alongside as an
# independent second reading, and disagreement is a hard error.
# ---------------------------------------------------------------------------
resolve_cards() {
    local arm="${1:?resolve_cards needs an arm name}"
    mkdir -p "$RUN"
    DEVICE_ORDER_JSON="$RUN/device_order.json"
    export DEVICE_ORDER_JSON
    PYTHONPATH="$WT/python" DEV_OUT="$DEVICE_ORDER_JSON" ARM="$arm" "$PY" - <<'PYEOF' \
        || die "card identity could not be resolved -- refusing to boot on a guessed index"
import json, os, sys

from sglang.srt.registry import nvml

imap = nvml.identity_map(allow_cuda_init=True)
cards = [
    {
        "uuid": c.uuid,
        "nvml_index": c.nvml_index,
        "cuda_ordinal": c.cuda_ordinal,
        "pci_bus_id": c.pci_bus_id,
        "name": c.name,
        "total_mib": int(c.total_bytes) // (1024 * 1024),
    }
    for c in imap
]

# Independent second reading straight from torch, so the bridge is checked
# rather than trusted. A disagreement here is exactly the silent defect this
# whole helper exists to prevent.
torch_view = []
try:
    import torch

    for ordinal in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(ordinal)
        torch_view.append(
            {
                "cuda_ordinal": ordinal,
                "name": props.name,
                "uuid": str(getattr(props, "uuid", "")) or None,
            }
        )
except Exception as exc:  # noqa: BLE001 - recorded, then judged below
    torch_view = [{"error": str(exc)}]

by_cuda = {c["cuda_ordinal"]: c for c in cards if c["cuda_ordinal"] is not None}
for row in torch_view:
    if "error" in row:
        continue
    bridged = by_cuda.get(row["cuda_ordinal"])
    if bridged is None:
        print(
            f"torch reports cuda:{row['cuda_ordinal']} ({row['name']}) but the "
            f"NVML/PCI bridge has no card at that ordinal",
            file=sys.stderr,
        )
        raise SystemExit(4)
    if bridged["name"] != row["name"]:
        print(
            f"BRIDGE DISAGREEMENT at cuda:{row['cuda_ordinal']}: NVML/PCI says "
            f"{bridged['name']!r}, torch says {row['name']!r}",
            file=sys.stderr,
        )
        raise SystemExit(5)
    if row["uuid"] and row["uuid"].replace("GPU-", "") not in bridged["uuid"]:
        print(
            f"BRIDGE DISAGREEMENT at cuda:{row['cuda_ordinal']}: NVML uuid "
            f"{bridged['uuid']!r}, torch uuid {row['uuid']!r}",
            file=sys.stderr,
        )
        raise SystemExit(6)

five = [c for c in cards if "5090" in c["name"]]
if len(five) != 1:
    print(f"expected exactly one 5090, found {[c['name'] for c in cards]}", file=sys.stderr)
    raise SystemExit(2)
f = five[0]
if f["cuda_ordinal"] is None:
    print(
        "the 5090's CUDA ordinal is unresolved; --speculative-draft-gpu and "
        "--rank-gpu-id both take CUDA ordinals, so a guess is not acceptable",
        file=sys.stderr,
    )
    raise SystemExit(3)

doc = {
    "arm": os.environ["ARM"],
    "note": (
        "NVML index != CUDA ordinal on this rig, permanently. --rank-gpu-id, "
        "--speculative-draft-gpu and every per-rank vector live in CUDA space; "
        "nvidia-smi lives in NVML space. Bridge by uuid/pci_bus_id, never by "
        "index equality."
    ),
    "cards": cards,
    "torch_view": torch_view,
    "nvml_to_cuda": {str(c["nvml_index"]): c["cuda_ordinal"] for c in cards},
    "cuda_to_nvml": {
        str(c["cuda_ordinal"]): c["nvml_index"] for c in cards if c["cuda_ordinal"] is not None
    },
    "five090": f,
}
with open(os.environ["DEV_OUT"], "w") as fh:
    json.dump(doc, fh, indent=1)
PYEOF

    cp -a "$DEVICE_ORDER_JSON" "$RUN/device_order_${arm}.json"

    eval "$(DEV_OUT="$DEVICE_ORDER_JSON" "$PY" -c '
import json, os
d = json.load(open(os.environ["DEV_OUT"]))
f = d["five090"]
print("export CARD_5090_NVML=%d" % f["nvml_index"])
print("export CARD_5090_CUDA=%d" % f["cuda_ordinal"])
print("export CARD_5090_UUID=%s" % f["uuid"])
print("export CARD_5090_TOTAL_MIB=%d" % f["total_mib"])
')"

    log "cards: 5090 -> CUDA ${CARD_5090_CUDA} / NVML ${CARD_5090_NVML} / ${CARD_5090_TOTAL_MIB} MiB (${CARD_5090_UUID})"
    log "cards: both orderings recorded in $DEVICE_ORDER_JSON"
}

# NVML index of the card a given RANK runs on. The ONLY sanctioned way to line
# a per-rank quantity up against an nvidia-smi reading.
nvml_index_for_rank() {
    local rank="${1:?nvml_index_for_rank needs a rank}"
    [ -n "${DEVICE_ORDER_JSON:-}" ] || die "nvml_index_for_rank called before resolve_cards"
    DEV_OUT="$DEVICE_ORDER_JSON" RGI="$RANK_GPU_ID" RANK="$rank" "$PY" -c '
import json, os, sys
d = json.load(open(os.environ["DEV_OUT"]))
cuda = int(os.environ["RGI"].split(",")[int(os.environ["RANK"])])
nvml = d["cuda_to_nvml"].get(str(cuda))
if nvml is None:
    print(f"no NVML index for cuda ordinal {cuda}", file=sys.stderr)
    raise SystemExit(2)
print(nvml)
'
}

# Free MiB on the card a given RANK runs on, read through the UUID bridge.
vram_free_mib_for_rank() {
    local rank="${1:?vram_free_mib_for_rank needs a rank}"
    local nvml_idx; nvml_idx="$(nvml_index_for_rank "$rank")" || return 1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$nvml_idx"
}

# The recipe's per-rank vectors put the largest reserve and resident fraction
# on rank 0 because rank 0 is meant to be the 5090 (also the clock rank,
# #439). If rank 0 is not the 5090 the vectors are transposed and the boot is
# measuring something else.
assert_rank0_is_5090() {
    local rank_gpu_id="${1:?assert_rank0_is_5090 needs the --rank-gpu-id string}"
    local first="${rank_gpu_id%%,*}"
    [ "$first" = "$CARD_5090_CUDA" ] || die "rank 0 maps to CUDA ordinal ${first} but the 5090 is CUDA ordinal ${CARD_5090_CUDA}. The per-rank reserve/resident vectors give rank 0 the biggest budget because rank 0 is meant to be the 5090; booting like this silently transposes them. RANK_GPU_ID is in CUDA-ordinal space (server_args.py:8476-8477)."
}

# The DSpark solo draft head MUST land on the 5090: the MXFP4 Marlin path is
# SM90/SM120 only (ANALYSE_447 §1.5), so a 3080 would either refuse by name or
# silently run the wrong kernel. Passing an NVML index here does NOT error --
# NVML 1 is a legal CUDA ordinal too -- so it must be asserted.
assert_draft_gpu_is_5090() {
    local draft_gpu="${1:?assert_draft_gpu_is_5090 needs the --speculative-draft-gpu value}"
    [ -n "${DEVICE_ORDER_JSON:-}" ] || die "assert_draft_gpu_is_5090 called before resolve_cards"
    DEV_OUT="$DEVICE_ORDER_JSON" DG="$draft_gpu" "$PY" -c '
import json, os, sys
d = json.load(open(os.environ["DEV_OUT"]))
want = int(os.environ["DG"])
hit = [c for c in d["cards"] if c["cuda_ordinal"] == want]
def table():
    return "\n".join(
        "  cuda:%s  nvml:%s  %s  %s" % (c["cuda_ordinal"], c["nvml_index"], c["name"], c["uuid"])
        for c in d["cards"]
    )
if not hit:
    print("--speculative-draft-gpu=%d names no CUDA ordinal on this host.\n"
          "Both orderings:\n%s" % (want, table()), file=sys.stderr)
    raise SystemExit(2)
card = hit[0]
if "5090" not in card["name"]:
    print("--speculative-draft-gpu=%d is CUDA ordinal %d = %s (NVML index %s).\n"
          "The DSpark solo head must run on the RTX 5090: the MXFP4 Marlin path "
          "is SM90/SM120 only. This is the NVML-vs-CUDA index confusion -- the "
          "flag takes a CUDA ordinal (server_args.py:3581-3589).\n"
          "Both orderings:\n%s" % (want, want, card["name"], card["nvml_index"], table()),
          file=sys.stderr)
    raise SystemExit(3)
print("draft-gpu check OK: cuda:%d = %s (nvml:%s)" % (want, card["name"], card["nvml_index"]))
' || die "the resolved --speculative-draft-gpu does not name the RTX 5090"
}

# ---------------------------------------------------------------------------
# assert_metrics_flag -- every boot carries --enable-metrics. No exceptions.
# ---------------------------------------------------------------------------
assert_metrics_flag() {
    case " $* " in
        *" --enable-metrics "*) return 0 ;;
        *) die "the assembled boot line carries no --enable-metrics. Standing user order: EVERY server boot carries it." ;;
    esac
}

# ---------------------------------------------------------------------------
# wait_ready -- bounded poll of /health_generate. NEVER an unbounded wait.
#
# Sizing: DSV4F weight load alone measured 239-240 s and launch->ready ~5.5-6
# min for the 98 GiB IQ3_XXS stream. UD-Q3_K_XL measures 120 GiB (+22 GiB), so
# the Q3_K_XL arm raises the ceiling from its own script.
# ---------------------------------------------------------------------------
wait_ready() {
    local arm="${1:?wait_ready needs an arm name}"
    local port="${2:?wait_ready needs a port}"
    local max_iters="${3:-90}"     # 90 x 10s = 15 min
    local pidfile="$RUN/${arm}.pid"
    local t0 i now
    t0="$(date +%s)"
    for i in $(seq 1 "$max_iters"); do
        # MUST check the HTTP STATUS, not curl's exit code. curl exits 0 for
        # 4xx/5xx, and sglang binds the port and answers 503 while the engine
        # is still initialising -- so an exit-code test declares readiness the
        # moment the socket is up. That cost two full ~6 minute loads in this
        # window: the arm was declared ready, every probe was refused against a
        # still-loading server, and the boot tore itself down having measured
        # nothing. Require a literal 200, and require it from /health (cheap)
        # rather than /health_generate (runs a real generation, and this server
        # boots with --max-running-requests 1).
        code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' \
                 "http://127.0.0.1:${port}/health" 2>/dev/null || echo 000)"
        if [ "$code" = "200" ]; then
            now=$(( $(date +%s) - t0 ))
            printf 'arm=%s port=%s ready_after_s=%d iters=%d utc=%s\n' \
                "$arm" "$port" "$now" "$i" "$(utc)" | tee "$RUN/ready_${arm}.txt"
            return 0
        fi
        if [ -e "$pidfile" ] && ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
            now=$(( $(date +%s) - t0 ))
            printf 'arm=%s port=%s DIED_AFTER_S=%d\n' "$arm" "$port" "$now" | tee "$RUN/ready_${arm}.txt"
            log "server died before ready; last 40 log lines:"
            tail -40 "$RUN/boot_${arm}.log" >&2 2>/dev/null || true
            return 2
        fi
        sleep 10
    done
    now=$(( $(date +%s) - t0 ))
    printf 'arm=%s port=%s TIMEOUT_AFTER_S=%d iters=%d\n' "$arm" "$port" "$now" "$max_iters" | tee "$RUN/ready_${arm}.txt"
    log "TIMEOUT waiting for /health_generate; last 40 log lines:"
    tail -40 "$RUN/boot_${arm}.log" >&2 2>/dev/null || true
    return 3
}

# ---------------------------------------------------------------------------
# record_pids -- store pid AND pgid so teardown addresses the group without
# ever touching a foreign process. `setsid` makes the child a session leader,
# so pgid == pid; it is read back rather than assumed.
# ---------------------------------------------------------------------------
record_pids() {
    local arm="${1:?record_pids needs an arm name}"
    local pid="${2:?record_pids needs a pid}"
    echo "$pid" > "$RUN/${arm}.pid"
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    [ -n "$pgid" ] || pgid="$pid"
    echo "$pgid" > "$RUN/${arm}.pgid"
    log "arm=${arm} pid=${pid} pgid=${pgid}"
}

# ---------------------------------------------------------------------------
# stop_server -- py-spy FIRST (a wedge is evidence; killing destroys it), then
# TERM the recorded process GROUP only, bounded wait, then KILL. Never pkill,
# never a pattern.
# ---------------------------------------------------------------------------
stop_server() {
    local arm="${1:?stop_server needs an arm name}"
    local pidfile="$RUN/${arm}.pid"
    local pgidfile="$RUN/${arm}.pgid"
    [ -e "$pidfile" ] || { log "stop_server: no pidfile for ${arm}, nothing to stop"; return 0; }
    local pid pgid i child
    pid="$(cat "$pidfile")"
    pgid="$(cat "$pgidfile" 2>/dev/null || echo "$pid")"

    if kill -0 "$pid" 2>/dev/null; then
        log "stop_server: py-spy dump of pid ${pid} BEFORE any signal"
        "$VENV/bin/py-spy" dump --pid "$pid" > "$RUN/pyspy_${arm}.txt" 2>&1 || \
            log "py-spy dump of the launcher failed (recorded in $RUN/pyspy_${arm}.txt)"
        # The scheduler workers carry the interesting frames; dump them too.
        for child in $(pgrep -P "$pid" 2>/dev/null); do
            printf '\n===== child pid %s =====\n' "$child" >> "$RUN/pyspy_${arm}.txt"
            "$VENV/bin/py-spy" dump --pid "$child" >> "$RUN/pyspy_${arm}.txt" 2>&1 || true
        done
    else
        log "stop_server: pid ${pid} already gone; no py-spy dump possible"
    fi

    kill -TERM -"$pgid" 2>/dev/null || log "stop_server: TERM to pgid ${pgid} found nothing"
    for i in $(seq 1 60); do          # 60 x 2s = 2 min bounded
        kill -0 "$pid" 2>/dev/null || { log "stop_server: ${arm} down after $((i*2))s"; return 0; }
        sleep 2
    done
    log "stop_server: ${arm} still alive after 120s, sending KILL to pgid ${pgid}"
    kill -KILL -"$pgid" 2>/dev/null || true
    for i in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || { log "stop_server: ${arm} killed"; return 0; }
        sleep 2
    done
    log "stop_server: ${arm} SURVIVED KILL -- report it, do not escalate to a broad kill"
    return 4
}

# ---------------------------------------------------------------------------
# rammon -- anon vs file, separately, every 15 s.
#
# This split IS the feasibility argument: there is no swap on this host, so
# cgroup reclaim can only take page cache (`file`). `anon` is the structurally
# unreclaimable term -- if anon alone approaches MemTotal the arm cannot be
# rescued by dropping caches, and that is a different failure from a merely
# large current+cache figure.
# ---------------------------------------------------------------------------
rammon_start() {
    local arm="${1:?rammon_start needs an arm name}"
    local interval="${2:-15}"
    mkdir -p "$RUN"
    local out="$RUN/ram_${arm}.log"
    local stopfile="$RUN/ram_${arm}.stop"
    rm -f "$stopfile"
    printf 'utc\tmemory_current_bytes\tanon_bytes\tfile_bytes\tmem_available_kib\n' > "$out"
    setsid bash -c '
      stopfile="$1"; out="$2"; interval="$3"
      while [ ! -e "$stopfile" ]; do
        cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo -1)
        anon=$(awk "/^anon /{print \$2}" /sys/fs/cgroup/memory.stat 2>/dev/null)
        fpc=$(awk "/^file /{print \$2}" /sys/fs/cgroup/memory.stat 2>/dev/null)
        avail=$(awk "/^MemAvailable:/{print \$2}" /proc/meminfo)
        printf "%s\t%s\t%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
               "${cur:--1}" "${anon:--1}" "${fpc:--1}" "${avail:--1}" >> "$out"
        sleep "$interval"
      done' _ "$stopfile" "$out" "$interval" </dev/null >/dev/null 2>&1 &
    echo $! > "$RUN/ram_${arm}.pid"
    log "rammon started for ${arm} (pid $(cat "$RUN/ram_${arm}.pid"), every ${interval}s) -> $out"
}

rammon_stop() {
    local arm="${1:?rammon_stop needs an arm name}"
    local pidfile="$RUN/ram_${arm}.pid"
    [ -e "$pidfile" ] || { log "rammon_stop: no sampler for ${arm}"; return 0; }
    : > "$RUN/ram_${arm}.stop"
    local pid i; pid="$(cat "$pidfile")"
    for i in $(seq 1 25); do
        kill -0 "$pid" 2>/dev/null || { rm -f "$pidfile"; log "rammon stopped for ${arm}"; return 0; }
        sleep 1
    done
    kill -TERM "$pid" 2>/dev/null || true
    rm -f "$pidfile"
    log "rammon TERMed for ${arm} after the bounded wait"
}

# ---------------------------------------------------------------------------
# Log assertions used by the first-boot checks.
# ---------------------------------------------------------------------------
assert_log_contains() {
    local logfile="${1:?}"; local needle="${2:?}"; local why="${3:-}"
    grep -qF -- "$needle" "$logfile" \
        || die "boot log ${logfile} does not contain '${needle}'. ${why}"
    log "log check OK: '${needle}' present"
}

assert_log_absent() {
    local logfile="${1:?}"; local needle="${2:?}"; local why="${3:-}"
    if grep -qF -- "$needle" "$logfile"; then
        die "boot log ${logfile} contains '${needle}', which must not appear. ${why}"
    fi
    log "log check OK: '${needle}' absent"
}

count_log() {
    grep -cF -- "${2:?}" "${1:?}" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# The base env every arm shares. Verbatim from
# /spinning/gpu-battery-results/2026-08-02_394_linkshards/boot394.sh -- the
# ONLY recipe that has ever served DSV4F on this rig.
#
# SGLANG_EXPERT_STATS=1 is armed in EVERY arm because it is free: arm 4
# (#390/#394 expert statistics) is harvested from the other three boots rather
# than costing a boot of its own.
# ---------------------------------------------------------------------------
export_base_env() {
    local arm="${1:?export_base_env needs an arm name}"
    export PYTHONPATH="$WT/python"
    export SGLANG_MOE_SCRATCH_SLOTS="${SGLANG_MOE_SCRATCH_SLOTS:-6}"   # measured routed top-k is exactly 6
    export SGLANG_FORWARD_PEAK_PATH="$RUN/peak_$arm"
    export SGLANG_GGUF_STREAM_TRIM_SOFT_GIB="${SGLANG_GGUF_STREAM_TRIM_SOFT_GIB:-88}"
    export SGLANG_GGUF_STREAM_TRIM_TARGET_GIB="${SGLANG_GGUF_STREAM_TRIM_TARGET_GIB:-78}"
    export SGLANG_DSV4_FP4_EXPERTS=0
    export SGLANG_EXPERT_STATS=1
    export SGLANG_EXPERT_STATS_PATH="$RUN/expert_stats_$arm"
    export SGLANG_EXPERT_STATS_INTERVAL_SEC="${SGLANG_EXPERT_STATS_INTERVAL_SEC:-45}"
    export SGLANG_MOE_STAGING_TRACE=1
    export SGLANG_OPT_FUSE_WQA_WKV=0
    export SGLANG_OPT_USE_TOPK_V2=0
    # Refuted / out-of-scope switches, unset explicitly so an inherited
    # environment cannot smuggle them in (TICKET_462 §1).
    unset SGLANG_MOE_COLD_TIER_SHM || true
    unset SGLANG_MOE_OFFLOAD_CUDA_GRAPH || true
    unset SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE || true
    unset SGLANG_MOE_HOT_RESIDENCY || true
}

# The two per-rank vectors are env-overridable: the operator is recomputing
# them from a GGUF footprint analysis, so the boot394 values are DEFAULTS ONLY.
RESIDENT_FRACTION="${RESIDENT_FRACTION:-0.485,0.42,0.42}"
AUTO_RESERVE_MIB="${AUTO_RESERVE_MIB:-2200,1400,1400}"
RANK_GPU_ID="${RANK_GPU_ID:-0,1,2}"     # CUDA ordinals. rank0 -> cuda:0 -> 5090.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-512}"
MAX_RUNNING="${MAX_RUNNING:-1}"

# The parsers. VERIFIED registered in this tree:
#   deepseek-v4 -> parser/reasoning_parser.py:1132 (DetectorMap entry)
#   deepseekv4  -> function_call/function_call_parser.py:66 (ToolCallParserEnum)
REASONING_PARSER="${REASONING_PARSER:-deepseek-v4}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseekv4}"

assert_chat_template() {
    [ -s "$CHAT_TEMPLATE" ] || die "chat template $CHAT_TEMPLATE is missing or empty. Regenerate it: $PY $SCRIPT_DIR/extract_chat_template.py --write"
    "$PY" "$SCRIPT_DIR/extract_chat_template.py" --verify --template "$CHAT_TEMPLATE" \
        || die "chat template $CHAT_TEMPLATE does not match the GGUF metadata it was extracted from"
}
