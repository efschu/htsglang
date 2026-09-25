#!/usr/bin/env bash
# htsglang Release-Entrypoint (Upgrade des August-Containers) -- ENTWURF (27B-Sitz R, 24.09.2026).
# NICHT GEBAUT, NICHT GELAUFEN. Nur `bash -n` geprueft.
#
# DISPATCHER. MODE=server|planner (oder MODE ungesetzt) -> unveraendert das August-Entrypoint
# /usr/local/bin/htsglang-entrypoint.sh (docker/htsglang-entrypoint.sh, env-getriebenes
# launch_server bzw. Planner). Bestehende `docker run`-Zeilen des veroeffentlichten Containers
# laufen also weiter wie bisher. NEU: MODE=weg2 -> der weg2-P/D-Flip-Launcher mit Profil.
#
# weg2-Ablauf:
#  1. Profil laden (HTSGLANG_PROFILE: 27b, 27b-fp8, 27b-nvfp4, 27b-gguf | nf, nf-nvfp4, nf-gguf);
#     Profil-Linie == Image-Linie; Platzhalter-Profile verweigern mit Verweis auf den Eigentuemer.
#  2. Modellformat am gemounteten Modell erkennen und gegen PROFILE_FORMAT pruefen.
#  3. Umgebung: FLASHINFER_CUDA_ARCH_LIST und TORCH_CUDA_ARCH_LIST weg (JIT-Vorbau-Namen),
#     expandable_segments verweigert (launcher.py:5515).
#  4. Preflight: GPU-Inventar, /dev/shm, Speicher, Baum-Identitaet, Modelle, Zustands-Dirs
#     (Stufe B: SGLANG_WEG2_*), ARB-Saat, tmpfs-Store (NF), BAR1-Kette.
#  5. Transport (Operator-Entscheid F7, 24.09.): bar1 (DEFAULT) | nccl (nur ausdruecklich).
#     Kein stilles Umschalten und kein `auto`: bar1 ohne vollstaendige Kette = Verweigerung mit
#     Grund; nccl = barlink KOMPLETT aus (Launcher --transport nccl streicht die barlink-Flags
#     beider Gruppen, hier zusaetzlich SGLANG_BARLINK/_TRANSPORT/_PP_TRANSPORT aus der Umgebung).
#     Beide Transporte muessen in der Host-Abnahme booten (host_acceptance.sh serve bar1|nccl).
#  Instrumente (F8): HTSGLANG_INSTRUMENTS=0 ist der Release-Default; =1 schaltet die
#     Mess-Env des Profils zu (Paritaet mit den Rig-Referenzboots, Abnahme).
#  6. Launcher starten; er kehrt nach "LAUNCHED" mit rc 0 zurueck (launcher.py:14068-14069) ->
#     dieser Prozess beaufsichtigt die Front und baut ueber `--teardown <state.json>` ab.
#
# weg2-Untermodi (erstes Argument oder HTSGLANG_MODE): serve (Default) | dryrun | preflight |
#   selfcheck | version.  Weitere Argumente gehen unveraendert an den Launcher.

set -Eeuo pipefail
umask 022

if [ "${MODE:-server}" != "weg2" ]; then
  exec /usr/local/bin/htsglang-entrypoint.sh "$@"
fi

HOME_DIR=/opt/htsglang
VENV=${SGLANG_WEG2_VENV:-/opt/venv}
PY=$VENV/bin/python
TREE=${HTSGLANG_TREE:-/opt/htsglang/src}
PROFILE_DIR=$HOME_DIR/profiles
BUILD_INFO=$HOME_DIR/BUILD_INFO.json
JIT_REPORT=$HOME_DIR/JIT_PREBUILD.json
ARB_SEED=$HOME_DIR/arb-seed
# Stufe B (Commit bae049a3b4): der Launcher liest diese Pfade aus der Umgebung.
export SGLANG_WEG2_EVIDENCE_DIR=${SGLANG_WEG2_EVIDENCE_DIR:-/var/lib/htsglang/evidence}
export SGLANG_WEG2_GPU_ARB=${SGLANG_WEG2_GPU_ARB:-/var/lib/htsglang/arb}
export SGLANG_WEG2_DEVTOOLS_DIR=${SGLANG_WEG2_DEVTOOLS_DIR:-/opt/htsglang/devtools}
export SGLANG_WEG2_STORE_ROOT=${SGLANG_WEG2_STORE_ROOT:-/var/lib/htsglang/hicache-weg2}
export SGLANG_WEG2_VENV=$VENV
export SGLANG_WEG2_TMS_OUT_DIR=${SGLANG_WEG2_TMS_OUT_DIR:-/opt/htsglang/tms}
# Rang-seitige Diagnose-Ziele, die sonst auf Rig-Pfade zeigen (barlink_abort_gate.py:679,
# barlink_capture_census.py:111-115).
export SGLANG_WEG2_PYSPY_DIR=${SGLANG_WEG2_PYSPY_DIR:-$SGLANG_WEG2_EVIDENCE_DIR}
export SGLANG_BARLINK_CAPTURE_CENSUS_DIR=${SGLANG_BARLINK_CAPTURE_CENSUS_DIR:-$SGLANG_WEG2_EVIDENCE_DIR/capture_census}
EVIDENCE_DIR=$SGLANG_WEG2_EVIDENCE_DIR
GPU_ARB=$SGLANG_WEG2_GPU_ARB
STORE_ROOT=$SGLANG_WEG2_STORE_ROOT
FRONT_PORT=30030

say()    { printf '[htsglang-weg2 %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
refuse() { say "REFUSED $1: $2"; exit 3; }
usage()  { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; }

# --- Untermodus -----------------------------------------------------------------
SUB=${HTSGLANG_MODE:-serve}
if [ $# -gt 0 ]; then
  case "$1" in
    serve|dryrun|preflight|selfcheck|version) SUB=$1; shift ;;
    bash|sh|/bin/bash|/bin/sh) exec "$@" ;;
    python|python3) shift; exec "$PY" "$@" ;;
    -h|--help|help) usage; exit 0 ;;
  esac
fi

if [ "$SUB" = "version" ]; then
  cat "$BUILD_INFO" 2>/dev/null || echo '{"BUILD_INFO": "fehlt"}'
  jq -c '{verdict, archs, flashinfer: [.flashinfer[]? | {dirkey, n: (.rows|length), built: ([.rows[]?|select(.status=="built")]|length)}], barlink: [.barlink.barlink[]? | {ext, status: .status[0:60]}], cpu_ext: [.cpu_ext.cpu_ext[]? | {ext, status: .status[0:60]}], tms: .tms.so, stages: [.stages[]?.only], problems}' "$JIT_REPORT" 2>/dev/null || true
  exit 0
fi

# --- 1. Profil -------------------------------------------------------------------
: "${HTSGLANG_PROFILE:=${HTSGLANG_LINE:-27b}}"
PROFILE_FILE=$PROFILE_DIR/${HTSGLANG_PROFILE}.env
[ -f "$PROFILE_FILE" ] || refuse PROFILE "unbekanntes Profil '$HTSGLANG_PROFILE' (vorhanden: $(cd "$PROFILE_DIR" && ls ./*.env 2>/dev/null | sed 's#^\./##; s/\.env$//' | tr '\n' ' '))"
: "${HTSGLANG_TAG:=dkr${HTSGLANG_PROFILE//-/}$(date -u +%m%d%H%M%S)}"
export HTSGLANG_TAG
# F8: Instrument-Env im Release AUS, schaltbar. VOR dem Profil gesetzt, weil Profile den Schalter
# schon beim Sourcen lesen (nf.env: NF_ENV_*_INSTR in --env-p/--env-d).
: "${HTSGLANG_INSTRUMENTS:=0}"
case "$HTSGLANG_INSTRUMENTS" in 0|1) ;; *) refuse INSTRUMENTS "HTSGLANG_INSTRUMENTS='$HTSGLANG_INSTRUMENTS' (0|1)" ;; esac
export HTSGLANG_INSTRUMENTS
# shellcheck source=/dev/null
source "$PROFILE_FILE"
[ "${PROFILE_LINE:-}" = "${HTSGLANG_LINE:-}" ] || refuse PROFILE "Profil-Linie '${PROFILE_LINE:-?}' != Image-Linie '${HTSGLANG_LINE:-?}' (27B und NF strikt getrennt: ein Image traegt genau eine Linie)"
# Profil-Stand (Release 25.09.): abgenommen = laeuft; experimentell = laeuft nur mit HTSGLANG_ALLOW_EXPERIMENTAL=1 und
# lauter Warnung; vorbereitet / geplant / Platzhalter = verweigert mit Grund. Altprofile ohne PROFILE_STATUS gelten als
# abgenommen, solange sie nicht PROFILE_PLACEHOLDER=1 tragen.
_pstat=${PROFILE_STATUS:-}
if [ -z "$_pstat" ]; then _pstat=abgenommen; [ "${PROFILE_PLACEHOLDER:-0}" = "1" ] && _pstat=platzhalter; fi
case "$_pstat" in
  abgenommen) ;;
  experimentell)
    [ "${HTSGLANG_ALLOW_EXPERIMENTAL:-0}" = "1" ] \
      || refuse PROFILE "Profil '$HTSGLANG_PROFILE' ist EXPERIMENTELL (${PROFILE_OWNER:-?}) -- nur mit HTSGLANG_ALLOW_EXPERIMENTAL=1"
    say "WARN: Profil '$HTSGLANG_PROFILE' ist EXPERIMENTELL (${PROFILE_OWNER:-?}) -- nicht abgenommen, auf eigenes Risiko" ;;
  vorbereitet) refuse PROFILE "Profil '$HTSGLANG_PROFILE' ist VORBEREITET, nicht abgenommen (${PROFILE_OWNER:-?})" ;;
  geplant) refuse PROFILE "Profil '$HTSGLANG_PROFILE' ist GEPLANT (${PROFILE_OWNER:-?}) -- Form noch nicht geliefert" ;;
  *) refuse PROFILE "Profil '$HTSGLANG_PROFILE' ist ein PLATZHALTER (${PROFILE_OWNER:-Eigentuemer offen}) -- Form noch nicht geliefert" ;;
esac
[ "${#PROFILE_ARGS[@]}" -gt 0 ] || refuse PROFILE "Profil '$HTSGLANG_PROFILE' hat keine Launcher-Argumente (Form nicht geliefert)"

_form() {   # _form VAR BESTFORM -- setzt Bestform, meldet eine explizite Abweichung laut
  local var=$1 best=$2
  if [ -n "${!var+x}" ] && [ "${!var}" != "$best" ]; then
    say "FORM-OVERRIDE $var=${!var} (Bestform $best) -- per docker run -e gesetzt"
  else
    export "$var=$best"
  fi
}
profile_form_env
if [ "$HTSGLANG_INSTRUMENTS" = "1" ]; then profile_instr_env; say "Instrumente AN (HTSGLANG_INSTRUMENTS=1)"; fi

# --- 2. Modellformat ---------------------------------------------------------------
detect_format() {   # detect_format <modellpfad> -> int8|int4|int4-mixed|fp8|nvfp4-modelopt|nvfp4-ct|gguf|bf16|...
  # Am Rig geprueft (24.09.): INT8-gdncov-vocabembed=int8, Flash-Next-Minachist=int4-mixed, 27B-FP8=fp8,
  # 27B-NVFP4-RadixArk und Flash-Next-NVFP4-nvidia=nvfp4-modelopt, 27B-NVFP4 (unsloth)=nvfp4-ct,
  # 27B-GGUF-unsloth (Verzeichnis MIT config.json) und Flash-Next-GGUF (gguf nur in Unterordnern)=gguf.
  "$PY" - "$1" <<'EOF'
import glob, json, os, sys
p = sys.argv[1]
if os.path.isfile(p) and p.endswith(".gguf"):
    print("gguf"); sys.exit()
# GGUF-Verzeichnis: *.gguf (Ebene 0-2), keine *.safetensors. Eine config.json darf daneben liegen
# (unsloth legt die HF-Konfiguration fuer Tokenizer/Architektur bei), sie macht es nicht zu bf16.
if os.path.isdir(p) and not glob.glob(os.path.join(p, "*.safetensors")) and (
        glob.glob(os.path.join(p, "*.gguf")) or glob.glob(os.path.join(p, "*", "*.gguf"))):
    print("gguf"); sys.exit()
cfg = json.load(open(os.path.join(p, "config.json")))
q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
m = str(q.get("quant_method") or "").lower()
kinds = {(str((g.get("weights") or {}).get("type")), (g.get("weights") or {}).get("num_bits"))
         for g in (q.get("config_groups") or {}).values()}
hq = os.path.join(p, "hf_quant_config.json")
if m == "fp8":
    print("fp8")
elif m == "modelopt" or os.path.exists(hq):
    # modelopt-NVFP4 (nvidia, RadixArk; hf_quant_config.json, MIXED_PRECISION) ist ein anderer
    # Lader als compressed-tensors-NVFP4 (unsloth) -> eigene Namen, das Profil nennt genau einen.
    algo = str(q.get("quant_algo") or "").upper()
    print("nvfp4-modelopt" if ("float", 4) in kinds or "FP4" in algo else ("fp8" if "FP8" in algo else "modelopt:" + algo.lower()))
elif m == "compressed-tensors":
    if ("float", 4) in kinds: print("nvfp4-ct")
    elif kinds == {("int", 8)}: print("int8")
    elif kinds == {("int", 4)}: print("int4")
    elif ("int", 4) in kinds: print("int4-mixed")
    elif kinds and all(k == ("float", 8) for k in kinds): print("fp8")
    else: print("compressed-tensors:" + ",".join(f"{t}{b}" for t, b in sorted(kinds, key=str)))
elif m in ("awq", "gptq", "auto-round", "autoround", "awq_marlin", "gptq_marlin"):
    print("int4")
else:
    print(m or "bf16")
EOF
}

# --- 3. Umgebung saeubern ------------------------------------------------------------
if [ -n "${FLASHINFER_CUDA_ARCH_LIST+x}" ]; then
  say "WARN: FLASHINFER_CUDA_ARCH_LIST='$FLASHINFER_CUDA_ARCH_LIST' entfernt -- vor dem flashinfer-Import gesetzt verschoebe es das JIT-Cache-Verzeichnis (0.6.14/120f bzw. /86); der Server setzt es selbst nach dem Import (utils/common.py:1547)"
  unset FLASHINFER_CUDA_ARCH_LIST
fi
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  # Das August-Image setzt 7 Architekturen fuer den Server-Modus. Im weg2-Modus gewinnt
  # sonst diese Liste ueber die Gruppen-Union (barlink_device.py:652-657) und keiner der
  # vorgebauten *_cuda_86_120-Exts passt -> minutenlanger nvcc-Bau im Boot.
  say "HINWEIS: TORCH_CUDA_ARCH_LIST='$TORCH_CUDA_ARCH_LIST' fuer den weg2-Modus entfernt (Gruppen-Union entscheidet)"
  unset TORCH_CUDA_ARCH_LIST
fi
case "${PYTORCH_CUDA_ALLOC_CONF:-}" in
  *expandable_segments*) refuse ALLOC_CONF "PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF': torch_memory_saver verweigert expandable_segments (launcher.py:5515)" ;;
esac

# --- 4. Preflight ------------------------------------------------------------------
GIB=$((1024 * 1024 * 1024))
BDFS=()

preflight_gpus() {
  command -v nvidia-smi >/dev/null || refuse GPU "nvidia-smi fehlt im Container (Treiber-Userspace nicht eingebunden: --gpus all bzw. CDI)"
  local rows
  rows=$(nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader 2>&1) \
    || refuse GPU "nvidia-smi scheitert: $rows"
  say "GPU-Inventar (NVML):"
  while IFS= read -r r; do say "  $r"; done <<< "$rows"
  DRIVER_SEEN=$(awk -F', ' 'NR==1{print $6}' <<< "$rows")
  local bus dom rest
  while IFS= read -r bus; do
    dom=${bus%%:*}; rest=${bus#*:}
    BDFS+=("$(printf '%04x:%s' "$((16#$dom))" "${rest,,}")")
  done < <(awk -F', ' '{print $4}' <<< "$rows")
  if [ "${HTSGLANG_EXPECT_GPUS:-1}" = "1" ] && [ -n "${PROFILE_GPUS:-}" ]; then
    local want name n have w
    IFS=';' read -r -a want <<< "$PROFILE_GPUS"
    for w in "${want[@]}"; do
      name=${w%=*}; n=${w##*=}
      have=$(awk -F', ' -v nm="$name" '$2==nm' <<< "$rows" | wc -l)
      [ "$have" = "$n" ] || refuse GPU "Profil '$HTSGLANG_PROFILE' erwartet ${n}x '$name', sichtbar ${have}x (Rig-Profil, auf dieses Karteninventar kalibriert)"
    done
  fi
}

preflight_shm() {
  local size
  size=$(df -B1 --output=size /dev/shm | tail -1 | tr -d ' ')
  [ "$size" -ge $((PROFILE_SHM_MIN_GIB * GIB)) ] \
    || refuse SHM "/dev/shm = $((size / GIB)) GiB < ${PROFILE_SHM_MIN_GIB} GiB -- docker run --shm-size (27B: 48g, NF: 16g); nie --ipc=host (Launcher-#1217 saehe fremde Halter)"
  say "/dev/shm $((size / GIB)) GiB (Minimum ${PROFILE_SHM_MIN_GIB})"
}

preflight_memory() {
  local total avail
  total=$(awk '/^MemTotal:/{print $2}' /proc/meminfo); avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  say "meminfo: MemTotal $((total / 1048576)) GiB, MemAvailable $((avail / 1048576)) GiB; cgroup memory.max $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo ?)"
  if [ $((avail / 1048576)) -lt "${PROFILE_MEMAVAIL_MIN_GIB:-40}" ]; then
    [ "$SUB" = "dryrun" ] && say "WARN: MemAvailable < ${PROFILE_MEMAVAIL_MIN_GIB:-40} GiB (nur Trockenlauf)" \
      || refuse MEMORY "MemAvailable $((avail / 1048576)) GiB < ${PROFILE_MEMAVAIL_MIN_GIB:-40} GiB"
  fi
}

preflight_tree() {
  local head
  head=$(git -C "$TREE" rev-parse HEAD 2>/dev/null) || refuse TREE "$TREE ist kein git-Baum (Launcher-Stempel launcher.py:12418, Linien-Identitaet line_identity.py:82)"
  [ "$head" = "${HTSGLANG_REVISION:-}" ] || refuse TREE "HEAD $head != Image-Revision ${HTSGLANG_REVISION:-?}"
  [ -z "$(git -C "$TREE" status --porcelain)" ] || refuse TREE "Baum nicht sauber: $(git -C "$TREE" status --porcelain | head -3 | tr '\n' ' ')"
  say "Baum $TREE @ ${head:0:10} sauber"
}

preflight_model() {
  [ -e "$PROFILE_MODEL" ] || refuse MODEL "Modell fehlt: $PROFILE_MODEL (read-only unter demselben Pfad mounten)"
  [ -z "${PROFILE_DRAFT:-}" ] || [ -e "$PROFILE_DRAFT" ] || refuse MODEL "Draft fehlt: $PROFILE_DRAFT"
  local req
  for req in "${PROFILE_REQUIRED_PATHS[@]:-}"; do   # z.B. GGUF: das Tokenizer-Verzeichnis (--tokenizer-path)
    [ -z "$req" ] || [ -e "$req" ] || refuse MODEL "vom Profil verlangter Pfad fehlt: $req"
  done
  local fmt
  fmt=$(detect_format "$PROFILE_MODEL" 2>&1) || refuse MODEL "Formaterkennung scheitert: $fmt"
  say "Modellformat erkannt: $fmt ($PROFILE_MODEL)"
  if [ -n "${PROFILE_FORMAT:-}" ] && [ "$fmt" != "$PROFILE_FORMAT" ] && [ "${HTSGLANG_FORMAT_CHECK:-1}" = "1" ]; then
    refuse MODEL "Profil '$HTSGLANG_PROFILE' erwartet Format '$PROFILE_FORMAT', gemountet ist '$fmt'"
  fi
}

preflight_state() {
  local d
  for d in "$EVIDENCE_DIR" "$STORE_ROOT" "$GPU_ARB/weg2" "$SGLANG_WEG2_TMS_OUT_DIR"; do
    mkdir -p "$d" 2>/dev/null || true
    { touch "$d/.w" && rm -f "$d/.w"; } 2>/dev/null || refuse STATE "nicht beschreibbar: $d"
  done
  # ARB-Saat (PROBE_RING, calib, corridor-Samples, census): nur fehlende Dateien ergaenzen, nie ueberschreiben.
  if [ -d "$ARB_SEED" ]; then
    (cd "$ARB_SEED" && find . -type f -print0) | while IFS= read -r -d '' f; do
      [ -e "$GPU_ARB/$f" ] || { mkdir -p "$(dirname "$GPU_ARB/$f")"; cp -p "$ARB_SEED/$f" "$GPU_ARB/$f"; }
    done
  fi
  say "Zustand: evidence=$EVIDENCE_DIR ($(find "$EVIDENCE_DIR" -maxdepth 1 -name '*.D.log' 2>/dev/null | wc -l) D-Logs als Kalibrierquellen), arb=$GPU_ARB, store=$STORE_ROOT"
  if [ -n "${PROFILE_TMPFS_STORE:-}" ]; then
    [ "$(stat -f -c %T "$PROFILE_TMPFS_STORE" 2>/dev/null)" = "tmpfs" ] \
      || refuse STORE "$PROFILE_TMPFS_STORE ist kein tmpfs (cudaHostRegister auf ZFS-mmap -> cudaError 1) -- --mount type=tmpfs,dst=$PROFILE_TMPFS_STORE,tmpfs-size=${PROFILE_TMPFS_GIB:-72}g"
    local sz
    sz=$(df -B1 --output=size "$PROFILE_TMPFS_STORE" | tail -1 | tr -d ' ')
    [ "$sz" -ge $(( ${PROFILE_TMPFS_GIB:-72} * GIB )) ] || refuse STORE "$PROFILE_TMPFS_STORE = $((sz / GIB)) GiB < ${PROFILE_TMPFS_GIB:-72} GiB"
    [ -z "$(ls -A "$PROFILE_TMPFS_STORE" 2>/dev/null)" ] || say "WARN: $PROFILE_TMPFS_STORE nicht leer (Rest eines frueheren Laufs im selben Container?)"
  fi
}

BAR1_OK=1; BAR1_WHY=()
preflight_bar1() {
  local params regkeys cap_eff bdf want_drv hdr
  params=$(grep -E '^RegistryDwords:' /proc/driver/nvidia/params 2>/dev/null || true)
  regkeys=${params#RegistryDwords: }
  case "$regkeys" in *RMSmallBarP2PPeerBar1=1*) ;; *) BAR1_OK=0; BAR1_WHY+=("Regkey RMSmallBarP2PPeerBar1=1 fehlt (gepatchter smallbar-Treiber)");; esac
  cap_eff=$(awk '/^CapEff:/{print $2}' /proc/self/status)
  case "$regkeys" in
    *PeerMappingOverride=1*) ;;
    *) if (( (16#$cap_eff >> 21) & 1 )); then :; else BAR1_OK=0; BAR1_WHY+=("weder PeerMappingOverride=1 noch CAP_SYS_ADMIN (barlink_bar1.py:3075-3100)"); fi ;;
  esac
  if [ -c /dev/dmabuf_holder ] && [ -r /dev/dmabuf_holder ] && [ -w /dev/dmabuf_holder ]; then :; else
    BAR1_OK=0; BAR1_WHY+=("/dev/dmabuf_holder fehlt/nicht rw (--device /dev/dmabuf_holder)"); fi
  for bdf in "${BDFS[@]}"; do
    [ -w "/sys/bus/pci/devices/$bdf/resource1_wc" ] || { BAR1_OK=0; BAR1_WHY+=("/sys/bus/pci/devices/$bdf/resource1_wc nicht beschreibbar (-v /sys:/sys, Host-Rechte 0666)"); }
  done
  hdr=${SGLANG_BARLINK_BAR1_NV_SOURCE:-/opt/nvidia-open-595}/src/common/sdk/nvidia/inc/nvos.h
  [ -f "$hdr" ] || { BAR1_OK=0; BAR1_WHY+=("NV-Header fehlen unter ${SGLANG_BARLINK_BAR1_NV_SOURCE:-/opt/nvidia-open-595} (Image mit WITH_NV_HEADERS=0 gebaut; Header des LAUFENDEN Treibers mounten oder mit WITH_NV_HEADERS=1 bauen)"); }
  want_drv=$(jq -r '.driver_expected // empty' "$BUILD_INFO" 2>/dev/null || true)
  if [ -f "$hdr" ] && [ -n "$want_drv" ] && [ "${DRIVER_SEEN:-}" != "$want_drv" ] && [ "${SGLANG_BARLINK_BAR1_NV_SOURCE:-/opt/nvidia-open-595}" = "/opt/nvidia-open-595" ]; then
    BAR1_OK=0; BAR1_WHY+=("Treiber ${DRIVER_SEEN:-?} != Header-Stand im Image $want_drv (UAPI versionsgebunden, barlink_bar1_ext.py:14-21)")
  fi
  if [ "$BAR1_OK" = "1" ]; then say "BAR1-Kette vollstaendig (Regkeys: $regkeys)"; else say "BAR1-Kette unvollstaendig: ${BAR1_WHY[*]}"; fi
}

run_preflight() {
  preflight_gpus
  preflight_shm
  preflight_memory
  preflight_tree
  preflight_model
  preflight_state
  preflight_bar1
}

nccl_barlink_off() {   # barlink KOMPLETT aus: keine Rest-Variablen, die ohne Flags doch barlink bauen liessen
  local v
  for v in SGLANG_BARLINK SGLANG_BARLINK_TRANSPORT SGLANG_BARLINK_PP_TRANSPORT; do
    if [ -n "${!v+x}" ]; then say "nccl: $v='${!v}' aus der Umgebung entfernt"; unset "$v"; fi
  done
}

resolve_transport() {
  : "${HTSGLANG_TRANSPORT:=bar1}"
  local nccl_status=${PROFILE_NCCL_STATUS:-unproven}
  case "$HTSGLANG_TRANSPORT" in
    bar1)
      [ "$BAR1_OK" = "1" ] || refuse TRANSPORT "bar1 verlangt, BAR1-Kette unvollstaendig: ${BAR1_WHY[*]} -- kein Wechsel auf NCCL ohne ausdrueckliches HTSGLANG_TRANSPORT=nccl"
      TRANSPORT=bar1 ;;
    nccl)
      TRANSPORT=nccl
      [ "$nccl_status" = "proven" ] || say "WARN: NCCL ist fuer Profil '$HTSGLANG_PROFILE' UNBELEGT (nie gebootet) -- ausdruecklich verlangt, laeuft" ;;
    auto)
      refuse TRANSPORT "'auto' gibt es nicht (F7: bar1 ist Default, nccl nur ausdruecklich) -- HTSGLANG_TRANSPORT=bar1 oder =nccl" ;;
    *) refuse TRANSPORT "HTSGLANG_TRANSPORT='$HTSGLANG_TRANSPORT' (bar1|nccl)" ;;
  esac
  [ "$TRANSPORT" = "nccl" ] && nccl_barlink_off
  say "TRANSPORT=$TRANSPORT (angefordert $HTSGLANG_TRANSPORT). Beleg nach dem Boot: je Gruppe 'ACHIEVED=bar1' bzw. keine barlink-Zeile"
}

# --- Selbsttest ohne GPU ------------------------------------------------------------
if [ "$SUB" = "selfcheck" ]; then
  preflight_tree
  "$PY" - <<'EOF'
import os, pathlib
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    print("nccl", torch.cuda.nccl.version())
except Exception as e:  # noqa: BLE001
    print("nccl version unreadable:", e)
import flashinfer
print("flashinfer", flashinfer.__version__)
root = pathlib.Path.home() / ".cache/flashinfer" / flashinfer.__version__
for d in ("120f", "86"):
    ops = sorted(p.name for p in (root / d / "cached_ops").glob("*") if (p / f"{p.name}.so").exists())
    print(f"flashinfer {d}: {len(ops)} vorgebaute Module")
ext = pathlib.Path(os.environ.get("TORCH_EXTENSIONS_DIR", pathlib.Path.home() / ".cache/torch_extensions"))
print("torch_extensions:", sorted(str(p.relative_to(ext)) for p in ext.glob("*/*") if p.is_dir()))
print("tvm-ffi Saat:", len(list((pathlib.Path.home() / ".cache/tvm-ffi").glob("*"))), "Eintraege")
tms = [p.name for p in pathlib.Path(os.environ["SGLANG_WEG2_TMS_OUT_DIR"]).glob("*.so")]
print("tms preload:", tms)
cu13 = pathlib.Path(os.environ.get("SGLANG_WEG2_VENV", "/opt/venv")) / "lib/python3.12/site-packages/nvidia/cu13/lib"
print("cu13 libcudart.so (Rig-Link):", (cu13 / "libcudart.so").exists())
if not tms:
    # Der Launcher baut den Hook bei jedem Boot (build_tms_preload) und verweigert, wenn das scheitert.
    print("SELFCHECK FEHLER: kein TMS-Preload im Image -- jeder weg2-Boot wuerde verweigert (Weg2LaunchRefused)")
    raise SystemExit(1)
nv = pathlib.Path(os.environ.get("SGLANG_BARLINK_BAR1_NV_SOURCE", "/opt/nvidia-open-595"))
print("NV-Header im Image:", (nv / "src/common/sdk/nvidia/inc/nvos.h").is_file())
EOF
  "$PY" -I "$TREE/python/sglang/srt/utils/kernel_dist_guard.py" \
      --site-packages "$VENV/lib/python3.12/site-packages" --require-arm --expect-pinned-sha256
  exit 0
fi

run_preflight
resolve_transport
[ "$SUB" = "preflight" ] && { say "PREFLIGHT fertig -- kein Launch"; exit 0; }

LAUNCH=(-m sglang.srt.weg2.launcher --tree "$TREE" --tag "$HTSGLANG_TAG" --transport "$TRANSPORT" "${PROFILE_ARGS[@]}")
cd "$TREE"

if [ "$SUB" = "dryrun" ]; then
  say "DRY-RUN: $PY ${LAUNCH[*]} --dry-run $*"
  CUDA_VISIBLE_DEVICES="" exec "$PY" "${LAUNCH[@]}" --dry-run "$@"
fi
[ "$SUB" = "serve" ] || refuse MODE "unbekannter weg2-Untermodus '$SUB'"

STATE=$GPU_ARB/weg2/boot_${HTSGLANG_TAG}.json   # launcher.py:14258-14259
LOGCOPY=$EVIDENCE_DIR/docker_${HTSGLANG_TAG}
mkdir -p "$LOGCOPY"

archive_artifacts() {   # Laufzeit-Artefakte aus GPU_ARB sichern (nur benannte Dateien, nie Schluesseldateien)
  local f
  for f in "$GPU_ARB"/deadman_"${HTSGLANG_TAG}"_*.out "$GPU_ARB"/memts_weg2_"${HTSGLANG_TAG}".csv \
           "$GPU_ARB"/preflight_weg2_"${HTSGLANG_TAG}".log "$STATE" "$GPU_ARB/weg2/boot_${HTSGLANG_TAG}.logpath"; do
    [ -f "$f" ] && cp -p "$f" "$LOGCOPY/" 2>/dev/null || true
  done
}

TORN=0
LPID=""
teardown() {
  [ "$TORN" = "1" ] && return 0; TORN=1
  say "TEARDOWN ($1)"
  [ -n "$LPID" ] && kill -TERM "$LPID" 2>/dev/null && sleep 2
  if [ -f "$STATE" ]; then
    "$PY" -m sglang.srt.weg2.launcher --tree "$TREE" --tag "$HTSGLANG_TAG" --teardown "$STATE" \
      >> "$LOGCOPY/teardown.log" 2>&1 || say "teardown rc=$? (siehe $LOGCOPY/teardown.log)"
  fi
  archive_artifacts
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  nach Teardown: /' >&2 || true
}
# Signale kommen waehrend `wait` sofort an. docker stop -t 180 geben (TERM -> warten -> KILL).
trap 'teardown SIGTERM; exit 0' TERM INT

say "LAUNCH: $PY ${LAUNCH[*]} $*"
"$PY" "${LAUNCH[@]}" "$@" > >(tee -a "$LOGCOPY/launcher.log") 2>&1 &
LPID=$!
set +e
wait "$LPID"
rc=$?
set -e
LPID=""
if [ "$rc" != "0" ]; then
  # Eine Verweigerung nach dem ersten Spawn baut der Launcher selbst ab (cli(), #1248).
  say "Launcher rc=$rc -- Boot nicht zustande gekommen (Launcher-Log: $LOGCOPY/launcher.log)"
  archive_artifacts
  exit "$rc"
fi

say "LAUNCHED -- Aufsicht: Front :$FRONT_PORT (Liveness /health, Readiness /weg2/state == serving)"
while :; do
  if ! pgrep -f "sglang[.]srt[.]weg2[.]front .*--port ${FRONT_PORT}" >/dev/null; then
    say "Front-Prozess ist weg -> Teardown"
    teardown front-dead
    exit 1
  fi
  sleep 5 & wait $!
done
