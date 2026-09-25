#!/bin/bash
# Abnahme des htsglang-Upgrade-Images AUF DEM PROXMOX-HOST -- ENTWURF (27B-Sitz R, 24.09.2026).
# NICHT GELAUFEN. Ausfuehrung nur durch Nutzer oder Operator nach Freigabe, als root auf dem Host:
#
#   CTX=/spinning/gpu-arb/docker/ctx/27b-<sha10> LINE=27b HOUSE_GUARD=memlimit|ct999-ruht [GPUQ_ID=<id>] \
#     bash /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/host_acceptance.sh <schritt> [transport]
#
# Schritte (einzeln oder `all`):
#   check            nur lesen: Treiber/BAR1-Kette, Karten leer, Haus-Schutz (HOUSE_GUARD), Host-RAM, Platz, Fenster
#   build            ruft host_build.sh (eigener Builder mit Speicher-/CPU-Deckel, Manifest-Pruefung, F12-Liste,
#                    D1 selfcheck); braucht EXPECT_MANIFEST aus Rs Bericht
#   seed             Bind-Verzeichnisse anlegen und aus dem Rig saeen (Referenz-Logs, sglang-Zustand, Triton)
#   d2 <transport>   Trockenlauf im Container (nur NVML, kein CUDA-Kontext)
#   t0               Preflight + BAR1-Graph-Probe (bar1_graph_check 0,1,2) -- braucht das Fenster
#   serve <t>        Boot bar1|nccl mit Instrumenten (Paritaet zu den Rig-Referenzboots), Readiness, Proben
#                    (GEN, Nadel, Flips, Decode), Belege (Transport, NCCL-Version, Idle-Politik), Teardown
#   release          Boot bar1 in der Veroeffentlichungs-Form (HTSGLANG_INSTRUMENTS=0, F8), GEN + Nadel
#   negative         bar1 ohne /dev/dmabuf_holder muss mit Grund verweigern
#   all              check build seed d2 bar1 d2 nccl t0 serve bar1 serve nccl release negative
#
# HAUS-SCHUTZ (F13, Entscheid beim Nutzer) -- Schalter HOUSE_GUARD, fuer check/t0/serve/release/negative PFLICHT:
#   memlimit    CT999 laeuft weiter (Router 30099, gpuq, Agenten-Sitzungen), aber OHNE Boot. Der Container bekommt
#               --memory ${MEM_LIMIT} (= --memory-swap, kein Swap) und --oom-score-adj 500: er stirbt bei Host-Druck
#               vor den Haus-Diensten (und vor CT999). Waechst CT999 waehrend des Boots (Agenten-Tests!), stirbt
#               also der Abnahme-Boot, nicht das Haus.
#   ct999-ruht  CT999 ruht (gestoppt oder eingefroren). Das tut der NUTZER vorher selbst -- nie dieses Skript und nie
#               ein Agent: Router 30099, gpuq und alle Agenten-Sitzungen leben in CT999 und ruhen mit. Der Container
#               bekommt dieselbe Speicherdecke wie die nativen Referenzboots (die CT999-LXC-Grenze aus `pct config`,
#               24.09.: 120880 MiB) und --oom-score-adj 500. gpuq ist dann nicht erreichbar: die Karten gehoeren fuer
#               die Dauer dem Nutzer; bindend ist die Pruefung "Karten leer".
#   Beide Varianten verlangen Host-MemAvailable >= MEMAVAIL_MIN_GIB (gemessene Spitze + Luft; 27B 80, NF 90).
#   Gemessen 24.09. ~22:55Z (nur lesend, waehrend eines CT999-Boots): Host 125,7 GiB; ausserhalb CT999 ~20 GiB
#   (system.slice mit den Haus-Containern 9,5, SUnreclaim inkl. ZFS-ARC 8,2 (ARC-Deckel 5), andere LXC 1,8).
#   Fuer CT999 + Container bleiben also hoechstens ~105 GiB -- die fruehere Bedingung "MemAvailable >= 105 GiB"
#   war nie erfuellbar und ist ersetzt. Mit ruhendem CT999 stehen ~100 GiB bereit, mit laufendem (ohne Boot)
#   entsprechend weniger; ob 96 GiB dann reichen, sagt erst `check`.
#   In beiden Varianten: nie `docker system prune`, nie Haus-Container anfassen, Router 30099 nie beruehren; die
#   Front wird nur auf dem Host-Loopback 127.0.0.1:${PORT} veroeffentlicht (Proben laufen auf dem Host). MEM_LIMIT in
#   ganzen GiB mit 'g'. /proc/meminfo im Container ist die des HOSTS (kein lxcfs) -> Plan R-2.
#
# F12/F14: das oeffentliche ghcr.io/efschu/htsglang:cu130-nccl2307 wird von diesem Skript nie geloescht, nie
# ueberschrieben und nie gepusht. Gebaut wird ein NEUER versionierter Tag (IMAGE unten); Ersetzen des oeffentlichen
# Tags und jeder Push nur mit Go des Nutzers, ausserhalb dieses Skripts.
set -euo pipefail

S=/spinning/subvol-999-disk-0                 # CT999-Wurzel, wie der Host sie sieht
LXC=999
LINE=${LINE:-27b}
PROFILE=${PROFILE:-$LINE}
RELEASE=${RELEASE:-$([ "$LINE" = nf ] && echo rc1 || echo rc5)}   # Tag-Bestandteil wie host_build.sh (27B: rc5, NF: rc1)
CTX=${CTX:?CTX = Kontext im LXC-Pfad, z.B. /spinning/gpu-arb/docker/ctx/27b-<sha10>}
HCTX=$S$CTX
REV=$(jq -r .revision "$HCTX/BUILD_INFO.json")
IMAGE=${IMAGE:-htsglang:cu129-weg2-${LINE}-${RELEASE}-${REV:0:10}}
PORT=${PORT:-31030}
MODEL_NAME=${MODEL_NAME:-$([ "$LINE" = 27b ] && echo Qwen3.8-27B || echo Qwen3.8-Flash-Next)}
HOUSE_GUARD=${HOUSE_GUARD:-}
MEM_LIMIT=${MEM_LIMIT:-104g}                  # HOUSE_GUARD=memlimit (ganze GiB mit 'g')
# Untergrenze Host-MemAvailable (beide Varianten) = gemessene Spitze + Luft, je Profil. 27B RC2-final (25.09.,
# hostram_<tag>, nonreclaim in CT999-Sicht, also MIT dem CT999-Grundstock): INT8 70,1 / FP8 71,7 GiB -> 80.
# NF (Operator 25.09., Release-Stand x177): Host-RAM-Spitze INT4-Mixed 83 GiB -> 90, NVFP4 91 GiB -> 98 -- 98 GiB gibt
# der Host nur mit ruhendem CT999 her (ausserhalb CT999 liegen ~20 GiB), also HOUSE_GUARD=ct999-ruht fuer nf-nvfp4.
case "$PROFILE" in
  27b*) _mem_default=80 ;;
  nf-nvfp4) _mem_default=98 ;;
  nf*) _mem_default=90 ;;
  *) _mem_default=90 ;;
esac
MEMAVAIL_MIN_GIB=${MEMAVAIL_MIN_GIB:-$_mem_default}
INSTRUMENTS=${INSTRUMENTS:-1}                 # serve: Paritaet mit den Referenzboots; `release` faehrt 0
SHM=${SHM:-$([ "$LINE" = 27b ] && echo 48g || echo 16g)}   # NF: Profil-Minimum 16 GiB (gemessen <= 8,7); ueberschreibbar
GPUQ_URL=${GPUQ_URL:-http://192.168.0.88:8770}   # gpuq lebt in CT999
ACC=$S/spinning/docker-acceptance/$LINE       # im LXC sichtbar als /spinning/docker-acceptance/<linie>
MODELS=$S/spinning/llm_stuff/club-3090/models-cache
REF_TAG=${REF_TAG:-}                          # nativer Referenzboot derselben Linie (Kalibrier-Saat), z.B. weg2rc2
LOG=$ACC/acceptance_$(date -u +%m%d_%H%M%S).log
mkdir -p "$ACC"

say(){ echo "[host-acc $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
die(){ say "ABBRUCH: $*"; exit 1; }
HB=""
cleanup() {   # laeuft bei JEDEM Ende: kein Abnahme-Container und kein Herzschlag bleibt zurueck
  [ -n "$HB" ] && kill "$HB" 2>/dev/null
  for c in $(docker ps -a --format '{{.Names}}' | grep -E "^htsglang-acc-${LINE}-" || true); do
    docker stop -t 180 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null 2>&1
    say "cleanup: $c gestoppt und entfernt"
  done
}
trap cleanup EXIT

# --- Haus-Schutz (F13) ----------------------------------------------------------------
ct999_state() {   # nur lesen: running | stopped | frozen
  local st; st=$(pct status "$LXC" 2>/dev/null | awk '{print $2}')
  if [ "$st" = running ] && lxc-info -n "$LXC" -s 2>/dev/null | grep -q FROZEN; then st=frozen; fi
  echo "${st:-unbekannt}"
}
ct999_mem_mib() { pct config "$LXC" 2>/dev/null | awk '/^memory:/{print $2}'; }

house_args() {   # docker-run-Speicherargumente je Variante
  case "$HOUSE_GUARD" in
    memlimit)   printf '%s\n' --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT" --oom-score-adj 500 ;;
    ct999-ruht) local m; m=$(ct999_mem_mib); [ -n "$m" ] || die "pct config $LXC: keine memory-Zeile"
                printf '%s\n' --memory "${m}m" --memory-swap "${m}m" --oom-score-adj 500 ;;
    "")         # nur fuer d2 (Trockenlauf, keine Rechenlast): Decke wie memlimit
                printf '%s\n' --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT" --oom-score-adj 500 ;;
    *)          die "HOUSE_GUARD='$HOUSE_GUARD' (memlimit|ct999-ruht)" ;;
  esac
}

house_check() {   # nur lesen
  local st avail
  st=$(ct999_state)
  avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
  case "$HOUSE_GUARD" in
    memlimit)
      [ "$st" = running ] || die "HOUSE_GUARD=memlimit, aber CT999 ist '$st' -- diese Variante setzt ein laufendes CT999 voraus"
      if pct exec "$LXC" -- pgrep -f -- 'sglang::schedule[r]|-m sglang\.srt\.weg2\.launche[r]|-m sglang\.launch_serve[r]' >/dev/null 2>&1; then
        die "in CT999 laeuft ein Boot -- der Host hat 125 GiB, beide zugleich gehen nicht"
      fi
      [ "$avail" -ge "$MEMAVAIL_MIN_GIB" ] || die "Host-MemAvailable ${avail} GiB < ${MEMAVAIL_MIN_GIB} GiB (Spitze + Luft): CT999 belegt zu viel -- erst CT999 entlasten oder HOUSE_GUARD=ct999-ruht"
      say "Haus-Schutz memlimit: CT999 laeuft ohne Boot, Container --memory ${MEM_LIMIT} --oom-score-adj 500, Host-MemAvailable ${avail} GiB"
      ;;
    ct999-ruht)
      case "$st" in
        stopped|frozen) ;;
        *) die "HOUSE_GUARD=ct999-ruht, aber CT999 ist '$st'. Ruhen lassen tut der Nutzer selbst (pct shutdown/suspend $LXC) -- nie dieses Skript: Router 30099, gpuq und die Agenten leben dort" ;;
      esac
      [ "$avail" -ge "$MEMAVAIL_MIN_GIB" ] || die "Host-MemAvailable ${avail} GiB < ${MEMAVAIL_MIN_GIB} GiB trotz ruhendem CT999 -- Haus-Dienste pruefen"
      say "Haus-Schutz ct999-ruht: CT999 $st, Container --memory $(ct999_mem_mib)m (CT999-LXC-Grenze) --oom-score-adj 500, Host-MemAvailable ${avail} GiB"
      ;;
    *) die "HOUSE_GUARD fehlt (F13 offen beim Nutzer): HOUSE_GUARD=memlimit oder HOUSE_GUARD=ct999-ruht setzen" ;;
  esac
}

run_args() {   # gemeinsame docker-run-Argumente; $1 = Transport (bar1|nccl), $2 = mit Holder (1|0), $3 = Instrumente (0|1)
  local t=$1 holder=${2:-1} instr=${3:-$INSTRUMENTS}
  local a=(--gpus all --security-opt apparmor=unconfined -v /sys:/sys
           --shm-size="$SHM" --ulimit memlock=-1:-1 --init
           -v "$MODELS:/spinning/llm_stuff/club-3090/models-cache:ro"
           -v "$ACC/evidence:/var/lib/htsglang/evidence" -v "$ACC/arb:/var/lib/htsglang/arb"
           -v "$ACC/store:/var/lib/htsglang/hicache-weg2" -v "$ACC/sglang:/root/.cache/sglang"
           -v "htsglang-${LINE}-${REV:0:10}-flashinfer:/root/.cache/flashinfer"
           -v "htsglang-${LINE}-${REV:0:10}-torchext:/root/.cache/torch_extensions"
           -v "htsglang-${LINE}-${REV:0:10}-tvmffi:/root/.cache/tvm-ffi"
           -v "$ACC/triton:/root/.triton"
           -e MODE=weg2 -e HTSGLANG_PROFILE="$PROFILE" -e HTSGLANG_TRANSPORT="$t" -e HTSGLANG_INSTRUMENTS="$instr")
  mapfile -t -O "${#a[@]}" a < <(house_args)
  [ "$holder" = "1" ] && a+=(--device /dev/dmabuf_holder)
  if [ "$(jq -r .nv_headers.in_image "$HCTX/BUILD_INFO.json")" != "true" ]; then
    a+=(-v "$S/spinning/nvidia-open-595:/opt/nvidia-open-595:ro")   # Header des LAUFENDEN Treibers
  fi
  # NF-Expertenspeicher: tmpfs (cudaHostRegister auf ZFS-mmap scheitert). Belegung INT4-Mixed ~39 GiB, NVFP4 ~46 GiB;
  # 72 GiB = PROFILE_TMPFS_GIB (der Entrypoint prueft die Groesse). Zaehlt als shmem gegen den Container-Deckel.
  if [ "$LINE" = nf ]; then a+=(--mount "type=tmpfs,dst=/mnt/nf-experts,tmpfs-size=${NF_TMPFS_BYTES:-77309411328},tmpfs-mode=1777"); fi
  printf '%s\n' "${a[@]}"
}

load_args() {   # load_args <transport> <holder> [instrumente] -> Array A; ohne Speicherdecke kein Lauf
  mapfile -t A < <(run_args "$@")
  printf '%s\n' "${A[@]}" | grep -qx -- --memory && printf '%s\n' "${A[@]}" | grep -qx -- --oom-score-adj \
    || die "docker-run-Argumente ohne --memory/--oom-score-adj (Haus-Schutz nicht aufgeloest, HOUSE_GUARD='$HOUSE_GUARD')"
}

step_check() {
  say "== check (nur lesen)"
  grep -q "595.58.03" /proc/driver/nvidia/version || die "Treiber ist nicht 595.58.03"
  grep -qE '^RegistryDwords:.*RMSmallBarP2PPeerBar1=1.*PeerMappingOverride=1' /proc/driver/nvidia/params || say "WARN: BAR1-Regkeys fehlen -> nur nccl moeglich"
  [ -c /dev/dmabuf_holder ] || say "WARN: /dev/dmabuf_holder fehlt -> nur nccl moeglich"
  for c in 0000:05:00.0 0000:0a:00.0 0000:0b:00.0; do [ "$(stat -c %a /sys/bus/pci/devices/$c/resource1_wc)" = 666 ] || say "WARN: resource1_wc $c nicht 0666"; done
  local apps; apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$apps" = 0 ] || die "Karten belegt ($apps Prozesse) -- erst das Fenster, dann leere Karten"
  house_check
  say "Host: Pool spinning frei: $(zfs list -H -o avail spinning)"
  docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' || die "nvidia-Runtime fehlt in docker info"
  if [ "$HOUSE_GUARD" = ct999-ruht ]; then
    say "gpuq ruht mit CT999 -- Karten gehoeren fuer diese Abnahme dem Nutzer (Pruefung 'Karten leer' oben ist bindend)"
  elif [ -n "${GPUQ_ID:-}" ]; then
    local st; st=$(curl -s -m 10 "$GPUQ_URL/api/v1/bookings/$GPUQ_ID" | jq -r '.state // empty')
    [ "$st" = "running" ] || die "gpuq-Fenster $GPUQ_ID nicht running (state=${st:-?})"
    say "gpuq-Fenster $GPUQ_ID laeuft"
  else
    die "HOUSE_GUARD=memlimit ohne GPUQ_ID -- GPU-Schritte nur im gebuchten Fenster"
  fi
}

step_build() {   # der Bau selbst steht in host_build.sh (ein Befehl, ein Deckel, eine Stelle)
  say "== build -> host_build.sh"
  CTX="$CTX" RELEASE="$RELEASE" IMAGE="$IMAGE" bash "$(dirname "$(readlink -f "$0")")/host_build.sh" \
    || die "host_build.sh gescheitert"
}

step_seed() {
  say "== seed ($ACC)"
  mkdir -p "$ACC"/{evidence,arb,store,sglang,triton}
  if [ -n "$REF_TAG" ]; then   # Kalibrierquellen des nativen Referenzboots (Launcher liest *.P/D/front.log)
    cp -pn "$S"/spinning/evidence-665-f1/boot_weg2_"${REF_TAG}"_*.{P,D,front}.log "$ACC/evidence/" 2>/dev/null || say "WARN: keine Logs fuer REF_TAG=$REF_TAG"
  fi
  [ -f "$S/spinning/evidence-665-f1/weg2_measured_record.json" ] && cp -pn "$S/spinning/evidence-665-f1/weg2_measured_record.json" "$ACC/evidence/"
  cp -pn "$S"/root/.cache/sglang/*.json "$ACC/sglang/" 2>/dev/null || true      # card_probe, phase_footprint, hw_profile, kv_budget
  # Das Image setzt TRITON_CACHE_DIR=/root/.triton (August-ENV), der Rig nutzt ~/.triton/cache:
  # die EINTRAEGE flach nach $ACC/triton (Triton-Cache ist inhaltsadressiert, ~6,4 GB).
  if [ "${SEED_TRITON:-1}" = "1" ]; then cp -a -n "$S/root/.triton/cache/." "$ACC/triton/" 2>/dev/null || true; fi
  say "Saat: evidence $(ls "$ACC/evidence" | wc -l) Dateien, sglang $(ls "$ACC/sglang" | wc -l), triton $(du -sh "$ACC/triton" | cut -f1)"
}

step_d2() {
  local t=${1:?transport}; say "== D2 Trockenlauf $t"
  load_args "$t" 1
  set +e; docker run --rm "${A[@]}" "$IMAGE" dryrun >"$ACC/d2_${t}.log" 2>&1; local rc=$?; set -e
  say "D2 $t rc=$rc; $(grep -a -oE 'WEG2-P-FORM key=[^ ]+|=> (FUNDABLE|refused)|WEG2-LAUNCH REFUSED: W[0-9]+ [A-Za-z]+|REAP MODEL: MemTotal=[0-9.]+|IDLE POLICY [^:]*: \(a\) --idle-layout [a-z]+|\(b\) --d-short-drain-tokens [0-9]+|\(c\) --d-hold-s [0-9.]+' "$ACC/d2_${t}.log" | sort -u | tr '\n' ' ')"
}

step_t0() {
  say "== T0 Preflight + BAR1-Graph-Probe"
  load_args bar1 1
  set +e
  docker run --rm "${A[@]}" "$IMAGE" preflight 2>&1 | tee -a "$LOG"; say "T0a preflight rc=${PIPESTATUS[0]}"
  docker run --rm "${A[@]}" --entrypoint /opt/venv/bin/python -w /opt/htsglang/src "$IMAGE" \
    benchmark/bar1_graph_check.py 0,1,2 29700 >"$ACC/t0b_bar1_graph.log" 2>&1; say "T0b rc=$? (Soll 0, 10/10)"
  set -e
  say "T0b: $(tail -3 "$ACC/t0b_bar1_graph.log" | tr '\n' ' ')"
}

holder_on()  { echo "docker-acc-$LINE guard=$HOUSE_GUARD gpuq=${GPUQ_ID:-?} cards=0,1,2 $(date -u +%FT%TZ)" > "$S/spinning/gpu-arb/holder"
               ( while :; do touch "$S/spinning/gpu-arb/holder"; sleep 20; done ) & HB=$!; }
# Herzschlag VOR der Fensterfreigabe stoppen (Rig-Regel); die Freigabe (gpu_release) macht der Aufrufer.
holder_off() { [ -n "${HB:-}" ] && kill "$HB" 2>/dev/null; HB=""; }

boot_and_probe() {   # $1 transport, $2 instrumente (0|1), $3 name-suffix, $4 probe-satz (full|release)
  local t=$1 instr=$2 sfx=$3 set=$4 name="htsglang-acc-${LINE}-$3"
  local tag="dkr${LINE}${sfx}$(date -u +%m%d%H%M)"
  say "== boot $sfx (transport $t, HTSGLANG_INSTRUMENTS=$instr, $name, tag $tag)"
  load_args "$t" 1 "$instr"
  docker rm -f "$name" >/dev/null 2>&1 || true
  local mark; mark=$(mktemp); holder_on
  docker run -d --name "$name" -p "127.0.0.1:${PORT}:30030" -e HTSGLANG_TAG="$tag" \
    "${A[@]}" "$IMAGE" serve >>"$LOG"
  local ok=0
  for _ in $(seq 1 300); do
    docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true || { say "Container beendet"; break; }
    if curl -s -m 5 "http://127.0.0.1:${PORT}/weg2/state" | jq -e '.state == "serving"' >/dev/null 2>&1; then ok=1; break; fi
    sleep 5
  done
  if [ "$ok" = 1 ]; then
    say "serving nach $(( $(date +%s) - $(stat -c %Y "$mark") )) s"
    local P="python3 $S/spinning/gpu-arb/docker/probes.py --base http://127.0.0.1:${PORT} --model $MODEL_NAME"
    local CODE="$HCTX/src/python/sglang/srt/weg2/launcher.py"      # eingefrorener Code-Kontext (Image-Baum)
    set +e   # eine gescheiterte Probe beendet nicht die Abnahme; der Teardown unten laeuft immer
    if [ "$set" = full ]; then
      { $P gen; $P needle --sentences 5500 --position 0.10; $P flip; $P gen; $P flip; $P gen
        for k in code prosa thinking code; do $P decode --kind "$k" --depth 10000 --max-tokens 1024 --code-file "$CODE"; done
      } 2>&1 | tee -a "$ACC/probes_${sfx}.jsonl" | tee -a "$LOG"
    else
      { $P gen; $P needle --sentences 5500 --position 0.10; $P gen; } 2>&1 | tee -a "$ACC/probes_${sfx}.jsonl" | tee -a "$LOG"
    fi
    set -e
  else
    say "NICHT serving -- Logs: docker logs $name; Evidenz $ACC/evidence"
  fi
  local ev="$ACC/evidence/boot_weg2_${tag}"
  say "Belege $sfx (Transport): $(grep -a -h -oE "barlink enabled for group '[^']+': requested=[a-z0-9]+, ACHIEVED=[a-z0-9]+|barlink group '[^']+': requested=[a-z0-9]+, ACHIEVED=[a-z0-9]+" "$ev"_*.{P,D}.log 2>/dev/null | sort | uniq -c | tr '\n' ';')"
  say "Belege $sfx (NCCL, F10 Soll 2.28.9): $(grep -a -h -oE 'sglang is using nccl==[0-9.]+' "$ev"_*.{P,D}.log 2>/dev/null | sort | uniq -c | tr '\n' ';')"
  say "Belege $sfx (Idle-Politik): $(grep -a -h -m1 -oE 'WEG2-IDLE-POLICY idle_layout=[A-Za-z]+ d_short_drain_tokens=[0-9]+ X=[0-9]+ flip_min_work_tokens=[0-9]+ d_hold_s=[^ ]+' "$ev"_*.front.log 2>/dev/null | head -1)"
  say "JIT-Bauten waehrend des Boots (Dateien juenger als der Start in den Cache-Volumes):"
  for v in flashinfer torchext tvmffi; do
    local mp; mp=$(docker volume inspect -f '{{.Mountpoint}}' "htsglang-${LINE}-${REV:0:10}-$v" 2>/dev/null || true)
    [ -n "$mp" ] && say "  $v: $(find "$mp" -newer "$mark" -name '*.so' 2>/dev/null | wc -l) neue .so"
  done
  docker stop -t 180 "$name" >>"$LOG" 2>&1 || true; docker rm "$name" >>"$LOG" 2>&1 || true
  holder_off; rm -f "$mark"
  sleep 5; say "nach Teardown: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"
}

step_serve()   { local t=${1:?transport}; boot_and_probe "$t" "$INSTRUMENTS" "$t" full; }
step_release() { boot_and_probe bar1 0 release release; }

step_negative() {
  say "== negative: bar1 ohne Holder muss verweigern"
  load_args bar1 0
  set +e; docker run --rm "${A[@]}" "$IMAGE" preflight >"$ACC/neg_bar1_noholder.log" 2>&1; local rc=$?; set -e
  say "rc=$rc (Soll 3), $(grep -a -m1 REFUSED "$ACC/neg_bar1_noholder.log")"
}

case "${1:-}" in
  check) step_check ;;
  build) step_build ;;
  seed) step_seed ;;
  d2) step_d2 "${2:-bar1}" ;;
  t0) house_check; step_t0 ;;
  serve) house_check; step_serve "${2:-bar1}" ;;
  release) house_check; step_release ;;
  negative) house_check; step_negative ;;
  all) step_check; step_build; step_seed; step_d2 bar1; step_d2 nccl
       house_check; step_t0
       house_check; step_serve bar1
       house_check; step_serve nccl
       house_check; step_release
       step_negative ;;
  *) sed -n '2,43p' "$0"; exit 2 ;;
esac
say "Protokoll: $LOG"
