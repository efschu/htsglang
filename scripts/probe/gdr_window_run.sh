#!/bin/bash
# SPDX-License-Identifier: MIT
#
# gdr_window_run.sh -- der eigentliche Fensterlauf fuer die dmabuf-GPU-RDMA-
# Sondierung (Naming: dmabuf_rdma, kein Markenname).
#
# STUFE 0 -- Smoke-Test 5090<->CX-4 ("geht es hier ueberhaupt"), IMMER ZUERST,
# vor jeder Leiter, per gpurdma_00_smoke:
#   (a) regcheck  -- ibv_reg_dmabuf_mr auf einer kleinen VMM-Allokation der
#                    5090 (per PCI-Adresse 0A:00.0, nicht CUDA-Index), ohne
#                    Netzwerk/Gegenstelle.
#   (b) target    -- ein kleiner Transfer NIC->5090 mit Inhaltspruefung
#                    (FILLVAL/PATVAL+Sequenzmarken, cuMemcpyDtoH-Readback --
#                    das Muster aus gpurdma_03_transfer.c).
#   (c) source    -- ein Transfer mit der 5090 als RDMA-Quelle (umgekehrte
#                    Richtung, Inhaltspruefung per Host-memcmp).
# Ergebnis je Richtung als JA/NEIN (+errno bei Fehlschlag) als ERSTE Zeilen
# der Ergebnisdatei -- das macht die Zielrichtung zum garantierten
# Erstergebnis, falls das Fenster frueh abgebrochen werden muss. Die
# Uebergabe hat die 5090 bereits als Quelle UND als Registrierungsziel
# belegt; Stufe 0 wiederholt genau das, bevor Zeit in die Leitern faellt.
#
# DANACH drei Leiter-Arme in fester Reihenfolge, je Arm eine feste
# Groessenleiter (8 B/4 KiB/64 KiB/1 MiB), +/- Relaxed-Ordering, in beiden
# Schreibrichtungen. Topologie hart belegt per lspci -tv + nvidia-smi topo -m
# auf Rig 1: die NIC (ConnectX-4, 04:00.0) und die erste 3080 (05:00.0)
# haengen BEIDE am X570-Switch -> PIX zueinander. Die 5090 (0a:00.0) und die
# zweite 3080 (0b:00.0) haengen dagegen an eigenen CPU-Root-Ports -> PHB zur
# NIC (Pfad: NIC -> X570-Uplink -> Root Complex -> Karte). "RC-Arm" im
# Namen unten meint diesen PHB-Pfad, nicht "die NIC haengt am Root Complex".
#
#   (a) PIX-Arm    NIC <-> 3080 #1 (05:00.0), teilt den X570-Switch mit der NIC
#   (b) RC-Arm     NIC <-> 5090 (0a:00.0, volles 32-GiB-BAR1, PHB-Pfad --
#                  die W1-Frage aus docs/EVAL_gdr_uebernahme.md)
#   (c) RC-Arm 2   NIC <-> 3080 #2 (0b:00.0, 256-MiB-BAR1 wie 3080 #1,
#                  ebenfalls PHB-Pfad -- Kontrollkarte)
#
# Weil je Arm-Paar nur EINE Variable wechselt, ergeben sich zwei sauber
# getrennte Achsen, die compute_derived_differences() explizit ausrechnet
# (siehe dort):
#   TOPOLOGIE-ACHSE  3080 #1 (PIX) vs 3080 #2 (PHB)  -- gleiche Karte/BAR,
#                    NUR der Pfad unterscheidet sich -> Root-Complex-Umweg
#                    ALLEIN.
#   BAR-ACHSE        3080 #2 (PHB) vs 5090 (PHB)      -- gleicher Pfadtyp,
#                    NUR die BAR-Groesse unterscheidet sich -> Voll-BAR-
#                    Wirkung ALLEIN.
#
# Jeder Arm braucht immer nur EINE physische Karte gleichzeitig (die andere
# Seite jeder Messung ist Host-Speicher, ord=-1) -- das haelt den Fussabdruck
# klein und Kartenlocks kurz gehalten.
#
# Physische Geraete werden ausschliesslich ueber PCI-Adresse/UUID aufgeloest,
# NIEMALS ueber den CUDA-Index direkt (Device-Order-Falle: torch/CUDA-Default-
# Reihenfolge ("fastest first") ist nicht notwendig die PCI-Reihenfolge).
# Abhilfe: CUDA_DEVICE_ORDER=PCI_BUS_ID wird fuer beide Prozesse gesetzt --
# damit ist der CUDA-Ordinal exakt der nvidia-smi-Index (der seinerseits
# immer aufsteigend nach PCI-Bus-Id sortiert ist).
#
# Kartenlocks: identisches Protokoll wie
# python/sglang/srt/planner/comm_suite.py::_CardWindow (Runbook §7.1 v2) --
# gleiche Pfade, gleiches info-Dateiformat, damit dieses Skript mit der
# bestehenden Fenster-Disziplin interoperiert statt ein Parallelschema
# einzufuehren.
#
# Aufruf (Normalfall, ein Einzeiler):
#   ./gdr_window_run.sh
#
# Trockenlauf (KEIN GPU-Zugriff, KEIN echter Lock auf /tmp/gpu-card-*.lock):
#   ./gdr_window_run.sh --dry-run
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIN="${GDR_WINDOW_BIN:-$HERE/bin/gpurdma_04_bench}"
SMOKE_BIN="${GDR_WINDOW_SMOKE_BIN:-$HERE/bin/gpurdma_00_smoke}"
RESULTS_DIR="${GDR_WINDOW_RESULTS_DIR:-$HERE/results}"
NIC="${GDR_WINDOW_NIC:-rocep4s0f0}"
OWNER=gdr-window
FREE_MIB=512
WARMUP=200          # muss zu SIZES/ITERS im C-Programm passen (informativ)

# §7.1-Lockpfade -- identisch zu comm_suite.py, NICHT eigene Namen erfinden.
LOCK_DIR_FMT="${GDR_WINDOW_LOCK_PREFIX:-/tmp/gpu-card-}%s.lock"
LEGACY_LOCK_DIR="${GDR_WINDOW_LEGACY_LOCK:-/tmp/gpu-owner.lock}"
QUIET_LOCK_DIR="${GDR_WINDOW_QUIET_LOCK:-/tmp/gpu-quiet.lock}"
QUIET_STALE_S=900

DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      echo "Usage: $0 [--dry-run]"
      echo "  --dry-run: keine GPU/NIC-Nutzung, kein echter Kartenlock,"
      echo "             nur Argumentpruefung + Lock-Logik gegen Dummy-Pfade"
      echo "             (setze GDR_WINDOW_LOCK_PREFIX auf einen Testpfad)."
      exit 0
      ;;
    *) echo "[FAIL] unbekanntes Argument: $a" >&2; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_FILE="$RESULTS_DIR/gdr_window_${TS}.tsv"
LOG_FILE="$RESULTS_DIR/gdr_window_${TS}.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE" >&2; }

if [ "$DRY_RUN" = "0" ]; then
  if [ ! -x "$BIN" ]; then
    log "FAIL: $BIN fehlt oder ist nicht ausfuehrbar -- erst scripts/probe/build.sh laufen lassen"
    exit 1
  fi
  if [ ! -x "$SMOKE_BIN" ]; then
    log "FAIL: $SMOKE_BIN fehlt oder ist nicht ausfuehrbar -- erst scripts/probe/build.sh laufen lassen"
    exit 1
  fi
fi

# ===========================================================================
# §7.1-Kartenlocks -- ein Lock zur Zeit, siehe Kopfkommentar.
# ===========================================================================
declare -a HELD_LOCKS=()

lock_path() { printf "$LOCK_DIR_FMT" "$1"; }

owner_of() {
  local path="$1"
  if [ -r "$path/info" ]; then
    grep -m1 '^owner=' "$path/info" | cut -d= -f2-
  else
    echo "unknown"
  fi
}

release_all_locks() {
  local idx
  for idx in "${HELD_LOCKS[@]}"; do
    local p; p=$(lock_path "$idx")
    rm -f "$p/info" 2>/dev/null || true
    rmdir "$p" 2>/dev/null || true
  done
  HELD_LOCKS=()
}
trap release_all_locks EXIT

acquire_card_lock() {
  local idx="$1"
  if [ -d "$QUIET_LOCK_DIR" ]; then
    local age=$(( $(date +%s) - $(stat -c %Y "$QUIET_LOCK_DIR" 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$QUIET_STALE_S" ]; then
      log "FAIL: ein Quiet-Fenster ist offen ($(owner_of "$QUIET_LOCK_DIR"), ${age}s alt) -- §7.1 verbietet eine neue GPU-Phase"
      return 1
    fi
  fi
  if [ -d "$LEGACY_LOCK_DIR" ]; then
    log "FAIL: rig-weites Legacy-Lock gehalten von $(owner_of "$LEGACY_LOCK_DIR")"
    return 1
  fi
  local path; path=$(lock_path "$idx")
  if ! mkdir "$path" 2>/dev/null; then
    log "FAIL: Karte $idx wird gehalten von $(owner_of "$path") -- fremder Owner, Abbruch"
    return 1
  fi
  {
    echo "owner=$OWNER"
    echo "purpose=probe/gdr-window -- dmabuf GPU-RDMA Fensterlauf"
    echo "acquired=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
  } > "$path/info"
  HELD_LOCKS+=("$idx")

  if [ "$DRY_RUN" = "1" ]; then
    return 0   # Dummy-Lock-Test: kein nvidia-smi-Memory-Check auf echten Karten
  fi
  local used
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' -v i="$idx" '$1==i {print $2}')
  if [ -n "$used" ] && [ "$used" -gt "$FREE_MIB" ]; then
    log "FAIL: Lock fuer Karte $idx frei, aber Karte nicht frei: ${used} MiB belegt (>${FREE_MIB} MiB). Ein Host-seitiger Prozess ist hier unsichtbar -- memory.used ist der massgebliche Check (§7.1)."
    return 1
  fi
  return 0
}

release_card_lock() {
  local idx="$1"
  local path; path=$(lock_path "$idx")
  rm -f "$path/info" 2>/dev/null || true
  rmdir "$path" 2>/dev/null || true
  HELD_LOCKS=("${HELD_LOCKS[@]/$idx}")
}

# ===========================================================================
# Topologie-Aufloesung -- ausschliesslich ueber PCI-Adresse/NVML, nie ueber
# eine feste CUDA-Index-Annahme (Device-Order-Falle).
# ===========================================================================
resolve_topology() {
  if [ "$DRY_RUN" = "1" ]; then
    log "dry-run: ueberspringe nvidia-smi-Topologie-Aufloesung"
    PIX_PCI="05:00.0"; PIX_ORD="0"; PIX_NAME="(dry-run 3080)"
    RC5090_PCI="0a:00.0"; RC5090_ORD="1"; RC5090_NAME="(dry-run 5090)"
    RC3080B_PCI="0b:00.0"; RC3080B_ORD="2"; RC3080B_NAME="(dry-run 3080)"
    return 0
  fi

  local csv
  csv=$(nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader,nounits)
  log "nvidia-smi-Karten (Index -> PCI -> Name):"
  echo "$csv" | while IFS= read -r line; do log "  $line"; done

  local topo
  topo=$(nvidia-smi topo -m)

  # Spalte fuer die gewaehlte NIC in der Topo-Matrix finden (NIC0/NIC1 ->
  # tatsaechlicher rocepXsYfZ-Name steht in der Legende am Fussende).
  local nic_col
  nic_col=$(echo "$topo" | grep -A2 "NIC Legend" | grep -i "$NIC" | grep -oE 'NIC[0-9]+' | head -1)
  if [ -z "$nic_col" ]; then
    log "FAIL: NIC '$NIC' nicht in 'nvidia-smi topo -m' gefunden"
    return 1
  fi
  local header; header=$(echo "$topo" | sed -n '1p')
  local col_idx
  col_idx=$(echo "$header" | tr '\t' '\n' | grep -n "^${nic_col}\$" | cut -d: -f1)

  PIX_PCI=""; RC5090_PCI=""; RC3080B_PCI=""
  # bewusst OHNE "IFS=" vor read: hier wird auf Woerter gesplittet (idx/pci/
  # name als drei Felder), nicht ganze Zeilen gelesen. "IFS= read" wuerde
  # ueberhaupt nicht splitten und die ganze Zeile in $idx packen.
  while read -r idx pci name; do
    idx=$(echo "$idx" | tr -d ' ,')
    pci=$(echo "$pci" | tr -d ' ,' | tr 'A-F' 'a-f')
    # "00000000:0a:00.0" -> "0a:00.0" (letzte zwei Doppelpunkt-Felder)
    local shortpci; shortpci=$(echo "$pci" | awk -F: '{print $(NF-1)":"$NF}')
    local row
    row=$(echo "$topo" | grep -E "^GPU${idx}[[:space:]]")
    local rel
    rel=$(echo "$row" | tr -s '\t ' '\n' | sed -n "${col_idx}p")
    if [ "$rel" = "PIX" ]; then
      PIX_PCI="$shortpci"; PIX_ORD="$idx"; PIX_NAME="$name"
    elif echo "$name" | grep -qi "5090"; then
      RC5090_PCI="$shortpci"; RC5090_ORD="$idx"; RC5090_NAME="$name"
    else
      RC3080B_PCI="$shortpci"; RC3080B_ORD="$idx"; RC3080B_NAME="$name"
    fi
  done < <(echo "$csv" | awk -F', *' '{print $1, $2, $3}')

  if [ -z "$PIX_PCI" ] || [ -z "$RC5090_PCI" ] || [ -z "$RC3080B_PCI" ]; then
    log "FAIL: Topologie nicht wie erwartet aufgeloest (PIX=$PIX_PCI 5090=$RC5090_PCI 3080b=$RC3080B_PCI)."
    log "      Erwartet: genau eine 3080 PIX zur NIC, eine 5090 PHB, eine weitere 3080 PHB."
    log "      Kein Rueckfall auf feste Indizes -- bitte Hardware/Topologie pruefen."
    return 1
  fi
  log "aufgeloest: PIX-Arm=$PIX_NAME @ $PIX_PCI (ord=$PIX_ORD), RC-Arm(5090)=$RC5090_NAME @ $RC5090_PCI (ord=$RC5090_ORD), RC-Arm2(3080)=$RC3080B_NAME @ $RC3080B_PCI (ord=$RC3080B_ORD)"
}

# ===========================================================================
# setpci -- NUR LESEN. MaxPayload/MaxReadReq aus dem PCIe Device-Control-
# Register (CAP_EXP+08.w), Bits 5-7 bzw. 12-14, Werte 0..5 -> 128*2^n Bytes.
# Faellt sauber auf "n/a" zurueck, wenn die erweiterte Config-Space-Capability
# nicht lesbar ist (z.B. in einem Container ohne vollen PCI-Zugriff) --
# bricht den Lauf dafuer NICHT ab, das ist nur ein Protokoll-Zusatz.
# ===========================================================================
read_devctl() {
  local pci="$1"
  local raw
  raw=$(setpci -s "$pci" CAP_EXP+08.w 2>/dev/null) || { echo "n/a (setpci fehlgeschlagen/kein Zugriff)"; return 0; }
  if [ -z "$raw" ]; then echo "n/a (leere Antwort)"; return 0; fi
  local val=$((16#$raw))
  local mp_bits=$(( (val >> 5) & 0x7 ))
  local mrr_bits=$(( (val >> 12) & 0x7 ))
  echo "MaxPayload=$((128 << mp_bits))B MaxReadReq=$((128 << mrr_bits))B (raw=0x$raw)"
}

log_topology_facts() {
  {
    echo "# Topologie-Fakten (nur lesend, setpci/lspci), $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# device        pci           devctl"
    for pair in "NIC $NIC_PCI" "PIX-3080 $PIX_PCI" "RC-5090 $RC5090_PCI" "RC-3080b $RC3080B_PCI"; do
      local dev="${pair%% *}" pci="${pair#* }"
      echo "# $dev  $pci  $(read_devctl "$pci")"
    done
  } | tee -a "$RESULTS_FILE" >&2
}

# ===========================================================================
# Auf ein gebundenes+lauschendes TCP-Port warten, statt auf eine Stdout-
# Statuszeile zu greppen: beide Tools drucken ihre Statuszeile ERST NACH dem
# vollstaendigen QP-Handshake (der seinerseits den bereits verbundenen Client
# voraussetzt) -- ein grep auf diese Zeile wuerde also IMMER auf das
# Timeout laufen, weil der Client noch gar nicht gestartet ist. bind()+
# listen() passieren dagegen vor dem GPU-Setup-abhaengigen accept()-Aufruf
# nur beim Client-Fall, beim Server-Fall (ord>=0) sogar erst NACH dem
# dmabuf-Setup -- deshalb hier auf den Socket-Zustand pollen, nicht auf eine
# feste Sleep-Zeit raten (CUDA-Kaltstart kann je nach Treiberzustand length
# variieren).
# ===========================================================================
wait_for_listen() {
  local port="$1" pid="$2" timeout_s="${3:-10}"
  local waited=0
  while true; do
    if ss -ltn 2>/dev/null | awk -v p=":$port\$" '$4 ~ p {f=1} END{exit !f}'; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 0.1; waited=$((waited + 1))
    if [ "$waited" -gt $((timeout_s * 10)) ]; then
      return 1
    fi
  done
}

# ===========================================================================
# Ein Server/Client-Paar, eine Groessenleiter, Ergebniszeilen anhaengen.
# ===========================================================================
run_pair() {
  local arm="$1" gpu_pci="$2" gpu_ord="$3" direction="$4" modus="$5" ro="$6"
  local port=$((23456 + RANDOM % 5000))
  local ro_flag=""; [ "$ro" = "on" ] && ro_flag="--ro"

  local server_ord client_ord
  if [ "$direction" = "gpu_is_server" ]; then
    server_ord="$gpu_ord"; client_ord="-1"
  else
    server_ord="-1"; client_ord="$gpu_ord"
  fi

  local srv_log; srv_log=$(mktemp)
  local cli_log; cli_log=$(mktemp)

  CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$server_ord" "$NIC" "$port" "$modus" $ro_flag \
    > "$srv_log" 2>&1 &
  local srv_pid=$!

  if ! wait_for_listen "$port" "$srv_pid" 10; then
    log "FAIL: Server kam nicht hoch/lauscht nicht auf Port $port ($arm/$direction/$modus/ro=$ro): $(cat "$srv_log")"
    kill "$srv_pid" 2>/dev/null || true
    rm -f "$srv_log" "$cli_log"
    return 1
  fi

  if ! CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC" "$port" "$modus" "$client_ord" $ro_flag \
       > "$cli_log" 2>&1; then
    log "FAIL: Client-Lauf fehlgeschlagen ($arm/$direction/$modus/ro=$ro): $(tail -3 "$cli_log")"
    kill "$srv_pid" 2>/dev/null || true
    wait "$srv_pid" 2>/dev/null || true
    rm -f "$srv_log" "$cli_log"
    return 1
  fi
  kill "$srv_pid" 2>/dev/null || true
  wait "$srv_pid" 2>/dev/null || true

  # Zeilen wie: "GDR             8 B   p10     3.21 us   MEDIAN     3.45 us   p90     3.89 us"
  awk -v arm="$arm" -v pci="$gpu_pci" -v dir="$direction" -v modus="$modus" -v ro="$ro" \
      'match($0, /^(GDR|STAGE)[[:space:]]+([0-9]+) B[[:space:]]+p10[[:space:]]+([0-9.]+) us[[:space:]]+MEDIAN[[:space:]]+([0-9.]+) us[[:space:]]+p90[[:space:]]+([0-9.]+) us/, m) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", arm, pci, dir, modus, ro, m[2], m[3], m[4], m[5]
      }' "$cli_log" >> "$RESULTS_FILE"

  rm -f "$srv_log" "$cli_log"
}

# ===========================================================================
# STUFE 0 -- Smoke-Test 5090<->CX-4, "geht es hier ueberhaupt", IMMER ZUERST.
# Schreibt seine JA/NEIN-Zeilen als ERSTES in die Ergebnisdatei (vor der
# Leiter-Tabelle), damit die Ziel-Richtung ein garantiertes Erstergebnis ist,
# falls das Fenster frueh abgebrochen werden muss.
# ===========================================================================
smoke_result_line() {
    # Extrahiert "RESULT <name> PASS|FAIL errno=<n> <detail...>" aus einem Log
    # und schreibt eine TSV-Zeile ins Ergebnisfile. Fehlt die Zeile ganz (Absturz
    # vor dem ersten result()-Aufruf), wird das explizit als FAIL protokolliert.
    local phase="$1" log_file="$2"
    local line; line=$(grep -m1 '^RESULT ' "$log_file" 2>/dev/null || true)
    if [ -z "$line" ]; then
        echo -e "smoke\t$phase\tFAIL\tn/a\tkein RESULT im Log (Absturz/Timeout vor Ergebnis): $(tail -2 "$log_file" | tr '\n' ' ')" \
            >> "$RESULTS_FILE"
        return 1
    fi
    # RESULT <name> PASS|FAIL errno=<n> <detail...>
    echo "$line" | awk -v phase="$phase" \
        '{ name=$2; verdict=$3; err=$4; sub(/^errno=/,"",err);
           detail=""; for(i=5;i<=NF;i++) detail = detail (i>5?" ":"") $i;
           printf "smoke\t%s\t%s\t%s\t%s\n", phase, verdict, err, detail }' \
        >> "$RESULTS_FILE"
    grep -q '^RESULT .* PASS ' <<<"$line"
}

run_smoke_stage0() {
    log "STUFE 0: Smoke-Test 5090<->CX-4 (regcheck, target, source)"
    {
        echo "# STUFE 0 -- Smoke-Test 5090<->CX-4 (immer zuerst, vor den Leitern)"
        echo "# phase	check	verdict	errno	detail"
    } >> "$RESULTS_FILE"

    if [ "$DRY_RUN" = "1" ]; then
        echo -e "smoke\tregcheck\tSKIP\tn/a\tdry-run, kein GPU-Zugriff" >> "$RESULTS_FILE"
        echo -e "smoke\ttarget(NIC->5090)\tSKIP\tn/a\tdry-run, kein GPU-Zugriff" >> "$RESULTS_FILE"
        echo -e "smoke\tsource(5090->NIC)\tSKIP\tn/a\tdry-run, kein GPU-Zugriff" >> "$RESULTS_FILE"
        return 0
    fi

    if ! acquire_card_lock "$RC5090_ORD"; then
        echo -e "smoke\tall\tFAIL\tn/a\tKartenlock fuer 5090 (ord=$RC5090_ORD) nicht verfuegbar, siehe Log" >> "$RESULTS_FILE"
        return 1
    fi

    # (a) regcheck: kein Netzwerk, keine Gegenstelle.
    local rc_log; rc_log=$(mktemp)
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" regcheck "$RC5090_ORD" "$NIC" > "$rc_log" 2>&1 || true
    smoke_result_line "regcheck" "$rc_log" || log "  regcheck: FAIL, siehe $RESULTS_FILE"
    rm -f "$rc_log"

    # (b) target: NIC -> 5090 (Server haelt die GPU).
    local port1=$((23456 + RANDOM % 5000))
    local srv1; srv1=$(mktemp); local cli1; cli1=$(mktemp)
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" server "$RC5090_ORD" "$NIC" "$port1" > "$srv1" 2>&1 &
    local srv1_pid=$!
    if wait_for_listen "$port1" "$srv1_pid" 10; then
        CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" client 127.0.0.1 "$NIC" "$port1" -1 > "$cli1" 2>&1 || true
    else
        echo "kein Listener" >> "$srv1"
    fi
    kill "$srv1_pid" 2>/dev/null || true; wait "$srv1_pid" 2>/dev/null || true
    # Das massgebliche Urteil (Inhalt angekommen) faellt der SERVER (er sieht
    # die Empfangsseite); der Client meldet nur "Write abgeschickt".
    smoke_result_line "target(NIC->5090)" "$srv1" || log "  target: FAIL, siehe $RESULTS_FILE"
    rm -f "$srv1" "$cli1"

    # (c) source: 5090 -> NIC (Client haelt die GPU, Server ist Host).
    local port2=$((23456 + RANDOM % 5000))
    local srv2; srv2=$(mktemp); local cli2; cli2=$(mktemp)
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" server -1 "$NIC" "$port2" > "$srv2" 2>&1 &
    local srv2_pid=$!
    if wait_for_listen "$port2" "$srv2_pid" 10; then
        CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" client 127.0.0.1 "$NIC" "$port2" "$RC5090_ORD" > "$cli2" 2>&1 || true
    else
        echo "kein Listener" >> "$srv2"
    fi
    kill "$srv2_pid" 2>/dev/null || true; wait "$srv2_pid" 2>/dev/null || true
    # Hier ist der SERVER (Host-Ziel) massgeblich, gleiches Prinzip.
    smoke_result_line "source(5090->NIC)" "$srv2" || log "  source: FAIL, siehe $RESULTS_FILE"
    rm -f "$srv2" "$cli2"

    release_card_lock "$RC5090_ORD"
    echo "" >> "$RESULTS_FILE"
}

run_arm() {
  local arm="$1" gpu_pci="$2" gpu_ord="$3"
  log "Arm '$arm' (PCI $gpu_pci, ord $gpu_ord): nehme Kartenlock $gpu_ord"
  if [ "$DRY_RUN" = "1" ]; then
    if ! acquire_card_lock "$gpu_ord"; then return 1; fi
    log "dry-run: ueberspringe echte Server/Client-Laeufe fuer Arm '$arm'"
    release_card_lock "$gpu_ord"
    return 0
  fi
  if ! acquire_card_lock "$gpu_ord"; then return 1; fi
  local direction modus ro
  for direction in gpu_is_server gpu_is_client; do
    for modus in gdr stage; do
      for ro in off on; do
        log "  -> $arm dir=$direction modus=$modus ro=$ro"
        run_pair "$arm" "$gpu_pci" "$gpu_ord" "$direction" "$modus" "$ro" || true
      done
    done
  done
  release_card_lock "$gpu_ord"
}

# ===========================================================================
# Abgeleitete Differenzen -- zwei Achsen, je nur EINE Variable wechselt
# (Topologie-Fakten hart belegt per lspci -tv + nvidia-smi topo -m: NIC und
# 3080 #1/05:00.0 haengen BEIDE am X570-Switch -> PIX; 5090/0a:00.0 und
# 3080 #2/0b:00.0 haengen an eigenen CPU-Root-Ports -> PHB zur NIC, Pfad
# NIC -> X570-Uplink -> Root Complex -> Karte):
#
#   TOPOLOGIE-ACHSE: 3080 #1 (PIX) vs 3080 #2 (PHB) -- gleiche Karte, gleiche
#     256-MiB-BAR, NUR der Pfad unterscheidet sich -> beziffert den
#     Root-Complex-Umweg ALLEIN.
#   BAR-ACHSE: 3080 #2 (PHB, 256 MiB) vs 5090 (PHB, 32 GiB) -- gleicher
#     Pfadtyp, NUR die BAR-Groesse unterscheidet sich -> beziffert die
#     Voll-BAR-Wirkung ALLEIN.
#
# Beide Differenzen werden je Leiter-Groesse (und je direction/modus/ro)
# explizit ausgerechnet und als eigene Zeilen ausgewiesen, nicht nur implizit
# in den drei Rohreihen belassen. diff_us = B - A (negativ = A schneller).
# ===========================================================================
compute_derived_differences() {
  log "Berechne abgeleitete Differenzen (Topologie-Achse, BAR-Achse)"
  local snapshot; snapshot=$(mktemp)
  cp "$RESULTS_FILE" "$snapshot"
  {
    echo ""
    echo "# Abgeleitete Differenzen -- Topologie-Achse (3080 PIX vs 3080 PHB,"
    echo "# gleiche Karte/BAR, isoliert NUR den Root-Complex-Umweg) und"
    echo "# BAR-Achse (3080 PHB vs 5090 PHB, gleicher Pfadtyp PHB, isoliert NUR"
    echo "# die Voll-BAR-Wirkung). diff_us = B - A (negativ = A schneller)."
    echo "# axis	direction	modus	ro	size_bytes	a_arm	a_median_us	b_arm	b_median_us	diff_us	diff_pct"
  } >> "$RESULTS_FILE"

  local derived; derived=$(awk -F'\t' '
    $1=="pix" || $1=="rc5090" || $1=="rc3080b" {
      key = $3 SUBSEP $4 SUBSEP $5 SUBSEP $6
      median[$1, key] = $8
      have[key] = 1
    }
    END {
      for (k in have) {
        split(k, parts, SUBSEP)
        dir=parts[1]; modus=parts[2]; ro=parts[3]; size=parts[4]
        if ((("pix" SUBSEP k) in median) && (("rc3080b" SUBSEP k) in median)) {
          a = median["pix" SUBSEP k]; b = median["rc3080b" SUBSEP k]
          diff = b - a; pct = (a+0 != 0) ? (diff/a*100) : 0
          printf "topology\t%s\t%s\t%s\t%s\tpix\t%.2f\trc3080b\t%.2f\t%.2f\t%.1f\n", dir, modus, ro, size, a, b, diff, pct
        }
        if ((("rc3080b" SUBSEP k) in median) && (("rc5090" SUBSEP k) in median)) {
          a = median["rc3080b" SUBSEP k]; b = median["rc5090" SUBSEP k]
          diff = b - a; pct = (a+0 != 0) ? (diff/a*100) : 0
          printf "bar\t%s\t%s\t%s\t%s\trc3080b\t%.2f\trc5090\t%.2f\t%.2f\t%.1f\n", dir, modus, ro, size, a, b, diff, pct
        }
      }
    }
  ' "$snapshot" | sort)
  rm -f "$snapshot"

  if [ -z "$derived" ]; then
    echo "# (keine Leiterzeilen vorhanden -- dry-run oder alle Sub-Runs fehlgeschlagen)" >> "$RESULTS_FILE"
  else
    echo "$derived" >> "$RESULTS_FILE"
  fi
}

# ===========================================================================
# Main
# ===========================================================================
log "gdr_window_run.sh Start (dry_run=$DRY_RUN)"
log "Binary: $BIN"
log "Ergebnisse: $RESULTS_FILE"

resolve_topology || exit 1
NIC_PCI=$(lspci -d 15b3: 2>/dev/null | grep "^04:00.0" | awk '{print $1}')
NIC_PCI="${NIC_PCI:-04:00.0}"

echo "# GDR-Fensterlauf $TS -- dmabuf GPU-RDMA Sondierung" > "$RESULTS_FILE"

# STUFE 0 zuerst -- siehe Kopfkommentar: garantiertes Erstergebnis, falls das
# Fenster danach abgebrochen werden muss. "|| true": ein Fehlschlag hier
# (z.B. 5090-Lock fremd belegt) darf die Arme auf den ANDEREN Karten nicht
# mitreissen -- unter "set -e" wuerde ein ungeschuetzter Fehlschlag sonst das
# gesamte Skript sofort beenden.
run_smoke_stage0 || log "STUFE 0 mit Fehler beendet, siehe Ergebnisdatei -- Leiter-Arme laufen trotzdem weiter"

echo "# arm	gpu_pci	direction	modus	ro	size_bytes	p10_us	median_us	p90_us" >> "$RESULTS_FILE"
log_topology_facts

# Jeder Arm ist unabhaengig (eigene Karte) -- "|| true" aus demselben Grund
# wie oben: ein Lock-Fehlschlag auf EINEM Arm darf die anderen nicht killen.
run_arm "pix"    "$PIX_PCI"    "$PIX_ORD"    || log "Arm 'pix' mit Fehler beendet, siehe Log"
run_arm "rc5090" "$RC5090_PCI" "$RC5090_ORD" || log "Arm 'rc5090' mit Fehler beendet, siehe Log"
run_arm "rc3080b" "$RC3080B_PCI" "$RC3080B_ORD" || log "Arm 'rc3080b' mit Fehler beendet, siehe Log"

compute_derived_differences

log "fertig. Ergebnistabelle: $RESULTS_FILE"
log "Log: $LOG_FILE"
