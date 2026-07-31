#!/bin/bash
# SPDX-License-Identifier: MIT
#
# crossrig_ladder_run.sh -- Cross-Rig-Groessenleiter: RDMA direkt in/aus
# GPU-Speicher gegen Host-Staging, ueber die Rechnergrenze (40G-RoCE-Strecke
# zwischen Rig 1 und Rig 2).
#
# ===========================================================================
# WOZU -- die zu klaerende Behauptung
# ===========================================================================
# Die Nutzer-Uebergabe (/spinning/gdr-uebergabe/GDR_BEFUNDE.md, Abschnitt 6.2)
# behauptet fuer die Strecke ueber die Rechnergrenze einen Umschlagpunkt: der
# direkte Weg gewinne nur unter 4 KiB und sei bei 1 MiB 3,4-fach langsamer als
# Host-Staging. Diese Zahlen wurden von uns nie reproduziert. Die frische
# Intra-Rig-Leiter (gdr_window_run.sh, gleiches Binary) zeigt KEINEN Umschlag --
# dort gewinnt der direkte Weg auf jeder Groesse.
#
# Zwei Teilfragen, die dieses Skript trennt:
#   (a) Gibt es den Cross-Rig-Umschlag ueberhaupt?
#   (b) Falls ja: ist er ein HARTER Bandbreiten-Deckel des PCIe-Peer-Pfades
#       (die NIC schreibt/liest ueber den Root Complex in die BAR-Apertur der
#       Karte), oder ein WEICHER Rueckstau (zu wenige ausstehende
#       Transaktionen), den man mit Pipelining-Tiefe mildern kann?
# Frage (b) beantwortet die Tiefen-Achse (--depth) direkt: sinkt die Zeit je
# Nachricht mit steigender Tiefe, war es Rueckstau; bleibt sie konstant, ist
# der Deckel hart.
#
# ===========================================================================
# PRUEFBARE VORHERSAGEN (vor der Messung notiert, damit sie falsifizierbar
# bleiben -- nicht hinterher passend erzaehlt)
# ===========================================================================
# V1  Richtung "NIC liest aus der BAR" (VRAM ist QUELLE) ist teurer als
#     "NIC schreibt in die BAR" (VRAM ist ZIEL), weil Lesen non-posted ist
#     (jede Leseanforderung wartet auf Daten), Schreiben posted.
# V2  Der Aufschlag aus V1 trifft die PIX-Karte (3080 #1, 05:00.0, teilt den
#     X570-Switch mit der NIC) deutlich staerker als die beiden Karten an
#     eigenen CPU-Root-Ports (5090 0a:00.0, 3080 #2 0b:00.0). Grundlage: die
#     Intra-Rig-Messung zeigt fuer non-posted Reads ueber den Switch +53 %.
#     Wenn V2 faellt, war der Switch-Aufschlag ein Intra-Rig-Artefakt.
# V3  Falls der Umschlag aus der Uebergabe existiert, sollte er BAR-abhaengig
#     sein: die 5090 (32-GiB-BAR, volle Apertur) muesste ihn spaeter oder gar
#     nicht zeigen, die 3080er (256-MiB-BAR) frueher. Die Uebergabe hat nur
#     mit einer 3080 gemessen.
# V4  Rig-2-GPU-Seite: GDR_BEFUNDE.md Abschnitt 7 erklaert die 2080 Ti fuer
#     untauglich als Quelle ("local protection error" bei jedem Zugriff,
#     Verdacht: Topologie/Peer-Routing des Cezanne-Root-Complex). Dieses
#     Verdikt entstand VOR dem iommu=pt-Fix auf Rig 2 -- die damaligen Fehler
#     koennen ganz oder teilweise AMD-Vi-Faults gewesen sein. Behandelt als
#     offene Frage, nicht als Fakt: Phase "ti" faengt deshalb mit regcheck +
#     verifiziertem Klein-Transfer + Fault-Zaehler an. Geht es jetzt, ist der
#     alte Befund ueberholt; scheitert es weiter, wird der exakte Fehler
#     dokumentiert -- mit hartem Deckel von 2-3 Diagnoseversuchen, kein
#     Debug-Marathon.
#
# ===========================================================================
# AUFBAU
# ===========================================================================
# Laeuft auf dem RIG-1-HOST (Proxmox), nicht im Container: nur dort liegen
# NIC, GPUs und die ssh-Kennung fuer Rig 2.
#
#   Rig 1  10.10.10.1  NIC rocep4s0f1  GPUs: 3080 (05:00.0, PIX zur NIC),
#                                            5090 (0a:00.0, eigener Root-Port),
#                                            3080 (0b:00.0, eigener Root-Port)
#   Rig 2  10.10.10.2  NIC rocep1s0f1 (ConnectX-5 Ex am CPU-Root-Port)
#                      GPU: 2080 Ti (04:00.0, hinter der Chipsatz-Bruecke)
#
# Das Bench-Programm ist ein Ping-Pong ueber ein RC-QP: der CLIENT ist immer
# der Sender (er schreibt seine Nutzlast per RDMA-WRITE in den Puffer des
# SERVERS), der Server antwortet mit einem 8-Byte-Flag. Daraus folgt die
# Richtungslogik:
#   a2b  Rig 1 -> Rig 2   Client auf Rig 1, Server auf Rig 2
#   b2a  Rig 2 -> Rig 1   Client auf Rig 2, Server auf Rig 1
# Liegt der Endpunkt einer Seite im VRAM, ist diese Seite in Richtung "sendet"
# die BAR-LESEN-Seite und in Richtung "empfaengt" die BAR-SCHREIBEN-Seite --
# genau die Achse aus V1.
#
# Modus:
#   gdr    Wer ein GPU-Ordinal hat, registriert eine dmabuf-MR auf dem VRAM;
#          die NIC greift direkt zu.
#   stage  Wer ein GPU-Ordinal hat, benutzt eine Host-MR und kopiert je Runde
#          mit der Copy-Engine (DtoH beim Sender, HtoD beim Empfaenger) --
#          der heute gefahrene Weg. Haben BEIDE Seiten eine GPU, kostet stage
#          zwei Kopien, sonst eine.
#   Sind beide Endpunkte Host-RAM, sind gdr und stage derselbe Codepfad; die
#   Draht-Basislinie faehrt deshalb nur EINEN davon und heisst "wire".
#
# ===========================================================================
# PHASEN (--phase)
# ===========================================================================
#   wire     Draht-Basislinie Host-RAM <-> Host-RAM ueber dasselbe Portpaar.
#            KEINE GPU, KEINE Kartenlocks, blockiert niemanden. Das ist die
#            Obergrenze, gegen die jeder Staging-Wert zu lesen ist.
#   neutral  A-vs-A-artiger Neutralitaetstest: Alt-Binary (zwei getrennte
#            ibv_post_send fuer Nutzlast und Flag) gegen Neu-Binary (eine
#            verkettete Postung) bei Tiefe 1, beide Host-RAM. Beziffert, wie
#            viel der Codepfad-Wechsel allein ausmacht, damit die Tiefen-Achse
#            nicht auf einem unbezifferten Sockel steht. Ebenfalls ohne GPU.
#   ti       Alle Paare mit der 2080 Ti auf Rig 2, deren Rig-1-Endpunkt
#            HOST-RAM ist. Braucht deshalb KEIN Rig-1-GPU-Fenster und keine
#            Kartenlocks. Beginnt mit regcheck + verifiziertem Klein-Transfer
#            (V4).
#   gpu      Alles mit Rig-1-VRAM: die drei Karten je in beiden Richtungen
#            gegen Rig-2-Host-RAM, die Tiefen-Achse, und VRAM<->VRAM gegen die
#            2080 Ti. NUR mit Kartenlock und nur im freigegebenen Fenster.
#   intra    Rig-1-intern, jede Karte gegen jede (3 Paare, beide Richtungen):
#            NIC-Relay GPU_A -> CX-4 -> GPU_B direkt gegen NIC-Relay mit
#            Host-Staging, plus der ECHTE heutige Pfad (NCCL ueber System-RAM)
#            als dritter Arm. Kartenlocks noetig (zwei Karten gleichzeitig).
#   pressure Parallel-Druck: alle drei Paare gleichzeitig, je EIN kurzer Burst
#            decode-artig (kleine Nachrichten) und prefill-artig (grosse),
#            solo gegen parallel. Alle drei Karten gleichzeitig gelockt.
#   all      wire, neutral, ti, gpu, intra, pressure in dieser Reihenfolge.
#
# Die Phasen intra und pressure aendern EINE Sache am System und machen sie
# rueckgaengig: der Intra-Rig-Loopback braucht eine IPv4 auf dem sonst
# IPv4-losen Port (RoCEv2-GID). Sie wird vor der Phase gesetzt und im
# EXIT-Trap wieder entfernt -- auch bei Abbruch. Es wird nichts anderes am
# Netz veraendert und nichts installiert.
#
# ===========================================================================
# AUSDUENNUNG DER KOMBINATORIK -- explizit, kein stilles Cap
# ===========================================================================
# Die volle Kreuzung (5 Karten-Endpunkte x 2 Richtungen x 2 Modi x 2 RO x 3
# Tiefen x 9 Groessen) waere Stunden. Gestrichen wird nach diesem Prinzip:
# jede Achse wird an der Stelle voll ausgefahren, wo sie die Kernfrage
# beantwortet, und an allen anderen auf ihren Referenzwert festgenagelt.
#
#   RO-Achse (+/-RO):  voll auf allen Paaren Rig-1-VRAM <-> Rig-2-Host-RAM und
#                      auf der Draht-Basislinie. FESTGENAGELT auf "off" bei
#                      VRAM<->VRAM und bei der Tiefen-Achse. Begruendung: die
#                      Intra-Rig-Leiter zeigt RO-Effekte unterhalb des
#                      Rauschbodens (<0,1 %); RO bleibt trotzdem auf der
#                      Hauptachse drauf, weil genau dort das BAR-Fenster im
#                      Spiel ist.
#   Tiefen-Achse (1/4/16): NUR auf der 5090, beide Richtungen, beide Modi.
#                      Begruendung: die Tiefe beantwortet "harter Deckel oder
#                      Rueckstau" -- das ist eine Pfad-Eigenschaft, keine
#                      Karten-Eigenschaft, und die 5090 ist die Karte mit der
#                      vollen BAR, also der Fall, bei dem ein weicher
#                      Rueckstau am ehesten sichtbar wird. Auf allen anderen
#                      Karten Tiefe 1.
#   VRAM<->VRAM:       beide Richtungen, beide Modi, RO off, Tiefe 1.
#   Groessenleiter:    immer voll (9 Punkte), inklusive der Kollektivgroessen
#                      20 KiB und 80 KiB -- das ist die Achse der Kernfrage
#                      und wird nirgends gekuerzt.
#
# Zusaetzlich fuer die Intra-Rig-Phasen, Vorgabe "kein Benchmark-Marathon,
# lieber 3 aussagekraeftige Groessen sauber als 10 duenn":
#   Groessen intra:    NUR 20 KiB / 80 KiB / 1 MiB (zwei Kollektivgroessen und
#                      ein Prefill-artiger Punkt) statt der 9-Punkt-Leiter.
#   RO intra:          GANZ WEGGELASSEN. Begruendung: die Intra-Rig-Leiter von
#                      heute zeigt den RO-Effekt unter 0,1 %, also unter dem
#                      Rauschboden -- die Achse traegt hier nichts und kostet
#                      die Haelfte der Laufzeit.
#   Tiefe intra:       fest 1.
#   Rauschboden:       ein A-vs-A-Wiederholpunkt je Phase (dasselbe Paar, derselbe
#                      Modus, zweimal gefahren) -- ohne ihn ist kein Unterschied
#                      als echt behauptbar.
#   Parallel-Druck:    genau ZWEI Szenarien (decode-artig 80 KiB, prefill-artig
#                      1 MiB), je ein kurzer Burst, keine Groessenachse darin.
#
# ===========================================================================
# BETRIEBSDISZIPLIN
# ===========================================================================
# - Reste-Check vor JEDEM Start auf BEIDEN Seiten: ein ssh-Abbruch toetet den
#   fernen Prozess NICHT. Gesucht wird ausschliesslich nach dem eigenen
#   Binary-Namen (gpurdma_04_bench*), nie breit gekillt.
# - Fault-Zaehler (AMD-Vi/DMAR aus dmesg) vor und nach jeder Phase auf beiden
#   Seiten; jede Zunahme wird in die Ergebnisdatei geschrieben.
# - Kartenlocks nur in Phase "gpu", Format identisch zu comm_suite.py
#   (/tmp/gpu-card-<idx>.lock + info mit owner=).
# - Physische Karten werden ueber die PCI-Adresse aufgeloest, nie ueber eine
#   feste CUDA-Index-Annahme; CUDA_DEVICE_ORDER=PCI_BUS_ID fuer alle Prozesse.
# - Es wird nichts installiert, nichts an der Netzkonfiguration geaendert,
#   nichts persistent veraendert.
#
# Aufruf:
#   ./crossrig_ladder_run.sh --phase wire
#   ./crossrig_ladder_run.sh --phase gpu          (nur im freigegebenen Fenster)
#   ./crossrig_ladder_run.sh --phase all --dry-run
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- Rig 1 (lokal) --------------------------------------------------------
BIN="${CR_BIN:-$HERE/bin/gpurdma_04_bench}"
BIN_PRE="${CR_BIN_PRE:-$HERE/bin/gpurdma_04_bench_pre}"
SMOKE_BIN="${CR_SMOKE_BIN:-$HERE/bin/gpurdma_00_smoke}"
NIC1="${CR_NIC1:-rocep4s0f1}"
IP1="${CR_IP1:-10.10.10.1}"

# ---- Rig 2 (fern) ---------------------------------------------------------
R2_SSH_KEY="${CR_R2_KEY:-/root/.ssh/id_rig2}"
R2_HOST="${CR_R2_HOST:-10.10.10.2}"
IP2="${CR_IP2:-10.10.10.2}"
NIC2="${CR_NIC2:-rocep1s0f1}"
R2_DIR="${CR_R2_DIR:-/root/gdr-crossrig}"
R2_BIN="${CR_R2_BIN:-$R2_DIR/bin/gpurdma_04_bench_cuda}"      # CUDA-Zweig, kann auch ord=-1
R2_BIN_HOST="$R2_DIR/bin/gpurdma_04_bench"      # Host-Zweig, nur ord=-1
R2_BIN_PRE="$R2_DIR/bin/gpurdma_04_bench_pre"
R2_SMOKE="$R2_DIR/bin/gpurdma_00_smoke_cuda"

RESULTS_DIR="${CR_RESULTS_DIR:-$HERE/results}"
OWNER=agent-crossrig-ladder
FREE_MIB=512
LOCK_DIR_FMT="${CR_LOCK_PREFIX:-/tmp/gpu-card-}%s.lock"
LEGACY_LOCK_DIR="${CR_LEGACY_LOCK:-/tmp/gpu-owner.lock}"
QUIET_LOCK_DIR="${CR_QUIET_LOCK:-/tmp/gpu-quiet.lock}"
QUIET_STALE_S=900

# Groessenleiter: 8 B .. 4 MiB, inklusive der Kollektivgroessen 20/80 KiB.
SIZES_SMALL="8,4096,16384,20480,65536,81920"
SIZES_LARGE="262144,1048576,4194304"
ITERS_SMALL=2000;  WARMUP_SMALL=200
ITERS_LARGE=300;   WARMUP_LARGE=50

# Intra-Rig: drei Groessen, kurze Laeufe (siehe Ausduennung im Kopf).
SIZES_INTRA="20480,81920,1048576"
ITERS_INTRA=400;   WARMUP_INTRA=50
# Intra-Rig-Loopback: derselbe Port wie im bisherigen Fensterlauf. Er hat
# keine IPv4, RoCEv2 braucht aber eine fuer den GID -- deshalb temporaer
# setzen und im Trap wieder entfernen.
NIC_INTRA="${CR_NIC_INTRA:-rocep4s0f0}"
IF_INTRA="${CR_IF_INTRA:-enp4s0f0np0}"
IP_INTRA="${CR_IP_INTRA:-169.254.77.1/24}"
IP_INTRA_ADDED=0
# Parallel-Druck
PRESSURE_DECODE=81920
PRESSURE_PREFILL=1048576
PRESSURE_ITERS=300

# ---- Messdauer je Punkt -------------------------------------------------
# CR_SECS>0 schaltet das Zeitbudget im Bench ein: je Groesse wird nach dem
# (unveraenderten) Warmup so lange gemessen, bis das Budget voll ist. Das
# ersetzt die feste Iterationszahl -- gegen P-State-Rampen und Ausreisser
# zaehlt Dauer, nicht Rundenzahl.
CR_SECS="${CR_SECS:-0}"
SECS_OPT=""
case "$CR_SECS" in
  0|0.0|"") ;;
  *) SECS_OPT="--secs=$CR_SECS" ;;
esac

# CR_LONG=1: Ausduennung fuer den Lauf mit langer Messdauer. Die RO-Achse
# faellt weg (im 1-s-Lauf durchgehend unter 0,2 % gemessen, also unter dem
# Rauschboden -- die Daten dazu liegen bereits vor und werden nicht besser,
# wenn man sie 5-mal so lange misst), und die Tiefen- sowie VRAM<->VRAM-Arme
# laufen auf der 3-Punkt-Leiter statt auf 9 Punkten. Ohne diese Ausduennung
# braeuchte allein die gpu-Phase bei 5 s je Punkt ueber eine Stunde.
CR_LONG="${CR_LONG:-0}"
if [ "$CR_LONG" = "1" ]; then RO_LIST="off"; else RO_LIST="off on"; fi

PHASE=all
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --phase) PHASE="${2:-}"; shift 2 ;;
    --phase=*) PHASE="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
      sed -n '1,140p' "$0" | sed -n '/^# Aufruf:/,$p'
      echo "Phasen: wire | neutral | ti | gpu | all"
      exit 0 ;;
    *) echo "[FAIL] unbekanntes Argument: $1" >&2; exit 1 ;;
  esac
done
case "$PHASE" in wire|neutral|ti|gpu|intra|pressure|mix|loadsym|all) ;; *)
  echo "[FAIL] --phase muss wire|neutral|ti|gpu|intra|pressure|mix|loadsym|all sein" >&2; exit 1 ;;
esac

# Trockenlauf fasst weder den echten Lock-Namensraum noch das Netz an.
if [ "$DRY_RUN" = "1" ]; then
  LOCK_DIR_FMT="/tmp/cr-dryrun-card-%s.lock"
  LEGACY_LOCK_DIR="/tmp/cr-dryrun-none.lock"
  QUIET_LOCK_DIR="/tmp/cr-dryrun-none-quiet.lock"
fi

mkdir -p "$RESULTS_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS_FILE="${CR_RESULTS_FILE:-$RESULTS_DIR/crossrig_${PHASE}_${TS}.tsv}"
LOG_FILE="${CR_LOG_FILE:-$RESULTS_DIR/crossrig_${PHASE}_${TS}.log}"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG_FILE" >&2; }
note() { echo "$*" >> "$RESULTS_FILE"; }

R2() { ssh -n -i "$R2_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
           -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
           "root@$R2_HOST" "$@"; }

# Fernen Hintergrundprozess starten und die PID zurueckgeben.
#
# Warum nicht einfach 'ssh "... & echo $!"': ssh haelt die Sitzung offen,
# solange irgendein ferner Prozess die Kanal-Fds (stdout/stderr) geerbt hat.
# Der gestartete Server selbst hat seine Fds zwar umgeleitet, die ihn
# startende Hintergrund-Subshell aber nicht -- das Ergebnis war ein ssh, das
# nie zurueckkommt und das Skript genau an dieser Stelle einfriert (statt zu
# messen). Deshalb: die GESAMTE Hintergrundgruppe von den Kanal-Fds trennen,
# die PID in eine Datei schreiben und in einem ZWEITEN, kurzen ssh-Aufruf
# abholen.
r2_start_bg() {
  local logname="$1"; shift
  R2 "cd $R2_DIR && rm -f $logname $logname.pid && \
      { setsid nohup env $* > $logname 2>&1 < /dev/null & echo \$! > $logname.pid ; } \
      > /dev/null 2>&1 < /dev/null" || return 1
  R2 "cat $R2_DIR/$logname.pid 2>/dev/null" | tr -d '\r\n '
}
r2_kill_bg() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  R2 "kill $pid 2>/dev/null; sleep 0.2; kill -9 $pid 2>/dev/null" >/dev/null 2>&1 || true
}

# ===========================================================================
# Reste-Disziplin. Gesucht wird NUR der eigene Binary-Name -- kein breites
# pkill: auf beiden Kisten laufen fremde Server, die nichts davon mitbekommen
# duerfen. Ein ssh-Abbruch toetet den fernen Prozess nicht, deshalb wird der
# Check vor JEDEM Paar wiederholt, nicht nur einmal am Anfang.
# ===========================================================================
PAT='gpurdma_04_bench'
leftovers_local()  { pgrep -f "$PAT" 2>/dev/null | tr '\n' ' '; }
leftovers_remote() { R2 "pgrep -f '$PAT' 2>/dev/null | tr '\n' ' '" 2>/dev/null; }

clear_leftovers() {
  local l r
  l=$(leftovers_local); r=$(leftovers_remote)
  if [ -n "${l// }" ]; then
    log "WARN: Reste auf Rig 1 ($l) -- eigener Binary-Name, werden beendet"
    # shellcheck disable=SC2086
    kill $l 2>/dev/null; sleep 0.3
    # shellcheck disable=SC2086
    kill -9 $l 2>/dev/null || true
  fi
  if [ -n "${r// }" ]; then
    log "WARN: Reste auf Rig 2 ($r) -- eigener Binary-Name, werden beendet"
    R2 "kill $r 2>/dev/null; sleep 0.3; kill -9 $r 2>/dev/null" >/dev/null 2>&1 || true
  fi
}

# ===========================================================================
# Fault-Zaehler. Kein Sysfs-Zaehler vorhanden -- die belastbare Quelle ist der
# Kernel-Log. Gezaehlt werden IOMMU-Faults auf beiden Seiten; jede Zunahme
# waehrend einer Phase landet in der Ergebnisdatei.
# ===========================================================================
fault_count_local()  { dmesg 2>/dev/null | grep -cE 'AMD-Vi.*(IO_PAGE_FAULT|event)|DMAR.*fault' || true; }
fault_count_remote() { R2 "dmesg 2>/dev/null | grep -cE 'AMD-Vi.*(IO_PAGE_FAULT|event)|DMAR.*fault'" 2>/dev/null || echo 0; }

FAULTS_L0=0; FAULTS_R0=0
faults_begin() { FAULTS_L0=$(fault_count_local); FAULTS_R0=$(fault_count_remote); }
faults_end() {
  local tag="$1" l1 r1
  l1=$(fault_count_local); r1=$(fault_count_remote)
  note "# faults	$tag	rig1 $FAULTS_L0 -> $l1	rig2 $FAULTS_R0 -> $r1"
  if [ "${l1:-0}" != "${FAULTS_L0:-0}" ] || [ "${r1:-0}" != "${FAULTS_R0:-0}" ]; then
    log "ACHTUNG: IOMMU-Fault-Zaehler gestiegen in Phase $tag (rig1 $FAULTS_L0->$l1, rig2 $FAULTS_R0->$r1)"
  fi
}

# ===========================================================================
# Kartenlocks (nur Phase gpu) -- Format identisch zu comm_suite.py::_CardWindow
# ===========================================================================
declare -a HELD_LOCKS=()
# Karten, die beim Start schon UNS gehoerten (aeusseres Fenster-Lock des
# Operators/dieser Sitzung). Sie werden mitbenutzt, aber NICHT freigegeben --
# sonst gaebe das Skript ein Fenster zurueck, das es nie genommen hat.
declare -A PREHELD=()
lock_path() { printf "$LOCK_DIR_FMT" "$1"; }
owner_of() { [ -r "$1/info" ] && grep -m1 '^owner=' "$1/info" | cut -d= -f2- || echo unknown; }
release_all_locks() {
  local idx p
  for idx in ${HELD_LOCKS[@]+"${HELD_LOCKS[@]}"}; do
    [ -n "${PREHELD[$idx]:-}" ] && continue
    p=$(lock_path "$idx"); rm -f "$p/info" 2>/dev/null; rmdir "$p" 2>/dev/null
  done
  HELD_LOCKS=()
}
# Die temporaere Intra-Rig-IPv4 MUSS auch bei Abbruch verschwinden -- deshalb
# im selben Trap wie die Locks, nicht am Ende der Phase.
intra_ip_add() {
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: keine IPv4 auf $IF_INTRA gesetzt"
    return 0
  fi
  if ip -4 addr show dev "$IF_INTRA" 2>/dev/null | grep -q "inet "; then
    log "  $IF_INTRA hat bereits eine IPv4 -- unveraendert gelassen"
    return 0
  fi
  if ip addr add "$IP_INTRA" dev "$IF_INTRA" 2>/dev/null; then
    IP_INTRA_ADDED=1
    log "  temporaere IPv4 $IP_INTRA auf $IF_INTRA gesetzt (wird am Ende entfernt)"
  else
    log "  FAIL: konnte $IP_INTRA nicht auf $IF_INTRA setzen"
    return 1
  fi
}
intra_ip_del() {
  if [ "$IP_INTRA_ADDED" = "1" ]; then
    ip addr del "$IP_INTRA" dev "$IF_INTRA" 2>/dev/null
    IP_INTRA_ADDED=0
    log "  temporaere IPv4 $IP_INTRA von $IF_INTRA entfernt"
  fi
}
cleanup_all() { release_all_locks; intra_ip_del; clear_leftovers >/dev/null 2>&1 || true; }
trap cleanup_all EXIT

acquire_card_lock() {
  local idx="$1" path age used
  if [ -d "$QUIET_LOCK_DIR" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$QUIET_LOCK_DIR" 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$QUIET_STALE_S" ]; then
      log "FAIL: Quiet-Fenster offen ($(owner_of "$QUIET_LOCK_DIR"), ${age}s) -- keine GPU-Phase"; return 1
    fi
  fi
  if [ -d "$LEGACY_LOCK_DIR" ]; then
    log "FAIL: rig-weites Legacy-Lock gehalten von $(owner_of "$LEGACY_LOCK_DIR")"; return 1
  fi
  path=$(lock_path "$idx")
  if ! mkdir "$path" 2>/dev/null; then
    if [ "$(owner_of "$path")" = "$OWNER" ]; then
      # Eigenes aeusseres Fensterlock -- wiedereintrittsfaehig mitbenutzen.
      PREHELD[$idx]=1
      # MUSS trotzdem in HELD_LOCKS: das Schutzgitter in run_unit/smoke_pair
      # prueft genau diese Liste, bevor es eine Rig-1-Karte anfasst.
      HELD_LOCKS+=("$idx")
      log "  Karte $idx: eigenes Fensterlock bereits gehalten, wird mitbenutzt"
      return 0
    fi
    log "FAIL: Karte $idx gehalten von $(owner_of "$path") -- fremder Owner, Abbruch"; return 1
  fi
  { echo "owner=$OWNER"
    echo "purpose=probe/gdr-window -- Cross-Rig-Groessenleiter"
    echo "acquired=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"; } > "$path/info"
  HELD_LOCKS+=("$idx")
  [ "$DRY_RUN" = "1" ] && return 0
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', *' -v i="$idx" '$1==i {print $2}')
  if [ -n "$used" ] && [ "$used" -gt "$FREE_MIB" ]; then
    log "FAIL: Lock fuer Karte $idx frei, aber Karte belegt (${used} MiB > ${FREE_MIB})"; return 1
  fi
  return 0
}
release_card_lock() {
  if [ -n "${PREHELD[$1]:-}" ]; then return 0; fi   # aeusseres Fensterlock bleibt
  local p; p=$(lock_path "$1"); rm -f "$p/info" 2>/dev/null; rmdir "$p" 2>/dev/null
  local n=(); local i
  for i in ${HELD_LOCKS[@]+"${HELD_LOCKS[@]}"}; do [ "$i" != "$1" ] && n+=("$i"); done
  HELD_LOCKS=(${n[@]+"${n[@]}"})
}

# ===========================================================================
# Topologie -- ausschliesslich ueber PCI-Adresse (Device-Order-Falle:
# CUDA-Default-Reihenfolge ist "fastest first", nicht PCI-Reihenfolge). Mit
# CUDA_DEVICE_ORDER=PCI_BUS_ID ist der CUDA-Ordinal exakt der nvidia-smi-Index.
# ===========================================================================
declare -A ORD1 NAME1
ORD2=""; NAME2=""; PCI2=""

resolve_cards() {
  local csv line idx pci name short
  if [ "$DRY_RUN" = "1" ]; then
    ORD1[05:00.0]=0; NAME1[05:00.0]="(dry) 3080 PIX"
    ORD1[0a:00.0]=1; NAME1[0a:00.0]="(dry) 5090"
    ORD1[0b:00.0]=2; NAME1[0b:00.0]="(dry) 3080b"
    ORD2=0; NAME2="(dry) 2080 Ti"; PCI2="04:00.0"
    return 0
  fi
  csv=$(CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,pci.bus_id,name \
        --format=csv,noheader,nounits)
  while IFS=, read -r idx pci name; do
    idx=$(echo "$idx" | tr -d ' '); pci=$(echo "$pci" | tr -d ' ' | tr 'A-F' 'a-f')
    short=$(echo "$pci" | awk -F: '{print $(NF-1)":"$NF}')
    ORD1[$short]=$idx; NAME1[$short]=$(echo "$name" | sed 's/^ *//')
    log "rig1 Karte: ord=$idx pci=$short name=${NAME1[$short]}"
  done <<< "$csv"

  local rcsv
  rcsv=$(R2 "CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader,nounits" 2>/dev/null | head -1)
  if [ -n "$rcsv" ]; then
    IFS=, read -r idx pci name <<< "$rcsv"
    ORD2=$(echo "$idx" | tr -d ' ')
    PCI2=$(echo "$pci" | tr -d ' ' | tr 'A-F' 'a-f' | awk -F: '{print $(NF-1)":"$NF}')
    NAME2=$(echo "$name" | sed 's/^ *//')
    log "rig2 Karte: ord=$ORD2 pci=$PCI2 name=$NAME2"
  else
    log "WARN: keine Rig-2-GPU aufgeloest -- 2080-Ti-Arme entfallen"
  fi
}

# ===========================================================================
# Warten, bis ein Server-Prozess wirklich lauscht. NICHT auf eine Statuszeile
# greppen: beide Rollen drucken ihre Statuszeile erst NACH dem vollstaendigen
# QP-Handshake, der seinerseits den bereits verbundenen Client voraussetzt --
# ein grep darauf liefe immer ins Timeout.
# ===========================================================================
wait_listen_local() {
  local port="$1" pid="$2" t="${3:-25}" w=0
  while :; do
    ss -ltn 2>/dev/null | awk -v p=":$port\$" '$4 ~ p {f=1} END{exit !f}' && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 0.2; w=$((w+1)); [ "$w" -gt $((t*5)) ] && return 1
  done
}
# Fern: EIN ssh-Aufruf mit Warteschleife auf der Gegenseite, nicht eine
# ssh-Runde je Versuch (das kostete sonst mehr als die Messung selbst).
# Geprueft wird die Adress-SPALTE, nicht das Zeilenende: "ss -ltn" haengt die
# Peer-Adresse hinten an, ein '$'-Anker auf den Port trifft deshalb NIE.
wait_listen_remote() {
  local port="$1" t="${2:-25}"
  R2 "for i in \$(seq 1 $((t*5))); do
        ss -ltn 2>/dev/null | awk -v p=':$port\$' '\$4 ~ p {f=1} END{exit !f}' && exit 0
        sleep 0.2
      done; exit 1"
}

# ===========================================================================
# EIN Messpunkt-Buendel: ein Server/Client-Paar, eine Groessenliste.
#
# Argumente:
#   pair_id  a_ord  b_ord  direction(a2b|b2a)  modus  ro  depth  sizes  iters  warmup
# a_ord/b_ord: CUDA-Ordinal der jeweiligen Seite, -1 = Host-RAM.
# ===========================================================================
run_unit() {
  local pair="$1" a_ord="$2" b_ord="$3" dir="$4" modus="$5" ro="$6" depth="$7" \
        sizes="$8" iters="$9" warmup="${10}"
  local opts="--sizes=$sizes --depth=$depth --iters=$iters --warmup=$warmup"
  [ -n "$SECS_OPT" ] && opts="$opts $SECS_OPT"
  [ "$ro" = "on" ] && opts="$opts --ro"
  if [ "$DRY_RUN" = "1" ]; then
    # Trockenlauf: kein Prozessstart, keine NIC, keine GPU -- nur die
    # Matrix-Abfolge sichtbar machen. Frueher lief --dry-run in den echten
    # Messpfad hinein; das ist der Fehler, den dieser Zweig schliesst.
    log "  dry-run: $pair dir=$dir a_ord=$a_ord b_ord=$b_ord modus=$modus ro=$ro depth=$depth [$opts]"
    note "# dry-run	$pair	$dir	$modus	$ro	$depth	$sizes"
    return 0
  fi
  # Schutzgitter wie in smoke_pair: eine Rig-1-Karte nur mit gehaltenem Lock.
  if [ "$a_ord" != "-1" ] && [ "${#HELD_LOCKS[@]}" = "0" ]; then
    log "  ABBRUCH $pair: Rig-1-Ordinal $a_ord ohne gehaltenen Kartenlock"
    return 1
  fi
  local port=$((23456 + RANDOM % 5000))
  local srv_log cli_log rc=0
  srv_log=$(mktemp); cli_log=$(mktemp)

  # Rig-2-Binary: der CUDA-Zweig kann beides; ohne ihn nur Host-Endpunkte.
  local r2bin="$R2_BIN"
  if ! R2 "test -x $R2_BIN" 2>/dev/null; then
    if [ "$b_ord" != "-1" ]; then
      log "  SKIP $pair/$dir/$modus/ro=$ro/d=$depth: $R2_BIN fehlt (GPU-Zweig auf Rig 2 nicht gebaut)"
      rm -f "$srv_log" "$cli_log"; return 1
    fi
    r2bin="$R2_BIN_HOST"
  fi

  clear_leftovers

  if [ "$dir" = "b2a" ]; then
    # Server lokal (Rig 1), Client fern (Rig 2)
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$a_ord" "$NIC1" "$port" "$modus" $opts \
      > "$srv_log" 2>&1 &
    local spid=$!
    if ! wait_listen_local "$port" "$spid" 30; then
      log "  FAIL $pair/$dir/$modus/ro=$ro/d=$depth: lokaler Server lauscht nicht: $(tail -2 "$srv_log"|tr '\n' ' ')"
      kill "$spid" 2>/dev/null; rm -f "$srv_log" "$cli_log"; return 1
    fi
    R2 "cd $R2_DIR && CUDA_DEVICE_ORDER=PCI_BUS_ID $r2bin client $IP1 $NIC2 $port $modus $b_ord $opts" \
      > "$cli_log" 2>&1 || rc=1
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  else
    # Server fern (Rig 2), Client lokal (Rig 1). Der ferne Prozess ueberlebt
    # einen ssh-Abbruch, deshalb wird seine PID gemerkt und explizit beendet.
    local rpid
    rpid=$(r2_start_bg srv.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $r2bin server $b_ord $NIC2 $port $modus $opts")
    if ! wait_listen_remote "$port" 30; then
      log "  FAIL $pair/$dir/$modus/ro=$ro/d=$depth: ferner Server lauscht nicht: $(R2 "tail -3 $R2_DIR/srv.log" 2>/dev/null | tr '\n' ' ')"
      r2_kill_bg "$rpid"
      rm -f "$srv_log" "$cli_log"; return 1
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client "$IP2" "$NIC1" "$port" "$modus" "$a_ord" $opts \
      > "$cli_log" 2>&1 || rc=1
    r2_kill_bg "$rpid"
    R2 "cat $R2_DIR/srv.log" > "$srv_log" 2>/dev/null
  fi

  local n
  n=$(awk -F'\t' -v pair="$pair" -v dir="$dir" -v a="$a_ord" -v b="$b_ord" '
        $1=="DATA" {
          # DATA modus size depth ro iters p10 median p90 bytes_per_round
          med=$8+0; bpr=$10+0
          # MB/s ueber die volle Runde: der halbe Round-trip wird verdoppelt.
          mbs = (med>0) ? (bpr / (2*med)) : 0
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.1f\n",
                 pair, dir, $2, $5, $4, $3, $6, $7, $8, $9, mbs
          c++
        } END { print c+0 > "/dev/stderr" }
      ' "$cli_log" 2>&1 >> "$RESULTS_FILE" )
  if [ "${n:-0}" = "0" ]; then
    log "  FAIL $pair/$dir/$modus/ro=$ro/d=$depth: keine DATA-Zeilen. cli: $(tail -2 "$cli_log"|tr '\n' ' ') | srv: $(tail -2 "$srv_log"|tr '\n' ' ')"
    rc=1
  else
    log "  ok   $pair dir=$dir modus=$modus ro=$ro depth=$depth -> $n Punkte"
  fi
  rm -f "$srv_log" "$cli_log"
  return $rc
}

# Ein Paar = kleine + grosse Groessen (getrennte Iterationszahl, sonst kostet
# 4 MiB x Tiefe 16 unnoetig Minuten).
run_pair() {
  local pair="$1" a_ord="$2" b_ord="$3" dir="$4" modus="$5" ro="$6" depth="${7:-1}" \
        profile="${8:-full}"
  local il=$(( ITERS_LARGE / depth )); [ "$il" -lt 40 ] && il=40
  if [ "$profile" = "short" ]; then
    # 3-Punkt-Leiter in EINEM Aufruf -- fuer Achsen, die keine volle
    # Groessenaufloesung brauchen (Tiefe, VRAM<->VRAM).
    run_unit "$pair" "$a_ord" "$b_ord" "$dir" "$modus" "$ro" "$depth" \
             "$SIZES_INTRA" "$ITERS_INTRA" "$WARMUP_INTRA"
    return
  fi
  run_unit "$pair" "$a_ord" "$b_ord" "$dir" "$modus" "$ro" "$depth" \
           "$SIZES_SMALL" "$ITERS_SMALL" "$WARMUP_SMALL"
  run_unit "$pair" "$a_ord" "$b_ord" "$dir" "$modus" "$ro" "$depth" \
           "$SIZES_LARGE" "$il" "$WARMUP_LARGE"
}

header() {
  note "# pair	direction	mode	ro	depth	size_bytes	iters	p10_us	median_us	p90_us	MB_per_s"
}

# ===========================================================================
# Kleiner Transfer MIT Inhaltspruefung ueber die Rechnergrenze (Stufe-0-Muster
# aus gdr_window_run.sh, hier cross-rig statt ueber 127.0.0.1).
#   smoke_pair <label> <a_ord> <b_ord>
# Der Client ist der Sender, der Server die Empfangsseite -- das massgebliche
# Urteil faellt deshalb IMMER der Server, der Client meldet nur "abgeschickt".
# a_ord = Rig-1-Seite, b_ord = Rig-2-Seite; wer nicht -1 ist, haelt die GPU.
# Rolle: die Seite mit b_ord != -1 als Sender heisst "source", sonst "target".
# ===========================================================================
smoke_pair() {
  local label="$1" a_ord="$2" b_ord="$3"
  local port=$((23456 + RANDOM % 5000))
  local srv cli line
  srv=$(mktemp); cli=$(mktemp)
  # Schutzgitter: ausserhalb der Phase gpu/intra/pressure darf hier NIE ein
  # Rig-1-Ordinal stehen -- eine vertauschte Argumentreihenfolge wuerde sonst
  # unbemerkt eine ungelockte Rig-1-Karte anfassen (genau das ist einmal
  # passiert). Lieber hart abbrechen als still eine fremde Karte benutzen.
  if [ "$a_ord" != "-1" ] && [ "${#HELD_LOCKS[@]}" = "0" ]; then
    log "  ABBRUCH smoke_pair '$label': Rig-1-Ordinal $a_ord ohne gehaltenen Kartenlock"
    note "ti_pre	$label	FAIL	Rig-1-Ordinal ohne Kartenlock -- Aufruf abgelehnt"
    rm -f "$srv" "$cli"; return 1
  fi
  clear_leftovers
  if [ "$label" = "source" ]; then
    # Rig 2 sendet (GPU als Quelle), Rig 1 empfaengt im Host-RAM und urteilt.
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" server "$a_ord" "$NIC1" "$port" > "$srv" 2>&1 &
    local spid=$!
    if wait_listen_local "$port" "$spid" 30; then
      R2 "cd $R2_DIR && CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_SMOKE client $IP1 $NIC2 $port $b_ord" > "$cli" 2>&1 || true
    else
      echo "kein Listener" >> "$srv"
    fi
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  else
    # Rig 1 sendet aus dem Host-RAM, Rig 2 empfaengt im VRAM und urteilt.
    local rpid
    rpid=$(r2_start_bg smk.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_SMOKE server $b_ord $NIC2 $port")
    if wait_listen_remote "$port" 30; then
      CUDA_DEVICE_ORDER=PCI_BUS_ID "$SMOKE_BIN" client "$IP2" "$NIC1" "$port" "$a_ord" > "$cli" 2>&1 || true
    fi
    r2_kill_bg "$rpid"
    R2 "cat $R2_DIR/smk.log" > "$srv" 2>/dev/null
  fi
  line=$(grep -m1 '^RESULT ' "$srv" 2>/dev/null)
  if [ -z "$line" ]; then
    note "ti_pre	$label	FAIL	kein RESULT (srv: $(tail -2 "$srv" | tr '\n' ' ') | cli: $(tail -2 "$cli" | tr '\n' ' '))"
    log "  smoke $label: FAIL (kein RESULT)"
    rm -f "$srv" "$cli"; return 1
  fi
  note "ti_pre	$label	$line"
  log "  smoke $label: $line"
  rm -f "$srv" "$cli"
  case "$line" in *" PASS "*) return 0 ;; *) return 1 ;; esac
}

# ===========================================================================
# Phase wire -- Draht-Basislinie ohne jede GPU. Keine Locks, blockiert nichts.
# ===========================================================================
phase_wire() {
  log "PHASE wire: Host-RAM <-> Host-RAM ueber $NIC1 <-> $NIC2 (keine GPU, keine Locks)"
  note "# ---- Phase wire: Draht-Basislinie Host<->Host, keine GPU beteiligt ----"
  note "# Obergrenze fuer jeden Staging-Wert. gdr/stage sind hier derselbe"
  note "# Codepfad (beide Endpunkte Host-MR), deshalb nur EIN Modus, 'stage'."
  header
  faults_begin
  local dir ro d
  for dir in b2a a2b; do
    for ro in $RO_LIST; do
      run_pair "wire" -1 -1 "$dir" stage "$ro" 1
    done
    # Tiefen-Achse auch auf dem Draht: sonst waere spaeter nicht trennbar, ob
    # ein Tiefengewinn am BAR-Pfad haengt oder schon auf dem Draht auftritt.
    for d in 4 16; do
      run_pair "wire" -1 -1 "$dir" stage off "$d" short
    done
  done
  faults_end wire
}

# ===========================================================================
# Phase neutral -- was kostet der Codepfad-Wechsel allein?
# ===========================================================================
phase_neutral() {
  log "PHASE neutral: Alt-Binary gegen Neu-Binary, Tiefe 1, Host<->Host"
  note "# ---- Phase neutral: Codepfad-Neutralitaet ----"
  note "# Alt-Stand a0c67380c4 (zwei getrennte ibv_post_send fuer Nutzlast und"
  note "# Flag) gegen den neuen Stand (eine verkettete Postung), beide Host-RAM,"
  note "# Tiefe 1, identische Leiter. Die Differenz ist der Sockel, auf dem die"
  note "# spaetere Tiefen-Achse steht. Das Alt-Binary kennt --sizes/--iters"
  note "# nicht, es faehrt seine eingebaute Leiter 8/4096/65536/1048576 mit"
  note "# 2000 Iterationen -- deshalb wird der neue Stand hier mit exakt diesen"
  note "# Parametern dagegengehalten, nicht mit der vollen 9-Punkt-Leiter."
  header
  faults_begin
  local port rc
  # Alt gegen Alt (eigener Wire-Formatstand, laeuft nur mit sich selbst).
  if [ -x "$BIN_PRE" ] && R2 "test -x $R2_BIN_PRE" 2>/dev/null; then
    clear_leftovers
    port=$((23456 + RANDOM % 5000))
    local srv cli; srv=$(mktemp); cli=$(mktemp)
    "$BIN_PRE" server -1 "$NIC1" "$port" stage > "$srv" 2>&1 &
    local spid=$!
    if wait_listen_local "$port" "$spid" 30; then
      R2 "cd $R2_DIR && $R2_BIN_PRE client $IP1 $NIC2 $port stage -1" > "$cli" 2>&1 || true
    fi
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
    awk '/^(GDR|STAGE) /{ printf "neutral_pre\tb2a\tstage\toff\t1\t%s\t2000\t%s\t%s\t%s\t-\n", $2, $5, $8, $11 }' \
        "$cli" >> "$RESULTS_FILE"
    rc=$(grep -c '^neutral_pre' "$RESULTS_FILE")
    log "  Alt-Binary: $rc Punkte"
    rm -f "$srv" "$cli"
  else
    log "  SKIP Alt-Binary (nicht auf beiden Seiten vorhanden)"
    note "# (Alt-Binary nicht vorhanden -- Neutralitaetsvergleich entfaellt)"
  fi
  # Neu mit exakt der Alt-Leiter.
  run_unit "neutral_new" -1 -1 b2a stage off 1 "8,4096,65536,1048576" 2000 200
  faults_end neutral
}

# ===========================================================================
# Phase ti -- alles mit der 2080 Ti, dessen Rig-1-Endpunkt Host-RAM ist.
# Braucht KEIN Rig-1-Fenster.
# ===========================================================================
phase_ti() {
  if [ -z "$ORD2" ]; then log "PHASE ti: keine Rig-2-GPU -- uebersprungen"; return 0; fi
  log "PHASE ti: 2080 Ti ($PCI2, ord=$ORD2) gegen Rig-1-Host-RAM"
  note "# ---- Phase ti: Rig-2-VRAM (2080 Ti $PCI2) <-> Rig-1-Host-RAM ----"
  note "# V4: GDR_BEFUNDE.md Abschnitt 7 haelt die 2080 Ti fuer untauglich als"
  note "# Quelle (local protection error). Das Verdikt stammt von VOR dem"
  note "# iommu=pt-Fix und wird hier als offene Frage geprueft, nicht"
  note "# uebernommen. Erst regcheck + verifizierter Klein-Transfer, dann Leiter."
  faults_begin

  # --- Vorpruefung: registrieren + kleiner Transfer MIT Inhaltspruefung ---
  # Erst registrieren (kein Netzwerk), dann beide Richtungen als echter
  # Transfer mit memcmp/Readback -- die Uebergabe berichtet, dass genau der
  # Schritt vom Registrieren zum Zugriff scheitert, deshalb reicht regcheck
  # allein als Antwort auf V4 NICHT.
  note "# phase	check	verdict	detail"
  local ok=0
  if R2 "test -x $R2_SMOKE" 2>/dev/null; then
    local out
    out=$(R2 "cd $R2_DIR && CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_SMOKE regcheck $ORD2 $NIC2 2>&1 | grep -m1 '^RESULT' ")
    note "ti_pre	regcheck	${out:-KEIN-RESULT}"
    log "  regcheck: ${out:-KEIN-RESULT}"
    case "$out" in *" PASS "*) ok=1 ;; esac
  else
    note "ti_pre	regcheck	SKIP	$R2_SMOKE nicht gebaut"
    log "  regcheck uebersprungen ($R2_SMOKE fehlt)"
  fi
  if [ "$ok" = "0" ]; then
    log "  2080 Ti registriert nicht -> Leiter entfaellt, alter Befund bestaetigt"
    note "# 2080 Ti: Registrierung fehlgeschlagen -- Befund aus Abschnitt 7 bestaetigt"
    faults_end ti
    return 0
  fi

  # Argumentreihenfolge ist IMMER <rig1-ord> <rig2-ord>. In beiden Aufrufen
  # ist die Rig-1-Seite Host-RAM (-1) und die Rig-2-Seite die 2080 Ti --
  # unterschiedlich ist nur, WER sendet (das steckt im Label).
  smoke_pair target "-1" "$ORD2"   # Rig 1 sendet -> 2080 Ti empfaengt (BAR schreiben)
  local src_ok=1
  smoke_pair source "-1" "$ORD2" || src_ok=0   # 2080 Ti sendet (BAR lesen) <- V4
  if [ "$src_ok" = "0" ]; then
    # Begrenzte Diagnose: die letzten Kernel-Zeilen beider Seiten sichern,
    # dann als bestaetigte Wand berichten. Kein Debug-Marathon.
    note "# 2080 Ti als QUELLE scheitert weiter -- Befund GDR_BEFUNDE.md Abschnitt 7"
    note "# bestaetigt, auch nach iommu=pt. dmesg-Auszug:"
    R2 "dmesg | tail -15" 2>/dev/null | sed 's/^/#   rig2: /' >> "$RESULTS_FILE"
    dmesg 2>/dev/null | tail -8 | sed 's/^/#   rig1: /' >> "$RESULTS_FILE"
    log "  2080 Ti als Quelle: weiterhin defekt -- Leiterarme mit ihr als Quelle laufen trotzdem,"
    log "  damit der Fehler in der Tabelle als Datenpunkt steht statt als Luecke"
  fi

  header
  local dir modus ro
  # Beide Richtungen, beide Modi. RO-Achse hier voll auf dir=b2a (2080 Ti ist
  # Quelle, also die BAR-LESEN-Richtung -- der interessante Fall), auf a2b
  # festgenagelt auf off. Ausduennung bewusst und hier benannt.
  for modus in gdr stage; do
    for ro in $RO_LIST; do
      run_pair "ti_src" -1 "$ORD2" b2a "$modus" "$ro" 1
    done
    run_pair "ti_dst" -1 "$ORD2" a2b "$modus" off 1
  done
  faults_end ti
}

# ===========================================================================
# Phase gpu -- alles mit Rig-1-VRAM. NUR im freigegebenen Fenster.
# ===========================================================================
phase_gpu() {
  log "PHASE gpu: Rig-1-VRAM-Arme (Kartenlocks noetig)"
  note "# ---- Phase gpu: Rig-1-VRAM <-> Rig-2 ----"
  note "# Ausduennung (siehe Kopf): RO voll nur gegen Rig-2-Host-RAM; Tiefen-"
  note "# Achse 1/4/16 NUR auf der 5090; VRAM<->VRAM mit RO off und Tiefe 1."
  header
  faults_begin

  local pci ord label modus ro d
  for pci in 05:00.0 0a:00.0 0b:00.0; do
    ord="${ORD1[$pci]:-}"
    if [ -z "$ord" ]; then log "  Karte $pci nicht aufgeloest -- uebersprungen"; continue; fi
    label=$(echo "${NAME1[$pci]}" | tr ' ' '_')
    log " Karte $pci ($label, ord=$ord): Lock nehmen"
    if ! acquire_card_lock "$ord"; then
      note "# Karte $pci (ord=$ord): Lock nicht verfuegbar -- Arm entfaellt"
      continue
    fi

    # (1) gegen Rig-2-Host-RAM, beide Richtungen, beide Modi, +/-RO.
    #     b2a = Rig 2 schreibt in die BAR (posted)
    #     a2b = Rig 1 liest aus der BAR (non-posted)  <- V1/V2
    for modus in gdr stage; do
      for ro in $RO_LIST; do
        run_pair "c_${pci}_hostpeer" "$ord" -1 b2a "$modus" "$ro" 1
        run_pair "c_${pci}_hostpeer" "$ord" -1 a2b "$modus" "$ro" 1
      done
    done

    # (2) Tiefen-Achse nur auf der 5090 -- Budget, siehe Kopf.
    if [ "$pci" = "0a:00.0" ]; then
      for d in 4 16; do
        for modus in gdr stage; do
          run_pair "c_${pci}_depth" "$ord" -1 b2a "$modus" off "$d" short
          run_pair "c_${pci}_depth" "$ord" -1 a2b "$modus" off "$d" short
        done
      done
    fi

    # (3) VRAM <-> VRAM gegen die 2080 Ti, beide Richtungen, RO off.
    if [ -n "$ORD2" ]; then
      for modus in gdr stage; do
        run_pair "c_${pci}_x_ti" "$ord" "$ORD2" b2a "$modus" off 1 short
        run_pair "c_${pci}_x_ti" "$ord" "$ORD2" a2b "$modus" off 1 short
      done
    fi

    release_card_lock "$ord"
    log " Karte $pci: Lock freigegeben"
  done
  faults_end gpu
}

# ===========================================================================
# Intra-Rig: ein Paar Rig-1-Karten ueber die NIC (Loopback), beide Prozesse
# lokal. Der "direkte" Weg ist hier ein NIC-RELAY: GPU_A -> CX-4 -> GPU_B,
# beidseitig dmabuf. Kein GPUDirect-P2P zwischen den Karten (dieses Rig kann
# das nicht), deshalb ist die heutige Alternative der Weg ueber den
# System-RAM -- der wird als dritter Arm mit NCCL gemessen, nicht geschaetzt.
# ===========================================================================
intra_unit() {
  local pair="$1" ord_a="$2" ord_b="$3" modus="$4" tag="${5:-}"
  local opts="--sizes=$SIZES_INTRA --depth=1 --iters=$ITERS_INTRA --warmup=$WARMUP_INTRA"
  [ -n "$SECS_OPT" ] && opts="$opts $SECS_OPT"
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: intra $pair$tag ord_a=$ord_a ord_b=$ord_b modus=$modus [$opts]"
    note "# dry-run	$pair$tag	intra	$modus	off	1	$SIZES_INTRA"
    return 0
  fi
  local port=$((23456 + RANDOM % 5000))
  local srv cli n
  srv=$(mktemp); cli=$(mktemp)
  # Server haelt GPU_B (Empfaenger, BAR schreiben), Client GPU_A (Sender,
  # BAR lesen) -- beide lokal, TCP-Austausch ueber 127.0.0.1.
  CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$ord_b" "$NIC_INTRA" "$port" "$modus" $opts \
    > "$srv" 2>&1 &
  local spid=$!
  if ! wait_listen_local "$port" "$spid" 30; then
    log "  FAIL intra $pair/$modus: Server lauscht nicht: $(tail -2 "$srv"|tr '\n' ' ')"
    kill "$spid" 2>/dev/null; rm -f "$srv" "$cli"; return 1
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$port" "$modus" "$ord_a" $opts \
    > "$cli" 2>&1
  kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  n=$(awk -F'\t' -v pair="$pair$tag" '
        $1=="DATA" { med=$8+0; bpr=$10+0; mbs=(med>0)?(bpr/(2*med)):0
          printf "%s\tintra\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.1f\n",
                 pair, $2, $5, $4, $3, $6, $7, $8, $9, mbs; c++ }
        END { print c+0 > "/dev/stderr" }' "$cli" 2>&1 >> "$RESULTS_FILE")
  [ "${n:-0}" = "0" ] && log "  FAIL intra $pair/$modus: keine DATA ($(tail -2 "$cli"|tr '\n' ' '))" \
                      || log "  ok   intra $pair$tag modus=$modus -> $n Punkte"
  rm -f "$srv" "$cli"
}

# NCCL ueber System-RAM: der ECHTE heutige Pfad zwischen zwei Karten dieses
# Rigs. Nicht neu gebaut -- der Worker aus comm_suite wird wiederverwendet
# (er kennt Rangbindung, Aufwaermen und Zeitmessung bereits). Gemessen wird
# all_reduce auf denselben Groessen; die Zahl ist NICHT direkt eine
# Punkt-zu-Punkt-Latenz, sondern der Kollektiv-Boden, und wird genau so
# beschriftet (arm=nccl_allreduce), damit niemand sie als send/recv liest.
intra_nccl() {
  local pair="$1" ord_a="$2" ord_b="$3"
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: nccl $pair ord_a=$ord_a ord_b=$ord_b"
    note "# dry-run	$pair	nccl_allreduce	-	-	1	20/80/1024 KiB"
    return 0
  fi
  local worker="$HERE/../r3val/comm_suite_worker.py"
  local py="${CR_PYTHON:-$HERE/../../.venv/bin/python}"
  if [ ! -x "$py" ] || [ ! -r "$worker" ]; then
    log "  SKIP nccl $pair: kein venv-Python oder Worker ($py / $worker)"
    note "# nccl $pair uebersprungen: $py oder $worker fehlt"
    return 0
  fi
  local out; out=$(mktemp --suffix=.json); rm -f "$out"
  local port; port=$((29500 + RANDOM % 3000))
  local kib="20,80,1024"
  # Beide Raenge sehen genau diese zwei Karten, Rang r nimmt Karte r.
  local i
  for i in 0 1; do
    local extra=""; [ "$i" = "0" ] && extra="--out $out"
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$port CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$ord_a,$ord_b" TORCH_NCCL_BLOCKING_WAIT=1 \
      "$py" "$worker" --backend nccl --rank "$i" --world 2 \
        --iters 200 --warmup 20 --sizes "$kib" $extra > /tmp/nccl_r$i.log 2>&1 &
  done
  wait
  if [ ! -r "$out" ]; then
    log "  FAIL nccl $pair: kein Ergebnis ($(tail -2 /tmp/nccl_r0.log|tr '\n' ' '))"
    note "# nccl $pair fehlgeschlagen: $(tail -1 /tmp/nccl_r0.log)"
    return 1
  fi
  "$py" - "$out" "$pair" >> "$RESULTS_FILE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); pair = sys.argv[2]
for key, val in sorted(d.items()):
    if not isinstance(val, dict) or "all_reduce" not in key:
        continue
    kib = key.split("/")[-1].replace("KiB", "")
    try:
        nbytes = int(float(kib) * 1024)
    except ValueError:
        continue
    med = val.get("median_us") or val.get("median") or 0
    mbs = (nbytes / (2 * med)) if med else 0
    print("%s\tintra\tnccl_allreduce\toff\t1\t%d\t-\t-\t%.3f\t-\t%.1f"
          % (pair, nbytes, med, mbs))
PYEOF
  rm -f "$out"
  log "  ok   nccl $pair"
}

phase_intra() {
  log "PHASE intra: Rig-1-intern, jede Karte gegen jede (NIC-Relay gegen NCCL/System-RAM)"
  note "# ---- Phase intra: Rig-1-Karte <-> Rig-1-Karte ----"
  note "# Direktarm = NIC-RELAY (GPU_A -> CX-4 -> GPU_B, dmabuf beidseitig),"
  note "# Stagingarm = derselbe Relay-Weg mit Host-Kopie, dritter Arm = NCCL"
  note "# ueber System-RAM (der heute gefahrene Weg). Ausduennung: nur 3"
  note "# Groessen (20/80 KiB, 1 MiB), kein RO, Tiefe 1 -- siehe Kopf."
  header
  faults_begin
  intra_ip_add || { log "  ohne IPv4 auf $IF_INTRA kein RoCEv2-GID -- Phase entfaellt"; return 1; }

  local -a pcis=(05:00.0 0a:00.0 0b:00.0)
  local i j pa pb oa ob pair
  for i in 0 1 2; do
    for j in 0 1 2; do
      [ "$i" -ge "$j" ] && continue
      pa=${pcis[$i]}; pb=${pcis[$j]}
      oa="${ORD1[$pa]:-}"; ob="${ORD1[$pb]:-}"
      [ -z "$oa" ] || [ -z "$ob" ] && continue
      log " Paar $pa <-> $pb: beide Locks nehmen"
      acquire_card_lock "$oa" || { note "# intra $pa<->$pb: Lock $oa fehlt"; continue; }
      if ! acquire_card_lock "$ob"; then
        note "# intra $pa<->$pb: Lock $ob fehlt"; release_card_lock "$oa"; continue
      fi
      pair="i_${pa}_to_${pb}"
      intra_unit "$pair" "$oa" "$ob" gdr
      intra_unit "$pair" "$oa" "$ob" stage
      # Gegenrichtung: Sender und Empfaenger tauschen -- BAR-Lesen wechselt
      # damit die Karte, das ist die Achse aus V1/V2.
      intra_unit "i_${pb}_to_${pa}" "$ob" "$oa" gdr
      intra_unit "i_${pb}_to_${pa}" "$ob" "$oa" stage
      # A-vs-A-Rauschboden: EIN Punkt doppelt gefahren, sonst ist kein
      # Unterschied als echt behauptbar.
      intra_unit "$pair" "$oa" "$ob" gdr "_avsa"
      intra_nccl "n_${pa}_x_${pb}" "$oa" "$ob"
      release_card_lock "$ob"; release_card_lock "$oa"
    done
  done
  intra_ip_del
  faults_end intra
}

# ===========================================================================
# Phase pressure -- Parallel-Druck, EIN kurzer Burst je Szenario.
#
# PRUEFBARE VORHERSAGE V5: das NIC-Relay teilt sich EINEN Gen3-x4-Endpunkt.
# Solo mag es den System-RAM-Weg schlagen; laufen alle drei Paare gleichzeitig,
# muesste es an dieser einen NIC serialisieren (Degradationsfaktor nahe 3),
# waehrend der NCCL/System-RAM-Weg ueber die RAM-Bandbreite skaliert (Faktor
# deutlich unter 3). Bestaetigt sich das, ist NIC-Relay hoechstens ein
# Solo-/Einzelpaar-Hebel. Bestaetigt es sich nicht, ist das ein echter Fund.
# ===========================================================================
pressure_burst() {
  local scenario="$1" size="$2"
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: pressure-Burst $scenario size=$size auf allen drei Paaren"
    note "# dry-run	pressure	$scenario	$size	drei Paare gleichzeitig"
    return 0
  fi
  local -a pcis=(05:00.0 0a:00.0 0b:00.0)
  local -a pids=() outs=()
  local i j pa pb oa ob k=0
  # Drei Paare: (0,1) (1,2) (0,2) -- jede Karte in genau zwei Paaren.
  local -a PAIRS=("0 1" "1 2" "0 2")
  local p
  for p in "${PAIRS[@]}"; do
    set -- $p; i=$1; j=$2
    pa=${pcis[$i]}; pb=${pcis[$j]}
    oa="${ORD1[$pa]:-}"; ob="${ORD1[$pb]:-}"
    [ -z "$oa" ] || [ -z "$ob" ] && continue
    local port=$((23456 + RANDOM % 5000))
    local o; o=$(mktemp); outs+=("$o:$pa:$pb")
    ( CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$ob" "$NIC_INTRA" "$port" gdr \
        --sizes=$size --depth=1 --iters=$PRESSURE_ITERS --warmup=20 > /dev/null 2>&1 &
      sp=$!
      wait_listen_local "$port" "$sp" 30 && \
      CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$port" gdr "$oa" \
        --sizes=$size --depth=1 --iters=$PRESSURE_ITERS --warmup=20 > "$o" 2>&1
      kill $sp 2>/dev/null ) &
    pids+=($!)
    k=$((k+1))
  done
  wait ${pids[@]+"${pids[@]}"} 2>/dev/null
  local e
  for e in ${outs[@]+"${outs[@]}"}; do
    local f="${e%%:*}" rest="${e#*:}"
    awk -F'\t' -v sc="$scenario" -v pr="${rest//:/_}" '
      $1=="DATA" { med=$8+0; bpr=$10+0; mbs=(med>0)?(bpr/(2*med)):0
        printf "p_%s\t%s\tgdr\toff\t1\t%s\t%s\t%s\t%s\t%s\t%.1f\n",
               pr, sc, $3, $6, $7, $8, $9, mbs }' "$f" >> "$RESULTS_FILE"
    rm -f "$f"
  done
}

phase_pressure() {
  log "PHASE pressure: Parallel-Druck auf allen drei Paaren, je ein kurzer Burst"
  note "# ---- Phase pressure: solo gegen parallel ----"
  note "# V5: das NIC-Relay teilt EINEN Gen3-x4-Endpunkt -- unter Parallel-Druck"
  note "# aller drei Paare muesste es serialisieren (Faktor nahe 3), waehrend"
  note "# der System-RAM-Weg ueber die RAM-Bandbreite skaliert. Zwei Szenarien:"
  note "# decode-artig ($PRESSURE_DECODE B) und prefill-artig ($PRESSURE_PREFILL B),"
  note "# je EIN Burst, keine Groessenachse darin (Budgetvorgabe)."
  header
  faults_begin
  intra_ip_add || return 1
  local pci ord
  for pci in 05:00.0 0a:00.0 0b:00.0; do
    ord="${ORD1[$pci]:-}"
    [ -n "$ord" ] && { acquire_card_lock "$ord" || { log "  Lock $ord fehlt -- Phase entfaellt"; return 1; }; }
  done
  # solo: nur EIN Paar aktiv (die anderen ruhen) -- Referenz fuer den Faktor.
  local -a pcis=(05:00.0 0a:00.0 0b:00.0)
  intra_unit "p_solo_decode"  "${ORD1[${pcis[0]}]}" "${ORD1[${pcis[1]}]}" gdr
  pressure_burst decode  "$PRESSURE_DECODE"
  pressure_burst prefill "$PRESSURE_PREFILL"
  local pci2
  for pci2 in 0b:00.0 0a:00.0 05:00.0; do
    ord="${ORD1[$pci2]:-}"; [ -n "$ord" ] && release_card_lock "$ord"
  done
  intra_ip_del
  faults_end pressure
}

# ===========================================================================
# Phase mix -- Head-of-Line-Blocking, das einzige Szenario mit gemischten
# Nachrichtengroessen.
#
# Warum ueberhaupt: Inferencing hat keine uniforme Nachrichtengroesse. Ein
# Sweep misst 20 KiB sauber fuer sich und 1 MiB sauber fuer sich und sieht
# deshalb per Konstruktion NICHT, was passiert, wenn beides auf demselben
# Pfad durcheinanderlaeuft: die kleine Nachricht wartet, bis der Brocken vor
# ihr durch ist. Dieser Effekt steht nicht im Median, er steht im p99 der
# KLEINEN Nachrichten.
#
# Fahrplan (im Bench, --mix): Dauer-Kadenz 20 KiB, jede 10. Runde 80 KiB,
# und alle ~100 ms ein 1-MiB-Burst. Je Arm zweimal gefahren -- einmal MIT
# Bursts, einmal mit --mix-no-burst bei sonst identischer Kadenz. Die eine
# Zahl, auf die es ankommt, ist der Quotient
#     p99(klein, mit Bursts) / p99(klein, ohne Bursts).
#
# Bewusst KEINE neue Batterie: ein Paar (5090), je ein kurzer 5-s-Burst je
# Arm. Cross-rig laeuft derselbe Mix nur, wenn das Fensterbudget es hergibt.
# ===========================================================================
MIX_SIZES="${CR_MIX_SIZES:-20480,81920,1048576}"
MIX_MID_EVERY="${CR_MIX_MID_EVERY:-10}"
MIX_BURST_MS="${CR_MIX_BURST_MS:-100}"
MIX_SECS="${CR_MIX_SECS:-5}"

# mix_unit <pair> <a_ord> <b_ord> <ort:intra|cross> <modus> <burst:mit|ohne>
mix_unit() {
  local pair="$1" a_ord="$2" b_ord="$3" ort="$4" modus="$5" burst="$6"
  local opts="--mix --sizes=$MIX_SIZES --depth=1 --iters=1000 --warmup=200"
  opts="$opts --secs=$MIX_SECS --mix-mid-every=$MIX_MID_EVERY --mix-burst-ms=$MIX_BURST_MS"
  [ "$burst" = "ohne" ] && opts="$opts --mix-no-burst"
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: mix $pair ort=$ort modus=$modus burst=$burst [$opts]"; return 0
  fi
  if [ "$a_ord" != "-1" ] && [ "${#HELD_LOCKS[@]}" = "0" ]; then
    log "  ABBRUCH mix $pair: Rig-1-Ordinal ohne Kartenlock"; return 1
  fi
  local port=$((23456 + RANDOM % 5000))
  local srv cli n
  srv=$(mktemp); cli=$(mktemp)
  clear_leftovers
  if [ "$ort" = "intra" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$b_ord" "$NIC_INTRA" "$port" "$modus" $opts \
      > "$srv" 2>&1 &
    local spid=$!
    if ! wait_listen_local "$port" "$spid" 30; then
      log "  FAIL mix $pair: Server lauscht nicht: $(tail -2 "$srv"|tr '\n' ' ')"
      kill "$spid" 2>/dev/null; rm -f "$srv" "$cli"; return 1
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$port" "$modus" "$a_ord" $opts \
      > "$cli" 2>&1
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  else
    # cross-rig: Rig-1-Karte sendet, Rig-2-Host-RAM empfaengt.
    local rpid
    rpid=$(r2_start_bg srv.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_BIN server -1 $NIC2 $port $modus $opts")
    if ! wait_listen_remote "$port" 30; then
      log "  FAIL mix $pair: ferner Server lauscht nicht"; r2_kill_bg "$rpid"
      rm -f "$srv" "$cli"; return 1
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client "$IP2" "$NIC1" "$port" "$modus" "$a_ord" $opts \
      > "$cli" 2>&1
    r2_kill_bg "$rpid"
  fi
  n=$(awk -F'\t' -v pair="$pair" -v ort="$ort" -v b="$burst" '
        $1=="MIXDATA" {
          # MIXDATA modus klasse size n burst_every mit|ohne p10 p50 p99
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                 pair, ort, $2, $3, $4, $5, b, $8, $9, $10; c++
        } END { print c+0 > "/dev/stderr" }' "$cli" 2>&1 >> "$RESULTS_FILE")
  if [ "${n:-0}" = "0" ]; then
    log "  FAIL mix $pair/$ort/$modus/burst=$burst: keine MIXDATA ($(tail -2 "$cli"|tr '\n' ' '))"
  else
    log "  ok   mix $pair ort=$ort modus=$modus burst=$burst -> $n Klassen"
    grep -m1 "^MIX-Fahrplan" "$cli" | sed 's/^/#   /' >> "$RESULTS_FILE"
  fi
  rm -f "$srv" "$cli"
}

# ---------------------------------------------------------------------------
# Nebenlast-Arm: der Teil, der Head-of-Line-Blocking WIRKLICH messen kann.
#
# Warum es den braucht: das Ping-Pong im Bench ist strikt im Gleichtakt --
# es ist immer genau EINE Nachricht unterwegs. Eine Warteschlange, in der
# sich eine kleine Nachricht hinter einem 1-MiB-Brocken anstellen koennte,
# existiert dort per Konstruktion nicht. Der Fahrplan-Mix (--mix) zeigt
# deshalb die KADENZ-Kosten, aber kein Blocking. Und selbst wenn es Blocking
# gaebe: bei ~100 ms Burst-Abstand und ~7 us je kleiner Nachricht liegen nur
# rund 0,02 % der kleinen Nachrichten neben einem Burst -- ein p99 kann das
# gar nicht sehen, dafuer braucht es p99,9/p99,99.
#
# Der ehrliche Aufbau ist deshalb: ein ZWEITER Prozess schiebt dauerhaft
# 1-MiB-Nachrichten ueber DENSELBEN Pfad, waehrend die Messstrecke ihre
# 20-KiB-Kadenz faehrt. Erst dann teilen sich zwei Stroeme eine NIC und ein
# BAR-Fenster, und erst dann ist "die kleine Nachricht wartet auf den
# Brocken" ueberhaupt moeglich. Gemessen wird die kleine Strecke einmal
# allein und einmal unter dieser Nebenlast.
# ---------------------------------------------------------------------------
mix_load_unit() {
  local pair="$1" a_ord="$2" b_ord="$3" ort="$4" modus="$5" load="$6"
  local sopts="--sizes=20480 --depth=1 --iters=1000 --warmup=200 --secs=$MIX_SECS"
  local lopts="--sizes=1048576 --depth=1 --iters=100000 --warmup=50 --secs=$(( MIX_SECS + 8 ))"
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: mixload $pair ort=$ort modus=$modus last=$load"; return 0
  fi
  if [ "$a_ord" != "-1" ] && [ "${#HELD_LOCKS[@]}" = "0" ]; then
    log "  ABBRUCH mixload $pair: Rig-1-Ordinal ohne Kartenlock"; return 1
  fi
  clear_leftovers
  local lpid="" lrpid="" lport=$((23456 + RANDOM % 5000))
  if [ "$load" = "mit" ]; then
    # Nebenlast starten und kurz anlaufen lassen, damit sie beim Messen
    # wirklich schon auf dem Pfad ist und nicht erst hochfaehrt.
    if [ "$ort" = "intra" ]; then
      ( CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$b_ord" "$NIC_INTRA" "$lport" "$modus" $lopts \
          > /dev/null 2>&1 & echo $! > /tmp/cr_load_srv.pid
        sleep 0.5
        CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$lport" "$modus" "$a_ord" $lopts \
          > /dev/null 2>&1 ) &
      lpid=$!
    else
      lrpid=$(r2_start_bg load.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_BIN server -1 $NIC2 $lport $modus $lopts")
      wait_listen_remote "$lport" 20 || log "  WARN Nebenlast-Server (fern) nicht bereit"
      ( CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client "$IP2" "$NIC1" "$lport" "$modus" "$a_ord" $lopts \
          > /dev/null 2>&1 ) &
      lpid=$!
    fi
    sleep 2
  fi

  local port=$((23456 + RANDOM % 5000))
  local srv cli n
  srv=$(mktemp); cli=$(mktemp)
  if [ "$ort" = "intra" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$b_ord" "$NIC_INTRA" "$port" "$modus" $sopts \
      > "$srv" 2>&1 &
    local spid=$!
    if wait_listen_local "$port" "$spid" 30; then
      CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$port" "$modus" "$a_ord" $sopts \
        > "$cli" 2>&1
    fi
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  else
    local rpid
    rpid=$(r2_start_bg srv.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_BIN server -1 $NIC2 $port $modus $sopts")
    if wait_listen_remote "$port" 30; then
      CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client "$IP2" "$NIC1" "$port" "$modus" "$a_ord" $sopts \
        > "$cli" 2>&1
    fi
    r2_kill_bg "$rpid"
  fi

  # Nebenlast beenden -- erst die eigenen PIDs, dann der Reste-Check.
  [ -n "$lpid" ] && { kill "$lpid" 2>/dev/null; wait "$lpid" 2>/dev/null; }
  [ -r /tmp/cr_load_srv.pid ] && { kill "$(cat /tmp/cr_load_srv.pid)" 2>/dev/null; rm -f /tmp/cr_load_srv.pid; }
  [ -n "$lrpid" ] && r2_kill_bg "$lrpid"
  clear_leftovers

  n=$(awk -F'\t' -v pair="$pair" -v ort="$ort" -v l="$load" '
        $1=="DATA" {
          printf "%s\t%s\t%s\tklein_unter_last\t%s\t%s\tlast=%s\t%s\t%s\t%s\n",
                 pair, ort, $2, $3, $6, l, $7, $8, $11; c++
        } END { print c+0 > "/dev/stderr" }' "$cli" 2>&1 >> "$RESULTS_FILE")
  if [ "${n:-0}" = "0" ]; then
    log "  FAIL mixload $pair/$ort/$modus/last=$load: keine DATA ($(tail -2 "$cli"|tr '\n' ' '))"
  else
    log "  ok   mixload $pair ort=$ort modus=$modus last=$load"
  fi
  rm -f "$srv" "$cli"
}

phase_mix() {
  log "PHASE mix: Head-of-Line-Blocking (gemischte Groessen auf einem Pfad)"
  note "# ---- Phase mix: Groessen-Mix auf EINEM Pfad ----"
  note "# Kadenz 20 KiB, jede $MIX_MID_EVERY. Runde 80 KiB, alle $MIX_BURST_MS ms ein 1-MiB-Burst."
  note "# Je Arm zweimal: MIT und OHNE Bursts bei identischer Kadenz. Die"
  note "# Kernzahl ist p99(klein,mit)/p99(klein,ohne) -- Head-of-Line-Blocking."
  note "# NCCL/SHM-Referenz ist hier NICHT dabei: der comm_suite-Worker faehrt"
  note "# feste Groessen je Lauf, ein gemischter Fahrplan waere neuer Messcode"
  note "# und damit genau die neue Batterie, die ausgeschlossen war."
  note "# pair	ort	modus	klasse	size_bytes	n	burst_every|last	p10_us	p50_us	p99_us"
  faults_begin

  local o5090="${ORD1[0a:00.0]:-}" o3080="${ORD1[05:00.0]:-}"
  if [ -z "$o5090" ] || [ -z "$o3080" ]; then
    log "  Karten nicht aufgeloest -- Phase entfaellt"; return 1
  fi
  acquire_card_lock "$o5090" || return 1
  acquire_card_lock "$o3080" || { release_card_lock "$o5090"; return 1; }

  intra_ip_add || { release_card_lock "$o3080"; release_card_lock "$o5090"; return 1; }
  local modus burst
  for modus in gdr stage; do
    for burst in ohne mit; do
      mix_unit "mix_5090_x_3080" "$o5090" "$o3080" intra "$modus" "$burst"
    done
  done
  intra_ip_del

  # Cross-rig derselbe Mix, wenn das Budget es hergibt (5090 sendet).
  for modus in gdr stage; do
    for burst in ohne mit; do
      mix_unit "mix_5090_x_rig2host" "$o5090" -1 cross "$modus" "$burst"
    done
  done

  # Nebenlast-Arm: das eigentliche Head-of-Line-Mass (siehe Kopfkommentar
  # von mix_load_unit). Intra-rig, beide Modi, je einmal ohne und mit Last.
  intra_ip_add && {
    for modus in gdr stage; do
      for burst in ohne mit; do
        mix_load_unit "mixload_5090_x_3080" "$o5090" "$o3080" intra "$modus" "$burst"
      done
    done
    intra_ip_del
  }
  for modus in gdr stage; do
    for burst in ohne mit; do
      mix_load_unit "mixload_5090_x_rig2host" "$o5090" -1 cross "$modus" "$burst"
    done
  done

  release_card_lock "$o3080"; release_card_lock "$o5090"
  faults_end mix
}

# ===========================================================================
# Phase loadsym -- symmetrischer Lastvergleich (04_OFFEN 1a), NIC-Seite.
#
# Der bisherige Lastvergleich stand auf drei wackligen Beinen: nur eine
# Groesse (20 KiB), NCCL p50 gegen NIC p99, und die Fremdlast lief ueber die
# NIC, traf NCCL also nur indirekt. Hier wird die Last symmetrisch definiert:
# ein ZWEITER 1-MiB-Ping-Pong-Strom desselben Kartenpaars ueber DENSELBEN
# Pfad wie die Messstrecke. Die NCCL-Seite faehrt exakt dieselbe
# Lastdefinition ueber ihren eigenen Pfad (System-RAM/SHM):
# nccl_loadsym_run.sh, laeuft im Container. Beide Seiten berichten p99.
#
# Unabhaengige Lastverifikation (Antwort auf 04_OFFEN 1c, den verdaechtigen
# 0,99x-Punkt): die ibv-Portzaehler (port_xmit_data/port_rcv_data) werden vor
# und nach dem Messfenster gelesen. Liegt der Byte-Umsatz des Fensters nur
# knapp ueber dem der Messstrecke selbst, hat die Last das Fenster NICHT
# getroffen -- dann ist die Zeile als unverifiziert markiert zu lesen.
# ===========================================================================
LOADSYM_SIZES="${CR_LOADSYM_SIZES:-20480,81920,1048576}"
LOADSYM_SECS="${CR_LOADSYM_SECS:-5}"
LOADSYM_LOAD_SECS=$(( LOADSYM_SECS * 3 + 30 ))

ibv_counters() { # <ibdev> -> "<xmit_bytes> <rcv_bytes>" (Zaehler sind 4-Byte-Worte)
  local d="/sys/class/infiniband/$1/ports/1/counters"
  awk -v x="$(cat "$d/port_xmit_data" 2>/dev/null || echo 0)" \
      -v r="$(cat "$d/port_rcv_data" 2>/dev/null || echo 0)" \
      'BEGIN { printf "%.0f %.0f", x*4, r*4 }'
}

# loadsym_unit <pair> <a_ord> <b_ord> <intra|cross> <modus> <depth> <ohne|mit>
# a_ord sendet (Client), b_ord empfaengt (Server); cross: b_ord=-1 ist
# Rig-2-Host-RAM. Nebenlast laeuft zwischen denselben Endpunkten.
loadsym_unit() {
  local pair="$1" a_ord="$2" b_ord="$3" ort="$4" modus="$5" depth="$6" load="$7"
  local sopts="--sizes=$LOADSYM_SIZES --depth=$depth --iters=2000 --warmup=200 --secs=$LOADSYM_SECS"
  local lopts="--sizes=1048576 --depth=1 --iters=200000 --warmup=50 --secs=$LOADSYM_LOAD_SECS"
  local nicdev; [ "$ort" = "intra" ] && nicdev="$NIC_INTRA" || nicdev="$NIC1"
  if [ "$DRY_RUN" = "1" ]; then
    log "  dry-run: loadsym $pair ort=$ort modus=$modus d=$depth last=$load"
    note "# dry-run	loadsym	$pair	$ort	$modus	$depth	$load"
    return 0
  fi
  if [ "$a_ord" != "-1" ] && [ "${#HELD_LOCKS[@]}" = "0" ]; then
    log "  ABBRUCH loadsym $pair: Rig-1-Ordinal ohne gehaltenen Kartenlock"
    return 1
  fi
  clear_leftovers
  local lpid="" lrpid="" lport=$((23456 + RANDOM % 5000))
  if [ "$load" = "mit" ]; then
    if [ "$ort" = "intra" ]; then
      ( CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$b_ord" "$NIC_INTRA" "$lport" "$modus" $lopts \
          > /dev/null 2>&1 & echo $! > /tmp/cr_loadsym_srv.pid
        sleep 0.5
        CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$lport" "$modus" "$a_ord" $lopts \
          > /dev/null 2>&1 ) &
      lpid=$!
    else
      lrpid=$(r2_start_bg load.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_BIN server -1 $NIC2 $lport $modus $lopts")
      wait_listen_remote "$lport" 20 || log "  WARN Nebenlast-Server (fern) nicht bereit"
      ( CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client "$IP2" "$NIC1" "$lport" "$modus" "$a_ord" $lopts \
          > /dev/null 2>&1 ) &
      lpid=$!
    fi
    sleep 3   # Last anlaufen lassen, nicht in die Rampe messen
  fi

  local c0 c1 t0 t1
  c0=$(ibv_counters "$nicdev"); t0=$(date +%s.%N)

  local port=$((23456 + RANDOM % 5000))
  while [ "$port" = "$lport" ]; do port=$((23456 + RANDOM % 5000)); done
  local srv cli n rc=0
  srv=$(mktemp); cli=$(mktemp)
  if [ "$ort" = "intra" ]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" server "$b_ord" "$NIC_INTRA" "$port" "$modus" $sopts \
      > "$srv" 2>&1 &
    local spid=$!
    if wait_listen_local "$port" "$spid" 30; then
      CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client 127.0.0.1 "$NIC_INTRA" "$port" "$modus" "$a_ord" $sopts \
        > "$cli" 2>&1 || rc=1
    else
      log "  FAIL loadsym $pair: Server lauscht nicht: $(tail -2 "$srv"|tr '\n' ' ')"; rc=1
    fi
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  else
    local rpid
    rpid=$(r2_start_bg srv.log "CUDA_DEVICE_ORDER=PCI_BUS_ID $R2_BIN server -1 $NIC2 $port $modus $sopts")
    if wait_listen_remote "$port" 30; then
      CUDA_DEVICE_ORDER=PCI_BUS_ID "$BIN" client "$IP2" "$NIC1" "$port" "$modus" "$a_ord" $sopts \
        > "$cli" 2>&1 || rc=1
    else
      log "  FAIL loadsym $pair: ferner Server lauscht nicht"; rc=1
    fi
    r2_kill_bg "$rpid"
  fi

  c1=$(ibv_counters "$nicdev"); t1=$(date +%s.%N)

  # Nebenlast beenden -- eigene PIDs zuerst, dann der Reste-Check.
  [ -n "$lpid" ] && { kill "$lpid" 2>/dev/null; wait "$lpid" 2>/dev/null; }
  [ -r /tmp/cr_loadsym_srv.pid ] && { kill "$(cat /tmp/cr_loadsym_srv.pid)" 2>/dev/null; rm -f /tmp/cr_loadsym_srv.pid; }
  [ -n "$lrpid" ] && r2_kill_bg "$lrpid"
  clear_leftovers

  n=$(awk -F'\t' -v pair="$pair" -v ort="$ort" -v l="$load" '
        $1=="DATA" {
          # DATA modus size depth ro iters p10 median p90 bytes_per_round p99 max
          printf "%s\t%s\t%s\t%s\t%s\tlast=%s\t%s\t%s\t%s\t%s\t%s\n",
                 pair, ort, $2, $4, $3, l, $6, $7, $8, $9, $11; c++
        } END { print c+0 > "/dev/stderr" }' "$cli" 2>&1 >> "$RESULTS_FILE")
  note "$(awk -v pair="$pair" -v ort="$ort" -v m="$modus" -v d="$depth" -v l="$load" \
             -v t0="$t0" -v t1="$t1" -v c0="$c0" -v c1="$c1" 'BEGIN {
        split(c0, a, " "); split(c1, b, " ")
        printf "# loadsym_verify\t%s\t%s\t%s\td%s\tlast=%s\twindow_s=%.1f\txmit_MB=%.0f\trcv_MB=%.0f",
               pair, ort, m, d, l, t1-t0, (b[1]-a[1])/1e6, (b[2]-a[2])/1e6 }')"
  if [ "${n:-0}" = "0" ]; then
    log "  FAIL loadsym $pair/$ort/$modus/d=$depth/last=$load: keine DATA ($(tail -2 "$cli"|tr '\n' ' '))"
    rc=1
  else
    log "  ok   loadsym $pair ort=$ort modus=$modus d=$depth last=$load -> $n Punkte"
  fi
  rm -f "$srv" "$cli"
  return $rc
}

phase_loadsym() {
  log "PHASE loadsym: symmetrischer Lastvergleich (04_OFFEN 1a) -- NIC-Seite"
  note "# ---- Phase loadsym: p99 unter symmetrischer Nebenlast ----"
  note "# Nebenlast = zweiter 1-MiB-Ping-Pong-Strom desselben Kartenpaars ueber"
  note "# DENSELBEN Pfad. Die NCCL-Seite faehrt dieselbe Lastdefinition ueber"
  note "# System-RAM (nccl_loadsym_run.sh, Container). Zeitbudget je Punkt:"
  note "# ${LOADSYM_SECS}s. Tiefen-Achse (d=4) nur auf gdr intra -- prueft, ob"
  note "# die Dispatcher-Hypothese (immer direkt + Tiefe >= 4) auch unter Last"
  note "# traegt. us-Werte sind der halbe Round-trip EINER RUNDE (depth"
  note "# Nachrichten je Runde, Zeit je Nachricht = Wert/depth)."
  note "# pair	ort	modus	depth	size_bytes	last	n	p10_us	p50_us	p90_us	p99_us"
  faults_begin

  local o5090="${ORD1[0a:00.0]:-}" o3080="${ORD1[05:00.0]:-}"
  if [ -z "$o5090" ] || [ -z "$o3080" ]; then
    log "  Karten nicht aufgeloest -- Phase entfaellt"; return 1
  fi
  acquire_card_lock "$o5090" || return 1
  acquire_card_lock "$o3080" || { release_card_lock "$o5090"; return 1; }

  local d load modus
  intra_ip_add && {
    for d in 1 4; do
      for load in ohne mit; do
        loadsym_unit "ls_5090_x_3080" "$o5090" "$o3080" intra gdr "$d" "$load"
      done
    done
    for load in ohne mit; do
      loadsym_unit "ls_5090_x_3080" "$o5090" "$o3080" intra stage 1 "$load"
    done
    # A-vs-A-Rauschboden: derselbe Arm doppelt, sonst ist kein Unterschied
    # als echt behauptbar.
    loadsym_unit "ls_5090_x_3080_avsa" "$o5090" "$o3080" intra gdr 1 ohne
    intra_ip_del
  }

  # Cross-rig: der 1c-Verdachtspunkt (gdr unter Last 0,99x) mit
  # Zaehler-Beleg nachgefahren; stage als Kontrast.
  for modus in gdr stage; do
    for load in ohne mit; do
      loadsym_unit "ls_5090_x_rig2host" "$o5090" -1 cross "$modus" 1 "$load"
    done
  done

  release_card_lock "$o3080"; release_card_lock "$o5090"
  faults_end loadsym
}

# ===========================================================================
# Main
# ===========================================================================
log "crossrig_ladder_run.sh Start -- Phase=$PHASE dry_run=$DRY_RUN"
log "Ergebnisse: $RESULTS_FILE"
{
  echo "# Cross-Rig-Groessenleiter $TS -- Phase $PHASE"
  echo "# rig1 $IP1/$NIC1  <->  rig2 $IP2/$NIC2"
  echo "# median_us = halber Round-trip EINER RUNDE; eine Runde traegt depth"
  echo "# Nachrichten, die Zeit je Nachricht ist median/depth."
  echo "# MB_per_s = bytes_per_round / (2*median_us), also ueber die volle Runde."
} > "$RESULTS_FILE"

if [ "$DRY_RUN" = "0" ]; then
  for f in "$BIN"; do
    [ -x "$f" ] || { log "FAIL: $f fehlt -- erst scripts/probe/build.sh laufen lassen"; exit 1; }
  done
  R2 "test -x $R2_BIN_HOST || test -x $R2_BIN" >/dev/null 2>&1 || {
    log "FAIL: kein Bench-Binary auf Rig 2 unter $R2_DIR/bin -- erst dort bauen"; exit 1; }
fi

resolve_cards

case "$PHASE" in
  wire)     phase_wire ;;
  neutral)  phase_neutral ;;
  ti)       phase_ti ;;
  gpu)      phase_gpu ;;
  intra)    phase_intra ;;
  pressure) phase_pressure ;;
  mix)      phase_mix ;;
  loadsym)  phase_loadsym ;;
  all)      phase_wire; phase_neutral; phase_ti; phase_gpu; phase_intra
            phase_pressure; phase_mix ;;
esac

clear_leftovers
log "fertig. Ergebnistabelle: $RESULTS_FILE"
log "Log: $LOG_FILE"
