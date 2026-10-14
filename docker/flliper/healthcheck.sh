#!/bin/sh
# Docker-HEALTHCHECK fuer beide Modi -- VORSCHLAG 27.09. (SW, Release-Punkt "healthy trotz toter Gruppe").
#
# BEFUND (b1 dkr27breleasedraftbar1w109270932, 27.09.): P-Rang PP1 starb 09:45:58 an "#1233 W27 PP WIDTH DIVERGENCE
# REFUSED" und ging in den DEBUG-HOLD (Prozess lebt weiter, Marker debug_hold/rc12gb1..._rank1_..._094558Z.txt).
# Die HTTP-Seite der Gruppe P beantwortete /health weiter mit 200 (ein 200 ist nur ein Transport-Fakt), also
# lieferte die Front /health 200 und Docker blieb "healthy". Der erste WEG2-HEALTH-Fehlschlag kam 09:53:47, der
# W17-Stopp 09:54:27 -- 8,5 min nach dem Tod. b23 (10:27:56) zeigte dasselbe Bild.
#
# DIESE FASSUNG prueft VOR dem /health-Aufruf drei Fakten, die kein 200 wegerklaeren kann, und ist KLEBRIG:
# einmal tot heisst bis zum Container-Ende unhealthy (eine Gruppe mit totem Rang erholt sich nicht).
#   1. Marker /tmp/htsglang/hc_dead (vorheriger Befund dieses Laufs)
#   2. DEBUG-HOLD-Marker eines Rangs dieses Laufs in $SGLANG_DEBUG_HOLD_DIR
#   3. benannter Stopp oder process_alive=False in den Boot-Logs DIESES Tags (inkrementell ab gespeichertem Offset)
# Danach wie bisher: Front-/health (200 nur, wenn beide Gruppen 200 und state != STOP, front.py handle_health).
#
# Muster NUR benannte Stopps. KEIN nacktes REFUSED ('#915 PREFETCH REFUSED reason=too_short' ist harmlos), KEIN
# nacktes SPLIT ('FI-GRAPH-SPLIT off' ist harmlos).
# Testbar ohne Container: HC_STATE (sonst /tmp/htsglang), HC_EVIDENCE, HC_CURL (sonst curl).
HC_STATE=${HC_STATE:-/tmp/htsglang}
HC_EVIDENCE=${HC_EVIDENCE:-${SGLANG_WEG2_EVIDENCE_DIR:-${FLLIPER_PDFLIP_EVIDENCE_DIR:-/var/lib/htsglang/evidence}}}
HC_CURL=${HC_CURL:-curl}
# NUR Toedliches, weil der Befund klebrig ist (Operator-Review 27.09.): KEIN 'Traceback' und KEIN 'CUDA out of memory' --
# beide kommen auch in abgefangenen Pfaden vor (logger.exception, OOM-Retry beim Graph-Capture); ein falsches klebriges
# unhealthy wuerde bei einer Restart-Policy einen gesunden Server abschiessen. Der toedliche Scheduler-Pfad heisst
# 'Scheduler hit an exception' (der Traceback steht darunter).
# Kein nacktes 'W27 ' (27.09. Fehlalarm: die Erklaerzeile "WEG2-LAUNCH ... #1245 ARMED ... W27 ..." traf es; der echte
# W27-Stopp heisst '... W27 PP WIDTH DIVERGENCE REFUSED' und faellt unter 'DIVERGENCE REFUSED'). WEG2-LAUNCH-Zeilen sind
# Launcher-Prosa und werden vor dem Muster ausgefiltert (HC_SKIP).
# 28.09. PA: alle Muster in beiden Schreibweisen (sglang/weg2 und flliper/pdflip), RENAME_PLAN 8.7-1.
# 28.09.: Flip-Deadlock-Klasse (NF rc12z4 06:07) nur mit vollem Namen -- 'W109' allein teilen sich Weg2VisionTimeout
# und Weg2PhaseFootprintBootCollision, 'W126' kommt auch in Prosa vor.
HC_RE='DIVERGENCE REFUSED|DIGEST MISMATCH|LIVELOCK REFUSED|#791b|SPLIT refused|ADMISSION SPLIT|Scheduler hit an exception|DEBUG-HOLD rank=|process_alive=False|(WEG2|PDFLIP) STOP |W109 (Weg2|PdFlip)XchgCreditCycleRefused|W121 (Weg2Flip|PdFlip)PeerLegAborted|W126 (Weg2|PdFlip)WakeCreditCycle'
# 12.10. NF-Befund (Dauerstand int24f, front.log 11:25:07): die INFO-Erklaerzeile 'WEG2-FLIP-ORDER epoch=.. src=..' traegt in
# ihrer Modell-Prosa 'W126 Weg2WakeCreditCycle (model: credit order: ...)' und machte einen gesunden Server klebrig unhealthy
# (FailingStreak 624). Gleiche Klasse wie WEG2-LAUNCH 27.09.: Flip-Ordnungs-Prosa wird vor dem Muster ausgefiltert.
HC_SKIP='(WEG2|PDFLIP)-LAUNCH|(WEG2|PDFLIP)-FLIP-ORDER'

dead() {  # $1 = Grund; klebrig ablegen, ausgeben, unhealthy
  mkdir -p "$HC_STATE" 2>/dev/null
  [ -s "$HC_STATE/hc_dead" ] || printf '%s\n' "$1" > "$HC_STATE/hc_dead"
  echo "unhealthy: $(head -1 "$HC_STATE/hc_dead")"
  exit 1
}

if [ "${MODE:-server}" = "weg2" ] || [ "${MODE:-server}" = "pdflip" ]; then
  [ -s "$HC_STATE/hc_dead" ] && dead "$(head -1 "$HC_STATE/hc_dead")"
  # 2. DEBUG-HOLD: ein haltender Rang ist ein toter Rang, auch wenn sein Prozess lebt.
  # Doppelname (RENAME_PLAN 8.7-1): ein umgestelltes CONTAINER_ENV traegt FLLIPER_DEBUG_HOLD_*.
  HC_HOLD_DIR=${SGLANG_DEBUG_HOLD_DIR:-${FLLIPER_DEBUG_HOLD_DIR:-}}; HC_HOLD_TAG=${SGLANG_DEBUG_HOLD_TAG:-${FLLIPER_DEBUG_HOLD_TAG:-}}
  if [ -n "$HC_HOLD_DIR" ] && [ -d "$HC_HOLD_DIR" ]; then
    for m in "$HC_HOLD_DIR/$HC_HOLD_TAG"*_rank*.txt; do
      [ -f "$m" ] || continue
      # nur Marker dieses Container-Laufs (juenger als der Start-Stempel des Entrypoints, falls vorhanden)
      if [ -e "$HC_STATE/serve_port" ] && [ ! "$m" -nt "$HC_STATE/serve_port" ]; then continue; fi
      dead "DEBUG-HOLD $(basename "$m")"
    done
  fi
  # 3. Logs dieses Tags, nur neue Bytes (Offset je Datei in $HC_STATE/hc_off.<name>).
  if [ -n "${HTSGLANG_TAG:-}" ]; then
    for f in "$HC_EVIDENCE"/boot_weg2_"$HTSGLANG_TAG"_*.log "$HC_EVIDENCE"/boot_pdflip_"$HTSGLANG_TAG"_*.log; do
      [ -f "$f" ] || continue
      case "$f" in *.P.log|*.D.log|*.front.log) ;; *) continue ;; esac
      o=$(cat "$HC_STATE/hc_off.$(basename "$f")" 2>/dev/null || echo 0)
      case "$o" in *[!0-9]*|"") o=0 ;; esac
      n=$(wc -c < "$f")
      [ "$n" -lt "$o" ] && o=0   # Datei neu angelegt
      hit=$(tail -c +"$((o + 1))" "$f" 2>/dev/null | grep -a -v -E "$HC_SKIP" | grep -a -m1 -E "$HC_RE" | cut -c1-200)
      mkdir -p "$HC_STATE" 2>/dev/null; echo "$n" > "$HC_STATE/hc_off.$(basename "$f")"
      [ -n "$hit" ] && dead "$(basename "$f"): $hit"
    done
  fi
  P=$(cat "$HC_STATE/serve_port" 2>/dev/null || echo 30030)
  case "$P" in *[!0-9]*|"") P=30030 ;; esac
  [ -r "$HC_STATE/stand" ] && echo "stand $(cat "$HC_STATE/stand")"
  # curl-Codes (22, 28, ...) auf 1 abbilden: Docker kennt nur 0/1, 2 ist reserviert.
  $HC_CURL -fsS -m 8 -o /dev/null "http://127.0.0.1:${P}/health" || exit 1
  exit 0
fi
$HC_CURL -fsS -m 8 -o /dev/null "http://127.0.0.1:${PORT:-30000}/health" || exit 1
exit 0
