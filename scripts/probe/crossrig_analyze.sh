#!/bin/bash
# SPDX-License-Identifier: MIT
#
# crossrig_analyze.sh -- rechnet aus den TSV-Dateien von crossrig_ladder_run.sh
# genau die Groessen aus, an denen die Vorhersagen V1..V5 haengen. Reine
# Auswertung, kein Messcode: liest nur, schreibt nur nach stdout.
#
#   V1  BAR-Lesen (VRAM ist Quelle) gegen BAR-Schreiben (VRAM ist Ziel)
#   V2  Aufschlag aus V1 je Karte -- PIX-Karte gegen die beiden Root-Port-Karten
#   V3  Umschlagpunkt direkt/stage je Karte (BAR-Groesse)
#   V5  Solo gegen Parallel (Degradationsfaktor)
#   Tiefe: Zeit je Nachricht ueber die Tiefe, Wire-Sockel danebengestellt
#
# Aufruf: crossrig_analyze.sh <results-dir>
set -uo pipefail
DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"

pick() { ls -t "$DIR"/crossrig_$1_*.tsv 2>/dev/null | head -1; }
GPU=$(pick gpu); WIRE=$(pick wire); INTRA=$(pick intra); PRESS=$(pick pressure); TI=$(pick ti)

echo "== Dateien =="
for f in "$WIRE" "$TI" "$GPU" "$INTRA" "$PRESS"; do [ -n "$f" ] && echo "  $f"; done

# ---------------------------------------------------------------------------
# Hilfsfunktion: median_us fuer eine Zeile herausziehen
# Spalten: pair dir modus ro depth size iters p10 median p90 MB/s
# ---------------------------------------------------------------------------
med() {  # med <file> <pair-regex> <dir> <modus> <ro> <depth> <size>
  awk -F'\t' -v p="$2" -v d="$3" -v m="$4" -v r="$5" -v dp="$6" -v s="$7" \
    '$1 ~ p && $2==d && $3==m && $4==r && $5==dp && $6==s {print $9; exit}' "$1"
}

echo
echo "== V1/V3: Richtung und Umschlag je Rig-1-Karte (gegen Rig-2-Host-RAM) =="
echo "# b2a = Rig 2 schreibt IN die BAR (posted) | a2b = Rig 1 liest AUS der BAR (non-posted)"
printf "%-10s %-9s %10s %10s %10s %10s %9s\n" Karte Groesse "b2a_gdr" "b2a_stage" "a2b_gdr" "a2b_stage" "lesen/schr"
for pci in 05:00.0 0a:00.0 0b:00.0; do
  for sz in 8 20480 81920 1048576 4194304; do
    bg=$(med "$GPU" "^c_${pci}_hostpeer" b2a gdr off 1 $sz)
    bs=$(med "$GPU" "^c_${pci}_hostpeer" b2a stage off 1 $sz)
    ag=$(med "$GPU" "^c_${pci}_hostpeer" a2b gdr off 1 $sz)
    as=$(med "$GPU" "^c_${pci}_hostpeer" a2b stage off 1 $sz)
    [ -z "$bg" ] && continue
    ratio=$(awk -v a="$ag" -v b="$bg" 'BEGIN{ if(b>0) printf "%.2f", a/b; else printf "-" }')
    printf "%-10s %-9s %10s %10s %10s %10s %9s\n" "$pci" "$sz" "$bg" "$bs" "$ag" "$as" "${ratio}x"
  done
done

echo
echo "== V3: Umschlagpunkt (erste Groesse, ab der stage < gdr) =="
for pci in 05:00.0 0a:00.0 0b:00.0; do
  for dir in b2a a2b; do
    first="kein Umschlag (gdr gewinnt ueberall)"
    for sz in 8 4096 16384 20480 65536 81920 262144 1048576 4194304; do
      g=$(med "$GPU" "^c_${pci}_hostpeer" $dir gdr off 1 $sz)
      s=$(med "$GPU" "^c_${pci}_hostpeer" $dir stage off 1 $sz)
      [ -z "$g" ] || [ -z "$s" ] && continue
      if awk -v g="$g" -v s="$s" 'BEGIN{exit !(s<g)}'; then first="$sz B"; break; fi
    done
    printf "  %-10s %-4s -> %s\n" "$pci" "$dir" "$first"
  done
done

echo
echo "== Tiefen-Achse 5090 gegen Wire-Sockel: Zeit je Nachricht (us) =="
printf "%-9s %-4s %-6s %10s %10s %10s %12s\n" Groesse dir arm d1 d4 d16 "Gewinn_d16"
for sz in 20480 81920 1048576; do
  for dir in b2a a2b; do
    for arm in "5090:^c_0a:00.0_(hostpeer|depth)" "wire:^wire"; do
      name="${arm%%:*}"; pat="${arm#*:}"
      f="$GPU"; [ "$name" = "wire" ] && f="$WIRE"
      out=""
      for d in 1 4 16; do
        v=$(med "$f" "$pat" $dir gdr off $d $sz)
        [ -z "$v" ] && v=$(med "$f" "$pat" $dir stage off $d $sz)
        out="$out $(awk -v v="${v:-0}" -v d="$d" 'BEGIN{ if(v>0) printf "%.2f", v/d; else printf "-" }')"
      done
      set -- $out
      gain=$(awk -v a="$1" -v c="$3" 'BEGIN{ if(a>0&&c>0) printf "%.2fx", a/c; else printf "-" }')
      printf "%-9s %-4s %-6s %10s %10s %10s %12s\n" "$sz" "$dir" "$name" "$1" "$2" "$3" "$gain"
    done
  done
done

echo
echo "== Intra-Rig: NIC-Relay gegen NCCL/System-RAM =="
[ -n "$INTRA" ] && awk -F'\t' '/^(i_|n_)/ {printf "  %-26s %-16s %9s B  %10s us  %10s MB/s\n", $1, $3, $6, $9, $11}' "$INTRA"

echo
echo "== V5: Solo gegen Parallel =="
[ -n "$PRESS" ] && awk -F'\t' '/^p_/ {printf "  %-30s %-10s %9s B  %10s us  %10s MB/s\n", $1, $2, $6, $9, $11}' "$PRESS"

echo
echo "== Fault-Zaehler =="
for f in "$WIRE" "$TI" "$GPU" "$INTRA" "$PRESS"; do
  [ -n "$f" ] && grep -h "^# faults" "$f" 2>/dev/null
done
