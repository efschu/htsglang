#!/bin/bash
# Abgekoppelter Host-Bau (27B-Sitz R, 25.09.2026): startet host_build.sh (ECHTER Lauf, Vorbedingungen greifen, nichts
# uebersteuert) und zeichnet waehrenddessen alle 5 s Speicherwerte in eine CSV auf -- ein Instrument, keine Meldungen:
#   Host-MemAvailable, und fuer den Builder-Container buildx_buildkit_<BUILDER>0 (sobald es ihn gibt) memory.current,
#   anon, file aus seiner cgroup. Am Ende eine .rc-Datei mit Exit-Code und memory.peak des Builders.
# Aufruf auf dem Host, abgekoppelt (ueberlebt eine SSH-Trennung):
#   setsid nohup bash /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/host_build_detached.sh <TS> \
#     </dev/null >/dev/null 2>&1 &
# Ergebnisse unter /spinning/subvol-999-disk-0/spinning/docker-acceptance/<linie>/ (in CT999 lesbar als
# /spinning/docker-acceptance/<linie>/): build_detached_<TS>.out, build_mem_<TS>.csv, build_detached_<TS>.rc,
# dazu das eigene Log von host_build.sh (build_<zeit>.log).
set -u
TS=${1:?TS fehlt}
S=/spinning/subvol-999-disk-0
CTX=${CTX:?CTX (Kontext im LXC-Pfad) fehlt}
EXPECT_MANIFEST=${EXPECT_MANIFEST:?EXPECT_MANIFEST (Digest aus Rs Bericht) fehlt}
BUILDER=${BUILDER:-htsglang-build}
LINE=$(jq -r .line "$S$CTX/BUILD_INFO.json")
D=$S/spinning/docker-acceptance/$LINE
mkdir -p "$D"
OUT=$D/build_detached_$TS.out; SAMP=$D/build_mem_$TS.csv; RC=$D/build_detached_$TS.rc
echo "ts_utc,host_memavailable_kib,builder_mem_current_b,builder_anon_b,builder_file_b" > "$SAMP"
CTX="$CTX" EXPECT_MANIFEST="$EXPECT_MANIFEST" BUILDER="$BUILDER" \
  bash "$S/spinning/gpu-arb/docker/host_build.sh" > "$OUT" 2>&1 &
BPID=$!
echo "$BPID" > "$D/build_detached_$TS.pid"
CG=""
while kill -0 "$BPID" 2>/dev/null; do
  if [ -z "$CG" ]; then
    id=$(docker inspect -f '{{.Id}}' "buildx_buildkit_${BUILDER}0" 2>/dev/null || true)
    [ -n "$id" ] && [ -d "/sys/fs/cgroup/system.slice/docker-$id.scope" ] && CG=/sys/fs/cgroup/system.slice/docker-$id.scope
  fi
  ma=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
  cur=""; anon=""; file=""
  if [ -n "$CG" ] && [ -r "$CG/memory.current" ]; then
    cur=$(cat "$CG/memory.current" 2>/dev/null)
    anon=$(awk '/^anon /{print $2}' "$CG/memory.stat" 2>/dev/null)
    file=$(awk '/^file /{print $2}' "$CG/memory.stat" 2>/dev/null)
  fi
  echo "$(date -u +%FT%TZ),$ma,$cur,$anon,$file" >> "$SAMP"
  sleep 5
done
wait "$BPID"; rc=$?
# memory.peak gilt nur seit dem letzten (Neu-)Start des Builder-Containers -- darum zusaetzlich die Maxima der Proben
# und die Neustart-Zahl (25.09.: ein OOM-Neustart setzte memory.peak auf 7 MB zurueck).
peak=""; [ -n "$CG" ] && peak=$(cat "$CG/memory.peak" 2>/dev/null)
restarts=$(docker inspect -f '{{.RestartCount}}' "buildx_buildkit_${BUILDER}0" 2>/dev/null || echo ?)
smax=$(awk -F, 'NR>1 && $3!="" {if ($3>c) c=$3; if ($4>a) a=$4} NR>1 && $2!="" {if (m=="" || $2<m) m=$2} END {printf "samples_max_current_b=%s samples_max_anon_b=%s samples_min_memavailable_kib=%s", c, a, m}' "$SAMP")
echo "rc=$rc builder_memory_peak_b=${peak:-?} builder_restarts=$restarts $smax cgroup=${CG:-?} end_utc=$(date -u +%FT%TZ) out=$OUT samples=$SAMP" > "$RC"
