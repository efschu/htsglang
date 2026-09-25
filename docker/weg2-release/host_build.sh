#!/bin/bash
# Host-Bau des htsglang-Upgrade-Images MIT RESSOURCEN-DECKEL -- ENTWURF (27B-Sitz R, 25.09.2026). NICHT GELAUFEN.
# Ausfuehrung nur nach Go des Operators, als root auf dem Proxmox-Host:
#
#   CTX=/spinning/gpu-arb/docker/ctx/27b-b5c7d01614 EXPECT_MANIFEST=<sha256 von MANIFEST.sha256> \
#     bash /spinning/subvol-999-disk-0/spinning/gpu-arb/docker/host_build.sh [--dry-run]
#
#   --dry-run  nur die Vorbedingungen pruefen und die exakten Befehle ausgeben; STRIKT LESEND: kein Log-Verzeichnis,
#              keine Log-Datei (Ausgabe nur auf stdout), kein Builder, kein Pull, kein Schreiben in src/.git (git
#              laeuft mit --no-optional-locks), keine Prozesse in CT999 (Boot-Pruefung liest /proc des Hosts).
#
# DECKEL -- warum die Haus-Dienste sicher bleiben:
#   * Ein EIGENER buildx-Builder "htsglang-build" (Treiber docker-container). Jeder Bau-Prozess (apt, pip, nvcc,
#     cargo, der JIT-Vorbau) laeuft in DIESEM einen Container, dessen cgroup hart begrenzt ist:
#       memory = memory-swap = $BUILD_MEM   (kein Swap; ein OOM bleibt im Builder, nur der Bau stirbt)
#       cpu-quota = $BUILD_CPUS CPUs        (von 32 Threads des 5950X)
#       cpu-shares = $CPU_SHARES            (Gewicht; Haus-Container und CT999 laufen mit 1024 = Vorrang bei Konkurrenz)
#     KEIN oom_score_adj am Builder-Init: am 25.09. (02:36:53Z) liess er den memcg-OOM-Killer docker-init (PID 1 des
#     Builders, rss 292 kB, adj 500) statt des cicc-Prozesses toeten -- der ganze Bau starb. Ohne ihn trifft ein OOM im
#     Deckel den groessten Prozess (ein nvcc/cicc), der Vorbau meldet das Modul, der Bau laeuft weiter.
#     Der Standard-Builder dieses Hosts ist "ikbuilder" (fremd) -- er wird weder benutzt noch veraendert, und
#     `docker buildx create` ohne --use aendert die Auswahl nicht.
#   * Vorbedingungen, lesend: Host-MemAvailable >= BUILD_MEM + MARGIN_GIB; kein Boot in CT999 (die Last des Baus
#     wuerde dessen Messungen verfaelschen und CT999 braucht bis ~91 GiB; ALLOW_WITH_CT999_BOOT=1 uebersteuert
#     bewusst); ZFS-Dataset spinning/docker frei >= MIN_FREE_GIB.
#   * Der Kontext wird so gebaut, wie er geprueft wurde: MANIFEST.sha256 muss EXPECT_MANIFEST sein und alle
#     Dateien darin muessen stimmen; src/ muss auf der Revision aus BUILD_INFO.json stehen und sauber sein.
#   * Nie: pushen, `docker system prune`, Haus-Container, fremde Builder, ghcr.io/efschu/* oder *cu130-nccl2307*
#     anfassen (F12/F14). Das Image bekommt einen NEUEN versionierten Tag; ein vorhandener Tag wird verweigert.
#   * Platten-I/O laesst sich fuer ZFS nicht deckeln (ionice greift dort nicht) -- darum der CT999-Boot-Riegel.
#
# SCHAETZUNG (aus Rig-Messungen 25.09., siehe Bericht):
#   Zeit ~1-2 h: JIT-Vorbau ~35-45 min (FlashInfer 120f 78 min + 86 38 min Einzelschritte laut .ninja_log bei
#   MAX_JOBS=4, dazu barlink/htccl ~5-12 min), pip-Installation aus dem Lock ~10-25 min (16 GiB installiert,
#   Download je nach Leitung), apt/rustup/sglang-Rust-Ext ~10-15 min, Basis-Pull im Builder (~10 GB) und
#   Export per --load (~30 GB) je ~5-10 min.
#   Platz: Image ~30 GiB (Basis 10,2 + venv 16 + JIT/tvm-ffi ~1 + Rust/apt/src ~2-3); Builder-Cache ~40 GiB
#   (Schichten + pip-Cache); Spitze ~70-80 GiB scheinbar, auf spinning/docker (zstd, Verhaeltnis dort 1,53x)
#   eher ~45-55 GiB belegt. Frei am 25.09.: 824 GiB.
set -euo pipefail

DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
S=${S_ROOT-/spinning/subvol-999-disk-0}   # CT999-Wurzel, wie der Host sie sieht (Test in CT999: S_ROOT= leer)
LXC=999
CTX=${CTX:?CTX = Kontext im LXC-Pfad, z.B. /spinning/gpu-arb/docker/ctx/27b-b5c7d01614}
HCTX=$S$CTX
EXPECT_MANIFEST=${EXPECT_MANIFEST:?EXPECT_MANIFEST = sha256 von MANIFEST.sha256 aus Rs Bericht}
LINE=$(jq -r .line "$HCTX/BUILD_INFO.json")
REV=$(jq -r .revision "$HCTX/BUILD_INFO.json")
RELEASE=${RELEASE:-$([ "$LINE" = nf ] && echo rc1 || echo rc5)}   # Tag: 27B rc5, NF rc1 (Operator 25.09.)
IMAGE=${IMAGE:-htsglang:cu129-weg2-${LINE}-${RELEASE}-${REV:0:10}}
BUILDER=${BUILDER:-htsglang-build}
BUILD_MEM=${BUILD_MEM:-32g}          # ganze GiB mit 'g'; 24g reichte am 25.09. nicht (gemm_sm120 bei MAX_JOBS=4 > 23 GiB)
BUILD_CPUS=${BUILD_CPUS:-12}
CPU_SHARES=${CPU_SHARES:-256}
MAX_JOBS=${MAX_JOBS:-3}              # parallele nvcc-Schritte im JIT-Vorbau (Dockerfile ARG MAX_JOBS)
PREBUILD_STRICT=${PREBUILD_STRICT:-0}   # 1 = jeder Vorbau-Modulfehler bricht den Bau ab
MARGIN_GIB=${MARGIN_GIB:-8}
MIN_FREE_GIB=${MIN_FREE_GIB:-150}
BASE=${BASE:-nvidia/cuda:12.9.1-devel-ubuntu24.04}
BUILDKIT_IMAGE=${BUILDKIT_IMAGE:-moby/buildkit:buildx-stable-1}   # liegt auf dem Host (242 MB)
LOGDIR=${LOGDIR:-$S/spinning/docker-acceptance/$LINE}
if [ "$DRY" = 1 ]; then LOG=/dev/null; else LOG=$LOGDIR/build_$(date -u +%m%d_%H%M%S).log; mkdir -p "$LOGDIR"; fi

say(){ echo "[host-build $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
die(){ say "ABBRUCH: $*"; exit 1; }
# block: eine Vorbedingung des BAUS fehlt. Echter Lauf -> Abbruch. --dry-run -> benennen und weiter pruefen,
# damit der Trockenlauf das ganze Bild liefert (Deckel-Plan, RAM, Platz, Befehl) und am Ende urteilt.
BLOCKED=()
block(){ if [ "$DRY" = 1 ]; then say "VERWEIGERUNG (Bau): $*"; BLOCKED+=("$*"); else die "$*"; fi; }
run(){ say "\$ $*"; [ "$DRY" = 1 ] || "$@"; }

# --- 1. Vorbedingungen (nur lesen) ---------------------------------------------------------
say "== Vorbedingungen (Kontext $HCTX, Image $IMAGE, Builder $BUILDER, Deckel ${BUILD_MEM} / ${BUILD_CPUS} CPUs / shares ${CPU_SHARES})"
case "$BUILD_MEM" in *g) ;; *) die "BUILD_MEM in ganzen GiB mit 'g' angeben (z.B. 24g)";; esac
[ "$(sha256sum "$HCTX/MANIFEST.sha256" | cut -d' ' -f1)" = "$EXPECT_MANIFEST" ] \
  || die "MANIFEST.sha256 != EXPECT_MANIFEST -- der Kontext ist nicht der gepruefte"
( cd "$HCTX" && sha256sum --quiet -c MANIFEST.sha256 ) || die "Kontext-Dateien weichen vom Manifest ab"
# src/ gehoert im unprivilegierten LXC uid 100000 (safe.directory); --no-optional-locks: `git status` schreibt den
# aufgefrischten Index NICHT zurueck (sonst legte root des Hosts eine index-Datei an, die CT999 nicht mehr aendern kann).
G=(git --no-optional-locks -c safe.directory='*' -C "$HCTX/src")
[ "$("${G[@]}" rev-parse HEAD)" = "$REV" ] || die "src/ HEAD != Revision $REV aus BUILD_INFO.json"
[ -z "$("${G[@]}" status --porcelain)" ] || die "src/ nicht sauber"
say "Kontext: Manifest ok, src/ @ ${REV:0:10} sauber, Linie $LINE, $(du -sh "$HCTX" | cut -f1)"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then block "$IMAGE existiert schon -- neuer Tag noetig (F14)"; fi
avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
need=$(( ${BUILD_MEM%g} + MARGIN_GIB ))
[ "$avail" -ge "$need" ] || block "Host-MemAvailable ${avail} GiB < ${need} GiB (BUILD_MEM ${BUILD_MEM} + ${MARGIN_GIB}) -- CT999 entlasten oder BUILD_MEM/MAX_JOBS senken"
# Boot-Pruefung ueber /proc des Hosts (CT999-Prozesse sind dort sichtbar; Zuordnung ueber die cgroup), ohne pct exec.
# Muster eng: Prozesstitel sglang::scheduler* und `-m sglang.srt.weg2.launcher` / `-m sglang.launch_server` mit
# maskierten Punkten. Das alte Muster ([s]glang.srt.weg2.launcher, '.' = beliebiges Zeichen) traf am 25.09. 02:02:58Z
# einen Agenten-Befehl in CT999 mit dem PFAD python/sglang/srt/weg2/launcher.py und verweigerte den Bau falsch.
# Die [x]-Klammern verhindern den Selbsttreffer, wenn jemand dieses Muster in einer Befehlszeile traegt.
BOOT_RE='sglang::schedule[r]|-m sglang\.srt\.weg2\.launche[r]|-m sglang\.launch_serve[r]'
boots=""
for p in $(pgrep -f -- "$BOOT_RE" 2>/dev/null || true); do
  cg=$(cut -d: -f3 "/proc/$p/cgroup" 2>/dev/null | head -1)
  case "$cg" in /lxc/"$LXC"/*|/lxc.payload."$LXC"/*) boots+=" CT$LXC:$p" ;; *) boots+=" host:$p($cg)" ;; esac
done
if [ -n "$boots" ]; then
  if [ "${ALLOW_WITH_CT999_BOOT:-0}" = 1 ]; then say "WARN: Boot laeuft (${boots# }), Bau trotzdem (ALLOW_WITH_CT999_BOOT=1)"
  else block "sglang-Boot laeuft (${boots# }) -- Bau erst danach (Messungen, Host-RAM); ALLOW_WITH_CT999_BOOT=1 uebersteuert"; fi
else
  say "kein sglang-Boot auf dem Host oder in CT$LXC (CT$LXC: $(pct status "$LXC" 2>/dev/null | awk '{print $2}'))"
fi
free_gib=$(( $(zfs list -H -p -o avail spinning/docker) / 1073741824 ))
[ "$free_gib" -ge "$MIN_FREE_GIB" ] || block "spinning/docker frei ${free_gib} GiB < ${MIN_FREE_GIB} GiB"
docker buildx version >/dev/null 2>&1 || die "docker buildx fehlt"
say "Host: MemAvailable ${avail} GiB, spinning/docker frei ${free_gib} GiB, aktueller Builder $(docker buildx ls 2>/dev/null | awk 'NR>1 && $1 ~ /\*$/ {print $1; exit}') (bleibt)"

# --- 2. F12: Host-Images nur hier, nur gelistet ------------------------------------------------------
# Loeschen nur mit PRUNE_HOST_IMAGES=1 UND PRUNE_PATTERN (Regex auf repo:tag, vom Nutzer benannt) -- nie pauschal.
# "Genutzt" = die Image-ID irgendeines Containers (laufend oder gestoppt) oder dessen Referenz (ohne Tag = :latest).
say "== F12 Host-Images: Kandidaten (von keinem Container genutzt; nicht ghcr.io/efschu/*, nicht *cu130-nccl2307*, nicht nvidia/cuda, nicht moby/buildkit)"
used_ids=$(docker ps -aq | xargs -r docker inspect -f '{{.Image}}' 2>/dev/null | sort -u)
used_refs=$(docker ps -a --format '{{.Image}}' | awk '{r=$1; if (r !~ /[:@]/) r=r":latest"; print r}' | sort -u)
cand=()
while read -r id ref; do
  case "$ref" in ghcr.io/efschu/*|*cu130-nccl2307*|nvidia/cuda:*|moby/buildkit:*) continue ;; esac
  if grep -qxF -e "$id" <<< "$used_ids" || grep -qxF -e "$ref" <<< "$used_refs"; then continue; fi
  if docker images --no-trunc --format '{{.ID}} {{.Repository}}:{{.Tag}}' | awk -v i="$id" '$1==i{print $2}' \
       | grep -qE '^ghcr\.io/efschu/|cu130-nccl2307'; then continue; fi
  [ "$ref" = "<none>:<none>" ] && ref=$id
  cand+=("$ref"); say "  $ref ($(docker image inspect -f '{{.Size}}' "$ref" 2>/dev/null | awk '{printf "%.1f GB", $1/1e9}'))"
done < <(docker images --no-trunc --format '{{.ID}} {{.Repository}}:{{.Tag}}')
if [ "${PRUNE_HOST_IMAGES:-0}" = 1 ] && [ "${#cand[@]}" -gt 0 ]; then
  [ -n "${PRUNE_PATTERN:-}" ] || die "PRUNE_HOST_IMAGES=1 ohne PRUNE_PATTERN -- welche Kandidaten weg duerfen, benennt der Nutzer (Regex auf repo:tag)"
  for r in "${cand[@]}"; do
    [[ "$r" =~ $PRUNE_PATTERN ]] || continue
    run docker rmi "$r" || say "  WARN: rmi $r verweigert"
  done
else
  say "nur gelistet (${#cand[@]}); loeschen mit PRUNE_HOST_IMAGES=1 PRUNE_PATTERN='<regex auf repo:tag>'"
fi

# --- 3. Builder mit Deckel ------------------------------------------------------------------
if docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
  say "Builder $BUILDER vorhanden -- Deckel pruefen"
else
  run docker buildx create --name "$BUILDER" --driver docker-container \
    --driver-opt "image=$BUILDKIT_IMAGE" \
    --driver-opt "memory=$BUILD_MEM" --driver-opt "memory-swap=$BUILD_MEM" \
    --driver-opt cpu-period=100000 --driver-opt "cpu-quota=$(( BUILD_CPUS * 100000 ))" \
    --driver-opt "cpu-shares=$CPU_SHARES" --bootstrap
fi
BK=buildx_buildkit_${BUILDER}0
if [ "$DRY" = 0 ]; then
  lim=$(docker inspect -f '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}} {{.HostConfig.CpuQuota}} {{.HostConfig.CpuPeriod}} {{.HostConfig.CpuShares}}' "$BK")
  read -r m ms cq cp cs <<< "$lim"
  [ "$m" = "$(( ${BUILD_MEM%g} * 1073741824 ))" ] && [ "$ms" = "$m" ] && [ "$cq" = "$(( BUILD_CPUS * 100000 ))" ] \
    && [ "$cp" = 100000 ] && [ "$cs" = "$CPU_SHARES" ] \
    || die "Builder-Container $BK hat nicht den verlangten Deckel ($lim) -- 'docker buildx rm $BUILDER' und neu"
  say "Deckel aktiv: memory=$m memory-swap=$ms cpu-quota=$cq/$cp cpu-shares=$cs (kein oom_score_adj am Init, siehe Kopf)"
fi

# --- 4. Basis-Digest festschreiben ----------------------------------------------------------------
if ! docker image inspect "$BASE" >/dev/null 2>&1; then run docker pull "$BASE"; fi
BASE_PIN=$(docker image inspect --format '{{index .RepoDigests 0}}' "$BASE" 2>/dev/null || echo "$BASE")
say "Basis: $BASE_PIN (der Builder zieht sie per Digest selbst)"

# --- 5. Bau --------------------------------------------------------------------------------
NV=0; [ "$(jq -r .nv_headers.in_image "$HCTX/BUILD_INFO.json")" = true ] && NV=1
NCCL_SHA=$(jq -r '.nccl.sha256 // empty' "$HCTX/BUILD_INFO.json")
PUSH_STATE=$(jq -r '.push_state // "unbekannt"' "$HCTX/BUILD_INFO.json")
case "$PUSH_STATE" in UNPUSHED*) say "!! Revision ${REV:0:10}: $PUSH_STATE -- das Image traegt das Label htsglang.push_state, host_publish.sh sperrt" ;; esac
T0=$(date +%s)
say "== Bau (Log $LOG)"
BUILD=(docker buildx build --builder "$BUILDER" --load --progress=plain --provenance=false
       -f "$HCTX/Dockerfile"
       --build-arg "CUDA_BASE=$BASE_PIN" --build-arg "HTSGLANG_REVISION=$REV" --build-arg "HTSGLANG_LINE=$LINE"
       --build-arg "SGLANG_SCM_VERSION=0.0.0.dev0+${LINE}.${RELEASE}.${REV:0:10}" --build-arg "WITH_NV_HEADERS=$NV"
       --build-arg "NCCL_SHA256=$NCCL_SHA" --build-arg "MAX_JOBS=$MAX_JOBS" --build-arg "PREBUILD_STRICT=$PREBUILD_STRICT"
       --build-arg "PUSH_STATE=$PUSH_STATE"
       -t "$IMAGE" "$HCTX")
say "\$ ${BUILD[*]}"
if [ "$DRY" = 1 ]; then
  if [ "${#BLOCKED[@]}" -gt 0 ]; then say "DRY-RUN URTEIL: Bau wuerde JETZT verweigert (${#BLOCKED[@]} Grund/Gruende, oben VERWEIGERUNG)"
  else say "DRY-RUN URTEIL: alle Vorbedingungen erfuellt -- der Bau wuerde jetzt starten"; fi
  say "DRY-RUN: nichts angelegt, nichts gebaut"; exit 0
fi
"${BUILD[@]}" >>"$LOG" 2>&1 || die "Bau gescheitert nach $(( ($(date +%s) - T0) / 60 )) min (siehe $LOG)"
say "Bau fertig nach $(( ($(date +%s) - T0) / 60 )) min"

# --- 6. Belege ------------------------------------------------------------------------------
say "Image $IMAGE: id $(docker image inspect -f '{{.Id}}' "$IMAGE"), $(docker image inspect -f '{{.Size}}' "$IMAGE" | awk '{printf "%.1f GB", $1/1e9}')"
say "NCCL im Bau-Log: $(grep -a -m1 -oE 'bundled libnccl: /opt/.*' "$LOG" || echo '?')"
say "Builder-Cache: $(docker buildx du --builder "$BUILDER" 2>/dev/null | tail -1)"
JV=$(docker run --rm --network none --entrypoint cat "$IMAGE" /opt/htsglang/JIT_PREBUILD.json 2>/dev/null | jq -r '.verdict // "?"')
if [ "$JV" = OK ]; then say "JIT-Vorbau: verdict OK"
else say "WARN: JIT-Vorbau verdict=$JV -- Fehlerzeilen im Log ([prebuild]   FEHLER), fehlende Module baut der erste Boot"; fi
say "D1 selfcheck:"
set +e
docker run --rm --network none --security-opt apparmor=unconfined -e MODE=weg2 "$IMAGE" selfcheck 2>&1 | tee -a "$LOG"; d1=${PIPESTATUS[0]}
docker run --rm --network none --security-opt apparmor=unconfined -e MODE=weg2 "$IMAGE" version 2>&1 | tee -a "$LOG"
set -e
say "D1 selfcheck rc=$d1 (0 = bestanden)"
say "Protokoll: $LOG. Builder $BUILDER bleibt (inkrementelle Neubauten); freigeben: docker buildx rm $BUILDER"
# Image gebaut, aber Selbstpruefung rot: eigener Exit-Code, damit das nicht als gruener Bau durchgeht.
[ "$d1" = 0 ] || exit 3
