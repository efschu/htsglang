#!/bin/bash
# Veroeffentlichungsschritt fuer ein gebautes htsglang-Image -- ENTWURF (27B-Sitz R, 25.09.2026). NICHT GELAUFEN.
# Veroeffentlicht wird NUR mit ausdruecklichem Go des Nutzers (F3/F14); dieses Skript ist die Stelle, an der die
# Sperren sitzen. Ohne --push prueft es nur und gibt die Befehle aus.
#
#   IMAGE=htsglang:cu129-weg2-nf-rc1-d93a17316b TARGET=ghcr.io/efschu/htsglang:cu129-weg2-nf-rc1-d93a17316b \
#     USER_GO="<Wortlaut des Nutzer-Go mit Datum>" bash host_publish.sh [--push]
#
# SPERREN (jede einzeln ein Abbruch):
#   1. USER_GO fehlt                                   -> keine Veroeffentlichung ohne Wortlaut des Nutzers.
#   2. Revision (Label org.opencontainers.image.revision) liegt auf KEINEM Remote-Branch des Repos
#      -- das ist der Fall "UNPUSHED, Veroeffentlichung erst nach Push durch den Nutzer" (Label htsglang.push_state
#      bzw. BUILD_INFO.json .push_state). Geprueft wird der Git-Stand, nicht nur das Label: nach dem Push durch den
#      Nutzer und einem `git fetch` im Repo ist die Sperre von selbst offen, ohne Neubau.
#   3. TARGET waere der oeffentliche Alt-Tag *cu130-nccl2307* -> nur mit REPLACE_PUBLIC=1 (F14: Ersetzen erst mit Go).
#   4. TARGET existiert schon in der Registry (docker manifest inspect) -> nie ueberschreiben (versionierte Tags).
set -euo pipefail
PUSH=0; [ "${1:-}" = "--push" ] && PUSH=1
S=${S_ROOT-/spinning/subvol-999-disk-0}
REPO=$S/spinning/htsglang
IMAGE=${IMAGE:?IMAGE (lokaler Tag) fehlt}
TARGET=${TARGET:?TARGET (Registry-Tag) fehlt}
say(){ echo "[host-publish $(date -u +%H:%M:%SZ)] $*"; }
die(){ say "VERWEIGERT: $*"; exit 1; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "$IMAGE gibt es lokal nicht"
REV=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")
PSTATE=$(docker image inspect -f '{{index .Config.Labels "htsglang.push_state"}}' "$IMAGE")
LINE=$(docker image inspect -f '{{index .Config.Labels "htsglang.line"}}' "$IMAGE")
say "Image $IMAGE: Linie $LINE, Revision ${REV:-?}, Label push_state='${PSTATE:-?}'"

[ -n "${USER_GO:-}" ] || die "USER_GO fehlt -- Veroeffentlichung nur mit ausdruecklichem Go des Nutzers (Wortlaut + Datum)"
[ -n "$REV" ] || die "Image ohne Revision-Label"
ON=$(git --no-optional-locks -c safe.directory='*' -C "$REPO" branch -r --contains "$REV" 2>/dev/null | sed 's/^[* ]*//' | grep -v -- '->' | paste -sd, - || true)
if [ -z "$ON" ]; then
  die "Revision ${REV:0:10} liegt auf keinem Remote-Branch: UNPUSHED, Veroeffentlichung erst nach Push durch den Nutzer (danach git fetch im Repo, dann ist diese Sperre offen)"
fi
case "$PSTATE" in UNPUSHED*) say "Label sagt UNPUSHED (Stand beim Bau), Git sagt inzwischen: auf $ON -- Sperre 2 offen" ;; esac
case "$TARGET" in *cu130-nccl2307*) [ "${REPLACE_PUBLIC:-0}" = 1 ] || die "TARGET ist der oeffentliche Alt-Tag -- Ersetzen nur mit REPLACE_PUBLIC=1 und Nutzer-Go (F14)";; esac
if docker manifest inspect "$TARGET" >/dev/null 2>&1; then die "$TARGET existiert schon in der Registry -- nie ueberschreiben, neuen versionierten Tag waehlen"; fi

say "Go des Nutzers: $USER_GO"
say "\$ docker tag $IMAGE $TARGET"
say "\$ docker push $TARGET"
if [ "$PUSH" = 1 ]; then
  docker tag "$IMAGE" "$TARGET"
  docker push "$TARGET"
  say "gepusht: $TARGET ($(docker image inspect -f '{{index .RepoDigests 0}}' "$TARGET" 2>/dev/null || echo 'Digest nach dem Push lesen'))"
else
  say "nur geprueft (ohne --push)"
fi
