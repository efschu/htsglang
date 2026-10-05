#!/usr/bin/env bash
# rigdash (Release-Edition, nur Profil-Editor) im Container -- Auftrag 1995, Nutzer-Entscheid 05.10.: "Der Profil-Editor kommt INS
# Release; ein Release, das nur der Nutzer bedienen kann, ist keins."  Zweiter Dienst im Image neben der Front (und neben userdash).
#
# Zwei Verwendungen, EINE Datei:
#   a) GESOURCED vom Image-Entrypoint, direkt nach dem userdash-Start (MODE=weg2, Server laeuft):
#        . /opt/htsglang/rigdash/entrypoint_rigdash.sh && rigdash_start
#      Erwartet say/refuse, PY, LOGCOPY (und optional HOME_DIR, TREE, STAND, PROFILE_DIR, USER_PROFILE_DIR, FRONT_PORT) aus dem Entrypoint.
#   b) AUSGEFUEHRT (MODE=editor des Entrypoints, `docker run -e MODE=editor ...`): nur der Editor im Vordergrund, kein Server, keine GPU.
#        /opt/htsglang/rigdash/entrypoint_rigdash.sh        (oder:  bash entrypoint_rigdash.sh --check   druckt die Kommandozeile)
#
# Schalter (FLLIPER_<X> wird vom Entrypoint auf HTSGLANG_<X> abgebildet, EP_PRODUCT_NAMES):
#   HTSGLANG_RIGDASH=1|0              an (Default) | aus -- aus = kein Prozess, kein Port
#   HTSGLANG_RIGDASH_PORT=30081       Container-Port; nie 30030/30031/30032/30080/30097/30099/8890/8428, nie FRONT_PORT
#   HTSGLANG_RIGDASH_BIND=0.0.0.0     im Container; veroeffentlicht wird mit `-p 127.0.0.1:30081:30081` (der Editor hat KEINEN Zugriffsschutz)
#   HTSGLANG_RIGDASH_LINE=27b|nf      welcher Code-Stand des Images den Planer-Baum stellt (Profile werden gegen EINEN Baum geprueft);
#                                     im Server-Modus die Linie des gestarteten Profils, sonst 27b
#   HTSGLANG_RIGDASH_TRUST_PROXY=1    der Betreiber sitzt selbst hinter einem Reverse-Proxy (X-Forwarded-*): Editor antwortet auch dahinter.
#                                     Der PROXY muss anmelden -- der Editor schreibt ins State-Volume.
#   HTSGLANG_RIGDASH_MODEL_ROOTS=a:b  Verzeichnisse, unter denen der Editor Modelle schaetzen darf (Standard: der Modell-Cache wie im README)
#   HTSGLANG_PROFILES_DIR             wohin der Editor Nutzerprofile (JSON) schreibt und woher der Entrypoint sie liest
#                                     (FLLIPER_PROFILES_DIR, Standard /var/lib/flliper/profiles) -- EIN Ort fuer beide
# Der Dienst ist rigdash --edition release --editor-only: kein Probennehmer, kein /api/live, kein gpuq, keine Messung, nichts am Rig.
# Er baut ein Profil und startet nichts. Fehlt das Paket oder der Planer-Baum, bootet der Server trotzdem (rigdash haengt nie im
# Docker-HEALTHCHECK); im MODE=editor ist das ein Fehler (rc 3).
RIGDASH_DIR=${RIGDASH_DIR:-/opt/htsglang/rigdash}
RIGDASH_PID=""

declare -F say >/dev/null || say() { printf '[rigdash %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
declare -F refuse >/dev/null || refuse() { say "REFUSED $1: $2"; exit 3; }

# rigdash_cmd: baut die Kommandozeile in das Feld RIGDASH_ARGV, setzt RIGDASH_TREE_PY; Rueckgabe 0 = startbar, 1 = nicht startbar (Grund per say)
rigdash_cmd() {
  local home=${HOME_DIR:-/opt/htsglang} line tree p
  : "${HTSGLANG_RIGDASH:=1}"
  : "${HTSGLANG_RIGDASH_PORT:=30081}"
  : "${HTSGLANG_RIGDASH_BIND:=0.0.0.0}"
  : "${HTSGLANG_RIGDASH_TRUST_PROXY:=0}"
  case "$HTSGLANG_RIGDASH" in
    0) say "RIGDASH aus (HTSGLANG_RIGDASH=0)"; return 1 ;;
    1) ;;
    *) refuse RIGDASH "HTSGLANG_RIGDASH='$HTSGLANG_RIGDASH' (0|1)" ;;
  esac
  case "$HTSGLANG_RIGDASH_TRUST_PROXY" in 0|1) ;; *) refuse RIGDASH "HTSGLANG_RIGDASH_TRUST_PROXY='$HTSGLANG_RIGDASH_TRUST_PROXY' (0|1)" ;; esac
  [[ $HTSGLANG_RIGDASH_PORT =~ ^[0-9]+$ ]] || refuse RIGDASH "HTSGLANG_RIGDASH_PORT='$HTSGLANG_RIGDASH_PORT' ist keine Portnummer"
  case "$HTSGLANG_RIGDASH_PORT" in
    30030|30031|30032|30080|30097|30099|8890|8428|"${FRONT_PORT:-30030}")
      refuse RIGDASH "HTSGLANG_RIGDASH_PORT=$HTSGLANG_RIGDASH_PORT kollidiert (Front/P/D 30030-30032, userdash 30080, Router 30097/30099, rigdash des Rigs 8890, VM 8428, Server ${FRONT_PORT:-30030})" ;;
  esac
  if [ ! -f "$RIGDASH_DIR/rigdash/__main__.py" ] || [ ! -f "$RIGDASH_DIR/kartenplan_build/couplings_worker.py" ]; then
    say "WARN: RIGDASH an, aber $RIGDASH_DIR/rigdash oder kartenplan_build fehlt im Image -- kein Editor"
    return 1
  fi
  line=${HTSGLANG_RIGDASH_LINE:-${STAND:-27b}}
  case "$line" in 27b|nf) ;; *) refuse RIGDASH "HTSGLANG_RIGDASH_LINE='$line' (27b|nf)" ;; esac
  tree=$home/src-$line
  [ -d "$tree/python" ] || tree=${TREE:-$home/src}      # Image ohne zwei Staende (aelterer Bau): der eine Baum
  RIGDASH_TREE_PY=$tree/python
  if [ ! -f "$RIGDASH_TREE_PY/sglang/srt/weg2/profile_json.py" ]; then
    say "WARN: Planer-Baum $RIGDASH_TREE_PY traegt sglang/srt/weg2/profile_json.py nicht (Code-Stand ohne Profil-Editor-Module, oder umbenannter Baum python/flliper) -- kein Editor"
    return 1
  fi
  RIGDASH_ARGV=(-m rigdash --edition release --editor-only --host "$HTSGLANG_RIGDASH_BIND" --port "$HTSGLANG_RIGDASH_PORT"
                --docker-ssh '' --gpuq http://127.0.0.1:1 --vm-url ''
                --profil-tree "$RIGDASH_TREE_PY" --couplings-python "${PY:-/opt/venv/bin/python}"
                --profile-dir "${HTSGLANG_PROFILES_DIR:-/var/lib/flliper/profiles}"
                --profiles-release-dir "${PROFILE_DIR:-$home/profiles}")
  if [ -n "${HTSGLANG_RIGDASH_MODEL_ROOTS:-}" ]; then
    local IFS=:
    for p in $HTSGLANG_RIGDASH_MODEL_ROOTS; do [ -z "$p" ] || RIGDASH_ARGV+=(--model-root "$p"); done
  fi
  [ "$HTSGLANG_RIGDASH_TRUST_PROXY" != 1 ] || RIGDASH_ARGV+=(--trust-proxy)
  return 0
}

rigdash_start() {
  rigdash_cmd || return 0
  local logd=${LOGCOPY:-/tmp/htsglang}
  mkdir -p "$logd" "${HTSGLANG_PROFILES_DIR:-/var/lib/flliper/profiles}" 2>/dev/null || true
  ( cd "$RIGDASH_DIR" && exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" PYTHONPATH="$RIGDASH_DIR" \
      nice -n 10 "${PY:-/opt/venv/bin/python}" "${RIGDASH_ARGV[@]}" ) >> "$logd/rigdash.log" 2>&1 &
  RIGDASH_PID=$!
  say "RIGDASH :$HTSGLANG_RIGDASH_PORT (pid $RIGDASH_PID, Editor-only, Planer-Baum $RIGDASH_TREE_PY, Profile ${HTSGLANG_PROFILES_DIR:-/var/lib/flliper/profiles}, Log $logd/rigdash.log, Gesundheit GET /healthz; aus: HTSGLANG_RIGDASH=0)"
}

rigdash_stop() {
  [ -n "$RIGDASH_PID" ] && kill -TERM "$RIGDASH_PID" 2>/dev/null
  RIGDASH_PID=""
  return 0
}

# MODE=editor: ausgefuehrt (nicht gesourct) -> Vordergrund
rigdash_foreground() {
  local rc=0
  rigdash_cmd || rc=$?
  [ "$rc" = 0 ] || refuse RIGDASH "Editor nicht startbar (siehe oben)"
  if [ "${1:-}" = "--check" ]; then
    printf 'cd %s && env -u PYTHONPATH CUDA_VISIBLE_DEVICES= PYTHONPATH=%s %s' "$RIGDASH_DIR" "$RIGDASH_DIR" "${PY:-/opt/venv/bin/python}"
    printf ' %q' "${RIGDASH_ARGV[@]}"; printf '\n'
    return 0
  fi
  mkdir -p "${HTSGLANG_PROFILES_DIR:-/var/lib/flliper/profiles}" 2>/dev/null || true
  say "RIGDASH (MODE=editor, Vordergrund) :$HTSGLANG_RIGDASH_PORT, Planer-Baum $RIGDASH_TREE_PY, Profile ${HTSGLANG_PROFILES_DIR:-/var/lib/flliper/profiles}"
  cd "$RIGDASH_DIR" || refuse RIGDASH "$RIGDASH_DIR fehlt"
  exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" PYTHONPATH="$RIGDASH_DIR" "${PY:-/opt/venv/bin/python}" "${RIGDASH_ARGV[@]}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -Eeuo pipefail
  rigdash_foreground "$@"
fi
