#!/usr/bin/env bash
# make_rigdash_ctx.sh -- rigdash (Release-Edition, nur Profil-Editor) in einen FERTIGEN Bau-Kontext legen.
# Auftrag 1995 (Nutzer-Entscheid 05.10.: der Profil-Editor kommt INS Release; Lead: zweiter Dienst im Image neben der Front).
#
# Nachbearbeitung eines Kontexts, den make_flat_ctx.sh / make_delta_ctx.sh schon gebaut haben (Muster userdash_ctx.sh), NEUE Datei: die
# Originale bleiben unveraendert. Schritte (alle zusammen oder keiner):
#   1. tools/rigdash/{rigdash/,kartenplan_build/,entrypoint_rigdash.sh} aus REPO @ REV (git archive, OHNE rigdash/tests und rigdash/deploy);
#      kartenplan_build/couplings_worker.py ist der Kopplungs-Worker (Image-Python /opt/venv/bin/python, PYTHONPATH=<Baum>/python).
#   2. tools/entrypoint.sh des Kontexts durch docker/pdflip-release/entrypoint.sh @ REV ersetzen (traegt den Haken rigdash_start und
#      MODE=editor). Nur wenn der Kontext die bekannte Basis (md5 ENTRYPOINT_BASE_MD5) oder schon die neue Fassung hat -- sonst Abbruch
#      (jemand hat den Entrypoint seit dem Commit 0471c2c05c geaendert: den Haken von Hand nachziehen), oder --keep-entrypoint.
#   3. an die Dockerfile-KOPIE im Kontext einen Block anhaengen: COPY nach /opt/htsglang/rigdash, Importprobe im Image-venv,
#      LABEL io.github.efschu.flliper.rigdash=<sha>, EXPOSE 30081. Die Dockerfile-Vorlagen bleiben unveraendert.
#
#   make_rigdash_ctx.sh --check   --ctx <dir> --rev <sha|ref> [--repo <git-dir>]   prueft alles, schreibt NICHTS (Voreinstellung)
#   make_rigdash_ctx.sh --dry-run --ctx <dir> --rev <sha|ref> [--repo <git-dir>]   wie --check + Dateiliste + Dockerfile-Block, schreibt NICHTS
#   make_rigdash_ctx.sh --apply   --ctx <dir> --rev <sha|ref> [--repo <git-dir>]   fuehrt aus
#   Weitere: --keep-entrypoint (Schritt 2 auslassen). Env: REPO (Standard /spinning/htsglang), RIGDASH_REV.
#   Test-Haken: RIGDASH_ALLOW_UNPUSHED=1 (sonst muss REV auf einem Remote-Zweig liegen, wie bei userdash_ctx.sh).
# Exit: 0 ok, 2 Aufruf, 3 Pruefung gescheitert (jede Ursache auf stderr, nichts geschrieben), 4 Ausfuehrung gescheitert.
# NICHT gebaut, kein Docker, keine GPU. Der Aufrufer baut danach wie bisher (host_build.sh).

set -uo pipefail

RIGDASH_LABEL=io.github.efschu.flliper.rigdash
ENTRYPOINT_BASE_MD5=f2d683b93e80a4f0cccf32ca6c0bc8e0     # docker/pdflip-release/entrypoint.sh @ 0471c2c05c (= Live-Stand 03.10. 21:15)
EP_REPO_PATH=docker/pdflip-release/entrypoint.sh
RD_PATHS=(tools/rig_dashboard/rigdash tools/rig_dashboard/kartenplan_build tools/rig_dashboard/entrypoint_rigdash.sh)
RD_EXCLUDES=(':(exclude)tools/rig_dashboard/rigdash/tests' ':(exclude)tools/rig_dashboard/rigdash/deploy')

err() { echo "make_rigdash_ctx: $*" >&2; }

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; }

MODE=check; CTX=""; REV_IN=${RIGDASH_REV:-}; REPO=${REPO:-/spinning/htsglang}; KEEP_EP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check ;;
    --dry-run) MODE=dry ;;
    --apply) MODE=apply ;;
    --keep-entrypoint) KEEP_EP=1 ;;
    --ctx) shift; CTX=${1:-} ;;
    --rev) shift; REV_IN=${1:-} ;;
    --repo) shift; REPO=${1:-} ;;
    -h|--help) usage; exit 0 ;;
    *) err "unbekannte Option '$1'"; usage >&2; exit 2 ;;
  esac
  shift
done
[ -n "$CTX" ] || { err "--ctx <dir> fehlt"; exit 2; }
[ -n "$REV_IN" ] || { err "--rev <sha|ref> (oder RIGDASH_REV) fehlt"; exit 2; }
case "$CTX" in /*) ;; *) err "--ctx muss ein absoluter Pfad sein ($CTX)"; exit 2 ;; esac

FAILS=()
fail() { FAILS+=("$*"); }

# ---- Pruefungen (schreiben nichts) ----------------------------------------------------------------------------------------
REV=$(git -C "$REPO" rev-parse --verify -q "${REV_IN}^{commit}") || { err "Revision '$REV_IN' unbekannt in $REPO"; exit 3; }
has() { git -C "$REPO" cat-file -e "$REV:$1" 2>/dev/null; }
for f in tools/rig_dashboard/rigdash/__main__.py tools/rig_dashboard/rigdash/server.py tools/rig_dashboard/rigdash/static/index.html \
         tools/rig_dashboard/kartenplan_build/couplings_worker.py tools/rig_dashboard/entrypoint_rigdash.sh "$EP_REPO_PATH"; do
  has "$f" || fail "${REV:0:10} traegt $f nicht"
done
if has tools/rig_dashboard/rigdash/server.py; then
  SRV=$(git -C "$REPO" show "$REV:tools/rig_dashboard/rigdash/server.py")      # erst einlesen: grep -q in einer Pipe + pipefail = Fehlalarm (SIGPIPE)
  grep -q -- '--editor-only' <<<"$SRV" || fail "${REV:0:10}: server.py kennt --editor-only nicht (Revision ohne Auftrag 1995)"
  grep -q -- '--profil-tree' <<<"$SRV" || fail "${REV:0:10}: server.py kennt --profil-tree nicht (Revision ohne Auftrag 1984 B)"
fi
if has "$EP_REPO_PATH"; then
  EPNEW=$(git -C "$REPO" show "$REV:$EP_REPO_PATH")
  grep -q 'rigdash_start' <<<"$EPNEW" && grep -q 'MODE:-}" = "editor"' <<<"$EPNEW" || fail "${REV:0:10}: $EP_REPO_PATH traegt den Haken (rigdash_start / MODE=editor) nicht"
  EPNEW_MD5=$(git -C "$REPO" show "$REV:$EP_REPO_PATH" | md5sum | cut -d' ' -f1)
fi
if [ "${RIGDASH_ALLOW_UNPUSHED:-0}" != 1 ]; then
  git -C "$REPO" branch -r --contains "$REV" 2>/dev/null | grep -q . || fail "${REV:0:10} ist nicht gepusht (kein Remote-Zweig enthaelt sie)"
fi
[ -d "$CTX" ] && [ -f "$CTX/Dockerfile" ] || fail "$CTX/Dockerfile fehlt (erst make_flat_ctx.sh bzw. make_delta_ctx.sh)"
if [ -d "$CTX" ]; then
  [ ! -e "$CTX/tools/rigdash" ] || fail "$CTX/tools/rigdash existiert schon"
  [ ! -f "$CTX/Dockerfile" ] || ! grep -q "$RIGDASH_LABEL" "$CTX/Dockerfile" || fail "Dockerfile traegt den rigdash-Block schon"
  if [ "$KEEP_EP" = 0 ]; then
    if [ ! -f "$CTX/tools/entrypoint.sh" ]; then fail "$CTX/tools/entrypoint.sh fehlt"
    else
      CTX_EP_MD5=$(md5sum < "$CTX/tools/entrypoint.sh" | cut -d' ' -f1)
      [ "$CTX_EP_MD5" = "$ENTRYPOINT_BASE_MD5" ] || [ "$CTX_EP_MD5" = "${EPNEW_MD5:-x}" ] \
        || fail "tools/entrypoint.sh des Kontexts (md5 ${CTX_EP_MD5:0:10}) ist weder die Basis (${ENTRYPOINT_BASE_MD5:0:10}) noch die neue Fassung (${EPNEW_MD5:0:10}): Haken von Hand nachziehen oder --keep-entrypoint"
    fi
  fi
fi
if [ "${#FAILS[@]}" -gt 0 ]; then
  for m in "${FAILS[@]}"; do err "FEHLT/FALSCH: $m"; done
  err "REFUSED (${#FAILS[@]} Befund(e)); nichts geschrieben"
  exit 3
fi

NFILES=$(git -C "$REPO" archive "$REV" "${RD_PATHS[@]}" "${RD_EXCLUDES[@]}" | tar -t | grep -vc '/$')
block() {
  cat <<EOF

# --- rigdash Profil-Editor (Nutzer-Entscheid 05.10.2026, Auftrag 1995) ---------------------------------------------------
# Angehaengt von make_rigdash_ctx.sh (REV=${REV}). rigdash --edition release --editor-only: stdlib, Kopplungs-Worker mit dem
# Image-Python. Start im Entrypoint nur, wenn /opt/htsglang/rigdash/entrypoint_rigdash.sh existiert und HTSGLANG_RIGDASH
# (FLLIPER_RIGDASH) nicht 0 ist, ODER mit MODE=editor als einziger Dienst. Port 30081 (HTSGLANG_RIGDASH_PORT), GET /healthz.
COPY tools/rigdash/ /opt/htsglang/rigdash/
RUN cd /opt/htsglang/rigdash && /opt/venv/bin/python -B -c "import rigdash.server, rigdash.profil, rigdash.profile_recompute, rigdash.hwprofil, rigdash.modellprofil; print('rigdash', rigdash.server._version())" \\
    && test -s rigdash/static/index.html && test -s kartenplan_build/couplings_worker.py && test -x entrypoint_rigdash.sh \\
    && bash -n entrypoint_rigdash.sh
LABEL ${RIGDASH_LABEL}="${REV}"
EXPOSE 30081
EOF
}

echo "make_rigdash_ctx: ${REV:0:10} ok: ${NFILES} Dateien (ohne rigdash/tests, rigdash/deploy), Kontext $CTX, Entrypoint $([ "$KEEP_EP" = 1 ] && echo "bleibt" || echo "wird ersetzt (md5 ${EPNEW_MD5:0:10})")"
case "$MODE" in
  check) echo "make_rigdash_ctx: --check: nichts geschrieben"; exit 0 ;;
  dry)
    echo "make_rigdash_ctx: --dry-run: wuerde tun:"
    echo "  1. git archive ${REV:0:10} ${RD_PATHS[*]} -> $CTX/tools/rigdash/ (strip 2)"
    [ "$KEEP_EP" = 1 ] || echo "  2. $EP_REPO_PATH @ ${REV:0:10} -> $CTX/tools/entrypoint.sh"
    echo "  3. an $CTX/Dockerfile anhaengen:"
    block | sed 's/^/     | /'
    git -C "$REPO" archive "$REV" "${RD_PATHS[@]}" "${RD_EXCLUDES[@]}" | tar -t | grep -v '/$' | sed 's#^tools/rig_dashboard/#  Datei: tools/rigdash/#' | head -400
    echo "make_rigdash_ctx: --dry-run: nichts geschrieben"; exit 0 ;;
esac

# ---- --apply ----------------------------------------------------------------------------------------------------------------
mkdir -p "$CTX/tools/rigdash" || { err "mkdir $CTX/tools/rigdash gescheitert"; exit 4; }
if ! git -C "$REPO" archive "$REV" "${RD_PATHS[@]}" "${RD_EXCLUDES[@]}" | tar -x --strip-components=2 -C "$CTX/tools/rigdash"; then
  err "git archive/tar gescheitert; entferne $CTX/tools/rigdash"; rm -rf "$CTX/tools/rigdash"; exit 4
fi
if [ ! -f "$CTX/tools/rigdash/rigdash/__main__.py" ] || [ ! -s "$CTX/tools/rigdash/rigdash/static/index.html" ] \
   || [ ! -f "$CTX/tools/rigdash/kartenplan_build/couplings_worker.py" ] || [ ! -x "$CTX/tools/rigdash/entrypoint_rigdash.sh" ]; then
  err "Paket unvollstaendig; entferne $CTX/tools/rigdash"; rm -rf "$CTX/tools/rigdash"; exit 4
fi
if [ "$KEEP_EP" = 0 ]; then
  cp -p "$CTX/tools/entrypoint.sh" "$CTX/tools/entrypoint.sh.vor_rigdash" || { err "Sicherung des Kontext-Entrypoints gescheitert"; exit 4; }
  git -C "$REPO" show "$REV:$EP_REPO_PATH" > "$CTX/tools/entrypoint.sh.new" && chmod --reference="$CTX/tools/entrypoint.sh.vor_rigdash" "$CTX/tools/entrypoint.sh.new" \
    && mv -f "$CTX/tools/entrypoint.sh.new" "$CTX/tools/entrypoint.sh" || { err "Entrypoint ersetzen gescheitert"; exit 4; }
  rm -f "$CTX/tools/entrypoint.sh.vor_rigdash"
fi
block >> "$CTX/Dockerfile" || { err "Dockerfile-Block anhaengen gescheitert"; exit 4; }
echo "make_rigdash_ctx: ${REV:0:10}: tools/rigdash ($(find "$CTX/tools/rigdash" -type f | wc -l) Dateien) + Dockerfile-Block (COPY, Importprobe, LABEL $RIGDASH_LABEL, EXPOSE 30081)$([ "$KEEP_EP" = 1 ] || echo " + Entrypoint ersetzt")"
exit 0
