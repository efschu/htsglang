#!/usr/bin/env bash
# Hermetischer Test zu Auftrag 1995: make_rigdash_ctx.sh, entrypoint_rigdash.sh, der Haken MODE=editor im Entrypoint.
# Kein Docker, keine GPU, kein Netz, nichts ausserhalb eines Wegwerf-Verzeichnisses (mktemp -d), kein /opt. Aufruf aus dem Worktree:
#     bash docker/pdflip-release/test_rigdash_ctx_1995.sh            (RIGDASH_TEST_PYTHON=python3, RIGDASH_TEST_TREE=<baum>/python optional)
# Er baut aus den Dateien des Arbeitsbaums ein Wegwerf-Git-Repo (Zweig mit und ohne --editor-only), einen Wegwerf-Kontext mit Dockerfile und
# Basis-Entrypoint, und fuehrt make_rigdash_ctx.sh dagegen aus. Der Teil "Server" (Entrypoint MODE=editor startet rigdash wirklich) braucht
# einen Planer-Baum mit den Editor-Modulen (RIGDASH_TEST_TREE, Standard /spinning/wt-profil-release-27b/python) und entfaellt sonst mit SKIP.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PYTHON=${RIGDASH_TEST_PYTHON:-python3}
TREE=${RIGDASH_TEST_TREE:-/spinning/wt-profil-release-27b/python}
BASE_COMMIT=0471c2c05c
T=$(mktemp -d "${TMPDIR:-/tmp}/rdctx1995.XXXXXX")
SRV_PID=""
cleanup() { [ -z "$SRV_PID" ] || kill "$SRV_PID" 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  ok   $*"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $*" >&2; }
check() { local what=$1; shift; if "$@"; then ok "$what"; else bad "$what"; fi; }
rcis() { local want=$1 what=$2; shift 2; local out rc; out=$("$@" 2>&1); rc=$?; if [ "$rc" = "$want" ]; then ok "$what (rc $rc)"; else bad "$what: rc $rc, erwartet $want: $(echo "$out" | tail -3)"; fi; LAST_OUT=$out; }
snap() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1); }
MK=$HERE/make_rigdash_ctx.sh

echo "== Wegwerf-Repo ($T)"
R=$T/repo; mkdir -p "$R/tools/rig_dashboard" "$R/docker/pdflip-release"
cp -a "$ROOT/tools/rig_dashboard/rigdash" "$ROOT/tools/rig_dashboard/kartenplan_build" "$ROOT/tools/rig_dashboard/entrypoint_rigdash.sh" "$R/tools/rig_dashboard/"
cp -a "$HERE/entrypoint.sh" "$R/docker/pdflip-release/entrypoint.sh"
find "$R" -name __pycache__ -prune -exec rm -rf {} +
git -C "$R" init -q -b main && git -C "$R" add -A && git -C "$R" -c user.name=t -c user.email=t@t commit -q -m good && GOOD=$(git -C "$R" rev-parse HEAD)
sed -i 's/--editor-only/--nope-only/g' "$R/tools/rig_dashboard/rigdash/server.py"
git -C "$R" -c user.name=t -c user.email=t@t commit -q -am noeditor && NOED=$(git -C "$R" rev-parse HEAD)
git -C "$R" checkout -q "$GOOD"
export RIGDASH_ALLOW_UNPUSHED=1

# Basis-Entrypoint des Kontexts = Stand vor dem Haken (Commit $BASE_COMMIT des Arbeitsbaums); ohne diesen Commit (flaches Repo) entfaellt dieser Teil
mkctx() { local c=$1; rm -rf "$c"; mkdir -p "$c/tools"; printf 'FROM scratch\nLABEL x=1\n' > "$c/Dockerfile"
  git -C "$ROOT" show "$BASE_COMMIT:docker/pdflip-release/entrypoint.sh" > "$c/tools/entrypoint.sh" 2>/dev/null; chmod 755 "$c/tools/entrypoint.sh"; }
HAVE_BASE=1; git -C "$ROOT" cat-file -e "$BASE_COMMIT:docker/pdflip-release/entrypoint.sh" 2>/dev/null || HAVE_BASE=0

echo "== Aufruf"
rcis 2 "ohne --ctx -> 2" "$MK" --rev "$GOOD" --repo "$R"
rcis 2 "ohne --rev -> 2" env -u RIGDASH_REV "$MK" --ctx "$T/c" --repo "$R"
rcis 2 "relatives --ctx -> 2" "$MK" --ctx rel --rev "$GOOD" --repo "$R"
rcis 2 "unbekannte Option -> 2" "$MK" --ctx "$T/c" --rev "$GOOD" --repo "$R" --nonsense
rcis 3 "unbekannte Revision -> 3" "$MK" --ctx "$T/c" --rev deadbeef00 --repo "$R"

if [ "$HAVE_BASE" = 1 ]; then
  echo "== check / dry-run schreiben nichts"
  C=$T/ctx1; mkctx "$C"; S0=$(snap "$C")
  rcis 0 "--check" "$MK" --check --ctx "$C" --rev "$GOOD" --repo "$R"
  check "--check nennt die Dateizahl" grep -q "Dateien (ohne rigdash/tests, rigdash/deploy)" <<<"$LAST_OUT"
  rcis 0 "Voreinstellung = --check" "$MK" --ctx "$C" --rev "$GOOD" --repo "$R"
  rcis 0 "--dry-run" "$MK" --dry-run --ctx "$C" --rev "$GOOD" --repo "$R"
  check "--dry-run zeigt EXPOSE 30081" grep -q "EXPOSE 30081" <<<"$LAST_OUT"
  check "--dry-run zeigt das LABEL" grep -q "LABEL io.github.efschu.flliper.rigdash=" <<<"$LAST_OUT"
  check "--dry-run listet couplings_worker.py" grep -q "tools/rigdash/kartenplan_build/couplings_worker.py" <<<"$LAST_OUT"
  check "--dry-run listet KEINE Tests" bash -c '! grep -q "tools/rigdash/rigdash/tests/" <<<"$0"' "$LAST_OUT"
  check "--dry-run listet KEIN deploy" bash -c '! grep -q "tools/rigdash/rigdash/deploy/" <<<"$0"' "$LAST_OUT"
  check "Kontext nach check/dry-run byte-gleich" test "$(snap "$C")" = "$S0"

  echo "== Pruefungen verweigern (rc 3, nichts geschrieben)"
  rcis 3 "Revision ohne --editor-only" "$MK" --apply --ctx "$C" --rev "$NOED" --repo "$R"
  check "  Grund genannt" grep -q -- "--editor-only nicht" <<<"$LAST_OUT"
  check "  Kontext unveraendert" test "$(snap "$C")" = "$S0"
  rcis 3 "nicht gepusht (Haken aus)" env -u RIGDASH_ALLOW_UNPUSHED "$MK" --apply --ctx "$C" --rev "$GOOD" --repo "$R"
  check "  Grund genannt" grep -q "nicht gepusht" <<<"$LAST_OUT"
  rcis 3 "Kontext ohne Dockerfile" "$MK" --apply --ctx "$T/leer" --rev "$GOOD" --repo "$R"
  C2=$T/ctx2; mkctx "$C2"; echo '# fremd' >> "$C2/tools/entrypoint.sh"; S2=$(snap "$C2")
  rcis 3 "fremder Kontext-Entrypoint" "$MK" --apply --ctx "$C2" --rev "$GOOD" --repo "$R"
  check "  Grund genannt" grep -q "weder die Basis" <<<"$LAST_OUT"
  check "  Kontext unveraendert" test "$(snap "$C2")" = "$S2"
  rcis 0 "fremder Entrypoint mit --keep-entrypoint" "$MK" --apply --keep-entrypoint --ctx "$C2" --rev "$GOOD" --repo "$R"
  check "  Entrypoint des Kontexts bleibt" grep -q '^# fremd$' "$C2/tools/entrypoint.sh"
  check "  Paket liegt da" test -f "$C2/tools/rigdash/rigdash/__main__.py"

  echo "== --apply"
  rcis 0 "--apply" "$MK" --apply --ctx "$C" --rev "$GOOD" --repo "$R"
  P=$C/tools/rigdash
  check "rigdash/__main__.py" test -f "$P/rigdash/__main__.py"
  check "static/index.html" test -s "$P/rigdash/static/index.html"
  check "kartenplan_build/couplings_worker.py" test -f "$P/kartenplan_build/couplings_worker.py"
  check "entrypoint_rigdash.sh ausfuehrbar" test -x "$P/entrypoint_rigdash.sh"
  check "keine rigdash/tests" test ! -e "$P/rigdash/tests"
  check "kein rigdash/deploy" test ! -e "$P/rigdash/deploy"
  check "Entrypoint des Kontexts ersetzt (= Repo-Fassung)" cmp -s "$C/tools/entrypoint.sh" "$HERE/entrypoint.sh"
  check "Entrypoint ausfuehrbar" test -x "$C/tools/entrypoint.sh"
  check "Dockerfile-Block: COPY" grep -q '^COPY tools/rigdash/ /opt/htsglang/rigdash/$' "$C/Dockerfile"
  check "Dockerfile-Block: LABEL mit voller SHA" grep -q "^LABEL io.github.efschu.flliper.rigdash=\"$GOOD\"$" "$C/Dockerfile"
  check "Dockerfile-Block: EXPOSE 30081" grep -q '^EXPOSE 30081$' "$C/Dockerfile"
  check "Dockerfile-Vorspann unveraendert" bash -c 'head -2 "$0" | cmp -s - <(printf "FROM scratch\nLABEL x=1\n")' "$C/Dockerfile"
  check "Block genau einmal" test "$(grep -c 'rigdash Profil-Editor (Nutzer-Entscheid' "$C/Dockerfile")" = 1
  check "keine Reste (.new/.vor_rigdash)" test -z "$(ls "$C/tools" | grep -E '\.(new|vor_rigdash)$')"
  S1=$(snap "$C")
  rcis 3 "zweiter --apply verweigert" "$MK" --apply --ctx "$C" --rev "$GOOD" --repo "$R"
  check "  Kontext unveraendert" test "$(snap "$C")" = "$S1"
  echo "== Importprobe wie im Dockerfile-Block (mit $PYTHON, kein Image)"
  check "import rigdash.server ... (stdlib-Paket)" bash -c 'cd "$0" && CUDA_VISIBLE_DEVICES= PYTHONPATH="$0" "$1" -B -c "import rigdash.server, rigdash.profil, rigdash.profil_recompute, rigdash.hwprofil, rigdash.modellprofil; print(rigdash.server._version())"' "$P" "$PYTHON"
  check "bash -n entrypoint_rigdash.sh" bash -n "$P/entrypoint_rigdash.sh"

  echo "== Entrypoint: MODE=editor (Kommandozeile, --check)"
  H=$T/home; mkdir -p "$H/src-27b/python/flliper/srt/pdflip" "$H/src-nf/python/flliper/srt/pdflip" "$H/profiles" "$T/state/profiles"
  : > "$H/src-27b/python/flliper/srt/pdflip/profile_json.py"; : > "$H/src-nf/python/flliper/srt/pdflip/profile_json.py"
  EP=$C/tools/entrypoint.sh
  ed() { env -i PATH="$PATH" HOME="$T" MODE=editor RIGDASH_DIR="$P" RIGDASH_HOME="$H" FLLIPER_PDFLIP_VENV="$T/venv" "$@" "$EP" --check; }
  rcis 0 "MODE=editor --check" ed FLLIPER_PROFILES_DIR="$T/state/profiles"
  O=$LAST_OUT
  check "  --edition release --editor-only" grep -q -- "--edition release --editor-only" <<<"$O"
  check "  Port 30081, Bind 0.0.0.0" grep -q -- "--port 30081" <<<"$O"
  check "  Planer-Baum = src-27b/python (Standardlinie)" grep -q -- "--profil-tree $H/src-27b/python" <<<"$O"
  check "  Kopplungs-Python = Image-venv" grep -q -- "--couplings-python $T/venv/bin/python" <<<"$O"
  check "  --profile-dir = FLLIPER_PROFILES_DIR" grep -q -- "--profile-dir $T/state/profiles" <<<"$O"
  check "  --profiles-release-dir = <home>/profiles" grep -q -- "--profiles-release-dir $H/profiles" <<<"$O"
  check "  ohne --trust-proxy" bash -c '! grep -q -- "--trust-proxy" <<<"$0"' "$O"
  check "  nichts am Rig: --gpuq auf Totadresse" grep -qF -- "--gpuq http://127.0.0.1:1 " <<<"$O"
  check "  nichts am Rig: --docker-ssh leer" grep -qF -- "--docker-ssh '' " <<<"$O"
  check "  nichts am Rig: --vm-url leer" grep -qF -- "--vm-url '' " <<<"$O"
  rcis 0 "FLLIPER_RIGDASH_LINE=nf -> src-nf" ed FLLIPER_RIGDASH_LINE=nf
  check "  Baum src-nf" grep -q -- "--profil-tree $H/src-nf/python" <<<"$LAST_OUT"
  rcis 0 "FLLIPER_RIGDASH_TRUST_PROXY=1 -> --trust-proxy" ed FLLIPER_RIGDASH_TRUST_PROXY=1
  check "  --trust-proxy" grep -q -- "--trust-proxy" <<<"$LAST_OUT"
  rcis 0 "FLLIPER_RIGDASH_PORT=31999 + MODEL_ROOTS" ed FLLIPER_RIGDASH_PORT=31999 FLLIPER_RIGDASH_MODEL_ROOTS=/m/a:/m/b
  check "  Port 31999" grep -q -- "--port 31999" <<<"$LAST_OUT"
  check "  zwei --model-root" test "$(grep -o -- '--model-root' <<<"$LAST_OUT" | wc -l)" = 2
  for badport in 30030 30080 8890 abc; do rcis 3 "Port $badport verweigert" ed FLLIPER_RIGDASH_PORT=$badport; done
  rcis 3 "FLLIPER_RIGDASH_LINE=xx verweigert" ed FLLIPER_RIGDASH_LINE=xx
  rcis 3 "FLLIPER_RIGDASH_TRUST_PROXY=2 verweigert" ed FLLIPER_RIGDASH_TRUST_PROXY=2
  rcis 3 "HTSGLANG_RIGDASH=0 -> kein Editor, rc 3" ed HTSGLANG_RIGDASH=0
  rcis 3 "FLLIPER_RIGDASH=7 verweigert" ed FLLIPER_RIGDASH=7
  rm "$H/src-27b/python/flliper/srt/pdflip/profile_json.py"
  rcis 3 "Baum ohne profile_json.py (Code-Stand ohne Editor-Module) -> rc 3" ed
  check "  Grund genannt" grep -q "profile_json.py nicht" <<<"$LAST_OUT"
  : > "$H/src-27b/python/flliper/srt/pdflip/profile_json.py"
  rcis 3 "Image ohne Paket -> rc 3 mit Klartext" env -i PATH="$PATH" HOME="$T" MODE=editor RIGDASH_DIR="$T/gibtsnicht" "$EP" --check
  check "  Grund genannt" grep -q "traegt den Profil-Editor nicht" <<<"$LAST_OUT"
  echo "== Entrypoint: Default-Pfad unveraendert (kein MODE=editor): der Haken fasst nichts an"
  check "Haken in pdflip-Teil nur hinter Dateitest" grep -q '^if \[ -f /opt/htsglang/rigdash/entrypoint_rigdash.sh \]; then \. /opt/htsglang/rigdash/entrypoint_rigdash.sh && rigdash_start; fi$' "$HERE/entrypoint.sh"
  check "Teardown ruft rigdash_stop nur wenn definiert" grep -q 'declare -F rigdash_stop' "$HERE/entrypoint.sh"
  check "Diff zur Basis: nur Zusaetze (hoechstens die EP_PRODUCT_NAMES-Zeile umgebrochen)" bash -c 'git -C "$0" diff --numstat '"$BASE_COMMIT"' -- docker/pdflip-release/entrypoint.sh | awk "{d=\$2} END{exit !(d<=1)}"' "$ROOT"
else
  echo "SKIP: Commit $BASE_COMMIT nicht im Repo -- Kontext-/Entrypoint-Teile entfallen"
fi

echo "== Server: Entrypoint MODE=editor startet rigdash wirklich (Wegwerf-Port, Wegwerf-State)"
if [ -f "$TREE/flliper/srt/pdflip/profile_json.py" ] && [ "$HAVE_BASE" = 1 ]; then
  PORT=$("$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
  H2=$T/home2; mkdir -p "$H2/profiles" "$T/venv/bin" "$T/state2"; ln -s "$TREE/.." "$H2/src-27b"; ln -sf "$(command -v "$PYTHON")" "$T/venv/bin/python"
  env -i PATH="$PATH" HOME="$T" MODE=editor RIGDASH_DIR="$P" RIGDASH_HOME="$H2" FLLIPER_PDFLIP_VENV="$T/venv" FLLIPER_PROFILES_DIR="$T/state2" \
      FLLIPER_RIGDASH_PORT="$PORT" FLLIPER_RIGDASH_BIND=127.0.0.1 FLLIPER_RIGDASH_TRUST_PROXY=0 "$EP" > "$T/srv.log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 60); do curl -fsS -m 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 0.5; done
  g() { curl -sS -m 20 -o "$T/body" -w '%{http_code}' "$@"; }
  check "/healthz 200" test "$(g "http://127.0.0.1:$PORT/healthz")" = 200
  check "  editor_only true" grep -q '"editor_only": true' "$T/body"
  check "/ 200 mit data-editor-only" test "$(g "http://127.0.0.1:$PORT/")" = 200
  check "  Seite traegt data-editor-only" grep -q 'data-edition="release" data-editor-only="1"' "$T/body"
  check "/api/live 404" test "$(g "http://127.0.0.1:$PORT/api/live")" = 404
  check "/api/profil/list 200" test "$(g "http://127.0.0.1:$PORT/api/profil/list")" = 200
  check "  Planer-Baum im Editor = src-27b" grep -q "src-27b" "$T/body"
  check "Proxy-Header ohne Schalter -> 403" test "$(g -H 'X-Forwarded-For: 1.2.3.4' "http://127.0.0.1:$PORT/api/profil/list")" = 403
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""
  check "Prozess beendet (kein Rest auf dem Port)" bash -c '! curl -fsS -m 2 "http://127.0.0.1:$0/healthz" >/dev/null 2>&1' "$PORT"
  # zweiter Lauf mit --trust-proxy
  env -i PATH="$PATH" HOME="$T" MODE=editor RIGDASH_DIR="$P" RIGDASH_HOME="$H2" FLLIPER_PDFLIP_VENV="$T/venv" FLLIPER_PROFILES_DIR="$T/state2" \
      FLLIPER_RIGDASH_PORT="$PORT" FLLIPER_RIGDASH_BIND=127.0.0.1 FLLIPER_RIGDASH_TRUST_PROXY=1 "$EP" > "$T/srv2.log" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 60); do curl -fsS -m 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 0.5; done
  check "mit FLLIPER_RIGDASH_TRUST_PROXY=1: Proxy-Header -> 200" test "$(g -H 'X-Forwarded-For: 1.2.3.4' "http://127.0.0.1:$PORT/api/profil/list")" = 200
  kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""
else
  echo "SKIP: kein Planer-Baum ($TREE) oder kein Basis-Commit"
fi

echo
echo "ERGEBNIS: $PASS ok, $FAIL FAIL"
[ "$FAIL" = 0 ]
