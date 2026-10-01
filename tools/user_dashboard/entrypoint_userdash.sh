# userdash im Container (Nutzer-Order 01.10. ~12:55Z: "in den docker gehört auch noch ein 'userdashboard' ohne die
# entwicklungssachen"). Wird vom Image-Entrypoint GESOURCED (nicht ausgefuehrt), direkt vor dem Launcher-Start:
#   . /opt/htsglang/userdash/entrypoint_userdash.sh && userdash_start
# Erwartet aus dem Entrypoint: FRONT_PORT, PY, LOGCOPY und die Funktionen say/refuse.
#
# Schalter (FLLIPER_<X> wird vom Entrypoint auf HTSGLANG_<X> abgebildet, EP_PRODUCT_NAMES):
#   HTSGLANG_USERDASH=1|0          an (Default) | aus -- aus = kein Prozess, kein Port
#   HTSGLANG_USERDASH_PORT=30080   Container-Port; nie 30030/30031/30032/30097/30099/8890/8428, nie FRONT_PORT
#   HTSGLANG_USERDASH_BIND=0.0.0.0 (fuer docker run -p <host>:<port>)
# Nur lesend gegen http://127.0.0.1:$FRONT_PORT (/health, /metrics, /v1/models), keine CUDA (NVML allein legt keinen
# Kontext an; CUDA_VISIBLE_DEVICES="" sichert das zusaetzlich), nice 10. Ein falscher Schalterwert wird wie jede
# Entrypoint-Env VOR dem Launch verweigert (refuse, rc 3); fehlt das Paket oder stirbt der Prozess, bootet der Server
# trotzdem (das Dashboard haengt nie im Docker-HEALTHCHECK und nie in der Front-Aufsicht).
USERDASH_DIR=${USERDASH_DIR:-/opt/htsglang/userdash}
USERDASH_PID=""

userdash_start() {
  : "${HTSGLANG_USERDASH:=1}"
  : "${HTSGLANG_USERDASH_PORT:=30080}"
  : "${HTSGLANG_USERDASH_BIND:=0.0.0.0}"
  case "$HTSGLANG_USERDASH" in
    0) say "USERDASH aus (HTSGLANG_USERDASH=0)"; return 0 ;;
    1) ;;
    *) refuse USERDASH "HTSGLANG_USERDASH='$HTSGLANG_USERDASH' (0|1)" ;;
  esac
  [[ $HTSGLANG_USERDASH_PORT =~ ^[0-9]+$ ]] || refuse USERDASH "HTSGLANG_USERDASH_PORT='$HTSGLANG_USERDASH_PORT' ist keine Portnummer"
  case "$HTSGLANG_USERDASH_PORT" in
    30030|30031|30032|30097|30099|8890|8428|"$FRONT_PORT")
      refuse USERDASH "HTSGLANG_USERDASH_PORT=$HTSGLANG_USERDASH_PORT kollidiert (Front/P/D 30030-30032, Router 30097/30099, rigdash 8890, VM 8428, Server $FRONT_PORT)" ;;
  esac
  if [ ! -f "$USERDASH_DIR/userdash/__main__.py" ]; then
    say "WARN: USERDASH an, aber $USERDASH_DIR/userdash fehlt im Image -- kein Dashboard, Server bootet weiter"
    return 0
  fi
  mkdir -p "$LOGCOPY" 2>/dev/null || true
  ( cd "$USERDASH_DIR" && exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" USERDASH_ENABLE=1 \
      USERDASH_PORT="$HTSGLANG_USERDASH_PORT" USERDASH_BIND="$HTSGLANG_USERDASH_BIND" \
      USERDASH_FRONT="http://127.0.0.1:$FRONT_PORT" nice -n 10 "$PY" -m userdash ) >> "$LOGCOPY/userdash.log" 2>&1 &
  USERDASH_PID=$!
  say "USERDASH :$HTSGLANG_USERDASH_PORT (pid $USERDASH_PID, Server :$FRONT_PORT, Log $LOGCOPY/userdash.log, Gesundheit GET /healthz; aus: HTSGLANG_USERDASH=0)"
}

userdash_stop() {
  [ -n "$USERDASH_PID" ] && kill -TERM "$USERDASH_PID" 2>/dev/null
  USERDASH_PID=""
  return 0
}
