#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Pre-flight check for the #466 translator path. Run it from the rig, and run
# the phone half from the phone's browser.
#
#   source /root/rig-env.sh
#   scripts/translator/check_tunnel.sh
#
# Every check is bounded (curl -m, capped loops) and read-only. Nothing here
# changes state; it answers "would this work from Spain" and says which layer
# is broken when the answer is no.
#
set -uo pipefail

WG_IF="${TRANSLATOR_WG_IF:-wg0}"
TRANSLATOR_HOST="${TRANSLATOR_HOST:-127.0.0.1}"
TRANSLATOR_PORT="${TRANSLATOR_PORT:-30800}"
DOMAIN="${TRANSLATOR_DOMAIN:-}"
MT_URL="${TRANSLATOR_MT_BASE_URL:-http://127.0.0.1:30000/v1}"
TTS_URL="${TRANSLATOR_TTS_BASE_URL:-http://127.0.0.1:30810/v1}"

pass=0; fail=0; warn=0
ok()   { echo "  OK    $*"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $*"; fail=$((fail+1)); }
note() { echo "  WARN  $*"; warn=$((warn+1)); }

echo "== 1. WireGuard interface"
if wg show "$WG_IF" > /dev/null 2>&1; then
  ok "interface $WG_IF is up"
  handshake="$(wg show "$WG_IF" latest-handshakes 2>/dev/null | awk '{print $2}' | head -1)"
  if [ -n "${handshake:-}" ] && [ "${handshake:-0}" -gt 0 ]; then
    age=$(( $(date +%s) - handshake ))
    if [ "$age" -lt 300 ]; then
      ok "peer handshake ${age}s ago"
    else
      note "last peer handshake was ${age}s ago -- the phone is not connected right now"
    fi
  else
    note "no peer handshake yet; connect the phone once to confirm the path"
  fi
else
  bad "interface $WG_IF is not up (scripts/translator/wireguard_server_setup.sh up)"
fi

echo "== 2. IP forwarding"
if [ "$(sysctl -n net.ipv4.ip_forward 2>/dev/null)" = "1" ]; then
  ok "net.ipv4.ip_forward=1"
else
  bad "net.ipv4.ip_forward is 0; the tunnel will connect but route nothing"
fi

echo "== 3. Translator service"
health="$(curl -sf -m 5 "http://${TRANSLATOR_HOST}:${TRANSLATOR_PORT}/api/translator/health" 2>/dev/null)"
if [ -n "$health" ]; then
  ok "health endpoint answers"
  echo "$health" | head -c 400; echo
  langs="$(curl -sf -m 5 "http://${TRANSLATOR_HOST}:${TRANSLATOR_PORT}/api/translator/languages" 2>/dev/null)"
  if echo "$langs" | grep -q '"default_participants_supported": *true'; then
    ok "the default conversation is routable on this deployment"
  else
    bad "the default language pair is NOT supported here; see /api/translator/languages"
  fi
else
  bad "no translator on ${TRANSLATOR_HOST}:${TRANSLATOR_PORT}"
fi

echo "== 4. Upstream services"
if curl -sf -m 5 "${MT_URL}/models" > /dev/null 2>&1; then
  ok "LLM (MT) reachable at ${MT_URL}"
else
  bad "LLM not reachable at ${MT_URL} -- translation will fail on every turn"
fi
if curl -sf -m 5 "${TTS_URL}/models" > /dev/null 2>&1; then
  ok "TTS reachable at ${TTS_URL}"
else
  note "TTS not reachable at ${TTS_URL} (fine if running with --tts fake)"
fi

echo "== 5. HTTPS front door (the secure-context requirement)"
if [ -z "$DOMAIN" ]; then
  note "TRANSLATOR_DOMAIN unset; skipping. Without HTTPS the phone will refuse"
  note "microphone access -- see docs/dev/DESIGN_466_live_translator.md 6.2"
else
  code="$(curl -so /dev/null -m 10 -w '%{http_code}' "https://${DOMAIN}/api/translator/health" 2>/dev/null)"
  if [ "$code" = "200" ]; then
    ok "https://${DOMAIN} serves the translator"
  else
    bad "https://${DOMAIN} returned ${code}; the proxy location is not live"
  fi
  if curl -so /dev/null -m 10 -w '%{http_code}' "https://${DOMAIN}/" 2>/dev/null | grep -q 200; then
    ok "the PWA is served over TLS (getUserMedia will work)"
  else
    bad "the PWA is not reachable over TLS"
  fi
fi

echo
echo "== phone-side, do this once from MOBILE DATA (not home WiFi)"
cat <<EOF
  1. WireGuard app on, then open https://${DOMAIN:-<TRANSLATOR_DOMAIN>}/
  2. The connection dot must turn green and the language picker must fill.
  3. Hold the talk button, say one sentence, release -- audio should come back.
  4. Turn on aeroplane mode for 20 s mid-conversation, then off. The client
     must reconnect on its own and keep the same speakers.
EOF

echo
echo "summary: ${pass} ok, ${warn} warnings, ${fail} failures"
[ "$fail" -eq 0 ]
