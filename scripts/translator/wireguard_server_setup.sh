#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Rig-side WireGuard setup for the #466 translator.
#
# Every address, port and key is a PLACEHOLDER resolved from the environment,
# per the rig-env convention -- nothing real belongs in this repository.
#
#   source /root/rig-env.sh      # supplies the real values
#   scripts/translator/wireguard_server_setup.sh keys
#   scripts/translator/wireguard_server_setup.sh config
#   scripts/translator/wireguard_server_setup.sh up
#   scripts/translator/wireguard_server_setup.sh status
#
# The script prints what it would do and refuses to invent values it was not
# given: a tunnel that silently comes up on a guessed subnet is worse than one
# that does not come up.
#
set -euo pipefail

WG_IF="${TRANSLATOR_WG_IF:-wg0}"
WG_DIR="${TRANSLATOR_WG_DIR:-/etc/wireguard}"
WG_PORT="${TRANSLATOR_WG_PORT:-<51820>}"
WG_SERVER_ADDR="${TRANSLATOR_WG_SERVER_ADDR:-<10.x.y.1/24>}"
WG_PHONE_ADDR="${TRANSLATOR_WG_PHONE_ADDR:-<10.x.y.2/32>}"

die() { echo "error: $*" >&2; exit 1; }

check_placeholder() {
  case "$2" in
    "<"*) die "$1 is unset (still the placeholder '$2'). source /root/rig-env.sh first." ;;
  esac
}

cmd_keys() {
  umask 077
  mkdir -p "$WG_DIR"
  for role in server phone; do
    if [ -f "$WG_DIR/${role}.key" ]; then
      echo "${role}.key already exists, keeping it"
    else
      wg genkey > "$WG_DIR/${role}.key"
      echo "generated ${role}.key"
    fi
    wg pubkey < "$WG_DIR/${role}.key" > "$WG_DIR/${role}.pub"
  done
  echo
  echo "server public key (goes into the PHONE's peer config):"
  cat "$WG_DIR/server.pub"
  echo
  echo "phone private key (type into the phone app, then DELETE the file):"
  echo "  $WG_DIR/phone.key"
  echo "The phone should generate its own key in the app instead where possible;"
  echo "then only its PUBLIC key comes back here and no private key ever travels."
}

cmd_config() {
  check_placeholder TRANSLATOR_WG_PORT "$WG_PORT"
  check_placeholder TRANSLATOR_WG_SERVER_ADDR "$WG_SERVER_ADDR"
  check_placeholder TRANSLATOR_WG_PHONE_ADDR "$WG_PHONE_ADDR"
  [ -f "$WG_DIR/server.key" ] || die "no server key; run '$0 keys' first"
  [ -f "$WG_DIR/phone.pub" ] || die "no phone public key at $WG_DIR/phone.pub"

  umask 077
  cat > "$WG_DIR/${WG_IF}.conf" <<EOF
[Interface]
Address    = ${WG_SERVER_ADDR}
ListenPort = ${WG_PORT}
PrivateKey = $(cat "$WG_DIR/server.key")

[Peer]
# phone
PublicKey  = $(cat "$WG_DIR/phone.pub")
AllowedIPs = ${WG_PHONE_ADDR}
EOF
  echo "wrote $WG_DIR/${WG_IF}.conf"
}

cmd_up() {
  [ -f "$WG_DIR/${WG_IF}.conf" ] || die "no config; run '$0 config' first"
  sysctl -w net.ipv4.ip_forward=1 > /dev/null
  systemctl enable --now "wg-quick@${WG_IF}"
  echo "interface ${WG_IF} up"
  wg show "$WG_IF"
  cat <<EOF

Remaining, outside this script:
  1. Forward ${WG_PORT}/udp on the router to this host.
  2. Point a dynamic-DNS name at the home IP and put it in the phone's
     Endpoint field. A bare IP will break the moment the ISP rotates it,
     which it will do while abroad.
  3. Bind the translator to the tunnel address, never 0.0.0.0:
       python -m sglang.srt.translator.launch --host ${WG_SERVER_ADDR%%/*} --port 30800
EOF
}

cmd_status() {
  wg show "$WG_IF" 2>/dev/null || die "interface ${WG_IF} is not up"
}

case "${1:-}" in
  keys)   cmd_keys ;;
  config) cmd_config ;;
  up)     cmd_up ;;
  status) cmd_status ;;
  *) die "usage: $0 {keys|config|up|status}" ;;
esac
