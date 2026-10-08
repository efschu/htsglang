#!/bin/bash
# OpenWebUI-Proxy als systemd-Dienst einrichten bzw. aktualisieren (Nutzer 01.10.: "richte den proxy als systemd dienst ein").
# Code nach /opt/owui_proxy (unabhaengig vom Worktree), Unit nach /etc/systemd/system, Neustart, Probe auf :30032.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
install -d /opt/owui_proxy
install -m 0644 "$HERE/owui_proxy.py" /opt/owui_proxy/owui_proxy.py
install -m 0644 "$HERE/owui-proxy.service" /etc/systemd/system/owui-proxy.service
# Modellkatalog des frueheren Handstarts uebernehmen, damit die Liste ab dem ersten Start nicht leer ist
install -d /var/lib/owui_proxy
if [ ! -s /var/lib/owui_proxy/models.json ] && [ -s "$HOME/.owui_proxy_models.json" ]; then
  cp "$HOME/.owui_proxy_models.json" /var/lib/owui_proxy/models.json
fi
systemctl daemon-reload
systemctl enable owui-proxy >/dev/null
systemctl restart owui-proxy
sleep 2
systemctl is-active owui-proxy
curl -sf --max-time 5 http://127.0.0.1:30032/v1/models | head -c 300; echo
