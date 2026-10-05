#!/usr/bin/env bash
# Grafana neben VictoriaMetrics auf CT999 (LAN :3000, anonym lesend). Idempotent.
#   bash install_grafana.sh     # Tarball nur laden, wenn /opt/grafana fehlt; Config/Tafeln immer aus dem Repo
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GF_VER="${GF_VER:-13.2.3}"
DL="${TMPDIR:-/tmp}/rigdash-grafana-dl"
if [ ! -x /opt/grafana/bin/grafana ]; then
  mkdir -p "$DL" /opt/grafana-dist
  [ -d "/opt/grafana-dist/grafana-$GF_VER" ] || {
    curl -sSfL -o "$DL/grafana.tgz" "https://dl.grafana.com/oss/release/grafana-$GF_VER.linux-amd64.tar.gz"
    tar xzf "$DL/grafana.tgz" -C /opt/grafana-dist
  }
  ln -sfn "/opt/grafana-dist/grafana-$GF_VER" /opt/grafana
fi
mkdir -p /etc/grafana-rig/provisioning/datasources /etc/grafana-rig/provisioning/dashboards /etc/grafana-rig/dashboards \
         /var/lib/grafana/plugins /var/log/grafana
python3 "$HERE/make_dashboard.py" > "$HERE/dashboards/rig-verlauf.json"
install -m 0644 "$HERE/custom.ini" /etc/grafana-rig/custom.ini
install -m 0644 "$HERE/provisioning/datasources/vm.yaml" /etc/grafana-rig/provisioning/datasources/vm.yaml
install -m 0644 "$HERE/provisioning/dashboards/rig.yaml" /etc/grafana-rig/provisioning/dashboards/rig.yaml
install -m 0644 "$HERE/dashboards/"*.json /etc/grafana-rig/dashboards/
install -m 0644 "$HERE/grafana-rig.service" /etc/systemd/system/grafana-rig.service
systemctl daemon-reload
systemctl enable grafana-rig >/dev/null
systemctl restart grafana-rig
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:3000/api/health >/dev/null && break
  sleep 1
done
curl -s http://127.0.0.1:3000/api/health; echo
