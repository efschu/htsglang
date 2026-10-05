#!/usr/bin/env bash
# VictoriaMetrics + Exporter auf CT999 (Nutzer-Order 01.10. ~07:40Z, TSDB statt Eigenbau).
# Idempotent: Binaries nur laden, wenn sie fehlen; Units und scrape.yml immer aus dem Repo.
#   bash install_vm.sh            # installieren / aktualisieren, Dienste (neu) starten
#   bash install_vm.sh --check    # nur die scrape.yml pruefen
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VM_VER="${VM_VER:-v1.153.0}"
NE_VER="${NE_VER:-1.12.1}"
GE_VER="${GE_VER:-1.15.1}"
DL="${TMPDIR:-/tmp}/rigdash-vm-dl"
mkdir -p "$DL" /opt/victoriametrics /opt/node_exporter /opt/nvidia_gpu_exporter /etc/victoria-metrics /var/lib/victoria-metrics

if [ ! -x /opt/victoriametrics/victoria-metrics-prod ]; then
  curl -sSfL -o "$DL/vm.tgz" "https://github.com/VictoriaMetrics/VictoriaMetrics/releases/download/$VM_VER/victoria-metrics-linux-amd64-$VM_VER.tar.gz"
  tar xzf "$DL/vm.tgz" -C /opt/victoriametrics
fi
if [ ! -x /opt/node_exporter/node_exporter ]; then
  curl -sSfL -o "$DL/ne.tgz" "https://github.com/prometheus/node_exporter/releases/download/v$NE_VER/node_exporter-$NE_VER.linux-amd64.tar.gz"
  tar xzf "$DL/ne.tgz" -C "$DL"
  cp "$DL/node_exporter-$NE_VER.linux-amd64/node_exporter" /opt/node_exporter/
fi
if [ ! -x /opt/nvidia_gpu_exporter/nvidia_gpu_exporter ]; then
  curl -sSfL -o "$DL/ge.tgz" "https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v$GE_VER/nvidia_gpu_exporter-nvml_${GE_VER}_linux_x86_64.tar.gz"
  mkdir -p "$DL/ge" && tar xzf "$DL/ge.tgz" -C "$DL/ge"
  cp "$DL/ge/nvidia_gpu_exporter" /opt/nvidia_gpu_exporter/
fi

/opt/victoriametrics/victoria-metrics-prod -promscrape.config="$HERE/scrape.yml" -promscrape.config.strictParse=true -promscrape.config.dryRun
[ "${1:-}" = "--check" ] && exit 0

install -m 0644 "$HERE/scrape.yml" /etc/victoria-metrics/scrape.yml
for u in victoria-metrics node-exporter nvidia-gpu-exporter; do
  install -m 0644 "$HERE/$u.service" "/etc/systemd/system/$u.service"
done
systemctl daemon-reload
systemctl enable node-exporter nvidia-gpu-exporter victoria-metrics >/dev/null
systemctl restart node-exporter nvidia-gpu-exporter victoria-metrics
sleep 3
systemctl is-active node-exporter nvidia-gpu-exporter victoria-metrics
curl -sf http://127.0.0.1:8428/health && echo " vm ok"
