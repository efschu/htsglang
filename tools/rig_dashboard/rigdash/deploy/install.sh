#!/bin/bash
# Deploy rigdash (and the read-only planner) from a COMMITTED revision and
# (re)start both systemd units.   Usage: install.sh [<git-rev>]  (default HEAD)
#
# Neither service runs from a checkout: `git archive` writes the code into
# /opt/rigdash/releases/<sha> (rigdash) and /opt/rigdash/planner/releases/<tree>
# (python/sglang, keyed by its tree hash so an unchanged planner is not
# re-extracted), `current` is switched atomically, the units are restarted.
# Rolling back = install.sh <older-sha>.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
rev=${1:-HEAD}
sha=$(git -C "$repo" rev-parse --short=10 "$rev")
mkdir -p /opt/rigdash/releases /opt/rigdash/planner/releases

# --- rigdash --------------------------------------------------------------
dst=/opt/rigdash/releases/$sha
if [ ! -d "$dst/rigdash" ]; then
  tmp=$(mktemp -d /opt/rigdash/.stage.XXXXXX)
  git -C "$repo" archive "$sha" tools/rig_dashboard/rigdash | tar -x -C "$tmp"
  mkdir -p "$tmp/out"
  mv "$tmp/tools/rig_dashboard/rigdash" "$tmp/out/rigdash"
  rm -rf "$tmp/out/rigdash/tests"
  echo "$sha" > "$tmp/out/rigdash/VERSION"
  mv "$tmp/out" "$dst"
  rm -rf "$tmp"
fi
ln -sfn "releases/$sha" /opt/rigdash/current.new && mv -T /opt/rigdash/current.new /opt/rigdash/current

# --- planner (python/sglang) ----------------------------------------------
tree=$(git -C "$repo" rev-parse --short=10 "$sha:python/sglang")
pdst=/opt/rigdash/planner/releases/$tree
if [ ! -d "$pdst/python/sglang" ]; then
  tmp=$(mktemp -d /opt/rigdash/planner/.stage.XXXXXX)
  git -C "$repo" archive "$sha" python/sglang | tar -x -C "$tmp"
  echo "$sha" > "$tmp/python/sglang/srt/planner/DEPLOYED_FROM"
  mv "$tmp" "$pdst"
fi
ln -sfn "releases/$tree" /opt/rigdash/planner/current.new && mv -T /opt/rigdash/planner/current.new /opt/rigdash/planner/current

# Seed the planner's own state dir ONCE with the rig's existing profiles
# (copied, read from /root/.cache; nothing is ever written back there).
state=/var/lib/rig-planner
mkdir -p "$state/.cache/sglang" "$state/.cache/htsglang-planner"
for src in /root/.cache/sglang/{power_profile.json,split_probe.jsonl,card_library.json,card_library.json.by-uuid.json,barlink_matrix.json,graph_mem_anchors.json,mlp_crossover.json} \
           /root/.cache/sglang/card_probe-*.json /root/.cache/sglang/hw_profile-*.json; do
  dstf=$state/.cache/sglang/$(basename "$src")
  if [ -f "$src" ] && [ ! -e "$dstf" ]; then cp -p "$src" "$dstf"; fi
done
if [ -d /root/.cache/htsglang-planner/bench_runs ] && [ ! -e "$state/.cache/htsglang-planner/bench_runs" ]; then
  cp -rp /root/.cache/htsglang-planner/bench_runs "$state/.cache/htsglang-planner/bench_runs"
fi

# keep the three newest releases of each
ls -1dt /opt/rigdash/releases/*/ | tail -n +4 | grep -v "/$sha/" | xargs -r rm -rf || true
ls -1dt /opt/rigdash/planner/releases/*/ | tail -n +4 | grep -v "/$tree/" | xargs -r rm -rf || true

for u in rig-dashboard rig-planner; do
  install -m 0644 "$dst/rigdash/deploy/$u.service" "/etc/systemd/system/$u.service"
done
systemctl daemon-reload
systemctl enable rig-dashboard.service rig-planner.service >/dev/null 2>&1
systemctl restart rig-dashboard.service rig-planner.service
sleep 2
curl -fsS -m5 http://127.0.0.1:8890/api/health && echo
for i in $(seq 1 60); do curl -fsS -m3 -o /dev/null http://127.0.0.1:8780/api/version && break; sleep 1; done
curl -fsS -m5 http://127.0.0.1:8780/api/readonly && echo
