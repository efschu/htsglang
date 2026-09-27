#!/bin/bash
# Deploy rigdash from a COMMITTED revision into /opt/rigdash and (re)start the
# systemd unit.  Usage: install.sh [<git-rev>]   (default: HEAD of this checkout)
#
# The service never runs from the checkout itself: `git archive` writes the
# package into /opt/rigdash/releases/<sha>, `current` is switched atomically,
# and the unit is restarted.  Rolling back = install.sh <older-sha>.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
rev=${1:-HEAD}
sha=$(git -C "$repo" rev-parse --short=10 "$rev")
dst=/opt/rigdash/releases/$sha
if [ ! -d "$dst/rigdash" ]; then
  tmp=$(mktemp -d /opt/rigdash/.stage.XXXXXX 2>/dev/null || { mkdir -p /opt/rigdash; mktemp -d /opt/rigdash/.stage.XXXXXX; })
  git -C "$repo" archive "$sha" tools/rig_dashboard/rigdash | tar -x -C "$tmp"
  mkdir -p "$tmp/out"
  mv "$tmp/tools/rig_dashboard/rigdash" "$tmp/out/rigdash"
  rm -rf "$tmp/out/rigdash/tests"
  echo "$sha" > "$tmp/out/rigdash/VERSION"
  mkdir -p /opt/rigdash/releases
  mv "$tmp/out" "$dst"
  rm -rf "$tmp"
fi
ln -sfn "releases/$sha" /opt/rigdash/current.new
mv -T /opt/rigdash/current.new /opt/rigdash/current
install -m 0644 "$dst/rigdash/deploy/rig-dashboard.service" /etc/systemd/system/rig-dashboard.service
systemctl daemon-reload
systemctl enable rig-dashboard.service >/dev/null 2>&1
systemctl restart rig-dashboard.service
sleep 2
systemctl --no-pager --lines=0 status rig-dashboard.service | head -5
curl -fsS -m5 http://127.0.0.1:8890/api/health && echo
