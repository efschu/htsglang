#!/bin/bash
# Deploy rigdash (and the read-only planner) from a COMMITTED revision and
# (re)start both systemd units.   Usage: install.sh [<git-rev>]  (default HEAD)
#
# Neither service runs from a checkout: `git archive` writes the code into
# /opt/rigdash/releases/<sha> (rigdash) and /opt/rigdash/planner/releases/<tree>
# (python/flliper, keyed by its tree hash so an unchanged planner is not
# re-extracted), `current` is switched atomically, the units are restarted.
# Rolling back = RIGDASH_DEPLOY_ROLLBACK=1 install.sh <older-sha>.
#
# DEPLOY-LINIE (Order 29.09., DASHBOARD-AUS-IPC): desk/dashboard-ipc-0929 ist DIE Linie des
# rigdash.  Wer deployt, setzt darauf auf oder übernimmt sie ff.  Das Deploy verweigert
#   (1) eine Revision, die kein Nachfahre der Linienspitze (origin) ist, und
#   (2) eine Revision, die kein Nachfahre des laufenden Releases ist (sonst fiele gelieferte
#       Arbeit still heraus -- so wäre Stufe 1 beim nächsten Deploy der alten Linie verloren),
# außer mit RIGDASH_DEPLOY_ROLLBACK=1 (bewusster Rückschritt, Grund im Commit/Entscheidungslog).
# Nur prüfen, nichts ändern: install.sh --check [<git-rev>]
set -euo pipefail

# --- state dir of the dashboard (F0-F, rename) -----------------------------------------------------------------------
# New home: /var/lib/flliper (the product's state volume; hardware.json already lies there). A rig that ran the dashboard
# BEFORE the rename keeps card history, history.sqlite and the live ring in /var/lib/rigdash: that directory stays the
# state dir (fallback) as long as /var/lib/flliper holds no history of its own -- nothing is moved or copied, and a
# deploy never starts the panels on an empty history. RIGDASH_STATE_DIR names one explicitly; RIGDASH_VARLIB is a test hook.
# BEGIN state-dir
pick_state_dir() {
  local varlib=${RIGDASH_VARLIB:-/var/lib}
  if [ -n "${RIGDASH_STATE_DIR:-}" ]; then echo "$RIGDASH_STATE_DIR"; return; fi
  if [ ! -e "$varlib/flliper/history.sqlite" ] && [ -e "$varlib/rigdash/history.sqlite" ]; then
    echo "$varlib/rigdash"
  else
    echo "$varlib/flliper"
  fi
}
# END state-dir

here=$(cd "$(dirname "$0")" && pwd)
repo=${RIGDASH_REPO:-$(git -C "$here" rev-parse --show-toplevel)}
check_only=0
if [ "${1:-}" = "--check" ]; then check_only=1; shift; fi
rev=${1:-HEAD}
sha=$(git -C "$repo" rev-parse --short=10 "$rev")
LINE=${RIGDASH_DEPLOY_LINE:-desk/dashboard-ipc-0929}
git -C "$repo" fetch -q origin "$LINE" 2>/dev/null || true
line_tip=$(git -C "$repo" rev-parse -q --verify "origin/$LINE^{commit}" || git -C "$repo" rev-parse -q --verify "$LINE^{commit}") || {
  echo "REFUSED: Deploy-Linie $LINE nicht auflösbar (weder origin/$LINE noch lokal)" >&2; exit 3; }
if ! git -C "$repo" merge-base --is-ancestor "$line_tip" "$sha"; then
  echo "REFUSED: $sha ist kein Nachfahre der Deploy-Linie $LINE (${line_tip:0:10}) -- auf $LINE aufsetzen oder ff übernehmen" >&2
  exit 3
fi
cur=$(basename "$(readlink /opt/rigdash/current 2>/dev/null || true)")
if [ -n "$cur" ] && [ "${RIGDASH_DEPLOY_ROLLBACK:-0}" != 1 ] \
   && git -C "$repo" cat-file -e "$cur^{commit}" 2>/dev/null \
   && ! git -C "$repo" merge-base --is-ancestor "$cur" "$sha"; then
  echo "REFUSED: $sha enthält das laufende Release $cur nicht -- dessen Commits fielen heraus (RIGDASH_DEPLOY_ROLLBACK=1 für einen bewussten Rückschritt)" >&2
  exit 3
fi
echo "deploy-linie ok: $sha enthält $LINE@${line_tip:0:10} und das laufende Release ${cur:-(keins)}"
if [ "$check_only" = 1 ]; then exit 0; fi
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

# --- planner (python/flliper) ----------------------------------------------
tree=$(git -C "$repo" rev-parse --short=10 "$sha:python/flliper")
pdst=/opt/rigdash/planner/releases/$tree
if [ ! -d "$pdst/python/flliper" ]; then
  tmp=$(mktemp -d /opt/rigdash/planner/.stage.XXXXXX)
  git -C "$repo" archive "$sha" python/flliper | tar -x -C "$tmp"
  echo "$sha" > "$tmp/python/flliper/srt/planner/DEPLOYED_FROM"
  mv "$tmp" "$pdst"
fi
ln -sfn "releases/$tree" /opt/rigdash/planner/current.new && mv -T /opt/rigdash/planner/current.new /opt/rigdash/planner/current

# Seed the planner's own state dir ONCE with the rig's existing profiles
# (copied, read from /root/.cache; nothing is ever written back there).
state=/var/lib/rig-planner
mkdir -p "$state/.cache/flliper" "$state/.cache/htsglang-planner"
for src in /root/.cache/flliper/{power_profile.json,split_probe.jsonl,card_library.json,card_library.json.by-uuid.json,barlink_matrix.json,graph_mem_anchors.json,mlp_crossover.json} \
           /root/.cache/flliper/card_probe-*.json /root/.cache/flliper/hw_profile-*.json; do
  dstf=$state/.cache/flliper/$(basename "$src")
  if [ -f "$src" ] && [ ! -e "$dstf" ]; then cp -p "$src" "$dstf"; fi
done
if [ -d /root/.cache/htsglang-planner/bench_runs ] && [ ! -e "$state/.cache/htsglang-planner/bench_runs" ]; then
  cp -rp /root/.cache/htsglang-planner/bench_runs "$state/.cache/htsglang-planner/bench_runs"
fi

# keep the three newest releases of each
ls -1dt /opt/rigdash/releases/*/ | tail -n +4 | grep -v "/$sha/" | xargs -r rm -rf || true
ls -1dt /opt/rigdash/planner/releases/*/ | tail -n +4 | grep -v "/$tree/" | xargs -r rm -rf || true

state_dir=$(pick_state_dir)
echo "state dir of rig-dashboard: $state_dir"
# rig-dashboard.service is written for /var/lib/flliper; the fallback (or RIGDASH_STATE_DIR) is substituted at install time
# (--state-dir and StateDirectory= name the same directory; StateDirectory= takes the part below /var/lib).
sed -e "s#--state-dir /var/lib/flliper#--state-dir $state_dir#" \
    -e "s#^StateDirectory=flliper\$#StateDirectory=${state_dir#/var/lib/}#" \
    "$dst/rigdash/deploy/rig-dashboard.service" > /etc/systemd/system/rig-dashboard.service
chmod 0644 /etc/systemd/system/rig-dashboard.service
install -m 0644 "$dst/rigdash/deploy/rig-planner.service" /etc/systemd/system/rig-planner.service
systemctl daemon-reload
systemctl enable rig-dashboard.service rig-planner.service >/dev/null 2>&1
systemctl restart rig-dashboard.service rig-planner.service
sleep 2
curl -fsS -m5 http://127.0.0.1:8890/api/health && echo
for i in $(seq 1 60); do curl -fsS -m3 -o /dev/null http://127.0.0.1:8780/api/version && break; sleep 1; done
curl -fsS -m5 http://127.0.0.1:8780/api/readonly && echo
