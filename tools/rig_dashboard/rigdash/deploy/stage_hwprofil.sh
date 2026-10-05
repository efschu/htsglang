#!/bin/bash
# Deploy-VORSCHLAG Auftrag 950 (Hardwareprofil).  GESTAGED: der Lead führt es aus, nicht der Schreibtisch.
#
#   stage_hwprofil.sh <rev>      legt hardware_profile.py + weg2/card_identity.py der Revision <rev> (Zweig der 27B-Linie,
#                                z. B. desk/profil-s2-hw-py) unter /opt/rigdash/kartenplan/hw/<rev>/python/ ab und schaltet
#                                /opt/rigdash/kartenplan/hw/current darauf.  Danach in rig-dashboard.service:
#                                  --hw-tree /opt/rigdash/kartenplan/hw/current/python
#                                  --hw-measure-tree <voller sglang-Baum>/python   (Kindprozess; muss rigmon/card_probe.py tragen)
#                                  --hw-python <Interpreter mit torch + sgl_kernel>
#                                  --hw-prefix 'systemd-run --scope -q -p MemoryMax=6G'
#                                (der Dienst hat MemoryMax=1G; torch/CUDA des Messlaufs braucht einen eigenen cgroup-Rahmen)
set -euo pipefail
rev=${1:?usage: stage_hwprofil.sh <rev>}
repo=${RIGDASH_REPO:-$(git -C "$(cd "$(dirname "$0")" && pwd)" rev-parse --show-toplevel)}
sha=$(git -C "$repo" rev-parse --short=10 "$rev")
for f in rigmon/hardware_profile.py weg2/card_identity.py; do
  git -C "$repo" cat-file -e "$sha:python/sglang/srt/$f" || { echo "REFUSED: $f fehlt in $rev" >&2; exit 3; }
done
dst=/opt/rigdash/kartenplan/hw/$sha
mkdir -p "$dst/python/sglang/srt/rigmon" "$dst/python/sglang/srt/weg2"
git -C "$repo" show "$sha:python/sglang/srt/rigmon/hardware_profile.py" > "$dst/python/sglang/srt/rigmon/hardware_profile.py"
git -C "$repo" show "$sha:python/sglang/srt/weg2/card_identity.py" > "$dst/python/sglang/srt/weg2/card_identity.py"
echo "$sha" > "$dst/REV"
ln -sfn "$sha" /opt/rigdash/kartenplan/hw/current.new && mv -T /opt/rigdash/kartenplan/hw/current.new /opt/rigdash/kartenplan/hw/current
echo "gestagt: $dst (current -> $sha)"
