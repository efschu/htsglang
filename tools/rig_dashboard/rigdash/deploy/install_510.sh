#!/bin/bash
# Deploy-VORSCHLAG Item 510 (Kartenplaner).  Dieses Skript ist GESTAGED: der Lead führt es aus, nicht der Schreibtisch.
#
#   install_510.sh --check [<rev>]   nur prüfen (Deploy-Linie, Planer-Stufe), nichts ändern
#   install_510.sh [<rev>]           1. Planer-Dateien der NF-Linie in /opt/rigdash/kartenplan/releases/<rev>/ stagen
#                                    2. rigdash über das bestehende install.sh deployen (Linienriegel, Neustart von
#                                       rig-dashboard + rig-planner, Health-Check)
#
# Was der Kartenplaner zur Laufzeit braucht (alles nur lesend, keine GPU, kein Launcher, keine Prozesse):
#   * rigdash selbst (Python-Stdlib, im Release): Katalog, Transport, Gate-Aufruf, Records (kartenplan_data/*.json)
#   * card_identity.py + topology.py des Planers (stdlib-only, "PURE") unter /opt/rigdash/kartenplan/current/python/...
#     -> werden vom Dienst per Dateipfad geladen (kartenplan_gate.py), NICHT per `import sglang` (zieht torch).
# Der Planer-Kindprozess (launcher.budgets_from_dc, torch ~1 GB) läuft NICHT im Dienst (MemoryMax=1G), sondern nur am
# Schreibtisch beim Erneuern der Records:  python3 -m kartenplan_build.records && python3 -m kartenplan_build.bridge --trees-root <dir>
# Tab nur im Rig-Dashboard (Edition rig); die Release-Edition liefert Reiter, Skript und API nicht aus (404).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=${RIGDASH_REPO:-$(git -C "$here" rev-parse --show-toplevel)}
check_only=0
if [ "${1:-}" = "--check" ]; then check_only=1; shift; fi
rev=${1:-HEAD}
sha=$(git -C "$repo" rev-parse --short=10 "$rev")
#: Revision, deren Planer-Dateien das Gate liefern (NF-Linie mit HW-GENERIC card_identity.py, Boot nfint4h6abldauer 03.10.)
GATE_REV=${KARTENPLAN_GATE_REV:-044316dd1a}
git -C "$repo" cat-file -e "$GATE_REV^{commit}" || { echo "REFUSED: Gate-Revision $GATE_REV nicht im Repo" >&2; exit 3; }
for f in card_identity.py topology.py; do
  git -C "$repo" cat-file -e "$GATE_REV:python/sglang/srt/weg2/$f" || { echo "REFUSED: $f fehlt in $GATE_REV" >&2; exit 3; }
done
dst=/opt/rigdash/kartenplan/releases/$GATE_REV
echo "Planer-Stufe: $dst  (card_identity.py, topology.py aus $GATE_REV)"
if [ "$check_only" = 1 ]; then
  "$here/install.sh" --check "$sha"
  exit 0
fi
if [ ! -d "$dst/python" ]; then
  mkdir -p "$dst/python/sglang/srt/weg2"
  for f in card_identity.py topology.py; do
    git -C "$repo" show "$GATE_REV:python/sglang/srt/weg2/$f" > "$dst/python/sglang/srt/weg2/$f"
  done
  echo "$GATE_REV" > "$dst/GATE_REV"
fi
ln -sfn "releases/$GATE_REV" /opt/rigdash/kartenplan/current.new && mv -T /opt/rigdash/kartenplan/current.new /opt/rigdash/kartenplan/current
exec "$here/install.sh" "$sha"
