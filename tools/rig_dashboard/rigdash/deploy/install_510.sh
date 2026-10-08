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
#     -> werden vom Dienst per Dateipfad geladen (kartenplan_gate.py), NICHT per `import flliper` (zieht torch).
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
  git -C "$repo" cat-file -e "$GATE_REV:python/flliper/srt/pdflip/$f" || { echo "REFUSED: $f fehlt in $GATE_REV" >&2; exit 3; }
done
# Auftrag 930 (Profil-Editor S1): profile_json.py + refusals.py + profile_catalog.py (stdlib-rein) in dieselbe Stufe. Sie liegen auf dem Planer-Zweig
# desk/profil-editor-s1-py-1003 (nicht auf der Dashboard-Linie); PROFIL_REV nennt die Revision, aus der sie kommen (Env KARTENPLAN_PROFIL_REV).
# Das Dashboard laedt sie per Dateipfad (profil.py); der Katalog liegt im rigdash-Release (rigdash/profil_data/catalog.json), die Nutzerprofile
# in $FLLIPER_PROFILES_DIR (Standard /var/lib/flliper/profiles, derselbe Ort wie im Entrypoint).
PROFIL_REV=${KARTENPLAN_PROFIL_REV:-136a929fa5}
git -C "$repo" cat-file -e "$PROFIL_REV^{commit}" || { echo "REFUSED: Revision $PROFIL_REV (Planer-Zweig des Profil-Editors) nicht im Repo" >&2; exit 3; }
for f in profile_json.py refusals.py profile_catalog.py; do
  git -C "$repo" cat-file -e "$PROFIL_REV:python/flliper/srt/pdflip/$f" || { echo "REFUSED: $f fehlt in $PROFIL_REV" >&2; exit 3; }
done
# Auftrag 960 (S3): model_profile.py (Schaetzer, stdlib) liegt auf dem Zweig desk/profil-s3-modell-1003; MODELLPROFIL_REV nennt die Revision.
MODELLPROFIL_REV=${KARTENPLAN_MODELLPROFIL_REV:-3d729b672c}
git -C "$repo" cat-file -e "$MODELLPROFIL_REV:python/flliper/srt/pdflip/model_profile.py" || { echo "REFUSED: model_profile.py fehlt in $MODELLPROFIL_REV" >&2; exit 3; }
dst=/opt/rigdash/kartenplan/releases/$GATE_REV
echo "Planer-Stufe: $dst  (card_identity.py, topology.py aus $GATE_REV)"
if [ "$check_only" = 1 ]; then
  "$here/install.sh" --check "$sha"
  exit 0
fi
if [ ! -d "$dst/python" ]; then
  mkdir -p "$dst/python/flliper/srt/pdflip"
  for f in card_identity.py topology.py; do
    git -C "$repo" show "$GATE_REV:python/flliper/srt/pdflip/$f" > "$dst/python/flliper/srt/pdflip/$f"
  done
  echo "$GATE_REV" > "$dst/GATE_REV"
fi
mkdir -p "$dst/python/flliper/srt/pdflip"
for f in profile_json.py refusals.py profile_catalog.py; do
  git -C "$repo" show "$PROFIL_REV:python/flliper/srt/pdflip/$f" > "$dst/python/flliper/srt/pdflip/$f"
done
echo "Profil-Editor-Module (profile_json.py refusals.py profile_catalog.py) aus $PROFIL_REV"
git -C "$repo" show "$MODELLPROFIL_REV:python/flliper/srt/pdflip/model_profile.py" > "$dst/python/flliper/srt/pdflip/model_profile.py"
echo "Modellprofil-Schaetzer (model_profile.py) aus $MODELLPROFIL_REV"
ln -sfn "releases/$GATE_REV" /opt/rigdash/kartenplan/current.new && mv -T /opt/rigdash/kartenplan/current.new /opt/rigdash/kartenplan/current
exec "$here/install.sh" "$sha"
