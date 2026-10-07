#!/bin/bash
# Deploy-VORSCHLAG Auftrag 1984 (B): die Planer-Module des Profil-Editors stagen.  GESTAGED: der Lead fuehrt --apply aus, nicht der Schreibtisch.
#
# WARUM: install.sh deployt python/sglang der DASHBOARD-Revision nach /opt/rigdash/planner.  Diese Revision (Dashboard-Linie) traegt die
# Rechenmodule des Editors NICHT (profile_json, refusals, profile_catalog(+_curated), model_profile, hardware_profile, profile_couplings,
# expert_residency, pp_cut): sie liegen auf den PYTHON-Release-Zweigen (desk/profil-editor-release-27b-1005 bzw. -nf-1005).  Ohne sie
# antwortet der Editor mit "kein Planer-Baum ..." (Route /api/profil/*), das Hardwareprofil mit "kein Planer-Baum mit hardware_profile.py"
# und die Balken/Topologie mit "kein Planer-Baum mit planner/profile_couplings.py".
#
# WAS: legt den VOLLEN Baum python/sglang der Revision <rev> (git archive, ~80 MiB, wie install.sh es fuer den Planer tut) unter
#   <root>/profil/releases/<sha>/python/sglang
# ab und schaltet <root>/profil/current atomar darauf.  Voll, weil der Kindprozess (Kopplungs-Worker, couplings_worker.py) mit der Python der
# sglang-Umgebung ein echtes `import sglang.srt.planner.profile_couplings` / `sglang.srt.weg2.topology` macht (PYTHONPATH = dieser Baum geht
# dem editierbaren Install der Umgebung vor); die Dateipfad-Lader des Dashboards (profil.py, hwprofil.py, modellprofil.py) finden ihre
# Dateien im selben Baum.  Der bestehende Kartenplaner-Baum (<root>/current, GATE_REV) bleibt UNBERUEHRT.
#
# Aufruf:
#   stage_profil_modules.sh [--check] <rev>              nur pruefen (Voreinstellung): alle Dateien in <rev>? was ist gestagt? KEIN Schreiben
#   stage_profil_modules.sh --dry-run <rev>              wie --check, nennt zusaetzlich jede Aktion, die --apply ausfuehren wuerde; KEIN Schreiben
#   stage_profil_modules.sh --apply <rev>                schreibt (nur unter <root>), idempotent: schon gestagter Stand wird nicht neu ausgepackt,
#                                                        ein unveraendertes `current` nicht neu gesetzt
#   stage_profil_modules.sh --unit-flags [<rev>]         druckt die Unit-Zeilen (kein Schreiben, keine Pruefung der Revision noetig)
# Optionen: --root <dir>   Wurzel der Ablage (Voreinstellung /opt/rigdash/kartenplan, Env RIGDASH_STAGE_ROOT)
# Env:      RIGDASH_REPO   das Repo, aus dem archiviert wird (Voreinstellung: das Repo dieses Skripts)
# Exit:     0 ok / 3 verweigert (Revision unbekannt oder Dateien fehlen) / 2 Aufruffehler.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=${RIGDASH_REPO:-$(git -C "$here" rev-parse --show-toplevel)}
root=${RIGDASH_STAGE_ROOT:-/opt/rigdash/kartenplan}
mode=check
rev=""

#: relativ zu python/sglang/srt/ -- alles, was Dashboard-Prozess (per Dateipfad) und Kindprozess (per import) fuer den Editor brauchen
REQUIRED=(
  weg2/profile_json.py weg2/refusals.py weg2/profile_catalog.py weg2/profile_catalog_curated.py
  weg2/model_profile.py weg2/card_identity.py weg2/topology.py
  rigmon/hardware_profile.py
  planner/profile_couplings.py planner/expert_residency.py planner/pp_cut.py
)
#: die Paket-Wurzeln, ohne die der Kindprozess nicht importieren kann
#: (sglang/srt hat im Repo bewusst KEIN __init__.py: Namensraum-Paket, gemessen an fa9e7d5c4c)
REQUIRED_ROOT=(__init__.py srt/planner/__init__.py srt/weg2/__init__.py srt/rigmon/__init__.py)

usage() { sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -40 >&2; exit 2; }

unit_flags() {
  local tree="$root/profil/current/python"
  cat <<EOF
# --- rig-dashboard.service (Drop-in: systemctl edit rig-dashboard) ------------------------------------------------------
# Der Lead setzt sie; dieses Skript aendert keine Unit.
[Service]
# EIN Baum fuer Editor, Modellprofil, Hardwareprofil UND den Kopplungs-Worker (--profil-tree, Env RIGDASH_PROFIL_TREE).
# Der Kartenplaner-Baum (KARTENPLAN_TREE bzw. $root/current) bleibt, wie er ist.
Environment=RIGDASH_PROFIL_TREE=$tree
# Python der sglang-Umgebung fuer den Kopplungs-Worker (Balken, Topologie-Urteil des Trockenlaufs).  Rig-Standard, im Container die Image-Python:
Environment=RIGDASH_COUPLINGS_PYTHON=/spinning/htsglang-gpu/.venv/bin/python
# Der Worker (import sglang, torch) liegt im cgroup der Unit: gemessen RSS 612 MiB nach dem ersten Topologie-Aufruf, die Unit lag bei
# MemoryCurrent 487 MiB / MemoryPeak 715 MiB gegen MemoryMax=1G.  Ohne mehr Raum killt der OOM-Killer den Worker oder das Dashboard:
MemoryMax=2G
# ExecStart um die Flags (jedes hat eine Env-Entsprechung; Flag gewinnt):
#   --profil-tree            $tree
#   --couplings-python       /spinning/htsglang-gpu/.venv/bin/python        (Env RIGDASH_COUPLINGS_PYTHON)
#   --profiles-release-dir   /spinning/gpu-arb/docker/profiles_release     (Env RIGDASH_PROFILES_RELEASE_DIR; die Release-.env des Editors)
#   --profile-dir            /var/lib/flliper/profiles                     (Env FLLIPER_PROFILES_DIR; Nutzerprofile JSON, derselbe Ort wie der Entrypoint)
#   --model-root             /spinning/llm_stuff/club-3090/models-cache    (wiederholbar, Env RIGDASH_MODEL_ROOTS; Modellprofil schaetzen liest NUR darunter)
#   --edition release                                                       (Env RIGDASH_EDITION; die veroeffentlichte Ausgabe)
# NUR Rig-Ausgabe (--edition rig), in release gesperrt (bucht gpuq):
#   --hw-measure-tree <voller sglang-Baum>/python   (muss rigmon/card_probe.py tragen)
#   --hw-python       <Interpreter mit torch + sgl_kernel>
#   --hw-prefix       'systemd-run --scope -q -p MemoryMax=6G'
#   --hw-tree         nur noetig, wenn die Hardware aus einem anderen Baum als --profil-tree kommen soll (Env HWPROFIL_TREE)
# Pruefen nach dem Neustart:  curl -s localhost:8890/api/profil/list | head -c 300      (planner_tree = $tree)
#                             curl -s localhost:8890/api/hwprofil | head -c 300         (ok: true)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --check) mode=check ;;
    --dry-run) mode=dry ;;
    --apply) mode=apply ;;
    --unit-flags) mode=flags ;;
    --root) shift; root=${1:?--root braucht ein Verzeichnis} ;;
    -h|--help) usage ;;
    -*) echo "unbekannte Option $1" >&2; usage ;;
    *) rev=$1 ;;
  esac
  shift
done
case "$root" in /*) ;; *) echo "REFUSED: --root muss ein absoluter Pfad sein: $root" >&2; exit 2 ;; esac

if [ "$mode" = flags ]; then unit_flags; exit 0; fi
[ -n "$rev" ] || { echo "usage: $0 [--check|--dry-run|--apply|--unit-flags] [--root DIR] <rev>" >&2; exit 2; }

sha=$(git -C "$repo" rev-parse --verify -q --short=10 "$rev^{commit}") || { echo "REFUSED: Revision $rev nicht im Repo $repo" >&2; exit 3; }

missing=()
for f in "${REQUIRED[@]}"; do
  git -C "$repo" cat-file -e "$sha:python/sglang/srt/$f" 2>/dev/null || missing+=("python/sglang/srt/$f")
done
for f in "${REQUIRED_ROOT[@]}"; do
  git -C "$repo" cat-file -e "$sha:python/sglang/$f" 2>/dev/null || missing+=("python/sglang/$f")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "REFUSED: $sha traegt die Editor-Module nicht (das ist keine Python-Release-Revision des Profil-Editors, z. B. eine Dashboard-Revision):" >&2
  printf '  fehlt: %s\n' "${missing[@]}" >&2
  echo "  -> desk/profil-editor-release-27b-1005 oder desk/profil-editor-release-nf-1005 angeben" >&2
  exit 3
fi
echo "ok: $sha traegt alle ${#REQUIRED[@]} Editor-Module und die ${#REQUIRED_ROOT[@]} Paketwurzeln"

dst="$root/profil/releases/$sha"
cur_target=""
if [ -L "$root/profil/current" ]; then cur_target=$(readlink "$root/profil/current" || true); fi
staged="nein"
[ -f "$dst/STAGED_OK" ] && staged="ja"
echo "ablage: root=$root  ziel=$dst  gestagt=$staged  current->${cur_target:-(keins)}"

want_extract=0; want_switch=0
[ "$staged" = ja ] || want_extract=1
[ "$cur_target" = "releases/$sha" ] || want_switch=1
if [ "$want_extract" = 0 ] && [ "$want_switch" = 0 ]; then
  echo "STAND: aktuell (nichts zu tun, --apply waere ein No-op)"
else
  echo "STAND: wuerde stagen=$([ $want_extract = 1 ] && echo ja || echo nein) current-umschalten=$([ $want_switch = 1 ] && echo ja || echo nein)"
fi

case "$mode" in
  check) exit 0 ;;
  dry)
    echo "DRY-RUN (es wird nichts geschrieben); --apply wuerde:"
    if [ "$want_extract" = 1 ]; then
      echo "  mkdir -p $root/profil/releases"
      echo "  tmp=\$(mktemp -d $root/profil/.stage.XXXXXX)"
      echo "  git -C $repo archive $sha python/sglang | tar -x -C \$tmp     # ~80 MiB"
      echo "  echo $sha > \$tmp/REV ; touch \$tmp/STAGED_OK ; mv \$tmp $dst"
    fi
    [ "$want_switch" = 1 ] && echo "  ln -sfn releases/$sha $root/profil/current.new && mv -T $root/profil/current.new $root/profil/current"
    echo "  danach: Unit-Zeilen mit  $0 --unit-flags  (der Lead setzt sie, dann systemctl restart rig-dashboard)"
    exit 0 ;;
  apply)
    mkdir -p "$root/profil/releases"
    if [ "$want_extract" = 1 ]; then
      rm -rf "$dst"
      tmp=$(mktemp -d "$root/profil/.stage.XXXXXX")
      trap 'rm -rf "$tmp"' EXIT
      git -C "$repo" archive "$sha" python/sglang | tar -x -C "$tmp"
      echo "$sha" > "$tmp/REV"
      touch "$tmp/STAGED_OK"
      mv "$tmp" "$dst"
      trap - EXIT
      echo "gestagt: $dst"
    fi
    if [ "$want_switch" = 1 ]; then
      ln -sfn "releases/$sha" "$root/profil/current.new" && mv -T "$root/profil/current.new" "$root/profil/current"
      echo "current -> releases/$sha"
    fi
    echo "fertig. Unit-Zeilen: $0 --unit-flags   (dieses Skript aendert keine Unit und startet nichts neu)"
    ;;
esac
