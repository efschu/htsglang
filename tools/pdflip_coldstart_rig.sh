#!/bin/bash
# pdflip_coldstart_rig.sh -- the NF cold-start rig against ANY tree: one line per scenario, GRUEN / ROT / N/A / RIG-FEHLER.
#
#   tools/pdflip_coldstart_rig.sh <SHA|ref|worktree-dir|image:<docker-image>> [more targets ...]
#
#   <SHA|ref>          a detached worktree is made (or reused) under /spinning/htsglang/.claude/worktrees/coldstart-rig-<sha10>
#   <worktree-dir>     an existing checkout of htsglang (the rig file is copied in for the run and removed again)
#   image:<image>      the NF revision of a built image: label htsglang.revision.nf (docker image inspect, nothing runs)
#
# The rig is test/registered/unit/pdflip/test_pdflip_coldstart_rig_1015.py of THIS script's tree; it is copied into the
# target and run there through /spinning/gpu-arb/pytest_gedeckelt.sh (CPU only, no GPU, no image, no boot).
# Exit: 0 every scenario green (N/A allowed), 1 at least one ROT, 2 a RIG-FEHLER / the run itself failed.
# Env: RIG_KEEP_WT=1 keeps nothing special (worktrees are always kept for reuse); RIG_OUT=<dir> for the junit files.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
RIG_REL=test/registered/unit/pdflip/test_pdflip_coldstart_rig_1015.py
RIG_SRC=$HERE/$RIG_REL
REPO=${RIG_REPO:-/spinning/htsglang}
WT_BASE=${RIG_WT_BASE:-$REPO/.claude/worktrees}
OUT_BASE=${RIG_OUT:-/tmp/pdflip_coldstart_rig}
PYTEST=${RIG_PYTEST:-/spinning/gpu-arb/pytest_gedeckelt.sh}

[ -f "$RIG_SRC" ] || { echo "pdflip_coldstart_rig: rig file missing: $RIG_SRC" >&2; exit 2; }
[ "$#" -ge 1 ] || { sed -n '2,13p' "$0" >&2; exit 2; }

resolve_target() {   # prints "<dir> <label>"
  local t=$1 sha
  if [[ "$t" == image:* ]]; then
    local img=${t#image:}
    sha=$(docker image inspect -f '{{index .Config.Labels "htsglang.revision.nf"}}' "$img" 2>/dev/null)
    [ -n "$sha" ] && [ "$sha" != "<no value>" ] || { echo "pdflip_coldstart_rig: $img carries no htsglang.revision.nf label" >&2; return 1; }
    t=$sha
  fi
  if [ -d "$t" ] && [ -f "$t/python/flliper/srt/managers/scheduler_pp_mixin.py" ]; then
    echo "$(cd "$t" && pwd) $(git -C "$t" rev-parse --short=10 HEAD 2>/dev/null || echo tree)"
    return 0
  fi
  sha=$(git -C "$REPO" rev-parse --verify --quiet "$t^{commit}") || { echo "pdflip_coldstart_rig: unknown target $t" >&2; return 1; }
  local wt=$WT_BASE/coldstart-rig-${sha:0:10}
  if [ ! -d "$wt" ]; then
    git -C "$REPO" worktree add --detach "$wt" "$sha" >/dev/null 2>&1 || { echo "pdflip_coldstart_rig: worktree add failed for $sha" >&2; return 1; }
  fi
  [ "$(git -C "$wt" rev-parse HEAD)" = "$sha" ] || { echo "pdflip_coldstart_rig: $wt is not at $sha" >&2; return 1; }
  echo "$wt ${sha:0:10}"
}

worst=0
for target in "$@"; do
  read -r dir label < <(resolve_target "$target") || { worst=2; continue; }
  [ -n "${dir:-}" ] || { worst=2; continue; }
  out=$OUT_BASE/$label
  mkdir -p "$out"
  copied=0
  if [ "$(realpath "$dir/$RIG_REL" 2>/dev/null)" != "$(realpath "$RIG_SRC")" ]; then
    if [ -e "$dir/$RIG_REL" ] && ! cmp -s "$dir/$RIG_REL" "$RIG_SRC"; then
      cp "$dir/$RIG_REL" "$out/rig.orig.py"
    fi
    [ -e "$dir/$RIG_REL" ] || copied=1
    cp "$RIG_SRC" "$dir/$RIG_REL"
  fi
  t0=$(date +%s)
  ( cd "$dir" && "$PYTEST" "$RIG_REL" -p no:warnings -q --color=no --junitxml="$out/junit.xml" ) >"$out/pytest.log" 2>&1
  rc=$?
  t1=$(date +%s)
  if [ "$copied" = 1 ]; then rm -f "$dir/$RIG_REL"; fi
  [ -f "$out/rig.orig.py" ] && { cp "$out/rig.orig.py" "$dir/$RIG_REL"; rm -f "$out/rig.orig.py"; }
  python3 - "$out/junit.xml" "$label" "$((t1 - t0))" "$rc" <<'PY'
import sys, xml.etree.ElementTree as ET
path, label, secs, rc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    root = ET.parse(path).getroot()
except Exception as exc:
    print(f"== {label}: no junit result ({exc}); pytest rc={rc} -- see pytest.log next to it")
    sys.exit(2)
rows, worst = [], 0
for tc in root.iter("testcase"):
    name = tc.get("name", "")
    if not name.startswith(("test_S", "test_selfcheck_")):
        continue
    verdict, why = "GRUEN", ""
    for tag in ("failure", "error"):
        el = tc.find(tag)
        if el is not None:
            msg = (el.get("message") or el.text or "").strip().splitlines()[0] if (el.get("message") or el.text) else ""
            verdict = "RIG-FEHLER" if ("RIG-FEHLER" in msg or tag == "error" and "GESETZ" not in msg) else "ROT"
            why = msg
    sk = tc.find("skipped")
    if sk is not None:
        verdict, why = "N/A", (sk.get("message") or "").strip()
    if verdict == "ROT":
        worst = max(worst, 1)
    if verdict == "RIG-FEHLER":
        worst = 2
    rows.append((name, verdict, why))
print(f"== pdflip coldstart rig @ {label} ({secs} s)")
for name, verdict, why in rows:
    short = name.replace("test_", "", 1)
    print(f"  {verdict:<10} {short:<58} {why[:220]}")
sys.exit(worst)
PY
  r=$?
  [ "$r" -gt "$worst" ] && worst=$r
done
exit $worst
