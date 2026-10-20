#!/usr/bin/env bash
# S41 (review of S39) mutation check: one plausible WRONG edit per named danger direction; each MUST turn
# test_pdflip_s41_review_s39_1018.py red. Runs LOCALLY in a 4 GiB cgroup. ONLY=<name> runs one. Restores on exit.
#   Direction RANK DISAGREEMENT (R11): TP0 drops a subtree whose interior node a worker holds on its device
#     D1  TP0 ignores the workers' dv set        D2  a worker does not publish dv        D3  TP0 does not read dv
#   Direction PEEL DOES NOT DELIVER / LOOP (W50 before the drop, or never)
#     H1  PW-R ignores TP0's queued S39 drops     H2  PW-R is held although nothing is queued (never answers W50)
#     H3  any tree's queue holds PW-R
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/flliper/srt
TEST=registered/unit/mem_cache/test_pdflip_s41_review_s39_1018.py
PY=${PY:-/spinning/shvllm/.venv/bin/python}
FILES=("$SRT/pdflip/forma_l3_subtree.py" "$SRT/mem_cache/form_a_host_shadow.py" "$SRT/pdflip/d_wall_head.py")
BAKD=$(mktemp -d); for f in "${FILES[@]}"; do cp "$f" "$BAKD/$(echo "$f" | tr / _)"; done
restore() { for f in "${FILES[@]}"; do cp "$BAKD/$(echo "$f" | tr / _)" "$f"; done; }
trap restore EXIT
mutate() { python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read(); n = s.count(old)
if n != 1: sys.exit(f"mutation anchor found {n}x: {old!r}")
open(path, "w").write(s.replace(old, new))
PY
}
declare -A F OLD NEW
S=$SRT/pdflip/forma_l3_subtree.py
R=$SRT/mem_cache/form_a_host_shadow.py
H=$SRT/pdflip/d_wall_head.py
F[D1]=$S; OLD[D1]='if (not d.children and r12.node_key(d) in nd) or r12.node_key(d) in dv),'; NEW[D1]='if (not d.children and r12.node_key(d) in nd)),'
F[D2]=$R; OLD[D2]='        else:
            dv.add(k)'; NEW[D2]='        else:
            pass'
F[D3]=$R; OLD[D3]='            for h, d in doc.get("dv", ()):'; NEW[D3]='            for h, d in ():'
F[H1]=$H; OLD[H1]='        pending = _s39.room_pending(tree)'; NEW[H1]='        pending = 0'
F[H2]=$H; OLD[H2]='    if pending > 0:'; NEW[H2]='    if pending >= 0:'
F[H3]=$S; OLD[H3]='if tree is None or ent[0] is tree)'; NEW[H3]='if True)'
ORDER=(D1 D2 D3 H1 H2 H3)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  if ! mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}"; then echo "MUTANT $m: ANCHOR MISSING"; survived=$((survived+1)); continue; fi
  out=$(cd "$ROOT/test" && systemd-run --scope -p MemoryMax=4G --quiet env PYTHONPATH="$ROOT/python" timeout 600 "$PY" -m pytest -q -p no:cacheprovider "-opythonpath=registered/unit/pdflip" "-opythonpath=registered/unit/mem_cache" "$TEST" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -aE "[0-9]+ (passed|failed)" | tail -1)
  case "$out" in *failed*) r=red; killed=$((killed+1));; *) r=green; survived=$((survived+1));; esac
  echo "MUTANT $m: $r ($out)"
done
echo "S41 MUTANTS: killed=$killed survived=$survived of ${#ORDER[@]}"
[ "$survived" -eq 0 ]
