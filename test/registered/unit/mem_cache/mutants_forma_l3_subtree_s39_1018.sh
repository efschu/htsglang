#!/usr/bin/env bash
# S39 FORMA-L3-SUBTREE mutation check: one plausible WRONG edit per danger direction; each MUST turn
# test_weg2_s39_forma_l3_subtree_1018.py red. A survivor = that direction is asserted nowhere. Runs LOCALLY in a 4 GiB
# cgroup (systemd-run --scope -p MemoryMax=4G). ONLY=<name> runs one. Restores every file on exit.
#   Direction DATA LOSS ("Seite ohne jede Kopie geraeumt"):
#     M1  the broadcast does not re-check the leaf's store mark (a leaf without l3_present is dropped)
#     M2  a host-only child outside the store (no l3_present) is deleted
#     M3  a host-locked child (a load-back reads it) is deleted
#     M4  the leaf's device anchor is dropped although the store does not hold it
#     M5  a child's host anchor leaves L2 without its L3 copy
#     K1  the S22 worker-nodelete set is ignored (TP0 drops a subtree a worker holds on its device)
#   Direction RANK DISAGREEMENT (R11, raenge-nie-uneins):
#     R1  every rank drops inside its own peel (rank-local decision, no verdict)
#     R2  a worker ignores the L3DROP verdict
#     R3  TP0 drops at its broadcast but sends no verdict
#     R4  a worker applies the drop although its child is host-locked (no pending)
#   Bound / switch:
#     Q1  TP0 queues every refused leaf, not just the peel's need        E1  the feature defaults OFF
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/sglang/srt
TEST=registered/unit/mem_cache/test_weg2_s39_forma_l3_subtree_1018.py
PY=${PY:-/spinning/shvllm/.venv/bin/python}
FILES=("$SRT/weg2/forma_l3_subtree.py" "$SRT/mem_cache/form_a_host_shadow.py" "$SRT/mem_cache/unified_radix_cache.py")
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
S=$SRT/weg2/forma_l3_subtree.py
F[M1]=$S; OLD[M1]='    if not getattr(node, "l3_present", False):
        return "not_in_store"'; NEW[M1]='    if False:
        return "not_in_store"'
F[M2]=$S; OLD[M2]='        if tp0 and not getattr(d, "l3_present", False):'; NEW[M2]='        if False:'
F[M3]=$S; OLD[M3]='        if _host_locked(d):
            return None, ("child_host_locked", did)'; NEW[M3]='        if False:
            return None, ("child_host_locked", did)'
F[M4]=$S; OLD[M4]='    if held:
        return True, "held"'; NEW[M4]='    if True:
        return True, "held"'
F[M5]=$S; OLD[M5]='            swhy = _secure_child_anchors(t, order, budget)'; NEW[M5]='            swhy = None'
F[K1]=$S; OLD[K1]='                stuck = next((d for d in order if not d.children and r12.node_key(d) in nd), None) if nd else None'; NEW[K1]='                stuck = None'
F[R1]=$SRT/mem_cache/unified_radix_cache.py; OLD[R1]='            _s39.note_refused(self, node, "children" if node.children else "device_anchor")
            return False'; NEW[R1]='            _o, _w = _s39.subtree(self, node, tp0=False)
            if _o is not None and not anchor_dev:
                _s39._execute(self, node, _o)
                return True
            return False'
F[R2]=$SRT/mem_cache/form_a_host_shadow.py; OLD[R2]='            out = _s39.apply_drop(tree, node, (h, int(depth)))'; NEW[R2]='            out = "missing"'
F[R3]=$S; OLD[R3]='            r12.record_l3_drop(t, k)'; NEW[R3]='            pass'
F[R4]=$S; OLD[R4]='    order, why = subtree(tree, node, tp0=False)
    busy = None'; NEW[R4]='    order, why = subtree(tree, node, tp0=False)
    if order is None:
        order = [d for d in reversed(list(node.children.values()))]
        for d in order:
            for cd in d.component_data:
                cd.host_lock_ref = 0
        why = None
    busy = None'
F[Q1]=$S; OLD[Q1]='        if peel[1] >= peel[0]:
            return False'; NEW[Q1]='        if False:
            return False'
F[E1]=$S; OLD[E1]='    return str(e.get(ENV, "1")).strip().lower()'; NEW[E1]='    return str(e.get(ENV, "0")).strip().lower()'
ORDER=(M1 M2 M3 M4 M5 K1 R1 R2 R3 R4 Q1 E1)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  if ! mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}"; then echo "MUTANT $m: ANCHOR MISSING"; survived=$((survived+1)); continue; fi
  out=$(cd "$ROOT/test" && systemd-run --scope -p MemoryMax=4G --quiet env PYTHONPATH="$ROOT/python" timeout 600 "$PY" -m pytest -q -p no:cacheprovider "-opythonpath=registered/unit/weg2" "-opythonpath=registered/unit/mem_cache" "$TEST" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -aE "[0-9]+ (passed|failed)" | tail -1)
  case "$out" in *failed*) r=red; killed=$((killed+1));; *) r=green; survived=$((survived+1));; esac
  echo "MUTANT $m: $r ($out)"
done
echo "S39 MUTANTS: killed=$killed survived=$survived of ${#ORDER[@]}"
[ "$survived" -eq 0 ]
