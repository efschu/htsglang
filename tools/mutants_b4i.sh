#!/usr/bin/env bash
# Mutant harness for the #1273 B4i slice: the uncovered conv/5-D population
# (weight_exchange.py) and the WEG2-XCHG-RESERVE emission (launcher.py).
#
# SAME SNAPSHOT DISCIPLINE AS tools/mutants_b4h.sh AND FOR THE SAME MEASURED
# REASON: a harness that restores with `git checkout --` reverts to the last
# COMMIT and therefore DISCARDS uncommitted work. So the tree is snapshot by
# copy, every restore comes from the snapshot, the snapshot is verified
# byte-for-byte afterwards, and nothing here consults git for content.
#
# SELF-TEST: `--selfcheck` plants an UNCOMMITTED edit, runs one mutation cycle
# over it, and asserts the planted edit SURVIVED.
#
# EMPTY IS NOT A PASS: `mutate` REFUSES when its anchor is not unique (a stale
# anchor silently mutates nothing and scores GREEN-SURVIVED) and VERIFIES the
# file changed after writing.
set -uo pipefail

WT="${WT:-/spinning/wt-desk-b4c}"
SNAP=""

cleanup() { [ -n "$SNAP" ] && rm -rf -- "$SNAP"; }
trap cleanup EXIT

snapshot() {
  SNAP="$(mktemp -d /tmp/mutsnap.XXXXXX)" || exit 2
  local f
  for f in "$@"; do
    mkdir -p "$SNAP/$(dirname "$f")"
    cp -a -- "$WT/$f" "$SNAP/$f" || exit 2
  done
  echo "snapshot: $# file(s) -> $SNAP"
}

restore() {
  local f="$1"
  cp -a -- "$SNAP/$f" "$WT/$f" || { echo "RESTORE FAILED: $f"; exit 3; }
  cmp -s -- "$SNAP/$f" "$WT/$f" || { echo "RESTORE NOT BYTE-EQUAL: $f"; exit 3; }
}

verify_all() {
  local f rc=0
  while IFS= read -r f; do
    cmp -s -- "$SNAP/$f" "$WT/$f" || { echo "TREE DRIFTED: $f"; rc=1; }
  done < <(cd "$SNAP" && find . -type f | sed 's|^\./||')
  [ "$rc" -eq 0 ] && echo "tree verified: every snapshotted file byte-identical"
  return "$rc"
}

mutate() {  # mutate <file> <old> <new>
  python3 - "$WT/$1" "$2" "$3" <<'PY'
import hashlib, sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old, new = sys.argv[2], sys.argv[3]
n = s.count(old)
if n != 1:
    print(f"ANCHOR NOT UNIQUE ({n}) -- mutation skipped, NOT a pass")
    sys.exit(4)
before = hashlib.sha256(s.encode()).hexdigest()
p.write_text(s.replace(old, new))
after = hashlib.sha256(p.read_text().encode()).hexdigest()
if before == after:
    print("FILE UNCHANGED after write -- mutation did NOT apply, NOT a pass")
    sys.exit(4)
PY
}

WX=python/sglang/srt/weg2/weight_exchange.py
LA=python/sglang/srt/weg2/launcher.py
T="test/registered/unit/weg2/test_weg2_xchg_coverage_conv_1273.py
test/registered/unit/weg2/test_weg2_xchg_reserve_1273.py
test/registered/unit/weg2/test_weg2_w19_form_residue_1273.py
test/registered/unit/weg2/test_weg2_xchg_cover_1273.py
test/registered/unit/weg2/test_weg2_xchg_plan_1273.py
test/registered/unit/weg2/test_weg2_xchg_plan_provider_1273.py"

if [ "${1:-}" = "--selfcheck" ]; then
  PLANT="# MUTANT-HARNESS-SELFCHECK-PLANTED-$$"
  cd "$WT" || exit 2
  printf '\n%s\n' "$PLANT" >> "$WX"
  snapshot "$WX"
  mutate "$WX" 'def plan_bytes_from_descs(' 'def plan_bytes_from_descs_MUTANT('
  restore "$WX"
  if grep -qF "$PLANT" "$WX"; then
    echo "SELFCHECK PASS: the planted uncommitted edit SURVIVED a mutation cycle"
    rc=0
  else
    echo "SELFCHECK FAIL: the harness destroyed uncommitted work"; rc=1
  fi
  python3 - "$WT/$WX" "$PLANT" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
p.write_text(p.read_text().replace("\n" + sys.argv[2] + "\n", ""))
PY
  exit "$rc"
fi

cd "$WT" || exit 2
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" OMP_NUM_THREADS=1
run() { timeout 900 python3 -m pytest $T -q --tb=no -p no:cacheprovider \
          --color=no 2>&1 | grep -oE '[0-9]+ (passed|failed|error)' | tr '\n' ' '; }

snapshot "$WX" "$LA"
echo "BASELINE: $(run)"

m() { echo "$1"; if mutate "$2" "$3" "$4"; then run; else echo "(anchor stale -- FIX THE ANCHOR)"; fi; restore "$2"; echo; }

# ---- HALF 1: coverage of the conv / 5-D population -----------------------
m "M1 the ndim refusal comes back for REPLICATED (the weg2xsn15 defect itself)" "$WX" \
  '            and shard_axis != REPLICATED' \
  '            and shard_axis != 999'
m "M2 the refusal is dropped ENTIRELY (a non-contiguous block gets planned)" "$WX" \
  '        if (
            ndim > 2
            and shard_axis != REPLICATED
            and (shard_dim is None or int(shard_dim) not in (0, ndim - 1))
        ):' \
  '        if False:'
m "M3 the sharded arm borrows the replicated exemption (expert-major MoE planned as rows)" "$WX" \
  '            and (shard_dim is None or int(shard_dim) not in (0, ndim - 1))' \
  '            and False'
m "M4 a 5-D tensor is flattened on the LEADING axis only (claims 1/(3*2*16) of its bytes)" "$WX" \
  '            rows = 1
            for s in shape[:-1]:
                rows *= s' \
  '            rows = shape[0]'
m "M5 the descriptor declares ZERO bytes (coverage relabels the population as exempt)" "$WX" \
  '        add = 0 if getattr(d, "kind", None) == ZEROFILL else int(getattr(d, "nbytes", 0))' \
  '        add = 0'
m "M6 the conv weight is charged to the BASE tag instead of its layer tag" "$WX" \
  '    layer_id = layer_id_from_module_name(name)
    if layer_id is None:
        return GPU_MEMORY_TYPE_WEIGHTS' \
  '    layer_id = layer_id_from_module_name(name)
    if True:
        return GPU_MEMORY_TYPE_WEIGHTS'
m "M7 the exemption is widened by NAME instead of covering (what the order forbids)" "$WX" \
  '    return int(planned_bytes) == 0 and int(live_bytes) > 0' \
  '    return True'
m "M8 an aliasing attribute is judged WITHOUT its storage (144 lines come back)" "$WX" \
  '            if t.storage_key not in covered_storage:
                uncovered.append(t)' \
  '            if True:
                uncovered.append(t)'

# ---- HALF 2: the RESERVE line's emission and its one instrument ----------
m "M9 the emitter loses its production call site again (weg2xsn15: 0 lines in 4 logs)" "$LA" \
  '        xchg_form_dormant_reserve(cards, ns.weg2_xchg_census, log=log)' \
  '        pass'
m "M10 the call computes and never prints (the W84 shape one level down)" "$LA" \
  '        xchg_form_dormant_reserve(cards, ns.weg2_xchg_census, log=log)' \
  '        xchg_form_dormant_reserve(cards, ns.weg2_xchg_census)'
m "M11 the census path is truthy-checked again (#872: silent absence, no W71)" "$LA" \
  '    if _xchg_form:
        # B4i: THE LINE' \
  '    if _xchg_form and ns.weg2_xchg_census:
        # B4i: THE LINE'
m "M12 the line is emitted on EVERY form (the ring arm pays for a census it has no reason to read)" "$LA" \
  '    if _xchg_form:
        # B4i: THE LINE' \
  '    if True:
        # B4i: THE LINE'
m "M13 reserved_mib stops going through the ONE selector" "$LA" \
  '        out[c.uuid] = dc_measured_d_mib(c, WEIGHT_SOURCE_EXCHANGE)' \
  '        out[c.uuid] = DC_MEASURED_D_XCHG_MIB[1]'
m "M14 the host-side subtraction returns as a printed term" "$LA" \
  '            f"measured_mib={int(entry.dormant_proc_used_mib)} "' \
  '            f"measured_mib={int(entry.dormant_proc_used_mib)} region_mib=385 "'
m "M15 the two readings stop naming their shared instrument" "$LA" \
  '            f"instrument=WEG2-DC-at-sleep "' \
  '            f""'
m "M16 the delta is silently dropped (the attribution loses its number)" "$LA" \
  '            f"delta_mib={out[c.uuid] - int(entry.dormant_proc_used_mib)} "' \
  '            f""'

echo "BASELINE AFTER: $(run)"
verify_all
