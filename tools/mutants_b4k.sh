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

WT="${WT:-/spinning/wt-desk-b4k}"
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

MS=python/sglang/srt/managers/weg2_memory_saver.py
FR=python/sglang/srt/weg2/front.py
WX=python/sglang/srt/weg2/weight_exchange.py
SH=python/sglang/srt/weg2/weight_exchange_shadow.py
T="test/registered/unit/weg2/test_weg2_xchg_draft_family_1273.py
test/registered/unit/weg2/test_weg2_xchg_cover_1273.py
test/registered/unit/weg2/test_weg2_coverage_verdict_1273.py
test/registered/unit/weg2/test_weg2_xchg_shadow_1273.py
test/registered/unit/weg2/test_weg2_xchg_manifest_1311.py"

if [ "${1:-}" = "--selfcheck" ]; then
  PLANT="# MUTANT-HARNESS-SELFCHECK-PLANTED-$$"
  cd "$WT" || exit 2
  printf '\n%s\n' "$PLANT" >> "$MS"
  snapshot "$MS"
  mutate "$MS" 'def draft_tag_in_family(' 'def draft_tag_in_family_MUTANT('
  restore "$MS"
  if grep -qF "$PLANT" "$MS"; then
    echo "SELFCHECK PASS: the planted uncommitted edit SURVIVED a mutation cycle"
    rc=0
  else
    echo "SELFCHECK FAIL: the harness destroyed uncommitted work"; rc=1
  fi
  python3 - "$WT/$MS" "$PLANT" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
p.write_text(p.read_text().replace("\n" + sys.argv[2] + "\n", ""))
PY
  exit "$rc"
fi

cd "$WT" || exit 2
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" OMP_NUM_THREADS=1
PY=/spinning/htsglang-gpu/.venv/bin/python
run() { timeout 900 $PY -m pytest $T -q --tb=no -p no:cacheprovider \
          --color=no 2>&1 | grep -oE '[0-9]+ (passed|failed|error)' | tr '\n' ' '; }

snapshot "$MS" "$FR" "$WX" "$SH"
echo "BASELINE: $(run)"

m() { echo "$1"; if mutate "$2" "$3" "$4"; then run; else echo "(anchor stale -- FIX THE ANCHOR)"; fi; restore "$2"; echo; }

m "M1 the draft tag joins on EVERY arm (the ring arm stops pausing 1.3 GiB/card)" "$MS" \
  '    return bool(exchange_armed())' \
  '    return True'
m "M2 the draft tag never joins (the weg2xsn15 defect itself returns)" "$MS" \
  '    return bool(exchange_armed())' \
  '    return False'
m "M3 the family list appends the draft tag LAST (it silently becomes the base)" "$MS" \
  '    if draft_tag_in_family():
        tags.append(GPU_MEMORY_TYPE_WEIGHTS_DRAFT)
    return tags + [GPU_MEMORY_TYPE_WEIGHTS]' \
  '    out = tags + [GPU_MEMORY_TYPE_WEIGHTS]
    if draft_tag_in_family():
        out = out + [GPU_MEMORY_TYPE_WEIGHTS_DRAFT]
    return out'
m "M4 the predicate matches but the family LIST omits it (every plan refuses tag-not-in-family)" "$MS" \
  '    if draft_tag_in_family():
        tags.append(GPU_MEMORY_TYPE_WEIGHTS_DRAFT)' \
  '    if False:
        tags.append(GPU_MEMORY_TYPE_WEIGHTS_DRAFT)'
m "M5 weights_vision is admitted too (a naming rule instead of a measurement)" "$MS" \
  '        or (tag == GPU_MEMORY_TYPE_WEIGHTS_DRAFT and draft_tag_in_family())' \
  '        or (tag.startswith(WEIGHT_CHUNK_PREFIX) and draft_tag_in_family())'
m "M6 the pause order splits on the RAW prefix again (identity refusal, weg2dk4 order lost)" "$FR" \
  '    chunks = [t for t in tags if is_weights_chunk_tag(t)]
    rest = [t for t in tags if not is_weights_chunk_tag(t)]' \
  '    chunks = [t for t in tags if t.startswith("weights_")]
    rest = [t for t in tags if not t.startswith("weights_")]'
m "M7 the coverage arm goes back to the LITERAL tag compare (draft never censused)" "$WX" \
  '    if not is_weights_family_tag(region_tag):' \
  '    if region_tag != GPU_MEMORY_TYPE_WEIGHTS:'
m "M8 the rotation predicate is WIDENED to the base tag (six ranks cannot agree on classes_hash)" "$SH" \
  '        if ms.is_weights_family_tag(g.tag)
        and g.tag != wx.GPU_MEMORY_TYPE_WEIGHTS}))' \
  '        if ms.is_weights_family_tag(g.tag)}))'
m "M9 the rotation predicate reverts to CHUNK-only (the draft runner refuses its own plan)" "$SH" \
  '        if ms.is_weights_family_tag(g.tag)
        and g.tag != wx.GPU_MEMORY_TYPE_WEIGHTS}))' \
  '        if ms.is_weights_chunk_tag(g.tag)}))'

echo "BASELINE AFTER: $(run)"
verify_all
