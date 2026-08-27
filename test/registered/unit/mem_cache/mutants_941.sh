#!/usr/bin/env bash
# #941 MUTATION PROOF. Each mutant is a plausible wrong version of one
# load-bearing decision in `FullComponent._free_full`; each must be killed by a
# NAMED test. A guard nobody can red is a guard nobody has tested.
#
# EVERY PATCH VERIFIES THAT IT APPLIED (#875d's lesson): a textual mutation that
# silently matches nothing leaves the file correct and the test green, which
# reads exactly like a covered mutant. `_patch` exits non-zero on a no-op.
#
# Hermetic: CUDA_VISIBLE_DEVICES is forced empty. Restores the file on exit,
# including on failure.
set -u

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${GATE_PY:-/spinning/htsglang-gpu/.venv/bin/python3}"
FC="$WT/python/sglang/srt/mem_cache/unified_cache_components/full_component.py"
T941="$WT/test/registered/unit/mem_cache/test_evict_frees_the_bound_allocator_941.py"

cp "$FC" /tmp/941_fc.orig
restore() { cp /tmp/941_fc.orig "$FC"; }
trap restore EXIT

FAILED=0

_patch() {  # _patch <file> <old> <new>   -- refuses a no-op
  "$PY" - "$1" "$2" "$3" <<'EOF' || { echo "BROKEN MUTANT: the patch matched nothing"; exit 1; }
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
if old not in s:
    sys.exit(1)
open(path, "w").write(s.replace(old, new, 1))
EOF
}

run() {  # run <test-file> <-k expr> -> PASS/FAIL
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT/python" PYTHONDONTWRITEBYTECODE=1 \
    "$PY" -m pytest -q "$1" -k "$2" -p no:randomly 2>&1 \
    | sed -e 's/\x1b\[[0-9;]*m//g' | grep -qE "^[0-9]+ passed" && echo PASS || echo FAIL
}

check() {  # check <name> <test-file> <-k expr>
  local name="$1" file="$2" expr="$3"
  local got; got="$(run "$file" "$expr")"
  if [ "$got" = "FAIL" ]; then
    echo "KILLED   $name  <- $(basename "$file")::$expr"
  else
    echo "SURVIVED $name  -- $(basename "$file")::$expr stayed green. Not covered."
    FAILED=1
  fi
  restore
}

# The body under mutation, quoted once.
BODY='        allocator = self.cache.token_to_kv_pool_allocator
        # When SWA is present, only free full-attention KV here;
        # SWA KV will be freed by cascade via SWAComponent.evict_component.
        if ComponentType.SWA in self.cache.tree_components:
            allocator = allocator.full_attn_allocator
        allocator.free(values)'

INIT_OLD='        super().__init__(cache, params)
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None'
INIT_NEW='        super().__init__(cache, params)
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None
        _a = cache.token_to_kv_pool_allocator
        if ComponentType.SWA in cache.tree_components:
            _a = _a.full_attn_allocator
        self._captured_free = _a.free'
BODY_CAPTURED='        self._captured_free(values)'

m1() {  # the pre-#941 tree, verbatim: capture at CONSTRUCTION, use it forever
  _patch "$FC" "$INIT_OLD" "$INIT_NEW"
  _patch "$FC" "$BODY" "$BODY_CAPTURED"
}

echo "== M1: the binding goes back to a bound method captured in __init__"
echo "   (the pre-#941 tree, verbatim: the free is addressed to whichever"
echo "    allocator happened to be bound when the tree was BUILT)"
m1
check "M1 construction-captured free" "$T941" "test_the_bound_allocator_gets_its_rows_back"

m1
check "M1b same, seen from the other allocator's books" "$T941" \
  "test_the_unbound_allocator_receives_nothing"

m1
check "M1c same, through the seam drop the boot measured" "$T941" \
  "test_the_cutover_drop_returns_the_rows_to_the_bound_allocator"

m1
check "M1d same, caught as the CLASS by the future check" "$T941" \
  "test_no_component_attribute_still_carries_the_old_allocator"

echo "== M2: the free is dropped -- the row count would look 'freed' to the tree"
echo "   (the tree's own evictable book still decrements, so only an allocator-"
echo "    side assertion can see it)"
_patch "$FC" "$BODY" '        return'
check "M2 no free at all" "$T941" "test_the_free_happens_exactly_once_and_only_for_evicted_rows"

_patch "$FC" "$BODY" '        return'
check "M2b same, seen as rows that never came back" "$T941" \
  "test_the_bound_allocator_gets_its_rows_back"

echo "== M3: the free runs twice -- the double-free this change must not build"
_patch "$FC" "$BODY" '        allocator = self.cache.token_to_kv_pool_allocator
        if ComponentType.SWA in self.cache.tree_components:
            allocator = allocator.full_attn_allocator
        allocator.free(values)
        allocator.free(values)'
check "M3 double free" "$T941" "test_the_free_happens_exactly_once_and_only_for_evicted_rows"

echo "== M4: the DEFAULT path is re-routed -- a boot that never flips must not move"
_patch "$FC" "$BODY" '        if not getattr(self.cache, "hicache_binding_generation", 0):
            return
        allocator = self.cache.token_to_kv_pool_allocator
        if ComponentType.SWA in self.cache.tree_components:
            allocator = allocator.full_attn_allocator
        allocator.free(values)'
check "M4 unrebound path regressed" "$T941" "test_an_unrebound_cache_is_byte_identical"

echo
if [ "$FAILED" -eq 0 ]; then
  echo "ALL MUTANTS KILLED"
else
  echo "SURVIVORS PRESENT -- the guard is weaker than it looks"
fi
exit "$FAILED"
