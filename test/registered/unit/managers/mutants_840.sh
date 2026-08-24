#!/usr/bin/env bash
# #840 mutants: one dying test per load-bearing edge of the fix.
#
# Each mutant breaks exactly one decision and must turn the suite RED. A mutant
# that survives means the edge it broke is not actually pinned by any test.
set -u

ROOT=/spinning/wt-840-teardown
PY=/spinning/htsglang-gpu/.venv/bin/python
SUITE=test/registered/unit/managers/test_sigterm_drain_840.py
TM=$ROOT/python/sglang/srt/managers/tokenizer_manager.py
SG=$ROOT/python/sglang/srt/managers/shutdown_gate.py

run_suite() {
  ( cd "$ROOT" && timeout 300 env CUDA_VISIBLE_DEVICES="" \
      PYTHONPATH=$ROOT/python "$PY" -m pytest "$SUITE" -q 2>&1 | tail -1 )
}

mutate() {  # name file sed-expr
  local name=$1 file=$2 expr=$3
  cp "$file" "$file.orig"
  "$PY" - "$file" "$expr" <<'EOF'
import sys, re
path, expr = sys.argv[1], sys.argv[2]
old, new = expr.split("|||")
text = open(path).read()
assert old in text, f"mutation anchor not found: {old!r}"
open(path, "w").write(text.replace(old, new, 1))
EOF
  local out
  out=$(run_suite)
  cp "$file.orig" "$file"; rm -f "$file.orig"
  if echo "$out" | grep -q "failed"; then
    echo "KILLED  $name  -- $out"
  else
    echo "SURVIVED $name  -- $out   <-- EDGE NOT PINNED"
  fi
}

echo "=== #840 mutants ==="

# M1: the refill gate never fires -> a shutting-down server keeps admitting.
mutate "M1 gate removed" "$TM" \
  'if getattr(self, "gracefully_exit", False):|||if False:'

# M2: the gate fires on a HEALTHY server -- the dangerous direction.
mutate "M2 gate inverted" "$TM" \
  'if getattr(self, "gracefully_exit", False):|||if not getattr(self, "gracefully_exit", False):'

# M3: the drain budget is ignored -> the unbounded loop is back.
mutate "M3 deadline disabled" "$TM" \
  'time.monotonic() + drain_budget_s if drain_budget_s > 0.0 else None|||None'

# M4: the deadline is consulted BEFORE the empty check, so a drain that
# finishes on the expiring tick is reported as abandoned anyway.
mutate "M4 zero-check bypassed" "$TM" \
  'if remain_num_req <= 0:|||if remain_num_req < 0:'

# M5: an explicit 0 is swallowed by a falsy default -> the documented
# bisecting mode silently becomes the bounded one.
mutate "M5 falsy-default budget" "$SG" \
  'if value is None:|||if not value:'

# M6: a negative budget becomes a real (negative) deadline instead of
# "unbounded", so time.monotonic() >= deadline is true on the first tick.
mutate "M6 negative budget kept" "$SG" \
  'return value if value > 0.0 else 0.0|||return value'

# M7: the defaulting getattr defaults the WRONG way -- a manager that has not
# established the flag yet refuses every request instead of admitting it.
mutate "M7 getattr default inverted" "$TM" \
  'if getattr(self, "gracefully_exit", False):|||if getattr(self, "gracefully_exit", True):'

# M8: the anti-inertness pin is checked against a method that does not set the
# flag, which is the mistake this pin itself made first time round.
mutate "M8 wiring pin misaimed" \
  "$ROOT/test/registered/unit/managers/test_sigterm_drain_840.py" \
  'TokenizerManager.init_running_status)|||TokenizerManager.__init__)'
