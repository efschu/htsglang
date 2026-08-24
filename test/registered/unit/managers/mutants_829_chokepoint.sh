#!/usr/bin/env bash
# #829 -- can-fail proof for the ring choke-point pins.
#
# Each pin in test_pp_ring_rebuild_choke_point_829.py asserts a PREMISE of the
# unreachability argument rather than a behaviour, so a pin that cannot fail
# would be worse than no pin at all: it would certify a premise it never
# checked. Each mutant below re-opens one of the two choke points and must kill
# exactly the pin that guards it.
#
# Usage:  bash test/registered/unit/managers/mutants_829_chokepoint.sh
# Exits 0 only when every mutant is killed and the tree is restored green.

set -u

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
MIXIN="$ROOT/python/sglang/srt/managers/scheduler_pp_mixin.py"
TEST="test/registered/unit/managers/test_pp_ring_rebuild_choke_point_829.py"
PY="/spinning/htsglang-gpu/.venv/bin/python"
BAK="$(mktemp)"
FAILED=0

cd "$ROOT" || exit 1
cp "$MIXIN" "$BAK"
restore() { cp "$BAK" "$MIXIN"; }
trap restore EXIT

run_pins() {
  CUDA_VISIBLE_DEVICES="" PYTHONPATH="$ROOT/python" "$PY" -m pytest "$TEST" \
    -q --no-header -p no:warnings --tb=no 2>&1 | tail -1
}

expect_killed() {
  local name="$1" expect_test="$2" out
  out="$(run_pins)"
  if echo "$out" | grep -q "failed"; then
    echo "  KILLED   $name  ($expect_test)"
  else
    echo "  SURVIVED $name  -- the pin did not notice: $out"
    FAILED=1
  fi
  restore
}

echo "== #829 choke-point mutants =="

# Mutant 1: a SECOND ring rebuild outside init_pp_loop_state. This is the
# refactor that silently re-opens the #829 root -- the new site never reaches
# pp_flip_forget_ring_scoped_slots.
"$PY" - "$MIXIN" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = "    def init_pp_loop_state(self: Scheduler):"
assert s.count(anchor) == 1, "anchor moved"
inject = (
    "    def _mutant_second_rebuild(self):\n"
    "        self.mbs = [None] * self.pp_loop_size\n\n"
)
p.write_text(s.replace(anchor, inject + anchor, 1))
PY
expect_killed "second ring rebuild site" "test_only_init_pp_loop_state_rebuilds_the_ring"

# Mutant 2: the rebuild no longer clears the ring-scoped carriers. This is the
# exact pre-#829 state that killed boot_window2_0823_1554.
"$PY" - "$MIXIN" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
call = "        pp_flip_forget_ring_scoped_slots(self)\n"
assert s.count(call) == 1, "clear call moved"
p.write_text(s.replace(call, "        pass  # mutant: clear removed\n", 1))
PY
expect_killed "rebuild stops clearing carriers" "test_the_rebuild_clears_the_ring_scoped_carriers"

# Mutant 3: a SECOND writer of the resume slot, i.e. a ring jump that does not
# pass the epoch gate.
"$PY" - "$MIXIN" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = "    def init_pp_loop_state(self: Scheduler):"
inject = (
    "    def _mutant_second_jump(self, arm_mb):\n"
    "        self._pp_flip_resume_slot = int(arm_mb)\n\n"
)
p.write_text(s.replace(anchor, inject + anchor, 1))
PY
expect_killed "second resume-slot writer" "test_only_one_site_sets_a_resume_slot"

# Mutant 4: a slot recorded without its generation -- #829's defect in one line.
"$PY" - "$MIXIN" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
anchor = "    def init_pp_loop_state(self: Scheduler):"
inject = (
    "    def _mutant_arm_without_epoch(self, mb_id):\n"
    "        self._pp_flip_arm_mb_id = mb_id\n\n"
)
p.write_text(s.replace(anchor, inject + anchor, 1))
PY
expect_killed "arm slot recorded without its epoch" "test_only_one_site_records_an_arm_slot + pair"

echo "== restoring and confirming green =="
restore
GREEN="$(run_pins)"
echo "  $GREEN"
echo "$GREEN" | grep -q "failed" && FAILED=1

if [ "$FAILED" -eq 0 ]; then
  echo "ALL MUTANTS KILLED, tree restored green"
  exit 0
fi
echo "MUTANT SURVIVED or tree not restored"
exit 1
