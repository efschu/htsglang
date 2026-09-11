#!/usr/bin/env bash
# #1273 B4q + #1336 + B4r mutant harness -- DANGER DIRECTION FIRST.
#
# The three danger directions the operator named, plus the instrument one:
#   (i)   the `ring` arm must stay byte-identical
#   (ii)  `authoritative` must not become reachable through the widening
#   (iii) the `shadow` arm (XSN12's form) must stay armed
#   (iv)  #1336: the inject instrument must never lie towards authority
#
# Every mutant below makes the production code WRONG in one of those
# directions and must turn a test RED. Mutants mutate the PRODUCTION file, not
# the test -- a harness that can only break its own assertions proves nothing
# about the code.
#
#   bash test/registered/unit/weg2/tools_mutants_b4q.sh
set -u
T=test/registered/unit/weg2/test_weg2_xchg_gate_axis_1273.py
RING=test/registered/unit/weg2/test_weg2_xchg_shadow_1273.py
WX=python/sglang/srt/weg2/weight_exchange.py
SH=python/sglang/srt/weg2/weight_exchange_shadow.py
WU=python/sglang/srt/managers/scheduler_components/weight_updater.py
BX=python/sglang/srt/weg2/weight_exchange_bounce.py
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$(pwd)/python"

snapshot() { for f in "$WX" "$SH" "$WU" "$BX"; do cp "$f" "/tmp/$(basename $f).b4q.bak"; done; }
restore()  { for f in "$WX" "$SH" "$WU" "$BX"; do cp "/tmp/$(basename $f).b4q.bak" "$f"; done; }

run() {  # $1 label, $2 expected, $3.. test files
  local label="$1" expected="$2"; shift 2
  local out got
  out=$(timeout 1200 python3 -m pytest "$@" -q -p no:cacheprovider 2>&1 \
        | sed -r 's/\x1b\[[0-9;]*m//g' | tail -3)
  if echo "$out" | grep -qE "[0-9]+ (failed|error)"; then got=RED
  elif echo "$out" | grep -qE "[0-9]+ passed"; then got=GREEN
  else got=NO-VERDICT; fi
  printf '%-62s expected=%-5s got=%s\n' "$label" "$expected" "$got"
  [ "$got" = "$expected" ] || printf '    ^^ MUTANT SURVIVED / BASELINE BROKE\n%s\n' "$out"
}

snapshot
trap restore EXIT

run "M0 baseline, gate-axis file" GREEN "$T"
run "M0b baseline, the ring tripwire's own file" GREEN "$RING"

# ---- (i) the ring arm --------------------------------------------------
# The axis predicate stops excluding `ring`: the default boot would arm the
# lane. This is the one direction that changes a boot that has run 50 times.
restore
python3 - "$WX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("    return weight_source() != WEIGHT_SOURCE_RING",
            "    return True  # M1", 1); open(p,'w').write(s)
EOF
run "M1 axis arms the ring arm too" RED "$T" "$RING"

# ---- (iii) the shadow arm ---------------------------------------------
# Candidate A, the REJECTED form, built for real: it reduces to
# exchange_armed() and disarms the shadow arm XSN12 proved runs.
restore
python3 - "$WX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("    return weight_source() != WEIGHT_SOURCE_RING",
            "    return exchange_armed() and inject_mode() in INJECT_CHOICES  # M2",
            1); open(p,'w').write(s)
EOF
run "M2 candidate A (tautology) disarms the shadow arm" RED "$T" "$RING"

# The other half of the same direction: the axis reads only the exchange arm,
# i.e. the defect in its mirror image.
restore
python3 - "$WX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("    return weight_source() != WEIGHT_SOURCE_RING",
            "    return weight_source() == WEIGHT_SOURCE_EXCHANGE  # M3", 1)
open(p,'w').write(s)
EOF
run "M3 axis covers only the exchange arm" RED "$T" "$RING"

# ---- the two gate sites, each on its own ------------------------------
restore
python3 - "$WU" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("            if not sh.bounce_lane_armed():",
            "            if not sh.shadow_armed():  # M4", 1); open(p,'w').write(s)
EOF
run "M4 the updater gate goes back to one arm" RED "$T"

restore
python3 - "$SH" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("        armed = bounce_lane_armed()",
            "        armed = shadow_armed()  # M5", 1); open(p,'w').write(s)
EOF
run "M5 run_leg_hook's gate goes back to one arm" RED "$T"

# ---- the delegation shape ---------------------------------------------
# Re-spelling the comparison in the shadow module is the Zweitbuchhaltung the
# module's own comment refuses -- and it is how two arms drift apart.
restore
python3 - "$SH" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("    return bool(wxm.bounce_lane_armed())",
            "    return bool(wxm.weight_source() != wxm.WEIGHT_SOURCE_RING)  # M6",
            1); open(p,'w').write(s)
EOF
run "M6 the delegation re-spells the predicate" RED "$T"

# ---- (ii) authority may not move onto the axis ------------------------
restore
python3 - "$WU" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("if wx.exchange_armed() and wx.inject_authoritative():",
            "if wx.bounce_lane_armed():  # M7", 1); open(p,'w').write(s)
EOF
run "M7 authority moves onto the axis" RED "$T"

# ---- (iv) #1336 the instrument ----------------------------------------
restore
python3 - "$BX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace('f"mode={self.mode or wx.INJECT_MODE_UNSET} "',
            'f"mode={self.inject.mode if self.inject else wx.INJECT_AUTHORITATIVE} "  # M8',
            1); open(p,'w').write(s)
EOF
run "M8 the inject instrument lies towards authority again" RED "$T"

# The softer form of the same defect: default to a real mode rather than a
# named unset state. Safer in direction, still a second decision, still a lie.
restore
python3 - "$BX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace('f"mode={self.mode or wx.INJECT_MODE_UNSET} "',
            'f"mode={self.mode or wx.INJECT_SHADOW} "  # M9', 1); open(p,'w').write(s)
EOF
run "M9 unset prints as a real mode (shadow)" RED "$T"

restore
python3 - "$BX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("        mode=mode,\n", "", 1); open(p,'w').write(s)
EOF
run "M10 the mode stops travelling from its one source" RED "$T"

# ---- B4r, the instrument that must not be a constant ------------------
restore
python3 - "$WX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace('            verdict = "here" if found.tag == tag else "elsewhere"',
            '            verdict = "here"  # M11', 1); open(p,'w').write(s)
EOF
run "M11 B4r never reports elsewhere (the hypothesis' own case)" RED "$T"

restore
python3 - "$WX" <<'EOF'
import sys; p=sys.argv[1]; s=open(p).read()
s=s.replace("        for ln in plan_param_lines(", "        for ln in ():  # M12\n        _unused = (plan_param_lines(", 1)
open(p,'w').write(s)
EOF
run "M12 B4r's emitter is unwired from the arming path" RED "$T"

restore
run "M13 baseline again, both files" GREEN "$T" "$RING"
