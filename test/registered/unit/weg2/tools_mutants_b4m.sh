#!/usr/bin/env bash
# #1329 B4m mutant harness -- DANGER DIRECTION FIRST.  For a REPLAY the
# dangerous failure is a test that passes no matter what the recorded inputs
# say (seat 5's M9 lesson: an instrument pin that only checks a key's
# EXISTENCE survives a VALUE mutation).  So most mutants below change a
# RECORDED NUMBER or a recorded input FILE and require the file to go RED;
# the rest break the replay's own arithmetic.
# Run from the repo root: bash test/registered/unit/weg2/tools_mutants_b4m.sh
set -u
BASE=test/registered/unit/weg2/test_weg2_launch_replay_1329.py
WORK=test/registered/unit/weg2/_b4m_mutant_tmp.py
FX=test/registered/unit/weg2/fixtures/xchg_launch_replay_0911
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$(pwd)/python"

run() {
  out=$(timeout 900 python3 -m pytest "$WORK" -q -p no:cacheprovider 2>&1 \
        | sed -r 's/\x1b\[[0-9;]*m//g' | tail -3)
  if echo "$out" | grep -qE "[0-9]+ (failed|error)"; then got=RED
  elif echo "$out" | grep -qE "[0-9]+ passed"; then got=GREEN
  else got=RED; fi
  printf '%-58s expected=%-5s got=%s\n' "$1" "$2" "$got"
  [ "$got" = "$2" ] || printf '    ^^ MUTANT SURVIVED / BASELINE BROKE\n%s\n' "$out"
}

restore() { cp "$BASE" "$WORK"; cp "$FX/recorded.json" "$FX/_recorded.bak"; }
undo_fixture() { mv "$FX/_recorded.bak" "$FX/recorded.json"; }

restore; run "N0 baseline (unmutated)" GREEN; undo_fixture

# N1: a recorded ring byte count is off by one -> the fidelity pin must notice
restore
python3 - "$FX/recorded.json" <<'EOF'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["ring_table"]["ring_bytes"] += 1
json.dump(d, open(p,"w"), indent=2)
EOF
run "N1 recorded ring_bytes +1" RED; undo_fixture

# N2: a recorded W19 reserve row is off by 4 MiB
restore
python3 - "$FX/recorded.json" <<'EOF'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["wall_xsn14_w19_dormant_residue"]["rows"][0]["reserve_mib"] += 4
json.dump(d, open(p,"w"), indent=2)
EOF
run "N2 recorded W19 reserve +4 MiB" RED; undo_fixture

# N3: a recorded excess row is off -> the wall's own arithmetic must notice
restore
python3 - "$FX/recorded.json" <<'EOF'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["wall_xsn14_w19_dormant_residue"]["rows"][1]["excess_mib"] = 999
json.dump(d, open(p,"w"), indent=2)
EOF
run "N3 recorded W19 excess -> 999" RED; undo_fixture

# N4: the cards are listed in NVML order instead of ORDINAL order -- the
# silent-wrong-per-card trap this fixture was built through
restore
python3 - "$FX/recorded.json" <<'EOF'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["cards_ordinal_order"] = sorted(d["cards_ordinal_order"], key=lambda c: c["nvml_index"])
json.dump(d, open(p,"w"), indent=2)
EOF
run "N4 cards in nvml order, not ordinal order" RED; undo_fixture

# N5: the reserve ignores the ARM STRING (always the serving constants)
restore
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace("        return {c.uuid: L.dc_measured_d_mib(c, weight_source) + slack",
            "        return {c.uuid: L.dc_measured_d_mib(c, L.WEIGHT_SOURCE_DEFAULT) + slack  # N5", 1)
open(p,'w').write(s)
EOF
run "N5 reserve ignores the form selector" RED; undo_fixture

# N6: the reserve drops the named slack term
restore
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace("        slack = L.reserve_slack_mib(transport)", "        slack = 0  # N6", 1)
open(p,'w').write(s)
EOF
run "N6 reserve drops the slack term" RED; undo_fixture

# N7: the front's predicate is inverted in the replay
restore
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace("                if reserve.get(u) is not None and m > reserve[u]}",
            "                if reserve.get(u) is not None and m < reserve[u]}  # N7", 1)
open(p,'w').write(s)
EOF
run "N7 W19 predicate inverted" RED; undo_fixture

# N8: the ring evidence loses the source boot's NVML->CUDA ordinal map line
restore
python3 - "$FX/ring_evidence" <<'EOF'
import os,sys,shutil
d=sys.argv[1]
f=[n for n in os.listdir(d) if n.endswith(".front.log")][0]
p=os.path.join(d,f); shutil.copy(p,p+".bak")
keep=[l for l in open(p,errors="replace") if "ordinal map" not in l]
open(p,"w").write("".join(keep))
EOF
run "N8 ring evidence loses the ordinal map" RED
python3 - "$FX/ring_evidence" <<'EOF'
import os,sys,shutil
d=sys.argv[1]
f=[n for n in os.listdir(d) if n.endswith(".front.log")][0]
shutil.move(os.path.join(d,f+".bak"), os.path.join(d,f))
EOF
undo_fixture

# N9: the contradiction guard is fed a ZERO ring (the wall cannot fire)
restore
python3 - "$WORK" <<'EOF'
import sys
p=sys.argv[1]; s=open(p).read()
s=s.replace('            ring_bytes=RECORDED["ring_table"]["ring_bytes"],\n            ring_span1_bytes=RECORDED["ring_table"]["ring_span1_bytes"],\n            ring_absent_by_design=ring_absent,',
            '            ring_bytes=0,  # N9\n            ring_span1_bytes=0,\n            ring_absent_by_design=ring_absent,', 1)
open(p,'w').write(s)
EOF
run "N9 contradiction fed a zero ring" RED; undo_fixture

restore; run "N10 baseline again (unmutated)" GREEN; undo_fixture
rm -f "$WORK"
