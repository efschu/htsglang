#!/bin/bash
# weg2_coldstart_rig_mutants.sh -- source mutants of the S7/S8 fixes against the cold-start rig.
#
#   tools/weg2_coldstart_rig_mutants.sh <SHA|ref>        (the target tree must CONTAIN the fixes: h16o2 364ab5de5d)
#
# NOT covered by any mutant: ACK-DRAIN (TOLD_ACK_DRAIN off, R4 survives: the pass floor keeps the rid open in S7b) and GROUP-FLOOR.
# A scratch worktree (.claude/worktrees/coldstart-mut-<sha10>) is made at the SHA; per mutant ONE fix is turned back in
# the SOURCE (not by an env switch), the rig runs there (tools/weg2_coldstart_rig.sh <dir>), the expected scenarios must
# be ROT, the file is restored. A mutant whose scenarios stay GRUEN means the rig measures nothing for that fix.
# Exit 0 = every mutant killed, 1 = at least one survived, 2 = rig/patch error.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=${RIG_REPO:-/spinning/htsglang}
WT_BASE=${RIG_WT_BASE:-$REPO/.claude/worktrees}
OUT=${RIG_OUT:-/tmp/weg2_coldstart_rig}/mutants
[ "$#" -eq 1 ] || { sed -n '2,10p' "$0" >&2; exit 2; }
sha=$(git -C "$REPO" rev-parse --verify --quiet "$1^{commit}") || { echo "unknown target $1" >&2; exit 2; }
wt=$WT_BASE/coldstart-mut-${sha:0:10}
[ -d "$wt" ] || git -C "$REPO" worktree add --detach "$wt" "$sha" >/dev/null 2>&1 || { echo "worktree add failed" >&2; exit 2; }
mkdir -p "$OUT"
exec python3 - "$HERE" "$wt" "${sha:0:10}" "$OUT" <<'PY'
import subprocess, sys, re, os
here, wt, label, out = sys.argv[1:5]
TOLD = "python/sglang/srt/managers/weg2_store_told.py"
FB = "python/sglang/srt/managers/weg2_told_fallback.py"
RT = "python/sglang/srt/weg2/d_park_runtime.py"
HY = "python/sglang/srt/weg2/park_hold_yield.py"
# (name, file, old, new, scenarios that must go ROT)
MUTANTS = [
    ("M7a SETTLE-ACK-NOW back to 'short' (the settle that issued no read acks only when its read ends)", TOLD,
     'str(env.get(ENV_SETTLE_ACK_NOW, "1") or "1")', '"0"', ["S7_"]),
    ("M7b FRIST-PASSES back to the wall Frist alone", FB,
     "default = max(0, int(pp_size) + 1)", "default = 0", ["S7b_"]),
    ("M8a awake re-queue clock stamped before the park's phases (SINCE/LATE_HOLD at the retraction)", RT,
     "    _flip_list = [r for r in sched.weg2_d_parked if not d_lane.lane_held(r)]\n",
     "    _flip_list = [r for r in sched.weg2_d_parked if not d_lane.lane_held(r)]\n"
     "    for req in _flip_list:\n        setattr(req, d_seats.SINCE_ATTR, now)\n", ["S8_"]),
    ("M8b HY settle walks L3 before the first vote on every park", HY,
     "    refused = _end(tree)\n    if not envs.SGLANG_WEG2_ENABLE_PARK_HOLD_YIELD.get():\n        return None\n",
     "    refused = _end(tree)\n    if not envs.SGLANG_WEG2_ENABLE_PARK_HOLD_YIELD.get():\n        return None\n"
     "    for _r in parked:\n        held_facts(tree, str(_r.rid))\n", ["S8b_"]),
    ("M8c second HY vote entered by the refusing rank alone (rank-local branch)", HY,
     '    tree._all_reduce_attn_groups(t, torch.distributed.ReduceOp.MIN, label="park_hold_yield_facts")\n',
     '    if refused:\n        tree._all_reduce_attn_groups(t, torch.distributed.ReduceOp.MIN, label="park_hold_yield_facts")\n',
     ["S8c_"]),
    # R2 (review 1015): park_tick without the group MIN -- the rank-local clock verdict decides alone (G1)
    ("R2 park_tick decides from the rank-local clock, no group MIN", RT,
     "        due = bool(sched._weg2_group_min_flags([local])[0])\n",
     "        due = bool(local)\n", ["S8_"]),
    # R3 (review 1015): the awake clock stamped right after the HY yield phase instead of at the answer (partial
    # reversal of flipstall fix A: before #248 PARK-MARK, #59b and the clamp)
    ("R3 awake clock stamped right after the yield phase, park_answer_stamp stamps nothing", RT,
     '    _ph("yield")\n',
     '    _ph("yield")\n    for _r in sched.weg2_d_parked:\n        setattr(_r, d_seats.SINCE_ATTR, time.monotonic())\n'
     '    setattr(sched, LATE_HOLD_ATTR, time.monotonic() if late_hold_armed else None)\n', ["S8_"]),
]
# M8a also needs park_answer_stamp not to restamp later: neutralise its two writes
M8A_EXTRA = (RT, "    for req in flip_list:\n        setattr(req, d_seats.SINCE_ATTR, answered)\n"
                 "    setattr(sched, LATE_HOLD_ATTR, answered if late_hold else None)\n", "    pass\n")

def rig(extra_env=None):
    p = subprocess.run([os.path.join(here, "tools/weg2_coldstart_rig.sh"), wt], capture_output=True, text=True)
    rows = {}
    for line in p.stdout.splitlines():
        m = re.match(r"\s+(GRUEN|ROT|N/A|RIG-FEHLER)\s+(\S+)", line)
        if m:
            rows[m.group(2)] = m.group(1)
    return p.returncode, rows, p.stdout

def git(*a):
    return subprocess.run(["git", "-C", wt, *a], capture_output=True, text=True)

git("checkout", "--", ".")
rc, base, txt = rig()
open(os.path.join(out, f"{label}.baseline.txt"), "w").write(txt)
bad = [k for k, v in base.items() if v not in ("GRUEN", "N/A")]
print(f"== mutants @ {label}: baseline rig rc={rc}, {len(base)} scenarios, not green: {bad or 'none'}")
if rc == 2 or not base:
    print("RIG-FEHLER in the baseline run -- see", out); sys.exit(2)
survived = 0
for name, f, old, new, want in MUTANTS:
    path = os.path.join(wt, f)
    src = open(path).read()
    if src.count(old) != 1:
        print(f"  PATCH-FEHLER {name}: anchor found {src.count(old)}x in {f}"); sys.exit(2)
    open(path, "w").write(src.replace(old, new))
    if name.startswith(("M8a", "R3")):
        p2 = os.path.join(wt, M8A_EXTRA[0]); s2 = open(p2).read()
        if s2.count(M8A_EXTRA[1]) != 1:
            print(f"  PATCH-FEHLER {name.split()[0]} extra anchor"); git("checkout", "--", "."); sys.exit(2)
        open(p2, "w").write(s2.replace(M8A_EXTRA[1], M8A_EXTRA[2]))
    rc, rows, txt = rig()
    open(os.path.join(out, f"{label}.{name.split()[0]}.txt"), "w").write(txt)
    git("checkout", "--", ".")
    killed = all(any(k.startswith(w) and v == "ROT" for k, v in rows.items()) for w in want)
    others = sorted(k for k, v in rows.items() if v == "ROT" and not any(k.startswith(w) for w in want))
    survived += 0 if killed else 1
    print(f"  {'GETOETET ' if killed else 'UEBERLEBT'} {name}\n"
          f"             expected ROT {want}: {[k + '=' + v for k, v in rows.items() if any(k.startswith(w) for w in want)]}"
          f"{'; also ROT: ' + str(others) if others else ''}")
sys.exit(1 if survived else 0)
PY
