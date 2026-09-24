"""27B line desk gate (24.09.): reads the arm's own launcher calls and holds
the launch to the standard HARD. Hermetic: a synthetic mini-arm and synthetic
dry-run logs, no launcher run. The live run (arm_xsn420 on this tree) is in
the commit message.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE = os.path.join(_HERE, "..", "..", "..", "..", "python", "sglang", "srt", "weg2", "line_gate_27b.py")
_spec = importlib.util.spec_from_file_location("line_gate_27b_under_test", os.path.abspath(_GATE))
G = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = G
_spec.loader.exec_module(G)

MINI_ARM = r'''#!/bin/bash
# header comment
_args=(); while [ $# -gt 0 ]; do
  case "$1" in --approved-older) X="${2:-}"; shift 2;; *) _args+=("$1"); shift;; esac
done; set -- "${_args[@]}"
SHA="${1:?missing}"
CAP="${4:-262144}"
BDEPTH="${5:-1}"
TAG=weg2xsnT
export SGLANG_A=1   # comment
WT=${WT_OVERRIDE:-/spinning/wt-old}
EVID=/ev/${TAG}_0924
TIP=$(git -C /x rev-parse HEAD)
say(){ echo "$*"; }
die(){ echo "$*"; exit 1; }
for p in $(pgrep -f nothing); do
  kill -TERM "$p"
done
FLAG=""
[ "0" = "1" ] && FLAG=" --never"
EVIDPROBE="$EVID/gate_probe"
gate_run(){ bk="$1"; inject="$2"; PYTHONPATH="$WT/python" timeout 900 "$PY" -m sglang.srt.weg2.launcher --tree "$WT" --tag ${TAG}gate --dry-run --xchg-bounce-depth $BDEPTH --weg2-weights-cpu-backup "$bk" --weg2-xchg-inject "$inject" > "$EVIDPROBE/probe_${bk}_$inject.log" 2>&1; }
gates_run(){
gate_run off authoritative; RCA=$?
A_LOG="$EVIDPROBE/probe_off_authoritative.log"
[ "$RCA" = "0" ] || die "rc"
gate_assert "$A_LOG" 'RING ABSENT' 'XCHG-ARMED'
gate_refuse "$A_LOG" 'RING ARMED'
}
if [ "${GATES_DONE:-}" = "$TIP" ]; then
  say "skip"
else
  gates_run
fi
BACKUP=""
if grep -q -- 'x' /dev/null; then
  BACKUP=" --never-either"
else
  BACKUP=" --weg2-weights-cpu-backup off"
fi
BASE="--tree $WT --tag $TAG --max-kv-per-request $CAP
      --p-bs 1$FLAG$BACKUP"
EXTRA_D=(--spec-form DFLASH --d-tp-objective maxkv)
export SGLANG_B=2 SGLANG_C="two words"
setsid nohup bash -c 'while true; do
  sleep 20
done' >/dev/null 2>&1 &
run(){ n="$1"; shift; timeout 900 $PY -m sglang.srt.weg2.launcher $BASE "${EXTRA_D[@]}" "$@" --dry-run > "$EVID/dry_$n.log" 2>&1; echo "$n rc=$?"; }
run control
setsid nohup $PY -m sglang.srt.weg2.launcher $BASE "${EXTRA_D[@]}" \
  --host-riegel-gib "93.0" \
  > "$LOG" 2>&1 < /dev/null &
'''


class TestArmReader(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.arm = os.path.join(self._td.name, "arm_mini.sh")
        with open(self.arm, "w") as f:
            f.write(MINI_ARM)

    def tearDown(self):
        self._td.cleanup()

    def test_calls_in_order_with_argv_env_and_criteria(self):
        calls = G.read_arm(self.arm, "/tree/under/test")
        self.assertEqual([(c.kind, c.name) for c in calls],
                         [("probe", "probe_off_authoritative"), ("dry", "dry_control"), ("launch", "launch")])
        probe, dry, launch = calls
        self.assertEqual(probe.argv[:4], ["--tree", "/tree/under/test", "--tag", "weg2xsnTgate"])
        self.assertTrue(probe.must_rc0)
        self.assertEqual(probe.must_have, ("RING ABSENT", "XCHG-ARMED"))
        self.assertEqual(probe.must_not, ("RING ARMED",))
        self.assertEqual(probe.env, {"SGLANG_A": "1"})
        self.assertEqual(launch.argv, ["--tree", "/tree/under/test", "--tag", "weg2xsnT",
                                       "--max-kv-per-request", "262144", "--p-bs", "1",
                                       "--weg2-weights-cpu-backup", "off",
                                       "--spec-form", "DFLASH", "--d-tp-objective", "maxkv",
                                       "--host-riegel-gib", "93.0"])
        self.assertEqual(launch.env, {"SGLANG_A": "1", "SGLANG_B": "2", "SGLANG_C": "two words"})
        self.assertEqual(dry.argv[-1], "--dry-run")

    def test_statements_join_quotes_and_continuations(self):
        st = G._statements('A="x\ny"\nB=1 \\\n  C=2\n# c\n')
        self.assertEqual(st, ['A="x\ny"', "B=1    C=2"])


def _dry_log(path, cut="42,11,11", attn="10,3,3", floor_cap=262144, p_tokens=279226,
             p_budget="26032,15760,15488", d_budget="27192,16112,16104", m=600, rule="cap"):
    with open(path, "w") as f:
        if rule == "cap":
            f.write(f"[t] WEG2-LAUNCH PP-CUT POOL FLOOR RULE: source=cap+chunk (p_bs=1, user order 2026-09-16) "
                    f"floor={floor_cap + 4096} = max_kv_per_request {floor_cap} + chunk 4096 -- x\n")
        else:
            f.write("[t] WEG2-LAUNCH PP-CUT POOL FLOOR RULE: source=default-from-ordered-cut 39,13,12 -- x\n")
        f.write(f"[t] WEG2-LAUNCH PP-CUT solver: objective=makespan layers={cut} attn={attn} makespan_ms=1\n")
        f.write(f"[t] WEG2-LAUNCH WEG2-HOST-LEDGER CHOSEN S=1 GB (--hicache-size, both groups) M={m} MiB x "
                f"reap_headroom=11.69 GiB (hard bound 94.43 - predicted run peak 82.74 GiB) -- y\n")
        f.write(f"[t] WEG2-LAUNCH group P argv: /v/python -m sglang.launch_server --model-path /m/q "
                f"--pp-stage-ratio {cut} --pp-attn-stage-ratio {attn} --max-total-tokens {p_tokens} "
                f"--rank-gpu-memory-mib {p_budget} --page-size 1\n")
        f.write(f"[t] WEG2-LAUNCH group D argv: /v/python -m sglang.launch_server --model-path /m/q "
                f"--rank-gpu-memory-mib {d_budget} --tp-size 3\n")
        f.write("[t] WEG2-LAUNCH DRY-RUN complete: nothing started\n")


class TestLaunchStandard(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.ref = os.path.join(self._td.name, "ref.log")
        _dry_log(self.ref, p_tokens=277997, p_budget="26008,15736,15464", d_budget="27168,16088,16088")
        self.good = G.group_argv(self.ref)

    def tearDown(self):
        self._td.cleanup()

    def _judge(self, **kw):
        p = os.path.join(self._td.name, "new.log")
        _dry_log(p, **kw)
        return G.launch_problems(p, self.good, "600", "42,11,11", "10,3,3")

    def test_standard_passes(self):
        problems, facts = self._judge()
        self.assertEqual(problems, [])
        self.assertTrue(any("within +-256" in f for f in facts))

    def test_other_cut_fails(self):
        problems, _ = self._judge(cut="43,11,10")
        self.assertTrue(any("not the standard" in p for p in problems))

    def test_ordered_cut_floor_fails(self):
        problems, _ = self._judge(rule="ordered")
        self.assertTrue(any("not cap+chunk" in p for p in problems))

    def test_small_cap_fails(self):
        problems, _ = self._judge(floor_cap=131072)
        self.assertTrue(any("one full-context request" in p for p in problems))

    def test_pool_below_one_request_fails(self):
        problems, _ = self._judge(p_tokens=246155)
        self.assertTrue(any("holds no full-context request" in p for p in problems))

    def test_budget_beyond_tolerance_fails(self):
        problems, _ = self._judge(d_budget="27192,15000,16104")
        self.assertTrue(any("beyond +-256" in p for p in problems))

    def test_other_m_fails(self):
        problems, _ = self._judge(m=300)
        self.assertTrue(any("pinned M=600" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
