"""PROGRESS-STALL, the 4th deadman tier (30.09.).

dual13 11:00-11:04Z: the front had outstanding=3 at queue=0, P wrote WAIT lines, D ran with 0 running -- and no
watcher fired: the log was not silent, the heartbeats ran, /health answered 200. NF built the host-side
progress_watch.py (state/current/state.json: HAENGT when outstanding > 0 and served / served_tokens do not move
for STALL_S). This tier puts the same rule into the deadman the IMAGE runs, with the verdict as STATE:

  * a fake state.json history without progress (outstanding > 0, counters frozen) -> HAENGT once, field
    `progress` = {verdict HAENGT, ...} + event progress_stall, ONE line; nothing stopped (no stop_request);
  * a history with movement -> nothing;
  * resume -> LAEUFT WIEDER once; a new boot_id resets; outstanding 0 / not serving never stalls;
  * the threshold comes from the environment (PROGRESS_STALL_S), default 60 s;
  * the deadman's own shell function writes the line into the boot log it watches and only runs for the
    front deadman with a state dir.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

from sglang.srt.weg2 import state_file as SF

ROOT = pathlib.Path(__file__).resolve().parents[4]
DEADMAN = ROOT / "scripts" / "weg2" / "devtools" / "boot_deadman.sh"


def _front(out, served_d, served_p, prompt, completion, **kw):
    return {"outstanding": out, "queue": 0, "awake": "D", "state": "serving",
            "outstanding_by_group": {"D": out, "P": 0}, "served": {"D": served_d, "P": served_p},
            "served_tokens": {"D": {"prompt": prompt, "completion": completion},
                              "P": {"prompt": prompt, "completion": 0}}, **kw}


def _st(boot, front, lc="serving"):
    return {"boot_id": boot, "lifecycle": {"state": lc}, "front": front}


class TestProgressStep(unittest.TestCase):
    """The pure step: a fake state history in, events out."""

    def run_history(self, hist, stall_s=60.0):
        memo, evs = None, []
        for t, st in hist:
            memo, ev = SF.progress_step(memo, st, float(t), stall_s)
            evs.append(ev)
        return evs

    def test_frozen_counters_with_outstanding_hang_once(self):
        fr = _front(3, 23, 10, 436773, 4224)
        evs = self.run_history([(t, _st("b1", fr)) for t in range(0, 200, 10)])
        self.assertEqual([e for e in evs if e], ["HAENGT"])          # once, not repeated
        self.assertEqual(evs.index("HAENGT"), 6)                      # at t=60 (threshold reached)

    def test_movement_gives_nothing(self):
        hist = [(t, _st("b1", _front(3, 23, 10, 436773, 4224 + t))) for t in range(0, 300, 10)]
        self.assertEqual([e for e in self.run_history(hist) if e], [])
        hist = [(t, _st("b1", _front(3, 23 + t // 50, 10, 436773, 4224))) for t in range(0, 300, 10)]
        self.assertEqual([e for e in self.run_history(hist) if e], [])

    def test_resume_new_boot_idle_and_not_serving(self):
        fr = _front(3, 23, 10, 436773, 4224)
        hist = [(t, _st("b1", fr)) for t in range(0, 100, 10)]
        hist.append((100, _st("b1", _front(3, 24, 10, 436773, 4300))))   # progress again
        evs = self.run_history(hist)
        self.assertEqual([e for e in evs if e], ["HAENGT", "LAEUFT"])
        # a new boot_id resets: the old stall does not carry over, the clock restarts
        hist = [(t, _st("b1", fr)) for t in range(0, 100, 10)] + [(t, _st("b2", fr)) for t in range(100, 150, 10)]
        evs = self.run_history(hist)
        self.assertEqual([e for e in evs if e], ["HAENGT"])              # b1 only; b2 below 60 s, no LAEUFT
        # outstanding 0 (idle) or not serving never stalls
        self.assertEqual([e for e in self.run_history(
            [(t, _st("b1", _front(0, 23, 10, 1, 1))) for t in range(0, 300, 10)]) if e], [])
        self.assertEqual([e for e in self.run_history(
            [(t, _st("b1", fr, lc="loading")) for t in range(0, 300, 10)]) if e], [])

    def test_threshold_is_a_parameter(self):
        fr = _front(3, 23, 10, 436773, 4224)
        evs = self.run_history([(t, _st("b1", fr)) for t in range(0, 40, 10)], stall_s=20.0)
        self.assertEqual(evs.index("HAENGT"), 2)


class TestProgressCheckOnStateDir(unittest.TestCase):
    """deadman_progress / the CLI on a real state dir: the verdict is the field `progress`, not a stop."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.d = SF.init(self.root, "27bdm-boot-test", "boot", {})
        SF.transition(self.d, "serving", fields={"front": _front(3, 23, 10, 436773, 4224)})
        self.memo = os.path.join(self.root, "memo.json")

    def set_front(self, fr):
        SF.transition(self.d, None, fields={"front": fr})

    def test_hang_is_state_and_event_and_no_stop(self):
        self.assertEqual(SF.deadman_progress(self.d, self.memo, 60, now=1000.0), "")
        self.assertEqual(SF.deadman_progress(self.d, self.memo, 60, now=1030.0), "")
        line = SF.deadman_progress(self.d, self.memo, 60, now=1061.0)
        self.assertIn("DEADMAN[PROGRESS-STALL]", line)
        self.assertIn("HAENGT", line)
        st = SF.read(self.d)
        self.assertEqual(st["progress"]["verdict"], "HAENGT")
        self.assertEqual(st["progress"]["outstanding"], 3)
        self.assertEqual(st["lifecycle"]["state"], "serving")                 # no state change, no death
        self.assertFalse(os.path.exists(os.path.join(self.d, "stop_request.json")))
        self.assertEqual(SF.health(self.d)[0], SF.HEALTH_OK)                  # the healthcheck does not kill
        self.assertIn("progress_stall", [e.get("type") for e in SF.events(self.d)])
        self.assertEqual(SF.deadman_progress(self.d, self.memo, 60, now=1200.0), "")   # not repeated
        self.set_front(_front(3, 24, 10, 436900, 4300))                                 # progress
        line = SF.deadman_progress(self.d, self.memo, 60, now=1210.0)
        self.assertIn("LAEUFT WIEDER", line)
        self.assertEqual(SF.read(self.d)["progress"]["verdict"], "LAEUFT")

    def test_movement_writes_nothing(self):
        for i, t in enumerate(range(1000, 1300, 20)):
            self.set_front(_front(3, 23 + i, 10, 436773 + 100 * i, 4224 + i))
            self.assertEqual(SF.deadman_progress(self.d, self.memo, 60, now=float(t)), "")
        self.assertNotIn("progress", SF.read(self.d))

    def test_cli_threshold_from_env(self):
        py = [sys.executable, SF.__file__, "progress-check", "--dir", self.d, "--memo", self.memo]
        env = dict(os.environ, PROGRESS_STALL_S="0.3")
        self.assertEqual(subprocess.run(py, env=env, capture_output=True, text=True).stdout, "")
        import time
        time.sleep(0.4)
        r = subprocess.run(py, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("HAENGT", r.stdout)
        # default 60 s when the env is absent
        os.unlink(self.memo)
        env.pop("PROGRESS_STALL_S")
        subprocess.run(py, env=env, capture_output=True, text=True)
        time.sleep(0.4)
        self.assertEqual(subprocess.run(py, env=env, capture_output=True, text=True).stdout, "")
        self.assertEqual(SF.PROGRESS_STALL_S_DEFAULT, 60.0)

    def test_only_the_deadman_owns_progress(self):
        self.assertTrue(SF.owns("deadman", "progress"))
        with self.assertRaises(SF.StateFileError):
            SF.transition(self.d, None, fields={"progress": {}}, writer="front")


def _ns_front(stream_flags):
    """y5c 19:31:41-19:32:42Z: outstanding=2 (weg2-32-62 and a sibling), both NON-stream, D decoding bs2 --
    served / served_tokens frozen (a non-stream answer reaches the front only at its end)."""
    fr = _front(len(stream_flags), 131, 42, 2512000, 61234)
    fr["outstanding_n"] = len(stream_flags)
    fr["outstanding_nonstream_n"] = sum(1 for s in stream_flags if not s)
    return fr


class TestNonStreamRankWork(unittest.TestCase):
    """y5c died at 19:32:42Z on PROGRESS-STALL ("outstanding=2, served/tokens 61 s unbewegt") while D decoded
    bs2 until 19:32:59: both open requests were non-stream. With only non-stream requests open, the ranks'
    progress counters (rankstate/*/*.rankstats, the contract progress_watch reads) are the witness."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.d = SF.init(self.root, "nf-y5c-replay", "boot", {})
        self.memo = os.path.join(self.root, "memo.json")
        os.makedirs(os.path.join(self.d, "rankstate", "D"), exist_ok=True)

    def ranks(self, tokens_done, fwd_ct):
        for t in range(3):
            p = os.path.join(self.d, "rankstate", "D", "D.tp%dpp0.rankstats" % t)
            with open(p, "w") as f:
                json.dump({"schema": "weg2.rankstats/1", "progress": {
                    "fwd_ct": fwd_ct, "tokens_done": tokens_done, "prefill_tokens": 0,
                    "decode_tokens": tokens_done}}, f)

    def replay(self, flags, moving):
        SF.transition(self.d, "serving", fields={"front": _ns_front(flags)})
        lines = []
        for i, t in enumerate(range(1000, 1130, 10)):
            k = i if moving else 0
            self.ranks(100000 + 8 * k, 5000 + 4 * k)          # bs2 x 4 verify rows per round
            lines.append(SF.deadman_progress(self.d, self.memo, 60, now=float(t)))
        return [l for l in lines if l]

    def test_y5c_two_nonstream_with_moving_rank_work_do_not_stall(self):
        self.assertEqual(self.replay([False, False], moving=True), [])
        self.assertNotIn("progress", SF.read(self.d))

    def test_rank_work_standing_still_is_a_real_stall(self):
        lines = self.replay([False, False], moving=False)
        self.assertEqual(len(lines), 1)
        self.assertIn("HAENGT", lines[0])
        self.assertIn("nur Nicht-Stream", lines[0])
        self.assertEqual(SF.read(self.d)["progress"]["verdict"], "HAENGT")

    def test_a_stream_request_open_keeps_the_old_rule(self):
        # one stream request among them: its chunks would move served_tokens -- rank work is not read
        lines = self.replay([False, True], moving=True)
        self.assertEqual(len(lines), 1)
        self.assertIn("HAENGT", lines[0])

    def test_front_without_the_field_keeps_the_old_rule(self):
        fr = _front(2, 131, 42, 2512000, 61234)                  # a y5c image: no stream info
        SF.transition(self.d, "serving", fields={"front": fr})
        out = []
        for i, t in enumerate(range(1000, 1130, 10)):
            self.ranks(100000 + 8 * i, 5000 + 4 * i)
            out.append(SF.deadman_progress(self.d, self.memo, 60, now=float(t)))
        self.assertEqual(len([l for l in out if l]), 1)

    def test_no_rankstats_no_witness(self):
        SF.transition(self.d, "serving", fields={"front": _ns_front([False, False])})
        out = [SF.deadman_progress(self.d, self.memo, 60, now=float(t)) for t in range(1000, 1130, 10)]
        self.assertEqual(len([l for l in out if l]), 1)
        self.assertIsNone(SF.rank_work(self.d))

    def test_resumed_rank_work_ends_the_stall(self):
        self.replay([False, False], moving=False)
        self.ranks(200000, 9000)
        self.assertIn("LAEUFT WIEDER", SF.deadman_progress(self.d, self.memo, 60, now=1200.0))


@unittest.skipUnless(DEADMAN.exists(), "deadman not in the tree")
class TestDeadmanShell(unittest.TestCase):
    """The deadman's own shell function (extracted and run), and its wiring in the main loop."""

    def _fn(self):
        src = DEADMAN.read_text()
        i = src.index("PROGRESS_STALL_S=\"${PROGRESS_STALL_S:-60}\"")
        j = src.index("while true; do", i)
        return src[i:j]

    def _run(self, group, state_dir, log, stall="0.2"):
        script = self._fn() + "\nprogress_check\nsleep 0.35\nprogress_check\n"
        env = {"PATH": os.environ["PATH"], "WEG2_DEADMAN_GROUP": group, "WEG2_STATE_DIR": state_dir,
               "WEG2_STATE_FILE_PY": SF.__file__, "WEG2_PY": sys.executable, "PROGRESS_STALL_S": stall,
               "PROGRESS_MEMO": os.path.join(state_dir, "memo.json"), "LOG": log,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        return subprocess.run(["bash", "-c", "LOG=$LOG; " + script], env=env, capture_output=True, text=True)

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.d = SF.init(self.root, "27bdm-boot-sh", "boot", {})
        SF.transition(self.d, "serving", fields={"front": _front(3, 23, 10, 436773, 4224)})
        self.log = os.path.join(self.root, "front.log")
        open(self.log, "w").write("boot log\n")

    def test_front_deadman_writes_state_and_one_log_line(self):
        r = self._run("front", self.d, self.log)
        self.assertIn("HAENGT", r.stdout, r.stderr)
        lines = [l for l in open(self.log).read().splitlines() if "PROGRESS-STALL" in l]
        self.assertEqual(len(lines), 1)
        self.assertEqual(SF.read(self.d)["progress"]["verdict"], "HAENGT")

    def test_other_deadmen_and_no_state_dir_stay_silent(self):
        self.assertNotIn("PROGRESS", self._run("D", self.d, self.log).stdout)
        self.assertNotIn("PROGRESS", self._run("front", os.path.join(self.root, "none"), self.log).stdout)
        self.assertNotIn("progress", SF.read(self.d))

    def test_wired_first_in_the_main_loop_and_never_calls_verdict(self):
        src = DEADMAN.read_text()
        loop = src[src.rindex("while true; do"):]
        self.assertTrue(loop.split("\n")[2].strip().startswith("progress_check"), loop[:300])
        fn = src[src.index("progress_check() {"):src.index("while true; do", src.index("progress_check() {"))]
        self.assertNotIn("verdict ", fn)          # no exit of the watcher, no stop
        self.assertEqual(subprocess.run(["bash", "-n", str(DEADMAN)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()


class Test27bFrontPublishesTheOutstandingBookFields(unittest.TestCase):
    """27B has no OutstandingBook: the front's own state block publishes the two fields
    only_nonstream reads (outstanding_n, outstanding_nonstream_n; NF f0bc387040)."""

    def test_nonstream_outstanding_is_counted(self):
        import types

        from sglang.srt.weg2 import front as F

        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="t", store_dir="/tmp",
                    prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0, weight_chunks=2,
                    flip_min_work_tokens=1)
        import time as _t

        f._ipc_stream_of = {"weg2-32-62": (False, _t.time()), "weg2-32-63": (False, _t.time()),
                            "weg2-32-64": (True, _t.time())}
        f.groups["D"].outstanding = {"weg2-32-62": _t.time(), "weg2-32-63": _t.time()}
        out = f._ipc_front_fields()
        self.assertEqual((out["outstanding_n"], out["outstanding_nonstream_n"]), (2, 2))
        self.assertTrue(SF.only_nonstream(out))
        f.groups["D"].outstanding["weg2-32-64"] = _t.time()                # one stream request open
        out = f._ipc_front_fields()
        self.assertEqual((out["outstanding_n"], out["outstanding_nonstream_n"]), (3, 2))
        self.assertFalse(SF.only_nonstream(out))
