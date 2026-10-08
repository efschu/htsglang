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

from flliper.srt.pdflip import state_file as SF

ROOT = pathlib.Path(__file__).resolve().parents[4]
DEADMAN = ROOT / "scripts" / "pdflip" / "devtools" / "boot_deadman.sh"


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
    """y5c 19:31:41-19:32:42Z: outstanding=2 (pdflip-32-62 and a sibling), both NON-stream, D decoding bs2 --
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
                json.dump({"schema": "pdflip.rankstats/1", "progress": {
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
        self.assertIn("Rang-Arbeit", lines[0])
        self.assertEqual(SF.read(self.d)["progress"]["verdict"], "HAENGT")

    def test_y6b_six_stream_requests_with_moving_rank_work_do_not_stall(self):
        # NF y6b 01.10. 03:37:50Z: six long STREAM answers, D decoded round 3562 -> 4262 at bs5-6,
        # served/served_tokens move only when a request ends -> the old rule (rank work only for
        # non-stream) stopped a healthy boot. Rank work now counts for every open request.
        self.assertEqual(self.replay([True] * 6, moving=True), [])
        self.assertNotIn("progress", SF.read(self.d))

    def test_stream_requests_with_frozen_rank_work_still_stall(self):
        lines = self.replay([True] * 6, moving=False)
        self.assertEqual(len(lines), 1)
        self.assertIn("HAENGT", lines[0])

    def test_front_without_the_field_counts_rank_work_too(self):
        fr = _front(2, 131, 42, 2512000, 61234)                  # a y5c image: no stream info
        SF.transition(self.d, "serving", fields={"front": fr})
        out = []
        for i, t in enumerate(range(1000, 1130, 10)):
            self.ranks(100000 + 8 * i, 5000 + 4 * i)
            out.append(SF.deadman_progress(self.d, self.memo, 60, now=float(t)))
        self.assertEqual([l for l in out if l], [])

    def test_no_rankstats_no_witness(self):
        SF.transition(self.d, "serving", fields={"front": _ns_front([False, False])})
        out = [SF.deadman_progress(self.d, self.memo, 60, now=float(t)) for t in range(1000, 1130, 10)]
        self.assertEqual(len([l for l in out if l]), 1)
        self.assertIsNone(SF.rank_work(self.d))

    def test_resumed_rank_work_ends_the_stall(self):
        self.replay([False, False], moving=False)
        self.ranks(200000, 9000)
        self.assertIn("LAEUFT WIEDER", SF.deadman_progress(self.d, self.memo, 60, now=1200.0))


def _y6d_front(ts, last_token_s, age_s):
    """NF y6d 08:03:31-08:04:46Z: ONE stream request (pdflip-18-47, the user's OpenWebUI answer) on D,
    served=77 / tokens=2673954 frozen for 75 s -- served and served_tokens count a request at its end.
    The front's OutstandingBook row carries the token it pushed last (state.json at 08:05:13Z:
    last_token_s 1.1, age_s 297.2, stream 1, where D)."""
    fr = _front(1, 62, 15, 1300000, 73954, ts=ts, outstanding_n=1, outstanding_nonstream_n=0)
    fr["outstanding_stalest"] = [{"rid": "pdflip-18-47", "where": "D", "age_s": age_s, "stream": 1,
                                  "last_token_s": last_token_s, "no_token_s": last_token_s}]
    return fr


class TestY6dSingleStream(unittest.TestCase):
    """y6d died at 08:04:31Z (HAENGT -> the NF arm's relay wrote stop_request.json, rc 24) while D decoded
    round 5836 -> 8292 at bs1 (#full token 446272 -> 451136) and the user received the stream."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.d = SF.init(self.root, "nf-y6d-replay", "boot", {})
        self.memo = os.path.join(self.root, "memo.json")
        os.makedirs(os.path.join(self.d, "rankstate", "D"), exist_ok=True)

    def ranks(self, k):
        for t in range(3):
            with open(os.path.join(self.d, "rankstate", "D", "D.tp%dpp0.rankstats" % t), "w") as f:
                json.dump({"schema": "pdflip.rankstats/1", "progress": {
                    "fwd_ct": 5836 + 164 * k, "tokens_done": 80000 + 330 * k}}, f)

    def replay(self, *, ranks_move, tokens_move, rankstats=True, t0=1790841811.0):
        out = []
        for k, t in enumerate(range(0, 90, 5)):
            now = t0 + t
            lt = 0.3 if tokens_move else 0.3 + t               # last token 0.3 s ago, or frozen at t0
            SF.transition(self.d, "serving" if k == 0 else None,
                          fields={"front": _y6d_front(round(now, 3), round(lt, 1), round(225.0 + t, 1))})
            if rankstats:
                self.ranks(k if ranks_move else 0)
            out.append(SF.deadman_progress(self.d, self.memo, 60, now=now))
        return [l for l in out if l]

    def test_y6d_specimen_moving_rank_work_and_stream_tokens_no_stall(self):
        self.assertEqual(self.replay(ranks_move=True, tokens_move=True), [])
        self.assertNotIn("progress", SF.read(self.d))
        self.assertFalse(os.path.exists(os.path.join(self.d, "stop_request.json")))

    def test_stream_tokens_alone_are_progress_without_rankstats(self):
        # the front's own stamp is a witness even when no rank file is readable
        self.assertEqual(self.replay(ranks_move=False, tokens_move=True, rankstats=False), [])
        self.assertIsNone(SF.rank_work(self.d))

    def test_stream_tokens_alone_beat_frozen_rank_files(self):
        self.assertEqual(self.replay(ranks_move=False, tokens_move=True), [])

    def test_all_witnesses_frozen_is_still_a_stall(self):
        lines = self.replay(ranks_move=False, tokens_move=False)
        self.assertEqual(len(lines), 1)
        self.assertIn("HAENGT", lines[0])
        self.assertIn("letzter Stream-Token vor", lines[0])
        rec = SF.read(self.d)["progress"]
        self.assertEqual(rec["verdict"], "HAENGT")
        self.assertGreaterEqual(rec["stream_token_age_s"], 60.0)

    def test_a_frozen_front_snapshot_is_no_progress(self):
        # the front's IPC snapshot stops refreshing: ts and rows stand -> the stamp stands
        SF.transition(self.d, "serving", fields={"front": _y6d_front(1000.0, 0.3, 225.0)})
        out = [SF.deadman_progress(self.d, self.memo, 60, now=float(t)) for t in range(1000, 1090, 5)]
        self.assertEqual(len([l for l in out if l]), 1)

    def test_rounding_jitter_is_no_progress(self):
        # the same token in snapshots 5 s apart: front.ts (ms) and last_token_s (0.1 s) rounding
        memo, evs = None, []
        for k, t in enumerate(range(0, 90, 5)):
            fr = _y6d_front(1000.0 + t + 0.004 * (k % 2), round(t + 0.04 * (k % 3), 1), 225.0 + t)
            memo, ev = SF.progress_step(memo, _st("b1", fr), 1000.0 + t, 60.0)
            evs.append(ev)
        self.assertEqual([e for e in evs if e], ["HAENGT"])
        self.assertEqual(evs.index("HAENGT"), 12)        # t=60: the jitter never restarted the clock

    def test_stream_token_ts_reads_only_stream_rows_with_a_token(self):
        fr = {"ts": 100.0, "outstanding_stalest": [
            {"stream": 0, "last_token_s": 1.0}, {"stream": 1, "last_token_s": None},
            {"stream": 1, "last_token_s": 7.5}, {"stream": 1, "last_token_s": 2.5}]}
        self.assertEqual(SF.stream_token_ts(fr), 97.5)
        self.assertIsNone(SF.stream_token_ts({"outstanding_stalest": [{"stream": 1, "last_token_s": 1}]}))
        self.assertIsNone(SF.stream_token_ts({"ts": 5.0}))


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
        env = {"PATH": os.environ["PATH"], "PDFLIP_DEADMAN_GROUP": group, "PDFLIP_STATE_DIR": state_dir,
               "PDFLIP_STATE_FILE_PY": SF.__file__, "PDFLIP_PY": sys.executable, "PROGRESS_STALL_S": stall,
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
