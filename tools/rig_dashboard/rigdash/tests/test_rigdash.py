"""Unit tests for rigdash (stdlib unittest; run: python3 -m unittest discover -s tests).

The fixture lines are verbatim from the weg2 boot logs of 2026-09-27
(NF dkrnfh91bar1dauer09270859, 27B dkr27breleasedraftbar1w109270737).
"""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import energy, health, imagechanges, live, parse, server, sources, stops  # noqa: E402

P_RANK0 = ("[2026-09-27 09:20:28 PP0] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, "
           "gpu-ms: 3255.5 (compute 3255.5, wait 0.0) (wait by family: tp.all_reduce 0.0/29x) bubble_ms=16.5 "
           "(between forwards, mb=1)")
P_RANK1 = ("[2026-09-27 09:20:28 PP1] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, "
           "gpu-ms: 2730.0 (compute 2730.0, wait 0.0) (wait by family: tp.all_reduce 0.0/11x) bubble_ms=621.4 "
           "(between forwards, mb=0)")
P_BATCH = ("[2026-09-27 09:20:28 PP0] Prefill batch, #new-seq: 1, #new-token: 16384, #cached-token: 0, full token "
           "usage: 0.26, mamba usage: 0.09, #running-req: 0, #queue-req: 1, #pending-token: 80401, cuda graph: False, "
           "input throughput (token/s): 4898.99")
D_RANK_NOSPLIT = "[2026-09-27 09:18:53 TP1] Prefill rank batch, #new-token: 4, #cached-token: 52172, #chunks: 1"
D_RANK_27B = ("[2026-09-27 07:39:20 TP0] Prefill rank batch, #new-token: 25, #cached-token: 0, #chunks: 1, gpu-ms: "
              "568.2 (compute 440.8, wait 127.4) (wait by family: tp.all_reduce 111.4/129x, dcp.all_gather 15.4/16x)")
DEC = ("[2026-09-27 09:19:23 TP0] Decode batch, #running-req: 4, #full token: 214912, full token usage: 0.82, mamba "
       "num: 16, mamba usage: 0.42, accept len: 2.70, accept rate: 0.57, cuda graph: True, gen throughput "
       "(token/s): 123.73, #queue-req: 0")
DEC_RANK = ("[2026-09-27 09:21:17 TP0] Decode rank batch, rank: 0, #round: 7453, t: 1790500877.238, bs: 5, #rows: 20, "
            "#fwd: 1, gpu-ms: 117.7 (split unavailable: graph-replay-reader-off, graphed-fwd 1/1)")
CLOCK_PROSE = ("[2026-09-27 09:03:29 TP1] collective clock: graph reader OFF (...): Graphed 'Decode rank batch' lin")
FLIP_BEGIN = "[2026-09-27 09:20:42,596] INFO weg2.front: WEG2-FLIP begin epoch=17 sleep=P wake=D outstanding=0 queue=2"
FLIP_DONE = ("[2026-09-27 09:20:44,544] INFO weg2.front: WEG2-FLIP done epoch=18 slept=P woke=D drain+quiesce=177 ms "
             "sleep=1563 ms (kv RPC + the P leg of the gathered pair) wake=1744 ms (the D leg + kv RPC) interleave=1642 "
             "ms overlap=1540 ms critical_path=wake/D rank=0 card=GPU-31d7 ms=1520 flip_total=1948 ms weights_tags=17")
CORRIDOR = ("[2026-09-27 09:20:35,254] INFO weg2.front: WEG2-CORRIDOR phase=P(awake) epoch=17 "
            "instrument=nvml_v2_free,allocatable band=858-1314MiB")
HEALTH = "[2026-09-27 09:24:07,713] WARNING weg2.front: WEG2-HEALTH group=P http_ok=False process_alive=False streak=4"
WARN_TIMEOUT = ("[2026-09-27 07:41:36,865] WARNING weg2.front: WEG2 D-POOL UNREADABLE (cause=TimeoutError: ): group D "
                "published no reading")
ERR = ("[2026-09-27 09:07:08,831] ERROR weg2.front: WEG2 leg2 rid=weg2-0-9 failed: ClientConnectionResetError: Cannot "
       "write to closing transport")
BOOT = ("[2026-09-27T08:59:39Z] WEG2-LAUNCH === WEG2 BOOT tag=dkrnfh91bar1dauer09270859 tree=/opt/htsglang/src-nf "
        "@ 8f0bf40c2f (clean) stamp=0927_085939 dry=False")
FORM = ("[2026-09-27T08:59:39Z] WEG2-LAUNCH WEG2-FORM arch=moe experts=offload draft=mtp model=Qwen3.8-Flash-Next "
        "(sources: arch <- checkpoint)")
ARGS = ("[2026-09-27 08:59:56] server_args=ServerArgs(model_path='/m/Qwen3.8-Flash-Next', "
        "served_model_name='Qwen3.8-Flash-Next', tp_size=1, pp_size=3, x=1)")


class ParseTests(unittest.TestCase):
    def test_prefill_rank_compute_honest_fields(self):
        e = parse.parse_line(P_RANK0)
        self.assertEqual((e["kind"], e["rk"], e["rank"]), ("prefill_rank", "PP", 0))
        self.assertEqual(e["new_tok"], 16384)
        self.assertAlmostEqual(e["compute_ms"], 3255.5)
        self.assertAlmostEqual(e["wait_ms"], 0.0)

    def test_compute_not_gpu_ms_when_wait_present(self):
        e = parse.parse_line(D_RANK_27B)
        self.assertAlmostEqual(e["gpu_ms"], 568.2)
        self.assertAlmostEqual(e["compute_ms"], 440.8)
        self.assertAlmostEqual(e["wait_ms"], 127.4)

    def test_rank_line_without_split_is_unrated(self):
        e = parse.parse_line(D_RANK_NOSPLIT)
        self.assertEqual(e["kind"], "prefill_rank")
        self.assertIsNone(e["compute_ms"])

    def test_prefill_batch_wall_is_labelled_wall(self):
        e = parse.parse_line(P_BATCH)
        self.assertEqual(e["kind"], "prefill_batch")
        self.assertAlmostEqual(e["wall_tps"], 4898.99)
        self.assertEqual((e["queue"], e["pending_tok"]), (1, 80401))

    def test_decode_lines(self):
        e = parse.parse_line(DEC)
        self.assertEqual((e["kind"], e["running"]), ("decode_batch", 4))
        self.assertAlmostEqual(e["gen_tps"], 123.73)
        self.assertAlmostEqual(e["accept_len"], 2.70)
        r = parse.parse_line(DEC_RANK)
        self.assertEqual((r["kind"], r["bs"]), ("decode_rank", 5))
        self.assertAlmostEqual(r["gpu_ms"], 117.7)

    def test_prose_mentioning_decode_rank_batch_is_not_a_row(self):
        self.assertIsNone(parse.parse_line(CLOCK_PROSE))

    def test_front_families(self):
        b = parse.parse_line(FLIP_BEGIN)
        self.assertEqual((b["kind"], b["sleep"], b["wake"]), ("flip_begin", "P", "D"))
        d = parse.parse_line(FLIP_DONE)
        self.assertEqual((d["kind"], d["slept"], d["woke"]), ("flip_done", "P", "D"))
        self.assertEqual(d["total_ms"], 1948)
        self.assertEqual(d["sleep_ms"], 1563)
        self.assertEqual(d["wake_ms"], 1744)
        self.assertAlmostEqual(d["t"] % 1, 0.544, places=3)
        c = parse.parse_line(CORRIDOR)
        self.assertEqual((c["kind"], c["awake"]), ("phase", "P"))
        h = parse.parse_line(HEALTH)
        self.assertEqual((h["kind"], h["group"], h["alive"]), ("health", "P", False))

    def test_warning_naming_an_error_is_not_an_error(self):
        self.assertIsNone(parse.parse_line(WARN_TIMEOUT))
        e = parse.parse_line(ERR)
        self.assertEqual(e["kind"], "error")
        self.assertEqual(e["exc"], "ClientConnectionResetError")

    def test_identity_lines(self):
        self.assertEqual(parse.parse_line(BOOT)["tag"], "dkrnfh91bar1dauer09270859")
        self.assertEqual(parse.parse_line(FORM)["model"], "Qwen3.8-Flash-Next")
        a = parse.parse_line(ARGS)
        self.assertEqual((a["kind"], a["tp"], a["pp"]), ("server_args", 1, 3))

    def test_timestamps_are_utc(self):
        e = parse.parse_line(P_RANK0)
        self.assertEqual(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(e["t"])), "2026-09-27 09:20:28")


def _shift(line, t):
    """Re-stamp a fixture line to epoch t (keeps the rest verbatim)."""
    m = parse.RE_PREFIX.match(line)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    tail = ""
    if m.group(8):
        tail = " %s%s" % (m.group(8), m.group(9))
    return "[%s%s] %s" % (stamp, tail, line[m.end():])


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stem = os.path.join(self.tmp.name, "boot_weg2_dkrtest0927_abc_0927_000000")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, group, lines, mode="a"):
        with open("%s.%s.log" % (self.stem, group), mode) as fh:
            for ln in lines:
                fh.write(ln + "\n")

    def test_pipeline_rate_is_the_slowest_stage(self):
        now = time.time()
        self._write("front", [BOOT, FORM])
        self._write("P", [ARGS, _shift(P_RANK0, now - 5), _shift(P_RANK1, now - 5), _shift(P_BATCH, now - 5)])
        # two Decode batch lines: the boot's first one has no previous line, its
        # interval (and so its rate) is unknown -- mark_gen_artefacts drops it
        self._write("D", [_shift(DEC, now - 4), _shift(DEC, now - 3), _shift(DEC_RANK, now - 3)])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        [v] = ll.snapshot()
        self.assertTrue(v["live"])
        self.assertEqual(v["meta"]["tag"], "dkrnfh91bar1dauer09270859")
        p = v["prefill"]["P"]["now"]
        # PP0: 16384 / 3.2555 s = 5033 tok/s, PP1: 16384 / 2.73 s = 6001 -> PP0 bounds the pipeline
        self.assertAlmostEqual(p["tps"], 16384 / 3.2555, places=0)
        self.assertAlmostEqual(p["ranks"]["PP1"]["tps"], 16384 / 2.730, places=0)
        self.assertAlmostEqual(p["wall_confounded_tps"], 4898.99)
        d = v["decode"]["D"]
        self.assertAlmostEqual(d["gen_tps"], 123.73)
        self.assertIn("one_s", d)
        self.assertIn("one_s", v["prefill"]["P"])
        t = v["totals"]
        self.assertAlmostEqual(t["p_rate_gpu"], 16384 / 3.2555, places=0)   # since boot, slowest rank, GPU time
        self.assertIsNotNone(t["boot_wall_s"])
        # compute-derived: bs 5 * accept 2.70 / 0.1177 s
        self.assertAlmostEqual(d["compute_tps"], 5 * 2.70 / 0.1177, places=1)
        self.assertIn("P_prefill_tps", v["series"])
        vals = [x for x in v["series"]["P_prefill_tps"] if x]
        self.assertAlmostEqual(max(vals), 16384 / 3.2555, places=0)

    def test_idle_window_falls_back_to_last_burst(self):
        old = time.time() - 600
        self._write("P", [_shift(P_RANK0, old), _shift(P_RANK1, old)])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        [v] = ll.snapshot()
        self.assertIsNone(v["prefill"]["P"]["now"]["tps"])
        self.assertAlmostEqual(v["prefill"]["P"]["last_burst"]["tps"], 16384 / 3.2555, places=0)

    def test_tail_reads_only_appended_complete_lines(self):
        now = time.time()
        self._write("D", [_shift(DEC, now)])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        with open(self.stem + ".D.log", "a") as fh:
            fh.write(_shift(DEC, now)[:40])  # partial line: must not be parsed yet
        ll.poll()
        [v] = ll.snapshot()
        self.assertEqual(v["decode"]["D"]["rows"], 1)
        with open(self.stem + ".D.log", "a") as fh:
            fh.write(_shift(DEC, now)[40:] + "\n")
        ll.poll()
        [v] = ll.snapshot()
        self.assertEqual(v["decode"]["D"]["rows"], 2)

    def test_flip_and_health_state(self):
        self._write("front", [FLIP_BEGIN, FLIP_DONE, HEALTH])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        [v] = ll.snapshot()
        self.assertEqual(v["flip_count"], 1)
        self.assertIsNone(v["flip_open"])
        self.assertEqual(v["awake"]["awake"], "D")
        self.assertFalse(v["health"]["P"]["alive"])

    def test_max_rate_empty_is_none(self):
        self.assertIsNone(live.max_rate({}, 0.0, 120.0))

    def test_max_rate_constant_rank(self):
        iv = [(float(i), float(i + 1), 1000) for i in range(10)]
        self.assertAlmostEqual(live.max_rate({"r0": iv}, 0.0, 20.0, w=3.0, rate=live.compute_rate), 1000.0, delta=1)

    def test_max_rate_picks_the_spike(self):
        iv = [(float(i), float(i + 1), 1000) for i in range(20)] + [(20.0, 21.0, 5000)]
        # anchor at the spike's end: compute_rate over the newest 3 s of compute
        # walks back (5000 + 1000 + 1000) tokens in 3 s
        self.assertAlmostEqual(live.max_rate({"r0": iv}, 0.0, 30.0, w=3.0, rate=live.compute_rate),
                               7000.0 / 3.0, delta=1)

    def test_max_rate_anchors_outside_the_window_do_not_count(self):
        iv = [(float(i), float(i + 1), 1000) for i in range(20)] + [(20.0, 21.0, 5000)]
        # t_lo past the spike's end: the spike's anchor falls out of [t_lo, t_hi]
        self.assertIsNone(live.max_rate({"r0": iv}, 22.0, 40.0, w=3.0, rate=live.compute_rate))

    def test_max_rate_class_is_the_slowest_rank(self):
        r0 = [(float(i), float(i + 1), 1000) for i in range(10)]
        r1 = [(float(i), float(i + 1), 500) for i in range(10)]
        self.assertAlmostEqual(live.max_rate({"r0": r0, "r1": r1}, 0.0, 20.0, w=3.0, rate=live.compute_rate),
                               500.0, delta=1)

    def test_max3s_none_for_a_finished_boot(self):
        # no existing finished-boot test for one_s to copy: the guard is the
        # same newest_mtime test as _one_s's, exercised here with an old mtime
        now = time.time()
        self._write("P", [_shift(P_RANK0, now - 5), _shift(P_RANK1, now - 5)])
        old = now - 300.0
        os.utime(self.stem + ".P.log", (old, old))
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        b = next(iter(ll.boots.values()))
        self.assertIsNone(b._max3s("P", "prefill", now))

    def test_cache_view_last_burst_of_a_sleeping_group(self):
        now = int(time.time())
        tb = now - 90                       # the burst ends 30 s before the 60-s window opens
        self._write("front", [BOOT, FORM])
        self._write("P", [_shift(P_BATCH, tb - 30), _shift(P_BATCH, tb),
                          _shift(P_BATCH.replace("#new-token: 16384, #cached-token: 0",
                                                 "#new-token: 4096, #cached-token: 12288"), tb + 1)])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        [v] = ll.snapshot()
        p = v["cache"]["P"]
        self.assertIsNone(p["window"]["hit_share"])          # nothing inside 60 s: P sleeps
        self.assertEqual(p["last_burst_t"], tb + 1)
        lb = p["last_burst"]
        self.assertEqual((lb["new"], lb["cached"], lb["chunks"]), (20480, 12288, 2))
        self.assertAlmostEqual(lb["hit_share"], 12288 / 32768)   # the row at tb-30 lies outside the 20-s burst

    def test_cache_view_without_any_burst(self):
        now = int(time.time())
        self._write("front", [BOOT, FORM, _shift(SV_P, now - 5)])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        [v] = ll.snapshot()
        self.assertIsNone(v["cache"]["served_P"]["last_burst"])
        self.assertIsNone(v["cache"]["served_P"]["last_burst_t"])

    def test_cache_view_prefetch_refused_and_timeout_since_boot(self):
        now = int(time.time())
        self._write("front", [BOOT, FORM])
        self._write("P", [_shift(P_BATCH, now - 5),
                          _shift("[2026-09-27 09:20:28 PP0] #123 PREFETCH REFUSED rid=x reason=cap", now - 4),
                          _shift("[2026-09-27 09:20:28 PP0] #124 PREFETCH TIMEOUT rid=x", now - 3),
                          _shift("[2026-09-27 09:20:28 PP0] #125 PREFETCH LANDED rid=x", now - 2),
                          _shift("[2026-09-27 09:20:28 PP1] #126 PREFETCH REFUSED rid=y", now - 2)])
        ll = live.LiveLogs([os.path.join(self.tmp.name, "boot_*.log")])
        ll.poll()
        [v] = ll.snapshot()
        boot = v["cache"]["P"]["boot"]
        # rank0 only: the PP1 REFUSED line must not count; LANDED counts under its own key
        self.assertEqual((boot["prefetch_refused"], boot["prefetch_timeout"]), (1, 1))
        self.assertEqual(boot["prefetch_landed"], 1)


class LaunchLineTests(unittest.TestCase):
    def test_key_lines_kept_dedup_and_stripped(self):
        lines = [
            BOOT,
            FORM,
            "[2026-09-27T09:32:20Z] WEG2-LAUNCH #1217/#1233 shm residue: none of ours in /dev/shm",
            "[2026-09-27T09:32:20Z] WEG2-LAUNCH X PROVENANCE: X=4096 source=flag",
            "[2026-09-27T09:32:21Z] WEG2-LAUNCH X PROVENANCE: X=4096 source=flag",
        ]
        out = live.launch_lines(lines)
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0].startswith("=== WEG2 BOOT tag="))
        self.assertEqual(out[2], "X PROVENANCE: X=4096 source=flag")


# verbatim, 27B-b1 dkr27breleasedraftbar1w109270932 P log, 2026-09-27
TB_PREFIXED = "[2026-09-27 09:45:58 PP1] Scheduler hit an exception: Traceback (most recent call last):"
W27_UNPREFIXED = ("sglang.srt.managers.pp_admission_congruence.PPWidthDivergenceRefused: #1233 W27 PP WIDTH "
                  "DIVERGENCE REFUSED: received hidden_states with 1024 row(s) for a batch of 512 token(s)")
FI_SPLIT_OFF = ("[2026-09-27 09:33:06 PP0] FI-GRAPH-SPLIT off for this capture: flashinfer '0.7.0', this module "
                "mirrors 0.6.14")


class HealthTests(unittest.TestCase):
    NOW = 1790502000.0

    def _boot(self, **kw):
        b = {"live": True, "health": {}, "stops": [], "last_activity": {}, "last_activity_any": self.NOW - 5,
             "container": {"Names": "htsglang-acc-27b-x", "State": "running", "Status": "Up 5 minutes (healthy)"},
             "front": None, "queue": None}
        b.update(kw)
        return b

    def test_healthy_boot_has_no_state(self):
        self.assertIsNone(health.assess(self._boot(), self.NOW)["state"])

    def test_weg2_health_dead_group(self):
        b = self._boot(health={"P": {"t": self.NOW - 10, "alive": False, "http_ok": False, "streak": 4}})
        a = health.assess(b, self.NOW)
        self.assertEqual(a["state"], "TOT")
        self.assertIn("Gruppe P tot", a["reasons"][0]["text"])

    def test_stale_health_line_does_not_alarm(self):
        b = self._boot(health={"P": {"t": self.NOW - 600, "alive": False, "http_ok": False, "streak": 4}})
        self.assertIsNone(health.assess(b, self.NOW)["state"])

    def test_alive_but_http_down_is_a_hang(self):
        b = self._boot(health={"P": {"t": self.NOW - 10, "alive": True, "http_ok": False, "streak": 2}})
        self.assertEqual(health.assess(b, self.NOW)["state"], "HAENGT")

    def test_named_stop_without_later_activity_is_dead_and_named_wins(self):
        t = self.NOW - 300
        b = self._boot(stops=[{"t": t, "group": "P", "text": "Traceback (most recent call last):", "bare": True},
                              {"t": t, "group": "P", "text": W27_UNPREFIXED, "bare": False}],
                       last_activity={"P": t - 1, "D": self.NOW - 5})
        a = health.assess(b, self.NOW)
        self.assertEqual(a["state"], "TOT")
        self.assertIn("W27", a["reasons"][0]["text"])

    def test_stop_followed_by_activity_is_only_a_warning(self):
        t = self.NOW - 300
        b = self._boot(stops=[{"t": t, "group": "D", "text": "ADMISSION SPLIT x", "bare": False}],
                       last_activity={"D": t + 120})
        self.assertEqual(health.assess(b, self.NOW)["state"], "WARNUNG")

    def test_queue_without_progress_is_a_hang(self):
        b = self._boot(front={"queue": 3, "outstanding": {"P": 1, "D": 0}}, last_activity_any=self.NOW - 125)
        a = health.assess(b, self.NOW)
        self.assertEqual(a["state"], "HAENGT")
        self.assertIn("4 Anfrage(n) warten", a["reasons"][0]["text"])
        self.assertIn("seit 125 s", a["reasons"][0]["text"])

    def test_no_hang_verdict_while_the_log_is_still_being_read(self):
        b = self._boot(front={"queue": 4, "outstanding": {"P": 0, "D": 6}}, last_activity_any=self.NOW - 550,
                       totals={"read_progress": 0.62},
                       container={"Names": "c", "State": "running", "Status": "Up 51 minutes (unhealthy)"})
        a = health.assess(b, self.NOW)
        self.assertEqual(a["state"], "WARNUNG")
        self.assertIn("noch eingelesen", a["reasons"][0]["text"])

    def test_queue_with_recent_progress_is_fine(self):
        b = self._boot(front={"queue": 3, "outstanding": {}}, last_activity_any=self.NOW - 20)
        self.assertIsNone(health.assess(b, self.NOW)["state"])

    def test_docker_unhealthy_without_progress_is_a_hang(self):
        b = self._boot(container={"Names": "c", "State": "running", "Status": "Up 24 minutes (unhealthy)"},
                       last_activity_any=self.NOW - 300)
        a = health.assess(b, self.NOW)
        self.assertEqual(a["state"], "HAENGT")
        self.assertIn("unhealthy", a["reasons"][0]["text"])

    def test_docker_unhealthy_with_progress_is_only_a_contradiction_note(self):
        b = self._boot(container={"Names": "c", "State": "running", "Status": "Up 51 minutes (unhealthy)",
                                  "health_output": "UNHEALTHY: stop pattern 'W27 ' in the P log"},
                       last_activity_any=self.NOW - 5)
        a = health.assess(b, self.NOW)
        self.assertEqual(a["state"], "WARNUNG")
        self.assertIn("widerspricht dem Fortschritt", a["reasons"][0]["text"])
        self.assertIn("W27", a["reasons"][0]["text"])

    def test_finished_boot_not_judged_but_ended_on_named(self):
        t = self.NOW - 3000
        b = self._boot(live=False, container=None,
                       stops=[{"t": t, "group": "P", "text": W27_UNPREFIXED, "bare": False}])
        a = health.assess(b, self.NOW)
        self.assertIsNone(a["state"])
        self.assertIn("W27", a["ended_on"]["text"])


class StopScanTests(unittest.TestCase):
    def test_stop_lines_from_a_p_log(self):
        self.assertTrue(parse.stop_match(TB_PREFIXED))
        # launcher prose that merely NAMES the W27 guard is no stop (27.09. false alarm)
        self.assertFalse(parse.stop_match("[2026-09-27T14:22:10Z] WEG2-LAUNCH W27 PP WIDTH guard armed: a divergence "
                                          "raises PPWidthDivergenceRefused: #1233 at the receiver"))
        self.assertFalse(parse.stop_match("[2026-09-27 14:22:10 PP0] width census W27 ok rows=512"))
        self.assertTrue(parse.stop_match(W27_UNPREFIXED))
        self.assertFalse(parse.stop_match(FI_SPLIT_OFF))
        with tempfile.TemporaryDirectory() as d:
            stem = os.path.join(d, "boot_weg2_x_0927_000000")
            with open(stem + ".P.log", "w") as fh:
                fh.write("\n".join([FI_SPLIT_OFF, P_RANK0, TB_PREFIXED, "  File \"x.py\", line 1", W27_UNPREFIXED]) + "\n")
            ll = live.LiveLogs([os.path.join(d, "boot_*.log")])
            ll.poll()
            [v] = ll.snapshot()
            self.assertEqual(v["stop_count"], 2)
            named = [x for x in v["stops"] if not x["bare"]]
            self.assertEqual(len(named), 1)
            # the unprefixed exception line inherits the traceback's timestamp
            self.assertEqual(time.strftime("%H:%M:%S", time.gmtime(named[0]["t"])), "09:45:58")
            self.assertEqual(v["last_activity"]["P"], parse.parse_line(P_RANK0)["t"])


# verbatim, NF dkrnfh91bar1dauer09270859 (D log / front log), 2026-09-27
LB0 = ("[2026-09-27 09:05:29 TP0] #988 LOADBACK rid=weg2-0-1 prefix moved to 27456, extend_range re-derived to the "
       "parked shape at the mutation (seen=1) kv_applied=1 mamba_restored=1 kv_only=0 anchor_depth=27456 extent=27456")
MB1 = ("[2026-09-27 09:04:40 TP1] MAMBA-HOST-RESUME n=1: anchor accepted at depth=16384 on a HOST-backed state (device "
       "copy evicted); this match triggers load_back. interval=None")
SR0 = ("[2026-09-27 09:04:40 TP0] #1324 STORE READ INCOMPLETE rid=weg2-0-1 delivered=16384 deliverable=27456 "
       "shortfall=11072 site=drain occurrence=1 -- the read TERMINATED holding less than the prefix it asked for")
SV_D = ("[2026-09-27 09:05:39,208] INFO weg2.front: WEG2-SERVED group=D leg=2 rid=weg2-0-6 stream=1 status=200 "
        "prompt_tokens=26012 cached_tokens=25984 completion_tokens=28 uncached=28 verdict=serve priced=True")
SV_P = ("[2026-09-27 09:04:38,917] INFO weg2.front: WEG2-SERVED group=P leg=1 rid=weg2-0-1 prompt_tokens=27512 "
        "cached_tokens=0 wall=39.25s epoch=1")
D_BATCH = ("[2026-09-27 09:18:53 TP0] Prefill batch, #new-seq: 1, #new-token: 4, #cached-token: 52172, full token usage: "
           "0.89, mamba usage: 0.63, #running-req: 5, #queue-req: 0, #pending-token: 0, cuda graph: False, input "
           "throughput (token/s): 0.39")
P_BATCH_PP1 = P_BATCH.replace(" PP0]", " PP1]")
ADMIN = ("[2026-09-27T09:57:55Z] WEG2-LAUNCH WEG2 ADMIN-KEY minted for this boot -> /var/lib/htsglang/arb/weg2/"
         "boot_x.adminkey (mode 0600); both groups get --admin-api-key")


class TotalsAndCacheTests(unittest.TestCase):
    def test_counted_once_per_chunk_and_request(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as d:
            stem = os.path.join(d, "boot_weg2_x_0927_000000")
            with open(stem + ".P.log", "w") as fh:   # the same chunk logged by PP0 and PP1: count once
                fh.write("\n".join([_shift(P_BATCH, now - 5), _shift(P_BATCH_PP1, now - 5)]) + "\n")
            with open(stem + ".D.log", "w") as fh:
                fh.write("\n".join([_shift(D_BATCH, now - 4), _shift(LB0, now - 4), _shift(MB1, now - 4),
                                     _shift(SR0, now - 4), _shift(SR0.replace(" TP0]", " TP1]"), now - 4)]) + "\n")
            with open(stem + ".front.log", "w") as fh:
                fh.write("\n".join([ADMIN, _shift(SV_P, now - 6), _shift(SV_D, now - 3)]) + "\n")
            ll = live.LiveLogs([os.path.join(d, "boot_*.log")])
            ll.poll()
            [v] = ll.snapshot()
        t = v["totals"]
        self.assertEqual((t["p_new"], t["d_new"], t["decoded"]), (16384, 4, 28))
        self.assertEqual(t["read_progress"], 1.0)
        dc = v["cache"]["D"]["boot"]
        self.assertEqual((dc["new"], dc["cached"]), (4, 52172))
        self.assertAlmostEqual(dc["hit_share"], 52172 / 52176)
        self.assertEqual((dc["loadback_n"], dc["loadback_tok"]), (1, 27456))
        self.assertEqual(dc.get("mamba_n", 0), 0)            # the MAMBA line came from TP1: not counted
        self.assertEqual((dc["l3inc_n"], dc["l3inc_delivered"], dc["l3inc_deliverable"]), (1, 16384, 27456))
        self.assertEqual(v["cache"]["D"]["window"]["cached"], 52172)
        sd = v["cache"]["served_D"]["boot"]
        self.assertAlmostEqual(sd["req_hit_share"], 25984 / 26012)
        # the ADMIN-KEY launch line never reaches the view
        self.assertFalse(any("ADMIN" in x for x in v["meta"].get("launch", [])))


class SeriesFillTests(unittest.TestCase):
    TS = [100.0 + 5 * i for i in range(12)]

    def test_hold_between_lines_then_zero(self):
        # values every other bucket -> typical gap 2 buckets, hold <= 4
        vals = [None, 10, None, 12, None, None, None, None, None, None, None, None]
        out = live.fill_series(self.TS, vals, [], 100.0, 5.0)
        self.assertEqual(out[:5], [0.0, 10, 10, 12, 12])       # bucket 0 is after the boot's first line -> 0
        self.assertEqual(out[7], 12)                          # still within 2x the usual interval
        self.assertEqual(out[8], 0.0)                         # beyond -> no work / slept

    def test_flip_ends_the_hold(self):
        vals = [5, 5, None, None, None, None, None, None, None, None, None, None]
        out = live.fill_series(self.TS, vals, [111.0], 100.0, 5.0)
        self.assertEqual(out[2], 0.0)                          # flipped away in bucket 2

    def test_nothing_before_the_boot(self):
        vals = [None] * 6 + [7] + [None] * 5
        out = live.fill_series(self.TS, vals, [], 128.0, 5.0)
        self.assertEqual(out[:5], [None] * 5)

    def test_dead_boot_is_cut_and_power_divides(self):
        ts = self.TS[:4]
        b = {"series": {"t": ts, "D_decode_tps": [100.0, 100.0, 100.0, 100.0]}, "live": True,
             "container": {"State": "running"},
             "alarm": {"state": "TOT", "reasons": [{"level": "dead", "t": 111.0, "text": "x"}]}}
        gs = {"t": [101.0, 106.0, 107.0, 116.0], "power": [[100, 150, 250], [100, 100, 200], [100, 100, 300], [50, 50, 100]]}
        server.finish_series(b, gs, 125.0, 5.0)
        ser = b["series"]
        self.assertEqual(ser["D_decode_tps"], [100.0, 100.0, None, None])
        self.assertEqual(ser["gap_reason"], "GRUPPE TOT")
        self.assertEqual(ser["power_sum_w"], [500.0, 450.0, None, 200.0])
        self.assertAlmostEqual(ser["D_decode_tps_per_w"][0], 0.2)
        self.assertIsNone(ser["D_decode_tps_per_w"][2])


class EnergyTests(unittest.TestCase):
    def test_power_goes_to_the_classes_that_computed_idle_to_none(self):
        book = energy.EnergyBook(None, 5.0)
        now = 1000.0 + 5 * 4 + energy.LAG_S + 0.1         # buckets 1000..1015 closed
        act = [{"P": True, "D": False, "dec": False, "P_tok": 1000, "D_tok": 0, "dec_tok": 0},
               {"P": False, "D": True, "dec": True, "P_tok": 0, "D_tok": 50, "dec_tok": 40},
               {"P": False, "D": False, "dec": False, "P_tok": 0, "D_tok": 0, "dec_tok": 0},
               {"P": True, "D": False, "dec": False, "P_tok": 1000, "D_tok": 0, "dec_tok": 0}]
        pw = [600.0, 400.0, 150.0, None]                  # last bucket: no power sample -> not covered
        book.update("b", 1000.0, lambda s0, n, bs: act[:n], lambda s0, n, bs: pw[:n], now)
        v = book.view("b", 20.0)
        self.assertEqual(v["covered_s"], 15.0)
        self.assertAlmostEqual(v["coverage"], 0.75)
        self.assertEqual(v["cls"]["P"]["j"], 3000.0)       # 600 W x 5 s, P alone
        self.assertEqual(v["cls"]["P"]["tok"], 1000)       # the uncovered bucket's 1000 tokens are NOT counted
        self.assertAlmostEqual(v["cls"]["P"]["j_per_tok"], 3.0)
        self.assertEqual(v["cls"]["D"]["j"], 1000.0)       # 400 W x 5 s split between D-prefill and decode
        self.assertEqual(v["cls"]["dec"]["j"], 1000.0)
        self.assertAlmostEqual(v["cls"]["dec"]["j_per_tok"], 25.0)
        self.assertAlmostEqual(v["cls"]["dec"]["wh_per_1k"], 25.0 * 1000 / 3600)
        self.assertEqual(v["idle_j"], 750.0)
        # a second update accounts nothing twice
        book.update("b", 1000.0, lambda s0, n, bs: act[:n], lambda s0, n, bs: pw[:n], now)
        self.assertEqual(book.view("b", 20.0)["cls"]["P"]["j"], 3000.0)

    def test_power_buckets_mean_of_card_sums(self):
        gs = {"t": [1000.5, 1002.0, 1006.0], "power": [[100, 200, 300], [100, 100, 100], [50, 50, 50]]}
        self.assertEqual(energy.power_buckets(gs, 1000.0, 3, 5.0), [450.0, 150.0, None])


class RedactTests(unittest.TestCase):
    def test_key_lines_dropped_values_cut_door_closed(self):
        from rigdash import redact
        self.assertIsNone(redact.clean("WEG2 ADMIN-KEY minted for this boot -> /x/boot.adminkey (mode 0600)"))
        self.assertIsNone(redact.clean("RPC auth=bearer abcdefghijklmnop"))
        self.assertEqual(redact.clean("failed: api_key=sk-or-v1-abcdef0123456789 x"), "failed: api_key=<entfernt> x")
        self.assertEqual(redact.clean("#new-token: 16384, tokens: 5"), "#new-token: 16384, tokens: 5")
        self.assertNotIn("ADMIN-KEY", redact.guard('{"t": "WEG2 ADMIN-KEY x"}'))


class SourceTests(unittest.TestCase):
    def test_nvsmi_csv(self):
        txt = ("0, NVIDIA GeForce RTX 3080, GPU-5c64, 225.11, 230.00, 19334, 20480, 100, 65, 1710\n"
               "1, NVIDIA GeForce RTX 5090, GPU-31d7, [N/A], 400.00, 31306, 32607, 99, 57, 2902\n")
        c = sources.parse_nvsmi_csv(txt)
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0]["memory.used"], 19334)
        self.assertIsNone(c[1]["power.draw"])

    def test_container_dirs_and_attach(self):
        ins = [{"Name": "/htsglang-acc-nf-h91bar1dauer", "Mounts": [
            {"Destination": "/var/lib/htsglang/evidence",
             "Source": "/spinning/subvol-999-disk-0/spinning/docker-acceptance/nf/evidence"}]}]
        dirs = sources.container_log_dirs(ins, "/spinning/subvol-999-disk-0")
        self.assertEqual(dirs["htsglang-acc-nf-h91bar1dauer"], "/spinning/docker-acceptance/nf/evidence")
        boots = [{"stem": "x", "live": True, "meta": {"dir": "/spinning/docker-acceptance/nf/evidence",
                                                      "tag": "dkrnfh91bar1dauer09270859"}}]
        cont = [{"Names": "htsglang-acc-nf-h91bar1dauer", "State": "running",
                 "evidence_dir": dirs["htsglang-acc-nf-h91bar1dauer"]}]
        server.attach_containers(boots, cont)
        self.assertEqual(boots[0]["container"]["Names"], "htsglang-acc-nf-h91bar1dauer")

    def test_front_matched_by_tag(self):
        fr = {"front:http://127.0.0.1:30030": {"value": {"tag": "abc", "awake": "P"}, "age_s": 1.0}}
        self.assertEqual(sources.front_for_boot(fr, "abc")["awake"], "P")
        self.assertIsNone(sources.front_for_boot(fr, "other"))

    def test_docker_ps_json_lines(self):
        line = "\t".join(["htsglang-acc-27b-x", "htsglang:rc12", "Up 3 minutes (healthy)", "running",
                           "127.0.0.1:31030->30030/tcp", "2026-09-27 11:32:10 +0200 CEST", "3 minutes ago"])
        rows = sources.parse_docker_ps(line + "\n\n")
        self.assertEqual((rows[0]["Names"], rows[0]["State"]), ("htsglang-acc-27b-x", "running"))
        self.assertIn("{{.Names}}\\t{{.Image}}", sources.DOCKER_PS_FORMAT)


class PcieMeanTests(unittest.TestCase):
    """pcie_mean is pure: history in, per-card means out -- no NVML here."""

    def test_empty_history_has_no_values(self):
        self.assertEqual(sources.pcie_mean([], 100.0, 2.0), [])

    def test_all_samples_too_old_report_none(self):
        r = sources.pcie_mean([(90.0, [(1000, 2000)])], 101.0, 2.0)
        self.assertEqual(r[0]["index"], 0)
        self.assertIsNone(r[0]["rx_mb_s"])
        self.assertIsNone(r[0]["tx_mb_s"])
        self.assertEqual(r[0]["n"], 0)

    def test_two_samples_in_window_are_averaged(self):
        hist = [(100.0, [(1000, 2000)]), (101.0, [(3000, 4000)])]
        r = sources.pcie_mean(hist, 101.0, 2.0)
        self.assertAlmostEqual(r[0]["rx_mb_s"], 2.0)   # (1000 + 3000) / 2 KB/s
        self.assertAlmostEqual(r[0]["tx_mb_s"], 3.0)   # (2000 + 4000) / 2 KB/s
        self.assertEqual(r[0]["n"], 2)

    def test_sample_older_than_the_window_does_not_count(self):
        hist = [(98.0, [(1000, 1000)]), (100.5, [(3000, 3000)])]
        r = sources.pcie_mean(hist, 101.0, 2.0)
        self.assertAlmostEqual(r[0]["rx_mb_s"], 3.0)
        self.assertAlmostEqual(r[0]["tx_mb_s"], 3.0)
        self.assertEqual(r[0]["n"], 1)

    def test_kbps_becomes_decimal_mbytes(self):
        r = sources.pcie_mean([(100.0, [(123456, 654321)])], 100.0, 2.0)
        self.assertAlmostEqual(r[0]["rx_mb_s"], 123.456)
        self.assertAlmostEqual(r[0]["tx_mb_s"], 654.321)

    def test_two_cards_are_kept_apart(self):
        hist = [(100.0, [(1000, 1000), (5000, 5000)])]
        r = sources.pcie_mean(hist, 100.0, 2.0)
        self.assertEqual([c["index"] for c in r], [0, 1])
        self.assertAlmostEqual(r[0]["rx_mb_s"], 1.0)
        self.assertAlmostEqual(r[1]["rx_mb_s"], 5.0)


class PhaseTimelineTests(unittest.TestCase):
    """The phase bar: runs per class, flips grey, idle between, nothing guessed."""

    def _pf(self, t, tok, ms, rank=0, rk="PP"):
        return {"t": t, "s": t - ms / 1000.0, "cls": "P", "ev": {"kind": "prefill_rank", "rk": rk, "rank": rank,
                                                            "new_tok": tok, "compute_ms": ms, "t": t}}

    def _dec(self, t, gen=None, cls="dec"):
        return {"t": t, "s": t, "cls": cls, "ev": {"kind": "decode_batch" if gen else "decode_rank", "gen_tps": gen,
                                                   "running": 2 if gen else None, "t": t}}

    def test_decode_rank_line_carries_its_exact_stamp(self):
        ev = parse.parse_line("[2026-09-27 17:35:40 TP0] Decode rank batch, rank: 0, #round: 2862, "
                              "t: 1790530539.968, bs: 3, #rows: 12, #fwd: 1, gpu-ms: 46.2 (split unavailable)")
        self.assertEqual(ev["kind"], "decode_rank")
        self.assertAlmostEqual(ev["t_exact"], 1790530539.968)

    def test_flip_pairs_begin_with_done_of_next_epoch(self):
        fl = live._flip_intervals([{"t": 100.0, "epoch": 7, "sleep": "P", "wake": "D"}],
                                  [{"t": 102.5, "epoch": 8, "slept": "P", "woke": "D", "total_ms": 2512.0}], None, 200)
        self.assertEqual((fl[0]["b"], fl[0]["d"]), (100.0, 102.5))
        # no begin in memory: the start is done - flip_total
        fl = live._flip_intervals([], [{"t": 102.5, "epoch": 8, "slept": "P", "woke": "D", "total_ms": 2000.0}], None, 200)
        self.assertAlmostEqual(fl[0]["b"], 100.5)

    def test_prefill_flip_decode_sequence(self):
        acts = [self._pf(10, 1000, 1000), self._pf(12, 1000, 1000), self._pf(14, 1000, 2000),
                self._dec(25.0), self._dec(26.0, gen=100.0), self._dec(28.0, gen=120.0)]
        flips = [{"b": 15.0, "d": 17.0, "slept": "P", "woke": "D", "total_ms": 2000.0, "drain_ms": 100.0, "open": False}]
        segs = live.phase_timeline(acts, flips, 0.0, 30.0, first_t=5.0, work_from=9.0)
        self.assertEqual([x["k"] for x in segs], ["idle", "P", "idle", "flip", "idle", "dec"])
        p = segs[1]
        self.assertEqual((p["s"], p["e"]), (9.0, 14.0))          # first chunk's own gpu-ms, not a guess
        self.assertAlmostEqual(p["tps"], 3000 / 4.0)             # sum tok / sum compute-ms
        self.assertEqual((segs[3]["s"], segs[3]["e"]), (15.0, 17.0))
        self.assertEqual(segs[4]["awake"], "D")                  # after the flip D is awake, no work yet
        self.assertEqual(segs[5]["s"], 24.0)                    # >gap after the flip: 1 s before its first line
        self.assertAlmostEqual(segs[5]["tps"], 110.0)           # mean gen throughput of the run
        self.assertTrue(segs[5]["running"])
        self.assertEqual(segs[5]["e"], 30.0)
        self.assertEqual(segs[0]["s"], 5.0)                     # nothing before the boot's first line
        self.assertTrue(segs[0].get("boot"))                    # before the first work line: loading, not waiting
        self.assertFalse(segs[4].get("boot"))

    def test_short_gap_after_a_flip_is_filled(self):
        acts = [self._dec(20.0, gen=100.0), self._dec(21.0, gen=100.0)]
        flips = [{"b": 15.0, "d": 17.0, "slept": "P", "woke": "D", "total_ms": 2000.0, "drain_ms": 1.0, "open": False}]
        segs = live.phase_timeline(acts, flips, 10.0, 40.0)
        self.assertEqual([x["k"] for x in segs], ["idle", "flip", "dec", "idle"])
        self.assertEqual(segs[2]["s"], 17.0)

    def test_drain_work_inside_a_flip_is_work_not_grey(self):
        acts = [self._dec(10.0, gen=90.0), self._dec(40.0, gen=80.0)]
        flips = [{"b": 5.0, "d": 42.0, "slept": "D", "woke": "P", "total_ms": 37000.0, "drain_ms": 35000.0, "open": False}]
        segs = live.phase_timeline(acts, flips, 0.0, 50.0)
        grey = [x for x in segs if x["k"] == "flip"][0]
        self.assertEqual(grey["s"], 40.0)                       # grey only after the old group's last line
        self.assertEqual(sum(1 for x in segs if x["k"] == "dec"), 2)   # a 30-s silence splits the run

    def test_running_phase_reaches_now_and_short_gaps_are_filled(self):
        acts = [self._dec(10.0, gen=100.0), self._pf(13.0, 100, 500, rk="TP"), self._dec(16.0, gen=100.0)]
        for a in acts[1:2]:
            a["cls"] = "D"
        segs = live.phase_timeline(acts, [], 0.0, 18.0, first_t=0.0)
        ks = [x["k"] for x in segs]
        self.assertEqual(ks, ["idle", "dec", "D", "dec"])
        self.assertEqual(segs[2]["s"], 10.0)                    # D-Prefill fills the 2.5-s gap after decode
        self.assertEqual(segs[3]["s"], 13.0)
        self.assertTrue(segs[3]["running"])
        self.assertEqual(segs[3]["e"], 18.0)

    def test_open_flip_is_grey_until_now(self):
        fl = live._flip_intervals([], [], {"t": 20.0, "epoch": 3, "sleep": "D", "wake": "P"}, 25.0)
        segs = live.phase_timeline([self._dec(19.0, gen=50.0)], fl, 0.0, 25.0, first_t=0.0)
        self.assertEqual(segs[-1]["k"], "flip")
        self.assertTrue(segs[-1]["open"])
        self.assertEqual((segs[-1]["s"], segs[-1]["e"]), (20.0, 25.0))


# verbatim, /spinning/docker-acceptance/nf/abnahme_cu130.log and 27b/abnahme_cu130.log, 2026-09-27
H_DAUER_STOP = "[nf-dauer 2026-09-27T17:39:28Z] Stop-Datei -> Hold-Ende, Schleife endet"
H_HOLD_STOP = "[host-acc 17:39:36Z] AGENT-HOLD (h91dprbar1dauer) Ende: Stop-Datei"
H_HOLD_DEAD = "[host-acc 15:39:54Z] AGENT-HOLD (releasedraftbar1w1) Ende: Container-tot"
H_HOLD_TIME = "[host-acc 11:02:10Z] AGENT-HOLD (h91dprbar1dauer) Ende: Zeit"
H_DEADMAN = ("[nf-rc11b 16:28:55Z] DEADMAN-VERDICT deadman_dkrnfh91dprbar1dauer09271603_P: DEADMAN[CRASH] "
             "2026-09-27T16:28:41+00:00 port=30031 no process matches pattern")
H_DEADMAN_KICK = "[host-acc 16:29:10Z] AGENT-HOLD (h91dprbar1dauer) Ende: Stop-Datei"
H_ARMED = ("[nf-rc11b 17:47:34Z] DEADMAN-ARMED deadman_dkrnfh91dprbar1dauer09271741_D: DEADMAN armed "
           "2026-09-27T17:47:28+00:00 log=/var/lib/htsglang/evidence/boot_weg2_x")
STEM_S = "boot_weg2_dkrnfh91dprbar1dauer09271719_e39b37d011_0927_171931"
STEM_603 = "boot_weg2_dkrnfh91dprbar1dauer09271603_21d7e3b188_0927_160335"
STEM_27B = "boot_weg2_dkr27breleasedraftbar1w109271525_21d7e3b188_0927_152512"


def _utc(hh, mm, ss):
    import calendar
    return float(calendar.timegm((2026, 9, 27, hh, mm, ss)))


class PlannedStopTests(unittest.TestCase):
    """Operator 27.09. ~18Z: a planned stop is grey, a death stays red."""

    def _markers(self, *lines):
        return [m for m in (stops.parse_marker(x) for x in lines) if m]

    def test_parse_all_marker_kinds(self):
        self.assertEqual(stops.parse_marker(H_DAUER_STOP)["kind"], "planned")
        self.assertEqual(stops.parse_marker(H_DAUER_STOP)["t"], _utc(17, 39, 28))
        m = stops.parse_marker(H_HOLD_STOP)
        self.assertEqual((m["kind"], m["tag"], m["sod"]), ("planned", "h91dprbar1dauer", 17 * 3600 + 39 * 60 + 36))
        self.assertEqual(stops.parse_marker(H_HOLD_TIME)["kind"], "planned")
        self.assertEqual(stops.parse_marker(H_HOLD_DEAD)["kind"], "death")
        d = stops.parse_marker(H_DEADMAN)
        self.assertEqual((d["kind"], d["ident"], d["t"]), ("death", "dkrnfh91dprbar1dauer09271603", _utc(16, 28, 41)))
        self.assertIsNone(stops.parse_marker(H_ARMED))          # armed is not a verdict

    def test_planned_stop_of_rc12s(self):
        mk = self._markers(H_DAUER_STOP, H_HOLD_STOP)
        e = stops.classify(STEM_S, _utc(17, 19, 31), _utc(17, 40, 5), mk)
        self.assertIsNone(e["death"])
        self.assertEqual(e["planned"]["t"], _utc(17, 39, 28))

    def test_death_wins_over_the_kick_by_stop_file(self):
        # the loop kicks the hold with the stop file after a deadman verdict
        mk = self._markers(H_DEADMAN, H_DEADMAN_KICK)
        e = stops.classify(STEM_603, _utc(16, 3, 35), _utc(16, 28, 45), mk)
        self.assertIsNone(e["planned"])
        self.assertEqual(e["death"]["t"], _utc(16, 28, 41))

    def test_markers_of_other_boots_do_not_match(self):
        mk = self._markers(H_DAUER_STOP, H_HOLD_STOP, H_DEADMAN)
        e = stops.classify(STEM_603, _utc(16, 3, 35), _utc(16, 28, 45), self._markers(H_DAUER_STOP, H_HOLD_STOP))
        self.assertIsNone(e["planned"])                         # 17:39 is > 10 min after that boot's end
        e = stops.classify(STEM_S, _utc(17, 19, 31), _utc(17, 40, 5), mk)
        self.assertIsNone(e["death"])                           # the 16:28 deadman names another boot
        e = stops.classify(STEM_27B, _utc(15, 25, 12), _utc(15, 39, 50), self._markers(H_HOLD_DEAD, H_HOLD_STOP))
        self.assertEqual(e["death"]["t"], _utc(15, 39, 54))     # 27B hold end "Container-tot"
        self.assertIsNone(e["planned"])

    NOW = _utc(17, 41, 0)

    def _boot(self, end, **kw):
        b = {"live": True, "stops": [], "last_activity": {"D": self.NOW - 70}, "last_activity_any": self.NOW - 70,
             "container": {"Names": "htsglang-acc-nf-x", "State": "running", "Status": "Up 20 minutes (healthy)"},
             "front": None, "queue": {"t": self.NOW - 30, "queue": 4},
             "health": {"D": {"t": _utc(17, 40, 2), "alive": False, "http_ok": False, "streak": 1}}, "end": end}
        b.update(kw)
        return b

    def test_teardown_after_planned_stop_is_grey_not_tot(self):
        end = {"planned": {"t": _utc(17, 39, 28), "text": H_DAUER_STOP, "src": "nf-dauer"}, "death": None}
        a = health.assess(self._boot(end), self.NOW)
        self.assertEqual(a["state"], "GESTOPPT")
        self.assertEqual(a["reasons"], [])
        self.assertEqual(a["planned_stop"]["teardown_t"], _utc(17, 40, 2))
        # the same signs without the marker: TOT, as before
        self.assertEqual(health.assess(self._boot({"planned": None, "death": None}), self.NOW)["state"], "TOT")

    def test_death_before_the_planned_stop_stays_red(self):
        end = {"planned": {"t": _utc(17, 40, 30), "text": H_DAUER_STOP, "src": "nf-dauer"}, "death": None}
        self.assertEqual(health.assess(self._boot(end), self.NOW)["state"], "TOT")

    def test_named_stop_before_planned_stop_stays_red(self):
        end = {"planned": {"t": _utc(17, 40, 50), "text": H_DAUER_STOP, "src": "nf-dauer"}, "death": None}
        b = self._boot(end, health={}, stops=[{"t": _utc(17, 38, 0), "group": "P", "text": W27_UNPREFIXED, "bare": False}],
                       last_activity={"P": _utc(17, 37, 0)})
        self.assertEqual(health.assess(b, self.NOW)["state"], "TOT")

    def test_stop_requested_while_serving_changes_nothing(self):
        end = {"planned": {"t": self.NOW - 200, "text": H_DAUER_STOP, "src": "nf-dauer"}, "death": None}
        b = self._boot(end, health={}, last_activity_any=self.NOW - 2, queue=None)
        a = health.assess(b, self.NOW)
        self.assertIsNone(a["state"])
        self.assertFalse(a["planned_stop"]["stopping"])

    def test_finish_series_grey_cut_for_planned_stop(self):
        end = {"planned": {"t": _utc(17, 39, 28), "text": H_DAUER_STOP, "src": "nf-dauer"}, "death": None}
        b = self._boot(end)
        b["alarm"] = health.assess(b, self.NOW)
        b["last_log_t"] = self.NOW - 5
        ts = [self.NOW - 60 + 5 * i for i in range(12)]
        b["series"] = {"t": ts, "D_decode_tps": [100.0] * 12}
        b["timeline"] = {"segs": [{"k": "dec", "s": ts[0], "e": self.NOW}]}
        server.finish_series(b, None, self.NOW, 5.0)
        self.assertEqual((b["series"]["gap_kind"], b["series"]["gap_reason"]), ("planned", "gestoppt (geplant)"))
        self.assertEqual(b["series"]["gap_from"], _utc(17, 40, 2))
        self.assertEqual(b["timeline"]["cut_kind"], "planned")


# verbatim, NF rc12s P log boot_weg2_dkrnfh91dprbar1dauer09271719 (TP1-PP3), a P phase, tails cut after "wait 0.0)"
PP_CHUNKS = [
    '[2026-09-27 17:31:27 PP0] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 3260.0 (compute 3260.0, wait 0.0)',
    '[2026-09-27 17:31:27 PP1] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 2732.7 (compute 2732.7, wait 0.0)',
    '[2026-09-27 17:31:30 PP2] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 1561.6 (compute 1561.6, wait 0.0)',
    '[2026-09-27 17:31:30 PP1] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 2782.7 (compute 2782.7, wait 0.0)',
    '[2026-09-27 17:31:30 PP0] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 3283.2 (compute 3283.1, wait 0.0)',
    '[2026-09-27 17:31:33 PP2] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 1571.8 (compute 1571.8, wait 0.0)',
    '[2026-09-27 17:31:34 PP1] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 2797.0 (compute 2796.9, wait 0.0)',
    '[2026-09-27 17:31:34 PP0] Prefill rank batch, #new-token: 16378, #cached-token: 0, #chunks: 1, gpu-ms: 3553.5 (compute 3553.5, wait 0.0)',
    '[2026-09-27 17:31:37 PP2] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 1578.2 (compute 1578.2, wait 0.0)',
    '[2026-09-27 17:31:37 PP0] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 3290.4 (compute 3290.3, wait 0.0)',
    '[2026-09-27 17:31:37 PP1] Prefill rank batch, #new-token: 16378, #cached-token: 0, #chunks: 1, gpu-ms: 2768.4 (compute 2768.4, wait 0.0)',
    '[2026-09-27 17:31:40 PP2] Prefill rank batch, #new-token: 16378, #cached-token: 0, #chunks: 1, gpu-ms: 1573.1 (compute 1573.1, wait 0.0)',
    '[2026-09-27 17:31:40 PP1] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 2749.3 (compute 2749.3, wait 0.0)',
    '[2026-09-27 17:31:40 PP0] Prefill rank batch, #new-token: 16384, #cached-token: 0, #chunks: 1, gpu-ms: 3330.3 (compute 3330.2, wait 0.0)',
]


class OneSecondRateTests(unittest.TestCase):
    """User 27.09.: 'es zeigt meistens null an und springt dann manchmal auf
    5000 und manchmal auf 10000 -- dabei sollte es eher durchgehend etwas
    zwischen 4700-5000 anzeigen'."""

    def _replay(self, lines, group, t_from, t_to, step=0.5, kind="prefill"):
        evs = []
        for ln in lines:
            ev = parse.parse_line(ln)
            vis = ev["t"] + 1.0 if ev["t"] == int(ev["t"]) else ev["t"]   # a whole-second line is complete at its end
            evs.append((vis, ev))
        evs.sort(key=lambda x: x[0])
        b = live.Boot("x", "/tmp")
        out, i, now = [], 0, t_from
        while now <= t_to:
            while i < len(evs) and evs[i][0] <= now:
                b._ingest(group, evs[i][1])
                i += 1
            out.append((now, b._one_s(group, kind, now)))
            now += step
        return out

    def test_spread_rate_shares(self):
        self.assertAlmostEqual(live.spread_rate([(0.0, 4.0, 4000)], 3.0), 1000.0)
        self.assertAlmostEqual(live.spread_rate([(0.0, 2.0, 2000), (2.0, 3.0, 3000)], 3.0), 3000.0)
        self.assertAlmostEqual(live.spread_rate([(0.0, 2.0, 2000), (2.0, 3.0, 3000)], 2.5), 0.5 * 1000 + 0.5 * 3000)

    def test_compute_rate_walks_back_one_second_of_compute(self):
        # 27B form: 512-token micro-chunks, 220 ms each, several per whole-second stamp
        iv = [(10.28, 10.5, 512), (10.28, 10.5, 512), (11.28, 11.5, 512), (11.28, 11.5, 512), (11.28, 11.5, 512)]
        self.assertAlmostEqual(live.compute_rate(iv), 512 / 0.22, delta=1)
        self.assertAlmostEqual(live.compute_rate([(0.0, 3.3, 16384)]), 16384 / 3.3, delta=1)

    def test_prefill_chunks_give_a_steady_rate_without_zeros(self):
        t0 = parse.parse_line(PP_CHUNKS[0])["t"]
        vals = [v for _, v in self._replay(PP_CHUNKS, "P", t0 + 1.0, t0 + 14.0)]
        self.assertTrue(all(4500 <= v <= 5200 for v in vals), vals)   # slowest stage PP0, ~16384 / 3.3 s

    def test_inactive_after_the_activity_limit_and_after_a_flip(self):
        t0 = parse.parse_line(PP_CHUNKS[0])["t"]
        last = max(parse.parse_line(x)["t"] for x in PP_CHUNKS)
        vals = self._replay(PP_CHUNKS, "P", last + 1.0, last + 12.0, step=1.0)
        self.assertGreater(vals[0][1], 4500)
        self.assertEqual(vals[-1][1], 0.0)          # 3.3-s chunks: no line for > 1.5 x 3.3 + 1 s -> stopped
        b = live.Boot("x", "/tmp")
        for ln in PP_CHUNKS:
            b._ingest("P", parse.parse_line(ln))
        b._ingest("front", {"t": last + 0.2, "kind": "flip_begin", "epoch": 3, "sleep": "P", "wake": "D"})
        self.assertEqual(b._one_s("P", "prefill", last + 1.5), 0.0)

    def test_decode_rounds_carry_the_gen_throughput(self):
        base = 1790530380.0
        lines = []
        for k in range(80):                             # one round every 62.5 ms, bs 3
            t = base + k * 0.0625
            lines.append("[%s TP0] Decode rank batch, rank: 0, #round: %d, t: %.3f, bs: 3, #rows: 12, #fwd: 1, gpu-ms: 50.0 (x)"
                         % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)), 100 + k, t))
        for s_, gen in ((0, 100.0), (3, 160.0)):
            lines.append("[%s TP0] Decode batch, #running-req: 3, #full token: 1000, full token usage: 0.10, "
                         "accept len: 2.30, accept rate: 0.40, cuda graph: True, gen throughput (token/s): %.2f, #queue-req: 0"
                         % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(base + s_)), gen))
        vals = [v for _, v in self._replay(lines, "D", base + 4.0, base + 5.0, step=0.25, kind="decode")]
        self.assertTrue(all(145 <= v <= 175 for v in vals), vals)   # gen throughput 160, not bs x accept len x 16 = 110


# verbatim, NF D log boot_weg2_dkrnfh91dprbar1dauer09271756, 2026-09-27
DSEATS = ("[2026-09-27 18:06:42 TP0] WEG2 D-PHASE-SEATS (H95) epoch=1790532154.3 handoff_n=6 parked_n=4 -> n=6 of cap 6 "
          "(CLAMPED: the front handed more than --d-bs): decode batch bs6, GDN slots in use <= 38 of 38 (boot)")
SCHEDCAP = ("[2026-09-27 18:02:25 TP0] max_total_num_tokens=262144, chunked_prefill_size=4096, max_prefill_tokens=16384, "
            "max_running_requests=6, context_len=262144, available_gpu_mem=3.65 GB")
DEC_BATCH_Q = ("[2026-09-27 18:07:00 TP0] Decode batch, #running-req: 4, #full token: 179392, full token usage: 0.68, mamba num: 16, "
               "mamba usage: 0.42, accept len: 2.34, accept rate: 0.45, cuda graph: True, gen throughput (token/s): 110.45, #queue-req: 2")


class DecodeSeatsTests(unittest.TestCase):
    """User 27.09.: next to the 1-s decode rate, how many seats (bs) REALLY compute."""

    def test_parse_seats_and_cap(self):
        e = parse.parse_line(DSEATS)
        self.assertEqual((e["kind"], e["handoff_n"], e["parked_n"], e["n"], e["cap"], e["clamped"]),
                         ("d_seats", 6, 4, 6, 6, True))
        c = parse.parse_line(SCHEDCAP)
        self.assertEqual((c["kind"], c["max_running"]), ("sched_cap", 6))

    def test_decode_view_running_bs_seats_and_queue(self):
        b = live.Boot("x", "/tmp")
        for ln in (SCHEDCAP, DSEATS, DEC_BATCH_Q):
            b._ingest("D", parse.parse_line(ln))
        for k, bs in enumerate((6, 5, 4)):
            b._ingest("D", parse.parse_line(
                "[2026-09-27 18:07:01 TP0] Decode rank batch, rank: 0, #round: %d, t: %.3f, bs: %d, #rows: 16, #fwd: 1, "
                "gpu-ms: 50.0 (x)" % (10 + k, 1790532421.0 + 0.06 * k, bs)))
        v = b._decode_view("D", 1790532421.5)
        self.assertEqual(v["round_bs"]["bs"], 4)                 # the newest round, not the batch line's 4 by chance
        self.assertEqual((v["seats"]["n"], v["seats"]["cap"]), (6, 6))
        self.assertEqual((v["queue_req"], v["max_running"]), (2, 6))

    def test_phase_run_carries_bs_range_of_its_rounds(self):
        evs = [{"kind": "decode_rank", "bs": b_} for b_ in (6, 6, 5, 4)] + [{"kind": "decode_batch", "gen_tps": 100.0, "running": 5}]
        st = live._run_stats("dec", evs)
        self.assertEqual((st["bs_min"], st["bs_max"]), (4, 6))
        self.assertAlmostEqual(st["bs_mean"], 5.25)


# verbatim, NF rc12z20 boot_weg2_dkrnfh91dprsavisnoadoptstbar1dauer09281220_3a86888ba5, 2026-09-28:
# D decoded bs1 until 12:34:19, flipped away and back (12:34:19.9 .. 12:34:47.2), extended six
# requests, and its first Decode batch line after that carried the pause in its denominator
ART_D = [
    "[2026-09-28 12:34:18 TP0] Decode batch, #running-req: 1, #full token: 333184, full token usage: 0.64, mamba num: 4, "
    "mamba usage: 0.21, accept len: 2.70, accept rate: 0.57, cuda graph: True, gen throughput (token/s): 88.01, #queue-req: 0",
    "[2026-09-28 12:34:19 TP0] Decode batch, #running-req: 1, #full token: 333248, full token usage: 0.64, mamba num: 4, "
    "mamba usage: 0.21, accept len: 2.62, accept rate: 0.54, cuda graph: True, gen throughput (token/s): 93.22, #queue-req: 0",
    "[2026-09-28 12:34:55 TP0] Prefill rank batch, #new-token: 164, #cached-token: 18560, #chunks: 1, gpu-ms: 1670.7 "
    "(compute 1623.6, wait 47.1) (wait by family: tp.all_reduce 47.1/96x, ple.wait 0.0/1x)",
    "[2026-09-28 12:34:57 TP0] Decode batch, #running-req: 6, #full token: 387840, full token usage: 0.74, mamba num: 24, "
    "mamba usage: 0.63, accept len: 2.16, accept rate: 0.39, cuda graph: True, gen throughput (token/s): 13.65, #queue-req: 0",
    "[2026-09-28 12:35:01 TP0] Decode batch, #running-req: 5, #full token: 387776, full token usage: 0.74, mamba num: 16, "
    "mamba usage: 0.42, accept len: 2.58, accept rate: 0.53, cuda graph: True, gen throughput (token/s): 133.47, #queue-req: 0",
]
ART_FRONT = [
    "[2026-09-28 12:34:44,088] INFO weg2.front: WEG2-FLIP begin epoch=11 sleep=P wake=D outstanding=0 queue=0",
]


class GenArtefactTests(unittest.TestCase):
    """User 28.09.: 'messartefakte entfernen. unsinnige werte im dashboard nicht anzeigen'."""

    def _boot(self, with_front=True):
        b = live.Boot("x", "/tmp")
        for ln in ART_D:
            b._ingest("D", parse.parse_line(ln))
        if with_front:
            for ln in ART_FRONT:
                b._ingest("front", parse.parse_line(ln))
        return b

    def test_line_after_flip_and_extend_is_no_rate(self):
        b = self._boot()
        b._mark_gen()
        got = [(e["gen_tps"], e["gen_art"]) for e in b.ev["D_decode_batch"]]
        self.assertEqual(got, [(None, "first"), (93.22, None), (None, "pause"), (133.47, None)])
        self.assertEqual(b.ev["D_decode_batch"][2]["gen_tps_raw"], 13.65)   # tokens stay: raw kept

    def test_gap_alone_marks_it(self):
        b = self._boot(with_front=False)
        b.ev["D_prefill_rank"].clear()
        b._mark_gen()
        self.assertEqual(b.ev["D_decode_batch"][2]["gen_art"], "gap")          # 38 s after a 1-s rhythm

    def test_views_never_show_the_artefact(self):
        b = self._boot()
        t = parse.parse_line(ART_D[-1])["t"]
        v = b.view(t + 1.0, with_series=True)
        d = v["decode"]["D"]
        self.assertEqual(d["gen_tps_last"], 133.47)
        self.assertNotIn(13.65, [x for x in v["series"]["D_decode_tps"] if x])
        self.assertTrue(all((s.get("tps") or 100) > 70 for s in v["timeline"]["segs"] if s["k"] == "dec"))

    def test_idempotent(self):
        b = self._boot()
        b._mark_gen()
        b._mark_gen()
        self.assertEqual([e["gen_tps"] for e in b.ev["D_decode_batch"]], [None, 93.22, None, 133.47])


class WachOhneArbeitTests(unittest.TestCase):
    """WACH-OHNE-ARBEIT-0929: P/D-Arbeit mit ihrer echten Zeit, 'Flip-Nachlauf D' statt 'wach, keine Arbeit'."""

    @staticmethod
    def _stamp(t):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))

    def _line(self, b, group, t, rank, text):
        b._ingest(group, parse.parse_line("[%s %s] %s" % (self._stamp(t), rank, text)))

    def test_p_rank_line_gets_the_forward_it_reports_even_when_the_timing_comes_later(self):
        base = 1790665270.0
        b = live.Boot("x", "/tmp")
        # forward 11 starts at base+3.571: the flush of forward 10 runs at its head layer
        self._line(b, "P", base + 3, "PP0", "TIMING-FLUSH-WAIT instrument=attn forward=10 mode=event wait_ms=0.0 "
                   "events=5 t_unix_ms=%d (x)" % int((base + 3.571) * 1000))
        self._line(b, "P", base + 3, "PP0", "TIMING-FLUSH-WAIT instrument=moe_prefill forward=10 mode=event "
                   "wait_ms=0.0 events=5 t_unix_ms=%d (x)" % int((base + 3.589) * 1000))
        # the rank line of forward 11 comes after the pipeline, 07:01:16-style ...
        self._line(b, "P", base + 6, "PP0", "Prefill rank batch, #new-token: 1440, #cached-token: 21568, #chunks: 1, "
                   "gpu-ms: 1795.0 (compute 1795.0, wait 0.0) bubble_ms=1.0 (x)")
        r = b.ev["P_prefill_rank"][-1]
        self.assertIsNone(r.get("e_exact"))              # no timing yet: never guessed
        # ... and forward 11's own timing only at the head of forward 12, 46 s later
        self._line(b, "P", base + 52, "PP0", "FWD-TIMING-PREFILL forward=11 tokens=1440 layers=29 embed_ms=0.5 "
                   "total_ms=1795.0 marks=3")
        self.assertAlmostEqual(r["s_exact"], base + 3.571, places=3)
        self.assertAlmostEqual(r["e_exact"], base + 3.571 + 1.795, places=3)
        tl = b.timeline(base + 60, span=120)
        p = [s for s in tl["segs"] if s["k"] == "P"]
        self.assertEqual(len(p), 1)
        self.assertAlmostEqual(p[0]["s"], base + 3.57, places=1)

    def test_d_extend_is_drawn_from_host_anon_pass_not_the_pass_late_rank_line(self):
        base = 1790666370.0
        b = live.Boot("x", "/tmp")
        self._line(b, "D", base, "TP0", "HOST-ANON-PASS pass=9 phase=EXTEND tokens=460 anon_begin=1MiB anon_end=1MiB "
                   "peak=1MiB (x) wall_ms=2500 (instrument)")
        self._line(b, "D", base + 1, "TP0", "HOST-ANON-PASS pass=10 phase=EXTEND tokens=460 anon_begin=1MiB "
                   "anon_end=1MiB peak=1MiB (x) wall_ms=20 (empty follow-up)")
        self._line(b, "D", base + 3, "TP0", "Prefill rank batch, #new-token: 460, #cached-token: 0, #chunks: 1, "
                   "gpu-ms: 2400.0 (compute 2300.0, wait 100.0) (x)")
        r = b.ev["D_prefill_rank"][-1]
        self.assertEqual(r["exact_src"], "anon")
        self.assertAlmostEqual(r["e_exact"], base + 0.5)
        self.assertAlmostEqual(r["s_exact"], base + 0.5 - 2.5)

    def test_flip_tail_replaces_awake_idle_between_flip_done_and_first_decode(self):
        base = 1790667000.0
        b = live.Boot("x", "/tmp")
        FlipTimeTests()._front(b, base, "WEG2-FLIP begin epoch=2 sleep=P wake=D outstanding=0 queue=1")
        FlipTimeTests()._front(b, base + 2.0, "WEG2-FLIP done epoch=3 slept=P woke=D drain+quiesce=100 ms "
                               "sleep=1000 ms wake=1000 ms flip_total=2000 ms weights_tags=17")
        self._line(b, "D", base + 4, "TP0", "WEG2-POST-WAKE-PASS n=0 mode=EXTEND bs=5 gap_ms=-1 schedule_ms=1400 "
                   "run_ms=2500 prefetch_ms=7 prepare_ms=1300 ready_ms=1 (x)")
        for k in range(5):
            FlipTimeTests()._decode(b, base + 7.2 + k * 0.05)
        tails = b.flip_tails(base - 10, base + 20)
        self.assertEqual(len(tails), 1)
        t = tails[0]
        self.assertAlmostEqual(t["s"], base + 2.0, places=2)
        self.assertEqual(t["ms"], 5200)
        self.assertEqual(t["parts"]["prepare_ms"], 1300)
        self.assertEqual(t["parts"]["run_ms"], 2500)
        segs = b.timeline(base + 8, span=60)["segs"]
        kinds = [s["k"] for s in segs if s["s"] >= base + 1.9]
        self.assertIn("flip_tail", kinds)
        self.assertFalse(any(s["k"] == "idle" and s.get("awake") == "D" and base + 2.1 < s["s"] < base + 7
                             for s in segs))

    def test_phase_timeline_splits_idle_by_tails_only(self):
        acts = [{"t": 10.0, "s": 9.0, "cls": "dec", "ev": {"kind": "decode_rank"}}]
        segs = live.phase_timeline(acts, [], 0.0, 30.0, tails=[{"s": 15.0, "e": 20.0, "ms": 5000, "parts": None}])
        ks = [(s["k"], s["s"], s["e"]) for s in segs]
        self.assertIn(("flip_tail", 15.0, 20.0), ks)
        self.assertTrue(any(k == "idle" and s == 10.0 and e == 15.0 for k, s, e in ks))
        self.assertTrue(any(k == "idle" and s == 20.0 and e == 30.0 for k, s, e in ks))


if __name__ == "__main__":
    unittest.main()


def _boot(stem, tree, sha, live_=False, last=0.0, container=None):
    b = {"stem": stem, "live": live_, "last_log_t": last,
         "meta": {"stem": stem, "tree": tree, "sha": sha, "tag": stem.split("_")[2]}}
    if container:
        b["container"] = container
    return b


IMAGES = {
    "f833fcbb2d": {"rc": "rc12z29b", "base": "rc12z29", "built_utc": "2026-09-28T19:21Z", "changes": [
        {"id": "S3f", "who": "NF", "title": "Owner-Zeilen", "expected": "-st-cut bootet",
         "status": "belegt", "evidence": "rc12z29b serving"},
        {"id": "X", "who": "NF", "title": "t", "expected": "e", "status": "geraten", "evidence": ""}]},
    "70ac86e2bd": {"rc": "rc12z30b", "changes": []},
}


class FlipTimeTests(unittest.TestCase):
    """Flipzeit (Nutzer 29.09.) = WEG2-FLIP begin -> erstes Decode-Token, nicht flip_total."""

    @staticmethod
    def _stamp(t, frac=False):
        s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
        return s + (",%03d" % int(round((t % 1) * 1000)) if frac else "")

    def _front(self, b, t, line):
        b._ingest("front", parse.parse_line("[%s] INFO weg2.front: %s" % (self._stamp(t, True), line)))

    def _decode(self, b, t):
        b._ingest("D", parse.parse_line(
            "[%s TP0] Decode rank batch, rank: 0, #round: 7, t: %.3f, bs: 1, #rows: 4, #fwd: 1, gpu-ms: 30.0 (x)"
            % (self._stamp(t), t)))

    def _prefill(self, b, t):
        b._ingest("P", parse.parse_line(
            "[%s PP0] Prefill batch, #new-seq: 1, #new-token: 16384, #cached-token: 0, full token usage: 0.26, "
            "#running-req: 0, #queue-req: 1, input throughput (token/s): 4898.99" % self._stamp(t)))

    def test_p_to_d_is_begin_to_first_decode_token_and_flip_total_is_only_the_layer_swap(self):
        base = 1790500000.0
        b = live.Boot("x", "/tmp")
        for k in range(20):                                     # D decoded before (earlier phase)
            self._decode(b, base - 100 + k * 0.05)
        for i, (fliptime, layer) in enumerate(((2.4, 1948), (3.1, 2100), (2.0, 1800))):
            t = base + i * 60
            self._front(b, t, "WEG2-FLIP begin epoch=%d sleep=P wake=D outstanding=0 queue=1" % (2 * i))
            self._front(b, t + layer / 1000.0, "WEG2-FLIP done epoch=%d slept=P woke=D drain+quiesce=100 ms "
                        "sleep=1000 ms wake=1000 ms flip_total=%d ms weights_tags=17" % (2 * i + 1, layer))
            for k in range(10):
                self._decode(b, t + fliptime + k * 0.05)
            # D->P: first PP0 'Prefill batch' 5 s after its begin
            self._front(b, t + 30, "WEG2-FLIP begin epoch=%d sleep=D wake=P outstanding=0 queue=1" % (2 * i + 1))
            self._prefill(b, t + 35.4)
        ft = b.flip_times_view()
        pd = ft["P>D"]
        self.assertEqual(pd["n"], 3)
        self.assertEqual(pd["last"], 2000)
        self.assertEqual(pd["median"], 2400)
        self.assertEqual(pd["p90"], 3100)
        self.assertEqual(pd["layer_last"], 1800)            # flip_total rides beside it, never as the Flipzeit
        dp = ft["D>P"]
        self.assertEqual(dp["n"], 3)
        self.assertTrue(4000 <= dp["last"] <= 5000, dp)      # whole-second stamp of the PP0 line
        self.assertEqual(dp["resolution_s"], 1.0)
        self.assertEqual(b.view(base + 200, with_series=False)["flip_times"]["P>D"]["n"], 3)

    def test_a_flip_without_follow_up_work_has_no_flip_time_and_the_newest_is_open(self):
        base = 1790600000.0
        b = live.Boot("x", "/tmp")
        self._front(b, base, "WEG2-FLIP begin epoch=0 sleep=P wake=D outstanding=0 queue=0")
        self._front(b, base + 20, "WEG2-FLIP begin epoch=1 sleep=D wake=P outstanding=0 queue=0")
        self._decode(b, base + 21)                              # after the NEXT begin: not this flip's token
        self._front(b, base + 40, "WEG2-FLIP begin epoch=2 sleep=P wake=D outstanding=0 queue=0")
        ft = b.flip_times_view()
        self.assertEqual(ft["P>D"]["n"], 0)
        self.assertEqual(ft["P>D"]["no_work"], 1)
        self.assertTrue(ft["P>D"]["open"])
        self.assertEqual([r["state"] for r in ft["recent"]], ["ohne Folgearbeit", "ohne Folgearbeit", "offen"])

    def test_every_value_names_its_instrument_and_a_27b_boot_leads_with_flip_total(self):
        # 27B-Review 29.09.: the 27B history is flip_total -- no silent switch of definition under it
        for stem, tag, headline in (("boot_weg2_dkrnfh91x_0929", "dkrnfh91x", "first_token"),
                                    ("boot_weg2_dkr27bb1_0929", "dkr27bb1", "flip_total")):
            b = live.Boot(stem, "/tmp")
            b.meta["tag"] = tag
            ft = b.view(1790700000.0, with_series=False)["flip_times"]
            self.assertEqual(ft["headline"], headline)
            self.assertEqual(set(ft["instruments"]), {"first_token", "flip_total"})


class ImageChangesTests(unittest.TestCase):
    def test_seat_from_tree_then_form_then_dir(self):
        self.assertEqual(imagechanges.seat_of({"tree": "/opt/htsglang/src-nf"}), "NF")
        self.assertEqual(imagechanges.seat_of({"tree": "/opt/htsglang/src-27b/"}), "27B")
        self.assertEqual(imagechanges.seat_of({"form": "arch=dense profile=qwen27b model=x"}), "27B")
        self.assertEqual(imagechanges.seat_of({"dir": "/spinning/docker-acceptance/nf/evidence"}), "NF")
        self.assertIsNone(imagechanges.seat_of({"dir": "/tmp/x"}))

    def test_running_boot_wins_over_a_newer_dead_one(self):
        boots = [
            _boot("boot_weg2_nfa_70ac86e2bd_0928_2005", "/opt/htsglang/src-nf", "70ac86e2bd", last=200.0),
            _boot("boot_weg2_nfb_f833fcbb2d_0928_2011", "/opt/htsglang/src-nf", "f833fcbb2d", live_=True, last=150.0),
            _boot("boot_weg2_27a_85386b1df1_0928_1851", "/opt/htsglang/src-27b", "85386b1df1", last=100.0),
        ]
        cur = imagechanges.current_boot_per_seat(boots)
        self.assertEqual(cur["NF"]["meta"]["sha"], "f833fcbb2d")
        self.assertEqual(cur["27B"]["meta"]["sha"], "85386b1df1")

    def test_view_names_the_image_its_changes_and_the_missing_rev(self):
        boots = [
            _boot("boot_weg2_nfb_f833fcbb2d_0928_2011", "/opt/htsglang/src-nf", "f833fcbb2d", live_=True, last=1.0,
                  container={"State": "running", "Image": "htsglang:cu130-weg2-rc12z29b-27b-nf"}),
            _boot("boot_weg2_27a_85386b1df1_0928_1851", "/opt/htsglang/src-27b", "85386b1df1", last=1.0),
        ]
        v = imagechanges.view(boots, IMAGES)
        self.assertEqual([s["seat"] for s in v["seats"]], ["27B", "NF"])
        b27, nf = v["seats"]
        self.assertFalse(b27["found"])
        self.assertEqual(b27["rev"], "85386b1df1")
        self.assertTrue(nf["found"] and nf["running"])
        self.assertEqual((nf["rc"], nf["image"]), ("rc12z29b", "htsglang:cu130-weg2-rc12z29b-27b-nf"))
        self.assertEqual([c["status"] for c in nf["changes"]], ["belegt", "unbekannt"])
        self.assertEqual(nf["changes"][1]["status_raw"], "geraten")

    def test_short_or_long_rev_matches(self):
        self.assertEqual(imagechanges.entry_for(IMAGES, "f833fcbb2d0123")[0], "f833fcbb2d")
        self.assertEqual(imagechanges.entry_for(IMAGES, "f833fcb")[0], "f833fcbb2d")
        self.assertEqual(imagechanges.entry_for(IMAGES, "f83")[0], None)

    def test_secret_in_a_text_field_is_cut(self):
        boots = [_boot("boot_weg2_nfb_f833fcbb2d_0928_2011", "/opt/htsglang/src-nf", "f833fcbb2d", live_=True)]
        imgs = {"f833fcbb2d": {"changes": [{"id": "A", "status": "belegt", "evidence": "token=abcdefgh12345678"}]}}
        self.assertNotIn("abcdefgh12345678", json.dumps(imagechanges.view(boots, imgs)))

    def test_loader_rereads_on_change_and_keeps_the_last_good_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "image_changes.json")
            ic = imagechanges.ImageChanges(p)
            self.assertIsNotNone(ic.load()[1])  # missing file is named
            with open(p, "w") as fh:
                json.dump({"images": IMAGES}, fh)
            imgs, err = ic.load()
            self.assertIsNone(err)
            self.assertIn("f833fcbb2d", imgs)
            with open(p, "w") as fh:
                fh.write('{"images": {"broken')
            imgs, err = ic.load()
            self.assertIn("f833fcbb2d", imgs)
            self.assertIn("JSONDecodeError", err)

    def test_snapshot_carries_the_view(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ic.json")
            with open(p, "w") as fh:
                json.dump({"images": IMAGES}, fh)
            ns = argparse.Namespace(log_glob=[os.path.join(d, "none-*.log")], docker_ssh="", docker_host_prefix="",
                                    front=[], gpuq="", state_dir="", release_profile=[], image_changes=p,
                                    features=os.path.join(d, "features.json"), features_repo=d)
            app = server.App(ns)
            snap = app.snapshot(with_series=False)
            self.assertEqual(snap["image_changes"], {"path": p, "error": None, "seats": []})
            # the feature table rides the same snapshot; a missing file is named, never a crash
            self.assertIn("FileNotFoundError", snap["features"]["error"])
            self.assertEqual([m["model"] for m in snap["features"]["models"]], ["NF", "27B"])

