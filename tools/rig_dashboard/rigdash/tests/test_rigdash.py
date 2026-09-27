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

from rigdash import live, parse, server, sources  # noqa: E402

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
        self._write("D", [_shift(DEC, now - 3), _shift(DEC_RANK, now - 3)])
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
        # compute-derived: bs 5 * accept 2.70 / 0.1177 s
        self.assertAlmostEqual(d["compute_tps"], 5 * 2.70 / 0.1177, places=1)
        self.assertIn("P_prefill_tps", v["series"])
        self.assertEqual(len([x for x in v["series"]["P_prefill_tps"] if x]), 1)

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
        rows = sources.parse_docker_ps(json.dumps({"Names": "a", "State": "running"}) + "\n\n")
        self.assertEqual(rows[0]["Names"], "a")


if __name__ == "__main__":
    unittest.main()
