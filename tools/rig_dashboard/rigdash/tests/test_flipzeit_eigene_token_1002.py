"""Nutzer 02.10. ~17:50Z (fuenfte Ruege, "GEHT DAS JETZT MAL IN EUREN KOPF REIN???"):

  "letztes decode token wurde erzeugt ->(alles hier ist flipzeit)->erster chunk prefill -> letzer cunk prefill
   -> (alles hier ist flipzeit)->erstes decode token wurde erzeugt"

Endpunkte sind die echten Erzeugungszeitpunkte der Gruppen, nie ein Front-Marker.  Zahlen aus NF y7y
(boot ...171652Z-520c, 999b64781a): Erstflip D>P 17:20:33 -- D's letzte TP0-Runde 17:20:26,417, der Park-Stempel
(alter Start) 6,05 s spaeter; P>D 17:21:20 -- P's letztes Chunk-Ende (PP2) 51 ms vor dem Leg-1-Ende am Front.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import grouplog, ipcboot  # noqa: E402

T = 1790961600.0


def _ev(typ, ts, **data):
    return {"type": typ, "ts": ts, "data": data}


def _sum(x):
    return sum(x[k] for k in ipcboot.PARTS if x.get(k) is not None)


def _ipc_dp():
    """y7y Erstflip D>P: park_rpc_sent 32,463, begin 33,891, done 37,571, PP0 first forward 37,776."""
    ev = [_ev("flip_begin", T + 33.891, flip_begin_ts=T + 33.891, sleep="D", wake="P", epoch_before=0),
          _ev("flip_done", T + 37.571, flip_begin_ts=T + 33.891, t=T + 37.571, flip_ms=1703, epoch=1, sleep="D", wake="P")]
    ut = [{"dir": "D>P", "epoch": 1, "flip_user_ms": 5313, "idle_flip": False, "rid": "pdflip-1-1", "start_ts": T + 32.463,
           "start_source": "park_rpc_sent", "prefill_start_ts": T + 37.776, "prefill_start_source": "pp_first_forward"}]
    return {"ipc_events": ev, "flip_user_time": ut, "flip_first_work": []}


SEGS = [{"s": T + 0.0, "e": T + 120.0, "k": "unknown"}]
ARR = {"pdflip-1-1": T + 26.0}       # the waiter arrived before D's last token: start = that token
D_ROUNDS = [(T + 26.39, T + 26.405), (T + 26.40, T + 26.417)]


class DPStartIsDsLastToken(unittest.TestCase):
    def test_dp_starts_at_ds_last_round_not_the_park_stamp(self):
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 120.0, None, d_rounds=D_ROUNDS, arrivals=ARR)[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["start"], T + 26.417, places=3)
        self.assertAlmostEqual(x["total_ms"], (37.776 - 26.417) * 1000, delta=1)   # 11,36 s, not 5,31 s
        self.assertIn("D log last decode round", x["start_src"])
        self.assertAlmostEqual(x["start_front"], T + 32.463, places=3)              # named, no endpoint
        self.assertAlmostEqual(x["warmup_ms"], (33.891 - 26.417) * 1000, delta=1)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_round_after_flip_begin_still_counts_until_d_sleeps(self):
        # D finished a round 20 ms after flip_begin (drain): the flip starts there, vorlauf 0
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 120.0, None,
                               d_rounds=D_ROUNDS + [(T + 33.89, T + 33.911)], arrivals=ARR)[0]
        self.assertAlmostEqual(x["start"], T + 33.911, places=3)
        self.assertEqual(x["warmup_ms"], 0.0)
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_no_d_log_is_missing_never_the_front_stamp(self):
        x = ipcboot.flip_views(SEGS, _ipc_dp(), T + 120.0, None, d_rounds=None)[0]
        self.assertEqual((x["kind"], x["total_ms"]), ("fehlt", None))
        self.assertEqual(x["missing"], ipcboot.F_DP_START_NOLOG)
        y = ipcboot.flip_views(SEGS, _ipc_dp(), T + 120.0, None, d_rounds=[])[0]
        self.assertEqual((y["kind"], y["missing"]), ("fehlt", ipcboot.F_DP_START))


def _ipc_pd(first_work=23.06):
    """y7y P>D 17:21:20: front leg-1 end 20,0 (p_end_ts), P's last chunk end (PP2) 19,949, D first round 22,4."""
    ev = [_ev("flip_begin", T + 20.002, flip_begin_ts=T + 20.002, sleep="P", wake="D", epoch_before=5),
          _ev("flip_done", T + 22.6, flip_begin_ts=T + 20.002, t=T + 22.6, flip_ms=2424, epoch=6, sleep="P", wake="D")]
    fw = [{"dir": "P>D", "flip_begin_ts": T + 20.002, "first_work_ts": T + first_work, "what": "decode_token",
           "p_end_ts": T + 20.0, "p_end_source": "p_leg1_end", "epoch": 6}]
    return {"ipc_events": ev, "flip_first_work": fw, "flip_user_time": []}


SEGS_PD = [{"s": T + 5.0, "e": T + 19.949, "k": "P"}, {"s": T + 19.949, "e": T + 120.0, "k": "unknown"}]


class PDEndpointsAreTheGroupsOwn(unittest.TestCase):
    def test_pd_starts_at_ps_last_chunk_end(self):
        x = ipcboot.flip_views(SEGS_PD, _ipc_pd(), T + 120.0, None, d_rounds=[])[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["start"], T + 19.949, places=3)
        self.assertAlmostEqual(x["p_end_front"], T + 20.0, places=3)
        self.assertAlmostEqual(x["warmup_ms"], 53, delta=1)                    # chunk end -> flip_begin
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_pd_start_reads_the_last_stages_own_chunk_stamp_not_the_sample_clock(self):
        # the ring sample after the burst lands at 20,4 (the P segment ends there); the last stage's record says
        # its newest chunk ended at 19,949
        ring = [{"t": T + 18.4, "r": {"P.tp0pp0": {"ts": T + 18.3, "plast_t": T + 17.0},
                                      "P.tp0pp2": {"ts": T + 18.3, "plast_t": T + 17.5}}},
                {"t": T + 20.4, "r": {"P.tp0pp0": {"ts": T + 20.3, "plast_t": T + 19.1},
                                      "P.tp0pp2": {"ts": T + 20.3, "plast_t": T + 19.949}}}]
        segs = [{"s": T + 5.0, "e": T + 20.4, "k": "P"}, {"s": T + 20.4, "e": T + 120.0, "k": "unknown"}]
        x = ipcboot.flip_views(segs, _ipc_pd(), T + 120.0, ring, d_rounds=[])[0]
        self.assertAlmostEqual(x["start"], T + 19.949, places=3)
        self.assertIn("P.tp0pp2 prefill.last.t", x["start_src"])
        self.assertAlmostEqual(_sum(x), x["total_ms"], delta=1e-6)

    def test_pd_ends_at_the_earlier_real_token(self):
        # D's first round (open 22,40, 30 gpu-ms) ended before the front saw the handed-off rid's token
        x = ipcboot.flip_views(SEGS_PD, _ipc_pd(), T + 120.0, None, d_rounds=[(T + 22.40, T + 22.43)])[0]
        self.assertAlmostEqual(x["end"], T + 22.43, places=3)
        self.assertIn("D log first decode round", x["end_src"])
        # the front's token (from D's extend) came first: it stays the end
        y = ipcboot.flip_views(SEGS_PD, _ipc_pd(first_work=22.35), T + 120.0, None,
                               d_rounds=[(T + 22.40, T + 22.43)])[0]
        self.assertAlmostEqual(y["end"], T + 22.35, places=3)
        self.assertIn("front flip_first_work", y["end_src"])
        self.assertAlmostEqual(_sum(y), y["total_ms"], delta=1e-6)

    def test_pd_rounds_before_the_flip_are_not_the_first_token(self):
        x = ipcboot.flip_views(SEGS_PD, _ipc_pd(), T + 120.0, None, d_rounds=[(T + 10.0, T + 10.02)])[0]
        self.assertAlmostEqual(x["end"], T + 23.06, places=3)


class GroupLogReader(unittest.TestCase):
    LINE = ("[2026-10-02 17:21:14 TP%d] Decode rank batch, rank: %d, #round: %d, t: %.3f, bs: 1, #rows: 4, "
            "#fwd: 1, gpu-ms: %.1f (split unavailable: graph-replay-reader-off)\n")

    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "nf", "evidence"))
        self.state = os.path.join(self.root, "nf", "state", "boot-x")
        os.makedirs(self.state)
        self.stem = "boot_weg2_tagx_999b64781a_1002_171722"
        self.log = os.path.join(self.root, "nf", "evidence", self.stem + ".D.log")
        grouplog._READERS.clear()

    def tearDown(self):
        shutil.rmtree(self.root)
        grouplog._READERS.clear()

    def _ipc(self, manifest=True):
        env = {grouplog.MANIFEST_ENV: "/var/lib/htsglang/evidence/%s.shared_cache" % self.stem} if manifest else {}
        return {"dir": self.state, "tag": "tagx", "launch": {"D": {"env": env}}}

    def test_reads_tp0_rounds_open_plus_gpu_ms_incrementally(self):
        with open(self.log, "w") as fh:
            fh.write(self.LINE % (0, 0, 316, 1790961674.650, 23.1))
            fh.write(self.LINE % (1, 1, 316, 1790961674.651, 23.0))          # another rank: not D's TP0 clock
            fh.write("[2026-10-02 17:21:14 TP0] Decode batch, #running-req: 1\n")
        self.assertEqual(grouplog.d_log_path(self._ipc()), self.log)
        self.assertEqual(grouplog.d_log_path(self._ipc(manifest=False)), self.log)   # by the launcher tag
        r = grouplog.decode_rounds(self._ipc())
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0][0], 1790961674.650, places=3)
        self.assertAlmostEqual(r[0][1], 1790961674.6731, places=4)
        line = self.LINE % (0, 0, 317, 1790961674.675, 21.0)
        with open(self.log, "a") as fh:
            fh.write(line[:40])                                                 # half a line: not yet a round
        self.assertEqual(len(grouplog.decode_rounds(self._ipc())), 1)
        with open(self.log, "a") as fh:
            fh.write(line[40:])
        r = grouplog.decode_rounds(self._ipc())
        self.assertEqual(len(r), 2)
        self.assertAlmostEqual(r[1][1], 1790961674.696, places=3)

    def test_no_log_no_rounds(self):
        self.assertIsNone(grouplog.decode_rounds(self._ipc()))
        self.assertIsNone(grouplog.decode_rounds({}))


if __name__ == "__main__":
    unittest.main()
