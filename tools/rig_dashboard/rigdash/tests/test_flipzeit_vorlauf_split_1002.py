"""Nutzer 02.10. ~18:25Z via NF: der D>P-Vorlauf wird benannt, total bleibt die Flipzeit ab D's letztem Token.

  vorlauf = leer (D's letztes Token -> Ankunft des Requests, der P braucht) + halt (Ankunft -> Park/flip_begin)
            + park (Park-RPC) + vor_rest (Park-Quittung -> flip_begin)

Leerlauf ohne Request darf echte Halte nicht verdecken.  Zahlen vom Metall:
  27B N6i epoch 10 (18:14:51): letzte D-Runde 331 endet 47,603; weg2-10-12 kommt 51,758 (WEG2 SESSION), Preisverdikt
      51,865 (oldest_waiter_arrival), kein Park -> leer 4,155 s, halt ~0,107 s, park 0.  TP0 schrieb die Runden
      285-331 erst um 18:15:28 (nach dem naechsten Wake): bis dahin ist der Start vorlaeufig (Runde 283, 46,021).
  NF y7y epoch 17 (17:48:51): letzte D-Runde endet 325,628 (D extendiert danach Agenten-Turns), weg2-16-52 kommt
      329,405, Park-RPC gesendet 329,682, rpc 1681 ms -> leer 3,777 s (davon D-Prefill), halt 0,277 s, park 1,681 s.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rigdash import grouplog, ipcboot, vmpush  # noqa: E402


def _ev(typ, ts, **data):
    return {"type": typ, "ts": ts, "data": data}


def _sum(x, keys):
    return sum(x[k] for k in keys if x.get(k) is not None)


B27 = 1790964891.866            # 27B N6i epoch 10 flip_begin


def _ipc_27b():
    ev = [_ev("flip_begin", B27, flip_begin_ts=B27, sleep="D", wake="P", epoch_before=9),
          _ev("flip_done", B27 + 1.9, flip_begin_ts=B27, t=B27 + 1.9, flip_ms=1570, epoch=10, sleep="D", wake="P")]
    ut = [{"dir": "D>P", "epoch": 10, "rid": "weg2-10-12", "idle_flip": False, "start_ts": 1790964891.865,
           "start_source": "oldest_waiter_arrival", "parts": {"park_rpc_ms": None, "pre_begin_ms": 1},
           "prefill_start_ts": B27 + 2.2, "prefill_start_source": "pp_first_forward"}]
    return {"ipc_events": ev, "flip_user_time": ut, "flip_first_work": []}


#: the D log read at 18:14:5x (rounds up to 283) and after 18:15:28 (285-331 written, then the next phase's 333)
ROUNDS_EARLY = [(1790964885.926, 1790964885.9559), (1790964885.991, 1790964886.021)]
ROUNDS_LATE = ROUNDS_EARLY + [(1790964887.573, 1790964887.603), (1790964909.927, 1790964909.9586)]
ARR_27B = {"weg2-10-12": 1790964891.758}
SEGS = [{"s": 1790964800.0, "e": 1790965000.0, "k": "unknown"}]


class VorlaufSplit27B(unittest.TestCase):
    def test_split_sums_to_vorlauf_and_names_the_idle(self):
        x = ipcboot.flip_views(SEGS, _ipc_27b(), B27 + 60, None, d_rounds=ROUNDS_LATE, arrivals=ARR_27B)[0]
        self.assertEqual(x["kind"], "ok")
        self.assertFalse(x["provisional"])
        self.assertAlmostEqual(x["start"], 1790964887.603, places=3)
        self.assertAlmostEqual(x["leer_ms"], 4155, delta=2)
        self.assertAlmostEqual(x["halt_ms"], 108, delta=2)
        self.assertEqual(x["park_ms"], 0.0)
        self.assertAlmostEqual(_sum(x, ipcboot.VOR_PARTS), x["vorlauf_ms"], delta=1e-6)
        self.assertAlmostEqual(_sum(x, ipcboot.PARTS), x["total_ms"], delta=1e-6)       # total unchanged
        self.assertIn("WEG2 SESSION", x["arrival_src"])

    def test_without_the_front_log_the_pricing_verdict_is_the_arrival(self):
        x = ipcboot.flip_views(SEGS, _ipc_27b(), B27 + 60, None, d_rounds=ROUNDS_LATE, arrivals=None)[0]
        self.assertAlmostEqual(x["leer_ms"], 4262, delta=2)
        self.assertIn("oldest_waiter_arrival", x["arrival_src"])
        self.assertAlmostEqual(_sum(x, ipcboot.VOR_PARTS), x["vorlauf_ms"], delta=1e-6)

    def test_late_written_rounds_keep_the_start_provisional_and_out_of_vm(self):
        x = ipcboot.flip_views(SEGS, _ipc_27b(), B27 + 5, None, d_rounds=ROUNDS_EARLY, arrivals=ARR_27B)[0]
        self.assertTrue(x["provisional"])
        self.assertAlmostEqual(x["start"], 1790964886.021, places=3)    # too early by 1,58 s until 331 is written
        self.assertEqual(vmpush.flip_view_points([x], "27B", "b", set()), [])
        y = ipcboot.flip_views(SEGS, _ipc_27b(), B27 + 60, None, d_rounds=ROUNDS_LATE, arrivals=ARR_27B)[0]
        pts = vmpush.flip_view_points([y], "27B", "b", set())
        parts = {p.split('part="')[1].split('"')[0] for p in pts}
        self.assertTrue({"total", "vorlauf", "leer", "halt", "park", "vor_rest"} <= parts)
        # without a later round for too long the row is taken as it is
        z = ipcboot.flip_views(SEGS, _ipc_27b(), B27 + ipcboot.PROVISIONAL_MAX_S + 10, None,
                               d_rounds=ROUNDS_EARLY, arrivals=ARR_27B)[0]
        self.assertFalse(z["provisional"])


BNF = 1790963331.364            # NF y7y epoch 17 flip_begin
SEGS_NF = [{"s": 1790963300.0, "e": 1790963400.0, "k": "unknown"}]


def _ipc_nf():
    ev = [_ev("flip_begin", BNF, flip_begin_ts=BNF, sleep="D", wake="P", epoch_before=16),
          _ev("flip_done", BNF + 2.2, flip_begin_ts=BNF, t=BNF + 2.2, flip_ms=2160, epoch=17, sleep="D", wake="P")]
    ut = [{"dir": "D>P", "epoch": 17, "rid": "weg2-16-52", "idle_flip": False, "start_ts": 1790963329.682,
           "start_source": "park_rpc_sent", "parts": {"park_rpc_ms": 1681, "pre_begin_ms": 1682},
           "prefill_start_ts": 1790963334.634, "prefill_start_source": "pp_first_forward"}]
    return {"ipc_events": ev, "flip_user_time": ut, "flip_first_work": []}


class VorlaufSplitNF(unittest.TestCase):
    def test_park_wait_is_its_own_part_and_d_prefill_shows_inside_leer(self):
        rounds = [(1790963325.600, 1790963325.628), (BNF + 30.0, BNF + 30.03)]
        segs = [{"s": 1790963300.0, "e": 1790963325.7, "k": "dec"},
                {"s": 1790963325.7, "e": BNF, "k": "D"},             # D extends the agent turns
                {"s": BNF, "e": 1790963400.0, "k": "unknown"}]
        x = ipcboot.flip_views(segs, _ipc_nf(), BNF + 60, None, d_rounds=rounds,
                               arrivals={"weg2-16-52": 1790963329.405})[0]
        self.assertEqual(x["kind"], "ok")
        self.assertAlmostEqual(x["leer_ms"], 3777, delta=2)
        self.assertAlmostEqual(x["halt_ms"], 277, delta=2)
        self.assertAlmostEqual(x["park_ms"], 1681, delta=2)
        self.assertAlmostEqual(x["vor_rest_ms"], 1, delta=2)
        self.assertAlmostEqual(x["leer_d_prefill_ms"], 3705, delta=2)
        self.assertAlmostEqual(_sum(x, ipcboot.VOR_PARTS), x["vorlauf_ms"], delta=1e-6)
        self.assertAlmostEqual(_sum(x, ipcboot.PARTS), x["total_ms"], delta=1e-6)

    def test_request_waiting_before_ds_last_token_is_no_idle(self):
        # the request arrived before D's last round (D still decoded): leer 0, all of it is halt + park
        rounds = [(1790963329.6, 1790963329.62), (BNF + 30.0, BNF + 30.03)]
        x = ipcboot.flip_views(SEGS_NF, _ipc_nf(), BNF + 60, None, d_rounds=rounds,
                               arrivals={"weg2-16-52": 1790963320.0})[0]
        self.assertEqual(x["leer_ms"], 0.0)
        self.assertAlmostEqual(x["halt_ms"], 62, delta=2)
        self.assertAlmostEqual(_sum(x, ipcboot.VOR_PARTS), x["vorlauf_ms"], delta=1e-6)

    def test_no_arrival_no_split(self):
        u = _ipc_nf()
        u["flip_user_time"][0]["rid"] = None
        rounds = [(1790963325.600, 1790963325.628), (BNF + 30.0, BNF + 30.03)]
        x = ipcboot.flip_views(SEGS_NF, u, BNF + 60, None, d_rounds=rounds, arrivals={})[0]
        self.assertEqual(x["kind"], "ok")
        self.assertIsNone(x["leer_ms"])
        self.assertIsNone(x["halt_ms"])
        self.assertIsNotNone(x["vorlauf_ms"])


class FrontArrivalReader(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "27b", "evidence"))
        self.state = os.path.join(self.root, "27b", "state", "boot-x")
        os.makedirs(self.state)
        self.stem = "boot_weg2_tagx_f501e462d0_1002_180930"
        self.log = os.path.join(self.root, "27b", "evidence", self.stem + ".front.log")
        grouplog._FRONT.clear()

    def tearDown(self):
        shutil.rmtree(self.root)
        grouplog._FRONT.clear()

    def test_first_session_stamp_per_rid(self):
        with open(self.log, "w") as fh:
            fh.write("[2026-10-02 18:14:51,758] INFO weg2.front: WEG2 SESSION rid=weg2-10-12 sess=- src=none\n")
            fh.write("[2026-10-02 18:14:51,864] INFO weg2.front: WEG2 X-EXACT-PRICE rid=weg2-10-12 pending=74199\n")
            fh.write("[2026-10-02 18:14:59,000] INFO weg2.front: WEG2 SESSION rid=weg2-10-12 sess=- src=none\n")
        ipc = {"dir": self.state, "tag": "tagx", "launch": {}}
        self.assertEqual(grouplog.front_log_path(ipc), self.log)
        a = grouplog.front_arrivals(ipc)
        self.assertAlmostEqual(a["weg2-10-12"], 1790964891.758, places=3)
        self.assertIsNone(grouplog.front_arrivals({}))


if __name__ == "__main__":
    unittest.main()
