"""Verlauf aus IPC (Nutzer 30.09.: "token-durchsatz / input-tokens aus cache/neu gerechnet / decode je
stream / kv-cache belegung sind nicht durchgängig ... liegt vermutlich noch an den alten datenquellen,
den logs"): the model series come from the ranks' rankstats, sampled every 5 s; an idle but alive
group gives 0 (a continuous line), a dead rank gives nothing; the log backfill stops at the first
IPC sample; the power panel sums all cards."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import history  # noqa: E402


def rs(g, t, ts, pnew=0, pcached=0, dtok=0, rounds=0, dgpu=0.0, running=None, kv=None):
    return {"schema": "weg2.rankstats/1", "group": g, "ts": ts,
            "prefill": {"new_tokens": pnew, "cached_tokens": pcached},
            "decode": {"tokens": dtok, "rounds": rounds, "gpu_ms": dgpu, "running": running},
            "tokens": {"prefill_total": pnew, "decode_total": dtok},
            "sched": {"full_token_usage": kv, "running_req": running or 0}}


class TestIpcHelpers(unittest.TestCase):
    def test_model_src_label(self):
        self.assertEqual(history.model_src([None, 1.0]), history.IPC_LABEL)
        self.assertEqual(history.model_src([None, None]), "keine Daten (vor IPC-Aufzeichnung)")

    def test_boot_ts_from_id(self):
        self.assertEqual(history.boot_ts("nfh91-boot-20260930T153426Z-051f"), 1790782466.0)
        self.assertIsNone(history.boot_ts("27bbf-boot-x"))

    def test_model_of_ipc(self):
        self.assertEqual(history.model_of_ipc({"dir": "/spinning/docker-acceptance/27b/state/x"}), "27B")
        self.assertEqual(history.model_of_ipc({"dir": "/spinning/docker-acceptance/nf/state/x", "tag": "nfh91"}), "NF")


class _Ipc:
    def __init__(self, views):
        self.views = views
        self.last_poll = 1.0

    def boots(self, now):
        return self.views


class _Logs:
    def __init__(self, ipc):
        import threading
        self.ipc = ipc
        self.lock = threading.Lock()
        self.boots = {}


class TestRecorderIpc(unittest.TestCase):
    """The recorder writes the 5-s buckets from the same 1-s ring the boot cards read, the work at the
    time it was done (test_activity_0930 holds the simulated NF boot)."""

    def test_buckets_from_the_ring(self):
        from collections import deque
        from rigdash import ipcboot
        from rigdash.tests import test_activity_0930 as sim
        T0 = 1790000000.0
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nf", "state")
            d = os.path.join(root, "nfx-boot-20260930T153426Z-051f")
            os.makedirs(d)
            with open(os.path.join(d, "state.json"), "w") as fh:
                json.dump({"schema": "weg2.state/1", "boot_id": os.path.basename(d), "kind": "boot", "tag": "nfx",
                           "lifecycle": {"state": "serving"}, "front": {"awake": "D"}}, fh)
            with open(os.path.join(d, "events.jsonl"), "w") as fh:
                for typ, data in (("flip_done", dict(sim.FLIP_DONE[0], flip_begin_ts=T0 + 26.0, t=T0 + 28.0)),
                                  ("flip_first_work", dict(sim.FIRST_WORK[0], flip_begin_ts=T0 + 26.0)),
                                  ("flip_first_work", {"dir": "D>P", "flip_begin_ts": T0 + 50.0, "what": "none",
                                                       "flip_time_ms": 900, "flip_total_ms": 900})):
                    fh.write(json.dumps({"schema": "weg2.event/1", "type": typ, "ts": T0 + 28.0, "data": data}) + "\n")
            ib = ipcboot.IpcBoots(roots=(root,))
            ib.poll(T0 + 0.5)
            ring = sim.ring_until(60.0)
            for smp in ring:
                smp["t"] += T0
                for r in smp["r"].values():
                    for k in ("ts", "plast_t"):
                        if r.get(k) is not None:
                            r[k] += T0
            key = os.path.basename(d)
            ib.rings[key] = deque(ring)
            db = history.HistoryDB(None)
            rec = history.Recorder(db, ib)
            rec.ingest_ipc(T0 + 100.0)
            got = db.query(["mi.NF.p_tps", "mi.NF.dec_tps", "mi.NF.d_tps", "mi.NF.ipc", "mi.NF.stream_tps"],
                           T0, T0 + 70, 5, now=T0 + 100)
            p = got["mi.NF.p_tps"]
            self.assertLess(max(p.values()), 6000)                               # never a chunk per bucket
            self.assertAlmostEqual(sum(p.values()) * 5, 4 * sim.CH, delta=5)
            dec = got["mi.NF.dec_tps"]
            both = [t for t in p if p[t] > 0 and (dec.get(t, 0) > 0 or got["mi.NF.d_tps"].get(t, 0) > 0)]
            # P and D never at once; a 5-s bucket can only hold both around a phase change (one flip here)
            self.assertLessEqual(len(both), 1)
            self.assertTrue(all(abs(v - 100.0) < 1 for v in got["mi.NF.stream_tps"].values()))
            self.assertEqual(db.get("hcur." + key) % 5, 0)
            flips = [m for m in db.marks("NF", T0, T0 + 100) if m["kind"] == "flip"]
            self.assertEqual([(m["label"], m["v"]) for m in flips], [("P>D ipc", 2500), ("D>P ipc", None)])
            v = history.view(db, None, "NF", "15m", now=T0 + 100)
            self.assertEqual(v["tiles"]["flip_n"], 1)                            # what=none is no Flipzeit


class TestNoLogInHistory(unittest.TestCase):
    """Nutzer 30.09.: "warum ist da immernoch 'aus Log (übergang)'" -- no history series may come
    from a boot log.  Structural: history.py neither imports the log reader nor touches a log boot;
    behavioural: rows without an IPC sample (what an older rigdash derived from logs) are not shown."""

    def test_history_module_reads_no_log(self):
        src = open(history.__file__, encoding="utf-8").read()
        for w in (".D.log", ".P.log", ".front.log", "from . import live", " live.", "self.logs.boots",
                  "b.ev.get(", "served_legs", "flip_rows", "_prefill_batch", "_decode_batch", "aus Log"):
            self.assertNotIn(w, src, w)

    def test_recorder_never_opens_a_log(self):
        import builtins
        from rigdash import ipcboot
        opened = []
        real = builtins.open

        def spy(path, *a, **k):
            opened.append(str(path))
            return real(path, *a, **k)
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nf", "state")
            d = os.path.join(root, "nfx-boot-20260930T153426Z-051f")
            os.makedirs(os.path.join(d, "rankstate", "P"))
            with real(os.path.join(d, "state.json"), "w") as fh:
                json.dump({"schema": "weg2.state/1", "boot_id": os.path.basename(d), "kind": "boot",
                           "lifecycle": {"state": "serving"}}, fh)
            for g in ("P", "D", "front"):          # logs lying next to it must stay unopened
                real(os.path.join(tmp, "x.%s.log" % g), "w").close()
                real(os.path.join(d, "x.%s.log" % g), "w").close()
            ib = ipcboot.IpcBoots(roots=(root,))
            rec = history.Recorder(history.HistoryDB(None), ib)
            import time
            now = time.time()
            builtins.open = spy
            try:
                for i in range(3):
                    with real(os.path.join(d, "rankstate", "P", "P.tp0pp0.rankstats"), "w") as fh:
                        json.dump(rs("P", 0, now + i, pnew=5 * i), fh)
                    ib.poll(now + i + 0.2)
                    rec.ingest_ipc(now + i + 60.0)
            finally:
                builtins.open = real
        self.assertTrue(any(p.endswith(".rankstats") for p in opened))
        self.assertFalse([p for p in opened if p.endswith(".log")])

    def test_view_hides_rows_without_ipc_sample(self):
        db = history.HistoryDB(None)
        db.put([("mi.NF.p_tps", 9900, 500.0), ("mi.NF.p_tps", 9905, 700.0), ("mi.NF.ipc", 9905, 1.0)])
        db.mark(9800, "NF", "flip", "P>D log", 3000.0)
        db.mark(9810, "NF", "flip", "P>D ipc", 2000.0)
        db.mark(9700, "NF", "boot", "nfx")                 # an old log-derived boot mark
        v = history.view(db, None, "NF", "15m", now=10000.0)
        i = v["t"].index(9900)
        self.assertIsNone(v["series"]["m.p_tps"][i])       # log-derived: not shown
        self.assertEqual(v["series"]["m.p_tps"][i + 1], 700.0)
        self.assertEqual([(m["kind"], m["label"]) for m in v["marks"]], [("flip", "P>D")])
        self.assertEqual(v["src"]["prefill"], history.IPC_LABEL)
        self.assertNotIn("Log", json.dumps(v["src"], ensure_ascii=False))


class TestViewPowerSum(unittest.TestCase):
    def test_power_is_the_sum_of_all_cards(self):
        db = history.HistoryDB(None)
        db.set("cards", [{"index": 0, "short": "RTX 3080", "uuid": "a"}, {"index": 1, "short": "RTX 5090", "uuid": "b"}])
        now = 10000.0
        db.put([("g0.power", 9900, 100.0), ("g1.power", 9900, 250.0), ("g0.power", 9905, 90.0)])
        v = history.view(db, None, "27B", "15m", now=now)
        i = v["t"].index(9900)
        self.assertEqual(v["series"]["gsum.power"][i], 350.0)
        self.assertEqual(v["series"]["gsum.power"][i + 1], 90.0)     # one card reported: its value, not a gap
        self.assertEqual(v["src"]["power"], "NVML, Summe aller Karten")
        self.assertNotIn("now_tiles", v["src"])                        # the log "jetzt" tiles are gone


if __name__ == "__main__":
    unittest.main()
