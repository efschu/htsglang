"""Boot cards from IPC only (NF-Operator 30.09.: "Log-Rückfall für ALLE Werte entfernen. Keine Anzeige
liest mehr ein Boot-Log").  Three checks:

* behaviour: build_view turns a ring of rankstats samples into the card's numbers (rates, bursts,
  15-min curve without holes while the boot lives, totals, cache, flip times from events);
* no log is opened: IpcBoots.poll/snapshot and history.Recorder.ingest_ipc run with open() spied on,
  next to boot logs lying in the same directories;
* static scan over rigdash/*.py: the server does not start a log reader, and no module wired into a
  display opens *.log.  Named exceptions: live.py / parse.py (the old log reader, kept for its tests,
  not started by the server) and stops.py:HarnessLogs (the harness-log tail, not started either).
"""

import ast
import builtins
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import history, ipcboot  # noqa: E402

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rec(g, ts, pnew=0, pcached=0, pchunks=0, pcomp=0.0, dtok=0, rounds=0, dgpu=0.0, running=None, kv=None):
    return {"schema": "weg2.rankstats/1", "group": g, "ts": ts,
            "prefill": {"new_tokens": pnew, "cached_tokens": pcached, "chunks": pchunks, "compute_ms": pcomp,
                        "last": {"t": ts, "gpu_ms": 1000.0, "new": 1000} if pchunks else None},
            "decode": {"tokens": dtok, "rounds": rounds, "gpu_ms": dgpu, "running": running,
                       "accept_len_ewma": 2.5, "gpu_ms_by_bs": {"1": [10, 400.0]}},
            "sched": {"full_token_usage": kv, "queue_req": 0}, "cap": {"kv_tokens": 1000, "seats": 6}}


def ring_of(n, t0=1000.0):
    """P prefills 1000 tok/s for the first 20 s, then D decodes 50 tok/s with 2 streams."""
    out = []
    for i in range(n):
        t = t0 + i
        p = rec("P", t, pnew=1000 * min(i, 20), pchunks=min(i, 20), pcomp=500.0 * min(i, 20))
        d = rec("D", t, pnew=10, pcached=20000 if i > 20 else 0, dtok=50 * max(0, i - 20), rounds=20 * max(0, i - 20),
                dgpu=900.0 * max(0, i - 20), running=2, kv=0.4)
        out.append({"t": t + 0.2, "r": {k: ipcboot.compact(v) for k, v in
                                        (("P.tp0pp0", p), ("P.tp0pp1", dict(p, pp_rank=1)), ("D.tp0pp0", d))},
                    "front": {"awake": "P" if i <= 20 else "D", "served": {"P": 1, "D": 1}}})
    return out


class TestBuildView(unittest.TestCase):
    def setUp(self):
        self.ring = ring_of(61)
        self.now = self.ring[-1]["t"]
        self.ipc = {"boot_id": "nfx-boot-x", "dir": "/x/nf/state/nfx-boot", "tag": "nfx",
                    "model": "NF", "terminal": False, "lifecycle": "serving", "front": {"awake": "D"},
                    "ipc_events": [{"type": "flip_begin", "ts": 1020.5, "data": {"flip_begin_ts": 1020.5, "sleep": "P", "wake": "D"}},
                                   {"type": "flip_done", "ts": 1021.0,
                                    "data": {"sleep": "P", "wake": "D", "flip_ms": 1800, "flip_begin_ts": 1020.5, "t": 1021.0}}],
                    "flip_first_work": [{"dir": "P>D", "flip_begin_ts": 1020.5, "flip_time_ms": 2100, "what": "decode_token",
                                         "first_work_ts": 1022.6, "p_end_ts": 1020.4, "p_end_source": "p_leg1_end"}]}
        self.v = ipcboot.build_view(self.ipc, self.ring, {"rankstats": {}, "rankstate": {}}, {}, self.now)

    def test_rates_from_deltas(self):
        dec = self.v["decode"]["D"]
        self.assertAlmostEqual(dec["gen_tps"], 50.0)
        self.assertAlmostEqual(dec["one_s"], 50.0)
        self.assertEqual(dec["running"], 2)
        self.assertAlmostEqual(dec["compute_tps"], 50.0 / 0.9, places=3)
        self.assertEqual(dec["round_ms_by_bs"]["1"]["median_ms"], 40.0)
        pre = self.v["prefill"]["P"]
        self.assertEqual(pre["now"]["tokens"], 20000)                   # the 60-s window still holds P's burst
        self.assertAlmostEqual(pre["last_burst"]["wall_tps"], 1000.0)
        self.assertAlmostEqual(pre["last_burst"]["tps_gpu"], 2000.0)     # 1000 tok per 500 compute-ms
        self.assertEqual(self.v["live"], True)

    def test_series_has_no_hole_while_the_boot_lives(self):
        s = self.v["series"]
        dec = [v for t, v in zip(s["t"], s["D_decode_tps"]) if t >= self.ring[1]["t"]]
        self.assertTrue(dec and all(v is not None for v in dec))            # 0 when idle, never a gap

    def test_totals_cache_flips(self):
        t = self.v["totals"]
        self.assertEqual(t["p_new"], 20000)
        self.assertEqual(t["decoded"], 2000)
        self.assertAlmostEqual(t["p_rate_gpu"], 2000.0)
        c = self.v["cache"]["D"]["boot"]
        self.assertEqual((c["cached"], c["new"]), (20000, 10))
        ft = self.v["flip_times"]
        # Nutzer 02.10.: the Flipzeit is P-Ende -> erstes Decode-Token, never the layer-only 1,8 s; 17:50Z: P-Ende
        # is P's own last chunk stamp (prefill.last.t 1020,0 on the last stage), not the front's leg-1 end 1020,4
        self.assertEqual(ft["P>D"]["n"], 1)
        self.assertAlmostEqual(ft["P>D"]["last"], 2600.0, delta=0.5)
        self.assertNotIn("layer_newest", ft["P>D"])
        self.assertEqual(self.v["flip_count"], 1)
        self.assertTrue(any(x["k"] == "flip_pd" for x in self.v["timeline"]["segs"]))

    def test_missing_fields_name_their_writer(self):
        f = self.v["fields"]
        self.assertEqual(f["A4"]["src"], "fehlt")
        self.assertIn("SGLANG_WEG2_FORM", f["A4"]["missing"])
        self.assertIn("rankstats.py", f["C1"]["missing"])
        self.assertNotIn("Log", json.dumps(self.v["fields"], ensure_ascii=False))


class TestNoLogOpened(unittest.TestCase):
    def test_ipcboots_and_recorder_open_no_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nf", "state")
            d = os.path.join(root, "nfx-boot-20260930T153426Z-051f")
            os.makedirs(os.path.join(d, "rankstate", "D"))
            with open(os.path.join(d, "state.json"), "w") as fh:
                json.dump({"schema": "weg2.state/1", "boot_id": os.path.basename(d), "kind": "boot", "tag": "nfx",
                           "lifecycle": {"state": "serving"}, "front": {"awake": "D"}}, fh)
            with open(os.path.join(d, "events.jsonl"), "w") as fh:
                fh.write("")
            for g in ("P", "D", "front"):                   # boot logs lying right next to it
                open(os.path.join(d, "boot_x.%s.log" % g), "w").close()
                open(os.path.join(tmp, "boot_x.%s.log" % g), "w").close()
            rs = os.path.join(d, "rankstate", "D", "D.tp0pp0.rankstats")
            opened, real = [], builtins.open

            def spy(path, *a, **k):
                opened.append(str(path))
                return real(path, *a, **k)
            ib = ipcboot.IpcBoots(roots=(root,))
            hr = history.Recorder(history.HistoryDB(None), ib)
            import time
            now = time.time()
            builtins.open = spy
            try:
                for i in range(3):
                    with real(rs, "w") as fh:
                        json.dump(rec("D", now + i, dtok=10 * i, running=1, kv=0.5), fh)
                    ib.poll(now + i + 0.2)
                    hr.ingest_ipc(now + i + 0.2)
                views = ib.snapshot(now + 3)
            finally:
                builtins.open = real
            self.assertEqual(len(views), 1)
            self.assertTrue(any(p.endswith(".rankstats") for p in opened))
            self.assertEqual([p for p in opened if p.endswith(".log")], [])


class TestStaticScan(unittest.TestCase):
    """Red as soon as a display path opens a boot log again."""
    NOT_WIRED = {"live.py", "parse.py"}     # the old log reader: kept for its tests, never started

    def test_server_starts_no_log_reader(self):
        src = open(os.path.join(PKG, "server.py"), encoding="utf-8").read()
        for w in ("LiveLogs(", "HarnessLogs(", "--log-glob", "self.logs"):
            self.assertNotIn(w, src, w)
        self.assertIn("ipcboot.IpcBoots()", src)

    def test_no_wired_module_opens_a_log(self):
        for name in sorted(os.listdir(PKG)):
            if not name.endswith(".py") or name in self.NOT_WIRED:
                continue
            tree = ast.parse(open(os.path.join(PKG, name), encoding="utf-8").read())
            docs = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
            for node in ast.walk(tree):
                if id(node) in docs:
                    continue          # prose, not a path the code opens
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value
                    if name == "stops.py" and v in ("abnahme_cu130.log",):
                        continue     # HarnessLogs (not started by the server); constant kept for its tests
                    if name == "grouplog.py" and v in (".D.log", "boot_weg2_%s_*.D.log"):
                        continue     # Nutzer 02.10. ~17:50Z: Flipzeit-Endpunkte = D's eigene Decode-Runden (TP0
                                     # 'Decode rank batch' t:); rankstats hat keine Rundenzeit -- IPC-Nachfolger
                                     # decode.last_t beim Rang-Schreiber, dann faellt dieser eine Leser
                    self.assertFalse(v.endswith(".log") or "*.log" in v or "boot_*.log" in v,
                                     "%s: %r" % (name, v))
                if isinstance(node, ast.Attribute) and node.attr in ("LiveLogs", "HarnessLogs") and name != "stops.py":
                    self.fail("%s references %s" % (name, node.attr))


if __name__ == "__main__":
    unittest.main()
