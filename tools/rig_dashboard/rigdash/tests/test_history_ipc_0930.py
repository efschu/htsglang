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


class TestIpcRates(unittest.TestCase):
    def test_first_rank_only_and_rates(self):
        stats0 = {"P.tp0pp0": rs("P", 0, 100.0, pnew=1000, pcached=200),
                  "P.tp0pp1": rs("P", 0, 100.0, pnew=1000, pcached=200),     # same chunk on PP1: not again
                  "D.tp0pp0": rs("D", 0, 100.0, pnew=10, pcached=5000, dtok=100, rounds=10, dgpu=1000, running=2, kv=0.5)}
        stats1 = {"P.tp0pp0": rs("P", 0, 105.0, pnew=6000, pcached=700),
                  "P.tp0pp1": rs("P", 0, 105.0, pnew=6000, pcached=700),
                  "D.tp0pp0": rs("D", 0, 105.0, pnew=20, pcached=9000, dtok=300, rounds=60, dgpu=5000, running=2, kv=0.6)}
        a, b = history.rank_sample(stats0), history.rank_sample(stats1)
        self.assertEqual(sorted(b), ["D", "P"])
        v = history.ipc_rates(a, b, now=106.0)
        self.assertEqual(v["p_tps"], 1000.0)             # 5000 new tokens / 5 s, counted once
        self.assertEqual(v["d_tps"], 2.0)
        self.assertEqual(v["dec_tps"], 40.0)
        self.assertEqual(v["stream_tps"], 20.0)          # 40 tok/s / 2 running
        self.assertEqual(v["kv_pct"], 60.0)
        self.assertEqual(v["tok_cache"], 100.0)          # P cached = cache
        self.assertEqual(v["tok_handoff"], 800.0)        # D cached = Übergabe, never cache
        self.assertEqual(v["tok_comp_p"], 1000.0)
        self.assertEqual(v["tok_comp_d"], 2.0)

    def test_idle_group_gives_zero_dead_rank_gives_nothing(self):
        a = history.rank_sample({"P.tp0pp0": rs("P", 0, 100.0, pnew=50), "D.tp0pp0": rs("D", 0, 100.0, dtok=9, kv=0.1)})
        b = history.rank_sample({"P.tp0pp0": rs("P", 0, 105.0, pnew=50), "D.tp0pp0": rs("D", 0, 100.0, dtok=9, kv=0.1)})
        v = history.ipc_rates(a, b, now=121.0)            # D's file is 21 s old: rank gone
        self.assertEqual(v["p_tps"], 0.0)                 # P alive and idle: 0, a continuous line
        self.assertNotIn("d_tps", v)
        self.assertNotIn("dec_tps", v)
        self.assertNotIn("kv_pct", v)

    def test_stream_only_while_decoding_continuously(self):
        a = history.rank_sample({"D.tp0pp0": rs("D", 0, 100.0, dtok=0, rounds=0, dgpu=0.0, running=1)})
        b = history.rank_sample({"D.tp0pp0": rs("D", 0, 105.0, dtok=40, rounds=10, dgpu=1000.0, running=1)})
        v = history.ipc_rates(a, b, now=105.0)            # decode GPU time 1 s of 5 s: a flip was in there
        self.assertEqual(v["dec_tps"], 8.0)
        self.assertNotIn("stream_tps", v)

    def test_counter_restart_counts_from_zero(self):
        a = history.rank_sample({"single.tp0pp0": rs("single", 0, 100.0, pnew=9000, dtok=500)})
        b = history.rank_sample({"single.tp0pp0": rs("single", 0, 105.0, pnew=100, dtok=50)})
        v = history.ipc_rates(a, b, now=105.0)
        self.assertEqual(v["p_tps"], 20.0)
        self.assertEqual(v["dec_tps"], 10.0)

    def test_first_sample_levels_only(self):
        b = history.rank_sample({"D.tp0pp0": rs("D", 0, 100.0, kv=0.25)})
        self.assertEqual(history.ipc_rates(None, b, now=101.0), {"kv_pct": 25.0, "tok_cache": 0.0, "tok_comp_p": 0.0,
                                                                  "tok_comp_d": 0.0, "tok_handoff": 0.0})

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
    def test_samples_state_dir_and_marks_ipc(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "27b", "state", "27bbf-boot-x")
            os.makedirs(os.path.join(d, "rankstate", "D"))
            f = os.path.join(d, "rankstate", "D", "D.tp0pp0.rankstats")
            view = {"dir": d, "boot_id": "27bbf-boot-x", "kind": "boot", "terminal": False, "tag": "27bbf",
                    "flip_first_work": [{"dir": "P>D", "flip_begin_ts": 1000.0, "flip_time_ms": 2100.0}]}
            db = history.HistoryDB(None)
            rec = history.Recorder(db, _Logs(None), ipc=_Ipc([view]))
            with open(f, "w") as fh:
                json.dump(rs("D", 0, 1000.0, dtok=0, kv=0.5), fh)
            rec.ingest_ipc(1001.0)
            with open(f, "w") as fh:
                json.dump(rs("D", 0, 1005.0, dtok=50, rounds=10, dgpu=4500.0, running=2, kv=0.6), fh)
            rec.ingest_ipc(1006.0)
            got = db.query(["m.27B.dec_tps", "m.27B.kv_pct", "m.27B.ipc", "m.27B.stream_tps"], 995, 1010, 5, now=1010)
            self.assertEqual(got["m.27B.dec_tps"], {1005: 10.0})
            self.assertEqual(got["m.27B.stream_tps"], {1005: 5.0})
            self.assertEqual(got["m.27B.kv_pct"], {1000: 50.0, 1005: 60.0})
            self.assertEqual(set(got["m.27B.ipc"]), {1000, 1005})
            self.assertEqual(db.get("ipc0.27bbf-boot-x"), 1001.0)      # the log backfill stops here
            self.assertEqual([m["label"] for m in db.marks("27B", 900, 1100) if m["kind"] == "flip"], ["P>D ipc"])
            view["terminal"], view["lifecycle"], view["lifecycle_since"] = True, "stopped_clean", 1007.0
            rec.ingest_ipc(1008.0)
            self.assertEqual([m["kind"] for m in db.marks("27B", 900, 1100) if m["kind"] != "flip"], ["end"])


class TestNoLogInHistory(unittest.TestCase):
    """Nutzer 30.09.: "warum ist da immernoch 'aus Log (übergang)'" -- no history series may come
    from a boot log.  Structural: history.py neither imports the log reader nor touches a log boot;
    behavioural: rows without an IPC sample (what an older rigdash derived from logs) are not shown."""

    def test_history_module_reads_no_log(self):
        src = open(history.__file__, encoding="utf-8").read()
        for w in (".D.log", ".P.log", ".front.log", "from . import live", "live.", "self.logs.boots",
                  "b.ev.get(", "served_legs", "flip_rows", "_prefill_batch", "_decode_batch", "aus Log"):
            self.assertNotIn(w, src, w)

    def test_recorder_never_opens_a_log(self):
        import builtins
        opened = []
        real = builtins.open

        def spy(path, *a, **k):
            opened.append(str(path))
            return real(path, *a, **k)
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "nf", "state", "nfx-boot-20260930T153426Z-051f")
            os.makedirs(os.path.join(d, "rankstate", "P"))
            with real(os.path.join(d, "rankstate", "P", "P.tp0pp0.rankstats"), "w") as fh:
                json.dump(rs("P", 0, 1000.0, pnew=5), fh)
            for g in ("P", "D", "front"):          # logs lying next to it must stay unopened
                real(os.path.join(tmp, "x.%s.log" % g), "w").close()
            view = {"dir": d, "boot_id": os.path.basename(d), "kind": "boot", "terminal": False}
            rec = history.Recorder(history.HistoryDB(None), _Logs(None), ipc=_Ipc([view]))
            builtins.open = spy
            try:
                rec.ingest_ipc(1001.0)
                rec.ingest_ipc(1006.0)
            finally:
                builtins.open = real
        self.assertTrue(any(p.endswith(".rankstats") for p in opened))
        self.assertFalse([p for p in opened if p.endswith(".log")])

    def test_view_hides_rows_without_ipc_sample(self):
        db = history.HistoryDB(None)
        db.put([("m.NF.p_tps", 9900, 500.0), ("m.NF.p_tps", 9905, 700.0), ("m.NF.ipc", 9905, 1.0)])
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
