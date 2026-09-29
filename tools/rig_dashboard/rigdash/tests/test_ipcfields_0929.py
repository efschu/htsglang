"""DASHBOARD-AUS-IPC, the 25 "Übergang" rows (user 29.09. ~12:00Z via 27B: "warum sind die
ganzen werte im dashboard noch aus log"): one reader per field, switched by presence.

For every field: the IPC source present -> the IPC value (src=ipc); absent -> the log value
with the label "aus Log (Übergang)".  The fixtures follow the producers' schemas:
weg2/rankstats.py (weg2.rankstats/1, 2188e1bd98), weg2/front.py _ipc_front_fields and the
events flip_begin / flip_done / flip_first_work / group_health (front_state_ipc.py), and the
§3 blocks the inventory names for rankstats (prefill / decode / cache / work.spans).
"""

import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import ipcfields, ipcstate  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html")


def _ev(typ, data, **kw):
    return dict({"schema": "weg2.event/1", "ts": data.get("ts", 1790680000.0), "type": typ, "data": data}, **kw)


def _ipc():
    """ipcstate.boot_view of a boot whose producers are all in the image."""
    return {
        "boot_id": "nf-boot-x", "lifecycle": "serving", "serving_since_ts": 1790679000.0,
        "forms": {"D": "OWNED_CUT_X1=workers kv=S0"},
        "front": {"groups": {"P": {"http_ok": True, "alive": True, "streak": 0, "ts": 1.0},
                             "D": {"http_ok": True, "alive": True, "streak": 0, "ts": 1.0}},
                  "errors": {"n": 1, "last": [{"t": 5.0, "logger": "sglang", "level": "ERROR", "exc": None,
                                               "text": "front err"}]},
                  "served_tokens": {"P": {"n": 3, "prompt": 30000, "cached": 1000, "completion": 0}},
                  "d_phase_n": 2, "d_parked_n": 1},
        "ipc_events": [
            _ev("group_health", {"group": "D", "verdict": "ok", "prev": None}),
            _ev("flip_begin", {"epoch_before": 3, "sleep": "P", "wake": "D", "flip_begin_ts": 100.0}),
            _ev("flip_done", {"epoch": 4, "sleep": "P", "wake": "D", "flip_ms": 2100, "drain_quiesce_ms": 40,
                              "sleep_ms": 900, "wake_ms": 1100, "chunks_n": 5, "flip_begin_ts": 100.0}),
            _ev("flip_first_work", {"epoch": 4, "dir": "P>D", "flip_begin_ts": 100.0, "first_work_ts": 102.5,
                                    "flip_time_ms": 2500, "what": "decode_token", "rid": "r1"}),
            _ev("flip_first_work", {"epoch": 5, "dir": "D>P", "flip_begin_ts": 110.0, "first_work_ts": 113.0,
                                    "flip_time_ms": 3000, "what": "p_prefill", "rid": "r2"}),
            _ev("rank_stop", {"exception_type": "RuntimeError", "detail_full": "x"}, group="D", rank=0,
                code="W35"),
            _ev("post_wake_pass", {"epoch": 4, "mode": "decode", "schedule_ms": 3, "run_ms": 31, "prepare_ms": 2}),
            _ev("group_ready", {"group": "D", "after_s": 182.0}),
        ],
    }


def _stats(ts=1000.0, prefill_total=1000, decode_total=500, new_tokens=16384, compute_ms=3000.0):
    base = {"schema": ipcfields.RANKSTATS_SCHEMA, "pid": 1, "ts": ts, "seq": 1, "tp_rank": 0, "pp_rank": 0,
            "work": {"forward_ct": 7, "spans": [{"t0": 1.0, "t1": 2.0, "kind": "prefill"},
                                                {"t0": 3.0, "t1": 3.5, "kind": "extend"}]},
            "tokens": {"prefill_total": prefill_total, "decode_total": decode_total},
            "spec": {"accept_tokens_total": 300, "forward_ct_total": 100},
            "sched": {"waiting": 2, "running": 1},
            "errors": {"n": 2, "last": [{"t": 4.0, "logger": "sglang", "level": "ERROR", "exc": "OSError",
                                         "text": "rank err"}]},
            "last_post_wake": {"n": 0, "run_ms": 31},
            "prefill": {"chunks": 3, "new_tokens": new_tokens, "cached_tokens": 4096, "compute_ms": compute_ms,
                        "wait_ms": 10.0, "gpu_ms": compute_ms},
            "decode": {"rounds": 50, "tokens": 500, "running": 1, "accept_len_ewma": 2.9,
                       "gpu_ms_by_bs": {"1": [40, 1200.0]}},
            "cache": {"loadback_n": 2, "loadback_tok": 8192, "mamba_resume_n": 1, "store_incomplete_n": 0,
                      "prefetch": {"landed": 4, "refused": 0, "deferred": 1, "timeout": 0}}}
    return {"P.tp0pp0": dict(copy.deepcopy(base), group="P"), "D.tp0pp0": dict(copy.deepcopy(base), group="D")}


def _rank():
    return {"rankstats": _stats(),
            "rankstate": {"D.tp0pp0": {"schema": 2, "kv": {"kv_tokens": 262144}, "seats": {"n": 2, "cap": 6}}}}


def _logv():
    return {"stem": "boot_x", "live": True, "meta": {"form": "log-form"}, "health": {"D": {"alive": True}},
            "error_count": 9, "errors": [{"t": 1, "text": "log err"}], "stop_count": 1, "stops": [{"t": 1}],
            "last_activity": {"P": 1.0}, "flip_times": {"p2d": {"n": 1}, "d2p": {"n": 1}},
            "flips": [{"t": 1, "slept": "P", "woke": "D"}], "flip_count": 12, "flip_open": None,
            "prefill": {"P": {"now": None}}, "queue_log": 3, "decode": {"D": {"round_ms_by_bs": {"1": 30}}},
            "series": {"t": []}, "timeline": {"P": [1], "D": [2]}, "first_work_t": 5.0, "cache": {"P": {}},
            "totals": {"served": 7}}


class FieldSwitchTests(unittest.TestCase):
    """Every field: IPC present -> IPC value; absent -> log value with the label."""

    def setUp(self):
        self.on = ipcfields.resolve(_ipc(), _rank(), _logv(), rates={"D.tp0pp0": {"decode_tps": 88.0},
                                                                    "P.tp0pp0": {"prefill_tps_gpu": 5400.0}})
        self.off = ipcfields.resolve(None, None, _logv(), rates={})

    def test_all_keys_present_both_ways(self):
        self.assertEqual(set(self.on), set(ipcfields.KEYS))
        self.assertEqual(set(self.off), set(ipcfields.KEYS))

    def test_every_field_reads_ipc_when_the_source_is_there(self):
        not_ipc = [k for k, f in self.on.items() if f["src"] != "ipc"]
        self.assertEqual(not_ipc, [])
        for k, f in self.on.items():
            self.assertIsNone(f["label"], k)
            self.assertTrue(f["ipc_src"], k)

    def test_every_field_falls_back_to_the_log_with_its_label(self):
        for k, f in self.off.items():
            self.assertEqual(f["src"], "log", k)
            self.assertEqual(f["label"], "aus Log (Übergang)", k)

    def test_values(self):
        on, off = self.on, self.off
        self.assertEqual(on["A1"]["value"]["boot_id"], "nf-boot-x")
        self.assertEqual(on["A4"]["value"], {"D": "OWNED_CUT_X1=workers kv=S0"})
        self.assertEqual(off["A4"]["value"], "log-form")
        self.assertEqual(on["A12"]["value"]["D"]["verdict"], "ok")
        self.assertEqual(on["A13"]["value"]["n"], 1 + 2 + 2)          # front + two ranks
        self.assertEqual(off["A13"]["value"]["n"], 9)
        self.assertEqual(on["A14"]["value"][0]["code"], "W35")
        self.assertEqual(on["A15"]["value"]["D.tp0pp0"]["forward_ct"], 7)
        self.assertEqual(on["B1"]["value"]["last_ms"], 2500)
        self.assertEqual(on["B2"]["value"]["last_ms"], 3000)
        self.assertEqual(on["B3"]["value"]["flip_ms"], 2100)
        self.assertEqual(on["B4"]["value"]["n"], 1)
        self.assertEqual(off["B4"]["value"]["n"], 12)
        self.assertEqual(len(on["B5"]["value"]), 1)
        self.assertFalse(on["B6"]["value"]["open"])                    # epoch 3 -> done epoch 4
        self.assertEqual(on["B7"]["value"]["run_ms"], 31)
        self.assertEqual(on["C1"]["value"]["P.tp0pp0"]["prefill_tps_gpu"], 5400.0)
        self.assertEqual(on["C2"]["value"]["D.tp0pp0"]["waiting"], 2)
        self.assertEqual(off["C2"]["value"], 3)
        self.assertEqual(on["C3"]["value"]["D.tp0pp0"]["gen_tps"], 88.0)
        self.assertEqual(on["C3"]["value"]["D.tp0pp0"]["accept_len_mean"], 3.0)
        self.assertEqual(on["C4"]["value"]["D.tp0pp0"], {"1": [40, 1200.0]})
        self.assertEqual(on["C5"]["value"]["D.tp0pp0"]["kv_tokens"], 262144)
        self.assertEqual(on["C6"]["value"], {"n": 2, "parked_n": 1})
        self.assertIn("D.tp0pp0", on["C7"]["value"])
        self.assertEqual(on["D1"]["value"]["P.tp0pp0"][0]["kind"], "prefill")
        self.assertEqual(on["D2"]["value"]["D.tp0pp0"][0]["kind"], "extend")
        self.assertEqual(on["D3"]["value"]["done"]["flip_ms"], 2100)
        self.assertEqual(on["D4"]["value"]["group_ready_after_s"], {"D": 182.0})
        self.assertEqual(on["E1"]["value"]["P.tp0pp0"], 4096)
        self.assertEqual(on["E2"]["value"]["D.tp0pp0"]["loadback_n"], 2)
        self.assertEqual(on["E3"]["value"]["P"]["prompt"], 30000)
        self.assertEqual(off["E3"]["value"], 7)
        self.assertEqual(on["F3"]["value"], on["B1"]["value"])
        self.assertEqual(on["F4"]["value"], on["A4"]["value"])
        self.assertEqual(set(on["F6"]["value"]), {"C1", "C2", "C3", "C4", "C5", "C6"})

    def test_producer_of_today_switches_what_it_carries_and_nothing_more(self):
        """2188e1bd98 rankstats without the §3 blocks: C2/C3/A13/A15/B7 switch, C1/C4/E1/E2/D1 stay log."""
        st = _stats()
        for rec in st.values():
            for k in ("prefill", "decode", "cache"):
                rec.pop(k)
            rec["work"].pop("spans")
        f = ipcfields.resolve(None, {"rankstats": st, "rankstate": {}}, _logv(), rates={})
        self.assertEqual({k for k in ("A13", "A15", "B7", "C2", "C3") if f[k]["src"] == "ipc"},
                         {"A13", "A15", "B7", "C2", "C3"})
        self.assertEqual({k for k in ("C1", "C4", "E1", "E2", "D1", "D2") if f[k]["src"] == "log"},
                         {"C1", "C4", "E1", "E2", "D1", "D2"})

    def test_open_flip(self):
        ipc = _ipc()
        ipc["ipc_events"] = [e for e in ipc["ipc_events"] if e["type"] != "flip_done"]
        f = ipcfields.resolve(ipc, None, _logv(), rates={})
        self.assertTrue(f["B6"]["value"]["open"])
        self.assertEqual(f["B6"]["value"]["since_ts"], 100.0)

    def test_page_payload_carries_no_log_value_twice(self):
        page = ipcfields.for_page(self.off)
        self.assertTrue(all("value" not in v for v in page.values()))
        self.assertTrue(all("value" in v for v in ipcfields.for_page(self.on).values()))
        s = ipcfields.summary(self.off)
        self.assertEqual((s["ipc"], s["log"], s["n"]), (0, len(ipcfields.KEYS), len(ipcfields.KEYS)))


class RatesTests(unittest.TestCase):
    def test_delta_rates_and_restart(self):
        r = ipcfields.Rates()
        self.assertEqual(r.update("b", _stats(ts=10.0)), {})
        out = r.update("b", _stats(ts=12.0, prefill_total=9000, decode_total=700, new_tokens=32768,
                                   compute_ms=6000.0))
        self.assertEqual(out["P.tp0pp0"]["prefill_tps"], 4000.0)
        self.assertEqual(out["D.tp0pp0"]["decode_tps"], 100.0)
        self.assertAlmostEqual(out["P.tp0pp0"]["prefill_tps_gpu"], 16384 / 3.0, places=1)
        # a restarted rank (counters back to small numbers): no negative rate
        out = r.update("b", _stats(ts=14.0, prefill_total=10, decode_total=5))
        self.assertNotIn("prefill_tps", out["P.tp0pp0"])
        self.assertNotIn("decode_tps", out["D.tp0pp0"])

    def test_same_sample_keeps_the_last_rate(self):
        r = ipcfields.Rates()
        r.update("b", _stats(ts=10.0))
        a = r.update("b", _stats(ts=12.0, decode_total=700))
        self.assertEqual(r.update("b", _stats(ts=12.0, decode_total=700))["D.tp0pp0"], a["D.tp0pp0"])


class RankFilesTests(unittest.TestCase):
    def test_reads_rankstats_and_rankstate_next_to_the_group_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "boot_weg2_x_0929_1.D.log")
            open(log, "w").close()
            rs = log + ".rankstate"
            os.makedirs(rs)
            with open(os.path.join(rs, "D.tp0pp0.rankstats"), "w") as fh:
                json.dump(_stats()["D.tp0pp0"], fh)
            with open(os.path.join(rs, "D.tp1pp0.rankstats"), "w") as fh:
                json.dump({"schema": "weg2.rankstats/9"}, fh)            # another schema: refused
            with open(os.path.join(rs, "D.tp0pp0.json"), "w") as fh:
                json.dump({"schema": 2, "kv": {"kv_tokens": 1}}, fh)
            dirs = ipcfields.rankstate_dirs({"D": {"path": log}, "front": {"path": log}})
            self.assertEqual(dirs, [rs])
            got = ipcfields.read_rank_files(dirs)
            self.assertEqual(list(got["rankstats"]), ["D.tp0pp0"])
            self.assertEqual(list(got["rankstate"]), ["D.tp0pp0"])


class BootViewEventsTests(unittest.TestCase):
    def test_boot_view_keeps_the_field_events_and_forms(self):
        ev = ipcstate._Events("/nonexistent")
        ev.rows = [_ev("flip_first_work", {"dir": "P>D", "flip_time_ms": 1}), _ev("lifecycle", {"state": "serving"})]
        st = {"schema": "weg2.state/1", "groups": {"D": {"form": "F", "state": "ready"}}, "lifecycle": {}}
        v = ipcstate.boot_view("/d", st, ev, 0.0)
        self.assertEqual([e["type"] for e in v["ipc_events"]], ["flip_first_work"])
        self.assertEqual(v["forms"], {"D": "F"})
        for t in ("flip_begin", "flip_done", "flip_first_work", "group_health", "rank_stop", "post_wake_pass"):
            self.assertIn(t, ipcstate.EVENT_TYPES)


class PageTests(unittest.TestCase):
    """The page shows the source per field: srcOf() at every former fixed LOGSRC of these rows."""

    def test_surfaces_read_the_field_source(self):
        html = open(STATIC).read()
        for needle in ('srcOf(b, "D1", "D2", "D3", "D4")', 'srcOf(b, "C1", "C2", "C7")',
                       'srcOf(b, "C3", "C4", "C7")', 'srcOf(b, "E1", "E2", "E3")', 'srcOf(b, "A12")',
                       'srcOf(b, "B1", "B2", "B3", "B4", "B5", "B6")', 'srcOf(b, "A13")', 'fieldIpc(b, "A4")',
                       'fields_summary'):
            self.assertIn(needle, html)
        self.assertNotIn("Cache-Treffer (aus den Boot-Logs)${LOGSRC}", html)
        self.assertNotIn("Flips &middot; Warteschlange${LOGSRC}", html)


if __name__ == "__main__":
    unittest.main()
