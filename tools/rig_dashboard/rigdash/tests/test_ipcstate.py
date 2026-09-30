"""DASHBOARD-AUS-IPC (user order 29.09. via 27B): the page reads the boots' IPC state
(state.json / events.jsonl / stop_request.json, IPC-STATE-PLAN §2.2) before any log line.

The state.json fixture is shaped like the NF boot
nfh91dprsavisadoptstcutvsyncodx2bswre2cutdauer-boot-20260929T065359Z-4579 (fields trimmed).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import health, ipcstate, stops  # noqa: E402

TAG = "dkrnfh91dprsavisadoptstcutvsyncodx2bswre2cutbar1dauer09290654"
BOOT_ID = "nfh91dprsavisadoptstcutvsyncodx2bswre2cutdauer-boot-20260929T065359Z-4579"


def _state(lc="serving", cause=None, **kw):
    st = {
        "schema": "weg2.state/1", "seq": 17, "boot_id": BOOT_ID, "kind": "boot", "tag": TAG, "line": "nf",
        "rev": "4f23714e2d", "profile": "nf-h91-dpr-sa-vis-adopt-st-cut-vsync-odx-2b-swr-e2cut",
        "image": "htsglang:cu130-weg2-rc12z30u-27b-nf", "container": "htsglang-acc-nf-x",
        "lifecycle": {"state": lc, "since_ts": 1790667599.9, "prev": "flipping"},
        "serving_since_ts": 1790665146.37, "cause": cause,
        "heartbeat": {"host_acceptance": {"pid": 1, "ts": 1790667599.9, "seq": 1}},
        "front": {"awake": "P", "epoch": 165, "outstanding": 2, "queue": 1, "state": "serving"},
        "groups": {
            "P": {"state": "ready", "launch": {"argv": ["python", "-m", "sglang.launch_server", "--model-path",
                                                        "/m/Qwen3.8-Flash-Next-INT4", "--served-model-name", "NF-INT4",
                                                        "--pp-size", "3"],
                                               "env": {"HTSGLANG_TRANSPORT": "bar1"}}},
            "D": {"state": "ready", "launch": {"argv": ["python", "-m", "sglang.launch_server", "--tp-size", "3"],
                                               "env": {}}},
        },
    }
    st.update(kw)
    return st


def _write_boot(root, st, events=(), stop_request=None):
    d = os.path.join(root, st["boot_id"])
    os.makedirs(d)
    with open(os.path.join(d, "state.json"), "w") as fh:
        json.dump(st, fh)
    with open(os.path.join(d, "events.jsonl"), "w") as fh:
        for i, e in enumerate(events):
            fh.write(json.dumps(dict({"schema": "weg2.event/1", "seq": i + 1, "boot_id": st["boot_id"],
                                      "group": None, "rank": None, "code": None}, **e)) + "\n")
    if stop_request:
        with open(os.path.join(d, "stop_request.json"), "w") as fh:
            json.dump(stop_request, fh)
    return d


class TestBootView(unittest.TestCase):
    def test_topology_model_transport_from_launch_argv_not_log(self):
        v = ipcstate.boot_view("/x", _state(), None, 1790667600.0)
        self.assertEqual(v["topology"], {"P": {"tp": 1, "pp": 3}, "D": {"tp": 3, "pp": 1}})
        self.assertEqual(v["model"], "NF-INT4")
        self.assertEqual(v["transport"], "bar1")
        self.assertEqual(v["rev"], "4f23714e2d")
        self.assertFalse(v["terminal"])
        self.assertEqual(v["src"], "state.json")

    def test_terminal_states(self):
        for lc in ("stopped_clean", "dead", "refused_preflight"):
            self.assertTrue(ipcstate.boot_view("/x", _state(lc), None, 0)["terminal"], lc)


class TestClassifyIpc(unittest.TestCase):
    def test_planned_stop_from_lifecycle_and_cause(self):
        with tempfile.TemporaryDirectory() as root:
            _write_boot(root, _state("stopped_clean", {"code": "stop_file", "origin": "operator", "rc": 0}),
                        events=[{"type": "lifecycle", "ts": 1790667563.5, "data": {"state": "stopping", "prev": "serving"}}])
            s = ipcstate.IpcStates(roots=(root,))
            s.poll(now=os.stat(os.path.join(root, BOOT_ID, "state.json")).st_mtime + 1)
            v = s.for_tag(TAG)
            e = stops.classify_ipc(v)
            self.assertIsNone(e["death"])
            self.assertEqual(e["planned"]["t"], 1790667563.5)      # the stopping event, not the final write
            self.assertIn("stop_file", e["planned"]["text"])
            self.assertEqual(e["src"], "state.json")

    def test_dead_wins_over_stopping(self):
        v = ipcstate.boot_view("/x", _state("dead", {"code": "W98_Weg2HostRateLatched", "origin": "front", "rc": 24}),
                               None, 0)
        e = stops.classify_ipc(v)
        self.assertIsNone(e["planned"])
        self.assertIn("W98_Weg2HostRateLatched", e["death"]["text"])

    def test_front_stop_event_is_a_death_while_the_boot_still_runs(self):
        with tempfile.TemporaryDirectory() as root:
            _write_boot(root, _state("serving"),
                        events=[{"type": "front_stop", "ts": 1790666000.0, "code": "W98_Weg2HostRateLatched",
                                 "data": {"name": "W98 Weg2HostRateLatched"}}],
                        stop_request={"code": "W98_Weg2HostRateLatched", "origin": "front"})
            s = ipcstate.IpcStates(roots=(root,))
            s.poll(now=os.stat(os.path.join(root, BOOT_ID, "state.json")).st_mtime + 1)
            e = stops.classify_ipc(s.for_tag(TAG))
            self.assertEqual(e["death"]["t"], 1790666000.0)
            self.assertIn("front_stop", e["death"]["text"])

    def test_unknown_tag_has_no_state(self):
        with tempfile.TemporaryDirectory() as root:
            _write_boot(root, _state())
            s = ipcstate.IpcStates(roots=(root,))
            s.poll()
            self.assertIsNone(s.for_tag("dkrother"))
            self.assertIsNone(s.for_tag(None))

    def test_events_tail_waits_for_a_torn_line(self):
        with tempfile.TemporaryDirectory() as root:
            d = _write_boot(root, _state())
            with open(os.path.join(d, "events.jsonl"), "a") as fh:
                fh.write('{"schema": "weg2.event/1", "type": "hold_end", "ts": 1.0, "data": {"reason"')
            s = ipcstate.IpcStates(roots=(root,))
            s.poll()
            self.assertEqual(s.for_tag(TAG)["events"]["hold_end"], [])
            with open(os.path.join(d, "events.jsonl"), "a") as fh:
                fh.write(': "stop_file"}}\n')
            s.poll()
            self.assertEqual(s.for_tag(TAG)["events"]["hold_end"][0]["data"]["reason"], "stop_file")


def _fw(direction, begin, ms, what):
    # shaped like the y4j front's events (nfh91...clkpodauer-boot-20260930T101130Z-74d3)
    return {"type": "flip_first_work", "ts": begin + ms / 1000.0,
            "data": {"clock": "time.time front", "dir": direction, "epoch": 1, "flip_begin_ts": begin,
                     "first_work_ts": begin + ms / 1000.0, "flip_time_ms": ms, "rid": "weg2-0-2", "what": what}}


class TestFlipFirstWork(unittest.TestCase):
    """Flipzeit from the front's own clock (30.09.): the log's D->P mark ('Prefill batch' on
    PP0) comes only after the first chunk ran through all PP stages -- y4i showed 9.6 s median
    there against 2.0 s to the first P leg in events.jsonl."""

    def _log_view(self):
        # what flip_times_view gave for y4i: D->P from the late PP0 line (whole seconds)
        return {"P>D": {"n": 1, "last": 2600, "median": 2600, "p90": 2600, "resolution_s": 0.001,
                        "layer_median": 1400, "layer_n": 1, "no_work": 0, "open": False},
                "D>P": {"n": 2, "last": 9400, "median": 9400, "p90": 10386, "resolution_s": 1.0,
                        "layer_median": 1600, "layer_n": 2, "no_work": 0, "open": False},
                "recent": [{"t": 1790763310.679, "dir": "D>P", "ms": 9321, "state": "ok", "layer_ms": 1600},
                           {"t": 1790763330.0, "dir": "P>D", "ms": 2600, "state": "ok", "layer_ms": 1400},
                           {"t": 1790763350.25, "dir": "D>P", "ms": 10386, "state": "ok", "layer_ms": 1610}]}

    def test_events_keep_first_work_apart_from_the_row_window(self):
        with tempfile.TemporaryDirectory() as root:
            evs = [_fw("D>P", 1790763310.679, 4136, "p_leg1_dispatch")]
            evs += [{"type": "hold_end", "ts": 1.0, "data": {"reason": "x"}}] * (ipcstate.EVENTS_KEEP + 5)
            _write_boot(root, _state(), events=evs)
            s = ipcstate.IpcStates(roots=(root,))
            s.poll()
            fw = s.for_tag(TAG)["flip_first_work"]
            self.assertEqual([x["flip_time_ms"] for x in fw], [4136])

    def test_ipc_replaces_the_late_log_mark(self):
        from rigdash import live
        fw = [_fw("D>P", 1790763310.679, 4136, "p_leg1_dispatch")["data"],
              _fw("P>D", 1790763330.0, 2460, "decode_token")["data"],
              _fw("D>P", 1790763350.25, 1822, "p_leg1_dispatch")["data"]]
        ft = live.apply_ipc_first_work(self._log_view(), fw)
        self.assertEqual((ft["D>P"]["n"], ft["D>P"]["last"], ft["D>P"]["median"]), (2, 1822, 1822))
        self.assertEqual(ft["D>P"]["resolution_s"], 0.001)
        self.assertIn("p_leg1_dispatch", ft["D>P"]["src"])
        self.assertEqual(ft["D>P"]["layer_median"], 1600)     # layer swap stays from the log
        self.assertEqual([r["ms"] for r in ft["recent"]], [4136, 2460, 1822])
        self.assertEqual({r.get("src") for r in ft["recent"]}, {"ipc"})

    def test_without_ipc_the_log_view_stays(self):
        from rigdash import live
        view = self._log_view()
        ft = live.apply_ipc_first_work(view, [])
        self.assertEqual(ft["D>P"], view["D>P"])
        self.assertNotIn("src", ft["recent"][0])


class TestHealthFrontMirror(unittest.TestCase):
    def test_state_json_front_outstanding_is_one_number(self):
        # IPC §2.2 H4: the state.json mirror keeps outstanding as one number, /weg2/state per group
        b = {"live": True, "front": {"queue": 1, "outstanding": 2, "src": "state.json front (Host-Spiegel)"},
             "last_activity_any": 0.0, "health": {}, "stops": [], "end": {}}
        a = health.assess(b, now=1000.0)
        text = " ".join(r["text"] for r in a.get("reasons", []))
        self.assertIn("state.json front", text)


if __name__ == "__main__":
    unittest.main()
