"""IPC Phase 2 (27B-Verbraucherseite, IPC-VERBRAUCHER-27B A4/N4/D6/W1).

Nutzer 28.09. ~20:45Z: „diese kommunikation über logs? … das muss man professionell
ordentlich machen“. Gepinnt wird hier:

  * der Deadman schreibt sein Urteil über den EINEN Schreibcode (writer=deadman): Event
    deadman_verdict immer, stop_request.json im Format des NF-Arms (dm_request) nur in
    einem lebenden Zustand und nur, wenn noch keine liegt (die erste Ursache gewinnt);
    während `stopping` keine Stop-Anfrage (sonst würde ein geplantes Ende über finish
    zu dead rc 24);
  * der Schreiber `deadman` besitzt keine Felder und keine Zustände;
  * `health` (Docker-Healthcheck) entscheidet aus dem Zustand: dead / stop_request /
    Gruppe dead = tot, fehlender Zustand = HEALTH_NO_STATE, nie „gesund“ aus Stille;
  * launcher.arm_deadman gibt dem Deadman Gruppe, Interpreter und Schreibcode mit;
  * Testregel: jede Log-lesende Riegel-Stelle im Launcher trägt einen benannten
    Ausnahmevermerk (IPC-LOG-EXCEPTION).

Kein GPU, kein Server.
"""

import json
import os
import re
import tempfile
import unittest

from flliper.srt.pdflip import state_file as sf


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def _boot(self, *states, kind="boot"):
        self.n += 1
        d = sf.init(self.root, f"t{self.n}-{kind}-20260929T000000Z-00{self.n:02d}", kind, {})
        for s in states:
            if s == "stopping":
                sf.transition(d, s, cause=sf.make_cause("stop_file", "operator"))
            else:
                sf.transition(d, s)
        return d

    @staticmethod
    def _events(d, typ):
        return [e for e in sf.events(d) if e["type"] == typ]


class TestDeadmanVerdict(_Base):
    def test_verdict_in_serving_writes_event_and_stop_request(self):
        d = self._boot("launching", "loading", "serving")
        r = sf.deadman_verdict(d, "CRASH", "DEADMAN[CRASH] no process", name="deadman_D", group="D")
        self.assertEqual(r, {"written": True, "stop_request": True, "state": "serving"})
        (ev,) = self._events(d, "deadman_verdict")
        self.assertEqual((ev["code"], ev["group"]), ("DEADMAN_CRASH", "D"))
        self.assertEqual(ev["data"]["detail_full"], "deadman_D: DEADMAN[CRASH] no process")
        with open(os.path.join(d, "stop_request.json")) as f:
            req = json.load(f)
        # dasselbe Format wie acc_nf_rc11b_dauer_v2.sh dm_request
        self.assertEqual(req, {"code": "DEADMAN_CRASH", "origin": "deadman", "group": "D", "rank": None,
                               "detail_full": "deadman_D: DEADMAN[CRASH] no process"})
        # der Host-Schreiber macht daraus dead rc 24
        cause = sf.stop_request(d)
        self.assertEqual((cause["code"], cause["origin"], cause["rc"]), ("DEADMAN_CRASH", "deadman", 24))

    def test_first_cause_wins(self):
        d = self._boot("launching", "loading", "serving")
        sf.write_json_atomic(os.path.join(d, "stop_request.json"),
                             {"code": "GT_LIVELOCK_REFUSED", "origin": "rank", "group": "D", "rank": None,
                              "detail_full": "arm"})
        r = sf.deadman_verdict(d, "HANG-OR-LIVELOCK", "x", name="deadman_front")
        self.assertFalse(r["stop_request"])
        self.assertEqual(sf.stop_request(d)["code"], "GT_LIVELOCK_REFUSED")
        self.assertEqual(len(self._events(d, "deadman_verdict")), 1)

    def test_no_stop_request_while_the_planned_end_runs(self):
        for states in (("launching", "loading", "serving", "stopping"),
                       ("launching", "loading", "serving", "stopping", "stopped_clean")):
            d = self._boot(*states)
            r = sf.deadman_verdict(d, "CRASH", "teardown", name="deadman_P", group="P")
            self.assertFalse(r["stop_request"], states)
            self.assertFalse(os.path.exists(os.path.join(d, "stop_request.json")), states)
            (ev,) = self._events(d, "deadman_verdict")
            self.assertEqual(ev["data"]["after_state"], states[-1])

    def test_front_has_no_group_and_tier_is_normalised(self):
        d = self._boot("launching", "loading", "serving")
        sf.deadman_verdict(d, "flip-stall", "x", name="deadman_front", group="front")
        self.assertEqual(sf.stop_request(d)["code"], "DEADMAN_FLIP_STALL")
        self.assertIsNone(sf.stop_request(d)["group"])

    def test_without_state_nothing_is_written(self):
        d = os.path.join(self.root, "none")
        os.makedirs(d)
        self.assertEqual(sf.deadman_verdict(d, "CRASH", "x"), {"written": False, "stop_request": False, "state": None})
        self.assertEqual(os.listdir(d), [])

    def test_deadman_writer_owns_no_fields_and_no_states(self):
        d = self._boot("launching")
        with self.assertRaises(sf.StateFileError):
            sf.transition(d, "dead", cause=sf.make_cause("X", "deadman"), writer="deadman")
        with self.assertRaises(sf.StateFileError):
            sf.transition(d, None, fields={"front": {}}, writer="deadman")
        sf.add_event(d, "deadman_verdict", {"tier": "CRASH"}, writer="deadman")
        self.assertIn("deadman", sf.read(d)["heartbeat"])
        self.assertEqual(sf.read(d)["lifecycle"]["state"], "launching")


class TestHealth(_Base):
    def test_serving_is_ok(self):
        d = self._boot("launching", "loading", "serving")
        self.assertEqual(sf.health(d)[0], sf.HEALTH_OK)

    def test_dead_stop_request_and_dead_group_are_dead(self):
        d = self._boot("launching")
        sf.transition(d, "dead", cause=sf.make_cause("CONTAINER_EXIT_137", "container_exit"))
        code, why = sf.health(d)
        self.assertEqual(code, sf.HEALTH_DEAD)
        self.assertIn("CONTAINER_EXIT_137", why)

        d = self._boot("launching", "loading", "serving")
        sf.deadman_verdict(d, "CRASH", "x", name="deadman_D", group="D")
        self.assertEqual(sf.health(d), (sf.HEALTH_DEAD, "stop_request DEADMAN_CRASH origin=deadman group=D"))

        d = self._boot("launching", "loading")
        sf.transition(d, None, fields={"groups.P": {"state": "dead"}}, writer="launcher")
        self.assertEqual(sf.health(d), (sf.HEALTH_DEAD, "group P dead"))

    def test_missing_or_foreign_state_is_no_state_never_ok(self):
        d = os.path.join(self.root, "empty")
        os.makedirs(d)
        self.assertEqual(sf.health(d)[0], sf.HEALTH_NO_STATE)
        with open(os.path.join(d, "state.json"), "w") as f:
            json.dump({"schema": "pdflip.state/99"}, f)
        self.assertEqual(sf.health(d)[0], sf.HEALTH_NO_STATE)

    def test_cli_exit_codes(self):
        d = self._boot("launching", "loading", "serving")
        self.assertEqual(sf.main(["health", "--dir", d]), sf.HEALTH_OK)
        sf.main(["deadman", "--dir", d, "--tier", "CRASH", "--name", "deadman_D", "--group", "D", "--detail", "x"])
        self.assertEqual(sf.main(["health", "--dir", d]), sf.HEALTH_DEAD)


LAUNCHER = os.path.join(os.path.dirname(sf.__file__), "launcher.py")
#: Riegel-Helfer, die ein Gruppen-Log lesen (IPC-STATE-PLAN §1 A, IPC-VERBRAUCHER-27B A1)
_LOG_READERS = re.compile(r"\b(count_marker|canonical_marker_counts|gate_w10|gate_w11|_cc\.census|"
                          r"_cc\.form_a_worker_ranks)\(")


class TestLauncher(unittest.TestCase):
    def test_arm_deadman_hands_over_group_interpreter_and_writer(self):
        from flliper.srt.pdflip import launcher
        lines = []
        launcher.arm_deadman(lines.append, "/x/boot.D.log", 30031, "launch_server.*--port 30031", 120,
                             "tagx", "D", True)
        (cmd,) = [x for x in lines if "would arm deadman" in x]
        self.assertIn("PDFLIP_DEADMAN_GROUP=D ", cmd)
        self.assertIn(f"PDFLIP_STATE_FILE_PY={sf.__file__}", cmd)
        self.assertRegex(cmd, r"PDFLIP_PY=\S*python")

    def test_every_log_reading_gate_carries_a_named_exception(self):
        """Testregel (IPC-STATE-PLAN, Zielbild): kein Riegel liest Log-Text als Steuergröße,
        außer mit benanntem Ausnahmevermerk auf der Zeile oder in den 6 Zeilen davor."""
        with open(LAUNCHER) as f:
            src = f.read().splitlines()
        missing = []
        for i, line in enumerate(src):
            code = line.split("#", 1)[0]
            if not _LOG_READERS.search(code) or re.match(r"\s*def ", code):
                continue
            window = src[max(0, i - 6): i + 1]
            if not any("IPC-LOG-EXCEPTION" in w for w in window):
                missing.append(f"launcher.py:{i + 1}: {line.strip()[:100]}")
        self.assertEqual(missing, [], "log-reading gate without IPC-LOG-EXCEPTION:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
