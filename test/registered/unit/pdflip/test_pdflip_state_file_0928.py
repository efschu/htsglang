"""IPC Phase 1b (27B-Abnahme A1-A7, H1-H6 zu §2.2): der EINE Schreibcode.

pdflip/state_file.py schreibt state.json/events.jsonl für alle Schreiber (Host,
Launcher, Front). Gepinnt wird hier:

  * A1: eine Sperre für alle Schreiber (flock auf <boot_id>/.lock), kein
    verlorenes Update bei zwei gleichzeitigen Schreibern; Feld- und Zustands-
    Eigentum je Schreiber; Übergänge nur vorwärts, terminal bleibt terminal,
    dead schlägt jeden nicht-terminalen Zustand (nie dead -> serving);
  * A1b: die Host-Kopie /spinning/gpu-arb/docker/acc_state.py ist byte-gleich;
  * A2 origin host, A3 probes_done/d2_done + stopped_clean cause.rc 0,
    A4 Herzschlag mit container/host_pid, A5 stop_request -> dead rc 24,
    A6 current nur für kind=boot, H1 serving_since_ts, H2 Tod nur kind=boot;
  * der Launcher schreibt über diesen Code (groups, invariants, dead mit
    W-Code+Name, launcher_done), RankState liegt unter rankstate/<G>.

Kein GPU, kein Launch eines Servers.
"""

import json
import multiprocessing
import os
import tempfile
import time
import unittest

from flliper.srt.pdflip import rank_state as rs
from flliper.srt.pdflip import state_file as sf

HOST_COPY = "/spinning/gpu-arb/docker/acc_state.py"


def _bump(d, writer, n, key):
    for i in range(n):
        sf.transition(d, None, fields={key: i}, writer=writer)


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _boot(self, kind="boot", bid="b1"):
        return sf.init(self.root, bid, kind, {})


class TestHostCopy(unittest.TestCase):
    def test_host_copy_is_byte_identical(self):
        if not os.path.exists(HOST_COPY):
            self.skipTest(f"{HOST_COPY} not on this machine")
        with open(sf.__file__, "rb") as a, open(HOST_COPY, "rb") as b:
            self.assertEqual(a.read(), b.read(), "acc_state.py must be a copy of pdflip/state_file.py (A1b)")


class TestOrder(_Base):
    def test_forward_only_and_dead_wins(self):
        d = self._boot()
        sf.transition(d, "launching")
        sf.transition(d, "serving")
        sf.transition(d, "ready", writer="launcher")        # launcher behind the host: no-op
        self.assertEqual(sf.read(d)["lifecycle"]["state"], "serving")
        sf.transition(d, "flipping")
        sf.transition(d, "serving")                          # serving <-> flipping
        sf.transition(d, "stopping", cause=sf.make_cause("probes_done", "operator"))
        sf.transition(d, "dead", cause=sf.make_cause("X", "container_exit"))   # dead beats stopping
        sf.transition(d, "serving")                          # never dead -> serving
        st = sf.read(d)
        self.assertEqual(st["lifecycle"]["state"], "dead")
        self.assertEqual([e["data"]["state"] for e in sf.events(d)],
                         ["preflight", "launching", "serving", "flipping", "serving", "stopping", "dead"])

    def test_refused_preflight_only_from_preflight(self):
        d = self._boot()
        sf.transition(d, "launching")
        sf.transition(d, "refused_preflight", cause=sf.make_cause("PRE_X", "preflight"))
        self.assertEqual(sf.read(d)["lifecycle"]["state"], "launching")
        self.assertFalse(sf.may_transition("loading", "launching"))
        self.assertTrue(sf.may_transition("preflight", "refused_preflight"))
        self.assertFalse(sf.may_transition("serving", "stopped_clean"))

    def test_stopped_clean_keeps_reason_with_rc0(self):
        d = self._boot(kind="d2")
        sf.transition(d, "stopping", cause=sf.make_cause("d2_done", "operator"))
        sf.transition(d, "stopped_clean")
        c = sf.read(d)["cause"]
        self.assertEqual((c["code"], c["rc"]), ("d2_done", 0))
        self.assertIn("probes_done", sf.STOP_REASONS)
        self.assertIn("host", sf.ORIGINS)


class TestOwnership(_Base):
    def test_fields_and_states_per_writer(self):
        d = self._boot()
        with self.assertRaises(sf.StateFileError):
            sf.transition(d, "serving", writer="launcher")
        with self.assertRaises(sf.StateFileError):
            sf.transition(d, None, fields={"tag": "t"}, writer="launcher")
        with self.assertRaises(sf.StateFileError):
            sf.transition(d, None, fields={"groups.P": {}}, writer="host")
        with self.assertRaises(sf.StateFileError):
            sf.transition(d, "dead", cause=sf.make_cause("X", "container_exit"), writer="launcher")
        sf.transition(d, None, fields={"groups.P.state": "ready", "invariants.W7": {"verdict": "pass"}},
                      writer="launcher")
        st = sf.read(d)
        self.assertEqual(st["groups"]["P"]["state"], "ready")
        self.assertIn("launcher", st["heartbeat"])

    def test_two_writers_lose_no_update(self):
        d = self._boot()
        ctx = multiprocessing.get_context("fork")
        ps = [ctx.Process(target=_bump, args=(d, "launcher", 40, "groups.P.n")),
              ctx.Process(target=_bump, args=(d, "host", 40, "gpuq_id"))]
        for p in ps:
            p.start()
        for p in ps:
            p.join(60)
            self.assertEqual(p.exitcode, 0)
        st = sf.read(d)
        self.assertEqual(st["seq"], 1 + 80)
        self.assertEqual((st["groups"]["P"]["n"], st["gpuq_id"]), (39, 39))
        self.assertFalse([f for f in os.listdir(d) if ".tmp." in f])


class TestHeartbeatStopPointer(_Base):
    def test_heartbeat_names_container_and_verdict(self):
        d = self._boot()
        sf.transition(d, None, heartbeat_only=True, container="dkr27bx")
        hb = sf.read(d)["heartbeat"]["host_acceptance"]
        self.assertEqual(hb["container"], "dkr27bx")
        self.assertIn("host_pid", hb)
        now = time.time()
        self.assertEqual(sf.heartbeat_verdict(hb, now), "alive")
        old = dict(hb, ts=now - 60)
        self.assertEqual(sf.heartbeat_verdict(old, now), "unknown")
        self.assertEqual(sf.heartbeat_verdict(old, now, container_running=False), "dead")

    def test_stop_request_is_dead_rc24(self):
        d = self._boot()
        sf.transition(d, "launching")
        sf.write_json_atomic(os.path.join(d, "stop_request.json"),
                             {"code": "GT_LIVELOCK_REFUSED", "origin": "rank", "group": "D", "rank": "tp1pp0",
                              "detail_full": "x"})
        sf.transition(d, "dead", cause=sf.stop_request(d))
        st = sf.read(d)
        self.assertEqual((st["cause"]["code"], st["cause"]["rank"]), ("GT_LIVELOCK_REFUSED", "tp1pp0"))
        self.assertEqual(sf.rc_of(st, False), sf.RC_STOPPED_BY_WATCHER)
        self.assertTrue(sf.counts_as_death(st))

    def test_pointer_per_kind_and_death_only_for_boot(self):
        b = self._boot(kind="boot", bid="b-boot")
        d2 = self._boot(kind="d2", bid="b-d2")
        self.assertEqual(os.readlink(os.path.join(self.root, "current")), "b-boot")
        self.assertEqual(os.readlink(os.path.join(self.root, "current-d2")), "b-d2")
        sf.transition(d2, "dead", cause=sf.make_cause("D2_RC_3", "launcher"))
        self.assertFalse(sf.counts_as_death(sf.read(d2)))
        self.assertFalse(sf.counts_as_death(sf.read(b)))

    def test_serving_since_set_once(self):
        d = self._boot()
        sf.transition(d, "launching")
        sf.transition(d, "serving")
        t0 = sf.read(d)["serving_since_ts"]
        sf.transition(d, "flipping")
        sf.transition(d, "serving")
        self.assertEqual(sf.read(d)["serving_since_ts"], t0)


class TestLauncherWrites(_Base):
    def setUp(self):
        super().setUp()
        from flliper.srt.environ import envs
        from flliper.srt.pdflip import launcher

        self.launcher = launcher
        self.d = self._boot()
        sf.transition(self.d, "launching")
        self._ov = envs.PDFLIP_STATE_DIR.override(self.d)
        self._ov.__enter__()
        self.lines = []

    def tearDown(self):
        self._ov.__exit__(None, None, None)
        super().tearDown()

    def _log(self, s):
        self.lines.append(s)

    def test_gate_writes_ranks_and_invariant_under_rankstate(self):
        L = self.launcher
        spec = L.GroupSpec("D", 0, ["true"], os.path.join(self.root, "boot.D.log"), {})
        sd = L.rank_state_dir_for(spec)
        self.assertEqual(sd, os.path.join(self.d, "rankstate", "D"))
        for tp, worker, rows in ((0, False, (64, 0, 40)), (1, True, (64, 40, 52)), (2, True, (64, 52, 64))):
            rs.write_rank_state(rs.build_rank_state(
                group="D", tp_rank=tp, tp_size=3, pp_rank=0, pp_size=1, form_a_worker=worker,
                canonical_on=True, canonical_kv_built=True, canonical_blob_built=not worker,
                has_mamba_pool=True, page_size=64, owner_ctx=rows, seq=1), sd)
        L.canonical_state_gate(spec, 3, self._log)
        st = sf.read(self.d)
        self.assertEqual(len(st["groups"]["D"]["ranks"]), 3)
        self.assertEqual(st["invariants"]["W7_W10_CanonicalWindow"]["D"]["verdict"], "pass")
        self.assertEqual(st["lifecycle"]["state"], "launching")

    def test_refusal_is_dead_with_code_and_name(self):
        L = self.launcher
        e = L.PdFlipLaunchRefused("W7 PdFlipMambaBlobAbsent / W10 PdFlipCanonicalPageMissing (launcher half): x")
        L.boot_state_write(self._log, "dead", cause=L.refusal_cause(e))
        c = sf.read(self.d)["cause"]
        self.assertEqual((c["code"], c["origin"], c["name"]), ("W7_PdFlipMambaBlobAbsent", "launcher", "PdFlipMambaBlobAbsent"))
        self.assertEqual(sf.read(self.d)["lifecycle"]["state"], "dead")

    def test_cli_writes_dead_on_refusal_and_crash_and_done_on_success(self):
        L = self.launcher
        orig = L.main
        try:
            L.main = lambda argv=None: (_ for _ in ()).throw(L.PdFlipLaunchRefused("W9 PdFlipKeySchemeSplit: x"))
            self.assertEqual(L.cli([]), 2)
            self.assertEqual(sf.read(self.d)["cause"]["code"], "W9_PdFlipKeySchemeSplit")
            d2 = sf.init(self.root, "b2", "boot", {})
            with __import__("flliper.srt.environ", fromlist=["envs"]).envs.PDFLIP_STATE_DIR.override(d2):
                L.main = lambda argv=None: (_ for _ in ()).throw(RuntimeError("boom"))
                with self.assertRaises(RuntimeError):
                    L.cli([])
                self.assertEqual(sf.read(d2)["cause"]["code"], "LAUNCHER_EXCEPTION_RuntimeError")
            d3 = sf.init(self.root, "b3", "boot", {})
            with __import__("flliper.srt.environ", fromlist=["envs"]).envs.PDFLIP_STATE_DIR.override(d3):
                L.main = lambda argv=None: 0
                self.assertEqual(L.cli([]), 0)
                self.assertEqual(sf.events(d3)[-1]["type"], "launcher_done")
        finally:
            L.main = orig

    def test_missing_state_json_writes_nothing(self):
        from flliper.srt.environ import envs

        empty = os.path.join(self.root, "nope")
        os.makedirs(empty)
        with envs.PDFLIP_STATE_DIR.override(empty):
            self.launcher.boot_state_write(self._log, "loading")
        self.assertFalse(os.path.exists(os.path.join(empty, "state.json")))
        self.assertTrue(any("state.json missing" in l for l in self.lines))


class TestLaunchSnapshot(_Base):
    """Nutzer 29.09.: groups.<G>.launch trägt argv + Schalter-Env (rigdash „aktiv“), additiv in pdflip.state/1."""

    def test_snapshot_keeps_switches_and_drops_secrets(self):
        snap = sf.launch_snapshot(
            ["python", "-m", "flliper.launch_server", "--d-kv-token-cut", "owned"],
            {"FLLIPER_PDFLIP_ENABLE_D_STORE_ADOPT": "1", "PDFLIP_STATE_DIR": "/x", "PATH": "/bin",
             "HOME": "/root", "FLLIPER_ADMIN_KEY": "geheim", "HF_TOKEN": "t", "NCCL_P2P_LEVEL": 2})
        self.assertEqual(snap["argv"][-2:], ["--d-kv-token-cut", "owned"])
        self.assertEqual(snap["env"], {"NCCL_P2P_LEVEL": "2", "FLLIPER_PDFLIP_ENABLE_D_STORE_ADOPT": "1",
                                       "PDFLIP_STATE_DIR": "/x"})

    def test_launcher_writes_launch_into_group_additively(self):
        d = self._boot()
        snap = sf.launch_snapshot(["a", "--flag"], {"FLLIPER_X": "1"})
        sf.transition(d, "loading", fields={"groups.P": {"state": "loading", "launch": snap}}, writer="launcher")
        st = sf.read(d)
        self.assertEqual(st["schema"], sf.STATE_SCHEMA)
        self.assertEqual(st["groups"]["P"]["launch"], {"argv": ["a", "--flag"], "env": {"FLLIPER_X": "1"}})


if __name__ == "__main__":
    unittest.main()
