"""Sampler in its own process (Nutzer 30.09. ~21:40Z: "der probenehmer sollte doch nicht an zu viel last
scheitern? der sollte das doch irgendwie parallel davon tun können?").

The sampler role writes each sample to the shared ring store, the web server's reader role takes the
ring from there (never reading the moment itself), the supervisor restarts a dead sampler and says so."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rigdash import ipcboot, sampler, server  # noqa: E402


def _boot(root, ts):
    d = os.path.join(root, "nfx-boot-20260930T153426Z-051f")
    os.makedirs(os.path.join(d, "rankstate", "D"), exist_ok=True)
    with open(os.path.join(d, "state.json"), "w") as fh:
        json.dump({"schema": "pdflip.state/1", "boot_id": os.path.basename(d), "kind": "boot", "tag": "nfx",
                   "lifecycle": {"state": "serving"}, "front": {"awake": "D", "queue": 0}}, fh)
    with open(os.path.join(d, "rankstate", "D", "D.tp0pp0.rankstats"), "w") as fh:
        json.dump({"schema": "pdflip.rankstats/1", "ts": ts, "decode": {"tokens": int(100 * ts), "running": 2,
                   "gpu_ms_by_bs": {"2": [int(50 * ts), 1000.0 * ts]}}, "prefill": {"chunks": 0, "new_tokens": 0},
                   "sched": {"full_token_usage": 0.5}}, fh)
    return os.path.basename(d)


class TestRingStoreRoles(unittest.TestCase):
    def test_sampler_writes_reader_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nf", "state")
            store = sampler.RingStore(os.path.join(tmp, "ring.sqlite"))
            w = ipcboot.IpcBoots(roots=(root,), store=store, role="sampler")
            r = ipcboot.IpcBoots(roots=(root,), store=sampler.RingStore(os.path.join(tmp, "ring.sqlite")), role="reader")
            now = time.time()
            for i in range(5):
                key = _boot(root, now + i)
                w.poll(now + i + 0.3)
            r.poll(now + 5.0)
            self.assertEqual(len(r.rings[key]), 5)                               # the reader's ring = the sampler's
            self.assertEqual([s["t"] for s in r.rings[key]], [s["t"] for s in w.rings[key]])
            self.assertEqual(r.rings[key][-1]["r"]["D.tp0pp0"]["dtok"], w.rings[key][-1]["r"]["D.tp0pp0"]["dtok"])
            self.assertIn(key, r.rank)                                           # fields still from the rank files
            n_before = len(r.rings[key])
            r.poll(now + 5.5)
            self.assertEqual(len(r.rings[key]), n_before)                       # incremental: nothing twice
            # the reader never samples: its own polls add no ring entry
            r2 = ipcboot.IpcBoots(roots=(root,), store=sampler.RingStore(os.path.join(tmp, "empty.sqlite")), role="reader")
            r2.poll(now + 6)
            self.assertEqual(dict(r2.rings), {})
            m = r.model(key)
            self.assertGreater(len(m.dec), 0)

    def test_store_trims_to_the_writers_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = sampler.RingStore(os.path.join(tmp, "ring.sqlite"))
            st.append([("a", 1.0, {"t": 1.0}), ("a", 2.0, {"t": 2.0}), ("b", 2.0, {"t": 2.0})], {"a": 1.0, "b": 2.0})
            st.append([("a", 3.0, {"t": 3.0})], {"a": 2.0})                      # b gone, a's oldest is 2
            self.assertEqual(st.extent(), {"a": 2.0})
            self.assertEqual([(k, t) for k, t, _ in st.since(0)], [("a", 2.0), ("a", 3.0)])


class TestSupervisor(unittest.TestCase):
    def test_restarts_a_dead_sampler_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = sampler.RingStore(os.path.join(tmp, "ring.sqlite"))
            sup = sampler.Supervisor([sys.executable, "-c", "import time; time.sleep(0.2)"], store=st)
            stop = threading.Event()
            th = threading.Thread(target=sup.run_forever, args=(stop,), daemon=True)
            th.start()
            t0 = time.time()
            while sup.restarts < 1 and time.time() - t0 < 10:
                time.sleep(0.1)
            s = sup.status()
            stop.set()
            th.join(10)
            self.assertGreaterEqual(s["restarts"], 1)
            self.assertFalse(s["ok"])                          # no heartbeat: shown as not ok, with the reason
            self.assertTrue(s["why"])
            self.assertEqual(s["server_pid"], os.getpid())
            self.assertNotEqual(s["pid"], os.getpid())

    def test_server_uses_a_process_with_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = server.main.__globals__["argparse"].Namespace(
                docker_ssh="", docker_host_prefix="", front=[], gpuq="http://127.0.0.1:9", state_dir=tmp,
                image_changes=os.path.join(tmp, "ic.json"), features=os.path.join(tmp, "f.json"),
                features_repo=tmp, release_profile=[], edition="rig", sampler=None)
            app = server.App(args)
            self.assertEqual(app.sampler_mode, "prozess")
            self.assertEqual(app.boots.role, "reader")
            self.assertIsNone(app.hist_rec)                  # no recorder thread in the web process
            self.assertIn("rigdash.sampler", " ".join(app.sup.cmd))


class TestWebServerMeasuresNothing(unittest.TestCase):
    """Folgeauftrag 30.09. ~22Z: sources (cards, docker, gpuq, fronts), the energy book and the rank files
    are the sampler's; the web server reads only what it published."""

    def test_reader_never_opens_a_rank_file(self):
        import builtins
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "nf", "state")
            store = sampler.RingStore(os.path.join(tmp, "ring.sqlite"))
            w = ipcboot.IpcBoots(roots=(root,), store=store, role="sampler")
            r = ipcboot.IpcBoots(roots=(root,), store=sampler.RingStore(os.path.join(tmp, "ring.sqlite")), role="reader")
            now = time.time()
            key = _boot(root, now)
            w.poll(now + 0.3)
            opened, real = [], builtins.open

            def spy(path, *a, **k):
                opened.append(str(path))
                return real(path, *a, **k)
            builtins.open = spy
            try:
                r.poll(now + 1.0)
            finally:
                builtins.open = real
            self.assertFalse([p for p in opened if p.endswith((".rankstats", ".json")) and "rankstate" in p])
            self.assertIn(key, r.rank)                               # the fields' rank dict came from the store
            self.assertEqual(r.rank[key]["rankstats"].keys(), w.rank[key]["rankstats"].keys())

    def test_sources_and_energy_readers(self):
        from rigdash import energy, sources
        with tempfile.TemporaryDirectory() as tmp:
            st = sampler.RingStore(os.path.join(tmp, "ring.sqlite"))
            rd = sources.SourcesReader(st)
            self.assertEqual(rd.view(), {})
            st.set_raw("sources", json.dumps({"t": time.time() - 2, "view": {"gpus": {"value": [{"index": 0}], "age_s": 0.5}},
                                              "gpu_series": {"t": [1.0], "power": [[100.0]], "mem": [[1.0]], "util": [[5]]}}))
            v = rd.view()
            self.assertEqual(v["gpus"]["value"], [{"index": 0}])
            self.assertGreaterEqual(v["gpus"]["age_s"], 2.4)       # aged by the time since publishing
            self.assertEqual(rd.gpu_series()["power"], [[100.0]])
            er = energy.EnergyReader(st)
            self.assertIsNone(er.view("x", 10.0))
            book = energy.EnergyBook(None)
            book.books["x"] = energy._blank()
            book.books["x"]["covered_s"] = 5.0
            st.set_raw("energy.books", json.dumps(book.books))
            self.assertEqual(er.view("x", 10.0)["covered_s"], 5.0)

    def test_server_wires_readers(self):
        from rigdash import energy, sources
        with tempfile.TemporaryDirectory() as tmp:
            args = server.main.__globals__["argparse"].Namespace(
                docker_ssh="", docker_host_prefix="/x", front=["http://127.0.0.1:30030"], gpuq="http://127.0.0.1:9",
                state_dir=tmp, image_changes=os.path.join(tmp, "ic.json"), features=os.path.join(tmp, "f.json"),
                features_repo=tmp, release_profile=[], edition="rig", sampler=None)
            app = server.App(args)
            self.assertIsInstance(app.src, sources.SourcesReader)
            self.assertIsInstance(app.energy, energy.EnergyReader)
            cmd = " ".join(app.sup.cmd)
            for w in ("--front http://127.0.0.1:30030", "--gpuq http://127.0.0.1:9", "--docker-host-prefix /x"):
                self.assertIn(w, cmd)

    def test_nvml_cards_power_from_energy(self):
        import types
        from rigdash import sources
        clock = {"e": 0.0}

        class U:
            gpu = 40

        class M:
            used, total = 1048576.0 * 1000, 1048576.0 * 20000
        fake = types.SimpleNamespace(
            nvmlInit=lambda: None, nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: i,
            nvmlDeviceGetName=lambda h: b"NVIDIA GeForce RTX 5090", nvmlDeviceGetUUID=lambda h: "GPU-x",
            nvmlDeviceGetTotalEnergyConsumption=lambda h: clock["e"], nvmlDeviceGetPowerUsage=lambda h: 123000,
            nvmlDeviceGetEnforcedPowerLimit=lambda h: 575000, nvmlDeviceGetMemoryInfo=lambda h: M,
            nvmlDeviceGetUtilizationRates=lambda h: U, nvmlDeviceGetTemperature=lambda h, k: 55,
            nvmlDeviceGetClockInfo=lambda h, k: 2400, NVML_TEMPERATURE_GPU=0, NVML_CLOCK_SM=1)
        old = sys.modules.get("pynvml")
        sys.modules["pynvml"] = fake
        try:
            s = sources.Sources({"state_dir": None})
            c0 = s._nvml_cards()[0]
            self.assertEqual((c0["power.draw"], c0["power_src"]), (123.0, "nvml-moment"))   # first reading
            t0 = s._e_prev[0][0]
            time.sleep(0.2)
            clock["e"] = 300.0 * 1000.0 * (time.time() - t0)                               # 300 W since then
            c1 = s._nvml_cards()[0]
            self.assertAlmostEqual(c1["power.draw"], 300.0, delta=15)
            self.assertEqual(c1["power_src"], "nvml-energie")
            self.assertEqual((c1["name"], c1["power.limit"], c1["memory.used"], c1["temperature.gpu"]),
                             ("NVIDIA GeForce RTX 5090", 575.0, 1000.0, 55))
        finally:
            if old is not None:
                sys.modules["pynvml"] = old
            else:
                sys.modules.pop("pynvml", None)


if __name__ == "__main__":
    unittest.main()
