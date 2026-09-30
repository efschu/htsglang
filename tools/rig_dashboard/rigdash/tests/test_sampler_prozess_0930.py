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
        json.dump({"schema": "weg2.state/1", "boot_id": os.path.basename(d), "kind": "boot", "tag": "nfx",
                   "lifecycle": {"state": "serving"}, "front": {"awake": "D", "queue": 0}}, fh)
    with open(os.path.join(d, "rankstate", "D", "D.tp0pp0.rankstats"), "w") as fh:
        json.dump({"schema": "weg2.rankstats/1", "ts": ts, "decode": {"tokens": int(100 * ts), "running": 2,
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


if __name__ == "__main__":
    unittest.main()
