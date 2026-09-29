"""DASHBOARD-AUS-IPC (29.09.): the rigdash is fed from the boot's state directory,
not from log lines (user order via 27B: "das dashboard soll auch aus der inter
prozess kommunikation gespeist werden, nicht aus logs").

(a) the front writes each flip as events -- ``flip_begin``, ``flip_done`` (the
    flip_log record without the chunk list) and ``flip_first_work``: the woken
    group's first work, timed on the front's ONE clock from the flip's begin
    (P->D: the first streamed content = first decode token). No model switch:
    the 27B boot writes the same events.
(b) the front writes its own keys under state.json ``front`` through state_file
    (writer front); the host beat writes only its five mirror keys, dotted, so
    it no longer erases the front's keys every 5 s.
(e) the launcher records the boot form handed to each group as
    ``groups.<G>.form``.

RED on 50ae2014b0: publish_event / publish_front_fields / FirstWorkClock /
flip_done_payload / Front._ipc_* / launcher.group_form_block do not exist, and
the host beat replaces the whole ``front`` object.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import front_state_ipc as fsi
from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import state_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MIB = 1 << 20
GIB = 1 << 30
Z30U = {"file_gib": 55.17, "shmem_gib": 53.84, "current_gib": 81.00,
        "nonreclaim_gib": 79.68, "memfree_gib": 2.40}


def _boot(root, bid="nfdash-boot-20260929T120000Z-beef"):
    return state_file.init(root, bid, "boot", {})


def _of(sd, typ):
    return [e for e in state_file.events(sd) if e["type"] == typ]


class TestPublish(CustomTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="dashipc-")

    def test_event_and_front_fields_go_through_the_one_writer(self):
        sd = _boot(self.root)
        self.assertTrue(fsi.publish_event(sd, "flip_begin", {"sleep": "P", "wake": "D"}))
        ev = _of(sd, "flip_begin")
        self.assertEqual(len(ev), 1)
        self.assertIn(state_file.HEARTBEAT_NAME["front"], state_file.read(sd)["heartbeat"])
        self.assertTrue(fsi.publish_front_fields(sd, {"served": {"P": 3, "D": 2}}))
        self.assertEqual(state_file.read(sd)["front"]["served"], {"P": 3, "D": 2})

    def test_host_mirror_keys_are_refused(self):
        sd = _boot(self.root)
        for k in fsi.HOST_FRONT_KEYS:
            with self.assertRaises(ValueError):
                fsi.publish_front_fields(sd, {k: 1})

    def test_no_state_json_writes_nothing(self):
        empty = os.path.join(self.root, "none")
        os.makedirs(empty)
        self.assertFalse(fsi.publish_event(empty, "flip_begin", {}))
        self.assertFalse(fsi.publish_front_fields(empty, {"served": {}}))
        self.assertFalse(fsi.publish_event("", "flip_begin", {}))
        self.assertEqual(os.listdir(empty), [])

    def test_host_beat_keeps_the_fronts_own_keys(self):
        """RED before: beat wrote fields={"front": {...five keys...}} and erased
        every key the front had written under `front` on each beat."""
        sd = _boot(self.root)
        # written as the front writes it, straight through state_file (the base has it)
        state_file.transition(sd, None, fields={"front.served_tokens": {"D": {"n": 1}}}, writer="front")
        rc = state_file.main(["beat", "--dir", sd, "--front-json", json.dumps(
            {"state": "serving", "epoch": 7, "awake": "D", "queue": 2, "outstanding": {"P": 1, "D": 2}})])
        self.assertIn(rc, (0, None))
        fr = state_file.read(sd)["front"]
        self.assertEqual(fr["served_tokens"], {"D": {"n": 1}})
        self.assertEqual((fr["state"], fr["epoch"], fr["awake"], fr["queue"], fr["outstanding"]),
                         ("serving", 7, "D", 2, 3))

    def test_flip_done_payload_stays_one_valid_event_line(self):
        rec = {"epoch": 4, "sleep": "P", "wake": "D", "flip_ms": 2010,
               "chunks": [{"tag": f"t{i}", "bytes": i * MIB, "ms": 1.5} for i in range(2000)]}
        out = fsi.flip_done_payload(rec, 1000.0)
        self.assertNotIn("chunks", out)
        self.assertEqual((out["chunks_n"], out["flip_begin_ts"]), (2000, 1000.0))
        self.assertIn("chunks", rec)
        sd = _boot(self.root)
        fsi.publish_event(sd, "flip_done", out)
        with open(os.path.join(sd, "events.jsonl"), "rb") as fh:
            lines = fh.read().splitlines()
        self.assertLess(len(lines[-1]), state_file.EVENT_MAX)
        self.assertEqual(json.loads(lines[-1])["data"]["flip_ms"], 2010)


class TestFirstWorkClock(CustomTestCase):
    def test_fires_once_for_the_woken_group_only(self):
        c = fsi.FirstWorkClock()
        self.assertIsNone(c.seen("D", "decode_token", "r0", 5.0))  # nothing armed
        c.arm(3, "P", "D", 100.0)
        self.assertIsNone(c.seen("P", "p_leg1_dispatch", "r1", 100.5))  # the sleeping group
        ev = c.seen("D", "decode_token", "r2", 102.25)
        self.assertEqual((ev["epoch"], ev["dir"], ev["flip_time_ms"], ev["rid"]), (3, "P>D", 2250, "r2"))
        self.assertEqual(ev["clock"], "time.time front")
        self.assertIsNone(c.seen("D", "decode_token", "r3", 103.0))  # once per flip

    def test_next_flip_rearms(self):
        c = fsi.FirstWorkClock()
        c.arm(1, "P", "D", 10.0)
        c.arm(2, "D", "P", 20.0)  # D never worked before the next flip
        self.assertIsNone(c.seen("D", "decode_token", "r", 21.0))
        self.assertEqual(c.seen("P", "p_leg1_dispatch", "r", 21.5)["dir"], "D>P")


def _front():
    return front_mod.Front(
        prefill="http://p", decode="http://d", awake="D", tag="dashipc",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1)


async def _rpc(g, path, body, timeout):
    if path == "/flush_cache":
        return 200, "{}"
    tags = tuple((body or {}).get("tags", ()))
    return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                            "critical_path": "rank=0 card=GPU-x ms=1"})


class TestFrontWrites(CustomTestCase):
    """The real Front.flip and the real leg-1 dispatch hook, no switch set:
    only WEG2_STATE_DIR, which the launcher sets for every boot (27B and NF)."""

    def _wait(self, sd, typ, n=1):
        deadline = time.time() + 3.0
        while time.time() < deadline and len(_of(sd, typ)) < n:
            time.sleep(0.02)
        return _of(sd, typ)

    def test_flip_writes_begin_done_and_first_work_once(self):
        sd = _boot(tempfile.mkdtemp(prefix="dashipc-f-"))
        f = _front()
        f.rpc = _rpc
        f.p_leg1_stall_s = 0.0
        p = front_mod.Pending(rid="weg2-1-1", path="/generate", payload={}, text="x",
                              t_arrive=time.time(), fut=None)

        async def post():
            return 200, b"{}"

        async def body():
            await f.flip("D", "P")
            await f._leg1_bounded(p, post())
            await f._leg1_bounded(p, post())

        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}), \
                mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=dict(Z30U)), \
                mock.patch.object(host_ledger, "read_cgroup", return_value={"max": 84 * GIB}):
            asyncio.run(body())
        begin, done = self._wait(sd, "flip_begin"), self._wait(sd, "flip_done")
        fw = self._wait(sd, "flip_first_work")
        time.sleep(0.2)
        self.assertEqual((len(begin), len(done), len(_of(sd, "flip_first_work"))), (1, 1, 1))
        ts = begin[0]["data"]["flip_begin_ts"]
        self.assertEqual(done[0]["data"]["flip_begin_ts"], ts)
        w = fw[0]["data"]
        self.assertEqual((w["dir"], w["what"], w["rid"], w["flip_begin_ts"]),
                         ("D>P", "p_leg1_dispatch", "weg2-1-1", ts))
        self.assertEqual(w["epoch"], done[0]["data"]["epoch"])
        self.assertGreaterEqual(w["flip_time_ms"], 0)
        self.assertNotIn("chunks", done[0]["data"])

    def test_front_writer_publishes_own_keys_only(self):
        sd = _boot(tempfile.mkdtemp(prefix="dashipc-w-"))
        f = _front()
        f._ipc_note_served("D", 1000, 800, 50)
        f._ipc_note_served("D", 10, 0, 5)
        fields = f._ipc_front_fields()
        self.assertFalse(set(fields) & set(fsi.HOST_FRONT_KEYS))
        self.assertEqual(fields["served_tokens"]["D"],
                         {"n": 2, "prompt": 1010, "cached": 800, "completion": 55})

        async def body():
            t = asyncio.create_task(f.ipc_front_writer(period_s=0.01))
            await asyncio.sleep(0.1)
            t.cancel()

        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}):
            asyncio.run(body())
        deadline = time.time() + 3.0
        while time.time() < deadline and "served_tokens" not in (state_file.read(sd).get("front") or {}):
            time.sleep(0.02)
        fr = state_file.read(sd)["front"]
        self.assertEqual(fr["served_tokens"]["D"]["completion"], 55)
        self.assertIn("groups", fr)

    def test_front_writer_without_state_dir_ends_at_once(self):
        f = _front()
        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": ""}):
            asyncio.run(asyncio.wait_for(f.ipc_front_writer(period_s=0.01), 1.0))
        self.assertIsNone(f.__dict__.get("_ipc_pool"))


class TestLauncherForm(CustomTestCase):
    def test_group_form_block_from_the_published_form(self):
        from sglang.srt.weg2 import form as weg2_form
        from sglang.srt.weg2 import launcher

        fm = weg2_form.Weg2Form(arch="moe", experts="offload", draft="mtp", p_draft="cold",
                                kv="qsa_forma_dcp", flip="family", vision="off",
                                profile="nextflash", model="Qwen3.8-Flash-Next")
        blk = launcher.group_form_block({weg2_form.FORM_ENV: fm.env_value()})
        self.assertEqual((blk["arch"], blk["kv"], blk["profile"], blk["model"]),
                         ("moe", "qsa_forma_dcp", "nextflash", "Qwen3.8-Flash-Next"))
        self.assertEqual(blk["describe"], fm.describe())
        self.assertIsNone(launcher.group_form_block({}))

    def test_launch_group_writes_the_form_into_state_json(self):
        from sglang.srt.environ import envs
        from sglang.srt.weg2 import form as weg2_form
        from sglang.srt.weg2 import launcher as L

        root = tempfile.mkdtemp(prefix="dashipc-l-")
        sd = _boot(root)
        state_file.transition(sd, "launching")
        fm = weg2_form.Weg2Form(arch="dense", experts="none", draft="dflash", p_draft="compute",
                                kv="paged_dcp", flip="family", vision="off", profile="27b", model="m")
        spec = L.GroupSpec("P", 0, ["true"], os.path.join(root, "boot.P.log"),
                           {weg2_form.FORM_ENV: fm.env_value()})

        class _P:
            pid = 4242

        with envs.WEG2_STATE_DIR.override(sd), mock.patch.object(L.subprocess, "Popen", return_value=_P()):
            L.launch_group(spec, root, lambda s: None, False)
        g = state_file.read(sd)["groups"]["P"]
        self.assertEqual(g["pids"], [4242])
        self.assertEqual((g["form"]["arch"], g["form"]["profile"]), ("dense", "27b"))


if __name__ == "__main__":
    unittest.main()
