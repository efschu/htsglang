"""Dashboard Flipzeit 06.10. (Nutzerentscheid ~14:15Z, "einfach IDLE"): the front writes an ``idle_flip`` marker
into the ``flip_begin`` event of a P->D flip when nothing is waiting for D at the begin -- the mirror of the D->P
``idle_flip`` (DpFlipClock.begin: no waiter, no park).  The dashboard does not count such a flip as a Flipzeit.
Marker only: the flip itself is unchanged.  RED on 173161c595: ``pd_idle_flip`` does not exist, ``flip_begin`` has no
``idle_flip``."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from flliper.srt.pdflip import front as front_mod
from flliper.srt.pdflip import front_state_ipc as fsi
from flliper.srt.pdflip import host_ledger, state_file
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MIB = 1 << 20
GIB = 1 << 30
Z30U = {"file_gib": 55.17, "shmem_gib": 53.84, "current_gib": 81.00,
        "nonreclaim_gib": 79.68, "memfree_gib": 2.40}


def _front(awake):
    return front_mod.Front(
        prefill="http://p", decode="http://d", awake=awake, tag="dashidle",
        store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
        weight_chunks=2, flip_min_work_tokens=1)


async def _rpc(g, path, body, timeout):
    if path == "/flush_cache":
        return 200, "{}"
    tags = tuple((body or {}).get("tags", ()))
    return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                            "critical_path": "rank=0 card=GPU-x ms=1"})


def _begin(awake, src, dst, ready=0):
    sd = state_file.init(tempfile.mkdtemp(prefix="dashidle-"), "nfidle-boot-20260610T120000Z-beef", "boot", {})
    f = _front(awake)
    f.rpc = _rpc
    f.p_leg1_stall_s = 0.0
    for i in range(ready):       # prefilled requests waiting for D
        f._ready_for_d.append(front_mod.Pending(rid="pdflip-1-%d" % i, path="/generate", payload={}, text="x",
                                                 t_arrive=time.time(), fut=None))

    async def body():
        await f.flip(src, dst)
        await asyncio.sleep(0.05)

    with mock.patch.dict(os.environ, {"PDFLIP_STATE_DIR": sd}), \
            mock.patch.object(front_mod.Front, "_p_beacons", lambda self: {}), \
            mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=dict(Z30U)), \
            mock.patch.object(host_ledger, "read_cgroup", return_value={"max": 84 * GIB}):
        asyncio.run(body())
    deadline = time.time() + 3.0
    ev = []
    while time.time() < deadline and not ev:
        ev = [e for e in state_file.events(sd) if e["type"] == "flip_begin"]
        time.sleep(0.02)
    return ev[0]["data"]


class TestPureRule(unittest.TestCase):
    def test_idle_only_when_nothing_waits_for_d(self):
        self.assertTrue(fsi.pd_idle_flip(0, 0, 0, 0))
        for args in ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)):
            self.assertFalse(fsi.pd_idle_flip(*args), args)


class TestFrontWritesTheMarker(CustomTestCase):
    def test_pd_flip_with_nothing_for_d_is_marked_idle(self):
        d = _begin("P", "P", "D")
        self.assertEqual((d["sleep"], d["wake"]), ("P", "D"))
        self.assertIs(d["idle_flip"], True)

    def test_pd_flip_with_a_prefilled_request_for_d_is_not_idle(self):
        d = _begin("P", "P", "D", ready=1)
        self.assertIs(d["idle_flip"], False)

    def test_dp_flip_begin_carries_no_marker(self):
        d = _begin("D", "D", "P")
        self.assertNotIn("idle_flip", d)          # D->P keeps its marker in flip_user_time


if __name__ == "__main__":
    unittest.main()
