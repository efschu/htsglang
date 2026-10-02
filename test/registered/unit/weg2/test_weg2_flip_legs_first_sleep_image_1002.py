# SPDX-License-Identifier: Apache-2.0
"""FLIP-LEGS 02.10.: the first-sleep dormant image leaves the flip.

MEASURED (N4p f405217a61 1002_095319, N4q 58361d5471 1002_101341, front logs;
VM weg2_flip_user_view_ms): the user's flip total exceeds vorlauf + layer +
nachlauf by 1.49/1.52 s on the FIRST D->P and by 1.03/1.06 s on the FIRST P->D
of each boot -- and by ~0 on every later flip. FLIP-TIMELINE puts the gap
between the kv wake and ``done``: ``wake-kv@2992 dc@4511`` (first D->P) against
``wake-kv@1785 dc@1808`` (a later one). ``layer`` (flip_ms) ends at the kv
wake's answer, ``done`` after the dc reading -- which on a group's FIRST sleep
also takes the dormant-image sample: ``/proc/<pid>/smaps`` of every process of
the group (host_ledger.image_shmem_bytes), awaited by the flip although nothing
in this front gates on it (the record is the next boot's ledger input).

Pinned here, each red on the tree before this change:

1. under H78 (SGLANG_WEG2_DC_OFF_PATH=1) the flip awaits W19's residue reading
   (ps + nvidia-smi) and NOT the image: a slow image stand-in does not lengthen
   the flip; the record lands afterwards, stamped with the flip's epoch and the
   load witness of that moment, exactly as the awaited form stamped it;
2. a flip that WAKES the sampled group settles the sample before its legs (the
   image is read from a sleeping group, never from a waking one);
3. SGLANG_WEG2_DC_IMAGE_OFF_FLIP=0 restores the awaited form;
4. W19 still stops the first D sleep before the flip closes.

Hermetic: no GPU, no boot; group RPCs stubbed (the H78 suite's harness).
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from sglang.srt.weg2 import front as F
from sglang.srt.weg2 import host_ledger as hl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MIB = 1024 * 1024
ENV_DC = "SGLANG_WEG2_DC_OFF_PATH"
ENV_IMG = "SGLANG_WEG2_DC_IMAGE_OFF_FLIP"
CARD = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
IMAGE_DELAY = 0.8      # stand-in for the smaps walk (measured 1.0-1.5 s)
READ_DELAY = 0.05      # stand-in for ps / nvidia-smi


class _Env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.saved = {k: os.environ.pop(k, None) for k in self.kv}
        for k, v in self.kv.items():
            if v is not None:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k in self.kv:
            os.environ.pop(k, None)
            if self.saved[k] is not None:
                os.environ[k] = self.saved[k]
        return False


def _front(path: str, image_off: str | None = None, dc_reserve=None, log=None):
    with _Env(**{ENV_DC: "1", ENV_IMG: image_off}):
        f = F.Front(prefill="http://p", decode="http://d", awake="D", tag="legs1002",
                    store_dir="/tmp", prefill_sid=11, decode_sid=22,
                    dc_reserve=dict(dc_reserve or {CARD: 2000}), w_s=45.0, weight_chunks=2,
                    flip_min_work_tokens=1, measured_record=path, commit="legs")
    f.stops = []
    log = log if log is not None else []

    async def rpc(g, path_, body, timeout):
        if path_ == "/flush_cache":
            return 200, "{}"
        tags = tuple((body or {}).get("tags", ()))
        log.append((time.monotonic(), g.name if hasattr(g, "name") else str(g), path_, tags))
        await asyncio.sleep(0.001)
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})

    f.rpc = rpc
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    f.resolve_x_live = lambda: None
    return f


class _Base(CustomTestCase):
    def setUp(self):
        self._saved = (F._nvml_free, F._session_pids, F._nvml_process_mib)
        F._nvml_free = lambda: []
        self.image_done = []

        def pids(sid):
            time.sleep(READ_DELAY)
            return {101, 102}

        def nvml(p):
            time.sleep(READ_DELAY)
            return {CARD: self.residue}

        F._session_pids = pids
        F._nvml_process_mib = nvml
        self.residue = 1700
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "weg2_measured_record.json")
        real = hl.image_shmem_bytes

        def slow_image(p, reader=None):
            time.sleep(IMAGE_DELAY)
            self.image_done.append(time.monotonic())
            return 7 * (1 << 30), sorted(p)

        self._p = mock.patch.object(hl, "image_shmem_bytes", side_effect=slow_image)
        self._p.start()
        self._real_image = real
        # the live cgroup readings, fixed: two runs compare their records
        self._cg = mock.patch.object(hl, "read_cgroup", return_value={
            "current": 80 << 30, "reclaimable": 10 << 30})
        self._sh = mock.patch.object(hl, "read_cgroup_shmem_bytes", return_value=40 << 30)
        self._cg.start()
        self._sh.start()

    def tearDown(self):
        self._sh.stop()
        self._cg.stop()
        self._p.stop()
        F._nvml_free, F._session_pids, F._nvml_process_mib = self._saved
        self._tmp.cleanup()


class TheFirstSleepImageLeavesTheFlip(_Base):
    def test_switch_parse(self):
        self.assertTrue(F.dc_image_off_flip_on({}))
        self.assertTrue(F.dc_image_off_flip_on({ENV_IMG: "1"}))
        for off in ("0", "false", "no", "off", " OFF "):
            self.assertFalse(F.dc_image_off_flip_on({ENV_IMG: off}), off)

    def test_the_first_flip_does_not_await_the_image(self):
        """RED before: the flip awaited the smaps walk (IMAGE_DELAY on every first sleep)."""
        async def body():
            f = _front(self.path)
            t0 = time.monotonic()
            await f.flip("D", "P")
            wall = time.monotonic() - t0
            pending_after_flip = "D" in getattr(f, "_dormant_image_pending", {})
            await asyncio.wait(list(f._dormant_image_pending.values()) or [asyncio.sleep(0)])
            await f._sidecar_writes_settled()
            return f, wall, pending_after_flip

        f, wall, pending = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertLess(wall, IMAGE_DELAY * 0.75, f"the flip still awaited the image ({wall:.3f} s)")
        self.assertTrue(pending, "the image was not in flight when the flip returned")
        self.assertEqual(f.awake, "P")
        self.assertEqual(f.dc_measured_d, {CARD: 1700})   # W19's reading stayed on the flip
        rec = f.dormant_image["D"]
        self.assertEqual(rec["sampled_at_flip_epoch"], 0)  # the flip's epoch, not the one after
        self.assertEqual(rec["vram_residue_mib"], {CARD: 1700})
        back = hl.read_measured_record(self.path)
        self.assertEqual(back["D"]["boot_tag"], "legs1002")

    def test_the_record_equals_the_awaited_form(self):
        """Same record fields as the awaited form (apart from the wall-clock stamp)."""
        def run(image_off):
            async def body():
                f = _front(self.path, image_off=image_off)
                await f.flip("D", "P")
                for t in list(getattr(f, "_dormant_image_pending", {}).values()):
                    await t
                await f._sidecar_writes_settled()
                return f
            return asyncio.run(body()).dormant_image["D"]

        a, b = run("0"), run(None)
        skip = {"at"}
        self.assertEqual({k: v for k, v in a.items() if k not in skip},
                         {k: v for k, v in b.items() if k not in skip})

    def test_a_waking_flip_settles_the_sample_before_its_legs(self):
        """Back to back: D's image must be read before D's legs wake it."""
        log = []

        async def body():
            f = _front(self.path, log=log)
            await f.flip("D", "P")
            t_back = time.monotonic()
            await f.flip("P", "D")
            await f._sidecar_writes_settled()
            return f, t_back

        f, t_back = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertEqual(len(self.image_done), 2, self.image_done)   # D's image, then P's
        d_image_end = self.image_done[0]
        d_weight_wakes = [t for t, g, p, tags in log
                          if p == "/resume_memory_occupation" and t >= t_back
                          and any(str(x).startswith("weights") for x in tags)]
        self.assertTrue(d_weight_wakes, log)
        self.assertLess(d_image_end, min(d_weight_wakes),
                        "D's legs started while its first-sleep image was still reading")
        self.assertIn("D", f.dormant_image)
        self.assertIn("P", f.dormant_image)

    def test_switch_off_keeps_the_awaited_form(self):
        async def body():
            f = _front(self.path, image_off="0")
            t0 = time.monotonic()
            await f.flip("D", "P")
            return f, time.monotonic() - t0

        f, wall = asyncio.run(body())
        self.assertEqual(f.stops, [])
        self.assertGreater(wall, IMAGE_DELAY - 0.05, wall)
        self.assertIn("D", f.dormant_image)
        self.assertFalse(getattr(f, "_dormant_image_pending", {}))

    def test_w19_still_stops_the_first_d_sleep_before_it_closes(self):
        self.residue = 2600

        async def body():
            f = _front(self.path)
            await f.flip("D", "P")
            for t in list(getattr(f, "_dormant_image_pending", {}).values()):
                await t
            return f

        f = asyncio.run(body())
        self.assertEqual([n for n, _ in f.stops], ["W19 DormantResidueRefused"], f.stops)
        self.assertEqual(f.awake, "D")

    def test_cleanup_lands_a_pending_image(self):
        async def body():
            f = _front(self.path)
            await f.flip("D", "P")
            await f.cleanup({})
            return f

        f = asyncio.run(body())
        self.assertEqual(hl.read_measured_record(self.path)["D"]["boot_tag"], "legs1002")


if __name__ == "__main__":
    unittest.main()
