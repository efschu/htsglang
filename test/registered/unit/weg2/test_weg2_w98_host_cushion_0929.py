# SPDX-License-Identifier: Apache-2.0
"""W98 host cushion, part 2 (boot z30u, 2026-09-29): what 2b kept, and the cushion per flip.

MEASURED (z30u logs, same prefix ..._4f23714e2d_0929_065429):
  * P (SHARED_CACHE=keep) kept 3057 + 1914 + 1731 = 6702 MiB of non-expert
    ranges; D (=drop) advised 4225 + 173 + 122 + 122 = 4642 MiB away and found
    only 1746 MiB of them still cached (hit_mib). Nobody ever let go of the
    ~2 GiB of kept ranges D does not read.
  * mem csv: at 06:58:18 (D loaded) the cgroup's whole non-shmem page cache
    was 0.50 GiB -- the 84 GiB memory.max had already reclaimed most of it --
    and W98 fired at 07:14:10 on cushion 1.33 with nonreclaim 79.68 of 84.

Guarded here:
  * keep writes a manifest of exactly the ranges it kept; the launcher drops
    exactly those after D is ready (never in the load path), each byte once,
    and says advised vs evicted (mincore) plus memory.current before/after;
  * every completed flip writes one WEG2-FLIP-CUSHION line and one
    ``flip_cushion`` event -- the course of the cushion without a W98;
  * the derived ``absorb`` never invents a zero for an unreadable term.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from safetensors.torch import save_file

from sglang.srt.model_loader import weight_utils as W
from sglang.srt.weg2 import flip_cushion as fc
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import shared_cache_release as scr
from sglang.srt.weg2 import state_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MIB = 1 << 20
GIB = 1 << 30


def _mixed_file(root):
    g = torch.Generator().manual_seed(7)
    d = {}
    for L in (0, 1):
        for e in range(8):
            d[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight_packed"] = torch.randint(
                -2**31, 2**31 - 1, (64, 128), dtype=torch.int32, generator=g)
        d[f"model.layers.{L}.self_attn.q_proj.weight"] = torch.randn(256, 64, generator=g).to(torch.bfloat16)
    d["model.embed_tokens.weight"] = torch.randn(512, 64, generator=g).to(torch.bfloat16)
    path = os.path.join(root, "model-00001-of-00001.safetensors")
    save_file(d, path)
    return path


def _collect(paths, mode, manifest_dir):
    env = {W.COALESCE_ENV: "8", "SGLANG_WEIGHT_LOADER_SHARED_CACHE": mode,
           scr.MANIFEST_ENV: manifest_dir}
    with mock.patch.dict(os.environ, env, clear=False):
        return list(W.pread_safetensors_stream(paths, None, direct_io=False, workers=2, log=False))


class SharedCacheRelease(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = _mixed_file(self.tmp.name)
        self.mdir = os.path.join(self.tmp.name, "boot.shared_cache")

    def tearDown(self):
        self.tmp.cleanup()

    def test_keep_writes_the_manifest_of_exactly_what_it_kept(self):
        """RED before: keep recorded nothing, so the ranges D never reads (z30u:
        ~2 GiB of 6702 MiB) had no one to release them."""
        _collect([self.path], "keep", self.mdir)
        ranges, kept, n = scr.read_manifests(self.mdir)
        self.assertEqual(n, 1)
        self.assertGreater(kept, 0)
        self.assertEqual(sum(ln for _, _, ln in ranges), kept)
        # only non-expert (2b) bytes: a kept range reaches into an expert tensor
        # by at most the read alignment (the run is read from an aligned offset)
        header, base = W.read_safetensors_header(self.path)
        expert = [(base + int(i["data_offsets"][0]), base + int(i["data_offsets"][1]))
                  for k, i in header.items() if not W.is_shared_cache_tensor(k)]
        for _p, off, ln in ranges:
            for a, b in expert:
                overlap = min(b, off + ln) - max(a, off)
                self.assertLess(overlap, W._DIRECT_ALIGN, "expert bytes in the keep manifest")
        # drop (D) writes none, and "off" neither
        os.remove(os.path.join(self.mdir, os.listdir(self.mdir)[0]))
        _collect([self.path], "drop", self.mdir)
        _collect([self.path], "", self.mdir)
        self.assertEqual(scr.read_manifests(self.mdir)[2], 0)

    def test_release_advises_every_kept_byte_once_and_measures_it(self):
        """Two P ranks that kept overlapping ranges (embed on two PP stages):
        advised counts each byte once, kept counts what each rank said."""
        os.makedirs(self.mdir)
        scr.write_keep_manifest(self.mdir, [(self.path, 0, 8192), (self.path, 4096, 8192)], 16384)
        scr.write_keep_manifest(self.mdir, [(self.path, 0, 4096),
                                            ("/nonexistent/gone.safetensors", 0, 4096)], 8192)
        reads = iter([{"current_gib": 81.0, "file_gib": 55.17, "shmem_gib": 53.84},
                      {"current_gib": 80.9, "file_gib": 55.07, "shmem_gib": 53.84}])
        rec = scr.release(self.mdir, read_pressure=lambda: next(reads))
        self.assertEqual(rec["manifests"], 2)
        self.assertEqual(rec["kept_mib"], round(24576 / MIB))
        self.assertEqual(rec["files_missing"], 1)
        merged = scr.merge_ranges([(self.path, 0, 8192), (self.path, 4096, 8192), (self.path, 0, 4096)])
        self.assertEqual(merged[self.path], [(0, 12288)])
        self.assertEqual(rec["advised_mib"], round(12288 / MIB))
        self.assertGreaterEqual(rec["evicted_mib"], 0)
        self.assertLessEqual(rec["resident_after_mib"], rec["resident_before_mib"])
        self.assertEqual((rec["cg_current_before_gib"], rec["cg_current_after_gib"]), (81.0, 80.9))
        self.assertEqual(rec["cushion_before_gib"], 1.33)
        self.assertIn("advised_mib=", scr.line(rec))
        # no manifest (2b off) = no release, no line
        self.assertIsNone(scr.release(os.path.join(self.tmp.name, "empty")))

    def test_the_launcher_releases_only_after_d_is_ready(self):
        """Boot time: the release runs after D's wait_ready, never in the load
        path; and it writes the event through the ONE state writer."""
        from sglang.srt.weg2 import launcher as L

        src = inspect.getsource(L)
        ready = src.rindex('state.t_ready["D"] = wait_ready(PORT_D, spec_d.pid')
        call = src.index("release_shared_cache(spec_d, log)")
        self.assertLess(ready, call)
        self.assertLess(call - ready, 400, "the release must follow D's ready directly")
        # the group env names ONE directory for P and D
        self.assertEqual(scr.manifest_dir_for("/e/boot_weg2_x.P.log"),
                         scr.manifest_dir_for("/e/boot_weg2_x.D.log"))

        root = tempfile.mkdtemp(prefix="w98scr-")
        sd = state_file.init(root, "nfw98-boot-20260929T065429Z-abcd", "boot", {})
        glog = os.path.join(self.tmp.name, "boot_weg2_x.D.log")
        os.makedirs(scr.manifest_dir_for(glog))
        scr.write_keep_manifest(scr.manifest_dir_for(glog), [(self.path, 0, 4096)], 4096)
        lines = []

        class _Spec:
            log = glog

        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}):
            rec = L.release_shared_cache(_Spec(), lines.append)
        self.assertIsNotNone(rec)
        self.assertTrue(any(scr.MARKER in l for l in lines), lines)
        ev = [e for e in state_file.events(sd) if e["type"] == "shared_cache_release"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["data"]["advised_mib"], rec["advised_mib"])


# the z30u reading at 07:14:10 (mem csv): file 55.17, shmem 53.84, current 81.00
Z30U = {"file_gib": 55.17, "shmem_gib": 53.84, "current_gib": 81.00,
        "nonreclaim_gib": 79.68, "memfree_gib": 2.40}


class FlipCushion(CustomTestCase):
    def test_absorb_takes_the_smaller_pool_and_never_invents_a_zero(self):
        w = fc.FlipCushionWindow(84.0, host_ledger.RATE_LATCH_CUSHION_FLOOR_GIB)
        w.open(epoch=7, src="P", dst="D", pr=dict(Z30U, file_gib=55.40))
        w.note(Z30U)
        rec = w.close(dict(Z30U, shmem_gib=53.96, file_gib=55.23))
        self.assertEqual(rec["cushion_min_gib"], 1.27)
        self.assertEqual(rec["cg_room_min_gib"], 3.0)
        # absorb = cushion + min(MemFree 2.40, room 3.00): the host pool binds here
        self.assertAlmostEqual(rec["absorb_min_gib"], 1.27 + 2.40, places=2)
        self.assertEqual(rec["below_floor_samples"], 2)
        self.assertAlmostEqual(rec["shmem_delta_gib"], 0.12, places=3)
        # the cgroup room binds when the host has free pages (the Docker ceiling)
        self.assertAlmostEqual(fc.absorb_gib(1.33, 16.66, 3.0), 4.33, places=2)
        # an unreadable cushion is None, never "no cushion"
        self.assertIsNone(fc.absorb_gib(None, 16.66, 3.0))
        self.assertIsNone(fc.absorb_gib(1.33, None, None))
        self.assertIn("unreadable", fc.line(dict(rec, memfree_min_gib=None)))

    def test_every_completed_flip_writes_one_cushion_line_and_event(self):
        """RED before: the cushion was read by the latch alone and left no
        trace at a flip that went well -- W98 was the first word about it."""
        root = tempfile.mkdtemp(prefix="w98fc-")
        sd = state_file.init(root, "nfw98-boot-20260929T071410Z-beef", "boot", {})
        f = front_mod.Front(
            prefill="http://p", decode="http://d", awake="D", tag="w98fc",
            store_dir="/tmp", prefill_sid=0, decode_sid=0, dc_reserve={}, w_s=45.0,
            weight_chunks=2, flip_min_work_tokens=1)

        async def rpc(g, path, body, timeout):
            if path == "/flush_cache":
                return 200, "{}"
            tags = tuple((body or {}).get("tags", ()))
            return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                    "critical_path": "rank=0 card=GPU-x ms=1"})

        f.rpc = rpc

        async def body():
            await f.flip("D", "P")
            deadline = time.time() + 3.0
            while time.time() < deadline and not [
                    e for e in state_file.events(sd) if e["type"] == "flip_cushion"]:
                await asyncio.sleep(0.02)

        with mock.patch.dict(os.environ, {"WEG2_STATE_DIR": sd}), \
                mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=dict(Z30U)), \
                mock.patch.object(host_ledger, "read_cgroup", return_value={"max": 84 * GIB}), \
                self.assertLogs(front_mod.logger, "INFO") as cm:
            asyncio.run(body())
        lines = [l for l in cm.output if "WEG2-FLIP-CUSHION epoch=0" in l]
        self.assertEqual(len(lines), 1, cm.output[-5:])
        self.assertIn("cushion min=1.33", lines[0])
        self.assertIn("cg_room_min=3.00 (memory.max 84.00)", lines[0])
        ev = [e for e in state_file.events(sd) if e["type"] == "flip_cushion"]
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]["data"]["sleep"], ev[0]["data"]["wake"]), ("D", "P"))
        self.assertEqual(ev[0]["data"]["cushion_min_gib"], 1.33)


if __name__ == "__main__":
    unittest.main()
