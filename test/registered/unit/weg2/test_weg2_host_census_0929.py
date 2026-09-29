# SPDX-License-Identifier: Apache-2.0
"""Host census (z30w-park, 2026-09-29): the ~10.5 GiB the ledger did not book.

After fe3cdc442d (mark = memory.max, arena from the group env, cold_tier_shm)
z30w priced 68.92 GiB against 79.46 measured. The rest is measured, per role
and per class (creep_0929.jsonl 09:12:00Z, Pss per pid + du of /dev/shm):

  non-rank anon 7.41 GiB -- front 1.13, the two launch_server mains (tokenizer
  manager + HTTP) 2.17, detokenizers 2.18, inductor compile workers 1.77,
  PLE pread workers 0.14, resource tracker 0.02;
  shmem 1.50 lane ring (weg2-seq) + 0.26 arena sidecar (arena-49152.bin, the
  QSA index page) + 0.23 hand-off; 6.26 GiB shmem no class names -> ungebucht.

Guarded here: role/class classification, the creep and live census, the
max-merged record, the posts in the ledger (charged, both moments; ungebucht
on the WATERMARK line, not charged), the z30w reading within +-2 GiB, the run
residual's image from anonymous-shared Pss (not RssShmem: -98.95 on z30w),
and a dense (27B) boot without expert store pricing cold_tier_shm at 0.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_census as hc
from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GIB = int(host_ledger.GIB)
MIB = 1 << 20
CREEP = "/spinning/gpu-arb/tools/creep/creep_0929.jsonl"
Z30W_STORE_GIB = 40142.0 / 1024.0          # the map: 346 slots x 48 x 2.417 MiB
Z30W_WIDTHS = [786432, 58834944]            # KV page, mamba blob (draft tier off)
RING_KW = dict(ring_bytes=4096 * MIB, ring_span1_bytes=4096 * MIB)


def _z30w_row():
    rows = [json.loads(l) for l in open(CREEP) if l.strip()]
    return [r for r in rows if "z30wpark" in r.get("name", "") and r.get("shm_du")][-1]


class TestClassification(CustomTestCase):
    def test_roles_of_the_z30w_processes(self):
        c = hc.classify_process
        self.assertEqual(c("sglang::schedul", "sglang::scheduler_TP0"), "rank")
        self.assertEqual(c("sglang::detoken", "sglang::detokenizer"), "detokenizer")
        self.assertEqual(c("python", "/opt/venv/bin/python -m sglang.srt.weg2.front --prefill x"), "front")
        self.assertEqual(c("python", "/opt/venv/bin/python -m sglang.launch_server --model-path m"), "server_main")
        self.assertEqual(c("python", "python .../torch/_inductor/compile_worker/__main__.py"),
                         "inductor_compile_worker")
        self.assertEqual(c("python", "python -I -S .../qwen4_exp_ple_pread_worker.py"), "ple_pread_worker")

    def test_shm_classes(self):
        s = hc.shm_class
        self.assertEqual(s("/mnt/nf-experts/fnFL2/l0.bin", "/mnt/nf-experts/fnFL2"), "store")
        self.assertEqual(s("/dev/shm/weg2-seq-1/c0_s1_unit_buffer.bin"), "seq_ring")
        self.assertEqual(s("/dev/shm/weg2-arena-T/arena-786432.bin", arena_booked_widths=Z30W_WIDTHS),
                         "arena_booked")
        self.assertEqual(s("/dev/shm/weg2-arena-T/arena-49152.bin", arena_booked_widths=Z30W_WIDTHS),
                         "arena_sidecar")
        self.assertEqual(s("/dev/shm/weg2-arena-T/handoff"), "arena_handoff")
        self.assertEqual(s("/dev/shm/weg2-arena-T-l3idx/l3idx.bin"), "l3idx")
        self.assertEqual(s("/dev/zero"), "anon_shared")


SMAPS = """\
7f0000000000-7f0040000000 rw-s 00000000 00:2a 11 /mnt/nf-experts/fnFL2/l0.bin
Rss:            1048576 kB
Pss:             349525 kB
7f0040000000-7f0080000000 rw-s 00000000 00:01 12 /dev/zero (deleted)
Rss:            1048576 kB
Pss:            1048576 kB
7f0080000000-7f0090000000 rw-p 00000000 00:00 0
Rss:             262144 kB
Pss:             262144 kB
7f0090000000-7f00a0000000 r--s 00000000 08:02 13 /spinning/model/model-00001.safetensors
Rss:             262144 kB
Pss:             262144 kB
"""
MOUNTS = "tmpfs /mnt/nf-experts tmpfs rw 0 0\ntmpfs /dev/shm tmpfs rw 0 0\n/dev/sda2 /spinning xfs rw 0 0\n"


class TestLiveCensus(CustomTestCase):
    def test_smaps_takes_shared_pss_only(self):
        got = hc.smaps_shm_pss(SMAPS)
        self.assertEqual(got["/mnt/nf-experts/fnFL2/l0.bin"], 349525 * 1024)
        self.assertEqual(got["/dev/zero (deleted)"], 1048576 * 1024)
        self.assertNotIn("", got)                       # the private mapping
        mounts = hc.tmpfs_mounts(MOUNTS)
        self.assertFalse(hc.is_shmem_path("/spinning/model/model-00001.safetensors", mounts))
        self.assertTrue(hc.is_shmem_path("/dev/zero (deleted)", mounts))

    def test_sample_live_over_a_fake_proc(self):
        files = {
            "/sys/fs/cgroup/cgroup.procs": "10\n11\n",
            "/proc/mounts": MOUNTS,
            "/sys/fs/cgroup/memory.stat": f"anon {3 * GIB}\nshmem {3 * GIB}\n",
            "/proc/10/comm": "python\n", "/proc/10/cmdline": "python\0-m\0sglang.srt.weg2.front\0",
            "/proc/10/smaps_rollup": "Pss_Anon:     1048576 kB\n", "/proc/10/smaps": "",
            "/proc/11/comm": "sglang::schedul\n", "/proc/11/cmdline": "sglang::scheduler_TP0",
            "/proc/11/smaps_rollup": "Pss_Anon:     2097152 kB\n", "/proc/11/smaps": SMAPS,
        }
        c = hc.sample_live(store_dir="/mnt/nf-experts/fnFL2", reader=lambda p: files[p])
        self.assertAlmostEqual(c["roles_anon_gib"]["front"], 1.0, places=3)
        self.assertAlmostEqual(c["roles_anon_gib"]["rank"], 2.0, places=3)
        self.assertAlmostEqual(c["shm_classes_gib"]["store"], 349525 / 1048576, places=3)
        self.assertAlmostEqual(c["shm_classes_gib"]["anon_shared"], 1.0, places=3)
        self.assertNotIn("other_tmpfs", c["shm_classes_gib"])     # the disk mmap is not shmem

    def test_record_is_a_max_merge_per_key(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, hc.RECORD_NAME)
            hc.merge_into_record(p, "k", {"roles_anon_gib": {"front": 1.0}, "shm_classes_gib": {"seq_ring": 1.5},
                                          "unattributed_shm_gib": 2.0, "source": "a"})
            e = hc.merge_into_record(p, "k", {"roles_anon_gib": {"front": 0.5, "detokenizer": 2.0},
                                              "shm_classes_gib": {"seq_ring": 1.0},
                                              "unattributed_shm_gib": 3.0, "source": "b"})
            self.assertEqual(e["roles_anon_gib"], {"front": 1.0, "detokenizer": 2.0})
            self.assertEqual(e["shm_classes_gib"]["seq_ring"], 1.5)
            self.assertEqual(e["unattributed_shm_gib"], 3.0)
            self.assertEqual(e["samples"], 2)
            self.assertIsNone(hc.load_record(p).get("other-key"))
            t = hc.ledger_terms(None)
            self.assertEqual(t["nonrank_anon_gib"], 0.0)
            self.assertIn("UNMEASURED", t["census_source"])


@unittest.skipUnless(os.path.exists(CREEP), "creep_0929.jsonl absent")
class TestZ30wCensus(CustomTestCase):
    def setUp(self):
        self.c = hc.census_from_creep(_z30w_row(), store_gib=Z30W_STORE_GIB, arena_booked_widths=Z30W_WIDTHS)
        self.t = hc.ledger_terms({**self.c, "samples": 1, "last_at": self.c["at"]})

    def test_the_posts(self):
        r = self.c["roles_anon_gib"]
        self.assertAlmostEqual(r["front"], 1.13, delta=0.01)
        self.assertAlmostEqual(r["server_main"], 2.17, delta=0.01)
        self.assertAlmostEqual(r["detokenizer"], 2.18, delta=0.01)
        self.assertAlmostEqual(r["inductor_compile_worker"], 1.77, delta=0.01)
        self.assertAlmostEqual(self.t["nonrank_anon_gib"], 7.41, delta=0.02)
        self.assertAlmostEqual(self.t["seq_ring_gib"], 1.50, delta=0.01)
        self.assertAlmostEqual(self.t["arena_sidecar_gib"], 0.26, delta=0.01)
        self.assertAlmostEqual(self.t["arena_handoff_gib"], 0.23, delta=0.01)
        self.assertAlmostEqual(self.t["unbooked_shm_gib"], 6.26, delta=0.02)

    def test_the_reading_within_two_gib(self):
        # fe3cdc442d's composition (47.72 - 23.75 arena + 5.75 arena + 39.20 store)
        # plus the charged census posts, against nonreclaim 79.46
        charged = sum(float(self.t[k]) for k in
                      ("nonrank_anon_gib", "seq_ring_gib", "arena_sidecar_gib", "arena_handoff_gib"))
        predicted = 47.72 - 23.75 + 5.75 + Z30W_STORE_GIB + charged
        self.assertAlmostEqual(charged, 9.40, delta=0.03)
        self.assertLess(abs(predicted - 79.46), 2.0, predicted)

    def test_the_ledger_charges_the_posts_and_prints_ungebucht(self):
        a0 = host_ledger.price(int(125.70 * GIB), int(101.96 * GIB), 1, 600, **RING_KW,
                               cg_current_bytes=int(1.09 * GIB), cg_ceiling_bytes=84 * GIB)
        a1 = host_ledger.price(int(125.70 * GIB), int(101.96 * GIB), 1, 600, **RING_KW,
                               cg_current_bytes=int(1.09 * GIB), cg_ceiling_bytes=84 * GIB, census=self.t)
        self.assertAlmostEqual(a1.predicted_run_peak_gib() - a0.predicted_run_peak_gib(), 9.40, delta=0.03)
        line = host_ledger.arm_terms_line(a1)
        for k in ("nonrank_anon=", "seq_ring=", "arena_sidecar=", "arena_handoff="):
            self.assertIn(k, line)
        live = {"current_gib": 1.09, "nonreclaim_gib": 0.79, "file_reclaimable_gib": 0.3,
                "source": "test", "max_gib": None}
        with mock.patch.object(host_ledger, "read_cgroup_pressure", return_value=live):
            _arm, _h, lines = host_ledger.choose(
                int(125.70 * GIB), int(101.96 * GIB), **RING_KW, ring_provenance="test",
                cg_current_bytes=int(1.09 * GIB), cg_ceiling_bytes=84 * GIB,
                cg_ceiling_source="cgroup memory.max", cg_oom_kill=0, census=self.t)
        wm = [ln for ln in lines if ln.startswith("WEG2-HOST WATERMARK=")][0]
        self.assertIn("ungebucht=6.26 GiB", wm)


class TestRunResidualImage(CustomTestCase):
    def test_the_image_is_anonymous_shared_pss_not_rss_shmem(self):
        # three ranks mapping the same tmpfs store and each its own 1 GiB TMS image
        with mock.patch.object(hc, "_read", return_value=SMAPS):
            b, seen = host_ledger.image_shmem_bytes([1, 2, 3])
            rec = host_ledger.dormant_image_sample(
                group="D", shmem_before_bytes=0, shmem_after_bytes=0, pids=[1, 2, 3],
                weight_tags_gib=0.0, interleaved=False, boot_tag="t", commit="c",
                cg_current_bytes=int(79.46 * GIB), reclaimable_bytes=0,
                arm={"s_gb": 1, "m_mib": 600})
        self.assertEqual(seen, [1, 2, 3])
        self.assertEqual(b, 3 * 1048576 * 1024)        # the store counts nowhere, the images once
        self.assertAlmostEqual(rec["rss_shmem_gib"], 3.0, places=3)
        self.assertEqual(rec["image_instrument"], "pss_anon_shared")
        self.assertGreater(rec["run_residual_gib"], 0.0)


class TestDenseHasNoColdTier(CustomTestCase):
    def test_27b_without_store_dir(self):
        ns = SimpleNamespace(env_d="", env_p="")
        with mock.patch.dict(os.environ, {"SGLANG_MOE_EXPERT_STORE_DIR": ""}):
            gib, line = launcher.cold_tier_shm_post(ns, "/nonexistent/qwen27b")
        self.assertEqual(gib, 0.0)
        self.assertIn("no expert store", line)

    def test_27b_with_a_tmpfs_dir_but_no_map_is_zero_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            ns = SimpleNamespace(env_d=f"SGLANG_MOE_EXPERT_STORE_DIR={d}", env_p="", extra_d="", extra_p="",
                                 pp_cut_expert_device_fraction="")
            gib, line = launcher.cold_tier_shm_post(ns, "/nonexistent/qwen27b",
                                                    mounts_text=f"tmpfs {d} tmpfs rw 0 0\n")
        self.assertEqual(gib, 0.0)
        self.assertIn("UNPRICED", line)


class TestWiring(CustomTestCase):
    def test_every_pricing_path_takes_the_census(self):
        import inspect

        for fn in (host_ledger.charge_terms, host_ledger.price, host_ledger.choose,
                   launcher.choose_host_ledger):
            self.assertIn("census", inspect.signature(fn).parameters, fn.__name__)
        src = inspect.getsource(launcher.main)
        self.assertIn("census=_census_terms", src)
        self.assertIn("_hc.census_line(", src)
        fsrc = inspect.getsource(launcher.front_argv_for)
        self.assertIn("--host-census-record", fsrc)
        from sglang.srt.weg2 import front
        self.assertIn("start_host_census_sampler(", inspect.getsource(front.main))


if __name__ == "__main__":
    unittest.main()
