"""H13 (24.09.): name the code that grows a rank's private anonymous host memory.

Boot fnFL2x109: D-TP0 (Form A attention host) died by the cgroup OOM reaper
in a cold D-direct prefill (3891 tokens); cgroup anon +7.4 GiB in one
second, shmem flat, every other rank flat, TP0's last line the MoE-input
carrier of layer 31. Nothing in the rank log said which allocator took the
bytes or where. x108 ran the same prefill without the jump.

The instrument (debug_utils/host_anon_probe.py, SGLANG_DEBUG_HOST_ANON_PROBE)
must, at the next boot, turn that into ONE grep-able line naming the site:

* the checkpoints bracket a jump between two named forward-path sites
  (layer, MoE wave fetch/apply) -- driven here through the REAL token-major
  run_waves of a CPU offload cache whose apply takes 320 MiB on wave 1;
* the sampler names growth the checkpoints cannot see (another thread, or a
  rank that dies before its next checkpoint), with every thread's stack;
* the measurement is RssAnon (private anon), and a MAP_SHARED anonymous map
  -- what cudaHostAlloc / pin_memory produce -- does NOT move it;
* off by default, a no-op when off.

Before the fix there is no such module and no line (red); after, green.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import logging
import mmap
import tempfile
import threading
import unittest
import unittest.mock
from collections import namedtuple
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

MIB = 1 << 20


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _logger(name):
    cap = _Capture()
    log = logging.getLogger(name)
    log.handlers = [cap]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log, cap


class _Script:
    """A scripted anon reading (bytes), advanced one value per call."""

    def __init__(self, mib_values):
        self.values = [int(v * MIB) for v in mib_values]
        self.i = 0

    def __call__(self):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return v


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 0.05
        return self.t


def _probe(read, log, **kw):
    from sglang.srt.debug_utils.host_anon_probe import HostAnonProbe

    return HostAnonProbe(
        threshold_bytes=256 * MIB,
        sample_ms=0,
        read_anon=read,
        clock=kw.pop("clock", _Clock()),
        cheap=kw.pop("cheap", lambda: {}),
        extras=kw.pop("extras", lambda: {}),
        stacks=kw.pop("stacks", lambda skip: []),
        vmas=kw.pop("vmas", lambda: []),
        log=log,
        **kw,
    )


class TestMeasurement(CustomTestCase):
    def test_statm_resident_minus_shared_is_rss_anon(self):
        from sglang.srt.debug_utils import host_anon_probe as hap

        with tempfile.NamedTemporaryFile("w", suffix="statm", delete=False) as fh:
            fh.write("900000 1000 300 10 0 5000 0\n")
        try:
            self.assertEqual(hap.read_anon_bytes(fh.name), 700 * hap._PAGE)
        finally:
            os.unlink(fh.name)
        self.assertEqual(hap.read_anon_bytes("/nonexistent/statm"), -1)

    def test_private_pages_move_it_shared_anonymous_pages_do_not(self):
        """The x109 jump was anon with shmem flat. A MAP_SHARED|MAP_ANONYMOUS
        map (cudaHostAlloc, torch pin_memory) is RssShmem and must not be
        counted; touched private bytes must."""
        from sglang.srt.debug_utils import host_anon_probe as hap

        n = 320 * MIB
        a0 = hap.read_anon_bytes()
        shared = mmap.mmap(-1, n, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)
        for off in range(0, n, 4096):
            shared[off] = 1
        a1 = hap.read_anon_bytes()
        private = b"\x01" * n
        a2 = hap.read_anon_bytes()
        self.assertLess(abs(a1 - a0), 64 * MIB, "a shared anonymous map moved RssAnon")
        self.assertGreater(a2 - a1, 300 * MIB, "touched private bytes did not move RssAnon")
        del private
        shared.close()

    def test_glibc_names_a_large_malloc_as_mmapped_chunks(self):
        from sglang.srt.debug_utils import host_anon_probe as hap

        m0 = hap._mallinfo2()
        if not m0:
            self.skipTest("mallinfo2 unavailable (not glibc >= 2.33)")
        big = bytearray(64 * MIB)
        m1 = hap._mallinfo2()
        self.assertGreaterEqual(m1["malloc_mmap"] - m0["malloc_mmap"], 64 * MIB)
        del big

    def test_smaps_top_vmas_by_anonymous_pages(self):
        from sglang.srt.debug_utils import host_anon_probe as hap

        text = (
            "7f0000000000-7f0200000000 rw-p 00000000 00:00 0 \n"
            "Size:           8388608 kB\nAnonymous:      7602176 kB\n"
            "7f0300000000-7f0300100000 rw-s 00000000 00:05 123 /dev/zero (deleted)\n"
            "Size:              1024 kB\nAnonymous:            0 kB\n"
            "55d000000000-55d000400000 rw-p 00000000 00:00 0 [heap]\n"
            "Size:              4096 kB\nAnonymous:         4096 kB\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix="smaps", delete=False) as fh:
            fh.write(text)
        try:
            top = hap.top_anon_vmas(fh.name, top=2)
        finally:
            os.unlink(fh.name)
        self.assertEqual([(a // MIB, p, path) for a, _s, p, path in top],
                         [(7424, "rw-p", "[anon]"), (4, "rw-p", "[heap]")])

    def test_thread_stacks_innermost_first_and_named(self):
        from sglang.srt.debug_utils import host_anon_probe as hap

        stacks = hap.thread_stacks()
        mine = [s for s in stacks if s.startswith(threading.current_thread().name)]
        self.assertEqual(len(mine), 1, stacks)
        self.assertIn("test_thread_stacks_innermost_first_and_named", mine[0].split(" < ")[1])


class TestCheckpoints(CustomTestCase):
    def test_a_jump_is_named_by_both_sites_and_the_pass(self):
        log, cap = _logger("h13.cp")
        p = _probe(_Script([1000, 1000, 1100, 8500, 8490]), log)
        p.pass_begin("EXTEND", 3891)
        self.assertEqual(p.checkpoint("layer", layer=31), 0)
        self.assertEqual(p.checkpoint("moe.fetch", layer=31, wave="3/28"), 100 * MIB)
        self.assertEqual(p.checkpoint("moe.apply", layer=31, wave="3/28"), 7400 * MIB)
        self.assertEqual(p.checkpoint("moe.fetch", layer=31, wave="4/28"), -10 * MIB)
        deltas = [l for l in cap.lines if l.startswith("HOST-ANON-DELTA")]
        self.assertEqual(len(deltas), 1, cap.lines)
        d = deltas[0]
        self.assertTrue(d.startswith("HOST-ANON-DELTA +7400MiB anon=8500MiB src=checkpoint"), d)
        for part in ("where=moe.apply", "phase=EXTEND", "tokens=3891", "layer=31",
                     "wave=3/28", "since=moe.fetch"):
            self.assertIn(part, d)

    def test_allocator_deltas_between_the_two_sites(self):
        log, cap = _logger("h13.alloc")
        cheap = iter([{"malloc_mmap": 10 * MIB}, {"malloc_mmap": 7410 * MIB}])
        p = _probe(_Script([1000, 8400]), log, cheap=lambda: next(cheap),
                   extras=lambda: {"torch_host": 64 * MIB, "pinned_exact": -1})
        p.checkpoint("moe.fetch", layer=31)
        p.checkpoint("moe.apply", layer=31)
        d = [l for l in cap.lines if l.startswith("HOST-ANON-DELTA")][0]
        self.assertIn("malloc_mmap=7410MiB(+7400)", d)
        self.assertIn("torch_host=64MiB", d)
        self.assertNotIn("pinned_exact", d)  # unknown is omitted, never printed as 0

    def test_pass_line_carries_the_peak_and_its_site(self):
        log, cap = _logger("h13.pass")
        p = _probe(_Script([1000, 1000, 1200, 3000, 1100, 1100,   # extend pass
                            1100, 1100, 1110, 1105]), log)        # decode pass
        p.pass_begin("EXTEND", 3891)
        p.checkpoint("layer", layer=0)
        p.checkpoint("layer", layer=1)
        p.checkpoint("moe.apply", layer=1, wave="0/2")
        p.checkpoint("layer", layer=2)
        p.pass_end()
        p.pass_begin("DECODE", 4)
        p.checkpoint("layer", layer=0)
        p.checkpoint("layer", layer=1)
        p.pass_end()
        passes = [l for l in cap.lines if l.startswith("HOST-ANON-PASS")]
        self.assertEqual(len(passes), 1, cap.lines)  # the quiet decode pass logs nothing
        self.assertIn("phase=EXTEND tokens=3891 anon_begin=1000MiB anon_end=1100MiB peak=3000MiB "
                      "(+2000MiB over begin, at moe.apply layer=1)", passes[0])


class TestSampler(CustomTestCase):
    def test_growth_off_the_checkpoints_is_named_with_every_thread(self):
        log, cap = _logger("h13.sampler")
        clock = _Clock()
        p = _probe(_Script([1000, 1000, 1050, 4500, 8400, 8400, 1000, 1400]), log, clock=clock,
                   stacks=lambda skip: ["MainThread(1): expert_offload.py:4530 run_waves < layer.py:2652 x",
                                        "hicache-backup(2): cache_controller.py:4310 backup_thread_func"],
                   vmas=lambda: [(7 << 30, 0x7F0000000000, "rw-p", "[anon]")])
        p.checkpoint("moe.fetch", layer=31, wave="12/28")   # reads 1000 (sets the location)
        self.assertEqual(p.sample_once(), 0)                 # 1000: arms the level
        self.assertEqual(p.sample_once(), 0)                 # 1050: below threshold
        self.assertEqual(p.sample_once(), 3500 * MIB)        # 4500: reported
        self.assertEqual(p.sample_once(), 3900 * MIB)        # 8400: reported again (still growing)
        self.assertEqual(p.sample_once(), 0)                 # flat
        self.assertEqual(p.sample_once(), 0)                 # 1000: released -> re-arm quietly
        self.assertEqual(p.sample_once(), 400 * MIB)         # measured from the new floor
        deltas = [l for l in cap.lines if l.startswith("HOST-ANON-DELTA")]
        self.assertEqual(len(deltas), 3, cap.lines)
        self.assertIn("src=sampler where=moe.fetch layer=31 wave=12/28", deltas[0])
        stacks = [l for l in cap.lines if l.startswith("HOST-ANON-STACK")]
        self.assertEqual(len(stacks), 6)
        self.assertIn("hicache-backup(2): cache_controller.py:4310", stacks[1])
        vmas = [l for l in cap.lines if l.startswith("HOST-ANON-VMAS")]
        self.assertEqual(len(vmas), 1, "smaps is parsed at most every 5 s")
        self.assertIn("7168MiB rw-p 0x7f0000000000 [anon]", vmas[0])

    def test_real_sampler_thread_sees_a_real_allocation(self):
        log, cap = _logger("h13.thread")
        from sglang.srt.debug_utils.host_anon_probe import HostAnonProbe

        p = HostAnonProbe(threshold_bytes=256 * MIB, sample_ms=5, log=log,
                          vmas=lambda: [])
        self.assertTrue(p.start_sampler())
        try:
            hold = b"\x02" * (320 * MIB)
            ev = threading.Event()
            for _ in range(200):
                if any(l.startswith("HOST-ANON-DELTA") for l in cap.lines):
                    break
                ev.wait(0.01)
        finally:
            p.stop()
        deltas = [l for l in cap.lines if l.startswith("HOST-ANON-DELTA")]
        self.assertTrue(deltas, "the sampler never reported a 320 MiB private allocation")
        self.assertIn("src=sampler", deltas[0])
        self.assertTrue(any("test_real_sampler_thread_sees_a_real_allocation" in l
                            for l in cap.lines if l.startswith("HOST-ANON-STACK")))
        del hold


class TestWiring(CustomTestCase):
    def tearDown(self):
        from sglang.srt.debug_utils import host_anon_probe as hap

        hap._reset_for_tests()

    def test_off_by_default_and_a_noop_when_off(self):
        from sglang.srt.debug_utils import host_anon_probe as hap
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_DEBUG_HOST_ANON_PROBE.get())
        self.assertEqual(envs.SGLANG_DEBUG_HOST_ANON_PROBE_DELTA_MIB.get(), 256)
        self.assertEqual(envs.SGLANG_DEBUG_HOST_ANON_PROBE_SAMPLE_MS.get(), 50)
        hap._reset_for_tests()
        self.assertIsNone(hap.checkpoint("layer", layer=0))
        self.assertFalse(hap.enabled())

    def test_env_arms_the_singleton(self):
        from sglang.srt.debug_utils import host_anon_probe as hap
        from sglang.srt.environ import envs

        hap._reset_for_tests()
        with envs.SGLANG_DEBUG_HOST_ANON_PROBE.override(True), \
                envs.SGLANG_DEBUG_HOST_ANON_PROBE_SAMPLE_MS.override(0), \
                envs.SGLANG_DEBUG_HOST_ANON_PROBE_DELTA_MIB.override(512):
            self.assertTrue(hap.enabled())
            self.assertIsInstance(hap.checkpoint("layer", layer=0), int)
            self.assertEqual(hap._probe.threshold, 512 * MIB)

    def test_the_real_token_major_run_waves_names_the_wave_that_allocates(self):
        """A CPU offload cache (the #104 desk shape) through the REAL run_waves:
        wave 1's apply takes 320 MiB of private host memory and keeps it. The
        line must say moe.apply, layer 23, wave 1 of N, since moe.fetch."""
        import torch

        from sglang.srt.debug_utils import host_anon_probe as hap
        from sglang.srt.layers.moe import expert_offload as eo
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        E, R, C, W = 10, 2, 3, 4
        Dispatch = namedtuple("Dispatch", "hidden_states hidden_states_scale topk_output")
        Combine = namedtuple("Combine", "hidden_states")
        with unittest.mock.patch.dict(os.environ, {"SGLANG_MOE_SCRATCH_SLOTS": str(C),
                                                   "SGLANG_MOE_OFFLOAD_WAVE_ORDER": "token"}):
            cache = eo.MoEExpertOffloadCache(SimpleNamespace(num_local_experts=E, layer_id=23), R / E)
        cache._pinned = {"w13": torch.zeros((E - R, W))}
        cache._resident = {"w13": torch.zeros((R + C, W))}
        cache._installed = True
        # three tokens, each on three distinct spill experts -> three waves
        ids = [[2, 3, 4], [5, 6, 7], [7, 8, 9]]
        topk_ids = torch.tensor(ids, dtype=torch.int32)
        topk = StandardTopKOutput(topk_weights=torch.ones(topk_ids.shape), topk_ids=topk_ids,
                                  router_logits=None)
        held = []
        calls = {"n": 0}

        def apply(sub):
            if calls["n"] == 1:
                held.append(b"\x03" * (320 * MIB))
            calls["n"] += 1
            return Combine(hidden_states=sub.hidden_states)

        log, cap = _logger("h13.wiring")
        hap._reset_for_tests(hap.HostAnonProbe(threshold_bytes=256 * MIB, sample_ms=0, log=log),
                             resolved=True)
        hap.pass_begin("EXTEND", 3)
        cache.run_waves(Dispatch(torch.zeros(3, W), None, topk), apply)
        hap.pass_end()
        self.assertEqual(calls["n"], 3)
        deltas = [l for l in cap.lines if l.startswith("HOST-ANON-DELTA")]
        self.assertEqual(len(deltas), 1, cap.lines)
        for part in ("src=checkpoint where=moe.apply", "phase=EXTEND", "layer=23",
                     "wave=1/3", "since=moe.fetch"):
            self.assertIn(part, deltas[0])
        passes = [l for l in cap.lines if l.startswith("HOST-ANON-PASS")]
        self.assertEqual(len(passes), 1)
        self.assertIn("at moe.apply layer=23", passes[0])
        held.clear()


if __name__ == "__main__":
    unittest.main()
