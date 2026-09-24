# SPDX-License-Identifier: Apache-2.0
"""H46b: D's draft host image (H25, 1.5 GB pinned) is allocated at scheduler
init, not at the first sleep's park.

Befund: x148 ``WEG2-DRAFT-PARK ... host image allocated 7989 ms``, x151
10442 ms, beide in TP0s erstem Schlaf; TP0s erster Deposit startete 8,1 s
(x148) bzw. 10,6 s (x151) nach TP1/TP2, der erste D->P-Flip dauerte 24,35 s
bzw. 13,1 s gegen 1,7-1,8 s ab dem dritten Flip.

CPU-only (pin=False, fakes for the saver and the sync):
* the boot allocation is sized from the SAME population the park lays out;
  the first park then reports ``host image reused`` and uses the same tensor;
* the boot image holds no draft (``holds_image`` False until a park wrote it);
* a population that grew after the boot re-allocates (the disproof line);
* the rank method logs ``WEG2-DRAFT-PARK host image preallocated bytes= ms=``,
  honours the switch, skips where nothing parks, and is wired at boot.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import draft_park as dpk  # noqa: E402

Manager = wu.SchedulerWeightUpdaterManager


class _Draft(torch.nn.Module):
    def __init__(self, shared=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(16, dtype=torch.float32), requires_grad=False)
        self.register_buffer("cos_sin", torch.full((8,), 3.0))
        if shared is not None:
            self.embed = shared


class ParkObject(unittest.TestCase):
    def test_prealloc_then_first_park_reuses_the_same_image(self):
        draft = _Draft()
        park = dpk.DraftHostPark(pin=False)
        pop = dpk.park_population(draft, None)
        nbytes, ms = park.preallocate(pop)
        self.assertEqual(nbytes, 16 * 4 + 8 * 4)
        self.assertGreaterEqual(ms, 0.0)
        host = park.host
        self.assertEqual(int(host.numel()), nbytes)
        self.assertFalse(park.holds_image, "a boot image carries no draft yet")
        rec = park.park(dpk.park_population(draft, None), tag="weights_draft",
                        pause=lambda t: None, sync=lambda: None)
        self.assertIs(park.host, host)
        self.assertEqual(rec.alloc_ms, -1.0)
        self.assertIn("host image reused", rec.line())
        self.assertTrue(park.holds_image)
        # the bytes went into the preallocated image
        self.assertTrue(torch.equal(park.host[:64].view(torch.float32),
                                    torch.arange(16, dtype=torch.float32)))

    def test_second_prealloc_is_a_no_op(self):
        park = dpk.DraftHostPark(pin=False)
        pop = dpk.park_population(_Draft(), None)
        park.preallocate(pop)
        host = park.host
        self.assertEqual(park.preallocate(pop)[1], -1.0)
        self.assertIs(park.host, host)

    def test_a_population_that_grew_after_the_boot_allocates_again(self):
        draft = _Draft()
        park = dpk.DraftHostPark(pin=False)
        park.preallocate(dpk.park_population(draft, None))
        draft.register_buffer("late", torch.zeros(64))
        rec = park.park(dpk.park_population(draft, None), tag="weights_draft",
                        pause=lambda t: None, sync=lambda: None)
        self.assertGreaterEqual(rec.alloc_ms, 0.0)
        self.assertIn("host image allocated", rec.line())


def _manager(draft, target):
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=target))
    saver = SimpleNamespace(pause=lambda tag: None)
    return Manager(tp_worker=worker, draft_worker=None, tp_cpu_group=None,
                   memory_saver_adapter=saver, flush_cache=lambda *a, **k: True,
                   is_fully_idle=lambda *a, **k: True)


class _CpuPark(dpk.DraftHostPark):
    def __init__(self, **kw):
        kw["pin"] = False
        super().__init__(**kw)


class RankSide(unittest.TestCase):
    def setUp(self):
        self.target = torch.nn.Module()
        self.target.embed = torch.nn.Parameter(torch.ones(32), requires_grad=False)
        self.draft = _Draft(self.target.embed)
        self.m = _manager(self.draft, self.target)
        self.lines = []
        drafter = SimpleNamespace(model=self.draft)
        self._p = [
            mock.patch.object(wu, "_weg2_drafter_of", lambda s: drafter),
            mock.patch.object(Manager, "_weg2_draft_park_armed", lambda s: True),
            mock.patch.object(Manager, "_weg2_tag_bytes", lambda s, tag: 4096),
            mock.patch.object(dpk, "DraftHostPark", _CpuPark),
            mock.patch.object(wu.logger, "info",
                              lambda fmt, *a, **k: self.lines.append(fmt % a if a else fmt)),
        ]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in reversed(self._p):
            p.stop()

    def test_boot_line_then_the_first_park_reuses(self):
        self.m._weg2_draft_prealloc_at_boot()
        pre = [ln for ln in self.lines if "WEG2-DRAFT-PARK host image preallocated" in ln]
        self.assertEqual(len(pre), 1, self.lines)
        self.assertIn("bytes=96 ", pre[0])            # w + cos_sin; the target's embed is not parked
        self.assertIn(" ms=", pre[0])
        self.assertIn("storages=2", pre[0])
        host = self.m._weg2_draft_park.host
        with mock.patch.object(torch.cuda, "synchronize", lambda *a: None):
            self.m._weg2_park_draft_at_sleep(None)
        self.assertIs(self.m._weg2_draft_park.host, host)
        park = [ln for ln in self.lines if ln.startswith("WEG2-DRAFT-PARK tag=weights_draft bytes=")]
        self.assertEqual(len(park), 1, self.lines)
        self.assertIn("host image reused", park[0])

    def test_without_prealloc_the_first_park_allocates(self):
        with mock.patch.object(torch.cuda, "synchronize", lambda *a: None):
            self.m._weg2_park_draft_at_sleep(None)
        park = [ln for ln in self.lines if ln.startswith("WEG2-DRAFT-PARK tag=weights_draft bytes=")]
        self.assertIn("host image allocated", park[0])

    def test_switch_off(self):
        with envs.SGLANG_WEG2_DRAFT_PARK_PREALLOC.override(False):
            self.m._weg2_draft_prealloc_at_boot()
        self.assertIsNone(self.m._weg2_draft_park)
        self.assertTrue(any("SGLANG_WEG2_DRAFT_PARK_PREALLOC=0" in ln for ln in self.lines))

    def test_default_is_on(self):
        self.assertTrue(envs.SGLANG_WEG2_DRAFT_PARK_PREALLOC.get())

    def test_not_armed_or_nothing_to_park_allocates_nothing(self):
        with mock.patch.object(Manager, "_weg2_draft_park_armed", lambda s: False):
            self.m._weg2_draft_prealloc_at_boot()
        self.assertIsNone(self.m._weg2_draft_park)
        with mock.patch.object(Manager, "_weg2_tag_bytes", lambda s, tag: 0):
            self.m._weg2_draft_prealloc_at_boot()
        self.assertIsNone(self.m._weg2_draft_park)
        self.assertTrue(any("nothing to park on this rank" in ln for ln in self.lines))

    def test_wired_at_boot(self):
        from sglang.srt.weg2 import weight_exchange as wx

        called = []
        with mock.patch.object(wx, "exchange_armed", return_value=True), \
                mock.patch.object(Manager, "_weg2_bar1_start", lambda s: None), \
                mock.patch.object(Manager, "_weg2_join_prewarm_start", lambda s: None), \
                mock.patch.object(Manager, "_weg2_ring_preregister_start", lambda s: None), \
                mock.patch.object(Manager, "_weg2_draft_prealloc_at_boot",
                                  lambda s: called.append(1)):
            self.m._weg2_prewarm_lanes_start()
        self.assertEqual(called, [1])


if __name__ == "__main__":
    unittest.main()
