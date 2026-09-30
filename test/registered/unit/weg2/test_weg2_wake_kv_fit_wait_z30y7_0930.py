# SPDX-License-Identifier: Apache-2.0
"""Metal z30y7 (27b-row-authority, flip, ...09301715 @ead403b7ab).

17:26:55 PP0 (5090): "W114 Weg2KvResumeRefused: free=8668 MiB < need=8832
MiB (short by 163 MiB)". The creep line: "card_other_procs +1916 MiB" since
the previous wake, where every one of the 15 D->P wakes before had read
other=2528-2564 MiB (no leak). D TP0's sleep was slower that time (deposits
527/425/293/299 ms) and released 'weights_draft' + 'weights' (1824 MiB)
about 0.3 s after P's check: the waker's kv call runs concurrently with the
sleeper's leg.

The front's kv call then got 200, logged "WEG2-FLIP done woke=P", and
weg2-31-31 sat in PP0's dormant hold for good ("#1443 DORMANT-HOLD"),
showing outstanding P=1 and nothing else.

DANGER DIRECTIONS guarded here:
* a kv resume waits (bounded) for the card to fund it -- the sleeper
  releasing a moment later is not a refusal;
* still short after the bound: the refusal FAILS the kv call through the
  group fence (every rank raises, the front sees non-200 -> named W4); never
  a 200 over a dormant group;
* no reading, no wait: an unknown figure is not a refusal.
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import wake_kv as W
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1 << 20


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


class KvFitWait(CustomTestCase):
    def test_z30y7_the_sleeper_releases_a_moment_later(self):
        c = _Clock()
        need = 8832 * MIB
        # 8668 MiB at the check, D's last two tags (1824 MiB) land 0.3 s later
        read = lambda: (8668 if c.t < 0.3 else 8668 + 1824) * MIB
        unfit, free, waited = W.wait_for_kv_fit(read, need, 1055 * MIB, wait_s=20, poll_s=0.05,
                                                sleep=c.sleep, now=c.now)
        self.assertIsNone(unfit, "a sleeper still releasing refused the wake (z30y7)")
        self.assertEqual(free, (8668 + 1824) * MIB)
        self.assertLess(waited, 1.0)

    def test_still_short_after_the_bound_is_a_named_refusal(self):
        c = _Clock()
        unfit, free, waited = W.wait_for_kv_fit(lambda: 8668 * MIB, 8832 * MIB, 0, wait_s=2.0, poll_s=0.1,
                                                sleep=c.sleep, now=c.now)
        self.assertIn("short by 164 MiB", unfit)          # the metal line read 163 on byte figures
        self.assertGreaterEqual(waited, 2.0)

    def test_no_reading_no_wait(self):
        c = _Clock()
        unfit, free, waited = W.wait_for_kv_fit(lambda: None, 8832 * MIB, 0, wait_s=20, sleep=c.sleep, now=c.now)
        self.assertIsNone(unfit)
        self.assertEqual(waited, 0.0)

    def test_the_bound_is_configurable(self):
        self.assertEqual(W.kv_fit_wait_s({W.KV_FIT_WAIT_ENV: "5"}), 5.0)
        self.assertEqual(W.kv_fit_wait_s({}), W.KV_FIT_WAIT_DEFAULT_S)


class RefusedKvCallFails(CustomTestCase):
    def test_the_wake_rpc_fails_through_the_fence_instead_of_200(self):
        from sglang.srt.managers.scheduler_components import weight_updater as WU

        src = inspect.getsource(WU)
        i = src.index("_unfit = kv_resume_fit_refusal(_kv_free, _kv_need, _kv_floor)")
        self.assertIn("wait_for_kv_fit(", src[i:i + 900], "no bounded wait before W114")
        j = src.index("elif GPU_MEMORY_TYPE_KV_CACHE in tags:")
        self.assertIn("_weg2_kv_refusal = (", src[j:j + 900])
        k = src.index("if _weg2_kv_refusal:\n            store_failure =")
        self.assertLess(k, src.index("report = self._weg2_group_fence(", k),
                        "the refusal must ride the group fence (every rank, non-200 -> W4)")
