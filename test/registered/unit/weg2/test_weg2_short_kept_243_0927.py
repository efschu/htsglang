# SPDX-License-Identifier: Apache-2.0
"""SK (#243 front half, NF rc12r dkrnfh91dprbar1dauer09271632).

(b) weg2-12-39: SHORT verdict (uncached 186, D presence 76416 d_leg2_cached), the SHORT-GATE held
120 s -> "routing BATCH" -> P leg 1 prefilled all 76602 tokens (cached 0), then 255 s for a D seat.
weg2-13-44 (short, uncached 735) arrived while P was awake -> queued BATCH with no line of its own
-> the same full P prefill. Now: a SHORT with a measured D presence that falls through is KEPT for
D (no leg 1), in D's admission line when D is awake; the P-awake fall-through gets its own line.
(a) the kept SHORT's price is bound to the front epoch; its D admission re-prices it after a flip
and sends it to P when the presence is gone (over X).
"""

import asyncio
import collections
import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from sglang.srt.weg2 import front as F  # noqa: E402


def _front(new_price, epoch=17):
    f = object.__new__(F.Front)
    f.epoch = epoch
    f.x_exact = True
    f.tp_prefill_max_tokens = 4096
    f.counters = collections.Counter()
    f.queue = collections.deque()
    f.ftok = types.SimpleNamespace(ids_for=lambda text: [1, 2, 3])
    f.tspans = types.SimpleNamespace(pending=lambda ids, epoch=None: (new_price, 0, True, "d_leg2_cached"))
    return f


def _p(price_epoch=13, est=186):
    loop = asyncio.new_event_loop()
    return F.Pending("weg2-12-39", "/v1/messages", {}, "text", 0.0, loop.create_future(),
                     est_prompt=76602, est_uncached=est, span_known=True, skip_leg1=True,
                     short_kept=True, price_epoch=price_epoch)


class Reprice(unittest.TestCase):
    def test_same_epoch_no_reprice(self):
        f, p = _front(76602), _p(price_epoch=17)
        self.assertFalse(f._sk_admission_reprice(p))
        self.assertEqual(p.est_uncached, 186)

    def test_b_weg2_12_39_presence_gone_after_a_flip_goes_to_p(self):
        f, p = _front(76602), _p(price_epoch=13)
        with self.assertLogs(F.logger, level="INFO") as cap:
            self.assertTrue(f._sk_admission_reprice(p))
        self.assertEqual((p.est_uncached, p.skip_leg1, p.short_kept, p.d_direct), (76602, False, False, False))
        self.assertEqual(list(f.queue), [p])
        self.assertIn("SHORT-KEPT-REPRICE rid=weg2-12-39 est_uncached 186 -> 76602 X=4096", cap.output[0])

    def test_presence_still_there_stays_on_d(self):
        f, p = _front(512), _p(price_epoch=13)
        self.assertFalse(f._sk_admission_reprice(p))
        self.assertEqual((p.est_uncached, p.price_epoch, p.short_kept), (512, 17, True))
        self.assertEqual(list(f.queue), [])

    def test_no_exact_tokenizer_admits_unchanged(self):
        f, p = _front(76602), _p(price_epoch=13)
        f.x_exact = False
        self.assertFalse(f._sk_admission_reprice(p))
        self.assertEqual(p.est_uncached, 186)


class Wiring(unittest.TestCase):
    def setUp(self):
        self.src = open(F.__file__).read()

    def test_switch(self):
        self.assertTrue(F._short_keep_enabled({}))
        self.assertFalse(F._short_keep_enabled({"SGLANG_WEG2_SHORT_KEEP_PRESENCE": "0"}))

    def test_fall_through_keeps_a_measured_d_presence(self):
        i = self.src.index("_sk = (short_ok and _short_keep_enabled() and int(store_span or 0) > 0")
        blk = self.src[i:i + 6000]
        self.assertIn("WEG2 SHORT-BEHIND-P rid=%s", blk)
        self.assertIn("skip_leg1=_sk, short_kept=_sk, price_epoch=int(self.epoch)", blk)
        self.assertIn('if _sk and self.awake == "D" and self.admit_d and self.state == "serving":', blk)
        self.assertIn("self._ready_for_d.append(p)", blk)

    def test_admitter_reprices_a_kept_short_before_the_seat(self):
        # The whole d_admitter body, not a fixed 6000-char window: SA (#244)
        # added its backfill between the re-price and the seat and pushed the
        # acquire past the old window -- the ORDER is what is pinned.
        i = self.src.index("    async def d_admitter(self)")
        j = self.src.index("\n    async def ", i + 10)
        blk = self.src[i:j]
        a = blk.index('if getattr(p, "short_kept", False) and self._sk_admission_reprice(p):')
        self.assertLess(a, blk.index("await self._d_seat.acquire()"))

    def test_a_backfilled_kept_short_is_repriced_before_the_seat(self):
        # SA's backfill may seat a younger request past the blocked head; a
        # kept SHORT among them is re-priced first, like the head.
        i = self.src.index("    async def d_admitter(self)")
        j = self.src.index("\n    async def ", i + 10)
        blk = self.src[i:j]
        b = blk.index('if getattr(_q, "short_kept", False) and self._sk_admission_reprice(_q):')
        self.assertLess(b, blk.index("p = _q"))
        self.assertLess(b, blk.index("await self._d_seat.acquire()"))

    def test_p_drain_skips_leg1_for_a_kept_short(self):
        self.assertIn("if p.skip_leg1:  # route CARRIER-EXCEEDS: no leg 1, D prefills once", self.src)


if __name__ == "__main__":
    unittest.main()
