# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 metal replay dual14 (bdfefe, ...09301139 @5bfdf964a9).

D TP0 died at 11:48:34 growing 229376 -> 319488 tokens:
WEG2-TMS-RESUME cuMemCreate FAILED rc=2 (out of memory). No P-PAUSE fired.

The P accounting was right: after every grant the ledger held exactly
bytes(mapped), 2306867200 B for the 102400 mapping on the 5090. The budget was
wrong. D shrank its boot pool at 11:41:57 (2415919104 B unmapped), the second
P started loading in the same second, and it sized its KV from free memory
that included D's released pool: 4475322368 B, against 2122317824 in dual13,
where P sized while D still held its pool. The budget counted those bytes
twice: 6891241472 B, where the physical KV room was about 4.5 GB.

DANGER DIRECTIONS guarded here:
* a pool the other group contributed but has released is not counted a second
  time by a later contribution;
* D keeps its boot pool until P has joined its card (group decision);
* a failed cuMemCreate while D grows is "card short", never fatal. The grow is
  refused on EVERY rank, the partial map is rolled back, the ledger is
  reconciled against the physical free bytes, and the shortfall becomes
  pressure on P;
* after every map, grant and release the ledger covers this rank's mapped
  bytes. A violation is refused by name;
* the ledger/physical self-check is a throttled instrument.
"""
from __future__ import annotations

import os
import tempfile
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from sglang.srt.weg2 import card_kv_ledger as K
from sglang.srt.weg2 import dual_d_kv_stage as D
from sglang.srt.weg2 import dual_p_kv_stage as S
from sglang.srt.weg2.d_seat_vram import AllocInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

G = 2 << 20
ROW = 2048
ALLOC = (16384 + 64) * ROW
MIB = 1 << 20


class FakeSpans:
    available = True

    def __init__(self, fail_above_bytes=None):
        self.fail_above = fail_above_bytes
        self.calls = []

    def info(self, ptr):
        return AllocInfo(size=ALLOC, mapped=0, planned=0, active=True)

    def set_spans(self, ptr, spans, now):
        self.calls.append((ptr, tuple(spans)))
        mapped = sum(int(n) for _o, n in spans)
        if self.fail_above is not None and ptr == 2 and mapped > self.fail_above:
            return 2                                             # CUDA_ERROR_OUT_OF_MEMORY on the 2nd tensor
        return 0


def _geom():
    return S._geom_for(torch.zeros(16448, 512), 16384, 64, "k", ALLOC)


def _card():
    return os.path.join(tempfile.mkdtemp(prefix="wkvph"), "card")


class CoverInvariant(CustomTestCase):
    def test_65536_then_102400_on_one_mapping_stays_covered(self):
        # scaled: 12288 ~ 65536, 16384 ~ 102400
        path = _card()
        led = K.CardKvLedger(path, "P")
        led.contribute(100 * MIB)
        st = S.PKvStage([(1, _geom())], led, allocator=object(), pools=[], page_size=64, granule=G,
                        top_tokens=16384, spans=FakeSpans(), engage_cap=lambda *a: None)
        stages = [{"ledger": path, "step": 4096, "top": 16384, "bytes": st.table()}]
        open_p = lambda pth: K.CardKvLedger(pth, "P")
        for tokens in (12000, 16000, 12000, 12000, 16000):
            st.map_granted(S.group_grant(stages, tokens, open_p))
            self.assertEqual(K.peek(path).committed["P"], st.bytes_for(st.mapped_tokens) - st.bytes_for(0))
            S.check_cover(st, "test")
        led.release(MIB)                                         # someone under-counts P
        with self.assertRaises(S.Weg2DualKvCoverBreach):
            S.check_cover(st, "test")


class BudgetNotCountedTwice(CustomTestCase):
    def test_a_released_pool_is_not_contributed_again(self):
        path = _card()
        d = K.CardKvLedger(path, "D")
        d.contribute(40 * MIB, committed=40 * MIB)
        d.release(30 * MIB)                                      # D unmapped 30 MiB before P sized
        p = K.CardKvLedger(path, "P")
        p.contribute(34 * MIB)                                   # P's sizing saw those 30 MiB as free
        self.assertEqual(K.peek(path).budget, 44 * MIB, "the released pool was counted twice")

    def test_disjoint_sizing_counts_both(self):
        path = _card()
        d = K.CardKvLedger(path, "D")
        d.contribute(40 * MIB, committed=40 * MIB)
        K.CardKvLedger(path, "P").contribute(34 * MIB)
        self.assertEqual(K.peek(path).budget, 74 * MIB)

    def test_d_keeps_its_boot_pool_until_p_joined(self):
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        alloc = TokenToKVPoolAllocator(200, torch.float16, "cpu", None, False)
        path = _card()
        led = K.CardKvLedger(path, "D")
        geom = S._geom_for(torch.zeros(264, 8), 256, 1, "k", 264 * 32)
        a = D.DKvStage([(1, geom)], led, allocator=alloc, pools=[], page_size=1, granule=32, top_tokens=192,
                       spans=FakeSpans(), step=16, gmin=lambda v: v)
        b = a.bytes_for(96) - a.bytes_for(0)
        led.contribute(b, committed=b)
        a.mapped_tokens, a._committed = 96, b
        sched = types.SimpleNamespace(
            running_batch=types.SimpleNamespace(reqs=[]), chunked_req=None, waiting_queue=[], tree_cache=None,
            server_args=types.SimpleNamespace(chunked_prefill_size=16, speculative_num_draft_tokens=1),
            tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(dual_d_kv=a)),
            _weg2_group_min_ints=lambda v: list(v))
        for _ in range(200):
            D.tick(sched)
        self.assertEqual(a.mapped_tokens, 96, "D shrank before P sized its pool on this card")
        K.CardKvLedger(path, "P").contribute(0)
        for _ in range(200):
            D.tick(sched)
        self.assertLess(a.mapped_tokens, 96)


class GrowOomIsCardShort(CustomTestCase):
    def _d(self, spans, gmin=lambda v: v):
        path = _card()
        led = K.CardKvLedger(path, "D")
        a = D.DKvStage([(1, _geom()), (2, _geom())], led, allocator=object(), pools=[], page_size=64,
                       granule=G, top_tokens=16384, spans=spans, step=4096, engage_cap=lambda *a: None, gmin=gmin)
        b = a.bytes_for(4096) - a.bytes_for(0)
        led.contribute(200 * MIB, committed=b)
        a.mapped_tokens, a._committed = 4096, b
        p = K.CardKvLedger(path, "P")
        p.contribute(0)
        p.request(20 * MIB)                                      # P holds context on this card
        return a, path

    def test_rc2_refuses_the_grow_rolls_back_and_presses_p(self):
        spans = FakeSpans(fail_above_bytes=8192 * ROW)
        a, path = self._d(spans)
        before = dict(K.peek(path).committed)
        with mock.patch.object(S, "phys_free_bytes", lambda: 10 * MIB):
            self.assertFalse(a.group_grow(16384))              # no exception
        self.assertEqual(a.mapped_tokens, 4096)
        self.assertEqual(K.peek(path).committed["D"], before["D"])
        last = {}
        for ptr, sp in spans.calls:
            last[ptr] = sum(int(n) for _o, n in sp)
        self.assertEqual(last[1], last[2], "the partial map was not rolled back on every tensor")
        st = K.peek(path)
        self.assertLessEqual(st.free, 10 * MIB, "the ledger still promises what the card does not have")
        self.assertGreater(st.pressure["P"], 0, "the physical shortfall did not become pressure on P")

    def test_a_rank_whose_map_worked_rolls_back_when_another_failed(self):
        seen = []

        def gmin(vals):
            seen.append(list(vals))
            return [0] if len(seen) == 2 else list(vals)           # 2nd collective: another rank failed

        a, path = self._d(FakeSpans(), gmin=gmin)
        before = K.peek(path).committed["D"]
        with mock.patch.object(S, "phys_free_bytes", lambda: 500 * MIB):
            self.assertFalse(a.group_grow(16384))
        self.assertEqual(a.mapped_tokens, 4096)
        self.assertEqual(K.peek(path).committed["D"], before)

    def test_other_map_errors_stay_fatal(self):
        class Bad(FakeSpans):
            def set_spans(self, ptr, spans, now):
                return 1

        a, _ = self._d(Bad())
        with self.assertRaises(RuntimeError):
            a.group_grow(16384)


class PhysicalSelfCheck(CustomTestCase):
    def test_throttled_and_names_an_over_promise(self):
        path = _card()
        led = K.CardKvLedger(path, "D")
        led.contribute(100 * MIB, committed=10 * MIB)
        actor = types.SimpleNamespace(ledger=led)
        lines = []
        clock = [0.0]
        with mock.patch.object(S, "_now", lambda: clock[0]), \
                mock.patch.object(S, "phys_free_bytes", lambda: 50 * MIB), \
                mock.patch.object(S.logger, "warning", lambda m, *a: lines.append(m % a)), \
                mock.patch.object(S.logger, "info", lambda m, *a: lines.append(m % a)):
            for i in range(1000):
                clock[0] = i * 0.1                               # 100 s
                S.phys_check(actor, "D")
        self.assertTrue(any("LEDGER-PHYS" in l and "OVER-PROMISE" in l for l in lines), lines[:3])
        self.assertLessEqual(len(lines), 12)


if __name__ == "__main__":
    import unittest

    unittest.main()
