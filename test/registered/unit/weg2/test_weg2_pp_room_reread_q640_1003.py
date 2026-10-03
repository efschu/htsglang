# SPDX-License-Identifier: Apache-2.0
"""Q-640 PP-ROOM-REREAD (27B dual y8u, boot dkr27bnvfp4dual1mpsleepbar1fs10031206, 12:33:47).

rid weg2-0-51, kv_tokens=42084 load-back on the local-PP floor (TP=1/PP=3):

  PP0/PP1  evictable=84869 -> SF LOADBACK-ROOM SAME-PASS, admitted
  PP2      evictable=2081 (node 79 held by an in-flight write-back) -> evicted node 80,
           the eviction drained the write-back (#1465 WRITE-BACK DRAIN n=1), node 79
           (82788 tokens) became evictable -- but the verdict came from the ONE read
           before the eviction: SF LOADBACK-ROOM PP-RESIDUAL avail_after=3072
  PP2      #968 PREFIX MATERIALISATION SHORTFALL -> rank death

The room verdict re-reads the evictable set after each eviction round, so PP2 admits in
the same pass as its peers.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

KV_TOKENS = 42084
FLOOR = 991            # PP2 avail at the verdict
NODE_80 = 2081         # PP2's evictable set at the first read
NODE_79 = 82788        # held by the in-flight write-back until the first eviction drained it
PEER_EVICTABLE = 84869  # PP0/PP1


class _Alloc:
    def __init__(self, avail):
        self.avail = avail

    def available_size(self):
        return self.avail


class _Tree:
    """A rank's tree: ``held`` tokens become evictable when an eviction drains the
    in-flight write-backs (the real evict flushes them, #1465)."""

    def __init__(self, avail, evictable, held=0):
        self.token_to_kv_pool_allocator = _Alloc(avail)
        self._evictable = evictable
        self._held = held
        self.evicted = []
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def evictable_size(self):
        return self._evictable

    def evict(self, params):
        n = min(int(params.num_tokens), self._evictable)
        self._evictable -= n
        self.token_to_kv_pool_allocator.avail += n
        self.evicted.append(n)
        self._evictable += self._held  # the drain released the held node
        self._held = 0
        return SimpleNamespace(num_tokens_evicted=n)


def _on():
    return mock.patch.dict(os.environ, {SF.ENV: "1"})


class TheRoomVerdictRereadsTheEvictableSet(CustomTestCase):
    def test_y8u_pp2_admits_in_the_same_pass_as_its_peers(self):
        with _on():
            pp2 = _Tree(FLOOR, NODE_80, held=NODE_79)
            peers = [_Tree(FLOOR, PEER_EVICTABLE) for _ in range(2)]
            verdicts = [SF.local_pp_room(t, KV_TOKENS, FLOOR, "weg2-0-51") for t in peers + [pp2]]
        self.assertEqual(verdicts, [True, True, True], "PP2 refused the pass PP0/PP1 admitted (#968)")
        self.assertEqual(pp2.evicted, [NODE_80, KV_TOKENS - FLOOR - NODE_80])
        self.assertGreaterEqual(pp2.token_to_kv_pool_allocator.avail, KV_TOKENS)

    def test_the_line_names_the_reread(self):
        with _on(), self.assertLogs(SF.logger, level="INFO") as cap:
            SF.local_pp_room(_Tree(FLOOR, NODE_80, held=NODE_79), KV_TOKENS, FLOOR, "weg2-0-51")
        line = [l for l in cap.output if "SAME-PASS" in l][0]
        self.assertIn(
            "kv_tokens=42084 floor=991 avail=991 evictable=2081 evicted=41093 avail_after=42084 "
            "evictable_after=43776 rounds=2", line)

    def test_a_true_residual_still_refuses_and_stops(self):
        # nothing becomes evictable: one round, then the residual -- no spin.
        with _on(), self.assertLogs(SF.logger, level="WARNING") as cap:
            t = _Tree(FLOOR, NODE_80)
            self.assertFalse(SF.local_pp_room(t, KV_TOKENS, FLOOR, "weg2-0-51"))
        self.assertEqual(t.evicted, [NODE_80])
        self.assertIn("PP-RESIDUAL", cap.output[0])
        self.assertIn("evictable_after=0 rounds=1", cap.output[0])

    def test_a_round_that_frees_nothing_ends_the_loop(self):
        class _Stuck(_Tree):
            def evict(self, params):
                self.evicted.append(0)
                return SimpleNamespace(num_tokens_evicted=0)

        with _on():
            t = _Stuck(FLOOR, NODE_80)
            self.assertFalse(SF.local_pp_room(t, KV_TOKENS, FLOOR, "weg2-0-51"))
        self.assertEqual(t.evicted, [0])


if __name__ == "__main__":
    unittest.main()
