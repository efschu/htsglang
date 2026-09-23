"""SGLANG_OPT_MOE_POOL_KEEP_LRU: an eager forward keeps the decode LRU it did not write.

THE MEASUREMENT. Device-planned expert pool (SGLANG_MOE_OFFLOAD_GRAPH_MODE=pool):
every eager forward (the extend of a request, and under
SGLANG_SPEC_EAGER_VERIFY=first also the first verify) ends in ``sync_tables``,
which used to FREE every LRU row the eager pass had not written. The rows' bytes
had not changed -- only the mapping was thrown away. fnFL2x100 (23.09.) MID-2, a
radix-hit repeat of the prompt D had just decoded: TP0 ``pool.fetch`` 53-68 ms in
the first graph rounds, 8-13 ms once the LRU was warm again, i.e. the working set
of the previous request was re-fetched over PCIe row by row.

WHAT MUST HOLD. (1) A row the eager pass did not write keeps expert and recency;
a written row takes the eager pass's expert; staging rows are never owned.
(2) The one invariant ``pool_step`` depends on -- ``row_key[r] == e`` iff
``hot_phys[e] == r`` over the LRU rows -- survives an eager pass that wrote an
expert a kept row still held: otherwise evicting the stale twin clears
``hot_phys[e]`` in the same step that hits ``e`` in the live row, and ``e`` is
routed to -1 (its contribution silently dropped). (3) keep off = the old form,
byte-for-byte: every unwritten LRU row is freed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import random
import unittest

import torch

from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.test.test_utils import CustomTestCase

# residents 0,1 in rows 0,1; LRU rows 2..5; staging rows 6,7
E, ROWS, R, S = 10, 8, 2, 2
HOST = [-1, -1] + list(range(E - 2))


def _pool():
    t = ep.allocate_pool_tables("cpu", E, ROWS, R, S, {0: 0, 1: 1}, HOST)
    b = ep.allocate_step_buffers("cpu", E, 8)
    return t, b


def _ids(*v):
    return torch.tensor(v, dtype=torch.int32)


def _assert_bijection(tc, t):
    hot, key = t.hot_phys.tolist(), t.row_key.tolist()
    for r in range(t.lru_start, t.pool_rows):
        if key[r] >= 0:
            tc.assertEqual(
                hot[key[r]], r, f"row {r} holds {key[r]} but hot says {hot[key[r]]}"
            )
    for e, r in enumerate(hot):
        if r >= t.lru_start:
            tc.assertEqual(key[r], e, f"expert {e} -> row {r} but row holds {key[r]}")
    for r in range(t.pool_rows, len(key)):
        tc.assertEqual(key[r], -1, "a staging row is never owned")


class TestPoolSyncKeepLru(CustomTestCase):
    def test_unwritten_rows_keep_expert_and_recency_written_rows_take_the_eager_expert(
        self,
    ):
        t, b = _pool()
        ep.step_reference(
            t, _ids(2, 3, 4, 5), b
        )  # clock 1: LRU rows 2..5 = experts 2..5
        ep.step_reference(t, _ids(5), b)  # clock 2: row 5 is the most recent
        use_before = t.row_use.tolist()
        # the eager pass wrote row 2 (expert 7) and staging row 6 (expert 8)
        owned = ep.sync_tables(t, {2: 7, 6: 8}, keep_unwritten=True).owned
        self.assertEqual(t.row_key.tolist()[2:6], [7, 3, 4, 5])
        self.assertEqual(t.row_use.tolist()[3:6], use_before[3:6])
        self.assertEqual(owned, 4)
        hot = t.hot_phys.tolist()
        self.assertEqual((hot[2], hot[7], hot[8]), (-1, 2, -1))
        _assert_bijection(self, t)
        # the kept working set is a hit, not a PCIe re-fetch
        pairs, _ = ep.step_reference(t, _ids(3, 4, 5, 7), b)
        self.assertEqual(pairs, [])

    def test_keep_off_is_the_old_form_every_unwritten_row_is_freed(self):
        t, b = _pool()
        ep.step_reference(t, _ids(2, 3, 4, 5), b)
        owned = ep.sync_tables(t, {2: 7}, keep_unwritten=False).owned
        self.assertEqual(t.row_key.tolist(), [0, 1, 7, -1, -1, -1, -1, -1])
        self.assertEqual(owned, 1)
        pairs, _ = ep.step_reference(t, _ids(3), b)
        self.assertEqual(len(pairs), 1)  # the previous working set is gone

    def test_an_expert_rewritten_into_a_new_row_has_one_owner_and_is_never_routed_to_minus_one(
        self,
    ):
        t, b = _pool()
        ep.step_reference(t, _ids(2, 3, 4, 5), b)  # expert 3 in row 3
        # eager pass wrote expert 3 again, into row 2 (row 3 untouched)
        ep.sync_tables(t, {2: 3}, keep_unwritten=True)
        _assert_bijection(self, t)
        self.assertEqual(t.hot_phys.tolist()[3], 2)
        self.assertEqual(t.row_key.tolist()[3], -1)  # the stale twin is free
        # a step that hits 3 and misses enough to evict the old twin row
        ep.step_reference(t, _ids(3, 6, 7, 8), b)
        self.assertNotIn(-1, b.routes[:4].tolist())
        self.assertEqual(b.routes[0].item(), t.hot_phys.tolist()[3])
        _assert_bijection(self, t)

    def test_seeded_eager_decode_interleaving_keeps_every_valid_route_on_its_expert(
        self,
    ):
        rng = random.Random(516)
        for _ in range(40):
            t, b = _pool()
            for _round in range(12):
                if rng.random() < 0.3:
                    lo, rows = t.lru_start, int(t.row_key.shape[0])
                    written = rng.sample(range(lo, rows), rng.randint(0, rows - lo))
                    holds = {r: rng.randrange(2, E) for r in written}
                    ep.sync_tables(t, holds, keep_unwritten=True)
                    _assert_bijection(self, t)
                ids = [rng.randrange(-1, E) for _ in range(4)]
                ep.step_reference(t, torch.tensor(ids, dtype=torch.int32), b)
                key, routes = t.row_key.tolist(), b.routes.tolist()
                for lane, e in enumerate(ids):
                    if e < 0:
                        continue
                    row = routes[lane]
                    self.assertGreaterEqual(row, 0, f"expert {e} routed to -1")
                    if row < t.pool_rows:
                        self.assertEqual(key[row], e)
                _assert_bijection(self, t)


if __name__ == "__main__":
    unittest.main()
