"""#852 R3: the fourth term is the CUDA-graph private pool.

THE SPECIMEN. W25 (boot_w25_0824_1125.log) printed, five times, stable to the
MiB, while the general cache drifted 310 -> 308 -> 307 -> 305 MiB:

    PHASE-FLIP staging reclaim: driver free 2896 -> 2896 MiB
      (+0 returned, predicted releasable 88 MiB), 310 MiB still cached

A NONZERO PREDICTION AGAINST A ZERO DELIVERY. #852's own commit text names
that reading: it "falsifies it and indicts this estimator instead". The
phantom was down 3.6x from W24's ~309-324 MiB and the cause split moved from
43/2 to 8/2 phantom/scarcity -- but 88 MiB survived, and a term that does not
move while its neighbours do is not fragmentation. It is a fixed structure.

THE STRUCTURE. `speculative/adaptive_graph_memory.py:841` takes a PER-TAG
`torch.cuda.graph_pool_handle()` and `:873` captures into it with
`torch.cuda.graph(cuda_graph, pool=pool, stream=stream)`. This boot captured
decode graphs (`backend='full'`, bs 1..24). Torch's `release_cached_blocks`
frees whole blocks from the GENERAL `large_blocks`/`small_blocks` pools only;
a private pool goes back solely via `graph_pools_freeable`, once nothing
references the graph. Meanwhile `reserved_bytes.all` / `allocated_bytes.all` /
`inactive_split_bytes.all` are device-global and COUNT those segments. So
`reserved - allocated - inactive_split` promises bytes the driver can never
be handed.

Torch's own data structure is the citation: `SegmentInfo` carries
`owner_private_pool_id = {0, 0}` (c10/core/CachingDeviceAllocator.h:98) and
the snapshot exposes it as `segment_pool_id` -- which `torch/_inductor/
cudagraph_trees.py:1827` uses for exactly this purpose, selecting the
segments belonging to a graph pool.

RED-FIRST: `releasable_cache_bytes_from_segments` and
`graph_pool_free_bytes_from_segments` did not exist, and the shipped
three-term arithmetic returns the phantom on the specimen below (asserted in
`TestTheProxyReallyDoesOverPromise`, which is what makes this a measured
defect rather than a story about one).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    graph_pool_free_bytes_from_segments,
    releasable_cache_bytes_from_segments,
    releasable_cache_bytes_from_stats,
)
from sglang.test.test_utils import CustomTestCase

MIB = 1024 * 1024

#: The W25 shape: a general pool with everything fragmented inside in-use
#: segments (so nothing is releasable), plus a graph private pool holding
#: 88 MiB of free space that empty_cache() can never return.
GRAPH_POOL_ID = (7, 3)


def _seg(total_mib, allocated_mib, pool=(0, 0), expandable=False):
    return {
        "total_size": total_mib * MIB,
        "allocated_size": allocated_mib * MIB,
        "segment_pool_id": pool,
        "is_expandable": expandable,
    }


def _w25_segments():
    return [
        # General pool: in use, with free space SPLIT inside live segments.
        _seg(200, 150),
        _seg(120, 100),
        # The graph private pool: 88 MiB free, structurally unreleasable.
        _seg(64, 0, pool=GRAPH_POOL_ID),
        _seg(32, 8, pool=GRAPH_POOL_ID),
    ]


class TestTheProxyReallyDoesOverPromise(CustomTestCase):
    """The shipped arithmetic, on the specimen, produces the phantom."""

    def test_the_three_term_proxy_counts_the_graph_pool(self):
        segs = _w25_segments()
        reserved = sum(s["total_size"] for s in segs)
        allocated = sum(s["allocated_size"] for s in segs)
        # Every free byte in the GENERAL pool is split inside a live segment
        # (that is what W25's "310 MiB still cached, 0 returned" means), so
        # inactive_split covers the general pool's free space and nothing else.
        inactive_split = (200 - 150) * MIB + (120 - 100) * MIB
        promised = releasable_cache_bytes_from_stats(
            {
                "reserved_bytes.all.current": reserved,
                "allocated_bytes.all.current": allocated,
                "inactive_split_bytes.all.current": inactive_split,
            }
        )
        # 64 + 24 = 88 MiB, the exact figure W25 printed five times.
        self.assertEqual(promised, 88 * MIB)


class TestTheSegmentViewIsExact(CustomTestCase):
    def test_a_graph_pool_is_never_promised(self):
        self.assertEqual(releasable_cache_bytes_from_segments(_w25_segments()), 0)

    def test_the_fourth_term_is_named_and_measured(self):
        self.assertEqual(graph_pool_free_bytes_from_segments(_w25_segments()), 88 * MIB)

    def test_a_wholly_free_general_segment_IS_promised(self):
        # THE CAN-FAIL DIRECTION. An implementation that returned 0 for
        # everything would pass every assertion above while destroying the
        # draw that #852 exists to keep paying.
        segs = _w25_segments() + [_seg(48, 0)]
        self.assertEqual(releasable_cache_bytes_from_segments(segs), 48 * MIB)

    def test_a_partly_used_general_segment_is_not_promised(self):
        # empty_cache frees whole free segments only; one live byte pins it.
        self.assertEqual(releasable_cache_bytes_from_segments([_seg(48, 1)]), 0)

    def test_an_expandable_segment_is_not_promised(self):
        segs = [_seg(48, 0, expandable=True), _seg(16, 0)]
        self.assertEqual(releasable_cache_bytes_from_segments(segs), 16 * MIB)


class TestItAbstainsRatherThanGuesses(CustomTestCase):
    def test_expandable_segments_env_still_abstains(self):
        # UNCHANGED ON PURPOSE: under that allocator `reserved` is a virtual
        # extent and the comparison is void. Under-reporting here suppresses
        # a draw that would have paid and makes the flip STICKIER -- the
        # precise defect #852 exists to remove.
        self.assertIsNone(
            releasable_cache_bytes_from_segments(
                _w25_segments(), alloc_conf="expandable_segments:True"
            )
        )

    def test_an_unreadable_snapshot_is_None_not_zero(self):
        # "no verdict" and "no trapped bytes" must never collapse.
        self.assertIsNone(releasable_cache_bytes_from_segments([]))
        self.assertIsNone(releasable_cache_bytes_from_segments(None))
        self.assertIsNone(graph_pool_free_bytes_from_segments(None))

    def test_a_missing_pool_field_reads_as_the_general_pool(self):
        # A torch version that drops the field must degrade to the old,
        # already-shipped reading -- never to a wrong subtraction.
        seg = {"total_size": 16 * MIB, "allocated_size": 0}
        self.assertEqual(releasable_cache_bytes_from_segments([seg]), 16 * MIB)
        self.assertEqual(graph_pool_free_bytes_from_segments([seg]), 0)

    def test_the_owner_private_pool_id_spelling_is_accepted_too(self):
        seg = {
            "total_size": 16 * MIB,
            "allocated_size": 0,
            "owner_private_pool_id": GRAPH_POOL_ID,
        }
        self.assertEqual(releasable_cache_bytes_from_segments([seg]), 0)
        self.assertEqual(graph_pool_free_bytes_from_segments([seg]), 16 * MIB)


if __name__ == "__main__":
    unittest.main()
