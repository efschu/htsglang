"""#1068 WEG 1 slice 1 (WEG1_BUILD_SPEC_0901 section 4.1): the MiB branch of
``MambaPoolHost`` sizing is MIN-synced across ranks, like the GB branch.

THE DEFECT. ``--hicache-mamba-host-mib`` is an absolute per-rank budget whose
own help text demands rank invariance ("a divergent anchor capacity is a
divergent prefetch vote"). But the slot COUNT it buys is ``mib // per_slot``,
and per_slot differs per rank: PP stages own different mamba-layer counts
(measured 2026-08-30 on boot_855: 37.41 / 21.82 / 15.59 MiB on PP0 / PP1 /
PP2). 2400 MiB therefore bought 64 / 109 / 153 slots -- three different
anchor ceilings, and the flag's rank-invariance promise was broken by its
own arithmetic. The GB branch two lines below already routes through
``sync_fixed_hicache_size`` (group MIN); the MiB branch did not.

Hermetic: the constructor is driven with its allocation and registration
methods stubbed, and the group MIN is a recorded fake -- what is under test
is the sizing decision and the helper it must go through, not the pinned
allocation.

    CUDA_VISIBLE_DEVICES='' python -m pytest \\
        test/registered/unit/mem_cache/test_mamba_host_mib_synced_1068.py -q
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache import memory_pool_host as mph
from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024**2
ANCHOR_MIB = 2400
#: per-slot bytes of the two extreme PP ranks (spec section 5, log 1351/1331)
PER_SLOT_PP0 = int(37.41 * MIB)
PER_SLOT_PP2 = int(15.59 * MIB)
DEVICE_SLOTS = 20


def _device_pool():
    cache = types.SimpleNamespace(
        conv=[torch.zeros(1, 1, 4, dtype=torch.float16)],
        temporal=torch.zeros(1, 1, 4, dtype=torch.float16),
    )
    return types.SimpleNamespace(
        num_mamba_layers=1, mamba_cache=cache, device="cpu", size=DEVICE_SLOTS
    )


class _GroupMin:
    """A stand-in for the all_reduce MIN over the pp group: every rank's
    local count is recorded, and the group minimum is the fixed floor the
    PP0 rank sets (64 = 2400 MiB // 37.41 MiB)."""

    def __init__(self, group_min):
        self.group_min = group_min
        self.calls = []

    def __call__(self, size, host_size):
        self.calls.append((int(size), int(host_size)))
        return min(int(size), self.group_min)


def _build(per_slot, group_min_fake, **kwargs):
    with (
        mock.patch.object(MambaPoolHost, "get_size_per_token", return_value=per_slot),
        mock.patch.object(MambaPoolHost, "init_kv_buffer", lambda self: None),
        mock.patch.object(
            MambaPoolHost, "_init_write_back_staging_buffers", lambda self: None
        ),
        mock.patch.object(MambaPoolHost, "clear", lambda self: None),
        mock.patch.object(mph, "check_and_register_pinned_post", lambda **kw: None),
        mock.patch.object(mph, "sync_fixed_hicache_size", group_min_fake),
    ):
        return MambaPoolHost(_device_pool(), 1.0, 0, layout="layer_first", **kwargs)


class TestTheMiBBranchIsMinSynced(CustomTestCase):
    def test_mib_branch_is_min_synced(self):
        """T4. Two ranks with different per-slot bytes end with the SAME slot
        count, and both went through the group-MIN helper with their local
        count (64 and 153) and the MiB budget as the host_size gate."""
        group = _GroupMin(group_min=ANCHOR_MIB * MIB // PER_SLOT_PP0)  # 64
        pp0 = _build(PER_SLOT_PP0, group, anchor_host_mib=ANCHOR_MIB)
        pp2 = _build(PER_SLOT_PP2, group, anchor_host_mib=ANCHOR_MIB)
        self.assertEqual(
            group.calls,
            [(64, ANCHOR_MIB), (153, ANCHOR_MIB)],
            "the MiB branch must route through sync_fixed_hicache_size with "
            "the LOCAL slot count, exactly as the GB branch does",
        )
        # 64 synced slots + the one page of alignment every host pool adds
        # (memory_pool_host.py: page_num = size // page_size + 1).
        self.assertEqual(pp0.size, 65)
        self.assertEqual(pp2.size, pp0.size, "rank-divergent anchor ceiling")

    def test_the_provenance_line_names_the_local_count(self):
        """L9: the #1035 provenance line carries synced_from_local=<local>
        so a later reader can see which rank bound the group."""
        group = _GroupMin(group_min=64)
        with self.assertLogs("sglang.srt.mem_cache.memory_pool_host", level="INFO") as logs:
            _build(PER_SLOT_PP2, group, anchor_host_mib=ANCHOR_MIB)
        prov = [line for line in logs.output if "#1035 ANCHOR-POOL PROVENANCE" in line]
        self.assertEqual(len(prov), 1, logs.output)
        self.assertIn("synced_from_local=153", prov[0])
        self.assertIn("host_anchor_slots=65", prov[0])

    def test_the_auto_branch_is_untouched(self):
        """anchor_host_mib=0 and host_size=0 is the ratio path: no sync
        call with a positive gate (the helper returns early on host_size
        <= 0 in production; here it must simply not be reached)."""
        group = _GroupMin(group_min=64)
        pool = _build(PER_SLOT_PP2, group, anchor_host_mib=0)
        self.assertEqual(group.calls, [])
        self.assertEqual(pool.size, DEVICE_SLOTS + 1)


if __name__ == "__main__":
    unittest.main()
