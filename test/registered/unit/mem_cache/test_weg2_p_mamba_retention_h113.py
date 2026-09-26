"""H113: group P of a weg2 boot refuses a mamba retention pin budget of 0.

MEASURED, boot fnFL2h91bb2 (tree e17bd548b5, 2026-09-26), P log
``/spinning/evidence-665-f1/boot_weg2_fnFL2h91bb2_e17bd548b5_0926_142251.P.log``:

* Z. 2577 (14:25:54): ``MAMBA-FLOOR pool=32 floor=32 retention_budget=0 (8
  running requests x (1 active + 1 ping-pong + 1 donation + 1 pinned
  checkpoint) = 8 x 4 = 32 slots) -- ... every mamba host backup will be
  declined`` -- stated, not refused; the arm had passed
  ``--max-mamba-cache-size 32`` believing 2 slots per seat (budget 16).
* Z. 15270-15275 (14:30:41, chunk 2 of the 97k needle weg2-0-4): ``mamba
  write-through pin budget reached (0 in flight, budget=0, pool=32)``,
  ``#1421 BACKUP-REFUSED why=mamba_pin node=9`` (chunk 1's end node, the
  first one carrying a mamba value), then ``why=parent_unbacked`` for every
  node after it; RETAIN-PUBLISH ``issued=0 refused=217``, no ``MAMBA-ARENA``
  line at all.
* D log Z. 13060-14078: ``#1439 ARENA-PRESENT keys=1528 leading_complete=192``
  (the three KV-only pieces of chunk 1), ``#1028B FETCH CAP mamba (0,-1)``,
  ``#1035c ZERO-ANSWER cause=CAPPED by=mamba``, ``#1471 SETTLE read still
  short`` -- a read that can never complete.

Reference h91v1 (39fd662d9e): ``pool=24 floor=16 retention_budget=8`` at 4
seats, no refusal, ``leading_complete=1528`` and SETTLE-RELEASE after 0.3 s.

WHAT MUST HOLD.
(1) bb2 form (P, 8 seats, pool 32, write_back): the REAL UnifiedRadixCache
    refuses at construction, named, with the minimum and the reference value.
(2) h91v1 form (P, 4 seats, pool 24) and the fixed bb2 form (pool 40) build.
(3) Group D and a non-weg2 boot keep the #773 posture: budget 0 stated only.
(4) The 4 slots per seat and the write_back question: write_back takes no
    reorder rebate (4), write_through + reorder takes it (3), + #811
    ack-release (2) -- the 2 the launcher's pool model and the arm assumed.
"""

import os
import unittest
from unittest import mock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10)

NUM_LAYERS = 8
FULL_LAYER_IDS = (3, 7)
NON_FULL_LAYER_IDS = [i for i in range(NUM_LAYERS) if i not in set(FULL_LAYER_IDS)]


def _p_server_args(seats: int, write_policy: str = "write_back") -> ServerArgs:
    """Group P's floor-relevant surface as fnFL2h91bb2 ran it (P argv:
    --enable-hierarchical-cache --hicache-write-policy write_back
    --disable-overlap-schedule, extra-buffer strategy -> 1 ping-pong)."""
    sa = ServerArgs(model_path="dummy", page_size=1)
    sa._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    sa.max_running_requests = seats
    sa.enable_hierarchical_cache = True
    sa.hicache_write_policy = write_policy
    sa.disable_overlap_schedule = True
    sa.disable_radix_cache = False
    sa.mamba_radix_cache_strategy = "extra_buffer"
    return sa


def _build(pool_slots: int, sa: ServerArgs) -> UnifiedRadixCache:
    set_global_server_args_for_scheduler(sa)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
            head_dim=16, state_size=16, conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=NON_FULL_LAYER_IDS)
    pool = HybridReqToTokenPool(
        size=10, mamba_size=pool_slots, mamba_spec_state_size=10,
        max_context_len=256, device="cpu", enable_memory_saver=False,
        cache_params=cache_params, mamba_layer_ids=NON_FULL_LAYER_IDS,
        enable_mamba_extra_buffer=False, speculative_num_draft_tokens=3,
    )
    kv_pool = HybridLinearKVPool(
        size=256, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
        full_attention_layer_ids=list(FULL_LAYER_IDS), device="cpu",
        enable_memory_saver=False, mamba_pool=pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=256, dtype=torch.bfloat16, device="cpu", kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=pool, token_to_kv_pool_allocator=allocator,
        page_size=1, disable=False, sliding_window_size=None,
        tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
        eviction_policy="lru", is_eagle=False,
    )
    return UnifiedRadixCache(params=params)


def _env(group):
    env = {k: v for k, v in os.environ.items() if k not in ("SGLANG_WEG2_GROUP", "SGLANG_MAMBA_SLOT_REORDER")}
    if group is not None:
        env["SGLANG_WEG2_GROUP"] = group
    return mock.patch.dict(os.environ, env, clear=True)


class TestWeg2PMambaRetentionH113(CustomTestCase):
    def test_bb2_form_is_refused_at_construction(self):
        """(1) P, 8 seats, pool 32 = floor 32 -> budget 0 -> named refusal."""
        from sglang.srt.mem_cache import mamba_pool_floor as mpf

        refusal_cls = getattr(mpf, "Weg2PMambaRetentionZero", None)
        with _env("P"):
            try:
                _build(32, _p_server_args(8))
            except Exception as exc:  # noqa: BLE001 -- asserted below
                raised = exc
            else:
                raised = None
        self.assertIsNotNone(
            raised,
            "group P built a tree whose mamba pin budget is 0 -- every anchor "
            "backup will be declined and D's hand-off read can never complete "
            "(fnFL2h91bb2)",
        )
        self.assertIsNotNone(refusal_cls)
        self.assertIsInstance(raised, refusal_cls)
        msg = str(raised)
        self.assertIn("H113", msg)
        self.assertIn("8 x 4 = 32", msg)
        self.assertIn("--max-mamba-cache-size 33", msg)  # floor + budget 1
        self.assertIn("i.e. 40 here", msg)  # h91v1's budget 8
        self.assertIn("at most 7", msg)  # (32 - 1) // 4

    def test_reference_and_fixed_forms_build(self):
        """(2) h91v1 (4 seats, pool 24 -> budget 8) and bb2 with pool 40."""
        with _env("P"):
            c = _build(24, _p_server_args(4))
            self.assertEqual(c._mamba_pin_budget, 8)
            c = _build(40, _p_server_args(8))
            self.assertEqual(c._mamba_pin_budget, 8)

    def test_group_d_and_non_weg2_keep_the_posture(self):
        """(3) budget 0 outside group P is stated, not refused."""
        for group in ("D", None):
            with _env(group):
                c = _build(32, _p_server_args(8))
                self.assertEqual(c._mamba_pin_budget, 0, group)

    def test_p_without_host_tier_is_not_refused(self):
        """(3) no hierarchical cache -> no host backups are expected at all."""
        sa = _p_server_args(8)
        sa.enable_hierarchical_cache = False
        with _env("P"):
            c = _build(32, sa)
            self.assertEqual(c._mamba_pin_budget, 0)

    def test_four_slots_per_seat_and_the_write_back_question(self):
        """(4) the per-seat derivation the refusal prints."""
        from sglang.srt.mem_cache.mamba_pool_floor import (
            describe_mamba_floor,
            mamba_slots_per_running_req,
        )

        with _env("P"):
            wb = _p_server_args(8, "write_back")
            self.assertEqual(mamba_slots_per_running_req(wb), 4)
            self.assertIn("1 active + 1 ping-pong + 1 donation + 1 pinned checkpoint",
                          describe_mamba_floor(wb, 8))
            os.environ["SGLANG_MAMBA_SLOT_REORDER"] = "1"
            # --mamba-slot-reorder is on P's argv, but write_back blocks it.
            self.assertEqual(mamba_slots_per_running_req(wb), 4)
            wt = _p_server_args(8, "write_through")
            self.assertEqual(mamba_slots_per_running_req(wt), 3)
            wt.mamba_anchor_ack_release = True
            self.assertEqual(mamba_slots_per_running_req(wt), 2)


if __name__ == "__main__":
    unittest.main()
