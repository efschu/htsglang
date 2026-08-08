# SPDX-License-Identifier: Apache-2.0
"""#631 slice 5.3b, the production GDN state mover: hermetic tests.

Gates, mapped to DESIGN_631 3.4 and the operator's 5.3b holds:

* BIT IDENTITY (hold 1: CPU-sampled fixtures, cross-arch rule): reference
  GDN states sampled on CPU survive PP -> TP (layer-axis -> head-axis)
  exactly -- every rank's compact shard equals the reference slice -- and
  the TP -> PP return trip restores the PP pools byte-identically.
* REACHABLE REFUSAL (hold 2): the mover's preconditions re-validate on
  EVERY flip; each red arm breaks one structural assumption (ReplaySSM
  buffers, int8 checkpoint pool, shard-spec vs actual TP tensor shape,
  slot-space divergence) and must refuse loudly BEFORE any byte moves --
  a regression degrades to refusal, never to silent no-move (#212).
* channel discipline inherited from the KV runtime: size mismatch and
  checksum corruption from a peer are loud errors with pools untouched.
"""

import os
import threading
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.distributed.utils import (
    set_cp_token_ratios,
    set_tp_partition_ratios,
    tp_partition_size,
)
from sglang.srt.layers.dcp.gdn_flip_plan import conv_slice, temporal_slice
from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.managers.gdn_flip_mover import (
    GdnFlipMover,
    derive_pp_linear_layer_map,
    gdn_flip_preconditions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_phase_flip_runtime import _MailboxExchange  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

N_RANKS = 3
# Small GDN-like geometry: conv [q|k|v] = [12|12|36], partitioned in 6
# indivisible units (`units` is a UNIT COUNT: key unit 2 channels, value
# unit 6 channels = exactly 1 of the 6 temporal heads), 6 linear layers
# of an 8-layer model (full-attn at 3 and 7).
SUB_SIZES, UNITS, CONV_DIM = [12, 12, 36], 6, 60
HEADS, HEAD_D, STATE_N = 6, 2, 3
CONV_W = 4
LINEAR_IDS = [0, 1, 2, 4, 5, 6]
N_HIDDEN = 8
N_SLOTS = 5
DTYPE = torch.bfloat16


def _mamba_pool_stub(conv, temporal, size, replayssm=False):
    return SimpleNamespace(
        mamba_cache=SimpleNamespace(
            conv=[conv],
            temporal=temporal,
            replayssm_d=(torch.zeros(1) if replayssm else None),
        ),
        enable_linear_replayssm=replayssm,
        get_conv_subblock_spec=lambda: (list(SUB_SIZES), UNITS, CONV_DIM),
        size=size,
    )


def _req_pool_stub(mamba_pool, mamba_map, ckpt=None):
    return SimpleNamespace(
        mamba_pool=mamba_pool, mamba_map=dict(mamba_map), mamba_ckpt_pool=ckpt
    )


def _build_world(seed=11):
    """CPU-sampled reference states + per-rank PP (full) and TP (compact,
    zeroed) req-pool stubs, shard shapes derived through the REAL
    partition functions so fixtures cannot drift from the mover's math."""
    g = torch.Generator().manual_seed(seed)
    stage_ids = derive_pp_linear_layer_map(LINEAR_IDS, N_HIDDEN, N_RANKS)
    ref_conv = {
        gid: torch.randn(N_SLOTS, CONV_DIM, CONV_W, generator=g).to(DTYPE)
        for gid in LINEAR_IDS
    }
    ref_temp = {
        gid: torch.randn(N_SLOTS, HEADS, HEAD_D, STATE_N, generator=g).to(DTYPE)
        for gid in LINEAR_IDS
    }
    pp_req, tp_req = [], []
    for r in range(N_RANKS):
        mine = stage_ids[r]
        conv = torch.stack([ref_conv[gid] for gid in mine])  # [L,S,C,W]
        temp = torch.stack([ref_temp[gid] for gid in mine])
        pp_req.append(
            _req_pool_stub(
                _mamba_pool_stub(conv.clone(), temp.clone(), N_SLOTS),
                {gid: i for i, gid in enumerate(mine)},
            )
        )
        conv_c = sum(
            tp_partition_size(s, N_RANKS, r, UNITS) for s in SUB_SIZES
        )
        heads_r = tp_partition_size(SUB_SIZES[-1], N_RANKS, r, UNITS) // (
            SUB_SIZES[-1] // HEADS
        )
        tp_req.append(
            _req_pool_stub(
                _mamba_pool_stub(
                    torch.zeros(
                        len(LINEAR_IDS), N_SLOTS, conv_c, CONV_W, dtype=DTYPE
                    ),
                    torch.zeros(
                        len(LINEAR_IDS),
                        N_SLOTS,
                        heads_r,
                        HEAD_D,
                        STATE_N,
                        dtype=DTYPE,
                    ),
                    N_SLOTS,
                ),
                {gid: i for i, gid in enumerate(LINEAR_IDS)},
            )
        )
    return stage_ids, ref_conv, ref_temp, pp_req, tp_req


def _run_movers(movers, direction, rounds=1):
    errors = [None] * len(movers)

    def _worker(r):
        try:
            for _ in range(rounds):
                movers[r].move(direction)
        except BaseException as e:  # noqa: BLE001
            errors[r] = e

    threads = [
        threading.Thread(target=_worker, args=(r,)) for r in range(len(movers))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    if any(t.is_alive() for t in threads):
        raise AssertionError("GDN mover hang")
    return errors


def _make_movers(stage_ids, pp_req, tp_req, live, exchange_factory=None):
    mailbox = _MailboxExchange(N_RANKS)
    if exchange_factory is None:
        exchange_factory = mailbox.exchange_for
    movers = []
    for r in range(N_RANKS):
        movers.append(
            GdnFlipMover(
                n_ranks=N_RANKS,
                rank=r,
                stage_layer_ids=stage_ids,
                pools_fn=(
                    lambda r=r: gdn_flip_preconditions(
                        pp_req[r], tp_req[r], N_RANKS, r
                    )
                ),
                slots_fn=lambda: live,
                exchange=exchange_factory(r),
            )
        )
    return movers


class TestGdnFlipBitIdentity(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None, families=None)
        set_cp_token_ratios(None)
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)

    tearDown = setUp

    def test_pp_to_tp_shards_match_reference_and_roundtrip_restores(self):
        stage_ids, ref_conv, ref_temp, pp_req, tp_req = _build_world()
        live = torch.tensor([0, 2, 3], dtype=torch.int64)
        movers = _make_movers(stage_ids, pp_req, tp_req, live)

        errors = _run_movers(movers, PP_TO_TP)
        self.assertEqual([e for e in errors if e], [], errors)
        spec = gdn_flip_preconditions(pp_req[0], tp_req[0], N_RANKS, 0).spec
        for r in range(N_RANKS):
            pools = gdn_flip_preconditions(pp_req[r], tp_req[r], N_RANKS, r)
            for gid in LINEAR_IDS:
                li = pools.tp_index[gid]
                for s in live.tolist():
                    self.assertTrue(
                        torch.equal(
                            pools.tp_conv[li, s],
                            conv_slice(ref_conv[gid][s], spec, r),
                        ),
                        f"rank {r} layer {gid} slot {s} conv",
                    )
                    self.assertTrue(
                        torch.equal(
                            pools.tp_temporal[li, s],
                            temporal_slice(ref_temp[gid][s], spec, r),
                        ),
                        f"rank {r} layer {gid} slot {s} temporal",
                    )

        # Return trip: wipe the PP pools, then TP -> PP must restore the
        # live slots bit-identically from the shards alone.
        for r in range(N_RANKS):
            pp_req[r].mamba_pool.mamba_cache.conv[0].zero_()
            pp_req[r].mamba_pool.mamba_cache.temporal.zero_()
        errors = _run_movers(movers, TP_TO_PP)
        self.assertEqual([e for e in errors if e], [], errors)
        for r in range(N_RANKS):
            pools = gdn_flip_preconditions(pp_req[r], tp_req[r], N_RANKS, r)
            for gid in stage_ids[r]:
                li = pools.pp_index[gid]
                for s in live.tolist():
                    self.assertTrue(
                        torch.equal(pools.pp_conv[li, s], ref_conv[gid][s]),
                        f"rank {r} layer {gid} slot {s} conv restore",
                    )
                    self.assertTrue(
                        torch.equal(pools.pp_temporal[li, s], ref_temp[gid][s]),
                        f"rank {r} layer {gid} slot {s} temporal restore",
                    )

    def test_weighted_plan_fixture_consistency(self):
        """With the production-shaped weighted plan installed, the derived
        spec still partitions and validates against fixtures built through
        the same real partition functions."""
        set_tp_partition_ratios([30, 17, 17], families=None)
        stage_ids, ref_conv, ref_temp, pp_req, tp_req = _build_world(seed=13)
        live = torch.tensor([1, 4], dtype=torch.int64)
        movers = _make_movers(stage_ids, pp_req, tp_req, live)
        errors = _run_movers(movers, PP_TO_TP)
        self.assertEqual([e for e in errors if e], [], errors)
        # Shard widths differ across ranks under the plan (the point).
        widths = {
            int(tp_req[r].mamba_pool.mamba_cache.conv[0].shape[2])
            for r in range(N_RANKS)
        }
        self.assertGreater(len(widths), 1, "plan produced a uniform split")

    def test_empty_slot_set_is_a_cheap_noop_with_validation(self):
        stage_ids, _, _, pp_req, tp_req = _build_world(seed=17)
        live = torch.empty(0, dtype=torch.int64)
        movers = _make_movers(stage_ids, pp_req, tp_req, live)
        errors = _run_movers(movers, PP_TO_TP)
        self.assertEqual([e for e in errors if e], [], errors)
        self.assertEqual(movers[0].last_stats["slots"], 0)


class TestReachableRefusal(CustomTestCase):
    """Hold 2: every broken assumption refuses loudly, never no-moves."""

    def setUp(self):
        set_tp_partition_ratios(None, families=None)
        set_cp_token_ratios(None)

    tearDown = setUp

    def _world(self, **kw):
        return _build_world(**kw)

    def test_replayssm_refused(self):
        _, _, _, pp_req, tp_req = self._world()
        pp_req[0].mamba_pool.mamba_cache.replayssm_d = torch.zeros(1)
        with self.assertRaisesRegex(KvReshardError, "ReplaySSM"):
            gdn_flip_preconditions(pp_req[0], tp_req[0], N_RANKS, 0)

    def test_checkpoint_pool_refused(self):
        _, _, _, pp_req, tp_req = self._world()
        tp_req[1].mamba_ckpt_pool = object()
        with self.assertRaisesRegex(KvReshardError, "checkpoint pool"):
            gdn_flip_preconditions(pp_req[1], tp_req[1], N_RANKS, 1)

    def test_shard_spec_vs_actual_tensor_mismatch_refused(self):
        """THE regression arm: a TP pool whose conv width disagrees with
        the plan-derived spec must refuse -- the mis-sliced-conv-state
        class can never pass silently."""
        _, _, _, pp_req, tp_req = self._world()
        cache = tp_req[2].mamba_pool.mamba_cache
        cache.conv[0] = cache.conv[0][:, :, :-2, :].clone()  # wrong width
        with self.assertRaisesRegex(KvReshardError, "channels but the"):
            gdn_flip_preconditions(pp_req[2], tp_req[2], N_RANKS, 2)

    def test_slot_space_divergence_refused(self):
        _, _, _, pp_req, tp_req = self._world()
        cache = tp_req[0].mamba_pool.mamba_cache
        cache.conv[0] = torch.zeros(
            len(LINEAR_IDS), N_SLOTS + 2, cache.conv[0].shape[2], CONV_W,
            dtype=DTYPE,
        )
        with self.assertRaisesRegex(KvReshardError, "slot spaces differ"):
            gdn_flip_preconditions(pp_req[0], tp_req[0], N_RANKS, 0)

    def test_mover_revalidates_on_every_flip(self):
        """pools_fn runs per move: break a pool AFTER a good flip; the
        NEXT flip must refuse (the reachable-refusal contract survives the
        mover's landing)."""
        stage_ids, _, _, pp_req, tp_req = self._world()
        live = torch.tensor([0], dtype=torch.int64)
        movers = _make_movers(stage_ids, pp_req, tp_req, live)
        errors = _run_movers(movers, PP_TO_TP)
        self.assertEqual([e for e in errors if e], [], errors)
        for r in range(N_RANKS):
            pp_req[r].mamba_pool.mamba_cache.replayssm_d = torch.zeros(1)
        errors = _run_movers(movers, TP_TO_PP)
        for r, e in enumerate(errors):
            self.assertIsInstance(e, KvReshardError, f"rank {r}: {e!r}")
            self.assertIn("ReplaySSM", str(e))

    def test_corrupted_peer_payload_is_loud(self):
        stage_ids, _, _, pp_req, tp_req = self._world()
        live = torch.tensor([0, 1], dtype=torch.int64)
        mailbox = _MailboxExchange(N_RANKS)

        def _factory(r):
            inner = mailbox.exchange_for(r)

            def _exchange(outgoing, incoming_nbytes):
                received = inner(outgoing, incoming_nbytes)
                if r == 1:
                    for peer, payload in received.items():
                        payload[0] ^= 0xFF
                        break
                return received

            return _exchange

        movers = _make_movers(
            stage_ids, pp_req, tp_req, live, exchange_factory=_factory
        )
        errors = _run_movers(movers, PP_TO_TP)
        self.assertIsInstance(errors[1], KvReshardError)
        self.assertIn("checksum", str(errors[1]))

    def test_truncated_peer_payload_is_loud(self):
        stage_ids, _, _, pp_req, tp_req = self._world()
        live = torch.tensor([0], dtype=torch.int64)
        mailbox = _MailboxExchange(N_RANKS)

        def _factory(r):
            inner = mailbox.exchange_for(r)

            def _exchange(outgoing, incoming_nbytes):
                received = inner(outgoing, incoming_nbytes)
                if r == 0:
                    received = {
                        p: t[: t.numel() // 2] for p, t in received.items()
                    }
                return received

            return _exchange

        movers = _make_movers(
            stage_ids, pp_req, tp_req, live, exchange_factory=_factory
        )
        errors = _run_movers(movers, PP_TO_TP)
        self.assertIsInstance(errors[0], KvReshardError)
        self.assertIn("expected", str(errors[0]))


class TestLinearLayerMap(CustomTestCase):
    def setUp(self):
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)

    tearDown = setUp

    def test_stage_split_covers_exactly_once(self):
        stage_ids = derive_pp_linear_layer_map(LINEAR_IDS, N_HIDDEN, N_RANKS)
        self.assertEqual(
            sorted(g for s in stage_ids for g in s), LINEAR_IDS
        )

    def test_qwen36_recipe_split(self):
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "32,16,16"
        linear = [i for i in range(64) if i % 4 != 3]  # 48 linear layers
        stage_ids = derive_pp_linear_layer_map(linear, 64, 3)
        self.assertEqual([len(s) for s in stage_ids], [24, 12, 12])

    def test_can_fail_unsorted_refused(self):
        with self.assertRaisesRegex(KvReshardError, "ascending"):
            derive_pp_linear_layer_map([2, 1], N_HIDDEN, N_RANKS)


if __name__ == "__main__":
    unittest.main()


class TestRealConfigHeadGeometry(CustomTestCase):
    """3.4b item 3 + first-real-metal refusal (2026-08-08): with the
    Qwen3.6-27B GDN constants (conv 10240 = 2x2048 + 6144, 16 key heads,
    48 value heads x 128, INT8 -> gdn_tp_units passes through as 16) and
    the 30,17,17 vector, the channel-total split floors 48*30/64 = 22.5
    down to head shards (22, 12, 12) -- sum 46, loud refusal on metal.
    The model splits HEADS in whole gdn_tp_units: (24, 12, 12). The mover
    must replicate the model split when given the geometry."""

    K_HEADS, V_HEADS, HEAD_DIM, UNITS_R = 16, 48, 128, 16
    KEY_DIM = K_HEADS * HEAD_DIM  # 2048
    VALUE_DIM = V_HEADS * HEAD_DIM  # 6144
    CONV_R = 2 * KEY_DIM + VALUE_DIM  # 10240
    W, S, STATE = 4, 2, 128

    def setUp(self):
        set_tp_partition_ratios([30, 17, 17], families=None)
        set_cp_token_ratios([30, 17, 17])

    def tearDown(self):
        set_tp_partition_ratios(None, families=None)
        set_cp_token_ratios(None)

    def _pool_pair(self, rank, k_shard, v_shard):
        def _stub(conv, temporal):
            return SimpleNamespace(
                mamba_cache=SimpleNamespace(
                    conv=[conv], temporal=temporal, replayssm_d=None
                ),
                enable_linear_replayssm=False,
                get_conv_subblock_spec=lambda: (
                    [self.KEY_DIM, self.KEY_DIM, self.VALUE_DIM],
                    None,  # partition_units absent -- the real-metal case
                    self.CONV_R,
                ),
                size=self.S,
            )

        pp = _stub(
            torch.zeros(1, self.S, self.CONV_R, self.W, dtype=DTYPE),
            torch.zeros(
                1, self.S, self.V_HEADS, self.HEAD_DIM, self.STATE, dtype=DTYPE
            ),
        )
        conv_shard = 2 * k_shard * self.HEAD_DIM + v_shard * self.HEAD_DIM
        tp = _stub(
            torch.zeros(1, self.S, conv_shard, self.W, dtype=DTYPE),
            torch.zeros(
                1, self.S, v_shard, self.HEAD_DIM, self.STATE, dtype=DTYPE
            ),
        )
        return (
            _req_pool_stub(pp, {7: 0}),
            _req_pool_stub(tp, {7: 0}),
        )

    def test_model_identical_head_split(self):
        geometry = SimpleNamespace(
            linear_num_key_heads=self.K_HEADS,
            linear_num_value_heads=self.V_HEADS,
            gdn_tp_units=self.UNITS_R,
        )
        expected_v = (24, 12, 12)
        expected_k = (8, 4, 4)
        for rank in range(3):
            pp_req, tp_req = self._pool_pair(
                rank, expected_k[rank], expected_v[rank]
            )
            pools = gdn_flip_preconditions(
                pp_req, tp_req, 3, rank, gdn_geometry=geometry
            )
            self.assertEqual(pools.spec.head_shards, expected_v)
            self.assertEqual(sum(pools.spec.head_shards), self.V_HEADS)

    def test_channel_split_without_geometry_still_refuses(self):
        # Negative control: the legacy channel-total derivation with the
        # real constants and no partition_units yields (22, 12, 12) and
        # must refuse loudly -- the exact metal failure, kept reachable.
        pp_req, tp_req = self._pool_pair(0, 8, 24)
        with self.assertRaisesRegex(KvReshardError, "do not partition"):
            gdn_flip_preconditions(pp_req, tp_req, 3, 0)
