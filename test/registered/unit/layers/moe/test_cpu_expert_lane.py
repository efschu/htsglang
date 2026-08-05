"""Slice-1 falsifiers for the CPU expert compute lane.

Hermetic and CPU-only: no CUDA device is touched, so this suite runs while the
rig is serving.

The suite is built so that each test CAN fail. The accuracy tests carry a
control that deliberately breaks the thing under test and asserts the tolerance
rejects it, because a tolerance nobody has ever seen reject anything is not a
gate.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.moe.cpu_expert_lane import (
    CpuExpertExecutor,
    CpuExpertLaneConfig,
    CpuExpertLaneError,
    MODE_W8A8,
    MODE_W8A32,
    PREFER_ACCURACY,
    PREFER_SPEED,
    CpuExpertPool,
    CpuLaneSlotFeed,
    ExpertJob,
    Int8ExpertShard,
    build_jobs,
)

HIDDEN = 256
INTER = 128


def _silu(x):
    return x * torch.sigmoid(x)


def _reference(gate_w, up_w, down_w, x):
    """Full-precision fp32 reference for one expert's SwiGLU FFN."""
    g = torch.nn.functional.linear(x, gate_w)
    u = torch.nn.functional.linear(x, up_w)
    return torch.nn.functional.linear(_silu(g) * u, down_w)


def _rand_expert(seed=0, hidden=HIDDEN, inter=INTER):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(inter, hidden, generator=g) * 0.02,
        torch.randn(inter, hidden, generator=g) * 0.02,
        torch.randn(hidden, inter, generator=g) * 0.02,
    )


def _rel_err(got, ref):
    return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


class TestInt8ShardAccuracy(unittest.TestCase):
    """The int8 shard must track an fp32 reference within quantisation error."""

    # Two modes, two bands, both MEASURED rather than assumed:
    #   W8A32 (weight-only, fp32 activations) ~1.1-1.6e-2 -- the same band as
    #     the already-accepted marlin offload.
    #   W8A8 (activations quantised too)      ~3.4-5.1e-2 -- roughly 3x worse,
    #     which is the price of the batched GEMM needed for MTP verify.
    TOL_W8A32 = 2.0e-2
    # W8A8 degrades with M (measured 3.4e-2 at M=1 rising to 6.7e-2 at M=33):
    # torch's dynamic path picks ONE activation scale for the whole batch, so a
    # wider batch means a wider range and a coarser step. Per-token activation
    # scales would flatten this and are a named Slice-2 item.
    TOL_W8A8 = 8.0e-2

    def test_w8a32_matches_fp32_reference(self):
        gate_w, up_w, down_w = _rand_expert(seed=1)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        for m in (1, 2, 4, 8, 33):
            x = torch.randn(m, HIDDEN)
            err = _rel_err(
                shard.forward(x, mode=MODE_W8A32), _reference(gate_w, up_w, down_w, x)
            )
            self.assertLess(err, self.TOL_W8A32, f"W8A32 M={m} drifted {err:.4f}")

    def test_w8a8_matches_fp32_reference_in_its_wider_band(self):
        gate_w, up_w, down_w = _rand_expert(seed=1)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        for m in (1, 4, 8, 33):
            x = torch.randn(m, HIDDEN)
            err = _rel_err(
                shard.forward(x, mode=MODE_W8A8), _reference(gate_w, up_w, down_w, x)
            )
            self.assertLess(err, self.TOL_W8A8, f"W8A8 M={m} drifted {err:.4f}")

    def test_w8a8_error_grows_with_batch_width(self):
        """Pins the per-tensor activation-scale weakness rather than hiding it.

        One activation scale is chosen for the whole batch, so a wider batch is
        quantised more coarsely. Slice 2 should replace this with per-token
        scales; until then the mode-selection rule must not assume W8A8 accuracy
        is batch-independent.
        """
        gate_w, up_w, down_w = _rand_expert(seed=11)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        torch.manual_seed(0)
        narrow = torch.randn(4, HIDDEN)
        wide = torch.randn(64, HIDDEN)
        e_narrow = _rel_err(
            shard.forward(narrow, mode=MODE_W8A8), _reference(gate_w, up_w, down_w, narrow)
        )
        e_wide = _rel_err(
            shard.forward(wide, mode=MODE_W8A8), _reference(gate_w, up_w, down_w, wide)
        )
        self.assertGreater(e_wide, e_narrow, f"expected growth, got {e_narrow} -> {e_wide}")
        # W8A32 has no activation quantisation, so it must NOT show the effect.
        w32_narrow = _rel_err(
            shard.forward(narrow, mode=MODE_W8A32), _reference(gate_w, up_w, down_w, narrow)
        )
        w32_wide = _rel_err(
            shard.forward(wide, mode=MODE_W8A32), _reference(gate_w, up_w, down_w, wide)
        )
        self.assertLess(abs(w32_wide - w32_narrow), 1e-2, "W8A32 should be batch-flat")

    def test_w8a32_is_strictly_more_accurate_than_w8a8(self):
        """The accuracy ordering the mode-selection rule depends on."""
        gate_w, up_w, down_w = _rand_expert(seed=5)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        x = torch.randn(8, HIDDEN)
        ref = _reference(gate_w, up_w, down_w, x)
        e32 = _rel_err(shard.forward(x, mode=MODE_W8A32), ref)
        e8 = _rel_err(shard.forward(x, mode=MODE_W8A8), ref)
        self.assertLess(e32, e8, f"W8A32 {e32:.4f} should beat W8A8 {e8:.4f}")

    def test_speed_preference_always_picks_the_fast_gemm(self):
        """W8A8 measured faster at every row count, M=1 included."""
        gate_w, up_w, down_w = _rand_expert(seed=6)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        for m in (1, 2, 4, 64):
            self.assertEqual(shard.select_mode(m, prefer=PREFER_SPEED), MODE_W8A8)

    def test_accuracy_preference_hands_back_above_the_row_limit(self):
        """The accurate kernel is used only while it still beats the fetch."""
        gate_w, up_w, down_w = _rand_expert(seed=6)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        self.assertEqual(shard.select_mode(1, prefer=PREFER_ACCURACY), MODE_W8A32)
        self.assertEqual(shard.select_mode(2, prefer=PREFER_ACCURACY), MODE_W8A32)
        self.assertEqual(shard.select_mode(4, prefer=PREFER_ACCURACY), MODE_W8A8)
        self.assertEqual(shard.select_mode(64, prefer=PREFER_ACCURACY), MODE_W8A8)

    def test_single_mode_shard_falls_back(self):
        gate_w, up_w, down_w = _rand_expert(seed=7)
        only32 = Int8ExpertShard(gate_w, up_w, down_w, modes=(MODE_W8A32,))
        self.assertEqual(only32.select_mode(64, prefer=PREFER_SPEED), MODE_W8A32)
        only8 = Int8ExpertShard(gate_w, up_w, down_w, modes=(MODE_W8A8,))
        self.assertEqual(only8.select_mode(1, prefer=PREFER_ACCURACY), MODE_W8A8)

    def test_tolerance_actually_rejects_a_broken_shard(self):
        """Control: the gate must reject a shard whose weights are wrong.

        Without this, TOL could be loose enough to pass anything.
        """
        gate_w, up_w, down_w = _rand_expert(seed=2)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        x = torch.randn(8, HIDDEN)
        # Reference built from DIFFERENT down weights: a real defect.
        wrong_down = down_w + torch.randn_like(down_w) * 0.02
        err = _rel_err(shard.forward(x), _reference(gate_w, up_w, wrong_down, x))
        self.assertGreater(
            err, self.TOL_W8A8, "tolerance is too loose -- it accepted a wrong expert"
        )

    def test_no_weight_is_widened_to_float(self):
        """The prepacked shard must hold int8, not a float copy.

        Per-event widening is the defect that sank the fp32 variant of this lane
        (2.831 ms per expert). This pins that the packed module stores qint8.
        """
        gate_w, up_w, down_w = _rand_expert(seed=3)
        shard = Int8ExpertShard(gate_w, up_w, down_w)
        for name in ("gate_q", "up_q", "down_q"):
            self.assertEqual(
                getattr(shard, name).dtype, torch.int8,
                f"{name} is not int8 -- the lane must never hold widened weights",
            )
        for name in ("gate_d", "up_d", "down_d"):
            w = getattr(shard, name)[0].weight()
            self.assertEqual(
                w.dtype, torch.qint8,
                f"{name} is {w.dtype}, not qint8 -- the lane must never hold or "
                "produce widened weights",
            )

    def test_shape_mismatch_is_refused(self):
        gate_w, up_w, down_w = _rand_expert(seed=4)
        with self.assertRaises(CpuExpertLaneError):
            Int8ExpertShard(gate_w, up_w[:, :-1], down_w)
        with self.assertRaises(CpuExpertLaneError):
            Int8ExpertShard(gate_w, up_w, down_w.t().contiguous())


class TestBuildJobs(unittest.TestCase):
    def test_partitions_rows_and_ignores_gpu_experts(self):
        topk = torch.tensor([[0, 1], [2, 0], [1, 3], [0, 2]])
        jobs = build_jobs(topk, cpu_expert_ids=[0, 2])
        got = {j.expert_id: j.rows.tolist() for j in jobs}
        self.assertEqual(got, {0: [0, 1, 3], 2: [1, 3]})

    def test_empty_when_no_cpu_experts(self):
        topk = torch.tensor([[0, 1], [2, 3]])
        self.assertEqual(build_jobs(topk, cpu_expert_ids=[]), [])

    def test_handles_topk_1(self):
        jobs = build_jobs(torch.tensor([0, 1, 0]), cpu_expert_ids=[0])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].rows.tolist(), [0, 2])


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.cfg = CpuExpertLaneConfig(enabled=True, max_workers=3, intra_op_threads=2)
        self.ex = CpuExpertExecutor(self.cfg)
        self.pool = CpuExpertPool(layer_id=0, hidden=HIDDEN, inter=INTER)
        self.weights = {}
        for e in range(4):
            w = _rand_expert(seed=100 + e)
            self.weights[e] = w
            self.pool.add_expert(e, *w)

    def tearDown(self):
        self.ex.shutdown()

    def test_multi_expert_scatter_matches_reference(self):
        T = 12
        acts = torch.randn(T, HIDDEN)
        topk = torch.randint(0, 4, (T, 2), generator=torch.Generator().manual_seed(7))
        jobs = build_jobs(topk, cpu_expert_ids=[0, 1, 2, 3])
        # Each row may appear under several experts; restrict to a partition so
        # index_copy_ semantics are well defined for this test.
        seen, part = set(), []
        for j in jobs:
            keep = [r for r in j.rows.tolist() if r not in seen]
            seen.update(keep)
            if keep:
                part.append(ExpertJob(j.expert_id, torch.tensor(keep, dtype=torch.int64)))

        out = torch.zeros(T, HIDDEN)
        n = self.ex.run(self.pool, acts, part, out)
        self.assertEqual(n, len(part))

        for job in part:
            gate_w, up_w, down_w = self.weights[job.expert_id]
            ref = _reference(gate_w, up_w, down_w, acts.index_select(0, job.rows))
            err = _rel_err(out.index_select(0, job.rows), ref)
            self.assertLess(err, 6e-2, f"expert {job.expert_id} drifted {err:.4f}")

    def test_router_weights_are_applied(self):
        T = 6
        acts = torch.randn(T, HIDDEN)
        rows = torch.arange(T, dtype=torch.int64)
        job = [ExpertJob(0, rows)]
        w = torch.rand(T) + 0.5

        plain = torch.zeros(T, HIDDEN)
        self.ex.run(self.pool, acts, job, plain)
        scaled = torch.zeros(T, HIDDEN)
        self.ex.run(self.pool, acts, job, scaled, weights=w)

        torch.testing.assert_close(scaled, plain * w.unsqueeze(1), rtol=1e-4, atol=1e-5)

    def test_untouched_rows_are_preserved(self):
        T = 8
        acts = torch.randn(T, HIDDEN)
        out = torch.full((T, HIDDEN), 7.0)
        self.ex.run(self.pool, acts, [ExpertJob(0, torch.tensor([1, 3]))], out)
        for r in (0, 2, 4, 5, 6, 7):
            self.assertTrue(torch.all(out[r] == 7.0), f"row {r} was clobbered")

    def test_unknown_expert_is_refused_with_a_useful_message(self):
        with self.assertRaises(CpuExpertLaneError) as cm:
            self.ex.run(
                self.pool, torch.randn(2, HIDDEN),
                [ExpertJob(99, torch.tensor([0]))], torch.zeros(2, HIDDEN),
            )
        self.assertIn("99", str(cm.exception))

    def test_hidden_mismatch_is_refused(self):
        with self.assertRaises(CpuExpertLaneError):
            self.ex.run(
                self.pool, torch.randn(2, HIDDEN + 1),
                [ExpertJob(0, torch.tensor([0]))], torch.zeros(2, HIDDEN + 1),
            )

    def test_mtp_verify_batch_shape(self):
        """A verify batch is M = num_draft_tokens + 1 rows for one expert."""
        for num_draft in (1, 3, 7):
            m = num_draft + 1
            acts = torch.randn(m, HIDDEN)
            out = torch.zeros(m, HIDDEN)
            self.ex.run(
                self.pool, acts,
                [ExpertJob(1, torch.arange(m, dtype=torch.int64))], out,
            )
            gate_w, up_w, down_w = self.weights[1]
            err = _rel_err(out, _reference(gate_w, up_w, down_w, acts))
            self.assertLess(err, 6e-2, f"verify batch M={m} drifted {err:.4f}")


class TestSlotFeedContract(unittest.TestCase):
    """The #462 seam contract, pinned without a GPU."""

    def setUp(self):
        self.stage = torch.zeros(4, HIDDEN)
        self.buf = torch.zeros(4, HIDDEN)
        self.feed = CpuLaneSlotFeed(layer_id=3, stage=self.stage, buf=self.buf)

    def test_publish_before_compute_done_is_refused(self):
        self.feed.begin_step()
        with self.assertRaises(CpuExpertLaneError) as cm:
            self.feed.publish()
        self.assertIn("before mark_compute_done", str(cm.exception))

    def test_double_publish_is_refused(self):
        self.feed.begin_step()
        self.feed.mark_compute_done()
        self.feed.publish()
        with self.assertRaises(CpuExpertLaneError):
            self.feed.publish()

    def test_publish_transfers_stage_to_buf(self):
        s = self.feed.begin_step()
        s.copy_(torch.arange(4 * HIDDEN, dtype=torch.float32).reshape(4, HIDDEN))
        self.feed.mark_compute_done()
        self.feed.publish()
        torch.testing.assert_close(self.buf, self.stage)
        self.assertTrue(self.feed.published)

    def test_begin_step_resets_and_zeroes(self):
        self.feed.begin_step()
        self.feed.mark_compute_done()
        self.feed.publish()
        self.stage.fill_(5.0)
        s = self.feed.begin_step()
        self.assertFalse(self.feed.published)
        self.assertTrue(torch.all(s == 0.0))
        self.feed.mark_compute_done()
        self.feed.publish()  # must not raise

    def test_non_cpu_stage_is_refused(self):
        with self.assertRaises(CpuExpertLaneError):
            CpuLaneSlotFeed(3, stage=torch.zeros(2, 2, device="meta"), buf=self.buf)


class TestConfigValidation(unittest.TestCase):
    def test_bad_worker_count(self):
        with self.assertRaises(CpuExpertLaneError):
            CpuExpertLaneConfig(max_workers=0).validate()

    def test_bad_thread_count(self):
        with self.assertRaises(CpuExpertLaneError):
            CpuExpertLaneConfig(intra_op_threads=0).validate()

    def test_unknown_engine_names_the_supported_set(self):
        with self.assertRaises(CpuExpertLaneError) as cm:
            CpuExpertLaneConfig(quant_engine="nonexistent").validate()
        self.assertIn("supported", str(cm.exception))

    def test_lane_is_off_by_default(self):
        """Numerics-changing features never default on."""
        self.assertFalse(CpuExpertLaneConfig().enabled)

    def test_bad_preference_is_refused(self):
        with self.assertRaises(CpuExpertLaneError):
            CpuExpertLaneConfig(prefer="whatever").validate()

    def test_executor_honours_the_accuracy_preference(self):
        """The config knob must reach the kernel, not just validate."""
        pool = CpuExpertPool(0, hidden=HIDDEN, inter=INTER)
        gate_w, up_w, down_w = _rand_expert(seed=21)
        pool.add_expert(0, gate_w, up_w, down_w)
        acts = torch.randn(1, HIDDEN)
        ref = _reference(gate_w, up_w, down_w, acts)
        errs = {}
        for pref in (PREFER_SPEED, PREFER_ACCURACY):
            ex = CpuExpertExecutor(CpuExpertLaneConfig(enabled=True, prefer=pref))
            out = torch.zeros(1, HIDDEN)
            ex.run(pool, acts, [ExpertJob(0, torch.tensor([0]))], out)
            ex.shutdown()
            errs[pref] = _rel_err(out, ref)
        self.assertLess(
            errs[PREFER_ACCURACY], errs[PREFER_SPEED],
            f"accuracy preference did not reach the kernel: {errs}",
        )


class TestPoolAccounting(unittest.TestCase):
    def test_single_mode_int8_tier_is_half_of_bf16(self):
        """The RAM-wall argument the design rests on."""
        pool = CpuExpertPool(0, hidden=2048, inter=512, modes=(MODE_W8A32,))
        for e in range(2):
            pool.add_expert(e, *_rand_expert(seed=e, hidden=2048, inter=512))
        bf16_bytes = 2 * (2 * 3 * 2048 * 512)
        self.assertEqual(pool.bytes_resident(), bf16_bytes // 2)

    def test_dual_mode_tier_costs_the_same_as_bf16(self):
        """Carrying both kernels costs exactly what the bf16 pool costs.

        Two int8 copies == one bf16 copy. So the dual-mode tier is RAM-neutral
        against the pool it displaces, and the single-mode tier halves it.
        """
        pool = CpuExpertPool(0, hidden=2048, inter=512, modes=(MODE_W8A32, MODE_W8A8))
        for e in range(2):
            pool.add_expert(e, *_rand_expert(seed=e, hidden=2048, inter=512))
        bf16_bytes = 2 * (2 * 3 * 2048 * 512)
        self.assertEqual(pool.bytes_resident(), bf16_bytes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
