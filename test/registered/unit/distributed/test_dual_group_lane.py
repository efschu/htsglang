"""Multi-group runtime (#121/#274) slice B: lane shells, loader sighting.

Every test runs CPU-only (CUDA_VISIBLE_DEVICES=99 in the suite).  The claims:

* the shells compose a FULL-width forward out of N sharded parts exactly
  (column: pure data movement; row: the same one addition a 2-rank
  all-reduce performs),
* the v2 parameter loaders see the LANE geometry when -- and only when --
  the lane's own partition vector is scoped in (the tp_size=None trap from
  the slice-A findings, made visible),
* assembly + finalization + the shared-byte (data_ptr) gate work on a real
  ColumnParallelLinear/RowParallelLinear toy tree.
"""

import unittest

import torch
from torch import nn

from sglang.srt.distributed.dual_group import derive_nested_plan
from sglang.srt.distributed.utils import (
    partition_sizes,
    scoped_tp_partition_ratios,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.model_executor.dual_group_lane import (
    DualGroupLane,
    LaneColumnParallelShell,
    LaneFusedMoEShell,
    LaneRowParallelShell,
    assemble_lane_shells,
    verify_shared_bytes,
)

H = 32
OUT = 128
UNITS = 8  # 16-elem units of OUT
BIG = [2, 1, 1]
FAST = (2, 2)  # derive_nested_plan(BIG).fast_ratio


def _mk_column(
    tp_size, tp_rank, ratios, families=None, out=OUT, units=UNITS, load_full=None
):
    # Construction AND loading sit inside the scope: the loaders read the
    # installed vector at LOAD time (tp_loaded_shard_start), not only at
    # construction -- the exact sighting finding of this slice.
    with scoped_tp_partition_ratios(ratios, families):
        lin = ColumnParallelLinear(
            H,
            out,
            bias=False,
            gather_output=False,
            params_dtype=torch.float32,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_units=units,
        )
        if load_full is not None:
            lin.weight_loader(lin.weight, load_full)
    return lin


def _mk_row(tp_size, tp_rank, ratios, in_size=OUT, units=UNITS, load_full=None):
    with scoped_tp_partition_ratios(ratios):
        lin = RowParallelLinear(
            in_size,
            H,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            params_dtype=torch.float32,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_units=units,
        )
        if load_full is not None:
            lin.weight_loader(lin.weight, load_full)
    return lin


class TestLoaderSighting(unittest.TestCase):
    """The v2 loaders read the GLOBALLY installed vector.  In lane scope they
    must see the lane split; outside it they silently fall back to even."""

    def test_column_loader_sees_the_lane_split_in_scope(self):
        # Asymmetric on purpose: [4,1,1] over 6 units -> FAST (4,2) splits
        # [4,2], the even fallback would split [3,3].
        big, units, out = [4, 1, 1], 6, 96
        full = torch.randn(out, H)
        plan = derive_nested_plan(big)
        fast = list(plan.fast_ratio)
        sizes = partition_sizes(out, fast, units)
        self.assertEqual(sizes, [64, 32])  # [4,2] units of 16

        lin = _mk_column(2, 1, fast, out=out, units=units, load_full=full)
        self.assertEqual(lin.weight.shape[0], 32)
        # The complement rank's rows are the TAIL of the full weight -- the
        # prefix-sum offset (64), not the even-rank offset (48).
        torch.testing.assert_close(lin.weight.data, full[64:96])

    def test_without_the_scope_the_fallback_is_even_and_silent(self):
        big, units, out = [4, 1, 1], 6, 96
        full = torch.randn(out, H)
        # A 3-entry vector is installed (the serving group's), and a 2-rank
        # layer is built WITHOUT its own scope: tp_plan_active(2) is False,
        # so construction and loading fall back to the even split without
        # raising anything.  This test documents the trap the lane scope
        # exists to prevent.
        with scoped_tp_partition_ratios(big):
            lin = ColumnParallelLinear(
                H,
                out,
                bias=False,
                gather_output=False,
                params_dtype=torch.float32,
                tp_rank=1,
                tp_size=2,
                tp_units=units,
            )
            self.assertEqual(lin.weight.shape[0], 48)  # even, not [4,2]
            lin.weight_loader(lin.weight, full)
            torch.testing.assert_close(lin.weight.data, full[48:96])

    def test_row_loader_sees_the_lane_split_in_scope(self):
        big, units, in_size = [4, 1, 1], 6, 96
        full = torch.randn(H, in_size)
        plan = derive_nested_plan(big)
        lin = _mk_row(
            2, 1, list(plan.fast_ratio), in_size=in_size, units=units, load_full=full
        )
        self.assertEqual(lin.weight.shape[1], 32)
        torch.testing.assert_close(lin.weight.data, full[:, 64:96])


def _load_column_parts(full, specs):
    """specs: list of (tp_size, tp_rank, ratios). Returns loaded linears."""
    parts = []
    for tp_size, tp_rank, ratios in specs:
        parts.append(_mk_column(tp_size, tp_rank, ratios, load_full=full))
    return parts


class TestShells(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(5, H)

    def test_column_shell_over_lane_parts_matches_the_monolith(self):
        full = torch.randn(OUT, H)
        shared = _load_column_parts(full, [(3, 0, BIG)])[0]  # serving rank 0
        comp = _load_column_parts(full, [(2, 1, list(FAST))])[0]
        shell = LaneColumnParallelShell([shared, comp])
        out, bias = shell(self.x)
        self.assertIsNone(bias)
        ref = self.x @ full.t()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_column_shell_regroups_merged_sub_outputs(self):
        # Two sub-outputs (gate|up-like), 4 units each: rank slices must be
        # regrouped per SUB-output, not concatenated whole.
        sub = 64
        full_gate = torch.randn(sub, H)
        full_up = torch.randn(sub, H)
        parts = []
        for tp_size, tp_rank, ratios in [(3, 0, BIG), (2, 1, list(FAST))]:
            with scoped_tp_partition_ratios(ratios):
                lin = MergedColumnParallelLinear(
                    H,
                    [sub, sub],
                    bias=False,
                    params_dtype=torch.float32,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    tp_units=4,
                )
            g_sizes = partition_sizes(sub, ratios, 4)
            u_sizes = partition_sizes(sub, ratios, 4)
            g_off = sum(g_sizes[:tp_rank])
            u_off = sum(u_sizes[:tp_rank])
            lin.weight.data.copy_(
                torch.cat(
                    [
                        full_gate[g_off : g_off + g_sizes[tp_rank]],
                        full_up[u_off : u_off + u_sizes[tp_rank]],
                    ]
                )
            )
            parts.append(lin)
        shell = LaneColumnParallelShell(parts)
        out, _ = shell(self.x)
        ref = self.x @ torch.cat([full_gate, full_up]).t()
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_row_shell_is_the_two_rank_reduce(self):
        full = torch.randn(H, OUT)
        shared = _mk_row(3, 0, BIG, load_full=full)
        comp = _mk_row(2, 1, list(FAST), load_full=full)
        shell = LaneRowParallelShell([shared, comp])
        xin = torch.randn(5, OUT)
        out, bias = shell(xin)
        self.assertIsNone(bias)
        # Exactly the manual 2-part sum (the local all-reduce)...
        a = xin[:, :64] @ shared.weight.data.t()
        b = xin[:, 64:] @ comp.weight.data.t()
        torch.testing.assert_close(out, a + b)
        # ...and numerically-near the monolith (NOT bitwise by contract:
        # the accumulation order differs).
        ref = xin @ full.t()
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


class _ToyBlock(nn.Module):
    """One norm (replicated) + column + row, the shell-relevant skeleton."""

    def __init__(self, tp_size, tp_rank, ratios, families=None):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        with scoped_tp_partition_ratios(ratios, families):
            self.up = ColumnParallelLinear(
                H,
                OUT,
                bias=False,
                gather_output=False,
                params_dtype=torch.float32,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_units=UNITS,
            )
            self.down = RowParallelLinear(
                OUT,
                H,
                bias=False,
                input_is_parallel=True,
                reduce_results=False,
                params_dtype=torch.float32,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_units=UNITS,
            )

    def forward(self, x):
        x = self.norm(x)
        up, _ = self.up(x)
        out, _ = self.down(torch.nn.functional.silu(up))
        return out


class TestAssembly(unittest.TestCase):
    def test_assemble_finalize_and_shared_byte_gate(self):
        torch.manual_seed(11)
        full_up = torch.randn(OUT, H)
        full_down = torch.randn(H, OUT)

        shared = _ToyBlock(3, 0, BIG)  # the resident serving rank's tree
        comp = _ToyBlock(2, 1, list(FAST))
        for blk, ratios in ((shared, BIG), (comp, list(FAST))):
            with scoped_tp_partition_ratios(ratios):
                blk.up.weight_loader(blk.up.weight, full_up)
                blk.down.weight_loader(blk.down.weight, full_down)

        hull = _ToyBlock(1, 0, None)  # full-width view
        counts = assemble_lane_shells(hull, [shared, comp])
        self.assertEqual(counts["column"], 1)
        self.assertEqual(counts["row"], 1)

        # Finalize replicated params by aliasing (what _finalize_hull_params
        # does; done by hand here because the toy has no GDN vectors).
        hull.norm.weight.data = shared.norm.weight.data
        hull.norm.bias.data = shared.norm.bias.data

        checked = verify_shared_bytes(hull, shared, 0)
        self.assertGreaterEqual(checked, 4)  # up.w, down.w, norm.w, norm.b

        x = torch.randn(3, H)
        # Manual monolithic reference (a real RowParallelLinear.forward would
        # touch the -- uninitialized -- process group even at tp_size 1).
        h = torch.nn.functional.layer_norm(
            x, (H,), shared.norm.weight, shared.norm.bias
        )
        ref = torch.nn.functional.silu(h @ full_up.t()) @ full_down.t()
        torch.testing.assert_close(hull(x), ref, rtol=1e-4, atol=1e-4)

    def test_gate_fails_on_a_copied_shard(self):
        shared = _ToyBlock(3, 0, BIG)
        comp = _ToyBlock(2, 1, list(FAST))
        hull = _ToyBlock(1, 0, None)
        assemble_lane_shells(hull, [shared, comp])
        # Sever the sharing: replace the shell's shared part with a COPY.
        stolen = _ToyBlock(3, 0, BIG)
        stolen.load_state_dict(shared.state_dict())
        hull.up._lane_parts = (stolen.up, comp.up)
        with self.assertRaises(AssertionError) as ctx:
            verify_shared_bytes(hull, shared, 0)
        self.assertIn("data_ptr", str(ctx.exception).replace("storage", "data_ptr"))


class _MoEPart(nn.Module):
    """A FusedMoE stand-in: contributes a partial sum of the full output."""

    def __init__(self, scale, num_experts=8, top_k=2, hidden_size=H):
        super().__init__()
        self.scale = scale
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.seen = []

    def forward_local(self, hidden_states, topk_output):
        self.seen.append(topk_output)
        return hidden_states * self.scale


class TestMoEShell(unittest.TestCase):
    """#274 families slice C: the fifth shell class.

    The claim is narrow and checkable: a FusedMoE rank emits a PARTIAL SUM of
    the full-width output in both shard modes, so the lane's local stand-in
    for the group all-reduce is the same addition -- and the routing decision
    is made ONCE, over the global expert numbering, and handed to every part
    unchanged.
    """

    def test_shell_is_the_local_reduce_over_parts(self):
        parts = [_MoEPart(2.0), _MoEPart(3.0), _MoEPart(5.0)]
        shell = LaneFusedMoEShell(parts)
        x = torch.randn(4, H)
        torch.testing.assert_close(shell(x, "topk"), x * 10.0)

    def test_every_part_sees_the_same_global_routing(self):
        parts = [_MoEPart(1.0), _MoEPart(1.0)]
        shell = LaneFusedMoEShell(parts)
        shell(torch.randn(2, H), "the-global-topk")
        for p in parts:
            self.assertEqual(p.seen, ["the-global-topk"])

    def test_geometry_is_taken_from_the_parts(self):
        shell = LaneFusedMoEShell([_MoEPart(1.0, num_experts=128, top_k=4)])
        self.assertEqual(shell.num_experts, 128)
        self.assertEqual(shell.top_k, 4)

    def test_a_part_that_cannot_split_its_reduce_is_refused(self):
        class _Reduced(nn.Module):
            def forward(self, h, t):  # no forward_local
                return h

        with self.assertRaises(ValueError) as ctx:
            LaneFusedMoEShell([_MoEPart(1.0), _Reduced()])
        self.assertIn("forward_local", str(ctx.exception))

    def test_fused_moe_exposes_the_split_reduce_entry_point(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        self.assertTrue(hasattr(FusedMoE, "forward_local"))


class _FakeRunner:
    def __init__(self, pool):
        self.req_to_token_pool = pool


class _PoolNoMamba:
    pass


class _PoolWithMamba:
    class _Mamba:
        mamba_cache = object()

    mamba_pool = _Mamba()


class TestFamilyNoOp(unittest.TestCase):
    """#274 families slice A: the GDN branches must be NO-OPs, not crashes.

    ``_verify_state_buffers`` is the one place in the lane's speculative path
    that dereferences the recurrent pool. A dense (or MoE-without-GDN) family
    has no recurrent state at all, so there is nothing a rejected candidate
    could have advanced -- that must return quietly. A HYBRID pool that was
    simply built without the speculative axis is a different situation and
    still has to refuse loudly.
    """

    def _lane(self, pool):
        lane = DualGroupLane.__new__(DualGroupLane)
        lane.runner = _FakeRunner(pool)
        return lane

    def test_family_without_recurrence_needs_no_verify_state(self):
        self.assertIsNone(self._lane(_PoolNoMamba())._verify_state_buffers(4))

    def test_hybrid_pool_without_the_speculative_axis_still_refuses(self):
        with self.assertRaises(ValueError) as ctx:
            self._lane(_PoolWithMamba())._verify_state_buffers(4)
        self.assertIn("SpeculativeState", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
