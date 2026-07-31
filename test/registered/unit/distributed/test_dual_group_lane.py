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

import contextlib
import os
import types
import unittest
import unittest.mock

import torch
from torch import nn

import sglang.srt.model_executor.dual_group_lane as dgl
from sglang.srt.distributed.dual_group import (
    derive_nested_plan,
    lane_part_device_indices,
    lane_visible_physical_gpus,
)
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
    _assert_lane_moe_is_pure_tp,
    _finalize_hull_params,
    assemble_lane_shells,
    resolve_lane_part_gpu_ids,
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


# ---------------------------------------------------------------------------
# #274 families slice 2: lane parts that are NOT the host's shard, and lane
# parts that are not on the host's card.
# ---------------------------------------------------------------------------


class _FakeQuant:
    """Records which device each part was asked to compute on."""

    def __init__(self, log):
        self.log = log

    def apply(self, layer, x, bias=None):
        self.log.append(("apply", layer.tag, x.device.type))
        return torch.zeros(x.shape[0], layer.out_width, dtype=x.dtype)


class _FakePart(nn.Module):
    def __init__(self, tag, out_width, log, device):
        super().__init__()
        self.tag = tag
        self.out_width = out_width
        self.output_partition_sizes = [out_width]
        self.input_size_per_partition = H // 2
        self.bias = None
        self.skip_bias_add = False
        self.gather_output = False
        self.quant_method = _FakeQuant(log)
        self.register_parameter(
            "weight", nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
        )


class TestLanePartsAcrossCards(unittest.TestCase):
    """The shells move the ACTIVATION to the part and the result back.

    No CUDA here: the plumbing is observed by replacing the module-level
    ``_on`` helper with a recorder, which is exactly the function the shells
    route every cross-card hop through. What is asserted is the CALL PATTERN
    -- every part sees a tensor on its own device, every result comes home to
    the input's device -- not a copy engine.
    """

    def _record(self):
        hops = []
        real_on = dgl._on

        def spy(x, device):
            if device is not None and x.device != device:
                hops.append((x.device.type, device.type))
                return x  # cannot really move to a fake device on CPU
            return real_on(x, device)

        return hops, spy

    def test_one_card_lane_makes_no_hops_at_all(self):
        log = []
        parts = [_FakePart("a", 8, log, "cpu"), _FakePart("b", 8, log, "cpu")]
        shell = LaneColumnParallelShell(parts)
        hops, spy = self._record()
        with unittest.mock.patch.object(dgl, "_on", spy):
            out, _ = shell(torch.zeros(2, H))
        self.assertEqual(hops, [])
        self.assertEqual(out.shape, (2, 16))

    def test_column_shell_sends_the_activation_and_brings_the_result_home(self):
        log = []
        parts = [_FakePart("a", 8, log, "cpu"), _FakePart("b", 8, log, "meta")]
        shell = LaneColumnParallelShell(parts)
        self.assertEqual([d.type for d in shell._lane_part_devices], ["cpu", "meta"])
        hops, spy = self._record()
        with unittest.mock.patch.object(dgl, "_on", spy):
            shell(torch.zeros(2, H))
        # one hop out to the foreign part, one hop back with its result;
        # the resident part is untouched.
        self.assertEqual(hops, [("cpu", "meta")])

    def test_row_shell_splits_first_then_travels(self):
        log = []
        parts = [_FakePart("a", 8, log, "cpu"), _FakePart("b", 8, log, "meta")]
        shell = LaneRowParallelShell(parts)
        hops, spy = self._record()
        with unittest.mock.patch.object(dgl, "_on", spy):
            shell(torch.zeros(2, H))
        # the SLICE travels, not the whole activation -- the split happens
        # before the hop.
        self.assertEqual(hops, [("cpu", "meta")])
        self.assertEqual([e[2] for e in log], ["cpu", "cpu"])

    def test_expert_shell_refuses_to_span_cards(self):
        class _MoEPartOnDevice(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.register_parameter(
                    "w13_weight",
                    nn.Parameter(torch.zeros(1, device=device), requires_grad=False),
                )

            def forward_local(self, hidden_states, topk_output):
                return hidden_states

        with self.assertRaises(ValueError) as ctx:
            LaneFusedMoEShell([_MoEPartOnDevice("cpu"), _MoEPartOnDevice("meta")])
        self.assertIn("different cards", str(ctx.exception))


class _ContextRecorder:
    """Stands in for the CUDA context switch on a CPU-only host.

    Replaces ``dgl._active_device``. ``active`` is what
    ``torch.cuda.current_device()`` would report at any moment, so a part's
    quant method can assert it is being launched into its OWN context rather
    than the lane host's.
    """

    def __init__(self, ambient=None):
        # The AMBIENT card: what the lane worker thread pinned with
        # torch.cuda.set_device before any shell ran.
        self.active = ambient
        self.ambient = ambient
        self.entered = []
        self._stack = []

    def __call__(self, device, home=None):
        if device is None or device == home:
            return contextlib.nullcontext()
        return self._Scope(self, device)

    class _Scope:
        def __init__(self, rec, device):
            self.rec = rec
            self.device = device

        def __enter__(self):
            self.rec._stack.append(self.rec.active)
            self.rec.active = self.device
            self.rec.entered.append(None if self.device is None else self.device.type)
            return self

        def __exit__(self, *exc):
            self.rec.active = self.rec._stack.pop()
            return False


class _ContextSensitiveQuant:
    """A Triton-like quant method: refuses a pointer outside the active context.

    This is the whole failure mode of the un-guarded shells reproduced without
    a second card -- Triton raises ``ValueError: Pointer argument (at 0) cannot
    be accessed`` when the tensor's card is not the current one.
    """

    def __init__(self, rec, log):
        self.rec = rec
        self.log = log

    def _check(self, x, tag):
        active = self.rec.active
        if active is None or x.device != active:
            raise ValueError(
                f"Pointer argument (at 0) cannot be accessed from Triton "
                f"(part {tag} on {x.device}, active context {active})"
            )
        self.log.append((tag, x.device.type))

    def apply(self, layer, x, bias=None):
        self._check(x, layer.tag)
        return torch.zeros(x.shape[0], layer.out_width, dtype=x.dtype, device=x.device)

    def embedding(self, layer, x):
        self._check(x, layer.tag)
        return torch.zeros(
            x.shape[0], layer.out_width, dtype=torch.float32, device=x.device
        )


class _StrictPart(_FakePart):
    def __init__(self, tag, out_width, rec, log, device):
        super().__init__(tag, out_width, log, device)
        self.quant_method = _ContextSensitiveQuant(rec, log)


class TestLanePartsSwitchTheCudaContext(unittest.TestCase):
    """A part's compute runs with the PART's card as the current device.

    ``_on`` moves the tensor; it does not move the context. cuBLAS carries a
    device guard out of its operands, Triton does not -- so the un-guarded
    shells worked for dense bf16 and died on every FP8/INT4 lane that spanned
    two cards. Observed here without CUDA: ``_active_device`` is replaced by a
    recorder, and the parts' quant methods reject any activation whose device
    is not the recorded active one, the same way Triton does.
    """

    def _parts(self, rec, log, cls=None):
        cls = cls or _StrictPart
        # "meta" plays the foreign card: a device that is never the ambient
        # one on a cpu test host.
        return [
            cls("home", 8, rec, log, "cpu"),
            cls("foreign", 8, rec, log, "meta"),
        ]

    def _run(self, rec, fn, guarded=True):
        # ``_on`` cannot really move a tensor to the fake foreign device with
        # a copy, so it is spied the same way the sibling suite does -- but
        # unlike there it returns a tensor that ACTUALLY reports the
        # destination device, which is what the context check compares
        # against.
        def spy(x, device):
            if device is not None and x.device != device:
                return torch.zeros(x.shape, dtype=x.dtype, device=device)
            return x

        guard = rec if guarded else (lambda dev, home=None: contextlib.nullcontext())
        with unittest.mock.patch.object(dgl, "_active_device", guard):
            with unittest.mock.patch.object(dgl, "_on", spy):
                return fn()

    def test_column_shell_guards_every_part(self):
        rec, log = _ContextRecorder(torch.device("cpu")), []
        shell = LaneColumnParallelShell(self._parts(rec, log))
        self._run(rec, lambda: shell(torch.zeros(2, H)))
        self.assertEqual(rec.entered, ["meta"])
        self.assertEqual(log, [("home", "cpu"), ("foreign", "meta")])
        self.assertEqual(rec.active, rec.ambient)  # restored after the shell

    def test_row_shell_guards_every_part(self):
        rec, log = _ContextRecorder(torch.device("cpu")), []
        shell = LaneRowParallelShell(self._parts(rec, log))
        self._run(rec, lambda: shell(torch.zeros(2, H)))
        self.assertEqual(rec.entered, ["meta"])
        self.assertEqual(log, [("home", "cpu"), ("foreign", "meta")])
        self.assertEqual(rec.active, rec.ambient)

    def test_lm_head_shell_guards_every_part(self):
        rec, log = _ContextRecorder(torch.device("cpu")), []
        shell = dgl.LaneLmHeadShell(self._parts(rec, log))
        self._run(rec, lambda: shell.quant_method.apply(shell, torch.zeros(2, H)))
        self.assertEqual(rec.entered, ["meta"])
        self.assertEqual(log, [("home", "cpu"), ("foreign", "meta")])
        self.assertEqual(rec.active, rec.ambient)

    def test_vocab_embedding_shell_guards_every_part(self):
        rec, log = _ContextRecorder(torch.device("cpu")), []
        parts = self._parts(rec, log)
        for p in parts:
            p.shard_indices = types.SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=8,
                num_org_vocab_padding=0,
                added_vocab_start_index=8,
                added_vocab_end_index=8,
            )
        shell = dgl.LaneVocabEmbeddingShell(parts)
        self._run(rec, lambda: shell(torch.zeros(2, dtype=torch.long)))
        self.assertEqual(rec.entered, ["meta"])
        self.assertEqual([e[0] for e in log], ["home", "foreign"])
        self.assertEqual(rec.active, rec.ambient)

    def test_unguarded_shell_would_fail_the_way_triton_does(self):
        """The falsifier: without the guard the foreign part raises.

        Without this the four guards could be deleted and the four tests above
        would still pass on their ``_on`` spy alone.
        """
        rec, log = _ContextRecorder(torch.device("cpu")), []
        shell = LaneColumnParallelShell(self._parts(rec, log))
        with self.assertRaises(ValueError) as ctx:
            self._run(rec, lambda: shell(torch.zeros(2, H)), guarded=False)
        self.assertIn("cannot be accessed", str(ctx.exception))

    def test_active_device_switches_only_when_the_activation_travels(self):
        cuda1 = torch.device("cuda:1")
        # no part device, non-cuda parts, and the one-card lane: no switch,
        # and no allocation either -- the shared no-op instance comes back.
        for dev, home in [
            (None, None),
            (torch.device("cpu"), None),
            (torch.device("meta"), None),
            (cuda1, cuda1),
        ]:
            self.assertIs(dgl._active_device(dev, home), dgl._NO_DEVICE_SWITCH)
        guard = dgl._active_device(cuda1, torch.device("cuda:0"))
        self.assertIsInstance(guard, torch.cuda.device)
        self.assertEqual(guard.idx, 1)


class TestHostAndMaterializedRanks(unittest.TestCase):
    """Which lane rank is aliased, and which one the lane has to load itself.

    At BIG tp_size 2 BOTH segments are singletons, so the old
    shared/complement split called them both shared -- and the build refused,
    because the second one's bytes live in another process. The host view
    names the one rank this process can actually alias.
    """

    def test_three_rank_group_reproduces_the_slice_b_split(self):
        plan = derive_nested_plan([2, 1, 1])
        self.assertEqual(plan.host_fast_rank(0), 0)
        self.assertEqual(plan.materialized_fast_ranks(0), (1,))

    def test_two_rank_group_has_a_materialized_singleton(self):
        plan = derive_nested_plan([3, 1])
        self.assertEqual(plan.segments, ((0,), (1,)))
        # Both segments are byte-shareable in principle ...
        self.assertEqual(plan.shared_fast_ranks, (0, 1))
        # ... but only rank 0's bytes are in THIS process.
        self.assertEqual(plan.host_fast_rank(0), 0)
        self.assertEqual(plan.materialized_fast_ranks(0), (1,))

    def test_a_host_rank_with_no_singleton_segment_is_named(self):
        plan = derive_nested_plan([2, 1, 1])
        with self.assertRaises(ValueError) as ctx:
            plan.host_fast_rank(1)
        self.assertIn("no singleton segment for serving rank 1", str(ctx.exception))


class TestLanePartPlacement(unittest.TestCase):
    """Physical GPU ids -> in-process cuda indices, one rule for both sides."""

    def test_host_card_is_always_first(self):
        self.assertEqual(lane_visible_physical_gpus(1, [1, 0]), [1, 0])
        self.assertEqual(lane_visible_physical_gpus(1, [1, 1]), [1])

    def test_indices_follow_the_visible_list(self):
        self.assertEqual(lane_part_device_indices(1, [1, 0], [1, 0]), [0, 1])
        self.assertEqual(lane_part_device_indices(1, [1, 1], [1]), [0, 0])

    def test_all_cards_visible_means_physical_is_the_index(self):
        self.assertEqual(lane_part_device_indices(1, [1, 2], None), [1, 2])

    def test_a_card_the_process_cannot_see_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            lane_part_device_indices(1, [1, 2], [1, 0])
        self.assertIn("only sees [1, 0]", str(ctx.exception))

    def test_default_keeps_every_part_on_the_host_card(self):
        plan = derive_nested_plan([2, 1, 1])
        args = types.SimpleNamespace(dual_group_lane_part_gpu_id=None)
        self.assertEqual(resolve_lane_part_gpu_ids(args, plan, 0), (0, 0))

    def test_flag_resolves_against_cuda_visible_devices(self):
        plan = derive_nested_plan([2, 1, 1])
        args = types.SimpleNamespace(dual_group_lane_part_gpu_id=[1, 0])
        with unittest.mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"}, clear=False
        ):
            self.assertEqual(resolve_lane_part_gpu_ids(args, plan, 0), (0, 1))

    def test_the_host_rank_cannot_be_moved_off_its_card(self):
        plan = derive_nested_plan([2, 1, 1])
        args = types.SimpleNamespace(dual_group_lane_part_gpu_id=[0, 1])
        with unittest.mock.patch.dict(
            os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"}, clear=False
        ):
            with self.assertRaises(ValueError) as ctx:
                resolve_lane_part_gpu_ids(args, plan, 0)
        self.assertIn("cannot move off its card", str(ctx.exception))

    def test_length_is_checked_against_the_lane_group(self):
        plan = derive_nested_plan([2, 1, 1])
        args = types.SimpleNamespace(dual_group_lane_part_gpu_id=[1, 0, 2])
        with self.assertRaises(ValueError) as ctx:
            resolve_lane_part_gpu_ids(args, plan, 0)
        self.assertIn("per LANE rank", str(ctx.exception))


class TestExpertShellEndToEnd(unittest.TestCase):
    """The MoE shell through assembly + finalize + the data_ptr gate.

    Slice-C pinned the shell's reduce algebra with a stand-in that had no
    parameters at all, so the expert TENSORS never met the byte gate. This
    walks a hull whose expert module is a real ``FusedMoE`` instance carrying
    real ``w13_weight``/``w2_weight`` parameters, which is the path a MoE lane
    boot takes.
    """

    @staticmethod
    def _expert_module(seed):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mod = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(mod)
        g = torch.Generator().manual_seed(seed)
        mod.register_parameter(
            "w13_weight",
            nn.Parameter(torch.rand(2, 4, generator=g), requires_grad=False),
        )
        mod.register_parameter(
            "w2_weight",
            nn.Parameter(torch.rand(4, 2, generator=g), requires_grad=False),
        )
        mod.num_experts, mod.top_k, mod.hidden_size, mod.layer_id = 8, 2, H, 0
        mod.moe_ep_size = 1
        mod.forward_local = lambda hidden_states, topk_output: hidden_states
        return mod

    @staticmethod
    def _tree(expert):
        tree = nn.Module()
        tree.experts = expert
        return tree

    def test_expert_tensors_pass_the_shared_byte_gate(self):
        shared = self._tree(self._expert_module(1))
        other = self._tree(self._expert_module(2))
        hull = self._tree(self._expert_module(3))

        counts = assemble_lane_shells(hull, [shared, other])
        self.assertEqual(counts["moe"], 1)
        self.assertIsInstance(hull.experts, LaneFusedMoEShell)

        fill = _finalize_hull_params(hull, shared, [shared, other])
        # The shell holds its parts in a plain tuple, so the hull tree has no
        # expert parameters left to fill -- and none to leave unfilled.
        self.assertEqual(fill["aliased"], 0)

        checked = verify_shared_bytes(hull, shared, 0)
        # w13_weight and w2_weight of the shared part, by data_ptr identity.
        self.assertEqual(checked, 2)

    def test_a_copied_expert_tensor_fails_the_gate(self):
        shared = self._tree(self._expert_module(1))
        other = self._tree(self._expert_module(2))
        hull = self._tree(self._expert_module(3))
        # Assemble against a COPY of the resident expert module -- same
        # names, same shapes, different storage. That is exactly the failure
        # the gate exists for: a lane that computes on a copy of the serving
        # rank's experts instead of on its bytes.
        copy_of_shared = self._tree(self._expert_module(1))
        assemble_lane_shells(hull, [copy_of_shared, other])
        with self.assertRaises(AssertionError) as ctx:
            verify_shared_bytes(hull, shared, 0)
        self.assertIn("shared-byte gate FAILED", str(ctx.exception))


class TestPartGpuFlagValidation(unittest.TestCase):
    """The flag is checked BEFORE any card is touched, because the foreign
    card has to be in CUDA_VISIBLE_DEVICES at spawn time."""

    @staticmethod
    def _args(part_gpus, rank_gpu_id=(1, 0)):
        return types.SimpleNamespace(
            dual_group_lane_part_gpu_id=list(part_gpus),
            rank_gpu_id=list(rank_gpu_id) if rank_gpu_id is not None else None,
        )

    def _check(self, args):
        from sglang.srt.server_args import ServerArgs

        ServerArgs._validate_dual_group_lane_part_gpu_id(args)

    def test_two_entries_on_known_cards_are_accepted(self):
        self._check(self._args([1, 0]))

    def test_one_entry_per_lane_rank_not_per_serving_rank(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(self._args([1, 1, 0]))
        self.assertIn("per LANE rank", str(ctx.exception))

    def test_the_host_lane_rank_must_name_the_host_card(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(self._args([0, 1]))
        self.assertIn("same bytes, not a copy", str(ctx.exception))

    def test_a_card_with_no_serving_rank_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(self._args([1, 5]))
        self.assertIn("carries no serving rank", str(ctx.exception))

    def test_physical_ids_need_rank_gpu_id(self):
        with self.assertRaises(ValueError) as ctx:
            self._check(self._args([1, 0], rank_gpu_id=None))
        self.assertIn("--rank-gpu-id", str(ctx.exception))


class TestHullStorage(unittest.TestCase):
    """Which families may build a REAL full-width hull.

    The slice-A predicate asked "is this linear attention?", on the reasoning
    that a quantized checkpoint allocates its big weights lazily. That holds
    for GGUF (``GGUFUninitializedParameter``) and for nothing else: fp8's
    ``create_weights`` calls ``torch.empty``, so an FP8-GDN hull is the whole
    model a second time.
    """

    @staticmethod
    def _cfg(quantization=None, **attrs):
        text = types.SimpleNamespace(**attrs)
        return types.SimpleNamespace(
            hf_text_config=text, hf_config=text, quantization=quantization
        )

    def test_gguf_linear_attention_keeps_the_real_hull(self):
        self.assertTrue(
            dgl.hull_needs_real_storage(
                self._cfg(quantization="gguf", linear_num_key_heads=16)
            )
        )

    def test_fp8_linear_attention_goes_to_meta(self):
        self.assertFalse(
            dgl.hull_needs_real_storage(
                self._cfg(quantization="fp8", linear_num_key_heads=16)
            )
        )

    def test_dense_family_goes_to_meta(self):
        self.assertFalse(dgl.hull_needs_real_storage(self._cfg()))

    def test_unquantized_linear_attention_goes_to_meta_too(self):
        self.assertFalse(
            dgl.hull_needs_real_storage(self._cfg(linear_num_key_heads=16))
        )


class TestExpertParallelIsRefused(unittest.TestCase):
    """EP is a different decomposition, and the lane says so at build time."""

    @staticmethod
    def _moe(**attrs):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mod = FusedMoE.__new__(FusedMoE)
        nn.Module.__init__(mod)
        for k, v in attrs.items():
            setattr(mod, k, v)
        tree = nn.Module()
        tree.experts = mod
        return tree

    def test_pure_tp_experts_pass(self):
        _assert_lane_moe_is_pure_tp(self._moe(moe_ep_size=1))

    def test_expert_parallel_group_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            _assert_lane_moe_is_pure_tp(self._moe(moe_ep_size=2))
        self.assertIn("expert parallelism", str(ctx.exception))

    def test_an_all_to_all_dispatcher_is_refused(self):
        class _DeepEPish:
            pass

        with self.assertRaises(ValueError) as ctx:
            _assert_lane_moe_is_pure_tp(
                self._moe(moe_ep_size=1, dispatcher=_DeepEPish())
            )
        self.assertIn("all-to-all", str(ctx.exception))

    def test_expert_offload_is_refused_as_an_unbudgeted_post(self):
        with self.assertRaises(ValueError) as ctx:
            _assert_lane_moe_is_pure_tp(
                self._moe(moe_ep_size=1, _moe_offload_enabled=True)
            )
        self.assertIn("expert offload", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
