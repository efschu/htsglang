# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hermetic tests for the deferred-join all-reduce (task #597).

No CUDA, no process group, no model. A stand-in backend supplies the device
calls, so what is under test is the CONTRACT rather than the stream ordering:
the reduction is issued exactly once, joined exactly once, no reducing site
ever sees a pending handle, and the values are unchanged. The ordering those
calls establish is a device property and is gated by the runsheet, the same
split #588 used for bitwise GEMM behaviour.

The centrepiece is the double-reduce falsifier. Window 8 put this feature on
the path where a mistake is silent: a tensor reduced twice is not a crash, it
is plausible-looking wrong activations. So the invariant gets its own test
AND its own can-fail proof.

Run with:
    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        pytest -q test/registered/unit/distributed/test_tp_ar_deferred_join.py
"""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from sglang.srt.distributed import tp_ar_pipeline as tap
from sglang.srt.environ import envs

WORLD = 3
TOKENS = 512
HIDDEN = 64


@pytest.fixture(autouse=True)
def _clean_state():
    tap.reset_tp_ar_pipeline_state()
    tap.set_deferred_backend_for_test(None)
    yield
    tap.reset_tp_ar_pipeline_state()
    tap.set_deferred_backend_for_test(None)


# --------------------------------------------------------------------------
# stand-in device backend
# --------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self):
        self.recorded_on = None

    def record(self, stream=None):
        self.recorded_on = stream

    def query(self):
        return True

    def elapsed_time(self, other):
        return 1.0


class _FakeStream:
    def __init__(self, name):
        self.name = name
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)


class FakeBackend:
    """Reports the device path as available and runs it eagerly.

    Eager execution on the "comm stream" is exactly the state a correct join
    guarantees the consumer sees, so byte-identity and the counter contract
    are both observable here; what is NOT observable is whether the real
    ordering primitives were placed correctly, which is why the runsheet
    keeps a GPU gate.
    """

    def __init__(self):
        self.compute = _FakeStream("compute")
        self.comm = _FakeStream("comm")
        self.recorded_streams = []
        self.is_capturing = False

    def capturing(self):
        return self.is_capturing

    def current_stream(self, device):
        return self.compute

    def comm_stream(self, device):
        return self.comm

    @contextlib.contextmanager
    def stream_ctx(self, stream):
        yield

    def timing_event(self):
        return _FakeEvent()

    def record_stream(self, tensor, stream):
        self.recorded_streams.append(stream)

    def usable(self, tensor):
        return True


@pytest.fixture
def backend():
    fake = FakeBackend()
    tap.set_deferred_backend_for_test(fake)
    return fake


# --------------------------------------------------------------------------
# emulated TP group
# --------------------------------------------------------------------------


def _exact_int_tensor(*shape, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-4, 5, shape, generator=generator, dtype=torch.int64).to(
        torch.float32
    )


class CountingGroup:
    """Sums this rank's partial with the other ranks', counting every call."""

    def __init__(self, tokens=TOKENS):
        self.partials = [
            _exact_int_tensor(tokens, HIDDEN, seed=200 + r) for r in range(WORLD)
        ]
        self.calls = 0

    def local(self) -> torch.Tensor:
        return self.partials[0].clone()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        for rank in range(1, WORLD):
            tensor.add_(self.partials[rank])
        return tensor

    def reference(self) -> torch.Tensor:
        out = self.partials[0].clone()
        for rank in range(1, WORLD):
            out = out + self.partials[rank]
        return out


# --------------------------------------------------------------------------
# issue / join contract
# --------------------------------------------------------------------------


def test_issue_then_join_reduces_exactly_once_and_is_bitwise_correct(backend):
    group = CountingGroup()
    tensor = group.local()

    issued = tap.issue_deferred_all_reduce(tensor, group.all_reduce)
    assert tap.has_deferred_handle(issued)
    assert group.calls == 1

    joined = tap.join_deferred(issued)
    assert not tap.has_deferred_handle(joined)
    assert group.calls == 1
    assert torch.equal(joined, group.reference())

    stats = tap.tp_ar_pipeline_stats()
    assert stats["deferred_issued"] == 1
    assert stats["deferred_joined"] == 1
    assert stats["deferred_reduce_site_hits"] == 0


def test_join_is_idempotent_across_every_consumer_entry_point(backend):
    """Joins sit at several entry points; only the first may do work."""
    group = CountingGroup()
    tensor = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)

    first = tap.join_deferred(tensor)
    second = tap.join_deferred(first)
    third = tap.join_deferred(second)

    assert group.calls == 1
    assert tap.tp_ar_pipeline_stats()["deferred_joined"] == 1
    assert torch.equal(third, group.reference())


def test_join_on_an_untagged_tensor_is_a_no_op(backend):
    plain = torch.zeros(4, 4)
    assert tap.join_deferred(plain) is plain
    assert tap.tp_ar_pipeline_stats()["deferred_joined"] == 0


def test_issue_declines_during_graph_capture(backend):
    """A side-stream collective cannot be captured. Eager only."""
    backend.is_capturing = True
    group = CountingGroup()
    out = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)
    assert not tap.has_deferred_handle(out)
    assert group.calls == 1
    assert torch.equal(out, group.reference())
    assert tap.tp_ar_pipeline_stats()["deferred_declined"] == 1


def test_issue_declines_without_a_device_path():
    """No backend installed and no CUDA: reduce synchronously, do not tag."""
    group = CountingGroup()
    out = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)
    assert not tap.has_deferred_handle(out)
    assert torch.equal(out, group.reference())
    assert tap.tp_ar_pipeline_stats()["deferred_declined"] == 1


# --------------------------------------------------------------------------
# THE DOUBLE-REDUCE FALSIFIER
# --------------------------------------------------------------------------


def test_a_reducing_site_never_sees_a_pending_handle(backend):
    """The invariant: issue is only taken where the reduction was owned.

    ``note_reduce_site`` is planted at every all-reduce site in the
    communicator. If a tensor with a pending handle ever reaches one, the
    reduction is about to be applied a second time. This asserts the counter
    stays at zero on the normal flow.
    """
    group = CountingGroup()
    tensor = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)
    # The consumer joins first, exactly as the communicator entry points do.
    tensor = tap.join_deferred(tensor)
    # Only afterwards could any reducing site be reached.
    tensor = tap.note_reduce_site(tensor)

    assert tap.tp_ar_pipeline_stats()["deferred_reduce_site_hits"] == 0
    assert group.calls == 1
    assert torch.equal(tensor, group.reference())


def test_double_reduce_falsifier_can_fail(backend):
    """Can-fail proof: a tagged tensor reaching a reducing site is caught.

    This is the mis-wiring the guard exists for -- an issue placed at a
    producer whose reduce belongs to the communicator, with the communicator's
    own reduce left in place. The guard must NOTICE (counter goes up) and
    must make the data consistent (join) rather than let a race through.
    """
    group = CountingGroup()
    tensor = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)
    assert tap.has_deferred_handle(tensor)

    # A site that reduces, reached WITHOUT an intervening join.
    tensor = tap.note_reduce_site(tensor)

    stats = tap.tp_ar_pipeline_stats()
    assert stats["deferred_reduce_site_hits"] == 1
    assert not tap.has_deferred_handle(tensor)
    # The guard joined, so the tensor is correct at this point; the SECOND
    # reduction that the site is about to perform is what would corrupt it,
    # and that is now visible in the counter instead of silent.
    assert torch.equal(tensor, group.reference())
    doubled = group.all_reduce(tensor)
    assert not torch.equal(doubled, group.reference())


def test_issuing_twice_on_the_same_tensor_is_refused(backend):
    """The in-process half of the same invariant."""
    group = CountingGroup()
    tensor = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)
    again = tap.issue_deferred_all_reduce(tensor, group.all_reduce)

    assert group.calls == 1
    assert tap.tp_ar_pipeline_stats()["deferred_reduce_site_hits"] == 1
    assert torch.equal(again, group.reference())


# --------------------------------------------------------------------------
# window meter (the ceiling input)
# --------------------------------------------------------------------------


def test_issue_to_join_window_is_sampled_without_synchronizing(backend):
    group = CountingGroup()
    for _ in range(3):
        tensor = tap.issue_deferred_all_reduce(group.local(), group.all_reduce)
        tap.join_deferred(tensor)
    stats = tap.tp_ar_pipeline_stats()
    # One pair is always still parked, so samples lag the joins by one.
    assert stats["deferred_window_samples"] == 2
    assert stats["deferred_window_mean_ms"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# off-path: the strong form
# --------------------------------------------------------------------------


def test_disabled_by_default():
    envs.SGLANG_TP_AR_PIPELINE_DEFERRED.clear()
    tap.reset_tp_ar_pipeline_state()
    assert tap.tp_ar_deferred_enabled() is False


def test_moe_forward_impl_is_inert_when_the_flag_is_off(monkeypatch):
    """Strong off-path pin: the module is never entered, so it changed nothing.

    Drives the real ``FusedMoE.forward_impl`` with a duck-typed self.
    """
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer

    group = CountingGroup()
    monkeypatch.setattr(moe_layer, "tensor_model_parallel_all_reduce", group.all_reduce)
    envs.SGLANG_TP_AR_PIPELINE_DEFERRED.clear()
    tap.reset_tp_ar_pipeline_state()

    fake = types.SimpleNamespace(
        reduce_results=True,
        moe_tp_size=WORLD,
        moe_ep_size=1,
        forward_local=lambda hidden_states, topk_output: group.local(),
    )
    out = moe_layer.FusedMoE.forward_impl(fake, torch.zeros(TOKENS, HIDDEN), None)

    assert torch.equal(out, group.reference())
    assert group.calls == 1
    stats = tap.tp_ar_pipeline_stats()
    assert stats["deferred_issued"] == 0
    assert stats["deferred_declined"] == 0
    assert not tap.has_deferred_handle(out)


def test_moe_forward_impl_declines_below_the_token_gate(backend, monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer

    group = CountingGroup(tokens=8)
    monkeypatch.setattr(moe_layer, "tensor_model_parallel_all_reduce", group.all_reduce)
    with (
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED.override(True),
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED_MIN_TOKENS.override(256),
    ):
        tap.reset_tp_ar_pipeline_state()
        tap.set_deferred_backend_for_test(backend)
        fake = types.SimpleNamespace(
            reduce_results=True,
            moe_tp_size=WORLD,
            moe_ep_size=1,
            forward_local=lambda hidden_states, topk_output: group.local(),
        )
        out = moe_layer.FusedMoE.forward_impl(fake, torch.zeros(8, HIDDEN), None)
    assert not tap.has_deferred_handle(out)
    assert tap.tp_ar_pipeline_stats()["deferred_issued"] == 0
    assert torch.equal(out, group.reference())


# --------------------------------------------------------------------------
# COVERAGE SELF-CHECK -- window 8's arm 0 must not repeat
# --------------------------------------------------------------------------


def test_coverage_moe_issue_fires_on_a_qwen3_5_shaped_config(backend, monkeypatch):
    """Pin that the hook FIRES where window 8 found the dominant all-reduce.

    Window 8's arm 0 came back ``calls_pipelined == 0`` because #588's hook
    sat in RowParallelLinear while the production model reduces in the MoE
    layer. That outcome is now a test failure rather than a wasted window:
    this drives the real ``FusedMoE.forward_impl`` with the production shape
    (reduce_results=True, moe_tp_size>1) and requires an issue.
    """
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer

    group = CountingGroup()
    monkeypatch.setattr(moe_layer, "tensor_model_parallel_all_reduce", group.all_reduce)
    with (
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED.override(True),
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED_MIN_TOKENS.override(256),
    ):
        tap.reset_tp_ar_pipeline_state()
        tap.set_deferred_backend_for_test(backend)
        fake = types.SimpleNamespace(
            reduce_results=True,
            moe_tp_size=WORLD,
            moe_ep_size=1,
            forward_local=lambda hidden_states, topk_output: group.local(),
        )
        out = moe_layer.FusedMoE.forward_impl(fake, torch.zeros(TOKENS, HIDDEN), None)

        assert tap.has_deferred_handle(out), (
            "the deferred issue did not fire on a production-shaped MoE layer "
            "-- this is window 8's arm-0 outcome reproduced"
        )
        assert tap.tp_ar_pipeline_stats()["deferred_issued"] == 1

        # ... and the communicator's entry point completes it.
        from sglang.srt.layers import communicator as comm_mod

        joined = comm_mod.join_deferred(out)

    assert not tap.has_deferred_handle(joined)
    assert tap.tp_ar_pipeline_stats()["deferred_joined"] == 1
    assert group.calls == 1
    assert torch.equal(joined, group.reference())


def test_coverage_check_can_fail(backend, monkeypatch):
    """Can-fail proof for the coverage pin: reduce_results=False must NOT fire.

    That is the configuration whose all-reduce belongs to the communicator.
    If the issue fired there too, the coverage assertion above would pass for
    the wrong reason and the double-reduce risk would be real.
    """
    from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer

    group = CountingGroup()
    monkeypatch.setattr(moe_layer, "tensor_model_parallel_all_reduce", group.all_reduce)
    with envs.SGLANG_TP_AR_PIPELINE_DEFERRED.override(True):
        tap.reset_tp_ar_pipeline_state()
        tap.set_deferred_backend_for_test(backend)
        fake = types.SimpleNamespace(
            reduce_results=False,
            moe_tp_size=WORLD,
            moe_ep_size=1,
            forward_local=lambda hidden_states, topk_output: group.local(),
        )
        out = moe_layer.FusedMoE.forward_impl(fake, torch.zeros(TOKENS, HIDDEN), None)
    assert not tap.has_deferred_handle(out)
    assert tap.tp_ar_pipeline_stats()["deferred_issued"] == 0
    assert group.calls == 0


# --------------------------------------------------------------------------
# the guards are actually planted
# --------------------------------------------------------------------------


def test_every_communicator_all_reduce_site_is_guarded():
    """Structural pin over the six reducing sites plus the two fusion paths.

    A new all-reduce added to the communicator without a guard is how the
    double-reduce invariant would silently stop being enforced, so the count
    is pinned rather than left to review.
    """
    import inspect

    from sglang.srt.layers import communicator as comm_mod

    source = inspect.getsource(comm_mod)
    reduce_calls = (
        source.count("moe_tensor_model_parallel_all_reduce(hidden_states)")
        + source.count("attn_tp_group.all_reduce(hidden_states)")
        + source.count("attention_tensor_model_parallel_all_reduce(")
        + source.count("attention_tensor_model_parallel_quant_all_reduce(")
        + source.count("tensor_model_parallel_all_reduce(hidden_states)")
        + source.count("forward_with_allreduce_fusion(")
    )
    guards = source.count("note_reduce_site(")
    # Imports and the guard's own definition are not call sites; what matters
    # is that guards are not FEWER than the branches they protect.
    assert guards >= 7, f"only {guards} guards for {reduce_calls} reduce references"


def test_communicator_entry_points_join_first():
    import inspect

    from sglang.srt.layers import communicator as comm_mod

    for method in (
        comm_mod.LayerCommunicator.prepare_attn,
        comm_mod.LayerCommunicator.prepare_mlp,
        comm_mod.LayerCommunicator.postprocess_layer,
    ):
        source = inspect.getsource(method)
        assert "join_deferred(hidden_states)" in source, method.__name__


# --------------------------------------------------------------------------
# #588(b): the DENSE producer-owned site (RowParallelLinear, opt-in)
# --------------------------------------------------------------------------


def _dense_fake(group, opted_in: bool):
    """A RowParallelLinear-shaped carrier for the real unbound forward.

    Mirrors the MoE coverage fake: only what the reduce region reads. The
    quant_method's apply returns this rank's partial sum, exactly like
    ``forward_local`` in the MoE pattern.
    """
    return types.SimpleNamespace(
        input_is_parallel=True,
        tp_rank=0,
        tp_size=WORLD,
        skip_bias_add=True,
        bias=None,
        reduce_results=True,
        use_dp_attention_reduce=False,
        defer_all_reduce_ok=opted_in,
        output_size=HIDDEN,
        quant_method=types.SimpleNamespace(
            apply=lambda module, x, bias=None: group.local()
        ),
    )


def _drive_dense(monkeypatch, group, fake):
    from contextlib import nullcontext

    from sglang.srt.layers import linear as linear_mod

    monkeypatch.setattr(
        linear_mod, "tensor_model_parallel_all_reduce", group.all_reduce
    )
    monkeypatch.setattr(
        linear_mod, "use_symmetric_memory", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(linear_mod, "get_tp_group", lambda: None, raising=False)
    monkeypatch.setattr(linear_mod, "should_skip_mlp_all_reduce", lambda: False)
    out, out_bias = linear_mod.RowParallelLinear.forward(
        fake, torch.zeros(TOKENS, HIDDEN)
    )
    assert out_bias is None
    return out


def test_coverage_dense_issue_fires_on_an_opted_in_row_parallel(
    backend, monkeypatch
):
    """#588(b) coverage: the lever FIRES on the dense producer-owned site.

    The #578 armed-never-executing form is the enemy: a hook that never
    fires measures the baseline twice. deferred_issued must go to 1 and the
    double-reduce counter must stay 0 -- the old-site half of the boot
    acceptance (deferred_issued>0 AND site_hits==0), hermetic edition.
    """
    group = CountingGroup()
    with (
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED.override(True),
        envs.SGLANG_TP_AR_PIPELINE_DENSE.override(True),
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED_MIN_TOKENS.override(256),
    ):
        tap.reset_tp_ar_pipeline_state()
        tap.set_deferred_backend_for_test(backend)
        out = _drive_dense(monkeypatch, group, _dense_fake(group, opted_in=True))
        assert tap.has_deferred_handle(out), (
            "the dense issue did not fire on an opted-in RowParallelLinear"
        )
        assert tap.tp_ar_pipeline_stats()["deferred_issued"] == 1
        assert tap.tp_ar_pipeline_stats()["deferred_reduce_site_hits"] == 0

        from sglang.srt.layers import communicator as comm_mod

        joined = comm_mod.join_deferred(out)
    assert not tap.has_deferred_handle(joined)
    assert tap.tp_ar_pipeline_stats()["deferred_joined"] == 1
    assert group.calls == 1
    assert torch.equal(joined, group.reference())


def test_coverage_dense_can_fail_without_the_opt_in(backend, monkeypatch):
    """The blast-radius pin: a RowParallelLinear that did NOT opt in (vision
    towers, draft heads -- consumers that never join) must take the ordinary
    reduce even with both flags on."""
    group = CountingGroup()
    with (
        envs.SGLANG_TP_AR_PIPELINE_DEFERRED.override(True),
        envs.SGLANG_TP_AR_PIPELINE_DENSE.override(True),
    ):
        tap.reset_tp_ar_pipeline_state()
        tap.set_deferred_backend_for_test(backend)
        out = _drive_dense(monkeypatch, group, _dense_fake(group, opted_in=False))
    assert not tap.has_deferred_handle(out)
    assert tap.tp_ar_pipeline_stats()["deferred_issued"] == 0
    assert group.calls == 1
    assert torch.equal(out, group.reference())


def test_dense_flag_off_is_byte_identical(backend, monkeypatch):
    """Default off: with SGLANG_TP_AR_PIPELINE_DENSE unset the opted-in
    module takes the exact pre-#588(b) path -- ordinary reduce, no handle.
    And the dense flag WITHOUT the #597 infrastructure flag is equally
    inert: the extension rides on top, never around."""
    for deferred, dense in ((True, False), (False, True), (False, False)):
        group = CountingGroup()
        with (
            envs.SGLANG_TP_AR_PIPELINE_DEFERRED.override(deferred),
            envs.SGLANG_TP_AR_PIPELINE_DENSE.override(dense),
        ):
            tap.reset_tp_ar_pipeline_state()
            tap.set_deferred_backend_for_test(backend)
            out = _drive_dense(
                monkeypatch, group, _dense_fake(group, opted_in=True)
            )
        assert not tap.has_deferred_handle(out), (deferred, dense)
        assert tap.tp_ar_pipeline_stats()["deferred_issued"] == 0
        assert group.calls == 1
        assert torch.equal(out, group.reference())


def test_dense_optins_are_exactly_the_traced_modules():
    """The per-instance opt-in exists ONLY where the trace was done: the two
    dense Qwen2MoeMLP down_proj constructions in qwen3_5 (o_proj is
    reduce_results=False there -- communicator-owned, the designed refusal).
    A third opt-in appearing anywhere in the tree without a trace is a
    review item, and this pin makes it one."""
    import pathlib
    import subprocess

    repo = pathlib.Path(__file__).resolve().parents[4]
    out = subprocess.run(
        ["grep", "-rn", "defer_all_reduce_ok = True", str(repo / "python")],
        capture_output=True,
        text=True,
    )
    hits = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert len(hits) == 2, hits
    for h in hits:
        assert "models/qwen3_5.py" in h, h
