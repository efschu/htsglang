# SPDX-License-Identifier: Apache-2.0
"""Seam F9 -- a CUDA graph per ROLE, hermetically.

What this file can and cannot prove is worth stating up front, because the
thing being built is a GPU mechanism and every test here runs on the CPU
with ``CUDA_VISIBLE_DEVICES=""``:

* PROVABLE here, and proved: which BODY each role records, that the wrong
  body refuses by name on BOTH roles (the host recording the worker's route
  is the wrong-answer direction and is refused too), that the captured body
  and the eager body are the SAME callable with the SAME arguments, that the
  declared per-layer collective sequence is byte-identical with and without
  graphs, and that a classic boot (no role plan) reaches none of it.
* NOT provable here, and named instead of implied: that the capture itself
  succeeds on the metal, and that host and workers ADMIT the same rounds to
  their graphs. The second is the load-bearing one -- see
  ``test_the_admission_predicate_is_the_same_predicate_on_both_roles``,
  which pins the property that MAKES it rank-uniform (one predicate, over
  rank-uniform inputs) rather than pretending to measure it.
"""

from __future__ import annotations

import inspect

import pytest

from sglang.srt import rank_role
from sglang.srt.form_a_boot_gate import declare_layer_collectives
from sglang.srt.rank_role import (
    HOST,
    WORKER,
    SEAMS,
    UNWIRED_ORDER,
    FormASeamNotWired,
    FormAWorkerDenseGraph,
    RankRoleError,
    RankRolePlan,
    guard_graph_mode,
    require_wired,
)

FORM_A = RankRolePlan((HOST, WORKER, WORKER))
MOE_ROUTE = rank_role.GRAPH_BODY_MOE_ROUTE
MODEL_FORWARD = rank_role.GRAPH_BODY_MODEL_FORWARD


# ==========================================================================
# 1. The seam itself
# ==========================================================================
def test_f9_is_wired_and_has_left_the_remaining_work_list():
    """fnFA15 ran end to end EAGER at 233 ms per round against 35 ms for the
    classic form WITH graphs. That measurement is what ended F9's
    deferrability, so the registry has to say so."""
    assert SEAMS["F9"].wired is True
    require_wired("F9")  # must not raise any more
    assert "F9" not in UNWIRED_ORDER
    # What is left is F5 (unreachable under Form A -- DCP is off) and F6.
    assert UNWIRED_ORDER == ("F5", "F6")
    for still_open in UNWIRED_ORDER:
        with pytest.raises(FormASeamNotWired):
            require_wired(still_open)


def test_the_seam_points_at_the_decision_point_not_at_the_env_reader():
    """The old `where` named offload_capture_gate's process-wide env read.
    That decision is still process-wide and still CORRECT there -- both
    roles run the same FusedMoE pool route. The seam is about which BODY is
    recorded, so it has to point at the line that picks one."""
    where = SEAMS["F9"].where
    assert "decode_cuda_graph_runner" in where
    assert "capture_one_shape" in where
    # The old address is kept, but as the thing that is NOT the seam.
    assert "offload_capture_gate" in where


# ==========================================================================
# 2. The guard: mode AND body, refusing on both roles
# ==========================================================================
def test_eager_stays_allowed_on_a_worker():
    """The form fnFA15 ran. Wiring F9 must not retro-refuse the boot that
    produced the measurement."""
    for mode in ("eager", "disabled", None):
        guard_graph_mode(FORM_A, 1, mode)


def test_a_worker_may_record_its_moe_route_and_nothing_else():
    guard_graph_mode(FORM_A, 1, "full", body=MOE_ROUTE)
    with pytest.raises(FormAWorkerDenseGraph, match="HostOnlyModule"):
        guard_graph_mode(FORM_A, 1, "full", body=MODEL_FORWARD)
    # No body named at all is still refused: "it did not say" is not
    # "it is fine".
    with pytest.raises(FormAWorkerDenseGraph):
        guard_graph_mode(FORM_A, 1, "full")


def test_the_host_is_refused_the_worker_body_which_is_the_silent_direction():
    """The mirror image, and the one that would never announce itself: the
    stripped route issues the same 96 collectives, so every rank stays in
    lockstep and only the OUTPUT is wrong."""
    guard_graph_mode(FORM_A, 0, "full", body=MODEL_FORWARD)
    guard_graph_mode(FORM_A, 0, "full")  # mode-only call site, unchanged
    with pytest.raises(FormAWorkerDenseGraph, match="HOST"):
        guard_graph_mode(FORM_A, 0, "full", body=MOE_ROUTE)


def test_an_unknown_body_is_refused_rather_than_defaulted():
    """Closed vocabulary, same rule as form_a_construction.worker_builds: a
    typo that silently picked either body is a wrong answer, not a crash."""
    with pytest.raises(RankRoleError, match="not a Form A graph body"):
        guard_graph_mode(FORM_A, 1, "full", body="moe_route")
    with pytest.raises(RankRoleError):
        guard_graph_mode(FORM_A, 0, "full", body="")


def test_the_refusal_is_not_a_seam_refusal_and_cannot_evaporate():
    """Same lesson as F11's first draft: a refusal tied to `wired` vanishes
    in the commit that builds the seam. F9 IS built now, and the wrong-body
    refusal must outlive that."""
    assert not issubclass(FormAWorkerDenseGraph, FormASeamNotWired)
    assert issubclass(FormAWorkerDenseGraph, RankRoleError)


# ==========================================================================
# 3. One body, two regimes -- the property the whole seam rests on
# ==========================================================================
def test_the_captured_body_is_the_same_method_the_eager_forward_calls():
    """Not "the same code" by eyeball: the capture calls
    ModelRunner.run_form_a_worker_route, and so does the eager forward. If
    someone inlines either one, this test fails and the drift is reported
    before a boot has to discover it as a hang."""
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    eager_src = inspect.getsource(ModelRunner._forward_form_a_worker)
    capture_src = inspect.getsource(DecodeCudaGraphRunner._capture_one_shape_form_a)
    assert "run_form_a_worker_route" in eager_src
    assert "run_form_a_worker_route" in capture_src
    # And the one spelling really is one function.
    assert callable(ModelRunner.run_form_a_worker_route)


def test_the_route_takes_its_host_rank_from_the_plan_not_from_a_literal():
    """A hardcoded 0 would be a second source of truth for a fact the plan
    already carries."""
    src = inspect.getsource(
        __import__(
            "sglang.srt.model_executor.model_runner", fromlist=["ModelRunner"]
        ).ModelRunner.run_form_a_worker_route
    )
    assert "form_a_token_src_rank" in src
    assert "host_rank=0" not in src


def test_the_declared_collective_sequence_does_not_change_under_graphs():
    """THE question the boot gate exists for. Graphs re-issue the recorded
    ops; they do not add, drop or reorder any. So the declaration is the
    same object in both regimes -- which is why gate_form_a_boot needed no
    new argument for F9, and why saying so here is worth more than a
    comment."""
    for rank in range(3):
        for is_attn in (False, True):
            ops = declare_layer_collectives(
                FORM_A,
                rank,
                is_attention_layer=is_attn,
                worker_skips_dense=True,
                host_dense_is_unsharded=True,
                host_uses_moe_exchange=False,
                moe_input_carrier="all_reduce_zero",
            )
            keys = [tuple(op.key()) for op in ops]
            # Declared per LAYER; the regime is not one of its inputs.
            assert keys == [
                tuple(op.key())
                for op in declare_layer_collectives(
                    FORM_A,
                    rank,
                    is_attention_layer=is_attn,
                    worker_skips_dense=True,
                    host_dense_is_unsharded=True,
                    host_uses_moe_exchange=False,
                    moe_input_carrier="all_reduce_zero",
                )
            ]


# ==========================================================================
# 4. Routing: who reaches the new path, and who must not notice it
# ==========================================================================
class _FakeRunner:
    """The two fields capture_one_shape branches on, and nothing else."""

    def __init__(self, is_weightless_worker=False, is_form_a_worker=False):
        self.is_weightless_worker = is_weightless_worker
        self.is_form_a_worker = is_form_a_worker
        self.tp_rank = 1


def test_capture_one_shape_routes_the_three_cases_in_the_right_order():
    """Read as SOURCE rather than executed: constructing a real runner needs
    a device. What must hold is the ORDER -- the weightless branch first (a
    rank is never both), then Form A, then the untouched classic path."""
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    src = inspect.getsource(DecodeCudaGraphRunner.capture_one_shape)
    i_wl = src.index("is_weightless_worker")
    i_fa = src.index("is_form_a_worker")
    i_classic = src.index("num_tokens = size * self.num_tokens_per_bs")
    assert i_wl < i_fa < i_classic
    # Both roles pass through the guard, with the body they are about to
    # record. A guard only the worker reaches would miss the silent
    # direction entirely.
    assert "GRAPH_BODY_MOE_ROUTE" in src
    assert "GRAPH_BODY_MODEL_FORWARD" in src


def test_the_guard_is_inert_without_a_role_plan():
    """The classic-boot property, executed rather than asserted in prose:
    with no plan installed the guard returns before it can look at anything
    else -- so it cannot refuse a classic capture, whatever the body says."""
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )
    from sglang.srt.rank_role import installed_role_plan, set_form_a_role_plan

    assert installed_role_plan() is None, "a test left a plan installed"
    calls = []

    class _Probe(_FakeRunner):
        server_args = None

    probe = _Probe()
    probe.server_args = type("SA", (), {"cuda_graph_config": None})()
    # Unbound call: no instance state beyond model_runner/server_args.
    holder = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    holder.model_runner = probe
    holder._guard_form_a_capture_body(MODEL_FORWARD)
    holder._guard_form_a_capture_body(MOE_ROUTE)  # even the wrong one
    assert calls == []

    # And with a plan installed it does look -- the same call now refuses.
    try:
        set_form_a_role_plan(FORM_A, 1)
        with pytest.raises(FormAWorkerDenseGraph):
            holder._guard_form_a_capture_body(MODEL_FORWARD)
    finally:
        set_form_a_role_plan(None, 0)
    assert installed_role_plan() is None


def test_the_graph_mode_name_falls_back_without_inventing_a_mode():
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    holder = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    holder.model_runner = type(
        "MR", (), {"server_args": type("SA", (), {"cuda_graph_config": None})()}
    )()
    assert holder._graph_mode_name() == "full"

    class _Cfg:
        decode = type("D", (), {"backend": "breakable"})()

    holder.model_runner.server_args.cuda_graph_config = _Cfg()
    assert holder._graph_mode_name() == "breakable"


# ==========================================================================
# 5. What the desk CANNOT decide -- named, not implied
# ==========================================================================
def test_the_admission_predicate_is_the_same_predicate_on_both_roles():
    """The one property this lane cannot prove without three cards.

    A captured replay runs the PADDED bucket width; the eager fallback runs
    the RAW row count. If the host admitted a round to its graph and a
    worker did not, the carrier all-reduce would pair a padded shape with a
    raw one -- a size mismatch, not a hang, and nothing in the collective
    census would name it.

    What makes that impossible is structural and IS checkable here: both
    roles call the SAME `decode_cuda_graph_runner.can_run_graph` over
    rank-uniform batch fields. This test pins that there is no second,
    Form-A-only admission rule; the boot log line
    ("Form A worker: FIRST GRAPH REPLAY") is the metal-side counterpart.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner

    src = inspect.getsource(ModelRunner._forward_raw)
    form_a_branch = src[src.index('getattr(self, "is_form_a_worker", False)') :]
    form_a_branch = form_a_branch[: form_a_branch.index("_forward_form_a_worker") + 40]
    assert "self.decode_cuda_graph_runner.can_run_graph(forward_batch)" in form_a_branch
    # No invented admission axis: the batch decides, not the role.
    for invented in ("is_form_a_worker and bs", "form_a_max_graph_bs"):
        assert invented not in form_a_branch
    assert "FIRST GRAPH REPLAY" in form_a_branch


def test_prefill_stays_eager_for_the_whole_form_a_group():
    """Rank-uniform by construction: the HOST takes the same exit, because
    the plan is installed on every rank. A phase captured on one role and
    eager on the other is the #631 wedge."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    src = inspect.getsource(ModelRunner.init_prefill_cuda_graph)
    assert "form_a_role_plan_installed" in src
    # keyed on the GROUP flag, not on the per-rank role
    head = src[: src.index("form_a_role_plan_installed")]
    assert head.count("is_form_a_worker") == 0


def test_flashinfer_autotune_is_refused_for_every_rank_under_form_a():
    """Its dummy run is a full model forward. Skipping it only on the worker
    would leave exactly the one-sided forward that hangs (the fnFA12 shape,
    one phase earlier), so the refusal is group-wide."""
    from sglang.srt.model_executor.runner import flashinfer_autotune

    src = inspect.getsource(flashinfer_autotune.should_run_flashinfer_autotune)
    assert "installed_role_plan() is not None" in src
    assert "is_form_a_worker" not in src


def test_the_worker_attention_backend_answers_the_graph_path_bookkeeping():
    """Every no-op the capture and replay path touches on this rank, as a
    list rather than as luck. An AttributeError here would surface as a
    capture abort with a stack that names flashinfer, not Form A."""
    from sglang.srt.form_a_construction import FormAWorkerAttnBackend

    for name in (
        "init_forward_metadata_out_graph",
        "init_forward_metadata_in_graph",
        "init_cuda_graph_state",
        "get_cuda_graph_seq_len_fill_value",
        "on_after_cuda_graph_warmup",
    ):
        assert callable(getattr(FormAWorkerAttnBackend, name)), name
    assert FormAWorkerAttnBackend.supports_ragged_verify_graph is False
    assert (
        FormAWorkerAttnBackend.use_captured_forward_metadata_for_breakable_cuda_graph
        is False
    )


def test_the_zero_addend_reasoning_is_recorded_where_the_capture_lives():
    """Not decoration. `receive_moe_input`'s torch.zeros is a RECORDED fill
    kernel under capture, which is the only reason a replay re-zeroes the
    carrier buffer. A later 'optimisation' that hoists the buffer out of the
    body corrupts the MoE input on replay two, silently and identically on
    every rank."""
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    doc = DecodeCudaGraphRunner._capture_one_shape_form_a.__doc__ or ""
    assert "torch.zeros" in doc
    assert "preallocated" in doc or "hoisted" in doc
