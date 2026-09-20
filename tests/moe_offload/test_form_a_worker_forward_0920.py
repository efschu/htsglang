# SPDX-License-Identifier: Apache-2.0
"""Form A slice 6a, second half: what a WORKER rank RUNS.

Hermetic. No CUDA, no model, no process group: every collective is passed
in as a callable and every tensor is a small CPU one, which is the only way
the question "does the worker enter the same ops in the same order as the
host" can be answered before a GPU window is spent.

The question this file is built around is NOT "does the worker skip the
dense path" -- slice 6 answered that at construction time. It is the one
the collective-sequence model could not see: WHERE DOES A RANK WITH NO
DENSE CHAIN GET THE MoE INPUT? On a classic boot nowhere, because every
rank computes it (the dense path ends in an all-reduce and every rank holds
the same result afterwards). Under Form A that value exists on one card.
"""

import pytest
import torch

from sglang.srt.form_a_construction import HostOnlyModule
from sglang.srt.form_a_symmetry import (
    CollectiveMismatch,
    FormAWorkerWithoutMoeInput,
    RankTrace,
    probe_form_a_boot,
    trace_forward_edges,
)
from sglang.srt.form_a_worker_forward import (
    FormAMoeInputUnavailable,
    FormAWorkerForwardError,
    FormAWorkerLayerMismatch,
    MOE_INPUT_CARRIERS,
    form_a_moe_blocks,
    moe_input_carrier,
    publish_moe_input,
    receive_moe_input,
    run_form_a_worker_layers,
)
from sglang.srt.rank_role import (
    RankRoleError,
    RankRolePlan,
    set_form_a_role_plan,
)

FORM_A = RankRolePlan(("host", "worker", "worker"))


@pytest.fixture(autouse=True)
def _no_plan_leaks():
    """The role plan is process-global by design (every F3 site reads it
    instead of threading a server_args through the model). Leaking it out
    of a test changes every later test in the session."""
    try:
        yield
    finally:
        set_form_a_role_plan(None)


# ==========================================================================
# 1. The carrier: an all-reduce whose other addends are zero IS a broadcast
# ==========================================================================
def _fake_all_reduce(tensors):
    """Sum the per-rank tensors the way a real all-reduce would, and hand
    every rank the same result object's value."""
    total = sum(tensors[1:], tensors[0].clone())
    return [total.clone() for _ in tensors]


def test_the_host_value_reaches_every_worker_unchanged():
    host_value = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    contributions = [host_value]

    set_form_a_role_plan(FORM_A, 1)
    worker_buf = {}

    def _capture(t):
        worker_buf["t"] = t
        contributions.append(t)
        return t

    receive_moe_input(
        2, 3, dtype=torch.float32, device="cpu", all_reduce=_capture,
        carrier="all_reduce_zero",
    )
    # the worker's contribution is EXACTLY zero -- not "small", not
    # "ignored": it holds no dense weights, so its addend to the dense sum
    # is zero and the sum is the host's value untouched.
    assert torch.equal(worker_buf["t"], torch.zeros(2, 3))

    summed = _fake_all_reduce(contributions + [torch.zeros(2, 3)])
    for got in summed:
        assert torch.equal(got, host_value)


def test_the_host_publishes_the_value_it_then_computes_on():
    """The call site rebinds, so the host cannot compute its own experts on
    a tensor the workers never saw."""
    seen = []
    x = torch.ones(4, 8)
    out = publish_moe_input(
        x, carrier="all_reduce_zero", all_reduce=lambda t: (seen.append(t), t * 2)[1]
    )
    assert seen and seen[0] is x
    assert torch.equal(out, x * 2)


def test_an_empty_batch_is_not_carried_on_either_side():
    """The MoE block short-circuits a zero-row forward before its own
    all-reduce (qwen2_moe.py), so a carrier there would be an op the
    combine never answers. The row count is rank-uniform, so this branch is
    taken by every rank or by none."""
    empty = torch.zeros(0, 8)
    calls = []
    out = publish_moe_input(
        empty, carrier="all_reduce_zero", all_reduce=lambda t: calls.append(t)
    )
    assert out is empty and not calls

    with pytest.raises(FormAMoeInputUnavailable, match="rank-uniform"):
        receive_moe_input(0, 8, dtype=torch.float32, device="cpu",
                          carrier="all_reduce_zero")


def test_an_unknown_width_refuses_rather_than_guessing():
    with pytest.raises(FormAMoeInputUnavailable, match="hidden_size"):
        receive_moe_input(4, 0, dtype=torch.float32, device="cpu",
                          carrier="all_reduce_zero")


# ==========================================================================
# 2. The carrier is DECLARED, not assumed
# ==========================================================================
def test_the_carrier_defaults_to_the_one_that_needs_no_new_transport():
    assert moe_input_carrier({}) == "all_reduce_zero"
    assert moe_input_carrier({"SGLANG_FORM_A_MOE_INPUT": ""}) == "all_reduce_zero"
    assert moe_input_carrier({"SGLANG_FORM_A_MOE_INPUT": "broadcast"}) == "broadcast"
    assert set(MOE_INPUT_CARRIERS) == {"all_reduce_zero", "broadcast"}


def test_an_unknown_carrier_refuses_instead_of_defaulting():
    """Two ranks reading the carrier differently is a HANG, not a
    slowdown: they would enter different ops at the same position."""
    with pytest.raises(RankRoleError, match="hang, not a slowdown"):
        moe_input_carrier({"SGLANG_FORM_A_MOE_INPUT": "allreduce"})


def test_the_broadcast_carrier_uses_the_host_rank_as_source():
    got = {}

    def _bcast(t, src):
        got["src"] = src
        return t
    publish_moe_input(torch.ones(2, 3), host_rank=0, carrier="broadcast",
                      broadcast=_bcast)
    assert got["src"] == 0
    receive_moe_input(2, 3, dtype=torch.float32, device="cpu", host_rank=0,
                      carrier="broadcast", broadcast=_bcast)
    assert got["src"] == 0


# ==========================================================================
# 3. The layer list comes from the TREE, not from a config count
# ==========================================================================
class _FakeMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Identity()
        self.seen = []

    def forward(self, x, forward_batch=None):
        self.seen.append(x)
        return x


class _FakeLayer(torch.nn.Module):
    def __init__(self, layer_id, moe=True):
        super().__init__()
        self.layer_id = layer_id
        self.mlp = _FakeMoE() if moe else torch.nn.Identity()


class _FakeModel(torch.nn.Module):
    def __init__(self, kinds):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_FakeLayer(i, moe=k) for i, k in enumerate(kinds)]
        )
        self.start_layer = 0


def test_the_layer_list_is_the_hosts_order_and_skips_dense_mlps():
    model = _FakeModel([True, False, True, True])
    blocks = form_a_moe_blocks(model)
    assert [lid for lid, _ in blocks] == [0, 2, 3]


def test_a_model_without_layers_or_without_experts_refuses_by_name():
    with pytest.raises(FormAWorkerLayerMismatch, match="model.layers"):
        form_a_moe_blocks(object())
    with pytest.raises(FormAWorkerLayerMismatch, match="no routed-MoE block"):
        form_a_moe_blocks(_FakeModel([False, False]))


# ==========================================================================
# 4. The worker's forward: two ops per layer, in the host's order
# ==========================================================================
def test_the_worker_runs_only_the_moe_blocks_and_only_on_carried_rows():
    set_form_a_role_plan(FORM_A, 2)
    model = _FakeModel([True, True, True])
    blocks = form_a_moe_blocks(model)
    host_rows = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def _receive(num_tokens, hidden_size, **kw):
        assert (num_tokens, hidden_size) == (2, 3)
        return host_rows

    served = run_form_a_worker_layers(
        blocks, num_tokens=2, hidden_size=3, dtype=torch.float32,
        device="cpu", receive=_receive,
    )
    assert served == 3
    for _lid, mlp in blocks:
        assert len(mlp.seen) == 1
        assert torch.equal(mlp.seen[0], host_rows)


def test_the_stripped_forward_refuses_to_run_on_the_host():
    """Running it on rank 0 would skip the entire dense model and return
    numbers that look fine -- the one failure mode in this file that would
    not announce itself."""
    set_form_a_role_plan(FORM_A, 0)
    with pytest.raises(FormAWorkerForwardError, match="not a Form A worker"):
        run_form_a_worker_layers(
            form_a_moe_blocks(_FakeModel([True])), num_tokens=1,
            hidden_size=3, dtype=torch.float32, device="cpu",
        )


def test_it_is_inert_without_a_role_plan():
    set_form_a_role_plan(None)
    with pytest.raises(FormAWorkerForwardError, match="not a Form A worker"):
        run_form_a_worker_layers(
            form_a_moe_blocks(_FakeModel([True])), num_tokens=1,
            hidden_size=3, dtype=torch.float32, device="cpu",
        )


# ==========================================================================
# 5. The construction skip, at the modules the worker forward needs gone
# ==========================================================================
@pytest.mark.parametrize(
    "kind",
    ["self_attn", "linear_attn", "o_proj", "embed_tokens", "lm_head",
     "shared_expert", "hyper_connection", "ple", "draft"],
)
def test_every_dense_kind_the_worker_forward_skips_has_a_placeholder(kind):
    from sglang.srt.form_a_construction import skip_on_worker

    set_form_a_role_plan(FORM_A, 1)
    ph = skip_on_worker(kind, f"layers.0.{kind}")
    assert isinstance(ph, HostOnlyModule)
    assert list(ph.parameters()) == [] and list(ph.buffers()) == []


def test_the_router_is_the_one_moe_side_module_a_worker_still_builds():
    from sglang.srt.form_a_construction import skip_on_worker

    set_form_a_role_plan(FORM_A, 1)
    assert skip_on_worker("experts") is None
    assert skip_on_worker("moe_gate") is None


# ==========================================================================
# 6. The whole-boot probe with the carrier and the vocab edges (F13)
# ==========================================================================
def test_the_built_form_a_is_symmetric_end_to_end():
    """Host + two workers, 48 layers, carrier on, vocab host-only: the
    configuration this seat built, with no divergence and no missing
    input."""
    traces = probe_form_a_boot(
        FORM_A,
        num_layers=48,
        worker_skips_dense=True,
        host_uses_moe_exchange=False,
        host_dense_is_unsharded=True,
        moe_input_carrier="all_reduce_zero",
        vocab_is_host_only=True,
    )
    assert len({len(t.ops) for t in traces}) == 1
    assert len(traces[0].ops) == 96
    # every op is the same MoE pair, layer after layer, on every rank
    assert [o.site for o in traces[0].ops[:4]] == [
        "layer0.moe_in", "layer0.moe_combine",
        "layer1.moe_in", "layer1.moe_combine",
    ]
    for t in traces[1:]:
        assert [o.key() for o in t.ops] == [o.key() for o in traces[0].ops]


def test_leaving_the_vocab_even_split_hangs_the_boot_one_op_earlier():
    """F13. `tp_vocab_ratios` deliberately does NOT inherit the base ratio
    vector ("vocab always even"), so without this seam the host would
    all-reduce the embedding with two ranks that hold no vocab rows and
    never reach the module. That is the F12 failure one op before layer
    zero -- and the seam survey missed it because it is not per-layer."""
    with pytest.raises(CollectiveMismatch) as e:
        probe_form_a_boot(
            FORM_A,
            num_layers=4,
            worker_skips_dense=True,
            host_uses_moe_exchange=False,
            host_dense_is_unsharded=True,
            moe_input_carrier="all_reduce_zero",
            vocab_is_host_only=False,
        )
    assert "collective #0 differs" in str(e.value)
    assert "embed_tokens" in str(e.value)


def test_the_vocab_edges_are_symmetric_on_a_classic_boot():
    """The model has to agree with a boot that did not hang: today every
    rank holds a vocab shard and issues both edge collectives."""
    traces = [RankTrace(r, FORM_A.role_of(r)) for r in range(3)]
    for t in traces:
        trace_forward_edges(t, worker_skips_dense=False, vocab_is_host_only=False)
    assert all(len(t.ops) == 2 for t in traces)
    assert [o.key() for o in traces[1].ops] == [o.key() for o in traces[0].ops]


def test_the_probe_still_refuses_the_48_collective_configuration():
    """Belt and braces against the number coming back: the layout the
    switch matrix priced at 48 has no MoE input at all."""
    with pytest.raises(FormAWorkerWithoutMoeInput):
        probe_form_a_boot(
            FORM_A, num_layers=8, worker_skips_dense=True,
            host_uses_moe_exchange=False, host_dense_is_unsharded=True,
            moe_input_carrier=None,
        )


# ==========================================================================
# 7. The boot gate declares the carrier, so a divergent env is caught
# ==========================================================================
def test_the_boot_gate_catches_two_ranks_reading_different_carriers():
    from sglang.srt.form_a_boot_gate import (
        FormARanksDisagree,
        declare_layer_collectives,
        assert_ranks_agree,
    )
    from sglang.srt.form_a_boot_gate import _encode

    def _ops(rank, carrier):
        return declare_layer_collectives(
            FORM_A, rank, is_attention_layer=False, worker_skips_dense=True,
            host_dense_is_unsharded=True, host_uses_moe_exchange=False,
            moe_input_carrier=carrier,
        )

    host = _ops(0, "all_reduce_zero")
    blobs = [
        _encode(host),
        _encode(_ops(1, "broadcast")),
        _encode(_ops(2, "all_reduce_zero")),
    ]
    with pytest.raises(FormARanksDisagree, match="RANKS DISAGREE"):
        assert_ranks_agree(FORM_A, 0, host, lambda _blob: blobs)


def test_the_boot_gate_passes_when_every_rank_reads_the_same_carrier():
    from sglang.srt.form_a_boot_gate import gate_form_a_boot

    def _gather(blob):
        return [blob, blob, blob]

    gate_form_a_boot(
        FORM_A, 0, _gather, worker_skips_dense=True,
        host_dense_is_unsharded=True, host_uses_moe_exchange=False,
        moe_input_carrier="all_reduce_zero",
    )
    # and it stays a no-op on a classic boot
    gate_form_a_boot(
        None, 0, _gather, worker_skips_dense=False,
        host_dense_is_unsharded=False, host_uses_moe_exchange=False,
    )
