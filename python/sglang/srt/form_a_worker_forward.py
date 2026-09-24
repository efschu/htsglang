# SPDX-License-Identifier: Apache-2.0
"""Form A slice 6a, second half: what a WORKER rank actually RUNS.

Slice 6 decided what a worker BUILDS (`form_a_construction`). This decides
what it EXECUTES, and the two halves cannot ship apart: a worker whose dense
modules are placeholders and whose forward is still the full model forward
reaches the first placeholder and raises; a worker that skips the dense
forward while the host still issues the dense collectives hangs (the
slice-6a finding, seam F12).

THE ONE QUESTION THIS FILE ANSWERS: how does a rank with no dense chain get
the MoE input?

Today every rank computes it: the per-layer dense path ends in an
all-reduce, so after it every rank holds the SAME hidden states and each
runs its own router over them. Under Form A only the host has a dense chain,
so that value exists on exactly one card and has to be carried.

The carrier chosen here is deliberately the CHEAPEST THING THAT IS ALREADY
THERE: the same collective, with the workers contributing ZEROS.

    host   : all_reduce(moe_in)       -> moe_in
    worker : all_reduce(zeros_like)   -> moe_in

An all-reduce whose non-host addends are zero IS a broadcast; it needs no
new transport, no new window, no subgroup (seam F6 stays unwired), and the
sequence it produces is identical on every rank, which is what the boot gate
checks. `SGLANG_FORM_A_MOE_INPUT=broadcast` selects a real one-to-many
broadcast instead -- same position in the sequence, cheaper on the wire,
but it is a DIFFERENT op and every rank must agree about which one is used,
so the choice is declared to the gate rather than taken per rank.

**The correction this carrier forces on the design note.** The slice-6a
switch matrix (DESIGN_FORM_A_0920.md §5a-BIS) prices "the simple Form A" at
48 collectives per round. That count models only the MoE COMBINE and is
silent about the MoE INPUT, because on a classic boot the input needs no
collective at all. With the carrier the honest count is **96 per round**
(one carrier + one combine per MoE layer) -- the same as the full
broadcast/reduce exchange, against 108 today. The gain of Form A was never
the collective COUNT (§2.3: "der Gewinn von Form A ist nicht weniger
Transport, sondern das Verschwinden der Schiefe"); that sentence survives
the correction unchanged, the number does not.

**Why the worker keeps its ROUTER.** The alternative to running the router
on every rank is to transport the host's `topk_ids`/`topk_weights`. That is
a second payload per layer whose width grows with k and with the batch,
and the ids are GLOBAL expert ids that each rank would have to filter into
its own shard anyway. The router is a replicated `[hidden, num_experts]`
matmul -- 0.12 GiB over 48 layers, measured on boot fn8ah, and a worker
already carries exactly that today. So `moe_gate` moves from the host-only
list to the worker list, and the acceptance criterion becomes
`("experts", "moe_gate")` rather than `("experts",)`. The criterion is not
weakened by that: the real eight-category worker census from fn8ah still
fails it, which is the property the accompanying test pins.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

from sglang.srt.rank_role import RankRoleError, this_rank_is_form_a_worker

__all__ = [
    "FormAWorkerForwardError",
    "FormAMoeInputUnavailable",
    "FormAWorkerLayerMismatch",
    "MOE_INPUT_CARRIERS",
    "moe_input_carrier",
    "publish_moe_input",
    "receive_moe_input",
    "form_a_moe_blocks",
    "run_form_a_worker_layers",
]


class FormAWorkerForwardError(RuntimeError):
    """A Form A worker's stripped forward cannot proceed -- by name."""


class FormAMoeInputUnavailable(FormAWorkerForwardError):
    """The worker cannot derive the shape of the MoE input it must receive."""


class FormAWorkerLayerMismatch(FormAWorkerForwardError):
    """The worker's MoE layer list is not the host's."""


#: The ways the host's MoE input can reach a worker. Both are ONE op per
#: layer and both are rank-uniform; they differ only in what the wire moves.
MOE_INPUT_CARRIERS: Tuple[str, ...] = ("all_reduce_zero", "broadcast")

_CARRIER_ENV = "SGLANG_FORM_A_MOE_INPUT"


def moe_input_carrier(env: Optional[dict] = None) -> str:
    """Which carrier this process uses. Refuses an unknown spelling.

    Read from the environment rather than from `server_args` because both
    the model forward and the boot gate need it, and threading a
    `server_args` into the layer forward is exactly the coupling
    `rank_role.installed_role_plan()` exists to avoid. Every rank reads the
    same variable from the same launcher line; a rank that read a different
    value would declare a different sequence and the boot gate would stop
    the boot instead of letting it hang.
    """
    src = os.environ if env is None else env
    value = str(src.get(_CARRIER_ENV, "") or "all_reduce_zero").strip().lower()
    if value not in MOE_INPUT_CARRIERS:
        raise RankRoleError(
            f"{_CARRIER_ENV}={value!r} is not a Form A MoE-input carrier; "
            f"known: {list(MOE_INPUT_CARRIERS)}. Refusing rather than "
            "defaulting: the carrier is part of the collective sequence, so "
            "two ranks reading it differently is a hang, not a slowdown."
        )
    return value


# ---------------------------------------------------------------------------
# The carrier itself. Both sides are one function each, and both take their
# collective as an argument so the desk can run them without a process group.
# ---------------------------------------------------------------------------
def _default_all_reduce(tensor):
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(tensor)


def _default_broadcast(tensor, src: int):
    from sglang.srt.distributed import get_tp_group

    return get_tp_group().broadcast(tensor, src=src)


def publish_moe_input(
    hidden_states,
    *,
    host_rank: int = 0,
    carrier: Optional[str] = None,
    all_reduce: Optional[Callable[[Any], Any]] = None,
    broadcast: Optional[Callable[[Any, int], Any]] = None,
):
    """HOST side: hand this layer's MoE input to the workers.

    Returns the value the host then feeds to its own MoE block, so the call
    site is a single rebinding and the host cannot accidentally compute its
    experts on a tensor the workers never saw.

    An EMPTY batch is not carried: the MoE block short-circuits a zero-row
    forward before it reaches its own all-reduce (qwen2_moe.py:904-910 vs
    :931-938), so carrying the input would add a collective the combine does
    not answer. The row count is rank-uniform, so this branch is taken by
    every rank or by none.
    """
    if hidden_states.shape[0] == 0:
        return hidden_states
    mode = carrier or moe_input_carrier()
    if mode == "broadcast":
        return (broadcast or _default_broadcast)(hidden_states, host_rank)
    return (all_reduce or _default_all_reduce)(hidden_states)


def receive_moe_input(
    num_tokens: int,
    hidden_size: int,
    *,
    dtype,
    device,
    host_rank: int = 0,
    carrier: Optional[str] = None,
    all_reduce: Optional[Callable[[Any], Any]] = None,
    broadcast: Optional[Callable[[Any, int], Any]] = None,
    zeros: Optional[Callable[..., Any]] = None,
):
    """WORKER side: the same op, with nothing of its own to add.

    The zero tensor is not a placeholder for missing work -- it IS the
    worker's contribution: it holds no dense weights, so its addend to the
    dense sum is exactly zero, and the all-reduce therefore returns the
    host's value unchanged on every rank.
    """
    if num_tokens <= 0:
        raise FormAMoeInputUnavailable(
            "a Form A worker was asked to receive a MoE input for "
            f"{num_tokens} rows. The row count is rank-uniform, so the host "
            "would have short-circuited an empty batch before its own "
            "collective -- receiving here would leave this rank waiting on a "
            "collective nobody issues."
        )
    if hidden_size <= 0:
        raise FormAMoeInputUnavailable(
            f"hidden_size={hidden_size} for the Form A MoE-input carrier; "
            "the worker cannot allocate a receive buffer of unknown width."
        )
    mode = carrier or moe_input_carrier()
    make = zeros
    if make is None:
        import torch

        make = torch.zeros
    buf = make((num_tokens, hidden_size), dtype=dtype, device=device)
    if mode == "broadcast":
        return (broadcast or _default_broadcast)(buf, host_rank)
    return (all_reduce or _default_all_reduce)(buf)


# ---------------------------------------------------------------------------
# The worker's layer list
# ---------------------------------------------------------------------------
def form_a_moe_blocks(model) -> List[Tuple[int, Any]]:
    """`(layer_id, mlp)` for every layer whose MLP is a routed-MoE block.

    Derived from the SAME module tree the host walks, in the same order,
    rather than from a layer count in a config: the host's sequence is the
    order of `model.layers`, and a worker that re-derived it from anything
    else could drift by one layer and hang with no way to see why.

    A layer whose `mlp` has no `experts` attribute is a dense MLP layer --
    under Form A a worker holds none of it and it issues no collective on
    the host either, so it contributes nothing to the sequence.
    """
    blocks: List[Tuple[int, Any]] = []
    layers = getattr(model, "layers", None)
    if layers is None:
        raise FormAWorkerLayerMismatch(
            "the Form A worker forward needs `model.layers`; this model "
            f"({type(model).__name__}) does not expose it, so the worker "
            "cannot reproduce the host's per-layer collective sequence."
        )
    start = int(getattr(model, "start_layer", 0))
    for offset, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "experts", None) is None:
            continue
        layer_id = int(getattr(layer, "layer_id", start + offset))
        blocks.append((layer_id, mlp))
    if not blocks:
        raise FormAWorkerLayerMismatch(
            "a Form A worker found no routed-MoE block in the model it "
            "built. A worker holds experts and nothing else, so an empty "
            "list means it would issue no collective at all while the host "
            "issues two per layer -- a hang, reported here instead."
        )
    return blocks


def run_form_a_worker_layers(
    blocks: Sequence[Tuple[int, Any]],
    *,
    num_tokens: int,
    hidden_size: int,
    dtype,
    device,
    forward_batch=None,
    host_rank: int = 0,
    carrier: Optional[str] = None,
    receive: Optional[Callable[..., Any]] = None,
) -> int:
    """The worker's whole forward: per MoE layer, receive and compute.

    Runs NO dense module (it has none) and returns only the number of layers
    it served -- the MoE output is all-reduced into every rank by the block
    itself, and the worker has nothing downstream to do with it.
    """
    if not this_rank_is_form_a_worker():
        raise FormAWorkerForwardError(
            "run_form_a_worker_layers was called on a rank that is not a "
            "Form A worker. This path exists only for a rank with no dense "
            "weights; running it on the host would skip the whole dense "
            "model and silently produce garbage."
        )
    from sglang.srt.debug_utils import host_anon_probe as _hap

    recv = receive or receive_moe_input
    served = 0
    # H13: SGLANG_DEBUG_HOST_ANON_PROBE -- the worker's pass and its layers.
    _mode = getattr(forward_batch, "forward_mode", None)
    _hap.pass_begin(getattr(_mode, "name", "WORKER"), int(num_tokens), role="worker")
    for _layer_id, mlp in blocks:
        _hap.checkpoint("worker.layer", layer=_layer_id)
        moe_in = recv(
            num_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
            host_rank=host_rank,
            carrier=carrier,
        )
        mlp(moe_in, forward_batch)
        served += 1
    _hap.pass_end()
    return served
