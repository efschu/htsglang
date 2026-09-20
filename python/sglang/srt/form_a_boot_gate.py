# SPDX-License-Identifier: Apache-2.0
"""'RAENGE NIE UNEINS' as a boot-path refusal, not as a hope.

The user law is CRASH/STOP on disagreement, never a hang. Form A makes the
law load-bearing: the ranks no longer run the same forward, so "they agree
about the collectives" stops being true by construction and becomes a claim
that has to be checked.

`form_a_symmetry` checks a MODEL of the forward at the desk. This checks the
REAL thing at boot, once, before the first forward: every rank declares the
collective sequence it will issue for one layer, the host compares them, and
a disagreement raises here -- where there is a stack, a log line and three
live processes -- instead of parking every card in a different collective
with nothing to show for it.

Deliberately NOT a per-forward check: the declaration is a property of the
layout, not of a batch, so checking it once at boot costs one small
all-gather and checking it per layer would cost 48 per round forever.
"""

from __future__ import annotations

import json
from typing import Callable, List, Optional, Sequence

from sglang.srt.form_a_symmetry import CollectiveMismatch, CollectiveOp, RankTrace
from sglang.srt.rank_role import RankRolePlan

__all__ = [
    "FormARanksDisagree",
    "declare_layer_collectives",
    "assert_ranks_agree",
    "gate_form_a_boot",
]


class FormARanksDisagree(RuntimeError):
    """The ranks would enter different collectives. CRASH/STOP, never hang."""


def declare_layer_collectives(
    plan: RankRolePlan,
    rank: int,
    *,
    is_attention_layer: bool,
    worker_skips_dense: bool,
    host_dense_is_unsharded: bool,
    host_uses_moe_exchange: bool,
    moe_input_carrier: Optional[str] = None,
) -> List[CollectiveOp]:
    """What THIS rank will issue for one decoder layer.

    Built from the same `trace_layer` the desk probe uses, so the boot gate
    and the desk verdict cannot drift apart -- two spellings of "which
    collectives" is how a gate comes to pass while the boot hangs.

    `moe_input_carrier` is declared, not assumed: it is read per process
    from the environment (form_a_worker_forward.moe_input_carrier), so a
    rank that read a different spelling issues a different op here and the
    gate stops the boot instead of letting it wedge on op #1 of layer 0.
    """
    from sglang.srt.form_a_symmetry import trace_layer

    trace = RankTrace(rank=rank, role=plan.role_of(rank))
    trace_layer(
        trace,
        0,
        is_attention_layer,
        worker_skips_dense=worker_skips_dense,
        host_uses_moe_exchange=host_uses_moe_exchange,
        host_dense_is_unsharded=host_dense_is_unsharded,
        moe_input_carrier=moe_input_carrier,
    )
    return trace.ops


def _encode(ops: Sequence[CollectiveOp]) -> str:
    return json.dumps([list(op.key()) for op in ops], separators=(",", ":"))


def _decode(rank: int, role: str, blob: str) -> RankTrace:
    t = RankTrace(rank=rank, role=role)
    for kind, site, payload in json.loads(blob):
        t.issue(kind, site, payload)
    return t


def assert_ranks_agree(
    plan: RankRolePlan,
    rank: int,
    ops: Sequence[CollectiveOp],
    gather: Callable[[str], List[str]],
) -> None:
    """Compare every rank's declaration. Raises on ANY disagreement.

    `gather` is injected rather than imported: at boot it is a CPU-group
    all-gather of strings, in a test it is a list. That keeps this function
    -- the one that decides whether the rig starts -- runnable without a
    process group.
    """
    blobs = gather(_encode(ops))
    if len(blobs) != plan.tp_size:
        raise FormARanksDisagree(
            f"the symmetry gate gathered {len(blobs)} declarations for "
            f"{plan.tp_size} ranks. A missing declaration is itself a "
            "disagreement: that rank either never reached the gate or is "
            "not running this layout."
        )
    traces = [_decode(r, plan.role_of(r), b) for r, b in enumerate(blobs)]
    from sglang.srt.form_a_symmetry import check_symmetry

    try:
        check_symmetry(traces)
    except CollectiveMismatch as e:
        summary = "; ".join(
            f"rank {t.rank} ({t.role}): "
            + ",".join(f"{o.kind}@{o.site}" for o in t.ops)
            for t in traces
        )
        raise FormARanksDisagree(
            f"RANKS DISAGREE ABOUT THE COLLECTIVES -- stopping the boot "
            f"instead of hanging it. {e} Declarations: {summary}"
        ) from e


def gate_form_a_boot(
    plan: Optional[RankRolePlan],
    rank: int,
    gather: Callable[[str], List[str]],
    *,
    worker_skips_dense: bool,
    host_dense_is_unsharded: bool,
    host_uses_moe_exchange: bool,
    attention_every: int = 4,
    moe_input_carrier: Optional[str] = None,
) -> None:
    """The whole gate: both layer shapes, checked once, before forward one.

    A no-op when `plan` is None -- on a classic boot every rank runs the
    same code and the property holds by construction, so the gate must cost
    nothing there.

    `moe_input_carrier=None` means "ask this process" -- the gate then reads
    the same environment variable the forward reads, which is the only way
    the declaration can catch a rank whose env differs. Pass it explicitly
    only from a test.
    """
    if plan is None:
        return
    if moe_input_carrier is None and worker_skips_dense:
        from sglang.srt.form_a_worker_forward import moe_input_carrier as _carrier

        moe_input_carrier = _carrier()
    for is_attn in (False, True):
        ops = declare_layer_collectives(
            plan,
            rank,
            is_attention_layer=is_attn,
            worker_skips_dense=worker_skips_dense,
            host_dense_is_unsharded=host_dense_is_unsharded,
            host_uses_moe_exchange=host_uses_moe_exchange,
            moe_input_carrier=moe_input_carrier,
        )
        assert_ranks_agree(plan, rank, ops, gather)
