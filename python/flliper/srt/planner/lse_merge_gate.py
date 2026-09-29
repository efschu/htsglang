"""#704b: the byte-identity gate harness for the decoupled attention path.

DESIGN_704 originally demanded byte-identity "decoupled vs coupled". That is
A-vs-B, not A-vs-A: an LSE merge sums partial results in a different
floating-point order than monolithic attention, so bit-exact agreement is not a
property correct code has. As written the gate would fail forever on a correct
implementation, and the predictable outcome is that someone waives it -- which
is worse than having no gate at all.

Respecified as two gates, both harnessed here:

**Gate 1 -- DETERMINISM, byte-identical A-vs-A. Never waivable.** The same
inputs merged twice, across runs and across boots, must give a bit-identical
result. This is what catches the most likely silent defect in a distributed
merge: folding partials in ARRIVAL order rather than rank order, which produces
a different rounding every run and cannot be seen in any single run.

**Gate 2 -- AGREEMENT with the coupled reference within a tolerance fixed
BEFORE the run**, plus a greedy-decode token-sequence match.

Inputs are sampled on CPU and moved to device. ``torch.randn`` on-GPU is not
architecture-identical across the 3080s and the 5090, and this rig has been
bitten by that before; a harness that seeded on-GPU would make gate 1 fail for
a reason that has nothing to do with the merge.

:func:`merge_partials` is the CONTRACT the GPU path must satisfy. It matches
the semantics of ``layers/dcp/comm.py:228-262``
(``cp_lse_ag_out_ar_mha_uneven``), which all-gathers every rank's LSE and
reduces with a single ``logsumexp`` over the stacked axis -- so the merge order
is RANK order, fixed by the collective, and never arrival order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import torch


class LseMergeGateError(AssertionError):
    """A gate failure. An AssertionError because it is a test-time verdict."""


def cpu_sampled_inputs(
    n_ranks: int,
    tokens: int,
    heads: int,
    head_dim: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic per-rank ``(out, lse)`` partials, sampled on CPU.

    On CPU by construction, not by convention: ``torch.randn`` on-GPU differs
    across architectures, so seeding on device would make the determinism gate
    fail for a reason unrelated to the merge.
    """
    if n_ranks <= 0:
        raise LseMergeGateError("n_ranks must be positive.")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(int(n_ranks)):
        o = torch.randn(tokens, heads, head_dim, generator=gen, dtype=dtype)
        lse = torch.randn(tokens, heads, generator=gen, dtype=dtype)
        out.append((o, lse))
    return out


def merge_partials(
    partials: Sequence[tuple[torch.Tensor, torch.Tensor]],
    rank_order: Sequence[int] | None = None,
) -> torch.Tensor:
    """LSE-merge per-rank partial attention outputs, in RANK order.

    ``partials[i]`` is ``(out_i, lse_i)`` with ``out_i`` shaped
    ``(tokens, heads, head_dim)`` and ``lse_i`` shaped ``(tokens, heads)``,
    where ``out_i`` is already softmax-normalised within rank ``i``'s shard.

    ``rank_order[i]`` is the RANK that ``partials[i]`` came from. Supply it
    whenever the caller's list is not already in rank order; the merge then
    reorders before reducing. Without it the list order IS taken as rank order,
    deliberately: a caller that shuffles and stays silent has a bug, and the
    harness must not launder it into a plausible answer.
    """
    if not partials:
        raise LseMergeGateError("no partials to merge.")
    items = list(partials)
    if rank_order is not None:
        if sorted(rank_order) != list(range(len(items))):
            raise LseMergeGateError(
                f"rank_order {list(rank_order)} is not a permutation of "
                f"0..{len(items) - 1}."
            )
        ordered: list[tuple[torch.Tensor, torch.Tensor]] = [None] * len(items)  # type: ignore[list-item]
        for pos, rank in enumerate(rank_order):
            ordered[int(rank)] = items[pos]
        items = ordered

    o0, l0 = items[0]
    for i, (o, lse) in enumerate(items):
        if o.shape != o0.shape or lse.shape != l0.shape:
            raise LseMergeGateError(
                f"partial {i} has shape {tuple(o.shape)}/{tuple(lse.shape)} "
                f"against {tuple(o0.shape)}/{tuple(l0.shape)}; refusing to "
                "broadcast a shape mismatch into a plausible answer."
            )
    if len(items) == 1:
        # The all-local degenerate case is exactly a no-op, not a rescale.
        return o0

    lses = torch.stack([lse for _, lse in items], dim=0)
    total = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - total.unsqueeze(0)).unsqueeze(-1)
    outs = torch.stack([o for o, _ in items], dim=0)
    return (weights * outs).sum(dim=0)


def assert_deterministic(fn: Callable[[], torch.Tensor], runs: int = 3) -> None:
    """GATE 1. Run ``fn`` repeatedly and demand bit-identical results.

    Byte-identity, not closeness: the defect being hunted is an order-dependent
    reduction, whose signature is a difference in the last bits. A tolerance
    here would hide exactly what the gate is for.
    """
    if runs < 2:
        raise LseMergeGateError("determinism needs at least two runs to compare.")
    first = fn()
    for i in range(1, int(runs)):
        again = fn()
        if not torch.equal(first, again):
            diff = (first.double() - again.double()).abs().max().item()
            raise LseMergeGateError(
                f"the merge is not deterministic: run {i} differs from run 0 "
                f"by up to {diff:.3e}. A distributed merge that varies between "
                "identical runs is folding partials in arrival order rather "
                "than rank order; no tolerance may be applied to this gate."
            )


@dataclasses.dataclass(frozen=True)
class AgreementReport:
    passed: bool
    max_abs_seen: float
    max_rel_seen: float
    max_abs_allowed: float
    max_rel_allowed: float


def agreement_report(
    got: torch.Tensor,
    reference: torch.Tensor,
    max_abs: float,
    max_rel: float,
) -> AgreementReport:
    """GATE 2. Compare against the coupled reference within a fixed tolerance.

    The tolerances are arguments rather than defaults on purpose: they are to be
    fixed BEFORE the run and recorded with the result. A tolerance chosen after
    seeing the numbers is not a gate.
    """
    if got.shape != reference.shape:
        raise LseMergeGateError(
            f"shape {tuple(got.shape)} against reference {tuple(reference.shape)}."
        )
    a = got.double()
    b = reference.double()
    abs_err = (a - b).abs()
    rel_err = abs_err / b.abs().clamp_min(1e-30)
    max_a = float(abs_err.max().item())
    max_r = float(rel_err.max().item())
    return AgreementReport(
        passed=(max_a <= float(max_abs)) and (max_r <= float(max_rel)),
        max_abs_seen=max_a,
        max_rel_seen=max_r,
        max_abs_allowed=float(max_abs),
        max_rel_allowed=float(max_rel),
    )
